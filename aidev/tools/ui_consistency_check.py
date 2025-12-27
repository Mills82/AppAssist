"""
Small CLI wrapper around aidev.stages.consistency_checks.run_ui_consistency_check.

Usage (from project root):
  python -m aidev.tools.ui_consistency_check [--root PATH] [--output report.json] [--out report.json] [--verbose]

Behavior:
- Prefers to call run_ui_consistency_check(project_root) from aidev.stages.consistency_checks when available.
- Normalizes legacy shapes into a canonical report with top-level keys: errors (list), warnings (list), details (dict).
- Emits the JSON report to stdout (and to --output/--out file if provided).
- Exits with code 0 when no errors, 1 when errors were found, and 2 on import/IO/runtime errors.

Notes for maintainers:
- This module deliberately uses only the standard library for the fallback path.
- The wrapper is defensive about the shape returned by run_ui_consistency_check; it will pass through
  the canonical shape unchanged, and will attempt to map common legacy shapes into the canonical shape.

Fallback behavior (when the authoritative module isn't importable or raises):
- A built-in, stdlib-only fallback (run_local_ui_checks) inspects aidev/ui/index.html for required DOM landmarks
  and produces a compatible canonical report {errors, warnings, details}.
- The fallback prefers a deterministic selectors contract at aidev/ui/selectors.json, or extracts literal selectors
  from aidev/ui/app.js, or falls back to a small canonical whitelist. Parsing issues and informational notes are
  recorded as warnings; missing critical selectors and missing UI landmarks are recorded as errors. If index.html
  cannot be read or the verification step detects no critical selectors at all, a RuntimeError is raised and the
  CLI exits with code 2.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Small whitelist of canonical selectors for projects that cannot emit literal selectors from app.js.
# This list is intentionally minimal; add project-specific canonical selectors here when needed.
CANONICAL_SELECTORS: List[str] = [
    "#events-list",
    "aria-live",
    "#diff-view",
    "#approval-modal",
]
# Critical selectors: at least one should be present in index.html for the checks to be meaningful.
CRITICAL_SELECTORS: List[str] = ["#events-list", "aria-live"]


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    # If it's an iterable but not a list (e.g. tuple), convert to list
    if isinstance(value, (tuple, set)):
        return list(value)
    # Single value -> wrap
    return [value]


def _normalize_result(raw: Any) -> Dict[str, Any]:
    """Backward-compatible normalizer for legacy return shapes.

    Produces a dict shaped like the old code: { 'issues_found', 'summary', 'diagnostics' }.
    This helper is used as an intermediate when classifying legacy outputs into the new canonical
    {errors, warnings, details} shape.
    """
    normalized: Dict[str, Any] = {
        "issues_found": False,
        "summary": {},
        "diagnostics": [],
    }

    # None or missing -> empty report
    if raw is None:
        return normalized

    # If it's already a dict, try to map expected keys
    if isinstance(raw, dict):
        # diagnostics can be under several names
        diagnostics = raw.get("diagnostics")
        if diagnostics is None:
            diagnostics = raw.get("issues")
        if diagnostics is None:
            diagnostics = raw.get("problems")
        normalized["diagnostics"] = _ensure_list(diagnostics)

        # summary may be provided under 'summary' or 'meta' or 'details'
        summary = raw.get("summary")
        if summary is None:
            summary = raw.get("meta")
        if summary is None:
            summary = raw.get("details")
        normalized["summary"] = summary if isinstance(summary, dict) else {}

        # determine issues_found via explicit key, legacy 'ok', or diagnostics presence
        if "issues_found" in raw:
            normalized["issues_found"] = bool(raw.get("issues_found"))
        elif "has_issues" in raw:
            normalized["issues_found"] = bool(raw.get("has_issues"))
        elif "ok" in raw:
            # legacy: ok==True means no issues
            normalized["issues_found"] = not bool(raw.get("ok"))
        else:
            normalized["issues_found"] = bool(normalized["diagnostics"]) 

        return normalized

    # If it's a sequence (list/tuple), try to heuristically map
    if isinstance(raw, (list, tuple)):
        seq = list(raw)
        # Common legacy: (ok, issues) or (issues, summary)
        if len(seq) == 2:
            a, b = seq
            # (ok_bool, issues_list)
            if isinstance(a, bool):
                normalized["issues_found"] = not bool(a)
                normalized["diagnostics"] = _ensure_list(b)
                normalized["summary"] = {}
                return normalized
            # (issues_list, summary_dict)
            if isinstance(b, dict):
                normalized["diagnostics"] = _ensure_list(a)
                normalized["summary"] = b
                normalized["issues_found"] = bool(normalized["diagnostics"])
                return normalized
        # Fallback: treat entire sequence as diagnostics
        normalized["diagnostics"] = _ensure_list(seq)
        normalized["issues_found"] = bool(normalized["diagnostics"]) 
        return normalized

    # If it's any other object, attempt to extract attributes
    try:
        # e.g., object with attributes .issues, .summary, .ok
        diagnostics = getattr(raw, "diagnostics", None) or getattr(raw, "issues", None)
        normalized["diagnostics"] = _ensure_list(diagnostics)
        summary = getattr(raw, "summary", None) or getattr(raw, "meta", None) or getattr(raw, "details", None)
        normalized["summary"] = summary if isinstance(summary, dict) else {}
        if hasattr(raw, "issues_found"):
            normalized["issues_found"] = bool(getattr(raw, "issues_found"))
        elif hasattr(raw, "ok"):
            normalized["issues_found"] = not bool(getattr(raw, "ok"))
        else:
            normalized["issues_found"] = bool(normalized["diagnostics"]) 
        return normalized
    except Exception:
        # As a last resort, stringify the raw result into diagnostics
        normalized["diagnostics"] = [str(raw)]
        normalized["issues_found"] = True
        return normalized


def _to_canonical_report(raw: Any) -> Dict[str, Any]:
    """Convert various raw results into the canonical report shape:
      { 'errors': list, 'warnings': list, 'details': dict }

    If raw already exposes 'errors'/'warnings'/'details', pass through (coercing types).
    Otherwise attempt best-effort mapping from legacy shapes.
    """
    # If it's a dict with canonical keys, coerce and return
    if isinstance(raw, dict) and ("errors" in raw or "warnings" in raw or "details" in raw):
        errors = [str(e) for e in _ensure_list(raw.get("errors"))]
        warnings = [str(w) for w in _ensure_list(raw.get("warnings"))]
        details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
        # If there is a top-level 'details' absent but 'summary' present, use that
        if not details and isinstance(raw.get("summary"), dict):
            details = raw.get("summary")
        return {"errors": errors, "warnings": warnings, "details": details}

    # Fallback: use legacy normalizer then classify diagnostics
    legacy = _normalize_result(raw)
    diagnostics = [str(d) for d in legacy.get("diagnostics", [])]
    summary = legacy.get("summary") if isinstance(legacy.get("summary"), dict) else {}

    errors: List[str] = []
    warnings: List[str] = []

    # Heuristic classification: messages that contain 'missing' or 'not found' are errors;
    # messages that indicate success/loading or parsing problems are warnings.
    for d in diagnostics:
        ld = d.lower()
        if "missing" in ld or "not found" in ld or "no 'aria-live'" in ld or "sse" in ld and "missing" in ld:
            errors.append(d)
        elif "failed to read" in ld or "failed to extract" in ld or "failed to" in ld:
            # treat runtime/IO failures as errors
            errors.append(d)
        else:
            warnings.append(d)

    return {"errors": errors, "warnings": warnings, "details": summary}


def extract_selectors_from_app_js(app_js_path: Path) -> List[str]:
    """Conservatively extract literal selectors/attributes from aidev/ui/app.js.

    Returns a list of selector descriptors. Possible returned forms:
      - '#some-id' for getElementById usages
      - '.some-class' for getElementsByClassName usages
      - 'data-foo' for data- attributes found as string literals
      - 'aria-live' for explicit aria-live string literals
      - raw querySelector argument verbatim for querySelector/querySelectorAll usages

    This function is intentionally conservative: it only extracts literal string arguments
    and normalizes them where possible. It does not execute JS and may miss dynamically
    constructed selectors.
    """
    if not app_js_path.exists():
        return []
    try:
        text = app_js_path.read_text(encoding="utf-8")
    except Exception:
        return []

    selectors: List[str] = []

    # querySelector / querySelectorAll
    for m in re.finditer(r"querySelector(All)?\(\s*['\"]([^'\"]+)['\"]\)", text):
        arg = m.group(2).strip()
        if arg:
            selectors.append(arg)

    # getElementById -> '#id'
    for m in re.finditer(r"(?:document\.)?getElementById\(\s*['\"]([^'\"]+)['\"]\)", text):
        idv = m.group(1).strip()
        if idv:
            selectors.append(f"#{idv}")

    # getElementsByClassName -> '.class' (take first class if space-separated)
    for m in re.finditer(r"(?:document\.)?getElementsByClassName\(\s*['\"]([^'\"]+)['\"]\)", text):
        cls = m.group(1).strip()
        if cls:
            first = cls.split()[0]
            selectors.append(f".{first}")

    # string literals containing data- attributes
    for m in re.finditer(r"['\"](data-[a-zA-Z0-9_:-]+)['\"]", text):
        selectors.append(m.group(1))

    # explicit aria-live string literal
    if re.search(r"['\"]aria-live['\"]", text):
        selectors.append("aria-live")

    # Normalize and deduplicate while preserving order
    seen = set()
    normalized: List[str] = []
    for s in selectors:
        s_norm = s.strip()
        if not s_norm:
            continue
        if s_norm not in seen:
            seen.add(s_norm)
            normalized.append(s_norm)
    return normalized


def load_selectors_contract(root: Path) -> Tuple[List[str], Optional[Path], List[str]]:
    """Attempt to read a deterministic selectors contract from aidev/ui/selectors.json.

    Returns a tuple: (selectors_list, source_path_or_None, diagnostics).
    - selectors_list: list of selector strings (may be empty)
    - source_path_or_None: Path to selectors.json when present, otherwise None
    - diagnostics: parse/shape diagnostics (empty on success/no file)

    Accepts either a JSON array or an object with key 'SELECTORS' or 'selectors'.
    Additionally accepts when 'SELECTORS'/'selectors' maps to an object/dict of key->selector_string;
    in that case the values are taken deterministically (sorted by key) to form the selectors list.
    Malformed JSON or unexpected shapes produce a diagnostic and an empty selectors list.
    """
    selectors_path = root / "aidev" / "ui" / "selectors.json"
    if not selectors_path.exists():
        return ([], None, [])

    diags: List[str] = []
    try:
        text = selectors_path.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception as exc:
        diags.append(f"failed to read/parse selectors.json at {selectors_path}: {exc}")
        return ([], selectors_path, diags)

    # If it's a list, accept directly
    if isinstance(data, list):
        selectors = [str(s).strip() for s in data if isinstance(s, (str,)) and str(s).strip()]
        # record a helpful diagnostic that selectors were loaded successfully
        if selectors:
            diags = [f"selectors contract loaded from {selectors_path}"]
        return (selectors, selectors_path, diags)

    # If it's a dict, accept 'SELECTORS' or 'selectors'
    if isinstance(data, dict):
        arr = None
        # Prefer explicit top-level keys
        if "SELECTORS" in data:
            arr = data.get("SELECTORS")
        elif "selectors" in data:
            arr = data.get("selectors")

        # If the key exists and is a list, accept
        if isinstance(arr, list):
            selectors = [str(s).strip() for s in arr if isinstance(s, (str,)) and str(s).strip()]
            if selectors:
                diags = [f"selectors contract loaded from {selectors_path}"]
            return (selectors, selectors_path, diags)

        # If the key exists and is a dict (mapping), convert deterministically by sorting keys
        if isinstance(arr, dict):
            # Deterministically order by key
            selectors = []
            for key in sorted(arr.keys()):
                val = arr.get(key)
                if val is None:
                    continue
                # coerce to string and strip
                s = str(val).strip()
                if s:
                    selectors.append(s)
            if selectors:
                diags = [f"selectors contract loaded from {selectors_path} (mapping converted to list)"]
            return (selectors, selectors_path, diags)

        # As an additional convenience, accept the case where the top-level object IS the mapping
        # i.e. selectors.json = { "id": "#foo", "x": ".bar" }
        # but only if none of the conventional keys were present
        # Detect if all top-level values are string-like -> treat as mapping
        all_string_values = all(isinstance(v, (str, type(None), int, float)) for v in data.values()) and bool(data)
        if all_string_values:
            selectors = []
            for key in sorted(data.keys()):
                val = data.get(key)
                if val is None:
                    continue
                s = str(val).strip()
                if s:
                    selectors.append(s)
            if selectors:
                diags = [f"selectors contract loaded from {selectors_path} (top-level mapping converted to list)"]
            return (selectors, selectors_path, diags)

        diags.append(f"selectors.json at {selectors_path} does not contain a top-level array, a 'SELECTORS'/'selectors' list, or a mapping of selector keys -> values")
        return ([], selectors_path, diags)

    diags.append(f"selectors.json at {selectors_path} has unexpected JSON shape (must be array or object)")
    return ([], selectors_path, diags)


def run_local_ui_checks(root: Path, verbose: bool = False) -> Dict[str, Any]:
    """Stdlib-only fallback: inspect aidev/ui/index.html for required UI landmarks.

    Returns the canonical dict: { 'errors': list, 'warnings': list, 'details': dict }.

    Raises IOError/FileNotFoundError if index.html cannot be read (these are treated as runtime errors
    by main and cause exit code 2).
    """
    ui_index = root / "aidev" / "ui" / "index.html"
    if not ui_index.exists():
        raise FileNotFoundError(f"expected UI file not found: {ui_index}")

    text = ui_index.read_text(encoding="utf-8")
    lower_text = text.lower()

    errors: List[str] = []
    warnings: List[str] = []

    # First, prefer deterministic selectors.json if present
    selectors_from_json, json_path, loader_diags = load_selectors_contract(root)
    # loader_diags: treat parse errors and notes as warnings (non-fatal)
    warnings.extend(loader_diags)

    if selectors_from_json:
        if verbose:
            print(f"INFO: using selectors contract at {json_path}", file=sys.stderr)

        selectors = selectors_from_json
        warnings.append(f"using selectors from selectors.json: {json_path}")

        # Validate each selector conservatively against index.html
        for sel in selectors:
            # id selectors '#id'
            if sel.startswith("#"):
                idname = sel[1:]
                if not re.search(r"id\s*=\s*['\"]" + re.escape(idname) + r"['\"]", text, flags=re.I):
                    errors.append(f"missing selector '{sel}' referenced in {json_path}: id not found in {ui_index}")
                continue
            # class selectors '.cls'
            if sel.startswith("."):
                cls = sel[1:]
                if not re.search(r"class\s*=\s*['\"][^'\"]*\b" + re.escape(cls) + r"\b[^'\"]*['\"]", text, flags=re.I):
                    errors.append(f"missing selector '{sel}' referenced in {json_path}: class not found in {ui_index}")
                continue
            # data- attributes
            if sel.startswith("data-"):
                if sel.lower() not in lower_text:
                    errors.append(f"missing data attribute '{sel}' referenced in {json_path} in {ui_index}")
                continue
            # aria-live
            if sel == "aria-live":
                if "aria-live" not in lower_text:
                    errors.append(f"missing 'aria-live' region referenced in {json_path} in {ui_index}")
                continue
            # fallback: check selector string presence conservatively
            if sel.lower() not in lower_text:
                errors.append(f"missing selector or string " + repr(sel) + f" referenced in {json_path} in {ui_index}")

        details = {"file": str(ui_index), "checked": selectors, "source": str(json_path)}

        # Verification step: ensure at least one critical selector exists in index.html
        found_critical = False
        for crit in CRITICAL_SELECTORS:
            if crit == "aria-live":
                if "aria-live" in lower_text:
                    found_critical = True
                    break
            elif crit.startswith("#"):
                # look for id attribute
                idname = crit[1:]
                if re.search(r"id\s*=\s*['\"]" + re.escape(idname) + r"['\"]", text, flags=re.I):
                    found_critical = True
                    break
            else:
                if crit.lower() in lower_text:
                    found_critical = True
                    break
        if not found_critical:
            # This indicates the project's markup and the checker's expectations are misaligned.
            raise RuntimeError(
                f"UI consistency checks appear misaligned: none of the critical selectors {CRITICAL_SELECTORS} were found in {ui_index}. "
                f"This checker expects at least one of these selectors to be present. Expected/canonical selectors: {CANONICAL_SELECTORS}. "
                "If your app.js is generated and cannot expose literal selectors, please add the project's canonical selectors to CANONICAL_SELECTORS in this script."
            )

        return {"errors": errors, "warnings": warnings, "details": details}

    # If selectors.json was not present/usable, attempt app.js extraction when app.js exists
    app_js = root / "aidev" / "ui" / "app.js"
    if app_js.exists():
        try:
            selectors = extract_selectors_from_app_js(app_js)
        except Exception as exc:  # defensive
            warnings.append(f"failed to extract selectors from {app_js}: {exc}")
            selectors = []

        if not selectors:
            # app.js present but no selectors extracted -> emit clear warning and prefer canonical whitelist
            warnings.append(
                f"app.js present at {app_js} but no literal selectors were extracted; attempting canonical selector whitelist for deterministic checks"
            )
            warnings.append(
                "expected/canonical selectors: " + ", ".join(CANONICAL_SELECTORS)
            )
            # Prefer deterministic whitelist when the app cannot expose selectors literally
            selectors = list(CANONICAL_SELECTORS)
            source_label = "canonical-whitelist"
            warnings.append(f"using canonical selectors whitelist as selectors source: {source_label}")
            if verbose:
                print(f"INFO: using canonical selectors whitelist as no selectors were extracted from {app_js}", file=sys.stderr)
        else:
            source_label = str(app_js)
            warnings.append(f"selectors extracted from {app_js}")
            if verbose:
                print(f"INFO: using selectors extracted from {app_js}", file=sys.stderr)

        # Validate each selector conservatively against index.html
        for sel in selectors:
            # id selectors '#id'
            if sel.startswith("#"):
                idname = sel[1:]
                if not re.search(r"id\s*=\s*['\"]" + re.escape(idname) + r"['\"]", text, flags=re.I):
                    errors.append(f"missing selector '{sel}' referenced in {app_js}: id not found in {ui_index}")
                continue
            # class selectors '.cls'
            if sel.startswith("."):
                cls = sel[1:]
                if not re.search(r"class\s*=\s*['\"][^'\"]*\b" + re.escape(cls) + r"\b[^'\"]*['\"]", text, flags=re.I):
                    errors.append(f"missing selector '{sel}' referenced in {app_js}: class not found in {ui_index}")
                continue
            # data- attributes
            if sel.startswith("data-"):
                if sel.lower() not in lower_text:
                    errors.append(f"missing data attribute '{sel}' referenced in {app_js} in {ui_index}")
                continue
            # aria-live
            if sel == "aria-live":
                if "aria-live" not in lower_text:
                    errors.append(f"missing 'aria-live' region referenced in {app_js} in {ui_index}")
                continue
            # fallback: check selector string presence conservatively
            m_id = re.match(r"^#([A-Za-z0-9_:-]+)$", sel)
            if m_id:
                idname = m_id.group(1)
                if not re.search(r"id\s*=\s*['\"]" + re.escape(idname) + r"['\"]", text, flags=re.I):
                    errors.append(f"missing selector '{sel}' referenced in {app_js}: id not found in {ui_index}")
                continue
            # Otherwise, do a substring check (conservative)
            if sel.lower() not in lower_text:
                errors.append(f"missing selector or string " + repr(sel) + f" referenced in {app_js} in {ui_index}")

        details = {"file": str(ui_index), "checked": selectors, "source": source_label}

        # Verification step: ensure at least one critical selector exists in index.html
        found_critical = False
        for crit in CRITICAL_SELECTORS:
            if crit == "aria-live":
                if "aria-live" in lower_text:
                    found_critical = True
                    break
            elif crit.startswith("#"):
                # look for id attribute
                idname = crit[1:]
                if re.search(r"id\s*=\s*['\"]" + re.escape(idname) + r"['\"]", text, flags=re.I):
                    found_critical = True
                    break
            else:
                if crit.lower() in lower_text:
                    found_critical = True
                    break
        if not found_critical:
            # This indicates the project's markup and the checker's expectations are misaligned.
            raise RuntimeError(
                f"UI consistency checks appear misaligned: none of the critical selectors {CRITICAL_SELECTORS} were found in {ui_index}. "
                f"Expected/canonical selectors: {CANONICAL_SELECTORS}. Please update index.html or add the project's canonical selectors to CANONICAL_SELECTORS in this script."
            )

        return {"errors": errors, "warnings": warnings, "details": details}

    # Legacy permissive marker groups (kept as fallback) when no selectors.json and no app.js
    marker_groups = {
        "onboarding": [
            '<template id="onboarding"',
            'id="onboarding-banner"',
            'id="onboarding-template"',
            'data-aidev-target="onboarding-banner"',
            'data-aidev-target="onboarding-modal"',
        ],
        "help_button": [
            'id="btn-help"',
            'id="help-button"',
            'data-aidev-target="help-button"',
            'aria-label="help"',
        ],
        "sse_region": [
            'aria-live',  # any aria-live region is useful
            'id="sse-status"',
            'id="conn-status"',
            # fallback to id containing 'sse' or 'status' (regex below)
        ],
        "diff_viewer": [
            'id="diff-view"',
            'id="diff-viewer"',
            'id="diff-view-template"',
            'class="diff-container"',
            'id="diff"',
            'id="diff-template"',
        ],
        "approval_modal": [
            'id="approval-modal"',
            'aria-label="approval"',
            'data-modal="approval"',
            'data-aidev-target="plan-modal"',
            # presence of role="dialog" is checked via regex below
        ],
    }

    def has_any_marker(group_markers: List[str]) -> bool:
        for m in group_markers:
            if m.lower() in lower_text:
                return True
        return False

    # Check onboarding
    if not has_any_marker(marker_groups["onboarding"]):
        errors.append(
            f"missing onboarding template/banner (looked for: {', '.join(marker_groups['onboarding'])}) in {ui_index}"
        )

    # Check help button
    if not has_any_marker(marker_groups["help_button"]):
        errors.append(
            f"missing Help button (looked for: {', '.join(marker_groups['help_button'])}) in {ui_index}"
        )

    # Check SSE region: aria-live OR id containing 'sse' or 'status'
    sse_found = False
    if has_any_marker(marker_groups["sse_region"]):
        sse_found = True
    else:
        # regex for id attributes containing sse or status
        if re.search(r"id\s*=\s*['\"][^'\"]*(?:sse|status)[^'\"]*['\"]", text, flags=re.I):
            sse_found = True
    if not sse_found:
        errors.append(
            f"missing SSE aria-live/status region: no 'aria-live' or id containing 'sse'/'status' found in {ui_index}"
        )

    # Check diff viewer
    if not has_any_marker(marker_groups["diff_viewer"]):
        errors.append(
            f"missing diff viewer/template (looked for: {', '.join(marker_groups['diff_viewer'])}) in {ui_index}"
        )

    # Check approval modal: role="dialog" plus one of the approval markers
    dialog_present = bool(re.search(r"role\s*=\s*['\"]dialog['\"]", text, flags=re.I))
    approval_marker_present = any(m.lower() in lower_text for m in marker_groups["approval_modal"])
    if not (dialog_present and approval_marker_present):
        # Provide a combined helpful message
        errors.append(
            f"missing approval modal/plan dialog: need role=\"dialog\" and one of ({', '.join(marker_groups['approval_modal'])}) in {ui_index}"
        )

    details = {"file": str(ui_index), "checked": list(marker_groups.keys())}

    # Verification step for legacy path as well: ensure at least one critical selector exists
    found_critical = False
    for crit in CRITICAL_SELECTORS:
        if crit == "aria-live":
            if "aria-live" in lower_text:
                found_critical = True
                break
        elif crit.startswith("#"):
            idname = crit[1:]
            if re.search(r"id\s*=\s*['\"]" + re.escape(idname) + r"['\"]", text, flags=re.I):
                found_critical = True
                break
        else:
            if crit.lower() in lower_text:
                found_critical = True
                break
    if not found_critical:
        raise RuntimeError(
            f"UI consistency checks appear misaligned: none of the critical selectors {CRITICAL_SELECTORS} were found in {ui_index}. "
            f"Expected/canonical selectors: {CANONICAL_SELECTORS}. Please update index.html or add the project's canonical selectors to CANONICAL_SELECTORS in this script."
        )

    return {"errors": errors, "warnings": warnings, "details": details}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aidev.tools.ui_consistency_check",
        description="Run UI consistency checks and emit a JSON report.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root path to scan (default: current working directory)",
    )
    # --output (primary) and --out (alias) write the JSON report to a file
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help="Optional path to write the JSON report",
    )
    parser.add_argument(
        "--out",
        dest="output",
        type=str,
        default=None,
        help="Alias for --output",
    )
    parser.add_argument("--verbose", action="store_true", help="Write minimal human logs to stderr")
    parser.add_argument(
        "--dump-selectors",
        action="store_true",
        help="Dump the selectors contract (if any) and exit (for local verification).",
    )

    args = parser.parse_args(argv)

    root_path = Path(args.root) if args.root is not None else Path.cwd()

    # If requested, just dump the selectors contract and exit (useful as a simple local test)
    if args.dump_selectors:
        selectors, src, diags = load_selectors_contract(root_path)
        out = {"selectors": selectors, "source": str(src) if src is not None else None, "diagnostics": diags}
        sys.stdout.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        return 0

    # Import the target function lazily; if it's not available, fall back to run_local_ui_checks
    run_ui_consistency_check = None
    try:
        from aidev.stages.consistency_checks import run_ui_consistency_check as _rcc  # type: ignore
        run_ui_consistency_check = _rcc
    except Exception as exc:  # pragma: no cover - import-time failure path
        if args.verbose:
            print(
                f"INFO: could not import aidev.stages.consistency_checks.run_ui_consistency_check; will use local fallback: {exc}",
                file=sys.stderr,
            )

    if args.verbose and run_ui_consistency_check is not None:
        print(f"Running external run_ui_consistency_check in: {root_path}", file=sys.stderr)
    elif args.verbose and run_ui_consistency_check is None:
        print(f"Running local UI checks in: {root_path}", file=sys.stderr)

    # Try to run the preferred checker. If external exists, prefer it; on its runtime error, try fallback.
    raw_result = None
    if run_ui_consistency_check is not None:
        try:
            raw_result = run_ui_consistency_check(root_path)
        except Exception as exc:  # pragma: no cover - runtime failure path
            # Prefer to fall back to local checks rather than failing immediately
            print(f"WARNING: external run_ui_consistency_check raised an exception: {exc}", file=sys.stderr)
            try:
                raw_result = run_local_ui_checks(root_path, verbose=args.verbose)
            except Exception as exc2:
                print(f"ERROR: local fallback check failed: {exc2}", file=sys.stderr)
                error_report = {
                    "errors": [str(exc), str(exc2)],
                    "warnings": [],
                    "details": {"error": "runtime_exception", "source": "external_and_local_failed"},
                }
                out_json = json.dumps(error_report, indent=2, ensure_ascii=False)
                sys.stdout.write(out_json + "\n")
                return 2
    else:
        # No external checker available; run local fallback
        try:
            raw_result = run_local_ui_checks(root_path, verbose=args.verbose)
        except Exception as exc:  # pragma: no cover - IO/runtime error while reading index.html
            print(f"ERROR: failed to run local UI checks: {exc}", file=sys.stderr)
            error_report = {
                "errors": [str(exc)],
                "warnings": [],
                "details": {"error": "runtime_exception", "source": "local_failed"},
            }
            out_json = json.dumps(error_report, indent=2, ensure_ascii=False)
            sys.stdout.write(out_json + "\n")
            return 2

    # At this point we have raw_result from either external or local checker
    report = _to_canonical_report(raw_result)

    # Emit JSON to stdout
    out_json = json.dumps(report, indent=2, ensure_ascii=False)
    sys.stdout.write(out_json + "\n")

    # Optionally write to file (alias --out also maps here)
    if args.output:
        try:
            out_path = Path(args.output)
            if out_path.parent:
                out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(out_json, encoding="utf-8")
            if args.verbose:
                print(f"Wrote JSON report to: {out_path}", file=sys.stderr)
        except Exception as exc:  # pragma: no cover - IO error
            print(f"ERROR: failed to write output file: {exc}", file=sys.stderr)
            return 2

    # Exit code: 0 when no errors, 1 when errors found
    return 1 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
