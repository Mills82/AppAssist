# aidev/stages/consistency_checks.py
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    ".aidev",
    ".idea",
    ".vscode",
}


def _normalize_cfg(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    return {}


def _iter_files(root: Path, patterns: List[str], ignore_dirs: Set[str]) -> List[Path]:
    """
    Yield files under root matching glob patterns, skipping common tool/vendor dirs.
    """
    files: List[Path] = []
    for pattern in patterns:
        try:
            for p in root.rglob(pattern):
                if not p.is_file():
                    continue
                # Skip ignored directories anywhere in the path
                if any(part in ignore_dirs for part in p.parts):
                    continue
                files.append(p)
        except Exception:
            logging.debug("Failed rglob pattern %r under %s", pattern, root, exc_info=True)
    return files


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        logging.debug("Failed to read %s", path, exc_info=True)
        return ""


def _collect_html_ids_classes(
    paths: List[Path],
    root: Path,
) -> Dict[str, Dict[str, Set[str]]]:
    id_to_html_files: Dict[str, Set[str]] = {}
    class_to_html_files: Dict[str, Set[str]] = {}
    data_target_to_html_files: Dict[str, Set[str]] = {}

    id_re = re.compile(r'id=["\']([A-Za-z_][A-Za-z0-9_-]*)["\']')
    class_re = re.compile(r'class=["\']([^"\']+)["\']')
    # Conservative data-aidev-target attribute extraction (literal values only)
    data_attr_re = re.compile(r'data-aidev-target=["\']([^"\']+)["\']')

    for path in paths:
        text = _read_text(path)
        if not text:
            continue

        rel = str(path.relative_to(root))

        for ident in id_re.findall(text):
            id_to_html_files.setdefault(ident, set()).add(rel)

        for cls_group in class_re.findall(text):
            for token in cls_group.split():
                token = token.strip()
                if not token:
                    continue
                class_to_html_files.setdefault(token, set()).add(rel)

        for target in data_attr_re.findall(text):
            data_target_to_html_files.setdefault(target, set()).add(rel)

    return {
        "ids": id_to_html_files,
        "classes": class_to_html_files,
        "data_targets": data_target_to_html_files,
    }


def _collect_css_ids_classes(
    paths: List[Path],
    root: Path,
) -> Dict[str, Dict[str, Set[str]]]:
    id_to_css_files: Dict[str, Set[str]] = {}
    class_to_css_files: Dict[str, Set[str]] = {}
    data_target_to_css_files: Dict[str, Set[str]] = {}

    # Very lightweight selector extraction; may include some false positives.
    id_re = re.compile(r'(?<![A-Za-z0-9_-])#([A-Za-z_][A-Za-z0-9_-]*)')
    class_re = re.compile(r'(?<![A-Za-z0-9_-])\.([A-Za-z_][A-Za-z0-9_-]*)')
    # Conservative CSS attribute selector extraction for data-aidev-target
    data_attr_selector_re = re.compile(r'\[\s*data-aidev-target\s*=\s*["\']([^"\']+)["\']\s*\]')

    for path in paths:
        text = _read_text(path)
        if not text:
            continue

        rel = str(path.relative_to(root))

        for ident in id_re.findall(text):
            id_to_css_files.setdefault(ident, set()).add(rel)

        for cls in class_re.findall(text):
            class_to_css_files.setdefault(cls, set()).add(rel)

        for target in data_attr_selector_re.findall(text):
            data_target_to_css_files.setdefault(target, set()).add(rel)

    return {
        "ids": id_to_css_files,
        "classes": class_to_css_files,
        "data_targets": data_target_to_css_files,
    }


def _collect_js_ids_classes(
    paths: List[Path],
    root: Path,
) -> Dict[str, Dict[str, Set[str]]]:
    id_to_js_files: Dict[str, Set[str]] = {}
    class_to_js_files: Dict[str, Set[str]] = {}
    data_target_to_js_files: Dict[str, Set[str]] = {}
    data_target_usage_files: Set[str] = set()

    # DOM id references
    get_by_id_re = re.compile(
        r'getElementById\(\s*["\']([A-Za-z_][A-Za-z0-9_-]*)["\']\s*\)'
    )
    qs_id_re = re.compile(
        r'querySelector(?:All)?\(\s*["\']#([A-Za-z_][A-Za-z0-9_-]*)["\']\s*\)'
    )

    # CSS class references
    qs_class_re = re.compile(
        r'querySelector(?:All)?\(\s*["\']\.([A-Za-z_][A-Za-z0-9_-]*)["\']\s*\)'
    )
    classlist_re = re.compile(
        r'classList\.(?:add|remove|toggle|contains)\(\s*["\']([A-Za-z_][A-Za-z0-9_-]*)["\']\s*\)'
    )

    # Conservative JS detection for data-aidev-target literal selectors inside selector strings
    js_data_attr_selector_re = re.compile(r'\[data-aidev-target\s*=\s*["\']([^"\']+)["\']\]')
    # getAttribute('data-aidev-target') usage (no literal value)
    getattr_data_re = re.compile(r'getAttribute\(\s*["\']data-aidev-target["\']\s*\)')
    # dataset.aidevTarget usage (no literal value)
    dataset_usage_re = re.compile(r'\.dataset\.aidevTarget\b')

    for path in paths:
        text = _read_text(path)
        if not text:
            continue

        rel = str(path.relative_to(root))

        for ident in get_by_id_re.findall(text):
            id_to_js_files.setdefault(ident, set()).add(rel)
        for ident in qs_id_re.findall(text):
            id_to_js_files.setdefault(ident, set()).add(rel)

        for cls in qs_class_re.findall(text):
            class_to_js_files.setdefault(cls, set()).add(rel)
        for cls in classlist_re.findall(text):
            class_to_js_files.setdefault(cls, set()).add(rel)

        # Find literal attribute selectors used in JS selector strings
        for target in js_data_attr_selector_re.findall(text):
            data_target_to_js_files.setdefault(target, set()).add(rel)

        # Record that a file uses the attribute via getAttribute or dataset (no value inference)
        if getattr_data_re.search(text) or dataset_usage_re.search(text):
            data_target_usage_files.add(rel)

    return {
        "ids": id_to_js_files,
        "classes": class_to_js_files,
        "data_targets": data_target_to_js_files,
        "data_targets_usage_files": data_target_usage_files,
    }


def run_consistency_checks(*, root: Path | str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run lightweight, project-local consistency checks after edits have been applied.

    This function accepts `root` as a pathlib.Path or a string; it is coerced to Path at
    the start of the function. It returns a machine-friendly dict intended for
    programmatic consumption by a small CLI wrapper. For backward compatibility the
    legacy keys "ok" and "issues" are still present; new keys are:

      - issues_found: bool  (True when any issues were found)
      - summary: dict       (counts: total_files_scanned, total_html_files, total_css_files,
                             total_js_files, total_issues, ok)
      - diagnostics: list   (the list of issue objects, identical to the legacy "issues")

    Existing behavior is preserved: detection heuristics, issue object shapes, logging
    and exception handling remain unchanged.
    """
    # Coerce root to a Path so callers may pass either a string or a Path
    root = Path(root)

    cfg_dict = _normalize_cfg(cfg)
    ignore_dirs = set(cfg_dict.get("ignore_dirs") or []) | DEFAULT_IGNORE_DIRS
    max_issues = int(cfg_dict.get("max_issues", 100))

    html_globs = cfg_dict.get("html_globs") or ["**/*.html", "**/*.htm"]
    css_globs = cfg_dict.get("css_globs") or ["**/*.css", "**/*.scss"]
    js_globs = cfg_dict.get("js_globs") or [
        "**/*.js",
        "**/*.ts",
        "**/*.tsx",
        "**/*.jsx",
    ]

    html_paths = _iter_files(root, html_globs, ignore_dirs)
    css_paths = _iter_files(root, css_globs, ignore_dirs)
    js_paths = _iter_files(root, js_globs, ignore_dirs)

    issues: List[Dict[str, Any]] = []

    def _add_issue(
        kind: str,
        message: str,
        files: List[str],
        identifiers: Optional[List[str]] = None,
        severity: str = "medium",
    ) -> None:
        if len(issues) >= max_issues:
            return
        issue: Dict[str, Any] = {
            "kind": kind,
            "message": message,
            "files": sorted(set(files)),
            "severity": severity,
        }
        if identifiers:
            issue["identifiers"] = sorted(set(identifiers))
        issues.append(issue)

    try:
        html_info = _collect_html_ids_classes(html_paths, root)
        css_info = _collect_css_ids_classes(css_paths, root)
        js_info = _collect_js_ids_classes(js_paths, root)

        html_ids = html_info.get("ids", {})
        html_classes = html_info.get("classes", {})
        html_data_targets = html_info.get("data_targets", {})

        css_ids = css_info.get("ids", {})
        css_classes = css_info.get("classes", {})
        css_data_targets = css_info.get("data_targets", {})

        js_ids = js_info.get("ids", {})
        js_classes = js_info.get("classes", {})
        js_data_targets = js_info.get("data_targets", {})

        # 1. HTML ids without CSS selectors
        for ident, html_files in html_ids.items():
            if ident not in css_ids:
                _add_issue(
                    kind="html_id_without_css",
                    message=f"HTML id '{ident}' is used, but no matching CSS selector '#{ident}' was found.",
                    files=list(html_files),
                    identifiers=[f"#{ident}", ident],
                    severity="low",
                )

        # 2. HTML classes without CSS selectors
        for cls, html_files in html_classes.items():
            if cls not in css_classes:
                _add_issue(
                    kind="html_class_without_css",
                    message=f"HTML class '{cls}' is used, but no matching CSS selector '.{cls}' was found.",
                    files=list(html_files),
                    identifiers=[f".{cls}", cls],
                    severity="low",
                )

        # 3. JS ids referenced but missing in HTML
        for ident, js_files in js_ids.items():
            if ident not in html_ids:
                _add_issue(
                    kind="js_id_without_html",
                    message=f"JavaScript references DOM id '{ident}', but no matching HTML 'id' attribute was found.",
                    files=list(js_files),
                    identifiers=[ident, f"#{ident}"],
                    severity="medium",
                )

        # 4. JS classes referenced but missing in HTML
        for cls, js_files in js_classes.items():
            if cls not in html_classes:
                _add_issue(
                    kind="js_class_without_html",
                    message=f"JavaScript references CSS class '{cls}', but no matching HTML 'class' attribute was found.",
                    files=list(js_files),
                    identifiers=[cls, f".{cls}"],
                    severity="medium",
                )

        # 5. Data-aidev-target: CSS/JS literal references missing in HTML
        # Combine CSS and JS literal referenced targets
        referenced_targets: Dict[str, List[str]] = {}
        for target, files in css_data_targets.items():
            referenced_targets.setdefault(target, []).extend(list(files))
        for target, files in js_data_targets.items():
            referenced_targets.setdefault(target, []).extend(list(files))

        for target, ref_files in referenced_targets.items():
            if target not in html_data_targets:
                # Build a helpful message and identifiers
                identifier_selector = f'[data-aidev-target="{target}"]'
                message = (
                    f"Referenced data-aidev-target '{target}' ({identifier_selector}) was found in CSS/JS but no matching HTML element with data-aidev-target='{target}' was found."
                )
                _add_issue(
                    kind="data_aidev_target_missing_in_html",
                    message=message,
                    files=ref_files,
                    identifiers=[identifier_selector, target],
                    severity="medium",
                )

        # Note: JS files that use getAttribute('data-aidev-target') or dataset.aidevTarget
        # are recorded but we cannot infer literal values; do not emit missing-value diagnostics
        # for these usages. If desired, a separate info-level diagnostic could be added.

    except Exception:
        logging.exception("consistency checks failed")
        # Treat a failure of the checker itself as non-fatal but visible.
        _add_issue(
            kind="consistency_checker_error",
            message="The consistency checker itself failed; see logs for details.",
            files=[],
            identifiers=None,
            severity="high",
        )

    # Build summary counts using the (relative) file paths collected above.
    unique_html_files = {str(p.relative_to(root)) for p in html_paths}
    unique_css_files = {str(p.relative_to(root)) for p in css_paths}
    unique_js_files = {str(p.relative_to(root)) for p in js_paths}
    unique_files = unique_html_files | unique_css_files | unique_js_files

    total_issues = len(issues)
    ok = total_issues == 0
    issues_found = total_issues > 0

    summary = {
        "total_files_scanned": len(unique_files),
        "total_html_files": len(unique_html_files),
        "total_css_files": len(unique_css_files),
        "total_js_files": len(unique_js_files),
        "total_issues": total_issues,
        "ok": ok,
    }

    # diagnostics is the same list as the legacy "issues" but named for the CLI consumer
    diagnostics = issues

    return {
        # New programmatic keys
        "issues_found": issues_found,
        "summary": summary,
        "diagnostics": diagnostics,
        # Legacy keys kept for backward compatibility
        "ok": ok,
        "issues": issues,
    }


def run_ui_consistency_check(project_root: Path | str) -> Dict[str, Any]:
    """
    Run the canonical UI consistency check and return a stable, discoverable result.

    The returned dict has the top-level keys:
      - errors: List[Dict]   (diagnostics with severity 'high' or 'medium')
      - warnings: List[Dict] (diagnostics with severity 'low')
      - details: Dict        (the full raw result returned by run_consistency_checks)

    Severity-to-bucket mapping:
      - 'high' and 'medium' -> errors
      - 'low' -> warnings

    The function accepts either a pathlib.Path or a string for project_root. It does
    not perform any I/O at import-time and is safe to import from tooling scripts.
    """
    # Coerce to Path; Path() accepts Path or str. Let it raise on clearly invalid inputs.
    root = Path(project_root)

    # Call the full checker and preserve its raw result under 'details'.
    details = run_consistency_checks(root=root)

    # Prefer the programmatic diagnostics key; fall back to legacy 'issues' if needed.
    diagnostics = list(details.get("diagnostics") or details.get("issues") or [])

    errors = [d for d in diagnostics if d.get("severity") in ("high", "medium")]
    warnings = [d for d in diagnostics if d.get("severity") == "low"]

    return {"errors": errors, "warnings": warnings, "details": details}


# Export hint for callers/discovery
__all__ = [
    "run_consistency_checks",
    "run_ui_consistency_check",
]
