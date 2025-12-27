# aidev/io_utils.py
from __future__ import annotations

# fmt: off
import json
import logging
import os
import random
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict, Union

try:
    import chardet  # optional
except Exception:  # pragma: no cover
    chardet = None  # type: ignore

# Try to import the runtime path-safety helper. If it's not available,
# fall back to a conservative resolve+containment check implemented below.
try:
    import runtimes.path_safety as path_safety  # type: ignore
except Exception:  # pragma: no cover - runtime may not provide this module in tests
    path_safety = None  # type: ignore

# Try to import the structured logger; if not present, fall back to stdlib logging.
try:
    from aidev import logger as alogger  # type: ignore
except Exception:  # pragma: no cover - tests/runtime may not have new logger yet
    alogger = None  # type: ignore


def _emit_structured(level: str, op: str, meta: Dict[str, Any], msg: Optional[str] = None) -> None:
    """
    Emit a structured log record.

    - Always includes `op` in the emitted metadata (even when aidev.logger is used).
    - Never raises (best-effort diagnostics).
    """
    meta2: Dict[str, Any] = {}
    try:
        meta2.update(meta or {})
    except Exception:
        meta2 = {}

    meta2.setdefault("op", op)
    meta2.setdefault("module", __name__)

    payload: Dict[str, Any] = {"op": op, "meta": meta2}
    if msg:
        payload["msg"] = msg

    try:
        if alogger is not None:
            # Prefer structured logger; keep op visible regardless of signature.
            text = msg or op
            if level == "info":
                try:
                    alogger.info(text, meta=meta2)
                except Exception:
                    alogger.info(text)
            elif level == "warning":
                try:
                    alogger.warning(text, meta=meta2)
                except Exception:
                    alogger.warning(text)
            elif level == "error":
                try:
                    alogger.error(text, meta=meta2)
                except Exception:
                    alogger.error(text)
            else:
                try:
                    alogger.debug(text, meta=meta2)
                except Exception:
                    alogger.debug(text)
            return

        # stdlib fallback
        level_map = {"debug": 10, "info": 20, "warning": 30, "error": 40}
        ln = level_map.get(level, 20)
        logging.getLogger(__name__).log(ln, json.dumps(payload, separators=(",", ":")))
    except Exception:
        try:
            logging.getLogger(__name__).debug("Failed to emit structured log for op=%s", op)
        except Exception:
            pass


def _stable_json_text(obj: Any) -> str:
    """Serialize JSON deterministically for cache/artifact persistence.

    Policy:
      - sort_keys=True for stable key order
      - indent=2 for readability
      - ensure_ascii=False to preserve unicode
      - separators fixed to (",", ": ") to avoid platform/stdlib variability

    Note: If callers include nondeterministic fields (e.g., timestamps, random IDs), the
    resulting bytes will differ; such fields should be excluded or explicitly documented
    as allowed exceptions when testing determinism.
    """
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, separators=(",", ": "))


# Provide minimal runtime/type stubs for annotations used throughout this module.
# The real ProjectState / Orchestrator types are defined elsewhere in the application.
# Defining these here avoids lint/type errors and keeps runtime behavior safe; callers
# that rely on specific attributes (e.g., st.trace.write) should provide a proper
# ProjectState implementation at runtime.
try:
    # Prefer importing the real definitions if available in the package.
    from aidev.runtime_types import ProjectState, Orchestrator  # type: ignore
except Exception:  # pragma: no cover
    class ProjectState:  # pragma: no cover - lightweight stub for linters/runtime
        def __init__(self, *args, **kwargs):
            # trace attribute is used in this module; default to a stub that no-ops.
            class _TraceStub:
                def write(self, *a, **k):
                    return None
            self.trace = _TraceStub()

    class Orchestrator:  # pragma: no cover - lightweight stub
        def __init__(self, *args, **kwargs):
            # Minimal fields referenced by this module may be created by callers.
            pass


# ---------------------- Diff stats ----------------------

def diff_stats(old: str, new: str) -> Tuple[int, int, int]:
    """Return (#added, #removed, #replaced) line counts between two texts."""
    import difflib
    sm = difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines())
    added = removed = replaced = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            added += (j2 - j1)
        elif tag == "delete":
            removed += (i2 - i1)
        elif tag == "replace":
            rep = min(i2 - i1, j2 - j1)
            replaced += rep
            if (j2 - j1) > rep:
                added += (j2 - j1 - rep)
            if (i2 - i1) > rep:
                removed += (i2 - i1 - rep)
    return added, removed, replaced


# ---------------------- Encoding / newline helpers ----------------------

_BOM_UTF8 = b"\xef\xbb\xbf"

def _is_probably_binary(raw: bytes) -> bool:
    # Simple binary heuristic: any NUL byte or a very high ratio of non-text bytes
    if b"\x00" in raw:
        return True
    # Heuristic: if many bytes are outside common text ranges, assume binary
    textish = sum(1 for b in raw if 9 <= b <= 13 or 32 <= b <= 126 or b in (0x85, 0xA0) or b >= 0xC0)
    return len(raw) > 0 and (textish / max(1, len(raw))) < 0.85

def _detect_encoding(raw: bytes) -> Tuple[str, bool]:
    """Return (encoding, had_bom). Prefer utf-8/utf-8-sig; fallback via chardet."""
    if raw.startswith(_BOM_UTF8):
        return ("utf-8-sig", True)
    # assume utf-8 first
    try:
        raw.decode("utf-8")
        return ("utf-8", False)
    except Exception:
        pass
    if chardet:
        try:
            info = chardet.detect(raw)
            enc = info.get("encoding") or "utf-8"
            return (enc, False)
        except Exception:
            pass
    return ("utf-8", False)

def _detect_newline(raw: bytes) -> str:
    # Count CRLF vs LF
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n")
    if crlf > 0 and crlf >= (lf - crlf):
        return "\r\n"
    return "\n"

def _read_text_with_optional_detect(p: Path) -> str:
    """Read file as text, attempting encoding detection when available."""
    try:
        raw = p.read_bytes()
        enc, _had_bom = _detect_encoding(raw)
        return raw.decode(enc, errors="replace")
    except Exception:
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

def _read_text_and_meta_if_exists(p: Path) -> Tuple[str, Optional[str], Optional[str], bool, bytes]:
    """
    Return (text, encoding, newline, had_bom, raw_bytes). If file doesn't exist, text="", enc=None, nl=None, had_bom=False.
    """
    if not p.exists() or not p.is_file():
        return "", None, None, False, b""
    raw = p.read_bytes()
    enc, had_bom = _detect_encoding(raw)
    # binary guard still returns decoded best-effort (for diff previews), but caller should check separately
    try:
        text = raw.decode(enc, errors="replace")
    except Exception:
        text = ""
    nl = _detect_newline(raw)
    return text, enc, nl, had_bom, raw

def _normalize_newlines_for_edit(s: str) -> str:
    # Normalize to LF during transformations; we'll re-map to target newline on write
    return s.replace("\r\n", "\n").replace("\r", "\n")

def _normalize_edits_for_apply(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize edit objects for the IO/apply layer.

    Goals:
      - Drop null/blank patch fields so 'presence' checks are meaningful.
      - Map legacy 'patch' -> 'patch_unified' (only when non-empty).
      - Never manufacture fields; keep upstream schema ownership upstream.
    """
    out: List[Dict[str, Any]] = []
    for e in edits or []:
        if not isinstance(e, dict):
            continue
        ee = dict(e)

        # Legacy patch -> patch_unified (only if non-empty)
        if "patch_unified" not in ee and "patch" in ee:
            p = ee.get("patch")
            if isinstance(p, str) and p.strip():
                ee["patch_unified"] = p
            ee.pop("patch", None)

        # Drop empty/None patch_unified
        if "patch_unified" in ee:
            pu = ee.get("patch_unified")
            if pu is None or (isinstance(pu, str) and not pu.strip()):
                ee.pop("patch_unified", None)

        # Drop empty/None legacy patch (if it still exists somehow)
        if "patch" in ee:
            p = ee.get("patch")
            if p is None or (isinstance(p, str) and not p.strip()):
                ee.pop("patch", None)

        out.append(ee)

    return out

# ---------------------- Path safety helper ----------------------

def _resolve_safe_path(project_root: Path, rel_or_target: Union[str, Path]) -> Path:
    """
    Resolve rel_or_target to an absolute Path guaranteed to be inside project_root.

    Prefers runtimes.path_safety.resolve_safe_path when available, but is defensive
    about signature differences (some variants take (root, rel), others (rel, root)).
    Falls back to a strict resolve()+containment check.

    Raises ValueError if the resolved path is not inside project_root.
    """
    base = Path(project_root).resolve()
    candidate = rel_or_target

    # Prefer runtime helper if available, but be defensive about signature.
    if path_safety is not None and hasattr(path_safety, "resolve_safe_path"):
        fn = getattr(path_safety, "resolve_safe_path", None)
        if callable(fn):
            # Try common call orders:
            #  1) resolve_safe_path(rel_or_target, base)
            #  2) resolve_safe_path(base, rel_or_target)
            for args in ((candidate, base), (base, candidate)):
                try:
                    resolved = fn(*args)  # type: ignore[misc]
                    p = Path(resolved).resolve()
                    p.relative_to(base)  # containment check
                    return p
                except Exception:
                    continue
            # If runtime helper exists but we couldn't use it, fall through to safe fallback.

    # Fallback implementation: strict resolve + containment.
    rt = Path(candidate)
    resolved = rt.resolve() if rt.is_absolute() else (base / rt).resolve()
    try:
        resolved.relative_to(base)
    except Exception:
        raise ValueError(f"Refusing to write outside project_root: {resolved} (project_root={base})")
    return resolved


# ---------------------- Unified patch application ----------------------

class UnifiedPatchError(Exception):
    """
    Exception raised when applying a unified patch fails.

    Backwards-compatible subclass of Exception that exposes a stable `code` attribute
    to aid classification of failures by higher-level logic. The allowed codes are:
      - "invalid_diff": The patch is malformed / cannot be parsed (e.g. invalid hunk header,
        unexpected patch line, no hunks, produced no changes).
      - "context_mismatch": The patch could be parsed but its context or deletion lines
        didn't match the target file (e.g. context mismatch or deletion mismatch).
      - "unknown": Fallback for other/unspecified errors.

    The stringification of this exception (str(e)) returns the human-readable message
    to preserve compatibility with existing callers that inspect the message.
    """

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or "unknown"

    def __str__(self) -> str:  # pragma: no cover - trivial wrapper for compatibility
        return self.message


def apply_unified_patch(old_text: str, patch_text: str) -> str:
    """
    Apply a unified (git-style) diff to old_text and return the new text.

    Hardening vs the previous version:
    - Normalize line endings on both old_text and patch_text (CRLF/LF safe).
    - Track whether we actually saw/apply any hunks; if not, raise.
    - If the resulting text is effectively identical to the original (after
      normalization), raise instead of silently returning the old text.

    Fix A (NEW):
    - Fuzzy hunk placement: treat @@ header line numbers as hints, and if the hunk
      does not apply at that position, search for a position where ALL context
      and deletion lines match. This makes LLM-authored diffs resilient when
      their line numbers drift.

    On failure this routine raises UnifiedPatchError with a `code` attribute set to one of:
      - 'invalid_diff' for parse/format/no-hunk/no-op style errors
      - 'context_mismatch' for context/deletion mismatches
      - 'unknown' for other errors

    Instrumentation:
      - Always computes stable sha256 (12 hex chars) + repr excerpts for old/patch inputs.
      - Emits:
          * "patch_apply_inputs" when AIDEV_PATCH_DEBUG=1
          * "patch_apply_failed" / "patch_apply_exception" on errors
    """
    # Normalize both sides to LF so CRLF/LF mismatches don't break matching.
    old_norm = _normalize_newlines_for_edit(old_text)
    patch_norm = _normalize_newlines_for_edit(patch_text)
    plines = patch_norm.splitlines()

    # ---- DEFINITIVE PATCH INSTRUMENTATION (hashes + repr) ----
    def _sha256_12(s: str) -> str:
        try:
            return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:12]
        except Exception:
            return "sha_err"

    def _repr_excerpt(s: str, n: int = 400) -> str:
        try:
            return repr(s[:n])
        except Exception:
            return "<repr_err>"

    diag = {
        "old_len": len(old_norm),
        "old_sha12": _sha256_12(old_norm),
        "patch_len": len(patch_norm),
        "patch_sha12": _sha256_12(patch_norm),
        # helpful for suspected underscore corruption
        "patch_has___init__": ("__init__" in patch_norm),
        "patch_double_underscore_count": patch_norm.count("__"),
        "patch_excerpt_repr": _repr_excerpt(patch_norm, 500),
    }

    # Optional: log inputs only when explicitly enabled (avoids noise).
    if os.getenv("AIDEV_PATCH_DEBUG", "").strip() == "1":
        _emit_structured("info", "patch_apply_inputs", dict(diag))

    # Track where we were if we fail mid-parse.
    last_i = -1
    last_line = ""
    # ---------------------------------------------------------

    try:
        # discard common file header lines '---', '+++', and other git/unified headers
        i = 0
        while i < len(plines) and (
            plines[i].startswith("--- ")
            or plines[i].startswith("+++ ")
            or plines[i].startswith("diff ")
            or plines[i].startswith("index ")
            or plines[i].startswith("*** ")
        ):
            i += 1

        old_lines = old_norm.splitlines()  # no keepends; we add newlines when joining
        new_lines: List[str] = []
        cur_old = 0  # 0-based index into old_lines
        hunks_seen = 0

        def _parse_hunk_header(h: str) -> Tuple[int, int, int, int]:
            # @@ -l,s +l2,s2 @@
            # s (lengths) may be omitted -> default 1
            import re as _re
            m = _re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", h)
            if not m:
                raise UnifiedPatchError(f"Invalid hunk header: {h!r}", code="invalid_diff")
            l1 = int(m.group(1))
            s1 = int(m.group(2) or 1)
            l2 = int(m.group(3))
            s2 = int(m.group(4) or 1)
            return l1, s1, l2, s2

        def _collect_hunk_ops(start_idx: int) -> Tuple[List[Tuple[str, str]], int]:
            """
            Collect (op, text) pairs for a hunk body until the next @@ header or EOF.
            ops op is one of: " " (context), "+" (add), "-" (delete).
            Returns (ops, next_idx).
            """
            ops: List[Tuple[str, str]] = []
            j = start_idx
            while j < len(plines) and not plines[j].startswith("@@ "):
                ln = plines[j]

                if ln.startswith("\\ No newline at end of file"):
                    j += 1
                    continue

                # allow some noise lines inside patch
                if ln.startswith("diff ") or ln.startswith("index ") or ln.startswith("*** "):
                    j += 1
                    continue

                if ln.startswith("--- ") or ln.startswith("+++ "):
                    break

                if ln.startswith((" ", "+", "-")):
                    ops.append((ln[0], ln[1:]))
                    j += 1
                    continue

                if ln == "@@":
                    raise UnifiedPatchError(
                        "Invalid diff: found bare '@@' placeholder line (not a hunk header).",
                        code="invalid_diff",
                    )

                raise UnifiedPatchError(f"Unexpected patch line: {ln!r}", code="invalid_diff")

            return ops, j

        def _hunk_can_apply_at(pos: int, ops: List[Tuple[str, str]]) -> bool:
            """
            Return True if all context and deletion lines match starting at pos.
            Additions do not consume old lines.
            """
            k = pos
            for op, txt in ops:
                if op == " ":
                    if k >= len(old_lines) or old_lines[k] != txt:
                        return False
                    k += 1
                elif op == "-":
                    if k >= len(old_lines) or old_lines[k] != txt:
                        return False
                    k += 1
                elif op == "+":
                    continue
                else:
                    return False
            return True

        def _find_hunk_start(cur_old_idx: int, hint_idx: int, ops: List[Tuple[str, str]]) -> Optional[int]:
            """
            Fuzzy find a start position for the hunk:
              1) try the header hint (clamped)
              2) scan within +/- window around hint
              3) scan forward from cur_old_idx to EOF

            Requires at least one anchored line (context or deletion).
            """
            anchored = any(op in (" ", "-") for op, _ in ops)
            if not anchored:
                raise UnifiedPatchError(
                    "Invalid diff: hunk has no context/deletion lines to anchor placement.",
                    code="invalid_diff",
                )

            # clamp hint within bounds and never before cur_old_idx
            hint = max(cur_old_idx, min(len(old_lines), max(0, hint_idx)))

            # 1) exact hint
            if _hunk_can_apply_at(hint, ops):
                return hint

            # 2) window scan around hint (prefer closest match)
            try:
                win = int(os.getenv("AIDEV_PATCH_FUZZY_WINDOW", "").strip() or "200")
            except Exception:
                win = 200

            lo = max(cur_old_idx, hint - win)
            hi = min(len(old_lines), hint + win)

            for delta in range(1, win + 1):
                left = hint - delta
                if left >= lo and _hunk_can_apply_at(left, ops):
                    return left
                right = hint + delta
                if right <= hi and _hunk_can_apply_at(right, ops):
                    return right

            # 3) forward scan (git-like resilience)
            for pos in range(cur_old_idx, len(old_lines) + 1):
                if _hunk_can_apply_at(pos, ops):
                    return pos

            return None

        while i < len(plines):
            line = plines[i]
            last_i = i
            last_line = line

            if not line.startswith("@@ "):
                i += 1
                continue

            hunks_seen += 1

            l1, s1, l2, s2 = _parse_hunk_header(line)
            i += 1

            ops, next_i = _collect_hunk_ops(i)

            # Find where this hunk applies (fuzzy).
            hint_old_idx = max(0, l1 - 1)
            start_at = _find_hunk_start(cur_old, hint_old_idx, ops)
            if start_at is None:
                raise UnifiedPatchError("Context mismatch while applying patch", code="context_mismatch")

            # Copy unchanged lines before the hunk start.
            if start_at > cur_old:
                new_lines.extend(old_lines[cur_old:start_at])
                cur_old = start_at

            # Apply hunk ops (strict now, since we chose a matching start).
            for op, txt in ops:
                if op == " ":
                    if cur_old >= len(old_lines) or old_lines[cur_old] != txt:
                        raise UnifiedPatchError("Context mismatch while applying patch", code="context_mismatch")
                    new_lines.append(txt)
                    cur_old += 1
                elif op == "+":
                    new_lines.append(txt)
                elif op == "-":
                    if cur_old >= len(old_lines) or old_lines[cur_old] != txt:
                        raise UnifiedPatchError("Deletion mismatch while applying patch", code="context_mismatch")
                    cur_old += 1
                else:
                    raise UnifiedPatchError(f"Unexpected patch op: {op!r}", code="invalid_diff")

            i = next_i  # jump to next hunk/header

        if hunks_seen == 0:
            raise UnifiedPatchError(
                "Unified diff contained no hunks; refusing to treat as a no-op.",
                code="invalid_diff",
            )

        # copy remainder
        if cur_old < len(old_lines):
            new_lines.extend(old_lines[cur_old:])

        new_text = "\n".join(new_lines)

        # If, after normalization, nothing changed, treat that as an error so we don't
        # silently "succeed" when in reality the patch didn't apply.
        if _normalize_newlines_for_edit(new_text) == old_norm:
            raise UnifiedPatchError(
                "Unified patch produced no changes; refusing to apply as a no-op.",
                code="invalid_diff",
            )

        return new_text

    except UnifiedPatchError as ue:
        _emit_structured(
            "error",
            "patch_apply_failed",
            {
                **diag,
                "code": getattr(ue, "code", "unknown"),
                "error": str(ue),
                "last_i": last_i,
                "last_line_repr": _repr_excerpt(last_line, 300),
            },
        )
        raise

    except Exception as e:
        _emit_structured(
            "error",
            "patch_apply_exception",
            {
                **diag,
                "error": str(e),
                "last_i": last_i,
                "last_line_repr": _repr_excerpt(last_line, 300),
            },
        )
        raise


# ---------------------- Unified diff generation ----------------------

def generate_unified_diff(fromfile: str, tofile: str, old_text: str, new_text: str) -> str:
    """
    Generate a git-style unified diff between old_text and new_text, with file labels.
    Uses LF-only line endings in the diff text.
    """
    import difflib as _difflib
    a = (old_text or "").splitlines(keepends=True)
    b = (new_text or "").splitlines(keepends=True)
    return "".join(_difflib.unified_diff(a, b, fromfile=fromfile, tofile=tofile))


# ---------------------- FS helpers ----------------------

def ensure_dir(p: Path) -> None:
    """
    Ensure directory `p` exists. If a file exists at `p`, remove it first, then mkdir -p.
    """
    if p is None:
        return
    p = Path(p)
    if p.exists():
        if p.is_file():
            p.unlink()
        else:
            return
    p.mkdir(parents=True, exist_ok=True)


# ---------------------- Logged writers ----------------------

def write_text_logged(
    path: Path,
    content: str,
    *,
    project_root: Path,
    st: "ProjectState" | None = None,
    stats: "Orchestrator" | None = None,
    rec_id: str | None = None,
    preserve_newlines_from: Optional[Path] = None,
    preserve_encoding_from: Optional[Path] = None,
) -> None:
    """
    Write text atomically and log a compact diff. Also appends a trace record if `st` is provided.

    Preserves newline style and UTF-8 BOM based on:
      - newline source: preserve_newlines_from (if provided) else target
      - encoding/BOM source: preserve_encoding_from (if provided) else target

    Diff is always computed against the actual target's current content (not preserve_*).
    """
    target = _resolve_safe_path(project_root, path)

    existed_before = target.exists()

    # Read actual current target for diff (always).
    old_text, _t_enc, _t_nl, _t_bom, _raw_target = _read_text_and_meta_if_exists(target)

    # Determine newline style from preserve_newlines_from (if provided) else target.
    nl_src = preserve_newlines_from or target
    try:
        nl_src = _resolve_safe_path(project_root, nl_src)
    except Exception:
        nl_src = target
    _nl_text, _nl_enc, nl_style, _nl_bom, _nl_raw = _read_text_and_meta_if_exists(nl_src)

    # Determine BOM/encoding from preserve_encoding_from (if provided) else target.
    enc_src = preserve_encoding_from or target
    try:
        enc_src = _resolve_safe_path(project_root, enc_src)
    except Exception:
        enc_src = target
    _enc_text, _enc_enc, _enc_nl, had_bom, _enc_raw = _read_text_and_meta_if_exists(enc_src)

    # Apply cleanup policies
    def _cleanup(s: str) -> str:
        s = _normalize_newlines_for_edit(s)
        if stats is not None and getattr(stats, "_cleanup_remove_empty", False):
            s = "\n".join(ln for ln in s.split("\n") if ln.strip() != "")
        elif stats is not None and getattr(stats, "_cleanup_collapse_blank", False):
            out: List[str] = []
            prev_blank = False
            for ln in s.split("\n"):
                blank = (ln.strip() == "")
                if blank and prev_blank:
                    continue
                out.append(ln)
                prev_blank = blank
            s = "\n".join(out)
        return s

    content_norm = _cleanup(content)

    # Skip true no-ops (avoids pointless writes + churn)
    if _normalize_newlines_for_edit(old_text) == content_norm:
        rel = str(target.relative_to(Path(project_root).resolve()))
        _emit_structured(
            "info",
            "write_noop_skip",
            {"path": rel, "rec_id": rec_id, "project_root": str(Path(project_root).resolve())},
            msg=f"No-op: {rel}",
        )
        try:
            if stats is not None:
                stats._files_skipped += 1  # type: ignore[attr-defined]
        except Exception:
            pass
        return

    added, removed, replaced = diff_stats(_normalize_newlines_for_edit(old_text), content_norm)
    total_lines = len(content_norm.splitlines())

    nl = nl_style or "\n"
    content_to_write = content_norm.replace("\n", nl) if nl != "\n" else content_norm

    enc = "utf-8-sig" if had_bom else "utf-8"

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.aidev_tmp_{os.getpid()}_{random.randint(1000,9999)}")
    with open(tmp, "w", encoding=enc, newline="") as f:
        f.write(content_to_write)
    os.replace(tmp, target)

    try:
        bytes_written = os.path.getsize(target)
    except Exception:
        try:
            bytes_written = len(content_to_write.encode(enc or "utf-8"))
        except Exception:
            bytes_written = 0

    rel = str(target.relative_to(Path(project_root).resolve()))

    _emit_structured(
        "info",
        "write",
        {
            "path": rel,
            "bytes_written": bytes_written,
            "encoding": enc,
            "had_bom": bool(had_bom),
            "newline": nl,
            "added": added,
            "removed": removed,
            "replaced": replaced,
            "lines": total_lines,
            "unchanged": (added == 0 and removed == 0 and replaced == 0),
            "rec_id": rec_id,
            "project_root": str(Path(project_root).resolve()),
        },
        msg=f"Wrote: {rel}",
    )

    try:
        if st is not None:
            st.trace.write(
                "write",
                "write_text",
                {
                    "path": rel,
                    "added": added,
                    "removed": removed,
                    "replaced": replaced,
                    "lines": total_lines,
                    "unchanged": (added == 0 and removed == 0 and replaced == 0),
                    "rec_id": rec_id,
                    "project_root": str(Path(project_root).resolve()),
                },
            )
    except Exception:
        pass

    try:
        if stats is not None:
            if not hasattr(stats, "_file_change_summaries"):
                stats._file_change_summaries = []  # type: ignore[attr-defined]
            if not hasattr(stats, "_writes_by_rec"):
                stats._writes_by_rec = {}  # type: ignore[attr-defined]
            stats._file_change_summaries.append({  # type: ignore[attr-defined]
                "path": str(target),
                "added": added,
                "removed": removed,
                "replaced": replaced,
                "lines": total_lines,
                "unchanged": (added == 0 and removed == 0 and replaced == 0),
            })
            if rec_id:
                stats._writes_by_rec.setdefault(rec_id, []).append(str(target))  # type: ignore[attr-defined]

            if (added == 0 and removed == 0 and replaced == 0):
                stats._files_skipped += 1  # type: ignore[attr-defined]
            else:
                if existed_before:
                    stats._files_modified += 1  # type: ignore[attr-defined]
                else:
                    stats._files_created += 1  # type: ignore[attr-defined]
    except Exception:
        pass


def write_json_logged(
    path: Path,
    obj: dict | list,
    *,
    project_root: Path,
    st: "ProjectState" | None = None,
    stats: "Orchestrator" | None = None,
    rec_id: str | None = None,
) -> None:
    text = _stable_json_text(obj)
    write_text_logged(path, text, project_root=project_root, st=st, stats=stats, rec_id=rec_id)


def save_text(project_root: Path, p: Path, text: str) -> None:
    """
    Save text to `p`, but only if `p` resolves inside `project_root`.
    This function is root-locked for safety; callers must provide project_root.
    """
    target = _resolve_safe_path(project_root, p)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", errors="replace")


def save_json(project_root: Path, p: Path, data: dict) -> None:
    target = _resolve_safe_path(project_root, p)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_stable_json_text(data), encoding="utf-8")


def _read_file_text_if_exists(path_or_root: Path, rel_path: Optional[str] = None) -> str:
    """
    Read a file if it exists.
    - If called as _read_file_text_if_exists(Path('/abs/file.txt')) it reads that file.
    - If called as _read_file_text_if_exists(root, 'rel/path') it reads root/rel/path.
    """
    p = path_or_root if rel_path is None else (path_or_root / rel_path)
    p = Path(p)
    if p.exists() and p.is_file():
        try:
            return _read_text_with_optional_detect(p)
        except Exception:
            return ""
    return ""


# ---------------------- Transactional writes ----------------------

@dataclass
class WriteTransaction:
    """
    Stages writes under .aidev/staged/<session> and commits them atomically
    per-file (os.replace). On failure, attempts rollback via backups.
    """
    root: Path
    st: "ProjectState" | None = None

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.stage_root = self.root / ".aidev" / "staged" / f"txn_{os.getpid()}_{random.randint(1000,9999)}"
        self.backup_root = self.stage_root / "_backups"
        self._staged: List[Tuple[Path, Path]] = []     # (staged_path, target_path)
        self._replaced: List[Tuple[Optional[Path], Path]] = []  # (backup_path_or_none, target_path)
        self._staged_index: Dict[Path, int] = {}        # target_path -> index into _staged
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        _emit_structured(
            "info",
            "txn_create",
            {"root": str(self.root), "stage_root": str(self.stage_root)},
            msg="WriteTransaction created",
        )

    # --- core API

    def stage_write(self, rel_path: str, content: str, *, keep_bom: bool, newline: str) -> None:
        # Resolve and validate the target path inside the transaction's root.
        target = _resolve_safe_path(self.root, rel_path)

        # staged location should mirror the path relative to the project root
        rel = target.relative_to(self.root)
        staged = self.stage_root / rel
        staged.parent.mkdir(parents=True, exist_ok=True)
        enc = "utf-8-sig" if keep_bom else "utf-8"

        # Normalize content to the requested newline style once
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)

        existing_idx = self._staged_index.get(target)
        if existing_idx is not None:
            # We already staged this target in this transaction: overwrite the prior
            # staged file and update the tuple instead of appending a duplicate.
            old_staged, _old_target = self._staged[existing_idx]
            try:
                if old_staged.exists():
                    old_staged.unlink()
            except Exception:
                # Best-effort cleanup; we will overwrite `staged` below anyway.
                pass

            with open(staged, "w", encoding=enc, newline="") as f:
                f.write(normalized)

            self._staged[existing_idx] = (staged, target)
            _emit_structured(
                "info",
                "restage",
                {
                    "staged": str(staged),
                    "target": str(target),
                    "keep_bom": keep_bom,
                    "newline": newline,
                },
                msg=f"Restaged {staged} -> {target}",
            )
        else:
            # First time staging this target in this transaction.
            with open(staged, "w", encoding=enc, newline="") as f:
                f.write(normalized)
            self._staged.append((staged, target))
            self._staged_index[target] = len(self._staged) - 1
            _emit_structured(
                "info",
                "stage_write",
                {"staged": str(staged), "target": str(target), "keep_bom": keep_bom, "newline": newline},
                msg=f"Staged {staged} -> {target}",
            )

    def commit(self) -> None:
        _emit_structured("info", "txn_commit_start", {"staged_count": len(self._staged)}, msg="commit start")
        try:
            for staged, target in self._staged:
                exists = staged.exists()
                _emit_structured(
                    "info",
                    "txn_commit_file",
                    {"staged": str(staged), "staged_exists": exists, "target": str(target)},
                )
                if not exists:
                    # Hard invariant violation: staged file vanished before commit.
                    raise RuntimeError(
                        f"Staged file missing at commit time: {staged} -> {target}. "
                        "This indicates a bug (cleanup or race) elsewhere."
                    )

                target.parent.mkdir(parents=True, exist_ok=True)
                # backup current target (if exists)
                backup_path: Optional[Path] = None
                if target.exists():
                    backup_path = self.backup_root / target.relative_to(self.root)
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup_path)
                # replace
                os.replace(staged, target)
                self._replaced.append((backup_path, target))
        except Exception as e:
            # Keep the exception trace as before but also emit a structured error.
            logging.exception("Commit failed; attempting rollback.")
            _emit_structured("error", "txn_commit_failed", {"error": str(e)})
            self.rollback()
            raise e
        finally:
            # cleanup stage files that weren't moved
            for staged, _ in self._staged:
                try:
                    if staged.exists():
                        staged.unlink()
                except Exception:
                    pass
            # remove empty stage dirs
            try:
                shutil.rmtree(self.stage_root, ignore_errors=True)
            except Exception:
                pass
            _emit_structured("info", "txn_commit_done", {"replaced_count": len(self._replaced)})

    def rollback(self) -> None:
        # Structured notice about rollback start; list attempted targets for clarity.
        try:
            _emit_structured("warning", "txn_rollback", {"attempted_targets": [str(t) for _, t in self._replaced]})
        except Exception:
            pass
        # Restore backups for files that were replaced; remove new files without backup
        for backup_path, target in reversed(self._replaced):
            try:
                if backup_path and backup_path.exists():
                    os.replace(backup_path, target)
                else:
                    if target.exists():
                        target.unlink()
            except Exception:
                logging.debug("Rollback failed for: %s", target)
        # Clean staged leftovers
        try:
            shutil.rmtree(self.stage_root, ignore_errors=True)
        except Exception:
            pass

    # --- context manager

    def __enter__(self) -> "WriteTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.commit()
        else:
            self.rollback()

def _coalesce_edits_by_path(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Coalesce multiple edits targeting the same path into one edit (last wins).
    """
    coalesced: Dict[str, Dict[str, Any]] = {}
    for e in edits:
        rel = str(e.get("path") or "").strip()
        if not rel:
            continue
        if rel in coalesced:
            _emit_structured("warning", "apply_edits_duplicate_path", {"path": rel}, msg=f"Duplicate edit; last wins: {rel}")
        coalesced[rel] = e
    # Deterministic order: preserve last-write-wins but output sorted by path for stable logs.
    return [coalesced[k] for k in sorted(coalesced.keys())]

def _validate_edits(edits: List[Dict[str, Any]]) -> None:
    """
    Fail-fast validation of edit objects before applying them.

    Rules:
      - non-empty 'path'
      - 'content' if present must be a string (empty string allowed)
      - 'patch_unified' if present must be a non-empty string and contain >=1 valid @@ hunk header
      - at least one of: content or patch_unified must be provided
      - allow BOTH content + patch_unified (apply tries patch first, fallback to content if patch fails)

    Tolerant of schema-emitted nullable keys:
      - patch_unified=None or patch_unified="" is treated as absent
    """
    import re as _re

    hunk_re = _re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    for e in edits:
        if not isinstance(e, dict):
            raise ValueError(f"Edit entry is not a dict: {type(e)}")

        rel = str(e.get("path") or "").strip()
        if not rel:
            raise ValueError(f"Edit entry missing 'path': {e}")

        # content validation (allow empty string)
        has_content = ("content" in e and e.get("content") is not None)
        if has_content and not isinstance(e.get("content"), str):
            raise ValueError(f"Edit for {rel} has non-string content: {type(e.get('content'))}")

        # patch validation (tolerate None/blank as absent)
        patch_val = e.get("patch_unified")
        has_patch = isinstance(patch_val, str) and bool(patch_val.strip())

        if has_patch:
            patch_str: str = patch_val  # type: ignore[assignment]

            # Reject any bare placeholder '@@' line
            for ln in patch_str.splitlines():
                if ln.strip() == "@@":
                    raise ValueError(f"Edit for {rel} contains an invalid bare '@@' placeholder line.")

            # Reject any '@@' line that isn't a valid hunk header
            bad_hunk_lines = []
            has_any_hunk = False
            for ln in patch_str.splitlines():
                if ln.startswith("@@"):
                    if hunk_re.match(ln):
                        has_any_hunk = True
                    else:
                        bad_hunk_lines.append(ln)

            if bad_hunk_lines:
                raise ValueError(f"Edit for {rel} contains invalid hunk header lines: {bad_hunk_lines[:3]!r}")

            if not has_any_hunk:
                _emit_structured(
                    "warning",
                    "patch_rejected_header_only",
                    {
                        "path": rel,
                        "rec_id": e.get("rec_id"),
                        "reason": "no_valid_hunk_headers",
                        "source_field": "patch_unified",
                    },
                    msg=f"Rejecting patch with no valid hunks for {rel}",
                )
                raise ValueError(f"Edit for {rel} contains a patch_unified with no valid hunks; rejecting as invalid.")

        if not has_content and not has_patch:
            raise ValueError(f"Edit for {rel} must specify either 'content' or a non-empty 'patch_unified' field.")

def apply_edits_transactionally(
    root: Path,
    edits: List[Dict[str, Any]],
    *,
    dry_run: bool,
    stats: "Orchestrator" | None,
    st: "ProjectState" | None,
    allow_binary: bool = False,
    project_root: Optional[Path] = None,
) -> None:
    """
    Apply a list of edit objects (JSONL items) in one transaction.
    """
    root = Path(root).resolve()

    # Resolve project_root safely:
    # - if absolute, use it
    # - if relative, resolve relative to `root`
    if project_root is not None:
        pr = Path(project_root)
        base = pr.resolve() if pr.is_absolute() else (root / pr).resolve()
    else:
        base = root

    _emit_structured("info", "apply_edits_start", {"dry_run": dry_run, "root": str(base), "raw_edits": len(edits)})

    edits = _normalize_edits_for_apply(edits)
    edits = _coalesce_edits_by_path(edits)
    _validate_edits(edits)

    _emit_structured("info", "apply_edits_coalesced", {"paths": [str(e.get("path")) for e in edits]})

    def _get_patch(e: Dict[str, Any]) -> str:
        p = e.get("patch_unified")
        return p if isinstance(p, str) else ""

    def _get_content(e: Dict[str, Any]) -> Optional[str]:
        c = e.get("content")
        return c if isinstance(c, str) else None

    # ---- DRY RUN ----
    if dry_run:
        for e in edits:
            rel = str(e.get("path") or "").strip()
            try:
                target = _resolve_safe_path(base, rel)
            except ValueError:
                _emit_structured("warning", "apply_edits_outside_root", {"path": rel, "root": str(base)})
                continue

            old_text, _enc, _nl, _had_bom, raw = _read_text_and_meta_if_exists(target)
            if raw and _is_probably_binary(raw) and not allow_binary:
                _emit_structured("warning", "apply_edits_binary_skip", {"path": rel})
                continue

            patch = _get_patch(e)
            content = _get_content(e)

            try:
                if patch:
                    try:
                        new_text = apply_unified_patch(old_text, patch)
                    except UnifiedPatchError as ue:
                        if content is not None:
                            _emit_structured(
                                "warning",
                                "patch_failed_fallback_to_content_dryrun",
                                {"path": rel, "error": str(ue), "code": getattr(ue, "code", "unknown")},
                            )
                            new_text = _normalize_newlines_for_edit(content)
                        else:
                            msg_l = str(ue).lower()
                            if getattr(ue, "code", "") == "invalid_diff" and (
                                "no hunks" in msg_l or "no changes" in msg_l or "no-op" in msg_l
                            ):
                                _emit_structured("warning", "patch_rejected_noop_dryrun", {"path": rel, "error": str(ue)})
                                continue
                            _emit_structured(
                                "error",
                                "patch_conflict_dryrun",
                                {"path": rel, "error": str(ue), "code": getattr(ue, "code", "unknown")},
                            )
                            continue
                else:
                    # content-only
                    if content is None:
                        _emit_structured("warning", "apply_edits_unknown_format", {"path": rel, "keys": list(e.keys())})
                        continue
                    new_text = _normalize_newlines_for_edit(content)
            except Exception as ex:
                _emit_structured("error", "dry_run_exception", {"path": rel, "error": str(ex)})
                continue

            old_norm = _normalize_newlines_for_edit(old_text)
            new_norm = _normalize_newlines_for_edit(new_text)

            if old_norm == new_norm:
                _emit_structured("info", "dry_run_noop", {"path": rel})
                continue

            a, r, rep = diff_stats(old_norm, new_norm)
            _emit_structured("info", "dry_run_preview", {"path": rel, "added": a, "removed": r, "replaced": rep})

        _emit_structured("info", "apply_edits_finished", {"dry_run": True})
        return

    # ---- REAL APPLY (transactional) ----
    pending_summaries: List[Dict[str, Any]] = []
    pending_by_rec: Dict[str, List[str]] = {}
    pending_created = 0
    pending_modified = 0
    pending_skipped = 0

    with WriteTransaction(base, st=st) as txn:
        for e in edits:
            rel = str(e.get("path") or "").strip()
            _emit_structured("info", "apply_edits_consider", {"path": rel, "rec_id": e.get("rec_id")})

            try:
                target = _resolve_safe_path(base, rel)
            except ValueError:
                _emit_structured("warning", "apply_edits_outside_root", {"path": rel, "root": str(base)})
                pending_skipped += 1
                continue

            old_text, _old_enc, old_nl, had_bom, raw = _read_text_and_meta_if_exists(target)
            existed_before = target.exists()

            if raw and _is_probably_binary(raw) and not allow_binary:
                _emit_structured("warning", "apply_edits_binary_skip", {"path": rel, "reason": "binary"})
                pending_skipped += 1
                continue

            patch = _get_patch(e)
            content = _get_content(e)

            # Determine new text
            if patch:
                try:
                    new_text = apply_unified_patch(old_text, patch)
                except UnifiedPatchError as ue:
                    if content is not None:
                        _emit_structured(
                            "warning",
                            "patch_failed_fallback_to_content",
                            {"path": rel, "error": str(ue), "code": getattr(ue, "code", "unknown")},
                        )
                        new_text = _normalize_newlines_for_edit(content)
                    else:
                        msg_l = str(ue).lower()
                        if getattr(ue, "code", "") == "invalid_diff" and (
                            "no hunks" in msg_l or "no changes" in msg_l or "no-op" in msg_l
                        ):
                            _emit_structured("warning", "patch_rejected_noop", {"path": rel, "error": str(ue)})
                            pending_skipped += 1
                            continue
                        _emit_structured(
                            "error",
                            "patch_conflict",
                            {"path": rel, "error": str(ue), "code": getattr(ue, "code", "unknown")},
                        )
                        raise
            else:
                if content is None:
                    _emit_structured("warning", "apply_edits_unknown_format", {"path": rel, "keys": list(e.keys())})
                    pending_skipped += 1
                    continue
                new_text = _normalize_newlines_for_edit(content)

            # Skip true no-ops to avoid churn + bogus "modified/created"
            old_norm = _normalize_newlines_for_edit(old_text)
            new_norm = _normalize_newlines_for_edit(new_text)

            if old_norm == new_norm:
                _emit_structured("info", "apply_edits_noop_skip", {"path": rel, "rec_id": e.get("rec_id")})
                pending_skipped += 1
                continue

            # Stage write preserving existing newline and BOM
            keep_bom = bool(had_bom)
            newline = old_nl or "\n"
            txn.stage_write(rel, new_norm, keep_bom=keep_bom, newline=newline)

            a, r, rep = diff_stats(old_norm, new_norm)
            pending_summaries.append(
                {
                    "path": str(target),
                    "added": a,
                    "removed": r,
                    "replaced": rep,
                    "lines": len(new_norm.splitlines()),
                    "unchanged": (a == 0 and r == 0 and rep == 0),
                }
            )

            rid = e.get("rec_id")
            if isinstance(rid, str) and rid:
                pending_by_rec.setdefault(rid, []).append(str(target))

            if existed_before:
                pending_modified += 1
            else:
                pending_created += 1

            try:
                if st is not None:
                    st.trace.write(
                        "write_staged",
                        "edit",
                        {
                            "path": str(target.relative_to(base)),
                            "added": a,
                            "removed": r,
                            "replaced": rep,
                            "rec_id": e.get("rec_id"),
                            "project_root": str(base),
                        },
                    )
            except Exception:
                pass

    # Commit succeeded if we got here (context manager exited cleanly).
    try:
        if stats is not None:
            if not hasattr(stats, "_file_change_summaries"):
                stats._file_change_summaries = []  # type: ignore[attr-defined]
            if not hasattr(stats, "_writes_by_rec"):
                stats._writes_by_rec = {}  # type: ignore[attr-defined]
            stats._file_change_summaries.extend(pending_summaries)  # type: ignore[attr-defined]
            for rid, paths in pending_by_rec.items():
                stats._writes_by_rec.setdefault(rid, []).extend(paths)  # type: ignore[attr-defined]

            if not hasattr(stats, "_files_created"):
                stats._files_created = 0  # type: ignore[attr-defined]
            if not hasattr(stats, "_files_modified"):
                stats._files_modified = 0  # type: ignore[attr-defined]
            if not hasattr(stats, "_files_skipped"):
                stats._files_skipped = 0  # type: ignore[attr-defined]

            stats._files_created += pending_created  # type: ignore[attr-defined]
            stats._files_modified += pending_modified  # type: ignore[attr-defined]
            stats._files_skipped += pending_skipped  # type: ignore[attr-defined]
    except Exception:
        pass

    _emit_structured(
        "info",
        "apply_edits_finished",
        {
            "dry_run": False,
            "files_created": pending_created,
            "files_modified": pending_modified,
            "files_skipped": pending_skipped,
        },
    )

# ---------------------- Back-compat writer ----------------------

def write_code_under_root(
    root: Path,
    rel_path: str,
    content: str,
    *,
    dry_run: bool,
    stats: "Orchestrator" | None,
    st: "ProjectState" | None,
    rec_id: str | None,
    tx: WriteTransaction | None = None,
    allow_binary: bool = False,
) -> None:
    """
    Backwards-compatible single-file write with optional staging.
    If `tx` is provided, the write is staged and committed with the transaction.
    Otherwise, it is atomically replaced in place (per-file).

    NOTE: `root` is treated as the project_root and is enforced via path-safety.
    """
    try:
        target = _resolve_safe_path(root, rel_path)
    except ValueError:
        _emit_structured("warning", "write_code_refuse", {"path": rel_path})
        if stats:
            try:
                stats._files_skipped += 1  # type: ignore[attr-defined]
            except Exception:
                pass
        return

    # Binary guard
    if target.exists():
        raw = target.read_bytes()
        if _is_probably_binary(raw) and not allow_binary:
            _emit_structured("warning", "write_code_binary_skip", {"path": rel_path, "allow_binary": allow_binary})
            return

    if dry_run:
        old = _read_text_with_optional_detect(target) if target.exists() else ""
        added, removed, replaced = diff_stats(_normalize_newlines_for_edit(old), _normalize_newlines_for_edit(content))
        _emit_structured(
            "info",
            "write_code_dry_run",
            {"path": rel_path, "added": added, "removed": removed, "replaced": replaced, "dry_run": True},
        )
        return

    if tx is not None:
        # Stage and let caller commit/rollback
        old_text, _enc, old_nl, had_bom, _raw = _read_text_and_meta_if_exists(target)
        keep_bom = bool(had_bom)
        newline = old_nl or "\n"
        tx.stage_write(rel_path, _normalize_newlines_for_edit(content), keep_bom=keep_bom, newline=newline)
        # Also emit structured staged write so summary counters still work after commit
        try:
            a, r, rep = diff_stats(_normalize_newlines_for_edit(old_text), _normalize_newlines_for_edit(content))
            _emit_structured(
                "info",
                "write_code_staged",
                {"path": str(target), "added": a, "removed": r, "replaced": rep, "rec_id": rec_id},
            )
            if st is not None:
                st.trace.write("write_staged", "write_text", {"path": str(target), "added": a, "removed": r, "replaced": rep, "rec_id": rec_id, "project_root": str(Path(root).resolve())})
        except Exception:
            pass
        return

    # Immediate atomic write
    write_text_logged(
        target,
        _normalize_newlines_for_edit(content),
        project_root=root,
        st=st,
        stats=stats,
        rec_id=rec_id,
        preserve_encoding_from=target if target.exists() else None,
        preserve_newlines_from=target if target.exists() else None,
    )


# ---------------------- DeepResearchEngine cache helpers ----------------------

def write_json_cache(
    project_root: Path,
    rel_path: Union[str, Path],
    obj: Any,
    *,
    st: "ProjectState" | None = None,
    stats: "Orchestrator" | None = None,
    rec_id: str | None = None,
) -> None:
    """
    Atomically write a JSON artifact intended for use as a cache entry by
    DeepResearchEngine or similar components. The target path is validated
    to be inside `project_root`.

    This function writes UTF-8 JSON using os.replace for atomicity and emits
    structured logs and optional trace records for observability.
    """
    try:
        target = _resolve_safe_path(project_root, rel_path)
    except ValueError:
        _emit_structured("warning", "cache_write_refuse", {"path": str(rel_path)})
        if stats:
            try:
                stats._files_skipped += 1  # type: ignore[attr-defined]
            except Exception:
                pass
        return

    text = _stable_json_text(obj)
    enc = "utf-8"

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.aidev_cache_tmp_{os.getpid()}_{random.randint(1000,9999)}")
    try:
        with open(tmp, "w", encoding=enc, newline="") as f:
            f.write(text)
        os.replace(tmp, target)

        try:
            bytes_written = os.path.getsize(target)
        except Exception:
            try:
                bytes_written = len(text.encode(enc))
            except Exception:
                bytes_written = 0

        rel = str(target.relative_to(Path(project_root).resolve()))
        _emit_structured(
            "info",
            "cache_write",
            {
                "path": rel,
                "bytes_written": bytes_written,
                "encoding": enc,
                "rec_id": rec_id,
                "project_root": str(Path(project_root).resolve()),
            },
            msg=f"Wrote cache: {rel}",
        )

        if st is not None:
            try:
                st.trace.write("cache_write", "write_json_cache", {"path": rel, "bytes_written": bytes_written, "rec_id": rec_id})
            except Exception:
                pass

        if stats is not None:
            try:
                # best-effort counters
                if not hasattr(stats, "_cache_writes"):
                    stats._cache_writes = []  # type: ignore[attr-defined]
                stats._cache_writes.append(rel)  # type: ignore[attr-defined]
            except Exception:
                pass

    finally:
        # Ensure tmp file gone
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def read_json_cache(project_root: Path, rel_path: Union[str, Path]) -> Optional[Any]:
    """
    Read a JSON cache artifact if it exists and appears to be a text JSON file.
    Returns the parsed object or None if missing, binary, or invalid JSON.
    """
    try:
        target = _resolve_safe_path(project_root, rel_path)
    except ValueError:
        _emit_structured("warning", "cache_read_refuse", {"path": str(rel_path)})
        return None

    if not target.exists() or not target.is_file():
        _emit_structured("debug", "cache_read_miss", {"path": str(rel_path)})
        return None

    try:
        raw = target.read_bytes()
    except Exception as e:
        _emit_structured("warning", "cache_read_failed", {"path": str(rel_path), "error": str(e)})
        return None

    if raw and _is_probably_binary(raw):
        _emit_structured("warning", "cache_read_binary", {"path": str(rel_path)})
        return None

    text, enc, nl, had_bom, _ = _read_text_and_meta_if_exists(target)
    if not text:
        return None

    try:
        obj = json.loads(text)
        _emit_structured("info", "cache_read_hit", {"path": str(rel_path)})
        return obj
    except Exception as e:
        _emit_structured("warning", "cache_read_invalid_json", {"path": str(rel_path), "error": str(e)})
        return None


# ---------------------- Repository scanning & evidence helpers ----------------------

def deterministic_repo_scan(
    project_root: Path,
    *,
    exclude_dirs: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    max_total_bytes: Optional[int] = None,
    max_file_bytes: Optional[int] = None,
) -> List[str]:
    """
    Deterministically list files under project_root suitable for scanning.

    Changes vs your version:
      - Uses os.walk with sorted dirs/files for deterministic traversal.
      - Skips symlinks (prevents scanning outside root via symlink tricks).
      - Uses resolved containment checks before returning paths.
    """
    base = Path(project_root).resolve()
    excluded = set(exclude_dirs or [".git", ".aidev", "__pycache__"])

    entries: List[Tuple[str, int]] = []  # (posix_rel_path, size)

    try:
        for dirpath, dirnames, filenames in os.walk(base):
            # deterministically order + prune excluded dirs
            dirnames[:] = sorted(d for d in dirnames if d not in excluded)
            filenames = sorted(filenames)

            dpath = Path(dirpath)

            # Skip symlinked directories defensively
            try:
                if dpath.is_symlink():
                    continue
            except Exception:
                pass

            for fn in filenames:
                p = dpath / fn
                try:
                    if not p.is_file():
                        continue
                    # skip symlinks
                    if p.is_symlink():
                        continue

                    # Resolve and ensure inside base (symlink/file-system trick defense)
                    resolved = p.resolve()
                    try:
                        rel = resolved.relative_to(base)
                    except Exception:
                        continue

                    try:
                        size = resolved.stat().st_size
                    except Exception:
                        size = 0

                    entries.append((rel.as_posix(), size))
                except Exception:
                    continue
    except Exception as e:
        _emit_structured("warning", "repo_scan_failed", {"root": str(base), "error": str(e)})
        return []

    # Already deterministic due to walk sorting, but keep final sort for safety
    entries.sort(key=lambda x: x[0])

    if max_file_bytes is not None:
        entries = [e for e in entries if e[1] <= max_file_bytes]

    selected: List[str] = []
    if max_total_bytes is not None:
        acc = 0
        for rel, size in entries:
            if acc + size > max_total_bytes:
                continue
            selected.append(rel)
            acc += size
    else:
        selected = [rel for rel, _ in entries]

    if max_files is not None:
        selected = selected[:max_files]

    _emit_structured("info", "repo_scan_done", {"root": str(base), "files": len(selected)})
    return selected


def gather_basic_evidence(
    project_root: Path,
    *,
    paths: Optional[List[str]] = None,
    max_files: int = 10000,
    max_bytes_per_file: int = 4096,
    max_lines_per_file: Optional[int] = None,
    compute_hash: bool = False,
    allow_full_hash: bool = False,
    rec_id: Optional[str] = None,
) -> Dict[str, Any]:
    base = Path(project_root).resolve()

    # Normalize/validate provided paths.
    if paths is None:
        valid_paths = deterministic_repo_scan(base)
    else:
        seen: List[str] = []
        for raw_rel in paths:
            try:
                if not raw_rel:
                    continue
                candidate = (base / raw_rel)

                # skip symlinks up front
                try:
                    if candidate.is_symlink():
                        continue
                except Exception:
                    pass

                resolved = candidate.resolve()
                try:
                    relpath = resolved.relative_to(base)
                except Exception:
                    _emit_structured("warning", "gather_evidence_path_outside_root", {"path": str(raw_rel)})
                    continue

                pp = relpath.as_posix()
                if pp not in seen:
                    seen.append(pp)
            except Exception:
                _emit_structured("warning", "gather_evidence_path_invalid", {"path": str(raw_rel)})
                continue

        valid_paths = sorted(seen)

    valid_paths = list(valid_paths)[:max_files]

    files_out: List[Dict[str, Any]] = []
    total_bytes = 0

    for rel in valid_paths:
        try:
            p = (base / rel)

            if not p.exists() or not p.is_file():
                continue
            if p.is_symlink():
                continue

            resolved = p.resolve()
            try:
                resolved.relative_to(base)
            except Exception:
                continue

            size = resolved.stat().st_size

            sha_hex: Optional[str] = None
            sha_truncated: bool = False
            is_bin = False

            with open(resolved, "rb") as fh:
                # Sample must never exceed max_bytes_per_file
                sample_size = min(2048, max(0, max_bytes_per_file))
                sample = fh.read(sample_size) if sample_size > 0 else b""
                is_bin = _is_probably_binary(sample) if sample else False

                if compute_hash:
                    h = hashlib.sha256()
                    bytes_read = 0

                    if sample:
                        h.update(sample)
                        bytes_read += len(sample)

                    if allow_full_hash:
                        while True:
                            chunk = fh.read(8192)
                            if not chunk:
                                break
                            h.update(chunk)
                            bytes_read += len(chunk)
                        sha_hex = h.hexdigest()
                        sha_truncated = False
                    else:
                        while bytes_read < max_bytes_per_file:
                            toread = min(8192, max_bytes_per_file - bytes_read)
                            if toread <= 0:
                                break
                            chunk = fh.read(toread)
                            if not chunk:
                                break
                            h.update(chunk)
                            bytes_read += len(chunk)

                        if bytes_read >= size:
                            sha_hex = h.hexdigest()
                            sha_truncated = False
                        else:
                            sha_hex = None
                            sha_truncated = True

            files_out.append(
                {
                    "path": rel,
                    "size": size,
                    "is_binary": bool(is_bin),
                    "sha256": sha_hex,
                    "sha256_truncated": bool(sha_truncated),
                }
            )
            total_bytes += size

        except Exception as e:
            _emit_structured("warning", "gather_evidence_entry_failed", {"path": rel, "error": str(e)})
            continue

    evidence = {
        "project_root": str(base),
        "rec_id": rec_id,
        "file_count": len(files_out),
        "total_bytes": total_bytes,
        "files": files_out,
        "max_lines_per_file": max_lines_per_file,  # carry-through for downstream policy if you want it
    }
    _emit_structured("info", "gather_basic_evidence_done", {"root": str(base), "file_count": len(files_out)})
    return evidence
