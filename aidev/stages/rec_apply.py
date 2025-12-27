# aidev/stages/rec_apply.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, TYPE_CHECKING
import subprocess
import difflib
import logging
import os
import hashlib

# Defensive imports: some companion modules in this repo may be in an
# intermediate broken state during iterative edits. Import them in a
# best-effort way and provide lightweight fallbacks so importing this
# module doesn't raise SyntaxError/ImportError and block test discovery.
# NOTE: Per the 'fail-fast' recommendation we no longer provide silent
# runtime fallbacks for core components like KnowledgeBase or
# apply_and_refresh. Those are imported lazily where used and will raise
# clear ImportError messages if missing so developers can fix the root
# issue instead of running a degraded no-op flow.
if TYPE_CHECKING:
    from ..cards import KnowledgeBase

try:
    from ..state import ProjectState
except Exception:
    ProjectState = object

try:
    from ..io_utils import _read_file_text_if_exists, generate_unified_diff
except Exception:
    # Fallback implementations: best-effort, used only if the real helpers
    # cannot be imported at module import time. These avoid expensive
    # dependencies and keep behavior reasonably similar for tests that don't
    # rely on full fidelity of diff generation.
    def _read_file_text_if_exists(root: Path, rel_path: str) -> Optional[str]:
        try:
            p = (root / rel_path)
            if not p.exists():
                return None
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    # IMPORTANT: keep signature compatible with call sites in this module:
    #   generate_unified_diff("a/x", "b/x", old_text, new_text)
    def generate_unified_diff(fromfile: str, tofile: str, old_text: str, new_text: str) -> str:
        try:
            old_lines = (old_text or "").splitlines(keepends=True)
            new_lines = (new_text or "").splitlines(keepends=True)
            diff = difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile)
            return "".join(diff)
        except Exception:
            return ""

try:
    from ..schemas import (
        file_edit_schema,
        validate_instance as _schema_validate_instance,
        SchemaValidationUnavailable,
    )
except Exception:
    # If schema helpers are unavailable, provide minimal stand-ins so the
    # validation wrapper can behave as-if jsonschema is not installed.
    def file_edit_schema() -> Dict[str, Any]:
        return {}

    class SchemaValidationUnavailable(Exception):
        pass

    def _schema_validate_instance(edit: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> None:  # pragma: no cover - fallback
        # Signal to callers that strict validation isn't available so they can
        # fall back to permissive behavior.
        raise SchemaValidationUnavailable("schema validation backend unavailable")

# Note: approval_gate (evaluate_approval, ApprovalContext) is imported
# lazily where it is used so missing modules raise clear ImportError with
# remediation hints instead of silently falling back to permissive no-ops.

try:
    from .. import events as _events
except Exception:
    # Lightweight events shim with the same API shape used here; all methods
    # are no-ops but return plausible sentinel values for callers.
    class _EventsShim:
        def progress_start(self, *args, **kwargs):
            return "progress-id"

        def checks_started(self, *args, **kwargs):
            return None

        def progress_finish(self, *args, **kwargs):
            return None

        def checks_result(self, *args, **kwargs):
            return None

        def status(self, *args, **kwargs):
            return None

    _events = _EventsShim()

# Use the structured logging helpers introduced by the recommendation.
# These functions are expected to have the signature: fn(message: str, meta: Optional[Dict]=None)
try:
    from ..logger import info, debug, warning, error
except Exception:
    # Fallback to the stdlib logger if the structured logger isn't available.
    _std = logging.getLogger("aidev")

    def info(msg: str, meta: Optional[Dict[str, Any]] = None) -> None:
        _std.info("%s %s", msg, repr(meta) if meta else "")

    def debug(msg: str, meta: Optional[Dict[str, Any]] = None) -> None:
        _std.debug("%s %s", msg, repr(meta) if meta else "")

    def warning(msg: str, meta: Optional[Dict[str, Any]] = None) -> None:
        _std.warning("%s %s", msg, repr(meta) if meta else "")

    def error(msg: str, meta: Optional[Dict[str, Any]] = None) -> None:
        _std.error("%s %s", msg, repr(meta) if meta else "")

# Prefer debug/warning implementations from the events module (if available).
# This ensures calls to debug(...) and warning(...) in this module resolve to
# the canonical events helpers when present, while preserving the logger
# fallback above.
try:
    from ..events import warning as _events_warning_fn, debug as _events_debug_fn

    warning = _events_warning_fn  # type: ignore
    debug = _events_debug_fn  # type: ignore
except Exception:
    # If the import fails, stick with the logger fallback defined above.
    pass

ProgressFn = Callable[[str, Dict[str, Any]], None]
ErrorFn = Callable[[str, Dict[str, Any]], None]


# ----------------------------- EDIT SCHEMA -----------------------------

try:
    EDIT_SCHEMA: Dict[str, Any] = file_edit_schema()
except Exception as e:  # pragma: no cover - best-effort
    # Emit a structured warning so operators see that schema loading failed.
    try:
        warning("schema.load_failed", meta={"error": str(e), "note": "falling back to empty schema"})
    except Exception:
        # best-effort: swallow logging failures
        pass
    EDIT_SCHEMA = {}

# ----------------------------- V5 FileEdit helpers -----------------------------

# We keep these helpers local to this module because this is the place where
# "action synthesized edits" and "sanitization" previously emitted legacy/partial
# edit dicts. The current schema (FileEdit v5) requires a stable set of keys,
# including some nullable ones, so sanitization must preserve/restore them.

_CFN_KEYS = ("changed_interfaces", "new_identifiers", "deprecated_identifiers", "followup_requirements")


def _empty_cross_file_notes() -> Dict[str, Any]:
    return {k: [] for k in _CFN_KEYS}


def _normalize_cross_file_notes(raw: Any) -> Dict[str, Any]:
    """
    Return a schema-friendly cross_file_notes dict with all required list keys.
    """
    base = _empty_cross_file_notes()
    if not isinstance(raw, dict):
        return base
    for k in _CFN_KEYS:
        v = raw.get(k)
        if isinstance(v, list):
            base[k] = v
        else:
            base[k] = []
    return base


def _infer_edit_kind(content: Optional[str], patch_unified: Optional[str], raw_kind: Any = None) -> Optional[str]:
    """
    Infer edit_kind with best-effort normalization. Returns None if it can't be inferred.
    """
    if isinstance(raw_kind, str):
        k = raw_kind.strip()
        if k in ("full", "patch_unified"):
            return k
    if content is not None:
        return "full"
    if patch_unified is not None:
        return "patch_unified"
    return None


def _mk_file_edit_v5(
    *,
    path: str,
    rec_id: str,
    is_new: bool,
    edit_kind: str,
    content: Optional[str],
    patch_unified: Optional[str],
    summary: str = "",
    cross_file_notes: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct a FileEdit dict that matches the canonical schema shape:
      - required keys always present
      - content/patch_unified always present (nullable)
      - details always present and includes nested cross_file_notes
    """
    cfn = _normalize_cross_file_notes(cross_file_notes)
    det: Dict[str, Any] = dict(details) if isinstance(details, dict) else {}
    # Ensure details.cross_file_notes exists and is dict-shaped
    det_cfn = det.get("cross_file_notes")
    if not isinstance(det_cfn, dict):
        det["cross_file_notes"] = dict(cfn)
    else:
        # normalize its required keys too
        det["cross_file_notes"] = _normalize_cross_file_notes(det_cfn)

    # Enforce discriminator exclusivity
    if edit_kind == "full":
        patch_unified = None
    elif edit_kind == "patch_unified":
        content = None

    return {
        "path": path,
        "rec_id": rec_id,
        "is_new": bool(is_new),
        "edit_kind": edit_kind,
        "content": content,  # required (nullable)
        "patch_unified": patch_unified,  # required (nullable)
        "summary": summary or "",
        "cross_file_notes": cfn,
        "details": det,
    }


# ----------------------------- Helpers ---------------------------------


def _is_safe_path(root: Path, rel_path: str) -> bool:
    """
    Root-lock helper: ensure rel_path stays inside the project root and is not the root itself.
    """
    try:
        if not rel_path:
            return False
        p = (root / rel_path).resolve()
        root_resolved = root.resolve()
        if p == root_resolved:
            return False
        # Will raise if p is not under root_resolved
        p.relative_to(root_resolved)
        return True
    except Exception:
        return False


def _tail(s: str, nbytes: int) -> str:
    if not s:
        return ""
    enc = s.encode("utf-8", errors="ignore")
    if len(enc) <= nbytes:
        return s
    return enc[-nbytes:].decode("utf-8", errors="ignore")


def _contains_glob_chars(p: str) -> bool:
    """Return True if the path contains common glob characters that indicate an unresolved pattern.

    We treat '*', '?', '[' as glob indicators.
    """
    if not p:
        return False
    return any(c in p for c in ("*", "?", "["))


def _get_pending_commands(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Recommendations may accumulate queued commands under rec["_pending_commands"].
    This returns a normalized list of dicts, without mutating rec.
    """
    raw = rec.get("_pending_commands")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in raw:
        if isinstance(it, dict):
            out.append(it)
    return out


def _pop_pending_commands(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pop queued commands from rec["_pending_commands"] (if present) and return them.
    """
    cmds = _get_pending_commands(rec)
    try:
        if "_pending_commands" in rec:
            del rec["_pending_commands"]
    except Exception:
        pass
    return cmds


def _run_queued_commands(
    *,
    root: Path,
    rec_id: str,
    commands: List[Dict[str, Any]],
    dry_run: bool,
    progress_cb: Optional[ProgressFn] = None,
    progress_error_cb: Optional[ErrorFn] = None,
    trace_obj: Optional[Any] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Execute queued commands (post-approval). Returns (ok, results).

    This is post-approval and post-apply by design: it prevents side effects
    from occurring before a user has approved the recommendation.
    """
    if not commands:
        return True, []

    def emit(event: str, **payload: Any) -> None:
        payload.setdefault("project_root", str(root.resolve()))
        if progress_cb:
            try:
                progress_cb(event, payload)
                return
            except Exception:
                try:
                    debug("progress_cb_failed", meta={"event": event, "rec_id": rec_id})
                except Exception:
                    pass
        try:
            info(f"apply_rec_commands.{event}", meta={"rec_id": rec_id, **payload})
        except Exception:
            pass

    def emit_err(where: str, **payload: Any) -> None:
        payload.setdefault("project_root", str(root.resolve()))
        if progress_error_cb:
            try:
                progress_error_cb(where, payload)
                return
            except Exception:
                try:
                    debug("progress_error_cb_failed", meta={"where": where, "rec_id": rec_id})
                except Exception:
                    pass
        try:
            error("apply_rec_commands.error", meta={"where": where, "rec_id": rec_id, **payload})
        except Exception:
            pass

    results: List[Dict[str, Any]] = []
    all_ok = True

    for cmd_spec in commands:
        try:
            cmd = str(cmd_spec.get("command") or "").strip()
        except Exception:
            cmd = ""
        if not cmd:
            continue

        timeout_sec = 1200.0
        try:
            timeout_sec = float(cmd_spec.get("timeout_sec") or 1200.0)
        except Exception:
            timeout_sec = 1200.0

        if dry_run:
            emit("run_command_skipped", command=cmd, reason="dry_run")
            results.append({"command": cmd, "skipped": True, "reason": "dry_run"})
            continue

        emit("run_command", command=cmd, cwd=str(root), timeout_sec=timeout_sec)

        try:
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            ok = (res.returncode == 0)
            if not ok:
                all_ok = False
                emit_err(
                    "run_command",
                    command=cmd,
                    returncode=res.returncode,
                    stderr_tail=_tail(res.stderr or "", 2000),
                )

            if trace_obj is not None:
                try:
                    trace_obj.write(
                        "ACTION",
                        "run_command",
                        {
                            "rec_id": rec_id,
                            "command": cmd,
                            "returncode": res.returncode,
                            "stdout_tail": _tail(res.stdout or "", 4000),
                            "stderr_tail": _tail(res.stderr or "", 4000),
                            "project_root": str(root.resolve()),
                            "phase": "post_apply",
                        },
                    )
                except Exception:
                    try:
                        debug("trace_obj_write_failed", meta={"rec_id": rec_id})
                    except Exception:
                        pass

            results.append(
                {
                    "command": cmd,
                    "returncode": res.returncode,
                    "ok": ok,
                    "stdout_tail": _tail(res.stdout or "", 4000),
                    "stderr_tail": _tail(res.stderr or "", 4000),
                }
            )
        except Exception as e:
            all_ok = False
            emit_err("run_command", command=cmd, error=str(e))
            results.append({"command": cmd, "ok": False, "error": str(e)})

    return all_ok, results


def sanitize_edits_and_proposed(
    *,
    root: Path,
    edits: List[Dict[str, Any]],
    proposed: List[Dict[str, Any]],
    rec_id: str,
    progress_cb: Optional[ProgressFn] = None,
    progress_error_cb: Optional[ErrorFn] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Best-effort cleanup / validation layer for LLM-generated edits.

    Enforces:
      - non-empty relative path
      - path is inside project root (root-lock)
      - rejects unresolved globs in paths
      - at least one of {content, patch_unified, patch} is present and non-empty

    Normalizes:
      - rec_id always set
      - ALWAYS emits schema-required keys for FileEdit (including nullable ones):
          path, rec_id, is_new, edit_kind, content, patch_unified, summary, cross_file_notes, details
      - patch -> patch_unified

    Aligns:
      - proposed list is filtered to only include paths present in sanitized edits,
        while preserving extra metadata fields.
    """
    def emit_err(where: str, **payload: Any) -> None:
        payload.setdefault("project_root", str(root.resolve()))
        if progress_error_cb:
            try:
                progress_error_cb(where, payload)
                return
            except Exception:
                try:
                    debug("progress_error_cb_failed", meta={"where": where, "rec_id": rec_id})
                except Exception:
                    pass
        try:
            warning("sanitize_edits.error", meta={"where": where, **payload})
        except Exception:
            pass

    def emit(event: str, **payload: Any) -> None:
        payload.setdefault("project_root", str(root.resolve()))
        if progress_cb:
            try:
                progress_cb(event, payload)
                return
            except Exception:
                try:
                    debug("progress_cb_failed", meta={"event": event, "rec_id": rec_id})
                except Exception:
                    pass
        try:
            info(f"sanitize_edits.{event}", meta={"rec_id": rec_id, **payload})
        except Exception:
            pass

    safe_edits: List[Dict[str, Any]] = []

    # Allowed keys from EDIT_SCHEMA (if present) are used only to preserve extra metadata
    # on top of the canonical v5 shape; we never drop v5 required keys.
    try:
        allowed_props = set(EDIT_SCHEMA.get("properties", {}).keys())
    except Exception:
        allowed_props = set()

    for raw in edits or []:
        if not isinstance(raw, dict):
            continue

        path = str(raw.get("path") or "").strip()
        if not path:
            emit_err("sanitize_edits", rec_id=rec_id, reason="empty_path")
            continue

        if not _is_safe_path(root, path):
            emit_err("sanitize_edits", rec_id=rec_id, path=path, reason="unsafe_path_outside_root")
            continue

        if _contains_glob_chars(path):
            emit_err("sanitize_edits", rec_id=rec_id, path=path, reason="contains_glob")
            continue

        # Normalize patch -> patch_unified
        patch_unified_raw = raw.get("patch_unified")
        if patch_unified_raw is None and "patch" in raw:
            patch_unified_raw = raw.get("patch")

        content_raw = raw.get("content")

        content: Optional[str] = content_raw if isinstance(content_raw, str) and content_raw else None
        patch_unified: Optional[str] = patch_unified_raw if isinstance(patch_unified_raw, str) and patch_unified_raw else None

        if content is None and patch_unified is None:
            emit_err("sanitize_edits", rec_id=rec_id, path=path, reason="no_content_or_patch")
            continue

        rec_val = str(raw.get("rec_id") or rec_id)

        # Determine edit_kind (or infer) and enforce exclusivity.
        edit_kind = _infer_edit_kind(content, patch_unified, raw.get("edit_kind"))
        if edit_kind is None:
            emit_err("sanitize_edits", rec_id=rec_id, path=path, reason="cannot_infer_edit_kind")
            continue

        # Determine is_new (default False; try to infer if missing)
        is_new_raw = raw.get("is_new")
        if isinstance(is_new_raw, bool):
            is_new = is_new_raw
        else:
            # Best-effort inference: if file doesn't exist on disk, treat as new
            try:
                is_new = not (root / path).exists()
            except Exception:
                is_new = False

        # Normalize cross_file_notes and details
        cfn = _normalize_cross_file_notes(raw.get("cross_file_notes"))
        details_raw = raw.get("details")
        details: Dict[str, Any] = dict(details_raw) if isinstance(details_raw, dict) else {}
        det_cfn = details.get("cross_file_notes")
        if not isinstance(det_cfn, dict):
            details["cross_file_notes"] = dict(cfn)
        else:
            details["cross_file_notes"] = _normalize_cross_file_notes(det_cfn)

        summary = raw.get("summary")
        summary_str = summary if isinstance(summary, str) else ""

        cleaned = _mk_file_edit_v5(
            path=path,
            rec_id=rec_val,
            is_new=is_new,
            edit_kind=edit_kind,
            content=content,
            patch_unified=patch_unified,
            summary=summary_str,
            cross_file_notes=cfn,
            details=details,
        )

        # Preserve any additional allowed props (if schema provides them) without
        # ever deleting required keys. This is mainly to keep forward-compatible
        # metadata fields if you extend the schema later (e.g. "confidence").
        if allowed_props:
            for k in allowed_props:
                if k in cleaned:
                    continue
                if k in raw:
                    cleaned[k] = raw[k]

        safe_edits.append(cleaned)

    # Align proposed diff bundle with the sanitized edits.
    safe_paths = {e["path"] for e in safe_edits}
    safe_proposed: List[Dict[str, Any]] = []
    for raw in proposed or []:
        if not isinstance(raw, dict):
            continue
        path = str(raw.get("path") or "").strip()
        if not path or path not in safe_paths:
            continue

        diff_text = raw.get("diff")
        if not isinstance(diff_text, str):
            diff_text = ""

        entry = dict(raw)
        entry["path"] = path
        entry["diff"] = diff_text
        if "why" not in entry:
            entry["why"] = raw.get("why") or ""
        entry["project_root"] = str(root.resolve())
        safe_proposed.append(entry)

    dropped = len(edits or []) - len(safe_edits)
    if dropped > 0:
        emit("sanitize_edits_dropped", rec_id=rec_id, dropped=dropped, total=len(edits or []))

    return safe_edits, safe_proposed


def validate_file_edits_schema(
    edits: List[Dict[str, Any]],
    *,
    rec_id: str,
    schema: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[ProgressFn] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate a list of FileEdit objects against the canonical schema.

    Returns (ok, errors), where:
      - ok is True if either:
          * schema is empty (no-op), or
          * jsonschema is available and all edits validate successfully.
      - errors is a list of human-readable error strings when validation
        fails (may be empty when ok=True).

    If jsonschema is not installed, we log a one-time status and treat
    validation as a no-op (ok=True).
    """
    def emit(event: str, **payload: Any) -> None:
        if progress_cb:
            try:
                progress_cb(event, payload)
                return
            except Exception:
                try:
                    debug("progress_cb_failed", meta={"event": event, "rec_id": rec_id})
                except Exception:
                    pass
        try:
            info("validate_file_edits_schema", meta={"rec_id": rec_id, **payload})
        except Exception:
            pass

    # IMPORTANT: empty edits can be valid (e.g., a commands-only recommendation).
    if not edits:
        emit("no_edits", where="validate_edits", rec_id=rec_id, message="No file edits provided; skipping file_edit validation.")
        return True, []

    if schema is None:
        schema = EDIT_SCHEMA

    # If we have no schema loaded, don't block the pipeline; just warn.
    if not schema:
        emit(
            "schema_missing",
            where="validate_edits",
            rec_id=rec_id,
            message=(
                "EDIT_SCHEMA is empty; skipping strict file_edit validation. "
                "Ensure aidev/schemas/file_edit.schema.json is present."
            ),
        )
        return True, []

    errors: List[str] = []

    for idx, edit in enumerate(edits):
        try:
            _schema_validate_instance(edit, schema=schema)
        except SchemaValidationUnavailable:
            emit(
                "schema_validation_unavailable",
                where="validate_edits",
                rec_id=rec_id,
                message=(
                    "jsonschema is not installed; skipping strict file_edit "
                    "validation. Install `jsonschema` to enable it."
                ),
            )
            return True, []
        except Exception as e:
            msg = f"edit[{idx}] invalid for rec {rec_id}: {e}"
            errors.append(msg)

    if errors:
        condensed = "; ".join(errors[:3])
        emit(
            "validate_edits_schema_failed",
            rec_id=rec_id,
            error=condensed,
        )
        return False, errors

    return True, []


def apply_rec_actions(
    *,
    root: Path,
    rec: Dict[str, Any],
    edits: List[Dict[str, Any]],
    proposed: List[Dict[str, Any]],
    dry_run: bool,
    progress_cb: Optional[ProgressFn] = None,
    progress_error_cb: Optional[ErrorFn] = None,
    trace_obj: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Consume structured actions from a recommendation and translate them into
    real workspace changes:

    - type == "run_command":
        IMPORTANT: Commands are QUEUED here and executed POST-APPROVAL (and typically post-apply)
        by apply_single_recommendation(). This prevents side effects before user approval.

    - type in {"create_file", "write_test"}:
        Append a FileEdit object that creates/overwrites the file with
        full content, plus a corresponding 'proposed' diff entry.

    - type in {"add_dependency", "update_config"}:
        Append a patch_unified edit for the target file (e.g. requirements.txt),
        by generating a unified diff that adds the requested line(s).
        If a valid diff cannot be produced, fall back to a full edit.

    Unknown types are ignored with a small status event.
    """
    rec_id = str(rec.get("id") or "rec")
    actions = rec.get("actions") or []
    if not isinstance(actions, list) or not actions:
        return edits, proposed

    def emit(event: str, **payload: Any) -> None:
        payload.setdefault("project_root", str(root.resolve()))
        if progress_cb:
            try:
                progress_cb(event, payload)
                return
            except Exception:
                try:
                    debug("progress_cb_failed", meta={"event": event, "rec_id": rec_id})
                except Exception:
                    pass
        try:
            info(f"apply_rec_actions.{event}", meta={"rec_id": rec_id, **payload})
        except Exception:
            pass

    def emit_err(where: str, **payload: Any) -> None:
        payload.setdefault("project_root", str(root.resolve()))
        if progress_error_cb:
            try:
                progress_error_cb(where, payload)
                return
            except Exception:
                try:
                    debug("progress_error_cb_failed", meta={"where": where, "rec_id": rec_id})
                except Exception:
                    pass
        try:
            error("apply_rec_actions.error", meta={"where": where, "rec_id": rec_id, **payload})
        except Exception:
            pass

    new_edits: List[Dict[str, Any]] = list(edits or [])
    new_proposed: List[Dict[str, Any]] = list(proposed or [])

    # Ensure queue exists if we add commands
    def _queue_cmd(cmd: str, timeout_sec: float) -> None:
        try:
            q = rec.get("_pending_commands")
            if not isinstance(q, list):
                q = []
                rec["_pending_commands"] = q
            q.append({"command": cmd, "timeout_sec": timeout_sec})
        except Exception:
            # best-effort: if we can't queue, emit an error
            emit_err("action_run_command", rec_id=rec_id, command=cmd, reason="failed_to_queue")

    for action in actions:
        if not isinstance(action, dict):
            continue

        a_type = (action.get("type") or "").strip()
        if not a_type:
            continue

        # ---------------- run_command (QUEUE ONLY) ----------------
        if a_type == "run_command":
            cmd = (action.get("command") or "").strip()
            if not cmd:
                continue

            try:
                timeout_sec = float(action.get("timeout_sec") or 1200.0)
            except Exception:
                timeout_sec = 1200.0

            # Queue regardless of dry_run; execution is controlled later.
            _queue_cmd(cmd, timeout_sec)

            emit(
                "action_run_command_queued",
                rec_id=rec_id,
                command=cmd,
                timeout_sec=timeout_sec,
                note="queued_for_post_approval_execution",
                dry_run=bool(dry_run),
            )
            continue

        # ---------------- create_file / write_test ----------------
        if a_type in {"create_file", "write_test"}:
            rel_path = (action.get("path") or "").strip()
            if not rel_path or not _is_safe_path(root, rel_path):
                emit_err(
                    "action_create_file",
                    rec_id=rec_id,
                    path=rel_path,
                    reason="unsafe_or_empty_path",
                )
                continue

            # If there's already an explicit edit for this path (e.g. from the
            # per-file Edit-File LLM call), do NOT synthesize a stub that
            # overwrites it. We may still add a proposed diff for UI purposes.
            existing_edit = next(
                (e for e in new_edits if isinstance(e, dict) and e.get("path") == rel_path),
                None,
            )
            if existing_edit is not None:
                has_proposed = any(isinstance(p, dict) and p.get("path") == rel_path for p in new_proposed)
                if not has_proposed:
                    old_text = _read_file_text_if_exists(root, rel_path) or ""

                    # Prefer diffing against the *effective* new text if available.
                    new_text = old_text
                    try:
                        c = existing_edit.get("content")
                        if isinstance(c, str):
                            new_text = c
                        else:
                            # If it's a patch-only edit, preview_content is ambiguous; fall back to old_text
                            new_text = old_text
                    except Exception:
                        new_text = old_text

                    try:
                        diff = generate_unified_diff(
                            f"a/{rel_path}",
                            f"b/{rel_path}",
                            old_text,
                            new_text,
                        )
                    except Exception:
                        diff = ""

                    new_proposed.append(
                        {
                            "path": rel_path,
                            "diff": diff,
                            "why": action.get("why") or rec.get("why") or "",
                            "preview_content": new_text,
                            "preview_bytes": len(new_text.encode("utf-8")),
                            "project_root": str(root.resolve()),
                        }
                    )

                emit(
                    "action_create_file_skipped_existing_edit",
                    rec_id=rec_id,
                    path=rel_path,
                )
                continue

            # No existing edit – fall back to the action's content/skeleton,
            # or, as a last resort, a tiny stub.
            raw_content = action.get("content")
            skeleton = action.get("skeleton")

            if isinstance(raw_content, str) and raw_content.strip():
                content = raw_content
            elif isinstance(skeleton, str) and skeleton.strip():
                content = skeleton
            else:
                content = (
                    "# TODO: Initial stub created by AI Dev Bot. "
                    "Fill in implementation to satisfy the recommendation.\n"
                )

            old_text = _read_file_text_if_exists(root, rel_path) or ""
            new_text = content

            try:
                diff = generate_unified_diff(
                    f"a/{rel_path}",
                    f"b/{rel_path}",
                    old_text,
                    new_text,
                )
            except Exception:
                diff = ""

            # Determine newness from on-disk state
            try:
                is_new = not (root / rel_path).exists()
            except Exception:
                is_new = False

            summary = str(action.get("summary") or action.get("why") or rec.get("why") or "")

            new_edits.append(
                _mk_file_edit_v5(
                    path=rel_path,
                    rec_id=rec_id,
                    is_new=is_new,
                    edit_kind="full",
                    content=new_text,
                    patch_unified=None,
                    summary=summary,
                    cross_file_notes=None,
                    details=None,
                )
            )
            new_proposed.append(
                {
                    "path": rel_path,
                    "diff": diff,
                    "why": action.get("why") or rec.get("why") or "",
                    "preview_content": new_text,
                    "preview_bytes": len(new_text.encode("utf-8")),
                    "project_root": str(root.resolve()),
                }
            )
            emit(
                "action_create_file",
                rec_id=rec_id,
                path=rel_path,
                bytes=len(new_text.encode("utf-8")),
            )
            continue

        # ---------------- add_dependency / update_config ----------------
        if a_type in {"add_dependency", "update_config"}:
            rel_path = (
                action.get("path")
                or action.get("file")
                or "requirements.txt"
            )
            rel_path = str(rel_path).strip()
            if not rel_path or not _is_safe_path(root, rel_path):
                emit_err(
                    "action_add_dependency",
                    rec_id=rec_id,
                    path=rel_path,
                    reason="unsafe_or_empty_path",
                )
                continue

            line = action.get("line") or action.get("dependency") or action.get("entry")
            if line is None:
                emit_err(
                    "action_add_dependency",
                    rec_id=rec_id,
                    path=rel_path,
                    reason="missing_line_or_dependency",
                )
                continue
            line_str = str(line).rstrip("\n")

            old_text = _read_file_text_if_exists(root, rel_path) or ""
            newline = "\n" if old_text and not old_text.endswith("\n") else ""
            new_text = old_text + f"{newline}{line_str}\n"

            # Determine newness from on-disk state
            try:
                file_exists = (root / rel_path).exists()
            except Exception:
                file_exists = True
            is_new = not file_exists

            summary = str(action.get("summary") or action.get("why") or rec.get("why") or "")

            # Prefer producing a valid unified diff that applies to old_text.
            try:
                diff = generate_unified_diff(
                    f"a/{rel_path}",
                    f"b/{rel_path}",
                    old_text,
                    new_text,
                )
            except Exception:
                diff = ""

            # If diff couldn't be produced (or file is new), fall back to a full edit.
            # This prevents sanitize_edits_and_proposed from dropping an empty patch edit.
            if not diff.strip() or is_new:
                new_edits.append(
                    _mk_file_edit_v5(
                        path=rel_path,
                        rec_id=rec_id,
                        is_new=is_new,
                        edit_kind="full",
                        content=new_text,
                        patch_unified=None,
                        summary=summary,
                        cross_file_notes=None,
                        details=None,
                    )
                )
                emit(
                    "action_add_dependency_full_fallback",
                    rec_id=rec_id,
                    path=rel_path,
                    line=line_str,
                    reason="diff_empty_or_new_file",
                )
            else:
                new_edits.append(
                    _mk_file_edit_v5(
                        path=rel_path,
                        rec_id=rec_id,
                        is_new=is_new,
                        edit_kind="patch_unified",
                        content=None,
                        patch_unified=diff,
                        summary=summary,
                        cross_file_notes=None,
                        details=None,
                    )
                )
                emit(
                    "action_add_dependency",
                    rec_id=rec_id,
                    path=rel_path,
                    line=line_str,
                )

            new_proposed.append(
                {
                    "path": rel_path,
                    "diff": diff,
                    "why": action.get("why") or rec.get("why") or "",
                    "preview_content": new_text,
                    "preview_bytes": len(new_text.encode("utf-8")),
                    "project_root": str(root.resolve()),
                }
            )
            continue

        # ---------------- unknown type ----------------
        emit(
            "action_ignored",
            rec_id=rec_id,
            action_type=a_type,
            reason="unknown_type",
        )

    return new_edits, new_proposed


# ----------------------------- ApplyRecResult --------------------------


@dataclass
class ApplyRecResult:
    """
    Structured result of applying a single recommendation.

    - rec_id: stable identifier for the recommendation
    - title: human-readable label
    - applied: True if edits were actually applied (and not dry-run)
    - skipped: True if the recommendation was intentionally not applied
    - reason: short machine-readable reason for skip/failure (e.g. "dry_run",
              "user_rejected", "apply_failed")
    - changed_paths: relative paths of files that were changed on disk
    - details: extra diagnostic info (e.g. errors, pre-apply metadata)
    """
    rec_id: str
    title: str = ""
    applied: bool = False
    skipped: bool = False
    reason: Optional[str] = None
    changed_paths: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def _normalize_items_to_dicts(items: Any) -> List[Dict[str, Any]]:
    """Normalize a possibly-heterogeneous 'items' value into a list of dicts.

    - If items is None -> [].
    - If a single dict -> [dict].
    - If a list/tuple of dicts -> returned as-is.
    - If a list/tuple and elements are tuples/lists, map the first two positions to
      'path' and 'diff' respectively and stash any extras under 'extra'.
    - Otherwise wrap non-dict elements as {'raw': element}.
    """
    if items is None:
        return []
    if isinstance(items, dict):
        return [items]
    if not isinstance(items, (list, tuple)):
        return [{"raw": items}]

    normalized: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            normalized.append(it)
            continue

        if isinstance(it, (list, tuple)):
            try:
                path = it[0] if len(it) > 0 else None
                diff = it[1] if len(it) > 1 else None
                extra = list(it[2:]) if len(it) > 2 else None
                d: Dict[str, Any] = {}
                if path is not None:
                    d["path"] = path
                if diff is not None:
                    d["diff"] = diff
                if extra:
                    d["extra"] = extra
                normalized.append(d)
                continue
            except Exception:
                try:
                    debug("normalize_tuple_failed", meta={"item": str(it)})
                except Exception:
                    pass

        normalized.append({"raw": it})
    return normalized


def _ensure_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    return _normalize_items_to_dicts(value)


def _normalize_changed_paths(root: Path, raw_paths: Any, rec_id: Optional[str] = None) -> List[str]:
    """
    Coerce raw_paths to a list of repo-relative POSIX strings.

    - Accepts None/any and coerces to list of strings.
    - Preserves original order while removing duplicates (first occurrence kept).
    - Ignores empty/non-string items.
    - Ensures each path is repo-relative and under the project root. For absolute
      paths that resolve inside the project root, convert to a repo-relative
      POSIX string. For other invalid/unsafe entries, skip and emit a structured
      warning via warning("apply.changed_paths_filtered", meta=...).

    Note: do not raise on invalid entries.

    Unit-test hint: Expect order-preserving dedupe and that invalid/absolute-outside-root
    entries are filtered out (first occurrence wins).
    """
    try:
        root_res = root.resolve()
    except Exception:
        # If resolving root fails for any reason, avoid crashing and return empty list.
        try:
            warning("apply.changed_paths_filtered", meta={"rec_id": rec_id or "", "filtered_count": 0, "project_root": str(root)})
        except Exception:
            pass
        return []

    seen: Set[str] = set()
    out: List[str] = []
    filtered_count = 0

    # Normalize raw_paths into an iterable of items
    items: List[Any]
    if raw_paths is None:
        items = []
    elif isinstance(raw_paths, str):
        items = [raw_paths]
    elif isinstance(raw_paths, (list, tuple, set)):
        items = list(raw_paths)
    else:
        # If it's an object with attribute changed_paths, try to read it (defensive)
        try:
            candidate = getattr(raw_paths, "changed_paths", None)
            if isinstance(candidate, (list, tuple)):
                items = list(candidate)
            else:
                items = [raw_paths]
        except Exception:
            items = [raw_paths]

    for itm in items:
        try:
            if not isinstance(itm, str):
                filtered_count += 1
                continue
            s = itm.strip()
            if not s:
                filtered_count += 1
                continue

            # Reject unresolved globs here as they are not concrete paths.
            if _contains_glob_chars(s):
                filtered_count += 1
                continue

            p = Path(s)
            # If absolute, check it's inside root and derive repo-relative path.
            if p.is_absolute():
                try:
                    resolved = p.resolve()
                    rel = resolved.relative_to(root_res).as_posix()
                except Exception:
                    filtered_count += 1
                    continue
            else:
                # Treat s as repo-relative: ensure it is safe and canonicalize.
                # Use the same root-lock semantics as _is_safe_path.
                try:
                    candidate = (root_res / s).resolve()
                    # Will raise if not under root_res
                    rel = candidate.relative_to(root_res).as_posix()
                except Exception:
                    filtered_count += 1
                    continue

            # Dedupe while preserving order (first occurrence kept)
            if rel in seen:
                continue
            seen.add(rel)
            out.append(rel)
        except Exception:
            # Be resilient to any unexpected issues per constraints.
            filtered_count += 1
            try:
                debug("normalize_changed_paths_item_failed", meta={"item": str(itm), "rec_id": rec_id})
            except Exception:
                pass
            continue

    if filtered_count > 0:
        try:
            warning(
                "apply.changed_paths_filtered",
                meta={"rec_id": rec_id or "", "filtered_count": filtered_count, "project_root": str(root_res)},
            )
        except Exception:
            pass

    return out


def apply_single_recommendation(
    *,
    root: Path,
    st: ProjectState,
    kb: "KnowledgeBase",
    meta: Dict[str, Any],
    rec: Dict[str, Any],
    rec_edits: List[Dict[str, Any]],
    rec_proposed: List[Dict[str, Any]],
    dry_run: bool,
    auto_approve: bool,
    approval_cb: Optional[Callable[[List[Dict[str, Any]]], bool]],
    session_id: Optional[str],
    job_id: Optional[str],
    writes_by_rec: Dict[str, List[str]],
    stats_obj: Any,
    progress_cb: Optional[ProgressFn] = None,
    progress_error_cb: Optional[ErrorFn] = None,
    approval_timeout_sec: Optional[float] = None,
    job_update_cb: Optional[Callable[..., None]] = None,
    allowed_rec_ids: Optional[Set[str]] = None,
) -> ApplyRecResult:
    """
    Run preapply checks + approval gate + apply edits for a single recommendation.

    Returns an ApplyRecResult with:
      - applied/skipped flags
      - reason for skip/failure (if any)
      - changed_paths of files that were actually written

    Note: an allowlist may be provided via allowed_rec_ids; if provided and the
    recommendation's id is not present, the function will immediately skip the
    recommendation without invoking any side-effecting helpers (no subprocesses,
    no file writes, and no events that record applied changes).
    """
    rid = str(rec.get("id") or "rec")
    title = (rec.get("title") or rid).strip() or rid

    result = ApplyRecResult(rec_id=rid, title=title)
    # Record project_root early for tests/observability
    result.details["project_root"] = str(root.resolve())

    # Optional per-rec self-review blob injected by the orchestrator.
    # Shape is defined by self_review.schema.json and typically includes:
    #   - overall_status
    #   - warnings[]
    #   - file_update_requests[]
    self_review = rec.get("_self_review")
    if isinstance(self_review, dict):
        result.details["self_review"] = self_review
    else:
        self_review = None

    # Early allowlist guard: if an explicit allowlist is provided, skip quickly
    # without performing any side-effecting operations.
    if allowed_rec_ids is not None and rid not in allowed_rec_ids:
        result.skipped = True
        result.reason = "not_selected"
        if progress_cb:
            try:
                progress_cb(
                    "apply_skip",
                    {
                        "reason": result.reason,
                        "approved": False,
                        "rec_id": rid,
                        "project_root": str(root.resolve()),
                    },
                )
            except Exception:
                try:
                    debug("progress_cb_failed", meta={"where": "apply_skip", "rec_id": rid})
                except Exception:
                    pass
        try:
            info("apply_skip", meta={"reason": result.reason, "approved": False, "rec_id": rid, "project_root": str(root.resolve())})
        except Exception:
            pass
        return result

    # Normalize incoming recommendation shapes
    try:
        rec_proposed = _ensure_list_of_dicts(rec_proposed)
    except Exception as e:
        try:
            error("normalize_failed", meta={"stage": "rec_proposed", "rec_id": rid, "error": str(e)})
        except Exception:
            pass
        rec_proposed = []

    try:
        rec_edits = _ensure_list_of_dicts(rec_edits)
    except Exception as e:
        try:
            error("normalize_failed", meta={"stage": "rec_edits", "rec_id": rid, "error": str(e)})
        except Exception:
            pass
        rec_edits = []

    # ------------------ Apply structured 'actions' (if any) before validation ------------------
    # This is where recommendations can add create_file/add_dependency, etc.
    # NOTE: run_command actions are queued here and executed later (post-approval).
    try:
        rec_edits, rec_proposed = apply_rec_actions(
            root=root,
            rec=rec,
            edits=rec_edits,
            proposed=rec_proposed,
            dry_run=dry_run,
            progress_cb=progress_cb,
            progress_error_cb=progress_error_cb,
            trace_obj=getattr(st, "trace", None) if st is not None else None,
        )
    except Exception as e:
        try:
            warning("apply_rec_actions_failed", meta={"rec_id": rid, "error": str(e), "project_root": str(root.resolve())})
        except Exception:
            pass
        result.details["apply_rec_actions_error"] = str(e)

    # Surface any queued commands for UI/diagnostics (no execution here)
    try:
        pending = _get_pending_commands(rec)
        if pending:
            result.details["pending_commands"] = pending
    except Exception:
        pass

    # ------------------ Sanitize edits and proposed to v5 schema shape ------------------
    rec_edits, rec_proposed = sanitize_edits_and_proposed(
        root=root,
        edits=rec_edits,
        proposed=rec_proposed,
        rec_id=rid,
        progress_cb=progress_cb,
        progress_error_cb=progress_error_cb,
    )

    # ------------------ Reject unresolved glob-like or unsafe paths early ------------------
    offending: List[Dict[str, str]] = []
    for e in rec_edits:
        try:
            path = str(e.get("path") or "").strip()
        except Exception:
            path = ""
        if not path:
            offending.append({"path": path, "reason": "empty_path"})
            continue
        if _contains_glob_chars(path):
            offending.append({"path": path, "reason": "contains_glob"})
            continue
        if not _is_safe_path(root, path):
            offending.append({"path": path, "reason": "unsafe_path_outside_root"})
            continue

    if offending:
        if any(o.get("reason") == "contains_glob" for o in offending):
            result.reason = "unresolved_glob_target"
        else:
            result.reason = "unsafe_path_outside_root"
        result.skipped = True
        result.details["offending_paths"] = offending
        result.details["message"] = (
            "One or more edits contain unresolved glob patterns or paths outside the project root. "
            "Resolve glob specs via the targets resolver (e.g. stages/targets) so each target.path is a concrete, "
            "repo-relative path before attempting to apply edits."
        )
        result.details["suggested_fix"] = (
            "Use runtimes.path_safety.resolve_glob_within_root or stages/targets resolver to expand glob specs into concrete paths."
        )
        payload = {"rec_id": rid, "offending": offending, "reason": result.reason, "project_root": str(root.resolve())}
        if progress_error_cb:
            try:
                progress_error_cb("apply_validation", payload)
            except Exception:
                try:
                    debug("progress_error_cb_failed", meta={"where": "apply_validation", "rec_id": rid})
                except Exception:
                    pass
        elif progress_cb:
            try:
                progress_cb("apply_validation", payload)
            except Exception:
                try:
                    debug("progress_cb_failed", meta={"where": "apply_validation", "rec_id": rid})
                except Exception:
                    pass

        try:
            warning("apply_validation.rejected_paths", meta={"rec_id": rid, "offending": offending, "reason": result.reason, "project_root": str(root.resolve())})
        except Exception:
            pass

        return result

    # ------------------ Validate against schema (if available) ------------------
    ok, schema_errors = validate_file_edits_schema(rec_edits, rec_id=rid, schema=EDIT_SCHEMA, progress_cb=progress_cb)
    if not ok:
        result.skipped = True
        result.reason = "edit_schema_validation"
        result.details["schema_errors"] = schema_errors
        try:
            warning("apply_validation.schema_failed", meta={"rec_id": rid, "errors": schema_errors[:3], "project_root": str(root.resolve())})
        except Exception:
            pass
        return result

    # ---------- preapply_checks (non-fatal; informational) ----------
    rid_checks = _events.progress_start(
        "preapply_checks",
        detail=f"Running build/test checks in a temporary workspace for {title}…",
        session_id=session_id,
        job_id=job_id,
        project_root=str(root.resolve()),
    )
    _events.checks_started(total=None, session_id=session_id, job_id=job_id, project_root=str(root.resolve()))

    if job_update_cb:
        try:
            job_update_cb(
                stage="preapply_checks",
                message=f"Running preapply checks for {title}…",
                progress_pct=60,
            )
        except Exception:
            pass

    # Lazy import of run_preapply_checks so missing runtime implementations fail fast
    try:
        from ..runtime import run_preapply_checks as _run_preapply_checks, build_file_overlays_from_edits as _build_overlays
    except Exception as e:
        raise ImportError(
            "Missing required module 'aidev.runtime.run_preapply_checks'.\n"
            "This function is required for preapply checks in aidev.stages.rec_apply.\n"
            "Ensure 'aidev/runtime.py' exports run_preapply_checks and is importable (check PYTHONPATH / package layout or .aidev project map).\n"
            f"Original error: {e}"
        ) from e

    # Build file overlays from the edits so preapply checks and repair logic can
    # operate on the updated in-memory content rather than disk.
    file_overlays, overlay_errors = _build_overlays(root, rec_edits)
    if overlay_errors:
        # recommended: fail fast so preapply isn't run on the wrong code
        result.skipped = True
        result.reason = "overlay_patch_apply_failed"
        result.details["overlay_errors"] = overlay_errors[:5]
        try:
            warning("preapply.overlay_build_failed", meta={"rec_id": rid, "errors": overlay_errors[:3], "project_root": str(root.resolve())})
        except Exception:
            pass
        return result

    # Emit trace events for baseline, overlays, and check inputs to aid debugging
    try:
        baseline: Dict[str, str] = {}
        try:
            for e in rec_edits or []:
                try:
                    p = e.get("path") if isinstance(e, dict) else None
                    if not p or not isinstance(p, str):
                        continue
                    baseline[p] = _read_file_text_if_exists(root, p) or ""
                except Exception:
                    continue
        except Exception:
            baseline = {}

        check_inputs = {
            "rec_id": rid,
            "rec_edits": rec_edits,
            "rec_proposed": rec_proposed,
            "file_overlays": file_overlays,
            "project_root": str(root.resolve()),
        }

        try:
            overlay_summary: List[Dict[str, Any]] = []
            try:
                for p, cont in (file_overlays or {}).items():
                    try:
                        if isinstance(cont, str):
                            b = cont.encode("utf-8")
                            sha = hashlib.sha256(b).hexdigest()[:10]
                            overlay_summary.append({"path": p, "preview_bytes": len(b), "sha": sha})
                        else:
                            overlay_summary.append({"path": p, "preview_bytes": 0})
                    except Exception:
                        try:
                            overlay_summary.append({"path": p, "preview_bytes": len(str(cont).encode("utf-8"))})
                        except Exception:
                            overlay_summary.append({"path": p})
            except Exception:
                overlay_summary = []

            baseline_id = ""
            try:
                items = []
                for p, txt in (baseline or {}).items():
                    try:
                        if not isinstance(txt, str):
                            txt = str(txt)
                        h = hashlib.sha256((p + "\x00" + txt).encode("utf-8")).hexdigest()
                        items.append(h)
                    except Exception:
                        continue
                if items:
                    baseline_id = hashlib.sha256("".join(sorted(items)).encode("utf-8")).hexdigest()[:12]
            except Exception:
                baseline_id = ""

            try:
                if hasattr(_events, "trace_routing"):
                    try:
                        _events.trace_routing(
                            decision="preapply_checks",
                            baseline_id=baseline_id,
                            overlay_summary=overlay_summary,
                            reason="preapply_checks_start",
                            session_id=session_id,
                            job_id=job_id,
                            recId=rid,
                        )
                    except Exception:
                        pass
                else:
                    if st is not None and hasattr(st, "trace") and getattr(st, "trace") and hasattr(st.trace, "write"):
                        try:
                            st.trace.write("TRACE", "trace.routing", {"decision": "preapply_checks", "baseline_id": baseline_id, "overlay_summary": overlay_summary, "reason": "preapply_checks_start", "session_id": session_id, "job_id": job_id, "recId": rid})
                        except Exception:
                            try:
                                debug("trace_routing_fallback_failed", meta={"rec_id": rid})
                            except Exception:
                                pass
            except Exception:
                try:
                    debug("emit_trace_routing_failed", meta={"rec_id": rid})
                except Exception:
                    pass
        except Exception:
            try:
                debug("build_overlay_summary_failed", meta={"rec_id": rid})
            except Exception:
                pass

        try:
            if hasattr(_events, "trace_baseline"):
                try:
                    _events.trace_baseline(baseline=baseline, rec_id=rid)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if hasattr(_events, "trace_overlay"):
                try:
                    _events.trace_overlay(overlay=file_overlays, rec_id=rid)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if hasattr(_events, "trace_check_inputs"):
                try:
                    _events.trace_check_inputs(inputs=check_inputs, rec_id=rid)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if st is not None and hasattr(st, "trace") and getattr(st, "trace") and hasattr(st.trace, "write"):
                try:
                    st.trace.write("TRACE", "trace.baseline", {"rec_id": rid, "baseline": baseline, "project_root": str(root.resolve())})
                except Exception:
                    pass
                try:
                    st.trace.write("TRACE", "trace.overlay", {"rec_id": rid, "overlay": file_overlays, "project_root": str(root.resolve())})
                except Exception:
                    pass
                try:
                    st.trace.write("TRACE", "trace.check_inputs", {"rec_id": rid, "inputs": check_inputs, "project_root": str(root.resolve())})
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        try:
            debug("emit_trace_events_failed", meta={"rec_id": rid})
        except Exception:
            pass

    try:
        try:
            checks_ok, checks_results = _run_preapply_checks(root, rec_edits, file_overlays=file_overlays)
        except TypeError:
            if file_overlays:
                try:
                    edits_with_overlay = list(rec_edits) if isinstance(rec_edits, (list, tuple)) else [rec_edits]
                except Exception:
                    edits_with_overlay = [rec_edits] if rec_edits is not None else []
                edits_with_overlay = list(edits_with_overlay)
                edits_with_overlay.append({"file_overlays": file_overlays})
                try:
                    if st is not None and hasattr(st, "trace") and getattr(st, "trace") and hasattr(st.trace, "write"):
                        st.trace.write("VERIFY", "preapply_overlay_fallback", {"rec_id": rid, "used_edits_overlay": True})
                except Exception:
                    try:
                        debug("preapply_overlay_fallback_trace_failed", meta={"rec_id": rid})
                    except Exception:
                        pass
                checks_ok, checks_results = _run_preapply_checks(root, edits_with_overlay)
            else:
                checks_ok, checks_results = _run_preapply_checks(root, rec_edits)
    except Exception:
        raise

    _events.progress_finish(
        "preapply_checks",
        ok=checks_ok,
        recId=rid_checks,
        ok_flag=checks_ok,
        session_id=session_id,
        job_id=job_id,
        project_root=str(root.resolve()),
    )
    _events.checks_result(
        ok=checks_ok,
        results=checks_results,
        session_id=session_id,
        job_id=job_id,
        project_root=str(root.resolve()),
    )

    result.details["preapply_checks"] = {
        "ok": checks_ok,
        "results": checks_results,
    }

    try:
        warnings_list: List[str] = []
        if isinstance(checks_results, dict):
            raw_w = checks_results.get("warnings") or checks_results.get("warnings_list") or []
            if isinstance(raw_w, (list, tuple)):
                warnings_list = [str(x) for x in raw_w]
        elif isinstance(checks_results, (list, tuple)):
            warnings_list = [str(x) for x in checks_results]

        info("preapply_checks.complete", meta={
            "phase": "preapply_checks",
            "rec_id": rid,
            "ok": bool(checks_ok),
            "warnings": warnings_list,
            "project_root": str(root.resolve()),
        })
    except Exception:
        try:
            debug("emit_preapply_checks_failed", meta={"rec_id": rid})
        except Exception:
            pass

    # ---------- approval_gate ----------
    if auto_approve:
        approved = True
        _events.status(
            "Auto-approving edits for recommendation",
            stage="approval",
            recId=rid,
            session_id=session_id,
            job_id=job_id,
            project_root=str(root.resolve()),
        )
    else:
        rid_approval = _events.progress_start(
            "approval_gate",
            detail=f"Awaiting user approval for {title} ({len(rec_edits)} edits)…",
            session_id=session_id,
            job_id=job_id,
            project_root=str(root.resolve()),
        )
        if job_update_cb:
            try:
                job_update_cb(
                    stage="approval",
                    message=f"Awaiting user approval for {title}…",
                    progress_pct=75,
                )
            except Exception:
                pass

        timeout = approval_timeout_sec if approval_timeout_sec is not None else 3600.0

        try:
            from .approval_gate import evaluate_approval, ApprovalContext
        except Exception as e:
            raise ImportError(
                "Missing required stage 'aidev.stages.approval_gate'.\n"
                "The approval_gate implementation is required to evaluate user approvals.\n"
                "Ensure 'aidev/stages/approval_gate.py' exists and is importable (check PYTHONPATH / package layout or .aidev config).\n"
                f"Original error: {e}"
            ) from e

        approved = evaluate_approval(
            proposed=rec_proposed,
            approval_cb=approval_cb,
            dry_run=dry_run,
            context=ApprovalContext(
                session_id=session_id,
                job_id=job_id,
                timeout_sec=timeout,
                use_inbox=True,
                rec_id=rid,
                rec_title=title,
                auto_apply_followups=False,
            ),
            self_review=self_review,
        )

        _events.progress_finish(
            "approval_gate",
            ok=bool(approved) or dry_run,
            recId=rid_approval,
            approved=bool(approved),
            dry_run=dry_run,
            session_id=session_id,
            job_id=job_id,
            project_root=str(root.resolve()),
        )

    try:
        info("approval.decision", meta={
            "phase": "approval",
            "rec_id": rid,
            "approved": bool(approved),
            "auto_approve": bool(auto_approve),
            "dry_run": bool(dry_run),
            "project_root": str(root.resolve()),
        })
    except Exception:
        try:
            debug("emit_approval_decision_failed", meta={"rec_id": rid})
        except Exception:
            pass

    if not approved and not dry_run:
        result.skipped = True
        result.reason = "user_rejected"

        if progress_cb:
            try:
                progress_cb(
                    "apply_skip",
                    {
                        "reason": result.reason,
                        "approved": False,
                        "rec_id": rid,
                        "project_root": str(root.resolve()),
                    },
                )
            except Exception:
                try:
                    debug("progress_cb_failed", meta={"where": "apply_skip", "rec_id": rid})
                except Exception:
                    pass

        _events.status(
            "Recommendation skipped",
            where="rec_skipped",
            rec_id=rid,
            title=title,
            reason=result.reason,
            session_id=session_id,
            job_id=job_id,
            project_root=str(root.resolve()),
        )

        try:
            warning("apply_skip.user_rejected", meta={"rec_id": rid, "reason": result.reason, "project_root": str(root.resolve())})
        except Exception:
            pass

        return result

    if job_update_cb:
        try:
            job_update_cb(
                stage="apply",
                message=f"Applying edits for {title}…",
                progress_pct=85,
            )
        except Exception:
            pass

    def _noop_progress(event: str, payload: Dict[str, Any]) -> None:  # pragma: no cover
        return

    def _noop_error(where: str, payload: Dict[str, Any]) -> None:  # pragma: no cover
        return

    try:
        from ..cards import KnowledgeBase as _KB  # noqa: F401 - presence check
    except Exception as e:
        raise ImportError(
            "Missing required module 'aidev.cards.KnowledgeBase'.\n"
            "The KnowledgeBase implementation is required for in-memory KB refreshes and other operations.\n"
            "Ensure 'aidev/cards.py' defines KnowledgeBase and is importable (check PYTHONPATH / package layout or .aidev config).\n"
            f"Original error: {e}"
        ) from e

    # Pull queued commands now (post-approval). We will execute them:
    #  - after apply_and_refresh if there are file edits, OR
    #  - immediately if this is a commands-only recommendation.
    queued_commands = _get_pending_commands(rec)

    # Commands-only path: no file edits to apply.
    if not rec_edits:
        # In dry_run, do not execute; just return preview.
        if dry_run:
            result.skipped = True
            result.reason = "dry_run"
            result.changed_paths = []
            result.details["run_commands"] = [{"command": c.get("command"), "skipped": True, "reason": "dry_run"} for c in queued_commands]
            _events.status(
                "Recommendation dry-run preview (commands only)",
                where="rec_dry_run",
                rec_id=rid,
                title=title,
                changed_paths=[],
                dry_run=True,
                session_id=session_id,
                job_id=job_id,
                project_root=str(root.resolve()),
            )
            return result

        # Execute queued commands (post-approval).
        cmds = _pop_pending_commands(rec)
        ok_cmds, cmd_results = _run_queued_commands(
            root=root,
            rec_id=rid,
            commands=cmds,
            dry_run=False,
            progress_cb=progress_cb,
            progress_error_cb=progress_error_cb,
            trace_obj=getattr(st, "trace", None) if st is not None else None,
        )
        result.details["run_commands_ok"] = bool(ok_cmds)
        result.details["run_commands"] = cmd_results
        result.details["commands_executed"] = bool(cmd_results)
        result.applied = bool(cmd_results)
        result.skipped = not result.applied
        if result.applied:
            result.reason = "commands_only"
            _events.status(
                "Recommendation applied (commands only)",
                where="rec_applied",
                rec_id=rid,
                title=title,
                changed_paths=[],
                dry_run=False,
                session_id=session_id,
                job_id=job_id,
                project_root=str(root.resolve()),
            )
        else:
            result.reason = "no_actions"
            _events.status(
                "Recommendation produced no actions",
                where="rec_skipped",
                rec_id=rid,
                title=title,
                reason=result.reason,
                changed_paths=[],
                session_id=session_id,
                job_id=job_id,
                project_root=str(root.resolve()),
            )
        return result

    # Normal path: apply file edits, then run queued commands post-apply.
    try:
        try:
            from .apply_and_refresh import apply_and_refresh as _apply_and_refresh
        except Exception as e:
            raise ImportError(
                "Missing required stage 'aidev.stages.apply_and_refresh.apply_and_refresh'.\n"
                "The apply-and-refresh pipeline must be present to actually apply recommendation edits.\n"
                "Ensure 'aidev/stages/apply_and_refresh.py' exists and is importable, or restore the module that provides apply_and_refresh.\n"
                f"Original error: {e}"
            ) from e

        try:
            apply_result = _apply_and_refresh(
                root=root,
                kb=kb,
                meta=meta,
                st=st,
                rec=rec,
                rec_edits=rec_edits,
                file_overlays=file_overlays,
                stats_obj=stats_obj,
                progress_cb=progress_cb or _noop_progress,
                progress_error_cb=progress_error_cb or _noop_error,
                job_id=job_id,
                writes_by_rec=writes_by_rec,
                dry_run=dry_run,
            )
        except TypeError:
            if file_overlays:
                try:
                    edits_with_overlay = list(rec_edits) if isinstance(rec_edits, (list, tuple)) else [rec_edits]
                except Exception:
                    edits_with_overlay = [rec_edits] if rec_edits is not None else []
                edits_with_overlay = list(edits_with_overlay)
                edits_with_overlay.append({"file_overlays": file_overlays})
                try:
                    if st is not None and hasattr(st, "trace") and getattr(st, "trace") and hasattr(st.trace, "write"):
                        st.trace.write("APPLY", "apply_overlay_fallback", {"rec_id": rid, "used_edits_overlay": True})
                except Exception:
                    try:
                        debug("apply_overlay_fallback_trace_failed", meta={"rec_id": rid})
                    except Exception:
                        pass
                apply_result = _apply_and_refresh(
                    root=root,
                    kb=kb,
                    meta=meta,
                    st=st,
                    rec=rec,
                    rec_edits=edits_with_overlay,
                    stats_obj=stats_obj,
                    progress_cb=progress_cb or _noop_progress,
                    progress_error_cb=progress_error_cb or _noop_error,
                    job_id=job_id,
                    writes_by_rec=writes_by_rec,
                    dry_run=dry_run,
                )
            else:
                apply_result = _apply_and_refresh(
                    root=root,
                    kb=kb,
                    meta=meta,
                    st=st,
                    rec=rec,
                    rec_edits=rec_edits,
                    stats_obj=stats_obj,
                    progress_cb=progress_cb or _noop_progress,
                    progress_error_cb=progress_error_cb or _noop_error,
                    job_id=job_id,
                    writes_by_rec=writes_by_rec,
                    dry_run=dry_run,
                )
    except Exception as e:
        result.skipped = True
        result.reason = "apply_failed"
        result.details["error"] = str(e)

        try:
            error("apply_failed", meta={"rec_id": rid, "error": str(e), "project_root": str(root.resolve())})
        except Exception:
            pass

        if progress_error_cb:
            try:
                progress_error_cb(
                    "apply_edits",
                    {
                        "error": str(e),
                        "rec_id": rid,
                        "project_root": str(root.resolve()),
                    },
                )
            except Exception:
                try:
                    debug("progress_error_cb_failed", meta={"where": "apply_edits", "rec_id": rid})
                except Exception:
                    pass

        _events.status(
            "Recommendation failed during apply",
            where="rec_skipped",
            rec_id=rid,
            title=title,
            reason=result.reason,
            session_id=session_id,
            job_id=job_id,
            project_root=str(root.resolve()),
        )
        return result

    # Post-apply: execute queued commands ONLY if approved and not dry_run.
    commands_executed = False
    if queued_commands:
        if dry_run:
            # Don't execute; just surface preview
            result.details["run_commands"] = [{"command": c.get("command"), "skipped": True, "reason": "dry_run"} for c in queued_commands]
            result.details["run_commands_ok"] = True
            result.details["commands_executed"] = False
        else:
            cmds = _pop_pending_commands(rec)
            ok_cmds, cmd_results = _run_queued_commands(
                root=root,
                rec_id=rid,
                commands=cmds,
                dry_run=False,
                progress_cb=progress_cb,
                progress_error_cb=progress_error_cb,
                trace_obj=getattr(st, "trace", None) if st is not None else None,
            )
            commands_executed = bool(cmd_results)
            result.details["run_commands_ok"] = bool(ok_cmds)
            result.details["run_commands"] = cmd_results
            result.details["commands_executed"] = commands_executed

    try:
        raw_changed = []
        raw_refresh = None

        if isinstance(apply_result, dict):
            raw_changed = apply_result.get("changed_paths") or []
            raw_refresh = apply_result.get("refresh") if isinstance(apply_result.get("refresh"), dict) else apply_result.get("refresh")
        else:
            try:
                raw_changed = getattr(apply_result, "changed_paths", None) or getattr(apply_result, "changed_paths", [])
            except Exception:
                raw_changed = []
            try:
                raw_refresh = getattr(apply_result, "refresh", None)
            except Exception:
                raw_refresh = None

        sanitized_changed = _normalize_changed_paths(root, raw_changed, rec_id=rid)

        refresh_info = None
        if isinstance(raw_refresh, dict):
            refresh_info = dict(raw_refresh)
            refresh_info["changed_paths"] = sanitized_changed
        elif raw_refresh is not None:
            try:
                refresh_info = {"raw": raw_refresh, "changed_paths": sanitized_changed}
            except Exception:
                refresh_info = {"changed_paths": sanitized_changed}

    except Exception as e:
        try:
            error("normalize_apply_result_failed", meta={"rec_id": rid, "error": str(e)})
        except Exception:
            pass
        sanitized_changed = []
        refresh_info = None

    if dry_run:
        result.skipped = True
        result.reason = "dry_run"
        result.changed_paths = sanitized_changed
        result.details["refresh"] = refresh_info or {}
        result.details.setdefault("recent_changed_files", result.changed_paths)

        _events.status(
            "Recommendation dry-run preview",
            where="rec_dry_run",
            rec_id=rid,
            title=title,
            changed_paths=result.changed_paths,
            refresh=result.details["refresh"],
            dry_run=True,
            session_id=session_id,
            job_id=job_id,
            project_root=str(root.resolve()),
        )

        try:
            info("apply.dry_run", meta={"rec_id": rid, "changed_paths": result.changed_paths, "project_root": str(root.resolve())})
        except Exception:
            pass

        return result

    if refresh_info:
        result.details["refresh"] = refresh_info
        if not refresh_info.get("ok", True):
            _events.status(
                "Recommendation refresh failed",
                where="rec_refresh_failed",
                rec_id=rid,
                title=title,
                refresh=refresh_info,
                session_id=session_id,
                job_id=job_id,
                project_root=str(root.resolve()),
            )
    else:
        try:
            if progress_cb:
                try:
                    progress_cb("kb_refresh_start", {"rec_id": rid, "project_root": str(root.resolve())})
                except Exception:
                    pass

            if hasattr(kb, "load_card_index") and callable(getattr(kb, "load_card_index")):
                kb.load_card_index()
            elif hasattr(kb, "_ensure_fresh_index") and callable(getattr(kb, "_ensure_fresh_index")):
                kb._ensure_fresh_index()
            else:
                try:
                    debug("kb_refresh_missing_method", meta={"rec_id": rid})
                except Exception:
                    pass

            if progress_cb:
                try:
                    progress_cb("kb_refresh_done", {"rec_id": rid, "project_root": str(root.resolve())})
                except Exception:
                    pass
            result.details["kb_refresh"] = {"ok": True}
        except Exception as e:
            try:
                error("kb_refresh_failed", meta={"rec_id": rid, "error": str(e)})
            except Exception:
                pass
            result.details["kb_refresh"] = {"ok": False, "error": str(e)}

    result.changed_paths = sanitized_changed
    result.details.setdefault("recent_changed_files", result.changed_paths)

    # Applied if either files changed OR commands executed.
    result.applied = bool(result.changed_paths) or bool(result.details.get("commands_executed"))
    result.skipped = not result.applied
    if not result.applied and not result.reason:
        result.reason = "no_files_changed"
    if result.applied and not result.changed_paths and result.reason is None:
        result.reason = "commands_only"

    if result.applied:
        _events.status(
            "Recommendation applied",
            where="rec_applied",
            rec_id=rid,
            title=title,
            changed_paths=result.changed_paths,
            dry_run=False,
            session_id=session_id,
            job_id=job_id,
            project_root=str(root.resolve()),
        )
        try:
            info("rec.applied", meta={"rec_id": rid, "changed_paths": result.changed_paths, "project_root": str(root.resolve())})
        except Exception:
            pass
    else:
        _events.status(
            "Recommendation produced no changed paths",
            where="rec_skipped",
            rec_id=rid,
            title=title,
            reason=result.reason,
            changed_paths=result.changed_paths,
            session_id=session_id,
            job_id=job_id,
            project_root=str(root.resolve()),
        )
        try:
            warning("rec.no_files_changed", meta={"rec_id": rid, "reason": result.reason, "project_root": str(root.resolve())})
        except Exception:
            pass

    return result


# Optional self-test guard: when AIDEV_SELFTEST=1 in the environment, perform a small
# import-time check that _normalize_changed_paths filters globs/absolute-outside-root
# entries and preserves order/dedupe behavior. This is intentionally opt-in so tests
# don't run it unless explicitly requested by the test harness.
if os.getenv("AIDEV_SELFTEST") == "1":
    try:
        try:
            test_root = Path.cwd()
            outside_abs = str(test_root.parent.joinpath("_aidev_selftest_outside.txt"))
            test_items = ["a.txt", outside_abs, "dir/*", "a.txt", "b.txt"]
            res = _normalize_changed_paths(test_root, test_items, rec_id="selftest")
            assert all("*" not in p for p in res)
            assert all(outside_abs != (test_root / p).resolve().as_posix() for p in res)
            assert res.count("a.txt") <= 1
        except AssertionError:
            try:
                warning("selftest.normalize_changed_paths_failed", meta={"rec_id": "selftest"})
            except Exception:
                pass
    except Exception:
        try:
            debug("selftest.normalize_changed_paths_error")
        except Exception:
            pass
