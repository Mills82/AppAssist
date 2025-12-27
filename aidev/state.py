# aidev/state.py
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
import logging
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TraceLogger:
    """
    Append-only JSONL tracer at .aidev/trace.jsonl, written atomically when
    possible, with a safe fallback on platforms (e.g. Windows) where the
    trace file may be temporarily locked.

    Provides:
      - write(kind, action, payload)
      - timer(kind, action, **fields) -> context manager that logs start/finish

    Added helpers below provide best-effort edit-related trace records used
    by the tiered edit-file strategy. Helpers are defensive and never raise.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.dir = self.root / ".aidev"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trace.jsonl"

    # ---------- low level ----------

    def _atomic_append_line(self, line: str) -> None:
        """
        Best-effort atomic append using a temp file + replace, with a
        Windows-friendly fallback to non-atomic append when PermissionError
        (or other OS errors) occur.

        This is designed for single-user / single-writer scenarios (local dev,
        CI). If multiple processes hammer this file concurrently, traces may be
        interleaved but should not crash the pipeline.
        """
        # Ensure the trace directory exists.
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Directory creation failure will surface on write/replace below.
            logger.debug("TraceLogger: failed to ensure trace dir exists", exc_info=True)

        # Try atomic read+append+replace first.
        existing = ""
        if self.path.exists():
            try:
                existing = self.path.read_text(encoding="utf-8")
            except Exception:
                # If we can't read the existing trace, just treat it as empty
                # rather than failing the caller.
                logger.debug("TraceLogger: failed to read existing trace.jsonl", exc_info=True)
                existing = ""

        final = existing + line

        try:
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
                tmp.write(final)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)

            try:
                # Primary path: atomic replace.
                tmp_path.replace(self.path)
                return
            except PermissionError as e:
                # Common on Windows when another process briefly holds a handle
                # to the trace file. Fall back to non-atomic append.
                logger.warning(
                    "TraceLogger: PermissionError replacing %s -> %s: %s; "
                    "falling back to non-atomic append.",
                    tmp_path,
                    self.path,
                    e,
                )
                try:
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(line)
                except Exception:
                    logger.debug("TraceLogger: fallback append failed after PermissionError", exc_info=True)
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        logger.debug("TraceLogger: failed to remove temp trace file after PermissionError", exc_info=True)
                return
            except Exception:
                # Any other unexpected error during replace: log and fall back.
                logger.debug("TraceLogger: unexpected error during atomic replace; falling back to append", exc_info=True)
                try:
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(line)
                except Exception:
                    logger.debug("TraceLogger: fallback append failed after generic replace error", exc_info=True)
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        logger.debug("TraceLogger: failed to remove temp trace file after generic replace error", exc_info=True)
                return
        except Exception:
            # If the whole atomic path (including temp file creation) fails,
            # do a last-ditch append so we never crash callers due to tracing.
            logger.debug("TraceLogger: atomic write path failed; final fallback to append", exc_info=True)
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except Exception:
                # At this point we give up on tracing rather than raising.
                logger.debug("TraceLogger: final fallback append failed; dropping trace line", exc_info=True)

    def write(self, kind: str, action: str, payload: Dict[str, Any]) -> None:
        """
        Emit a single JSONL trace record for the given kind/action with an
        optional payload. The emitted record will always include a
        project_root field set to this TraceLogger's resolved root. If the
        caller provides project_root inside payload, it will be overwritten.

        Rationale: tracing must unambiguously record the workspace/project
        targeted by operations. Enforcing the TraceLogger's project_root here
        prevents inconsistent traces when callers run from a different CWD or
        forget to propagate the selected project/workspace.
        """
        # NOTE: project_root is authoritative and always set from this TraceLogger's root.
        # Even if caller provides a project_root in payload, it will be overwritten so audits can trust this field.
        rec = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "kind": kind,
            "action": action,
            **(payload or {}),
        }
        # Ensure the project_root in emitted traces is always the resolved root of this TraceLogger.
        rec["project_root"] = str(self.root)

        # model normalization for auditability/tests:
        # Some callers embed the resolved LLM model under nested keys (e.g. payload.model or payload.data.model).
        # Promote any discovered model to a top-level `model` field so traces and SSE events consistently expose it.
        try:
            if not rec.get("model"):
                model_candidate = None
                if isinstance(payload, dict):
                    # Prefer top-level payload['model'] (already merged into rec, but check payload directly for clarity)
                    if payload.get("model"):
                        model_candidate = payload.get("model")
                    else:
                        for container_key in ("payload", "data"):
                            inner = payload.get(container_key)
                            if isinstance(inner, dict) and inner.get("model"):
                                model_candidate = inner.get("model")
                                break
                if model_candidate:
                    try:
                        rec["model"] = str(model_candidate)
                    except Exception:
                        # Best-effort only: do not allow model normalization to raise.
                        logger.debug("TraceLogger: failed to stringify model candidate", exc_info=True)
        except Exception:
            # Swallow any unexpected errors from normalization to preserve tracing best-effort behavior.
            logger.debug("TraceLogger: model normalization failed", exc_info=True)

        line = json.dumps(rec, ensure_ascii=False) + "\n"
        self._atomic_append_line(line)

    # ---------- edit-event helpers (best-effort) ----------

    def log_edit_attempt_start(self, file_path: str, attempt_number: int, model: str) -> None:
        """
        Writes kind='edit', action='edit_attempt_start' with payload:
        { file_path, attempt_number, model }

        This helper is best-effort and will swallow/log any exceptions so it
        never interferes with the edit pipeline.
        """
        try:
            payload = {
                "file_path": file_path,
                "attempt_number": int(attempt_number),
                "model": str(model) if model is not None else None,
            }
            self.write("edit", "edit_attempt_start", payload)
        except Exception:
            logger.debug("TraceLogger: log_edit_attempt_start failed", exc_info=True)

    def log_edit_attempt_result(
        self,
        file_path: str,
        attempt_number: int,
        model: str,
        output_type: str,
        ok: bool,
        last_error: Optional[str] = None,
        classifier_label: Optional[str] = None,
    ) -> None:
        """
        Writes kind='edit', action='edit_attempt_result' with payload:
        { file_path, attempt_number, model, output_type, ok, last_error, classifier_label }

        `output_type` is expected to be one of: 'patch_unified', 'content', 'none'.
        """
        try:
            payload: Dict[str, Any] = {
                "file_path": file_path,
                "attempt_number": int(attempt_number),
                "model": str(model) if model is not None else None,
                "output_type": output_type,
                "ok": bool(ok),
            }
            if last_error:
                payload["last_error"] = str(last_error)
            if classifier_label:
                payload["classifier_label"] = str(classifier_label)
            self.write("edit", "edit_attempt_result", payload)
        except Exception:
            logger.debug("TraceLogger: log_edit_attempt_result failed", exc_info=True)

    def log_patch_apply_failed(self, file_path: str, attempt_number: int, model: str, error_text: str, classifier_label: str) -> None:
        """
        Writes kind='edit', action='patch_apply_failed' with payload including:
        { file_path, attempt_number, model, error_text, classifier_label }
        """
        try:
            payload = {
                "file_path": file_path,
                "attempt_number": int(attempt_number),
                "model": str(model) if model is not None else None,
                "error_text": str(error_text),
                "classifier_label": str(classifier_label),
            }
            self.write("edit", "patch_apply_failed", payload)
        except Exception:
            logger.debug("TraceLogger: log_patch_apply_failed failed", exc_info=True)

    def log_fallback_full_content(self, file_path: str, attempt_number: int, model: str) -> None:
        """
        Writes kind='edit', action='fallback_full_content' to mark transition to attempt #2.
        Payload: { file_path, attempt_number, model }
        """
        try:
            payload = {
                "file_path": file_path,
                "attempt_number": int(attempt_number),
                "model": str(model) if model is not None else None,
            }
            self.write("edit", "fallback_full_content", payload)
        except Exception:
            logger.debug("TraceLogger: log_fallback_full_content failed", exc_info=True)

    def log_edit_finalized(self, file_path: str, success: bool, final_type: str, attempts_summary: List[Dict[str, Any]]) -> None:
        """
        Writes kind='edit', action='edit_finalized' with payload:
        { file_path, success, final_type, attempts_summary }

        attempts_summary should be a compact list of per-attempt dicts with keys:
        attempt_number, model, output_type, ok, classifier_label (optional), last_error (optional)
        """
        try:
            payload = {
                "file_path": file_path,
                "success": bool(success),
                "final_type": final_type,
                "attempts_summary": attempts_summary,
            }
            self.write("edit", "edit_finalized", payload)
        except Exception:
            logger.debug("TraceLogger: log_edit_finalized failed", exc_info=True)

    # ---------- high level ----------

    class _Timer(ContextDecorator):
        def __init__(self, parent: "TraceLogger", kind: str, action: str, **fields: Any) -> None:
            self.parent = parent
            self.kind = kind
            self.action = action
            self.fields = dict(fields or {})
            self._t0 = 0.0

        def __enter__(self):
            self._t0 = time.perf_counter()
            self.parent.write(self.kind, self.action, {**self.fields, "event": "start"})
            return self

        def __exit__(self, exc_type, exc, tb):
            dt_ms = max(0.0, (time.perf_counter() - self._t0) * 1000.0)
            payload = {
                **self.fields,
                "event": "finish",
                "duration_ms": round(dt_ms, 2),
            }
            if exc is not None:
                payload["ok"] = False
                payload["error"] = str(exc)
                try:
                    payload["traceback"] = "".join(
                        traceback.format_exception(exc_type, exc, tb)
                    )[-4000:]
                except Exception:
                    pass
                self.parent.write(self.kind, self.action, payload)
                # re-raise so callers still see the exception
                return False
            else:
                payload["ok"] = True
                self.parent.write(self.kind, self.action, payload)
                return False

    def timer(self, kind: str, action: str, **fields: Any) -> "_Timer":
        """Usage: with trace.timer('llm_plan','focus_card', phase='plan'): ..."""
        return TraceLogger._Timer(self, kind, action, **fields)


@dataclass
class ProjectState:
    project_root: Path
    env: Dict[str, Any] = field(default_factory=dict)
    attempt_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        self.trace = TraceLogger(self.project_root)

    # ---------- attempt-history helpers ----------

    def record_edit_attempt(self, file_path: str, attempt_meta: Dict[str, Any]) -> None:
        """
        Append attempt_meta to in-memory attempt_history for file_path and emit
        corresponding edit_attempt_start and edit_attempt_result trace records.

        attempt_meta is expected to contain at least:
          - attempt_number (int)
          - model (str)
          - output_type (str)  # 'patch_unified'|'content'|'none'
          - ok (bool)
        Optional keys:
          - classifier_label (str)
          - last_error (str)

        This method is defensive and will not raise if tracing fails.
        """
        try:
            fh = self.attempt_history.setdefault(file_path, [])
            # Normalize stored entry to known keys so downstream consumers can rely on shape.
            entry: Dict[str, Any] = {
                "attempt_number": int(attempt_meta.get("attempt_number", len(fh) + 1)),
                "model": str(attempt_meta.get("model")) if attempt_meta.get("model") is not None else None,
                "output_type": str(attempt_meta.get("output_type", "none")),
                "ok": bool(attempt_meta.get("ok", False)),
            }
            if attempt_meta.get("classifier_label") is not None:
                entry["classifier_label"] = str(attempt_meta.get("classifier_label"))
            if attempt_meta.get("last_error") is not None:
                entry["last_error"] = str(attempt_meta.get("last_error"))

            fh.append(entry)

            # Emit start + result traces. Start uses the attempt_number and model.
            try:
                self.trace.log_edit_attempt_start(file_path, entry["attempt_number"], entry["model"])
            except Exception:
                # log_edit_attempt_start is already defensive, but be extra paranoid
                logger.debug("ProjectState: trace.log_edit_attempt_start threw", exc_info=True)

            try:
                # Map keys to the trace helper signature (last_error/classifier_label optional)
                self.trace.log_edit_attempt_result(
                    file_path,
                    entry["attempt_number"],
                    entry["model"],
                    entry["output_type"],
                    entry["ok"],
                    last_error=entry.get("last_error"),
                    classifier_label=entry.get("classifier_label"),
                )
            except Exception:
                logger.debug("ProjectState: trace.log_edit_attempt_result threw", exc_info=True)
        except Exception:
            # Never let tracing/storage failure bubble out to callers of the edit pipeline.
            logger.debug("ProjectState.record_edit_attempt failed", exc_info=True)

    def record_patch_failure(self, file_path: str, attempt_number: int, model: str, error_text: str, classifier_label: str) -> None:
        """
        Record a patch application failure for file_path and emit a
        patch_apply_failed trace record. Also append a corresponding
        attempt entry (ok=False) to attempt_history.

        Stored entry keys: attempt_number, model, output_type='patch_unified', ok=False, classifier_label, last_error
        """
        try:
            entry = {
                "attempt_number": int(attempt_number),
                "model": str(model) if model is not None else None,
                "output_type": "patch_unified",
                "ok": False,
                "classifier_label": str(classifier_label),
                "last_error": str(error_text),
            }
            self.attempt_history.setdefault(file_path, []).append(entry)

            try:
                self.trace.log_patch_apply_failed(file_path, attempt_number, model, error_text, classifier_label)
            except Exception:
                logger.debug("ProjectState: trace.log_patch_apply_failed threw", exc_info=True)

            # Also emit an edit_attempt_result for consistency with attempts timeline
            try:
                self.trace.log_edit_attempt_result(
                    file_path,
                    attempt_number,
                    model,
                    "patch_unified",
                    False,
                    last_error=error_text,
                    classifier_label=classifier_label,
                )
            except Exception:
                logger.debug("ProjectState: trace.log_edit_attempt_result (after patch failure) threw", exc_info=True)
        except Exception:
            logger.debug("ProjectState.record_patch_failure failed", exc_info=True)

    def finalize_edit(self, file_path: str, success: bool, final_type: str) -> None:
        """
        Emit a single edit_finalized trace record with aggregated attempt
        history for file_path. After emitting the finalized record, the in-memory
        attempt history for that file is cleared (best-effort).

        final_type is expected to be one of: 'patch_unified', 'content', 'none'.
        """
        try:
            attempts_summary = list(self.attempt_history.get(file_path, []))
            try:
                self.trace.log_edit_finalized(file_path, success, final_type, attempts_summary)
            except Exception:
                logger.debug("ProjectState: trace.log_edit_finalized threw", exc_info=True)

            # Clear in-memory history for this file to avoid unbounded growth.
            try:
                if file_path in self.attempt_history:
                    del self.attempt_history[file_path]
            except Exception:
                logger.debug("ProjectState: failed to clear attempt_history for %s", file_path, exc_info=True)
        except Exception:
            logger.debug("ProjectState.finalize_edit failed", exc_info=True)


__all__ = ["TraceLogger", "ProjectState"]
