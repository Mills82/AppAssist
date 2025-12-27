# aidev/events_bus.py
from __future__ import annotations
import asyncio
import logging
import os
import json
import datetime
from typing import Callable, Awaitable, Optional, Any
from .validators import redact_secrets


# Lazy / resilient logging helpers: import structured aidev.logger only when available.
# This avoids importing other modules at package-import time (which can surface
# unrelated syntax/import errors) and provides a sensible stdlib fallback.
_logger_functions = None

def _ensure_logger_funcs():
    global _logger_functions
    if _logger_functions is None:
        try:
            from .logger import info as _info, debug as _debug, warning as _warning, error as _error

            # Wrap to accept meta kwarg uniformly (some callers pass meta=...)
            def _wrap(fn):
                def _w(msg: str, meta: Optional[dict] = None, **kwargs):
                    # If the structured logger accepts meta, pass it; else include meta in the message.
                    try:
                        return fn(msg, meta=meta, **kwargs)
                    except TypeError:
                        if meta:
                            return fn(f"{msg} | meta={meta}")
                        return fn(msg)
                return _w

            _logger_functions = (_wrap(_info), _wrap(_debug), _wrap(_warning), _wrap(_error))
        except Exception:
            # Fallback to standard library logging. Include meta in the message when present.
            std = logging.getLogger("aidev.events_bus")

            def _std_info(msg: str, meta: Optional[dict] = None, **_kwargs):
                if meta:
                    std.info(f"{msg} | meta={meta}")
                else:
                    std.info(msg)

            def _std_debug(msg: str, meta: Optional[dict] = None, **_kwargs):
                if meta:
                    std.debug(f"{msg} | meta={meta}")
                else:
                    std.debug(msg)

            def _std_warning(msg: str, meta: Optional[dict] = None, **_kwargs):
                if meta:
                    std.warning(f"{msg} | meta={meta}")
                else:
                    std.warning(msg)

            def _std_error(msg: str, meta: Optional[dict] = None, **_kwargs):
                if meta:
                    std.error(f"{msg} | meta={meta}")
                else:
                    std.error(msg)

            _logger_functions = (_std_info, _std_debug, _std_warning, _std_error)
    return _logger_functions


def info(msg: str, meta: Optional[dict] = None, **kwargs):
    return _ensure_logger_funcs()[0](msg, meta=meta, **kwargs)


def debug(msg: str, meta: Optional[dict] = None, **kwargs):
    return _ensure_logger_funcs()[1](msg, meta=meta, **kwargs)


def warning(msg: str, meta: Optional[dict] = None, **kwargs):
    return _ensure_logger_funcs()[2](msg, meta=meta, **kwargs)


def error(msg: str, meta: Optional[dict] = None, **kwargs):
    return _ensure_logger_funcs()[3](msg, meta=meta, **kwargs)


class Emitter:
    """Thread-safe event emitter for session queues.

    This emitter will normally forward redacted events into the provided
    coroutine queue. In addition, when an "applied_changes" event is
    emitted it records a JSONL trace entry to .aidev/trace.jsonl and writes a
    human-readable line to app.log before forwarding the event. If the
    recording step fails the event will not be forwarded and an exception
    will be raised so callers (the apply stage) can surface the failure.

    For Deep Research phases 0–2 (and beyond), use the dedicated helpers:
      - emit_deep_phase_start(...)
      - emit_deep_phase_done(...)
      - emit_cache_hit(...)
      - emit_cache_miss(...)
      - emit_artifact_written(...)
      - emit_budget_update(...)

    These helpers normalize project_root and repo-relative refs, redact secrets,
    write a trace.jsonl record (best-effort), and forward a stable event envelope
    (event/type + payload) to the session queue.
    """

    # Hard bounds to keep event payloads small and JSON-serializable.
    _MAX_STRING_LEN = 4096
    _MAX_LIST_ITEMS = 200
    _MAX_DICT_ITEMS = 200
    _MAX_DEPTH = 5

    def __init__(self, loop: asyncio.AbstractEventLoop, put_coro: Callable[[dict], Awaitable[None]]):
        self._loop = loop
        self._put_coro = put_coro

    # ----------------------------
    # Deep Research helper methods
    # ----------------------------

    def emit_deep_phase_start(
        self,
        *,
        run_id: str,
        phase: str,
        project_root: Optional[str] = None,
        cache_key: Optional[str] = None,
        cache_ref: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Emit and trace a Deep Research phase start.

        Minimal payload recorded/forwarded:
          {
            "run_id": "...",
            "phase": "preflight"|"plan"|"gather"|...,
            "cache_key": "..." (optional),
            "cache_ref": "..." (optional)
          }
        """
        self._emit_deep_research_event(
            event="deep_research.phase_started",
            run_id=run_id,
            phase=phase,
            project_root=project_root,
            cache_key=cache_key,
            cache_ref=cache_ref,
            payload=payload,
        )

    def emit_deep_phase_done(
        self,
        *,
        run_id: str,
        phase: str,
        project_root: Optional[str] = None,
        cache_key: Optional[str] = None,
        cache_ref: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Emit and trace a Deep Research phase completion."""
        self._emit_deep_research_event(
            event="deep_research.phase_done",
            run_id=run_id,
            phase=phase,
            project_root=project_root,
            cache_key=cache_key,
            cache_ref=cache_ref,
            payload=payload,
        )

    def emit_cache_hit(
        self,
        *,
        run_id: str,
        phase: Optional[str] = None,
        artifact_type: str,
        cache_key: Optional[str] = None,
        cache_ref: Optional[str] = None,
        cache_path: Optional[str] = None,
        artifact_path: Optional[str] = None,
        project_root: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Emit and trace a Deep Research cache hit.

        The event includes artifact_type and either cache_ref or cache_path.
        Any provided paths are normalized to repo-relative refs.
        """
        self._emit_deep_research_event(
            event="deep_research.cache_hit",
            run_id=run_id,
            phase=phase,
            project_root=project_root,
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            cache_key=cache_key,
            cache_ref=cache_ref,
            cache_path=cache_path,
            payload=payload,
        )

    def emit_cache_miss(
        self,
        *,
        run_id: str,
        phase: Optional[str] = None,
        artifact_type: str,
        cache_key: Optional[str] = None,
        cache_ref: Optional[str] = None,
        cache_path: Optional[str] = None,
        artifact_path: Optional[str] = None,
        project_root: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Emit and trace a Deep Research cache miss."""
        self._emit_deep_research_event(
            event="deep_research.cache_miss",
            run_id=run_id,
            phase=phase,
            project_root=project_root,
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            cache_key=cache_key,
            cache_ref=cache_ref,
            cache_path=cache_path,
            payload=payload,
        )

    def emit_artifact_written(
        self,
        *,
        run_id: str,
        phase: Optional[str] = None,
        artifact_type: str,
        artifact_path: str,
        project_root: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Emit and trace an artifact write for Deep Research.

        artifact_path must be repo-relative (or will be rejected if absolute).
        """
        self._emit_deep_research_event(
            event="deep_research.artifact_written",
            run_id=run_id,
            phase=phase,
            project_root=project_root,
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            payload=payload,
        )

    def emit_budget_update(
        self,
        *,
        run_id: str,
        phase: Optional[str] = None,
        budget_before: Any,
        budget_after: Any,
        reason: str,
        project_root: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Emit and trace a budget update.

        Includes before/after (numeric) and reason (short string).
        """
        # Validate numeric types (accept ints/floats or strings convertible to float).
        bb = self._coerce_number(budget_before, field_name="budget_before")
        ba = self._coerce_number(budget_after, field_name="budget_after")

        self._emit_deep_research_event(
            event="deep_research.budget_update",
            run_id=run_id,
            phase=phase,
            project_root=project_root,
            budget_before=bb,
            budget_after=ba,
            reason=reason,
            payload=payload,
        )

    # ----------------------------
    # AI summary helper methods
    # ----------------------------

    def emit_ai_summary_run(
        self,
        *,
        run_id: str,
        project_root: Optional[str] = None,
        total: int = 0,
        successes: int = 0,
        failures: Optional[list] = None,
    ) -> None:
        """Emit a run-level summary for AI summary card generation.

        Callers should invoke this once after per-file summarize tasks complete.
        This normalizes and sorts failure paths deterministically, writes a single
        trace.jsonl record (best-effort), writes a concise app.log line (best-effort),
        and forwards a stable event envelope to the session queue.
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id is required")

        # Normalize project_root; do not include absolute project_root in forwarded/trace payloads.
        pr = self._normalize_project_root({"run_id": run_id, "event": "ai_summary.run"}, project_root)

        # Coerce totals to ints (defensive, but keep strictness for invalid inputs).
        try:
            total_i = int(total)
        except Exception as e:
            raise ValueError("total must be an int") from e
        try:
            successes_i = int(successes)
        except Exception as e:
            raise ValueError("successes must be an int") from e

        raw_failures = failures or []
        if not isinstance(raw_failures, list):
            raise ValueError("failures must be a list of objects with at least 'path' and 'error'")

        normalized_failures: list[dict[str, str]] = []
        for i, item in enumerate(raw_failures):
            if not isinstance(item, dict):
                raise ValueError(f"failures[{i}] must be a dict")
            p = item.get("path")
            err = item.get("error")
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"failures[{i}].path must be a non-empty string")
            # Normalize to repo-relative path; reject absolute/escaping paths.
            norm_p = self._repo_relative(pr, p, field_name="failures[].path")
            if err is None:
                err_s = ""
            elif isinstance(err, str):
                err_s = err
            else:
                err_s = str(err)
            normalized_failures.append({"path": norm_p, "error": err_s})

        # Deterministic ordering regardless of task completion timing.
        normalized_failures.sort(key=lambda d: d.get("path", ""))

        payload = {
            "run_id": run_id,
            "total": total_i,
            "successes": successes_i,
            "failures": normalized_failures,
        }

        # Redact secrets and bound size.
        redacted = redact_secrets({"event": "ai_summary.run", "payload": payload})
        redacted_payload = redacted.get("payload") if isinstance(redacted, dict) else payload
        if not isinstance(redacted_payload, dict):
            redacted_payload = payload
        bounded_payload = self._trim_and_summarize(redacted_payload)

        # Best-effort trace and human log writes; do not raise on IO failures.
        try:
            self._write_trace_record_best_effort(project_root=pr, event="ai_summary.run", payload=bounded_payload)
        except Exception:
            pass

        try:
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            app_log_path = os.path.join(pr, "app.log")
            failures_count = len(normalized_failures)
            human_line = (
                f"{ts} - AI summary run: run_id={run_id} total={total_i} successes={successes_i} "
                f"failures={failures_count}\n"
            )
            with open(app_log_path, "a", encoding="utf-8") as lf:
                lf.write(human_line)
                lf.flush()
                try:
                    os.fsync(lf.fileno())
                except Exception:
                    pass
        except Exception as e:
            error(
                "failed to write app.log entry for ai_summary.run",
                meta={
                    "module": "events_bus",
                    "event": "ai_summary.run",
                    "run_id": run_id,
                    "app_log_path": os.path.join(pr, "app.log"),
                    "exc": str(e),
                },
            )

        forwarded = {
            "event": "ai_summary.run",
            "type": "ai_summary.run",
            "payload": bounded_payload,
            "run_id": run_id,
        }
        self._forward_event(forwarded)

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _coerce_number(self, value: Any, *, field_name: str) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            v = value.strip()
            if not v:
                raise ValueError(f"{field_name} must be a number")
            try:
                return float(v)
            except Exception as e:
                raise ValueError(f"{field_name} must be a number") from e
        raise ValueError(f"{field_name} must be a number")

    def _normalize_project_root(self, event_dict: dict, project_root: Optional[str]) -> str:
        # Determine project_root from explicit arg or event/payload candidates; otherwise cwd.
        if isinstance(project_root, str) and project_root.strip():
            return os.path.abspath(os.path.normpath(project_root))

        sanitized = event_dict
        root = None
        for key in ("project_root", "project_path", "repo_root"):
            if key in sanitized:
                cand = sanitized.get(key)
                if isinstance(cand, str) and cand.strip():
                    root = os.path.abspath(os.path.normpath(cand))
                else:
                    debug("ignored invalid project_root candidate in top-level event", meta={"module": "events_bus", "candidate_key": key})
                break
        if root is None and isinstance(sanitized.get("payload"), dict):
            for key in ("project_root", "project_path", "repo_root"):
                if key in sanitized["payload"]:
                    cand = sanitized["payload"].get(key)
                    if isinstance(cand, str) and cand.strip():
                        root = os.path.abspath(os.path.normpath(cand))
                    else:
                        debug("ignored invalid project_root candidate in payload", meta={"module": "events_bus", "candidate_key": key})
                    break

        if not root:
            root = os.path.abspath(os.getcwd())
        return root

    def _repo_relative(self, project_root: str, p: str, *, field_name: str = "path") -> str:
        if not isinstance(p, str) or not p.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        if os.path.isabs(p):
            raise ValueError(f"absolute paths are not allowed for {field_name}: {p}")

        repo_root_norm = os.path.normpath(project_root)
        prefix = repo_root_norm if repo_root_norm.endswith(os.sep) else repo_root_norm + os.sep
        joined = os.path.normpath(os.path.join(project_root, p))
        if not (joined == repo_root_norm or joined.startswith(prefix)):
            raise ValueError(f"{field_name} escapes repository root: {p}")

        rel = os.path.relpath(joined, project_root)
        return rel.replace("\\", "/")

    def _trim_and_summarize(self, value: Any, *, depth: int = 0) -> Any:
        # Keep values JSON-serializable and bounded.
        if depth > self._MAX_DEPTH:
            return "<truncated: depth>"

        if value is None:
            return None
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            s = value
            if len(s) > self._MAX_STRING_LEN:
                return s[: self._MAX_STRING_LEN] + f"...<truncated:{len(s)} chars>"
            return s
        if isinstance(value, (list, tuple)):
            out = []
            for i, item in enumerate(value):
                if i >= self._MAX_LIST_ITEMS:
                    out.append(f"<truncated:{len(value)} items>")
                    break
                out.append(self._trim_and_summarize(item, depth=depth + 1))
            return out
        if isinstance(value, dict):
            out = {}
            # deterministic iteration for stability
            for i, (k, v) in enumerate(list(value.items())):
                if i >= self._MAX_DICT_ITEMS:
                    out["<truncated>"] = f"{len(value)} items"
                    break
                # ensure key is a string
                key = k if isinstance(k, str) else str(k)
                out[key] = self._trim_and_summarize(v, depth=depth + 1)
            return out

        # Fallback: stringify unknown types.
        s = str(value)
        if len(s) > self._MAX_STRING_LEN:
            return s[: self._MAX_STRING_LEN] + f"...<truncated:{len(s)} chars>"
        return s

    def _write_trace_record_best_effort(self, *, project_root: str, event: str, payload: dict) -> None:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trace_dir = os.path.join(project_root, ".aidev")
        try:
            os.makedirs(trace_dir, exist_ok=True)
        except Exception as e:
            error("failed to create trace dir", meta={"module": "events_bus", "event": event, "trace_dir": trace_dir, "exc": str(e)})
            return

        record = {
            "ts": ts,
            "event": event,
            "payload": payload,
        }

        trace_path = os.path.join(trace_dir, "trace.jsonl")
        try:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            info("wrote trace event", meta={"module": "events_bus", "event": event, "project_root": project_root})
        except Exception as e:
            error("failed to append trace record", meta={
                "module": "events_bus",
                "event": event,
                "project_root": project_root,
                "trace_path": trace_path,
                "exc": str(e),
            })

    def _forward_event(self, forwarded: dict) -> None:
        try:
            asyncio.run_coroutine_threadsafe(self._put_coro(forwarded), self._loop)
        except RuntimeError as e:
            debug("emit after loop closed", meta={"module": "events_bus", "reason": str(e)})

    def _emit_deep_research_event(
        self,
        *,
        event: str,
        run_id: str,
        project_root: Optional[str] = None,
        phase: Optional[str] = None,
        cache_key: Optional[str] = None,
        cache_ref: Optional[str] = None,
        cache_path: Optional[str] = None,
        artifact_type: Optional[str] = None,
        artifact_path: Optional[str] = None,
        budget_before: Optional[float] = None,
        budget_after: Optional[float] = None,
        reason: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        # Build a canonical envelope that remains JSON-serializable and bounded.
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id is required")
        if event in ("deep_research.phase_started", "deep_research.phase_done"):
            if not isinstance(phase, str) or not phase.strip():
                raise ValueError("phase is required for phase events")
        if event == "deep_research.budget_update":
            if budget_before is None or budget_after is None:
                raise ValueError("budget_before and budget_after are required for budget_update")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("reason is required for budget_update")

        # Start with explicit fields; add optional payload (small dict) under payload.extra
        base_payload: dict[str, Any] = {
            "run_id": run_id,
        }
        if phase is not None:
            base_payload["phase"] = phase
        if cache_key is not None:
            base_payload["cache_key"] = cache_key
        if cache_ref is not None:
            base_payload["cache_ref"] = cache_ref
        if artifact_type is not None:
            base_payload["artifact_type"] = artifact_type
        if budget_before is not None:
            base_payload["budget_before"] = budget_before
        if budget_after is not None:
            base_payload["budget_after"] = budget_after
        if reason is not None:
            base_payload["reason"] = reason

        # Prepare an event dict for project_root normalization.
        event_dict = {"event": event, "payload": base_payload, "run_id": run_id, "phase": phase}
        project_root_norm = self._normalize_project_root(event_dict, project_root)
        # Do NOT include absolute project_root in trace or forwarded payloads to avoid leaking local paths.
        # Instead, record a non-sensitive repo identifier when helpful (repo basename), or omit entirely.
        repo_name = os.path.basename(project_root_norm) if project_root_norm else ""
        if repo_name:
            base_payload["repo_name"] = repo_name

        # Normalize any paths/refs to repo-relative; reject absolute.
        if cache_path is not None:
            base_payload["cache_path"] = self._repo_relative(project_root_norm, cache_path, field_name="cache_path")
        if artifact_path is not None:
            base_payload["artifact_path"] = self._repo_relative(project_root_norm, artifact_path, field_name="artifact_path")

        # Add optional extra payload (bounded). Avoid known large keys by default.
        extra = payload if isinstance(payload, dict) else None
        if extra:
            extra = dict(extra)
            for forbidden in (
                "content",
                "full_content",
                "file_content",
                "artifact_blob",
                "raw",
                "response_text",
                "prompt",
            ):
                if forbidden in extra:
                    # Replace with a tiny summary to avoid leaking large text.
                    v = extra.get(forbidden)
                    if isinstance(v, str):
                        extra[forbidden] = f"<omitted:{len(v)} chars>"
                    else:
                        extra[forbidden] = "<omitted>"

            # If extra includes paths, normalize common keys.
            for key in ("artifact_path", "cache_path"):
                if key in extra and isinstance(extra.get(key), str):
                    extra[key] = self._repo_relative(project_root_norm, extra[key], field_name=key)

            base_payload["extra"] = extra

        # Redact secrets and bound size.
        redacted = redact_secrets({"event": event, "payload": base_payload})
        redacted_payload = redacted.get("payload") if isinstance(redacted, dict) else base_payload
        if not isinstance(redacted_payload, dict):
            redacted_payload = base_payload
        bounded_payload = self._trim_and_summarize(redacted_payload)

        # Best-effort trace write; do not raise if it fails (per recommendation + constraints).
        try:
            self._write_trace_record_best_effort(project_root=project_root_norm, event=event, payload=bounded_payload)
        except Exception:
            # _write_trace_record_best_effort is already defensive, but keep this ultra-safe.
            pass

        # Forward stable envelope.
        forwarded = {
            "event": event,
            "type": event,
            "payload": bounded_payload,
            "run_id": run_id,
        }
        if phase is not None:
            forwarded["phase"] = phase
        self._forward_event(forwarded)

    def emit(self, evt: dict) -> None:
        sanitized = redact_secrets(evt)

        # Detect applied_changes events by common keys used for event name/type.
        evt_name = None
        for k in ("type", "event", "name", "evt"):
            if k in sanitized:
                evt_name = sanitized.get(k)
                break

        if evt_name == "applied_changes":
            # Attempt to write structured trace and a human-readable log entry
            try:
                # Normalize identifiers: prefer explicit run_id or change_id
                run_id = sanitized.get("run_id")
                change_id = sanitized.get("change_id") or sanitized.get("rec_id")

                # Extract applied IDs list if present, supporting several common keys.
                applied_ids = None
                for k in ("applied_ids", "applied_rec_ids", "applied_recommendation_ids", "applied"):
                    if k in sanitized:
                        applied_ids = sanitized.get(k)
                        break
                # Also accept applied ids nested under payload
                if applied_ids is None and isinstance(sanitized.get("payload"), dict):
                    for k in ("applied_ids", "applied_rec_ids", "applied"):
                        if k in sanitized["payload"]:
                            applied_ids = sanitized["payload"].get(k)
                            break

                # Normalize applied_ids: allow single string -> list, allow missing when change_id present
                if applied_ids is None:
                    if change_id:
                        applied_ids = [change_id]
                    elif run_id:
                        # run-level event without specific recommendation ids: normalize to empty list
                        applied_ids = []
                    else:
                        raise ValueError("applied_changes event must include a non-empty run_id, change_id, or applied_ids")

                # Validate applied_ids
                if applied_ids is None:
                    applied_ids = []
                if isinstance(applied_ids, str):
                    applied_ids = [applied_ids]
                if not isinstance(applied_ids, (list, tuple)):
                    raise ValueError("applied_ids must be a list of strings")
                for aid in applied_ids:
                    if not isinstance(aid, str):
                        raise ValueError(f"invalid applied_id (not a string): {aid!r}")

                # Accept several common keys for changed paths
                raw_paths = sanitized.get("changed_paths") or sanitized.get("paths") or sanitized.get("changed_files") or []
                if raw_paths is None:
                    raw_paths = []
                if not isinstance(raw_paths, (list, tuple)):
                    raise ValueError("changed_paths/paths must be a list of repo-relative paths")

                # Determine project_root from event or payload candidates (prefer explicit project selection)
                project_root = None
                # Check top-level keys first
                for key in ("project_root", "project_path", "repo_root"):
                    if key in sanitized:
                        cand = sanitized.get(key)
                        if isinstance(cand, str) and cand.strip():
                            project_root = os.path.abspath(os.path.normpath(cand))
                        else:
                            # Structured debug for ignored candidate
                            debug("ignored invalid project_root candidate in top-level event", meta={"module": "events_bus", "candidate_key": key})
                        break
                # If not found on top-level, inspect payload dict
                if project_root is None and isinstance(sanitized.get("payload"), dict):
                    for key in ("project_root", "project_path", "repo_root"):
                        if key in sanitized["payload"]:
                            cand = sanitized["payload"].get(key)
                            if isinstance(cand, str) and cand.strip():
                                project_root = os.path.abspath(os.path.normpath(cand))
                            else:
                                debug("ignored invalid project_root candidate in payload", meta={"module": "events_bus", "candidate_key": key})
                            break

                # Fall back to current working directory if no valid project_root provided
                if not project_root:
                    project_root = os.path.abspath(os.getcwd())

                # Normalize and validate paths to repo-relative paths that do not escape the project_root
                repo_root = project_root

                validated_paths = []
                repo_root_norm = os.path.normpath(repo_root)
                prefix = repo_root_norm if repo_root_norm.endswith(os.sep) else repo_root_norm + os.sep
                for p in raw_paths:
                    if not isinstance(p, str):
                        raise ValueError(f"invalid path entry (not a string): {p!r}")
                    if os.path.isabs(p):
                        # Reject absolute paths to ensure all paths are repo-relative
                        raise ValueError(f"absolute paths are not allowed in applied_changes payload: {p}")
                    joined = os.path.normpath(os.path.join(repo_root, p))
                    if not (joined == repo_root_norm or joined.startswith(prefix)):
                        raise ValueError(f"path escapes repository root: {p}")
                    rel = os.path.relpath(joined, repo_root)
                    validated_paths.append(rel.replace("\\", "/"))

                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

                # Ensure .aidev directory exists under the selected project_root
                trace_dir = os.path.join(repo_root, ".aidev")
                os.makedirs(trace_dir, exist_ok=True)

                # Build the structured trace record with the required shape
                payload = {
                    "project_root": project_root,
                    "paths": validated_paths,
                }
                # Include run_id and change_id if present
                if run_id:
                    payload["run_id"] = run_id
                if change_id:
                    payload["change_id"] = change_id
                # Include explicitly provided applied recommendation ids (only these are recorded)
                payload["applied_ids"] = list(applied_ids)

                record = {
                    "ts": ts,
                    "event": "applied_changes",
                    "payload": payload,
                }

                # Emit a small structured info that we are about to write the trace (keep it concise)
                info("writing applied_changes trace", meta={
                    "module": "events_bus",
                    "event": "applied_changes",
                    "run_id": run_id,
                    "change_id": change_id,
                    "paths_count": len(validated_paths),
                })

                # Append JSONL record to .aidev/trace.jsonl atomically (write + flush + fsync)
                trace_path = os.path.join(trace_dir, "trace.jsonl")
                try:
                    with open(trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False))
                        f.write("\n")
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            # If fsync is not available on the platform, ignore.
                            pass
                except Exception as e:
                    # Structured error; preserve raising so callers can react
                    error("failed to append trace record", meta={
                        "module": "events_bus",
                        "event": "applied_changes",
                        "run_id": run_id,
                        "change_id": change_id,
                        "trace_path": trace_path,
                        "exc": str(e),
                    })
                    raise

                # Write a human-readable line to app.log at project root (so it's easy to find)
                app_log_path = os.path.join(repo_root, "app.log")
                id_display = run_id or change_id or ""
                human_line = f"{ts} - Applied changes: id={id_display} applied_ids={payload['applied_ids']} paths={validated_paths} project_root={project_root}\n"
                try:
                    with open(app_log_path, "a", encoding="utf-8") as lf:
                        lf.write(human_line)
                        lf.flush()
                        try:
                            os.fsync(lf.fileno())
                        except Exception:
                            pass
                except Exception as e:
                    error("failed to write app.log entry", meta={
                        "module": "events_bus",
                        "event": "applied_changes",
                        "run_id": run_id,
                        "change_id": change_id,
                        "app_log_path": app_log_path,
                        "exc": str(e),
                    })
                    raise

                # All recording succeeded; forward a standardized event into the session queue
                try:
                    # Preserve other non-sensitive top-level fields from the original event,
                    # but ensure we use the normalized payload and canonical event name. This helps
                    # the consumers (frontends / SSE handlers) rely on a stable envelope while
                    # keeping the normalized 'payload' contents we recorded.
                    forwarded = dict(sanitized) if isinstance(sanitized, dict) else {}
                    forwarded["payload"] = payload
                    forwarded["event"] = "applied_changes"
                    # Ensure 'type' is present for consumers that route on either 'type' or 'event'.
                    # Keep existing 'type' if provided; otherwise mirror the chosen 'event' value.
                    forwarded["type"] = forwarded.get("type") or forwarded.get("event")
                    # Remove duplicate or raw keys that are represented in payload to reduce noise
                    for k in ("changed_paths", "paths", "changed_files", "applied_ids", "applied_rec_ids", "applied_recommendation_ids", "applied"):
                        forwarded.pop(k, None)

                    asyncio.run_coroutine_threadsafe(self._put_coro(forwarded), self._loop)
                except RuntimeError as e:
                    # Happens on shutdown; don't crash the app for this, but log.
                    debug("emit after loop closed", meta={"module": "events_bus", "reason": str(e)})

                # Emit a structured info indicating recording succeeded. Avoid logging full paths list to reduce noise; include count.
                info("applied_changes recorded", meta={
                    "module": "events_bus",
                    "event": "applied_changes",
                    "outcome": "recorded",
                    "run_id": run_id,
                    "change_id": change_id,
                    "applied_ids": payload.get("applied_ids"),
                    "paths_count": len(validated_paths),
                    "project_root": project_root,
                })
            except Exception as e:
                # Use structured error logging and re-raise so callers (apply stage) can present a failure instead of clearing planned changes
                error("error handling applied_changes event; emitting aborted and error propagated", meta={
                    "module": "events_bus",
                    "event": "applied_changes",
                    "run_id": locals().get("run_id", None),
                    "change_id": locals().get("change_id", None),
                    "exc": str(e),
                })
                raise
        elif evt_name in ("patch_apply_failed", "fallback_full_content", "edit_file_fallback"):
            # Enrich patch-failure / fallback events with diagnostic fields and record to trace.jsonl.
            # Failures to write this trace should not abort the main flow; they are best-effort diagnostics.
            record_payload = {}
            try:
                # Determine project_root preference (top-level then payload), otherwise cwd
                project_root = None
                for key in ("project_root", "project_path", "repo_root"):
                    if key in sanitized:
                        cand = sanitized.get(key)
                        if isinstance(cand, str) and cand.strip():
                            project_root = os.path.abspath(os.path.normpath(cand))
                        else:
                            debug("ignored invalid project_root candidate in top-level event", meta={"module": "events_bus", "candidate_key": key})
                        break
                if project_root is None and isinstance(sanitized.get("payload"), dict):
                    for key in ("project_root", "project_path", "repo_root"):
                        if key in sanitized["payload"]:
                            cand = sanitized["payload"].get(key)
                            if isinstance(cand, str) and cand.strip():
                                project_root = os.path.abspath(os.path.normpath(cand))
                            else:
                                debug("ignored invalid project_root candidate in payload", meta={"module": "events_bus", "candidate_key": key})
                            break
                if not project_root:
                    project_root = os.path.abspath(os.getcwd())

                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                trace_dir = os.path.join(project_root, ".aidev")
                os.makedirs(trace_dir, exist_ok=True)

                # Start with payload if provided, else use top-level sanitized event copy
                record_payload = sanitized.get("payload") if isinstance(sanitized.get("payload"), dict) else dict(sanitized)
                # Ensure we have a dict copy we can modify
                if isinstance(record_payload, dict):
                    record_payload = dict(record_payload)

                # Canonicalize and enrich fields consumers expect when a patch fails or a fallback to full-file edit is used.
                # full_content: prefer payload.full_content then top-level candidates
                for cand_key in ("full_content", "content", "file_content", "fallback_full_content"):
                    if "full_content" not in record_payload and cand_key in sanitized:
                        maybe = sanitized.get(cand_key)
                        if isinstance(maybe, str):
                            record_payload["full_content"] = maybe
                            break
                if "full_content" not in record_payload and isinstance(sanitized.get("payload"), dict):
                    for cand_key in ("full_content", "content", "file_content", "fallback_full_content"):
                        if cand_key in sanitized["payload"]:
                            maybe = sanitized["payload"].get(cand_key)
                            if isinstance(maybe, str):
                                record_payload["full_content"] = maybe
                                break

                # original_patch: preserve raw patch/diff if present
                for cand_key in ("original_patch", "patch", "diff"):
                    if "original_patch" not in record_payload and cand_key in sanitized:
                        maybe = sanitized.get(cand_key)
                        if isinstance(maybe, str):
                            record_payload["original_patch"] = maybe
                            break
                if "original_patch" not in record_payload and isinstance(sanitized.get("payload"), dict):
                    for cand_key in ("original_patch", "patch", "diff"):
                        if cand_key in sanitized["payload"]:
                            maybe = sanitized["payload"].get(cand_key)
                            if isinstance(maybe, str):
                                record_payload["original_patch"] = maybe
                                break

                # --- New: capture unified patch attempt and apply error explicitly ---
                # patch_unified_attempt: store the attempted unified-diff text (if available)
                if "patch_unified_attempt" not in record_payload:
                    for cand_key in ("patch_unified_attempt", "patch_unified", "patch_unified_text", "patch_attempt", "patch"):
                        if cand_key in sanitized:
                            maybe = sanitized.get(cand_key)
                            if isinstance(maybe, str) and maybe.strip():
                                record_payload["patch_unified_attempt"] = maybe
                                break
                    if "patch_unified_attempt" not in record_payload and isinstance(sanitized.get("payload"), dict):
                        for cand_key in ("patch_unified_attempt", "patch_unified", "patch_unified_text", "patch_attempt", "patch"):
                            if cand_key in sanitized["payload"]:
                                maybe = sanitized["payload"].get(cand_key)
                                if isinstance(maybe, str) and maybe.strip():
                                    record_payload["patch_unified_attempt"] = maybe
                                    break
                # patch_apply_error: store the error message / reason from the failed apply attempt
                if "patch_apply_error" not in record_payload:
                    for cand_key in ("patch_apply_error", "apply_error", "patch_error", "error", "reason"):
                        if cand_key in sanitized:
                            maybe = sanitized.get(cand_key)
                            if isinstance(maybe, str) and maybe.strip():
                                record_payload["patch_apply_error"] = maybe
                                break
                    if "patch_apply_error" not in record_payload and isinstance(sanitized.get("payload"), dict):
                        for cand_key in ("patch_apply_error", "apply_error", "patch_error", "error", "reason"):
                            if cand_key in sanitized["payload"]:
                                maybe = sanitized["payload"].get(cand_key)
                                if isinstance(maybe, str) and maybe.strip():
                                    record_payload["patch_apply_error"] = maybe
                                    break

                # fallback_reason: single-line reason for UI/diagnostics
                if "fallback_reason" not in record_payload:
                    for cand_key in ("fallback_reason", "reason", "error"):
                        if cand_key in sanitized:
                            maybe = sanitized.get(cand_key)
                            if isinstance(maybe, str):
                                record_payload["fallback_reason"] = maybe
                                break
                    if "fallback_reason" not in record_payload and isinstance(sanitized.get("payload"), dict):
                        for cand_key in ("fallback_reason", "reason", "error"):
                            if cand_key in sanitized["payload"]:
                                maybe = sanitized["payload"].get(cand_key)
                                if isinstance(maybe, str):
                                    record_payload["fallback_reason"] = maybe
                                    break

                # fallback_attempts: integer attempts count (best-effort parse)
                if "fallback_attempts" not in record_payload:
                    for cand_key in ("fallback_attempts", "attempts", "retry_count"):
                        val = None
                        if cand_key in sanitized:
                            val = sanitized.get(cand_key)
                        elif isinstance(sanitized.get("payload"), dict) and cand_key in sanitized["payload"]:
                            val = sanitized["payload"].get(cand_key)
                        if val is not None:
                            try:
                                record_payload["fallback_attempts"] = int(val)
                            except Exception:
                                # If it can't be parsed, store raw representation
                                record_payload["fallback_attempts"] = val
                            break

                # trace: allow structured trace metadata (list/dict) or synthesize a minimal trace entry
                if "trace" not in record_payload:
                    if "trace" in sanitized:
                        record_payload["trace"] = sanitized.get("trace")
                    else:
                        record_payload["trace"] = {"ts": ts, "reason": record_payload.get("fallback_reason")}

                # Always include project_root for clarity
                record_payload.setdefault("project_root", project_root)

                # Use a dedicated trace event name for patch-fallback diagnostics so it's
                # distinct from UI-facing edit strategy events.
                trace_event_name = "trace.patch_fallback"

                record = {
                    "ts": ts,
                    "event": trace_event_name,
                    "payload": record_payload,
                }

                trace_path = os.path.join(trace_dir, "trace.jsonl")
                try:
                    with open(trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False))
                        f.write("\n")
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            pass
                    info("wrote patch/fallback trace event", meta={"module": "events_bus", "event": trace_event_name, "project_root": project_root})
                except Exception as e:
                    error("failed to append trace record for patch/fallback event", meta={
                        "module": "events_bus",
                        "event": evt_name,
                        "project_root": project_root,
                        "trace_path": trace_path,
                        "exc": str(e),
                    })
            except Exception as e:
                # Unexpected errors should be logged but not raised to avoid interrupting producers
                error("unexpected error while handling patch/fallback event", meta={"module": "events_bus", "event": evt_name, "exc": str(e)})
            finally:
                # Forward the enriched event to the session queue regardless of trace write outcome
                try:
                    forwarded = dict(sanitized)
                    forwarded["payload"] = record_payload
                    forwarded["event"] = evt_name
                    forwarded["type"] = forwarded.get("type") or forwarded.get("event")
                    asyncio.run_coroutine_threadsafe(self._put_coro(forwarded), self._loop)
                except RuntimeError as e:
                    debug("emit after loop closed", meta={"module": "events_bus", "reason": str(e)})
        elif evt_name in ("trace.baseline", "trace.overlay", "trace.check_inputs", "trace.routing"):
            # Lightweight trace events for baseline/overlay/check inputs and routing decisions.
            # These are recorded to .aidev/trace.jsonl but failures here should not abort normal application flow:
            # log errors and continue forwarding.
            try:
                # Determine project_root preference (top-level then payload), otherwise cwd
                project_root = None
                for key in ("project_root", "project_path", "repo_root"):
                    if key in sanitized:
                        cand = sanitized.get(key)
                        if isinstance(cand, str) and cand.strip():
                            project_root = os.path.abspath(os.path.normpath(cand))
                        else:
                            debug("ignored invalid project_root candidate in top-level event", meta={"module": "events_bus", "candidate_key": key})
                        break
                if project_root is None and isinstance(sanitized.get("payload"), dict):
                    for key in ("project_root", "project_path", "repo_root"):
                        if key in sanitized["payload"]:
                            cand = sanitized["payload"].get(key)
                            if isinstance(cand, str) and cand.strip():
                                project_root = os.path.abspath(os.path.normpath(cand))
                            else:
                                debug("ignored invalid project_root candidate in payload", meta={"module": "events_bus", "candidate_key": key})
                            break
                if not project_root:
                    project_root = os.path.abspath(os.getcwd())

                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                trace_dir = os.path.join(project_root, ".aidev")
                os.makedirs(trace_dir, exist_ok=True)

                # Prefer recording the event's payload (already redacted); fallback to the whole sanitized event
                record_payload = sanitized.get("payload") if isinstance(sanitized.get("payload"), dict) else dict(sanitized)
                # Include project_root for clarity
                if isinstance(record_payload, dict):
                    record_payload = dict(record_payload)
                    record_payload.setdefault("project_root", project_root)

                record = {
                    "ts": ts,
                    "event": evt_name,
                    "payload": record_payload,
                }

                trace_path = os.path.join(trace_dir, "trace.jsonl")
                try:
                    with open(trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False))
                        f.write("\n")
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            pass
                    info("wrote trace event", meta={"module": "events_bus", "event": evt_name, "project_root": project_root})
                except Exception as e:
                    error("failed to append trace record for trace event", meta={
                        "module": "events_bus",
                        "event": evt_name,
                        "project_root": project_root,
                        "trace_path": trace_path,
                        "exc": str(e),
                    })
            except Exception as e:
                # Unexpected errors should be logged but not raised to avoid interrupting producers of optional traces
                error("unexpected error while handling trace event", meta={"module": "events_bus", "event": evt_name, "exc": str(e)})
            finally:
                # Forward the event to the session queue regardless of trace write outcome
                try:
                    asyncio.run_coroutine_threadsafe(self._put_coro(sanitized), self._loop)
                except RuntimeError as e:
                    debug("emit after loop closed", meta={"module": "events_bus", "reason": str(e)})
        else:
            # Not an applied_changes event: normal forwarding
            # Structured debug so forwarding is observable in logs
            try:
                debug("forwarding event to session queue", meta={
                    "module": "events_bus",
                    "event": evt_name,
                })

                # Special-case llm_call events: ensure payload.model is preserved/promoted so subscribers and trace writers see the resolved model.
                # Do not mutate the original sanitized dict used for logging; create a shallow copy and a dict copy of payload.
                if evt_name == "llm_call":
                    forwarded = dict(sanitized)
                    payload = dict(sanitized.get("payload") or {})

                    # Prefer payload-level model if present; otherwise consider top-level model/model_id candidates.
                    candidate = None
                    for k in ("model", "model_id"):
                        v = payload.get(k)
                        if isinstance(v, str) and v.strip():
                            candidate = v
                            break
                    if candidate is None:
                        for k in ("model", "model_id"):
                            v = sanitized.get(k)
                            if isinstance(v, str) and v.strip():
                                candidate = v
                                break

                    # Only set payload['model'] when absent and we have a candidate
                    if "model" not in payload and candidate:
                        payload["model"] = candidate

                    forwarded["payload"] = payload
                    asyncio.run_coroutine_threadsafe(self._put_coro(forwarded), self._loop)
                else:
                    asyncio.run_coroutine_threadsafe(self._put_coro(sanitized), self._loop)
            except RuntimeError as e:
                # Happens on shutdown; don't crash the app for this.
                debug("emit after loop closed", meta={"module": "events_bus", "reason": str(e)})
