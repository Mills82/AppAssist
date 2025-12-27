# runtimes/logger.py
"""
Structured file logger for the project.

Behavior & goals

- Writes one JSON object per line to app.log (default: ./app.log). Each record contains:
  - ts: ISO8601 UTC timestamp
  - level: one of INFO, WARN, ERROR
  - msg: short human-readable message
  - ctx: optional mapping with contextual data
  - exc: optional exception text (traceback or exception message)
  - dur_ms: optional duration in milliseconds for actions

- Ensures app.log is created if absent.

- Performs simple rotation when file size exceeds `max_size`:
  - Rotates by renaming app.log -> app.log.1, app.log.1 -> app.log.2, ... up to rotate_count
  - The newest backup is always app.log.1; older backups are bumped. If rotate_count is 0 no rotation occurs.
  - This is intentionally simple (move-based) and documented here to set expectations. No compression is performed.

- Thread-safe writes via an internal lock.

- Exposes convenience helpers: setup(), info(), warn(), error(), exception(), start_action() (context manager).
- Also exposes setup_logging() alias and get_logger() to obtain a proxy logger with optional bound context.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from contextlib import ContextDecorator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

_lock = threading.Lock()
_log_path: Optional[Path] = None
_max_size = 5_000_000
_rotate_count = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(obj: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable types for ctx values.
    """
    # Fast path for primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Datetime → ISO
    if isinstance(obj, datetime):
        try:
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.isoformat()
        except Exception:
            return str(obj)

    # Path → string
    if isinstance(obj, Path):
        return str(obj)

    # Exceptions → string
    if isinstance(obj, BaseException):
        return f"{type(obj).__name__}: {obj}"

    # Mapping → recurse
    if isinstance(obj, Mapping):
        try:
            return {str(k): _json_safe(v) for k, v in obj.items()}
        except Exception:
            return {str(k): str(v) for k, v in obj.items()}

    # Sequence (but not str/bytes) → list of safe items
    if isinstance(obj, (list, tuple, set)):
        try:
            return [_json_safe(x) for x in obj]
        except Exception:
            return [str(x) for x in obj]

    # Fallback
    try:
        json.dumps(obj)  # type: ignore[arg-type]
        return obj
    except Exception:
        return str(obj)


def format_progress_annotation(current: int, total: Optional[int] = None, percent: Optional[float] = None) -> str:
    """
    Build a concise progress annotation string that can be attached to both
    user-facing messages and structured logs/traces. Examples:
      - "1/4 complete (25%)"
      - "75%"
      - "3"
    """
    try:
        if total is not None and total > 0:
            pct = round((current / total) * 100)
            return f"{current}/{total} complete ({pct}%)"
        if percent is not None:
            pct = round(percent)
            return f"{pct}%"
        return str(current)
    except Exception:
        # Never fail logging because of formatting
        try:
            return f"{current}/{total}"
        except Exception:
            return str(current)


def setup(log_path: Optional[str] = None, max_size: int = 5_000_000, rotate_count: int = 3) -> None:
    """Configure the module-level logger.

    Args:
        log_path: path to the app.log file. Defaults to ./app.log
        max_size: rotation threshold in bytes
        rotate_count: number of rotated backups to keep (0 to disable rotation)
    """
    global _log_path, _max_size, _rotate_count
    _max_size = int(max_size)
    _rotate_count = int(rotate_count)

    # Allow env override if not explicitly provided
    if log_path is None:
        log_path = os.getenv("APP_LOG_PATH") or "app.log"

    _log_path = Path(log_path)
    parent = _log_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort: emit to stderr if we cannot create dir
        print(f"WARN: could not create log dir {parent}", flush=True)

    # Ensure file exists
    try:
        with _log_path.open("a", encoding="utf-8", newline="\n"):
            pass
    except Exception:
        print(f"ERROR: could not create log file {_log_path}", flush=True)


# Backwards/forwards friendly alias
setup_logging = setup


def _rotate_if_needed() -> None:
    """Rotate log if it exceeds _max_size. Simple move-based rotation.

    Implemented with the convention:
      app.log -> app.log.1
      app.log.1 -> app.log.2
      ...
    """
    global _log_path, _max_size, _rotate_count
    if _log_path is None or _rotate_count <= 0:
        return
    try:
        if not _log_path.exists():
            return
        if _log_path.stat().st_size <= _max_size:
            return

        # bump existing backups (N-1 → N, ..., 1 → 2)
        for i in range(_rotate_count - 1, 0, -1):
            src = (
                _log_path.with_suffix(_log_path.suffix + f".{i}")
                if _log_path.suffix
                else Path(str(_log_path) + f".{i}")
            )
            dst = (
                _log_path.with_suffix(_log_path.suffix + f".{i+1}")
                if _log_path.suffix
                else Path(str(_log_path) + f".{i+1}")
            )
            if src.exists():
                try:
                    if dst.exists():
                        dst.unlink()
                    src.replace(dst)
                except Exception:
                    # non-fatal
                    pass

        # rotate current to .1
        first = (
            _log_path.with_suffix(_log_path.suffix + ".1")
            if _log_path.suffix
            else Path(str(_log_path) + ".1")
        )
        try:
            if first.exists():
                first.unlink()
            _log_path.replace(first)
        except Exception:
            # if rename fails, copy & truncate
            try:
                from shutil import copy2

                copy2(str(_log_path), str(first))
                with _log_path.open("w", encoding="utf-8", newline="\n"):
                    pass
            except Exception:
                # give up on rotation
                pass
    except Exception:
        # Never allow logger to crash caller
        return


def _write_record(rec: Dict[str, Any]) -> None:
    global _log_path
    if _log_path is None:
        setup()
    assert _log_path is not None

    # Ensure ctx is json-safe if present
    if "ctx" in rec:
        try:
            rec["ctx"] = _json_safe(rec["ctx"])
        except Exception:
            rec["ctx"] = str(rec.get("ctx"))

    line = json.dumps(rec, ensure_ascii=False)
    try:
        with _lock:
            _rotate_if_needed()
            with _log_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    # ignore fsync errors
                    pass
    except Exception:
        # Last resort: stderr
        try:
            print(f"LOGGER WRITE ERROR: {rec}", flush=True)
        except Exception:
            pass


def _make_record(
    level: str,
    msg: str,
    ctx: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
    dur_ms: Optional[float] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "ts": _now_iso(),
        "level": level,
        "msg": msg,
    }
    if ctx:
        rec["ctx"] = ctx

    # If a progress_annotation exists in ctx, mirror it as a top-level field
    if ctx and isinstance(ctx, dict) and "progress_annotation" in ctx:
        try:
            rec["progress_annotation"] = _json_safe(ctx["progress_annotation"])
        except Exception:
            rec["progress_annotation"] = str(ctx.get("progress_annotation"))

    if exc is not None:
        try:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            tb = f"{type(exc).__name__}: {exc}"
        rec["exc"] = tb
    if dur_ms is not None:
        rec["dur_ms"] = dur_ms
    return rec


def info(msg: str, ctx: Optional[Dict[str, Any]] = None) -> None:
    _write_record(_make_record("INFO", msg, ctx=ctx))


def warn(msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[BaseException] = None) -> None:
    _write_record(_make_record("WARN", msg, ctx=ctx, exc=exc))


def error(msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[BaseException] = None) -> None:
    _write_record(_make_record("ERROR", msg, ctx=ctx, exc=exc))


def exception(msg: str, ctx: Optional[Dict[str, Any]] = None) -> None:
    """Log the current exception (use inside an except: block)."""
    try:
        exc_text = traceback.format_exc()
    except Exception:
        exc_text = "(could not format exception)"
    rec = {
        "ts": _now_iso(),
        "level": "ERROR",
        "msg": msg,
        "ctx": _json_safe(ctx) if ctx else None,
        "exc": exc_text,
    }
    if rec["ctx"] is None:
        del rec["ctx"]
    _write_record(rec)


class Action(ContextDecorator):
    """Context manager for logging start/stop of an action.

    Example:
        with Action("lint", {"target": "src/"}):
            run_linter()
    """

    def __init__(self, name: str, ctx: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.ctx = ctx
        self._t0: Optional[datetime] = None

    def __enter__(self) -> "Action":
        self._t0 = datetime.now(timezone.utc)
        _write_record(_make_record("INFO", f"action_start: {self.name}", ctx=self.ctx))
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        t1 = datetime.now(timezone.utc)
        dur = (t1 - self._t0).total_seconds() * 1000.0 if self._t0 else None
        if exc is not None:
            _write_record(
                _make_record("ERROR", f"action_error: {self.name}", ctx=self.ctx, exc=exc, dur_ms=dur)
            )
            return False
        else:
            _write_record(_make_record("INFO", f"action_stop: {self.name}", ctx=self.ctx, dur_ms=dur))
            return False

    def progress(self, current: int, total: Optional[int] = None, percent: Optional[float] = None, message: Optional[str] = None) -> None:
        """Emit a progress update for this action. The same progress_annotation is included
        in the ctx and mirrored as a top-level 'progress_annotation' field in the log record.
        """
        ann = format_progress_annotation(current, total=total, percent=percent)
        merged = dict(self.ctx) if self.ctx else {}
        merged["progress"] = {"current": current, "total": total, "percent": percent}
        merged["progress_annotation"] = ann
        msg = message if message is not None else f"action_progress: {self.name} {ann}"
        _write_record(_make_record("INFO", msg, ctx=merged))


def start_action(name: str, ctx: Optional[Dict[str, Any]] = None) -> Action:
    return Action(name, ctx=ctx)


def log_progress(name: str, current: int, total: Optional[int] = None, percent: Optional[float] = None, ctx: Optional[Dict[str, Any]] = None) -> None:
    """Convenience helper to emit a progress update for a named action.

    The function builds a single progress_annotation string and includes it both in
    the ctx (under 'progress_annotation') and mirrored at the top-level of the record
    as 'progress_annotation' so trace/jsonl and app.log consumers see the identical string.
    """
    annotation = format_progress_annotation(current, total=total, percent=percent)
    merged = dict(ctx) if ctx else {}
    merged.setdefault("progress", {"current": current, "total": total, "percent": percent})
    merged["progress_annotation"] = annotation
    _write_record(_make_record("INFO", f"action_progress: {name} {annotation}", ctx=merged))


class _LoggerProxy:
    """Thin proxy that mirrors module-level helpers and injects default ctx."""

    def __init__(self, default_ctx: Optional[Dict[str, Any]] = None) -> None:
        self._default_ctx = default_ctx or {}

    def _merge(self, ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not self._default_ctx and ctx is None:
            return {}
        merged: Dict[str, Any] = {}
        merged.update(self._default_ctx)
        if ctx:
            merged.update(ctx)
        return merged

    def info(self, msg: str, ctx: Optional[Dict[str, Any]] = None) -> None:
        info(msg, ctx=self._merge(ctx) or None)

    def warn(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[BaseException] = None) -> None:
        warn(msg, ctx=self._merge(ctx) or None, exc=exc)

    def error(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[BaseException] = None) -> None:
        error(msg, ctx=self._merge(ctx) or None, exc=exc)

    def exception(self, msg: str, ctx: Optional[Dict[str, Any]] = None) -> None:
        exception(msg, ctx=self._merge(ctx) or None)

    def start_action(self, name: str, ctx: Optional[Dict[str, Any]] = None) -> Action:
        merged = self._merge(ctx) or None
        return start_action(name, ctx=merged)

    def progress(self, name: str, current: int, total: Optional[int] = None, percent: Optional[float] = None, ctx: Optional[Dict[str, Any]] = None) -> None:
        merged = self._merge(ctx) or None
        log_progress(name, current, total=total, percent=percent, ctx=merged)


def get_logger(name: Optional[str] = None, default_ctx: Optional[Dict[str, Any]] = None) -> _LoggerProxy:
    """
    Return a logger proxy that uses this module's structured writer.
    If `name` is provided, it is added to the bound context under 'logger'.

    Example:
        log = get_logger(__name__)
        log.info("boot", ctx={"version": "1.2.3"})
    """
    bound = dict(default_ctx or {})
    if name:
        # Don't overwrite if caller already passed a 'logger' key
        bound.setdefault("logger", name)
    return _LoggerProxy(bound if bound else None)


# Ensure there is a default logger configured on import (safe defaults)
try:
    if _log_path is None:
        setup()
except Exception:
    # never raise on import
    pass


__all__ = [
    "setup",
    "setup_logging",
    "info",
    "warn",
    "error",
    "exception",
    "Action",
    "start_action",
    "get_logger",
    "format_progress_annotation",
    "log_progress",
]
