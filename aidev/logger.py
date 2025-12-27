# aidev/logger.py
from __future__ import annotations

from typing import Optional, Dict, Any

# Prefer the structured JSON logger if available
try:
    from runtimes import logger as _rt_logger  # requires runtimes/__init__.py to exist
    import json

    # Normalize upstream: pick whichever exists, prefer .warning over .warn
    _rt_warning_fn = getattr(_rt_logger, "warning", None) or getattr(_rt_logger, "warn", None)
    if _rt_warning_fn is None:
        # Last-resort shim: fall back to info so we never crash on import
        _rt_warning_fn = getattr(_rt_logger, "info")

    info = getattr(_rt_logger, "info")
    debug = getattr(_rt_logger, "debug", info)
    warning = _rt_warning_fn
    warn = _rt_warning_fn  # alias for backward compatibility
    error = getattr(_rt_logger, "error")
    exception = getattr(_rt_logger, "exception")
    Action = getattr(_rt_logger, "Action")
    start_action = getattr(_rt_logger, "start_action")

    def get_logger(name=None, default_ctx=None):
        # Get upstream logger (if provided), then normalize its API to have both names
        base = getattr(_rt_logger, "get_logger", lambda *_a, **_k: _rt_logger)()
        if not hasattr(base, "warning") and hasattr(base, "warn"):
            setattr(base, "warning", getattr(base, "warn"))
        if not hasattr(base, "warn") and hasattr(base, "warning"):
            setattr(base, "warn", getattr(base, "warning"))
        if not hasattr(base, "debug"):
            setattr(base, "debug", getattr(base, "info"))
        return base

    def _coerce_num(v: Optional[Any]) -> Optional[Any]:
        if v is None:
            return None
        try:
            # prefer int when possible
            if float(v).is_integer():
                return int(float(v))
            return float(v)
        except Exception:
            return v

    def log_llm_call(
        phase: str,
        response_id: Optional[str],
        status: Optional[str],
        model: Optional[str],
        latency_ms: Optional[float],
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        request_id: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        """Emit a structured LLM-related log entry.

        This will try to call the upstream logger with ctx=meta if supported,
        otherwise it will fall back to emitting the meta JSON appended to the
        message so nothing raises.
        """
        meta: Dict[str, Any] = {
            "phase": phase,
            "response_id": response_id,
            "status": status,
            "model": model,
            "latency_ms": _coerce_num(latency_ms),
            "tokens_in": _coerce_num(tokens_in),
            "tokens_out": _coerce_num(tokens_out),
            "request_id": request_id,
        }
        # merge extra_meta without mutating caller's dict
        if extra_meta:
            m = dict(extra_meta)
            # only put non-conflicting keys under 'extra' to avoid accidental overrides
            meta["extra"] = m

        msg = f"[llm_client.chat_json] success phase={phase}"

        level_fn = {
            "debug": debug,
            "info": info,
            "warning": warning,
            "warn": warning,
            "error": error,
        }.get(level, info)

        # Try upstream 'ctx' style if it accepts it, else append JSON to message
        try:
            # many runtimes loggers accept ctx kwarg
            level_fn(msg, ctx=meta)
        except TypeError:
            try:
                # fallback: pass merged json as part of the message
                level_fn(f"{msg} {json.dumps(meta, default=str)}")
            except Exception:
                # do not swallow unexpected exceptions from upstream logger: re-raise
                raise

    def _emit_structured(level: str, op: str, meta: Optional[Dict[str, Any]], msg: Optional[str] = None) -> None:
        """Thin wrapper to emit a structured JSON-style log entry.

        This prefers the upstream runtime logger (using ctx=meta) when available
        and falls back to appending a JSON payload to the message if the
        upstream logger does not accept structured kwargs.
        """
        meta = dict(meta or {})
        # record the operation name under a concise key
        meta["op"] = op
        message = msg or op

        level_fn = {
            "debug": debug,
            "info": info,
            "warning": warning,
            "warn": warning,
            "error": error,
        }.get(level, info)

        try:
            level_fn(message, ctx=meta)
        except TypeError:
            try:
                level_fn(f"{message} {json.dumps(meta, default=str)}")
            except Exception:
                raise

    class _Proxy:
        def info(self, msg: str, ctx: Optional[Dict[str, Any]] = None):
            info(msg, ctx=ctx) if ctx is not None else info(msg)

        def debug(self, msg: str, ctx: Optional[Dict[str, Any]] = None):
            debug(msg, ctx=ctx) if ctx is not None else debug(msg)

        def warning(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
            warning(msg, ctx=ctx, exc=exc)

        def warn(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
            warning(msg, ctx=ctx, exc=exc)  # alias

        def error(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
            error(msg, ctx=ctx, exc=exc)

        def exception(self, msg: str, ctx: Optional[Dict[str, Any]] = None):
            exception(msg, ctx=ctx)

        def start_action(self, name: str, ctx: Optional[Dict[str, Any]] = None):
            return start_action(name, ctx=ctx)

        def log_llm_call(self, *a, **k):
            return log_llm_call(*a, **k)

    logger = _Proxy()

except Exception:
    # Fallback: standard logging with similar signature, emit compact JSON lines
    import logging, sys, json
    from datetime import datetime, timezone
    from typing import Optional, Dict, Any

    _lg = logging.getLogger("aidev")
    if not _lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        # Handler prints only the message; module build the JSON "message"
        h.setFormatter(logging.Formatter("%(message)s"))
        _lg.addHandler(h)
    _lg.setLevel(logging.INFO)

    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _coerce_num(v: Optional[Any]) -> Optional[Any]:
        if v is None:
            return None
        try:
            if float(v).is_integer():
                return int(float(v))
            return float(v)
        except Exception:
            return v

    def _emit(level_name: str, msg: str, meta: Optional[Dict[str, Any]] = None) -> None:
        event: Dict[str, Any] = {
            "ts": _ts(),
            "level": level_name,
            "module": __name__,
            "msg": msg,
            "meta": meta or {},
        }
        # Emit raw JSON as the log message so the handler prints compact JSON lines
        payload = json.dumps(event, default=str)
        if level_name == "debug":
            _lg.debug(payload)
        elif level_name in ("warn", "warning"):
            _lg.warning(payload)
        elif level_name == "error":
            _lg.error(payload)
        else:
            _lg.info(payload)

    def info(msg: str, ctx: Optional[Dict[str, Any]] = None) -> None:
        _emit("info", msg, ctx)

    def debug(msg: str, ctx: Optional[Dict[str, Any]] = None) -> None:
        _emit("debug", msg, ctx)

    def warning(msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None) -> None:
        meta = dict(ctx or {})
        if exc is not None:
            meta["exc"] = str(exc)
        _emit("warning", msg, meta)

    # Back-compat alias
    def warn(msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None) -> None:
        warning(msg, ctx=ctx, exc=exc)

    def error(msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None) -> None:
        meta = dict(ctx or {})
        if exc is not None:
            meta["exc"] = str(exc)
        _emit("error", msg, meta)

    def exception(msg: str, ctx: Optional[Dict[str, Any]] = None) -> None:
        _emit("error", msg, dict(ctx or {}))
        _lg.exception(msg)

    class Action:
        def __init__(self, name: str, ctx: Optional[Dict[str, Any]] = None):
            self.name, self.ctx = name, ctx

        def __enter__(self):
            meta = dict(self.ctx or {})
            meta["action"] = self.name
            info("action_start", ctx=meta)
            return self

        def __exit__(self, exc_type, exc, tb):
            meta = dict(self.ctx or {})
            meta["action"] = self.name
            if exc:
                error("action_error", ctx=meta, exc=exc)
            else:
                info("action_stop", ctx=meta)
            return False

    def start_action(name: str, ctx: Optional[Dict[str, Any]] = None):
        return Action(name, ctx=ctx)

    def get_logger(name=None, default_ctx=None):
        # minimal shim: return a proxy with the same API as module
        class _LocalProxy:
            def info(self, msg: str, ctx: Optional[Dict[str, Any]] = None):
                info(msg, ctx=ctx)

            def debug(self, msg: str, ctx: Optional[Dict[str, Any]] = None):
                debug(msg, ctx=ctx)

            def warning(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
                warning(msg, ctx=ctx, exc=exc)

            def warn(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
                warning(msg, ctx=ctx, exc=exc)

            def error(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
                error(msg, ctx=ctx, exc=exc)

            def exception(self, msg: str, ctx: Optional[Dict[str, Any]] = None):
                exception(msg, ctx=ctx)

            def start_action(self, name: str, ctx: Optional[Dict[str, Any]] = None):
                return start_action(name, ctx=ctx)

            def log_llm_call(self, *a, **k):
                return log_llm_call(*a, **k)

        return _LocalProxy()

    def log_llm_call(
        phase: str,
        response_id: Optional[str],
        status: Optional[str],
        model: Optional[str],
        latency_ms: Optional[float],
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        request_id: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        meta: Dict[str, Any] = {
            "phase": phase,
            "response_id": response_id,
            "status": status,
            "model": model,
            "latency_ms": _coerce_num(latency_ms),
            "tokens_in": _coerce_num(tokens_in),
            "tokens_out": _coerce_num(tokens_out),
            "request_id": request_id,
        }
        if extra_meta:
            meta["extra"] = dict(extra_meta)
        msg = f"[llm_client.chat_json] success phase={phase}"
        _emit(level, msg, meta)

    def _emit_structured(level: str, op: str, meta: Optional[Dict[str, Any]], msg: Optional[str] = None) -> None:
        """Fallback thin wrapper to emit a structured JSON-style log entry via stdlib logging.

        This will attach op into the meta under 'op' and emit a compact JSON line
        as produced by _emit().
        """
        meta = dict(meta or {})
        meta["op"] = op
        message = msg or op
        _emit(level, message, meta)

    class _Proxy:
        def info(self, msg: str, ctx: Optional[Dict[str, Any]] = None):
            info(msg, ctx=ctx)

        def debug(self, msg: str, ctx: Optional[Dict[str, Any]] = None):
            debug(msg, ctx=ctx)

        def warning(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
            warning(msg, ctx=ctx, exc=exc)

        def warn(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
            warning(msg, ctx=ctx, exc=exc)  # alias

        def error(self, msg: str, ctx: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
            error(msg, ctx=ctx, exc=exc)

        def exception(self, msg: str, ctx: Optional[Dict[str, Any]] = None):
            exception(msg, ctx=ctx)

        def start_action(self, name: str, ctx: Optional[Dict[str, Any]] = None):
            return start_action(name, ctx=ctx)

        def log_llm_call(self, *a, **k):
            return log_llm_call(*a, **k)

    logger = _Proxy()

__all__ = [
    "info", "debug", "warning", "warn", "error", "exception",
    "Action", "start_action", "get_logger", "logger", "log_llm_call", "_emit_structured",
]
