# aidev/logging_utils.py
from __future__ import annotations
import logging
import logging.handlers as lh
import os
import sys
import json
import datetime


class _HttpNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        noisy = (
            "HTTP Request:" in msg
            or "HTTP Response:" in msg
            or "httpx" in record.name
            or "urllib3" in record.name
            or "azure.core.pipeline.policies.http_logging_policy" in record.name
        )
        return not noisy


class JsonFormatter(logging.Formatter):
    """Compact single-line JSON formatter for logs.

    Emits objects with keys: ts (ISO8601 UTC), level, module, msg, meta.
    meta is taken from record.__dict__.get('meta') and enriched with common fields
    that may be attached to LogRecord (phase, rec_id, job_id, latency_ms, model, etc.).
    This formatter is intentionally compact (no spaces) so it is safe for JSONL files/streams.
    """

    def __init__(self):
        super().__init__()

    def _safe(self, v):
        # Ensure value is JSON serializable; fallback to str()
        try:
            json.dumps(v)
            return v
        except Exception:
            try:
                return str(v)
            except Exception:
                return None

    def format(self, record: logging.LogRecord) -> str:
        rec_ts = datetime.datetime.utcnow().isoformat() + "Z"
        msg = record.getMessage()
        module = record.name
        # Start with explicit meta if provided by caller
        meta = {}
        raw_meta = record.__dict__.get("meta")
        if isinstance(raw_meta, dict):
            meta.update(raw_meta)

        # Enrich meta with common well-known fields if present on the record
        for k in (
            "phase",
            "rec_id",
            "job_id",
            "run_id",
            "request_id",
            "response_id",
            "latency_ms",
            "elapsed",
            "tokens_in",
            "tokens_out",
            "model",
            "status",
            "path",
            "op",
            "event_type",
            "chars",
        ):
            if k in record.__dict__ and record.__dict__[k] is not None:
                meta[k] = self._safe(record.__dict__[k])

        payload = {
            "ts": rec_ts,
            "level": record.levelname.lower(),
            "module": module,
            "msg": msg,
            "meta": meta,
        }
        # If exception info present, include a short representation
        if record.exc_info:
            try:
                payload["meta"]["exc"] = self._safe(self.formatException(record.exc_info))
            except Exception:
                payload["meta"]["exc"] = "<exc?>"

        return json.dumps(payload, separators=(",", ":"))


def configure_quiet_http(quiet_http: bool) -> None:
    """Reduce noisy HTTP-level logs from httpx/urllib3/openai/azure namespaces.

    This function is safe to call multiple times; it sets affected loggers to WARNING and
    prevents propagation where appropriate.
    """
    if not quiet_http:
        return
    for name in (
        "httpx",
        "openai",
        "azure",
        "azure.core",
        "azure.identity",
        "azure.core.pipeline.policies.http_logging_policy",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.propagate = False
    root = logging.getLogger()
    for h in root.handlers:
        # only add filter if none of this type already present on handler
        has = any(isinstance(f, _HttpNoiseFilter) for f in h.filters)
        if not has:
            h.addFilter(_HttpNoiseFilter())
    os.environ.setdefault("OPENAI_LOG_LEVEL", "error")


def configure_logging(
    quiet_http: bool = True, verbose: bool = False, structured: bool = True
) -> None:
    """Configure global logging for the application.

    Backwards-compatible: previously callers passed (quiet_http, verbose). We keep the same
    parameter order but make quiet_http default to True and add `structured` to enable
    compact JSON output (JSONL) on stdout and the rotating file.

    To keep test harnesses deterministic, tests may call configure_logging(structured=False).
    """
    lg = logging.getLogger()
    lg.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Idempotent handler additions: avoid adding duplicate StreamHandler or RotatingFileHandler

    # Prepare formatters
    json_fmt = JsonFormatter()
    human_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Rotating file handler for app.log - keep existing filename and rotation policy
    add_fh = True
    app_log_path = os.path.abspath("app.log")
    for h in lg.handlers:
        # compare file handlers by their baseFilename where available
        base = getattr(h, "baseFilename", None)
        if base and os.path.abspath(base) == app_log_path:
            add_fh = False
            # ensure formatter type matches requested mode; update if different
            if structured and not isinstance(h.formatter, JsonFormatter):
                h.setFormatter(json_fmt)
            elif not structured and isinstance(h.formatter, JsonFormatter):
                h.setFormatter(human_fmt)
            break
    if add_fh:
        fh = lh.RotatingFileHandler(
            "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(json_fmt if structured else human_fmt)
        lg.addHandler(fh)

    # Stream handler for stdout - avoid duplicates bound to stdout
    add_sh = True
    for h in lg.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout:
            add_sh = False
            # ensure formatter type matches requested mode; update if different
            if structured and not isinstance(h.formatter, JsonFormatter):
                h.setFormatter(json_fmt)
            elif not structured and isinstance(h.formatter, JsonFormatter):
                h.setFormatter(human_fmt)
            break
    if add_sh:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(json_fmt if structured else human_fmt)
        lg.addHandler(sh)

    # Quiet noisy HTTP/logging libraries when requested
    configure_quiet_http(quiet_http)
