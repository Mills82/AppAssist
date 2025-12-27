# runtimes/trace_writer.py
"""Per-run, atomic JSONL trace writer for .aidev/traces/run-<run_id>.jsonl

Key behaviors
-------------
- **Per-run file**: each process/run writes to its own JSONL file
  (.aidev/traces/run-<run_id>.jsonl), easy to upload/inspect.
- **Atomic, minimal appends**: uses os.open(..., O_APPEND|O_CREAT|O_WRONLY)
  and a single write per line, plus fsync for durability.
- **Structured fields**: phase, tool, latency_ms, tokens_in/out,
  sse_events_emitted are first-class optional fields.
- **Redaction**: best-effort secret masking (keys + token-like patterns).
- **Soft JSON Schema**: validates shape if jsonschema is installed.
- **Pruning**: keep only the N most recent run files (configurable).

Usage
-----
    writer = TraceWriter(project_root="/path/to/repo")
    writer.write("start", "run", {"root": "..."}, phase="init")

    # LLM call with metrics
    with writer.timer(event="llm_call", kind="openai", phase="generate", tool="recommend"):
        # ... perform call ...
        writer.write_llm(event="llm_call_done", kind="openai",
                         tokens_in=1234, tokens_out=456)

    writer.write("end", "run", {"ok": True}, phase="finalize")
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import logging
import os
import re
import socket
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---- Optional JSON Schema support ------------------------------------------------
try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore

EVENT_SCHEMA: Dict[str, Any] = {
    "title": "AIDevTraceEvent",
    "type": "object",
    "properties": {
        "ts": {"type": "string"},  # ISO8601
        "event": {"type": "string"},
        "kind": {"type": "string"},
        "data": {"type": ["object", "array", "string", "number", "boolean", "null"]},
        "session_id": {"type": ["string", "null"]},
        "user_id": {"type": ["string", "null"]},
        "host": {"type": "string"},
        "pid": {"type": "integer"},
        "model": {"type": ["string", "null"]},
        "level": {"type": ["string", "null"]},

        #  first-class observability fields
        "phase": {"type": ["string", "null"]},
        "tool": {"type": ["string", "null"]},
        "latency_ms": {"type": ["number", "null"]},
        "tokens_in": {"type": ["number", "null"]},
        "tokens_out": {"type": ["number", "null"]},
        "sse_events_emitted": {"type": ["number", "null"]},
    },
    "required": ["ts", "event", "kind", "data"],
    "additionalProperties": True,
}

# ---- Redaction -------------------------------------------------------------------

# Keys that should be redacted when found in dicts (case-insensitive substring match)
_REDACT_KEYS = [
    "api_key", "apikey", "api-key", "secret", "password", "pass", "pwd",
    "token", "access_token", "refresh_token", "id_token", "bearer",
    "authorization", "cookie", "set-cookie", "session", "private", "key",
]

# Regexes that will be redacted inside ANY string value
_REDACT_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    # OpenAI-like keys
    (re.compile(r"(?i)\b(?:sk|rk|pk|oc|azd)-[A-Za-z0-9]{20,}\b"), "***REDACTED***"),
    # AWS Access Keys
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "***REDACTED***"),
    # AWS Secret Keys (loose heuristic)
    (re.compile(r"(?i)\baws_secret_access_key\s*=\s*['\"][A-Za-z0-9/+=]{30,}['\"]"),
     "aws_secret_access_key=\"***REDACTED***\""),
    # Bearer tokens
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-_.=+/]{10,}\b"), "Bearer ***REDACTED***"),
    # Generic JWT (xxx.yyy.zzz)
    (re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"),
     "***REDACTED_JWT***"),
    # Long hex secrets
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "***REDACTED_HEX***"),
    # Cookie-like key=value
    (re.compile(r"(?i)\b(?:session|sid|csrftoken|xsrf|auth|token)=[^;,\s]{10,}"),
     "cookie=***REDACTED***"),
)


def _iso_now() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return os.environ.get("HOSTNAME", "") or ""


def _redact_string(s: str) -> str:
    out = s
    for pat, repl in _REDACT_PATTERNS:
        try:
            out = pat.sub(repl, out)
        except Exception:
            pass
    return out


def _should_redact_key(k: str) -> bool:
    low = k.lower()
    return any(tag in low for tag in _REDACT_KEYS)


def _redact_obj(obj: Any, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return obj
    if isinstance(obj, dict):
        red: Dict[str, Any] = {}
        for k, v in obj.items():
            if _should_redact_key(str(k)):
                red[k] = "***REDACTED***"
            else:
                red[k] = _redact_obj(v, depth + 1, max_depth)
        return red
    if isinstance(obj, list):
        return [_redact_obj(v, depth + 1, max_depth) for v in obj]
    if isinstance(obj, str):
        return _redact_string(obj)
    return obj

# ---- Trace Writer ----------------------------------------------------------------

@dataclass
class TraceWriter:
    """
    Write newline-delimited JSON (JSONL) to a *per-run* file with atomic appends.

    By default the file is:
        <project_root>/.aidev/traces/run-<run_id>.jsonl

    Tip: set AIDEV_RUN_ID to control the filename (e.g., CI build number).
    """

    project_root: Optional[str] = None
    traces_rel_dir: str = ".aidev/traces"
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    # Per-run file pruning (how many most-recent .jsonl/.jsonl.gz to keep)
    keep_runs: int = 20
    # Optional compression of *completed* runs; call .finalize(compress=True)
    compress_on_finalize: bool = False

    # observed SSE counter (optional: feed via .accumulate_sse_emitted)
    _sse_emitted_counter: int = 0

    def __post_init__(self) -> None:
        self.project_root = os.path.abspath(self.project_root or os.getcwd())
        self.traces_dir = os.path.abspath(os.path.join(self.project_root, self.traces_rel_dir))

        # Safety: ensure directory is inside project_root
        common = os.path.commonpath([self.project_root, self.traces_dir])
        if common != self.project_root:
            raise ValueError(f"traces dir '{self.traces_dir}' is outside project root '{self.project_root}'")

        self.run_id = self.run_id or os.environ.get("AIDEV_RUN_ID") or self._make_run_id()
        self.trace_path = os.path.join(self.traces_dir, f"run-{self.run_id}.jsonl")
        os.makedirs(self.traces_dir, exist_ok=True)

        # Session/user
        self.session_id = self.session_id or os.environ.get("AIDEV_SESSION_ID")
        self.user_id = self.user_id or os.environ.get("AIDEV_USER_ID")

        # Open the file descriptor once for the run (append-only)
        self._fd: Optional[int] = None
        self._ensure_fd()

        # Write a tiny header event
        self.write("run_start", "run", {"traces_dir": self.traces_dir, "path": self.trace_path},
                   phase="init")

        # Prune older runs if over the limit
        try:
            self._prune_old_runs()
        except Exception:
            logger.debug("Trace run pruning failed", exc_info=True)

    # --- Public API ---------------------------------------------------------------

    def path(self) -> str:
        return self.trace_path

    def accumulate_sse_emitted(self, n: int = 1) -> None:
        try:
            self._sse_emitted_counter += int(n)
        except Exception:
            pass

    def write_llm(self, event: str, kind: str, *,
                  tokens_in: Optional[int] = None,
                  tokens_out: Optional[int] = None,
                  latency_ms: Optional[float] = None,
                  model: Optional[str] = None,
                  phase: Optional[str] = None,
                  tool: Optional[str] = None,
                  data: Any = None,
                  level: Optional[str] = None) -> None:
        self.write(event, kind, data or {}, model=model, level=level,
                   phase=phase, tool=tool, latency_ms=latency_ms,
                   tokens_in=tokens_in, tokens_out=tokens_out)

    def write(self, event: str, kind: str, data: Any, *,
              model: Optional[str] = None,
              level: Optional[str] = None,
              duration_ms: Optional[float] = None,     # kept for back-compat
              phase: Optional[str] = None,
              tool: Optional[str] = None,
              latency_ms: Optional[float] = None,
              tokens_in: Optional[int] = None,
              tokens_out: Optional[int] = None) -> None:
        """
        Preferred structured event writer.
        - `duration_ms` is accepted for back-compat and mapped to `latency_ms` if not given.
        """
        if latency_ms is None and duration_ms is not None:
            latency_ms = duration_ms

        entry = {
            "ts": _iso_now(),
            "event": str(event),
            "kind": str(kind),
            "data": _redact_obj(data),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "host": _hostname(),
            "pid": os.getpid(),
            "model": model,
            "level": level,

            # observability
            "phase": phase,
            "tool": tool,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "sse_events_emitted": self._sse_emitted_counter or None,
        }
        self._validate(entry)
        self._append_jsonl(entry)

    def write_applied_changes(self, rec_id: str, files: Iterable[str]) -> None:
        """
        Emit an 'applied_changes' structured trace event and also append a human
        readable line to the per-repo app.log. The JSON trace entry will include
        an ISO-8601 ts and a data.files list containing the provided repo-relative
        file paths.

        This method writes the event to the current per-run trace file and also
        appends the same JSON line to <project_root>/.aidev/trace.jsonl so there is
        a canonical, easily-consumable record, and writes a short line to
        <project_root>/.aidev/app.log documenting the apply.
        """
        files_list = list(files)
        ts = _iso_now()
        entry: Dict[str, Any] = {
            "ts": ts,
            "event": "applied_changes",
            "kind": "apply",
            "data": _redact_obj({"rec_id": rec_id, "files": files_list}),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "host": _hostname(),
            "pid": os.getpid(),
            "model": None,
            "level": None,
            "phase": None,
            "tool": None,
            "latency_ms": None,
            "tokens_in": None,
            "tokens_out": None,
            "sse_events_emitted": self._sse_emitted_counter or None,
        }

        # Write to the per-run trace file
        try:
            self._validate(entry)
            self._append_jsonl(entry)
        except Exception:
            logger.exception("Failed writing applied_changes to per-run trace")

        # Also append the same JSONL line to a global .aidev/trace.jsonl (single-file)
        try:
            global_trace_dir = os.path.join(self.project_root, ".aidev")
            os.makedirs(global_trace_dir, exist_ok=True)
            global_trace_path = os.path.join(global_trace_dir, "trace.jsonl")
            line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            fd = os.open(global_trace_path, flags, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
                try:
                    os.fsync(fd)
                except Exception:
                    pass
            finally:
                try:
                    os.close(fd)
                except Exception:
                    pass
        except Exception:
            logger.exception("Failed appending applied_changes to global trace.jsonl")

        # Append a human-readable line to app.log documenting the apply
        try:
            app_log_path = os.path.join(self.project_root, ".aidev", "app.log")
            # Keep the files array serialized so it exactly matches the JSON payload
            human_line = f"{ts} applied_changes rec_id={rec_id} files={json.dumps(files_list, ensure_ascii=False)}\n"
            with open(app_log_path, "a", encoding="utf-8") as fh:
                fh.write(human_line)
                try:
                    fh.flush()
                    os.fsync(fh.fileno())
                except Exception:
                    pass
        except Exception:
            logger.exception("Failed appending applied_changes to app.log")

    @contextmanager
    def timer(self, event: str, kind: str, *,
              phase: Optional[str] = None,
              tool: Optional[str] = None,
              model: Optional[str] = None,
              level: Optional[str] = None,
              data_start: Any = None) -> Iterator[None]:
        """
        Context manager to time a block and emit start/end events.
        Emits <event>_start and <event>_end with latency_ms.
        """
        t0 = _dt.datetime.utcnow()
        self.write(f"{event}_start", kind, data_start or {}, phase=phase, tool=tool, model=model, level=level)
        try:
            yield
            ok = True
        except Exception as e:
            ok = False
            self.write(f"{event}_error", kind, {"message": str(e)}, phase=phase, tool=tool, model=model, level="error")
            raise
        finally:
            dt = (_dt.datetime.utcnow() - t0).total_seconds() * 1000.0
            self.write(f"{event}_end", kind, {"ok": ok}, phase=phase, tool=tool, model=model,
                       level=level, latency_ms=dt)

    def finalize(self, *, compress: Optional[bool] = None) -> None:
        """
        Close the current run. Optionally gzip the file.
        """
        self.write("run_end", "run", {"ok": True}, phase="finalize")
        try:
            if self._fd is not None:
                try:
                    os.fsync(self._fd)
                except Exception:
                    pass
                os.close(self._fd)
        finally:
            self._fd = None

        do_compress = self.compress_on_finalize if compress is None else bool(compress)
        if do_compress:
            try:
                self._gzip_inplace()
            except Exception:
                logger.debug("Trace compression failed", exc_info=True)

    # --- Internals ----------------------------------------------------------------

    def _make_run_id(self) -> str:
        ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        # short uuid suffix to avoid collisions in parallel runs
        return f"{ts}-{uuid.uuid4().hex[:6]}"

    def _validate(self, entry: Dict[str, Any]) -> None:
        if jsonschema is None:
            return
        try:
            jsonschema.validate(entry, EVENT_SCHEMA)  # type: ignore[attr-defined]
        except Exception as e:
            # Soft-fail: log but still write (traces should be resilient)
            logger.debug("Trace schema validation failed: %s; entry=%r", e, entry)

    def _ensure_fd(self) -> None:
        if self._fd is not None:
            return
        os.makedirs(self.traces_dir, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        # 0o644 by default
        self._fd = os.open(self.trace_path, flags, 0o644)

    def _append_jsonl(self, obj: Any) -> None:
        line = self._dumps(obj) + "\n"
        try:
            self._ensure_fd()
            assert self._fd is not None
            os.write(self._fd, line.encode("utf-8"))
            # For durability (optional, can be tuned if high QPS)
            os.fsync(self._fd)
        except Exception:
            logger.exception("Failed appending to trace file")
            # Best-effort fallback: write atomically by replace
            self._append_via_replace([line])

    def _append_via_replace(self, lines: Iterable[str]) -> None:
        """
        Fallback path similar to the previous implementation: read + replace atomically.
        This is only used on failure; normal path uses O_APPEND.
        """
        try:
            try:
                with open(self.trace_path, "r", encoding="utf-8") as f:
                    existing = f.read()
            except FileNotFoundError:
                existing = ""
            new_contents = existing
            if new_contents and not new_contents.endswith("\n"):
                new_contents += "\n"
            new_contents += "".join(lines)
            # Atomic replace
            dir_for_tmp = self.traces_dir or os.path.dirname(self.trace_path)
            fd, tmp_path = tempfile.mkstemp(prefix="trace.jsonl.", dir=dir_for_tmp)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                    f.write(new_contents)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.trace_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        except Exception:
            logger.exception("Replace-append fallback also failed")

    def _dumps(self, obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:  # never break tracing on serialization errors
            try:
                safe = _redact_obj(self._to_json_safe(obj))
                return json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                return json.dumps({"event": "trace_error", "kind": "serialize", "data": str(obj)[:1000]})

    def _to_json_safe(self, obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {str(k): self._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_json_safe(v) for v in obj]
        return repr(obj)

    # ---- Run pruning & compression ----------------------------------------------

    def _prune_old_runs(self) -> None:
        if self.keep_runs <= 0:
            return
        p = Path(self.traces_dir)
        cand = list(p.glob("run-*.jsonl")) + list(p.glob("run-*.jsonl.gz"))
        cand.sort(key=lambda q: q.stat().st_mtime, reverse=True)
        for path in cand[self.keep_runs:]:
            try:
                path.unlink()
            except Exception:
                logger.debug("Failed removing old run file: %s", path, exc_info=True)

    def _gzip_inplace(self) -> None:
        src = Path(self.trace_path)
        if not src.exists():
            return
        dst = src.with_suffix(src.suffix + ".gz")
        with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
            fo.writelines(fi)
        # keep original around if you prefer; here we remove to save space
        try:
            src.unlink()
        except Exception:
            pass


# ---- Demo ------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    tw = TraceWriter()
    tw.write("demo", "start", {"msg": "hello", "openai_api_key": "sk-XYZSHOULDHIDE"}, phase="init")
    with tw.timer("demo_block", "note", phase="work", tool="example"):
        tw.write_llm("llm_call_done", "openai", tokens_in=321, tokens_out=123, latency_ms=42.0,
                     model="gpt-5-mini", phase="work", tool="recommend")
    tw.accumulate_sse_emitted(3)
    tw.write("demo", "end", {"ok": True}, phase="finalize")
    tw.finalize(compress=False)
    print(f"Trace: {tw.path()}")