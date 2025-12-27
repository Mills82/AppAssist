# aidev/llm_client.py
from __future__ import annotations

import asyncio
import json
import os
import time
import math
import random
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

import httpx

try:
    # OpenAI Python SDK v1+
    from openai import OpenAI
    from openai import APIConnectionError, RateLimitError, APITimeoutError, APIStatusError
except Exception:
    # Soft dependency: tests/environments may not have openai installed.
    OpenAI = None  # type: ignore

    class APIConnectionError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class APIStatusError(Exception):
        pass


from .llm_utils import parse_json_object, parse_json_array, parse_analyze_plan_text, ParseError

# -------------------- Optional soft deps --------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # soft dependency

try:
    from jsonschema import validate as jsonschema_validate  # type: ignore
    from jsonschema.exceptions import ValidationError as JsonSchemaValidationError  # type: ignore
except Exception:
    jsonschema_validate = None
    JsonSchemaValidationError = Exception  # sentinel

# Lightweight logger shim
try:
    from . import logger  # aidev/logger.py shim if present
except Exception:
    import logging as _logging

    _log = _logging.getLogger("aidev.llm_client")

    class _Shim:
        def info(self, msg, ctx=None):
            _log.info(msg)

        def warn(self, msg, ctx=None, exc=None):
            _log.warning(msg)

        def warning(self, msg, ctx=None, exc=None):
            _log.warning(msg)  # ensure .warning exists

        def error(self, msg, ctx=None, exc=None):
            _log.error(msg)

        def exception(self, msg, ctx=None):
            _log.exception(msg)

    logger = _Shim()

# -------------------- Structured emit helper --------------------
def _emit_llm_log(level: str, msg: str, meta: Dict[str, Any], *, exc: Optional[Exception] = None) -> None:
    """
    Emit an LLM-oriented structured log via aidev.logger.log_llm_call when available,
    falling back to stdlib logging with the provided message + meta payload.
    """
    # 1) Preferred: structured logger pipeline
    try:
        fn = getattr(logger, "log_llm_call", None)
        if callable(fn):
            try:
                if exc is not None:
                    fn(level=level, msg=msg, meta=meta, exc=exc)
                else:
                    fn(level=level, msg=msg, meta=meta)
                return
            except TypeError:
                # Backward-compatible calling convention
                try:
                    if exc is not None:
                        fn(level, msg, meta, exc)
                    else:
                        fn(level, msg, meta)
                    return
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    # 2) Fallback: stdlib logging, but include meta so you don't lose diagnostics.
    try:
        # Best-effort JSON for meta
        try:
            meta_json = json.dumps(meta, ensure_ascii=False, default=str)
        except Exception:
            meta_json = str(meta)

        # Prevent giant log lines
        meta_json = _safe_truncate(meta_json, 8000)
        full_msg = f"{msg} | meta={meta_json}"

        # Use a stable non-root logger name (helps routing; avoids accidental filters)
        _stdlog = logging.getLogger("aidev.llm_client")

        if level == "info":
            _stdlog.info(full_msg)
        elif level in ("warn", "warning"):
            _stdlog.warning(full_msg)
        else:
            if exc is not None:
                _stdlog.error(full_msg, exc_info=True)
            else:
                _stdlog.error(full_msg)
    except Exception:
        # Absolute last resort
        try:
            logging.getLogger("aidev.llm_client").error(msg, exc_info=bool(exc))
        except Exception:
            pass


# Public wrapper for other modules -------------------------------------------------
def log_llm_call(
    level: str,
    msg: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    *,
    exc: Optional[Exception] = None,
    stage: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Public wrapper for structured LLM logging.
    """
    m = dict(meta or {})
    if stage:
        m.setdefault("stage", stage)
    if model:
        m.setdefault("model", model)
    if not msg:
        parts: List[str] = ["LLM call"]
        if stage:
            parts.append(f"stage={stage}")
        if model:
            parts.append(f"model={model}")
        msg = " ".join(parts)

    try:
        _emit_llm_log(level, msg, m, exc=exc)
    except Exception:
        try:
            if exc is not None:
                logging.error(msg, exc_info=True)
            else:
                if level == "info":
                    logging.info(msg)
                elif level in ("warn", "warning"):
                    logging.warning(msg)
                else:
                    logging.error(msg)
        except Exception:
            pass


# -------------------- Event emission helper (non-fatal, lazy) --------------------
def _emit_llm_event_model(stage: Optional[str], model: Optional[str]) -> None:
    """
    Best-effort emission of an llm_call event with the resolved model.
    """
    try:
        if not model:
            return
        try:
            from . import events
        except Exception:
            return

        meta = {"model": model, "stage": stage}

        fn = getattr(events, "_emit_llm_event_model", None)
        if callable(fn):
            try:
                try:
                    fn("llm_call", meta, session_id=None)
                    return
                except TypeError:
                    fn(payload=meta, model=model, session_id=None)
                    return
            except Exception:
                pass

        fn2 = getattr(events, "llm_call", None)
        if callable(fn2):
            try:
                try:
                    fn2(payload=meta, model=model, session_id=None)
                    return
                except TypeError:
                    fn2(meta, model, None)
                    return
            except Exception:
                pass

        fn3 = getattr(events, "emit", None)
        if callable(fn3):
            try:
                fn3("llm_call", meta)
                return
            except Exception:
                pass
    except Exception:
        pass


# -------------------- Small env helpers --------------------
def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)))
        return max(1, v)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """
    Parse a boolean-ish env var.
    Treats 0/false/no/off/"" as False; anything else as True.
    """
    try:
        v = os.getenv(name)
    except Exception:
        return default
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def _first_of_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


# -------------------- Public result container --------------------
@dataclass
class ChatResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: Any = None
    model: Optional[str] = None

    def __str__(self) -> str:
        return self.text or ""


# -------------------- Prompt loading / presets --------------------
_PROMPT_ALIASES: Dict[str, List[str]] = {
    "intent_classify": ["system.intent_classifier.md", "intent_classifier.md"],
    "recommendations": ["system.recommendations.md", "recommendations.md"],
    "select_targets": ["system.target_select.md", "select_targets.md"],
    "target_select": ["system.target_select.md", "select_targets.md"],
    "create_file": ["system.create_file.md", "create_file.md"],
    "edit_analyze": ["system.edit_analyze_file.md", "edit_analyze_file.md"],
    "edit_file": ["system.edit_file.md", "edit_file.md"],
    "edit_file_full": ["system.edit_file_full.md", "edit_file_full.md"],
    "repair_file": ["system.repair_file.md", "repair_file.md"],
    "summarize_file": ["system.card_summarize.md"],
    "card_summarize": ["system.card_summarize.md"],
    "brief_compile": ["system.brief_compile.md", "brief_compile.md"],
    "project_create": ["system.project_create.md", "project_create.md"],
    "qa": ["system.qa.md", "qa.md"],
    "analyze": ["system.analyze.md", "analyze.md"],
}

INCREMENTAL_GUIDELINES = (
    "INCREMENTAL_GUIDELINES:\n"
    "When producing recommendations, return a short ordered list of small, independently shippable steps. "
    "Each step must be independently testable and implementable, and must include a brief acceptance criteria entry that specifies how to verify success. "
    "Prefer minimal, localized changes over large refactors; ensure each recommendation can be completed and validated in isolation."
)


def _read_if_exists(p: Path) -> Optional[str]:
    try:
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return None


def load_prompt_any(
    candidates: List[str],
    base_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
) -> str:
    """
    Try aidev/prompts/<name> in package dir then CWD.
    Returns first prompt found, else "".
    """
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    search_roots = [base / "prompts", Path.cwd() / "aidev" / "prompts", Path.cwd() / "prompts"]

    for name in candidates:
        txt = _read_if_exists(search_roots[0] / name)
        if txt is not None:
            return txt
        for root in search_roots[1:]:
            txt = _read_if_exists(root / name)
            if txt is not None:
                return txt
    return ""


def system_preset(kind: str) -> str:
    aliases = _PROMPT_ALIASES.get(kind, [])
    if not aliases:
        raise RuntimeError(f"Unknown system prompt preset {kind!r}")

    txt = load_prompt_any(aliases)
    if not txt:
        raise RuntimeError(
            f"No prompt file found for preset {kind!r}; expected one of {aliases} "
            f"under aidev/prompts/ or ./aidev/prompts/ or ./prompts/"
        )
    return txt


# -------------------- Token helpers / packing --------------------
def _encoding_for_model(model: str):
    if not tiktoken:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(model: str, text: str) -> int:
    enc = _encoding_for_model(model)
    if not enc:
        return max(1, math.ceil(len(text) / 4))
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def _pack_messages_to_budget(model: str, messages: List[Dict[str, Any]], max_input_tokens: Optional[int]) -> List[Dict[str, Any]]:
    if not max_input_tokens:
        return messages

    def _as_text(c: Any) -> str:
        if isinstance(c, str):
            return c
        if isinstance(c, (dict, list)):
            try:
                return json.dumps(c, ensure_ascii=False)
            except Exception:
                return str(c)
        return str(c)

    toks = []
    total = 0
    for i, m in enumerate(messages):
        t = _as_text(m.get("content", ""))
        n = count_tokens(model, t)
        toks.append((i, n, t))
        total += n

    if total <= max_input_tokens:
        return messages

    result = [dict(m) for m in messages]

    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    def _is_jsonish(s: str) -> bool:
        ss = s.lstrip()
        return ss.startswith("{") or ss.startswith("[")

    drop_order = []
    for i, n, t in toks:
        role = messages[i].get("role")
        if i == last_user_idx:
            continue
        if role == "system":
            continue
        pri = 0 if role == "assistant" else (1 if role == "user" else 2)
        drop_order.append((pri, i, n, t))
    drop_order.sort(key=lambda x: (x[0], x[1]))

    for pri, i, n, t in drop_order:
        if total <= max_input_tokens:
            break
        if n < 64:
            continue
        result[i]["content"] = "[dropped to fit context budget]"
        total -= n

    if total <= max_input_tokens:
        return result

    for i, n, t in toks:
        if total <= max_input_tokens:
            break
        if i == last_user_idx:
            continue
        role = messages[i].get("role")
        if role in ("system",):
            continue

        cur = _as_text(result[i].get("content", ""))
        if cur == "[dropped to fit context budget]":
            continue
        if _is_jsonish(cur):
            result[i]["content"] = "[dropped JSON-like content to fit context budget]"
            total -= count_tokens(model, cur)
            continue

        keep_chars = max(256, int(len(cur) * 0.6))
        truncated = cur[:keep_chars] + "\n...[truncated]..."
        result[i]["content"] = truncated
        total -= max(0, count_tokens(model, cur) - count_tokens(model, truncated))

    return result


def _estimate_messages_chars(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, (dict, list)):
            try:
                total += len(json.dumps(c, ensure_ascii=False))
            except Exception:
                total += len(str(c))
        else:
            total += len(str(c))
    return total


# -------------------- Retry / backoff --------------------
def _retry_after_seconds(exc: Exception) -> Optional[float]:
    resp = getattr(exc, "response", None)
    try:
        if resp and getattr(resp, "headers", None):
            h = resp.headers
            if "Retry-After" in h:
                return float(h.get("Retry-After"))
            if any(k.lower().startswith("x-ratelimit") for k in h.keys()):
                return 1.0
    except Exception:
        pass
    return None


def _is_rate_limited_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "rate limit" in s or "too many requests" in s or "429" in s


def _is_timeout_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "timeout" in s or "timed out" in s


def _is_transient_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(
        k in s
        for k in (
            "temporarily",
            "unavailable",
            "gateway",
            "502",
            "503",
            "504",
            "connection reset",
            "service unavailable",
        )
    )


def _is_bad_structured_output_param_error(exc: Exception) -> bool:
    """
    Detect common mis-wires between:
      - legacy `response_format` (chat.completions)
      - Responses structured outputs via `text.format`
    """
    s = str(exc).lower()
    return any(
        k in s
        for k in (
            "response_format",
            "text.format",
            "text_format",
            "json_schema",
            "unknown parameter",
            "invalid parameter",
            "json object",
        )
    )


def _httpx_openai_error_hook(resp: httpx.Response) -> None:
    """
    httpx response hook: log OpenAI HTTP error bodies with request id + truncated body.
    Non-fatal; never raises.
    """
    try:
        if resp.status_code < 400:
            return

        req = getattr(resp, "request", None)
        url = str(getattr(req, "url", ""))

        # Only log OpenAI API endpoints (avoid noise if this client is reused)
        if "/v1/" not in url:
            return

        # Ensure body is available (buffers it). Safe: httpx keeps it once read.
        try:
            resp.read()
        except Exception:
            pass

        x_request_id = None
        try:
            x_request_id = (resp.headers or {}).get("x-request-id")
        except Exception:
            x_request_id = None

        body_json = None
        body_text = None
        try:
            body_json = resp.json()
        except Exception:
            try:
                body_text = resp.text
            except Exception:
                body_text = None

        meta = {
            "status_code": int(resp.status_code),
            "method": str(getattr(req, "method", "")),
            "url": url,
            "x_request_id": x_request_id,
            "body": body_json if body_json is not None else _safe_truncate(body_text or "", 4000),
        }

        # Route through your structured logger pipeline
        try:
            _emit_llm_log("error", "openai_http_error", meta)
        except Exception:
            # last resort
            try:
                logger.error("openai_http_error", ctx=meta)
            except Exception:
                pass
    except Exception:
        return


def _safe_truncate(s: Any, limit: int = 4000) -> str:
    try:
        if s is None:
            return ""
        s2 = str(s)
        if len(s2) <= limit:
            return s2
        return s2[:limit] + f"...(truncated, {len(s2)} chars total)"
    except Exception:
        return ""


def _extract_openai_error_fields(obj: Any) -> Dict[str, Any]:
    """
    Expected OpenAI error shape:
      {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}
    """
    if not isinstance(obj, dict):
        return {}
    err = obj.get("error")
    if isinstance(err, dict):
        out = {}
        for k in ("message", "type", "param", "code"):
            if k in err:
                out[k] = err.get(k)
        return out
    # fallback: sometimes error info can be top-level-ish
    out = {}
    for k in ("message", "type", "param", "code"):
        if k in obj:
            out[k] = obj.get(k)
    return out


def _have_any_api_key() -> Tuple[bool, str]:
    openai_key = _first_of_env("OPENAI_API_KEY")
    if openai_key:
        return True, ""
    return False, "Missing OPENAI_API_KEY."


# -------------------- Core client --------------------
class LLMClient:
    """
    OpenAI-only client adapter (Responses API primary).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 1800.0,
        max_retries: int = 0,
        backoff_base: float = 1.2,
        backoff_cap: float = 10.0,
        max_input_tokens: Optional[int] = None,
        *,
        stage: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> None:
        openai_key_env = _first_of_env("OPENAI_API_KEY")
        self.api_key = api_key or openai_key_env
        self.base_url = base_url or _first_of_env("OPENAI_BASE_URL", "OPENAI_API_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")

        if model_override and isinstance(model_override, str) and model_override.strip():
            self.model = model_override.strip()
        else:
            if stage:
                resolved = None
                try:
                    resolved = self._resolve_model_for_stage(stage)
                except Exception:
                    resolved = None
                if resolved:
                    self.model = resolved

        # Prefer Responses API
        self.use_responses = _env_bool("AIDEV_USE_RESPONSES", True)
        self.web_search_mode = (os.getenv("AIDEV_WEB_SEARCH", "auto") or "auto").strip().lower()

        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.max_input_tokens = max_input_tokens

        self.default_max_output_tokens = _env_int("AIDEV_DEFAULT_MAX_OUTPUT_TOKENS", 80000)
        self.max_output_cap = _env_int("AIDEV_OUTPUT_TOKENS_CAP", 80000)

        # Long-call mode settings (Responses background+polling)
        try:
            raw_thr = os.getenv("AIDEV_LONGCALL_CHAR_THRESHOLD", "0") or "0"
            self.longcall_char_threshold = int(str(raw_thr).replace("_", ""))
        except Exception:
            self.longcall_char_threshold = 0
        if self.longcall_char_threshold < 0:
            self.longcall_char_threshold = 0

        try:
            raw_poll = os.getenv("AIDEV_LONGCALL_POLL_INTERVAL_SEC", "5") or "5"
            self.longcall_poll_interval = float(str(raw_poll).strip())
        except Exception:
            self.longcall_poll_interval = 5.0
        if self.longcall_poll_interval <= 0:
            self.longcall_poll_interval = 5.0

        # Prompt cache / reasoning / service tier knobs (Responses-only)
        self.prompt_cache_enabled = _env_bool("AIDEV_PROMPT_CACHE", True)

        # ---- Structured Outputs enforcement knobs ----
        # chat_json MUST have a schema; fail-fast to surface miswired call paths.
        self.require_schema_for_chat_json = _env_bool("AIDEV_REQUIRE_SCHEMA_FOR_CHAT_JSON", True)

        # analyze-mode should also require schema when provided/expected; default True for fail-fast debugging.
        self.require_schema_for_analyze = _env_bool("AIDEV_REQUIRE_SCHEMA_FOR_ANALYZE", True)

        self.prompt_cache_retention = (os.getenv("AIDEV_PROMPT_CACHE_RETENTION", "24h") or "").strip() or None
        self.prompt_cache_retention_supported = _env_bool("AIDEV_PROMPT_CACHE_RETENTION_SUPPORTED", True)
        self.service_tier_default = (os.getenv("AIDEV_LLM_SERVICE_TIER", "auto") or "auto").strip()
        self.reasoning_default_effort = (os.getenv("AIDEV_LLM_REASONING_EFFORT", "medium") or "medium").strip()
        self.reasoning_enabled_default = _env_bool("AIDEV_LLM_REASONING", True)

        # Shared HTTP client with explicit timeouts
        http_timeout = httpx.Timeout(timeout=self.timeout, connect=30.0)
        self._http_client = httpx.Client(
            timeout=http_timeout,
            event_hooks={"response": [_httpx_openai_error_hook]},
        )
        self._closed = False

        try:
            logger.info(
                "LLM vendor selection",
                ctx={
                    "vendor": "openai",
                    "base_url": self.base_url or "default",
                    "model": self.model,
                    "timeout_sec": self.timeout,
                    "max_retries": self.max_retries,
                    "use_responses": self.use_responses,
                },
            )
        except Exception:
            pass

        # Instantiate OpenAI client
        try:
            if OpenAI is None:
                raise RuntimeError("openai package not installed.")
            if not self.api_key:
                raise RuntimeError("Missing OPENAI_API_KEY.")

            if self.base_url:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, http_client=self._http_client)
            else:
                self._client = OpenAI(api_key=self.api_key, http_client=self._http_client)

        except Exception as e:
            try:
                self.close()
            except Exception:
                pass
            raise RuntimeError(
                "Failed to initialize OpenAI client. Verify OPENAI_API_KEY and optional OPENAI_BASE_URL."
            ) from e

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        try:
            hc = getattr(self, "_http_client", None)
            if hc is not None:
                hc.close()
        except Exception:
            pass
        self._closed = True

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ----------------- Model resolution helper -----------------
    def _resolve_model_for_stage(self, stage: Optional[str]) -> Optional[str]:
        if not stage:
            return None
        try:
            key = re.sub(r"[^A-Za-z0-9]", "_", stage).upper()
            env_name = f"AIDEV_MODEL_{key}"
            m = os.getenv(env_name)
            if m and m.strip():
                return m.strip()
        except Exception:
            pass

        try:
            from . import config

            if hasattr(config, "get_model_for"):
                try:
                    mm = config.get_model_for(stage)
                    if mm:
                        if isinstance(mm, tuple) and len(mm) and isinstance(mm[0], str):
                            return mm[0]
                        if isinstance(mm, str):
                            return mm
                except Exception:
                    pass
            if hasattr(config, "get_model_for_stage"):
                try:
                    mm = config.get_model_for_stage(stage)
                    if mm:
                        if isinstance(mm, tuple) and len(mm) and isinstance(mm[0], str):
                            return mm[0]
                        if isinstance(mm, str):
                            return mm
                except Exception:
                    pass
            if hasattr(config, "model_map") and isinstance(getattr(config, "model_map"), dict):
                mm = config.model_map.get(stage)
                if mm:
                    if isinstance(mm, tuple) and len(mm) and isinstance(mm[0], str):
                        return mm[0]
                    if isinstance(mm, str):
                        return mm
        except Exception:
            pass

        try:
            default_env = os.getenv("AIDEV_DEFAULT_MODEL")
            if default_env and default_env.strip():
                return default_env.strip()
        except Exception:
            pass

        return None

    def _resolve_web_search_mode(self, extra: Optional[Dict[str, Any]]) -> str:
        mode = (self.web_search_mode or "auto").strip().lower()
        if not extra:
            return mode

        if isinstance(extra.get("disable_web_search"), bool):
            return "off" if extra["disable_web_search"] else mode

        m = extra.get("web_search_mode")
        if isinstance(m, str) and m.strip():
            m2 = m.strip().lower()
            if m2 in ("off", "on", "auto"):
                return m2
        return mode

    # --------------- High-level APIs ---------------
    def _should_add_web_search(self, messages: List[Dict[str, Any]], payload: Dict[str, Any]) -> bool:
        if "tools" in payload or "tool_choice" in payload:
            return False

        try:
            text = " ".join(
                (m.get("content") if isinstance(m.get("content"), str) else "")
                for m in messages
                if m.get("role") == "user"
            ).lower()
        except Exception:
            text = ""

        strong = (
            "today",
            "latest",
            "news",
            "breaking",
            "price",
            "prices",
            "changelog",
            "release",
            "version",
            "api docs",
            "documentation",
            "cve",
            "security",
            "vulnerability",
            "travel",
            "weather",
            "schedule",
            "stock",
            "crypto",
            "up-to-date",
            "as of",
            "who is",
            "what happened",
        )
        weak = ("best", "top", "review", "compare", "benchmarks", "buy", "cost")

        if any(k in text for k in strong):
            return True

        freshness = ("today", "latest", "current", "as of", "right now", "this week", "this month")
        if any(k in text for k in weak) and any(k in text for k in freshness):
            return True

        return False

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> ChatResponse:
        payload: Dict[str, Any] = {"model": self.model}
        phase: Optional[str] = None
        if extra:
            phase = extra.get("phase") or extra.get("llm_phase")

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if system:
            messages = [{"role": "system", "content": system}] + messages
        if extra:
            payload.update(extra)

        effective_model = None
        if model_override and isinstance(model_override, str) and model_override.strip():
            effective_model = model_override.strip()
        else:
            resolved = None
            try:
                resolved = self._resolve_model_for_stage(stage)
            except Exception:
                resolved = None
            if resolved:
                effective_model = resolved
        if not effective_model:
            effective_model = self.model

        payload["model"] = effective_model

        # gpt-5* rejects temperature; avoid a guaranteed 400.
        if self._model_disallows_temperature(effective_model):
            payload.pop("temperature", None)

        try:
            _emit_llm_log("info", "LLM call: resolved model", {"stage": stage or phase, "model": effective_model})
        except Exception:
            pass
        try:
            _emit_llm_event_model(stage or phase, effective_model)
        except Exception:
            pass

        effective_web_mode = self._resolve_web_search_mode(extra)
        if self.use_responses and effective_web_mode != "off":
            want_web = (
                effective_web_mode == "on"
                or (effective_web_mode == "auto" and self._should_add_web_search(messages, payload))
            )
            if want_web:
                tools = list(payload.get("tools", []))
                if not any(isinstance(t, dict) and t.get("type") == "web_search" for t in tools):
                    tools.append({"type": "web_search"})
                    payload["tools"] = tools
                payload.setdefault("tool_choice", "auto")

        payload = self._attach_metadata_and_cache(payload, phase=phase, extra=extra, effective_model=effective_model)
        messages = _pack_messages_to_budget(effective_model, messages, self.max_input_tokens)

        def _call() -> ChatResponse:
            return self._chat_once(messages, payload=payload, stream=stream, effective_model=effective_model)

        return self._with_retries(_call)

    def chat_json(
        self,
        messages: List[Dict[str, Any]],
        *,
        schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Tuple[Any, ChatResponse]:
        payload: Dict[str, Any] = {"model": self.model}
        phase: Optional[str] = None
        if extra:
            phase = extra.get("phase") or extra.get("llm_phase")
        phase_label = phase or "llm"

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if system:
            messages = [{"role": "system", "content": system}] + messages
        if extra:
            payload.update(extra)

        # Prefer Responses structured outputs via text.format; keep response_format only for chat fallback.
        # chat_json is schema-bound: fail fast if schema missing/empty.
        if self.require_schema_for_chat_json:
            schema = self._require_schema_or_raise(schema, phase=phase_label, model=self.model, where="LLMClient.chat_json")

        fmt = self._build_structured_text_format(schema)
        if not fmt:
            # This should never happen if schema is present; fail loudly.
            _emit_llm_log(
                "error",
                "Structured Outputs required but text.format could not be built",
                {"phase": phase_label, "model": self.model, "schema_id": self._schema_label(schema), "where": "LLMClient.chat_json"},
            )
            raise ValueError(f"LLMClient.chat_json: failed to build text.format for schema (phase={phase_label!r})")

        payload["_aidev_text_format"] = fmt  # internal key; applied only in Responses path
        payload["_aidev_require_structured"] = True

        effective_model = None
        if model_override and isinstance(model_override, str) and model_override.strip():
            effective_model = model_override.strip()
        else:
            resolved = None
            try:
                resolved = self._resolve_model_for_stage(stage or phase_label)
            except Exception:
                resolved = None
            if resolved:
                effective_model = resolved
        if not effective_model:
            effective_model = self.model

        payload["model"] = effective_model

        # gpt-5* rejects temperature; avoid a guaranteed 400.
        if self._model_disallows_temperature(effective_model):
            payload.pop("temperature", None)

        try:
            _emit_llm_log("info", "LLM call (json): resolved model", {"phase": phase_label, "model": effective_model})
        except Exception:
            pass
        try:
            _emit_llm_event_model(phase_label, effective_model)
        except Exception:
            pass

        effective_web_mode = self._resolve_web_search_mode(extra)
        if self.use_responses and effective_web_mode != "off":
            want_web = (
                effective_web_mode == "on"
                or (effective_web_mode == "auto" and self._should_add_web_search(messages, payload))
            )
            if want_web:
                tools = list(payload.get("tools", []))
                if not any(isinstance(t, dict) and t.get("type") == "web_search" for t in tools):
                    tools.append({"type": "web_search"})
                    payload["tools"] = tools
                payload.setdefault("tool_choice", "auto")

        payload = self._attach_metadata_and_cache(payload, phase=phase_label, extra=extra, effective_model=effective_model)
        messages = _pack_messages_to_budget(effective_model, messages, self.max_input_tokens)

        def _call() -> Tuple[Any, ChatResponse]:
            t0 = time.time()
            try:
                res = self._chat_once(messages, payload=payload, stream=False, effective_model=effective_model)

                # Prefer structured parsed output if available (Responses structured outputs).
                data: Any = None
                try:
                    if hasattr(res, "parsed") and res.parsed is not None:
                        data = res.parsed
                    elif getattr(res, "raw", None) is not None:
                        raw = res.raw
                        # Best-effort common locations for parsed output
                        for attr in ("parsed", "output_parsed"):
                            if hasattr(raw, attr) and getattr(raw, attr) is not None:
                                data = getattr(raw, attr)
                                break
                except Exception:
                    data = None

                text = (res.text or "").strip() if data is None else ""

                if text.startswith("```"):
                    m = re.search(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
                    if m:
                        text = m.group(1).strip()

                if data is None and text:
                    try:
                        data = json.loads(text)
                    except Exception:
                        try:
                            data = parse_json_object(text)
                        except Exception:
                            data = parse_json_array(text)

                # Schema-bound call must return JSON (dict/list). Never allow None to pass.
                if schema is not None:
                    if data is None:
                        raise ValueError("Structured (schema-bound) call returned no parseable JSON.")
                    if not isinstance(data, (dict, list)):
                        raise ValueError(
                            f"Structured (schema-bound) call returned non-JSON type: {type(data).__name__}"
                        )

                if schema and jsonschema_validate and data is not None:
                    try:
                        jsonschema_validate(data, schema)  # type: ignore[arg-type]
                    except JsonSchemaValidationError as ve:  # type: ignore[misc]
                        raise ValueError(f"Model returned JSON that failed schema validation: {ve}") from ve

                dt_ms = int((time.time() - t0) * 1000)
                resp_id = getattr(getattr(res, "raw", None), "id", None) or getattr(res, "raw", None) and getattr(res.raw, "response_id", None)  # type: ignore[attr-defined]
                meta = {
                    "phase": phase_label,
                    "response_id": resp_id,
                    "model": res.model or effective_model,
                    "latency_ms": int(dt_ms),
                    "tokens_in": int(res.prompt_tokens or 0),
                    "tokens_out": int(res.completion_tokens or 0),
                }
                _emit_llm_log("info", f"[llm_client.chat_json] success phase={phase_label}", meta)

                return data, res

            except Exception as e:
                dt_ms = int((time.time() - t0) * 1000)
                try:
                    _emit_llm_log(
                        "error",
                        f"[chat_json] exception phase={phase_label} after={dt_ms}ms type={type(e).__name__} msg={str(e)}",
                        {"phase": phase_label, "latency_ms": int(dt_ms), "exc_type": type(e).__name__, "exc_msg": str(e)},
                        exc=e,
                    )
                except Exception:
                    pass
                raise

        return self._with_retries(_call)

    # --------------- High-level: compiled project brief ---------------
    def compile_project_brief(
        self,
        app_text: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any], ChatResponse]:
        if schema is None:
            schema = project_brief_json_schema()

        file_prompt = system_preset("brief_compile") or ""
        default_system = (
            "You are a senior staff engineer.\n"
            "The user will paste the contents of app_descrip.txt: a free-form description of a software project.\n\n"
            "Your job is to:\n"
            "1) Rewrite it into a clean, concise Markdown project brief (project_description_md).\n"
            "2) Emit a small machine-readable metadata object (project_metadata).\n\n"
            "Respond with JSON ONLY, matching this schema:\n"
            "{\n"
            '  "project_description_md": "<markdown brief>",\n'
            '  "project_metadata": { ... arbitrary additional keys ... }\n'
            "}\n"
            "Do not wrap the JSON in markdown fences. Do not add extra top-level fields."
        )
        sys_prompt = system or file_prompt or default_system

        messages = [{"role": "user", "content": app_text}]
        data, resp = self.chat_json(
            messages,
            schema=schema,
            system=sys_prompt,
            max_tokens=max_tokens or _env_int("AIDEV_BRIEF_MAX_OUTPUT_TOKENS", 4096),
            extra={"phase": "brief_compile", "disable_web_search": True},
            stage="brief_compile",
        )

        if not isinstance(data, dict) or not data:
            md = (app_text or "").strip()
            return md, {}, resp

        md = str(data.get("project_description_md") or "").strip()
        meta = data.get("project_metadata")
        if not isinstance(meta, dict):
            meta = {}

        if not md:
            md = (app_text or "").strip()

        return md, meta, resp

    # --------------- Internals ---------------
    def _with_retries(self, call: Callable[[], Any]) -> Any:
        last_exc = None
        for attempt in range(int(self.max_retries) + 1):
            try:
                return call()
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break

                retryable = False
                try:
                    if isinstance(
                        e,
                        (
                            RateLimitError,
                            APITimeoutError,
                            APIConnectionError,
                            httpx.ConnectError,
                            httpx.TimeoutException,
                            httpx.ReadError,
                            httpx.RemoteProtocolError,
                        ),
                    ):
                        retryable = True
                    elif isinstance(e, APIStatusError):
                        status = getattr(e, "status_code", None) or getattr(e, "status", None)
                        if status in (429, 500, 502, 503, 504):
                            retryable = True
                except Exception:
                    retryable = False

                if not retryable:
                    retryable = (_is_rate_limited_error(e) or _is_timeout_error(e) or _is_transient_error(e))

                if not retryable:
                    break

                delay = None
                try:
                    delay = _retry_after_seconds(e)
                except Exception:
                    delay = None
                if not delay or delay <= 0:
                    base = min(self.backoff_cap, self.backoff_base * (2**attempt))
                    delay = base * (0.5 + random.random() * 0.5)
                else:
                    delay = min(float(delay), float(self.backoff_cap))

                try:
                    logger.warning(
                        "LLM call failed; backing off",
                        ctx={
                            "attempt": attempt + 1,
                            "of": self.max_retries + 1,
                            "err": str(e),
                            "sleep_s": round(delay, 3),
                        },
                    )
                except Exception:
                    pass

                time.sleep(delay)
                continue

        assert last_exc is not None
        raise last_exc

    # ----------------- Analyze-mode helper (private) -----------------
    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        if t.startswith("```"):
            m = re.search(r"^```(?:json)?\s*(.*?)\s*```$", t, flags=re.DOTALL | re.IGNORECASE)
            if m:
                return (m.group(1) or "").strip()
            return t.strip("`").strip()
        return t

    def _make_error_envelope(self, err_type: str, message: str, diagnostic: str = "") -> Dict[str, Any]:
        # Centralized envelope builder so analyze-mode returns deterministic structured diagnostics.
        diag = (diagnostic or "").strip()
        if len(diag) > 400:
            diag = diag[:400].rstrip() + "..."
        return {"error": {"type": err_type, "message": (message or "").strip(), "diagnostic": diag}}

    def _parse_json_deterministic(self, raw_text: str, *, schema: Optional[Dict[str, Any]] = None) -> Any:
        """Parse analyze-mode JSON using canonical parse_analyze_plan_text/ParseError.

        Returns parsed data on success, or a structured error envelope dict on parse/schema failure.
        The envelope shape is {"error": {"type": ..., "message": ..., "diagnostic": ...}} so callers/UI
        can deterministically render diagnostics without raw stack traces.
        """
        text = self._strip_markdown_fences(raw_text)
        if not text:
            return self._make_error_envelope(
                "parse_error",
                "Model returned empty output; expected JSON.",
                "Return JSON only, matching the required schema.",
            )

        try:
            # Delegate parsing and schema-aware checks to the shared utility.
            parsed = parse_analyze_plan_text(text, schema=schema) if schema is not None else parse_analyze_plan_text(text)

            return parsed

        except ParseError as pe:
            # Surface deterministic diagnostics derived from ParseError fields.
            try:
                msg = getattr(pe, "message", None) or str(pe)
                snippet = getattr(pe, "snippet", None) or getattr(pe, "text_snippet", None) or None
                suggestion = getattr(pe, "suggestion", None) or getattr(pe, "hint", None) or None

                parts: List[str] = []
                if snippet:
                    parts.append(f"snippet: {_safe_truncate(snippet, 300)}")
                if suggestion:
                    parts.append(f"suggestion: {_safe_truncate(suggestion, 300)}")
                diag = "; ".join(parts).strip()
                if not diag:
                    diag = _safe_truncate(msg, 400)

                return self._make_error_envelope("parse_error", msg, diag)
            except Exception:
                return self._make_error_envelope("parse_error", str(pe), "")
        except Exception:
            # Fallback deterministic envelope for unexpected parse errors.
            snip = _safe_truncate(text, 400)
            return self._make_error_envelope(
                "parse_error",
                "Could not parse model output as JSON.",
                f"Output snippet: {snip}",
            )

    def _call_analyze_model(
        self,
        messages: List[Dict[str, Any]],
        *,
        schema: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stage: str = "analyze",
        model_override: Optional[str] = None,
        max_parse_retries: int = 1,
    ) -> Tuple[str, Any, Optional[str], Dict[str, int]]:
        """
        Perform one analyze-mode request with bounded parse/schema retry.
        Returns (raw_text, parsed_or_error_envelope, model_id, usage_dict).
        """

        def _one_attempt(
            attempt: int,
            *,
            retry_note: Optional[str] = None,
        ) -> Tuple[str, Any, Optional[str], Dict[str, int]]:
            payload: Dict[str, Any] = {"model": self.model}
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

            if self.require_schema_for_analyze:
                schema = self._require_schema_or_raise(schema, phase=stage, model=self.model, where="LLMClient._call_analyze_model")

            fmt = self._build_structured_text_format(schema)
            if not fmt:
                _emit_llm_log(
                    "error",
                    "Structured Outputs required but text.format could not be built",
                    {"phase": stage, "model": self.model, "schema_id": self._schema_label(schema), "where": "LLMClient._call_analyze_model"},
                )
                raise ValueError(f"LLMClient._call_analyze_model: failed to build text.format for schema (stage={stage!r})")

            payload["_aidev_text_format"] = fmt
            payload["_aidev_require_structured"] = True

            effective_model = None
            if model_override and isinstance(model_override, str) and model_override.strip():
                effective_model = model_override.strip()
            else:
                resolved = None
                try:
                    resolved = self._resolve_model_for_stage(stage)
                except Exception:
                    resolved = None
                if resolved:
                    effective_model = resolved
            if not effective_model:
                effective_model = self.model
            payload["model"] = effective_model

            # gpt-5* rejects temperature; avoid a guaranteed 400.
            if self._model_disallows_temperature(effective_model):
                payload.pop("temperature", None)

            msgs = list(messages)
            if system:
                msgs = [{"role": "system", "content": system}] + msgs

            if retry_note:
                repair = (
                    "The previous response was invalid for this request.\n"
                    f"Error: {retry_note}\n"
                    "Return valid JSON only, matching the required schema. Do not include markdown fences or extra text."
                )
                if msgs and msgs[0].get("role") == "system":
                    msgs = [dict(msgs[0], content=str(msgs[0].get("content", "")) + "\n\n" + repair)] + msgs[1:]
                else:
                    msgs = [{"role": "system", "content": repair}] + msgs

            payload = self._attach_metadata_and_cache(payload, phase=stage, extra={"phase": stage}, effective_model=effective_model)
            msgs = _pack_messages_to_budget(effective_model, msgs, self.max_input_tokens)

            t0 = time.time()
            res = self._chat_once(msgs, payload=payload, stream=False, effective_model=effective_model)
            dt_ms = int((time.time() - t0) * 1000)

            raw_text = (res.text or "")
            model_id = res.model or effective_model
            usage = {
                "prompt_tokens": int(res.prompt_tokens or 0),
                "completion_tokens": int(res.completion_tokens or 0),
                "total_tokens": int(res.total_tokens or 0),
            }

            try:
                _emit_llm_log(
                    "info",
                    "analyze result",
                    {
                        "stage": stage,
                        "model": model_id,
                        "attempt": int(attempt),
                        "latency_ms": int(dt_ms),
                        "tokens_in": usage["prompt_tokens"],
                        "tokens_out": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                    },
                )
            except Exception:
                pass
            try:
                _emit_llm_event_model(stage, model_id)
            except Exception:
                pass

            parsed = self._parse_json_deterministic(raw_text, schema=schema)
            return raw_text, parsed, model_id, usage

        raw1, parsed1, model1, usage1 = _one_attempt(1)
        # If ParseError-derived envelope (deterministic error) and retry allowed, do exactly one retry.
        if isinstance(parsed1, dict) and isinstance(parsed1.get("error"), dict):
            err = parsed1["error"]
            if max_parse_retries and int(max_parse_retries) > 0 and err.get("type") in ("parse_error", "schema_error"):
                note = f"{err.get('type')}: {err.get('message') or ''}".strip()
                raw2, parsed2, model2, usage2 = _one_attempt(2, retry_note=note)
                # Return the second attempt's result; callers can inspect error envelope.
                return raw2, parsed2, model2, usage2
        return raw1, parsed1, model1, usage1

    @staticmethod
    def _sanitize_text_format_name(name: Any, *, fallback: str = "response", max_len: int = 64) -> str:
        """
        OpenAI Responses API requires text.format.name to match: ^[a-zA-Z0-9_-]+$
        Schemas often use $id URLs or titles with spaces/punctuation. Sanitize deterministically.
        """
        try:
            s = str(name or "").strip()
        except Exception:
            s = ""
        if not s:
            s = fallback
        # Replace invalid chars with underscore
        s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
        # Trim underscores
        s = s.strip("_-")
        if not s:
            s = fallback
        if max_len and len(s) > max_len:
            s = s[:max_len].rstrip("_-")
            if not s:
                s = fallback
        return s

    @staticmethod
    def _schema_label(schema: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Human/debug label for a schema (stable-ish).
        Prefer $id, then title, then name.
        """
        if not isinstance(schema, dict) or not schema:
            return None
        try:
            v = schema.get("$id") or schema.get("title") or schema.get("name")
            if v is None:
                return None
            s = str(v).strip()
            return s or None
        except Exception:
            return None

    def _require_schema_or_raise(
        self,
        schema: Optional[Dict[str, Any]],
        *,
        phase: Optional[str],
        model: Optional[str],
        where: str,
    ) -> Dict[str, Any]:
        """
        Enforce that schema is present and non-empty for schema-bound calls.
        """
        if isinstance(schema, dict) and schema:
            return schema

        meta = {
            "where": where,
            "phase": phase,
            "model": model,
            "schema_present": bool(schema),
            "schema_type": type(schema).__name__ if schema is not None else None,
        }
        _emit_llm_log("error", "Structured Outputs required but schema is missing/empty", meta)
        raise ValueError(f"{where}: schema is required for structured output but was missing/empty (phase={phase!r})")

    def _structured_meta_from_text_format(self, tf: Any) -> Dict[str, Any]:
        """
        Extract structured-output diagnostics from a text.format dict.
        """
        if not isinstance(tf, dict) or not tf:
            return {
                "has_text_format": False,
                "text_format_type": None,
                "text_format_name": None,
                "text_format_strict": None,
                "text_format_has_schema": False,
                "schema_id": None,
            }

        schema = tf.get("schema") if isinstance(tf, dict) else None
        schema_id = self._schema_label(schema) if isinstance(schema, dict) else None

        return {
            "has_text_format": True,
            "text_format_type": tf.get("type"),
            "text_format_name": tf.get("name"),
            "text_format_strict": tf.get("strict"),
            "text_format_has_schema": bool(isinstance(schema, dict) and schema),
            "schema_id": schema_id,
        }

    def _build_structured_text_format(self, schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Responses API Structured Outputs via text.format.

        Expected shape:
        text: { format: { type: "json_schema", name: "...", schema: {...}, strict: true } }
        """
        if not schema:
            return {}

        raw_name = schema.get("$id") or schema.get("title") or schema.get("name")
        name = self._sanitize_text_format_name(raw_name, fallback="response")

        return {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": True,
        }

    @staticmethod
    def _build_chat_response_format_json_mode() -> Dict[str, Any]:
        """
        Chat Completions fallback: JSON mode only (no schema).
        """
        return {"type": "json_object"}

    def _normalize_messages_for_chat(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                parts_text: List[str] = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") in ("text", "output_text", "input_text"):
                        t = p.get("text")
                        if isinstance(t, str):
                            parts_text.append(t)
                if parts_text:
                    out.append({"role": role, "content": "\n\n".join(parts_text)})
                else:
                    out.append({"role": role, "content": str(content)})
            elif isinstance(content, dict):
                try:
                    out.append({"role": role, "content": json.dumps(content, ensure_ascii=False)})
                except Exception:
                    out.append({"role": role, "content": str(content)})
            else:
                out.append({"role": role, "content": str(content)})
        return out

    def _messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list) and all(isinstance(x, dict) and "type" in x for x in content):
                out.append({"role": role, "content": content})
                continue
            if isinstance(content, dict):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)
            out.append({"role": role, "content": [{"type": "input_text", "text": str(content)}]})
        return out

    def _extract_instructions_and_strip_system(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        instructions = None
        kept: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system" and instructions is None:
                c = m.get("content", "")
                if isinstance(c, list):
                    parts = []
                    for p in c:
                        if isinstance(p, dict) and p.get("type") in ("text", "output_text", "input_text"):
                            t = p.get("text")
                            if isinstance(t, str):
                                parts.append(t)
                    instructions = "\n\n".join(parts) if parts else (c if isinstance(c, str) else str(c))
                    continue
                instructions = c if isinstance(c, str) else str(c)
                continue
            kept.append(m)
        return instructions, kept

    @staticmethod
    def _first_text_from_responses(resp: Any) -> str:
        try:
            ot = getattr(resp, "output_text", None)
            if isinstance(ot, str) and ot.strip():
                return ot
            if isinstance(ot, list):
                buf = [str(t) for t in ot if isinstance(t, str) and t.strip()]
                if buf:
                    return "\n".join(buf)
        except Exception:
            pass

        try:
            out = getattr(resp, "output", None)
            if isinstance(out, list):
                chunks = []
                for item in out:
                    content = getattr(item, "content", None)
                    if isinstance(content, list):
                        for part in content:
                            t = getattr(part, "text", None)
                            if isinstance(t, str) and t.strip():
                                chunks.append(t)
                            if isinstance(part, dict):
                                t2 = part.get("text") or part.get("output_text")
                                if isinstance(t2, str) and t2.strip():
                                    chunks.append(t2)
                if chunks:
                    return "\n".join(chunks)
        except Exception:
            pass

        def _deep_texts(obj):
            acc = []
            if isinstance(obj, dict):
                if isinstance(obj.get("text"), str) and obj.get("text").strip():
                    acc.append(obj["text"])
                for v in obj.values():
                    acc.extend(_deep_texts(v))
            elif isinstance(obj, list):
                for v in obj:
                    acc.extend(_deep_texts(v))
            else:
                t = getattr(obj, "text", None)
                if isinstance(t, str) and t.strip():
                    acc.append(t)
            return acc

        try:
            if hasattr(resp, "model_dump"):
                dumped = resp.model_dump()  # type: ignore[attr-defined]
                texts = _deep_texts(dumped)
                if texts:
                    return "\n".join(texts)
        except Exception:
            pass
        return ""

    @staticmethod
    def _pluck(d: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
        return {k: d[k] for k in keys if k in d}

    @staticmethod
    def _remap_max_tokens_for_responses(payload: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "max_output_tokens" in payload and payload["max_output_tokens"] is not None:
            out["max_output_tokens"] = payload["max_output_tokens"]
        elif "max_tokens" in payload and payload["max_tokens"] is not None:
            out["max_output_tokens"] = payload["max_tokens"]
        return out

    @staticmethod
    def _remap_max_tokens_for_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "max_tokens" in payload and payload["max_tokens"] is not None:
            out["max_tokens"] = payload["max_tokens"]
        return out

    def _effective_max_out(self, explicit: Optional[int]) -> int:
        val = int(explicit) if explicit is not None else int(self.default_max_output_tokens)
        return min(max(1, val), int(self.max_output_cap))

    def _compute_prompt_cache_key(
        self,
        payload: Dict[str, Any],
        phase: Optional[str],
        effective_model: Optional[str] = None,
    ) -> Optional[str]:
        if not (self.prompt_cache_enabled and self.use_responses):
            return None

        explicit = payload.get("prompt_cache_key")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()

        st = getattr(self, "st", None)
        project_id = None
        project_root = None
        git_sha = None

        if st is not None:
            for attr in ("project_id", "project_name", "id"):
                project_id = getattr(st, attr, None)
                if project_id:
                    break
            project_root = getattr(st, "project_root", None) or getattr(st, "root", None)
            for attr in ("git_sha", "git_commit", "git_head", "git_rev"):
                git_sha = getattr(st, attr, None)
                if git_sha:
                    break

        base = project_id or (str(project_root) if project_root else None) or "aidev-project"
        phase_label = phase or payload.get("phase") or "generic"
        model_label = effective_model or self.model
        key_raw = f"{base}:{phase_label}:{model_label}"
        if git_sha:
            key_raw += f":{git_sha}"

        try:
            import hashlib

            return hashlib.sha256(key_raw.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return key_raw[:128]

    def _attach_metadata_and_cache(
        self,
        payload: Dict[str, Any],
        *,
        phase: Optional[str],
        extra: Optional[Dict[str, Any]],
        effective_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            meta: Dict[str, Any] = dict(payload.get("metadata") or {})
        except Exception:
            meta = {}

        st = getattr(self, "st", None)
        project_root = None
        if st is not None:
            project_root = getattr(st, "project_root", None) or getattr(st, "root", None)
        if project_root and "project_root" not in meta:
            meta["project_root"] = str(project_root)[:256]

        phase_label = phase or meta.get("phase")
        if extra:
            phase_label = phase_label or extra.get("phase") or extra.get("llm_phase")
            job_id = extra.get("job_id")
            rec_id = extra.get("rec_id")
        else:
            job_id = None
            rec_id = None

        job_id = job_id or getattr(self, "job_id", None) or meta.get("job_id")
        rec_id = rec_id or getattr(self, "rec_id", None) or meta.get("rec_id")

        if phase_label and "phase" not in meta:
            meta["phase"] = str(phase_label)
        if job_id and "job_id" not in meta:
            meta["job_id"] = str(job_id)[:128]
        if rec_id and "rec_id" not in meta:
            meta["rec_id"] = str(rec_id)[:128]

        if meta:
            payload["metadata"] = meta

        cache_key = self._compute_prompt_cache_key(payload, phase_label, effective_model=effective_model)
        if cache_key:
            payload["prompt_cache_key"] = cache_key
            if (
                self.prompt_cache_retention
                and getattr(self, "prompt_cache_retention_supported", True)
                and not payload.get("prompt_cache_retention")
            ):
                payload["prompt_cache_retention"] = self.prompt_cache_retention

        if self.use_responses:
            svc = None
            if extra:
                svc = extra.get("service_tier")
            svc = (svc or self.service_tier_default or "auto").strip()
            if svc:
                payload.setdefault("service_tier", svc)

        model_lower = str(effective_model or self.model).lower()
        if self.use_responses and self.reasoning_enabled_default and model_lower.startswith("gpt-5") and "reasoning" not in payload:
            phase_for_reasoning = (phase_label or "").lower()
            reasoning_phases = {"recommendations", "plan", "cards", "card_summarize", "analyze", "edit_analyze"}
            if phase_for_reasoning in reasoning_phases:
                effort = self.reasoning_default_effort or "medium"
                payload["reasoning"] = {"effort": effort}
                include = list(payload.get("include") or [])
                if "reasoning.encrypted_content" not in include:
                    include.append("reasoning.encrypted_content")
                payload["include"] = include

        return payload

    @staticmethod
    def _usage_numbers(usage_obj: Any) -> Tuple[int, int, int]:
        if usage_obj is None:
            return 0, 0, 0
        if isinstance(usage_obj, dict):
            pt = int(usage_obj.get("prompt_tokens", 0) or usage_obj.get("input_tokens", 0) or 0)
            ct = int(usage_obj.get("completion_tokens", 0) or usage_obj.get("output_tokens", 0) or 0)
            tt = int(usage_obj.get("total_tokens", 0) or (pt + ct))
            return pt, ct, tt
        pt = int(getattr(usage_obj, "prompt_tokens", getattr(usage_obj, "input_tokens", 0)) or 0)
        ct = int(getattr(usage_obj, "completion_tokens", getattr(usage_obj, "output_tokens", 0)) or 0)
        total = getattr(usage_obj, "total_tokens", None)
        tt = int(total if total is not None else (pt + ct))
        return pt, ct, tt

    @staticmethod
    def _strip_temperature(d: Dict[str, Any]) -> Dict[str, Any]:
        if "temperature" in d:
            d = dict(d)
            d.pop("temperature", None)
        return d

    @staticmethod
    def _is_bad_content_type_error(e: Exception) -> bool:
        s = str(e).lower()
        return ("invalid type for 'messages" in s or "invalid type for messages" in s) and "content" in s

    @staticmethod
    def _is_bad_temperature_error(e: Exception) -> bool:
        s = str(e).lower()
        return (
            "temperature" in s
            and (
                "unsupported parameter" in s
                or "not supported" in s
                or "only the default" in s
                or "param': 'temperature" in s
            )
        )

    @staticmethod
    def _model_supports_chat(model: str) -> bool:
        m = model.lower()
        return m.startswith(("gpt-4o", "gpt-4.1", "gpt-3.5"))

    @staticmethod
    def _model_disallows_temperature(model: str) -> bool:
        m = (model or "").lower().strip()
        return m.startswith("gpt-5")

    def _log_api_error(self, e: Exception) -> None:
        """
        Best-effort error-body logging for OpenAI SDK errors (esp. APIStatusError).
        Logs status, x-request-id, parsed error fields (message/type/param/code), and a truncated body.
        """
        try:
            resp = getattr(e, "response", None)
            if resp is None:
                return

            status = getattr(resp, "status_code", None)

            # headers / request id
            try:
                x_request_id = None
                hdrs = getattr(resp, "headers", None)
                if hdrs:
                    x_request_id = hdrs.get("x-request-id")
            except Exception:
                x_request_id = None

            parsed = None
            raw_text = None
            try:
                parsed = resp.json()
            except Exception:
                try:
                    raw_text = resp.text
                except Exception:
                    raw_text = None

            err_fields = _extract_openai_error_fields(parsed) if parsed is not None else {}

            # log safely (avoid huge dumps)
            ctx = {
                "status": status,
                "x_request_id": x_request_id,
                "error_type": err_fields.get("type"),
                "error_code": err_fields.get("code"),
                "error_param": err_fields.get("param"),
                "error_message": _safe_truncate(err_fields.get("message"), 1000),
                "body": _safe_truncate(json.dumps(parsed, ensure_ascii=False) if parsed is not None else (raw_text or ""), 4000),
            }

            # Use your shim logger so it matches the rest of your logs
            logger.error("OpenAI API error", ctx=ctx, exc=e)
        except Exception:
            # never crash on logging
            try:
                logger.error("OpenAI API error (logging failed)", ctx={"exc_type": type(e).__name__, "exc_msg": str(e)[:400]})
            except Exception:
                pass

    def _should_use_long_call(self, payload: Dict[str, Any], messages: List[Dict[str, Any]]) -> bool:
        if not self.use_responses:
            return False

        raw_hint = None
        if "long_call_hint" in payload:
            raw_hint = payload["long_call_hint"]
        elif "long_call_mode" in payload:
            raw_hint = payload["long_call_mode"]

        if isinstance(raw_hint, bool):
            return raw_hint
        if isinstance(raw_hint, (int, str)):
            try:
                s = str(raw_hint).strip().lower()
            except Exception:
                s = ""
            if s in ("1", "true", "yes", "on", "long", "background"):
                return True
            if s in ("0", "false", "no", "off", "short", "normal"):
                return False

        thr = getattr(self, "longcall_char_threshold", 0) or 0
        if thr <= 0:
            return False

        try:
            total_chars = _estimate_messages_chars(messages)
        except Exception:
            return False

        try:
            logger.info(
                "LLM long-call decision",
                ctx={"total_chars": total_chars, "threshold": thr, "use_long_call": bool(total_chars >= thr)},
            )
        except Exception:
            pass

        return total_chars >= thr

    # -------------------- Core call --------------------
    def _chat_once(
        self,
        messages: List[Dict[str, Any]],
        *,
        payload: Dict[str, Any],
        stream: bool,
        effective_model: Optional[str] = None,
    ) -> ChatResponse:
        model_to_use = effective_model or self.model

        try:
            phase_label = None
            if isinstance(payload, dict):
                phase_label = (payload.get("metadata") or {}).get("phase") or payload.get("phase")
            _emit_llm_event_model(phase_label, model_to_use)
        except Exception:
            pass

        messages_chat = self._normalize_messages_for_chat(messages)
        payload_any = dict(payload)

        require_structured = bool(payload_any.pop("_aidev_require_structured", False))

        # Compatibility shim:
        # - If caller passed legacy response_format, don't send it to Responses; keep only for chat fallback.
        legacy_rf = payload_any.pop("response_format", None)

        # Apply text.format for Responses from our internal key
        aidev_text_format = payload_any.pop("_aidev_text_format", None)

        # ---- Definitive structured-output preflight (one line per call) ----
        try:
            phase_label = None
            if isinstance(payload, dict):
                phase_label = (payload.get("metadata") or {}).get("phase") or payload.get("phase")

            tf_meta = self._structured_meta_from_text_format(aidev_text_format)

            _emit_llm_log(
                "info",
                "llm_call.preflight",
                {
                    "model": model_to_use,
                    "phase": phase_label,
                    "require_structured": bool(require_structured),
                    **tf_meta,
                },
            )

            # Fail fast if caller required structured outputs but we don't actually have a schema attached.
            if require_structured and not tf_meta.get("has_text_format"):
                _emit_llm_log(
                    "error",
                    "require_structured=True but no text.format was attached (host-side bug)",
                    {"model": model_to_use, "phase": phase_label, **tf_meta},
                )
                raise ValueError("require_structured=True but payload is missing text.format (schema not attached)")

            # Also fail fast if it isn't strict json_schema mode.
            if require_structured:
                if tf_meta.get("text_format_type") != "json_schema" or not tf_meta.get("text_format_has_schema"):
                    _emit_llm_log(
                        "error",
                        "require_structured=True but text.format is not strict json_schema with schema",
                        {"model": model_to_use, "phase": phase_label, **tf_meta},
                    )
                    raise ValueError("require_structured=True but text.format is not strict json_schema with schema")

        except Exception:
            # If we raised intentionally above, let it propagate. Otherwise don't break the call due to logging.
            raise

        # ---------- OpenAI path (prefer Responses API) ----------
        if self.use_responses and hasattr(self._client, "responses"):
            def _responses_call(_payload: Dict[str, Any]) -> ChatResponse:
                instr, msgs_wo_system = self._extract_instructions_and_strip_system(messages)
                extra_instr = {"instructions": instr} if instr else {}

                remapped = self._remap_max_tokens_for_responses(_payload)
                eff = self._effective_max_out(_payload.get("max_tokens") or _payload.get("max_output_tokens"))
                if "max_output_tokens" not in remapped or not remapped["max_output_tokens"]:
                    remapped["max_output_tokens"] = eff
                else:
                    remapped["max_output_tokens"] = self._effective_max_out(remapped["max_output_tokens"])

                input_payload: Dict[str, Any] = {
                    "model": model_to_use,
                    "input": self._messages_to_responses_input(msgs_wo_system),
                    **extra_instr,
                    **remapped,
                    **self._pluck(
                        _payload,
                        {
                            "metadata",
                            "seed",
                            "temperature",
                            "tools",
                            "tool_choice",
                            "include",
                            "reasoning",
                            "service_tier",
                            "parallel_tool_calls",
                            "max_tool_calls",
                            "top_p",
                            "top_logprobs",
                            "prompt_cache_key",
                            "prompt_cache_retention",
                            "safety_identifier",
                        },
                    ),
                }

                # Attach structured output format via Responses text.format, if present.
                # Shape: text={"format": {...}}
                text_format = aidev_text_format  # avoid assigning to the closed-over var
                if isinstance(text_format, dict) and text_format:
                    # Defensive: ensure name matches OpenAI regex ^[a-zA-Z0-9_-]+$
                    tf = dict(text_format)
                    if "name" in tf:
                        tf["name"] = self._sanitize_text_format_name(tf.get("name"))
                    input_payload["text"] = {"format": tf}

                if require_structured and not ((input_payload.get("text") or {}).get("format")):
                    raise ValueError("require_structured=True but payload.text.format is missing")

                # Some models (e.g., gpt-5*) reject temperature entirely.
                if self._model_disallows_temperature(model_to_use):
                    input_payload.pop("temperature", None)

                tf_obj = (input_payload.get("text") or {}).get("format")
                tf_type = tf_obj.get("type") if isinstance(tf_obj, dict) else None
                tf_name = tf_obj.get("name") if isinstance(tf_obj, dict) else None
                tf_strict = tf_obj.get("strict") if isinstance(tf_obj, dict) else None
                tf_has_schema = bool(tf_obj.get("schema")) if isinstance(tf_obj, dict) else False

                _emit_llm_log(
                    "info",
                    "responses.create payload structured?",
                    {
                        "model": model_to_use,
                        "phase": (payload.get("metadata") or {}).get("phase") or payload.get("phase"),
                        "require_structured": bool(require_structured),
                        "has_text_format": bool(tf_obj),
                        "text_format_type": tf_type,
                        "text_format_name": tf_name,
                        "text_format_strict": tf_strict,
                        "text_format_has_schema": tf_has_schema,
                    },
                )

                # Extra guardrail log: if you required structured, warn if we're not actually in strict json_schema mode.
                if require_structured and (tf_type != "json_schema" or not tf_has_schema):
                    _emit_llm_log(
                        "warning",
                        "require_structured=True but Responses text.format is not strict json_schema",
                        {
                            "model": model_to_use,
                            "phase": (payload.get("metadata") or {}).get("phase") or payload.get("phase"),
                            "text_format_type": tf_type,
                            "text_format_name": tf_name,
                            "text_format_strict": tf_strict,
                            "text_format_has_schema": tf_has_schema,
                        },
                    )

                use_background = self._should_use_long_call(_payload, messages)

                def _responses_create_with_retries(p: Dict[str, Any]):
                    max_attempts = getattr(self, "responses_create_max_attempts", 2)
                    base_delay = getattr(self, "responses_create_retry_delay", 1.0)
                    delay = float(base_delay)
                    attempt = 0
                    last_exc: Optional[Exception] = None

                    while attempt < max_attempts:
                        attempt += 1
                        try:
                            return self._client.responses.create(**p)
                        except (
                            APIConnectionError,
                            APITimeoutError,
                            httpx.ConnectError,
                            httpx.TimeoutException,
                            httpx.ReadError,
                            httpx.RemoteProtocolError,
                        ) as e:
                            last_exc = e
                            try:
                                _emit_llm_log(
                                    "warning",
                                    "[llm_client] transient connection error on responses.create",
                                    {
                                        "model": model_to_use,
                                        "attempt": attempt,
                                        "max_attempts": max_attempts,
                                        "delay_s": round(delay, 2),
                                        "error": str(e),
                                    },
                                    exc=e,
                                )
                            except Exception:
                                pass
                            if attempt >= max_attempts:
                                break
                            time.sleep(delay)
                            delay = min(delay * 2.0, 10.0)

                    if last_exc is not None:
                        raise last_exc
                    raise RuntimeError("responses.create failed with no exception captured")

                def _do_call(p: Dict[str, Any]) -> ChatResponse:
                    if stream and not use_background:
                        chunks: List[str] = []
                        with self._client.responses.stream(**p) as s:
                            for event in s:
                                if event.type == "response.output_text.delta":
                                    chunks.append(event.delta)
                            s.close()
                        return ChatResponse(text="".join(chunks), raw=None, total_tokens=0, model=model_to_use)

                    resp = _responses_create_with_retries(p)
                    text = self._first_text_from_responses(resp)
                    usage = getattr(resp, "usage", None) or {}
                    pt, ct, tt = self._usage_numbers(usage)
                    return ChatResponse(text=text or "", prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, raw=resp, model=model_to_use)

                def _long_call(p: Dict[str, Any]) -> ChatResponse:
                    p2 = dict(p)
                    p2["background"] = True
                    p2.setdefault("store", True)

                    start_ts = time.time()
                    poll_count = 0

                    hard_timeout = float(getattr(self, "timeout", None) or 900.0)
                    hard_deadline = start_ts + hard_timeout

                    resp = _responses_create_with_retries(p2)
                    response_id = getattr(resp, "id", None) or getattr(resp, "response_id", None)
                    status = getattr(resp, "status", None) or getattr(resp, "status_code", None)

                    interval = float(getattr(self, "longcall_poll_interval", 5.0) or 5.0)
                    try:
                        max_interval = float(
                            getattr(self, "longcall_poll_max_interval", None)
                            or (os.getenv("AIDEV_LONGCALL_POLL_MAX_INTERVAL_SEC", "") or "").strip()
                            or 30.0
                        )
                    except Exception:
                        max_interval = 30.0
                    if max_interval <= 0:
                        max_interval = 30.0
                    max_interval = max(max_interval, interval)

                    def _jitter(x: float) -> float:
                        return max(0.5, x * (0.9 + random.random() * 0.2))

                    interval = _jitter(interval)

                    final = resp
                    if status not in ("completed", "done"):
                        if not response_id:
                            raise RuntimeError("Responses background call did not return a response_id.")
                        while True:
                            now = time.time()
                            if now >= hard_deadline:
                                raise TimeoutError(
                                    f"Long-call polling exceeded deadline for response_id={response_id} "
                                    f"(timeout={hard_timeout}s). Last status={status!r}"
                                )

                            time.sleep(interval)
                            poll_count += 1
                            try:
                                final = self._client.responses.retrieve(response_id)
                            except (
                                APIConnectionError,
                                APITimeoutError,
                                httpx.ConnectError,
                                httpx.TimeoutException,
                                httpx.ReadError,
                                httpx.RemoteProtocolError,
                                RateLimitError,
                            ) as e:
                                try:
                                    _emit_llm_log(
                                        "warning",
                                        "[llm_client] long-call transient error on responses.retrieve",
                                        {
                                            "model": model_to_use,
                                            "response_id": response_id,
                                            "poll": poll_count,
                                            "interval_s": round(interval, 2),
                                            "error": str(e),
                                        },
                                        exc=e,
                                    )
                                except Exception:
                                    pass

                                ra = None
                                try:
                                    ra = _retry_after_seconds(e)
                                except Exception:
                                    ra = None
                                if ra and ra > 0:
                                    interval = min(max_interval, max(interval, float(ra)))
                                else:
                                    interval = min(interval * 1.5, max_interval)
                                interval = _jitter(interval)
                                continue

                            except APIStatusError as e:
                                st = getattr(e, "status_code", None) or getattr(e, "status", None)
                                if st == 429:
                                    ra = None
                                    try:
                                        ra = _retry_after_seconds(e)
                                    except Exception:
                                        ra = None
                                    if ra and ra > 0:
                                        interval = min(max_interval, max(interval, float(ra)))
                                    else:
                                        interval = min(interval * 1.5, max_interval)
                                    interval = _jitter(interval)
                                    continue
                                raise

                            status = getattr(final, "status", None) or getattr(final, "status_code", None)
                            if status in ("completed", "done"):
                                break
                            if status in ("failed", "cancelled"):
                                # Try to extract structured error info if present
                                err_obj = getattr(final, "error", None)
                                err_dump = None
                                try:
                                    if err_obj is not None:
                                        # OpenAI SDK objects sometimes have model_dump()
                                        if hasattr(err_obj, "model_dump"):
                                            err_dump = err_obj.model_dump()  # type: ignore[attr-defined]
                                        elif isinstance(err_obj, dict):
                                            err_dump = err_obj
                                        else:
                                            err_dump = {"error": str(err_obj)}
                                except Exception:
                                    err_dump = {"error": str(err_obj)[:2000]} if err_obj is not None else None

                                try:
                                    _emit_llm_log(
                                        "error",
                                        "[llm_client] long-call failed",
                                        {
                                            "model": model_to_use,
                                            "response_id": response_id,
                                            "status": status,
                                            "polls": int(poll_count),
                                            "error": err_dump,
                                        },
                                    )
                                except Exception:
                                    pass

                                raise RuntimeError(
                                    f"Responses background call {response_id} ended with status={status}. error={_safe_truncate(err_dump, 2000)}"
                                )

                            interval = _jitter(min(interval * 1.05, max_interval))

                    text = self._first_text_from_responses(final)
                    usage = getattr(final, "usage", None) or {}
                    pt, ct, tt = self._usage_numbers(usage)

                    elapsed = time.time() - start_ts
                    _emit_llm_log(
                        "info",
                        "[llm_client] long-call done",
                        {
                            "model": model_to_use,
                            "response_id": response_id,
                            "status": status,
                            "polls": int(poll_count),
                            "latency_ms": int(elapsed * 1000),
                            "tokens_in": int(pt or 0),
                            "tokens_out": int(ct or 0),
                            "total_tokens": int(tt or 0),
                        },
                    )
                    return ChatResponse(text=text or "", prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, raw=final, model=model_to_use)

                caller = _long_call if use_background else _do_call

                try:
                    return caller(input_payload)

                except Exception as e:
                    self._log_api_error(e)
                    es = str(e).lower()

                    # prompt_cache_retention not supported -> disable and retry
                    if "prompt_cache_retention" in es and ("not support" in es or "invalid" in es or "unknown" in es):
                        self.prompt_cache_retention_supported = False
                        self.prompt_cache_retention = None
                        p4 = dict(input_payload)
                        p4.pop("prompt_cache_retention", None)
                        return caller(p4)

                    # If structured output param is rejected, drop it and retry once (unstructured)
                    if _is_bad_structured_output_param_error(e):
                        if require_structured:
                            raise  # schema was required; don't silently degrade
                        p5 = dict(input_payload)
                        p5.pop("text", None)
                        return caller(p5)

                    # If server complains about max_output_tokens limits, clamp and retry
                    if "max_output_tokens" in es and ("require" in es or "exceed" in es):
                        p3 = dict(input_payload)
                        p3["max_output_tokens"] = min(512, int(p3.get("max_output_tokens", 512) or 512))
                        return caller(p3)

                    # If Responses isn't supported in this env, fall back to chat.completions
                    if "responses" in es and ("not supported" in es or "use chat.completions" in es):
                        self.use_responses = False
                        raise

                    raise

            try:
                return _responses_call(payload_any)
            except Exception as e:
                # IMPORTANT: if structured was required, do NOT fall through.
                # Surface the real root cause (status=failed, server error, etc.).
                try:
                    _emit_llm_log(
                        "error",
                        "[llm_client] Responses call failed",
                        {
                            "model": model_to_use,
                            "phase": (payload.get("metadata") or {}).get("phase") or payload.get("phase"),
                            "require_structured": bool(require_structured),
                            "exc_type": type(e).__name__,
                            "exc_msg": str(e)[:2000],
                        },
                        exc=e,
                    )
                except Exception:
                    pass

                if require_structured:
                    raise

                # If we get obvious format miswire, retry once without text.format
                if (_is_bad_structured_output_param_error(e) or self._is_bad_temperature_error(e)) and not require_structured:
                    saved = aidev_text_format
                    try:
                        aidev_text_format = None
                        safe_payload = self._strip_temperature(dict(payload_any))
                        return _responses_call(safe_payload)
                    finally:
                        aidev_text_format = saved

                # Otherwise fall through to chat fallback below

        # ---------- OpenAI fallback: chat.completions (only for chat-capable models) ----------
        def _chat_call(_payload: Dict[str, Any]) -> ChatResponse:
            if "max_tokens" not in _payload or _payload["max_tokens"] is None:
                _payload = dict(_payload, max_tokens=self._effective_max_out(None))
            else:
                _payload = dict(_payload, max_tokens=self._effective_max_out(_payload["max_tokens"]))

            payload2 = dict(_payload)
            payload2.pop("tools", None)
            payload2.pop("tool_choice", None)
            payload2.pop("service_tier", None)
            payload2.pop("include", None)
            payload2.pop("reasoning", None)
            payload2.pop("prompt_cache_key", None)
            payload2.pop("prompt_cache_retention", None)

            # If we wanted JSON, use JSON mode in chat fallback (response_format).
            if isinstance(legacy_rf, dict) and legacy_rf:
                payload2["response_format"] = legacy_rf
            elif isinstance(aidev_text_format, dict) and aidev_text_format:
                # We were in structured mode; chat fallback can only do json_object.
                payload2["response_format"] = self._build_chat_response_format_json_mode()

            if stream:
                chunks: List[str] = []
                for ev in self._client.chat.completions.create(
                    model=model_to_use,
                    messages=messages_chat,
                    **self._remap_max_tokens_for_chat(payload2),
                    stream=True,
                    **self._pluck(payload2, {"response_format", "metadata", "seed", "temperature"}),
                ):
                    delta = (ev.choices and ev.choices[0].delta.content) or ""
                    if delta:
                        chunks.append(delta)
                return ChatResponse(text="".join(chunks), model=model_to_use)
            else:
                resp = self._client.chat.completions.create(
                    model=model_to_use,
                    messages=messages_chat,
                    **self._remap_max_tokens_for_chat(payload2),
                    **self._pluck(payload2, {"response_format", "metadata", "seed", "temperature"}),
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                pt, ct, tt = self._usage_numbers(getattr(resp, "usage", None))
                return ChatResponse(text=text, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt, raw=resp, model=model_to_use)

        if require_structured:
            raise RuntimeError("Structured Outputs required but Responses call failed; refusing chat.completions fallback.")

        if self._model_supports_chat(model_to_use):
            try:
                return _chat_call(payload_any)
            except Exception as e:
                self._log_api_error(e)
                if self._is_bad_content_type_error(e) or self._is_bad_temperature_error(e) or _is_bad_structured_output_param_error(e):
                    safe_payload = self._strip_temperature(dict(payload_any))
                    safe_payload.pop("response_format", None)
                    return _chat_call(safe_payload)
                raise

        raise RuntimeError(
            f"Model '{model_to_use}' does not support chat.completions fallback; Responses API failed or was unavailable."
        )

    # -------------------- File summarization (instance) --------------------
    def summarize_file(self, rel_path: str, text: str, max_tokens: int = 4096) -> str:
        card = self.summarize_file_card(rel_path=rel_path, text=text or "", max_tokens=max_tokens)

        raw_lines = card.get("lines") or card.get("summary_lines") or []
        if isinstance(raw_lines, str):
            raw_lines = [raw_lines]

        parts: List[str] = []
        for line in raw_lines:
            if isinstance(line, str):
                s = line.strip()
                if s:
                    parts.append(s)

        if not parts:
            fallback = card.get("summary") or card.get("text") or card.get("description") or ""
            if isinstance(fallback, str):
                parts.append(fallback.strip())

        raw = " ".join(parts).strip()
        if not raw:
            return ""
        return _clean_summary_output(raw)

    async def summarize_file_async(self, rel_path: str, text: str, max_tokens: int = 4096) -> str:
        try:
            to_thread = getattr(asyncio, "to_thread", None)
            if callable(to_thread):
                return await to_thread(self.summarize_file, rel_path, text, max_tokens)
        except Exception:
            pass
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.summarize_file, rel_path, text, max_tokens)

    async def summarize_file_card_async(
        self,
        rel_path: str,
        text: str,
        *,
        max_tokens: int = 4096,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            to_thread = getattr(asyncio, "to_thread", None)
            if callable(to_thread):
                return await to_thread(self.summarize_file_card, rel_path, text, max_tokens=max_tokens, schema=schema)
        except Exception:
            pass

        loop = asyncio.get_running_loop()

        def _call() -> Dict[str, Any]:
            return self.summarize_file_card(rel_path, text, max_tokens=max_tokens, schema=schema)

        return await loop.run_in_executor(None, _call)

    def summarize_file_card(
        self,
        rel_path: str,
        text: str,
        *,
        max_tokens: int = 4096,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        system = system_preset("card_summarize")

        input_limit = int(os.getenv("AIDEV_CARD_INPUT_TOKENS", os.getenv("AIDEV_SUMMARY_INPUT_TOKENS", "60000")))
        summary_input = _prepare_summary_input(
            model=self.model,
            system=system,
            rel_path=rel_path,
            text=text or "",
            input_limit=input_limit,
        )

        payload_schema = schema or _load_ai_summary_schema()
        messages = [{"role": "user", "content": summary_input}]

        data, resp = self.chat_json(
            messages,
            schema=payload_schema,
            system=system,
            max_tokens=max_tokens,
            extra={"phase": "card_summarize"},
            stage="card_summarize",
        )

        if not isinstance(data, dict):
            raise ValueError(f"summarize_file_card: expected dict from model (ai_summary), got {type(data).__name__}")

        ai_summary: Dict[str, Any] = dict(data)

        card: Dict[str, Any] = {"path": rel_path, "ai_summary": ai_summary}
        card["title"] = str(Path(rel_path).name)

        def _coerce_lines(val: Any) -> List[str]:
            out: List[str] = []
            if isinstance(val, str):
                s = val.strip()
                if s:
                    out.append(s)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        s = item.strip()
                        if s:
                            out.append(s)
            return out

        lines: List[str] = []

        summaries_obj = ai_summary.get("summaries")
        if isinstance(summaries_obj, dict):
            lines.extend(_coerce_lines(summaries_obj.get("summary_short")))
            if not lines:
                lines.extend(_coerce_lines(summaries_obj.get("summary_long")))

        if not lines:
            lines.extend(_coerce_lines(ai_summary.get("what")))
        if not lines:
            lines.extend(_coerce_lines(ai_summary.get("why")))
        if not lines:
            lines.extend(_coerce_lines(ai_summary.get("how")))

        seen = set()
        safe_lines: List[str] = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                safe_lines.append(line)
            if len(safe_lines) >= 8:
                break

        if not safe_lines:
            fallback_text = ""
            if isinstance(resp.text, str):
                fallback_text = _clean_summary_output(resp.text)
            if fallback_text:
                safe_lines.append(fallback_text)

        card["lines"] = safe_lines
        return card


# -------------------- Summary utilities --------------------
def _prepare_summary_input(
    *,
    model: str,
    system: str,
    rel_path: str,
    text: str,
    input_limit: int,
    min_body_tokens: int = 256,
) -> str:
    overhead_estimate = count_tokens(model, system) + 512
    body_allow = max(min_body_tokens, input_limit - overhead_estimate)
    snippet = _truncate_to_tokens(model, text or "", body_allow)

    ext = Path(rel_path).suffix.lower()
    language_map = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".rst": "rst",
        ".txt": "text",
        ".py": "python",
        ".ipynb": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "javascriptreact",
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".conf": "ini",
        ".cfg": "ini",
        ".env": "dotenv",
        ".php": "php",
        ".rb": "ruby",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".swift": "swift",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".hh": "cpp",
        ".hxx": "cpp",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".dart": "dart",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".ps1": "powershell",
        ".psm1": "powershell",
        ".bat": "batch",
        ".cmd": "batch",
        ".lua": "lua",
        ".luau": "lua",
        ".sql": "sql",
        ".csv": "text",
        ".tsv": "text",
        ".xml": "xml",
        ".xhtml": "xml",
        ".vue": "vue",
        ".svelte": "svelte",
    }
    language = language_map.get(ext, "text")

    contents_wrapped = f"BEGIN_FILE_CONTENT\n{snippet}\nEND_FILE_CONTENT\n"

    payload: Dict[str, Any] = {
        "mode": "summarize_file_card",
        "file": {
            "path": rel_path,
            "language": language,
            "contents": contents_wrapped,
        },
        "context": {},
    }
    return json.dumps(payload, ensure_ascii=False)


def _clean_summary_output(raw: str) -> str:
    out = (raw or "").strip()
    if not out:
        return ""

    if out.startswith("```"):
        m = re.search(r"^```(?:\w+)?\s*(.*?)\s*```$", out, flags=re.DOTALL)
        if m:
            out = m.group(1).strip()
        else:
            out = out.strip("`").strip()

    out = " ".join(out.split())

    try:
        max_chars = int(os.getenv("AIDEV_SUMMARY_MAX_CHARS", "800"))
    except Exception:
        max_chars = 800

    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "..."
    return out


def _truncate_to_tokens(model: str, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if tiktoken is None:
        approx_chars = max_tokens * 4
        return text[:approx_chars] + ("...[truncated]..." if len(text) > approx_chars else "")
    try:
        enc = _encoding_for_model(model) or tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        cut = enc.decode(toks[:max_tokens])
        return cut + " ...[truncated]..."
    except Exception:
        approx_chars = max_tokens * 4
        return text[:approx_chars] + ("...[truncated]..." if len(text) > approx_chars else "")


def _compact_index_for_deep(card_index: Dict[str, Any], *, model: str, token_budget: int = 7000) -> str:
    if not card_index:
        return "(no cards)"
    items: List[Tuple[str, Dict[str, Any]]] = []
    if "nodes" in card_index and "cards" in card_index:
        for rel in card_index.get("nodes") or []:
            entry = (card_index.get("cards") or {}).get(rel) or {}
            items.append((str(rel), dict(entry)))
    else:
        for rel, entry in card_index.items():
            if isinstance(entry, dict):
                items.append((str(rel), dict(entry)))

    lines: List[str] = []
    for rel, entry in items[:]:
        ttl = (entry.get("title") or "")[:120]
        ai = entry.get("ai_summary") or {}
        ai_text = ai.get("text") if isinstance(ai, dict) else (ai if isinstance(ai, str) else "")
        if not ai_text:
            ai_text = entry.get("summary") or ""
        ai_text = str(ai_text or "")
        ai_text = " ".join(ai_text.split())
        lines.append(f"- {rel} — {ttl or 'untitled'} :: {ai_text}")

    blob = "\n".join(lines)
    return _truncate_to_tokens(model, blob, token_budget)


# -------- Repo map helpers (load + compact) --------
def _load_repo_map(root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pmap = _safe_load_json(root / ".aidev" / "project_map.json") or {}
    if isinstance(pmap, dict) and "cards" in pmap and isinstance(pmap["cards"], dict):
        card_index = {"nodes": list(pmap["cards"].keys()), "cards": pmap["cards"]}
    elif isinstance(pmap, dict) and pmap:
        card_index = {"nodes": list(pmap.keys()), "cards": pmap}
    else:
        card_index = {"nodes": [], "cards": {}}
    return pmap, card_index


def _repo_map_compact_for_prompt(client: "LLMClient", project_root: Optional[str], *, token_budget: int) -> str:
    if not project_root:
        return ""
    root = Path(project_root).resolve()
    _pmap, card_index = _load_repo_map(root)
    if not card_index.get("nodes"):
        return ""
    compact = _compact_index_for_deep(card_index, model=client.model, token_budget=max(500, token_budget))
    return "Project file index (compact):\n" + compact + "\n\nGuidance: prefer minimal, surgical edits and use the existing layout."


# -------------------- Optional tool schema --------------------
def summarize_file_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "emit_summary",
            "description": "Emit a 1–3 sentence plain-English summary of a source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Concise summary of the file's purpose and behavior."}
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    }


def _load_targets_schema() -> Dict[str, Any]:
    try:
        from .schemas import targets_schema

        return targets_schema()
    except Exception as e:
        try:
            logger.warning("Failed to load canonical 'targets' schema; using fallback schema", ctx={"err": str(e)})
        except Exception:
            pass
        return {
            "$id": "targets_envelope",
            "title": "TargetsEnvelope",
            "type": "object",
            "properties": {
                "targets": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": True},
                },
                "notes": {"type": "string"},
            },
            "required": ["targets"],
            "additionalProperties": True,
        }


def _load_ai_summary_schema() -> Dict[str, Any]:
    try:
        from .schemas import ai_summary_schema

        return ai_summary_schema()
    except Exception as e:
        try:
            logger.warning("Failed to load canonical 'ai_summary' schema; using fallback schema", ctx={"err": str(e)})
        except Exception:
            pass
        return {"$id": "ai_summary_fallback", "title": "AIDevAiSummaryFallback", "type": "object", "additionalProperties": True}


# -------------------- Lightweight fallbacks for missing schema helpers --------------------
def diff_json_schema() -> Dict[str, Any]:
    try:
        from .schemas import diff_json_schema as s
    except Exception:
        s = None
    if s:
        try:
            return s()
        except Exception:
            pass
    return {
        "$id": "diffs_v1",
        "title": "DiffsEnvelope",
        "type": "object",
        "properties": {
            "diffs": {
                "type": "array",
                "items": {"type": "object", "properties": {"path": {"type": "string"}, "patch": {"type": "string"}}, "additionalProperties": True},
            }
        },
        "required": ["diffs"],
        "additionalProperties": True,
    }


def project_brief_json_schema() -> Dict[str, Any]:
    try:
        from .schemas import project_brief_json_schema as s
    except Exception:
        s = None
    if s:
        try:
            return s()
        except Exception:
            pass
    return {
        "$id": "project_brief_v1",
        "title": "ProjectBrief",
        "type": "object",
        "properties": {"project_description_md": {"type": "string"}, "project_metadata": {"type": "object", "additionalProperties": True}},
        "required": ["project_description_md", "project_metadata"],
        "additionalProperties": True,
    }


def select_targets_for_rec(
    client: Optional[LLMClient],
    recommendation_text: str,
    *,
    project_root: Optional[str] = None,
    max_tokens: int = 8192,
) -> Tuple[List[Dict[str, Any]], ChatResponse]:
    data, resp = select_targets_envelope_for_rec(client=client, recommendation_text=recommendation_text, project_root=project_root, max_tokens=max_tokens)
    targets = []
    if isinstance(data, dict):
        targets = data.get("targets") or []
    return targets, resp


def select_targets_envelope_for_rec(
    client: Optional["LLMClient"],
    recommendation_text: str,
    *,
    project_root: Optional[str] = None,
    max_tokens: int = 8192,
) -> Tuple[Dict[str, Any], ChatResponse]:
    client = client or LLMClient()
    system = system_preset("select_targets")
    schema = _load_targets_schema()

    use_repo_map = _env_bool("AIDEV_USE_REPO_MAP", True)
    map_snip = (
        _repo_map_compact_for_prompt(client, project_root, token_budget=int(os.getenv("AIDEV_REPO_MAP_TOKENS", "2500")))
        if use_repo_map
        else ""
    )

    spec_blob = "Change request / spec:\n\n" + recommendation_text
    if map_snip:
        spec_blob += "\n\n" + map_snip
    spec_blob += "\n\nReturn JSON only."

    messages = [{"role": "user", "content": spec_blob}]
    if max_tokens is None:
        max_tokens = _env_int("AIDEV_DIFFS_MAX_OUTPUT_TOKENS", 60000)

    data, resp = client.chat_json(messages, schema=schema, system=system, max_tokens=max_tokens)

    # In structured mode, never return a schema-invalid fallback. Surface the real failure.
    if not isinstance(data, dict):
        raise ValueError(
            f"Model returned non-object JSON for TargetsEnvelope (type={type(data).__name__})."
        )

    return data, resp



def compile_project_brief(
    app_text: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, Dict[str, Any], ChatResponse]:
    client = LLMClient(model=model, max_input_tokens=int(os.getenv("AIDEV_BRIEF_INPUT_TOKENS", "8000")))
    try:
        return client.compile_project_brief(app_text, system=system, max_tokens=max_tokens)
    finally:
        try:
            client.close()
        except Exception:
            pass


def summarize_file(
    path: Path,
    content: str,
    context: Dict[str, Any] | None = None,
    *,
    model: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    client = LLMClient(model=model)
    try:
        return client.summarize_file(rel_path=str(path), text=content, max_tokens=max_tokens)
    finally:
        try:
            client.close()
        except Exception:
            pass


async def summarize_file_async(
    path: Path,
    content: str,
    context: Dict[str, Any] | None = None,
    *,
    model: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    client = LLMClient(model=model)
    try:
        return await client.summarize_file_async(rel_path=str(path), text=content, max_tokens=max_tokens)
    finally:
        try:
            client.close()
        except Exception:
            pass


async def summarize_file_card_async(
    rel_path: Union[str, Path],
    text: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = LLMClient(model=model)
    try:
        return await client.summarize_file_card_async(rel_path=str(rel_path), text=text, max_tokens=max_tokens, schema=schema)
    finally:
        try:
            client.close()
        except Exception:
            pass


# -------------------- Endpoint-facing async helpers --------------------
async def summarize_changed(payload: Dict[str, Any]) -> Dict[str, Any]:
    have_key, key_err = _have_any_api_key()
    if not have_key:
        return {"ok": False, "error": key_err, "message": key_err, "summarized_count": 0, "skipped_count": 0}

    project_root = await _resolve_project_root_from_payload(payload)
    root = Path(project_root).resolve()

    try:
        from .config import load_project_config
        from .structure import discover_structure
        from .cards import KnowledgeBase
    except Exception as e:
        msg = f"Failed to import KnowledgeBase/config/structure: {e}"
        logger.warning("summarize_changed: imports failed", ctx={"err": str(e)})
        return {"ok": False, "error": msg, "message": msg, "summarized_count": 0, "skipped_count": 0}

    try:
        cfg, _ = load_project_config(root, None)
        includes = list((cfg.get("discovery", {}) or {}).get("includes", []))
        excludes = list((cfg.get("discovery", {}) or {}).get("excludes", []))

        struct, _ctx = discover_structure(root, includes, excludes, max_total_kb=128, strip_comments=False)
        kb = KnowledgeBase(root, struct)

        paths = payload.get("files") or payload.get("paths")
        compute_embeddings = payload.get("compute_embeddings")
        max_files = payload.get("max_files")

        result = kb.summarize_changed(paths=paths, model=payload.get("model"), compute_embeddings=compute_embeddings, max_files=max_files)
        if not isinstance(result, dict):
            raise RuntimeError("KnowledgeBase.summarize_changed returned non-dict")

        summarized = int(result.get("updated", 0))
        skipped = int(result.get("skipped", 0))
        message = result.get("message") or f"Summarized {summarized} file(s), skipped {skipped}."

        return {
            "ok": bool(result.get("ok", True)),
            "error": result.get("error"),
            "message": message,
            "summarized_count": summarized,
            "skipped_count": skipped,
        }
    except Exception as e:
        msg = str(e)
        logger.warning("summarize_changed: failure", ctx={"err": msg[:400]})
        return {"ok": False, "error": msg, "message": msg, "summarized_count": 0, "skipped_count": 0}


# -------------------- Support: project root, file IO, discovery --------------------
async def _resolve_project_root_from_payload(payload: Dict[str, Any]) -> str:
    for k in ("project_root", "root"):
        v = payload.get(k)
        if v:
            return str(Path(v).resolve())

    sess_id = payload.get("session_id")
    if sess_id:
        try:
            from .session_store import SESSIONS  # type: ignore

            session = await SESSIONS.get(sess_id)
            pr = getattr(session, "project_root", None) or getattr(session, "root", None)
            if pr:
                return str(Path(pr).resolve())
        except Exception:
            pass
    return str(Path.cwd().resolve())


_TEXT_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".md",
    ".txt",
    ".rst",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".dart",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".hpp",
    ".cpp",
    ".cs",
    ".m",
    ".mm",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
    ".twig",
    ".vue",
    ".svelte",
}


def _safe_read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(errors="ignore")
        except Exception:
            return ""


def _safe_load_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _safe_dump_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def _too_large(path: Path, max_bytes: int = 2_000_000) -> bool:
    try:
        return path.stat().st_size > max_bytes
    except Exception:
        return True


def _discover_changed_files(root: Path, *, ttl_days: int = 3) -> List[Path]:
    for name in ("changed_files.json", "changed_files.json300s"):
        ch = _safe_load_json(root / ".aidev" / name)
        if isinstance(ch, dict) and isinstance(ch.get("files"), list):
            out_files: List[Path] = []
            for f in ch["files"]:
                p = Path(f)
                if not p.is_absolute():
                    p = root / f
                if p.exists() and p.is_file():
                    out_files.append(p)
            if out_files:
                return out_files

    out: List[Path] = []
    try:
        import subprocess

        cmd = ["git", "-C", str(root), "diff", "--name-only", "HEAD"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                p = (root / line.strip()).resolve()
                if p.suffix.lower() in _TEXT_EXTS and p.exists() and p.is_file():
                    out.append(p)
            if out:
                return out
    except Exception:
        pass

    return _discover_recent_text_files(root, within_days=ttl_days)


def _discover_recent_text_files(root: Path, *, within_days: int = 3, cap: int = 200) -> List[Path]:
    now = time.time()
    max_age = within_days * 86400
    candidates: List[Tuple[float, Path]] = []
    for p in root.rglob("*"):
        try:
            if p.is_file() and p.suffix.lower() in _TEXT_EXTS:
                mt = p.stat().st_mtime
                if (now - mt) <= max_age:
                    candidates.append((mt, p))
        except Exception:
            continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates[:cap]]


# -------------------- Exports --------------------
ChatGPT = LLMClient

__all__ = [
    "LLMClient",
    "ChatGPT",
    "ChatResponse",
    "load_prompt_any",
    "system_preset",
    "summarize_file",
    "summarize_changed",
    "diff_json_schema",
    "project_brief_json_schema",
    "select_targets_envelope_for_rec",
    "select_targets_for_rec",
    "compile_project_brief",
    "INCREMENTAL_GUIDELINES",
    "log_llm_call",
    "summarize_file_async",
    "summarize_file_card_async",
]
