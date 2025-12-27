# aidev/orchestration/edit_strategy.py
"""
Tiered per-file edit runner.

Implements a two-attempt edit flow per file:
 - Attempt #1: patch-capable edit via preset "edit_file" (allows `patch_unified` OR `content`).
   - If `patch_unified` returned: apply strict unified patch against the exact snapshot provided.
   - If patch application fails (invalid diff OR context mismatch OR unknown): do NOT attempt patch repair.
     Immediately fallback to Attempt #2.
   - If the model response is invalid (schema/empty output/path mismatch), fallback to Attempt #2.
 - Attempt #2: full-file edit via preset "edit_file_full" (must return full `content`, patch fields rejected).

Critical invariants enforced here:
 - Patch baseline MUST be the exact full snapshot the model saw: `file.current` == base_snapshot (LF-normalized).
 - preview/snippet content MUST NOT be used as canonical content or patch baseline.
 - When a context payload is provided (recommended), it MUST be used for BOTH attempts (patch + full fallback),
   so the payload matches the edit_file / edit_file_full system prompt contract.

Model selection (supports tiering):
 - Patch attempt model: env AIDEV_MODEL_EDIT_PATCH, else AIDEV_MODEL_GENERATE_EDITS, else (explicit `model`), else "gpt-5-mini"
 - Full-file attempt model: env AIDEV_MODEL_EDIT_FULL, else AIDEV_MODEL_GENERATE_EDITS, else (explicit `model`), else "gpt-5-mini"

IMPORTANT:
 - The `model` argument is treated as a *fallback base* only. It MUST NOT override
   the tiered env vars when they are present.
 - If you pass `model_override`, it is a fallback only; this file enforces tiered env precedence.

Observability:
 - Debug log of baseline snapshot hash/len.
 - Patch failure emits a failure_class and logs an INFO-level patch excerpt.
 - Patch text in payload.details is truncated to avoid giant SSE/trace payloads.

Exports:
 - apply_tiered_edit_to_file(...)
 - classify_patch_failure(error_text)
 - TieredEditError
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

JsonSchema = Dict[str, Any]

PATCH_PROMPT_PRESET = "edit_file"
FULL_PROMPT_PRESET = "edit_file_full"


# Try importing repo modules.
try:
    from aidev import io_utils  # provides apply_unified_patch
except Exception as e:  # pragma: no cover
    # In production/runtime usage we MUST fail rather than silently using a fake patcher.
    raise ImportError(
        "Failed to import aidev.io_utils (required). "
        "Refusing to fall back to a shim. "
        "Set AIDEV_ALLOW_IOUTILS_SHIM=1 only for standalone tests."
    ) from e

    io_utils = _LocalIOUtils()

try:
    from aidev.orchestration import edit_prompts
except Exception:  # pragma: no cover - fallback simple prompts
    class _LocalPrompts:
        @staticmethod
        def get_patch_capable_prompt(
            file_path: str, file_current: str, create_mode: bool = False
        ) -> Dict[str, Any]:
            return {
                "system": PATCH_PROMPT_PRESET,
                "file_path": file_path,
                "file_current": file_current,
                "create_mode": bool(create_mode),
            }

        @staticmethod
        def get_full_file_prompt(
            file_path: str, file_current: str, create_mode: bool = False
        ) -> Dict[str, Any]:
            return {
                "system": FULL_PROMPT_PRESET,
                "file_path": file_path,
                "file_current": file_current,
                "create_mode": bool(create_mode),
            }

    edit_prompts = _LocalPrompts()

try:
    from aidev import events
except Exception:
    class _LocalEvents:
        @staticmethod
        def emit_event(name: str, payload: Dict[str, Any]) -> None:
            print(f"[event]{name}: {json.dumps(payload)}")

        @staticmethod
        def sse_edit_attempt_start(**kwargs):
            pass

        @staticmethod
        def sse_edit_attempt_result(**kwargs):
            pass

        @staticmethod
        def sse_patch_apply_failed(**kwargs):
            pass

        @staticmethod
        def sse_fallback_full_content(**kwargs):
            pass

        @staticmethod
        def sse_edit_finalized(**kwargs):
            pass

    events = _LocalEvents()

try:
    from aidev import state as global_state
except Exception:
    class _LocalState:
        def __init__(self):
            self.trace: Dict[str, Any] = {}

        def record_edit_attempt(self, file_path: str, attempt_meta: Dict[str, Any]) -> None:
            ent = self.trace.setdefault(file_path, {})
            ent_attempts = ent.setdefault("attempts", [])
            ent_attempts.append(attempt_meta)

        def record_patch_failure(
            self,
            file_path: str,
            attempt_number: int,
            model: str,
            error_text: str,
            classifier_label: str,
        ) -> None:
            ent = self.trace.setdefault(file_path, {})
            ent_attempts = ent.setdefault("attempts", [])
            ent_attempts.append(
                {
                    "attempt": attempt_number,
                    "model": model,
                    "ok": False,
                    "error": error_text,
                    "classifier_label": classifier_label,
                }
            )

        def finalize_edit(self, file_path: str, success: bool, final_type: Optional[str]) -> None:
            ent = self.trace.setdefault(file_path, {})
            ent["finalized"] = {"success": success, "final_type": final_type}

    global_state = _LocalState()

# Optional: resolve system presets into concrete system_text when available
try:
    from aidev.llm_client import system_preset  # type: ignore
except Exception:
    system_preset = None  # type: ignore


@dataclass
class TieredEditError(Exception):
    """Structured error when tiered edit fails after allowed attempts."""

    file_path: str
    last_error: str
    attempt_history: List[Dict[str, Any]]

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"TieredEditError(file={self.file_path!r}, last_error={self.last_error!r}, "
            f"attempts={len(self.attempt_history)})"
        )


def classify_patch_failure(error_text: str) -> Literal["invalid_diff", "context_mismatch", "unknown"]:
    if not error_text:
        return "unknown"

    text = error_text.lower()

    invalid_patterns = [
        r"invalid patch",
        r"malformed hunk",
        r"cannot parse",
        r"could not parse",
        r"unexpected hunk",
        r"hunk header",
        r"parse error",
        r"failed to parse",
    ]
    for p in invalid_patterns:
        if re.search(p, text):
            return "invalid_diff"

    context_patterns = [
        r"context mismatch",
        r"offset",
        r"hunk context",
        r"fuzzy",
        r"does not match",
        r"context lines",
    ]
    for p in context_patterns:
        if re.search(p, text):
            return "context_mismatch"

    return "unknown"


def _emit(event_name: str, payload: Dict[str, Any]) -> None:
    helper_name = f"sse_{event_name}"

    def _map_payload(p: Dict[str, Any]) -> Dict[str, Any]:
        mapping: Dict[str, Any] = {}
        for k, v in p.items():
            if k == "file_path":
                mapping["file"] = v
            elif k == "classifier":
                mapping["classifier_label"] = v
            else:
                mapping[k] = v
        return mapping

    mapped = _map_payload(payload)

    if hasattr(events, helper_name):
        try:
            getattr(events, helper_name)(**mapped)
            return
        except Exception:
            pass

    if hasattr(events, "emit_event"):
        try:
            events.emit_event(event_name, payload)
            return
        except Exception:
            pass


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _sha256_12(text: str) -> str:
    try:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unknown"


def _excerpt_lines(text: str, max_lines: int = 12, max_chars: int = 2000) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = text.splitlines()
    chunk = "\n".join(lines[:max_lines])
    if len(chunk) > max_chars:
        chunk = chunk[:max_chars] + "\n...[excerpt truncated]..."
    if len(lines) > max_lines:
        chunk += "\n...[more lines truncated]..."
    return chunk


def _truncate_for_details(text: str, max_chars: int = 15000) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[patch_unified_attempt truncated]..."


def _choose_models(explicit_model: Optional[str] = None) -> Dict[str, str]:
    """
    Select per-attempt models.

    NOTE: `explicit_model` is treated as a fallback *base* only. It must NOT override
    AIDEV_MODEL_EDIT_PATCH / AIDEV_MODEL_EDIT_FULL when those env vars exist.

    Priority:
      base = AIDEV_MODEL_GENERATE_EDITS or explicit_model or "gpt-5-mini"
      patch = AIDEV_MODEL_EDIT_PATCH or base
      full  = AIDEV_MODEL_EDIT_FULL  or base
    """
    base = os.getenv("AIDEV_MODEL_GENERATE_EDITS") or (explicit_model or "") or "gpt-5-mini"
    patch_model = os.getenv("AIDEV_MODEL_EDIT_PATCH") or base
    full_model = os.getenv("AIDEV_MODEL_EDIT_FULL") or base
    return {"patch": patch_model, "full": full_model}


def _guess_language_from_path(path: str) -> str:
    p = (path or "").lower()
    if p.endswith(".py"):
        return "python"
    if p.endswith((".js", ".jsx", ".ts", ".tsx")):
        return "javascript"
    if p.endswith(".css"):
        return "css"
    if p.endswith((".html", ".htm")):
        return "html"
    if p.endswith((".md", ".markdown")):
        return "markdown"
    return "text"


def validate_edit_response(
    resp: Dict[str, Any],
    *,
    file_path: str,
    require_no_patch_unified: bool = False,
) -> Literal["content", "patch_unified"]:
    if not isinstance(resp, dict):
        raise ValueError(f"edit response must be a dict (got {type(resp).__name__})")

    if "path" in resp and resp.get("path") not in (None, "") and resp.get("path") != file_path:
        raise ValueError(
            f"response.path mismatch (got {resp.get('path')!r}, expected {file_path!r})"
        )

    raw_content = resp.get("content", None)
    raw_patch = resp.get("patch_unified", None)

    has_content = isinstance(raw_content, str) and raw_content.strip() != ""
    has_patch = isinstance(raw_patch, str) and raw_patch.strip() != ""

    if require_no_patch_unified and has_patch:
        raise ValueError(
            "response contained non-empty 'patch_unified' but full-file prompt forbids patches"
        )

    if has_content == has_patch:
        raise ValueError(
            "response must contain exactly one of non-empty 'content' or non-empty 'patch_unified'"
        )

    return "content" if has_content else "patch_unified"


def _resolve_system_text(system_field: Any, *, fallback_preset: str) -> str:
    """
    If prompt['system'] is a preset key like 'edit_file', try to resolve it via system_preset().
    Otherwise treat it as a raw system prompt string. Fall back to a minimal built-in prompt.
    """
    if isinstance(system_field, str) and system_field.strip():
        s = system_field.strip()
        if system_preset is not None and s in (
            PATCH_PROMPT_PRESET,
            FULL_PROMPT_PRESET,
            "repair_file",
            "create_file",
        ):
            try:
                resolved = system_preset(s)
                if isinstance(resolved, str) and resolved.strip():
                    return resolved
            except Exception:
                pass
        # If it looks like a real system prompt, use as-is.
        if len(s) > 80 and ("you are" in s.lower() or "return" in s.lower()):
            return s
        # Otherwise it's probably a preset name; try resolving; if not, fall through.
        if system_preset is not None:
            try:
                resolved = system_preset(s)
                if isinstance(resolved, str) and resolved.strip():
                    return resolved
            except Exception:
                pass

    # Built-in fallback
    if fallback_preset == FULL_PROMPT_PRESET:
        return (
            "You are an expert software engineer. Produce a FULL updated file.\n"
            "Return exactly one JSON object with:\n"
            "- path (must match input file path)\n"
            "- content (full updated contents)\n"
            "Do NOT return patches. No markdown."
        )
    return (
        "You are an expert software engineer. You may return either a full file or a unified diff.\n"
        "Return exactly one JSON object with:\n"
        "- path (must match input file path)\n"
        "- EITHER content (full updated file) OR patch_unified (unified diff)\n"
        "No markdown."
    )


_EDIT_FILE_CONTRACT_KEYS = {
    "rec",
    "file",
    "goal",
    "acceptance_criteria",
    "criteria_summary",
    "file_local_plan",
    "file_constraints",
    "file_notes_for_editor",
    "file_role",
    "file_importance",
    "file_kind_hint",
    "file_related_paths",
    "file_context_summary",
    "context_files",
    "cross_file_notes",
    "analysis_cross_file_notes",
    "details",
    "create_mode",
}

def _filter_edit_file_contract_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k in _EDIT_FILE_CONTRACT_KEYS}

def _build_contract_user_payload(
    *,
    file_path: str,
    base_snapshot: str,
    create_mode: bool,
    context_payload: Optional[Dict[str, Any]] = None,
    goal: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a user payload that is contract-aligned and enforces:
      - file.current == base_snapshot (LF-normalized)
      - file.path == file_path
      - create_mode propagated
      - goal propagated when provided (non-empty)
      - details merged under payload.details when provided
    """
    payload: Dict[str, Any] = {}
    if isinstance(context_payload, dict):
        # shallow copy so we don't mutate caller state
        payload = dict(context_payload)
        # Remove any accidental non-contract keys (aliases/debug mirrors, etc.)
        payload = _filter_edit_file_contract_payload(payload)

    # Ensure file object exists and is authoritative
    file_obj: Dict[str, Any] = {}
    raw_file_obj = payload.get("file")
    if isinstance(raw_file_obj, dict):
        file_obj = dict(raw_file_obj)

    file_obj["path"] = file_path
    file_obj.setdefault("language", _guess_language_from_path(file_path))
    file_obj["current"] = base_snapshot  # enforce baseline == model snapshot
    payload["file"] = file_obj

    # Enforce create_mode in payload
    payload["create_mode"] = bool(create_mode)

    # Goal propagation: if explicit non-empty goal provided, override; else keep context if present
    if isinstance(goal, str) and goal.strip():
        payload["goal"] = goal.strip()
    else:
        if "goal" not in payload:
            payload["goal"] = ""

    # Merge details under payload.details
    if isinstance(details, dict) and details:
        d = payload.get("details")
        if not isinstance(d, dict):
            d = {}

        # Normalize patch attempt newlines if present
        p_attempt = details.get("patch_unified_attempt")
        if isinstance(p_attempt, str):
            details = dict(details)
            details["patch_unified_attempt"] = _normalize_newlines(p_attempt)

        d.update(details)
        payload["details"] = d

    # Final filter pass in case details/goal introduced anything unexpected
    payload = _filter_edit_file_contract_payload(payload)
    return payload


def _prompt_to_chat_args(
    prompt: Dict[str, Any],
    *,
    file_path: str,
    file_current: str,
    create_mode: bool,
    phase: str,
    temperature: float,
    max_tokens: Optional[int],
    stage: Optional[str],
    model_override: Optional[str],
    schema: Optional[JsonSchema],
) -> Tuple[str, Dict[str, Any], JsonSchema, float, str, Optional[int], Optional[str], Optional[str]]:
    """
    Convert a prompt dict into (system_text, user_payload, schema, temperature, phase, max_tokens, stage, model_override)
    suitable for adapter-style llm_client.generate_edits(...)

    NOTE: If prompt doesn't provide a user payload, we synthesize a minimal one. In normal
    repo usage, callers SHOULD provide a contract-complete user_payload via context_payload.
    """
    system_field = prompt.get("system_text") or prompt.get("system") or ""
    preset_name = str(prompt.get("system") or "").strip()
    fallback_preset = FULL_PROMPT_PRESET if preset_name == FULL_PROMPT_PRESET else PATCH_PROMPT_PRESET
    system_text = _resolve_system_text(system_field, fallback_preset=fallback_preset)

    user_payload = prompt.get("user_payload") or prompt.get("payload") or prompt.get("user")
    if not isinstance(user_payload, dict):
        user_payload = {
            "file": {
                "path": file_path,
                "language": _guess_language_from_path(file_path),
                "current": file_current,
            },
            "goal": "",
            "create_mode": bool(create_mode),
        }

    schema_arg: JsonSchema = schema if isinstance(schema, dict) else {}
    return (
        system_text,
        user_payload,
        schema_arg,
        float(temperature),
        str(phase),
        max_tokens,
        stage,
        model_override,
    )


def _call_llm_generate_edits(
    llm_client: Any,
    *,
    prompt: Dict[str, Any],
    model: str,
    schema: Optional[JsonSchema] = None,
    temperature: float = 0.0,
    phase: str = "generate_edits",
    max_tokens: Optional[int] = None,
    stage: Optional[str] = None,
    model_override: Optional[str] = None,
    create_mode: bool = False,
    **extra_llm_kwargs: Any,
) -> Dict[str, Any]:
    """
    Calls llm_client in a way that supports BOTH:
      A) prompt-style: generate_edits(prompt=..., model=..., schema=..., temperature=..., max_tokens=..., stage=..., model_override=...)
      B) adapter-style: generate_edits(system_text, user_payload, schema, temperature, phase, max_tokens, ...)

    Important nuance:
      - Some wrappers accept **kwargs (VAR_KEYWORD). In that case stage/model_override/etc
        will NOT appear in inspect.signature(...).parameters even though they are supported.
      - We therefore detect VAR_KEYWORD and treat it as "accept all kwargs".
    """
    fn = getattr(llm_client, "generate_edits", None) or getattr(llm_client, "generate", None)
    if fn is None:
        raise TypeError("llm_client missing generate_edits(...)")

    # Inspect signature to determine which kwargs are safe to pass.
    allowed: set[str] = set()
    has_varkw = False
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        allowed = set(params.keys())
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except Exception:
        allowed = set()
        has_varkw = False

    def _kw_ok(name: str) -> bool:
        return has_varkw or (name in allowed)

    # Mode A: prompt-style (generate_edits(prompt=..., model=..., ...))
    if "prompt" in allowed or has_varkw:
        kwargs: Dict[str, Any] = {"prompt": prompt, "model": model}

        if schema is not None and _kw_ok("schema"):
            kwargs["schema"] = schema
        if _kw_ok("temperature"):
            kwargs["temperature"] = temperature
        if max_tokens is not None and _kw_ok("max_tokens"):
            kwargs["max_tokens"] = max_tokens
        if stage is not None and _kw_ok("stage"):
            kwargs["stage"] = stage
        if model_override is not None and _kw_ok("model_override"):
            kwargs["model_override"] = model_override
        if _kw_ok("create_mode"):
            kwargs["create_mode"] = bool(create_mode)

        if extra_llm_kwargs:
            if has_varkw:
                kwargs.update(extra_llm_kwargs)
            else:
                for k, v in extra_llm_kwargs.items():
                    if k in allowed:
                        kwargs[k] = v

        out = fn(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            data, _res = out
            if isinstance(data, dict):
                return data
            raise ValueError(f"llm_client returned non-dict data: {type(data).__name__}")
        if isinstance(out, dict):
            return out
        raise ValueError(f"llm_client returned non-dict: {type(out).__name__}")

    # Mode B: adapter-style (generate_edits(system_text, user_payload, schema, temperature, phase, max_tokens, ...))
    system_text, user_payload, schema_arg, temp, ph, mt, st, mo = _prompt_to_chat_args(
        prompt,
        file_path=str(prompt.get("file_path") or ""),
        file_current=str(prompt.get("file_current") or ""),
        create_mode=create_mode,
        phase=phase,
        temperature=temperature,
        max_tokens=max_tokens,
        stage=stage,
        model_override=model_override,
        schema=schema,
    )

    call_args = [system_text, user_payload, schema_arg, temp, ph, mt]
    call_kwargs: Dict[str, Any] = {}

    if st is not None and _kw_ok("stage"):
        call_kwargs["stage"] = st
    if mo is not None and _kw_ok("model_override"):
        call_kwargs["model_override"] = mo
    if _kw_ok("create_mode"):
        call_kwargs["create_mode"] = bool(create_mode)

    if extra_llm_kwargs:
        if has_varkw:
            call_kwargs.update(extra_llm_kwargs)
        else:
            for k, v in extra_llm_kwargs.items():
                if k in allowed:
                    call_kwargs[k] = v

    out = fn(*call_args, **call_kwargs)

    if isinstance(out, tuple) and len(out) == 2:
        data, _res = out
        if isinstance(data, dict):
            return data
        raise ValueError(f"llm_client returned non-dict data: {type(data).__name__}")
    if isinstance(out, dict):
        return out
    raise ValueError(f"llm_client returned non-dict: {type(out).__name__}")


def _build_patch_prompt(file_path: str, base_snapshot: str, create_mode: bool) -> Dict[str, Any]:
    if hasattr(edit_prompts, "get_patch_capable_prompt"):
        try:
            return edit_prompts.get_patch_capable_prompt(
                file_path, base_snapshot, create_mode=create_mode
            )
        except TypeError:
            return edit_prompts.get_patch_capable_prompt(file_path, base_snapshot)
    return {
        "system": PATCH_PROMPT_PRESET,
        "file_path": file_path,
        "file_current": base_snapshot,
        "create_mode": bool(create_mode),
    }


def _build_full_prompt(file_path: str, base_snapshot: str, create_mode: bool) -> Dict[str, Any]:
    if hasattr(edit_prompts, "get_full_file_prompt"):
        try:
            return edit_prompts.get_full_file_prompt(file_path, base_snapshot, create_mode=create_mode)
        except TypeError:
            return edit_prompts.get_full_file_prompt(file_path, base_snapshot)
    return {
        "system": FULL_PROMPT_PRESET,
        "file_path": file_path,
        "file_current": base_snapshot,
        "create_mode": bool(create_mode),
    }


def _attempt_full_file(
    *,
    file_path: str,
    base_snapshot: str,
    llm_client: Any,
    state: Any,
    model: str,
    attempt_no: int,
    attempts: List[Dict[str, Any]],
    schema: Optional[JsonSchema],
    temperature: float,
    max_tokens: Optional[int],
    stage: Optional[str],
    create_mode: bool,
    model_override: Optional[str],
    phase: str,
    context_payload: Optional[Dict[str, Any]] = None,
    goal: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    **extra_llm_kwargs: Any,
) -> Dict[str, Any]:
    prompt2 = _build_full_prompt(file_path, base_snapshot, create_mode=create_mode)

    # Attach a contract-complete user payload (context + enforced file.current + create_mode + goal + details)
    user_payload = _build_contract_user_payload(
        file_path=file_path,
        base_snapshot=base_snapshot,
        create_mode=create_mode,
        context_payload=context_payload,
        goal=goal,
        details=details,
    )
    prompt2["user_payload"] = user_payload

    _emit(
        "edit_attempt_start",
        {"file_path": file_path, "attempt": attempt_no, "model": model, "type": "full_file"},
    )

    try:
        resp2 = _call_llm_generate_edits(
            llm_client,
            prompt=prompt2,
            model=model,
            schema=schema,
            temperature=temperature,
            phase=phase,
            max_tokens=max_tokens,
            stage=stage,
            model_override=model_override,
            create_mode=create_mode,
            **extra_llm_kwargs,
        )
    except Exception as e2:
        attempt_meta2 = {
            "attempt": attempt_no,
            "model": model,
            "output_type": None,
            "ok": False,
            "last_error": str(e2),
            "base_snapshot_sha256_12": _sha256_12(base_snapshot),
            "base_snapshot_len": len(base_snapshot),
        }
        attempts.append(attempt_meta2)
        try:
            state.record_edit_attempt(file_path, attempt_meta2)
        except Exception:
            pass
        raise TieredEditError(file_path=file_path, last_error=str(e2), attempt_history=attempts)

    try:
        output_type2 = validate_edit_response(
            resp2, file_path=file_path, require_no_patch_unified=True
        )
    except Exception as e2:
        attempt_meta2 = {
            "attempt": attempt_no,
            "model": model,
            "output_type": None,
            "ok": False,
            "last_error": str(e2),
            "raw_response": resp2,
            "base_snapshot_sha256_12": _sha256_12(base_snapshot),
            "base_snapshot_len": len(base_snapshot),
        }
        attempts.append(attempt_meta2)
        try:
            state.record_edit_attempt(file_path, attempt_meta2)
        except Exception:
            pass
        _emit(
            "edit_attempt_result",
            {
                "file_path": file_path,
                "attempt": attempt_no,
                "model": model,
                "valid": False,
                "error": str(e2),
            },
        )
        raise TieredEditError(file_path=file_path, last_error=str(e2), attempt_history=attempts)

    content2 = _normalize_newlines(resp2["content"])
    attempt_meta2 = {
        "attempt": attempt_no,
        "model": model,
        "output_type": output_type2,
        "ok": True,
        "base_snapshot_sha256_12": _sha256_12(base_snapshot),
        "base_snapshot_len": len(base_snapshot),
    }
    attempts.append(attempt_meta2)
    try:
        state.record_edit_attempt(file_path, attempt_meta2)
    except Exception:
        pass

    return {"file_path": file_path, "content": content2, "source": "full_after_fallback", "attempts": attempts}


def apply_tiered_edit_to_file(
    file_path: str,
    file_current: str,
    llm_client: Any,
    state: Any = global_state,
    model: Optional[str] = None,
    max_attempts: int = 2,
    *,
    schema: Optional[JsonSchema] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    stage: Optional[str] = None,
    create_mode: bool = False,
    model_override: Optional[str] = None,
    phase: str = "generate_edits",
    # supply the full contract payload (rec/recommendation, acceptance_criteria, plans, constraints, notes, etc.)
    context_payload: Optional[Dict[str, Any]] = None,
    # optional explicit goal override; if provided (non-empty) it is enforced into payload.goal
    goal: Optional[str] = None,
    **extra_llm_kwargs: Any,
) -> Dict[str, Any]:
    """
    Main entrypoint. Accepts schema and standard LLM knobs, so callers like generate_edits.py
    can pass them without TypeError.

    Use context_payload for contract compliance:
      - Ensures edit_file / edit_file_full receive recommendation + acceptance_criteria + per-file guidance.
      - Ensures goal propagation.
      - Ensures file.current baseline integrity for patch application.
    """
    if max_attempts < 2:
        raise ValueError("max_attempts must be >= 2 for tiered strategy")

    models = _choose_models(explicit_model=model)

    # Effective models must respect env vars first, then model_override, then base selection.
    patch_env = os.getenv("AIDEV_MODEL_EDIT_PATCH")
    full_env = os.getenv("AIDEV_MODEL_EDIT_FULL")

    patch_model_effective = patch_env or model_override or models["patch"]
    full_model_effective = full_env or model_override or models["full"]

    patch_stage = "edit_patch"
    full_stage = "edit_full"

    attempts: List[Dict[str, Any]] = []
    base_snapshot = _normalize_newlines(file_current)

    # Baseline consistency observability
    try:
        logging.debug(
            "[apply_tiered_edit_to_file] baseline file=%s len=%d sha256_12=%s",
            file_path,
            len(base_snapshot),
            _sha256_12(base_snapshot),
        )
    except Exception:
        pass

    # Optional sanity warning for goal propagation regressions
    try:
        if isinstance(context_payload, dict):
            rec = context_payload.get("rec") or context_payload.get("recommendation") or {}
            focus = rec.get("focus") if isinstance(rec, dict) else None
            if isinstance(focus, str) and focus.strip():
                effective_goal = ""
                if isinstance(goal, str) and goal.strip():
                    effective_goal = goal.strip()
                elif isinstance(context_payload.get("goal"), str) and str(context_payload.get("goal")).strip():
                    effective_goal = str(context_payload.get("goal")).strip()
                if not effective_goal:
                    logging.warning(
                        "[apply_tiered_edit_to_file] focus present but goal empty file=%s",
                        file_path,
                    )
    except Exception:
        pass

    # Attempt #1 (patch-capable)
    attempt_no = 1
    _emit(
        "edit_attempt_start",
        {"file_path": file_path, "attempt": attempt_no, "model": patch_model_effective, "type": "patch_capable"},
    )

    prompt = _build_patch_prompt(file_path, base_snapshot, create_mode=create_mode)

    # Attach contract-complete payload for patch attempt (same payload shape as full fallback)
    prompt["user_payload"] = _build_contract_user_payload(
        file_path=file_path,
        base_snapshot=base_snapshot,
        create_mode=create_mode,
        context_payload=context_payload,
        goal=goal,
        details=None,
    )

    try:
        resp = _call_llm_generate_edits(
            llm_client,
            prompt=prompt,
            model=patch_model_effective,
            schema=schema,
            temperature=temperature,
            phase=phase,
            max_tokens=max_tokens,
            stage=patch_stage,
            model_override=patch_model_effective,
            create_mode=create_mode,
            **extra_llm_kwargs,
        )
    except Exception as e:
        attempt_meta = {
            "attempt": attempt_no,
            "model": patch_model_effective,
            "output_type": None,
            "ok": False,
            "last_error": str(e),
            "base_snapshot_sha256_12": _sha256_12(base_snapshot),
            "base_snapshot_len": len(base_snapshot),
        }
        attempts.append(attempt_meta)
        try:
            state.record_edit_attempt(file_path, attempt_meta)
        except Exception:
            pass

        _emit(
            "fallback_full_content",
            {
                "file_path": file_path,
                "attempt_from": 2,
                "trace": attempts,
                "fallback_reason": "patch_model_error",
                "original_patch": None,
                "fallback_attempts": attempts[:],
            },
        )

        # fallback to Attempt #2
        try:
            fallback = _attempt_full_file(
                file_path=file_path,
                base_snapshot=base_snapshot,
                llm_client=llm_client,
                state=state,
                model=full_model_effective,
                attempt_no=2,
                attempts=attempts,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
                stage=full_stage,
                create_mode=create_mode,
                model_override=full_model_effective,
                phase=phase,
                context_payload=context_payload,
                goal=goal,
                details=None,
                **extra_llm_kwargs,
            )
            try:
                state.finalize_edit(file_path, True, "full_after_llm_error")
            except Exception:
                pass
            _emit(
                "edit_finalized",
                {
                    "file_path": file_path,
                    "status": "applied",
                    "source": "full_after_llm_error",
                    "attempts": attempts,
                    "trace": attempts,
                    "full_content": fallback.get("content"),
                    "fallback_reason": "patch_model_error",
                    "original_patch": None,
                    "fallback_attempts": attempts[:],
                },
            )
            fallback["source"] = "full_after_llm_error"
            return fallback
        except TieredEditError as te:
            try:
                state.finalize_edit(file_path, False, None)
            except Exception:
                pass
            _emit("edit_finalized", {"file_path": file_path, "status": "failed", "attempts": attempts})
            raise TieredEditError(
                file_path=file_path,
                last_error=f"attempt1 failed; fallback failed: {te.last_error}",
                attempt_history=attempts,
            )

    # Validate attempt #1 response; if invalid, fallback
    try:
        output_type = validate_edit_response(resp, file_path=file_path, require_no_patch_unified=False)
    except Exception as e:
        attempt_meta = {
            "attempt": attempt_no,
            "model": patch_model_effective,
            "output_type": None,
            "ok": False,
            "last_error": str(e),
            "raw_response": resp,
            "base_snapshot_sha256_12": _sha256_12(base_snapshot),
            "base_snapshot_len": len(base_snapshot),
        }
        attempts.append(attempt_meta)
        try:
            state.record_edit_attempt(file_path, attempt_meta)
        except Exception:
            pass
        _emit(
            "edit_attempt_result",
            {"file_path": file_path, "attempt": attempt_no, "model": patch_model_effective, "valid": False, "error": str(e)},
        )

        original_patch = None
        if isinstance(resp, dict):
            rp = resp.get("patch_unified")
            if isinstance(rp, str) and rp.strip():
                original_patch = _normalize_newlines(rp)

        _emit(
            "fallback_full_content",
            {
                "file_path": file_path,
                "attempt_from": 2,
                "trace": attempts,
                "fallback_reason": "invalid_response",
                "original_patch": original_patch,
                "fallback_attempts": attempts[:],
            },
        )
        fallback = _attempt_full_file(
            file_path=file_path,
            base_snapshot=base_snapshot,
            llm_client=llm_client,
            state=state,
            model=full_model_effective,
            attempt_no=2,
            attempts=attempts,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            stage=full_stage,
            create_mode=create_mode,
            model_override=full_model_effective,
            phase=phase,
            context_payload=context_payload,
            goal=goal,
            details=None,
            **extra_llm_kwargs,
        )
        try:
            state.finalize_edit(file_path, True, "full_after_invalid_attempt1")
        except Exception:
            pass
        _emit(
            "edit_finalized",
            {
                "file_path": file_path,
                "status": "applied",
                "source": "full_after_invalid_attempt1",
                "attempts": attempts,
                "trace": attempts,
                "full_content": fallback.get("content"),
                "fallback_reason": "invalid_response",
                "original_patch": original_patch,
                "fallback_attempts": attempts[:],
            },
        )
        fallback["source"] = "full_after_invalid_attempt1"
        return fallback

    attempt_meta = {
        "attempt": attempt_no,
        "model": patch_model_effective,
        "output_type": output_type,
        "ok": None,
        "base_snapshot_sha256_12": _sha256_12(base_snapshot),
        "base_snapshot_len": len(base_snapshot),
    }
    attempts.append(attempt_meta)
    try:
        state.record_edit_attempt(file_path, attempt_meta)
    except Exception:
        pass
    _emit(
        "edit_attempt_result",
        {"file_path": file_path, "attempt": attempt_no, "model": patch_model_effective, "output_type": output_type},
    )

    # If model returned full content in attempt #1
    if output_type == "content":
        content = _normalize_newlines(resp["content"])
        attempts[-1].update({"ok": True})
        try:
            state.finalize_edit(file_path, True, "full")
        except Exception:
            pass
        _emit("edit_finalized", {"file_path": file_path, "status": "applied", "source": "full", "attempts": attempts})
        return {"file_path": file_path, "content": content, "source": "full", "attempts": attempts}

    # Otherwise apply patch_unified (strictly against base_snapshot)
    patch_text = _normalize_newlines(resp.get("patch_unified", ""))
    if not patch_text.strip():
        raise TieredEditError(file_path=file_path, last_error="missing patch_unified in patch attempt", attempt_history=attempts)

    # Enforce invariant: model snapshot == patch baseline == base_snapshot
    # (We already forced payload.file.current to base_snapshot before calling the model.)
    try:
        patched = io_utils.apply_unified_patch(base_snapshot, patch_text)
        patched = _normalize_newlines(patched)
        attempts[-1].update({"ok": True})
        try:
            state.finalize_edit(file_path, True, "patch")
        except Exception:
            pass
        _emit("edit_finalized", {"file_path": file_path, "status": "applied", "source": "patch", "attempts": attempts})
        return {"file_path": file_path, "content": patched, "source": "patch", "attempts": attempts}
    except Exception as e:
        err_text = str(e)

        classifier: Literal["invalid_diff", "context_mismatch", "unknown"]
        if hasattr(e, "code") and getattr(e, "code"):
            code_val = getattr(e, "code")
            if code_val in ("invalid_diff", "context_mismatch", "unknown"):
                classifier = code_val
            else:
                classifier = classify_patch_failure(err_text)
        else:
            classifier = classify_patch_failure(err_text)

        attempts[-1].update({"ok": False, "last_error": err_text, "failure_class": classifier})
        try:
            state.record_patch_failure(file_path, attempt_no, patch_model_effective, err_text, classifier)
        except Exception:
            try:
                state.record_edit_attempt(file_path, attempts[-1])
            except Exception:
                pass

        normalized_patch_attempt = _normalize_newlines(patch_text)
        if not isinstance(normalized_patch_attempt, str) or normalized_patch_attempt.strip() == "":
            normalized_patch_attempt = "<missing_patch_unified_attempt>"

        patch_apply_error = err_text or repr(e)

        # Truncate only for details payload to prevent giant SSE payloads
        details: Dict[str, Any] = {
            "patch_unified_attempt": _truncate_for_details(normalized_patch_attempt),
            "patch_apply_error": patch_apply_error,
            "failure_class": classifier,
            "base_snapshot_sha256_12": _sha256_12(base_snapshot),
            "base_snapshot_len": len(base_snapshot),
        }

        # INFO log excerpt for easier debugging
        try:
            logging.info(
                "[patch_apply_failed] file=%s classifier=%s base_sha=%s err=%s\npatch_excerpt:\n%s",
                file_path,
                classifier,
                _sha256_12(base_snapshot),
                (err_text or "")[:500],
                _excerpt_lines(normalized_patch_attempt),
            )
        except Exception:
            pass

        _emit(
            "patch_apply_failed",
            {
                "file_path": file_path,
                "attempt": attempt_no,
                "error": err_text,
                "classifier": classifier,
                "trace": attempts,
                "original_patch": patch_text,
                # Keep the full normalized attempt here if you want; details already truncated.
                "patch_unified_attempt": normalized_patch_attempt,
                "patch_apply_error": patch_apply_error,
                "details": details,
                "fallback_attempts": attempts[:],
            },
        )

        _emit(
            "fallback_full_content",
            {
                "file_path": file_path,
                "attempt_from": 2,
                "trace": attempts,
                "fallback_reason": classifier,
                "original_patch": patch_text,
                "details": details,
                "fallback_attempts": attempts[:],
            },
        )

        fallback = _attempt_full_file(
            file_path=file_path,
            base_snapshot=base_snapshot,
            llm_client=llm_client,
            state=state,
            model=full_model_effective,
            attempt_no=2,
            attempts=attempts,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            stage=full_stage,
            create_mode=create_mode,
            model_override=full_model_effective,
            phase=phase,
            context_payload=context_payload,
            goal=goal,
            details=details,
            **extra_llm_kwargs,
        )
        try:
            state.finalize_edit(file_path, True, "full_after_patch_failure")
        except Exception:
            pass
        _emit(
            "edit_finalized",
            {
                "file_path": file_path,
                "status": "applied",
                "source": "full_after_patch_failure",
                "attempts": attempts,
                "trace": attempts,
                "full_content": fallback.get("content"),
                "fallback_reason": classifier,
                "original_patch": patch_text,
                "details": details,
                "fallback_attempts": attempts[:],
            },
        )
        fallback["source"] = "full_after_patch_failure"
        return fallback


def tiered_edit_file(
    file: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
    current: Optional[str] = None,
    llm_client: Any = None,
    state: Any = global_state,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    if file is not None:
        file_path = file.get("path") or file.get("file_path")
        file_current = file.get("current") or file.get("file_current") or file.get("content")
    else:
        file_path = path
        file_current = current

    if file_path is None or file_current is None:
        raise ValueError(
            "tiered_edit_file requires either file dict with 'path' and 'current' or path & current kwargs"
        )

    result = apply_tiered_edit_to_file(
        str(file_path),
        str(file_current),
        llm_client=llm_client,
        state=state,
        model=model,
        **kwargs,
    )
    return {
        "file_path": result.get("file_path"),
        "final_content": result.get("content"),
        "attempts": result.get("attempts"),
    }
