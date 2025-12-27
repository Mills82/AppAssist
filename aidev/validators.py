# aidev/validators.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


class ValidationError(Exception):
    """Controlled validation error for callers that want exceptions.

    NOTE: Public helpers in this module currently return (ok, message) rather
    than raising by default.
    """


# Common secret-ish keys that should be masked
REDACT_KEYS = {
    "authorization",
    "api_key",
    "openai_api_key",
    "token",
    "password",
    "secret",
    "set-cookie",
    "cookie",
    "bearer",
    "client_secret",
    "private_key",
}

# Regex masks for secret-like data appearing inside strings
# NOTE: These are conservative and intentionally broad; keep false positives
# acceptable for logs.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # OpenAI keys or similar tokens (sk-...)
    (re.compile(r"sk-[A-Za-z0-9]{16,}", re.I), "sk-********"),
    # Bearer JWTs or opaque tokens
    (
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-\._~\+/=]{10,}"),
        "Bearer ********",
    ),
    # Generic api_key=... in URLs or text
    (
        re.compile(r"(?i)(api[_-]?key)\s*=\s*[^&\s]+"),
        r"\1=********",
    ),
    # Azure keys (base64-ish 24+ chars)
    (
        re.compile(
            r"(?i)\b(azure[_-]?openai[_-]?api[_-]?key)\s*[:=]\s*"
            r"[A-Za-z0-9\+/_-]{20,}"
        ),
        r"\1=********",
    ),
    # Cookies
    (
        re.compile(r"(?i)\b(Set-Cookie|Cookie):\s*[^;\r\n]+"),
        r"\1: ********",
    ),
    # Private key blocks (single-line)
    (
        re.compile(
            r"-----BEGIN [^-]+ PRIVATE KEY-----.*?-----END [^-]+ PRIVATE KEY-----",
            re.S,
        ),
        "********",
    ),
)


def _redact_in_string(s: str) -> str:
    out = s
    for pat, repl in _SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate `data` against a JSON Schema dict.

    Note: This is used for HTTP payloads and internal request shapes.
    The LLM-facing schemas live under aidev/schemas/*.schema.json and are
    typically passed directly into chat_json() rather than via this helper.
    """
    if jsonschema is None:
        # Best-effort if jsonschema isn't installed
        return True, None
    try:
        jsonschema.validate(data, schema)  # type: ignore
        return True, None
    except Exception as e:
        return False, str(e)


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({"type": "error", "message": "serialization failed"})


def redact_secrets(obj: Any) -> Any:
    """
    Recursively mask values for common secret keys, and scrub tokens inside
    strings.
    """
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if k.lower() in REDACT_KEYS:
                out[k] = "********"
            else:
                out[k] = redact_secrets(v)
        return out
    if isinstance(obj, list):
        return [redact_secrets(x) for x in obj]
    if isinstance(obj, (str, bytes)):
        s = obj.decode("utf-8", errors="ignore") if isinstance(obj, bytes) else obj
        return _redact_in_string(s)
    return obj


def is_unified_diff(text: str) -> bool:
    return text.lstrip().startswith("--- ") or bool(
        re.search(r"^\+\+\+ ", text, flags=re.M)
    )


# -------- JSON Schemas for new request bodies (optional runtime validation) --------

CHAT_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "message": {"type": ["string", "null"]},
        "text": {"type": ["string", "null"]},
        "session_id": {"type": ["string", "null"]},
        "workspace_root": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

CREATE_PROJECT_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "project_name": {"type": ["string", "null"]},
        "base_dir": {"type": ["string", "null"]},
        "brief": {"type": "string"},
    },
    "required": ["brief"],
    "additionalProperties": True,
}

SELECT_PROJECT_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"project_path": {"type": "string"}},
    "required": ["project_path"],
    "additionalProperties": False,
}

UPDATE_DESCRIPTIONS_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"instructions": {"type": "string"}},
    "required": ["instructions"],
    "additionalProperties": False,
}

RUN_CHECKS_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}

APPLY_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": ["string", "null"]},
        "approve": {"type": "boolean"},
    },
    "required": ["approve"],
    "additionalProperties": True,
}


def validate_payload(name: str, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Convenience wrapper for ad hoc endpoints that don't use Pydantic models.

    `name` is a short identifier for the endpoint shape. This function does NOT
    look at aidev/schemas/*.schema.json files; those are primarily used for
    validating LLM responses via chat_json().
    """
    schemas = {
        "chat": CHAT_REQUEST_SCHEMA,
        "create_project": CREATE_PROJECT_REQUEST_SCHEMA,
        "select_project": SELECT_PROJECT_REQUEST_SCHEMA,
        "update_descriptions": UPDATE_DESCRIPTIONS_REQUEST_SCHEMA,
        "run_checks": RUN_CHECKS_REQUEST_SCHEMA,
        "apply": APPLY_REQUEST_SCHEMA,
    }
    schema = schemas.get(name)
    if not schema:
        return True, None
    return validate_json_schema(data, schema)


# -------- Repository schema helpers --------


def validate_repo_schema(
    schema_filename: str, data: Any, base_dir: Optional[Path] = None
) -> Tuple[bool, Optional[str]]:
    """
    Load a JSON Schema from the repository (aidev/schemas/<schema_filename>) and
    validate `data` against it. Returns (True, None) on success, otherwise
    (False, error_message).

    base_dir may be provided to indicate the repository root; if omitted the
    function will try the path relative to this file's directory.
    """
    # Build candidate locations to search for the schema file.
    candidates = []
    if base_dir is not None:
        try:
            base = Path(base_dir)
        except Exception:
            base = Path(str(base_dir))
        candidates.append(base / "aidev" / "schemas" / schema_filename)
        candidates.append(base / "schemas" / schema_filename)
    # Path relative to this module (useful for installs or running from different CWDs)
    candidates.append(Path(__file__).resolve().parent / "schemas" / schema_filename)

    for p in candidates:
        if p and p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    schema = json.load(f)
            except Exception as e:
                return False, f"failed to load schema {p}: {e}"
            return validate_json_schema(data, schema)

    looked = ", ".join(str(x) for x in candidates)
    return False, (
        f"schema file not found: {schema_filename} (looked at: {looked})"
    )


def _normalize_error(error_type: str, payload: Dict[str, Any]) -> str:
    """
    Return a deterministic, structured error string (JSON) with sorted keys.
    This avoids embedding stack traces or arbitrary object representations.
    """
    obj = {"error_type": error_type}
    obj.update(payload)
    # sort_keys=True ensures deterministic key ordering for logs/tests
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _redacted_normalized_error(error_type: str, payload: Dict[str, Any]) -> str:
    """Normalize to deterministic JSON and redact any secrets in the payload."""
    safe_payload = redact_secrets(payload)
    # Ensure payload is a dict after redaction; if not, stringify deterministically.
    if not isinstance(safe_payload, dict):
        safe_payload = {"payload": safe_json(safe_payload)}
    return _normalize_error(error_type, safe_payload)


def _redact_normalized_json_string(s: str) -> str:
    """Attempt to redact secrets from an already-normalized JSON string."""
    try:
        obj = json.loads(s)
    except Exception:
        # Not JSON; redact as plain string and return a normalized wrapper.
        return _redacted_normalized_error(
            "validation_error",
            {"message": _redact_in_string(s)},
        )

    redacted_obj = redact_secrets(obj)
    # Keep deterministic JSON; ensure an error_type exists for structured logs.
    if isinstance(redacted_obj, dict):
        if "error_type" in redacted_obj:
            return json.dumps(redacted_obj, sort_keys=True, ensure_ascii=False)
        # If it somehow lacked error_type, wrap but preserve detail.
        return _redacted_normalized_error("validation_error", {"detail": redacted_obj})

    return _redacted_normalized_error(
        "validation_error",
        {"detail": redacted_obj},
    )


def validate_repo_schema_normalized(
    schema_filename: str, data: Any, base_dir: Optional[Path] = None
) -> Tuple[bool, Optional[str]]:
    """
    Like validate_repo_schema but returns deterministic, structured error
    messages suitable for logs/traces. On success returns (True, None). On
    failure returns (False, <json-string>) where the JSON has keys such as
    'error_type' and other deterministic fields.
    """
    # Build candidate locations to search for the schema file (stable order)
    candidates = []
    if base_dir is not None:
        try:
            base = Path(base_dir)
        except Exception:
            base = Path(str(base_dir))
        candidates.append(base / "aidev" / "schemas" / schema_filename)
        candidates.append(base / "schemas" / schema_filename)
    candidates.append(Path(__file__).resolve().parent / "schemas" / schema_filename)

    looked_paths = [str(p) for p in candidates]

    for p in candidates:
        if p and p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    schema = json.load(f)
            except Exception as e:
                # Return a normalized load error (no stack trace)
                return False, _redacted_normalized_error(
                    "schema_load_error",
                    {
                        "schema_filename": schema_filename,
                        "path": str(p),
                        "exception": type(e).__name__,
                        "message": str(e),
                    },
                )
            ok, msg = validate_json_schema(data, schema)
            if ok:
                return True, None
            # Validation failed; normalize the message
            detail = msg if isinstance(msg, str) else str(msg)
            return False, _redacted_normalized_error(
                "validation_error",
                {
                    "schema_filename": schema_filename,
                    "path": str(p),
                    "message": detail,
                },
            )

    # Not found: return deterministic list of paths looked
    return False, _redacted_normalized_error(
        "schema_not_found",
        {"schema_filename": schema_filename, "looked": looked_paths},
    )


def _validate_via_repo_or_helper(
    *,
    schema_filename: str,
    data: Any,
    base_dir: Optional[Path],
    helper: Any,
) -> Tuple[bool, Optional[str]]:
    """Single call site for schema validation with preferred delegation.

    Preferred path: delegate to `aidev.schemas` helpers when present (so tests
    can mock them and callers share consistent behavior).

    Fallback path: validate using the repository schema loader in this module.

    If jsonschema is missing, keep best-effort behavior and return (True, None).
    """
    if jsonschema is None:
        # CI/environment without jsonschema: keep best-effort behavior.
        return True, None

    try:
        if helper is not None:
            # Call the helper and robustly interpret its return type. Helpers may
            # return (ok, msg), a bool, or a dict like {'valid': bool, 'errors': [...]}.
            try:
                res = helper(data, base_dir)
            except TypeError:
                # Some helpers may accept only (data,) not (data, base_dir)
                res = helper(data)
            except Exception as e:
                return False, _redacted_normalized_error(
                    "validation_error",
                    {
                        "schema_filename": schema_filename,
                        "exception": type(e).__name__,
                        "message": str(e),
                    },
                )

            # Normalize the helper result into (ok, msg)
            ok: bool = False
            msg: Optional[str] = None

            if isinstance(res, tuple):
                # Expect (ok, msg) but be permissive about shapes
                if len(res) >= 1:
                    ok = bool(res[0])
                    if len(res) >= 2:
                        raw_msg = res[1]
                        msg = _redact_normalized_json_string(str(raw_msg)) if raw_msg is not None else None
                else:
                    ok = False
            elif isinstance(res, bool):
                ok = res
                msg = None
            elif isinstance(res, dict):
                # Interpret common dict shapes
                if "valid" in res:
                    ok = bool(res.get("valid"))
                elif "is_valid" in res:
                    ok = bool(res.get("is_valid"))
                elif "ok" in res:
                    ok = bool(res.get("ok"))
                else:
                    # If errors/messages present, treat non-empty as failure
                    errs = None
                    for k in ("errors", "error", "messages", "message"):
                        if k in res:
                            errs = res.get(k)
                            break
                    ok = not bool(errs)

                if not ok:
                    # Convert the structured dict into the module's normalized/redacted JSON string
                    try:
                        msg = _redact_normalized_json_string(json.dumps(res))
                    except Exception:
                        msg = _redact_normalized_json_string(str(res))
                else:
                    msg = None
            elif res is None:
                ok = True
                msg = None
            else:
                # Unknown return type: stringify and treat non-empty as failure
                sval = str(res)
                ok = False
                msg = _redact_normalized_json_string(sval)

            if ok:
                return True, None

            # Helper reported failure. Ensure we return a normalized/redacted string.
            if msg is None:
                return False, _redacted_normalized_error(
                    "validation_error",
                    {"schema_filename": schema_filename, "message": "validation failed"},
                )
            return False, msg

        ok, msg = validate_repo_schema_normalized(schema_filename, data, base_dir)
        if ok:
            return True, None
        if msg is None:
            return False, _redacted_normalized_error(
                "validation_error",
                {"schema_filename": schema_filename, "message": "validation failed"},
            )
        return False, _redact_normalized_json_string(str(msg))
    except Exception as e:
        # Never leak a stack trace; return deterministic error.
        return False, _redacted_normalized_error(
            "validation_error",
            {
                "schema_filename": schema_filename,
                "exception": type(e).__name__,
                "message": str(e),
            },
        )


# Prefer delegation to aidev.schemas helpers when available.
try:  # pragma: no cover
    from aidev.schemas import validate_research_plan as _schemas_validate_research_plan
except Exception:  # pragma: no cover
    _schemas_validate_research_plan = None

try:  # pragma: no cover
    from aidev.schemas import validate_research_brief as _schemas_validate_research_brief
except Exception:  # pragma: no cover
    _schemas_validate_research_brief = None


def validate_research_plan(
    data: Any, base_dir: Optional[Path] = None
) -> Tuple[bool, Optional[str]]:
    """
    Convenience wrapper that validates `data` against the repository's
    research_plan.schema.json. The schema file is expected at
    aidev/schemas/research_plan.schema.json (or under <base_dir>/aidev/schemas).

    Returns (True, None) on success or (False, <normalized-json-string>) on
    failure. The failure string is a small JSON object with an 'error_type'
    and deterministic details.

    Note: Prefer delegating to `aidev.schemas.validate_research_plan` when
    present; tests may mock that helper.
    """
    return _validate_via_repo_or_helper(
        schema_filename="research_plan.schema.json",
        data=data,
        base_dir=base_dir,
        helper=_schemas_validate_research_plan,
    )


def validate_research_brief(
    data: Any, base_dir: Optional[Path] = None
) -> Tuple[bool, Optional[str]]:
    """
    Convenience wrapper that validates `data` against research_brief.schema.json
    using the normalized repository schema loader.

    Returns (True, None) on success or (False, <normalized-json-string>) on
    failure.

    Note: Prefer delegating to `aidev.schemas.validate_research_brief` when
    present; tests may mock that helper.
    """
    return _validate_via_repo_or_helper(
        schema_filename="research_brief.schema.json",
        data=data,
        base_dir=base_dir,
        helper=_schemas_validate_research_brief,
    )
