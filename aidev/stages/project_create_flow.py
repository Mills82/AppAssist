# aidev/stages/project_create_flow.py
# -*- coding: utf-8 -*-
"""
Helper for conversational project creation flow.

Exposes create_project_flow(brief, llm_client, answers=None) -> dict which returns either:
- {'follow_up_questions': [str|dict, ...]} when more information is needed, or
- {'created_files': [{'path': str, 'content': str}, ...]} when the LLM produced scaffold files.

The function is pure (no filesystem writes) and tolerant of LLM errors: on unexpected responses
it returns a generic follow-up question to prompt the caller for clarification.

New helpers in this module allow callers (typically the HTTP route) to persist and
resume transient session envelopes under .aidev/project_create_sessions/<token>.json.
These helpers are intentionally separate so create_project_flow remains pure:
- session_dir() -> pathlib.Path
- save_session(token: str, envelope: dict) -> None
- load_session(token: str) -> dict
- delete_session(token: str) -> None

Integration contract / usage guidance (summarized):
- Call create_project_flow(brief, llm_client) to get either follow_up_questions or created_files.
- If follow_up_questions are returned, the route should generate a session token (sanitized
  string matching r'^[A-Za-z0-9._-]+$'), call save_session(token, envelope) where envelope is the
  LLM response plus any metadata the caller wants to persist, and return the questions + token
  to the client.
- When the client submits answers, the route should load the saved envelope with load_session(token),
  pass answers into create_project_flow(..., answers=answers) to resume, and when created_files are
  returned persist them atomically using existing IO utilities.

Important: create_project_flow remains filesystem-free. Session helpers are provided for callers
that need to persist/restore intermediate state.
"""

from typing import Any, Dict, List, Optional
import logging
import pathlib
import json
import re
import tempfile
import os
import uuid
import datetime
import hashlib

# Minimal heuristic thresholds and keywords used to hint at ambiguity.
_MIN_BRIEF_LEN = 80
_KEYWORDS = {"goal", "platform", "features", "auth", "data", "users", "example", "ui", "api"}

# Path to the system prompt that instructs the LLM to emit either follow_up_questions or created_files.
_SYSTEM_PROMPT_PATH = "aidev/prompts/system.project_create.md"

# Generic clarifying question used as a safe fallback.
_FALLBACK_QUESTION = (
    "Could you clarify goals, target platform, main features, auth/data needs, "
    "and an example of expected input/output?"
)

# Exported integration hint for callers/maintainers indicating the expected endpoint.
EXPECTED_ENDPOINT = "aidev/routes/workspaces.py:POST /projects/create"

logger = logging.getLogger(__name__)

# Try to import atomic write helper from aidev.io_utils if present.
try:
    from aidev import io_utils  # type: ignore
    _HAS_IO_UTILS = True
except Exception:
    io_utils = None  # type: ignore
    _HAS_IO_UTILS = False

_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Canonical sessions dir used by session helpers. Other modules should reuse this constant.
SESSIONS_DIR = pathlib.Path(".aidev") / "project_create_sessions"


# Try importing events and apply helpers; keep them optional and guarded so tests
# and environments without these modules won't fail on import.
try:
    from aidev import events as _events  # type: ignore
except Exception:
    _events = None  # type: ignore

try:
    from aidev.stages import apply_and_refresh as _apply_and_refresh  # type: ignore
except Exception:
    _apply_and_refresh = None  # type: ignore


def _is_ambiguous_brief(brief: str) -> bool:
    """Return True when the brief looks under-specified.

    Heuristic: short briefs or briefs missing obvious keywords are considered ambiguous.
    This is only a hint; the LLM's response ultimately decides whether follow-ups
    questions are required.
    """
    if not brief:
        return True
    if len(brief.strip()) < _MIN_BRIEF_LEN:
        return True
    lowered = brief.lower()
    # If none of the important keywords appear, treat as ambiguous
    if not any(k in lowered for k in _KEYWORDS):
        return True
    return False


def _normalize_created_files(raw: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize created_files entries to {'path': str, 'content': str}.

    Coerces to strings and strips whitespace. Ignores entries missing a path.
    """
    normalized = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content = item.get("content", "")
        if not path:
            # skip invalid entries
            continue
        try:
            path_str = str(path).strip()
            content_str = str(content)
        except Exception:
            continue
        normalized.append({"path": path_str, "content": content_str})
    return normalized


def session_dir() -> pathlib.Path:
    """Return the directory Path used for transient project-create sessions.

    Creates the directory if it does not exist. Canonical path is SESSIONS_DIR.
    """
    p = SESSIONS_DIR
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Log but do not raise; callers that rely on filesystem persistence will
        # see errors when saving/loading sessions.
        logger.debug("Could not create session dir %s", p, exc_info=True)
    return p


def _validate_token(token: str) -> None:
    if not isinstance(token, str) or not token:
        raise ValueError("session token must be a non-empty string")
    if not _TOKEN_RE.match(token):
        raise ValueError("invalid session token; allowed: A-Za-z0-9._-")


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write text to path atomically.

    Prefer aidev.io_utils.atomic_write if available; otherwise write to a
    tempfile in the same directory and os.replace.
    """
    if _HAS_IO_UTILS and getattr(io_utils, "atomic_write", None):
        try:
            # Some io_utils.atomic_write implementations expect (path, data)
            io_utils.atomic_write(path, text)
            return
        except Exception:
            logger.debug("io_utils.atomic_write failed; falling back", exc_info=True)
    # Fallback implementation
    dirpath = str(path.parent)
    fd = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=dirpath, prefix=path.name + ".") as tf:
            tf.write(text)
            tmp_path = pathlib.Path(tf.name)
        # os.replace is atomic on supported platforms
        os.replace(str(tmp_path), str(path))
        try:
            os.chmod(str(path), 0o600)
        except Exception:
            # best-effort; ignore if not permitted
            pass
    finally:
        # ensure temp file doesn't remain on error
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def save_session(token: str, envelope: Dict[str, Any]) -> None:
    """Persist an envelope dict under .aidev/project_create_sessions/<token>.json.

    Raises ValueError for invalid tokens. Writes JSON with ensure_ascii=False and indent=2.
    """
    _validate_token(token)
    if not isinstance(envelope, dict):
        raise ValueError("envelope must be a dict")
    p = session_dir() / f"{token}.json"
    text = json.dumps(envelope, ensure_ascii=False, indent=2)
    try:
        _atomic_write_text(p, text)
        logger.debug("Saved session %s -> %s", token, p)
    except Exception as exc:
        logger.warning("Failed to save session %s: %s", token, exc)
        raise


def load_session(token: str) -> Dict[str, Any]:
    """Load and return the session envelope for token.

    Raises ValueError for invalid token and FileNotFoundError if missing.
    """
    _validate_token(token)
    p = session_dir() / f"{token}.json"
    if not p.exists():
        raise FileNotFoundError(f"session not found: {token}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("session file does not contain a JSON object")
        return data
    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.warning("Failed to load session %s: %s", token, exc)
        raise


def delete_session(token: str) -> None:
    """Remove a persisted session file if present. Ignore if absent.

    Raises ValueError for invalid tokens.
    """
    _validate_token(token)
    p = session_dir() / f"{token}.json"
    try:
        if p.exists():
            p.unlink()
            logger.debug("Deleted session %s", token)
    except Exception:
        logger.debug("Failed to delete session %s", token, exc_info=True)


def normalize_follow_up_questions(raw: Any) -> List[Dict[str, Any]]:
    """Normalize follow_up_questions returned by LLM into a canonical structured form.

    Accepts either a list of strings, a list of dicts, or mixed content and returns
    a list of dicts with at least the keys: 'question' (str). When possible preserves
    'id', 'type', 'required', and 'choices'. Legacy plain strings are converted to
    {'question': <string>}.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if item is None:
            continue
        if isinstance(item, dict):
            # keep canonical keys, coerce question to str if present
            q: Dict[str, Any] = {}
            if "id" in item:
                q["id"] = item["id"]
            # Prefer explicit 'question' field; fall back to 'text' or 'label'
            question = item.get("question") or item.get("text") or item.get("label")
            if question is None:
                # If no question-like key, stringify the whole item
                try:
                    q["question"] = json.dumps(item, ensure_ascii=False)
                except Exception:
                    q["question"] = str(item)
            else:
                q["question"] = str(question).strip()
            # optional fields
            if "type" in item:
                q["type"] = item["type"]
            if "required" in item:
                q["required"] = bool(item["required"])
            if "choices" in item:
                q["choices"] = item["choices"]
            out.append(q)
        else:
            # coerce plain values to a minimal structured form
            out.append({"question": str(item).strip()})
    return out


def _create_project_flow_impl(brief: str, llm_client: Any, answers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Internal LLM-driven implementation of the conversational project-create flow.

    This function preserves the historical behavior (brief, llm_client, answers).
    A thin compatibility wrapper named create_project_flow is provided below and
    handles both the older calling convention and a new one that accepts an
    optional project_root and keyword llm_client for clarity.
    """
    brief_text = (brief or "").strip()

    # Heuristic check that can be used by callers or for logging; the LLM still
    # ultimately decides whether follow-ups are needed.
    _is_ambiguous_brief(brief_text)

    # Prepare a conservative default response to fall back to on errors/invalid shapes.
    fallback = {"follow_up_questions": [_FALLBACK_QUESTION]}

    # Always call the LLM with the system prompt and brief so the model can decide the
    # correct action (ask follow-ups or produce files). Include answers if provided so
    # the model can resume the flow.
    payload: Dict[str, Any] = {"brief": brief_text}
    if answers is not None:
        payload["answers"] = answers

    try:
        resp = llm_client.chat_json(_SYSTEM_PROMPT_PATH, payload)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LLM call failed in create_project_flow: %s", exc)
        # If the heuristic says the brief was clearly rich, still return the fallback
        # (we avoid raising to keep callers simple). The UI can retry or show follow-ups.
        return fallback

    # Validate response shape
    if not isinstance(resp, dict):
        logger.debug("LLM returned non-dict response: %r", resp)
        return fallback

    # If the model asked follow-up questions, normalize and return them.
    if "follow_up_questions" in resp:
        q = resp.get("follow_up_questions")
        if isinstance(q, list) and q:
            # If the LLM returned structured objects (dicts), preserve them so callers
            # can render slot-filling prompts. For backward compatibility, if the
            # LLM returned plain strings, return a list of strings as before.
            has_struct = any(isinstance(x, dict) for x in q)
            if has_struct:
                questions: List[Any] = []
                for x in q:
                    if isinstance(x, dict):
                        # preserve object as-is (caller/UI expected to handle shape)
                        questions.append(x)
                    else:
                        # coerce non-dict items into a minimal structured form so the
                        # UI can unify rendering (keeps original content).
                        if x is None:
                            continue
                        questions.append({"question": str(x).strip()})
                if questions:
                    logger.debug("Returning structured follow_up_questions from LLM")
                    return {"follow_up_questions": questions}
            else:
                # Historical behavior: normalize to list of strings
                questions = [str(x).strip() for x in q if x is not None]
                if questions:
                    return {"follow_up_questions": questions}
        # fall through to fallback if empty/invalid

    # If the model returned created files, normalize and return them.
    if "created_files" in resp:
        raw_files = resp.get("created_files")
        if isinstance(raw_files, list) and raw_files:
            created = _normalize_created_files(raw_files)
            if created:
                # Warn if expected scaffold files are missing, but still return what we got.
                expected = {"app_descrip.txt", "project_description.md"}
                present = {f.get("path") for f in created}
                missing = expected - present
                if missing:
                    logger.info("LLM produced created_files but missing expected files: %s", missing)
                return {"created_files": created}
        # fall through to fallback if empty/invalid

    # If the response didn't contain an explicit branch or was invalid, fall back.
    # If the brief seemed very rich but we didn't get files, still ask clarifying question.
    logger.debug("LLM response lacked required keys; returning fallback. resp=%r", resp)
    return fallback


def create_project_flow(brief: str, project_root: Optional[str] = None, answers: Optional[Dict[str, Any]] = None, llm_client: Optional[Any] = None) -> Dict[str, Any]:
    """Compatibility wrapper for the project-create flow.

    New signature accepts (brief, project_root=None, answers=None, llm_client=None) and is
    suitable for callers that want to pass a project_root path. For backward compatibility
    callers that used the historical signature create_project_flow(brief, llm_client, answers)
    will still work: the wrapper detects when the second positional argument is an LLM
    client (has chat_json) and adjusts accordingly.

    If llm_client is provided (or inferred from the second arg) this forwards to the
    internal LLM-driven implementation. If no llm_client is present the function raises
    a ValueError explaining that an llm_client is required — callers (e.g. route handlers)
    must supply an LLM adapter implementing chat_json(prompt_path, payload) -> dict.
    """
    # Back-compat: if caller passed llm_client as the second positional argument
    inferred_llm = None
    if llm_client is None and project_root is not None:
        # If project_root looks like an llm_client (has chat_json), treat it as such.
        if not isinstance(project_root, (str, bytes, pathlib.Path)) and hasattr(project_root, "chat_json"):
            inferred_llm = project_root
            project_root = None
    if llm_client is None and inferred_llm is None:
        raise ValueError(
            "llm_client is required: provide an object with chat_json(prompt_path, payload) -> dict"
        )
    final_llm = llm_client if llm_client is not None else inferred_llm
    # forward to the internal implementation
    return _create_project_flow_impl(brief, final_llm, answers=answers)


# ---------------------------------------------------------------------------
# New exported sessioned entrypoints required by the recommendation
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_emit(event_name: str, session_id: str, payload: Any) -> None:
    """Safely emit an event using aidev.events if available.

    Event helper names expected:
      - emit_project_create_questions(session_id, payload)
      - emit_project_create_scaffold(session_id, payload)
      - emit_project_create_applied(session_id, payload)

    Emit payloads are validated against aidev/schemas/events.schema.json; keep shapes stable.
    If events module or function is unavailable, log debug and continue.
    """
    if not _events:
        logger.debug("Events module unavailable; skipping emit %s", event_name)
        return
    fn = getattr(_events, event_name, None)
    if not fn:
        logger.debug("Events module missing %s; skipping", event_name)
        return
    try:
        fn(session_id, payload)
    except Exception:
        logger.exception("Failed to emit event %s for session %s", event_name, session_id)


def _ensure_answers_shape(raw: Any) -> Dict[str, Any]:
    """Normalize provided answers into a dict of id->value.

    Accepts dict or list forms (e.g. [{'id': 'goal', 'answer': 'x'}, ...]) and returns a
    merged dict. Values from a dict input are used as-is.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    out: Dict[str, Any] = {}
    if isinstance(raw, list):
        for item in raw:
            if not item:
                continue
            if isinstance(item, dict):
                if "id" in item and ("answer" in item or "value" in item):
                    out[item["id"]] = item.get("answer", item.get("value"))
                elif "id" in item and "question" in item and "value" in item:
                    out[item["id"]] = item.get("value")
                else:
                    # fallback: if dict has single k/v, use it
                    if len(item) == 1:
                        k = next(iter(item.keys()))
                        out[k] = item[k]
            else:
                # cannot map plain list items without context; store under auto-generated key
                out_key = f"answer_{len(out)}"
                out[out_key] = item
    return out


def _v2_response_for_followups(session_id: str, questions: Any) -> Dict[str, Any]:
    # questions may be strings or dicts; normalize to list of dicts using existing helper
    normalized = normalize_follow_up_questions(questions)
    # If normalization produced empty but original was list of strings, try fallback
    if not normalized and isinstance(questions, list):
        normalized = [{"question": str(q)} for q in questions if q is not None]
    return {"session_id": session_id, "follow_up_questions": normalized}


def _v2_response_for_created_files(session_id: str, created_files: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"session_id": session_id, "created_files": created_files}


def _created_files_emit_payload(created: List[Dict[str, str]]) -> Dict[str, Any]:
    """Build the payload for emit_project_create_scaffold.

    Schema expected: {'created_files': [..], 'total_files': N, 'preview': False}
    Each created_file entry should include 'path', 'summary', 'size', 'sha'.
    """
    payload_files: List[Dict[str, Any]] = []
    if not isinstance(created, list):
        created = []
    for item in created:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content = item.get("content", "") if item.get("content") is not None else ""
        if not path:
            continue
        # summary: first 240 chars of content (preserve newlines trimmed)
        summary = (content.strip().replace("\r\n", "\n"))[:240]
        try:
            size = len(content.encode("utf-8"))
        except Exception:
            size = len(str(content))
        # compute sha256 of content bytes
        try:
            sha = hashlib.sha256(content.encode("utf-8") if isinstance(content, str) else bytes(content)).hexdigest()
        except Exception:
            sha = ""
        payload_files.append({"path": path, "summary": summary, "size": size, "sha": sha})
    return {"created_files": payload_files, "total_files": len(payload_files), "preview": False}


def _ensure_questions_for_emit(questions: Any) -> List[Dict[str, Any]]:
    """Ensure questions is a list of dicts suitable for the events payload.

    Uses normalize_follow_up_questions and guarantees a list of dicts.
    """
    normalized = normalize_follow_up_questions(questions)
    if not normalized and isinstance(questions, list):
        normalized = [{"question": str(q)} for q in questions if q is not None]
    # As a last resort, if questions was a single string, wrap it
    if not normalized and isinstance(questions, str):
        normalized = [{"question": questions}]
    return normalized


def start_session(brief: str, llm_client: Any, apply: bool = False, token: Optional[str] = None) -> Dict[str, Any]:
    """Start a new conversational project-create session.

    Calls create_project_flow to get either follow_up_questions or created_files,
    persists a session envelope under a generated token (or provided token), and
    emits events for follow-up questions when returned.

    Returns a v2-shaped dict: {session_id, follow_up_questions[]} or {session_id, created_files[]}.
    """
    if token is None:
        token = uuid.uuid4().hex
    else:
        # validate provided token
        _validate_token(token)
    session_id = token

    now = _now_iso()
    envelope: Dict[str, Any] = {
        "brief": brief,
        "created_at": now,
        "last_updated": now,
        "answers": {},
        "apply_requested": bool(apply),
        "applied": False,
    }

    try:
        result = create_project_flow(brief, llm_client=llm_client)
    except Exception as exc:
        # convert error into follow-up fallback
        logger.exception("create_project_flow raised in start_session: %s", exc)
        result = {"follow_up_questions": [_FALLBACK_QUESTION]}

    envelope["last_llm_response"] = result
    # Persist follow-ups or created files into the envelope appropriately
    if "follow_up_questions" in result:
        questions = result.get("follow_up_questions")
        normalized = _ensure_questions_for_emit(questions)
        envelope["follow_up_questions"] = normalized
        try:
            save_session(session_id, envelope)
        except Exception:
            # save_session logs and raises; to keep start_session robust, capture and continue
            logger.exception("Failed to persist session %s", session_id)
        # emit questions event - payload must include 'questions' key per events.schema.json
        try:
            _safe_emit("emit_project_create_questions", session_id, {"questions": normalized})
        except Exception:
            logger.exception("Failed to emit project_create_questions for %s", session_id)
        return _v2_response_for_followups(session_id, normalized)

    if "created_files" in result:
        raw_files = result.get("created_files")
        created = _normalize_created_files(raw_files if isinstance(raw_files, list) else [])
        envelope["created_files"] = created
        envelope["last_llm_response"] = result
        try:
            save_session(session_id, envelope)
        except Exception:
            logger.exception("Failed to persist session %s", session_id)
        # Build scaffold event payload using normalized created files. The payload shape is
        # validated against aidev/schemas/events.schema.json and must include created_files, total_files, preview.
        scaffold_payload = _created_files_emit_payload(created)
        try:
            _safe_emit("emit_project_create_scaffold", session_id, scaffold_payload)
        except Exception:
            logger.exception("Failed to emit project_create_scaffold for %s", session_id)

        # If apply requested, attempt to apply now
        if apply and created:
            apply_result = {"applied": False}
            try:
                if not _apply_and_refresh:
                    raise RuntimeError("apply_and_refresh module unavailable")
                apply_fn = getattr(_apply_and_refresh, "apply", None) or getattr(_apply_and_refresh, "apply_and_refresh", None)
                if not apply_fn:
                    raise RuntimeError("apply function not found in apply_and_refresh module")
                # Best-effort call: many apply implementations accept created_files list.
                apply_res = apply_fn(created)
                apply_result = {"applied": True, "result": apply_res}
                envelope["applied"] = True
                envelope["applied_at"] = _now_iso()
                envelope["apply_result"] = apply_result
                save_session(session_id, envelope)
                # Wrap apply outcome under 'result' key per events schema
                _safe_emit("emit_project_create_applied", session_id, {"result": apply_result})
            except Exception as exc:
                logger.exception("Failed to apply created files for session %s: %s", session_id, exc)
                apply_result = {"applied": False, "error": str(exc)}
                envelope["apply_result"] = apply_result
                try:
                    save_session(session_id, envelope)
                except Exception:
                    logger.exception("Failed to persist session after apply error %s", session_id)
            # Return created files and include apply_result for caller
            resp = _v2_response_for_created_files(session_id, created)
            resp["apply_result"] = apply_result
            return resp

        return _v2_response_for_created_files(session_id, created)

    # fallback: persist session with fallback question
    envelope["follow_up_questions"] = [ {"question": _FALLBACK_QUESTION} ]
    try:
        save_session(session_id, envelope)
    except Exception:
        logger.exception("Failed to persist fallback session %s", session_id)
    try:
        _safe_emit("emit_project_create_questions", session_id, {"questions": envelope["follow_up_questions"]})
    except Exception:
        logger.exception("Failed to emit fallback project_create_questions for %s", session_id)
    return _v2_response_for_followups(session_id, envelope["follow_up_questions"])


def submit_answers(session_id: str, answers: Any, llm_client: Any, apply: bool = False) -> Dict[str, Any]:
    """Submit answers for an existing session and advance the conversation.

    Loads the persisted envelope, merges answers, calls create_project_flow to resume,
    persists updated envelope, emits events when follow-ups or created files are returned,
    and if apply=True attempts to write files via the apply_and_refresh helper.

    Returns a v2-shaped dict: {session_id, follow_up_questions[]} or {session_id, created_files[], (apply_result?)}.
    """
    # Validate and load
    _validate_token(session_id)
    try:
        envelope = load_session(session_id)
    except Exception as exc:
        logger.exception("Failed to load session %s: %s", session_id, exc)
        raise

    # Merge answers
    normalized_answers = _ensure_answers_shape(answers)
    existing_answers = envelope.get("answers") or {}
    if not isinstance(existing_answers, dict):
        existing_answers = {}
    merged = dict(existing_answers)
    merged.update(normalized_answers)
    envelope["answers"] = merged
    envelope["last_updated"] = _now_iso()

    brief = envelope.get("brief", "")

    try:
        result = create_project_flow(brief, llm_client=llm_client, answers=merged)
    except Exception as exc:
        logger.exception("create_project_flow raised in submit_answers for %s: %s", session_id, exc)
        result = {"follow_up_questions": [_FALLBACK_QUESTION]}

    envelope["last_llm_response"] = result

    if "follow_up_questions" in result:
        questions = result.get("follow_up_questions")
        normalized = _ensure_questions_for_emit(questions)
        envelope["follow_up_questions"] = normalized
        try:
            save_session(session_id, envelope)
        except Exception:
            logger.exception("Failed to persist session %s after follow-ups", session_id)
        try:
            _safe_emit("emit_project_create_questions", session_id, {"questions": normalized})
        except Exception:
            logger.exception("Failed to emit project_create_questions for %s", session_id)
        return _v2_response_for_followups(session_id, normalized)

    if "created_files" in result:
        raw_files = result.get("created_files")
        created = _normalize_created_files(raw_files if isinstance(raw_files, list) else [])
        envelope["created_files"] = created
        envelope["last_updated"] = _now_iso()
        try:
            save_session(session_id, envelope)
        except Exception:
            logger.exception("Failed to persist session %s after created_files", session_id)
        scaffold_payload = _created_files_emit_payload(created)
        try:
            _safe_emit("emit_project_create_scaffold", session_id, scaffold_payload)
        except Exception:
            logger.exception("Failed to emit project_create_scaffold for %s", session_id)

        # Optionally apply
        if apply and created:
            apply_result = {"applied": False}
            try:
                if not _apply_and_refresh:
                    raise RuntimeError("apply_and_refresh module unavailable")
                apply_fn = getattr(_apply_and_refresh, "apply", None) or getattr(_apply_and_refresh, "apply_and_refresh", None)
                if not apply_fn:
                    raise RuntimeError("apply function not found in apply_and_refresh module")
                apply_res = apply_fn(created)
                apply_result = {"applied": True, "result": apply_res}
                envelope["applied"] = True
                envelope["applied_at"] = _now_iso()
                envelope["apply_result"] = apply_result
                save_session(session_id, envelope)
                _safe_emit("emit_project_create_applied", session_id, {"result": apply_result})
            except Exception as exc:
                logger.exception("Failed to apply created files for session %s: %s", session_id, exc)
                apply_result = {"applied": False, "error": str(exc)}
                envelope["apply_result"] = apply_result
                try:
                    save_session(session_id, envelope)
                except Exception:
                    logger.exception("Failed to persist session after apply error %s", session_id)
            resp = _v2_response_for_created_files(session_id, created)
            resp["apply_result"] = apply_result
            return resp

        return _v2_response_for_created_files(session_id, created)

    # fallback: return follow-up fallback question
    envelope["follow_up_questions"] = [ {"question": _FALLBACK_QUESTION} ]
    try:
        save_session(session_id, envelope)
    except Exception:
        logger.exception("Failed to persist session %s after fallback in submit_answers", session_id)
    try:
        _safe_emit("emit_project_create_questions", session_id, {"questions": envelope["follow_up_questions"]})
    except Exception:
        logger.exception("Failed to emit fallback project_create_questions for %s", session_id)
    return _v2_response_for_followups(session_id, envelope["follow_up_questions"])


def get_status(session_id: str) -> Dict[str, Any]:
    """Return a stable snapshot of the session state without mutating it.

    Returns v2-shaped dict: {session_id, follow_up_questions[]} or {session_id, created_files[]}.
    """
    _validate_token(session_id)
    try:
        envelope = load_session(session_id)
    except Exception as exc:
        logger.exception("Failed to load session %s in get_status: %s", session_id, exc)
        raise

    # Prefer created_files when present, else follow_up_questions
    if envelope.get("created_files"):
        created = envelope.get("created_files") or []
        return _v2_response_for_created_files(session_id, created)
    if envelope.get("follow_up_questions"):
        questions = envelope.get("follow_up_questions") or []
        return _v2_response_for_followups(session_id, questions)
    # If neither present, attempt to derive from last_llm_response
    last = envelope.get("last_llm_response") or {}
    if "created_files" in last:
        created = _normalize_created_files(last.get("created_files") or [])
        return _v2_response_for_created_files(session_id, created)
    if "follow_up_questions" in last:
        questions = normalize_follow_up_questions(last.get("follow_up_questions") or [])
        return _v2_response_for_followups(session_id, questions)

    # final fallback
    return _v2_response_for_followups(session_id, [{"question": _FALLBACK_QUESTION}])


if __name__ == "__main__":
    # Minimal examples demonstrating expected behavior. This is not a unit test framework
    # but can be run to see how the helper normalizes mocked LLM responses.

    class MockLLM:
        def chat_json(self, prompt_path, payload):
            brief = (payload.get("brief") or "").strip()
            answers = payload.get("answers")
            # If brief is short and no answers, simulate follow-up questions
            if len(brief) < 50 and not answers:
                return {
                    "follow_up_questions": [
                        {"id": "goal", "question": "What is the main goal of the app?", "required": True, "type": "text"},
                        {"id": "platform", "question": "Which platforms should it run on?", "required": True, "type": "choice", "choices": ["web", "mobile", "desktop"]},
                    ]
                }
            # If answers are provided or brief long, simulate created files
            return {
                "created_files": [
                    {"path": "app_descrip.txt", "content": f"Expanded brief:\n{brief}\nanswers:\n{json.dumps(answers or {}, ensure_ascii=False)}"},
                    {"path": "project_description.md", "content": "# Project\nDetailed description..."},
                ]
            }

    client = MockLLM()

    short_brief = "A simple todo app"
    long_brief = (
        "A web-based team todo app with authentication, real-time sync, "
        "and CSV export. Target: web (React + Flask API). Users: teams of 2-50."
    )

    print("--- Short brief (expect follow-ups) ---")
    print(start_session(short_brief, llm_client=client))

    print("\n--- Short brief with answers (expect created_files) ---")
    s = start_session(short_brief, llm_client=client)
    sid = s.get("session_id")
    print(submit_answers(sid, {"goal": "team todo", "platform": "web"}, llm_client=client))

    print("\n--- Long brief (expect created_files) ---")
    print(start_session(long_brief, llm_client=client))
