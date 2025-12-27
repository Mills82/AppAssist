# aidev/events.py
"""
aidev.events — SSE event envelope contract

Canonical envelope emitted by the server (JSON):
  {
    "type": "<string>",          # e.g. "status", "progress", "log", "diff", "qa_answer"
    "payload": { ... },           # structured payload for the event type
    "ts": 1671234567.123,         # unix timestamp (float)
    # optional: session_id, job_id, meta, etc.
  }

Backwards/legacy compatibility: this module previously mirrored common payload keys to
top-level aliases and a `data` alias (see _overlay_backcompat and
_ensure_data_alias). To simplify and standardize client behavior we now emit the
canonical envelope only (type/payload/ts). The compatibility helpers remain in
this module (marked deprecated) so callers can migrate gradually, but they are
no longer applied to emitted envelopes.

Front-end consumers should:
  - parse each SSE message's data as JSON (fall back to legacy string handling if parsing fails),
  - prefer the canonical shape (type/payload) and read fields from payload (e.g. event.payload.text),
    rather than relying on top-level mirrors such as event.text.

Additionally: LLM-related emitters in this module may attach payload.model (string)
when a resolved model id is provided. Callers should pass an explicit `model`
parameter to those helpers to ensure the model metadata is present in the
canonical event payload. If `model` is provided as an explicit parameter it will
override any `model` value present in extras or the initial payload dict.

See: aidev/schemas/events.schema.json for the authoritative schema and
_overlay_backcompat/_ensure_data_alias for the old mirroring rules (deprecated).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional
from pathlib import Path

def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

# If STRICT_EVENTS=1 we hard-fail on schema violations; otherwise we log at DEBUG and continue.
STRICT_EVENTS = _env_truthy("STRICT_EVENTS", "0")

try:
    import jsonschema
    _EVENT_SCHEMA = json.loads(
        (Path(__file__).resolve().parent / "schemas" / "events.schema.json").read_text(encoding="utf-8")
    )
    _EVENT_VALIDATOR = jsonschema.Draft7Validator(_EVENT_SCHEMA)

    # Known event types from schema enum (used to avoid noisy validation failures for new telemetry events)
    _KNOWN_EVENT_TYPES = set(
        ((_EVENT_SCHEMA.get("properties") or {}).get("type") or {}).get("enum") or []
    )
except Exception:  # jsonschema not installed in prod or schema missing
    _EVENT_VALIDATOR = None
    _KNOWN_EVENT_TYPES = set()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emit utilities only (no FastAPI routes here).
# Job-scoped streaming lives in /jobs/stream (see server.py), bridged via observers.
# ---------------------------------------------------------------------------

# Optional observers (metrics/tracing and stream bridging)
# token -> callback
_emit_observers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

# Back-compat mapping for function-based observers (add_observer/remove_observer)
# so we can support both token-based and function-based APIs.
_observer_tokens: Dict[Callable[[Dict[str, Any]], None], str] = {}


@dataclass
class _TopicState:
    last_where: Optional[str] = None
    last_stage: Optional[str] = None


# If your flows still pass session_id, we keep lightweight "last topic" tracking
_last_topic: Dict[str, _TopicState] = {}


def register_emit_observer(fn: Callable[[Dict[str, Any]], None]) -> str:
    """Register a hook called with each emitted event dict. Returns a token for unregister."""
    token = uuid.uuid4().hex
    if callable(fn):
        _emit_observers[token] = fn
    return token


def unregister_emit_observer(token: str) -> None:
    _emit_observers.pop(token, None)


# ----- Back-compat observer APIs -----


def add_observer(fn: Callable[[Dict[str, Any]], None]) -> None:
    """
    Legacy API: register an observer by function reference.
    Compatible with server._attach_events_bridge(add_observer/remove_observer).
    """
    if not callable(fn):
        return
    if fn in _observer_tokens:
        return
    token = register_emit_observer(fn)
    _observer_tokens[fn] = token


def remove_observer(fn: Callable[[Dict[str, Any]], None]) -> None:
    """Legacy API: remove an observer previously added via add_observer()."""
    token = _observer_tokens.pop(fn, None)
    if token:
        unregister_emit_observer(token)


def subscribe(fn: Callable[[Dict[str, Any]], None]) -> str:
    """
    Token-based subscription API.
    Compatible with server._attach_events_bridge(subscribe/unsubscribe).
    """
    return register_emit_observer(fn)


def unsubscribe(token: str) -> None:
    """Token-based unsubscribe API."""
    unregister_emit_observer(token)


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return '{"type":"error","payload":{"message":"serialization_failed"}}'


def _overlay_backcompat(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    (Deprecated) Mirror selected payload fields to top-level keys for existing UI bits.

    NOTE: The canonical envelope is:
        {"type": ..., "payload": {...}, "ts": ...}
    This helper only adds legacy mirrors (top-level keys + ev.data) for older frontends.

    Kept for gradual migration; NOT applied to emitted envelopes anymore.
    """
    if "payload" not in ev:
        return ev

    p = ev.get("payload")

    # --- Harden against bad payload types ---
    if not isinstance(p, dict):
        if p is None:
            p = {}
        else:
            try:
                # handles list of (k, v) pairs
                p = dict(p)  # type: ignore[arg-type]
            except Exception:
                # last resort: stash raw value
                p = {"value": p}
        ev["payload"] = p

    # Common status fields
    for k in ("message", "msg", "stage", "where", "detail", "file", "progress_pct"):
        if k in p:
            ev.setdefault(k, p[k])

    # Assistant text
    if ev.get("type") == "assistant":
        txt = p.get("text") or p.get("content") or ""
        ev.setdefault("text", txt)

    # Q&A answer: expose answer as top-level text/message for generic chat UIs
    if ev.get("type") == "qa_answer":
        ans = p.get("answer") or ""
        if ans:
            ev.setdefault("text", ans)
            ev.setdefault("message", ans)

    # Analyze result: expose analysis as top-level text/message ONLY.
    if ev.get("type") == "analysis_result":
        analysis = p.get("analysis") or ""
        if analysis:
            ev.setdefault("text", analysis)
            ev.setdefault("message", analysis)

        plan = p.get("plan")
        if isinstance(plan, dict):
            recs = plan.get("recommendations")
            if recs is not None:
                ev.setdefault("recommendations", recs)
                # If no more specific data alias is set, mirror recommendations + plan.
                ev.setdefault("data", {"recommendations": recs, "plan": plan})

    # Plan modal expects ev.data.recommendations OR ev.recommendations
    if ev.get("type") == "plan":
        recs = p.get("recommendations") or p.get("plan")
        if recs is not None:
            ev.setdefault("recommendations", recs)
            ev.setdefault("data", {"recommendations": recs})

    # Generic recommendations event (legacy shape)
    if ev.get("type") == "recommendations":
        # Prefer payload.recommendations, then items for older callers
        items = p.get("recommendations") or p.get("items") or []
        ev.setdefault("recommendations", items)
        ev.setdefault("count", p.get("count", len(items)))
        ev.setdefault("data", {"recommendations": items})

    # diff_ready expects ev.payload.unified or ev.data.unified
    if ev.get("type") in ("diff", "diffs", "diff_ready"):
        uni = p.get("unified")
        if uni is not None:
            ev.setdefault("payload", {**p, "unified": uni})
            ev.setdefault("unified", uni)  # for some older handlers
            ev.setdefault("data", {"unified": uni})

    # need_plan_approval expects suggested_files at top-level or in data
    if ev.get("type") == "need_plan_approval":
        suggestion = p.get("suggested_files") or []
        ev.setdefault("suggested_files", suggestion)
        ev.setdefault("data", {"suggested_files": suggestion})

    # project_selected expects root at top-level or in data
    if ev.get("type") == "project_selected":
        root = p.get("root")
        if root is not None:
            ev.setdefault("root", root)
            ev.setdefault("data", {"root": root})

    # approval_summary expects ev.data with summary/risk/files
    if ev.get("type") == "approval_summary":
        if "data" not in ev:
            ev["data"] = {k: v for k, v in p.items() if k in ("summary", "risk", "files")}

    # done/result expect ok and where
    if ev.get("type") in ("done", "result"):
        if "ok" in p:
            ev.setdefault("ok", p["ok"])
        if "where" in p:
            ev.setdefault("where", p["where"])

    return ev


def _ensure_data_alias(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    (Deprecated) Ensure ev.data mirrors ev.payload for older frontends that read `data`
    instead of `payload`. If a more specific data shape was already provided
    by _overlay_backcompat, it is left untouched.

    Kept for gradual migration; NOT applied to emitted envelopes anymore.
    """
    if "payload" in ev and "data" not in ev:
        ev["data"] = ev["payload"]
    return ev


def _s(v: Any) -> Optional[str]:
    """Coerce to string, but keep None as None (avoid 'None')."""
    if v is None:
        return None
    return str(v)

def _ensure_type_alias(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Back-compat: allow callers to use 'event' instead of 'type'."""
    if "type" not in ev and "event" in ev:
        ev["type"] = ev.get("event")
    if "event" not in ev and "type" in ev:
        ev["event"] = ev.get("type")
    return ev

def _normalize_payload(payload: Any) -> Any:
    """
    Keep payload as object/array/null. If it's some other scalar, wrap it.
    This avoids schema failures when a caller passes payload='some string'.
    """
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    # scalar / unknown → wrap
    return {"text": str(payload)}

def _validate_event(event: Dict[str, Any]) -> None:
    """
    Validate event against JSON schema.

    Default (STRICT_EVENTS=0):
      - validate only events whose type is in schema enum (known types)
      - log DEBUG on first violation and continue

    Strict (STRICT_EVENTS=1):
      - validate all events
      - raise on first violation
    """
    if _EVENT_VALIDATOR is None:
        return

    ev_type = event.get("type")
    if not STRICT_EVENTS and ev_type in _KNOWN_EVENT_TYPES:
        # validate known types only
        errors = _EVENT_VALIDATOR.iter_errors(event)
        for err in errors:
            log.debug("Event schema violation: %s (event=%s)", err.message, event)
            return
        return

    if STRICT_EVENTS:
        for err in _EVENT_VALIDATOR.iter_errors(event):
            raise ValueError(f"Event schema violation: {err.message} (type={ev_type})")


def _emit_raw(event: Dict[str, Any], *, session_id: Optional[str] = None) -> None:
    """
    Fan out to registered observers (server bridge, metrics, tests, etc.).
    """
    for _, obs in list(_emit_observers.items()):
        try:
            obs(event)
        except Exception:
            pass

    # Optional: structured logging
    try:
        # Include contextual metadata (model/job_id/recId when present) to aid tracing of LLM calls.
        meta: Dict[str, Any] = {"type": event.get("type"), "sid": session_id}
        payload = event.get("payload") if isinstance(event, dict) else None
        if isinstance(payload, dict):
            if "model" in payload:
                meta["model"] = payload.get("model")
            if "job_id" in payload:
                meta["job_id"] = payload.get("job_id")
            if "recId" in payload:
                meta["recId"] = payload.get("recId")
        # also respect a top-level model key if present (rare; do not mutate event)
        if isinstance(event, dict) and "model" in event:
            meta.setdefault("model", event.get("model"))
        log.debug("emit: %s", _json(meta))
    except Exception:
        pass


def _emit(ev_type: str, payload: Any, *, session_id: Optional[str] = None) -> None:
    # Normalize payload shape (dict/list/null; scalars get wrapped)
    payload_norm = _normalize_payload(payload)

    # Ensure payload is dict if we need to inject session_id
    if isinstance(payload_norm, dict):
        payload_dict = dict(payload_norm)
        if session_id is not None and "session_id" not in payload_dict:
            payload_dict["session_id"] = session_id
        payload_norm = payload_dict

    # Canonical envelope
    event: Dict[str, Any] = {"type": ev_type, "payload": payload_norm, "ts": time.time()}

    # Back-compat alias support ('event' <-> 'type') for downstream observers/tools
    _ensure_type_alias(event)

    # Validate (non-strict by default; strict if STRICT_EVENTS=1)
    try:
        _validate_event(event)
    except Exception as e:
        if STRICT_EVENTS:
            raise
        log.debug("Event validation skipped/failed non-strict: %s", e)

    _emit_raw(event, session_id=session_id)


def _clamp_pct(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    return 0.0 if f < 0 else 100.0 if f > 100 else f


def _infer_where(message: Any, stage: Optional[str]) -> Optional[str]:
    """
    Best-effort guess for 'where' this status belongs, based on a short
    human-readable message. If the input isn't a string, we just give up.
    """
    if stage:
        return stage
    if not isinstance(message, str):
        return None
    base = message.split(":", 1)[0] if ":" in message else message
    base = base.strip()
    return base or None


def _ensure_rec_id(rec_id: Optional[str]) -> str:
    return rec_id or uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Public emit API (typed helpers)
# Allowed types commonly used: "status" | "token" | "assistant" | "plan" | "recommendations"
#                               "diff" | "checks" | "result" | "error" | "analysis_result" | "qa_answer"
# Legacy helpers kept for UI compatibility: "diff_ready", "approval_summary",
#                                           "project_selected", "need_plan_approval", "done"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Deep Research (v1) event type constants
#
# NOTE: These types should be added to aidev/schemas/events.schema.json enum in
# a follow-up to enable validation when STRICT_EVENTS=1.
# ---------------------------------------------------------------------------

DEEP_RESEARCH_PHASE_STARTED = "deep_research.phase_started"
DEEP_RESEARCH_PHASE_DONE = "deep_research.phase_done"
DEEP_RESEARCH_CACHE_HIT = "deep_research.cache_hit"
DEEP_RESEARCH_CACHE_MISS = "deep_research.cache_miss"
DEEP_RESEARCH_START = "deep_research.start"
DEEP_RESEARCH_DONE = "deep_research.done"
DEEP_RESEARCH_ATTACHED = "deep_research.attached_to_payload"
DEEP_RESEARCH_ARTIFACT_WRITTEN = "deep_research.artifact_written"
DEEP_RESEARCH_BUDGET_UPDATE = "deep_research.budget_update"
DEEP_RESEARCH_CACHE_WRITE = "deep_research.cache_write"

# Deep Research lifecycle events (v1)
DEEP_RESEARCH_REQUEST_RECEIVED = "deep_research.request_received"
DEEP_RESEARCH_PLAN_CREATED = "deep_research.plan_created"
DEEP_RESEARCH_GATHER_STATS = "deep_research.gather_stats"
DEEP_RESEARCH_SYNTH_DONE = "deep_research.synthesize_done"
DEEP_RESEARCH_VERIFY_DONE = "deep_research.verify_done"
DEEP_RESEARCH_VERIFY_SKIPPED = "deep_research.verify_skipped"


def _coerce_int(v: Any, default: int = 0) -> int:
    """Best-effort int coercion for metrics; never raises."""
    try:
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _whitelist_dict(src: Any, allowed_keys: Iterable[str]) -> Dict[str, Any]:
    """Shallow-copy only whitelisted keys from a dict-like object.

    Defensive: never raises, never returns non-dict.
    """
    if not isinstance(src, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in allowed_keys:
        if k in src:
            out[k] = src.get(k)
    return out


def _repo_relative(path: Optional[str], repo_root: Optional[str] = None) -> Optional[str]:
    """Return a repo-relative path for emission, never an absolute path.

    Best effort:
      - if repo_root is provided and path resolves under it, emit the relative path
      - otherwise emit only the basename

    This is used by Deep Research helpers to avoid leaking absolute paths.
    """
    if not path:
        return None
    try:
        p = Path(path)
        if repo_root:
            try:
                root = Path(repo_root).resolve()
                return str(p.resolve().relative_to(root))
            except Exception:
                # fall back below
                pass
        return p.name
    except Exception:
        # extremely defensive: never raise from emit helpers
        try:
            return str(path).split("/")[-1].split("\\")[-1]
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Analyze plan validation + completion event
# ---------------------------------------------------------------------------

def _truncate(s: Any, max_len: int = 1200) -> str:
    """Deterministic truncation for diagnostics; never raises."""
    try:
        txt = "" if s is None else str(s)
    except Exception:
        txt = "<unprintable>"
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 3] + "..."


def _safe_path(parts: Iterable[Any], max_parts: int = 20) -> str:
    """Render a jsonschema error path deterministically."""
    out: List[str] = []
    try:
        for i, p in enumerate(list(parts)[:max_parts]):
            if isinstance(p, int):
                out.append(f"[{p}]")
            else:
                # dot-escape is overkill; keep simple
                out.append(str(p))
    except Exception:
        return ""
    # join: first token as-is, subsequent tokens with dot unless already [idx]
    rendered = ""
    for tok in out:
        if tok.startswith("["):
            rendered += tok
        else:
            rendered = tok if not rendered else rendered + "." + tok
    return rendered


def _coerce_plan_payload(plan_payload: Any) -> Any:
    """Best-effort coercion to dict for validation/emission; never raises."""
    if plan_payload is None:
        return None
    if isinstance(plan_payload, dict):
        return dict(plan_payload)
    try:
        return dict(plan_payload)  # type: ignore[arg-type]
    except Exception:
        return {"value": plan_payload}


def _validate_analyze_plan_payload(plan_payload: Any) -> List[Dict[str, Any]]:
    """Validate analyze plan payload against analyze_plan.schema.json.

    Returns a list of diagnostics (empty when valid). Never raises.

    Preferred: use aidev.schemas validation helper if present.
    Fallback: load schemas/analyze_plan.schema.json adjacent to this file and validate with jsonschema.
    """
    # Try centralized helper first (if present in this repo)
    try:
        from aidev import schemas as _schemas  # type: ignore

        validate_fn = getattr(_schemas, "validate_schema", None) or getattr(_schemas, "validate_payload", None)
        if callable(validate_fn):
            try:
                res = validate_fn("analyze_plan", plan_payload)
                # Accept a few possible return shapes:
                # - [] / list[dict]
                # - (ok: bool, diagnostics: list)
                # - {ok: bool, diagnostics: list}
                if isinstance(res, tuple) and len(res) == 2:
                    ok, diags = res
                    if ok:
                        return []
                    if isinstance(diags, list):
                        return [d if isinstance(d, dict) else {"message": _truncate(d)} for d in diags]
                    return [{"message": _truncate(diags)}]
                if isinstance(res, dict):
                    if res.get("ok") is True:
                        return []
                    diags = res.get("diagnostics") or res.get("errors") or []
                    if isinstance(diags, list):
                        return [d if isinstance(d, dict) else {"message": _truncate(d)} for d in diags]
                    return [{"message": _truncate(diags)}]
                if isinstance(res, list):
                    return [d if isinstance(d, dict) else {"message": _truncate(d)} for d in res]
            except Exception as e:
                return [{"message": _truncate(f"schema_validation_error: {e}")}]  # helper failed
    except Exception:
        # ignore and fall back
        pass

    # Fallback: local schema + jsonschema
    try:
        import jsonschema as _jsonschema  # type: ignore

        schema_path = Path(__file__).resolve().parent / "schemas" / "analyze_plan.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = _jsonschema.Draft7Validator(schema)

        instance = _coerce_plan_payload(plan_payload)
        diags: List[Dict[str, Any]] = []
        for err in validator.iter_errors(instance):
            diags.append(
                {
                    "path": _safe_path(err.path),
                    "message": _truncate(err.message),
                    "validator": _truncate(getattr(err, "validator", None)),
                }
            )
            if len(diags) >= 50:
                diags.append({"message": "too_many_validation_errors"})
                break
        return diags
    except Exception as e:
        return [{"message": _truncate(f"schema_validation_unavailable: {e}")}]


def emit_analyze_result(
    plan_payload: Any,
    *,
    status: str,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
    job_id: Optional[str] = None,
    recId: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit canonical analyze completion event.

    Event type: 'analyze_result'
    Payload keys: { status, plan, diagnostics, job_id, recId, ... }

    Validation behavior:
      - Always attempts to validate plan_payload against analyze_plan.schema.json.
      - Never raises on schema-invalid payload; emits status='invalid' with diagnostics.
      - If STRICT_EVENTS=1, still does not raise; logs at ERROR and emits diagnostics.

    Note: This is a completion event for the analyze pipeline (success vs invalid vs error).
    """
    rec = _ensure_rec_id(recId)

    # Validate and merge diagnostics deterministically
    computed_diags = _validate_analyze_plan_payload(plan_payload)
    merged_diags: List[Dict[str, Any]] = []
    if diagnostics:
        for d in diagnostics:
            merged_diags.append(d if isinstance(d, dict) else {"message": _truncate(d)})
    for d in computed_diags:
        merged_diags.append(d if isinstance(d, dict) else {"message": _truncate(d)})

    # If caller says success but schema invalid, force invalid (no silent fallback)
    final_status = str(status)
    if final_status == "success" and computed_diags:
        final_status = "invalid"

    if computed_diags and STRICT_EVENTS:
        try:
            log.error("Analyze plan schema invalid (STRICT_EVENTS=1): %s", _json(computed_diags[:5]))
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "status": final_status,
        "plan": _coerce_plan_payload(plan_payload),
        "diagnostics": merged_diags,
        "job_id": job_id,
        "recId": rec,
    }

    # Merge extras conservatively
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v

    _emit("analyze_result", payload, session_id=session_id)


def analyze_result(
    plan_payload: Any,
    *,
    ok: bool = True,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    analysis: Optional[str] = None,
    session_id: Optional[str] = None,
    job_id: Optional[str] = None,
    recId: Optional[str] = None,
    **extras: Any,
) -> None:
    """Legacy wrapper for analyze completion.

    Emits BOTH:
      - legacy 'analysis_result' (older callers) with a backward-compatible payload
      - canonical 'analyze_result' (via emit_analyze_result) that includes explicit
        status ('success'|'invalid'|'error') and deterministic diagnostics.

    Behavior:
      - If ok==True and the plan validates against the schema, status='success'.
      - If ok==True but validation produces diagnostics, status='invalid'.
      - If ok==False, status='error'.

    This ensures no silent fallback: schema-invalid outputs become status='invalid'
    and are surfaced in diagnostics for both legacy and canonical events.
    """
    # Determine schema diagnostics to map status consistently for legacy event
    try:
        computed_diags = _validate_analyze_plan_payload(plan_payload)
    except Exception:
        computed_diags = [{"message": "schema_validation_unavailable"}]

    # Map ok + diagnostics -> final status
    if ok:
        status = "success" if not computed_diags else "invalid"
    else:
        status = "error"

    # Legacy payload (keeps keys older consumers expect: analysis, plan, diagnostics, ok)
    legacy_payload: Dict[str, Any] = {
        "analysis": analysis,
        "plan": _coerce_plan_payload(plan_payload),
        "diagnostics": list(diagnostics or computed_diags or []),
        "ok": True if status == "success" else False,
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    # Merge extras conservatively
    for k, v in extras.items():
        if k not in legacy_payload:
            legacy_payload[k] = v

    # Emit legacy compatibility event first (so older bridges see it)
    _emit("analysis_result", legacy_payload, session_id=session_id)

    # Delegate to canonical emitter which will re-validate/merge diagnostics and emit 'analyze_result'
    # Pass user-supplied diagnostics (if any) so they are preserved/merged.
    try:
        emit_analyze_result(
            plan_payload,
            status=status,
            diagnostics=diagnostics,
            session_id=session_id,
            job_id=job_id,
            recId=recId,
            **extras,
        )
    except Exception:
        # emit_analyze_result is defensive and should not raise; but if it does, emit an error canonical event
        _emit(
            "analyze_result",
            {
                "status": "error",
                "plan": _coerce_plan_payload(plan_payload),
                "diagnostics": [{"message": "emit_analyze_result_failed"}],
                "job_id": job_id,
                "recId": _ensure_rec_id(recId),
            },
            session_id=session_id,
        )


def deep_research_phase_started(
    run_id: str,
    phase: str,
    cache_key: Optional[str] = None,
    cache_ref: Optional[str] = None,
    *,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a canonical Deep Research phase start event.

    See: aidev/schemas/events.schema.json (TODO: add DEEP_RESEARCH_* types to schema).

    Payload keys: run_id, phase, cache_key?, cache_ref?, recId, job_id?
    """
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "phase": str(phase),
        "cache_key": cache_key,
        "cache_ref": cache_ref,
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_PHASE_STARTED, payload, session_id=session_id)


def deep_research_phase_done(
    run_id: str,
    phase: str,
    *,
    ok: bool = True,
    summary: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a canonical Deep Research phase done event.

    Payload keys: run_id, phase, ok, summary?, recId, job_id?
    """
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "phase": str(phase),
        "ok": bool(ok),
        "summary": summary,
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_PHASE_DONE, payload, session_id=session_id)


def deep_research_start(
    run_id: str,
    *,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit Deep Research gating start marker.

    Acceptance tests assert that, when research gating is enabled, the stream
    includes deep_research.start -> deep_research.done before downstream LLM calls.

    NOTE: These event types should be added to aidev/schemas/events.schema.json enum
    in a follow-up to allow STRICT_EVENTS=1 validation.
    """
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_START, payload, session_id=session_id)


def deep_research_done(
    run_id: str,
    ok: bool = True,
    *,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit Deep Research gating done marker.

    This provides the exact deep_research.done name required by acceptance tests
    while preserving existing phase_* events used elsewhere.
    """
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "ok": bool(ok),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_DONE, payload, session_id=session_id)


def deep_research_attached_to_payload(
    run_id: str,
    evidence_items: int,
    findings: int,
    truncated: bool,
    *,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    artifact_ref: Optional[str] = None,
    repo_root: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit an attachment event indicating Deep Research output was added to an LLM payload.

    Payload is intentionally small and audit-safe:
      - includes counts (evidence_items, findings) and truncated flag
      - may include artifact_ref (repo-relative/basename redacted)
      - MUST NOT include raw file contents or artifact text

    This helper will NOT allow overwriting the canonical keys and will drop
    obvious raw-content extras. It also blocks 'research_brief' to avoid
    accidentally attaching large human-readable content.

    NOTE: Add DEEP_RESEARCH_ATTACHED to the schema enum in a follow-up.
    """
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "evidence_items": _coerce_int(evidence_items, 0),
        "findings": _coerce_int(findings, 0),
        "truncated": bool(truncated),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }

    if artifact_ref is not None:
        payload["artifact_ref"] = _repo_relative(str(artifact_ref), repo_root=repo_root)

    # Merge extras conservatively, but drop obvious raw-content keys and block research_brief
    for k, v in extras.items():
        if k in payload:
            continue
        lk = str(k).lower()
        if any(s in lk for s in ("content", "contents", "file_text", "raw", "text", "prompt", "research_brief")):
            continue
        payload[k] = v

    _emit(DEEP_RESEARCH_ATTACHED, payload, session_id=session_id)


def deep_research_cache_event(
    kind: str,
    run_id: str,
    artifact_type: str,
    *,
    cache_ref: Optional[str] = None,
    repo_path: Optional[str] = None,
    repo_root: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit cache hit/miss events for Deep Research.

    kind: 'hit' | 'miss' (stringly typed to avoid importing Literal)

    Payload keys: run_id, artifact_type, cache_ref?, repo_path?, recId, job_id?

    NOTE: repo_path is redacted to a repo-relative path or basename via _repo_relative.
    """
    ev_type = DEEP_RESEARCH_CACHE_HIT if str(kind).lower() == "hit" else DEEP_RESEARCH_CACHE_MISS
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "artifact_type": str(artifact_type),
        "cache_ref": cache_ref,
        "repo_path": _repo_relative(repo_path, repo_root=repo_root),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(ev_type, payload, session_id=session_id)


def deep_research_cache_hit(
    run_id: str,
    artifact_type: str,
    *,
    cache_ref: Optional[str] = None,
    deep_research_digest: Optional[Any] = None,
    repo_path: Optional[str] = None,
    repo_root: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """First-class helper for emitting a deep_research.cache_hit event.

    Mirrors the pattern of deep_research_start/done/attached helpers and
    ensures a consistent payload shape that can include a sanitized
    deep_research_digest. The digest is sanitized to only include audit-safe
    fields (evidence_items, findings, truncated, artifact_ref) to avoid
    embedding raw file contents.
    """
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "artifact_type": str(artifact_type),
        "cache_ref": cache_ref,
        "repo_path": _repo_relative(repo_path, repo_root=repo_root),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }

    # Sanitize provided digest: only allow specific safe keys and map findings_count -> findings
    if deep_research_digest is not None:
        sanitized: Dict[str, Any] = {}
        if isinstance(deep_research_digest, dict):
            for k, v in deep_research_digest.items():
                lk = str(k).lower()
                if lk in ("evidence_items", "evidence_count"):
                    sanitized["evidence_items"] = _coerce_int(v, 0)
                elif lk in ("findings", "findings_count"):
                    sanitized["findings"] = _coerce_int(v, 0)
                elif lk == "truncated":
                    sanitized["truncated"] = bool(v)
                elif lk in ("artifact_ref", "artifact"):
                    sanitized["artifact_ref"] = _repo_relative(str(v), repo_root=repo_root)
        else:
            # Non-dict digest: include only its string representation under 'digest'
            try:
                sanitized["digest"] = str(deep_research_digest)
            except Exception:
                sanitized["digest"] = "<unserializable>"
        payload["deep_research_digest"] = sanitized

    # Merge extras conservatively (do not overwrite canonical keys)
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v

    _emit(DEEP_RESEARCH_CACHE_HIT, payload, session_id=session_id)


def deep_research_cache_miss(
    run_id: str,
    artifact_type: str,
    *,
    cache_ref: Optional[str] = None,
    repo_path: Optional[str] = None,
    repo_root: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """First-class helper for emitting a deep_research.cache_miss event."""
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "artifact_type": str(artifact_type),
        "cache_ref": cache_ref,
        "repo_path": _repo_relative(repo_path, repo_root=repo_root),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_CACHE_MISS, payload, session_id=session_id)


def deep_research_cache_write(
    run_id: str,
    cache_key: str,
    artifact_type: str,
    *,
    cache_ref: Optional[str] = None,
    repo_path: Optional[str] = None,
    repo_root: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a Deep Research cache write event.

    Payload keys: run_id, cache_key (stable identifier), artifact_type, cache_ref?, repo_path?, recId, job_id?

    NOTE: This must not include raw artifact contents. repo_path is redacted using _repo_relative.
    """
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "cache_key": "" if cache_key is None else str(cache_key),
        "artifact_type": str(artifact_type),
        "cache_ref": cache_ref,
        "repo_path": _repo_relative(repo_path, repo_root=repo_root),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_CACHE_WRITE, payload, session_id=session_id)


def deep_research_artifact_written(
    run_id: str,
    artifact_type: str,
    *,
    repo_path: Optional[str] = None,
    repo_root: Optional[str] = None,
    artifact_ref: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a Deep Research artifact-written event (no contents).

    Payload keys: run_id, artifact_type, repo_path?, artifact_ref?, recId, job_id?

    The intent is to point to where an artifact was written (path/ref), without
    embedding file contents or absolute paths.
    """
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "artifact_type": str(artifact_type),
        "repo_path": _repo_relative(repo_path, repo_root=repo_root),
        "artifact_ref": artifact_ref,
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_ARTIFACT_WRITTEN, payload, session_id=session_id)


def deep_research_budget_update(
    run_id: str,
    before: float,
    after: float,
    reason: str,
    *,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a Deep Research budget update event.

    Payload keys: run_id, before, after, delta, reason, recId, job_id?

    Values are coerced to float when possible to keep payload JSON-serializable.
    """
    try:
        b = float(before)
    except Exception:
        b = 0.0
    try:
        a = float(after)
    except Exception:
        a = b

    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "before": b,
        "after": a,
        "delta": a - b,
        "reason": "" if reason is None else str(reason),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_BUDGET_UPDATE, payload, session_id=session_id)


def deep_research_request_received(
    run_id: str,
    profile: Dict[str, Any],
    budget: Dict[str, Any],
    *,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a Deep Research 'request received' event.

    This event is intended for auditability and should not include secrets or raw file contents.
    Only a small whitelist of profile/budget keys is emitted.
    """
    # Conservative whitelist: deterministic, non-secret-ish metadata only.
    # (Callers can still pass additional safe metadata via **extras.)
    profile_allow = (
        "name",
        "profile",
        "mode",
        "strategy",
        "provider",
        "model",
        "temperature",
        "top_p",
        "max_output_tokens",
        "max_tokens",
        "seed",
    )
    budget_allow = (
        "before",
        "after",
        "delta",
        "reason",
        "max_steps",
        "max_queries",
        "max_files",
        "max_bytes",
        "time_budget_s",
        "token_budget",
        "cost_budget",
        "currency",
    )

    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "profile": _whitelist_dict(profile, profile_allow),
        "budget": _whitelist_dict(budget, budget_allow),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }

    for k, v in extras.items():
        if k in payload:
            continue
        # Avoid accidental secrets/raws via obvious key names.
        lk = str(k).lower()
        if any(s in lk for s in ("secret", "token", "apikey", "api_key", "password", "content", "contents", "file_text", "raw")):
            continue
        payload[k] = v

    _emit(DEEP_RESEARCH_REQUEST_RECEIVED, payload, session_id=session_id)


def deep_research_plan_created(
    run_id: str,
    scoped_paths_count: int,
    queries_count: Optional[int] = None,
    planned_steps_count: Optional[int] = None,
    *,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a Deep Research 'plan created' event with deterministic counts."""
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "scoped_paths_count": _coerce_int(scoped_paths_count, 0),
        "queries_count": (_coerce_int(queries_count, 0) if queries_count is not None else None),
        "planned_steps_count": (
            _coerce_int(planned_steps_count, 0) if planned_steps_count is not None else None
        ),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_PLAN_CREATED, payload, session_id=session_id)


def deep_research_gather_stats(
    run_id: str,
    files_examined: int,
    bytes_read: int,
    evidence_count: int,
    *,
    repo_root: Optional[str] = None,
    sample_paths: Optional[Iterable[str]] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit deterministic gather stats for Deep Research.

    Payload keys: files_examined, bytes_read, evidence_count.
    Optionally includes sample_paths (repo-relative/basename redacted).
    """
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "files_examined": _coerce_int(files_examined, 0),
        "bytes_read": _coerce_int(bytes_read, 0),
        "evidence_count": _coerce_int(evidence_count, 0),
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }

    if sample_paths is not None:
        try:
            payload["sample_paths"] = [
                p
                for p in (
                    _repo_relative(str(sp), repo_root=repo_root) for sp in list(sample_paths or [])
                )
                if p
            ]
        except Exception:
            # don't fail instrumentation
            pass

    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_GATHER_STATS, payload, session_id=session_id)


def deep_research_synthesize_done(
    run_id: str,
    findings: int,
    output_bytes: int,
    *,
    summary: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a Deep Research synthesize completion event (once per run)."""
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "findings": _coerce_int(findings, 0),
        "output_bytes": _coerce_int(output_bytes, 0),
        "summary": summary,
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_SYNTH_DONE, payload, session_id=session_id)


def deep_research_verify_done(
    run_id: str,
    *,
    ok: bool = True,
    summary: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a Deep Research verify completion event."""
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "ok": bool(ok),
        "summary": summary,
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_VERIFY_DONE, payload, session_id=session_id)


def deep_research_verify_skipped(
    run_id: str,
    *,
    reason: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a Deep Research verify skipped event."""
    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "reason": reason,
        "job_id": job_id,
        "recId": _ensure_rec_id(recId),
    }
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit(DEEP_RESEARCH_VERIFY_SKIPPED, payload, session_id=session_id)

# Public stable emitter alias exported for other modules (e.g., deep_research_cache.py)
def emit_event(event_type: str, payload: Any, *, session_id: Optional[str] = None) -> None:
    """Public stable emitter alias expected across the codebase.

    Other modules import `emit_event` from aidev.events; this wrapper delegates
    to the internal _emit function and preserves the expected signature
    (event_type: str, payload: Any, *, session_id: Optional[str]=None).
    """
    _emit(event_type, payload, session_id=session_id)


def emit_status(
  message: Any,
  *,
  stage: Optional[str] = None,
  progress_pct: Optional[float] = None,
  file: Optional[str] = None,
  where: Optional[str] = None,
  session_id: Optional[str] = None,
  detail: Optional[str] = None,
  recId: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Generic status/progress line (type: 'status').

    `message` can be:
      - a string: becomes the human-readable message
      - a dict: used as base payload; we derive a short text from common keys
        like 'message', 'msg', 'status', or 'event' for where inference and
        for the top-level message fields.
    """
    progress_pct = _clamp_pct(progress_pct)

    base_payload: Dict[str, Any]
    if isinstance(message, dict):
        base_payload = dict(message)  # shallow copy
        text_for_where = (
            base_payload.get("message")
            or base_payload.get("msg")
            or base_payload.get("status")
            or base_payload.get("event")
            or ""
        )
    else:
        base_payload = {}
        text_for_where = "" if message is None else str(message)

    where = where or _infer_where(text_for_where, stage)

    if session_id:
        st = _last_topic.setdefault(session_id, _TopicState())
        st.last_where = where
        st.last_stage = stage or st.last_stage

    # Prefer explicit recId argument, then any recId on the base payload, then generate one.
    rec_id_value = _ensure_rec_id(recId or base_payload.get("recId"))

    payload = {
        **base_payload,
        "message": text_for_where,
        "msg": text_for_where,  # legacy key kept inside payload only
        "stage": stage,
        "where": where,
        "progress_pct": progress_pct,
        "file": file,
        "detail": detail,
        "recId": rec_id_value,
        **extras,
    }
    _emit("status", payload, session_id=session_id)


def progress_start(
  stage: str,
  *,
  detail: Optional[str] = None,
  file: Optional[str] = None,
  recId: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> str:
    """Convenience start-of-stage status (stage examples: 'plan', 'apply', 'checks')."""
    rid = _ensure_rec_id(recId)
    emit_status(
        f"{stage}: start",
        stage=stage,
        progress_pct=0,
        file=file,
        detail=detail,
        recId=rid,
        session_id=session_id,
        **extras,
    )
    return rid


def progress_update(
  stage: str,
  *,
  progress_pct: float,
  detail: Optional[str] = None,
  file: Optional[str] = None,
  recId: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    emit_status(
        f"{stage}: progress",
        stage=stage,
        progress_pct=progress_pct,
        file=file,
        detail=detail,
        recId=recId,
        session_id=session_id,
        **extras,
    )


def progress_finish(
  stage: str,
  *,
  detail: Optional[str] = None,
  file: Optional[str] = None,
  ok: bool = True,
  recId: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    rid = _ensure_rec_id(recId)
    emit_status(
        f"{stage}: finish",
        stage=stage,
        progress_pct=100,
        file=file,
        detail=detail,
        recId=rid,
        session_id=session_id,
        **extras,
    )
    done(ok=ok, where=stage, session_id=session_id, recId=rid, **extras)


def emit_token(delta: str, *, session_id: Optional[str] = None, model: Optional[str] = None, **extras: Any) -> None:
    """Streamed token chunk from an LLM (for live typing UX).

    When `model` is provided it will be attached to payload.model in the emitted event.
    """
    payload: Dict[str, Any] = {"delta": delta}
    # Attach explicit model param (explicit param wins over extras)
    if model is not None:
        payload["model"] = model
    # Merge extras conservatively (do not overwrite explicit model)
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit("token", payload, session_id=session_id)


def assistant(text: Optional[str] = None, *, session_id: Optional[str] = None, model: Optional[str] = None, **extras: Any) -> None:
    """Chat bubble from the assistant (UI renders inline).

    When `model` is provided it will be attached to payload.model in the emitted event.
    """
    payload: Dict[str, Any] = {"text": text or ""}
    if model is not None:
        payload["model"] = model
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit("assistant", payload, session_id=session_id)


def chat(text: str, *, session_id: Optional[str] = None, model: Optional[str] = None, **extras: Any) -> None:
    """Alias for assistant(). Accepts `model` which will be attached as payload.model when provided."""
    assistant(text=text, session_id=session_id, model=model, **extras)


def emit_plan(
  plan_items: List[Dict[str, Any]],
  *,
  session_id: Optional[str] = None,
  summary: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Planner output: list of recommended actions/targets with rationale.
    type: "plan"
    payload: { "recommendations": [...], "summary": "..." }
    """
    payload = {"recommendations": plan_items, "summary": summary, **extras}
    _emit("plan", payload, session_id=session_id)


def recommendations(
  items: List[Dict[str, Any]],
  *,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Broadcast a recommendations list.

    Canonical envelope:
      {
        "type": "recommendations",
        "payload": {
          "recommendations": [...],
          "count": <int>,
          ...
        }
      }
    Note: legacy "items" mirror removed from payload to enforce canonical shape.
    """
    payload = {
        "recommendations": list(items),
        "count": len(items),
        **extras,
    }
    _emit("recommendations", payload, session_id=session_id)


def need_plan_approval(
  suggested_files: Optional[List[str]] = None,
  *,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """Ask UI to open plan modal even without full recs yet (type: 'need_plan_approval')."""
    payload = {"suggested_files": list(suggested_files or []), **extras}
    _emit("need_plan_approval", payload, session_id=session_id)


def awaiting_approval(
  message: str = "Awaiting approval to apply changes.",
  *,
  session_id: Optional[str] = None,
  detail: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Explicit status when the bot is blocked on UI approval of a plan/diff.

    UI can watch for:
      type: "status"
      payload.stage == "awaiting_approval"
    """
    emit_status(
        message,
        stage="awaiting_approval",
        detail=detail,
        session_id=session_id,
        **extras,
    )


def no_recommendations(
  message: str = "No recommendations were generated for this request.",
  *,
  reason: Optional[str] = None,
  session_id: Optional[str] = None,
  stage: str = "plan",
  **extras: Any,
) -> None:
    """
    Explicit "no recommendations" event for UI friendliness.

    Emits BOTH:
      - status event (stage typically 'plan') so status bar can show a message
      - recommendations event with an empty list so the plan modal / list UI
        can render a friendly "nothing to do" state.
    """
    # Status line for status bar / run log
    emit_status(
        message,
        stage=stage,
        detail=reason,
        session_id=session_id,
        no_recommendations=True,
        **extras,
    )
    # Empty recommendations payload for plan UI
    recommendations([], session_id=session_id, reason=reason, **extras)


def emit_diff(
  unified: Optional[str] = None,
  *,
  files_changed: Optional[int] = None,
  bundle: Optional[Dict[str, Any]] = None,
  summary: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """Modern diff event (type: 'diff')."""
    payload = {
        "unified": unified,
        "files_changed": files_changed,
        "bundle": bundle,
        "summary": summary,
        **extras,
    }
    _emit("diff", payload, session_id=session_id)


def diff_ready(
  unified: Optional[str] = None,
  *,
  files_changed: Optional[int] = None,
  summary: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """Legacy/compat diff event (type: 'diff_ready')."""
    payload = {"unified": unified, "files_changed": files_changed, "summary": summary, **extras}
    _emit("diff_ready", payload, session_id=session_id)

# ---------- New SSE helpers for edit-attempt lifecycle (tiered edit strategy) ----------

def sse_edit_attempt_start(
    attempt: Any,
    file: Optional[str] = None,
    output_type: Optional[str] = None,
    *,
    model: Optional[str] = None,
    classifier: Optional[str] = None,
    error: Optional[str] = None,
    attempt_meta: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit 'edit_attempt_start'.

    Payload keys: attempt (int), file (str), output_type (str), model?, classifier?, error?, attempt_meta?, recId, job_id?.

    Back-compat: tolerate single-dict payload as first positional argument.
    """
    # Backwards-compatibility: caller may pass a single dict payload as the first arg.
    if isinstance(attempt, dict) and file is None and output_type is None:
        p = dict(attempt)
        # extract common names; accept a few legacy variants
        attempt = p.get("attempt") or p.get("attempt_num")
        file = p.get("file") or p.get("file_path") or p.get("path")
        output_type = p.get("output_type") or p.get("outputType") or p.get("type")
        model = p.get("model", model)
        classifier = p.get("classifier") or p.get("classification")
        
        error = p.get("error", error)
        attempt_meta = p.get("attempt_meta") or p.get("meta") or attempt_meta
        session_id = p.get("session_id") or p.get("session") or session_id
        recId = p.get("recId") or p.get("rec_id") or recId
        job_id = p.get("job_id") or p.get("jobId") or job_id
        # Merge remaining keys into extras, without overwriting existing extras
        for k, v in p.items():
            if k not in ("attempt", "attempt_num", "file", "file_path", "path", "output_type", "outputType", "type", "model", "classifier", "classification", "error", "attempt_meta", "meta", "session_id", "session", "recId", "rec_id", "job_id", "jobId") and k not in extras:
                extras[k] = v

    # Now build the canonical payload as before
    payload: Dict[str, Any] = {"attempt": int(attempt), "file": _s(file), "output_type": _s(output_type)}
    if model is not None:
        payload["model"] = model
    if classifier is not None:
        payload["classifier"] = classifier
    if error is not None:
        payload["error"] = error
    if attempt_meta is not None:
        payload["attempt_meta"] = dict(attempt_meta)
    if job_id is not None:
        payload["job_id"] = job_id

    payload["recId"] = _ensure_rec_id(recId)

    # Merge extras conservatively
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit("edit_attempt_start", payload, session_id=session_id)


def sse_edit_attempt_result(
    attempt: Any,
    file: Optional[str] = None,
    output_type: Optional[str] = None,
    *,
    success: Optional[bool] = None,
    result: Optional[Any] = None,
    model: Optional[str] = None,
    attempt_meta: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit 'edit_attempt_result'.

    Payload keys: attempt, file, output_type, success (bool), result (optional), model?, attempt_meta?, recId, job_id?

    Back-compat: tolerate single-dict payload as first positional argument.
    """
    if isinstance(attempt, dict) and file is None and output_type is None:
        p = dict(attempt)
        attempt = p.get("attempt") or p.get("attempt_num")
        file = p.get("file") or p.get("file_path") or p.get("path")
        output_type = p.get("output_type") or p.get("outputType") or p.get("type")
        success = p.get("success") if success is None else success
        result = p.get("result", result)
        model = p.get("model", model)
        attempt_meta = p.get("attempt_meta") or p.get("meta") or attempt_meta
        session_id = p.get("session_id") or p.get("session") or session_id
        recId = p.get("recId") or p.get("rec_id") or recId
        job_id = p.get("job_id") or p.get("jobId") or job_id
        for k, v in p.items():
            if k not in ("attempt", "attempt_num", "file", "file_path", "path", "output_type", "outputType", "type", "success", "result", "model", "attempt_meta", "meta", "session_id", "session", "recId", "rec_id", "job_id", "jobId") and k not in extras:
                extras[k] = v

    payload: Dict[str, Any] = {
        "attempt": int(attempt),
        "file": _s(file),
        "output_type": _s(output_type),
        "success": (bool(success) if success is not None else None),
    }

    if result is not None:
        payload["result"] = result
    if model is not None:
        payload["model"] = model
    if attempt_meta is not None:
        payload["attempt_meta"] = dict(attempt_meta)
    if job_id is not None:
        payload["job_id"] = job_id

    payload["recId"] = _ensure_rec_id(recId)

    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit("edit_attempt_result", payload, session_id=session_id)


def sse_patch_apply_failed(
    attempt: Any,
    file: Optional[str] = None,
    *,
    classifier: Optional[str] = None,
    error: Optional[str] = None,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    attempt_meta: Optional[Dict[str, Any]] = None,
    **extras: Any,
) -> None:
    """Emit 'patch_apply_failed'.

    Payload keys: attempt, file, classifier ('invalid_diff'|'context_mismatch'|'unknown'), error (text), model?, recId, job_id, attempt_meta?

    Back-compat: tolerate single-dict payload as first positional argument.
    """
    if isinstance(attempt, dict) and file is None:
        p = dict(attempt)
        attempt = p.get("attempt") or p.get("attempt_num")
        file = p.get("file") or p.get("file_path") or p.get("path")
        classifier = p.get("classifier") or p.get("classification") or classifier
        error = p.get("error", error)
        model = p.get("model", model)
        attempt_meta = p.get("attempt_meta") or p.get("meta") or attempt_meta
        session_id = p.get("session_id") or p.get("session") or session_id
        recId = p.get("recId") or p.get("rec_id") or recId
        job_id = p.get("job_id") or p.get("jobId") or job_id
        for k, v in p.items():
            if k not in ("attempt", "attempt_num", "file", "file_path", "path", "classifier", "classification", "error", "model", "attempt_meta", "meta", "session_id", "session", "recId", "rec_id", "job_id", "jobId") and k not in extras:
                extras[k] = v

    payload: Dict[str, Any] = {"attempt": int(attempt), "file": _s(file)}
    if classifier is not None:
        payload["classifier"] = classifier
    if error is not None:
        payload["error"] = error
    if model is not None:
        payload["model"] = model
    if attempt_meta is not None:
        payload["attempt_meta"] = dict(attempt_meta)
    if job_id is not None:
        payload["job_id"] = job_id

    payload["recId"] = _ensure_rec_id(recId)

    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit("patch_apply_failed", payload, session_id=session_id)


def sse_fallback_full_content(
    file: Any,
    from_attempt: Optional[int] = -1,
    *,
    reason: Optional[str] = None,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    attempt_meta: Optional[Dict[str, Any]] = None,
    **extras: Any,
) -> None:
    """Emit 'fallback_full_content' indicating a forced full-file fallback.

    Payload keys: file, from_attempt (int), reason?, model?, recId?, job_id?, attempt_meta?

    Back-compat: tolerate single-dict payload as first positional argument.
    """
    if isinstance(file, dict) and from_attempt is None:
        p = dict(file)
        file = p.get("file") or p.get("file_path") or p.get("path")
        from_attempt = p.get("from_attempt") or p.get("fromAttempt") or p.get("attempt")
        reason = p.get("reason", reason)
        model = p.get("model", model)
        attempt_meta = p.get("attempt_meta") or p.get("meta") or attempt_meta
        session_id = p.get("session_id") or p.get("session") or session_id
        recId = p.get("recId") or p.get("rec_id") or recId
        job_id = p.get("job_id") or p.get("jobId") or job_id
        for k, v in p.items():
            if k not in ("file", "file_path", "path", "from_attempt", "fromAttempt", "attempt", "reason", "model", "attempt_meta", "meta", "session_id", "session", "recId", "rec_id", "job_id", "jobId") and k not in extras:
                extras[k] = v

    payload: Dict[str, Any] = {"file": _s(file), "from_attempt": int(from_attempt)}
    if reason is not None:
        payload["reason"] = reason
    if model is not None:
        payload["model"] = model
    if attempt_meta is not None:
        payload["attempt_meta"] = dict(attempt_meta)
    if job_id is not None:
        payload["job_id"] = job_id

    payload["recId"] = _ensure_rec_id(recId)

    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit("fallback_full_content", payload, session_id=session_id)


def sse_edit_finalized(
    file: Any,
    ok: Optional[bool] = None,
    *,
    attempt_history: Optional[List[Dict[str, Any]]] = None,
    final_content_summary: Optional[str] = None,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    recId: Optional[str] = None,
    job_id: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit 'edit_finalized' summarizing the final outcome for a file edit.

    Payload keys: file, ok (bool), attempt_history? (list), final_content_summary? (str), model?, recId?, job_id?

    Back-compat: tolerate single-dict payload as first positional argument.
    """
    if isinstance(file, dict) and ok is None:
        p = dict(file)
        file = p.get("file") or p.get("file_path") or p.get("path")
        ok = p.get("ok")
        attempt_history = p.get("attempt_history") or attempt_history
        final_content_summary = p.get("final_content_summary") or p.get("summary") or final_content_summary
        model = p.get("model", model)
        session_id = p.get("session_id") or p.get("session") or session_id
        recId = p.get("recId") or p.get("rec_id") or recId
        job_id = p.get("job_id") or p.get("jobId") or job_id
        for k, v in p.items():
            if k not in ("file", "file_path", "path", "ok", "attempt_history", "final_content_summary", "summary", "model", "session_id", "session", "recId", "rec_id", "job_id", "jobId") and k not in extras:
                extras[k] = v

    payload: Dict[str, Any] = {"file": _s(file), "ok": (bool(ok) if ok is not None else None)}
    if attempt_history is not None:
        payload["attempt_history"] = list(attempt_history)
    if final_content_summary is not None:
        payload["final_content_summary"] = final_content_summary
    if model is not None:
        payload["model"] = model
    if job_id is not None:
        payload["job_id"] = job_id

    payload["recId"] = _ensure_rec_id(recId)

    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit("edit_finalized", payload, session_id=session_id)

# ---------- End of new SSE helpers ----------


def checks_started(
  total: Optional[int] = None,
  *,
  stage: str = "checks",
  recId: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> str:
    """Signal that build/tests/checks are starting."""
    rid = _ensure_rec_id(recId)
    _emit(
        "checks",
        {"stage": stage, "ok": None, "total": total, "recId": rid, **extras},
        session_id=session_id,
    )
    # Also emit a status to kick progress bar
    emit_status(f"{stage}: start", stage=stage, progress_pct=0, recId=rid, session_id=session_id)
    return rid


def checks_result(
  ok: bool,
  *,
  duration_ms: Optional[int] = None,
  passed: Optional[int] = None,
  failed: Optional[int] = None,
  artifacts: Optional[Iterable[str]] = None,
  summary: Optional[str] = None,
  stage: str = "checks",
  recId: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """Publish the final checks outcome."""
    rid = _ensure_rec_id(recId)
    payload = {
        "stage": stage,
        "ok": bool(ok),
        "duration_ms": duration_ms,
        "passed": passed,
        "failed": failed,
        "artifacts": list(artifacts or []),
        "summary": summary,
        "recId": rid,
        **extras,
    }
    _emit("checks", payload, session_id=session_id)
    progress_finish(stage, ok=ok, detail=summary, recId=rid, session_id=session_id)


def approval_summary(
  summary: str,
  risk: str,
  files: List[Dict[str, Any]],
  *,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Broadcast a headline summary for the approval gate.
    Payload keys: summary (str), risk (low|medium|high), files ([{path, added, removed, why?}])
    """
    payload = {"summary": summary, "risk": risk, "files": files, **extras}
    _emit("approval_summary", payload, session_id=session_id)


def project_selected(root: str, *, session_id: Optional[str] = None, **extras: Any) -> None:
    """Notify that a project was selected."""
    payload = {"root": root, **extras}
    _emit("project_selected", payload, session_id=session_id)


def cards_refresh_start(changed_paths: Iterable[str], *, refreshed_count: Optional[int] = None, session_id: Optional[str] = None, **extras: Any) -> None:
    """Emit 'cards.refresh.start' with payload {changed_paths, refreshed_count}.

    changed_paths: iterable of repo-relative paths changed by the recommendation.
    refreshed_count: optional explicit count; if None, len(changed_paths) is used.

    Extras are merged conservatively and will not overwrite the canonical
    changed_paths/refreshed_count keys.

    Canonical payload keys: 'changed_paths' (List[str]) and 'refreshed_count' (int).
    Consumers should rely on those keys when reacting to cards refresh events.
    """
    paths = [str(p) for p in list(changed_paths or [])]
    count = int(refreshed_count) if refreshed_count is not None else len(paths)
    # Preserve canonical keys from being overwritten by extras
    safe_extras = {k: v for k, v in extras.items() if k not in ("changed_paths", "refreshed_count")}
    payload = {"changed_paths": paths, "refreshed_count": count, **safe_extras}
    _emit("cards.refresh.start", payload, session_id=session_id)


def cards_refresh_done(changed_paths: Iterable[str], *, refreshed_count: Optional[int] = None, session_id: Optional[str] = None, **extras: Any) -> None:
    """Emit 'cards.refresh.done' with payload {changed_paths, refreshed_count}.

    Mirrors cards_refresh_start; use this at the end of the refresh operation.
    Extras are merged conservatively and will not overwrite the canonical
    changed_paths/refreshed_count keys.

    Canonical payload keys: 'changed_paths' (List[str]) and 'refreshed_count' (int).
    These keys indicate which files had their cards updated and how many were refreshed.
    """
    paths = [str(p) for p in list(changed_paths or [])]
    count = int(refreshed_count) if refreshed_count is not None else len(paths)
    safe_extras = {k: v for k, v in extras.items() if k not in ("changed_paths", "refreshed_count")}
    payload = {"changed_paths": paths, "refreshed_count": count, **safe_extras}
    _emit("cards.refresh.done", payload, session_id=session_id)


def emit_result(
  ok: bool,
  *,
  summary: Optional[str] = None,
  artifacts: Optional[Iterable[str]] = None,
  next_steps: Optional[List[str]] = None,
  where: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """Terminal result for the run (success/failure + short summary)."""
    payload = {
        "ok": bool(ok),
        "summary": summary,
        "artifacts": list(artifacts or []),
        "next_steps": next_steps or [],
        "where": where,
        **extras,
    }
    _emit("result", payload, session_id=session_id)


def emit_error(
  message: str,
  *,
  code: Optional[str] = None,
  where: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Non-fatal error (LLM/IO/etc.). Prefer rich messages over tracebacks.

    We expose the text under BOTH `message` and `error` so UIs that expect
    `payload.error` won't fall back to "unknown error".
    """
    payload = {
        "message": message,
        "msg": message,     # legacy alias kept inside payload only
        "error": message,   # explicit error field for SSE consumers
        "code": code,
        "where": where,
        **extras,
    }
    _emit("error", payload, session_id=session_id)


# ---------- New helpers for Analyze + Q&A modes ----------


def analyze_result(
  focus: str,
  analysis: str,
  cards: List[Dict[str, Any]],
  *,
  plan: Any = None,
  session_id: Optional[str] = None,
  job_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Structured result for analyze mode.
    type: "analysis_result"
    payload: { focus, analysis, cards, plan, job_id, ... }
    """
    payload = {
        "focus": focus,
        "analysis": analysis,
        "cards": list(cards or []),
        "plan": plan,
        "job_id": job_id,
        **extras,
    }
    _emit("analysis_result", payload, session_id=session_id)


def emit_qa_answer(event_payload: Dict[str, Any], *, session_id: Optional[str] = None, model: Optional[str] = None, **extras: Any) -> None:
    """
    Emit a canonical 'qa_answer' event from a pre-built payload.

    (DEPRECATED signature retained for backward compatibility)
    This version accepts a single positional dict (event_payload) and behaves
    like the newer signature with event_type 'qa_answer'. It shallow-copies
    the provided payload and merges extras without overwriting existing keys.

    When `model` is provided it will be attached to payload.model in the emitted event.

    See the newer emit_qa_answer signature below for the preferred usage.
    """
    # Back-compat: call the new implementation with explicit event_type
    # Note: keep behavior: do not allow extras to overwrite existing payload keys
    payload = dict(event_payload or {})
    # Explicit model param should take precedence over extras/payload content
    if model is not None:
        payload["model"] = model
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v

    # Conservative normalization: ensure question/answer keys exist and are strings
    q = payload.get("question")
    a = payload.get("answer")
    payload["question"] = "" if q is None else str(q)
    payload["answer"] = "" if a is None else str(a)

    _emit("qa_answer", payload, session_id=session_id)


def emit_qa_answer_new(
  event_type_or_payload: Any,
  payload: Optional[Dict[str, Any]] = None,
  *,
  session_id: Optional[str] = None,
  job_id: Optional[str] = None,
  model: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Newer, flexible signature for emitting Q&A-like events.

    Signature:
      emit_qa_answer(event_type: str, payload: Dict[str, Any], *, session_id=None, job_id=None, model=None, **extras)

    Backward compatibility: callers may still call emit_qa_answer(payload_dict, session_id=...)
    — this file exposes the legacy emit_qa_answer wrapper above that delegates to the canonical event type 'qa_answer'.

    Behavior:
      - If called with a dict as the first arg and no second arg, treats it as the legacy payload and emits type 'qa_answer'.
      - Otherwise treats the first arg as the event_type string and the second as payload.
      - Shallow-copies the payload, merges extras only for keys that do NOT already exist (to avoid unintentionally overwriting normalized values), and preserves any payload-provided job_id over the job_id argument.
      - Ensures 'question' and 'answer' exist as strings on the final payload.
      - Emits using the provided event_type (do not hardcode 'qa_answer').
      - When `model` is provided it will be attached to payload.model and will override any
        `model` provided via extras or the payload itself.
    """
    # Detect legacy single-arg form: emit_qa_answer(payload_dict, session_id=...)
    if payload is None and isinstance(event_type_or_payload, dict):
        event_type = "qa_answer"
        final_payload = dict(event_type_or_payload or {})
    else:
        event_type = str(event_type_or_payload)
        final_payload = dict(payload or {})

    # Attach explicit model param (explicit param wins)
    if model is not None:
        final_payload["model"] = model

    # Merge extras but do NOT overwrite existing payload keys
    for k, v in extras.items():
        if k not in final_payload:
            final_payload[k] = v

    # Preserve job_id provided in argument only if payload doesn't already have it
    if job_id is not None and "job_id" not in final_payload:
        final_payload["job_id"] = job_id

    # Conservative normalization: ensure question/answer keys exist and are strings
    q = final_payload.get("question")
    a = final_payload.get("answer")
    final_payload["question"] = "" if q is None else str(q)
    final_payload["answer"] = "" if a is None else str(a)

    _emit(event_type, final_payload, session_id=session_id)

# Keep a stable reference to the legacy implementation
_emit_qa_answer_legacy = emit_qa_answer


def emit_qa_answer_dispatcher(
  event_type_or_payload: Any,
  payload: Optional[Dict[str, Any]] = None,
  *,
  session_id: Optional[str] = None,
  job_id: Optional[str] = None,
  model: Optional[str] = None,
  **extras: Any,
) -> None:
    # Legacy single-arg dict form
    if payload is None and isinstance(event_type_or_payload, dict):
        # forward model and extras to legacy implementation which accepts model param
        _emit_qa_answer_legacy(event_type_or_payload, session_id=session_id, model=model, **extras)
        return

    # New form
    emit_qa_answer_new(
        event_type_or_payload,
        payload,
        session_id=session_id,
        job_id=job_id,
        model=model,
        **extras,
    )

# Rebind the public name
emit_qa_answer = emit_qa_answer_dispatcher


def qa_answer(
  question: str,
  answer: str,
  *,
  session_id: Optional[str] = None,
  job_id: Optional[str] = None,
  model: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Structured result for Q&A mode (thin backward-compatible wrapper).

    This function remains as a convenience wrapper for callers that supply
    question/answer strings; it delegates to emit_qa_answer which performs
    conservative normalization and emits the canonical 'qa_answer' event. If
    `model` is provided it will be attached to payload.model.
    """
    payload = {"question": question, "answer": answer, "job_id": job_id, **extras}
    emit_qa_answer("qa_answer", payload, session_id=session_id, model=model)


# ---------- New helpers for chat intent + mode events ----------


def chat_intent_detected(
  intent: str,
  *,
  confidence: float,
  rationale: str = "",
  slots: Optional[Dict[str, Any]] = None,
  session_id: Optional[str] = None,
  job_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    High-level event telling the UI what intent was inferred for this chat turn.

    type: "chat_intent_detected"
    payload: {
      intent: <str>,             # e.g. "MAKE_RECOMMENDATIONS"
      confidence: <float 0-1>,
      rationale: <str>,
      slots: { ... },            # optional structured slots
      job_id: <str> | null,
      ...
    }
    """
    payload = {
        "intent": intent,
        "confidence": confidence,
        "rationale": rationale,
        "slots": dict(slots or {}),
        "job_id": job_id,
        **extras,
    }
    _emit("chat_intent_detected", payload, session_id=session_id)


def chat_mode_chosen(
  mode: str,
  *,
  reason: str = "",
  intent: Optional[str] = None,
  confidence: Optional[float] = None,
  session_id: Optional[str] = None,
  job_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    High-level event telling the UI which mode ("Q&A" | "analyze" | "edit")
    was chosen for this chat turn and why.

    type: "chat_mode_chosen"
    payload: {
      mode: <str>,
      reason: <str>,
      intent: <str> | null,
      confidence: <float> | null,
      job_id: <str> | null,
      ...
    }
    """
    payload = {
        "mode": mode,
        "reason": reason,
        "intent": intent,
        "confidence": confidence,
        "job_id": job_id,
        **extras,
    }
    _emit("chat_mode_chosen", payload, session_id=session_id)


def emit_mode_choice_event(
  *args,
  event_payload: Any = None,
  session_id: Optional[str] = None,
  job_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Emit a standardized 'mode_choice' event that the UI can consume to show
    a small inline chip (e.g. "Auto chose: Q&A").

    Supported call forms:
      - emit_mode_choice_event(payload)
      - emit_mode_choice_event(session_id, payload)
      - emit_mode_choice_event(event_payload=payload, session_id=session_id)

    Behavior:
      - Accepts a dict-like payload or other values; robust to malformed inputs.
      - If called with positional args, this function parses them as described above
        (len(args) == 1 => payload; len(args) >= 2 => session_id=args[0], payload=args[1]).
      - Shallow-copies the provided payload; merges extras only for keys that do not
        already exist (to avoid overwriting caller-provided values).
      - Coerces confidence to float when possible, otherwise sets to 0.0.
      - Defaults: mode -> 'unknown' if missing, reason -> '', source -> 'auto'.
      - Preserves a caller-provided job_id/session_id; session_id is attached to the
        _emit call rather than injected into the payload unless the caller provided it.
      - Emits via the canonical internal emitter so overlay/backcompat logic is applied.
    """
    # Positional-arg parsing to support (payload), (session_id, payload)
    final_payload: Dict[str, Any]
    
    if len(args) == 0:
        # No positional args: prefer explicit event_payload kwarg
        if isinstance(event_payload, dict):
            final_payload = dict(event_payload)
        elif event_payload is None:
            final_payload = {}
        else:
            try:
                final_payload = dict(event_payload)
            except Exception:
                final_payload = {"mode": str(event_payload)}
    elif len(args) == 1:
        # Single positional arg -> treated as payload
        p = args[0]
        if isinstance(p, dict):
            final_payload = dict(p)
        else:
            try:
                final_payload = dict(p)
            except Exception:
                final_payload = {"mode": str(p)}
    else:
        # Two-or-more positional args: (session_id, payload, ...)
        session_id = args[0]
        p = args[1]
        if isinstance(p, dict):
            final_payload = dict(p)
        else:
            try:
                final_payload = dict(p)
            except Exception:
                final_payload = {"mode": str(p)}

    # Merge extras but do NOT overwrite existing payload keys
    for k, v in extras.items():
        if k not in final_payload:
            final_payload[k] = v

    # Preserve job_id provided in argument only if payload doesn't already have it
    if job_id is not None and "job_id" not in final_payload:
        final_payload["job_id"] = job_id

    # Ensure canonical keys with safe coercion
    mode_val = final_payload.get("mode")
    final_payload["mode"] = "unknown" if mode_val is None else str(mode_val)

    conf = final_payload.get("confidence")
    if conf is None:
        final_payload["confidence"] = 0.0
    else:
        try:
            final_payload["confidence"] = float(conf)
        except Exception:
            final_payload["confidence"] = 0.0

    reason_val = final_payload.get("reason")
    final_payload["reason"] = "" if reason_val is None else str(reason_val)

    # Default source is 'auto' when absent
    final_payload.setdefault("source", "auto")

    _emit("mode_choice", final_payload, session_id=session_id)


# Convenience alias
mode_choice = emit_mode_choice_event


def chat_message(
  role: str,
  content: str,
  *,
  session_id: Optional[str] = None,
  job_id: Optional[str] = None,
  model: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Structured chat message event (optional; in addition to 'assistant' / user logs).

    type: "chat_message"
    payload: { role: "user" | "assistant", content: <str>, job_id: <str> | null, ... }

    When `model` is provided it will be attached to payload.model in the emitted event.
    """
    payload: Dict[str, Any] = {"role": role, "content": content}
    if job_id is not None:
        payload["job_id"] = job_id
    if model is not None:
        payload["model"] = model
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v
    _emit("chat_message", payload, session_id=session_id)


# ---------- Project-create lifecycle emit helpers (reworked signatures) ----------

def emit_project_create_questions(session_id: Optional[str], payload: Optional[Dict[str, Any]] = None, **extras: Any) -> None:
    """
    Emit 'project_create.questions' with payload shaped as:
      { "questions": [ {question, id?, type?, choices?, ...} ], "recId": <str>, "model": <str?>, ... }

    Signature intentionally accepts (session_id, payload=None, **extras).
    If callers pass a list as payload it will be treated as the questions array.
    Extras are merged conservatively (do not overwrite explicit payload keys).
    """
    # Normalize payload to dict. Allow payload to be a list -> treat as questions.
    if isinstance(payload, list):
        payload_dict: Dict[str, Any] = {"questions": [dict(q) if isinstance(q, dict) else {"question": q} for q in payload]}
    elif isinstance(payload, dict):
        payload_dict = dict(payload)
    else:
        payload_dict = {}

    # Merge extras conservatively
    for k, v in extras.items():
        if k not in payload_dict:
            payload_dict[k] = v

    # Ensure questions is an array of objects
    qs = payload_dict.get("questions") or []
    normalized_qs: List[Dict[str, Any]] = []
    for q in qs:
        if isinstance(q, dict):
            normalized_qs.append(dict(q))
        else:
            normalized_qs.append({"question": q})
    payload_dict["questions"] = normalized_qs

    # session_id and recId handling
    if session_id is not None:
        payload_dict.setdefault("session_id", session_id)
    rec = payload_dict.get("recId") or extras.get("recId")
    payload_dict["recId"] = _ensure_rec_id(rec)

    _emit("project_create.questions", payload_dict, session_id=session_id)


def emit_project_create_scaffold(session_id: Optional[str], payload: Optional[Dict[str, Any]] = None, **extras: Any) -> None:
    """
    Emit 'project_create.scaffold' with payload shaped as:
      { "created_files": [ {path, summary?, size?, sha?}, ... ], "preview": <str?>, "summary": <str?>, "recId": <str>, ... }

    Signature intentionally accepts (session_id, payload=None, **extras).
    If callers pass a list as payload it will be treated as the created_files array.
    Extras are merged conservatively (do not overwrite explicit payload keys).
    """
    # Normalize payload to dict. Allow payload to be a list -> treat as created_files.
    if isinstance(payload, list):
        payload_dict: Dict[str, Any] = {"created_files": [dict(f) if isinstance(f, dict) else {"path": str(f)} for f in payload]}
    elif isinstance(payload, dict):
        payload_dict = dict(payload)
    else:
        payload_dict = {}

    # Merge extras conservatively
    for k, v in extras.items():
        if k not in payload_dict:
            payload_dict[k] = v

    # Ensure created_files is a list of descriptors
    files = payload_dict.get("created_files") or []
    normalized_files: List[Dict[str, Any]] = []
    for f in files:
        if isinstance(f, dict):
            # keep only expected keys but allow extras
            fd = dict(f)
            if "path" in fd:
                fd["path"] = str(fd["path"])
            normalized_files.append(fd)
        else:
            normalized_files.append({"path": str(f)})
    payload_dict["created_files"] = normalized_files

    # session_id and recId handling
    if session_id is not None:
        payload_dict.setdefault("session_id", session_id)
    rec = payload_dict.get("recId") or extras.get("recId")
    payload_dict["recId"] = _ensure_rec_id(rec)

    _emit("project_create.scaffold", payload_dict, session_id=session_id)

def emit_project_create_applied(session_id: Optional[str], payload: Optional[Dict[str, Any]] = None, **extras: Any) -> None:
    """
    Emit 'project_create.applied' with payload shaped as:
      { "result": { ... , "ok": <bool> }, "applied_paths": [<str>], "recId": <str>, ... }

    Signature intentionally accepts (session_id, payload=None, **extras).
    Extras are merged conservatively (do not overwrite explicit payload keys).
    Ensures that payload['result'] exists and contains at least an 'ok' boolean.
    """
    if isinstance(payload, dict):
        payload_dict = dict(payload)
    else:
        payload_dict = {}

    # Merge extras conservatively
    for k, v in extras.items():
        if k not in payload_dict:
            payload_dict[k] = v

    # Normalize result: must be a dict and include at least 'ok'
    res = dict(payload_dict.get("result") or {})
    # prefer explicit top-level ok if provided, then fallback to True
    if "ok" not in res:
        res["ok"] = bool(payload_dict.get("ok", True))
    payload_dict["result"] = res

    # Normalize applied_paths if present
    if "applied_paths" in payload_dict:
        payload_dict["applied_paths"] = [str(p) for p in (payload_dict.get("applied_paths") or [])]

    # session_id and recId handling
    if session_id is not None:
        payload_dict.setdefault("session_id", session_id)
    rec = payload_dict.get("recId") or extras.get("recId")
    payload_dict["recId"] = _ensure_rec_id(rec)

    _emit("project_create.applied", payload_dict, session_id=session_id)


# ---------- New trace events for baselines, overlays, and check inputs ----------

def trace_baseline(
    baseline: Any,
    *,
    session_id: Optional[str] = None,
    job_id: Optional[str] = None,
    recId: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a trace event describing the baseline data used for a run.

    Event type: 'trace.baseline'
    Payload contains a shallow copy of the provided baseline (if dict) or
    wraps the provided value under the 'baseline' key. Adds job_id and recId
    metadata when provided. Extras are merged conservatively.
    """
    if isinstance(baseline, dict):
        payload: Dict[str, Any] = dict(baseline)
    else:
        payload = {"baseline": baseline}

    if job_id is not None:
        payload.setdefault("job_id", job_id)

    payload["recId"] = _ensure_rec_id(recId)

    for k, v in extras.items():
        if k not in payload:
            payload[k] = v

    _emit("trace.baseline", payload, session_id=session_id)


def trace_overlay(
    overlay: Any,
    *,
    session_id: Optional[str] = None,
    job_id: Optional[str] = None,
    recId: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a trace event describing overlays applied or considered during a run.

    Event type: 'trace.overlay'
    Payload contains a shallow copy of the provided overlay (if dict) or
    wraps the provided value under the 'overlay' key. Adds job_id and recId
    metadata when provided. Extras are merged conservatively.
    """
    if isinstance(overlay, dict):
        payload: Dict[str, Any] = dict(overlay)
    else:
        payload = {"overlay": overlay}

    if job_id is not None:
        payload.setdefault("job_id", job_id)

    payload["recId"] = _ensure_rec_id(recId)

    for k, v in extras.items():
        if k not in payload:
            payload[k] = v

    _emit("trace.overlay", payload, session_id=session_id)


def trace_check_inputs(
    inputs: Any,
    *,
    session_id: Optional[str] = None,
    job_id: Optional[str] = None,
    recId: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a trace event capturing inputs fed into checks/validators.

    Event type: 'trace.check_inputs'
    Payload contains a shallow copy of the provided inputs (if dict) or
    wraps the provided value under the 'inputs' key. Adds job_id and recId
    metadata when provided. Extras are merged conservatively.
    """
    if isinstance(inputs, dict):
        payload: Dict[str, Any] = dict(inputs)
    else:
        payload = {"inputs": inputs}

    if job_id is not None:
        payload.setdefault("job_id", job_id)

    payload["recId"] = _ensure_rec_id(recId)

    for k, v in extras.items():
        if k not in payload:
            payload[k] = v

    _emit("trace.check_inputs", payload, session_id=session_id)


def trace_routing(
    decision: str,
    *,
    baseline_id: Optional[str] = None,
    overlay_summary: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    session_id: Optional[str] = None,
    job_id: Optional[str] = None,
    recId: Optional[str] = None,
    **extras: Any,
) -> None:
    """Emit a routing decision trace event describing why a run chose edit vs repair.

    Event type: 'trace.routing'

    Parameters:
      decision: str - routing decision identifier (e.g. 'edit', 'repair', 'preapply-check')
      baseline_id: optional str - identifier for the baseline used to make the decision
      overlay_summary: optional dict - summary of overlays considered/applied (paths, hashes, counts)
      reason: optional str - short human-readable reason for the decision
      session_id/job_id/recId: optional metadata attached to the payload
      **extras: merged conservatively (do not overwrite canonical keys)

    The payload will include at minimum the keys { 'decision', 'reason', 'baseline_id', 'overlay_summary', 'recId' }.
    Extras are merged only when they don't clobber those canonical keys. Do not include raw secrets in overlay_summary; callers should redact before passing, and we rely on downstream redactors when recording traces.
    """
    payload: Dict[str, Any] = {"decision": decision}
    if reason is not None:
        payload["reason"] = reason
    if baseline_id is not None:
        payload["baseline_id"] = baseline_id
    if overlay_summary is not None:
        # shallow-copy overlay_summary when it's a dict to avoid accidental mutation
        payload["overlay_summary"] = dict(overlay_summary) if isinstance(overlay_summary, dict) else overlay_summary

    if job_id is not None:
        payload.setdefault("job_id", job_id)

    payload["recId"] = _ensure_rec_id(recId)

    # Merge extras conservatively
    for k, v in extras.items():
        if k not in payload:
            payload[k] = v

    _emit("trace.routing", payload, session_id=session_id)

# ---------- End trace events ----------


# Aliases used throughout the codebase
status = emit_status
token = emit_token
plan = emit_plan
diff = emit_diff
result = emit_result
error = emit_error


def done(
  ok: bool = True,
  *,
  where: Optional[str] = None,
  session_id: Optional[str] = None,
  **extras: Any,
) -> None:
    """
    Terminal 'done'. If where omitted, use last remembered stage from this session.
    """
    if where is None and session_id:
        st = _last_topic.get(session_id)
        if st:
            where = st.last_stage or st.last_where
    payload = {"ok": bool(ok), "where": where, **extras}
    _emit("done", payload, session_id=session_id)
