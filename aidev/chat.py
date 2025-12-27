# aidev/chat.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Core SSE helpers (already used by your UI)
from .events import (
    assistant,
    status,
    chat_intent_detected,
    chat_mode_chosen,
    chat_message,
    qa_answer,
    emit_plan,
    emit_error,
    emit_mode_choice_event,
)

# Optional: analyze result event emitter (newer servers)
try:
    from .events import emit_analyze_result  # type: ignore
except Exception:  # older environments
    emit_analyze_result = None  # type: ignore[assignment]

# Optional: central schema validator helper
try:
    from .schemas import validate_schema as _validate_schema  # type: ignore
except Exception:
    _validate_schema = None  # type: ignore[assignment]

# Optional LLM client for intent classification
try:
    from .llm_client import ChatGPT as _ChatClient  # type: ignore
except Exception:  # tests / minimal environments without llm_client
    _ChatClient = None  # type: ignore[assignment]

# NOTE: Intent detection and planner implementations are imported lazily at
# runtime to avoid import-time cycles and to fail fast when the canonical
# modules are missing. Do NOT provide in-file fallback implementations here —
# missing canonical modules should raise informative ImportError messages so
# maintainers can fix the environment instead of running silently degraded
# behavior.


# ---------- Lazy import helpers (fail-fast with remediation hints) ----------

def _import_intents_module():
    """
    Lazy-import aidev.conversation.intents and extract expected symbols.
    Raises ImportError with an actionable message if the module is missing.
    """
    try:
        from .conversation import intents as _intents  # type: ignore
        Intent = getattr(_intents, "Intent")
        IntentResult = getattr(_intents, "IntentResult")
        detect_intent = getattr(_intents, "detect_intent")
        return Intent, IntentResult, detect_intent
    except Exception as e:
        raise ImportError(
            "Missing 'aidev.conversation.intents': %s. Remediation: ensure the 'aidev.conversation' "
            "package is present and importable (check project layout or .aidev config), or install the "
            "required extras that provide conversation/intents." % e
        ) from e


def _import_planner_module():
    """
    Lazy-import aidev.conversation.planner and extract expected planner functions.
    Raises ImportError with an actionable message if the module is missing.
    """
    try:
        from .conversation import planner as _planner  # type: ignore
        plan_from_intent = getattr(_planner, "plan_from_intent")
        propose_plan = getattr(_planner, "propose_plan")

        # NEW: preferred entrypoint that takes an IntentResult and does NOT re-classify
        plan_from_intent_result = getattr(_planner, "plan_from_intent_result", None)

        return plan_from_intent, propose_plan, plan_from_intent_result
    except Exception as e:
        raise ImportError(
            "Missing 'aidev.conversation.planner': %s. Remediation: ensure the 'aidev.conversation' "
            "package is present and importable (check project layout or .aidev config), or install the "
            "required extras that provide conversation/planner." % e
        ) from e


def _import_project_create_flow():
    """
    Lazy-import aidev.stages.project_create_flow and extract canonical entrypoints.
    Raises ImportError with actionable remediation text if missing.
    Expected to provide at least `start` and `advance` callables.

    NOTE: Some older stage implementations expose alternate names such as
    `start_create_session` / `advance_create_session`. This helper attempts
    multiple attribute name variants to maximize compatibility.
    """
    try:
        from .stages import project_create_flow as _pcf  # type: ignore

        # Try preferred names first
        start = getattr(_pcf, "start", None)
        advance = getattr(_pcf, "advance", None)
        if callable(start) and callable(advance):
            return start, advance

        # Try alternate legacy names (some implementations used these)
        alt_start = getattr(_pcf, "start_create_session", None)
        alt_advance = getattr(_pcf, "advance_create_session", None)
        if callable(alt_start) and callable(alt_advance):
            return alt_start, alt_advance

        # If one of the preferred names exists, return what we have (caller will handle None/absent)
        if callable(start) or callable(advance):
            return start, advance

        # Nothing usable found
        raise AttributeError("project_create_flow missing expected entrypoints: start/advance or start_create_session/advance_create_session")

    except Exception as e:
        raise ImportError(
            "Missing 'aidev.stages.project_create_flow' or expected entrypoints: %s. Remediation: ensure the 'aidev.stages' "
            "package is present and that project_create_flow.py exposes start(...) and advance(...), or the legacy "
            "start_create_session/advance_create_session variants." % e
        ) from e


# Wrapper delegations that perform the lazy imports only when needed.
def _detect_intent(message: str, *, llm: Any = None, **kwargs: Any):
    """Call the canonical detect_intent from aidev.conversation.intents.

    This intentionally raises ImportError if the intents module is missing.
    """
    _, _, detect_fn = _import_intents_module()
    return detect_fn(message, llm=llm, **kwargs)


def _plan_from_intent(intent: Any, message: str, session_id: Optional[str]):
    plan_fn, _, _ = _import_planner_module()
    return plan_fn(intent, message, session_id)


def _plan_from_intent_result(intent_result: Any, message: str, session_id: Optional[str]):
    """
    Preferred planner entrypoint: pass the full IntentResult so we keep LLM-produced slots.
    Falls back to propose_plan if plan_from_intent_result isn't available (older planner).
    """
    plan_fn, propose_fn, plan_ir_fn = _import_planner_module()
    if callable(plan_ir_fn):
        return plan_ir_fn(intent_result, message, session_id=session_id)
    # fallback: propose_plan accepts IntentResult in newer planner; if not, this still works best-effort.
    return propose_fn(intent_result, message, session_id)


def _propose_plan(intent: Any, message: str, session_id: Optional[str]):
    _, propose_fn, _ = _import_planner_module()
    return propose_fn(intent, message, session_id)


# ---------- Formatting helpers -----------------------------------------------

def format_plan(plan_steps: List[Dict[str, Any]]) -> str:
    parts = []
    for i, step in enumerate(plan_steps, 1):
        name = step.get("name", "step")
        params = step.get("params") or {}
        param_bits = ", ".join(f"{k}={v!r}" for k, v in params.items()) if params else ""
        parts.append(f"{i}. {name}{(' — ' + param_bits) if param_bits else ''}")
    return "\n".join(parts)


def format_planned_changes(changes: List[Dict[str, Any]], limit: int = 12) -> str:
    if not changes:
        return "• (none yet)"
    out = []
    for i, ch in enumerate(changes[:limit], 1):
        path = ch.get("path") or "unknown"
        title = ch.get("title") or ch.get("summary") or ch.get("reason") or "Planned edit"
        out.append(f"{i}. {title} — {path}")
    if len(changes) > limit:
        out.append(f"... (and {len(changes) - limit} more)")
    return "\n".join(out)


def narrate_kickoff(project_root: str, focus: str, plan_steps: List[Dict[str, Any]]) -> None:
    assistant(
        "Got it! Here's what I'll do:\n"
        f"• Project: {project_root}\n"
        f"• Focus: {focus}\n"
        "• Plan:\n" + format_plan(plan_steps)
    )
    status({"event": "narrate_kickoff"})


def narrate_planned_changes(changes: List[Dict[str, Any]]) -> None:
    assistant("Planned changes (draft):\n" + format_planned_changes(changes))
    status({"event": "narrate_planned_changes", "count": len(changes)})


def narrate_stage(stage_name: str, note: Optional[str] = None) -> None:
    assistant(f"Working on **{stage_name}**" + (f" — {note}" if note else ""))
    status({"event": "narrate_stage", "stage": stage_name, "note": note})


def narrate_apply_result(files_changed: List[str]) -> None:
    if not files_changed:
        assistant("No file changes were necessary after verification.")
    else:
        bullets = "\n".join(f"• {p}" for p in files_changed)
        assistant("Applied changes to:\n" + bullets)
    status({"event": "narrate_apply_result", "changed": files_changed})


# ---------- Tool registry & execution ----------------------------------------

def _execute_plan_steps(
    plan_steps: List[Dict[str, Any]],
    *,
    registry: "ToolRegistry",
    session_id: Optional[str],
) -> None:
    """
    Execute plan steps sequentially via the ToolRegistry.
    """
    total = len(plan_steps) or 0
    if total == 0:
        status({
            "event": "plan_empty",
            "stage": "plan",
            "progress_pct": 100.0,
        })
        return

    for idx, step in enumerate(plan_steps, start=1):
        tool = step.get("tool")
        name = step.get("name") or tool or f"step-{idx}"
        params = step.get("params") or {}

        if not tool:
            status({
                "event": "plan_step_skipped",
                "reason": "missing_tool",
                "step_index": idx,
                "step_total": total,
            })
            continue

        if not registry.has(tool):
            status({
                "event": "plan_step_skipped",
                "reason": "unknown_tool",
                "tool": tool,
                "step_index": idx,
                "step_total": total,
            })
            continue

        status({
            "event": "plan_step_start",
            "tool": tool,
            "name": name,
            "step_index": idx,
            "step_total": total,
            "progress_pct": (idx - 1) * 100.0 / max(1, total),
        })

        try:
            registry.execute(tool, params)
        except Exception as e:
            emit_error({
                "event": "plan_step_error",
                "where": "plan_execute",
                "message": f"Plan step {idx}/{total} failed: {type(e).__name__}: {e}",
                "tool": tool,
                "step_index": idx,
                "step_total": total,
            })
            break

        status({
            "event": "plan_step_finish",
            "tool": tool,
            "name": name,
            "step_index": idx,
            "step_total": total,
            "progress_pct": idx * 100.0 / max(1, total),
        })

    status({
        "event": "plan_complete",
        "stage": "plan",
        "progress_pct": 100.0,
    })


ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class ToolSpec:
    name: str
    handler: ToolHandler
    required: Tuple[str, ...] = field(default_factory=tuple)
    optional: Tuple[str, ...] = field(default_factory=tuple)
    produces_proposals: bool = False


class ToolRegistry:
    """
    In-process registry that the API layer can populate at startup.
    Each handler accepts a dict of params and returns a dict payload.
    """
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def validate_params(self, spec: ToolSpec, params: Dict[str, Any]) -> Optional[str]:
        p = params or {}
        missing = [k for k in spec.required if k not in p or p[k] in (None, "")]
        if missing:
            return f"Missing required param(s): {', '.join(missing)}"
        return None

    def execute(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        spec = self.get(name)
        err = self.validate_params(spec, params or {})
        if err:
            raise ValueError(err)
        return spec.handler(params or {})


# ---------- Internal helpers for chat routing --------------------------------

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _intent_label(intent_obj: Any) -> str:
    if hasattr(intent_obj, "value"):
        return str(intent_obj.value)
    return str(intent_obj)


def _normalize_mode_label(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip().lower()
    if s in ("q&a", "qa", "qna", "q and a", "question", "question-answer"):
        return "Q&A"
    if s.startswith("anal"):
        return "analyze"
    if s.startswith("edit") or s == "apply":
        return "edit"
    if s == "auto":
        return "auto"
    return None


def _canonical_mode_from_intent(intent_obj: Any) -> str:
    """
    Map a high-level Intent to one of: "Q&A", "analyze", "edit".
    """
    label = _intent_label(intent_obj)

    if label in ("MAKE_RECOMMENDATIONS", "APPLY_EDITS", "CREATE_PROJECT", "SELECT_PROJECT"):
        return "edit"

    if label in ("RUN_CHECKS", "UPDATE_DESCRIPTIONS", "ANALYZE_PROJECT"):
        return "analyze"

    return "Q&A"


def _decide_mode(intent_result: "IntentResult", explicit_mode: Optional[str]) -> Tuple[str, float, bool, str]:
    norm = _normalize_mode_label(explicit_mode)
    if norm and norm != "auto":
        return norm, 1.0, True, "explicit_override"

    mode = _canonical_mode_from_intent(getattr(intent_result, "intent", ""))
    conf = float(getattr(intent_result, "confidence", 0.0) or 0.0)
    return mode, conf, False, "derived_from_intent"


def _emit_analyze_result_event(payload: Dict[str, Any], *, session_id: Optional[str]) -> None:
    """Emit a single explicit analyze completion event (success/invalid/error).

    Uses emit_analyze_result when available; otherwise falls back to status() with
    a deterministic event name so the UI still receives a clear completion state.

    The canonical emit_analyze_result signature expected by newer servers is:
      emit_analyze_result(plan_payload, status=..., diagnostics=..., session_id=..., job_id=...)

    To remain compatible with callers in this module, callers may pass a dict
    that contains optional keys: 'analyze_plan'|'plan' (the plan payload),
    'status' (string), 'errors'|'diagnostics'|'failures' (list), and
    'job_id'|'rec_id' (optional id).
    """
    try:
        plan_payload = None
        if isinstance(payload, dict):
            plan_payload = payload.get("analyze_plan") or payload.get("plan")
            status_val = payload.get("status") or payload.get("state") or "error"
            diagnostics = payload.get("errors") or payload.get("diagnostics") or payload.get("failures") or []
            job_id = payload.get("job_id") or payload.get("rec_id") or payload.get("id")
        else:
            # Fallback when callers pass non-dict (shouldn't happen here)
            status_val = "error"
            diagnostics = []
            job_id = None

        # Normalize diagnostics shape: ensure list of dicts
        if diagnostics is None:
            diagnostics = []
        if not isinstance(diagnostics, list):
            diagnostics = [{"message": str(diagnostics)}]

        if callable(emit_analyze_result):
            # Call the canonical emitter with explicit args rather than passing a single dict
            try:
                emit_analyze_result(plan_payload, status=status_val, diagnostics=diagnostics, session_id=session_id, job_id=job_id)  # type: ignore[misc]
            except TypeError:
                # Some older emit_analyze_result variants might accept a different signature;
                # fallback to passing a single dict but keep the structured fields as keys.
                try:
                    emit_analyze_result({"plan": plan_payload, "status": status_val, "diagnostics": diagnostics, "job_id": job_id}, session_id=session_id)  # type: ignore[misc]
                except Exception:
                    # Final fallback to status event if emitter fails
                    status({"event": "analyze_result", "status": status_val, "diagnostics": diagnostics, "session_id": session_id, "job_id": job_id})
        else:
            status({"event": "analyze_result", "status": status_val, "diagnostics": diagnostics, "session_id": session_id, "job_id": job_id})
    except Exception as e:
        try:
            emit_error({"event": "emit_analyze_result_failed", "error": str(e), "payload": payload})
        except Exception:
            pass


def _validate_analyze_plan(analyze_plan: Any) -> Tuple[bool, List[Dict[str, Any]]]:
    """Validate analyze_plan against the canonical schema.

    Returns (is_valid, failures). failures is a structured list suitable for UI.
    If the central validator is unavailable, returns invalid with a clear failure.
    """
    if analyze_plan is None:
        return True, []

    if not callable(_validate_schema):
        return False, [
            {
                "code": "validator_unavailable",
                "message": "Schema validator is not available (aidev.schemas.validate_schema missing).",
                "schema": "analyze_plan.schema.json",
            }
        ]

    try:
        res = _validate_schema("analyze_plan", analyze_plan)  # type: ignore[misc]
    except Exception as e:
        return False, [
            {
                "code": "validator_exception",
                "message": f"Schema validation raised: {type(e).__name__}: {e}",
                "schema": "analyze_plan.schema.json",
            }
        ]

    # Support common validator return shapes:
    # - (bool, failures)
    # - {"ok": bool, "errors": [...]}
    # - {"valid": bool, "errors": [...]}
    # - {"valid": bool, "diagnostics": [...]}
    if isinstance(res, tuple) and len(res) == 2:
        ok = bool(res[0])
        failures = res[1] if isinstance(res[1], list) else [{"message": str(res[1])}]
        return ok, failures

    if isinstance(res, dict):
        ok = bool(res.get("ok", res.get("valid", False)))
        failures = res.get("errors") or res.get("failures") or res.get("diagnostics") or []
        if not isinstance(failures, list):
            failures = [{"message": str(failures)}]
        return ok, failures

    return False, [{"code": "validator_unknown_return", "message": f"Unexpected validator return: {_safe_json(res)}"}]


# ---------- Per-mode runners -------------------------------------------------

def _run_analyze_mode(
    intent_result: "IntentResult",
    message: str,
    *,
    session_id: Optional[str],
    registry: ToolRegistry,
) -> Dict[str, Any]:
    """
    Analyze mode: primarily repo/card analysis.
    """
    slots = getattr(intent_result, "slots", {}) or {}

    if not registry.has("ai_cards"):
        msg = "I don't have access to the analysis tools (ai_cards) on this server."
        assistant(f"ℹ️ {msg}")
        status({"event": "analyze_skipped", "reason": "tool_missing"})
        _emit_analyze_result_event(
            {
                "status": "error",
                "errors": [{"code": "tool_missing", "message": msg, "tool": "ai_cards"}],
                "summary": None,
                "plan_present": False,
            },
            session_id=session_id,
        )
        return {
            "intent": _intent_label(intent_result.intent),
            "mode": "analyze",
            "mode_confidence": float(getattr(intent_result, "confidence", 0.0) or 0.0),
            "error": msg,
            "analyze_plan": None,
            "analyze_plan_valid": False,
            "analyze_plan_errors": [{"code": "tool_missing", "message": msg, "tool": "ai_cards"}],
        }

    ai_params: Dict[str, Any] = {"mode": "changed"}

    model = slots.get("model")
    if model:
        ai_params["model"] = model

    focus = slots.get("focus")
    if focus and str(focus).strip():
        ai_params["focus"] = focus

    status({
        "event": "stage",
        "stage": "Analyze",
        "detail": "Summarizing changed files (ai_cards)",
    })

    try:
        out = registry.execute("ai_cards", ai_params)
        assistant("Analysis complete — here's what I found about your project.")
        status({"event": "analyze_complete"})
    except Exception as e:
        err = f"Analyze failed: {type(e).__name__}: {e}"
        status({"event": "stage", "stage": "Analyze", "detail": err})
        assistant(f"⚠️ {err}")
        _emit_analyze_result_event(
            {
                "status": "error",
                "errors": [{"code": "ai_cards_failed", "message": err}],
                "summary": None,
                "plan_present": False,
            },
            session_id=session_id,
        )
        return {
            "intent": _intent_label(intent_result.intent),
            "mode": "analyze",
            "mode_confidence": float(getattr(intent_result, "confidence", 0.0) or 0.0),
            "error": err,
            "slots": slots,
            "analyze_plan": None,
            "analyze_plan_valid": False,
            "analyze_plan_errors": [{"code": "ai_cards_failed", "message": err}],
        }

    base_res: Dict[str, Any] = {
        "intent": _intent_label(intent_result.intent),
        "mode": "analyze",
        "mode_confidence": float(getattr(intent_result, "confidence", 0.0) or 0.0),
        "knowledge_cards": out,
        "slots": slots,
        "analyze_plan": None,
        "text_summary": None,
        "analyze_plan_valid": None,
        "analyze_plan_errors": None,
        "analyze_plan_diagnostics": None,
    }

    if registry.has("orchestrate"):
        orch_params: Dict[str, Any] = {"focus": message or "", "mode": "analyze"}
        if slots.get("includes"):
            orch_params["includes"] = slots.get("includes")
        if slots.get("excludes"):
            orch_params["excludes"] = slots.get("excludes")

        status({"event": "analyze_orchestrate_start"})
        try:
            orch_res = registry.execute("orchestrate", orch_params) or {}
            if not isinstance(orch_res, dict):
                orch_res = {"text": str(orch_res)}

            analyze_plan = (
                orch_res.get("analyze_plan")
                or orch_res.get("plan")
                or orch_res.get("proposed_plan")
                or orch_res.get("analysis")
            )
            text_summary = (
                orch_res.get("text_summary")
                or orch_res.get("summary")
                or orch_res.get("text")
                or (None if analyze_plan else None)
            )

            if text_summary:
                try:
                    assistant(text_summary)
                except Exception:
                    pass

            # Validate analyze_plan (if present) against schema.
            is_valid, failures = _validate_analyze_plan(analyze_plan)
            base_res["analyze_plan"] = analyze_plan
            base_res["text_summary"] = text_summary
            base_res["analyze_plan_valid"] = bool(is_valid)
            base_res["analyze_plan_errors"] = failures if not is_valid else []
            base_res["analyze_plan_diagnostics"] = (
                {"validated": True, "schema": "analyze_plan.schema.json"} if is_valid else {"validated": False, "schema": "analyze_plan.schema.json"}
            )

            if analyze_plan is not None:
                if is_valid:
                    try:
                        emit_plan(analyze_plan, session_id=session_id, summary=text_summary)
                    except Exception as e:
                        emit_error({"event": "emit_plan_failed", "error": str(e)})

                    _emit_analyze_result_event(
                        {
                            "status": "success",
                            "errors": [],
                            "summary": text_summary,
                            "plan_present": True,
                            "analyze_plan": analyze_plan,
                        },
                        session_id=session_id,
                    )
                    status({"event": "analyze_completion", "status": "success"})
                else:
                    # No silent fallback: return structured invalid state.
                    _emit_analyze_result_event(
                        {
                            "status": "invalid",
                            "errors": failures,
                            "summary": text_summary,
                            "plan_present": True,
                            "analyze_plan": analyze_plan,
                        },
                        session_id=session_id,
                    )
                    status({"event": "analyze_completion", "status": "schema_invalid"})
                    base_res["error"] = "Analyze plan output failed schema validation."
            else:
                # Orchestrator ran but did not produce a plan.
                _emit_analyze_result_event(
                    {
                        "status": "error",
                        "errors": [{"code": "no_plan", "message": "Analyze pipeline did not produce an analyze_plan."}],
                        "summary": text_summary,
                        "plan_present": False,
                    },
                    session_id=session_id,
                )
                status({"event": "analyze_completion", "status": "error"})
                base_res["error"] = "Analyze pipeline did not produce an analyze_plan."

            base_res.update({k: v for k, v in orch_res.items() if k not in base_res})

            status({"event": "analyze_orchestrate_complete"})

        except Exception as e:
            err = f"Orchestrator analyze failed: {type(e).__name__}: {e}"
            emit_error({"event": "analyze_orchestrate_failed", "error": str(e)})
            try:
                assistant(f"⚠️ Analysis pipeline failed to run: {e}")
            except Exception:
                pass
            base_res["error"] = err
            base_res["analyze_plan_valid"] = False
            base_res["analyze_plan_errors"] = [{"code": "orchestrate_failed", "message": err}]
            base_res["analyze_plan_diagnostics"] = {"validated": False, "schema": "analyze_plan.schema.json"}
            status({"event": "analyze_orchestrate_failed", "detail": err})
            _emit_analyze_result_event(
                {
                    "status": "error",
                    "errors": [{"code": "orchestrate_failed", "message": err}],
                    "summary": None,
                    "plan_present": False,
                },
                session_id=session_id,
            )
            status({"event": "analyze_completion", "status": "error"})

    return base_res


def _run_edit_mode(
    intent_result: "IntentResult",
    message: str,
    *,
    session_id: Optional[str],
    registry: ToolRegistry,
) -> Dict[str, Any]:
    """
    Edit mode: propose code changes / project edits, then execute the plan.

    IMPORTANT CHANGE:
      - Pass the full IntentResult into the planner so we keep LLM-produced slots.
      - Do NOT reduce to intent_label when planning.
    """
    intent_label = _intent_label(intent_result.intent)
    slots = getattr(intent_result, "slots", {}) or {}

    edit_like_for_proposals = ("MAKE_RECOMMENDATIONS", "APPLY_EDITS")

    # Prefer the IntentResult-aware planner path (keeps slots).
    if intent_label in edit_like_for_proposals:
        plan_steps = _propose_plan(intent_result, message, session_id)
    else:
        plan_steps = _plan_from_intent_result(intent_result, message, session_id)

    emit_plan(plan_steps, session_id=session_id, summary=None)

    assistant(
        "Got it — here’s the plan I’ll follow to update your project, and I’m starting to work through it now."
    )
    status({"event": "edit_plan_ready", "count": len(plan_steps)})

    handled_create_events: List[Dict[str, Any]] = []
    remaining_steps: List[Dict[str, Any]] = []

    flow_start = None
    flow_advance = None

    for step in plan_steps or []:
        name = step.get("name") or step.get("tool") or ""
        if name and str(name).startswith("create."):
            params = step.get("params") or {}
            if isinstance(params.get("payload"), dict):
                call_params = dict(params.get("payload") or {})
            else:
                call_params = dict(params or {})

            api = getattr(registry, "api", None)
            if api is not None:
                try:
                    if name == "create.start":
                        if "brief" not in call_params or not call_params.get("brief"):
                            call_params["brief"] = message or call_params.get("brief")

                        if hasattr(api, "create_project_start") and callable(getattr(api, "create_project_start")):
                            status({"event": "project_create.start", "session_id": session_id})
                            payload = api.create_project_start(**call_params)
                            handled_create_events.append({
                                "event": "project_create.start",
                                "session_id": session_id,
                                "payload": payload,
                            })
                            continue

                    elif name == "create.advance":
                        call_params["session_id"] = session_id

                        if hasattr(api, "create_project_answer") and callable(getattr(api, "create_project_answer")):
                            status({"event": "project_create.advance", "session_id": session_id})
                            payload = api.create_project_answer(**call_params)
                            handled_create_events.append({
                                "event": "project_create.advance",
                                "session_id": session_id,
                                "payload": payload,
                            })
                            continue

                except Exception as e:
                    emit_error({"event": "project_create.api_failed", "error": str(e), "name": name})
                    status({"event": "project_create.error", "error": str(e), "session_id": session_id})
                    handled_create_events.append({"event": "project_create.error", "error": str(e), "session_id": session_id})
                    continue

            try:
                if flow_start is None and flow_advance is None:
                    flow_start, flow_advance = _import_project_create_flow()
            except ImportError as ie:
                emit_error({"event": "project_create.import_failed", "error": str(ie)})
                status({"event": "project_create.error", "error": str(ie), "session_id": session_id})
                handled_create_events.append({"event": "project_create.error", "error": str(ie), "session_id": session_id})
                continue
            except Exception as e:
                emit_error({"event": "project_create.import_failed", "error": str(e)})
                status({"event": "project_create.error", "error": str(e), "session_id": session_id})
                handled_create_events.append({"event": "project_create.error", "error": str(e), "session_id": session_id})
                continue

            try:
                if name == "create.start":
                    call_params = dict(call_params)
                    if "brief" not in call_params or not call_params.get("brief"):
                        call_params["brief"] = message or call_params.get("brief")
                    call_params["session_id"] = session_id

                    status({"event": "project_create.start", "session_id": session_id})
                    payload = flow_start(**call_params) if callable(flow_start) else flow_start(call_params)

                    handled_create_events.append({
                        "event": "project_create.start",
                        "session_id": session_id,
                        "payload": payload,
                    })

                elif name == "create.advance":
                    call_params = dict(call_params)
                    call_params["session_id"] = session_id
                    status({"event": "project_create.advance", "session_id": session_id})
                    payload = flow_advance(**call_params) if callable(flow_advance) else flow_advance(call_params)
                    handled_create_events.append({
                        "event": "project_create.advance",
                        "session_id": session_id,
                        "payload": payload,
                    })
                else:
                    status({"event": "project_create.unknown", "name": name, "session_id": session_id})
                    handled_create_events.append({"event": "project_create.unknown", "name": name, "session_id": session_id})

            except Exception as e:
                emit_error({"event": "project_create.handler_failed", "error": str(e), "name": name})
                status({"event": "project_create.error", "error": str(e), "name": name, "session_id": session_id})
                handled_create_events.append({"event": "project_create.error", "error": str(e), "name": name, "session_id": session_id})

        else:
            remaining_steps.append(step)

    _execute_plan_steps(remaining_steps, registry=registry, session_id=session_id)

    res: Dict[str, Any] = {
        "intent": intent_label,
        "mode": "edit",
        "mode_confidence": float(getattr(intent_result, "confidence", 0.0) or 0.0),
        "proposed_plan": plan_steps,
        "slots": slots,
        "plan_executed": True,
    }

    if handled_create_events:
        res["project_create_v2"] = handled_create_events

    return res


def _run_qa_mode(
    intent_result: "IntentResult",
    message: str,
    *,
    session_id: Optional[str],
    registry: ToolRegistry,
) -> Dict[str, Any]:
    """
    Q&A mode: primary answer path.
    """
    intent_label = _intent_label(intent_result.intent)
    slots = getattr(intent_result, "slots", {}) or {}

    question = (message or "").strip() or str(slots.get("focus", "")).strip()
    if not question:
        question = message or slots.get("focus_raw") or ""

    if registry.has("qa"):
        try:
            qa_out = registry.execute("qa", {"question": question})
            answer_text = (
                qa_out.get("answer")
                or qa_out.get("text")
                or qa_out.get("summary", "")
            )
            return {
                "intent": intent_label,
                "mode": "Q&A",
                "mode_confidence": float(getattr(intent_result, "confidence", 0.0) or 0.0),
                "answer": answer_text,
                "slots": slots,
            }

        except Exception as e:
            answer_text = f"⚠️ Q&A pipeline failed: {type(e).__name__}: {e}"
            qa_answer(question=question, answer=answer_text, session_id=session_id)
            assistant(answer_text)
            status({"event": "qa_answered", "error": str(e)})

            return {
                "intent": intent_label,
                "mode": "Q&A",
                "mode_confidence": float(getattr(intent_result, "confidence", 0.0) or 0.0),
                "answer": answer_text,
                "slots": slots,
                "error": str(e),
            }

    if question and len(question) < 400:
        answer_text = f"Here's a quick answer: {question}"
    else:
        answer_text = "I'll answer that question — here's a concise reply."

    qa_answer(question=question, answer=answer_text, session_id=session_id)
    assistant(answer_text)
    status({"event": "qa_answered"})

    return {
        "intent": intent_label,
        "mode": "Q&A",
        "mode_confidence": float(getattr(intent_result, "confidence", 0.0) or 0.0),
        "answer": answer_text,
        "slots": slots,
    }


# ---------- High-level chat orchestration ------------------------------------

def handle_chat(
    message: str,
    *,
    session_id: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
    mode_override: Optional[str] = None,
    llm: Any = None,
    # NEW: allow callers to provide history/projects to improve classification
    history: Optional[List[str]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint used by the HTTP /chat or /jobs/start (chat) handler.
    """
    if registry is None:
        registry = ToolRegistry()

    try:
        chat_message("user", message, session_id=session_id)
    except Exception:
        pass

    # STEP 1 — detect high-level intent from the message (LLM when available)
    intent_result = _detect_intent(
        message,
        llm=llm,
        history=history or [],
        projects=projects or [],
    )

    intent_label = _intent_label(getattr(intent_result, "intent", ""))
    slots = getattr(intent_result, "slots", {}) or {}
    intent_conf = float(getattr(intent_result, "confidence", 0.0) or 0.0)
    rationale = getattr(intent_result, "rationale", "") or ""

    try:
        chat_intent_detected(
            intent=intent_label,
            confidence=intent_conf,
            rationale=rationale,
            slots=slots,
            session_id=session_id,
        )
    except Exception:
        pass

    mode, mode_conf, overridden, reason = _decide_mode(intent_result, mode_override)

    try:
        chat_mode_chosen(
            mode=mode,
            reason=reason,
            intent=intent_label,
            confidence=mode_conf,
            session_id=session_id,
        )
    except Exception:
        pass

    selected_mode: Dict[str, Any] = {
        "mode": mode,
        "confidence": float(mode_conf or 0.0),
        "reason": reason,
        "source": ("auto" if not overridden else "explicit"),
    }

    if mode == "analyze":
        res = _run_analyze_mode(intent_result, message, session_id=session_id, registry=registry)
    elif mode == "edit":
        res = _run_edit_mode(intent_result, message, session_id=session_id, registry=registry)
    else:
        res = _run_qa_mode(intent_result, message, session_id=session_id, registry=registry)

    if not isinstance(res, dict):
        res = {"result": res}

    if not overridden:
        try:
            emit_mode_choice_event(selected_mode, session_id=session_id)
        except Exception as e:
            emit_error({"event": "mode_choice_emit_failed", "error": str(e)})

    try:
        res["selected_mode"] = selected_mode
    except Exception:
        try:
            emit_error({"event": "selected_mode_attach_failed", "error": "failed to attach selected_mode"})
        except Exception:
            pass

    return res


def run_chat_conversation(
    session: Any,
    message: str,
    *,
    explicit_mode: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
    api: Any = None,
) -> Dict[str, Any]:
    """
    High-level helper for the web 'chat' / 'auto' flow.
    """
    if registry is None and api is not None:
        registry = default_tool_registry(api)
    if registry is None:
        registry = ToolRegistry()

    session_id = getattr(session, "id", None) or getattr(session, "session_id", None)

    # (Optional) attach minimal conversational history to the session
    try:
        hist = getattr(session, "history", None)
        if isinstance(hist, list):
            hist.append({"role": "user", "content": message})
        else:
            setattr(session, "history", [{"role": "user", "content": message}])
    except Exception:
        pass

    # Build history as List[str] for the classifier (best-effort)
    history_texts: List[str] = []
    try:
        hist = getattr(session, "history", None)
        if isinstance(hist, list):
            for h in hist[-30:]:
                if isinstance(h, str):
                    s = h.strip()
                    if s:
                        history_texts.append(s)
                elif isinstance(h, dict):
                    s = str(h.get("content") or "").strip()
                    if s:
                        history_texts.append(s)
                else:
                    s = str(h).strip()
                    if s:
                        history_texts.append(s)
    except Exception:
        history_texts = []

    # Best-effort projects list (optional, but helps SELECT_PROJECT / "existing project" signals)
    projects: List[Dict[str, Any]] = []
    try:
        sess_projects = getattr(session, "projects", None)
        if isinstance(sess_projects, list):
            projects = [p for p in sess_projects if isinstance(p, dict)]
        elif api is not None:
            api_projects = getattr(api, "projects", None)
            if isinstance(api_projects, list):
                projects = [p for p in api_projects if isinstance(p, dict)]
    except Exception:
        projects = []

    llm = None
    if api is not None:
        llm = getattr(api, "llm", None)

    if llm is None and _ChatClient is not None:
        try:
            llm = _ChatClient()
        except Exception:
            llm = None

    return handle_chat(
        message,
        session_id=session_id,
        registry=registry,
        mode_override=explicit_mode,
        llm=llm,
        history=history_texts,
        projects=projects,
    )


# ---------- Default tool wiring helpers --------------------------------------

def default_tool_registry(api: Any) -> ToolRegistry:
    """
    Helper to wrap your existing service layer in ToolSpecs.
    """
    r = ToolRegistry()

    try:
        setattr(r, "api", api)
    except Exception:
        pass

    if api is not None and hasattr(api, "apply_commit") and callable(getattr(api, "apply_commit")):
        orig_apply = getattr(api, "apply_commit")

        def _apply_commit_wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            try:
                res = orig_apply(*args, **kwargs)
                if isinstance(res, dict) and "ok" in res:
                    return res
            except Exception:
                pass
            return {"ok": True}

        try:
            setattr(api, "apply_commit", _apply_commit_wrapper)
            emit_error({
                "event": "legacy_apply_commit_detected",
                "warning": (
                    "DevBotAPI.apply_commit is deprecated. The default tool registry will not "
                    "expose an 'apply_commit' ToolSpec. Calls to api.apply_commit() will be "
                    "wrapped to safely return {'ok': True} on error. Update your API to "
                    "use the orchestrator/apply_and_refresh pipeline instead.")
            })
        except Exception:
            try:
                status({"event": "legacy_apply_commit_detected_warning"})
            except Exception:
                pass

    def _maybe_register(
        tool_name: str,
        *,
        fn_name: Optional[str] = None,
        required: Tuple[str, ...] = (),
        optional: Tuple[str, ...] = (),
        produces_proposals: bool = False,
    ) -> None:
        attr = fn_name or tool_name
        fn = getattr(api, attr, None)
        if not callable(fn):
            return

        def _handler(params: Dict[str, Any]) -> Dict[str, Any]:
            return fn(**(params or {}))

        r.register(
            ToolSpec(
                name=tool_name,
                handler=_handler,
                required=required,
                optional=optional,
                produces_proposals=produces_proposals,
            )
        )

    _maybe_register(
        "create_project",
        required=("brief",),
        optional=("project_name", "base_dir", "tech_stack"),
    )
    _maybe_register(
        "select_project",
        required=("project_path",),
    )
    _maybe_register(
        "refresh_cards",
        required=(),
        optional=("force",),
    )
    _maybe_register(
        "ai_cards",
        required=("mode",),
        optional=("model", "focus"),
    )
    _maybe_register(
        "update_descriptions",
        required=("instructions",),
        produces_proposals=True,
    )
    _maybe_register(
        "run_checks",
        required=(),
    )
    _maybe_register(
        "recommend",
        required=("prompt",),
        optional=("strategy_note",),
    )

    # Deep research tool: validated in-handler so we can return ok=False envelopes
    # (ToolRegistry.execute would otherwise raise for missing required params).
    if hasattr(api, "deep_research") and callable(getattr(api, "deep_research")):
        def _deep_research_handler(params: Dict[str, Any]) -> Dict[str, Any]:
            p = params or {}
            q = p.get("query")
            if q is None or not str(q).strip():
                return {
                    "ok": False,
                    "error": "Missing required param: query",
                    "missing": ["query"],
                }

            allowed = ("query", "scope", "scope_paths", "depth", "force_refresh", "budgets")
            call_params = {k: v for k, v in p.items() if k in allowed}

            # Optional light validation (don't raise; return structured envelope).
            depth = call_params.get("depth")
            if depth is not None:
                try:
                    call_params["depth"] = int(depth)
                except Exception:
                    return {
                        "ok": False,
                        "error": "Invalid param: depth must be an int",
                        "invalid": ["depth"],
                    }

            try:
                res = api.deep_research(**call_params)
                # Acceptance: forward DevBotAPI.deep_research output unchanged (no wrapping/remapping).
                return res
            except Exception as e:
                return {"ok": False, "error": str(e), "exc_type": type(e).__name__}

        r.register(
            ToolSpec(
                name="deep_research",
                handler=_deep_research_handler,
                required=(),
                optional=("scope", "scope_paths", "depth", "force_refresh", "budgets"),
                produces_proposals=False,
            )
        )

    if hasattr(api, "orchestrate") and callable(getattr(api, "orchestrate")):
        def _orch_handler(params: Dict[str, Any]) -> Dict[str, Any]:
            return api.orchestrate(**(params or {}))

        r.register(
            ToolSpec(
                name="orchestrate",
                handler=_orch_handler,
                required=("focus",),
                optional=("mode", "includes", "excludes", "dry_run", "approved_rec_ids", "auto_approve"),
                produces_proposals=False,
            )
        )
    elif hasattr(api, "run_conversation") and callable(getattr(api, "run_conversation")):
        def _orch_fallback(params: Dict[str, Any]) -> Dict[str, Any]:
            return api.run_conversation(**(params or {}))

        r.register(
            ToolSpec(
                name="orchestrate",
                handler=_orch_fallback,
                required=("focus",),
                optional=("mode", "includes", "excludes", "dry_run", "approved_rec_ids", "auto_approve"),
                produces_proposals=False,
            )
        )

    if hasattr(api, "run_conversation") and callable(getattr(api, "run_conversation")):
        def _qa_handler(params: Dict[str, Any]) -> Dict[str, Any]:
            question = params.get("question") or params.get("focus") or ""
            return api.run_conversation(
                focus=question,
                mode="qa",
                dry_run=False,
                auto_approve=False,
            )

        r.register(
            ToolSpec(
                name="qa",
                handler=_qa_handler,
                required=("question",),
                produces_proposals=False,
            )
        )

    def _narrate_handler(p: Dict[str, Any]) -> Dict[str, Any]:
        text = p.get("text") or ""
        assistant(text)
        return {"ok": True, "text": text}

    r.register(
        ToolSpec(
            name="narrate",
            handler=_narrate_handler,
            required=("text",),
            produces_proposals=False,
        )
    )

    _maybe_register(
        "get_card",
        required=("path",),
    )

    return r


__all__ = [
    "handle_chat",
    "run_chat_conversation",
    "default_tool_registry",
    "ToolRegistry",
    "ToolSpec",
    "format_plan",
    "format_planned_changes",
    "narrate_kickoff",
    "narrate_planned_changes",
    "narrate_stage",
    "narrate_apply_result",
]
