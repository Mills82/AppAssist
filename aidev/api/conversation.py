# aidev/api/conversation.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..conversation.intents import classify_intent_slots, IntentResult
from ..conversation.planner import plan_from_intent_result
from ..events import status

import importlib

# 'plan' event is optional in some codebases; import defensively.
try:
    from ..events import plan as emit_plan  # type: ignore
except Exception:  # pragma: no cover
    def emit_plan(_: Dict[str, Any]) -> None:
        return

# Attempt to import mode inference helper; fall back to a safe noop.
try:
    from ..mode_inference import infer_mode  # type: ignore
except Exception:  # pragma: no cover
    def infer_mode(_: Any) -> Dict[str, Any]:
        return {"mode": None, "confidence": 0.0, "intent": None, "rationale": ""}

# Backwards-compatible alias: some callers / tests expect a top-level
# `classify_intent` callable.
def classify_intent(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper around classify_intent_slots."""
    return classify_intent_slots(*args, **kwargs)

# Try to inject the compatibility alias into commonly-checked modules.
for _mod_name in ("aidev.conversation.intents", "aidev.conversation", "aidev"):
    try:
        mod = importlib.import_module(_mod_name)
        if not hasattr(mod, "classify_intent"):
            setattr(mod, "classify_intent", classify_intent)
    except Exception:
        pass

router = APIRouter(prefix="/conversation", tags=["conversation"])


class DebugIntentRequest(BaseModel):
    message: str
    history: Optional[List[str]] = None
    projects: Optional[List[Dict[str, Any]]] = None


class DebugIntentResponse(BaseModel):
    intent: str
    slots: Dict[str, Any]
    confidence: float
    rationale: str
    steps: List[Dict[str, Any]]

    mode: Optional[str] = None
    mode_confidence: float = 0.0

    mode_infer_intent: Optional[str] = None
    mode_infer_rationale: str = ""

    intent_raw: Dict[str, Any] = Field(default_factory=dict)


@router.post("/debug-intent", response_model=DebugIntentResponse)
def debug_intent(req: DebugIntentRequest) -> DebugIntentResponse:
    """
    Classify a single user utterance and produce a normalized execution plan.
    Also emits 'plan' + 'status' SSE events so the UI can preview it immediately.
    """
    # NOTE: this endpoint remains offline-friendly by default (llm=None).
    # Production chat routing should pass a real llm client to classify_intent_slots.
    ir: IntentResult = classify_intent_slots(
        req.message,
        history=req.history or [],
        projects=req.projects or [],
        llm=None,
    )

    steps = plan_from_intent_result(ir, req.message)

    # Steps are dicts; keep them stable/serializable as-is.
    steps_payload: List[Dict[str, Any]] = []
    for s in steps:
        if isinstance(s, dict):
            steps_payload.append(s)
        else:
            # Defensive fallback if any caller returns objects
            steps_payload.append({
                "tool": getattr(s, "tool", None),
                "name": getattr(s, "name", None),
                "params": getattr(s, "params", None),
                "meta": getattr(s, "meta", None),
            })

    intent_raw: Dict[str, Any] = {
        "intent": getattr(getattr(ir, "intent", None), "value", None),
        "slots": getattr(ir, "slots", {}) or {},
        "confidence": float(getattr(ir, "confidence", 0.0) or 0.0),
        "rationale": getattr(ir, "rationale", "") or "",
        "matched_rules": getattr(ir, "matched_rules", None),
    }

    try:
        mode_info = infer_mode(ir) or {}
    except Exception:
        mode_info = {}

    mode = mode_info.get("mode") if isinstance(mode_info, dict) else None
    mode_confidence = float(mode_info.get("confidence", 0.0) or 0.0) if isinstance(mode_info, dict) else 0.0

    mode_infer_intent = mode_info.get("intent") if isinstance(mode_info, dict) else None
    mode_infer_rationale = mode_info.get("rationale") if isinstance(mode_info, dict) else ""

    try:
        emit_plan({
            "intent": intent_raw.get("intent"),
            "slots": intent_raw.get("slots"),
            "steps": steps_payload,
            "mode": mode,
            "mode_confidence": mode_confidence,
        })
        status({"event": "debug_intent", "ok": True, "intent": intent_raw.get("intent"), "mode": mode})
    except Exception:
        pass

    return DebugIntentResponse(
        intent=intent_raw.get("intent") or "",
        slots=intent_raw.get("slots") or {},
        confidence=float(intent_raw.get("confidence", 0.0) or 0.0),
        rationale=intent_raw.get("rationale") or "",
        steps=steps_payload,
        mode=mode,
        mode_confidence=mode_confidence,
        mode_infer_intent=mode_infer_intent,
        mode_infer_rationale=mode_infer_rationale,
        intent_raw=intent_raw,
    )
