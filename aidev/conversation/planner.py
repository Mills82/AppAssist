# aidev/conversation/planner.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from .intents import Intent, IntentResult

# --------------------------------------------------------------------
# Utilities

_TECH_RE = re.compile(
    r"\b(flutter|react|next(?:\.js)?|fastapi|django|node|express|php|laravel|rails|dotnet|unity|unreal)\b",
    re.I,
)

_INCREMENTAL_STRATEGY_NOTE = (
    "When proposing recommendations, prefer a short (about 3–5 items), "
    "prioritized list of small, independently safe steps. For each "
    "recommendation, include:\n"
    "- A clear, testable acceptance criterion\n"
    "- A concrete definition of \"done\" for that step\n"
    "Treat these as meta-guidelines: follow them, but do not restate them "
    "verbatim in your output.\n\n"
    "Assume the orchestrator will apply changes one recommendation at a time "
    "and re-scan the project map and cards between steps."
)


def _extract_probable_path(text: str) -> Optional[str]:
    m = re.search(r"[\"']([^\"']*[\\/][^\"']+)[\"']", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"([A-Za-z]:[\\/][^\s]+|[./][^\s]+|[^ \t\n\r\f\v]+[\\/][^ \t\n\r\f\v]+)", text)
    return m.group(1).strip() if m else None


def _extract_project_name(text: str) -> Optional[str]:
    for pat in (
        r"\bcalled\s+([A-Za-z0-9._-]+)\b",
        r"\bname(?:d)?\s+([A-Za-z0-9._-]+)\b",
        r"\b(?:create|new)\s+([A-Za-z0-9._-]+)\b",
    ):
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1)
    return None


def _extract_stack(text: str) -> Optional[str]:
    m = _TECH_RE.search(text or "")
    return m.group(1).lower() if m else None


def _intent_label(intent: Any) -> str:
    if hasattr(intent, "value"):
        return str(intent.value)
    return str(intent)


def _ai_cards_changed_params(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic policy:
      - Always summarize changed files.
      - Optionally pass model/focus if provided by intent classification.
    """
    params: Dict[str, Any] = {"mode": "changed"}

    model = str(slots.get("model", "") or "").strip()
    if model:
        params["model"] = model

    focus = str(slots.get("focus", "") or "").strip()
    if focus:
        params["focus"] = focus

    return params


def _normalize_targets(val: Any) -> Optional[List[str]]:
    """Normalize slots.targets or path-like signals to list[str] or None."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else None
    if isinstance(val, list):
        out = [str(x).strip() for x in val if str(x).strip()]
        return out or None
    return None


# --------------------------------------------------------------------
# Planner API (used by chat.py / api)


def propose_plan(
    intent_or_result: Union[IntentResult, Intent, str, Any],
    message: str,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Translate (intent, message) into ordered, incremental tool calls.

    IMPORTANT: If an IntentResult is provided, we will use it directly and
    will NOT re-classify the message. This is required to "rely on the LLM call".

    Tools expected by chat.default_tool_registry():
      - create_project
      - select_project
      - refresh_cards
      - ai_cards
      - update_descriptions
      - run_checks
      - recommend
      - apply_commit
      - orchestrate
    """
    if isinstance(intent_or_result, IntentResult):
        ir = intent_or_result
        I = ir.intent.value
        slots = ir.slots or {}
    else:
        # Back-compat: allow callers to pass intent only (but then slots are empty).
        I = _intent_label(intent_or_result)
        slots = {}

    steps: List[Dict[str, Any]] = []

    # deep_research explicit tool request
    # NOTE: This is intentionally slot-driven (no new Intent enum required), so
    # existing intent branching behavior remains unchanged.
    if slots.get("tool") == "deep_research" or slots.get("deep_research"):
        # Build a normalized, non-empty query string. The planner MUST NOT emit
        # a deep_research step without a non-empty 'query' per acceptance rules.
        raw_q = slots.get("query")
        if raw_q is None:
            q = (message or "").strip()
        else:
            if isinstance(raw_q, str):
                q = raw_q.strip()
            else:
                try:
                    q = str(raw_q).strip()
                except Exception:
                    q = ""

        if q:
            params: Dict[str, Any] = {"query": q}

            # scope can be any value; planner will pass through without strict coercion
            if slots.get("scope") is not None:
                params["scope"] = slots.get("scope")

            sp = _normalize_targets(slots.get("scope_paths"))
            if sp is not None:
                params["scope_paths"] = sp

            # depth: treat as string enum, prefer 'standard'/'quick' values; emit normalized lowercase string
            d = slots.get("depth")
            if d is not None:
                try:
                    params["depth"] = str(d).strip().lower()
                except Exception:
                    pass

            # force_refresh: coerce common string/bool inputs safely
            fr = slots.get("force_refresh")
            if fr is not None:
                if isinstance(fr, str):
                    params["force_refresh"] = fr.strip().lower() in ("1", "true", "yes", "y", "on")
                else:
                    params["force_refresh"] = bool(fr)

            # budgets: pass-through, runtime/tool will validate shape
            if slots.get("budgets") is not None:
                params["budgets"] = slots.get("budgets")

            # Mark deep_research calls as fail_safe so executor may continue if it fails
            steps.append({
                "tool": "deep_research",
                "name": "Deep research",
                "params": params,
                "meta": {"fail_safe": True},
            })
        else:
            # No non-empty query available: do NOT emit a deep_research step. This
            # keeps the planner from producing invalid tool calls that would
            # violate the tool contract requiring a 'query' parameter.
            pass
        # Fall through: session_id meta will be set later.

    if I == Intent.CREATE_PROJECT.value:
        brief = slots.get("focus") or message.strip()
        project_name = slots.get("project_name") or _extract_project_name(message) or None
        base_dir = slots.get("base_dir") or None
        tech = slots.get("tech_stack") or slots.get("framework") or _extract_stack(message)

        create_params: Dict[str, Any] = {"brief": brief}
        if project_name:
            create_params["project_name"] = project_name
        if base_dir:
            create_params["base_dir"] = base_dir
        if tech:
            create_params["tech_stack"] = tech

        # Determine if the message contains answer key/values using slots or a simple key:value regex
        answers_present = False
        if any(slots.get(k) for k in ("project_name", "base_dir", "tech_stack", "answers")):
            answers_present = True
        else:
            if re.search(r"\b\w+\s*[:=]\s*[^,;\n]+", message):
                answers_present = True

        step_name = "create.advance" if answers_present else "create.start"

        if answers_present:
            answers: Dict[str, Any] = {}
            for k in ("project_name", "base_dir", "tech_stack"):
                v = slots.get(k)
                if v:
                    answers[k] = v

            slot_answers = slots.get("answers")
            if slot_answers:
                if isinstance(slot_answers, dict):
                    for kk, vv in slot_answers.items():
                        if vv is not None:
                            answers[kk] = vv
                else:
                    answers["text"] = slot_answers

            if answers:
                create_params["answers"] = answers

        steps.append({
            "tool": "create_project",
            "name": step_name,
            "params": create_params,
        })

        steps.append({"tool": "refresh_cards", "name": "Build project map", "params": {"force": False}})
        steps.append({
            "tool": "ai_cards",
            "name": "AI cards (changed)",
            "params": _ai_cards_changed_params(slots),
        })

    elif I == Intent.SELECT_PROJECT.value:
        p = slots.get("project_path") or _extract_probable_path(message)
        steps.append({"tool": "select_project", "name": "Select project", "params": {"project_path": p}})
        steps.append({"tool": "refresh_cards", "name": "Refresh cards", "params": {"force": False}})
        steps.append({
            "tool": "ai_cards",
            "name": "AI cards (changed)",
            "params": _ai_cards_changed_params(slots),
        })

    elif I == Intent.UPDATE_DESCRIPTIONS.value:
        instructions = slots.get("instructions") or message.strip()
        steps.append({
            "tool": "ai_cards",
            "name": "AI cards (changed)",
            "params": _ai_cards_changed_params(slots),
        })
        steps.append({
            "tool": "update_descriptions",
            "name": "Update descriptions",
            "params": {"instructions": instructions},
        })

    elif I == Intent.RUN_CHECKS.value:
        steps.append({"tool": "run_checks", "name": "Run checks", "params": {}})

    elif I == Intent.ANALYZE_PROJECT.value:
        # Ensure project/card refresh happens before analysis; insert refresh_cards
        # if it wasn't already part of this flow. Do not reorder existing steps.
        steps.append({"tool": "refresh_cards", "name": "Refresh cards", "params": {"force": False}})

        steps.append({
            "tool": "ai_cards",
            "name": "AI cards (changed)",
            "params": _ai_cards_changed_params(slots),
        })

        base_prompt = slots.get("focus") or message.strip()

        if base_prompt:
            # Insert a single fail-safe deep_research step after the cards refresh
            # and ai_cards refresh; depth must be the string enum 'standard'.
            steps.append({
                "tool": "deep_research",
                "name": "Deep research",
                "params": {"query": base_prompt, "depth": "standard"},
                "meta": {"fail_safe": True},
            })
        analyze_params: Dict[str, Any] = {"mode": "analyze", "prompt": base_prompt, "focus": base_prompt}

        targets = _normalize_targets(slots.get("targets")) or _normalize_targets(slots.get("project_path")) or _normalize_targets(_extract_probable_path(message))
        if targets:
            analyze_params["targets"] = targets

        steps.append({
            "tool": "orchestrate",
            "name": "Analyze project",
            "params": analyze_params,
            "meta": {"strategy": "Run orchestrator to produce structured analyze_plan"},
        })

    elif I == Intent.Q_AND_A.value:
        query_text = message.strip()

        top_k = slots.get("top_k") or 5
        try:
            top_k = int(top_k)
        except Exception:
            top_k = 5

        steps.append({
            "tool": "ai_cards",
            "name": "AI cards (query)",
            "params": {"mode": "query", "query": query_text, "top_k": top_k},
        })

        targets = _normalize_targets(slots.get("targets")) or _normalize_targets(slots.get("project_path")) or _normalize_targets(_extract_probable_path(message))
        orchestrate_params: Dict[str, Any] = {
            "mode": "qa",
            "prompt": query_text,
            "focus": query_text,
            "fallback": {"mode": "research", "max_retries": 1},
        }
        if targets:
            orchestrate_params["targets"] = targets

        steps.append({
            "action": "qa",
            "tool": "orchestrate",
            "name": "Run QA pipeline",
            "params": orchestrate_params,
            "meta": {"strategy": "Run orchestrator QA mode"},
        })

    elif I == Intent.MAKE_RECOMMENDATIONS.value:
        base_prompt = slots.get("focus") or message.strip()
        raw_prompt = str(slots.get("focus_raw") or "").strip() or message.strip()
        raw_prompt = raw_prompt[:5000]

        steps.append({
            "tool": "ai_cards",
            "name": "AI cards (changed)",
            "params": _ai_cards_changed_params(slots),
        })
        if base_prompt:
            # Insert a single fail-safe deep_research step immediately before recommendations
            steps.append({
                "tool": "deep_research",
                "name": "Deep research",
                "params": {"query": base_prompt, "depth": "quick"},
                "meta": {"fail_safe": True},
            })
        steps.append({
            "tool": "recommend",
            "name": "Make incremental recommendations",
            "params": {
                "prompt": base_prompt,
                "strategy_note": raw_prompt,
            },
            "meta": {
                "strategy": "Propose a few small, independently safe recommendations with explicit acceptance criteria and done states.",
                "incremental_guidelines": _INCREMENTAL_STRATEGY_NOTE,
            },
        })

    elif I == Intent.APPLY_EDITS.value:
        base_prompt = slots.get("focus") or message.strip()
        raw_prompt = str(slots.get("focus_raw") or "").strip() or message.strip()
        raw_prompt = raw_prompt[:5000]

        steps.append({
            "tool": "ai_cards",
            "name": "AI cards (changed)",
            "params": _ai_cards_changed_params(slots),
        })
        steps.append({
            "tool": "orchestrate",
            "name": "Orchestrate edit pipeline",
            "params": {
                "mode": "edit",
                "auto_approve": True,
                "prompt": base_prompt,
                "focus": base_prompt,
                "strategy_note": raw_prompt,
            },
            "meta": {"strategy": "Run orchestrator to apply edits automatically."},
        })

    else:
        # Safe default: freshen cards then recommend.
        base_prompt = slots.get("focus") or message.strip()
        raw_prompt = str(slots.get("focus_raw") or "").strip() or message.strip()
        raw_prompt = raw_prompt[:5000]
        
        steps.append({
            "tool": "ai_cards",
            "name": "AI cards (changed)",
            "params": _ai_cards_changed_params(slots),
        })
        steps.append({
            "tool": "recommend",
            "name": "Make incremental recommendations",
            "params": {"prompt": base_prompt, "strategy_note": raw_prompt},
            "meta": {
                "strategy": "Propose a few small, independently safe recommendations with explicit acceptance criteria and done states.",
                "incremental_guidelines": _INCREMENTAL_STRATEGY_NOTE,
            },
        })

    if session_id:
        for st in steps:
            st.setdefault("meta", {})["session_id"] = session_id

    return steps


def plan_from_intent_result(ir: IntentResult, message: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Preferred entrypoint: does NOT re-classify; uses the provided IntentResult."""
    return propose_plan(ir, message, session_id=session_id)


def plan_from_intent(intent: Any, message: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Compatibility wrapper delegating to propose_plan()."""
    return propose_plan(intent, message, session_id=session_id)


__all__ = ["propose_plan", "plan_from_intent", "plan_from_intent_result"]