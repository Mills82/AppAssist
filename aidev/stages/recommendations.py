# aidev/stages/recommendations.py
from __future__ import annotations

import json
import os
from importlib.resources import files as pkg_files
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..llm_utils import parse_json_array, parse_json_object
from ..orchestration.edit_prompts import (
    recommendations_system_prompt,
    build_recommendations_user_payload,
)

# Try to import canonical incremental guidelines string from llm_client.
# If unavailable (staged edits), fall back to None and make the helper a no-op.
try:
    from ..llm_client import INCREMENTAL_GUIDELINES
except Exception:
    INCREMENTAL_GUIDELINES = None

# Type aliases for the orchestrator-supplied LLM wrappers
ChatJsonFn = Callable[
    [str, Any, Dict[str, Any], float, str, Optional[int]],
    Tuple[Any, Any],
]
ChatTextFn = Callable[[str, Any, str, Optional[int]], str]

ProgressFn = Callable[[str, Dict[str, Any]], None]
ErrorFn = Callable[[str, Dict[str, Any]], None]


def _load_schema(name: str) -> dict:
    """Load a JSON schema from aidev.schemas/<name>."""
    try:
        p = pkg_files("aidev.schemas").joinpath(name)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover - simple I/O wrapper
        raise RuntimeError(f"Failed to load JSON schema '{name}': {e}")


RECOMMENDATIONS_SCHEMA: Dict[str, Any] = _load_schema("recommendations.schema.json")


# ---------------------------------------------------------------------------
# Local normalization helpers
# ---------------------------------------------------------------------------

# Keys allowed by recommendations.schema.json (v2)
_ALLOWED_REC_KEYS = {
    "schema_version",
    "id",
    "rationale",
    "title",
    "reason",
    "summary",
    "risk",
    "budget_estimate",
    "files",
    "actions",
    "acceptance_criteria",
}


def _coerce_recommendations_sequence(data: Any) -> List[Dict[str, Any]]:
    """
    Coerce various shapes into a list of recommendation dicts:

      - [ {...}, {...} ]
      - { "recommendations": [ {...}, ... ] }
      - { ... }  (single recommendation dict)
    """
    if isinstance(data, list):
        seq = data
    elif isinstance(data, dict):
        if isinstance(data.get("recommendations"), list):
            seq = data["recommendations"]
        else:
            seq = [data]
    else:
        return []

    out: List[Dict[str, Any]] = []
    for item in seq:
        if isinstance(item, dict):
            out.append(item)
    return out


def normalize_recommendations(data: Any) -> List[Dict[str, Any]]:
    """
    Normalize raw LLM recommendation output into a list of dicts that:

      - Always set schema_version = 2.
      - Only include keys allowed by recommendations.schema.json.
      - Normalize acceptance_criteria to a non-empty list of strings
        when possible.

    This runs *after* the primary JSON-schema call, and is also used
    for fallback text parsing paths.
    """
    raw_recs = _coerce_recommendations_sequence(data)
    normed: List[Dict[str, Any]] = []

    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue

        # Start with a shallow copy so we can inspect extra fields.
        r = dict(rec)

        # Base normalized object with forced schema_version.
        out: Dict[str, Any] = {"schema_version": 2}

        # Copy whitelisted keys when present.
        for key in _ALLOWED_REC_KEYS:
            if key == "schema_version":
                continue
            if key in r:
                out[key] = r[key]

        # Normalize risk to a simple string; leave fine-grained
        # validation (enum: low/medium/high) to later stages.
        risk_val = out.get("risk", "low")
        if not isinstance(risk_val, str):
            risk_val = str(risk_val)
        risk_val = (risk_val or "low").strip() or "low"
        out["risk"] = risk_val

        # Normalize acceptance_criteria:
        # - Prefer array if present
        # - If missing but a single acceptance-like string exists,
        #   wrap it into a list so downstream shaping can still work.
        ac_raw = out.get("acceptance_criteria")
        ac: List[str] = []

        if isinstance(ac_raw, list):
            for c in ac_raw:
                s = str(c).strip()
                if s:
                    ac.append(s)
        elif isinstance(ac_raw, str):
            s = ac_raw.strip()
            if s:
                ac.append(s)

        # If we have nothing, leave it empty; llm_recommendations will
        # inject a conservative default if needed.
        out["acceptance_criteria"] = ac

        normed.append(out)

    return normed


# ---------------------------------------------------------------------------
# Meta helper: deterministic reporting of ai_summary/card refresh failures
# ---------------------------------------------------------------------------


def _normalize_ai_summary_failures(meta: Any) -> List[Dict[str, str]]:
    """Normalize summarize failures from meta into meta['ai_summary_failures'].

    Expected meta contract (best-effort, tolerant of variations):
      - meta['ai_summary_failures'] OR meta['cards_refresh_failures'] may contain
        a list of objects with at least: {path, error/message}.

    This helper coerces entries into {'path': str, 'error': str}, drops malformed
    items, and sorts by 'path' to ensure deterministic ordering for UI/events.
    """
    if not isinstance(meta, dict):
        return []

    raw = meta.get("ai_summary_failures") or meta.get("cards_refresh_failures") or []
    out: List[Dict[str, str]] = []
    for e in (raw or []):
        if not isinstance(e, dict):
            continue
        p = e.get("path")
        err = e.get("error") or e.get("message") or ""
        if not p:
            continue
        out.append({"path": str(p), "error": str(err)})

    out.sort(key=lambda x: x["path"])
    meta["ai_summary_failures"] = out
    return out


# ---------------------------------------------------------------------------
# Guidelines helper: ensure INCREMENTAL_GUIDELINES is present in user payload
# ---------------------------------------------------------------------------


def _ensure_guidelines_in_user_payload(user_payload: Dict[str, Any], system_text: str) -> None:
    """
    Idempotently ensure the canonical INCREMENTAL_GUIDELINES string is present
    in the user_payload (under the 'incremental_guidelines' key) unless the
    same text already appears in the system_text or any top-level string
    value of the payload.

    This is intentionally conservative: if INCREMENTAL_GUIDELINES is not
    importable or empty, this becomes a no-op to avoid hard dependencies
    during staged edits.
    """
    if not INCREMENTAL_GUIDELINES:
        return
    g = str(INCREMENTAL_GUIDELINES).strip()
    if not g:
        return

    # If the system prompt already includes the guidelines, nothing to do.
    if isinstance(system_text, str) and g in system_text:
        return

    # If payload already has the explicit key containing the guidelines, do nothing.
    existing = user_payload.get("incremental_guidelines")
    if isinstance(existing, str) and g in existing:
        return

    # Check any top-level string values to avoid duplicate insertion.
    for v in user_payload.values():
        if isinstance(v, str) and g in v:
            return

    # Otherwise, insert the canonical guidelines.
    try:
        user_payload["incremental_guidelines"] = g
    except Exception:
        # Defensive: if payload is not mutable for some reason, silently skip.
        return


# ---------------------------------------------------------------------------
# LLM entrypoint
# ---------------------------------------------------------------------------


def llm_recommendations(
    *,
    chat_json_fn: ChatJsonFn,
    chat_text_fn: ChatTextFn,
    project_brief_text: str,
    meta: Dict[str, Any],
    developer_focus: str,
    strategy_note: Optional[str] = None,
    max_tokens: Optional[int] = None,
    progress_cb: Optional[ProgressFn] = None,
    error_cb: Optional[ErrorFn] = None,
) -> List[Dict[str, Any]]:
    """
    Ask the LLM for recommendations, normalize to:

        [
          {
            id,
            title,
            reason,
            summary,
            rationale,
            risk,
            files?,        # optional
            actions?,      # optional, passed through
            acceptance_criteria[]
          },
          ...
        ]

    This is LLM-specific plumbing extracted out of orchestrator.py.
    The orchestrator passes its wrappers for chat_json/chat so this
    stays stateless and testable.

    NOTE: Context for this call is derived from project meta (project_map,
    cards, etc.) inside build_recommendations_user_payload; this function
    no longer accepts a raw ctx_blob.
    """
    # If an upstream concurrent ai_summary refresh recorded per-file failures,
    # normalize and emit them deterministically. This must be non-fatal.
    try:
        failures = _normalize_ai_summary_failures(meta)
    except Exception:
        failures = []

    if failures:
        # Ensure the canonical key name is used in the emitted payload so
        # downstream orchestrator logic can consistently pick it up.
        payload = {"ai_summary_failures": failures}
        if progress_cb:
            progress_cb("ai_summary_failures", payload)
        elif error_cb:
            error_cb(
                "ai_summary_failures",
                {
                    "reason": "ai_summary_failures",
                    "message": "One or more ai_summary refreshes failed; continuing with available context.",
                    "ai_summary_failures": failures,
                },
            )

    system_text = recommendations_system_prompt()

    # Budget / shaping hints; orchestrator can still override via env
    try:
        max_items = int(os.getenv("AIDEV_RECS_MAX_ITEMS", "12"))
    except Exception:
        max_items = 12

    try:
        max_chars = int(os.getenv("AIDEV_RECS_MAX_CHARS", "1200"))
    except Exception:
        max_chars = 1200

    try:
        max_context_chars = int(os.getenv("AIDEV_RECS_MAX_CONTEXT_CHARS", "120000"))
    except Exception:
        max_context_chars = 120_000

    budget_limits: Dict[str, int] = {
        "max_items": max_items,
        "max_chars_per_item": max_chars,
        "max_context_chars": max_context_chars,
    }

    # Try to locate an upstream research brief in meta (planner should have
    # run deep_research before MAKE_RECOMMENDATIONS). If absent, lazily
    # attempt to call the deep research engine; in either case we do NOT
    # construct payload-level digest here. Instead, pass the brief (if any)
    # into build_recommendations_user_payload via deep_research_brief so the
    # prompt builder (single source of truth) can compute any compact digest.
    deep_brief: Optional[Dict[str, Any]] = None
    try:
        if isinstance(meta, dict):
            for k in ("deep_research", "deep_research_result", "research_brief"):
                if k in meta:
                    candidate = meta.get(k)
                    if isinstance(candidate, dict):
                        # If the candidate is a wrapper that contains a nested
                        # research_brief, prefer it.
                        if candidate.get("ok") and isinstance(candidate.get("research_brief"), dict):
                            deep_brief = candidate.get("research_brief")
                        else:
                            # Otherwise treat the dict as the brief itself.
                            deep_brief = candidate
                        break

        if deep_brief is None:
            try:
                from ..orchestration import deep_research_engine as dre
            except Exception:
                dre = None

            if dre is not None:
                fn = getattr(dre, "deep_research", None)
                if callable(fn):
                    deep_res = None
                    try:
                        deep_res = fn(depth="quick", developer_focus=developer_focus, project_brief=project_brief_text)
                    except TypeError:
                        try:
                            deep_res = fn(depth="quick", developer_focus=developer_focus)
                        except TypeError:
                            try:
                                deep_res = fn(depth="quick", project_brief=project_brief_text)
                            except TypeError:
                                try:
                                    deep_res = fn("quick")
                                except Exception:
                                    deep_res = None
                    except Exception:
                        deep_res = None

                    # If the engine returned a successful wrapper, extract the brief
                    if isinstance(deep_res, dict) and deep_res.get("ok"):
                        brief = deep_res.get("research_brief") or deep_res.get("brief") or deep_res.get("result")
                        if isinstance(brief, dict):
                            deep_brief = brief
    except Exception as e:  # pragma: no cover - defensive
        # Non-fatal: surface to callbacks and continue without a brief.
        if error_cb:
            error_cb(
                "deep_research",
                {
                    "reason": "deep_research_failed",
                    "message": "Failed to locate or run deep_research; continuing without research brief.",
                    "error": str(e),
                },
            )
        elif progress_cb:
            progress_cb(
                "deep_research",
                {"reason": "deep_research_failed", "error": str(e)},
            )

    # Build the user payload via the shared prompt helper. Prefer passing
    # the deep_research_brief into the prompt builder (single digest source).
    try:
        # Try the newer signature that accepts deep_research_brief. If that
        # hasn't been updated in the staged edit environment, fall back to
        # calling without it to remain backward compatible.
        try:
            user_payload = build_recommendations_user_payload(
                project_brief_text=project_brief_text,
                meta=meta,
                developer_focus=developer_focus,
                strategy_note=strategy_note,
                budget_limits=budget_limits,
                deep_research_brief=deep_brief,
            )
        except TypeError:
            user_payload = build_recommendations_user_payload(
                project_brief_text=project_brief_text,
                meta=meta,
                developer_focus=developer_focus,
                strategy_note=strategy_note,
                budget_limits=budget_limits,
            )
    except Exception as e:
        # If building the payload fails for any reason, surface and abort
        # the recommendations attempt gracefully by reporting the error and
        # returning an empty list.
        if error_cb:
            error_cb(
                "recommendations.payload",
                {
                    "reason": "payload_build_failed",
                    "message": "Failed to construct recommendations payload.",
                    "error": str(e),
                },
            )
        return []

    # Inject related KB cards (path + short summary) into user_payload.context.related_cards.
    # This is done lazily to avoid import-time side-effects. Failures here should
    # not prevent the main recommendations flow.
    try:
        # Lazy import of KB selector module (aidev.cards)
        try:
            from .. import cards as kb_module
            kb = kb_module
        except Exception:
            kb = None

        # Read config overrides if available; fall back to safe defaults.
        try:
            from ..config import (
                RECOMMENDATIONS_RELATED_CARDS_TOP_K as cfg_top_k,
                RECOMMENDATIONS_RELATED_CARDS_CHAR_CAP as cfg_char_cap,
            )
        except Exception:
            cfg_top_k = None
            cfg_char_cap = None

        top_k = int(cfg_top_k) if cfg_top_k is not None else 6
        char_cap = int(cfg_char_cap) if cfg_char_cap is not None else 4000

        related_cards: List[Dict[str, str]] = []
        if kb is not None:
            select_fn = getattr(kb, "select_cards", None)
            if callable(select_fn):
                cards = select_fn(developer_focus, top_k=top_k)
                total = 0
                for c in (cards or [])[:top_k]:
                    # Accept either dict-like or object-like cards; only use path and summary.
                    if isinstance(c, dict):
                        path = c.get("path")
                        summary = c.get("summary") or ""
                    else:
                        path = getattr(c, "path", None)
                        summary = getattr(c, "summary", None) or ""

                    if not path or not summary:
                        continue

                    remaining = char_cap - total
                    if remaining <= 0:
                        break

                    s = str(summary)
                    if len(s) > remaining:
                        # leave room for an ellipsis; if only 1 char left, drop this card.
                        if remaining > 1:
                            s = s[: max(0, remaining - 1)] + "…"
                        else:
                            break

                    related_cards.append({"path": path, "summary": s})
                    total += len(s)

        if related_cards:
            ctx = user_payload.setdefault("context", {})
            ctx["related_cards"] = related_cards
    except Exception as e:
        # Non-fatal: report via error_cb if available and continue.
        if error_cb:
            error_cb(
                "recommendations.related_cards",
                {
                    "reason": "related_cards_failed",
                    "message": "Failed to attach related KB cards to recommendations payload.",
                    "error": str(e),
                },
            )

    # Ensure incremental guidelines are present in the payload unless they
    # are already included in the system prompt or payload.
    try:
        _ensure_guidelines_in_user_payload(user_payload, system_text)
    except Exception:
        # Defensive: do not let guidelines injection break the main flow.
        pass

    # Primary path: structured JSON via schema
    recs: List[Dict[str, Any]] = []
    data: Any = None

    try:
        data, _res = chat_json_fn(
            system_text,
            user_payload,
            RECOMMENDATIONS_SCHEMA,
            0.0,  # temperature
            "recommendations",
            max_tokens,
        )
    except Exception as e:
        if error_cb:
            error_cb(
                "recommendations",
                {
                    "reason": "llm_error",
                    "message": "The AI could not generate recommendations (JSON schema call failed).",
                    "error": str(e),
                },
            )
        data = None

    if isinstance(data, list) and data:
        recs = normalize_recommendations(data)
    elif isinstance(data, dict) and data:
        recs = normalize_recommendations(data)

    # Fallback: try again with stricter budget and manual JSON extraction
    if not recs:
        user_payload_fallback = dict(user_payload)
        try:
            base_items = int(budget_limits.get("max_items", 8))
        except Exception:
            base_items = 8
        try:
            base_chars = int(budget_limits.get("max_chars_per_item", 400))
        except Exception:
            base_chars = 400

        user_payload_fallback["budget_limits"] = {
            "max_items": max(3, base_items // 2),
            "max_chars_per_item": max(200, base_chars // 2),
            # keep same context budget; only tighten per-item size/count
            "max_context_chars": budget_limits.get("max_context_chars", max_context_chars),
        }

        # Ensure fallback payload also contains the incremental guidelines.
        try:
            _ensure_guidelines_in_user_payload(user_payload_fallback, system_text)
        except Exception:
            pass

        raw: str = ""
        try:
            raw = chat_text_fn(
                system_text,
                user_payload_fallback,
                "recommendations",
                max_tokens,
            )
        except Exception as e:
            if error_cb:
                error_cb(
                    "recommendations",
                    {
                        "reason": "llm_error_fallback",
                        "message": "The AI could not generate recommendations (fallback text call failed).",
                        "error": str(e),
                    },
                )
            raw = ""

        if raw:
            arr = parse_json_array(raw)
            if arr:
                recs = normalize_recommendations(arr)
            else:
                obj = parse_json_object(raw)
                if obj:
                    recs = normalize_recommendations(obj)

    # Final shaping and ID assignment
    if not recs and error_cb:
        error_cb(
            "recommendations",
            {
                "reason": "no_usable_recommendations",
                "message": "The AI could not generate usable recommendations for this run.",
            },
        )

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(recs[:12], 1):
        title = (r.get("title") or "").strip()
        reason = (r.get("reason") or "").strip()
        ac = r.get("acceptance_criteria") or []

        # fields we want to preserve for the UI / downstream logic
        summary = (r.get("summary") or "").strip()
        rationale = (r.get("rationale") or "").strip()

        risk_val = r.get("risk", "low")
        if not isinstance(risk_val, str):
            risk_val = str(risk_val)
        risk = (risk_val or "low").strip() or "low"

        # Optional: propagate 'files' and 'actions' if present and sane
        files = r.get("files")
        if not isinstance(files, list):
            files = None

        actions = r.get("actions")
        if not isinstance(actions, list):
            actions = None

        if not title or not reason:
            continue
        if not isinstance(ac, list):
            ac = [str(ac)]
        ac = [str(x).strip() for x in ac if str(x).strip()]
        if not ac:
            ac = [f"Demonstrate measurable progress on '{title}' with a small PR."]

        rec_out: Dict[str, Any] = {
            "id": r.get("id") or f"rec-{i}",
            "title": title,
            "reason": reason,
            "summary": summary,
            "rationale": rationale,
            "risk": risk,
            "acceptance_criteria": ac,
        }

        if files is not None:
            rec_out["files"] = files
        if actions is not None:
            rec_out["actions"] = actions

        out.append(rec_out)

    return out
