# aidev/orchestration/analyze_prompts.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..llm_client import system_preset

logger = logging.getLogger(__name__)

# Tunables for how much context we send into Analyze mode.
DEFAULT_MAX_CARD_SUMMARY_CHARS = 800
DEFAULT_MAX_STRUCTURE_CHARS = 10_000
DEFAULT_MAX_META_CHARS = 6_000
DEFAULT_MAX_CARDS = 25

# Explicit overall + per-section budgets (character-based, post-serialization).
# NOTE: These are conservative defaults intended to keep the final payload well
# under typical model context limits once wrapped into chat messages.
DEFAULT_MAX_PAYLOAD_CHARS = 30_000
DEFAULT_MAX_BRIEF_CHARS = 6_000
DEFAULT_MAX_CARDS_CHARS = 12_000


def analyze_system_prompt() -> str:
    """
    System prompt for Analyze mode.

    Prefer loading from /prompts via system_preset("analyze"), with a small
    baked-in fallback if no prompt file is found.
    """
    try:
        txt = system_preset("analyze")
        if txt:
            return txt
    except Exception:
        # Fall through to builtin fallback
        pass

    # Fallback prompt if no system.analyze.md / analyze.md is present.
    # This is intentionally similar to your saved system.analyze.md, but shorter.
    return (
        "*** ANALYZE MODE SYSTEM PROMPT ***\n"
        "# System Prompt — AnalyzePlan (read-only analysis, object format)\n\n"
        "You are an expert software architect and code reviewer for THIS repository.\n"
        "\n"
        "In analyze mode you:\n"
        "- Read the project context and analysis_focus.\n"
        "- Identify the most important themes and opportunities for improvement.\n"
        "- Propose read-only, prioritized recommendations grouped by theme.\n"
        "- Return a single JSON object that matches the Analyze Plan schema (schema_version = 1).\n"
        "\n"
        "You are NOT allowed to emit code patches, diffs, or machine-editable JSONL edits.\n"
        "\n"
        "INPUTS (user payload):\n"
        "- analysis_focus: what to analyze or focus on.\n"
        "- project_brief: markdown summary of the repo.\n"
        "- project_meta: compact structure / metadata.\n"
        "- structure_overview: compact representation of the project tree.\n"
        "- top_cards: relevant Knowledge Cards with path, title, summary, language, score.\n"
        "\n"
        "Treat analysis_focus as the primary question or lens.\n"
        "Assume the user understands the overall project but may not be an expert developer.\n"
        "\n"
        "OUTPUT (JSON object only):\n"
        "- schema_version: 1\n"
        "- focus: short, refined focus string.\n"
        "- overview: 2–6 sentences summarizing project health and key findings.\n"
        "- themes: array of 1–5 themes with id, title, summary, impact, effort, files, recommendations.\n"
        "- next_steps: short list of concrete next steps.\n"
        "\n"
        "Each recommendation must be read-only: describe what to change and why, but do not include patches or diffs.\n"
        "Do not wrap the JSON in markdown fences; the output is consumed by a parser.\n"
    )


def _compact_json_blob(obj: Any, max_chars: int) -> str:
    """
    Turn an arbitrary object into a compact JSON string and truncate it to a
    bounded length. This protects us from giant payloads.
    """
    if obj is None:
        return ""
    try:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(obj)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "… (truncated)"
    return text


def _json_len(obj: Any) -> int:
    """Length of compact JSON serialization (best-effort)."""
    try:
        return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        return len(str(obj))


def _normalize_path(p: Any) -> str:
    if not isinstance(p, str):
        return ""
    # Normalize separators and strip leading ./ for stable comparisons.
    p = p.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def _is_non_target_path(path: str) -> bool:
    """Filter out obvious generated/artifact content by default."""
    if not path:
        return True
    # Exclude .aidev artifacts anywhere in the path.
    if path.startswith(".aidev/") or "/.aidev/" in path or path == ".aidev":
        return True
    return False


def _coerce_score(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _prepare_cards(
    top_cards: Optional[Iterable[Dict[str, Any]]],
    *,
    max_cards: int,
    max_card_summary_chars: int,
    allow_non_target_paths: bool,
) -> List[Dict[str, Any]]:
    """Deterministically filter, sort, and dedupe cards."""
    cards: List[Dict[str, Any]] = []
    for c in top_cards or []:
        if not isinstance(c, dict):
            continue
        path = _normalize_path(c.get("path"))
        if not path:
            continue
        if (not allow_non_target_paths) and _is_non_target_path(path):
            continue
        cards.append(c)

    # Deterministic ordering: score desc, then path asc.
    cards.sort(key=lambda c: (-_coerce_score(c.get("score")), _normalize_path(c.get("path")) or ""))

    # Dedupe by normalized path, keep first (highest score due to sort).
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for c in cards:
        p = _normalize_path(c.get("path"))
        if not p or p in seen:
            continue
        seen.add(p)
        summary = c.get("summary")
        if isinstance(summary, str) and len(summary) > max_card_summary_chars:
            summary = summary[: max_card_summary_chars] + "…"
        unique.append(
            {
                "path": p,
                "title": c.get("title"),
                "summary": summary,
                "language": c.get("language"),
                "score": c.get("score"),
            }
        )
        if len(unique) >= max_cards:
            break

    return unique


def _shrink_cards_to_budget(
    cards: List[Dict[str, Any]],
    *,
    max_cards_chars: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop lowest-ranked cards until serialized size fits within budget."""
    # Cards are already sorted best->worst. Drop from the end.
    cur = list(cards)
    cur_len = _json_len(cur)
    while cur and cur_len > max_cards_chars:
        cur.pop()
        cur_len = _json_len(cur)
    return cur, cur_len


def build_analyze_user_payload(
    *,
    analysis_focus: str,
    project_brief: str,
    project_meta: Any,
    structure_overview: Any,
    top_cards: List[Dict[str, Any]],
    max_cards: int = DEFAULT_MAX_CARDS,
    max_card_summary_chars: int = DEFAULT_MAX_CARD_SUMMARY_CHARS,
    max_structure_chars: int = DEFAULT_MAX_STRUCTURE_CHARS,
    max_meta_chars: int = DEFAULT_MAX_META_CHARS,
    overall_max_chars: int = DEFAULT_MAX_PAYLOAD_CHARS,
    include_trace: bool = False,
    allow_non_target_paths: bool = False,
    max_brief_chars: int = DEFAULT_MAX_BRIEF_CHARS,
    max_cards_chars: int = DEFAULT_MAX_CARDS_CHARS,
    research_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct a compact, LLM-friendly user payload for Analyze mode.

    Responsibilities:
      - deterministic card filtering/sorting/deduping
      - cap the number of cards
      - truncate card summaries
      - turn meta/structure into bounded-size JSON strings
      - enforce per-section and overall payload budgets
      - optionally attach a bounded DeepResearch context bundle under the key "research"

    Determinism notes:
      - Cards are sorted by (score desc, path asc) and deduped by normalized path.
      - Non-target artifacts (e.g. .aidev) are excluded unless allow_non_target_paths=True.

    Research bundle notes:
      - If research_bundle is provided and is a dict, it is attached as a pre-serialized
        JSON string using json.dumps(..., sort_keys=True, separators=(",",":"), ensure_ascii=False)
        to ensure deterministic, byte-identical serialization across callers.
      - If research_bundle is provided but is not a dict, it is coerced to a compact JSON string
        via _compact_json_blob() and attached under "research".
      - This function accounts for the research string's serialized size when enforcing the overall
        payload budget. If budget reductions are necessary, research is deterministically truncated
        as a last resort.

    Debugging/verification:
      - If include_trace=True, injects "_aidev_payload_trace" with per-section sizes.
    """
    # ---- cards: deterministic filter/sort/dedupe + truncate summaries ----
    prepared_cards = _prepare_cards(
        top_cards,
        max_cards=max_cards,
        max_card_summary_chars=max_card_summary_chars,
        allow_non_target_paths=allow_non_target_paths,
    )

    # ---- structure + meta blobs (bounded) ----
    structure_blob = _compact_json_blob(structure_overview, max_structure_chars)
    meta_blob = _compact_json_blob(project_meta, max_meta_chars)

    # ---- project brief (bounded) ----
    brief = project_brief or ""
    if len(brief) > max_brief_chars:
        # Keep deterministic truncation marker.
        brief = brief[: max(0, max_brief_chars - 20)] + "… (truncated)"

    # ---- enforce cards section budget by dropping lowest-ranked cards ----
    final_cards, cards_chars = _shrink_cards_to_budget(prepared_cards, max_cards_chars=max_cards_chars)

    payload: Dict[str, Any] = {
        "analysis_focus": analysis_focus,
        "project_brief": brief,
        "structure_overview": structure_blob,
        "project_meta": meta_blob,
        "top_cards": final_cards,
    }

    # ---- optional research bundle (bounded upstream; included only when supplied) ----
    # IMPORTANT: Do not expand/concatenate research internals here. The upstream builder is expected
    # to bound this object; we only account for its size in the overall payload budget.
    if research_bundle is not None:
        if isinstance(research_bundle, dict):
            # Requirement: deterministic, byte-identical serialization for dict research bundles.
            payload["research"] = json.dumps(
                research_bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        else:
            # Defensive: avoid crashes from unexpected types and cap via existing blob logic.
            payload["research"] = _compact_json_blob(research_bundle, max_structure_chars)

    # ---- prepare caps for iterative reduction, including research as a last-resort target ----
    def _payload_chars(p: Dict[str, Any]) -> int:
        return _json_len(p)

    def _truncate_str(s: str, max_len: int) -> str:
        if len(s) <= max_len:
            return s
        return s[: max(0, max_len - 20)] + "… (truncated)"

    # Iteratively reduce until within budget or nothing left to reduce.
    # Use fixed step reductions for determinism.
    step = 500
    min_floor = 200
    structure_cap = max(0, max_structure_chars)
    meta_cap = max(0, max_meta_chars)
    brief_cap = max(0, max_brief_chars)
    # Research cap: if research is present and a string, start from its current length.
    research_cap = 0
    if "research" in payload and isinstance(payload["research"], str):
        research_cap = len(payload["research"])

    # If already over budget, attempt reductions.
    for _ in range(200):
        total_chars = _payload_chars(payload)
        if total_chars <= overall_max_chars:
            break

        # 1) Drop more cards if possible (beyond max_cards_chars) by halving budget.
        if payload["top_cards"]:
            # Reduce cards budget gradually.
            max_cards_chars = max(min_floor, max_cards_chars - step)
            payload["top_cards"], cards_chars = _shrink_cards_to_budget(
                payload["top_cards"], max_cards_chars=max_cards_chars
            )
            continue

        # 2) Shrink structure/meta/brief strings.
        if structure_cap > min_floor:
            structure_cap = max(min_floor, structure_cap - step)
            payload["structure_overview"] = _truncate_str(str(payload["structure_overview"]), structure_cap)
            continue
        if meta_cap > min_floor:
            meta_cap = max(min_floor, meta_cap - step)
            payload["project_meta"] = _truncate_str(str(payload["project_meta"]), meta_cap)
            continue
        if brief_cap > min_floor:
            brief_cap = max(min_floor, brief_cap - step)
            payload["project_brief"] = _truncate_str(str(payload["project_brief"]), brief_cap)
            continue

        # 3) As a last resort, deterministically truncate the research string if present and a string.
        if research_cap > min_floor and "research" in payload and isinstance(payload["research"], str):
            research_cap = max(min_floor, research_cap - step)
            payload["research"] = _truncate_str(payload["research"], research_cap)
            continue

        # Nothing left to shrink meaningfully.
        break

    # ---- optional trace metadata (must not push payload over overall_max_chars) ----
    if include_trace:
        # Helper to measure payload size when a given trace is attached.
        def _measure_with_trace(p: Dict[str, Any], t: Dict[str, Any]) -> int:
            temp = dict(p)
            temp["_aidev_payload_trace"] = t
            return _json_len(temp)

        # Build a full-detail trace reflecting current state.
        def _build_full_trace() -> Dict[str, Any]:
            research_val = payload.get("research")
            if "research" in payload:
                if isinstance(research_val, str):
                    research_len = len(research_val)
                else:
                    research_len = _json_len(research_val)
            else:
                research_len = 0

            return {
                "overall_max_chars": overall_max_chars,
                "payload_chars": 0,  # placeholder, will be updated after measuring
                "section_chars": {
                    "analysis_focus": len(analysis_focus or ""),
                    "project_brief": len(payload.get("project_brief") or ""),
                    "structure_overview": len(payload.get("structure_overview") or ""),
                    "project_meta": len(payload.get("project_meta") or ""),
                    "top_cards": _json_len(payload.get("top_cards")),
                    # research is optional; when it's a pre-serialized string we report its raw length
                    "research": research_len,
                },
                "cards": {
                    "input_count": len(top_cards or []),
                    "prepared_count": len(prepared_cards),
                    "final_count": len(payload.get("top_cards") or []),
                    "max_cards": max_cards,
                    "max_cards_chars": max_cards_chars,
                    "allow_non_target_paths": allow_non_target_paths,
                },
                "research": {
                    "present": ("research" in payload),
                    # The upstream builder may or may not have bounded the object; we don't assume.
                    "is_bounded_by_builder": None,
                },
            }

        trace = _build_full_trace()
        total_with_trace = _measure_with_trace(payload, trace)

        # If adding the trace pushes us over overall_max_chars, attempt further
        # deterministic reductions to the payload first (same strategy as above),
        # then progressively compact the trace itself.
        if total_with_trace > overall_max_chars:
            for _ in range(200):
                # Try reducing payload further with the same deterministic steps.
                # NOTE: this includes the research section if present (we may mutate it here).
                current_total = _json_len(payload)
                approx_with_trace = current_total + _json_len(trace)
                if approx_with_trace <= overall_max_chars:
                    break

                if payload["top_cards"]:
                    max_cards_chars = max(min_floor, max_cards_chars - step)
                    payload["top_cards"], cards_chars = _shrink_cards_to_budget(
                        payload["top_cards"], max_cards_chars=max_cards_chars
                    )
                    # rebuild trace to reflect updated card counts
                    trace = _build_full_trace()
                    continue

                if structure_cap > min_floor:
                    structure_cap = max(min_floor, structure_cap - step)
                    payload["structure_overview"] = _truncate_str(str(payload["structure_overview"]), structure_cap)
                    trace = _build_full_trace()
                    continue

                if meta_cap > min_floor:
                    meta_cap = max(min_floor, meta_cap - step)
                    payload["project_meta"] = _truncate_str(str(payload["project_meta"]), meta_cap)
                    trace = _build_full_trace()
                    continue

                if brief_cap > min_floor:
                    brief_cap = max(min_floor, brief_cap - step)
                    payload["project_brief"] = _truncate_str(str(payload["project_brief"]), brief_cap)
                    trace = _build_full_trace()
                    continue

                # Try truncating research as a last-resort deterministic step.
                if research_cap > min_floor and "research" in payload and isinstance(payload["research"], str):
                    research_cap = max(min_floor, research_cap - step)
                    payload["research"] = _truncate_str(payload["research"], research_cap)
                    trace = _build_full_trace()
                    continue

                # Nothing more to shrink in payload; break and compact trace.
                break

            # Recompute measurement with updated payload and full trace
            trace = _build_full_trace()
            total_with_trace = _measure_with_trace(payload, trace)

        # If still too large, progressively compact the trace to preserve required keys
        if total_with_trace > overall_max_chars:
            # Compact the trace: remove verbose cards metadata, keep counts only.
            research_val = payload.get("research")
            if "research" in payload:
                if isinstance(research_val, str):
                    research_len = len(research_val)
                else:
                    research_len = _json_len(research_val)
            else:
                research_len = 0

            compact_trace = {
                "overall_max_chars": overall_max_chars,
                "payload_chars": 0,
                "section_chars": {
                    "analysis_focus": len(analysis_focus or ""),
                    "project_brief": len(payload.get("project_brief") or ""),
                    # keep only lengths for the largest blobs
                    "structure_overview": len(str(payload.get("structure_overview") or ""))
                    if payload.get("structure_overview")
                    else 0,
                    "project_meta": len(str(payload.get("project_meta") or "")) if payload.get("project_meta") else 0,
                    "top_cards": _json_len(payload.get("top_cards")),
                    "research": research_len,
                },
                "cards": {
                    "input_count": len(top_cards or []),
                    "prepared_count": len(prepared_cards),
                    "final_count": len(payload.get("top_cards") or []),
                },
                "research": {
                    "present": ("research" in payload),
                    "is_bounded_by_builder": None,
                },
            }
            total_with_compact = _measure_with_trace(payload, compact_trace)
            if total_with_compact <= overall_max_chars:
                trace = compact_trace
                total_with_trace = total_with_compact
            else:
                # Last-resort minimal trace: only declare overall cap and final payload size
                minimal_trace = {"overall_max_chars": overall_max_chars, "payload_chars": 0}
                total_with_minimal = _measure_with_trace(payload, minimal_trace)
                # populate payload_chars with the measured total (including the minimal trace)
                minimal_trace["payload_chars"] = total_with_minimal
                trace = minimal_trace
                total_with_trace = total_with_minimal

        # Finalize trace.payload_chars to reflect the real serialized size including trace
        final_total_with_trace = _measure_with_trace(payload, trace)
        trace["payload_chars"] = final_total_with_trace

        # Attach the deterministic, possibly compacted trace.
        payload["_aidev_payload_trace"] = trace

        # Ensure that the final payload (including trace) does not exceed overall_max_chars.
        # If it does (extremely unlikely after above steps), log a warning and still return the
        # payload with the compact/minimal trace so the caller can inspect sizes deterministically.
        final_check = _json_len(payload)
        if final_check > overall_max_chars:
            # Include whether research was present so regressions are obvious in logs.
            try:
                logger.warning(
                    "Analyze payload (with trace) exceeds overall_max_chars (%d > %d). final_check=%d research_present=%s",
                    overall_max_chars,
                    overall_max_chars,
                    final_check,
                    "research" in payload,
                )
            except Exception:
                pass
        else:
            try:
                logger.debug(
                    "Analyze payload final size (with trace): %d research_present=%s",
                    final_check,
                    "research" in payload,
                )
            except Exception:
                pass

    return payload
