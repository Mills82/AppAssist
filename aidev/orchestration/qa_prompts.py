# aidev/orchestration/qa_prompts.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from ..llm_client import system_preset

DEFAULT_MAX_CARD_SUMMARY_CHARS = 800
DEFAULT_MAX_STRUCTURE_CHARS = 8000
DEFAULT_MAX_META_CHARS = 4000
DEFAULT_MAX_CARDS = 10


def qa_system_prompt() -> str:
    """
    System prompt for Q&A mode.

    Prefer loading from /prompts via system_preset("qa"), with a small
    baked-in fallback if no prompt file is found.
    """
    # Maintainers: prompt must list keys expected by aidev/schemas/qa_answer.schema.json
    # and prefer JSON-only output so qa_mixin can validate and emit SSE events.
    try:
        txt = system_preset("qa")
        if txt:
            return txt
    except Exception:
        # Fall through to builtin fallback
        pass

    # Fallback prompt if no system.qa.md / qa.md is present
    return (
        "You are an AI developer assistant for this specific repository.\n"
        "\n"
        "You answer ONE question about the project at a time (purpose, architecture,\n"
        "important files, how pieces fit).\n"
        "\n"
        "STYLE:\n"
        "- Assume the user is a capable developer but new to THIS codebase.\n"
        "- Start with ONE short, direct sentence that answers the question.\n"
        "- Then list 3–5 concrete items (files, components, steps) in bullet form.\n"
        "- Avoid long prose, nested lists, or checklists.\n"
        "- DO NOT ask the user to run commands, paste file listings, or provide more data.\n"
        "- DO NOT restate the question or add meta commentary like 'If you want, I can...'.\n"
        "\n"
        "LENGTH:\n"
        "- Max ~150 words total.\n"
        "- No more than 6 bullet points.\n"
        "- If you’re unsure, say so briefly, but still give your best guess.\n"
        "\n"
        "OUTPUT:\n"
        "- You will receive a JSON payload with `question`, `project_brief`,\n"
        "  `structure_overview`, `project_meta`, and `top_cards`.\n"
        "- Respond with a single, valid JSON object and nothing else. Do NOT wrap the JSON in\n"
        "  markdown fences, code blocks, or add surrounding prose.\n"
        "- The JSON MUST contain these top-level keys: `answer` (string), `files` (array of\n"
        "  objects, each with at least `path` and optional `excerpt`), and `follow_ups`\n"
        "  (array of strings). For missing data use null or empty arrays; do NOT omit keys.\n"
        "- If you cannot strictly follow a JSON-only response, still place a valid JSON\n"
        "  object as the very first token(s) of your response (no leading text, no fences).\n"
        "  Any additional commentary may only appear after a blank line following that JSON\n"
        "  object. The consumer will parse the leading JSON object.\n"
        "- Keep the `answer` concise and developer-focused.\n"
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



def build_qa_user_payload(
    *,
    question: str,
    project_brief: str,
    project_meta: Any,
    structure_overview: Any,
    top_cards: List[Dict[str, Any]],
    max_cards: int = DEFAULT_MAX_CARDS,
    max_card_summary_chars: int = DEFAULT_MAX_CARD_SUMMARY_CHARS,
    max_structure_chars: int = DEFAULT_MAX_STRUCTURE_CHARS,
    max_meta_chars: int = DEFAULT_MAX_META_CHARS,
) -> Dict[str, Any]:
    """
    Construct a compact, LLM-friendly user payload for Q&A mode.

    This function is responsible for:
      - capping the number of cards
      - truncating card summaries
      - turning meta/structure into bounded-size strings
    """
    # ---- cards: cap count + truncate summaries ----
    trimmed_cards: List[Dict[str, Any]] = []
    for card in (top_cards or [])[: max_cards]:
        if not isinstance(card, dict):
            continue
        summary = card.get("summary")
        if isinstance(summary, str) and len(summary) > max_card_summary_chars:
            summary = summary[: max_card_summary_chars] + "…"
        trimmed_cards.append(
            {
                "path": card.get("path"),
                "title": card.get("title"),
                "summary": summary,
                "language": card.get("language"),
                "score": card.get("score"),
            }
        )

    # ---- structure + meta blobs ----
    structure_blob = _compact_json_blob(structure_overview, max_structure_chars)
    meta_blob = _compact_json_blob(project_meta, max_meta_chars)

    # Optionally cap project brief as well, though it is usually short.
    brief = project_brief or ""
    if len(brief) > 4000:
        brief = brief[:3980] + "… (truncated)"

    return {
        "question": question,
        "project_brief": brief,
        "structure_overview": structure_blob,
        "project_meta": meta_blob,
        "top_cards": trimmed_cards,
    }
