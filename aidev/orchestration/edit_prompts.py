# aidev/orchestration/edit_prompts.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import os

from ..llm_client import system_preset, INCREMENTAL_GUIDELINES

# Defaults for recommendation budget shaping
DEFAULT_MAX_ITEMS = 12
DEFAULT_MAX_CHARS_PER_ITEM = 1200
DEFAULT_MAX_CONTEXT_CHARS = 120_000

# How aggressively to trim project_map summaries for LLM payloads
DEFAULT_MAX_PROJECT_MAP_SUMMARY_LEN = 300
DEFAULT_MAX_PROJECT_MAP_FILES = 2000  # just a safety valve for enormous repos


# Prompt preset identifier constants for the tiered edit-file strategy.
# These MUST be preset keys accepted by system_preset(...) and present in _PROMPT_ALIASES.
#
# PATCH_CAPABLE_PROMPT_NAME corresponds to the prompt that permits either
# `patch_unified` (preferred) or full `content` in the model response.
PATCH_CAPABLE_PROMPT_NAME = "edit_file"

# FULL_FILE_PROMPT_NAME corresponds to the prompt that requires full `content`
# only and forbids returning patch fields. Use this as the guaranteed fallback.
FULL_FILE_PROMPT_NAME = "edit_file_full"


def _slim_project_map_for_llm(
    project_meta: Dict[str, Any],
    *,
    max_summary_len: int = DEFAULT_MAX_PROJECT_MAP_SUMMARY_LEN,
    max_files: int = DEFAULT_MAX_PROJECT_MAP_FILES,
) -> None:
    """
    If project_meta contains a 'project_map' key, replace it with a
    slimmed-down, LLM-friendly version based on the v2 repo map.

    Mutates project_meta in place; no-op if project_map is missing or malformed.
    """
    pm = project_meta.get("project_map")
    if not isinstance(pm, dict):
        return

    files = pm.get("files") or []
    if not isinstance(files, list):
        files = []

    slim_files: List[Dict[str, Any]] = []
    for i, f in enumerate(files):
        if i >= max_files:
            break
        if not isinstance(f, dict):
            continue
        path = f.get("path")
        if not isinstance(path, str) or not path:
            continue
        kind = f.get("kind")
        if not isinstance(kind, str) or not kind:
            kind = "other"
        summary = f.get("summary") or ""
        if isinstance(summary, str) and summary:
            summary = summary[:max_summary_len]
        else:
            summary = ""

        slim_entry: Dict[str, Any] = {
            "path": path,
            "kind": kind,
        }
        if summary:
            slim_entry["summary"] = summary
        if f.get("changed"):
            slim_entry["changed"] = True

        slim_files.append(slim_entry)

    slim_map: Dict[str, Any] = {
        "version": pm.get("version", 1),
        "root": pm.get("root"),
        "total_files": pm.get("total_files"),
        "language_kinds": pm.get("language_kinds"),
        "by_ext": pm.get("by_ext"),
        "by_top": pm.get("by_top"),
        "files": slim_files,
    }

    project_meta["project_map"] = slim_map


def _build_context_excerpt(
    project_meta: Dict[str, Any],
    max_context_chars: int,
    max_files: int = 100,
) -> str:
    """
    Build the 'context.excerpt' string for the recommendations payload.
    Token-lean, structural listing derived from project_map.
    """
    pm = project_meta.get("project_map")
    if not isinstance(pm, dict):
        return ""

    files = pm.get("files") or []
    if not isinstance(files, list) or not files:
        return ""

    def _sort_key(f: Dict[str, Any]) -> tuple:
        changed = bool(f.get("changed"))
        path = str(f.get("path") or "")
        return (0 if changed else 1, path)

    ordered = sorted(
        (f for f in files if isinstance(f, dict) and f.get("path")),
        key=_sort_key,
    )

    lines: List[str] = []
    lines.append("PROJECT STRUCTURE (subset):")

    for i, f in enumerate(ordered):
        if i >= max_files:
            lines.append("... (more files omitted)")
            break
        path = str(f.get("path") or "").strip()
        summary = str(f.get("summary") or "").strip()
        changed = bool(f.get("changed"))

        prefix = "- "
        if changed:
            prefix = "- [changed] "

        if summary:
            lines.append(f"{prefix}{path}: {summary}")
        else:
            lines.append(f"{prefix}{path}")

    text = "\n".join(lines)
    return text[:max_context_chars]


def recommendations_system_prompt() -> str:
    """
    System prompt for the recommendations (edit) pipeline.

    Prefer loading from /prompts via system_preset("recommendations"), with a
    small baked-in fallback if no prompt file is found.
    """
    try:
        txt = system_preset("recommendations")
        if txt:
            return txt
    except Exception:
        pass

    return INCREMENTAL_GUIDELINES


def _summarize_focus_for_recs(developer_focus: str) -> str:
    dev_raw = (developer_focus or "").strip()
    if dev_raw:
        return dev_raw
    return "Identify the highest-impact, concrete improvements for this project."


def _compute_deep_research_digest(research_brief: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a compact digest from a deep research brief.

    Returns {} on any parsing issue.

    DECIDED output shape (when non-empty) to match recommendations.py and
    the acceptance criteria:
      {
        "top_findings": ["Title1", ...],  # up to top 5 finding titles
        "referenced_paths": [{"path": "...", "lines": "12-18"|"42"|None}, ...]
      }

    This function recognizes multiple evidence line-range field variants
    (preformatted 'lines', 'line_start'/'line_end', and 'start_line'/'end_line').
    """

    def _format_lines(e: Dict[str, Any]) -> Optional[str]:
        try:
            # Some schemas provide a pre-formatted 'lines' field
            if "lines" in e and e.get("lines") is not None:
                v = e.get("lines")
                s = str(v).strip()
                return s or None

            # Recognize both variants: line_start/line_end and start_line/end_line
            s_raw = None
            t_raw = None
            for a, b in (("line_start", "line_end"), ("start_line", "end_line")):
                if a in e or b in e:
                    s_raw = e.get(a)
                    t_raw = e.get(b)
                    break

            s_int: Optional[int] = None
            t_int: Optional[int] = None
            try:
                if s_raw is not None and str(s_raw).strip() != "":
                    s_int = int(s_raw)
            except Exception:
                s_int = None
            try:
                if t_raw is not None and str(t_raw).strip() != "":
                    t_int = int(t_raw)
            except Exception:
                t_int = None

            if s_int is not None and t_int is not None:
                return str(s_int) if s_int == t_int else f"{s_int}-{t_int}"
            if s_int is not None:
                return str(s_int)
            if t_int is not None:
                return str(t_int)
        except Exception:
            return None

        return None

    try:
        if not isinstance(research_brief, dict):
            return {}

        # Findings / results can vary by schema
        findings_raw = research_brief.get("findings")
        if not isinstance(findings_raw, list):
            findings_raw = research_brief.get("results")
        if not isinstance(findings_raw, list):
            findings_raw = []

        top_findings: List[str] = []
        for f in findings_raw:
            if len(top_findings) >= 5:
                break
            if isinstance(f, str):
                t = f.strip()
                if t:
                    top_findings.append(t)
                continue
            if isinstance(f, dict):
                t = f.get("title") or f.get("name") or f.get("summary")
                if isinstance(t, str):
                    t2 = t.strip()
                    if t2:
                        top_findings.append(t2)

        if not top_findings:
            return {}

        # Collect evidence from multiple plausible locations
        evidence_items: List[Dict[str, Any]] = []

        for container in (research_brief,):
            ev = container.get("evidence") if isinstance(container, dict) else None
            if isinstance(ev, list):
                evidence_items.extend([x for x in ev if isinstance(x, dict)])
            ev2 = container.get("evidences") if isinstance(container, dict) else None
            if isinstance(ev2, list):
                evidence_items.extend([x for x in ev2 if isinstance(x, dict)])

        # Also harvest evidence nested under findings/results
        for f in findings_raw:
            if not isinstance(f, dict):
                continue
            for k in ("evidence", "evidences"):
                ev = f.get(k)
                if isinstance(ev, list):
                    evidence_items.extend([x for x in ev if isinstance(x, dict)])

        seen: set[tuple[str, Optional[str]]] = set()
        referenced_paths: List[Dict[str, Any]] = []

        for e in evidence_items:
            try:
                p = e.get("path") or e.get("file_path") or e.get("source")
                if not isinstance(p, str):
                    continue
                path = p.strip()
                if not path:
                    continue

                lines = _format_lines(e)
                key = (path, lines)
                if key in seen:
                    continue
                seen.add(key)

                entry: Dict[str, Any] = {"path": path, "lines": lines}
                referenced_paths.append(entry)
            except Exception:
                continue

        return {
            "top_findings": top_findings,
            "referenced_paths": referenced_paths,
        }
    except Exception:
        return {}


def build_recommendations_user_payload(
    *,
    project_brief_text: str,
    meta: Dict[str, Any],
    developer_focus: str,
    budget_limits: Dict[str, int],
    strategy_note: Optional[str] = None,
    deep_research_brief: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dev_focus_str = _summarize_focus_for_recs(developer_focus)

    project_meta: Dict[str, Any] = dict(meta or {})
    _slim_project_map_for_llm(project_meta)

    budget: Dict[str, int] = dict(budget_limits or {})
    try:
        max_items = int(budget.get("max_items", DEFAULT_MAX_ITEMS))
    except Exception:
        max_items = DEFAULT_MAX_ITEMS

    try:
        max_chars_per_item = int(budget.get("max_chars_per_item", DEFAULT_MAX_CHARS_PER_ITEM))
    except Exception:
        max_chars_per_item = DEFAULT_MAX_CHARS_PER_ITEM

    try:
        max_context_chars = int(budget.get("max_context_chars", DEFAULT_MAX_CONTEXT_CHARS))
    except Exception:
        max_context_chars = DEFAULT_MAX_CONTEXT_CHARS

    budget["max_items"] = max_items
    budget["max_chars_per_item"] = max_chars_per_item
    budget["max_context_chars"] = max_context_chars

    context_excerpt = _build_context_excerpt(
        project_meta=project_meta,
        max_context_chars=max_context_chars,
    )

    # Normalize strategy_note (keep it small + clean)
    strategy_note_str: Optional[str] = None
    try:
        s = (strategy_note or "").strip()
        if s:
            strategy_note_str = s[:5000]
    except Exception:
        strategy_note_str = None

    envelope: Dict[str, Any] = {
        "schema_version": 1,
        "project": {
            "summary": project_brief_text or "",
            "meta": project_meta,
        },
        "run": {
            "developer_focus": dev_focus_str,
            "budget_limits": budget,
            **({"strategy_note": strategy_note_str} if strategy_note_str else {}),  # <-- ADD
        },
        "context": {
            "excerpt": context_excerpt,
        },
        "debug": {
            "source": "aidev.orchestration.edit_prompts",
            "run_id": project_meta.get("run_id"),
        },
    }

    # Optional: attach deep research digest without changing default payload shape
    try:
        if isinstance(deep_research_brief, dict) and deep_research_brief:
            digest = _compute_deep_research_digest(deep_research_brief)
            # Use the decided canonical schema keys: top_findings + referenced_paths
            if isinstance(digest, dict) and digest.get("top_findings"):
                envelope["deep_research_digest"] = digest
    except Exception:
        pass

    try:
        if not envelope.get("incremental_guidelines"):
            envelope["incremental_guidelines"] = INCREMENTAL_GUIDELINES
    except Exception:
        pass

    return envelope


def get_edit_prompt(
    prompt_name: Optional[str] = None,
    *,
    patch_capable: Optional[bool] = None,
) -> Dict[str, Optional[str]]:
    """
    Resolve an edit prompt preset key to its system prompt text and the model id
    to use for edit-generation calls.

    IMPORTANT: `prompt_name` must be a preset KEY recognized by system_preset(...)
    (i.e., present in _PROMPT_ALIASES), not a filename.
    """
    # Determine chosen preset key
    if prompt_name:
        chosen = str(prompt_name)
    else:
        if patch_capable is True:
            chosen = PATCH_CAPABLE_PROMPT_NAME
        elif patch_capable is False:
            chosen = FULL_FILE_PROMPT_NAME
        else:
            chosen = PATCH_CAPABLE_PROMPT_NAME

    # Load prompt text via system_preset (preset keys only)
    prompt_text: Optional[str]
    try:
        prompt_text = system_preset(chosen)
        if prompt_text is None:
            prompt_text = None
    except Exception:
        prompt_text = None

    # Resolve model id (tiered): prefer per-attempt env vars if available
    # patch-capable -> AIDEV_MODEL_EDIT_PATCH
    # full-file      -> AIDEV_MODEL_EDIT_FULL
    # fallback       -> AIDEV_MODEL_GENERATE_EDITS
    model_env_key = "AIDEV_MODEL_GENERATE_EDITS"
    if patch_capable is True:
        model_env_key = "AIDEV_MODEL_EDIT_PATCH"
    elif patch_capable is False:
        model_env_key = "AIDEV_MODEL_EDIT_FULL"

    model: Optional[str] = None

    # Try aidev.config first (import-safe), then env vars
    try:
        import aidev.config as _cfg  # type: ignore

        maybe = getattr(_cfg, model_env_key, None)
        if isinstance(maybe, str) and maybe:
            model = maybe
        else:
            maybe2 = getattr(_cfg, "CONFIG", None) or getattr(_cfg, "settings", None)
            if isinstance(maybe2, dict):
                maybe3 = maybe2.get(model_env_key)
                if isinstance(maybe3, str) and maybe3:
                    model = maybe3
    except Exception:
        model = None

    if not model:
        model = os.environ.get(model_env_key) or os.environ.get("AIDEV_MODEL_GENERATE_EDITS")

    return {"prompt_name": chosen, "prompt_text": prompt_text, "model": model}


def get_patch_capable_prompt(file_path: str, file_current: str) -> Dict[str, Any]:
    """
    Return a small descriptor for a patch-capable edit LLM call.
    """
    info = get_edit_prompt(prompt_name=PATCH_CAPABLE_PROMPT_NAME, patch_capable=True)
    return {
        "system": PATCH_CAPABLE_PROMPT_NAME,  # preset key
        "prompt_name": info.get("prompt_name"),
        "prompt_text": info.get("prompt_text"),
        "model": info.get("model"),
        "file_path": file_path,
        "file_current": file_current,
    }


def get_full_file_prompt(file_path: str, file_current: str) -> Dict[str, Any]:
    """
    Return a small descriptor for a full-file-only edit LLM call.
    """
    info = get_edit_prompt(prompt_name=FULL_FILE_PROMPT_NAME, patch_capable=False)
    return {
        "system": FULL_FILE_PROMPT_NAME,  # preset key
        "prompt_name": info.get("prompt_name"),
        "prompt_text": info.get("prompt_text"),
        "model": info.get("model"),
        "file_path": file_path,
        "file_current": file_current,
    }
