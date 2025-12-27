# aidev/stages/analyze_file.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import ast
import logging
import re

from ..cards import KnowledgeBase
from ..context.edit_context import build_context_bundle_for_paths
from ..llm_client import system_preset
from ..schemas import target_analysis_schema

JsonSchema = Dict[str, Any]
# ChatJsonFn: wrapper used to call the orchestrator LLM.
# The runtime wrapper evolved: older clients accepted a positional signature
# (system_text, payload, schema, temp, tag, stage, max_tokens) while newer
# clients accept stage and max_tokens as keywords. To remain compatible with
# both, use a permissive Callable type and handle both call forms at runtime.
# Expected runtime usage (preferred):
#   chat_json_fn(system_text, payload, schema, temp, tag, stage=<stage>, max_tokens=<int>)
# Legacy positional usage is supported via a TypeError fallback.
ChatJsonFn = Callable[..., Tuple[Any, Any]]

# Hard-ish caps to keep the analyze payload compact.
MAX_ANALYZE_FILE_BYTES = 150_000
# Much smaller per-snippet budget for neighbor context.
MAX_SNIPPET_CHARS = 1500
MAX_SNIPPETS_PER_FILE = 8  # allow a couple “contract” snippets without losing local neighbors
MAX_EXTRA_CONTEXT_PATHS = 5  # targeted schema/events/engine files (kept small)
MAX_PROJECT_BRIEF_CHARS = 4_000


@dataclass
class PerFileAnalysis:
    """
    Per-file result from the analyze stage.

    This object always contains a stable `analysis` dict with a predictable
    shape so downstream aggregation logic can rely on keys being present.

    Expected minimal analysis shape (defaults are used when the LLM call
    fails or returns unexpected data):

    {
      "schema_version": 2,
      "rec_id": "<recommendation id or 'rec'>",
      "focus": "<run focus string>",
      "path": "repo/relative/path",
      "role": "unknown",                # string describing the file's role
      "should_edit": False,              # bool
      "local_plan": "...",             # string or list
      "issues": [],                      # list of issue dicts or strings
      "suggestions": [],                 # list
      "recommended_actions": [],        # list
      "failures": [],                   # list of {path, error_type, message, partial?}
      # optional metadata
      "llm_model": "<model-name>" or None,
      "notes": "Human-friendly note about fallback or partial data",
    }

    The driver and tests expect these core keys to exist; when the LLM fails
    we populate sensible defaults rather than returning None so aggregation
    can proceed.
    """

    path: str
    analysis: Dict[str, Any]
    cross_file_notes: Optional[str]
    updated_targets: List[Dict[str, Any]]


def _trim_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]..."


def _normalize_to_lf(text: str) -> str:
    """
    Normalize all newlines to LF so the model sees consistent content.
    """
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _safe_parse_action_obj(s: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort parse for:
      - JSON object strings
      - Python dict literal strings (your current "{'type': ..., 'path': ...}" format)
    """
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    if not (t.startswith("{") and t.endswith("}")):
        return None

    # Try JSON first
    try:
        import json

        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try Python literal
    try:
        obj = ast.literal_eval(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def _extract_context_hint_paths(rec: Dict[str, Any]) -> List[str]:
    """
    Pull high-signal context paths from the recommendation itself:
      - rec.dependencies (list)
      - rec.actions / rec.steps (list of dicts or stringified dicts)
    """
    out: List[str] = []

    def _add(p: Any) -> None:
        if not isinstance(p, str):
            return
        p = p.strip().replace("\\", "/")
        if not p or p.startswith(("/", "\\")) or ".." in p:
            return
        out.append(p)

    deps = rec.get("dependencies")
    if isinstance(deps, (list, tuple)):
        for d in deps:
            _add(d)

    actions = rec.get("actions") or rec.get("steps") or []
    if isinstance(actions, str):
        actions = [actions]
    if isinstance(actions, (list, tuple)):
        for a in actions:
            if isinstance(a, dict):
                _add(a.get("path"))
            elif isinstance(a, str):
                obj = _safe_parse_action_obj(a)
                if isinstance(obj, dict):
                    _add(obj.get("path"))
                    continue
                # last-ditch regex extraction
                m = re.search(r"['\"]path['\"]\s*:\s*['\"]([^'\"]+)['\"]", a)
                if m:
                    _add(m.group(1))

    # de-dup preserve order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _build_project_map_excerpt(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a slim project-map snapshot for the analyze prompt.

    v2 behavior:
      - Prefer meta["project_map"] when present (repo_map.build_project_map output).
      - Fall back to meta directly (for older callers).
      - Only include tiny, aggregate info: totals + language/extension/top-dir
        breakdowns, plus small-by-kind / by-tag counts derived from files[].

    We intentionally keep this tiny: it's only meant to give the model a
    high-level sense of repository size / shape, not a full structure dump.
    """
    if not isinstance(meta, dict):
        return {}

    # Prefer the canonical project_map attached by orchestrator; fall back to
    # the top-level meta shape if that's all we have.
    src = meta.get("project_map")
    if not isinstance(src, dict):
        src = meta

    out: Dict[str, Any] = {}

    # Core aggregates from repo_map.build_project_map
    for key in ("total_files", "language_kinds", "by_ext", "by_top"):
        val = src.get(key)
        if val is not None:
            out[key] = val

    # Tiny derived aggregates from files[]: by_kind + by_tag.
    files = src.get("files")
    if isinstance(files, list):
        by_kind: Dict[str, int] = {}
        by_tag: Dict[str, int] = {}

        for f in files:
            if not isinstance(f, dict):
                continue

            k = f.get("kind")
            if isinstance(k, str) and k:
                by_kind[k] = by_kind.get(k, 0) + 1

            tags = f.get("tags") or []
            if isinstance(tags, list):
                for t in tags:
                    if isinstance(t, str) and t:
                        by_tag[t] = by_tag.get(t, 0) + 1

        if by_kind:
            out["by_kind"] = by_kind

        if by_tag:
            # Keep only the top few tags to stay token-lean.
            top_items = sorted(
                by_tag.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[:8]
            out["by_tag"] = {k: v for k, v in top_items}

    return out


def _build_file_payload(
    *,
    kb: Optional[KnowledgeBase],
    project_root: Path,
    path: str,
    extra_context_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build the single `file` object for the analyze payload.

    Includes:
      - `content`: full or trimmed primary file contents.
      - `context_snippets`: tiny neighbour hints derived from the shared
        context-bundle helper so analyze + edit see a consistent notion of
        neighbours / related files.
    """
    rel = path.strip()
    if not rel:
        return {"path": "", "content": "", "context_snippets": []}

    project_root = project_root.resolve()

    # Try to use the shared context bundle helper so we stay in sync with edit.
    try:
        hint_paths: List[str] = []
        if isinstance(extra_context_paths, list):
            for p in extra_context_paths:
                if isinstance(p, str):
                    pp = p.strip().replace("\\", "/")
                    if pp and pp != rel:
                        hint_paths.append(pp)
        hint_paths = hint_paths[:MAX_EXTRA_CONTEXT_PATHS]

        bundles = build_context_bundle_for_paths(
            kb=kb,
            project_root=project_root,
            paths=[rel] + hint_paths,
            max_neighbors=8,
            max_tests=4,
            max_bytes_per_file=MAX_ANALYZE_FILE_BYTES,
        )
    except Exception:
        logging.exception(
            "[analyze_file] build_context_bundle_for_paths failed; "
            "falling back to simple file read."
        )
        # Fallback: best-effort minimal payload – just primary file contents,
        # no rich neighbour info.
        abs_path = (project_root / rel).resolve()
        try:
            text = abs_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        text = _normalize_to_lf(text)
        return {
            "path": rel,
            "content": _trim_text(text, MAX_ANALYZE_FILE_BYTES),
            "context_snippets": [],
        }

    # Resolve bundle for this path; the helper may normalize keys.
    key = rel.replace("\\", "/")
    bundle = bundles.get(key) or bundles.get(rel)
    if bundle is None:
        # Last-resort: match by normalized key against whatever the helper used.
        for b_key, b_val in bundles.items():
            try:
                if str(b_key).replace("\\", "/") == key:
                    bundle = b_val
                    break
            except Exception:
                continue

    if not bundle:
        # No bundle (e.g., file missing); still send an entry with empty content.
        return {"path": rel, "content": "", "context_snippets": []}

    # Pick primary file.
    primary = None
    for cf in bundle:
        if getattr(cf, "role", None) == "primary":
            primary = cf
            break
    if primary is None:
        primary = bundle[0]

    primary_content = _normalize_to_lf(getattr(primary, "content", "") or "")
    content = _trim_text(primary_content, MAX_ANALYZE_FILE_BYTES)

    def _classify_kind(p: str, fallback_kind: str) -> str:
        pp = (p or "").replace("\\", "/").lower()
        if "/schemas/" in pp or pp.endswith(".schema.json"):
            return "schema"
        if "events" in pp and pp.endswith(".py"):
            return "events"
        if "deep_research" in pp:
            return "engine"
        if "/test" in pp or pp.endswith("_test.py") or pp.endswith(".spec.ts") or pp.endswith(".test.ts"):
            return "test"
        return fallback_kind or "neighbor"

    def _score(kind: str, p: str) -> int:
        k = (kind or "").lower()
        if k == "schema":
            return 100
        if k == "events":
            return 90
        if k == "engine":
            return 85
        if k == "test":
            return 80
        return 50

    candidates: List[Dict[str, str]] = []

    # 1) Neighbor snippets (including tests) for the primary file
    for cf in bundle:
        if cf is primary:
            continue

        cf_path = str(getattr(cf, "path", "") or "").strip()
        if not cf_path:
            continue

        summary = getattr(cf, "summary", None)
        cf_content = getattr(cf, "content", "") or ""
        snippet_source = summary if isinstance(summary, str) and summary.strip() else cf_content
        snippet_source = _normalize_to_lf(snippet_source)
        snippet = _trim_text(snippet_source, MAX_SNIPPET_CHARS)
        if not snippet.strip():
            continue

        fallback_kind = str(getattr(cf, "role", None) or getattr(cf, "kind", None) or "neighbor")
        kind = _classify_kind(cf_path, fallback_kind)
        candidates.append({"path": cf_path, "kind": kind, "snippet": snippet})

    # 2) Explicit “contract” context files passed via extra_context_paths
    if isinstance(extra_context_paths, list):
        for p in extra_context_paths[:MAX_EXTRA_CONTEXT_PATHS]:
            if not isinstance(p, str):
                continue
            p = p.strip().replace("\\", "/")
            if not p or p == rel:
                continue

            b2 = bundles.get(p) or bundles.get(p.replace("\\", "/"))
            if not b2:
                continue

            p_primary = None
            for cf in b2:
                if getattr(cf, "role", None) == "primary":
                    p_primary = cf
                    break
            if p_primary is None and len(b2) > 0:
                p_primary = b2[0]
            if not p_primary:
                continue

            p_content = getattr(p_primary, "content", "") or ""
            p_summary = getattr(p_primary, "summary", None)
            src = p_summary if isinstance(p_summary, str) and p_summary.strip() else p_content
            src = _normalize_to_lf(src)
            snippet = _trim_text(src, MAX_SNIPPET_CHARS)
            if not snippet.strip():
                continue

            kind = _classify_kind(p, "context")
            candidates.append({"path": p, "kind": kind, "snippet": snippet})

    # De-dup by path, keep highest score
    best: Dict[str, Dict[str, str]] = {}
    for c in candidates:
        p = c["path"].replace("\\", "/")
        cur = best.get(p)
        if cur is None or _score(c.get("kind", ""), p) > _score(cur.get("kind", ""), p):
            best[p] = c

    ranked = sorted(best.values(), key=lambda c: (-_score(c.get("kind", ""), c.get("path", "")), c.get("path", "")))
    context_snippets = ranked[:MAX_SNIPPETS_PER_FILE]

    return {
        "path": rel,
        "content": content,
        "context_snippets": context_snippets,
    }


def _load_target_analysis_schema() -> JsonSchema:
    try:
        raw = target_analysis_schema()
        if isinstance(raw, dict):
            return raw
    except Exception:
        logging.exception("[analyze_file] failed to load target_analysis_schema")
    # Caller will treat an empty dict as "no schema" if desired.
    return {}


def analyze_file(
    *,
    rec: Dict[str, Any],
    kb: Optional[KnowledgeBase],
    project_root: Path,
    path: str,
    focus: str,
    chat_json_fn: ChatJsonFn,
    max_tokens: Optional[int] = None,
    project_brief_text: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    target_meta: Optional[Dict[str, Any]] = None,
) -> Optional[PerFileAnalysis]:
    """
    Run the intermediate 'analyze file' LLM stage for a single target file and
    recommendation.

    Inputs:
      - rec:          recommendation object (full, but we send a compact view).
      - kb:           KnowledgeBase for neighbour discovery (may be None).
      - project_root: Path to the repository root.
      - path:         repo-relative path of the target file.
      - focus:        high-level run focus string.
      - chat_json_fn: Orchestrator-style ChatJsonFn wrapper.
      - project_brief_text: optional project brief markdown/text.
      - meta:         optional project-map / structure summary used to build
                      a slim project_map_excerpt.
      - target_meta:  optional per-file target metadata from the selection
                      stage (e.g., original intent, notes).

    Returns:
      PerFileAnalysis on success. This function is defensive: it will not
      return None even if the LLM call fails or returns unexpected data. In
      those cases a fallback analysis dict with sensible defaults will be
      returned. Callers should still treat analysis contents as best-effort.
    """
    rid = str(rec.get("id") or "rec")
    rel_path = path.strip()

    # Local helper to construct a defensive fallback analysis dict when the LLM
    # call fails or returns unexpected data. Tests and the driver rely on the
    # presence of the following keys (see file_local_plan for details).
    def _make_fallback_analysis(
        note: Optional[str] = None,
        partial: Optional[str] = None,
        model: Optional[str] = None,
        error_type: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        msg = (message or note or "see logs").strip() if isinstance((message or note), str) else (message or note or "see logs")
        msg = _trim_text(str(msg), 400)
        return {
            "schema_version": 2,
            "rec_id": rid,
            "focus": focus,
            "path": rel_path,
            "role": "unknown",
            "should_edit": False,
            "local_plan": "No reliable per-file analysis available due to LLM failure; no edits recommended.",
            "issues": [],
            "suggestions": [],
            "recommended_actions": [],
            "failures": [
                {
                    "path": rel_path,
                    "error_type": (error_type or "unknown_error"),
                    "message": msg,
                    "partial": _trim_text(partial or "", 1000) if partial else None,
                }
            ],
            "llm_model": model if model else None,
            "notes": note or "LLM call failed or returned unexpected data; see logs for details",
        }

    if not rel_path:
        logging.warning("[analyze_file] empty path for rec_id=%s; producing fallback analysis.", rid)
        fallback = _make_fallback_analysis(
            note="Empty path provided to analyze_file; no analysis performed.",
            error_type="invalid_path",
            message="Empty path provided to analyze_file",
        )
        fallback["local_plan"] = "No reliable per-file analysis available due to empty path; no edits recommended."
        return PerFileAnalysis(path=rel_path, analysis=fallback, cross_file_notes=None, updated_targets=[])

    extra_context_paths = _extract_context_hint_paths(rec)

    file_payload = _build_file_payload(
        kb=kb,
        project_root=project_root,
        path=rel_path,
        extra_context_paths=extra_context_paths,
    )

    # If the file appears missing (no content and path doesn't exist), short-circuit
    # with a standardized failure record rather than calling the LLM.
    try:
        abs_path = (project_root.resolve() / rel_path).resolve()
        exists = abs_path.exists()
    except Exception:
        exists = False

    if (file_payload.get("content") or "") == "" and not exists:
        logging.warning("[analyze_file] file missing for rec_id=%s path=%s", rid, rel_path)
        fallback = _make_fallback_analysis(
            note="Target file missing; no analysis performed.",
            error_type="file_missing",
            message="Target file does not exist on disk",
        )
        fallback["local_plan"] = "Target file missing; no edits recommended."
        return PerFileAnalysis(path=rel_path, analysis=fallback, cross_file_notes=None, updated_targets=[])

    project_brief = (project_brief_text or "").strip()
    if project_brief and len(project_brief) > MAX_PROJECT_BRIEF_CHARS:
        project_brief = _trim_text(project_brief, MAX_PROJECT_BRIEF_CHARS)
    if project_brief:
        project_brief = _normalize_to_lf(project_brief)

    # Compact rec stub for the prompt.
    actions = rec.get("actions") or rec.get("steps") or []

    if isinstance(actions, str):
        actions_raw = [actions]
    elif isinstance(actions, (list, tuple)):
        actions_raw = list(actions)
    else:
        actions_raw = []

    actions_structured: List[Any] = []
    actions_summary: List[str] = []

    for a in actions_raw:
        if isinstance(a, dict):
            actions_structured.append(a)
            t = str(a.get("type") or a.get("intent") or "action")
            p = str(a.get("path") or "").strip()
            actions_summary.append(f"{t}: {p}" if p else t)
        elif isinstance(a, str):
            obj = _safe_parse_action_obj(a)
            if isinstance(obj, dict):
                actions_structured.append(obj)
                t = str(obj.get("type") or obj.get("intent") or "action")
                p = str(obj.get("path") or "").strip()
                actions_summary.append(f"{t}: {p}" if p else t)
            else:
                actions_structured.append(a.strip())
                actions_summary.append(_trim_text(a.strip(), 240))
        else:
            s = str(a)
            actions_structured.append(s)
            actions_summary.append(_trim_text(s, 240))

    rec_stub: Dict[str, Any] = {
        "id": rid,
        "title": (rec.get("title") or "").strip(),
        "summary": (rec.get("summary") or "").strip(),
        "why": (rec.get("why") or rec.get("reason") or "").strip(),
        "acceptance_criteria": rec.get("acceptance_criteria") or [],
        "actions": actions_structured,
        "actions_summary": actions_summary,
        "dependencies": rec.get("dependencies") or [],
    }

    project_map_excerpt = _build_project_map_excerpt(meta)

    payload: Dict[str, Any] = {
        "payload_version": 1,
        "rec": rec_stub,
        "focus": focus,
        "project_brief": project_brief,
        "project_map_excerpt": project_map_excerpt,
        "target": target_meta or {},
        "file": file_payload,
    }

    preset = system_preset("edit_analyze")
    system_text = preset or (
        "You are the planning lead for a codebase. For a SINGLE target file, you will\n"
        "analyze how that file should be treated in order to implement one high-level\n"
        "recommendation.\n\n"
        "You receive ONE JSON object with fields:\n"
        "- payload_version: integer (input envelope version).\n"
        "- rec: compact recommendation metadata {id, title, summary, why,\n"
        "  acceptance_criteria[], actions[]}.\n"
        "- focus: high-level goal string for this run.\n"
        "- project_brief: optional short project brief string.\n"
        "- project_map_excerpt: tiny structure summary.\n"
        "- target: per-file target metadata from the selection stage (may be empty).\n"
        "- file: { path, content, context_snippets[] } describing the primary file.\n"
        "  Each context_snippet is { path, kind, snippet } and is short.\n\n"
        "Your job for THIS ONE FILE is to:\n"
        "1) Decide what role it plays for the recommendation and whether it should be\n"
        "   edited.\n"
        "2) If it should be edited, describe concretely what should change in this\n"
        "   file (local_plan), plus any important constraints and invariants.\n"
        "3) Summarise any important context gleaned from neighbouring files.\n"
        "4) Provide concise notes_for_editor that can be injected directly into the\n"
        "   downstream edit_file prompt.\n"
        "5) If you identify broader cross-file contracts or follow-ups, include them\n"
        "   in cross_file_notes.\n"
        "6) If you discover additional files that should be treated as targets\n"
        "   (especially tests or docs), list them in updated_targets[], each with\n"
        "   path, intent, is_primary, rationale, and optional success_criteria.\n\n"
        "Return ONE JSON object that matches the per-file target_analysis_schema\n"
        "(v2). It MUST include at least:\n"
        "- schema_version (set to 2),\n"
        "- rec_id (matching rec.id),\n"
        "- focus (copied from input focus),\n"
        "- path (matching file.path),\n"
        "- role, should_edit, local_plan.\n"
        "If nothing should be changed in this file, set should_edit=false but still\n"
        "fill in role and a short rationale in local_plan.\n"
        "Do NOT include markdown or any commentary outside the JSON object."
    )

    schema = _load_target_analysis_schema()

    model_used = None

    try:
        # Prefer calling with keyword args for stage and max_tokens so newer
        # clients can route per-stage models.
        data, _res = chat_json_fn(
            system_text,
            payload,
            schema,
            0.0,  # deterministic planning
            "target_analysis_file",
            stage="analyze",
            max_tokens=max_tokens,
        )
    except TypeError:
        # Legacy wrappers may expect positional 'stage' then 'max_tokens'. Try that
        # form before giving up.
        try:
            data, _res = chat_json_fn(
                system_text,
                payload,
                schema,
                0.0,  # deterministic planning
                "target_analysis_file",
                "analyze",
                max_tokens,
            )
        except Exception as e:
            logging.exception(
                "[analyze_file] LLM call failed for rec_id=%s path=%s (positional fallback)",
                rid,
                rel_path,
            )
            fallback = _make_fallback_analysis(
                note="LLM call failed (positional fallback); see logs.",
                error_type="llm_call_failed",
                message=f"LLM call failed (positional fallback): {type(e).__name__}",
            )
            return PerFileAnalysis(path=rel_path, analysis=fallback, cross_file_notes=None, updated_targets=[])
    except Exception as e:
        logging.exception(
            "[analyze_file] LLM call failed for rec_id=%s path=%s (keyword call)",
            rid,
            rel_path,
        )
        fallback = _make_fallback_analysis(
            note="LLM call failed (keyword call); see logs.",
            error_type="llm_call_failed",
            message=f"LLM call failed (keyword call): {type(e).__name__}",
        )
        return PerFileAnalysis(path=rel_path, analysis=fallback, cross_file_notes=None, updated_targets=[])

    # Try to extract a resolved model string from the returned metadata and
    # inject it into the analysis JSON under `llm_model` so downstream tests /
    # logs can assert which model was used. We are tolerant: don't overwrite an
    # existing `llm_model` key if present.
    # NOTE: _res may be None when the wrapper returned only data; handle safely.

    # Case 1: dict-style metadata (future/alternate wrappers)
    if isinstance(_res, dict):
        # Common top-level keys
        for k in ("llm_model", "model", "model_name", "effective_model", "used_model"):
            v = _res.get(k)
            if isinstance(v, str) and v:
                model_used = v
                break
        # Check nested metadata containers if not found yet.
        if not model_used:
            for container in ("meta", "metadata", "model_info", "response"):
                sub = _res.get(container)
                if isinstance(sub, dict):
                    for k in ("llm_model", "model", "model_name", "effective_model"):
                        v = sub.get(k)
                        if isinstance(v, str) and v:
                            model_used = v
                            break
                    if model_used:
                        break

    # Case 2: object-style metadata (ChatResponse from ChatGPT.chat_json)
    elif _res is not None:
        # Try direct attributes on the response object.
        for attr in ("llm_model", "model", "model_name", "effective_model", "used_model"):
            v = getattr(_res, attr, None)
            if isinstance(v, str) and v:
                model_used = v
                break

        # If still not found, look into a nested `raw` object / dict.
        if not model_used:
            raw = getattr(_res, "raw", None)

            # raw as dict
            if isinstance(raw, dict):
                for k in ("llm_model", "model", "model_name", "effective_model"):
                    v = raw.get(k)
                    if isinstance(v, str) and v:
                        model_used = v
                        break
            # raw as object
            elif raw is not None:
                for attr in ("llm_model", "model", "model_name", "effective_model"):
                    v = getattr(raw, attr, None)
                    if isinstance(v, str) and v:
                        model_used = v
                        break

    if model_used:
        logging.info("[analyze_file] rec_id=%s path=%s llm_model=%s", rid, rel_path, model_used)
    else:
        logging.info(
            "[analyze_file] rec_id=%s path=%s llm_model not present in response metadata",
            rid,
            rel_path,
        )

    # If the LLM returned a non-dict (string/array/etc), be defensive and build
    # a fallback analysis rather than returning None. Include a trimmed repr
    # of the partial response in `partial` so downstream tests can inspect.
    if not isinstance(data, dict):
        logging.warning(
            "[analyze_file] non-dict response for rec_id=%s path=%s; got type=%s",
            rid,
            rel_path,
            type(data).__name__,
        )
        partial_str = None
        try:
            partial_str = _trim_text(repr(data), 1000)
        except Exception:
            partial_str = None
        fallback = _make_fallback_analysis(
            note=f"Non-dict response from LLM: {type(data).__name__}",
            partial=partial_str,
            model=model_used,
            error_type="llm_non_dict_response",
            message=f"Non-dict response from LLM: {type(data).__name__}",
        )
        return PerFileAnalysis(path=rel_path, analysis=fallback, cross_file_notes=None, updated_targets=[])

    # Ensure path is set and normalized in the response.
    resp_path = data.get("path")
    if not isinstance(resp_path, str) or not resp_path.strip():
        data["path"] = rel_path
        resp_path = rel_path
    else:
        resp_path = resp_path.strip()
        data["path"] = resp_path

    # If we found a model from the metadata and the analysis JSON didn't include
    # it already, inject it now.
    if model_used and isinstance(data, dict) and not data.get("llm_model"):
        try:
            data["llm_model"] = model_used
        except Exception:
            # If something odd prevents setting the key, don't fail the analysis.
            logging.debug("[analyze_file] failed to set llm_model on analysis data")

    # Be tolerant of future/alternate field names.
    cross_file_notes_raw = (
        data.get("cross_file_notes")
        or data.get("analysis_cross_file_notes")
        or data.get("global_cross_file_notes")
    )
    if isinstance(cross_file_notes_raw, str) and cross_file_notes_raw.strip():
        cross_file_notes = cross_file_notes_raw
    else:
        cross_file_notes = None

    updated_targets = data.get("updated_targets") or []
    if not isinstance(updated_targets, list):
        updated_targets = []

    # Normalise presence of core keys in the analysis dict: ensure the minimal
    # set exists so callers can rely on them.
    def _ensure_core_keys(d: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(d, dict):
            d = {}
        d.setdefault("schema_version", 2)
        d.setdefault("rec_id", rid)
        d.setdefault("focus", focus)
        d.setdefault("path", resp_path)
        d.setdefault("role", d.get("role") or "unknown")
        d.setdefault("should_edit", bool(d.get("should_edit", False)))
        d.setdefault("local_plan", d.get("local_plan") or "")
        d.setdefault("issues", d.get("issues") or [])
        d.setdefault("suggestions", d.get("suggestions") or [])
        d.setdefault("recommended_actions", d.get("recommended_actions") or [])
        d.setdefault("failures", d.get("failures") or [])
        # Preserve llm_model if present; otherwise leave as-is (may be None)
        if model_used and not d.get("llm_model"):
            d["llm_model"] = model_used
        return d

    data = _ensure_core_keys(data)

    return PerFileAnalysis(
        path=resp_path,
        analysis=data,
        cross_file_notes=cross_file_notes,
        updated_targets=updated_targets,
    )
