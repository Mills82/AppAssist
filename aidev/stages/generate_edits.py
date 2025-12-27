# aidev/stages/generate_edits.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from pathlib import Path
import json
import hashlib
import logging
import os

from ..llm_client import system_preset
from ..schemas import file_edit_schema
from ..context.edit_context import build_context_bundle_for_paths

JsonSchema = Dict[str, Any]

# Standard ChatJsonFn signature used by the orchestrator:
#   (system_text, user_payload, schema, temperature, phase, max_tokens, stage=None, model_override=None) -> (data, res)
#
# NOTE:
# - stage and model_override are optional keyword arguments that wrappers may accept.
# - Callers should use a TypeError-safe fallback when invoking older chat_json_fn
#   implementations that do not accept these kwargs.
ChatJsonFn = Callable[
    [str, Any, JsonSchema, float, str, Optional[int], Optional[str], Optional[str]],
    Tuple[Any, Any],
]

_SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"

# Hard cap to keep each file payload reasonably sized.
MAX_PREVIEW_CHARS = 20000


def _assert_path_is_concrete(path: str) -> None:
    """
    Validate that a target path is a concrete, repo-relative path and not a glob
    specification. This helper intentionally does not attempt to expand globs;
    its purpose is to fail fast and give callers a clear, actionable error so
    unresolved glob patterns are handled earlier in the pipeline (stages/targets.py
    should resolve glob specs via runtimes.path_safety.resolve_glob_within_root).

    Behavior:
      - If path is falsy or not a string: no-op (leave further validation to callers).
      - If path contains any glob characters ('*','?','[') we raise ValueError with
        a concise message pointing callers to stages/targets.py or the runtime helper.
    """
    if not isinstance(path, str) or not path.strip():
        return
    if any(c in path for c in ("*", "?", "[")):
        raise ValueError(
            f"Unresolved glob pattern in target.path: '{path}'. Targets must be concrete repo-relative paths without glob characters ('*','?','['). Resolve patterns via stages/targets.py or runtimes/path_safety.resolve_glob_within_root."
        )


def _load_targets_schema_for_stage() -> Optional[JsonSchema]:
    """
    Load the canonical Targets JSON Schema used for target selection.

    This reads aidev/schemas/targets.schema.json so stage-level calls use the
    same envelope schema as llm_client.select_targets_for_rec / system.target_select.md.

    If the schema is missing or invalid, returns None and the caller can fall
    back to legacy behavior.
    """
    schema_path = _SCHEMAS_DIR / "targets.schema.json"
    try:
        if schema_path.exists():
            raw = json.loads(schema_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    return None


def _load_edit_schema_for_stage() -> Optional[JsonSchema]:
    """
    Load the canonical file_edit JSON Schema used for edit generation/repair.

    Prefer the helper from ..schemas so we stay in sync with the orchestrator
    and any schema caching. Fall back to reading the JSON file directly.
    """
    # Primary: use the shared helper.
    try:
        raw = file_edit_schema()
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass

    # Fallback: direct JSON read.
    schema_path = _SCHEMAS_DIR / "file_edit.schema.json"
    try:
        if schema_path.exists():
            raw = json.loads(schema_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass

    return None


def _sanitize_edit_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort cleanup of a FileEdit object before validation/usage.

    - Drop content/patch fields that are None or empty/whitespace-only strings.
    - Leave everything else unchanged.

    This keeps us compatible with the file_edit.schema.json contract even if
    the model occasionally returns `content: null` while using patch_unified.
    """
    e = dict(raw)

    for key in ("content", "patch_unified", "patch"):
        if key in e:
            val = e[key]
            if val is None:
                e.pop(key, None)
            elif isinstance(val, str) and not val.strip():
                e.pop(key, None)

    return e


def _coerce_file_index(meta: Any) -> List[Dict[str, Any]]:
    """
    Try to normalize various 'project map' / 'structure' shapes into a simple
    list of {path, language?, summary?, kind?, tags?, changed?} dicts.

    Handles shapes like:
      - project_map.json: { "version": 1, "files": [ { "path": "...", ... }, ... ] }
      - { "project_map": { "files": [...] } }
      - { "file_index": [ ... ] }

    If nothing usable is found, returns an empty list.
    """
    if not isinstance(meta, dict):
        return []

    raw_files: Any = None
    if isinstance(meta.get("files"), list):
        raw_files = meta["files"]
    elif isinstance(meta.get("project_map"), dict) and isinstance(
        meta["project_map"].get("files"), list
    ):
        raw_files = meta["project_map"]["files"]
    elif isinstance(meta.get("file_index"), list):
        raw_files = meta["file_index"]

    if not isinstance(raw_files, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in raw_files:
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            norm: Dict[str, Any] = {
                "path": item["path"],
                "language": item.get("language"),
                "summary": item.get("summary"),
            }
            # Preserve small, high-signal hints from project_map v2 if present.
            for k in ("kind", "tags", "changed", "role", "entrypoint"):
                if k in item:
                    norm[k] = item[k]
            out.append(norm)
    return out


def _build_target_select_payload(
    *,
    rec: Dict[str, Any],
    meta: Any,
    candidate_files: List[str],
) -> Dict[str, Any]:
    """
    Construct a stable, LLM-friendly payload for target selection that includes:
      - the recommendation
      - a lightweight structure summary
      - a full file path index (when available)
      - high-signal candidate files plus summaries and structural hints
    """
    file_index = _coerce_file_index(meta)

    if file_index:
        all_paths = sorted({f["path"] for f in file_index if isinstance(f.get("path"), str)})
    else:
        all_paths = sorted(set(candidate_files))

    # Guardrail against huge repos: cap the lists to keep token usage sane.
    MAX_ALL_FILES = 512
    MAX_CANDIDATES = 64

    all_files_limited = all_paths[:MAX_ALL_FILES]
    candidate_files_limited = candidate_files[:MAX_CANDIDATES]

    # Build candidate_summaries = [{path, language?, summary?, kind?, tags?, changed?}, ...]
    cand_summaries: List[Dict[str, Any]] = []
    if file_index:
        by_path = {f["path"]: f for f in file_index if isinstance(f.get("path"), str)}
        for path in candidate_files_limited:
            base: Dict[str, Any] = {"path": path}
            fi = by_path.get(path)
            if fi:
                if fi.get("language") is not None:
                    base["language"] = fi["language"]
                if fi.get("summary") is not None:
                    base["summary"] = fi["summary"]
                for k in ("kind", "tags", "changed", "role", "entrypoint"):
                    if k in fi and fi[k] is not None:
                        base[k] = fi[k]
            cand_summaries.append(base)
    else:
        cand_summaries = [{"path": p} for p in candidate_files_limited]

    # Light structure summary; avoid dumping huge meta objects directly.
    if isinstance(meta, dict):
        structure_summary: Dict[str, Any] = {
            "total_files": meta.get("total_files") or len(all_paths) or len(candidate_files),
            "language_kinds": meta.get("language_kinds"),
            "by_ext": meta.get("by_ext"),
            "by_top": meta.get("by_top"),
        }
    else:
        structure_summary = {
            "total_files": len(all_paths) or len(candidate_files),
        }

    return {
        "recommendation": rec,
        "structure_summary": structure_summary,
        "all_files": all_files_limited,
        "candidate_files": candidate_files_limited,
        "candidate_summaries": cand_summaries,
    }


def _build_llm_payload_preview(
    project_root: Path,
    selected_files: List[str],
) -> List[Dict[str, Any]]:
    """
    Host-side builder for llm_payload_preview.

    For the selected files (and a small halo of high-signal neighbors), we:

      - Read on-disk content via the shared edit/analyze context helper.
      - Decide full vs excerpt per file based on MAX_PREVIEW_CHARS.
      - Compute a SHA-256 checksum of the exact string provided to the LLM.

    This guarantees:
      - No hallucinated snippets (everything comes from disk).
      - Checksums match the actual payload text.
      - Target-selection, analysis, and edit stages see a consistent notion
        of "context files" (roles, summaries, neighbors).
    """
    previews: List[Dict[str, Any]] = []
    if not selected_files:
        return previews

    root = project_root.resolve()

    # Use the shared context helper so neighbor selection matches the edit stage.
    try:
        bundles = build_context_bundle_for_paths(
            kb=None,
            project_root=root,
            paths=selected_files,
            max_neighbors=8,
            max_tests=4,
            max_bytes_per_file=MAX_PREVIEW_CHARS,
        )
    except Exception:
        # Fallback: simple per-file preview.
        for rel_path in selected_files:
            if not isinstance(rel_path, str) or not rel_path.strip():
                continue

            abs_path = (root / rel_path).resolve()
            try:
                text = abs_path.read_text(encoding="utf-8", errors="ignore")
            except (FileNotFoundError, OSError):
                text = ""

            if text and len(text) <= MAX_PREVIEW_CHARS:
                content = text
                content_excerpt = None
                payload = content
            else:
                excerpt = text[:MAX_PREVIEW_CHARS] if text else ""
                content = None
                content_excerpt = excerpt
                payload = excerpt

            checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            previews.append(
                {
                    "path": rel_path,
                    "content": content,
                    "content_excerpt": content_excerpt,
                    "checksum": checksum,
                }
            )
        return previews

    # Flatten bundles into a single list, deduping by path.
    seen_paths: Set[str] = set()

    for _primary_path, bundle in bundles.items():
        for cf in bundle:
            rel = cf.path
            if not rel or rel in seen_paths:
                continue
            seen_paths.add(rel)

            text = cf.content or ""
            if text and len(text) <= MAX_PREVIEW_CHARS:
                content = text
                content_excerpt = None
                payload = content
            else:
                excerpt = text[:MAX_PREVIEW_CHARS] if text else ""
                content = None
                content_excerpt = excerpt
                payload = excerpt

            checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()

            entry: Dict[str, Any] = {
                "path": rel,
                "content": content,
                "content_excerpt": content_excerpt,
                "checksum": checksum,
            }

            if cf.summary:
                entry["summary"] = cf.summary
            entry["role"] = cf.role
            entry["language"] = cf.language
            entry["kind"] = cf.kind
            entry["size_bytes"] = cf.size

            previews.append(entry)

    return previews


def _contains_glob_chars(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    return any(c in path for c in ("*", "?", "["))


def _is_noop_or_header_only_diff(patch_text: str) -> bool:
    """
    Heuristic to detect header-only or effectively no-op unified diffs.

    - If the patch contains no hunk markers ('@@') we treat it as header-only.
    - Otherwise, if within the diff there are no substantive '+' or '-' lines
      (excluding the '+++ ' / '--- ' file header lines), treat as no-op.
    """
    if not isinstance(patch_text, str) or not patch_text.strip():
        return True

    # Header-only diffs won't have any @@ hunk markers.
    import re as _re
    if not _re.search(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", patch_text, flags=_re.M):
        return True

    for ln in patch_text.splitlines():
        if ln.startswith('+++ ') or ln.startswith('--- '):
            continue
        
        # Any real +/- line (not the file headers above) indicates a substantive hunk.
        if ln.startswith("+") or ln.startswith("-"):
            return False

    # No substantive +/- lines found in hunks -> treat as no-op.
    return True

# Try to import a resolver helper if present. This is optional — stages/targets.py
# should already expand globs earlier in the pipeline. When available we use it
# to expand any remaining specs into concrete repo-relative paths.
try:
    # Prefer top-level runtimes package
    from runtimes.path_safety import resolve_glob_within_root  # type: ignore
except Exception:
    try:
        # Fallback relative import for packaged installs
        from ..runtimes.path_safety import resolve_glob_within_root  # type: ignore
    except Exception:
        resolve_glob_within_root = None  # type: ignore

# Guarded import for the new tiered runner. If absent, we fall back to the
# legacy single-chat invocation behavior so the repo remains compatible.
# TODO: aidev/orchestration/edit_strategy.py implements the two-attempt cap,
# classifier, and SSE events listed in the recommendation.
try:
    from ..orchestration.edit_strategy import apply_tiered_edit_to_file  # type: ignore
except Exception:
    apply_tiered_edit_to_file = None  # type: ignore


def _expand_and_normalize_paths(root: Path, paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Given a list of repo-relative paths (some possibly glob specs), return a tuple:
      (concrete_paths, notes)
    - concrete_paths: list of repo-relative paths with no glob chars.
    - notes: human-readable notes describing any expansions or rejections.

    If resolve_glob_within_root is available it will be used to expand patterns;
    otherwise any glob-containing paths are skipped and noted.
    """
    concrete: List[str] = []
    notes: List[str] = []

    for p in paths:
        if not isinstance(p, str) or not p.strip():
            continue
        if not _contains_glob_chars(p):
            concrete.append(p)
            continue

        # p contains glob chars
        if resolve_glob_within_root:
            try:
                matched = resolve_glob_within_root(root, p)
                if not matched:
                    notes.append(f"Glob pattern '{p}' matched no files under project root; skipping.")
                    continue

                for m in matched:
                    try:
                        mp = Path(m)
                        rel = str(mp.relative_to(root)).replace("\\", "/")
                    except Exception:
                        try:
                            rel = os.path.relpath(str(m), str(root)).replace("\\", "/")
                        except Exception:
                            rel = str(m)
                    concrete.append(rel)
                notes.append(f"Expanded glob '{p}' to {len(matched)} concrete paths.")
            except Exception as e:
                notes.append(f"Failed to resolve glob '{p}': {e}")
        else:
            notes.append(
                f"Unresolved glob pattern '{p}' encountered but no resolver available; skipping."
            )

    # Dedup while preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for q in concrete:
        if q in seen:
            continue
        seen.add(q)
        deduped.append(q)

    return deduped, notes


def select_targets_for_recommendation(
    *,
    rec: Dict[str, Any],
    meta: Dict[str, Any],
    candidate_files: List[str],
    project_root: Path,
    chat_json_fn: ChatJsonFn,
    schema: Optional[JsonSchema],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    project_brief_text: Optional[str] = None,
    candidate_card_views: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Ask the model for a minimal set of file paths to touch for this recommendation.

    - Provides both a full file index (when available via `meta`) and a focused
      `candidate_files` list, plus optional structured `candidate_card_views`.
    - Uses a richer Targets schema (envelope with `targets`, `selected_files`,
      `llm_payload_preview`, and optional `notes`) when provided.
    - Returns a TargetsEnvelope-like dict, never a bare list. Legacy array
      responses from the model are wrapped into an envelope.
    - After the model responds, the host builds `llm_payload_preview` from the
      real on-disk files under `project_root`, overwriting any model-provided
      previews.
    """
    effective_schema: Optional[JsonSchema] = schema
    if effective_schema is None or (
        isinstance(effective_schema, dict) and effective_schema.get("type") == "array"
    ):
        loaded = _load_targets_schema_for_stage()
        if isinstance(loaded, dict):
            effective_schema = loaded

    preset = system_preset("target_select")
    system_text = preset or (
        "You are the Senior Maintainer & Planner for this repository.\n\n"
        "Select the minimal, highest-ROI set of files to modify or create in "
        "order to implement the provided recommendation.\n"
        "- Prefer editing existing files from the given file index (`all_files`) and `candidate_files`.\n"
        "- Only propose a truly new file path when there is no reasonable existing file to change.\n"
        "- Avoid near-duplicate names of existing files.\n"
        "- Use candidate_summaries as high-signal hints, but you MAY select additional files from all_files.\n\n"
        "Return ONLY one JSON value: a JSON object matching the TargetsEnvelope schema with:\n"
        "- `targets`: array of target objects with at least a `path`.\n"
        "- `selected_files`: array of repo-relative paths you will read/analyze.\n"
        "- `llm_payload_preview`: array of per-file objects with `path`, and either `content` or "
        "`content_excerpt`, plus a `checksum`.\n"
        "No markdown, no extra commentary."
    )

    user_payload = _build_target_select_payload(
        rec=rec,
        meta=meta,
        candidate_files=candidate_files,
    )

    if candidate_card_views is not None:
        user_payload["candidates"] = candidate_card_views

    if project_brief_text:
        user_payload["PROJECT_BRIEF"] = project_brief_text

    schema_arg: JsonSchema = effective_schema or {}

    # Resolve any per-recommendation model override and attempt to pass a stage
    # label to chat_json_fn so llm_client can do stage-aware model selection.
    rec_model_override: Optional[str] = None
    try:
        rec_model_override = rec.get("model_override") or rec.get("model")
    except Exception:
        rec_model_override = None
    effective_model_override: Optional[str] = rec_model_override
    stage_label = "target_select"

    # Call the chat_json_fn, preferring to forward stage & model_override when supported.
    try:
        try:
            data, _res = chat_json_fn(
                system_text,
                user_payload,
                schema_arg,
                temperature,
                stage_label,
                max_tokens,
                stage=stage_label,
                model_override=effective_model_override,
            )
        except TypeError:
            # Older chat_json_fn implementations may not accept stage/model_override kwargs.
            data, _res = chat_json_fn(
                system_text,
                user_payload,
                schema_arg,
                temperature,
                stage_label,
                max_tokens,
            )
    except Exception:
        # If the LLM call itself fails entirely, surface an envelope indicating failure.
        logging.exception("[select_targets_for_recommendation] llm call failed for rec_id=%s", rec.get("id"))
        return {
            "targets": [],
            "selected_files": [],
            "llm_payload_preview": [],
            "notes": ["LLM call failed during target selection."],
        }

    root = Path(project_root)

    # Helper to record notes about any unresolved/expanded globs.
    backfill_notes: List[str] = []

    if isinstance(data, dict) and isinstance(data.get("targets"), list):
        env: Dict[str, Any] = dict(data)

        raw_selected = env.get("selected_files") or []
        if isinstance(raw_selected, list):
            selected_files = [p for p in raw_selected if isinstance(p, str) and p.strip()]
        else:
            selected_files = []

        # Ensure selected_files are concrete; expand globs when possible.
        validated_selected, notes = _expand_and_normalize_paths(root, selected_files)
        if notes:
            backfill_notes.extend(notes)
        env["selected_files"] = validated_selected

        # Process env['targets'] and expand any glob specs into multiple concrete targets
        raw_targets = env.get("targets") or []
        new_targets: List[Dict[str, Any]] = []
        for t in raw_targets:
            if not isinstance(t, dict):
                continue
            tp = t.get("path")
            if not isinstance(tp, str) or not tp.strip():
                continue
            if not _contains_glob_chars(tp):
                new_targets.append(t)
                continue

            # Path is a glob spec: try to expand with resolver if available
            if resolve_glob_within_root:
                try:
                    matched = resolve_glob_within_root(root, tp)
                    if not matched:
                        backfill_notes.append(f"Target glob '{tp}' matched no files under project root; skipped.")
                        continue
                    for m in matched:
                        try:
                            mp = Path(m)
                            rel = str(mp.relative_to(root)).replace("\\", "/")
                        except Exception:
                            rel = os.path.relpath(str(m), str(root)).replace("\\", "/")
                        new_t = dict(t)
                        new_t["path"] = rel
                        # annotate origin
                        new_t.setdefault("origin_spec", tp)
                        new_t["is_glob_spec"] = True
                        new_targets.append(new_t)
                    backfill_notes.append(f"Expanded target glob '{tp}' to {len(matched)} concrete targets.")
                except Exception as e:
                    backfill_notes.append(f"Failed to resolve target glob '{tp}': {e}")
            else:
                backfill_notes.append(
                    f"Unresolved target glob '{tp}' encountered but no resolver available; skipping."
                )

        env["targets"] = new_targets

        try:
            env["llm_payload_preview"] = (
                _build_llm_payload_preview(root, validated_selected) if validated_selected else []
            )
        except Exception:
            env.setdefault("llm_payload_preview", [])

        if backfill_notes:
            prev_notes = env.get("notes")
            if isinstance(prev_notes, list):
                env["notes"] = prev_notes + backfill_notes
            else:
                existing = str(prev_notes) if prev_notes else ""
                env["notes"] = [existing] + backfill_notes if existing else backfill_notes

        return env

    if isinstance(data, list):
        paths = [p for p in data if isinstance(p, str) and p.strip()]

        concrete_paths, notes = _expand_and_normalize_paths(root, paths)
        backfill_notes.extend(notes)

        try:
            llm_preview = _build_llm_payload_preview(root, concrete_paths) if concrete_paths else []
        except Exception:
            llm_preview = []

        env_notes = ["Backfilled TargetsEnvelope from legacy array-of-paths response."]
        env_notes.extend(backfill_notes)

        return {
            "targets": [
                {
                    "path": p,
                    "intent": "edit",
                    "rationale": "",
                    "success_criteria": [],
                    "test_impact": "",
                    "effort": "S",
                    "risk": "low",
                    "confidence": 0.5,
                }
                for p in concrete_paths
            ],
            "selected_files": concrete_paths,
            "llm_payload_preview": llm_preview,
            "notes": env_notes,
        }

    env_notes = ["Model returned unrecognized shape for target selection; using empty targets."]
    if data is not None:
        env_notes.append(f"Model response type: {type(data).__name__}")
    return {
        "targets": [],
        "selected_files": [],
        "llm_payload_preview": [],
        "notes": env_notes,
    }


def _normalize_acceptance_criteria(raw: Any) -> List[str]:
    """
    Normalize acceptance_criteria from a rec/analysis object into a list of strings.
    """
    criteria: List[str] = []
    if isinstance(raw, list):
        for c in raw:
            cs = str(c).strip()
            if cs:
                criteria.append(cs)
    elif isinstance(raw, str):
        cs = raw.strip()
        if cs:
            criteria.append(cs)
    return criteria


def _attach_acceptance_criteria(
    rec: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    """
    Helper to attach acceptance_criteria / criteria_summary to a user payload
    based on a recommendation object, if present.
    """
    criteria_raw = rec.get("acceptance_criteria") or []
    criteria = _normalize_acceptance_criteria(criteria_raw)

    if criteria:
        payload["acceptance_criteria"] = criteria
        payload["criteria_summary"] = "; ".join(criteria[:5])


def _per_file_guidance_from_rec(
    rec: Dict[str, Any],
    rel_path: str,
) -> Dict[str, Any]:
    """
    Best-effort extraction of per-file guidance from rec._target_analysis.
    """
    guidance: Dict[str, Any] = {}

    analysis = rec.get("_target_analysis")
    if not isinstance(analysis, dict):
        return guidance

    rel_norm = str(rel_path).replace("\\", "/").strip()
    candidate: Optional[Dict[str, Any]] = None

    analysis_by_path = analysis.get("analysis_by_path")
    if isinstance(analysis_by_path, dict):
        for key in (rel_path, rel_norm):
            val = analysis_by_path.get(key)
            if isinstance(val, dict):
                candidate = val
                break

    if candidate is None:
        targets_by_path = analysis.get("targets_by_path")
        if isinstance(targets_by_path, dict):
            for key in (rel_path, rel_norm):
                val = targets_by_path.get(key)
                if isinstance(val, dict):
                    candidate = val
                    break

    if candidate is None:
        targets = analysis.get("targets")
        if isinstance(targets, list):
            for item in targets:
                if not isinstance(item, dict):
                    continue
                p = str(item.get("path") or "").strip()
                if not p:
                    continue
                if p == rel_path or p == rel_norm:
                    candidate = item
                    break

    if not isinstance(candidate, dict):
        return guidance

    src = candidate.get("analysis")
    if not isinstance(src, dict):
        src = candidate

    lp = src.get("local_plan") or src.get("plan") or src.get("file_plan")
    if isinstance(lp, str):
        lp = lp.strip()
        if lp:
            guidance["local_plan"] = lp

    raw_constraints = src.get("constraints") or src.get("hard_constraints")
    constraints: List[str] = []
    if isinstance(raw_constraints, list):
        for c in raw_constraints:
            cs = str(c).strip()
            if cs:
                constraints.append(cs)
    elif isinstance(raw_constraints, str):
        cs = raw_constraints.strip()
        if cs:
            constraints.append(cs)
    if constraints:
        guidance["constraints"] = constraints

    nfe = src.get("notes_for_editor") or src.get("notes") or src.get("editor_notes")
    if isinstance(nfe, str):
        nfe = nfe.strip()
        if nfe:
            guidance["notes_for_editor"] = nfe

    return guidance


def _ctx_get(file_ctx: Any, attr: str, default: Any = None) -> Any:
    """
    Small helper to read attributes from FileEditContext-like objects.

    Supports both dataclass instances (attribute access) and plain dicts.
    """
    if hasattr(file_ctx, attr):
        return getattr(file_ctx, attr)
    if isinstance(file_ctx, dict):
        return file_ctx.get(attr, default)
    return default


def _guess_language_from_path(path: str) -> str:
    """
    Very small heuristic to guess language from a file path.
    """
    p = path.lower()
    if p.endswith(".py"):
        return "python"
    if p.endswith((".js", ".jsx", ".ts", ".tsx")):
        return "javascript"
    if p.endswith(".css"):
        return "css"
    if p.endswith((".html", ".htm")):
        return "html"
    if p.endswith((".md", ".markdown")):
        return "markdown"
    return "text"


def _build_edit_file_user_payload(file_ctx: Any) -> Dict[str, Any]:
    """
    Build the user payload for system.edit_file.md from a FileEditContext-like object.

    Goals:
      - Deterministic, high-signal ordering for gpt-5.2
      - Avoid redundant alias keys (no `recommendation` mirror of `rec`)
      - Keep acceptance_criteria top-level only (not inside rec)
      - Avoid dumping giant `analysis` / `target_meta`; lift only keys the prompt uses
    """
    path = str(_ctx_get(file_ctx, "path", "") or "").strip()
    language = str(_ctx_get(file_ctx, "language", "") or "text").strip() or "text"
    content = _ctx_get(file_ctx, "content", "") or ""

    analysis = _ctx_get(file_ctx, "analysis", {}) or {}
    if not isinstance(analysis, dict):
        analysis = {}

    rec_meta = _ctx_get(file_ctx, "rec_meta", {}) or {}
    if not isinstance(rec_meta, dict):
        rec_meta = {}

    target_meta = _ctx_get(file_ctx, "target_meta", {}) or {}
    if not isinstance(target_meta, dict):
        target_meta = {}

    # rec_id anchor (prefer explicit IDs in rec_meta)
    rec_id_raw = rec_meta.get("id") or rec_meta.get("rec_id") or analysis.get("rec_id")
    rec_id = str(rec_id_raw or "").strip()

    # goal: high-level run focus; keep short and low precedence vs acceptance_criteria
    goal = rec_meta.get("focus") or rec_meta.get("goal") or analysis.get("goal") or ""
    goal = str(goal or "").strip()

    # Keep acceptance_criteria ONLY top-level; remove from rec to prevent duplication.
    rec_meta_for_payload = dict(rec_meta)
    rec_meta_for_payload.pop("acceptance_criteria", None)

    # acceptance_criteria: prefer rec_meta, then analysis
    criteria_raw: Any = rec_meta.get("acceptance_criteria")
    if not criteria_raw:
        criteria_raw = analysis.get("acceptance_criteria")
    criteria = _normalize_acceptance_criteria(criteria_raw)

    # file_local_plan (canonical)
    lp = (
        analysis.get("file_local_plan")
        or analysis.get("local_plan")
        or target_meta.get("local_plan")
        or target_meta.get("plan")
    )
    file_local_plan = lp.strip() if isinstance(lp, str) and lp.strip() else None

    # file_constraints (canonical)
    raw_constraints = (
        analysis.get("file_constraints")
        or analysis.get("constraints")
        or target_meta.get("constraints")
        or target_meta.get("edit_constraints")
    )
    file_constraints: Any = None
    if raw_constraints is not None:
        constraints_list: List[str] = []
        if isinstance(raw_constraints, list):
            for c in raw_constraints:
                cs = str(c).strip()
                if cs:
                    constraints_list.append(cs)
        elif isinstance(raw_constraints, str):
            cs = raw_constraints.strip()
            if cs:
                constraints_list.append(cs)

        if constraints_list:
            file_constraints = constraints_list
        elif not isinstance(raw_constraints, (list, str)):
            # Preserve odd types rather than silently dropping (observability).
            file_constraints = raw_constraints

    # file_notes_for_editor (canonical)
    nfe = (
        analysis.get("file_notes_for_editor")
        or analysis.get("notes_for_editor")
        or target_meta.get("notes_for_editor")
        or target_meta.get("editor_notes")
        or target_meta.get("notes")
    )
    file_notes_for_editor = nfe.strip() if isinstance(nfe, str) and nfe.strip() else None

    # context_files: already curated/snippets (pass-through)
    context_files = _ctx_get(file_ctx, "context_files", None)
    if not (isinstance(context_files, list) and context_files):
        context_files = None

    # cross_file_notes: prefer dict sources; keep canonical key only.
    # If we only have a string-ish blob, treat it as analysis_cross_file_notes instead.
    cf_primary = analysis.get("cross_file_notes")
    cf_target = target_meta.get("cross_file_notes")
    cf_rec = rec_meta.get("cross_file_notes")

    merged_cf: Dict[str, Any] = {}
    first_text_blob: Optional[str] = None

    for src in (cf_primary, cf_target, cf_rec):
        if src is None:
            continue
        if isinstance(src, dict):
            for k, v in src.items():
                merged_cf.setdefault(k, v)
        elif isinstance(src, str) and src.strip() and first_text_blob is None:
            first_text_blob = src.strip()

    # analysis_cross_file_notes: prompt expects a string (optional)
    # Prefer explicit analysis field; else reuse any text blob we found.
    acfn = analysis.get("analysis_cross_file_notes")
    analysis_cross_file_notes = acfn.strip() if isinstance(acfn, str) and acfn.strip() else first_text_blob

    # Optional file hints (canonical)
    role = analysis.get("file_role") or analysis.get("role") or target_meta.get("file_role") or target_meta.get("role")
    role = role.strip() if isinstance(role, str) and role.strip() else None

    importance = analysis.get("file_importance") or analysis.get("importance") or target_meta.get("importance")
    importance = importance.strip() if isinstance(importance, str) and importance.strip() else None

    kind_hint = analysis.get("file_kind_hint") or analysis.get("kind_hint") or target_meta.get("kind_hint")
    kind_hint = kind_hint.strip() if isinstance(kind_hint, str) and kind_hint.strip() else None

    related_paths = (
        analysis.get("file_related_paths")
        or analysis.get("related_paths")
        or target_meta.get("related_paths")
        or target_meta.get("dependencies")
    )
    file_related_paths: Optional[List[str]] = None
    if isinstance(related_paths, list):
        rel_norm = [str(p).strip() for p in related_paths if str(p).strip()]
        if rel_norm:
            file_related_paths = rel_norm

    context_summary = analysis.get("file_context_summary") or analysis.get("context_summary")
    file_context_summary = context_summary.strip() if isinstance(context_summary, str) and context_summary.strip() else None

    # details (optional): only include if you already curate it elsewhere
    details = _ctx_get(file_ctx, "details", None)
    if not isinstance(details, dict) or not details:
        details = None

    # create_mode: include only when true (or omit entirely if you always route to another prompt)
    create_mode = bool(_ctx_get(file_ctx, "create_mode", False) or False)

    # ---------------------------
    # Build payload in optimal order
    # ---------------------------
    payload: Dict[str, Any] = {
        "file": {"path": path, "language": language, "current": content},
    }

    if rec_id:
        payload["rec_id"] = rec_id

    if criteria:
        payload["acceptance_criteria"] = criteria
        payload["criteria_summary"] = "; ".join(criteria[:5])

    if file_local_plan:
        payload["file_local_plan"] = file_local_plan
    if file_constraints is not None:
        payload["file_constraints"] = file_constraints
    if file_notes_for_editor:
        payload["file_notes_for_editor"] = file_notes_for_editor

    if context_files is not None:
        payload["context_files"] = context_files

    if goal:
        payload["goal"] = goal

    if rec_meta_for_payload:
        payload["rec"] = rec_meta_for_payload
    else:
        payload["rec"] = {}

    if merged_cf:
        payload["cross_file_notes"] = merged_cf
    if analysis_cross_file_notes:
        payload["analysis_cross_file_notes"] = analysis_cross_file_notes

    if role:
        payload["file_role"] = role
    if importance:
        payload["file_importance"] = importance
    if kind_hint:
        payload["file_kind_hint"] = kind_hint
    if file_related_paths:
        payload["file_related_paths"] = file_related_paths
    if file_context_summary:
        payload["file_context_summary"] = file_context_summary

    if details is not None:
        payload["details"] = details

    if create_mode:
        payload["create_mode"] = True

    # Optional sanity warnings (lightweight observability)
    try:
        if (("focus" in rec_meta) or ("goal" in rec_meta)) and not goal:
            logging.warning(
                "[_build_edit_file_user_payload] expected non-empty goal but got empty path=%s rec_id=%s",
                path,
                rec_id or (rec_meta.get("id") or rec_meta.get("rec_id") or "rec"),
            )
    except Exception:
        pass

    return payload


def _build_repair_file_user_payload(
    file_ctx: Any,
    validation_feedback: str,
    goal: Optional[str],
) -> Dict[str, Any]:
    """
    Build the user payload for system.repair_file.md from a FileEditContext-like
    object and a validation feedback string.

    This prefers using preview content (if provided in file_ctx) as the
    baseline for repair operations so that model responses are judged against
    the same snippet the selector/analysis already used.
    """
    path = str(_ctx_get(file_ctx, "path", "") or "").strip()
    language = str(_ctx_get(file_ctx, "language", "") or "text").strip() or "text"

    # Prefer preview content if present; fall back to the full current content.
    preview = _ctx_get(file_ctx, "preview_content", None)
    if preview is None:
        preview = _ctx_get(file_ctx, "content_excerpt", None)
    if preview is None:
        preview = _ctx_get(file_ctx, "preview", None)

    # IMPORTANT: use preview as baseline whenever a preview value exists (even
    # if it is an empty string). This guarantees repairs are judged against the
    # same snippet the user saw during selection/analysis.
    if isinstance(preview, str):
        current_content = preview
        preview_used = True
    else:
        current_content = _ctx_get(file_ctx, "content", "") or ""
        preview_used = False

    analysis = _ctx_get(file_ctx, "analysis", {}) or {}
    if not isinstance(analysis, dict):
        analysis = {}

    rec_meta = _ctx_get(file_ctx, "rec_meta", {}) or {}
    if not isinstance(rec_meta, dict):
        rec_meta = {}

    target_meta = _ctx_get(file_ctx, "target_meta", {}) or {}
    if not isinstance(target_meta, dict):
        target_meta = {}

    rec_id_raw = rec_meta.get("id") or rec_meta.get("rec_id")
    rec_id = str(rec_id_raw or "rec").strip() or "rec"

    feedback = (validation_feedback or "").strip()
    if feedback and len(feedback) > 15000:
        feedback = feedback[:14900] + "\n...[validation feedback truncated]..."
    if not feedback:
        feedback = "Validation failed, but no detailed diagnostics were available."

    criteria_raw: Any = rec_meta.get("acceptance_criteria")
    if not criteria_raw:
        criteria_raw = analysis.get("acceptance_criteria")
    criteria = _normalize_acceptance_criteria(criteria_raw)

    recommendation: Dict[str, Any] = {}
    for key in (
        "id",
        "title",
        "summary",
        "reason",
        "focus",
        "risk",
        "impact",
        "effort",
        "tags",
    ):
        if key in rec_meta and rec_meta[key] is not None:
            recommendation[key] = rec_meta[key]

    recommendation.setdefault("id", rec_id)
    if goal and not recommendation.get("focus"):
        recommendation["focus"] = goal

    # Keep acceptance criteria only at top-level payload (avoid duplicate/confusion).
    recommendation.pop("acceptance_criteria", None)

    cf_notes: Dict[str, Any] = {}
    for source in (rec_meta.get("cross_file_notes"), analysis.get("cross_file_notes")):
        if isinstance(source, dict):
            for k, v in source.items():
                cf_notes.setdefault(k, v)

    # ---------------------------------------------------------------------
    # Payload order optimized for smaller models (e.g., gpt-5-mini):
    #   1) Identifiers/anchors first
    #   2) Intent/constraints/diagnostics next
    #   3) Large blob (current_content) last
    # ---------------------------------------------------------------------
    payload: Dict[str, Any] = {}

    # 1) Anchors / identity
    payload["file_path"] = path
    payload["rec_id"] = rec_id
    if language:
        payload["file_language"] = language

    # 2) Intent + constraints + diagnostics (high signal)
    payload["goal"] = goal or recommendation.get("focus") or ""
    if criteria:
        payload["acceptance_criteria"] = criteria
    payload["validation_feedback"] = feedback
    if cf_notes:
        payload["cross_file_notes"] = cf_notes

    # Optional small context (keep before current_content)
    if recommendation:
        payload["recommendation"] = recommendation
    if preview_used:
        payload["preview_used_as_baseline"] = True
    if analysis:
        payload["analysis"] = analysis
    if target_meta:
        payload["target"] = target_meta

    # 3) Largest field last
    payload["current_content"] = current_content

    return payload


# --- New helpers to support patch-first tiered behavior ---

def _normalize_unified_diff_headers(patch_text: str, rel_path: str) -> str:
    """
    Ensure the unified diff has sensible '---' / '+++ ' headers that reference
    the provided relative path. If headers are missing or reference different
    paths, replace them with '--- a/<rel_path>' and '+++ b/<rel_path>' while
    preserving any trailing timestamp/comments on the same header line.

    This is a best-effort, conservative normalization intended to help downstream
    patch-apply logic reliably match the hunk context to a target file path.
    """
    if not isinstance(patch_text, str) or not patch_text.strip():
        return patch_text

    lines = patch_text.splitlines()
    out_lines: List[str] = list(lines)

    # Find first header lines if present
    idx_minus = None
    idx_plus = None
    for i, ln in enumerate(lines[:8]):  # headers are expected near the top
        if ln.startswith("--- ") and idx_minus is None:
            idx_minus = i
        elif ln.startswith("+++ ") and idx_plus is None:
            idx_plus = i

    # Build header replacements
    rel_norm = str(rel_path).replace("\\", "/")
    minus_header = f"--- a/{rel_norm}"
    plus_header = f"+++ b/{rel_norm}"

    def replace_header(orig: str, new_base: str) -> str:
        # Preserve any trailing tab/space and timestamp/comment
        parts = orig.split(None, 1)
        if len(parts) == 2:
            return new_base + " " + parts[1]
        return new_base

    if idx_minus is None and idx_plus is None:
        # No headers: prepend them
        out_lines = [minus_header, plus_header, ""] + out_lines
    else:
        if idx_minus is not None:
            out_lines[idx_minus] = replace_header(lines[idx_minus], minus_header)
        else:
            # Insert minus header before plus or at top
            insert_at = idx_plus if idx_plus is not None else 0
            out_lines.insert(insert_at, minus_header)
            if idx_plus is not None:
                idx_plus += 1

        if idx_plus is not None:
            out_lines[idx_plus] = replace_header(lines[idx_plus], plus_header)
        else:
            # Insert plus header after minus
            insert_at = (idx_minus + 1) if idx_minus is not None else 1
            out_lines.insert(insert_at, plus_header)

    return "\n".join(out_lines) + ("\n" if patch_text.endswith("\n") else "")


def _prefer_patch_over_content(edit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Annotate and normalize an edit dict that contains both 'content' and a
    'patch_unified'/'patch'. We prefer using the patch when present (patch-first
    strategy) but keep the full content as a fallback. This helper:
      - Normalizes unified diff headers to reference edit['path'] when possible.
      - Sets 'patch_preferred' = True when a patch is present.

    This function mutates the provided dict in-place and returns it for convenience.
    """
    if not isinstance(edit, dict):
        return edit

    rel = str(edit.get("path") or "").strip()
    # Prefer patch_unified over patch, but normalize whichever is present.
    pu = edit.get("patch_unified") or edit.get("patch")
    if isinstance(pu, str) and pu.strip():
        try:
            if rel:
                normalized = _normalize_unified_diff_headers(pu, rel)
                # Prefer patch_unified key for downstream consumers
                edit["patch_unified"] = normalized
                if "patch" in edit:
                    # keep original 'patch' as an alias but keep consistency
                    edit["patch"] = normalized
            edit["patch_preferred"] = True
        except Exception:
            # If normalization fails for any reason, still mark preference
            edit.setdefault("patch_preferred", True)
    else:
        # No patch present
        edit.setdefault("patch_preferred", False)

    return edit

# --- end new helpers ---


def generate_edits_for_file(
    *,
    file_ctx: Any,
    chat_json_fn: ChatJsonFn,
    schema: Optional[JsonSchema] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    model_override: Optional[str] = None,
    stage: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    New single-file edit entrypoint used by propose_edits.py.

    Returns a single FileEdit dict matching file_edit.schema.json or None on failure.

    This function will attempt to pass stage + per-recommendation model override
    to chat_json_fn when available. If the provided chat_json_fn does not accept
    these kwargs, a TypeError-safe fallback to the legacy positional call is used.
    """
    path = str(_ctx_get(file_ctx, "path", "") or "")

    # Validate that target.path is concrete (no unresolved glob characters).
    _assert_path_is_concrete(path)

    rec_meta = _ctx_get(file_ctx, "rec_meta", {}) or {}
    if not isinstance(rec_meta, dict):
        rec_meta = {}
    rec_id = str(rec_meta.get("id") or "rec")

    # If the incoming file_ctx indicates this file failed validation or is in
    # repair-mode, route to the dedicated repair flow instead of the general
    # edit path. This enforces a clearer separation between edit vs repair logic
    # and ensures repair-specific heuristics (no-op rejection, patch heuristics)
    # are applied consistently.
    try:
        validation_feedback = _ctx_get(file_ctx, "validation_feedback", None)
        repair_flag = _ctx_get(file_ctx, "repair_mode", None)
        routing_reason = _ctx_get(file_ctx, "routing_reason", None)

        is_routing_repair = (
            isinstance(routing_reason, str) and routing_reason.strip().lower() == "repair"
        )

        if (
            (isinstance(validation_feedback, str) and validation_feedback.strip())
            or repair_flag is True
            or is_routing_repair
        ):
            logging.info("[generate_edits_for_file] routing to repair flow path=%s rec_id=%s", path, rec_id)
            # Ensure we pass a string (may be None)
            feedback_str = validation_feedback if isinstance(validation_feedback, str) else ""
            return generate_repair_for_file(
                file_ctx=file_ctx,
                validation_feedback=feedback_str,
                chat_json_fn=chat_json_fn,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
                goal=rec_meta.get("focus") or rec_meta.get("goal"),
                model_override=model_override,
                stage=stage,
            )
    except Exception:
        # If detection fails, continue with the normal edit flow rather than
        # raising; callers may still receive an edit response.
        logging.debug("[generate_edits_for_file] repair routing check failed for path=%s", path)

    # Detect create-mode (prefer explicit context over cwd heuristics).
    create_mode = False
    try:
        # If upstream already decided, respect it.
        is_new_ctx = _ctx_get(file_ctx, "is_new", None)
        if isinstance(is_new_ctx, bool):
            create_mode = is_new_ctx
        else:
            abs_ctx = _ctx_get(file_ctx, "abs_path", None)
            root_ctx = _ctx_get(file_ctx, "project_root", None) or _ctx_get(file_ctx, "repo_root", None)

            if isinstance(abs_ctx, str) and abs_ctx.strip():
                abs_path = Path(abs_ctx).expanduser().resolve()
            else:
                root = Path(str(root_ctx)).expanduser().resolve() if isinstance(root_ctx, (str, Path)) and str(root_ctx).strip() else Path.cwd().resolve()
                abs_path = (root / path).resolve() if path else root

            create_mode = not abs_path.exists()
    except Exception:
        create_mode = False

    if create_mode:
        try:
            preset = system_preset("create_file")
        except Exception:
            preset = ""
        system_text = preset or (
            "You are an expert software engineer creating ONE new file in a version-controlled repository.\n\n"
            "Produce a complete, runnable source file that includes all necessary imports, configuration, "
            "and a minimal usage example or test. Include brief inline documentation/comments where helpful.\n"
            "Return EXACTLY ONE JSON object matching the FileEdit schema with:\n"
            "- path: the repo-relative path for the new file.\n"
            "- content: full file contents (preferred).\n"
            "- optional rec_id and summary.\n"
            "Do not return partial skeletons — the file must be functional and importable/runable when possible.\n"
            "No markdown or extra commentary."
        )
    else:
        try:
            preset = system_preset("edit_file")
        except Exception:
            preset = ""
        system_text = preset or (
            "You are an expert software engineer editing ONE file at a time in a "
            "version-controlled repository.\n\n"
            "You receive a JSON payload with:\n"
            "- file: {path, language, current} for the file you are editing.\n"
            "- rec / recommendation: metadata about the higher-level recommendation "
            "(id, title, reason, focus, acceptance_criteria).\n"
            "- target: compact target-selection metadata for this file "
            "(intent, rationale, local_plan, constraints, notes_for_editor, etc.).\n"
            "- analysis: structured per-file analysis from earlier stages, including "
            "local_plan, constraints, notes_for_editor, and cross_file_notes.\n"
            "- goal: overall focus string for this run.\n\n"
            "Your job:\n"
            "- Update ONLY file.path.\n"
            "- Make minimal, high-quality changes that satisfy the relevant "
            "acceptance_criteria and keep the codebase coherent.\n"
            "- Return EXACTLY ONE JSON object matching the FileEdit schema.\n"
            "- Prefer returning the full updated file in `content`. "
            "(A `patch_unified` diff is still accepted for backwards compatibility.)\n\n"
            "Output JSON object:\n"
            "- path: MUST equal file.path.\n"
            "- content: full updated contents of the file (preferred), OR\n"
            "- patch_unified: unified diff against file.current.\n"
            "- optional rec_id and summary fields.\n"
            "No markdown or commentary; respond with the JSON object only."
        )

    user_payload = _build_edit_file_user_payload(file_ctx)

    if create_mode:
        user_payload["create_mode"] = True
        user_payload["create_prompt"] = "aidev/prompts/system.create_file.md"

    effective_schema: Optional[JsonSchema] = schema
    if effective_schema is None or not isinstance(effective_schema, dict) or not effective_schema:
        loaded = _load_edit_schema_for_stage()
        effective_schema = loaded or {}
    schema_arg: JsonSchema = effective_schema or {}

    rec_model_override: Optional[str] = None
    try:
        rec_model_override = rec_meta.get("model_override") or rec_meta.get("model")
    except Exception:
        rec_model_override = None

    effective_model_override: Optional[str] = model_override or rec_model_override
    stage_label = (stage or ("create_file" if create_mode else "edit_file")).strip().lower()

    # Local helper implementing the legacy single-call flow so we can delegate
    # to the new tiered runner when available and fall back cleanly otherwise.
    def _legacy_invoke_chat() -> Tuple[Optional[Any], Optional[Any]]:
        try:
            try:
                data, _res = chat_json_fn(
                    system_text,
                    user_payload,
                    schema_arg,
                    temperature,
                    "generate_edits",
                    max_tokens,
                    stage=stage_label,
                    model_override=effective_model_override,
                )
            except TypeError:
                data, _res = chat_json_fn(
                    system_text,
                    user_payload,
                    schema_arg,
                    temperature,
                    "generate_edits",
                    max_tokens,
                )
            return data, _res
        except Exception as e:
            logging.exception(
                "[generate_edits_for_file] LLM edit call failed path=%s: %s",
                path,
                e,
            )
            try:
                file_chars = len(str(_ctx_get(file_ctx, "content", "") or ""))
                analysis = _ctx_get(file_ctx, "analysis", {}) or {}
                analysis_keys = len(analysis) if isinstance(analysis, dict) else 0
                logging.info(
                    "[generate_edits_for_file] payload_stats path=%s file_chars=%d analysis_keys=%d",
                    path,
                    file_chars,
                    analysis_keys,
                )
            except Exception:
                logging.debug(
                    "[generate_edits_for_file] failed to compute payload_stats path=%s",
                    path,
                )
            return None, None

    # Delegate to the tiered runner if available. The runner enforces the two-attempt
    # patch-first strategy and is expected to return a final FileEdit dict or raise.
    data: Optional[Dict[str, Any]] = None
    _res: Optional[Any] = None
    delegated = False

    if apply_tiered_edit_to_file is not None:
        logging.info("[generate_edits_for_file] attempting delegation to tiered runner path=%s create_mode=%s", path, create_mode)
        # Build a small adapter that exposes a generate_edits(...) method compatible
        # with the expectations of the tiered runner. The adapter will call the
        # local chat_json_fn with the same backward-compatible TypeError-safe calling
        # convention used elsewhere in this module.
        class _LLMClientAdapter:
            def __init__(self, chat_fn: ChatJsonFn):
                self._chat = chat_fn
            def generate_edits(
                self,
                system_text: str,
                user_payload: Any,
                schema: JsonSchema,
                temperature: float,
                phase: str,
                max_tokens: Optional[int],
                stage: Optional[str] = None,
                model_override: Optional[str] = None,
            ) -> Tuple[Any, Any]:
                try:
                    try:
                        return self._chat(
                            system_text,
                            user_payload,
                            schema,
                            temperature,
                            phase,
                            max_tokens,
                            stage=stage,
                            model_override=model_override,
                        )
                    except TypeError:
                        return self._chat(
                            system_text,
                            user_payload,
                            schema,
                            temperature,
                            phase,
                            max_tokens,
                        )
                except Exception:
                    logging.exception("[LLMClientAdapter] generate_edits failure for path=%s", path)
                    return None, None
            # Backwards-compatible alias some runners may call.
            def generate(self, *args, **kwargs):
                return self.generate_edits(*args, **kwargs)

        adapter = _LLMClientAdapter(chat_json_fn)
        try:
            # Optional guardrail: ensure file_current arg matches payload file.current exactly
            try:
                payload_current = ""
                if isinstance(user_payload, dict):
                    f = user_payload.get("file")
                    if isinstance(f, dict):
                        payload_current = str(f.get("current") or "")
                file_current_arg = str(_ctx_get(file_ctx, "content", "") or "")
                if payload_current != file_current_arg:
                    logging.warning(
                        "[generate_edits_for_file] payload file.current differs from file_current arg path=%s payload_len=%d arg_len=%d",
                        path,
                        len(payload_current),
                        len(file_current_arg),
                    )
            except Exception:
                pass

            result = apply_tiered_edit_to_file(
                file_path=path,
                file_current=str(_ctx_get(file_ctx, "content", "") or ""),
                llm_client=adapter,
                model=effective_model_override,
                schema=schema_arg,
                temperature=temperature,
                max_tokens=max_tokens,
                stage=stage_label,
                create_mode=create_mode,
                context_payload=user_payload,
            )
            if isinstance(result, dict):
                data = result
                # Attempt to surface any llm metadata the runner may have attached.
                if isinstance(result.get("_llm_call_meta"), dict):
                    _res = result.get("_llm_call_meta")
                elif isinstance(result.get("_llm_model"), str):
                    _res = {"llm_model": result.get("_llm_model")}
                else:
                    _res = None
                delegated = True
                logging.info("[generate_edits_for_file] tiered runner used for path=%s", path)
            else:
                logging.info("[generate_edits_for_file] tiered runner returned non-dict for path=%s; falling back", path)
        except Exception:
            logging.exception("[generate_edits_for_file] tiered runner failed for path=%s", path)
            delegated = False

    if not delegated:
        # fall back to the legacy single LLM call flow
        logging.info("[generate_edits_for_file] falling back to legacy single-call flow for path=%s", path)
        data, _res = _legacy_invoke_chat()

    if not isinstance(data, dict):
        return None

    data = _sanitize_edit_payload(data)

    # Normalize legacy alias if the model emitted it.
    # (We still force path below, but popping avoids schema failures if additionalProperties=false.)
    if "file_path" in data and "path" not in data:
        try:
            data["path"] = str(data.get("file_path") or "").strip() or path
        except Exception:
            data["path"] = path
    data.pop("file_path", None)

    data.setdefault("path", path)
    data.setdefault("rec_id", rec_id)

    # REQUIRED by your FileEdit v5 schema: always provide is_new.
    # Force it to match the actual create_mode we computed.
    data["is_new"] = bool(create_mode)

    # If model returned both patch and full content, normalize headers and
    # annotate preference for patch-first application. Keep both values so the
    # downstream runner may attempt patch application and fall back to content.
    try:
        _prefer_patch_over_content(data)
    except Exception:
        logging.debug("[generate_edits_for_file] failed to normalize/annotate patch preference for path=%s", path)

    has_content = isinstance(data.get("content"), str)
    has_patch_unified = isinstance(data.get("patch_unified"), str)
    has_patch = isinstance(data.get("patch"), str)

    if not (has_content or has_patch_unified or has_patch):
        return None

    try:
        model_used = None

        if isinstance(_res, dict):
            for k in ("llm_model", "model", "model_name", "effective_model", "used_model"):
                v = _res.get(k)
                if isinstance(v, str) and v:
                    model_used = v
                    break
            if not model_used:
                raw = _res.get("raw")
                if isinstance(raw, dict):
                    for k in ("llm_model", "model", "model_name", "effective_model"):
                        v = raw.get(k)
                        if isinstance(v, str) and v:
                            model_used = v
                            break
        elif _res is not None:
            for attr in ("llm_model", "model", "model_name", "effective_model", "used_model"):
                v = getattr(_res, attr, None)
                if isinstance(v, str) and v:
                    model_used = v
                    break
            if not model_used:
                raw = getattr(_res, "raw", None)
                if isinstance(raw, dict):
                    for k in ("llm_model", "model", "model_name", "effective_model"):
                        v = raw.get(k)
                        if isinstance(v, str) and v:
                            model_used = v
                            break
                elif raw is not None:
                    for attr in ("llm_model", "model", "model_name", "effective_model"):
                        v = getattr(raw, attr, None)
                        if isinstance(v, str) and v:
                            model_used = v
                            break

        if model_used:
            data.setdefault("_llm_model", model_used)
        elif isinstance(_res, dict):
            data.setdefault("_llm_call_meta", _res)
    except Exception:
        logging.debug("[generate_edits_for_file] failed to attach llm metadata for path=%s", path)

    return data


def generate_repair_for_file(
    *,
    file_ctx: Any,
    validation_feedback: str,
    chat_json_fn: ChatJsonFn,
    schema: Optional[JsonSchema] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    goal: Optional[str] = None,
    model_override: Optional[str] = None,
    stage: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    New single-file repair entrypoint, used by the edit orchestration mixin.

    Returns a repaired FileEdit dict matching file_edit.schema.json or None on failure.

    This function will attempt to pass stage + per-recommendation model override
    to chat_json_fn when available. If the provided chat_json_fn does not accept
    these kwargs, a TypeError-safe fallback to the legacy positional call is used.
    """
    path = str(_ctx_get(file_ctx, "path", "") or "")

    # Validate that target.path is concrete (no unresolved glob characters).
    _assert_path_is_concrete(path)

    rec_meta = _ctx_get(file_ctx, "rec_meta", {}) or {}
    if not isinstance(rec_meta, dict):
        rec_meta = {}
    rec_id = rec_meta.get("id")

    effective_goal = goal or rec_meta.get("focus") or rec_meta.get("goal") or ""

    preset = system_preset("repair_file")
    system_text = preset or (
        "You are an expert software engineer specializing in repairing code that "
        "failed automated validation.\n\n"
        "You receive a JSON payload with:\n"
        "- file: {path, language, current} for the version of the file that FAILED checks.\n"
        "- file_path and current_content: aliases for backwards compatibility.\n"
        "- goal: short high-level intent for this recommendation or run.\n"
        "- validation_feedback: diagnostics from linters/tests/type-checkers/etc.\n"
        "- recommendation / rec: metadata for the higher-level recommendation "
        "(id, title, reason, acceptance_criteria).\n"
        "- analysis: structured per-file analysis, possibly including cross_file_notes.\n\n"
        "Your job is to repair ONLY this file so that:\n"
        "1) It addresses the problems described in validation_feedback.\n"
        "2) It preserves the original intent as much as possible.\n"
        "3) It stays aligned with goal and any acceptance_criteria.\n\n"
        "Return exactly one JSON object matching the FileEdit schema with:\n"
        "- path: the same as file.path.\n"
        "- EITHER content: full repaired file OR patch_unified: unified diff "
        "against current_content.\n"
        "- optional rec_id and summary.\n"
        "Do not include markdown or any extra commentary."
    )

    user_payload = _build_repair_file_user_payload(
        file_ctx=file_ctx,
        validation_feedback=validation_feedback,
        goal=effective_goal,
    )

    # Capture the baseline used (may be a preview excerpt) so we can judge
    # whether the model response actually changes that baseline.
    baseline_content = user_payload.get("current_content", "") or ""

    effective_schema: Optional[JsonSchema] = schema
    if effective_schema is None or not isinstance(effective_schema, dict) or not effective_schema:
        loaded = _load_edit_schema_for_stage()
        effective_schema = loaded or {}
    schema_arg: JsonSchema = effective_schema or {}

    rec_model_override: Optional[str] = None
    try:
        rec_model_override = rec_meta.get("model_override") or rec_meta.get("model")
    except Exception:
        rec_model_override = None

    effective_model_override: Optional[str] = model_override or rec_model_override
    stage_label = (stage or "repair_edits").strip().lower() or "repair_edits"

    try:
        try:
            data, _res = chat_json_fn(
                system_text,
                user_payload,
                schema_arg,
                temperature,
                "repair_edits",
                max_tokens,
                stage=stage_label,
                model_override=effective_model_override,
            )
        except TypeError:
            data, _res = chat_json_fn(
                system_text,
                user_payload,
                schema_arg,
                temperature,
                "repair_edits",
                max_tokens,
            )
    except Exception as e:
        rec_id_dbg = rec_id or "rec"
        logging.exception(
            "[generate_repair_for_file] LLM repair call failed rec_id=%s path=%s: %s",
            rec_id_dbg,
            path,
            e,
        )
        try:
            current_content = str(_ctx_get(file_ctx, "content", "") or "")
            file_chars = len(current_content)
            feedback_len = len(validation_feedback or "")
            logging.info(
                "[generate_repair_for_file] payload_stats rec_id=%s path=%s file_chars=%d feedback_len=%d",
                rec_id_dbg,
                path,
                file_chars,
                feedback_len,
            )
        except Exception:
            logging.debug(
                "[generate_repair_for_file] failed to compute payload_stats rec_id=%s path=%s",
                rec_id_dbg,
                path,
            )
        return None

    if not isinstance(data, dict):
        return None

    data = _sanitize_edit_payload(data)

    # Normalize legacy alias if the model emitted it.
    if "file_path" in data and "path" not in data:
        try:
            data["path"] = str(data.get("file_path") or "").strip() or path
        except Exception:
            data["path"] = path
    data.pop("file_path", None)

    data.setdefault("path", path)
    if "rec_id" not in data and isinstance(rec_id, str) and rec_id:
        data["rec_id"] = rec_id

    # REQUIRED by your FileEdit v5 schema: repairs are edits to existing files.
    # Force is_new = False unless you explicitly support “repairing” newly-created files.
    data["is_new"] = False

    # Normalize and annotate patch preference when both patch and content are present.
    try:
        _prefer_patch_over_content(data)
    except Exception:
        logging.debug("[generate_repair_for_file] failed to normalize/annotate patch preference for path=%s", path)

    has_content = isinstance(data.get("content"), str)
    has_patch_unified = isinstance(data.get("patch_unified"), str)
    has_patch = isinstance(data.get("patch"), str)

    if not (has_content or has_patch_unified or has_patch):
        return None

    # Reject no-op responses relative to the baseline preview/content used.
    try:
        # If the model returned a full content string equal to the baseline, treat as no-op.
        returned_content = data.get("content")
        if isinstance(returned_content, str):
            if returned_content.strip() == (baseline_content or "").strip():
                logging.info(
                    "[generate_repair_for_file] model returned no-op content equal to baseline rec_id=%s path=%s",
                    rec_id or "rec",
                    path,
                )
                return None

        # If the model returned a unified patch that is header-only or contains no real hunks/changes, reject it.
        pu = data.get("patch_unified") or data.get("patch")
        if isinstance(pu, str) and pu.strip():
            if _is_noop_or_header_only_diff(pu):
                logging.info(
                    "[generate_repair_for_file] model returned header-only or no-op patch rec_id=%s path=%s",
                    rec_id or "rec",
                    path,
                )
                return None
    except Exception:
        logging.debug("[generate_repair_for_file] failed while evaluating no-op/header-only heuristics for path=%s", path)

    try:
        model_used = None

        if isinstance(_res, dict):
            for k in ("llm_model", "model", "model_name", "effective_model", "used_model"):
                v = _res.get(k)
                if isinstance(v, str) and v:
                    model_used = v
                    break
            if not model_used and isinstance(_res.get("raw"), dict):
                raw = _res["raw"]
                for k in ("llm_model", "model", "model_name", "effective_model"):
                    v = raw.get(k)
                    if isinstance(v, str) and v:
                        model_used = v
                        break

        if model_used:
            data.setdefault("_llm_model", model_used)
        elif isinstance(_res, dict):
            data.setdefault("_llm_call_meta", _res)
    except Exception:
        logging.debug("[generate_repair_for_file] failed to attach llm metadata for path=%s", path)

    return data


def generate_repair_for_path(
    *,
    rec: Dict[str, Any],
    rel_path: str,
    current_content: str,
    validation_feedback: str,
    goal: str,
    chat_json_fn: ChatJsonFn,
    schema: Optional[JsonSchema],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Backwards-compatible wrapper that adapts legacy repair callers to
    generate_repair_for_file(FileEditContext-like).
    """
    rec_meta: Dict[str, Any] = {
        "id": rec.get("id") or "rec",
        "title": rec.get("title"),
        "summary": rec.get("summary"),
        "reason": rec.get("reason"),
        "focus": goal or rec.get("focus"),
    }
    if "acceptance_criteria" in rec:
        rec_meta["acceptance_criteria"] = rec["acceptance_criteria"]
    for key in ("risk", "impact", "effort", "tags"):
        if key in rec:
            rec_meta[key] = rec[key]

    file_ctx = {
        "path": rel_path,
        "language": _guess_language_from_path(rel_path),
        "content": current_content,
        "analysis": {},
        "rec_meta": rec_meta,
        "target_meta": {},
    }

    return generate_repair_for_file(
        file_ctx=file_ctx,
        validation_feedback=validation_feedback,
        chat_json_fn=chat_json_fn,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
        goal=goal,
    )
