# aidev/stages/analyze_stage_driver.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

from ..cards import KnowledgeBase
from .analyze_file import PerFileAnalysis, analyze_file

JsonSchema = Dict[str, Any]
# ChatJsonFn is intentionally permissive: callers may accept either the legacy
# positional signature or the extended signature that accepts stage/model_override
# as keyword args. We call the provided function using keyword arguments for the
# optional extensions and fall back to the legacy positional form on TypeError.
ChatJsonFn = Callable[..., Tuple[Any, Any]]
TargetsEnvelope = Dict[str, Any]


@dataclass
class AggregateAnalysisResult:
    """
    Aggregated result for the 'analyze' stage for a single recommendation.

    analysis_by_path:
        Mapping of repo-relative path -> per-file analysis object
        (the full object returned by target_analysis.schema.json).

    updated_targets_envelope:
        The refined TargetsEnvelope after merging any updated_targets discovered
        by per-file analysis back into the original env_for_rec.

    global_cross_file_notes:
        Optional dict of { path: notes_string } aggregating per-file
        cross_file_notes. EditAnalyzeMixin exposes this on the recommendation
        as both `analysis_cross_file_notes` (canonical) and `cross_file_notes`
        for downstream consumers.

    failures:
        Deterministic ordered list of per-file failures. Each entry is a dict:
        {"path": str, "error_type": str, "error_summary": str}.

    Note: analysis_by_path is keyed by the original selected_paths input order
    (i.e., the selected file path used to call analyze_file). This preserves a
    deterministic ordering source for downstream aggregation.
    """

    analysis_by_path: Dict[str, Dict[str, Any]]
    updated_targets_envelope: Dict[str, Any]
    global_cross_file_notes: Optional[Dict[str, str]]
    failures: List[Dict[str, str]]

    @property
    def per_file(self) -> Dict[str, Dict[str, Any]]:
        """
        Backwards-compatible alias for older callers that expected `per_file`.
        """
        return self.analysis_by_path


def _coerce_failure_record(path: str, error_type: str, error_summary: str) -> Dict[str, str]:
    return {
        "path": str(path or ""),
        "error_type": str(error_type or "Error"),
        "error_summary": str(error_summary or ""),
    }


def _extract_failure_from_result(rel_path: str, result: Any) -> Optional[Dict[str, str]]:
    """Best-effort extraction of a structured failure record from a non-success result.

    Supports:
      - None -> NoResult
      - dict with keys like error/failed/error_summary
      - object with attributes like error/failed/error_summary
    """
    if result is None:
        return _coerce_failure_record(rel_path, "NoResult", "analyze_file returned None")

    # Dict sentinel
    if isinstance(result, dict):
        # If dict explicitly marks failure, or contains common error keys, coerce.
        failed = result.get("failed")
        if failed is True or any(k in result for k in ("error", "error_summary", "error_type")):
            et = result.get("error_type") or "FailedResult"
            es = (
                result.get("error_summary")
                or result.get("error")
                or result.get("message")
                or str(result)
            )
            return _coerce_failure_record(rel_path, str(et), str(es))
        return None

    # Object sentinel
    try:
        failed_attr = getattr(result, "failed", None)
        if failed_attr is True or any(
            hasattr(result, k) for k in ("error", "error_summary", "error_type")
        ):
            et = getattr(result, "error_type", None) or "FailedResult"
            es = (
                getattr(result, "error_summary", None)
                or getattr(result, "error", None)
                or getattr(result, "message", None)
                or str(result)
            )
            return _coerce_failure_record(rel_path, str(et), str(es))
    except Exception:
        # If introspection fails, fall through to None.
        return None

    return None


def _extract_target_entries(env_for_rec: TargetsEnvelope) -> List[Dict[str, Any]]:
    """
    Extract a normalized list of target entries from the env_for_rec envelope.

    We:
      - look for env_for_rec["targets"]
      - filter to dicts with a non-empty string 'path'
      - dedupe by path while preserving first occurrence order
    """
    if not isinstance(env_for_rec, dict):
        return []

    raw_targets = env_for_rec.get("targets")
    if not isinstance(raw_targets, list):
        return []

    seen: set[str] = set()
    entries: List[Dict[str, Any]] = []

    for t in raw_targets:
        if not isinstance(t, dict):
            continue
        path = t.get("path")
        if not isinstance(path, str):
            continue
        rel = path.strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)
        # Keep the entire target dict as metadata; analyze_file can use it.
        entries.append(t)

    return entries


def _extract_selected_paths(
    env_for_rec: TargetsEnvelope, target_entries: List[Dict[str, Any]]
) -> List[str]:
    """
    Build the list of paths to analyze for this recommendation.

    Rules:
      - Start from env_for_rec["selected_files"] (if present, list of strings).
      - Ensure every target path is included (union of selected_files and targets).
      - Deduplicate by path while preserving the first-seen order.
    """
    seen: set[str] = set()
    paths: List[str] = []

    if isinstance(env_for_rec, dict):
        raw_selected = env_for_rec.get("selected_files")
        if isinstance(raw_selected, list):
            for p in raw_selected:
                if not isinstance(p, str):
                    continue
                rel = p.strip()
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                paths.append(rel)

    for t in target_entries:
        path = t.get("path")
        if not isinstance(path, str):
            continue
        rel = path.strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)
        paths.append(rel)

    return paths


def _merge_updated_targets(
    base_env: TargetsEnvelope, updates: List[Dict[str, Any]]
) -> TargetsEnvelope:
    """
    Merge updated_targets (from all files) back into a copy of the original
    targets envelope.

    Rules:
      - Start from a shallow copy of base_env.
      - Build path -> target mapping from existing base_env["targets"].
      - For each updated_target:
          * If path already exists in a target, overlay keys onto that dict,
            but avoid *downgrading* intent (edit/create -> inspect/context_only).
          * Otherwise, append a new target dict.
      - Never remove existing targets; this is additive/refining only.
    """
    if not isinstance(base_env, dict):
        base_env = {}

    merged_env: Dict[str, Any] = dict(base_env)

    raw_targets = base_env.get("targets")
    if isinstance(raw_targets, list):
        targets: List[Dict[str, Any]] = [
            t for t in raw_targets if isinstance(t, dict)
        ]
    else:
        targets = []

    path_to_target: Dict[str, Dict[str, Any]] = {}
    for t in targets:
        path = t.get("path")
        if isinstance(path, str) and path.strip():
            path_to_target[path.strip()] = t

    def _intent_priority(val: Any) -> int:
        """
        Higher number = stronger edit intent.

        context_only < inspect < edit/create
        """
        if not isinstance(val, str):
            return 0
        v = val.strip().lower()
        if v == "context_only":
            return 1
        if v == "inspect":
            return 2
        if v in ("edit", "create"):
            return 3
        return 0

    for upd in updates:
        if not isinstance(upd, dict):
            continue
        path = upd.get("path")
        if not isinstance(path, str):
            continue
        rel = path.strip()
        if not rel:
            continue

        existing = path_to_target.get(rel)
        if existing is not None:
            # Overlay, but treat 'intent' specially so we don't accidentally
            # downgrade a primary edit target to inspect/context_only.
            try:
                merged_update = dict(upd)

                if "intent" in merged_update:
                    existing_intent = existing.get("intent")
                    updated_intent = merged_update.get("intent")

                    if existing_intent is not None:
                        if _intent_priority(updated_intent) <= _intent_priority(
                            existing_intent
                        ):
                            # Keep the stronger (or equal) existing intent;
                            # drop the lower-priority update.
                            merged_update.pop("intent", None)

                existing.update(merged_update)
            except Exception:
                logging.debug(
                    "[analyze_stage_driver] failed to update target for path=%s",
                    rel,
                    exc_info=True,
                )
        else:
            # Brand-new target discovered during analysis; accept its intent.
            new_target = dict(upd)
            new_target["path"] = rel
            targets.append(new_target)
            path_to_target[rel] = new_target

    merged_env["targets"] = targets
    return merged_env


def _merge_cross_file_notes(
    notes_by_file: List[Tuple[str, Optional[str]]]
) -> Optional[Dict[str, str]]:
    """
    Combine per-file cross_file_notes into a dict { path: notes_string }.

    - Ignores empty/None notes.
    - If multiple notes exist for the same path, concatenates them with a
      blank-line separator.
    """
    merged: Dict[str, str] = {}

    for path, note in notes_by_file:
        if not isinstance(note, str):
            continue
        text = note.strip()
        if not text:
            continue

        existing = merged.get(path)
        if existing:
            merged[path] = f"{existing}\n\n{text}"
        else:
            merged[path] = text

    return merged or None


def run_analysis_for_recommendation(
    *,
    rec: Dict[str, Any],
    env_for_rec: TargetsEnvelope,
    kb: Optional[KnowledgeBase],
    project_root: Path,
    focus: str,
    chat_json_fn: ChatJsonFn,
    project_brief_text: str = "",
    meta: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
) -> Optional[AggregateAnalysisResult]:
    """
    High-level driver for the 'analyze' stage for a single recommendation.

    This replaces the old single-call analyze_targets_for_recommendation and
    instead makes one LLM call per selected file (targets plus selected_files)
    via analyze_file(...).

    Inputs:
      - rec:             full recommendation object.
      - env_for_rec:     TargetsEnvelope produced by target selection.
      - kb:              KnowledgeBase (may be None) for neighbor discovery.
      - project_root:    repository root path.
      - focus:           high-level run focus string.
      - chat_json_fn:    orchestrator-style ChatJsonFn wrapper. This is
                         typically a thin wrapper around orchestrator._chat_json
                         that injects rec_id/title/etc. The ChatJsonFn signature
                         has been extended to accept optional `stage` and
                         `model_override` arguments; this driver wraps the
                         provided chat_json_fn so that analyze-file calls will
                         have stage='analyze' by default.
      - project_brief_text: optional project brief text/markdown.
      - meta:            optional project map / structure metadata.
      - max_tokens:      optional max_tokens override for each analyze_file call.

    Returns:
      AggregateAnalysisResult on success, or None if no analyses were produced
    (e.g., no valid selected files or all per-file calls failed).
    """
    rid = str(rec.get("id") or "rec")

    if not isinstance(env_for_rec, dict):
        logging.warning(
            "[analyze_stage_driver] env_for_rec for rec %s is not a dict (got %s); "
            "skipping analysis.",
            rid,
            type(env_for_rec).__name__,
        )
        return None

    target_entries = _extract_target_entries(env_for_rec)
    selected_paths = _extract_selected_paths(env_for_rec, target_entries)

    if not selected_paths:
        logging.info(
            "[analyze_stage_driver] rec_id=%s has no selected files/targets; skipping.",
            rid,
        )
        # No selected files -> nothing to analyze, but we can still return a no-op result.
        return AggregateAnalysisResult(
            analysis_by_path={},
            updated_targets_envelope=dict(env_for_rec),
            global_cross_file_notes=None,
            failures=[],
        )

    # Build a quick lookup for targets by path so we can attach metadata
    # (intent, rationale, success_criteria, etc.) when analyzing.
    target_by_path: Dict[str, Dict[str, Any]] = {}
    for t in target_entries:
        path = t.get("path")
        if isinstance(path, str):
            rel = path.strip()
            if rel:
                target_by_path[rel] = t

    logging.info(
        "[analyze_stage_driver] starting for rec_id=%s selected_paths=%d targets=%d",
        rid,
        len(selected_paths),
        len(target_entries),
    )

    analysis_by_path: Dict[str, Dict[str, Any]] = {}
    updated_targets_accum: List[Dict[str, Any]] = []
    cross_file_notes_accum: List[Tuple[str, Optional[str]]] = []
    failures_accum: List[Dict[str, str]] = []

    # We'll create a small wrapper around the provided chat_json_fn so that any
    # analyze_file calls receive stage='analyze' by default and an optional
    # model_override sourced from the recommendation/target metadata.
    def _make_chat_wrapper(
        base_fn: ChatJsonFn, default_model_override: Optional[str]
    ) -> ChatJsonFn:
        def wrapped(
            system_text: str,
            payload: Any,
            schema: JsonSchema,
            temp: float,
            tag: str,
            stage: Optional[str] = None,
            model_override: Optional[str] = None,
            max_tokens_override: Optional[int] = None,
        ) -> Tuple[Any, Any]:
            # Default the stage to 'analyze' if the caller did not provide one.
            effective_stage = stage or "analyze"
            # model_override passed by the caller takes precedence; otherwise use
            # the default we captured from rec/target_meta.
            effective_model_override = model_override or default_model_override

            # Try calling the supplied chat_json_fn using keyword arguments for the
            # extended parameters first; fall back to the older positional signature
            # if the base function does not accept those keywords.
            try:
                return base_fn(
                    system_text=system_text,
                    payload=payload,
                    schema=schema,
                    temp=temp,
                    tag=tag,
                    stage=effective_stage,
                    model_override=effective_model_override,
                    max_tokens_override=max_tokens_override,
                )
            except TypeError:
                # Older callers expect: (system_text, payload, schema, temp, tag, Optional[int])
                # In that case we pass only the max_tokens_override as the final param.
                return base_fn(
                    system_text,
                    payload,
                    schema,
                    temp,
                    tag,
                    max_tokens_override,
                )

        return wrapped

    for rel_path in selected_paths:
        if not isinstance(rel_path, str):
            continue
        rel_path = rel_path.strip()
        if not rel_path:
            continue

        # Use the existing target metadata when available; otherwise treat this
        # as a context-only file for this recommendation.
        target_meta = target_by_path.get(rel_path)
        if target_meta is None:
            target_meta = {"path": rel_path, "intent": "context_only"}

        # Compute default model_override for this file (recommendation-level overrides
        # take precedence if present). We capture this into the wrapper so analyze_file
        # gets it when it calls chat_json_fn.
        default_model_override: Optional[str] = None
        if isinstance(rec, dict):
            default_model_override = rec.get("model_override")
        if isinstance(target_meta, dict):
            # target-specific override should override rec-level override if set.
            default_model_override = target_meta.get("model_override") or default_model_override

        chat_wrapper = _make_chat_wrapper(chat_json_fn, default_model_override)

        try:
            result: Optional[PerFileAnalysis] = analyze_file(
                rec=rec,
                kb=kb,
                project_root=project_root,
                path=rel_path,
                focus=focus,
                chat_json_fn=chat_wrapper,
                max_tokens=max_tokens,
                project_brief_text=project_brief_text,
                meta=meta,
                target_meta=target_meta,
            )
        except Exception as e:
            logging.warning(
                "[analyze_stage_driver] analyze_file failed for rec_id=%s path=%s: %s",
                rid,
                rel_path,
                str(e),
                exc_info=True,
            )
            # Record a deterministic failure entry preserving iteration order.
            try:
                failures_accum.append(
                    _coerce_failure_record(rel_path, type(e).__name__, str(e))
                )
            except Exception:
                # Defensive fallback if coercion fails.
                failures_accum.append(_coerce_failure_record(rel_path, "Exception", "analyze_file raised an exception"))
            continue

        # Detect non-exception failure sentinels (None or structured failure objects)
        try:
            failure_from_result = _extract_failure_from_result(rel_path, result)
        except Exception as e:
            logging.debug(
                "[analyze_stage_driver] failure-extraction error for rec_id=%s path=%s: %s",
                rid,
                rel_path,
                str(e),
                exc_info=True,
            )
            failures_accum.append(_coerce_failure_record(rel_path, type(e).__name__, str(e)))
            continue

        if failure_from_result is not None:
            logging.warning(
                "[analyze_stage_driver] per-file analysis failed for rec_id=%s path=%s: %s",
                rid,
                rel_path,
                failure_from_result.get("error_summary", ""),
            )
            failures_accum.append(failure_from_result)
            continue

        if result is None:
            # Defensive: should have been handled by _extract_failure_from_result.
            failures_accum.append(
                _coerce_failure_record(rel_path, "NoResult", "analyze_file returned None")
            )
            continue

        # Store per-file analysis object (full schema-validated payload).
        # Key by the original selected_paths entry (rel_path) to preserve canonical
        # ordering source for aggregation regardless of result.path values.
        analysis_by_path[rel_path] = result.analysis

        # Track any per-file cross_file_notes. Use rel_path as the canonical key
        # so downstream merging aligns with selected_paths ordering.
        if result.cross_file_notes:
            cross_file_notes_accum.append((rel_path, result.cross_file_notes))

        # Collect any updated/extra targets discovered by this file.
        if result.updated_targets:
            updated_targets_accum.extend(result.updated_targets)

    if not analysis_by_path and not updated_targets_accum and not cross_file_notes_accum and not failures_accum:
        logging.info(
            "[analyze_stage_driver] rec_id=%s produced no analyses; returning None.",
            rid,
        )
        return None

    updated_env = _merge_updated_targets(env_for_rec, updated_targets_accum)
    global_cross_file_notes = _merge_cross_file_notes(cross_file_notes_accum)

    logging.info(
        "[analyze_stage_driver] done rec_id=%s analysis_entries=%d updated_targets=%d failures=%d",
        rid,
        len(analysis_by_path),
        len(updated_env.get("targets", []))
        if isinstance(updated_env, dict)
        else 0,
        len(failures_accum),
    )

    return AggregateAnalysisResult(
        analysis_by_path=analysis_by_path,
        updated_targets_envelope=updated_env,
        global_cross_file_notes=global_cross_file_notes,
        failures=failures_accum,
    )


def aggregate(
    rec: Dict[str, Any],
    env_for_rec: TargetsEnvelope,
    kb: Optional[KnowledgeBase],
    project_root: Path,
    focus: str,
    chat_json_fn: ChatJsonFn,
    project_brief_text: str = "",
    meta: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Thin wrapper that runs per-file analysis for a recommendation and
    assembles an Analyze Plan-shaped dictionary suitable for the orchestrator.

    This function intentionally does not produce side effects (events); the
    orchestrator/analyze_mixin is responsible for emitting SSE events.

    Returned plan keys:
      - schema_version: int (1)
      - overview: dict with rec_id, title, summary, project_brief
      - themes: list (may be empty)
      - recommendations: list of per-file entries; each entry is a dict with
        keys 'path' and 'analysis' where 'analysis' is the PerFileAnalysis payload.
      - cross_file_notes: optional dict (copied from AggregateAnalysisResult)
      - updated_targets_envelope: optional dict (copied from AggregateAnalysisResult)
      - failures: optional list of per-file failure dicts (preserves input order)
      - warnings: optional list summarizing failures
    """
    rid = None
    if isinstance(rec, dict):
        rid = str(rec.get("id") or "rec")
    else:
        rid = "rec"

    logging.info("[analyze_stage_driver.aggregate] starting for rec_id=%s", rid)

    result = run_analysis_for_recommendation(
        rec=rec,
        env_for_rec=env_for_rec,
        kb=kb,
        project_root=project_root,
        focus=focus,
        chat_json_fn=chat_json_fn,
        project_brief_text=project_brief_text,
        meta=meta,
        max_tokens=max_tokens,
    )

    # Build a defensive overview using available rec fields and project_brief_text
    overview = {
        "rec_id": rid,
        "title": (rec.get("title") if isinstance(rec, dict) else None) or "",
        "summary": (rec.get("summary") if isinstance(rec, dict) else None) or project_brief_text or "",
        "project_brief": project_brief_text or "",
    }

    plan: Dict[str, Any] = {
        "schema_version": 1,
        "overview": overview,
        "themes": [],
        "recommendations": [],
    }

    if result is None:
        # No analysis produced; return minimal but schema-shaped plan.
        logging.info("[analyze_stage_driver.aggregate] no analysis result for rec_id=%s; returning empty plan", rid)
        return plan

    # To satisfy deterministic ordering requirements, build recommendations by
    # iterating the input-selected paths (selected_paths) first.
    target_entries = _extract_target_entries(env_for_rec)
    selected_paths = _extract_selected_paths(env_for_rec, target_entries)

    # Append successful per-file analyses for selected_paths in the same order.
    for path in selected_paths:
        if path in result.analysis_by_path:
            try:
                entry = {"path": path, "analysis": result.analysis_by_path[path]}
                plan["recommendations"].append(entry)
            except Exception:
                logging.debug(
                    "[analyze_stage_driver.aggregate] failed to append recommendation for path=%s",
                    path,
                    exc_info=True,
                )

    # If there are any additional analyzed paths (not in selected_paths), append
    # them deterministically (sorted) so output remains stable across runs.
    extra_paths = [p for p in result.analysis_by_path.keys() if p not in set(selected_paths)]
    if extra_paths:
        for path in sorted(extra_paths):
            try:
                entry = {"path": path, "analysis": result.analysis_by_path[path]}
                plan["recommendations"].append(entry)
            except Exception:
                logging.debug(
                    "[analyze_stage_driver.aggregate] failed to append extra recommendation for path=%s",
                    path,
                    exc_info=True,
                )

    # Include optional contextual fields when present to aid consumers.
    if result.global_cross_file_notes:
        plan["cross_file_notes"] = result.global_cross_file_notes
    if result.updated_targets_envelope:
        plan["updated_targets_envelope"] = result.updated_targets_envelope

    if result.failures:
        # Preserve deterministic ordering of failures as collected in run_analysis_for_recommendation
        plan["failures"] = result.failures
        plan["warnings"] = [
            f"{len(result.failures)} per-file analyses failed; see failures for details"
        ]

    logging.info(
        "[analyze_stage_driver.aggregate] done rec_id=%s recommendations=%d failures=%d",
        rid,
        len(plan.get("recommendations", [])),
        len(result.failures) if result and result.failures else 0,
    )

    return plan


def _smoke_test_chat_wrapper() -> None:
    """
    Simple smoke test to ensure our wrapper calls both the new (keyword) and
    legacy (positional) chat_json_fn signatures without raising TypeError.
    This is executed only when the module is run directly.
    """
    # Create a local copy of the wrapper factory similar to the one used in
    # run_analysis_for_recommendation so this smoke-test does not depend on
    # nested function visibility.
    def _local_make_chat_wrapper(
        base_fn: ChatJsonFn, default_model_override: Optional[str]
    ) -> ChatJsonFn:
        def wrapped(
            system_text: str,
            payload: Any,
            schema: JsonSchema,
            temp: float,
            tag: str,
            stage: Optional[str] = None,
            model_override: Optional[str] = None,
            max_tokens_override: Optional[int] = None,
        ) -> Tuple[Any, Any]:
            effective_stage = stage or "analyze"
            effective_model_override = model_override or default_model_override
            try:
                return base_fn(
                    system_text=system_text,
                    payload=payload,
                    schema=schema,
                    temp=temp,
                    tag=tag,
                    stage=effective_stage,
                    model_override=effective_model_override,
                    max_tokens_override=max_tokens_override,
                )
            except TypeError:
                return base_fn(system_text, payload, schema, temp, tag, max_tokens_override)

        return wrapped

    # New-style function: accepts stage/model_override/max_tokens_override as kwargs
    def new_style(system_text, payload, schema, temp, tag, stage=None, model_override=None, max_tokens_override=None):
        return ({"which": "new", "stage": stage, "model_override": model_override, "max_tokens_override": max_tokens_override}, None)

    # Old-style function: expects max_tokens as final positional arg
    def old_style(system_text, payload, schema, temp, tag, max_tokens=None):
        return ({"which": "old", "max_tokens": max_tokens}, None)

    wrapper_new = _local_make_chat_wrapper(new_style, "model-from-rec")
    wrapper_old = _local_make_chat_wrapper(old_style, None)

    # Call both wrappers; if there is a TypeError it will surface here.
    r1, _ = wrapper_new("sys", {}, {}, 0.5, "tag")
    r2, _ = wrapper_old("sys", {}, {}, 0.5, "tag")

    logging.info("_smoke_test_chat_wrapper new result=%s", r1)
    logging.info("_smoke_test_chat_wrapper old result=%s", r2)

if __name__ == "__main__":
    # Run the smoke test locally when invoked as a script. This should be a
    # no-op for normal imports but helps validate the wrapper behavior quickly.
    _smoke_test_chat_wrapper()
