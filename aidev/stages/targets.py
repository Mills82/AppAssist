# aidev/stages/targets.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence
from pathlib import Path

from ..cards import KnowledgeBase
from .generate_edits import select_targets_for_recommendation, ChatJsonFn

# Safe glob resolver: expands globs only within a repo root and raises on escapes
from runtimes.path_safety import resolve_glob_within_root

ProgressFn = Callable[[str, Dict[str, Any]], None]
ShouldCancelFn = Callable[[], bool]
TimeoutRunner = Callable[[Callable[[], Any], float, str], Any]
TraceFn = Callable[[str, str, Dict[str, Any]], None]


def _card_view_for_path(
    kb: KnowledgeBase,
    path: str,
    *,
    kb_score: Optional[float] = None,
    kb_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a compact, LLM-facing snapshot of a v2 card for target selection.

    This pulls from the KnowledgeBase card index + related-file graph, using:
      - path, language, kind
      - summaries.summary_short (or AI / heuristic summary as fallback)
      - summaries.capability_tags
      - contracts.public_api + config_contracts.env_required
      - routes / cli_args
      - neighbors via KnowledgeBase.get_related_files()
      - staleness.changed / staleness.needs_ai_refresh
      - kb_score / kb_rank from select_cards()
    """
    # Load the card meta from the index
    card = kb.get_card(path) or {}
    if not isinstance(card, dict):
        card = {}

    # Top-level, always-present-ish fields
    language = card.get("language") or "other"
    kind = card.get("kind") or ""

    summary_block = card.get("summary") or {}
    summaries = card.get("summaries") or {}
    contracts = card.get("contracts") or {}
    metrics = card.get("metrics") or {}
    role = card.get("role") or {}
    staleness = card.get("staleness") or {}

    routes = card.get("routes") or []
    cli_args = card.get("cli_args") or []

    # Short summary preference: summaries.summary_short -> AI text -> heuristic
    summary_short = (
        summaries.get("summary_short")
        or summary_block.get("ai_text")
        or summary_block.get("heuristic")
        or ""
    )

    # Capability tags (if enriched via structured summaries)
    cap_tags = summaries.get("capability_tags") or []
    if not isinstance(cap_tags, list):
        cap_tags = [cap_tags] if cap_tags else []
    capability_tags = [str(c) for c in cap_tags][:8]

    # Contracts: public API + env vars + test_neighbors
    public_api = contracts.get("public_api") or []
    if not isinstance(public_api, list):
        public_api = [public_api]

    cfg_contracts = contracts.get("config_contracts") or {}
    env_required = cfg_contracts.get("env_required") or []
    if not isinstance(env_required, list):
        env_required = [env_required]

    test_neighbors = contracts.get("test_neighbors") or []
    if not isinstance(test_neighbors, list):
        test_neighbors = [test_neighbors]

    # Graph-based neighbors from KnowledgeBase
    related = kb.get_related_files(path)
    neighbors_view = {
        "same_dir": related.get("same_dir") or [],
        "dependencies": related.get("dependencies") or [],
        "dependents": related.get("dependents") or [],
        "tests": related.get("tests") or [],
        # Expose test_neighbors from contracts as well; often overlaps but that’s fine.
        "contract_test_neighbors": test_neighbors[:8],
    }

    # Metrics – currently we only guarantee test_notes, but we expose the field
    metrics_view = {
        "test_notes": metrics.get("test_notes"),
    }

    view: Dict[str, Any] = {
        "path": path,
        "language": language,
        "kind": kind,
        "role": {
            "subsystem": role.get("subsystem"),
            "layer": role.get("layer"),
            "role_hint": role.get("role_hint"),
            # You can extend this later if you add more role fields into cards.schema.json
        },
        "summary_short": summary_short,
        "capability_tags": capability_tags,
        "contracts": {
            "public_api": [str(x) for x in public_api[:32]],
            "routes": [str(r) for r in routes[:32]],
            "cli_args": [str(a) for a in cli_args[:32]],
            "env_required": [str(e) for e in env_required[:32]],
        },
        "neighbors": neighbors_view,
        "metrics": metrics_view,
        "staleness": {
            "changed": bool(staleness.get("changed")),
            "needs_ai_refresh": bool(staleness.get("needs_ai_refresh")),
        },
    }

    if kb_score is not None:
        view["kb_score"] = float(kb_score)
    if kb_rank is not None:
        view["kb_rank"] = int(kb_rank)

    return view


def _normalize_envelope_targets_metadata(
    envelope: Dict[str, Any],
    rec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ensure each target entry has enough metadata for downstream stages like
    analyze_targets and propose_edits.

    We do **not** override any fields that the LLM has already set; we only
    attach sensible defaults when values are missing.

    Fields:
      - intent:            default "edit"
      - success_criteria:  falls back to rec["acceptance_criteria"] if absent
      - confidence:        default 0.5
      - risk:              default "medium"
      - effort:            default "M"
    """
    targets = envelope.get("targets")
    if not isinstance(targets, list):
        return envelope

    acceptance = rec.get("acceptance_criteria") or []
    if isinstance(acceptance, str):
        acceptance_list: List[str] = [acceptance.strip()] if acceptance.strip() else []
    elif isinstance(acceptance, (list, tuple)):
        acceptance_list = [str(a).strip() for a in acceptance if str(a).strip()]
    else:
        acceptance_list = []

    for t in targets:
        if not isinstance(t, dict):
            continue

        # Intent is used heavily by the analyze/edit prompts.
        t.setdefault("intent", "edit")

        # Success criteria help the analyze/edit stages align concrete changes
        # with "what good looks like" for this recommendation.
        sc = t.get("success_criteria")
        if not sc and acceptance_list:
            t["success_criteria"] = list(acceptance_list)

        # Lightweight planning hints; prompts treat these as soft signals.
        t.setdefault("confidence", 0.5)
        t.setdefault("risk", "medium")
        t.setdefault("effort", "M")

    return envelope


def build_targets_for_single_recommendation(
    *,
    rec: Dict[str, Any],
    kb: KnowledgeBase,
    meta: Dict[str, Any],
    includes: List[str],
    excludes: List[str],
    focus: str,
    top_k_select: int,
    chat_json_fn: ChatJsonFn,
    schema: Dict[str, Any],
    timeout_runner: TimeoutRunner,
    timeout_sentinel: Any,
    timeout_sec: float,
    fallback_n: int,
    should_cancel: ShouldCancelFn,
    project_root: str,
    progress_cb: Optional[ProgressFn] = None,
    trace_fn: Optional[TraceFn] = None,
    project_brief_text: Optional[str] = None,
    brief_hash: Optional[str] = None,
    idx: int = 1,
    total_recs: int = 1,
) -> Dict[str, Any]:
    """
    Select target files for a *single* recommendation.

    This is the per-rec version of build_rec_targets and is intended for
    pipelines that process one recommendation at a time:

        envelope = build_targets_for_single_recommendation(...)
        # envelope is a TargetsEnvelope matching aidev/schemas/targets.schema.json

    Returns:
        TargetsEnvelope dict with at minimum:
            {
                "targets": [...],
                "selected_files": [...],
                "llm_payload_preview": [...],
                "notes": "...",
            }

    On timeout / failure, a minimal fallback envelope is synthesized using the
    top-N candidate files from the KnowledgeBase.

    :param project_root:
        Filesystem path to the project root. Used to build host-side
        llm_payload_preview entries from real on-disk files for the selected
        paths returned by the target-selection LLM.

    :param project_brief_text:
        Optional markdown / string project brief to include in the
        TargetsEnvelope payload (as PROJECT_BRIEF) so the target selection
        stays aligned with the overall app/product goals.

    :param brief_hash:
        Optional hash of the project brief, logged in traces so we can
        correlate target-selection decisions with a specific brief version.
    """
    rid = str(rec.get("id", "rec"))
    title = (rec.get("title") or "").strip()
    query = f"{title} {focus}".strip() or "project overview"

    # ---- Acceptance criteria awareness ---------------------------------
    # Normalize acceptance_criteria into a list of strings and attach
    # a short criteria_summary back onto the recommendation so downstream
    # stages and prompts can rely on it.
    criteria_raw = rec.get("acceptance_criteria") or []
    criteria: List[str] = []

    if isinstance(criteria_raw, list):
        for c in criteria_raw:
            cs = str(c).strip()
            if cs:
                criteria.append(cs)
    elif isinstance(criteria_raw, str):
        cs = criteria_raw.strip()
        if cs:
            criteria.append(cs)

    if criteria:
        rec["acceptance_criteria"] = criteria
        rec["criteria_summary"] = "; ".join(criteria[:5])

        # Enrich the query used for KnowledgeBase selection so candidates
        # are biased toward files that help satisfy the criteria.
        query += "\n\nAcceptance criteria:\n" + "\n".join(
            f"- {c}" for c in criteria[:6]
        )

    if should_cancel():
        # Return an empty-but-well-formed envelope so downstream callers
        # don't have to special-case cancellation.
        envelope: Dict[str, Any] = {
            "targets": [],
            "selected_files": [],
            "llm_payload_preview": [],
            "notes": "Target selection cancelled before LLM call.",
        }
        envelope = _normalize_envelope_targets_metadata(envelope, rec)
        if trace_fn:
            trace_fn(
                "TARGETS",
                "cancelled",
                {
                    "rec_id": rid,
                    "envelope": envelope,
                    "brief_hash": brief_hash,
                },
            )
        return envelope

    if progress_cb:
        progress_cb(
            "target_select_iter",
            {
                "rec_id": rid,
                "title": title or "(untitled)",
                "idx": idx,
                "total": max(1, total_recs),
            },
        )

    # Use KnowledgeBase to find candidate files
    hits = kb.select_cards(
        query or "project",
        top_k=top_k_select,
        filter_includes=includes or None,
        filter_excludes=excludes or None,
    )

    candidate_files: List[str] = []
    candidate_card_views: List[Dict[str, Any]] = []

    for rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, dict):
            continue  # defensive; should not happen with the new contract

        path = (hit.get("path") or "").strip()
        if not path:
            continue

        # score is optional but should exist; coerce to float when present
        score_val = hit.get("score", None)
        try:
            kb_score = float(score_val) if score_val is not None else None
        except Exception:
            kb_score = None

        candidate_files.append(path)
        candidate_card_views.append(
            _card_view_for_path(
                kb,
                path,
                kb_score=kb_score,
                kb_rank=rank,
            )
        )

    def _select() -> Dict[str, Any]:
        # Pass the project brief into the LLM helper so it can add PROJECT_BRIEF
        # to its JSON payload for the TargetsEnvelope call, and pass the
        # project_root so the host can build llm_payload_preview from real files.
        return select_targets_for_recommendation(
            rec=rec,
            meta=meta,
            candidate_files=candidate_files,
            candidate_card_views=candidate_card_views,  # NEW
            project_root=Path(project_root),
            chat_json_fn=chat_json_fn,
            schema=schema,
            project_brief_text=project_brief_text,
        )

    # Call the LLM under a timeout wrapper
    res = timeout_runner(
        _select,
        timeout_sec,
        desc=f"select_targets:{rid}",
    )

    used_fallback = False

    if (
        res is timeout_sentinel
        or not res
        or not isinstance(res, dict)
        or "targets" not in res
    ):
        # If the call timed out or produced no usable envelope, fall back to
        # the top-N candidate files so we still make progress. We synthesize
        # a minimal TargetsEnvelope that passes the JSON schema.
        fallback_paths = candidate_files[:fallback_n]

        # Build a criteria-aware success_criteria list for the fallback case.
        if criteria:
            base_success_criteria = criteria[:6]
            if len(base_success_criteria) < 6:
                base_success_criteria.append(
                    "Changes to this file move the codebase toward the active recommendation's goal."
                )
        else:
            base_success_criteria = [
                "Changes to this file move the codebase toward the active recommendation's goal."
            ]

        targets = []
        for path in fallback_paths:
            # Preserve KnowledgeBase ranking as the implicit edit order for
            # the fallback case (highest-signal candidates first).
            targets.append(
                {
                    "path": path,
                    "intent": "edit",
                    "rationale": (
                        "Fallback target from KnowledgeBase.select_cards because "
                        "the target-selection LLM timed out or returned no usable envelope. "
                        "This file is a high-signal candidate for the current recommendation."
                    ),
                    "success_criteria": list(base_success_criteria),
                    "dependencies": [],
                    "test_impact": (
                        "Run existing tests relevant to this area of the codebase. "
                        "Add focused tests once concrete edits are proposed."
                    ),
                    "effort": "M",
                    "risk": "medium",
                    "confidence": 0.3,
                }
            )

        envelope = {
            "targets": targets,
            "selected_files": fallback_paths,
            "llm_payload_preview": [],
            "notes": (
                "Fallback TargetsEnvelope: using top candidate files because the "
                "target-selection LLM timed out or returned no usable targets."
            ),
        }
        used_fallback = True
    else:
        # Use the LLM-produced envelope as-is (it should already conform to
        # aidev/schemas/targets.schema.json). The LLM stage now overwrites
        # llm_payload_preview with host-built previews based on project_root.
        envelope = res

    # Make sure each target entry has the metadata that downstream stages
    # expect (intent, success_criteria, confidence, risk, effort).
    envelope = _normalize_envelope_targets_metadata(envelope, rec)

    # --- Resolve any glob-like target specs into concrete repo-relative paths
    # This ensures downstream stages never see raw glob patterns like '*.ts'
    # which can cause filesystem errors (notably on Windows).
    def _looks_like_glob(pat: str) -> bool:
        return any(ch in pat for ch in ("*", "?", "["))

    targets_list = envelope.get("targets") or []
    expanded_targets: List[Dict[str, Any]] = []

    for original in list(targets_list):
        # Preserve non-dict entries untouched (schema should use dicts, but be safe)
        if not isinstance(original, dict):
            expanded_targets.append(original)
            continue

        orig_path = original.get("path")
        if not isinstance(orig_path, str) or not _looks_like_glob(orig_path):
            # Not a glob-like path -> keep as-is
            expanded_targets.append(original)
            continue

        # It's a glob-like pattern: attempt to resolve it within project_root
        pattern = orig_path
        try:
            matches = resolve_glob_within_root(project_root, pattern)
        except Exception as e:
            # Re-raise a clear, testable ValueError so callers/tests can match the message.
            raise ValueError(
                f"Target selection pattern '{pattern}' for rec {rid} rejected by path_safety.resolve_glob_within_root: {e}"
            )

        # If the resolver returned no matches, surface a helpful error rather
        # than silently leaving a glob in the envelope.
        if not matches:
            raise ValueError(
                f"Target selection pattern '{pattern}' for rec {rid} resolved to no files by path_safety.resolve_glob_within_root"
            )

        # Expand into one target entry per matched concrete path. Keep a shallow
        # copy of the original metadata for provenance, add non-breaking fields
        # to mark this expansion.
        for concrete in matches:
            new_t = dict(original)
            new_t["path"] = concrete
            # Provenance for downstream diagnostics; optional fields per schema
            new_t.setdefault("is_glob_spec", True)
            # preserve original pattern under both 'glob_spec' and 'origin_spec'
            # so downstream code can look for either name.
            new_t.setdefault("glob_spec", pattern)
            new_t.setdefault("origin_spec", pattern)
            expanded_targets.append(new_t)

    # Replace the envelope targets with the expanded list. Fallback targets are
    # typically concrete and will not be affected by the above expansion.
    envelope["targets"] = expanded_targets

    # Rebuild selected_files as an ordered, deduplicated list of concrete paths
    # drawn from envelope["targets"]. This keeps selected_files consistent with
    # the final, resolved targets while preserving encounter order.
    sel_files: List[str] = []
    seen: set = set()
    for t in envelope.get("targets", []):
        if not isinstance(t, dict):
            continue
        p = t.get("path")
        if not isinstance(p, str):
            continue
        if p in seen:
            continue
        seen.add(p)
        sel_files.append(p)

    envelope["selected_files"] = sel_files

    if trace_fn:
        trace_fn(
            "TARGETS",
            "fallback" if used_fallback else "llm",
            {
                "rec_id": rid,
                "envelope": envelope,
                "brief_hash": brief_hash,
            },
        )

    # IMPORTANT: The order of `envelope["targets"]` is treated as the
    # logical edit order for this recommendation. Downstream stages
    # will iterate targets in this order so earlier edits can inform
    # later ones via cross-file notes and context.
    return envelope


def build_rec_targets(
    *,
    recs: Sequence[Dict[str, Any]],
    kb: KnowledgeBase,
    meta: Dict[str, Any],
    includes: List[str],
    excludes: List[str],
    focus: str,
    top_k_select: int,
    chat_json_fn: ChatJsonFn,
    schema: Dict[str, Any],
    timeout_runner: TimeoutRunner,
    timeout_sentinel: Any,
    timeout_sec: float,
    fallback_n: int,
    should_cancel: ShouldCancelFn,
    project_root: str,
    progress_cb: Optional[ProgressFn] = None,
    trace_fn: Optional[TraceFn] = None,
    project_brief_text: Optional[str] = None,
    brief_hash: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    For each recommendation, ask the LLM which files to target and return
    a full TargetsEnvelope.

    Returns:
        Mapping: rec_id -> TargetsEnvelope dict.
        Each envelope matches aidev/schemas/targets.schema.json and has at
        least a "targets" array (possibly empty on cancellation).

    - The LLM-based selection is done via select_targets_for_recommendation.
    - If the LLM call times out or fails / returns no envelope, we fall back to
      the top-N candidate files from KnowledgeBase.select_cards and synthesize
      a minimal envelope.

    :param project_root:
        Filesystem path to the project root. Passed through to
        build_targets_for_single_recommendation so the target-selection
        stage can build host-side llm_payload_preview entries from real
        files on disk.

    :param project_brief_text:
        Optional markdown / string project brief to include in the
        TargetsEnvelope payload (as PROJECT_BRIEF) so the target selection
        stays aligned with the overall app/product goals.

    :param brief_hash:
        Optional hash of the project brief, logged in traces so we can
        correlate target-selection decisions with a specific brief version.
    """
    rec_envelopes: Dict[str, Dict[str, Any]] = {}
    total_recs = max(1, len(recs))

    for idx, rec in enumerate(recs, start=1):
        if should_cancel():
            break

        rid = str(rec.get("id", "rec"))
        envelope = build_targets_for_single_recommendation(
            rec=rec,
            kb=kb,
            meta=meta,
            includes=includes,
            excludes=excludes,
            focus=focus,
            top_k_select=top_k_select,
            chat_json_fn=chat_json_fn,
            schema=schema,
            timeout_runner=timeout_runner,
            timeout_sentinel=timeout_sentinel,
            timeout_sec=timeout_sec,
            fallback_n=fallback_n,
            should_cancel=should_cancel,
            project_root=project_root,
            progress_cb=progress_cb,
            trace_fn=trace_fn,
            project_brief_text=project_brief_text,
            brief_hash=brief_hash,
            idx=idx,
            total_recs=total_recs,
        )
        rec_envelopes[rid] = envelope

    return rec_envelopes
