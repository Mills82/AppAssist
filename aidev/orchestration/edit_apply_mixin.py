# aidev/orchestration/edit_apply_mixin.py
from __future__ import annotations

import os
import logging
import traceback
import json
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..cards import KnowledgeBase
from ..stages.rec_apply import apply_single_recommendation
from ..stages.quality_gates import run_quality_gates
from ..io_utils import _read_file_text_if_exists, apply_unified_patch
from .. import events as _events


class EditApplyMixin:
    """
    Mixin that provides the per-recommendation apply phase of the edit pipeline:

      - target selection (via _run_targets_and_edits_for_rec)
      - edit generation + schema validation
      - optional pre-apply validation + auto-repair
      - diff preview + approval gating
      - apply + post-apply quality gates
      - optional git snapshot/push

    Assumes `self` is an Orchestrator-like object with:

      Attributes:
        - root, st, args, auto_approve, job_id
        - _llm, _session_id, _approval_job_id
        - _timeout_targets, _timeout_edit, _targets_fallback_n
        - _rid_recs, _rid_targets, _rid_edits, _rid_checks
        - _project_brief_text, _project_brief_hash
        - _writes_by_rec, _approval_cb
        - _preapply_patch_jsonl, _preapply_report, _apply_results

      Methods:
        - _phase_max_tokens(phase: str) -> int
        - _chat_json(...)
        - _with_timeout(...)
        - _should_cancel() -> bool
        - _progress(event: str, **payload)
        - _progress_error(where: str, **payload)
        - _emit_result_and_done(ok: bool, summary: str)
        - _job_update(...)
        - _coerce_str_list(...)
        - _arun(...)
        - validate_proposed_edits(...)
        - _run_targets_and_edits_for_rec(...)
        - _maybe_run_self_review_and_auto_repairs(...)
        - _approval_cb(...)
        - _build_commit_message_from_results(...)
    """

    # -------------------- apply phase + per-rec --------------------

    def _run_apply_phase(
        self,
        *,
        kb: KnowledgeBase,
        meta: Dict[str, Any],
        recs: List[Dict[str, Any]],
        includes: List[str],
        excludes: List[str],
        focus: str,
        top_k_select: int,
        dry_run: bool,
        approval_timeout_sec: float,
    ) -> None:
        """
        Handles per-recommendation work:

        - target selection
        - edit generation + schema validation
        - optional pre-apply validation + self-repair
        - diff preview + approval gating
        - apply + quality gates
        - final git snapshot
        """
        # Maximum times to re-call the edit LLM if schema validation fails.
        try:
            edit_schema_retries = int(
                self.args.get("edit_schema_retries")
                or os.getenv("AIDEV_EDIT_SCHEMA_RETRIES", "3")
            )
        except Exception:
            edit_schema_retries = 3
        if edit_schema_retries < 1:
            edit_schema_retries = 1

        # Maximum times to attempt self-healing when validation checks fail.
        try:
            validation_repair_retries = int(
                self.args.get("validation_repair_retries")
                or os.getenv("AIDEV_VALIDATION_REPAIR_RETRIES", "2")
            )
        except Exception:
            validation_repair_retries = 2
        if validation_repair_retries < 0:
            validation_repair_retries = 0

        applied_edits_total = 0
        total_recs = max(1, len(recs))

        # Defensive: respect approved_rec_ids again at the entry to the apply phase.
        approved = self._coerce_str_list(self.args.get("approved_rec_ids"))
        if approved:
            approved_set = set(approved)
            before = len(recs)
            recs = [
                r for r in recs
                if str(r.get("id") or "").strip() in approved_set
            ]
            _events.status(
                "Filtered recommendations by approved_rec_ids (apply entry)",
                where="apply_filtered",
                session_id=self._session_id,
                job_id=self.job_id,
                before=before,
                after=len(recs),
                approved_rec_ids=approved,
            )
            if not recs:
                # Nothing to apply after filtering — write a run-level trace and finish.
                try:
                    self.st.trace.write(
                        "applied_recommendations",
                        "run",
                        {
                            "applied_recommendation_ids": [],
                            "applied_edits_total": 0,
                            "brief_hash": self._project_brief_hash,
                        },
                    )
                except Exception:
                    pass
                self._emit_result_and_done(
                    ok=True,
                    summary=(
                        "No recommendations selected for apply after filtering."
                    ),
                )
                return

        # Structured per-rec results
        results: List[Any] = []
        applied_results: List[Any] = []
        applied_rec_ids: List[str] = []  # for trace/audit

        # Collect proposed entries across recs for dry-run preapply output.
        preapply_proposed_all: List[Dict[str, Any]] = []

        def _risk_from(files_meta: List[Dict[str, Any]]) -> str:
            total_add = sum(f["added"] for f in files_meta)
            total_rem = sum(f["removed"] for f in files_meta)
            n_files = len(files_meta)
            critical_hits = any(
                p["path"].endswith(("pubspec.yaml", ".gradle", ".yaml"))
                or p["path"].startswith(
                    ("android/", "ios/", ".github/", "scripts/")
                )
                for p in files_meta
            )
            if total_add + total_rem >= 200 or critical_hits:
                return "high"
            if total_add + total_rem >= 60 or n_files >= 6:
                return "medium"
            return "low"

        # Per-rec LLM wrappers for targets and edits
        def _chat_json_targets(
            system_text: str,
            user_payload: Any,
            schema: Dict[str, Any],
            temperature: float,
            phase: str = "target_select",
            max_tokens: Optional[int] = None,
        ):
            effective_phase = phase or "target_select"
            mt = (
                max_tokens
                if max_tokens is not None
                else self._phase_max_tokens(effective_phase)
            )
            return self._chat_json(
                system_text,
                user_payload,
                schema=schema,
                temperature=temperature,
                phase=effective_phase,
                inject_brief=False,
                max_tokens=mt,
            )

        def _chat_json_edits(
            system_text: str,
            user_payload: Any,
            schema: Dict[str, Any],
            temperature: float,
            phase: str = "generate_edits",
            max_tokens: Optional[int] = None,
        ):
            effective_phase = phase or "generate_edits"
            mt = (
                max_tokens
                if max_tokens is not None
                else self._phase_max_tokens(effective_phase)
            )
            return self._chat_json(
                system_text,
                user_payload,
                schema=schema,
                temperature=temperature,
                phase=effective_phase,
                inject_brief=False,
                max_tokens=mt,
            )

        def _normalize_file_edit(edit: Dict[str, Any]) -> Dict[str, Any]:
            e = dict(edit)
            path = str(e.get("path") or "").strip()
            if not path:
                raise ValueError(f"Edit entry missing 'path': {edit}")

            content = e.get("content")
            patch_val = (
                e.get("patch_unified")
                if "patch_unified" in e
                else e.get("patch")
            )

            has_content = content is not None and content != ""
            has_patch = patch_val is not None and patch_val != ""

            if has_content and has_patch:
                e["patch_unified"] = str(patch_val)
                e.pop("patch", None)
                e.pop("content", None)
                # Prefer content (or flip this if you’d rather prefer patch_unified)
                # e["content"] = str(content)
                # e.pop("patch_unified", None)
                # e.pop("patch", None)
            elif has_content:
                e["content"] = str(content)
                e.pop("patch_unified", None)
                e.pop("patch", None)
            elif has_patch:
                e["patch_unified"] = str(patch_val)
                e.pop("patch", None)
            else:
                raise ValueError(
                    f"Edit for {path} must have non-empty 'content' or "
                    f"'patch_unified'; got neither."
                )

            return e

        # --------- per-recommendation pipeline ----------
        for idx, rec in enumerate(recs, start=1):
            rid = str(rec.get("id") or f"rec-{idx}")

            if self._should_cancel():
                self._progress(
                    "apply_skip",
                    reason="cancelled",
                    stage="pipeline",
                    rec_id=rid,
                    job_id=self.job_id,
                )
                # Write out whatever was applied so far for auditability.
                try:
                    self.st.trace.write(
                        "applied_recommendations",
                        "run",
                        {
                            "applied_recommendation_ids": applied_rec_ids,
                            "applied_edits_total": applied_edits_total,
                            "brief_hash": self._project_brief_hash,
                        },
                    )
                except Exception:
                    pass
                self._emit_result_and_done(
                    ok=False,
                    summary="Run cancelled before applying edits.",
                )
                return

            # --- targets + edits for this recommendation ---
            edits, proposed = self._run_targets_and_edits_for_rec(
                kb=kb,
                meta=meta,
                includes=includes,
                excludes=excludes,
                focus=focus,
                rec=rec,
                rid=rid,
                idx=idx,
                total_recs=total_recs,
                top_k_select=top_k_select,
                dry_run=dry_run,
                edit_schema_retries=edit_schema_retries,
                validation_repair_retries=validation_repair_retries,
                chat_json_targets=_chat_json_targets,
                chat_json_edits=_chat_json_edits,
            )

            # Enforce a single representation per edit object.
            edits = [
                _normalize_file_edit(e)
                for e in edits
                if isinstance(e, dict)
            ]

            # De-duplicate proposed entries by path, preferring the one
            # with the biggest diff.
            def _diff_line_counts(diff_text: str) -> Tuple[int, int]:
                add = rem = 0
                for ln in (diff_text or "").splitlines():
                    if ln.startswith(("+++ ", "--- ", "@@")):
                        continue
                    if ln.startswith("+"):
                        add += 1
                    elif ln.startswith("-"):
                        rem += 1
                return add, rem

            if isinstance(proposed, list) and proposed:
                best_by_path: Dict[str, Dict[str, Any]] = {}
                for p in proposed:
                    path = p.get("path")
                    if not isinstance(path, str) or not path:
                        continue
                    a, r = _diff_line_counts(p.get("diff", "") or "")
                    score = a + r  # total changed lines
                    existing = best_by_path.get(path)
                    if existing is None or score > existing["score"]:
                        best_by_path[path] = {
                            "score": score,
                            "proposal": p,
                        }

                proposed = [
                    entry["proposal"] for entry in best_by_path.values()
                ]

            # Record proposed entries for dry-run preapply output
            if isinstance(proposed, list) and proposed:
                for p in proposed:
                    if isinstance(p, dict):
                        pcopy = dict(p)
                        pcopy.setdefault("rec_id", rid)
                        preapply_proposed_all.append(pcopy)

            # --------- optional self-review / consistency pass ----------
            rec_self_review: Optional[Dict[str, Any]] = None
            try:
                if not dry_run and edits and proposed:
                    (
                        edits,
                        proposed,
                        rec_self_review,
                    ) = self._maybe_run_self_review_and_auto_repairs(
                        rec=rec,
                        rid=rid,
                        focus=focus,
                        edits=edits,
                        proposed=proposed,
                        max_auto_passes=1,
                    )
            except Exception:
                # Best-effort only: self-review must never break the pipeline.
                logging.debug(
                    "self_review: failed for rec %s", rid, exc_info=True
                )
                rec_self_review = None

            if rec_self_review:
                # Attach to the recommendation so downstream stages / UI can surface it.
                try:
                    rec["_self_review"] = rec_self_review
                except Exception:
                    pass

            # ---------- diff preview + summary for this recommendation ----------
            unified = "\n\n".join(
                x.get("diff", "") for x in proposed if x.get("diff")
            )
            if len(unified.encode("utf-8")) > 1_000_000:
                unified = (
                    unified[:900_000]
                    + "\n\n... (truncated for preview) ..."
                )

            files_meta: List[Dict[str, Any]] = []
            for p in proposed:
                a, r = _diff_line_counts(p.get("diff", "") or "")
                files_meta.append(
                    {
                        "path": p["path"],
                        "added": a,
                        "removed": r,
                        "why": p.get("why") or "",
                    }
                )

            total_added = sum(x["added"] for x in files_meta)
            total_removed = sum(x["removed"] for x in files_meta)
            summary_text = (
                f"{len(files_meta)} file(s) changed, "
                f"+{total_added}/-{total_removed} lines"
                + (f" — {focus[:80]}" if focus else "")
            )
            risk_level = _risk_from(files_meta)

            # Default per-rec auto-approval; we may downgrade it for high risk
            rec_auto_approve = self.auto_approve
            if rec_auto_approve and risk_level == "high":
                rec_auto_approve = False
                _events.status(
                    "Auto-approval disabled for this recommendation due to high risk.",
                    stage="apply",
                    session_id=self._session_id,
                    job_id=self.job_id,
                    rec_id=rid,
                    risk=risk_level,
                )

            files_changed = len({p["path"] for p in proposed})
            _events.diff_ready(
                summary=(
                    f"{len(edits)} edit objects across {files_changed} files "
                    f"(rec {rid}, {idx}/{total_recs})"
                ),
                unified=unified,
                files_changed=files_changed,
                session_id=self._session_id,
                job_id=self.job_id,
            )
            _events.diff(
                unified=unified,
                files_changed=files_changed,
                bundle={
                    "proposed": proposed,
                    "edits": [
                        {"path": e["path"], "rec_id": e.get("rec_id")}
                        for e in edits
                    ],
                },
                summary=summary_text,
                session_id=self._session_id,
                job_id=self.job_id,
            )

            # --------- explicit "awaiting approval" phase ----------
            if rec_auto_approve:
                _events.status(
                    "Auto-approval enabled; applying changes immediately.",
                    stage="apply",
                    session_id=self._session_id,
                    job_id=self.job_id,
                    rec_id=rid,
                )
            else:
                _events.awaiting_approval(
                    message=(
                        f"Awaiting approval for recommendation {idx}/{total_recs} "
                        f"(rec {rid})."
                    ),
                    session_id=self._session_id,
                    rec_id=rid,
                    job_id=self.job_id,
                )

            logging.info(
                "[orchestrator] About to apply rec %s (idx=%d/%d) "
                "dry_run=%s auto_approve=%s edits=%d",
                rid,
                idx,
                total_recs,
                dry_run,
                rec_auto_approve,
                len(edits),
            )

            # --------- apply this recommendation ----------
            result = apply_single_recommendation(
                root=self.root,
                st=self.st,
                kb=kb,
                meta=meta,
                rec=rec,
                rec_edits=edits,
                rec_proposed=proposed,
                dry_run=dry_run,
                auto_approve=rec_auto_approve,
                approval_cb=self._approval_cb,
                session_id=self._session_id,
                job_id=self._approval_job_id,
                writes_by_rec=self._writes_by_rec,
                stats_obj=self,
                progress_cb=lambda event, payload: self._progress(
                    event, **payload
                ),
                progress_error_cb=lambda where, payload: self._progress_error(
                    where, **payload
                ),
                approval_timeout_sec=approval_timeout_sec,
                job_update_cb=self._job_update,
            )
            results.append(result)

            logging.info(
                "[orchestrator] Result for rec %s: applied=%s reason=%s "
                "changed_paths=%s",
                rid,
                getattr(result, "applied", None),
                getattr(result, "reason", None),
                getattr(result, "changed_paths", None),
            )

            if not getattr(result, "applied", False):
                # Treat this as a skipped recommendation (user rejected, dry_run,
                # or apply failed), but do not abort the entire run.
                self._progress(
                    "rec_skipped",
                    stage="apply",
                    rec_id=rid,
                    reason=(
                        getattr(result, "reason", None)
                        or "apply_failed_or_rejected"
                    ),
                    job_id=self.job_id,
                )
                _events.status(
                    "Recommendation apply failed or was skipped; continuing with remaining recommendations.",
                    stage="apply",
                    session_id=self._session_id,
                    job_id=self.job_id,
                    rec_id=rid,
                    reason=getattr(result, "reason", None),
                    changed_paths=getattr(result, "changed_paths", []),
                )
                continue

            # Count edits for metrics / commit summaries
            n_applied = len(edits)
            applied_edits_total += n_applied
            applied_results.append(result)
            try:
                applied_rec_ids.append(rid)
            except Exception:
                pass

            # Announce per-rec apply completion
            self._progress(
                "apply_done",
                stage="apply",
                rec_id=rid,
                applied_edits=n_applied,
                total_applied=applied_edits_total,
                message=(
                    f"Applied {n_applied} edits for recommendation "
                    f"{idx}/{total_recs} (rec {rid})."
                ),
                job_id=self.job_id,
            )

            # --------- post-apply quality gates for this recommendation ----------
            if not dry_run and n_applied > 0:
                self._job_update(
                    stage="checks",
                    message=(
                        f"Running post-apply quality gates for recommendation "
                        f"{idx}/{total_recs}…"
                    ),
                    progress_pct=92,
                )
                run_quality_gates(
                    root=self.root,
                    st=self.st,
                    cfg=self.args.get("cfg"),
                    session_id=self._session_id,
                    job_id=self.job_id,
                    progress_error_cb=self._progress_error,
                )

        # Make per-rec results available to the final result emitter
        self._apply_results = results

        # If this was a dry-run / preapply, build a JSONL patch + run checks in a temp workspace.
        if dry_run:
            patch_entries: List[Dict[str, Any]] = []
            tempd: Optional[str] = None
            report: Dict[str, Any] = {"checks_ok": True, "details": {}}
            try:
                tempd = tempfile.mkdtemp(prefix="aidev_preapply_")
                # Try to copy project tree to tempdir to give quality gates full context.
                try:
                    shutil.copytree(str(self.root), tempd, dirs_exist_ok=True)
                except Exception:
                    # Best-effort: if copytree fails (permissions/etc),
                    # continue and write per-file previews.
                    logging.debug(
                        "preapply: copytree failed, will write per-file previews",
                        exc_info=True,
                    )

                # Write each proposed preview (preferred) or reconstruct via diff.
                for p in preapply_proposed_all:
                    path = p.get("path")
                    if not path or not isinstance(path, str):
                        continue
                    dest = os.path.join(tempd, path)
                    parent = os.path.dirname(dest)
                    try:
                        os.makedirs(parent, exist_ok=True)
                    except Exception:
                        pass

                    preview = p.get("preview_content")
                    if not isinstance(preview, str) or not preview:
                        # Try to reconstruct from diff + on-disk baseline
                        base = (
                            _read_file_text_if_exists(self.root, path) or ""
                        )
                        diff_text = (
                            p.get("diff")
                            or p.get("patch_unified")
                            or p.get("patch")
                        )
                        if isinstance(diff_text, str) and diff_text.strip():
                            try:
                                preview = apply_unified_patch(
                                    base.replace("\r\n", "\n").replace(
                                        "\r", "\n"
                                    ),
                                    diff_text,
                                )
                            except Exception:
                                preview = base
                        else:
                            preview = base

                    try:
                        with open(dest, "w", encoding="utf-8") as fh:
                            fh.write(preview)
                    except Exception:
                        # best-effort; continue
                        logging.debug(
                            "preapply: failed to write preview for %s",
                            path,
                            exc_info=True,
                        )

                    # Build JSONL entry (concise)
                    entry = {
                        "path": path,
                        "rec_id": p.get("rec_id"),
                        "diff": p.get("diff"),
                        "preview_bytes": p.get("preview_bytes"),
                    }
                    patch_entries.append(entry)

                # Run quality gates against the temporary workspace to capture
                # formatter/linter/test results.
                try:
                    run_quality_gates(
                        root=Path(tempd),
                        st=self.st,
                        cfg=self.args.get("cfg"),
                        session_id=self._session_id,
                        job_id=self.job_id,
                        progress_error_cb=self._progress_error,
                    )
                    report["checks_ok"] = True
                except Exception as e:
                    report["checks_ok"] = False
                    report["error"] = str(e)
                    report["trace"] = traceback.format_exc()
            except Exception as e:
                report["checks_ok"] = False
                report["error"] = str(e)
                report["trace"] = traceback.format_exc()
            finally:
                # Compose JSONL patch text and attach to orchestrator state.
                try:
                    lines: List[str] = [
                        json.dumps(o, ensure_ascii=False)
                        for o in patch_entries
                    ]
                    patch_jsonl = "\n".join(lines)
                except Exception:
                    patch_jsonl = ""
                self._preapply_patch_jsonl = patch_jsonl
                self._preapply_report = report
                # Emit a run-level trace entry recording that no recommendations
                # were applied during dry-run.
                try:
                    self.st.trace.write(
                        "applied_recommendations",
                        "run",
                        {
                            "applied_recommendation_ids": applied_rec_ids,
                            "applied_edits_total": applied_edits_total,
                            "brief_hash": self._project_brief_hash,
                        },
                    )
                except Exception:
                    pass
                # Clean up tempdir if it was created
                if tempd and os.path.isdir(tempd):
                    try:
                        shutil.rmtree(tempd, ignore_errors=True)
                    except Exception:
                        pass

                # Emit a final dry-run result so outer orchestrator layers can
                # return the patch + report.
                try:
                    self._emit_result_and_done(
                        ok=report.get("checks_ok", False),
                        summary=(
                            "Dry-run pre-apply completed; no files were written."
                            if report.get("checks_ok")
                            else "Dry-run pre-apply completed with failing checks."
                        ),
                    )
                except Exception:
                    logging.debug(
                        "preapply: _emit_result_and_done failed",
                        exc_info=True,
                    )
                return

        # --------- optional git snapshot/push (after all applied recs) ----------
        if bool(self.args.get("git_snapshot")) or bool(
            self.args.get("git_push")
        ):
            if not dry_run and applied_edits_total > 0:
                try:
                    from ..vcs.commit import git_snapshot  # local import

                    commit_msg = self._build_commit_message_from_results(
                        applied_results,
                        applied_edits_total,
                    )
                    pushed = bool(self.args.get("git_push"))

                    git_snapshot(self.root, message=commit_msg, push=pushed)
                    self._progress(
                        "git_snapshot",
                        pushed=pushed,
                        message=commit_msg,
                        job_id=self.job_id,
                    )
                except Exception as e:
                    logging.debug("git snapshot failed: %s", e)
                    self._progress(
                        "git_snapshot_failed",
                        error=str(e),
                        job_id=self.job_id,
                    )
                    self._progress_error(
                        "git_snapshot",
                        error=str(e),
                        trace=traceback.format_exc(),
                        job_id=self.job_id,
                    )

        # Emit run-level trace entry with the exact set of recommendation IDs that were applied.
        try:
            self.st.trace.write(
                "applied_recommendations",
                "run",
                {
                    "applied_recommendation_ids": applied_rec_ids,
                    "applied_edits_total": applied_edits_total,
                    "brief_hash": self._project_brief_hash,
                },
            )
        except Exception:
            # best-effort: don't fail the run if tracing fails
            pass

        # --- NEW: run a post-apply repository refresh to update cards/project_map ---
        # Only run refresh for real (non-dry-run) applies where edits were made.
        # This preserves dry-run semantics: dry-run runs must not persist cards/project_map.
        self._post_apply_refresh_failed = False
        self._post_apply_refresh_error = None
        if (not dry_run) and applied_edits_total > 0:
            try:
                _events.status(
                    "Starting post-apply refresh of cards/project_map",
                    stage="apply",
                    session_id=self._session_id,
                    job_id=self.job_id,
                    applied_recommendation_ids=applied_rec_ids,
                )
                self._progress(
                    "refresh_started",
                    stage="apply",
                    applied_recommendation_ids=applied_rec_ids,
                    job_id=self.job_id,
                )

                # Local import to avoid potential import cycles; adapt kwargs to
                # the real apply_and_refresh signature if it differs.
                from ..stages.apply_and_refresh import apply_and_refresh  # type: ignore

                # TODO: ensure apply_and_refresh accepts these keyword args; adapt if necessary.
                refresh_res = apply_and_refresh(
                    root=self.root,
                    st=self.st,
                    applied_rec_ids=applied_rec_ids,
                    dry_run=False,
                )

                # Interpret result: accept dicts or objects with an 'ok' indicator.
                refresh_ok = True
                refresh_err = None
                if refresh_res is None:
                    refresh_ok = True
                elif isinstance(refresh_res, dict):
                    refresh_ok = bool(refresh_res.get("ok", True))
                    if not refresh_ok:
                        refresh_err = refresh_res.get("error") or refresh_res.get("message") or str(refresh_res)
                else:
                    # object-like result
                    refresh_ok = bool(getattr(refresh_res, "ok", True))
                    if not refresh_ok:
                        refresh_err = getattr(refresh_res, "error", None) or getattr(refresh_res, "message", None) or str(refresh_res)

                if refresh_ok:
                    _events.status(
                        "Post-apply refresh succeeded",
                        stage="apply",
                        session_id=self._session_id,
                        job_id=self.job_id,
                        applied_recommendation_ids=applied_rec_ids,
                    )
                    self._progress(
                        "refresh_succeeded",
                        stage="apply",
                        applied_recommendation_ids=applied_rec_ids,
                        job_id=self.job_id,
                    )
                    refresh_entry = {"refresh": {"ok": True, "result": refresh_res}}
                else:
                    self._post_apply_refresh_failed = True
                    self._post_apply_refresh_error = str(refresh_err)
                    _events.status(
                        "Post-apply refresh failed",
                        stage="apply",
                        session_id=self._session_id,
                        job_id=self.job_id,
                        error=str(refresh_err),
                        applied_recommendation_ids=applied_rec_ids,
                    )
                    self._progress_error(
                        "refresh_failed",
                        error=str(refresh_err),
                        job_id=self.job_id,
                    )
                    refresh_entry = {"refresh": {"ok": False, "error": str(refresh_err)}}
            except Exception as e:
                # Record exception details and surface via events/progress
                self._post_apply_refresh_failed = True
                self._post_apply_refresh_error = traceback.format_exc()
                _events.status(
                    "Post-apply refresh raised exception",
                    stage="apply",
                    session_id=self._session_id,
                    job_id=self.job_id,
                    error=str(e),
                    applied_recommendation_ids=applied_rec_ids,
                )
                self._progress_error(
                    "refresh_exception",
                    error=str(e),
                    trace=traceback.format_exc(),
                    job_id=self.job_id,
                )
                refresh_entry = {"refresh": {"ok": False, "error": str(e), "trace": traceback.format_exc()}}

            # Attach refresh outcome to the per-run apply results so external consumers
            # can observe refresh success/failure alongside applied recommendation data.
            try:
                if isinstance(self._apply_results, list):
                    self._apply_results.append(refresh_entry)
            except Exception:
                pass

    def _build_commit_message_from_results(
        self,
        applied_results: List[Any],
        applied_edits_total: int,
    ) -> str:
        """
        Build a commit message summarizing applied recommendations.

        Prefers human-readable recommendation titles; falls back to
        counts and the run focus if needed.
        """
        # No applied recommendations – fall back to edit count / focus.
        if not applied_results:
            if applied_edits_total > 0:
                focus_text = (self.args.get("focus") or "") or ""
                return (
                    f"aidev: applied {applied_edits_total} edits"
                    + (
                        f" — {str(focus_text)[:100]}"
                        if focus_text
                        else ""
                    )
                )
            return "aidev: no recommendations applied"

        # Collect human-readable titles where available.
        titles: List[str] = []
        for r in applied_results:
            title = ""
            if hasattr(r, "title"):
                title = str(getattr(r, "title") or "").strip()
            elif isinstance(r, dict):
                title = str(
                    r.get("title")
                    or r.get("summary")
                    or r.get("id")
                    or ""
                ).strip()
            if title:
                titles.append(title)

        if titles:
            joined = "; ".join(titles)
            return (
                f"aidev: applied {len(applied_results)} rec(s) – "
                f"{joined[:120]}"
            )

        # Fallback: summarize by number of changed files.
        total_files = 0
        for r in applied_results:
            paths = []
            if hasattr(r, "changed_paths"):
                paths = getattr(r, "changed_paths") or []
            elif isinstance(r, dict):
                paths = r.get("changed_paths") or []
            try:
                total_files += len(paths)
            except TypeError:
                pass

        if total_files:
            return (
                f"aidev: applied {len(applied_results)} rec(s), "
                f"changed {total_files} file(s)"
            )

        # Final fallback – match old behavior, using focus text if present.
        focus_text = (self.args.get("focus") or "") or ""
        return (
            f"aidev: applied {applied_edits_total} edits"
            + (f" — {str(focus_text)[:100]}" if focus_text else "")
        )
