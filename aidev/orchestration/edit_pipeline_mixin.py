# aidev/orchestration/edit_pipeline_mixin.py
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..cards import KnowledgeBase
from ..recommendations_io import save_recommendations
from ..stages.recommendations import llm_recommendations
from ..orchestration.approval_inbox import get_approval_inbox
from .. import events as _events


class EditPipelineMixin:
    """
    Mixin that provides the high-level LLM edit pipeline orchestration:

      1) Run recommendation phase (planning + plan-level approval).
      2) Run apply phase over the selected recommendations.

    Canonical pipeline sequence (per-recommendation semantics):
      propose → approval gate → apply → cards refresh → quality gates → proceed to next recommendation

    Notes:
      - Per-recommendation card refreshes must be performed by the apply stage
        (for example, using aidev/stages/apply_and_refresh.py). This mixin
        invokes the apply phase; it does not itself implement per-recommendation
        incremental refresh.
      - The post-pipeline heuristic refresh (performed after the whole run)
        is a fallback/aggregate refresh and MUST NOT be relied on to keep the
        KB current between recommendations.

    Assumes `self` is an Orchestrator-like object with:

      - root, st, args, auto_approve, job_id
      - _llm, _session_id, _approval_job_id
      - _phase_max_tokens, _chat_json, _chat, _with_timeout
      - _should_cancel, _progress, _progress_error
      - _emit_result_and_done, _job_update, _coerce_str_list, _arun
      - _run_apply_phase (implemented by a concrete subclass/mixin)
      - _project_brief_text, _project_brief_hash

    See aidev/stages/apply_and_refresh.py and aidev/stages/rec_apply.py for the
    recommended location to implement the per-recommendation refresh logic and
    to emit cards.refresh.start / cards.refresh.done events.
    """

    # ------------ public entry from Orchestrator.run() ------------

    def _run_llm_pipeline(
        self,
        *,
        kb: KnowledgeBase,
        meta: Dict[str, Any],
        ctx_blob: str,
        includes: List[str],
        excludes: List[str],
        focus: str,
    ) -> None:
        """
        High-level pipeline coordinator:

        1) Run recommendation phase (planning + plan-level approval).
        2) Run apply phase over the selected recommendations.
        """
        (
            recs,
            top_k_select,
            dry_run,
            approval_timeout_sec,
        ) = self._run_recommendations_phase(
            kb=kb,
            meta=meta,
            ctx_blob=ctx_blob,
            includes=includes,
            excludes=excludes,
            focus=focus,
        )

        # Nothing to do (no recs, or plan rejected / filtered away).
        if not recs:
            return

        # NOTE: The apply phase is expected to perform per-recommendation
        # incremental card refreshes (cards.refresh.start / cards.refresh.done
        # events) when REFRESH_CARDS_BETWEEN_RECS is true. The concrete apply
        # implementation (see aidev/stages/apply_and_refresh.py) should call into
        # the KB helper to refresh changed files and update the project_map
        # before returning so subsequent recommendations observe updated card
        # metadata. This mixin only invokes the apply phase; do not rely on the
        # post-pipeline heuristic refresh for between-rec freshness.
        self._run_apply_phase(
            kb=kb,
            meta=meta,
            recs=recs,
            includes=includes,
            excludes=excludes,
            focus=focus,
            top_k_select=top_k_select,
            dry_run=dry_run,
            approval_timeout_sec=approval_timeout_sec,
        )

        # Final heuristic-only refresh (fallback). Per-rec refresh should have
        # already occurred during apply. This block is an aggregate best-effort
        # pass that updates heuristic cards and persists a compact project map;
        # it is NOT a substitute for per-recommendation incremental refresh.
        try:
            # Emit a debug hint indicating whether a per-recommendation refresh
            # implementation is expected (heuristic). We don't change control
            # flow based on this hint.
            expected = (
                hasattr(kb, "refresh_cards")
                or os.getenv("REFRESH_CARDS_BETWEEN_RECS", "true").lower() != "false"
            )
            logging.debug("per-recommendation-refresh-expected=%s", expected)

            # emit a trace event so runs are auditable without LLM enrichment
            try:
                self.st.trace.write(
                    "project_map_refresh",
                    "repo_map",
                    {"brief_hash": self._project_brief_hash},
                )
            except Exception:
                # best-effort: don't fail the run if tracing fails
                pass

            # heuristic-only card refresh (changed-only to minimize work)
            try:
                kb.update_cards(changed_only=True)
            except Exception:
                # Some KB implementations may not have update_cards; best-effort.
                logging.debug(
                    "kb.update_cards failed or not available",
                    exc_info=True,
                )

            # Persist a compact project map if the KB exposes a saver/builder
            try:
                if hasattr(kb, "save_project_map"):
                    # prefer to pass Path if supported
                    try:
                        kb.save_project_map(self.root)
                    except TypeError:
                        kb.save_project_map(str(self.root))
                elif hasattr(kb, "build_project_map"):
                    try:
                        kb.build_project_map(self.root)
                    except TypeError:
                        kb.build_project_map(str(self.root))
            except Exception as e:
                logging.debug("project_map save failed: %s", e)
        except Exception:
            # swallow any unexpected errors in the non-critical post-refresh step
            logging.debug(
                "post-pipeline project map refresh encountered an error",
                exc_info=True,
            )

    # -------------------- recommendations phase --------------------

    def _run_recommendations_phase(
        self,
        *,
        kb: KnowledgeBase,
        meta: Dict[str, Any],
        ctx_blob: str,
        includes: List[str],
        excludes: List[str],
        focus: str,
    ) -> Tuple[List[Dict[str, Any]], int, bool, float]:
        """
        Handles the recommendations part of the pipeline:

        - Builds the developer_focus string
        - Calls llm_recommendations
        - Handles empty result
        - Applies approved_rec_ids filtering (if any)
        - Persists and emits recommendations
        - Runs plan-level approval (unless auto_approve)
        - Computes top_k_select, dry_run, approval_timeout_sec

        Returns:
            (recs, top_k_select, dry_run, approval_timeout_sec)
        """
        # --------- recommendations ----------
        self._rid_recs = _events.progress_start(
            "recommendations",
            detail="Generating improvement plan (high-impact recommendations)…",
            session_id=self._session_id,
            job_id=self.job_id,
        )
        self._job_update(
            stage="recommendations",
            message="Generating plan…",
            progress_pct=15,
        )

        # Compute the developer's goal string (focus/message)
        focus_str = str(self.args.get("focus") or self.args.get("message") or "")
        developer_focus = focus_str.strip()

        strategy_note = str(self.args.get("strategy_note") or "").strip() or None

        recs_max_tokens = self._phase_max_tokens("recommendations")

        # Heuristic-only card refresh before running recommendations. This avoids
        # invoking any LLM-based card enrichment during recommendation runs.
        try:
            try:
                self.st.trace.write(
                    "heuristic_card_refresh",
                    "cards",
                    {
                        "changed_only": True,
                        "brief_hash": self._project_brief_hash,
                    },
                )
            except Exception:
                # tracing best-effort
                pass

            try:
                kb.update_cards(changed_only=True)
            except Exception:
                logging.debug(
                    "kb.update_cards failed or not available (pre-recommendations)",
                    exc_info=True,
                )

            # Optionally persist a project map as part of the pre-refresh (best-effort)
            try:
                if hasattr(kb, "save_project_map"):
                    try:
                        kb.save_project_map(self.root)
                    except TypeError:
                        kb.save_project_map(str(self.root))
                elif hasattr(kb, "build_project_map"):
                    try:
                        kb.build_project_map(self.root)
                    except TypeError:
                        kb.build_project_map(str(self.root))
            except Exception:
                logging.debug(
                    "pre-recommendation project_map save failed",
                    exc_info=True,
                )
        except Exception:
            logging.debug(
                "pre-recommendation heuristic refresh encountered an error",
                exc_info=True,
            )

        # Wire orchestrator's LLM wrappers into the recommendations stage
        def _rec_chat_json(
            system_text: str,
            user_payload: Any,
            schema: Dict[str, Any],
            temperature: float,
            phase: str,
            max_tokens: Optional[int],
        ):
            return self._chat_json(
                system_text,
                user_payload,
                schema=schema,
                temperature=temperature,
                phase=phase,
                inject_brief=False,
                max_tokens=max_tokens,
            )

        def _rec_chat_text(
            system_text: str,
            user_payload: Any,
            phase: str,
            max_tokens: Optional[int],
        ) -> str:
            return self._chat(
                system_text,
                user_payload,
                phase=phase,
                inject_brief=False,
                max_tokens=max_tokens,
            )

        recs = llm_recommendations(
            chat_json_fn=_rec_chat_json,
            chat_text_fn=_rec_chat_text,
            project_brief_text=self._project_brief_text or "",
            meta=meta,
            developer_focus=developer_focus,
            strategy_note=strategy_note,
            max_tokens=recs_max_tokens,
            # progress_cb / error_cb are optional; you can wire them later if desired
        )

        # --------- empty recommendations => graceful "nothing to do" ---------
        if not recs:
            _events.progress_finish(
                "recommendations",
                ok=True,
                recId=self._rid_recs,
                count=0,
                session_id=self._session_id,
                job_id=self.job_id,
            )

            _events.no_recommendations(
                message="No recommendations were generated for this request.",
                reason="empty_result",
                session_id=self._session_id,
                job_id=self.job_id,
            )

            self._progress(
                "recommendations_empty",
                stage="plan",
                reason="empty_result",
                job_id=self.job_id,
            )
            return [], 0, bool(self.args.get("dry_run")), 0.0

        # --------- non-empty recommendations path ---------
        for i, r in enumerate(recs, 1):
            r.setdefault("id", f"rec-{i}")
            
        # --------- optional 'approved_rec_ids' filtering ---------
        approved = self._coerce_str_list(self.args.get("approved_rec_ids"))
        if approved:
            approved_set = set(approved)
            before = len(recs)
            recs = [
                r
                for r in recs
                if str(r.get("id") or "").strip() in approved_set
            ]

            _events.status(
                "Filtered recommendations by approved_rec_ids",
                where="plan_filtered",
                session_id=self._session_id,
                job_id=self.job_id,
                before=before,
                after=len(recs),
                approved_rec_ids=approved,
            )

            if not recs:
                self._progress(
                    "plan_filtered_empty",
                    message=(
                        "No recommendations left after filtering by "
                        "approved_rec_ids."
                    ),
                    job_id=self.job_id,
                )
                self._emit_result_and_done(
                    ok=False,
                    summary="No recommendations selected after filtering.",
                )
                return [], 0, bool(self.args.get("dry_run")), 0.0

        self.st.trace.write(
            "RECOMMENDATIONS",
            "llm",
            {
                "count": len(recs),
                "items": recs,
                "model": getattr(self._llm, "model", None),
                "brief_hash": self._project_brief_hash,
            },
        )
        _events.progress_finish(
            "recommendations",
            ok=True,
            recId=self._rid_recs,
            count=len(recs),
            session_id=self._session_id,
            job_id=self.job_id,
        )

        try:
            saved = save_recommendations(recs)
            _events.status(
                "recommendations_saved",
                stage="recommendations",
                session_id=self._session_id,
                count=len(saved),
                job_id=self.job_id,
            )
        except Exception as e:
            logging.debug("Failed to persist recommendations: %s", e)

        # Canonical recommendations event for the UI
        _events.recommendations(
            recs,
            session_id=self._session_id,
            job_id=self.job_id,
        )
        self._progress(
            "pipeline",
            stage="recommend_done",
            count=len(recs),
            job_id=self.job_id,
        )

        if self._should_cancel():
            self._progress(
                "apply_skip",
                reason="cancelled",
                stage="pipeline",
                job_id=self.job_id,
            )
            self._emit_result_and_done(
                ok=False,
                summary="Run cancelled after recommendations.",
            )
            return [], 0, bool(self.args.get("dry_run")), 0.0

        # Shared config used by per-rec stages
        cfg = self.args.get("cfg") or {}
        if not isinstance(cfg, dict):
            cfg = {}
        cards_cfg = (
            cfg.get("cards", {}) if isinstance(cfg.get("cards"), dict) else {}
        )
        top_k_select = int(
            self.args.get("cards_top_k")
            or cards_cfg.get("default_top_k", 24)
        )

        dry_run = bool(self.args.get("dry_run"))

        # Compute approval timeout once
        try:
            _env_to = os.getenv("AIDEV_APPROVAL_TIMEOUT_SEC")
            approval_timeout_sec = float(
                self.args.get("approval_timeout_sec")
                or (_env_to if _env_to is not None else 3600.0)
            )
        except Exception:
            approval_timeout_sec = 3600.0

        # Plan-level approval gate: ask the user once before any edits
        if not self.auto_approve:
            approvals = get_approval_inbox()

            # Build a lightweight "files" list for the UI. At the plan stage
            # we may not know exact files yet, so we just surface rec IDs/titles.
            files_meta: List[Dict[str, Any]] = []
            for rec in recs:
                title = str(
                    rec.get("title") or rec.get("summary") or ""
                ).strip()
                rec_id = str(rec.get("id") or "").strip()
                label = title or rec_id or "(recommendation)"
                files_meta.append(
                    {
                        "path": label,
                        "added": 0,
                        "removed": 0,
                        "why": rec.get("why") or "",
                    }
                )

            if not files_meta:
                files_meta.append(
                    {
                        "path": "(plan)",
                        "added": 0,
                        "removed": 0,
                        "why": "Approval for generated recommendations",
                    }
                )

            summary_line = f"Apply {len(recs)} recommendation(s)"
            meta_plan = {
                "stage": "plan",
                "recommendation_ids": [rec.get("id") for rec in recs],
            }

            async def _await_plan_approval():
                token = await approvals.request(
                    session_id=self._session_id or "",
                    job_id=self._approval_job_id,
                    summary=summary_line,
                    risk="low",
                    files=files_meta,
                    meta=meta_plan,
                )
                req = await approvals.wait(
                    token,
                    timeout=approval_timeout_sec,
                )
                return req

            try:
                req = self._arun(_await_plan_approval())
            except Exception as e:
                self._progress_error(
                    "plan_approval",
                    error=str(e),
                    job_id=self.job_id,
                )
                self._emit_result_and_done(
                    ok=False,
                    summary=f"Plan approval failed: {e}",
                )
                return [], top_k_select, dry_run, approval_timeout_sec

            if req.decision != "approved":
                reason = (req.reason or "").strip()
                if not reason:
                    reason = (
                        "Plan / recommendations were not approved by the user."
                    )
                _events.status(
                    "Plan not approved; stopping run.",
                    stage="plan",
                    session_id=self._session_id,
                    job_id=self.job_id,
                    detail=reason,
                )
                self._emit_result_and_done(
                    ok=False,
                    summary=f"Plan not approved: {reason}",
                )
                return [], top_k_select, dry_run, approval_timeout_sec

            _events.status(
                "Plan approved; continuing with target selection and edits.",
                stage="plan",
                session_id=self._session_id,
                job_id=self.job_id,
            )

        return recs, top_k_select, dry_run, approval_timeout_sec
