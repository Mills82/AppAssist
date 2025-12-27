# aidev/orchestration/edit_analyze_mixin.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..cards import KnowledgeBase
from .self_review_mixin import SelfReviewMixin


class EditAnalyzeMixin(SelfReviewMixin):
    """
    Mixin that encapsulates the per-recommendation target-analysis stage.

    It assumes `self` has:

      - root, _project_brief_text
      - _phase_max_tokens, _chat_json
      - _accumulate_cross_file_notes (from SelfReviewMixin)
    """

    def _run_target_analysis_for_rec(
        self,
        *,
        kb: KnowledgeBase,
        rec: Dict[str, Any],
        rid: str,
        focus: str,
        env_for_rec: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run the per-file 'analyze' stage for a single recommendation.

        This delegates to `run_analysis_for_recommendation` in
        `aidev/stages/analyze_stage_driver.py`, then:

          - merges any `updated_targets_envelope` back into `env_for_rec` in-place
          - attaches the aggregated analysis onto the recommendation as
            `rec["_analysis"]` (with `analysis_by_path` and `per_file` aliases)
          - also mirrors the payload to `rec["_target_analysis"]` for
            backwards-compat guidance helpers
          - seeds cross_file_notes (if the analysis returns them in a structured form)

        The return value is for introspection/testing only; the important effects
        are the mutations on `env_for_rec` and `rec`.
        """
        # Import here to avoid any circular import weirdness at module import time.
        try:
            from ..stages.analyze_stage_driver import (
                run_analysis_for_recommendation,
                AggregateAnalysisResult,
            )
        except Exception:
            logging.warning(
                "target_analysis: could not import run_analysis_for_recommendation; "
                "skipping analysis for rec %s",
                rid,
                exc_info=True,
            )
            return None

        # No targets => nothing to analyze.
        if not isinstance(env_for_rec, dict):
            logging.warning(
                "target_analysis: env_for_rec for rec %s is not a dict (got %s); "
                "skipping analysis.",
                rid,
                type(env_for_rec).__name__,
            )
            return None

        targets = env_for_rec.get("targets")
        if not (isinstance(targets, list) and targets):
            logging.info(
                "target_analysis: rec_id=%s has no targets after selection; skipping.",
                rid,
            )
            return None

        # Basic rec metadata
        try:
            title = str(rec.get("title") or rec.get("summary") or "").strip()
        except Exception:
            title = ""

        try:
            why = str(rec.get("why") or rec.get("reason") or "").strip()
        except Exception:
            why = ""

        # Chat wrapper for the analyze stage so it goes through orchestrator's
        # _chat_json and carries per-rec metadata + phase/max_tokens.
        def _chat_json_analyze(
            system_text: str,
            user_payload: Any,
            schema: Dict[str, Any],
            temperature: float,
            phase: str = "target_analysis",
            max_tokens: Optional[int] = None,
        ):
            payload = user_payload
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("rec_id", rid)
                if title:
                    payload.setdefault("rec_title", title)
                if why:
                    payload.setdefault("rec_reasoning", why)

            effective_phase = phase or "target_analysis"
            mt = max_tokens if max_tokens is not None else self._phase_max_tokens(
                effective_phase
            )

            # self._chat_json returns (data, raw_response), which matches ChatJsonFn
            return self._chat_json(
                system_text,
                payload,
                schema=schema,
                temperature=temperature,
                phase=effective_phase,
                inject_brief=False,
                max_tokens=mt,
            )

        max_tokens = self._phase_max_tokens("target_analysis")

        logging.info(
            "target_analysis: starting for rec_id=%s targets=%d",
            rid,
            len(targets),
        )

        try:
            result = run_analysis_for_recommendation(
                rec=rec,
                env_for_rec=env_for_rec,
                kb=kb,
                project_root=self.root,
                focus=focus,
                chat_json_fn=_chat_json_analyze,
                project_brief_text=self._project_brief_text or "",
                meta=meta,
                max_tokens=max_tokens,
            )
        except Exception:
            logging.warning(
                "target_analysis: failed for rec %s; skipping analysis.",
                rid,
                exc_info=True,
            )
            return None

        if result is None:
            logging.info(
                "target_analysis: rec_id=%s returned no result; skipping.",
                rid,
            )
            return None

        # Normalise AggregateAnalysisResult vs dict-shaped result (defensive).
        analysis_by_path: Dict[str, Any] = {}
        updated_envelope: Dict[str, Any] = {}
        global_cross_file_notes: Any = None

        if hasattr(result, "analysis_by_path"):
            # AggregateAnalysisResult-like
            analysis_by_path = getattr(result, "analysis_by_path", None) or {}
            updated_envelope = getattr(result, "updated_targets_envelope", None) or {}
            global_cross_file_notes = getattr(result, "global_cross_file_notes", None)
        elif isinstance(result, dict):
            # In case someone ever returns a plain dict from tests/hooks.
            analysis_by_path = (
                result.get("analysis_by_path")
                or result.get("per_file")
                or {}
            )
            updated_envelope = result.get("updated_targets_envelope") or {}
            global_cross_file_notes = (
                result.get("analysis_cross_file_notes")
                or result.get("cross_file_notes")
                or result.get("global_cross_file_notes")
            )
        else:
            logging.debug(
                "target_analysis: rec_id=%s returned unexpected result type=%s",
                rid,
                type(result).__name__,
            )

        if not isinstance(analysis_by_path, dict):
            analysis_by_path = {}
        if not isinstance(updated_envelope, dict):
            updated_envelope = {}

        # Merge any refined envelope into the in-place env_for_rec, so subsequent
        # stages (propose_edits) see updated intents/flags/targets.
        if updated_envelope:
            try:
                env_for_rec.update(updated_envelope)
            except Exception:
                logging.debug(
                    "target_analysis: failed to merge updated_targets_envelope "
                    "for rec %s",
                    rid,
                    exc_info=True,
                )

        # Build the new aggregated analysis payload and attach it to the rec for
        # downstream consumers (edit stage, UI, self_review, tests).
        analysis_payload: Dict[str, Any] = {
            "rec_id": rid,
            "focus": focus,
            # New canonical mapping:
            "analysis_by_path": analysis_by_path,
            # Back-compat alias used by propose_edits and older tooling:
            "per_file": analysis_by_path,
        }
        if global_cross_file_notes is not None:
            # Canonical name for downstream tools:
            analysis_payload["analysis_cross_file_notes"] = global_cross_file_notes
            # Back-compat alias used by existing stages:
            analysis_payload["cross_file_notes"] = global_cross_file_notes
        if updated_envelope:
            analysis_payload["updated_targets_envelope"] = updated_envelope

        try:
            # Primary attachment for the edit stage.
            rec["_analysis"] = analysis_payload
            # Legacy alias so older helpers (e.g. _per_file_guidance_from_rec)
            # that look at rec["_target_analysis"] still have data.
            rec["_target_analysis"] = analysis_payload
        except Exception:
            logging.debug(
                "target_analysis: failed to attach analysis to rec %s",
                rid,
                exc_info=True,
            )

        # If the stage returned structured cross_file_notes (dict-shaped), we can
        # seed the cross-file context bucket used by the edit phase.
        if isinstance(global_cross_file_notes, dict):
            try:
                # _run_targets_and_edits_for_rec already called _init_cross_file_notes_for_rec,
                # so we just accumulate here.
                self._accumulate_cross_file_notes(
                    rec_id=rid,
                    notes=global_cross_file_notes,
                )
            except Exception:
                logging.debug(
                    "target_analysis: failed to seed cross_file_notes for rec %s",
                    rid,
                    exc_info=True,
                )

        logging.info(
            "target_analysis: done rec_id=%s analysis_entries=%d",
            rid,
            len(analysis_by_path),
        )

        return analysis_payload
