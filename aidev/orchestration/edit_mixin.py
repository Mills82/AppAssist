# aidev/orchestration/edit_mixin.py
from __future__ import annotations

import os
import logging
import traceback
import json
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

from .. import events as _events
from ..cards import KnowledgeBase
from ..schemas import file_edit_schema, targets_schema
from ..stages.targets import build_rec_targets
from ..stages.propose_edits import propose_edits_for_recommendations
from ..stages.rec_apply import (
    apply_rec_actions,
    sanitize_edits_and_proposed,
    validate_file_edits_schema,
)
from ..stages.generate_edits import generate_repair_for_path

from ..io_utils import (
    _read_file_text_if_exists,
    apply_unified_patch,
    generate_unified_diff,
)
from .edit_pipeline_mixin import EditPipelineMixin
from .edit_apply_mixin import EditApplyMixin
from .edit_analyze_mixin import EditAnalyzeMixin

# Local schemas for this mixin – kept here so this module is self-contained.
try:
    TARGETS_SCHEMA: Dict[str, Any] = targets_schema()
except Exception as e:
    logging.warning(
        "edit_mixin: Failed to load TARGETS_SCHEMA; falling back to empty schema: %s",
        e,
    )
    TARGETS_SCHEMA = {}

try:
    EDIT_SCHEMA: Dict[str, Any] = file_edit_schema()
except Exception as e:
    logging.warning(
        "edit_mixin: Failed to load EDIT_SCHEMA; falling back to empty schema: %s",
        e,
    )
    EDIT_SCHEMA = {}


class _TimeoutSentinel:
    pass


_TIMEOUT = _TimeoutSentinel()


def _concat_text(a: Optional[str], b: Optional[str]) -> Optional[str]:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return None
    if not a:
        return b
    if not b:
        return a
    return f"{a}\n\n{b}"


def _merge_notes_values(a: Any, b: Any) -> Any:
    """
    Merge two note values that may be:
      - str
      - dict
      - None
      - other scalar types

    Rules:
      - str + str -> concatenated text
      - dict + dict -> shallow merge with recursive handling for overlapping keys
      - str + dict -> store/merge str into dict["_text"]
      - dict + str -> store/merge str into dict["_text"]
      - None -> other
      - otherwise -> prefer b
    """
    if a is None:
        return b
    if b is None:
        return a

    if isinstance(a, str) and isinstance(b, str):
        return _concat_text(a, b)

    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, bv in b.items():
            if k not in out:
                out[k] = bv
            else:
                out[k] = _merge_notes_values(out.get(k), bv)
        return out

    # If we have a dict on either side, tuck string content into "_text"
    if isinstance(a, dict) and isinstance(b, str):
        out = dict(a)
        out["_text"] = _concat_text(out.get("_text"), b)
        return out

    if isinstance(a, str) and isinstance(b, dict):
        out = dict(b)
        out["_text"] = _concat_text(a, out.get("_text"))
        return out

    # Fallback: prefer the newer value
    return b


def _merge_structured_and_text_notes(
    base: Any,
    incoming: Any,
) -> Any:
    """
    Merge cross_file_notes values from multiple sources.

    This is intentionally permissive because:
      - older flows may supply strings
      - newer analyze/edit flows may supply dicts
      - schemas allow additionalProperties

    Return type:
      - str if both sides are strings
      - dict if either side is a dict
      - None if both are empty/None
    """
    merged = _merge_notes_values(base, incoming)

    # Normalize empty strings/dicts to None
    if isinstance(merged, str) and not merged.strip():
        return None
    if isinstance(merged, dict):
        # Strip empty _text if present
        if "_text" in merged and isinstance(merged["_text"], str) and not merged["_text"].strip():
            merged.pop("_text", None)
        return merged or None

    return merged


class OrchestratorEditMixin(EditAnalyzeMixin, EditPipelineMixin, EditApplyMixin):
    """
    Mixin that holds the heavy LLM edit pipeline logic so aidev/orchestrator.py
    can stay lean. This class assumes `self` is an Orchestrator-like object
    with:

      - root, st, args, auto_approve, job_id
      - _llm, _session_id, _approval_job_id
      - _timeout_targets, _timeout_edit, _targets_fallback_n
      - _rid_recs, _rid_targets, _rid_edits, _rid_checks
      - _project_brief_text, _project_brief_hash
      - methods: _phase_max_tokens, _chat_json, _chat, _with_timeout,
                 _should_cancel, _progress, _progress_error,
                 _emit_result_and_done,
                 validate_proposed_edits,
                 _job_update, _coerce_str_list,
                 _arun
    """

    def _ensure_rec_acceptance_criteria(
        self,
        rec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ensure this recommendation has an 'acceptance_criteria' list in a normalized shape.

        - If the rec already has acceptance_criteria, normalise it to a list[str].
        - If not, leave the recommendation unchanged.
        """
        crit = rec.get("acceptance_criteria")

        if isinstance(crit, list):
            cleaned = [str(c).strip() for c in crit if str(c).strip()]
            if cleaned:
                new_rec = dict(rec)
                new_rec["acceptance_criteria"] = cleaned
                return new_rec
            # fallback default if we somehow lost all criteria
            title = str(rec.get("title") or rec.get("summary") or "").strip()
            if title:
                new_rec = dict(rec)
                new_rec["acceptance_criteria"] = [
                    f"Demonstrate measurable progress on '{title}' with a small PR."
                ]
                return new_rec
            return rec

        # If it's a single string, normalise to a one-element list.
        if isinstance(crit, str) and crit.strip():
            new_rec = dict(rec)
            new_rec["acceptance_criteria"] = [crit.strip()]
            return new_rec

        # Otherwise leave as-is (no acceptance criteria).
        return rec

    # ---------------- per-rec targets + edits + repair ----------------

    def _run_targets_and_edits_for_rec(
        self,
        *,
        kb: KnowledgeBase,
        meta: Dict[str, Any],
        includes: List[str],
        excludes: List[str],
        focus: str,
        rec: Dict[str, Any],
        rid: str,
        idx: int,
        total_recs: int,
        top_k_select: int,
        dry_run: bool,
        edit_schema_retries: int,
        validation_repair_retries: int,
        chat_json_targets: Callable[..., Any],
        chat_json_edits: Callable[..., Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        For a single recommendation:

        - select targets
        - generate edits with schema-enforced retries
        - run optional pre-apply checks with targeted self-repair

        Returns:
            (edits, proposed)
        Raises:
            LLMFailure / IOFailure on hard errors.
        """
        from ..orchestrator import LLMFailure, IOFailure  # avoid circular at import-time

        # Ensure rec has acceptance_criteria
        rec = self._ensure_rec_acceptance_criteria(rec)

        # Derive human-readable reasoning/title for this recommendation
        rec_title = ""
        try:
            rec_title = str(rec.get("title") or rec.get("summary") or "").strip()
        except Exception:
            rec_title = ""

        rec_why = ""
        try:
            rec_why = str(rec.get("why") or "").strip()
        except Exception:
            rec_why = ""

        rec_reasoning = rec_why or focus or ""

        # Helper trace wrapper to ensure per-rec explainability in traces.
        def _trace_with_rec(event: str, sub: str, data: Any) -> None:
            """
            Wrap self.st.trace.write so that all LLM-stage traces for this
            recommendation carry rec_id + reasoning metadata.
            """
            if isinstance(data, dict):
                enriched = dict(data)
                enriched.setdefault("rec_id", rid)
                if rec_title:
                    enriched.setdefault("rec_title", rec_title)
                if rec_reasoning:
                    enriched.setdefault("rec_reasoning", rec_reasoning)
                self.st.trace.write(event, sub, enriched)
            else:
                # Non-dict payloads are passed through unchanged.
                self.st.trace.write(event, sub, data)

        # Initialize per-recommendation cross-file context so that edit-file
        # calls for this recommendation can share and accumulate notes.
        self._init_cross_file_notes_for_rec(rid)

        # --------- target selection (per-rec) ----------
        self._rid_targets = _events.progress_start(
            "target_select",
            detail=f"Selecting files for recommendation {rid} ({idx}/{total_recs})…",
            session_id=self._session_id,
            job_id=self.job_id,
        )
        self._job_update(
            stage="target_select",
            message=f"Selecting targets for recommendation {idx}/{total_recs}…",
            progress_pct=25,
        )

        # Per-rec wrapper around target-selection chat_json to inject rec_id + title.
        def _chat_json_targets_for_rec(
            system_text: str,
            user_payload: Any,
            schema: Dict[str, Any],
            temperature: float,
            phase: str = "target_select",
            max_tokens: Optional[int] = None,
            stage: Optional[str] = None,
            **kwargs,
        ):
            payload = user_payload
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("rec_id", rid)
                if rec_title:
                    payload.setdefault("rec_title", rec_title)

            effective_phase = (stage or phase or "target_select")
            mt = max_tokens if max_tokens is not None else self._phase_max_tokens(effective_phase)

            return chat_json_targets(
                system_text,
                payload,
                schema,
                temperature,
                phase=effective_phase,
                max_tokens=mt,
            )

        rec_targets_map = build_rec_targets(
            recs=[rec],
            kb=kb,
            meta=meta,
            includes=includes,
            excludes=excludes,
            focus=focus,
            top_k_select=top_k_select,
            chat_json_fn=_chat_json_targets_for_rec,
            schema=TARGETS_SCHEMA,
            timeout_runner=self._with_timeout,
            timeout_sentinel=_TIMEOUT,
            timeout_sec=self._timeout_targets,
            fallback_n=self._targets_fallback_n,
            should_cancel=self._should_cancel,
            project_root=str(self.root),
            progress_cb=lambda event, payload: self._progress(event, **payload),
            trace_fn=lambda event, sub, data: _trace_with_rec(event, sub, data),
            project_brief_text=self._project_brief_text,
            brief_hash=self._project_brief_hash,
        )

        _events.progress_finish(
            "target_select",
            ok=bool(rec_targets_map),
            recId=self._rid_targets,
            recommendations=1,
            session_id=self._session_id,
            job_id=self.job_id,
        )

        # Extract this rec's envelope
        env_for_rec: Dict[str, Any] = {}
        if rec_targets_map:
            raw_env = rec_targets_map.get(rid)
            if isinstance(raw_env, dict):
                env_for_rec = raw_env
            else:
                # Fallback: take the first envelope-like value if rid wasn't found
                first_val = next(iter(rec_targets_map.values()), {})
                if isinstance(first_val, dict):
                    env_for_rec = first_val

        # --------- per-target analysis for this recommendation ----------
        try:
            self._run_target_analysis_for_rec(
                kb=kb,
                rec=rec,
                rid=rid,
                focus=focus,
                env_for_rec=env_for_rec,
                meta=meta,
            )
        except Exception:
            # Best-effort only; analysis must never break the pipeline, but we want
            # to *see* if it blows up.
            logging.warning(
                "target_analysis: hard failure for rec %s; skipping analysis.",
                rid,
                exc_info=True,
            )

        # Now count the (possibly refined later) target paths.
        targets_list: List[str] = []
        if isinstance(env_for_rec.get("targets"), list):
            for t in env_for_rec["targets"]:
                if isinstance(t, dict):
                    p = t.get("path")
                    if isinstance(p, str) and p.strip():
                        targets_list.append(p)

        total_files = len(targets_list)
        if total_files == 0:
            self._progress("no_targets", rec_id=rid, job_id=self.job_id)
            self._progress_error(
                "targets",
                reason="no_targets_selected",
                rec_id=rid,
                job_id=self.job_id,
            )
            raise LLMFailure(f"No target files selected for recommendation {rid}")

        def _on_edit_error(where: str, payload: Dict[str, Any]) -> None:
            # Always record the error as before
            self._progress_error(where, **payload)

            # Special-case: malformed / missing edits for this file
            if where == "generate_edit" and payload.get("reason") == "no_edit":
                msg = payload.get(
                    "message",
                    "The AI could not generate code changes for this step.",
                )
                extra = {k: v for k, v in payload.items() if k != "message"}
                _events.status(
                    str(msg),
                    where="generate_edit",
                    stage="generate_edits",
                    session_id=self._session_id,
                    job_id=self.job_id,
                    **extra,
                )

        # Per-rec wrapper around edit-phase chat_json to inject/accumulate cross-file context
        # and attach per-rec metadata (rec_id + title).
        def _chat_json_edits_for_rec(
            system_text: str,
            user_payload: Any,
            schema: Dict[str, Any],
            temperature: float,
            phase: str = "generate_edits",
            max_tokens: Optional[int] = None,
            stage: Optional[str] = None,
            **kwargs,
        ):
            """
            Per-recommendation wrapper around the edit-phase chat_json call.
            ...
            """

            payload = user_payload

            # Inject rec-scoped cross_file_notes + rec metadata into the payload if it's a dict.
            if isinstance(payload, dict):
                payload = dict(payload)

                analysis_notes = payload.pop("analysis_cross_file_notes", None)

                try:
                    ctx = self._get_cross_file_notes_for_rec(rid)
                except Exception:
                    ctx = None

                base_notes = payload.get("cross_file_notes")

                merged_notes = _merge_structured_and_text_notes(base_notes, ctx)
                merged_notes = _merge_structured_and_text_notes(merged_notes, analysis_notes)

                if merged_notes is not None:
                    payload["cross_file_notes"] = merged_notes

                payload.setdefault("rec_id", rid)
                if rec_title:
                    payload.setdefault("rec_title", rec_title)

            # Treat stage as an alias of phase if provided.
            effective_phase = (stage or phase or "generate_edits")
            mt = max_tokens if max_tokens is not None else self._phase_max_tokens(effective_phase)

            data, res = chat_json_edits(
                system_text,
                payload,
                schema,
                temperature,
                phase=effective_phase,
                max_tokens=mt,
            )

            # Extract and accumulate cross_file_notes if present in the response.
            try:
                if isinstance(data, dict):
                    # Case 0: a plain cross_file_notes at top level
                    if isinstance(data.get("cross_file_notes"), dict):
                        self._accumulate_cross_file_notes(
                            rec_id=rid,
                            notes=data["cross_file_notes"],
                        )

                    # Case 1: single object with nested details.cross_file_notes
                    details = data.get("details")
                    if isinstance(details, dict) and isinstance(
                        details.get("cross_file_notes"), dict
                    ):
                        self._accumulate_cross_file_notes(
                            rec_id=rid,
                            notes=details["cross_file_notes"],
                        )

                    # Case 2: multi-edit bundle with an 'edits' list, where each
                    # entry may carry its own details.cross_file_notes.
                    edits_list = data.get("edits")
                    if isinstance(edits_list, list):
                        for e in edits_list:
                            if not isinstance(e, dict):
                                continue
                            d = e.get("details")
                            if isinstance(d, dict) and isinstance(
                                d.get("cross_file_notes"), dict
                            ):
                                self._accumulate_cross_file_notes(
                                    rec_id=rid,
                                    notes=d["cross_file_notes"],
                                )
            except Exception:
                # Best-effort: never derail the pipeline if aggregation fails.
                logging.debug(
                    "cross_file_notes aggregation failed for rec %s", rid, exc_info=True
                )

            return data, res

        # --- generate + validate edits, with schema-enforced retries ---
        edits: List[Dict[str, Any]] = []
        proposed: List[Dict[str, Any]] = []
        valid_edits: bool = False
        last_schema_errors: List[str] = []

        for attempt in range(1, edit_schema_retries + 1):
            result = propose_edits_for_recommendations(
                root=self.root,
                recs=[rec],
                rec_targets=rec_targets_map,
                focus=focus,
                chat_json_fn=_chat_json_edits_for_rec,
                edit_schema=EDIT_SCHEMA,
                timeout_runner=self._with_timeout,
                timeout_sentinel=_TIMEOUT,
                timeout_sec=self._timeout_edit,
                should_cancel=self._should_cancel,
                progress_cb=lambda event, payload: self._progress(event, **payload),
                error_cb=_on_edit_error,
                trace_fn=lambda event, sub, data: _trace_with_rec(event, sub, data),
                brief_hash=self._project_brief_hash,
            )
            # Support both 2-tuple and 3-tuple returns for forward/backward compat
            if isinstance(result, tuple):
                if len(result) >= 2:
                    proposed_raw, edits_raw = result[0], result[1]
                else:
                    raise LLMFailure(
                        "propose_edits_for_recommendations returned too few values"
                    )
            else:
                raise LLMFailure(
                    f"propose_edits_for_recommendations returned unexpected type: {type(result)!r}"
                )

            # Apply any structured actions on the recommendation (stage helper)
            edits_tmp, proposed_tmp = apply_rec_actions(
                root=self.root,
                rec=rec,
                edits=edits_raw,
                proposed=proposed_raw,
                dry_run=dry_run,
                progress_cb=lambda event, payload: self._progress(event, **payload),
                progress_error_cb=lambda where, payload: self._progress_error(where, **payload),
                trace_obj=self.st.trace,
            )

            # Sanitize and align with the proposed bundle (stage helper)
            edits_tmp, proposed_tmp = sanitize_edits_and_proposed(
                root=self.root,
                edits=edits_tmp,
                proposed=proposed_tmp,
                rec_id=rid,
                progress_cb=lambda event, payload: self._progress(event, **payload),
                progress_error_cb=lambda where, payload: self._progress_error(where, **payload),
            )

            if not edits_tmp:
                # Nothing usable this attempt; optionally retry if we have attempts left.
                self._progress(
                    "no_edits_attempt",
                    rec_id=rid,
                    attempt=attempt,
                    attempts=edit_schema_retries,
                    job_id=self.job_id,
                )
                if attempt < edit_schema_retries:
                    self._attempt_schema_repair(
                        rec_id=rid,
                        attempt=attempt,
                        attempts=edit_schema_retries,
                        errors=["No usable edits generated"],
                    )
                    continue
                # fall through and fail after loop
                edits = []
                proposed = []
                break

            # Strict JSON Schema validation for each FileEdit (stage helper)
            ok_schema, schema_errors = validate_file_edits_schema(
                edits_tmp,
                rec_id=rid,
                schema=EDIT_SCHEMA,
                progress_cb=lambda event, payload: self._progress(event, **payload),
            )

            if ok_schema:
                edits = edits_tmp
                proposed = proposed_tmp
                valid_edits = True
                break

            # Schema invalid: record and optionally retry
            last_schema_errors = schema_errors or ["Unknown schema validation error"]
            if attempt < edit_schema_retries:
                self._attempt_schema_repair(
                    rec_id=rid,
                    attempt=attempt,
                    attempts=edit_schema_retries,
                    errors=last_schema_errors,
                )
            else:
                self._progress_error(
                    "edit_schema_validation",
                    rec_id=rid,
                    attempt=attempt,
                    attempts=edit_schema_retries,
                    error="; ".join(last_schema_errors[:3]),
                )

        # Final gate after schema retries for this generation cycle
        if not edits or not valid_edits:
            _events.progress_finish(
                "generate_edits",
                ok=False,
                recId=self._rid_edits,
                session_id=self._session_id,
                job_id=self.job_id,
            )
            self._progress("no_edits", rec_id=rid, job_id=self.job_id)
            self._progress_error(
                "edits",
                reason="none_generated_or_invalid",
                rec_id=rid,
                job_id=self.job_id,
                errors=last_schema_errors,
            )
            raise LLMFailure(
                f"No usable, schema-valid edits generated for recommendation {rid} "
                f"after {edit_schema_retries} attempt(s)."
            )

        _events.progress_finish(
            "generate_edits",
            ok=True,
            recId=self._rid_edits,
            files=total_files,
            session_id=self._session_id,
            job_id=self.job_id,
        )

        # --------- optional pre-apply checks in a temporary workspace ----------
        if not dry_run:
            validation = self.validate_proposed_edits(
                rec_id=rid,
                proposed=proposed,
            )
            if not getattr(validation, "ok", False):
                # validation failed; optionally perform targeted repair
                if validation_repair_retries <= 0:
                    # validate_proposed_edits already emitted detailed events
                    # and _progress_error on failure, so we just stop the pipeline.
                    self._progress_error(
                        "preapply_checks",
                        rec_id=rid,
                        job_id=self.job_id,
                        attempts=0,
                        reason="validation_failed_no_repairs_configured",
                    )
                    raise IOFailure(
                        f"Pre-apply checks failed for recommendation {rid}; "
                        f"not applying edits."
                    )

                edits, proposed = self._run_validation_repair_cycle(
                    rec=rec,
                    rid=rid,
                    focus=focus,
                    edits=edits,
                    proposed=proposed,
                    validation=validation,
                    validation_repair_retries=validation_repair_retries,
                    chat_json_edits=_chat_json_edits_for_rec,
                )

        # If we reach here, either dry_run is True or validation passed/was repaired.
        return edits, proposed

    # ------------------- small repair helpers -------------------

    def _attempt_schema_repair(
        self,
        *,
        rec_id: str,
        attempt: int,
        attempts: int,
        errors: List[str],
    ) -> None:
        """
        Emit events + guidance when schema validation fails, telling the model
        to try again with the exact file_edit schema.
        """
        summary = "; ".join(errors[:3])
        self._progress_error(
            "edit_schema_validation",
            rec_id=rec_id,
            attempt=attempt,
            attempts=attempts,
            error=summary,
        )
        _events.status(
            "Edit schema validation failed; asking the model to try again "
            "using this exact file_edit schema.",
            stage="generate_edits",
            session_id=self._session_id,
            job_id=self.job_id,
            rec_id=rec_id,
            attempt=attempt,
            attempts=attempts,
        )

    def _extract_validation_feedback_per_file(
        self,
        validation_obj: Any,
        max_len: int = 4000,
    ) -> Dict[str, str]:
        """
        Extract a mapping of file path -> joined diagnostic text from a
        ValidationResult-like object, based on .errors and .details structure.

        Returns:
            { "path/to/file.py": "error1\nerror2\n...", ... }
        """
        file_map: Dict[str, List[str]] = {}

        if validation_obj is None:
            return {}

        def _record(line: str, files: List[str]) -> None:
            if not line:
                return
            if not isinstance(files, list):
                files_local: List[str] = []
            else:
                files_local = [f for f in files if isinstance(f, str) and f]
            for f in files_local:
                file_map.setdefault(f, []).append(line)

        # Prefer explicit .errors list on the object
        errs = getattr(validation_obj, "errors", None)
        if isinstance(errs, (list, tuple)):
            for e in errs:
                files: List[str] = []
                if isinstance(e, str):
                    line = e
                elif isinstance(e, dict):
                    msg = str(e.get("message") or e)
                    cid = e.get("check_id") or e.get("tool")
                    files = e.get("files") or []
                    line = msg
                    if cid:
                        line = f"[{cid}] {line}"
                else:
                    continue
                _record(line, files)

        # Fallback: errors + failing checks inside .details
        details = getattr(validation_obj, "details", None)
        if isinstance(details, dict):
            for e in details.get("errors") or []:
                files = []
                if isinstance(e, str):
                    line = e
                elif isinstance(e, dict):
                    msg = str(e.get("message") or e)
                    cid = e.get("check_id") or e.get("tool")
                    files = e.get("files") or []
                    line = msg
                    if cid:
                        line = f"[{cid}] {line}"
                else:
                    continue
                _record(line, files)

            for chk in details.get("checks") or []:
                if not isinstance(chk, dict) or chk.get("ok"):
                    continue
                label = chk.get("label") or chk.get("id") or chk.get("tool") or "check"
                tail = (chk.get("stderr_tail") or chk.get("stdout_tail") or "").strip()
                line = f"{label}: {tail}" if tail else label
                files = chk.get("files") or []
                _record(line, files)

        # Join and truncate per-file diagnostics
        out: Dict[str, str] = {}
        for path, lines in file_map.items():
            text = "\n".join(lines)
            if len(text) > max_len:
                text = text[: max_len - 200] + "\n...[truncated]..."
            out[path] = text

        return out

    def _build_validation_feedback_text(self, validation_obj: Any, max_len: int = 6000) -> str:
        """
        Build a global, human-readable feedback string from a ValidationResult-like object.
        Used when there are no file-scoped diagnostics.
        """
        if validation_obj is None:
            return "Validation failed, but no diagnostics were provided."

        parts: list[str] = []

        errs = getattr(validation_obj, "errors", None)
        if isinstance(errs, (list, tuple)):
            for e in errs:
                if isinstance(e, str):
                    parts.append(e)
                elif isinstance(e, dict):
                    msg = str(e.get("message") or e)
                    cid = e.get("check_id") or e.get("tool")
                    if cid:
                        msg = f"[{cid}] {msg}"
                    parts.append(msg)

        details = getattr(validation_obj, "details", None)
        if isinstance(details, dict):
            for e in details.get("errors") or []:
                if isinstance(e, str):
                    parts.append(e)
                elif isinstance(e, dict):
                    msg = str(e.get("message") or e)
                    cid = e.get("check_id") or e.get("tool")
                    if cid:
                        msg = f"[{cid}] {msg}"
                    parts.append(msg)

            for chk in details.get("checks") or []:
                if not isinstance(chk, dict) or chk.get("ok"):
                    continue
                label = chk.get("label") or chk.get("id") or chk.get("tool") or "check"
                tail = (chk.get("stderr_tail") or chk.get("stdout_tail") or "").strip()
                line = f"{label}: {tail}" if tail else label
                parts.append(line)

        if not parts:
            return "Validation failed, but diagnostics were empty or unstructured."

        text = "\n".join(parts)
        if len(text) > max_len:
            text = text[: max_len - 200] + "\n...[truncated]..."
        return text

    def _record_preapply_warning(
        self,
        *,
        rec_id: str,
        status: str,
        validation_obj: Any,
    ) -> None:
        """
        Record a soft-failed pre-apply status for the given recommendation so
        that the UI can surface it to the user.

        - Builds a concise list of human-readable error strings.
        - Stashes them on `self._preapply_warnings[rec_id]`.
        - Emits a job_update + trace + status event with `preapply_status`
          and `preapply_errors` so the web UI can show a banner like:

              ⚠️ Pre-apply checks failed after automatic repairs (tests/linters may be red).
              You can still apply these changes, but you may need to fix issues manually.
        """
        # 1) Derive a list[str] of errors from the ValidationResult-like object.
        try:
            per_file = self._extract_validation_feedback_per_file(validation_obj)
        except Exception:
            per_file = {}

        errors: List[str] = []
        if per_file:
            for path, text in per_file.items():
                snippet = (text or "").strip()
                if snippet:
                    errors.append(f"{path}: {snippet}")
        else:
            try:
                global_text = self._build_validation_feedback_text(validation_obj)
            except Exception:
                global_text = "Pre-apply checks failed after automatic repairs."
            errors = [global_text]

        # 2) Stash on the orchestrator instance so other layers can inspect if needed.
        try:
            if not hasattr(self, "_preapply_warnings"):
                # { rec_id: { "status": str, "errors": list[str] } }
                self._preapply_warnings = {}
            self._preapply_warnings[rec_id] = {
                "status": status,
                "errors": errors,
            }
        except Exception:
            logging.debug(
                "edit_mixin: failed to stash _preapply_warnings for rec %s",
                rec_id,
                exc_info=True,
            )

        # 3) Emit an explicit job_update so the web UI has a simple hook.
        message = (
            "Pre-apply checks failed after automatic repairs; continuing with edits "
            "for manual review (tests/linters may be red)."
        )
        try:
            self._job_update(
                stage="checks",
                message=message,
                rec_id=rec_id,
                preapply_status=status,
                preapply_errors=errors,
            )
        except Exception:
            logging.debug(
                "edit_mixin: _job_update for preapply warning failed",
                exc_info=True,
            )

        # 4) Trace + status event for auditability / logs.
        try:
            self.st.trace.write(
                "preapply_warning",
                "checks",
                {
                    "rec_id": rec_id,
                    "status": status,
                    "errors": errors,
                    "brief_hash": getattr(self, "_project_brief_hash", None),
                },
            )
        except Exception:
            logging.debug(
                "edit_mixin: trace preapply_warning failed",
                exc_info=True,
            )

        try:
            _events.status(
                "Pre-apply checks failed after automatic repairs; continuing with edits for manual review.",
                stage="checks",
                session_id=getattr(self, "_session_id", None),
                job_id=getattr(self, "job_id", None),
                rec_id=rec_id,
                status=status,
                errors=errors,
            )
        except Exception:
            logging.debug(
                "edit_mixin: events.status preapply_warning failed",
                exc_info=True,
            )

    def _run_validation_repair_cycle(
        self,
        *,
        rec: Dict[str, Any],
        rid: str,
        focus: str,
        edits: List[Dict[str, Any]],
        proposed: List[Dict[str, Any]],
        validation: Any,
        validation_repair_retries: int,
        chat_json_edits: Callable[..., Any],
        file_overlays: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Attempt targeted, file-level repairs for a failing validation, without
        regenerating the entire edit set.

        Uses generate_repair_for_path with per-file diagnostics and the
        current preview_content as the baseline for each repair.

        If `file_overlays` (rel-path -> content) is provided, prefer those
        overlayed contents when reconstructing baselines/preview content so
        repairs are generated against the most up-to-date in-memory view.
        """
        from ..orchestrator import IOFailure  # avoid circular import at module load

        def _approx_size_for_path(path: str, proposed_bundle: List[Dict[str, Any]]) -> int:
            """
            Approximate size of the edited file content for `path`, in bytes.
            Prefer preview_content, then content, then diff length as a last resort.
            """
            for p in proposed_bundle:
                if not isinstance(p, dict):
                    continue
                if p.get("path") != path:
                    continue

                text = None
                # Prefer preview_content (what we'll actually write).
                if isinstance(p.get("preview_content"), str) and p["preview_content"]:
                    text = p["preview_content"]
                elif isinstance(p.get("content"), str) and p["content"]:
                    text = p["content"]
                elif isinstance(p.get("diff"), str) and p["diff"]:
                    text = p["diff"]
                elif isinstance(p.get("patch_unified"), str) and p["patch_unified"]:
                    text = p["patch_unified"]

                if isinstance(text, str):
                    return len(text.encode("utf-8"))

            return 0

        repair_attempt = 0
        current_edits = edits
        current_proposed = proposed
        current_validation = validation

        # Threshold above which we treat files as "large" when only soft issues are present.
        LARGE_FILE_THRESHOLD = int(os.getenv("AIDEV_LARGE_FILE_BYTES", "60000"))

        while repair_attempt < validation_repair_retries and not getattr(current_validation, "ok", False):
            repair_attempt += 1

            # Get severity classification from the current validation.
            v_details = getattr(current_validation, "details", {}) or {}
            has_blocking = bool(v_details.get("has_blocking"))
            has_soft = bool(v_details.get("has_soft"))
            severity = v_details.get("severity") or (
                "blocking" if has_blocking else "soft_only" if has_soft else "unknown"
            )

            # Build per-file diagnostics mapping; used to ensure each repair call only
            # sees errors for the file being fixed.
            per_file_feedback = self._extract_validation_feedback_per_file(current_validation)

            candidate_paths = {
                p.get("path") for p in current_proposed if isinstance(p.get("path"), str)
            }
            candidate_paths = {p for p in candidate_paths if p}

            if per_file_feedback:
                # We have file-scoped diagnostics: only repair files that appear in that map.
                failing_paths_raw = [p for p in candidate_paths if p in per_file_feedback]

                if not failing_paths_raw:
                    # Validation errors exist, but none of them correspond to these edited files.
                    # Don't try to "guess" repairs for clean files; let the caller handle the failure.
                    self._progress_error(
                        "validation_no_matching_files",
                        rec_id=rid,
                        job_id=self.job_id,
                        files=list(candidate_paths),
                    )
                    raise IOFailure(
                        "Validation failed, but no diagnostics referenced the edited files; "
                        "not attempting automatic per-file repair."
                    )

                # Apply size/severity rules:
                # - If any blocking issues exist, repair all failing files (even large).
                # - If only soft issues exist, skip very large files.
                failing_paths: List[str] = []
                for path in failing_paths_raw:
                    if has_blocking:
                        failing_paths.append(path)
                    else:
                        size_bytes = _approx_size_for_path(path, current_proposed)
                        if size_bytes and size_bytes > LARGE_FILE_THRESHOLD and severity == "soft_only":
                            # Soft-only issues on a large file: skip auto-repair, but log for the UI.
                            self._progress(
                                "validation_soft_skip_large_file",
                                rec_id=rid,
                                path=path,
                                size_bytes=size_bytes,
                                job_id=self.job_id,
                            )
                            _events.status(
                                "Skipping auto-repair for large file with soft-only issues.",
                                stage="checks",
                                session_id=self._session_id,
                                job_id=self.job_id,
                                rec_id=rid,
                                path=path,
                                size_bytes=size_bytes,
                            )
                        else:
                            failing_paths.append(path)

            else:
                # No per-file mapping at all: treat this as a global failure and
                # propagate a generic feedback blob to all candidate files (if you still want that behavior).
                failing_paths_raw = sorted(candidate_paths)
                global_text = self._build_validation_feedback_text(current_validation)
                per_file_feedback = {p: global_text for p in failing_paths_raw}

                failing_paths = []
                for path in failing_paths_raw:
                    if has_blocking:
                        failing_paths.append(path)
                    else:
                        size_bytes = _approx_size_for_path(path, current_proposed)
                        if size_bytes and size_bytes > LARGE_FILE_THRESHOLD and severity == "soft_only":
                            self._progress(
                                "validation_soft_skip_large_file",
                                rec_id=rid,
                                path=path,
                                size_bytes=size_bytes,
                                job_id=self.job_id,
                            )
                            _events.status(
                                "Skipping auto-repair for large file with soft-only issues.",
                                stage="checks",
                                session_id=self._session_id,
                                job_id=self.job_id,
                                rec_id=rid,
                                path=path,
                                size_bytes=size_bytes,
                            )
                        else:
                            failing_paths.append(path)

            if not failing_paths:
                # Nothing left to repair under the current severity/size policy.
                # Exit the repair loop and let the failure-handling logic decide whether
                # to treat this as advisory vs fatal.
                break

            _events.status(
                "Pre-apply checks failed; attempting targeted repair for affected files.",
                stage="checks",
                session_id=self._session_id,
                job_id=self.job_id,
                rec_id=rid,
                attempt=repair_attempt,
                attempts=validation_repair_retries,
                files=failing_paths,
            )
            self._progress(
                "validation_repair",
                rec_id=rid,
                attempt=repair_attempt,
                attempts=validation_repair_retries,
                job_id=self.job_id,
            )

            # Index proposed/edits by path for efficient updates.
            proposed_by_path: Dict[str, Dict[str, Any]] = {}
            for p in current_proposed:
                path = p.get("path")
                if isinstance(path, str) and path:
                    proposed_by_path[path] = p

            edit_idx_by_path: Dict[str, int] = {}
            for i, e in enumerate(current_edits):
                path = e.get("path")
                if isinstance(path, str) and path and path not in edit_idx_by_path:
                    edit_idx_by_path[path] = i

            max_tokens = self._phase_max_tokens("repair_edits")

            for path in failing_paths:
                p_entry = proposed_by_path.get(path)
                if not p_entry:
                    continue

                # Baseline for repair is the already-edited content that failed validation.
                current_content = p_entry.get("preview_content")
                if not isinstance(current_content, str):
                    # If caller provided overlayed content for this path, prefer that
                    # as the most up-to-date baseline instead of reading from disk.
                    if file_overlays and isinstance(file_overlays.get(path), str):
                        current_content = file_overlays[path]
                    else:
                        # Fallback: reconstruct from diff + original on-disk file.
                        baseline = _read_file_text_if_exists(self.root, path) or ""
                        diff_text = p_entry.get("diff")
                        if isinstance(diff_text, str) and diff_text.strip():
                            try:
                                current_content = apply_unified_patch(
                                    baseline.replace("\r\n", "\n").replace("\r", "\n"),
                                    diff_text,
                                )
                            except Exception as e:
                                self._progress_error(
                                    "repair_reconstruct_preview",
                                    rec_id=rid,
                                    path=path,
                                    error=str(e),
                                    attempt=repair_attempt,
                                    job_id=self.job_id,
                                )
                                current_content = baseline
                        else:
                            current_content = baseline

                feedback_for_path = per_file_feedback.get(path)
                if not feedback_for_path:
                    # Keep this strictly file-scoped: don't fall back to global multi-file diagnostics, which can confuse the repair model.
                    feedback_for_path = (
                        "Validation failed for this file, but no file-scoped diagnostics "
                        "were available. Please re-check this file carefully."
                    )

                # NOTE:
                # When invoking generate_repair_for_path we force the chat_json stage/phase
                # to 'propose_repairs' so that generate_edits.generate_edits_for_file
                # can choose repair-specific prompts/templates. This separates the
                # edit-generation vs. repair-generation paths.
                # Adapter: force the chat_json stage/phase to 'propose_repairs' so that
                # generate_repair_for_path (and downstream generate_edits_for_file) can
                # select repair-specific prompts/templates. This is intentional to route
                # validation-driven repairs through the repair flow rather than the normal
                # edit-generation flow.
                def repair_chat_json_fn(*a, **kw):
                    return chat_json_edits(*a, **{**kw, "phase": "propose_repairs", "stage": "propose_repairs"})

                repair_edit = generate_repair_for_path(
                    rec=rec,
                    rel_path=path,
                    current_content=current_content,
                    validation_feedback=feedback_for_path,
                    goal=focus,
                    chat_json_fn=repair_chat_json_fn,
                    schema=EDIT_SCHEMA,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )

                if not repair_edit:
                    self._progress_error(
                        "repair_generate_edit",
                        rec_id=rid,
                        path=path,
                        attempt=repair_attempt,
                        attempts=validation_repair_retries,
                        job_id=self.job_id,
                        error="LLM returned no usable repair edit",
                    )
                    continue

                # Ensure routing fields
                repair_edit.setdefault("path", path)
                repair_edit.setdefault("rec_id", rid)

                # REQUIRED by FileEdit v5 schema
                repair_edit.setdefault("is_new", False)

                # Compute repaired preview + diff against on-disk baseline.
                # If overlay was provided, use it as the "old" baseline for diff generation
                old_text = _read_file_text_if_exists(self.root, path) or ""
                if file_overlays and isinstance(file_overlays.get(path), str):
                    old_text = file_overlays[path]
                old_norm = old_text.replace("\r\n", "\n").replace("\r", "\n")

                preview_new: Optional[str] = None

                if isinstance(repair_edit.get("content"), str):
                    preview_new = repair_edit["content"]
                elif isinstance(repair_edit.get("patch_unified"), str):
                    try:
                        preview_new = apply_unified_patch(
                            old_norm,
                            repair_edit["patch_unified"],
                        )
                    except Exception as e:
                        self._progress_error(
                            "repair_apply_patch_preview",
                            rec_id=rid,
                            path=path,
                            error=str(e),
                            attempt=repair_attempt,
                            job_id=self.job_id,
                        )
                        preview_new = None
                elif isinstance(repair_edit.get("patch"), str):
                    try:
                        preview_new = apply_unified_patch(
                            old_norm,
                            repair_edit.get("patch"),
                        )
                        repair_edit["patch_unified"] = repair_edit.pop("patch")
                    except Exception as e:
                        self._progress_error(
                            "repair_apply_patch_preview",
                            rec_id=rid,
                            path=path,
                            error=str(e),
                            attempt=repair_attempt,
                            job_id=self.job_id,
                        )
                        preview_new = None

                if preview_new is None:
                    # If we can't build a preview, we can't confidently keep this repair.
                    continue

                diff_text = generate_unified_diff(
                    f"a/{path}",
                    f"b/{path}",
                    old_text,
                    preview_new,
                )

                # Update edits list
                if path in edit_idx_by_path:
                    current_edits[edit_idx_by_path[path]] = repair_edit
                else:
                    edit_idx_by_path[path] = len(current_edits)
                    current_edits.append(repair_edit)

                # Update proposed entry with new diff/preview
                p_entry["diff"] = diff_text
                p_entry["preview_content"] = preview_new
                p_entry["preview_bytes"] = len(preview_new.encode("utf-8"))
                if isinstance(repair_edit.get("summary"), str):
                    summary = repair_edit["summary"].strip()
                    if summary:
                        p_entry["summary"] = summary

            # Re-run validation with the updated proposed bundle
            current_validation = self.validate_proposed_edits(
                rec_id=rid,
                proposed=current_proposed,
            )
            if getattr(current_validation, "ok", False):
                return current_edits, current_proposed

        # If we reach here, all repair attempts failed (or were skipped by policy).
        self._progress_error(
            "preapply_checks",
            rec_id=rid,
            job_id=self.job_id,
            attempts=repair_attempt,
            reason="validation_failed_no_more_repairs",
        )

        mode = os.getenv("AIDEV_PREAPPLY_MODE", "advisory").strip().lower()
        allow_soft_fail = mode != "strict"

        if allow_soft_fail:
            logging.warning(
                "[preapply_checks] validation_failed_no_more_repairs for rec %s; "
                "continuing with edits but marking rec as having validation issues "
                "(AIDEV_PREAPPLY_MODE=%s)",
                rid,
                mode or "<empty>",
            )

            try:
                # Attach warnings so the UI can show them (preapply_status/preapply_errors).
                self._record_preapply_warning(
                    rec_id=rid,
                    status="validation_failed_no_more_repairs",
                    validation_obj=current_validation,
                )
            except Exception:
                logging.debug(
                    "edit_mixin: failed to record preapply warning for rec %s",
                    rid,
                    exc_info=True,
                )

            # SOFT FAILURE: return the current edits/proposed so the user can
            # still review/approve/apply them.
            return current_edits, current_proposed

        # STRICT MODE: preserve original behaviour and abort this recommendation.
        raise IOFailure(
            f"Pre-apply checks failed for recommendation {rid} after "
            f"{repair_attempt} targeted repair attempt(s); not applying edits."
        )
