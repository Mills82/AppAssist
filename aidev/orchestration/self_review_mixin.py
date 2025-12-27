# aidev/orchestration/self_review_mixin.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..io_utils import (
    _read_file_text_if_exists,
    apply_unified_patch,
    generate_unified_diff,
)
from ..schemas import file_edit_schema, self_review_schema
from ..stages.generate_edits import generate_repair_for_path


# Local schema + prompt loader for self-review so this module stays self-contained.
try:
    SELF_REVIEW_SCHEMA: Dict[str, Any] = self_review_schema()
except Exception as e:
    logging.warning(
        "self_review_mixin: Failed to load SELF_REVIEW_SCHEMA; "
        "falling back to empty schema: %s",
        e,
    )
    SELF_REVIEW_SCHEMA = {}

# Schema for follow-up repair edits (should match the edit-file output contract).
try:
    FILE_EDIT_SCHEMA: Dict[str, Any] = file_edit_schema()
except Exception as e:
    logging.warning(
        "self_review_mixin: Failed to load FILE_EDIT_SCHEMA; "
        "falling back to empty schema: %s",
        e,
    )
    FILE_EDIT_SCHEMA = {}

_SELF_REVIEW_SYSTEM_TEXT: Optional[str] = None


def _slim_repair_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    # ORDER MATTERS: use a tuple/list, not a set.
    allow_order = (
        "file_path",
        "rec_id",
        "file_language",
        "goal",
        "acceptance_criteria",
        "validation_feedback",
        "cross_file_notes",
        # optional small hints
        "rec_title",
        "rec_reasoning",
        # put the biggest field last
        "current_content",
    )

    out: Dict[str, Any] = {}
    for k in allow_order:
        if k in payload:
            out[k] = payload[k]

    # Back-compat aliases (best-effort)
    if "file_path" not in out:
        p = payload.get("path")
        if isinstance(p, str) and p:
            out["file_path"] = p

    if "current_content" not in out:
        cc = payload.get("content")
        if isinstance(cc, str) and cc:
            out["current_content"] = cc

    return out


def _load_self_review_system_text() -> Optional[str]:
    """
    Best-effort loader for the system.self_review.md prompt.

    Returns:
        The full system prompt text as a string, or None if it cannot be loaded.
    """
    global _SELF_REVIEW_SYSTEM_TEXT

    if _SELF_REVIEW_SYSTEM_TEXT is not None:
        return _SELF_REVIEW_SYSTEM_TEXT

    try:
        base = Path(__file__).resolve().parents[1]  # .../aidev
        prompt_path = base / "prompts" / "system.self_review.md"
        _SELF_REVIEW_SYSTEM_TEXT = prompt_path.read_text(encoding="utf-8")
        return _SELF_REVIEW_SYSTEM_TEXT
    except Exception as e:
        logging.warning(
            "self_review_mixin: Failed to load system.self_review.md: %s",
            e,
        )
        _SELF_REVIEW_SYSTEM_TEXT = None
        return None


def _as_nonempty_str_list(value: Any) -> List[str]:
    """Normalize a value into a list of non-empty strings."""
    if value is None:
        return []
    items: List[Any]
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    out: List[str] = []
    for it in items:
        if it is None:
            continue
        try:
            s = str(it).strip()
        except Exception:
            continue
        if s:
            out.append(s)
    return out


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _severity_rank(sev: str) -> int:
    s = (sev or "").strip().lower()
    return {"high": 3, "medium": 2, "low": 1}.get(s, 0)


class SelfReviewMixin:
    """
    Mixin that encapsulates:

      - self-review LLM calls for a single recommendation
      - automatic follow-up edits from self-review suggestions
      - cross-file context (cross_file_notes) accumulation helpers

    It assumes `self` is an Orchestrator-like object with:

      - _chat_json, _phase_max_tokens, _progress_error
      - root, job_id
    """

    # ------------------- self-review core -------------------

    def _run_self_review_for_rec(
        self,
        *,
        rec: Dict[str, Any],
        rec_id: str,
        proposed: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Run a self-review / consistency pass for a single recommendation.

        Uses system.self_review.md + SELF_REVIEW_SCHEMA to:

          - Inspect the full unified diffs for all changed files in this rec.
          - Cross-check against accumulated cross_file_notes (contracts).
          - Emit structured warnings and optional file_update_requests.

        Returns:
            A dict matching self_review.schema.json, or None on failure/skip.
        """
        if not proposed:
            return None
        if not isinstance(SELF_REVIEW_SCHEMA, dict) or not SELF_REVIEW_SCHEMA:
            return None

        system_text = _load_self_review_system_text()
        if not system_text:
            return None

        payload = self._build_self_review_payload(
            rec=rec,
            rec_id=rec_id,
            proposed=proposed,
        )
        if not payload:
            return None

        max_tokens = self._phase_max_tokens("self_review")

        raw = self._chat_json(
            system_text,
            payload,
            schema=SELF_REVIEW_SCHEMA,
            temperature=0.0,
            phase="self_review",
            inject_brief=False,
            max_tokens=max_tokens,
        )

        # Normalize (data, raw_response) -> data so callers see just the parsed payload.
        data = raw
        if isinstance(raw, tuple) and len(raw) == 2:
            data, _raw_response = raw

        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def _build_self_review_payload(
        self,
        *,
        rec: Dict[str, Any],
        rec_id: str,
        proposed: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Build the JSON payload for the self-review LLM call.

        Matches the SELF REVIEW prompt contract and deliberately avoids sending
        any "after" content or full files to prevent duplicate-context confusion.
        """
        if not proposed:
            return None

        files_payload: List[Dict[str, Any]] = []
        seen_paths: set[str] = set()

        for p in proposed:
            if not isinstance(p, dict):
                continue

            path = p.get("path")
            if not isinstance(path, str) or not path:
                continue

            if path in seen_paths:
                logging.debug(
                    "[self_review] rec_id=%s duplicate path=%s; keeping first occurrence",
                    rec_id,
                    path,
                )
                continue
            seen_paths.add(path)

            # Prefer an already-normalized unified diff field when available.
            diff_unified: Optional[str] = None
            for key in ("diff_unified", "diff", "patch_unified", "patch"):
                val = p.get(key)
                if isinstance(val, str) and val.strip():
                    diff_unified = val
                    if key not in ("diff_unified", "diff"):
                        logging.debug(
                            "[self_review] rec_id=%s path=%s using non-canonical diff key=%s",
                            rec_id,
                            path,
                            key,
                        )
                    break

            if not diff_unified:
                logging.debug(
                    "[self_review] rec_id=%s path=%s missing diff/patch; skipping file",
                    rec_id,
                    path,
                )
                continue

            # Language hint
            language = p.get("language")
            if not isinstance(language, str) or not language:
                ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
                language = {
                    "py": "python",
                    "js": "javascript",
                    "ts": "typescript",
                    "tsx": "typescriptreact",
                    "jsx": "javascriptreact",
                    "css": "css",
                    "html": "html",
                    "md": "markdown",
                    "dart": "dart",
                    "json": "json",
                    "yml": "yaml",
                    "yaml": "yaml",
                }.get(ext, "code")

            # Kind hint
            kind = p.get("kind")
            if not isinstance(kind, str) or not kind:
                if path.endswith((".html", ".htm")):
                    kind = "template"
                elif path.endswith(".css"):
                    kind = "stylesheet"
                elif path.endswith(("_test.py", "_spec.py", ".test.js", ".spec.js")):
                    kind = "test"
                elif path.endswith((".md", ".rst")):
                    kind = "docs"
                else:
                    kind = "code"

            file_obj: Dict[str, Any] = {
                "path": path,
                "language": language,
                "kind": kind,
                "diff_unified": diff_unified,
            }

            summary_before = p.get("summary_before")
            if isinstance(summary_before, str):
                sb = summary_before.strip()
                if sb:
                    file_obj["summary_before"] = sb

            files_payload.append(file_obj)

        if not files_payload:
            logging.debug(
                "[self_review] rec_id=%s has no files with diffs; skipping payload",
                rec_id,
            )
            return None

        title = str(rec.get("title") or rec.get("summary") or "").strip()
        why = str(rec.get("why") or rec.get("reasoning") or rec.get("reason") or "").strip()

        raw_criteria = rec.get("acceptance_criteria") or rec.get("rec_acceptance_criteria") or []
        if isinstance(raw_criteria, list):
            rec_acceptance_criteria = [str(c).strip() for c in raw_criteria if str(c).strip()]
        elif raw_criteria:
            c = str(raw_criteria).strip()
            rec_acceptance_criteria = [c] if c else []
        else:
            rec_acceptance_criteria = []

        # Normalize cross_file_notes to a predictable shape.
        DEFAULT_CF_KEYS = (
            "changed_interfaces",
            "new_identifiers",
            "deprecated_identifiers",
            "followup_requirements",
        )

        try:
            cf_raw = self._get_cross_file_notes_for_rec(rec_id) or {}
        except Exception:
            cf_raw = {}

        if not isinstance(cf_raw, dict):
            cf_raw = {}

        cf: Dict[str, Any] = dict(cf_raw)

        # Merge legacy rec["cross_file_notes"] if present (non-destructive).
        legacy_cf = rec.get("cross_file_notes") or {}
        if isinstance(legacy_cf, dict):
            for key, value in legacy_cf.items():
                cf.setdefault(key, value)

        # Merge any cross_file_notes attached by target-analysis.
        ta = rec.get("_target_analysis") or {}
        if isinstance(ta, dict):
            ta_cf = ta.get("cross_file_notes")
            if isinstance(ta_cf, dict):
                for key, value in ta_cf.items():
                    cf.setdefault(key, value)

        for key in DEFAULT_CF_KEYS:
            val = cf.get(key)
            if not isinstance(val, list):
                cf[key] = [] if val is None else [str(val)]

        payload: Dict[str, Any] = {
            "rec_id": rec_id,
            "rec_title": title or rec_id,
            "rec_reasoning": why,
            "rec_acceptance_criteria": rec_acceptance_criteria,
            "cross_file_notes": cf,
            "files": files_payload,
        }
        return payload

    def _maybe_run_self_review_and_auto_repairs(
        self,
        *,
        rec: Dict[str, Any],
        rid: str,
        focus: str,
        edits: List[Dict[str, Any]],
        proposed: List[Dict[str, Any]],
        max_auto_passes: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Run self-review (optionally multiple passes) and automatically apply any
        file_update_requests as small follow-up edits, without user approval.

        Returns:
            (updated_edits, updated_proposed, last_self_review_dict_or_None)
        """
        if not edits or not proposed:
            return edits, proposed, None

        passes = 0
        last_review: Optional[Dict[str, Any]] = None

        while passes < max_auto_passes:
            passes += 1

            review = self._run_self_review_for_rec(
                rec=rec,
                rec_id=rid,
                proposed=proposed,
            )

            if not isinstance(review, dict):
                break

            last_review = review

            # NEW: persist decisions + apply cross_file_notes_delta (additive).
            self._record_self_review_decisions(rid, review.get("decisions"))
            self._apply_self_review_cross_file_notes_delta(rid, review.get("cross_file_notes_delta"))

            overall = str(review.get("overall_status") or "").strip().lower()

            file_requests = review.get("file_update_requests") or []
            if not isinstance(file_requests, list):
                logging.debug(
                    "[self_review] rec_id=%s file_update_requests not a list (type=%s); ignoring",
                    rid,
                    type(file_requests).__name__,
                )
                file_requests = []

            warnings = review.get("warnings") or []
            if not isinstance(warnings, list):
                warnings = []

            logging.info(
                "[self_review] rec_id=%s pass=%d overall_status=%s file_update_requests=%d warnings=%d",
                rid,
                passes,
                overall or "<none>",
                len(file_requests),
                len(warnings),
            )

            if not file_requests:
                break

            # Safer gating:
            # - If model follows the prompt, required followups imply needs_followups.
            # - If older models omit/garble overall_status, still proceed.
            # - If overall_status says "warnings" but requests exist, only proceed if any request is high severity.
            proceed = False
            if not overall:
                proceed = True
            elif overall in ("needs_followups", "needs_followup", "needs_edits"):
                proceed = True
            elif overall in ("warnings",):
                max_sev = 0
                for r in file_requests:
                    if isinstance(r, dict):
                        max_sev = max(max_sev, _severity_rank(str(r.get("severity") or "")))
                proceed = max_sev >= _severity_rank("high")
            else:
                proceed = False

            if not proceed:
                break

            logging.info(
                "[self_review] rec_id=%s pass=%d applying follow-ups for paths=%s",
                rid,
                passes,
                [
                    str((req or {}).get("path") or "").strip()
                    for req in file_requests
                    if isinstance(req, dict)
                ],
            )

            edits, proposed = self._apply_self_review_followups(
                rec=rec,
                rid=rid,
                focus=focus,
                edits=edits,
                proposed=proposed,
                file_requests=file_requests,
                warnings=warnings,
            )

            if max_auto_passes <= passes:
                break

        return edits, proposed, last_review

    def _apply_self_review_followups(
        self,
        *,
        rec: Dict[str, Any],
        rid: str,
        focus: str,
        edits: List[Dict[str, Any]],
        proposed: List[Dict[str, Any]],
        file_requests: List[Dict[str, Any]],
        warnings: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Given a self-review result's file_update_requests, generate small follow-up
        edits using generate_repair_for_path and merge them into `edits`/`proposed`.

        This is fully automatic and does NOT require a separate user approval.
        """
        if not file_requests:
            return edits, proposed

        warnings = warnings if isinstance(warnings, list) else []

        logging.info(
            "[self_review] applying %d follow-up file_update_requests for rec_id=%s",
            len(file_requests),
            rid,
        )

        # Index current proposed entries and edits by path for easy updates.
        proposed_by_path: Dict[str, Dict[str, Any]] = {}
        for p in proposed:
            if not isinstance(p, dict):
                continue
            path = p.get("path")
            if isinstance(path, str) and path:
                proposed_by_path[path] = p

        edit_idx_by_path: Dict[str, int] = {}
        for i, e in enumerate(edits):
            if not isinstance(e, dict):
                continue
            path = e.get("path")
            if isinstance(path, str) and path and path not in edit_idx_by_path:
                edit_idx_by_path[path] = i

        def _chat_json_followup(
            system_text: str,
            user_payload: Any,
            schema: Dict[str, Any],
            temperature: float = 0.0,
            phase: Optional[str] = None,
            max_tokens: Optional[int] = None,
            stage: Optional[str] = None,
            **_kwargs: Any,
        ):
            """
            Minimal chat_json wrapper for repair/follow-up edits.

            generate_repair_for_path expects the same interface we use for
            edit-generation: (data, raw_response) from self._chat_json.
            """
            effective_phase = (stage or phase or "self_review_followup")
            mt = (
                max_tokens
                if max_tokens is not None
                else self._phase_max_tokens(effective_phase)
            )

            payload = user_payload
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("rec_id", rid)
                title = str(rec.get("title") or rec.get("summary") or "").strip()
                if title:
                    payload.setdefault("rec_title", title)
                why = str(
                    rec.get("why")
                    or rec.get("reason")
                    or rec.get("reasoning")
                    or ""
                ).strip()
                if why:
                    payload.setdefault("rec_reasoning", why)

                # Ensure acceptance_criteria is present when available (high signal).
                if "acceptance_criteria" not in payload:
                    ac = (
                        rec.get("acceptance_criteria")
                        or rec.get("rec_acceptance_criteria")
                        or []
                    )
                    if isinstance(ac, list) and ac:
                        payload["acceptance_criteria"] = ac

                # Ensure cross_file_notes is present (small, but helps consistency).
                if "cross_file_notes" not in payload:
                    try:
                        payload["cross_file_notes"] = (
                            self._get_cross_file_notes_for_rec(rid) or {}
                        )
                    except Exception:
                        payload["cross_file_notes"] = {}

                # Hard token diet: drop everything except the allowlist.
                payload = _slim_repair_payload(payload)

            return self._chat_json(
                system_text,
                payload,
                schema=schema,
                temperature=temperature,
                phase=effective_phase,
                inject_brief=False,
                max_tokens=mt,
            )

        max_tokens = self._phase_max_tokens("repair_edits")

        def _group_file_requests(reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            Collapse multiple self-review requests targeting the same path into a single
            merged request so we do 1 repair LLM call per file per pass.
            """
            grouped: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []

            for r in reqs:
                if not isinstance(r, dict):
                    continue
                p = str(r.get("path") or "").strip()
                if not p:
                    continue

                if p not in grouped:
                    grouped[p] = {
                        "path": p,
                        "severity": "",
                        "reasons": [],
                        "instructions": [],
                        "related_identifiers": [],
                    }
                    order.append(p)

                g = grouped[p]

                sev = str(r.get("severity") or "").strip()
                if _severity_rank(sev) > _severity_rank(str(g.get("severity") or "")):
                    g["severity"] = sev

                reason = str(r.get("reason") or "").strip()
                if reason:
                    g["reasons"].append(reason)

                instr = r.get("instructions")
                if isinstance(instr, list):
                    for item in instr:
                        s = str(item).strip()
                        if s:
                            g["instructions"].append(s)
                elif isinstance(instr, str):
                    s = instr.strip()
                    if s:
                        g["instructions"].append(s)

                rel = r.get("related_identifiers")
                if isinstance(rel, list):
                    for item in rel:
                        s = str(item).strip()
                        if s:
                            g["related_identifiers"].append(s)

            merged: List[Dict[str, Any]] = []
            for p in order:
                g = grouped[p]
                g["reasons"] = _dedupe_keep_order(_as_nonempty_str_list(g.get("reasons")))
                g["instructions"] = _dedupe_keep_order(_as_nonempty_str_list(g.get("instructions")))
                g["related_identifiers"] = _dedupe_keep_order(_as_nonempty_str_list(g.get("related_identifiers")))
                merged.append(g)

            return merged

        warnings_by_path: Dict[str, List[Dict[str, Any]]] = {}
        for w in warnings:
            if not isinstance(w, dict):
                continue
            files_involved = w.get("files_involved")
            if not isinstance(files_involved, list):
                continue
            for fp in files_involved:
                p = str(fp).strip()
                if not p:
                    continue
                warnings_by_path.setdefault(p, []).append(w)

        grouped_requests = _group_file_requests(file_requests)

        for req in grouped_requests:
            path = str(req.get("path") or "").strip()
            if not path:
                continue

            # For safety, only touch files already in the proposed bundle.
            p_entry = proposed_by_path.get(path)
            if not p_entry:
                logging.info(
                    "[self_review] followup rec_id=%s path=%s skipped (not in proposed bundle)",
                    rid,
                    path,
                )
                continue

            # Use the already-edited preview as the baseline for the follow-up.
            current_content = p_entry.get("preview_content")
            if not isinstance(current_content, str) or not current_content:
                baseline = _read_file_text_if_exists(self.root, path) or ""
                diff_text = (
                    p_entry.get("diff_unified")
                    or p_entry.get("diff")
                    or p_entry.get("patch_unified")
                    or p_entry.get("patch")
                    or ""
                )
                if isinstance(diff_text, str) and diff_text.strip():
                    try:
                        current_content = apply_unified_patch(
                            baseline.replace("\r\n", "\n").replace("\r", "\n"),
                            diff_text,
                        )
                    except Exception:
                        current_content = baseline
                else:
                    current_content = baseline

            severity_text = str(req.get("severity") or "").strip()
            reasons: List[str] = list(req.get("reasons") or [])
            instructions_list: List[str] = list(req.get("instructions") or [])
            related_ids: List[str] = list(req.get("related_identifiers") or [])

            # Include decisions (if any) so repairs stay consistent with the self-review contract.
            decisions_text = self._format_self_review_decisions_for_feedback(rid)

            # Include any relevant warnings + evidence anchors for this path (helps the repair model be deterministic).
            warn_blocks: List[str] = []
            for w in warnings_by_path.get(path, []):
                msg = str(w.get("message") or "").strip()
                kind = str(w.get("kind") or "").strip()
                sev = str(w.get("severity") or "").strip()
                header = f"[warning kind={kind} severity={sev}] " if (kind or sev) else "[warning] "
                if msg:
                    warn_blocks.append(header + msg)
                evidence = w.get("evidence")
                if isinstance(evidence, list):
                    for ev in evidence[:4]:
                        if not isinstance(ev, dict):
                            continue
                        if str(ev.get("path") or "").strip() != path:
                            continue
                        anchor = str(ev.get("diff_anchor") or "").strip()
                        why = str(ev.get("why_it_matters") or "").strip()
                        if anchor:
                            warn_blocks.append(f"- diff_anchor: {anchor}")
                        if why:
                            warn_blocks.append(f"- why_it_matters: {why}")

            header = f"[severity={severity_text}] " if severity_text else ""
            parts: List[str] = []

            if decisions_text:
                parts.append("Decisions to follow (do not contradict):")
                parts.append(decisions_text)

            if reasons:
                parts.append("Reasons:")
                for r in reasons:
                    parts.append(f"- {r}")

            if warn_blocks:
                parts.append("Related self-review warnings/evidence:")
                parts.extend(warn_blocks)

            if related_ids:
                parts.append("Related identifiers:")
                parts.append(", ".join(related_ids))

            if instructions_list:
                parts.append("Required changes (treat as acceptance criteria for this repair):")
                for idx2, inst in enumerate(instructions_list, 1):
                    parts.append(f"{idx2}. {inst}")

            if not parts:
                parts.append("Please apply the requested follow-up fix for this file.")

            feedback = header + "\n".join(parts)

            repair_edit = generate_repair_for_path(
                rec=rec,
                rel_path=path,
                current_content=current_content,
                validation_feedback=feedback,
                goal=focus,
                chat_json_fn=_chat_json_followup,
                schema=FILE_EDIT_SCHEMA,
                temperature=0.0,
                max_tokens=max_tokens,
            )

            if not repair_edit or not isinstance(repair_edit, dict):
                logging.warning(
                    "[self_review] followup rec_id=%s path=%s: LLM returned no usable edit",
                    rid,
                    path,
                )
                self._progress_error(
                    "self_review_followup_generate",
                    rec_id=rid,
                    path=path,
                    job_id=self.job_id,
                    error="LLM returned no usable follow-up edit",
                )
                continue

            # Ensure routing fields + schema-required is_new.
            repair_edit.setdefault("path", path)
            repair_edit.setdefault("rec_id", rid)
            repair_edit.setdefault("is_new", False)

            # Drop legacy alias fields if present (helps strict schemas / downstream consistency).
            repair_edit.pop("file_path", None)

            # Content-only contract: the repair prompt requires full `content`.
            old_text = _read_file_text_if_exists(self.root, path) or ""

            preview_new: Optional[str] = None
            content = repair_edit.get("content")
            if isinstance(content, str) and content.strip():
                preview_new = content

            # Defensive one-shot retry if model violates contract (should be rare).
            if preview_new is None:
                logging.warning(
                    "[self_review] followup rec_id=%s path=%s: missing/empty content; retrying once with explicit requirement",
                    rid,
                    path,
                )
                feedback_retry = (
                    feedback
                    + "\n\nIMPORTANT: You MUST return the FULL updated file in a non-empty 'content' field. "
                    + "Do NOT return patches or diffs."
                )
                repair_edit_retry = generate_repair_for_path(
                    rec=rec,
                    rel_path=path,
                    current_content=current_content,
                    validation_feedback=feedback_retry,
                    goal=focus,
                    chat_json_fn=_chat_json_followup,
                    schema=FILE_EDIT_SCHEMA,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                if isinstance(repair_edit_retry, dict):
                    c2 = repair_edit_retry.get("content")
                    if isinstance(c2, str) and c2.strip():
                        repair_edit = repair_edit_retry
                        repair_edit.setdefault("path", path)
                        repair_edit.setdefault("rec_id", rid)
                        repair_edit.setdefault("is_new", False)
                        repair_edit.pop("file_path", None)
                        preview_new = c2

            if preview_new is None:
                logging.warning(
                    "[self_review] followup rec_id=%s path=%s: could not build preview; skipping",
                    rid,
                    path,
                )
                self._progress_error(
                    "self_review_followup_preview_failed",
                    rec_id=rid,
                    path=path,
                    job_id=self.job_id,
                    error="repair output did not include non-empty 'content'",
                )
                continue

            # Enforce content-only downstream: remove any patch fields so nothing tries patch-first
            # against a mismatched baseline later.
            repair_edit["content"] = preview_new
            repair_edit.pop("patch_unified", None)
            repair_edit.pop("patch", None)
            repair_edit.pop("patch_preferred", None)

            diff_text = generate_unified_diff(
                f"a/{path}",
                f"b/{path}",
                old_text,
                preview_new,
            )

            # Update edits list
            if path in edit_idx_by_path:
                edits[edit_idx_by_path[path]] = repair_edit
            else:
                edit_idx_by_path[path] = len(edits)
                edits.append(repair_edit)

            # Update proposed entry with new diff/preview
            p_entry["diff"] = diff_text
            p_entry["diff_unified"] = diff_text  # canonical name for downstream stages
            p_entry["preview_content"] = preview_new
            p_entry["preview_bytes"] = len(preview_new.encode("utf-8"))

            if isinstance(repair_edit.get("summary"), str):
                summary = repair_edit["summary"].strip()
                if summary:
                    p_entry["summary"] = summary

        return edits, proposed

    # ------------------- decisions + cross_file_notes_delta helpers -------------------

    def _record_self_review_decisions(self, rec_id: str, decisions: Any) -> None:
        """
        Persist self-review decisions (if any) so follow-up repairs can stay aligned.
        Stored as a list of {key,value,rationale} dicts.
        """
        if not isinstance(decisions, list):
            return

        cleaned: List[Dict[str, str]] = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            k = str(d.get("key") or "").strip()
            v = str(d.get("value") or "").strip()
            r = str(d.get("rationale") or "").strip()
            if not k or not v:
                continue
            cleaned.append({"key": k, "value": v, "rationale": r})

        if not cleaned:
            return

        if not hasattr(self, "_self_review_decisions_by_rec"):
            self._self_review_decisions_by_rec: Dict[str, List[Dict[str, str]]] = {}
        self._self_review_decisions_by_rec[rec_id] = cleaned

    def _format_self_review_decisions_for_feedback(self, rec_id: str) -> str:
        """
        Render stored decisions into a short, stable text block for validation_feedback.
        """
        if not hasattr(self, "_self_review_decisions_by_rec"):
            return ""
        ds = self._self_review_decisions_by_rec.get(rec_id)
        if not isinstance(ds, list) or not ds:
            return ""
        lines: List[str] = []
        for d in ds:
            if not isinstance(d, dict):
                continue
            k = str(d.get("key") or "").strip()
            v = str(d.get("value") or "").strip()
            r = str(d.get("rationale") or "").strip()
            if not k or not v:
                continue
            if r:
                lines.append(f"- {k} = {v}  # {r}")
            else:
                lines.append(f"- {k} = {v}")
        return "\n".join(lines).strip()

    def _apply_self_review_cross_file_notes_delta(self, rec_id: str, delta: Any) -> None:
        """
        Apply an additive cross_file_notes_delta from self-review into the per-rec bucket.

        Expected delta shape (additive-only):
          {
            "changed_interfaces_add": [...],
            "new_identifiers_add": [...],
            "deprecated_identifiers_add": [...],
            "followup_requirements_add": [...]
          }
        """
        if not isinstance(delta, dict):
            return

        self._init_cross_file_notes_for_rec(rec_id)
        ctx = self._cross_file_notes_by_rec[rec_id]

        mapping = {
            "changed_interfaces_add": "changed_interfaces",
            "new_identifiers_add": "new_identifiers",
            "deprecated_identifiers_add": "deprecated_identifiers",
            "followup_requirements_add": "followup_requirements",
        }

        for src_key, dst_key in mapping.items():
            items = _as_nonempty_str_list(delta.get(src_key))
            if not items:
                continue
            bucket = ctx.setdefault(dst_key, [])
            for s in items:
                if s not in bucket:
                    bucket.append(s)

    # ------------------- cross-file context helpers -------------------

    def _init_cross_file_notes_for_rec(self, rec_id: str) -> None:
        """
        Ensure there is a cross-file context bucket for this recommendation.

        Canonical internal shape (all keys optional):

            {
                "changed_interfaces": [...],
                "new_identifiers": [...],
                "deprecated_identifiers": [...],
                "followup_requirements": [...]
            }

        Values are lists that are monotonically appended to as individual
        file edits contribute cross_file_notes.
        """
        if not hasattr(self, "_cross_file_notes_by_rec"):
            # { rec_id: { "changed_interfaces": [...], "new_identifiers": [...], ... } }
            self._cross_file_notes_by_rec: Dict[str, Dict[str, List[str]]] = {}

        if rec_id not in self._cross_file_notes_by_rec:
            self._cross_file_notes_by_rec[rec_id] = {
                "changed_interfaces": [],
                "new_identifiers": [],
                "deprecated_identifiers": [],
                "followup_requirements": [],
            }

    def _get_cross_file_notes_for_rec(self, rec_id: str) -> Dict[str, List[str]]:
        """
        Return the current aggregated cross-file context for the given rec_id.

        Callers should treat the returned object as read-only.
        """
        if not hasattr(self, "_cross_file_notes_by_rec"):
            return {}
        ctx = self._cross_file_notes_by_rec.get(rec_id) or {}
        return ctx

    def _accumulate_cross_file_notes(
        self,
        *,
        rec_id: str,
        notes: Dict[str, Any],
    ) -> None:
        """
        Merge a cross_file_notes payload from a single edit-file or analysis
        response into the per-recommendation context.

        Canonical internal shape (all keys optional):

            {
                "changed_interfaces": [...],
                "new_identifiers": [...],
                "deprecated_identifiers": [...],
                "followup_requirements": [...]
            }

        Values may be lists, single strings/objects, or None. They are normalised
        into lists of non-empty strings and appended with simple de-duplication.
        """
        if not isinstance(notes, dict):
            return

        self._init_cross_file_notes_for_rec(rec_id)
        ctx = self._cross_file_notes_by_rec[rec_id]

        # 1) New canonical keys (preferred).
        canonical_keys = (
            "changed_interfaces",
            "new_identifiers",
            "deprecated_identifiers",
            "followup_requirements",
        )
        for key in canonical_keys:
            if key in notes:
                items = _as_nonempty_str_list(notes.get(key))
                if not items:
                    continue
                bucket = ctx.setdefault(key, [])
                for s in items:
                    if s not in bucket:
                        bucket.append(s)

        # 2) Legacy aliases -> canonical keys.
        legacy_map = {
            "interface_changes": "changed_interfaces",
            "breaking_changes": "changed_interfaces",
            "new_symbols": "new_identifiers",
            "removed_symbols": "deprecated_identifiers",
            "followups": "followup_requirements",
        }
        for legacy_key, canonical in legacy_map.items():
            if legacy_key in notes:
                items = _as_nonempty_str_list(notes.get(legacy_key))
                if not items:
                    continue
                bucket = ctx.setdefault(canonical, [])
                for s in items:
                    if s not in bucket:
                        bucket.append(s)
