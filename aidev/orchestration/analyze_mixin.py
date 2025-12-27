# aidev/orchestration/analyze_mixin.py
from __future__ import annotations

import hashlib
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..cards import KnowledgeBase
from .. import events as _events

from .analyze_prompts import (
    analyze_system_prompt,
    build_analyze_user_payload,
)

from ..llm_utils import parse_analyze_plan_text, ParseError

try:
    from ..schemas import analyze_plan_schema
except Exception:  # pragma: no cover - defensive
    analyze_plan_schema = None  # type: ignore[assignment]

# Best-effort load of the Analyze Plan JSON schema.
try:
    ANALYZE_PLAN_SCHEMA: Dict[str, Any] = (
        analyze_plan_schema() if callable(analyze_plan_schema) else {}
    )
except Exception as e:  # pragma: no cover - defensive
    logging.warning(
        "Failed to load ANALYZE_PLAN_SCHEMA; falling back to loose object schema: %s",
        e,
    )
    ANALYZE_PLAN_SCHEMA = {"type": "object"}


class OrchestratorAnalyzeMixin:
    """
    Mixin providing the 'analyze' mode pipeline.

    This mode:
    - Performs a multi-file analysis of the project.
    - Uses existing Knowledge Cards and project metadata.
    - Calls the LLM to produce a structured Analyze Plan object
        (themes + suggested improvements).
    - Emits both the structured plan and a human-readable analysis summary.
    - Does NOT generate or apply code edits / patches.

    It expects the concrete Orchestrator class to provide:
      - self.args (dict-like)
      - self._should_cancel() -> bool
      - self._progress(event: str, **payload)
      - self._progress_error(where: str, **payload)
      - self._chat_json(...)
      - self._chat(...)
      - self._phase_max_tokens(phase: str) -> Optional[int]
      - self._to_jsonable(obj: Any) -> Any
      - self._project_brief_text
      - self._session_id
      - self.job_id
      - self.st.trace
      - self._job_update(...)
    """

    def _run_analyze_mode(
        self,
        *,
        kb: KnowledgeBase,
        meta: Dict[str, Any],
        ctx_blob: Any,
    ) -> Tuple[bool, str]:
        """
        Analysis mode: perform a deeper multi-file analysis and produce
        a structured Analyze Plan (themes + suggestions), without generating
        code edits or patches.

        Outputs:
        - Structured Analyze Plan (JSON) sent via SSE.
        - Human-readable analysis string for chat/legacy UIs.
        """
        if self._should_cancel():
            return False, "Run cancelled before analysis."

        # Focus is the "question" or high-level topic we want to analyze.
        raw_focus = (
            self.args.get("focus")
            or self.args.get("message")
            or self.args.get("question")
            or ""
        )
        focus = str(raw_focus).strip()
        if not focus:
            focus = "General project analysis and improvement opportunities"

        cfg = self.args.get("cfg") or {}
        if not isinstance(cfg, dict):
            cfg = {}
        cards_cfg = cfg.get("cards", {}) if isinstance(cfg.get("cards"), dict) else {}
        try:
            top_k = int(
                self.args.get("cards_top_k")
                or cards_cfg.get("default_top_k", 30)
            )
        except Exception:
            top_k = 30

        # Select the most relevant cards for this focus.
        try:
            hits = kb.select_cards(focus, top_k=top_k)
        except Exception as e:
            hits = []
            self._progress_error(
                "analyze_mode_select_cards",
                error=str(e),
                trace=traceback.format_exc(),
            )

        # Build LLM-facing card payloads from the v2 card structure.
        cards_payload: List[Dict[str, Any]] = []
        for hit in hits or []:
            try:
                # v2: hits are (rel_path, score). Be defensive in case older shapes exist.
                if not isinstance(hit, (tuple, list)) or len(hit) != 2:
                    continue
                rel_path, score = hit

                # Older implementations might return a card-like object as first element.
                if hasattr(rel_path, "path") and not isinstance(rel_path, str):
                    rel = getattr(rel_path, "path", None) or getattr(rel_path, "rel", None)
                    rel = str(rel) if rel is not None else ""
                else:
                    rel = str(rel_path or "")

                if not rel:
                    continue

                try:
                    card_meta = kb.get_card(rel)
                except Exception:
                    card_meta = None
                if not isinstance(card_meta, dict):
                    card_meta = {}

                # Title: prefer explicit title, else filename, else path.
                title = card_meta.get("title")
                if not isinstance(title, str) or not title.strip():
                    # fall back to filename; if that fails, use the rel path
                    filename = Path(rel).name
                    title = filename or rel

                # Summary: prefer summary_text if present, else summary.ai_text, then heuristic.
                summary_text = ""
                raw_summary_text = card_meta.get("summary_text")
                if isinstance(raw_summary_text, str) and raw_summary_text.strip():
                    summary_text = raw_summary_text.strip()
                else:
                    s = card_meta.get("summary")
                    if isinstance(s, dict):
                        ai = s.get("ai_text")
                        heur = s.get("heuristic")
                        if isinstance(ai, str) and ai.strip():
                            summary_text = ai.strip()
                        elif isinstance(heur, str) and heur.strip():
                            summary_text = heur.strip()
                    elif isinstance(s, str) and s.strip():
                        # Legacy/raw summary string
                        summary_text = s.strip()

                language = card_meta.get("language") or card_meta.get("lang") or None

                cards_payload.append(
                    {
                        "path": rel,
                        "title": title,
                        "summary": summary_text,
                        "language": language,
                        "score": float(score),
                    }
                )
            except Exception:
                # Best-effort; skip malformed entries without failing the whole run.
                continue

        # Load system prompt from /prompts via system_preset("analyze"), with fallback.
        system_text = analyze_system_prompt()

        # --------------------------------------------------------------
        # Optional deep research (GATED) - must run BEFORE analyze LLM call
        # --------------------------------------------------------------
        # Acceptance constraints:
        # - When disabled, emit no deep research events and do not change behavior.
        # - When enabled, emit deep_research.start -> deep_research.done before LLM call.
        # - On cache hit, emit deep_research.cache_hit (and skip start/done).
        # - Attach only a small digest (counts + truncated + cache_key) to LLM payload.
        # - Never include raw file contents / evidence bodies in events/payload.

        deep_research_enabled = False
        try:
            research_cfg = (
                cfg.get("deep_research")
                if isinstance(cfg.get("deep_research"), dict)
                else {}
            )
            # Accept a few common knobs; default must remain False.
            deep_research_enabled = bool(
                self.args.get("deep_research")
                or self.args.get("deep_research_enabled")
                or self.args.get("enable_deep_research")
                or research_cfg.get("enabled")
            )
        except Exception:
            deep_research_enabled = False

        research_depth = (
            self.args.get("deep_research_depth")
            or self.args.get("research_depth")
            or (research_cfg.get("depth") if isinstance(research_cfg, dict) else None)
            or "standard"
        )

        research_brief: Any = None
        research_digest: Optional[Dict[str, Any]] = None
        # Only used for tracing/debug; not included in LLM payload.
        _dr_cache_key: Optional[str] = None

        if deep_research_enabled:
            try:
                # Lazy imports: if missing/unavailable, analysis must proceed unchanged.
                dr_cache = None
                run_deep_research = None
                try:
                    from . import deep_research_cache as dr_cache  # type: ignore
                except Exception:
                    dr_cache = None
                try:
                    from .deep_research_engine import run_deep_research as run_deep_research
                except Exception:
                    run_deep_research = None

                if callable(run_deep_research):
                    # Compute deterministic cache key only via repo-local cache API.
                    cache_key: Optional[str] = None
                    cached: Any = None
                    try:
                        if dr_cache is not None:
                            fn_key = getattr(dr_cache, "compute_cache_key", None)
                            if callable(fn_key):
                                cache_key = fn_key(
                                    focus=focus,
                                    cards=cards_payload,
                                    depth=research_depth,
                                )
                            else:
                                # Do NOT use a local ad-hoc hashing fallback here; prefer the
                                # cache module's compute_cache_key so key generation is
                                # deterministic across orchestrator paths.
                                cache_key = None
                    except Exception:
                        cache_key = None

                    _dr_cache_key = cache_key

                    # Small helper to prefer first-class deep research event helpers
                    # in aidev.events, falling back to emit/status if they are missing.
                    def _emit_deep_research_event(kind: str, **kwargs: Any) -> None:
                        try:
                            # Map logical kind to expected helper names.
                            helper_map = {
                                "start": "deep_research_start",
                                "done": "deep_research_done",
                                "cache_hit": "deep_research_cache_hit",
                                "attached_to_payload": "deep_research_attached_to_payload",
                            }
                            helper_name = helper_map.get(kind)
                            if helper_name:
                                fn = getattr(_events, helper_name, None)
                                if callable(fn):
                                    try:
                                        fn(**kwargs)
                                        return
                                    except TypeError:
                                        try:
                                            fn(kwargs)
                                            return
                                        except Exception:
                                            # fallthrough to generic emit
                                            pass
                                # If specific helper not present or failed, try generic emit
                            fn_emit = getattr(_events, "emit", None)
                            if callable(fn_emit):
                                # event name like 'deep_research.start'
                                fn_emit(f"deep_research.{kind}", **kwargs)
                                return
                            # Final fallback: status-based emit with 'where' set
                            fn_status = getattr(_events, "status", None)
                            if callable(fn_status):
                                fn_status(f"deep_research.{kind}", where=f"deep_research.{kind}", **kwargs)
                        except Exception:
                            # Events must never break analysis.
                            return

                    def _safe_counts_from_brief(brief_obj: Any) -> Tuple[int, int, bool]:
                        """Extract (evidence_items, findings, truncated) counts from brief/res dict."""
                        evidence_items = 0
                        findings = 0
                        truncated = False
                        try:
                            if isinstance(brief_obj, dict):
                                ev = (
                                    brief_obj.get("evidence_items")
                                    or brief_obj.get("evidence")
                                    or brief_obj.get("sources")
                                    or []
                                )
                                fi = brief_obj.get("findings") or []
                                if isinstance(ev, list):
                                    evidence_items = len(ev)
                                elif ev is not None:
                                    evidence_items = 1
                                if isinstance(fi, list):
                                    findings = len(fi)
                                elif fi is not None:
                                    findings = 1
                                truncated = bool(
                                    brief_obj.get("truncated")
                                    or brief_obj.get("is_truncated")
                                    or brief_obj.get("truncation")
                                )
                        except Exception:
                            return 0, 0, False
                        return evidence_items, findings, truncated

                    if cache_key is not None:
                        try:
                            fn_get = (
                                getattr(dr_cache, "get", None)
                                or getattr(dr_cache, "read", None)
                                or getattr(dr_cache, "load", None)
                            )
                            if callable(fn_get):
                                try:
                                    cached = fn_get(cache_key)
                                except TypeError:
                                    cached = fn_get(key=cache_key)  # type: ignore[call-arg]
                        except Exception:
                            cached = None

                    if cached is not None:
                        # Treat cached as either the brief itself or a wrapper.
                        brief_obj = None
                        if isinstance(cached, dict) and (
                            "brief" in cached or "research_brief" in cached
                        ):
                            brief_obj = cached.get("brief") or cached.get("research_brief")
                        else:
                            brief_obj = cached

                        evidence_items, findings, truncated = _safe_counts_from_brief(brief_obj)
                        research_brief = self._to_jsonable(brief_obj)
                        research_digest = {
                            "cache_key": cache_key,
                            "evidence_items": evidence_items,
                            "findings": findings,
                            "truncated": truncated,
                        }

                        # Emit cache-hit event (no raw contents) via canonical helper if available.
                        try:
                            _emit_deep_research_event(
                                "cache_hit",
                                mode="analyze",
                                cache_key=cache_key,
                                focus=focus,
                                evidence_items=evidence_items,
                                findings=findings,
                                truncated=truncated,
                            )
                        except Exception:
                            pass
                    else:
                        # Cache miss: emit start -> run -> done.
                        try:
                            _emit_deep_research_event(
                                "start",
                                mode="analyze",
                                cache_key=cache_key,
                                focus=focus,
                            )
                        except Exception:
                            pass

                        research_res: Any = None
                        try:
                            research_res = run_deep_research(
                                focus=focus,
                                cards=cards_payload,
                                depth=research_depth,
                            )
                        except Exception as e:
                            self._progress_error(
                                "analyze_mode_deep_research_run",
                                error=str(e),
                                trace=traceback.format_exc(),
                            )
                            research_res = None

                        ok_dr = False
                        if isinstance(research_res, dict):
                            ok_dr = bool(research_res.get("ok"))
                        else:
                            ok_dr = bool(research_res)

                        brief_obj = None
                        if ok_dr and isinstance(research_res, dict):
                            brief_obj = (
                                research_res.get("brief")
                                or research_res.get("research_brief")
                                or research_res.get("result")
                            )

                        evidence_items, findings, truncated = _safe_counts_from_brief(
                            brief_obj if brief_obj is not None else research_res
                        )

                        # Emit done event with only aggregate counts + truncated.
                        try:
                            _emit_deep_research_event(
                                "done",
                                mode="analyze",
                                cache_key=cache_key,
                                focus=focus,
                                ok=ok_dr,
                                evidence_items=evidence_items,
                                findings=findings,
                                truncated=truncated,
                            )
                        except Exception:
                            pass

                        # If ok, persist to cache (best-effort) and set digest/brief.
                        if ok_dr and brief_obj is not None:
                            research_brief = self._to_jsonable(brief_obj)
                            research_digest = {
                                "cache_key": cache_key,
                                "evidence_items": evidence_items,
                                "findings": findings,
                                "truncated": truncated,
                            }
                            try:
                                if dr_cache is not None and cache_key:
                                    fn_set = (
                                        getattr(dr_cache, "set", None)
                                        or getattr(dr_cache, "write", None)
                                        or getattr(dr_cache, "save", None)
                                    )
                                    if callable(fn_set):
                                        try:
                                            fn_set(cache_key, research_brief)
                                        except TypeError:
                                            fn_set(key=cache_key, value=research_brief)  # type: ignore[call-arg]
                            except Exception:
                                # Cache writes must never break analysis.
                                pass
            except Exception as e:
                # Fully fail-safe. Also: if this fails, do not emit partial extra events here.
                self._progress_error(
                    "analyze_mode_deep_research",
                    error=str(e),
                    trace=traceback.format_exc(),
                )
                research_brief = None
                research_digest = None
                _dr_cache_key = None

        # Shape and bound the user payload (similar to Q&A mode).
        payload = build_analyze_user_payload(
            analysis_focus=focus,
            project_brief=self._project_brief_text or "",
            project_meta=self._to_jsonable(meta),
            structure_overview=self._to_jsonable(ctx_blob),
            top_cards=cards_payload,
        )

        # If research is enabled and we produced a digest, attach it to the payload
        # before the analyze LLM call and emit an attachment event.
        if deep_research_enabled and isinstance(research_digest, dict):
            try:
                if isinstance(payload, dict):
                    payload = dict(payload)
                    payload["deep_research_digest"] = dict(research_digest)

                # Emit attached_to_payload event (counts + truncated + cache_key only)
                # Prefer first-class helper when available.
                try:
                    helper = getattr(_events, "deep_research_attached_to_payload", None)
                    if callable(helper):
                        try:
                            helper(
                                mode="analyze",
                                cache_key=research_digest.get("cache_key"),
                                evidence_items=research_digest.get("evidence_items"),
                                findings=research_digest.get("findings"),
                                truncated=research_digest.get("truncated"),
                                attached=True,
                            )
                        except TypeError:
                            try:
                                helper({
                                    "mode": "analyze",
                                    "cache_key": research_digest.get("cache_key"),
                                    "evidence_items": research_digest.get("evidence_items"),
                                    "findings": research_digest.get("findings"),
                                    "truncated": research_digest.get("truncated"),
                                    "attached": True,
                                })
                            except Exception:
                                # Fall through to generic emit
                                getattr(_events, "emit", lambda *a, **k: None)(
                                    "deep_research.attached_to_payload",
                                    mode="analyze",
                                    cache_key=research_digest.get("cache_key"),
                                    evidence_items=research_digest.get("evidence_items"),
                                    findings=research_digest.get("findings"),
                                    truncated=research_digest.get("truncated"),
                                    attached=True,
                                )
                    else:
                        fn_emit = getattr(_events, "emit", None)
                        if callable(fn_emit):
                            fn_emit(
                                "deep_research.attached_to_payload",
                                mode="analyze",
                                cache_key=research_digest.get("cache_key"),
                                evidence_items=research_digest.get("evidence_items"),
                                findings=research_digest.get("findings"),
                                truncated=research_digest.get("truncated"),
                                attached=True,
                            )
                        else:
                            _events.status(
                                "deep_research.attached_to_payload",
                                where="deep_research.attached_to_payload",
                                mode="analyze",
                                cache_key=research_digest.get("cache_key"),
                                evidence_items=research_digest.get("evidence_items"),
                                findings=research_digest.get("findings"),
                                truncated=research_digest.get("truncated"),
                                attached=True,
                            )
                except Exception:
                    pass
            except Exception as e:
                # Attachment must never break analysis.
                self._progress_error(
                    "analyze_mode_deep_research_attach",
                    error=str(e),
                    trace=traceback.format_exc(),
                )

        self._job_update(
            stage="analyze",
            message="Analyzing project and preparing structured improvement plan…",
            progress_pct=15,
        )
        self._progress(
            "analyze_mode_start",
            focus=focus,
            top_cards=len(cards_payload),
        )

        def _trace_model_usage(where: str, res: Any) -> None:
            """Best-effort trace of model id and token usage; never raises."""
            try:
                if not isinstance(res, dict):
                    return
                model = res.get("model") or res.get("model_id") or res.get("id")
                usage = res.get("usage") or res.get("token_usage") or res.get("tokens")
                if not model and not usage:
                    return
                self.st.trace.write(
                    "ANALYSIS",
                    "model_usage",
                    {
                        "where": where,
                        "model": model,
                        "usage": usage,
                        "session_id": getattr(self, "_session_id", None),
                        "job_id": getattr(self, "job_id", None),
                    },
                )
            except Exception:
                # Trace is best-effort; do not affect run.
                return

        def _build_retry_payload(base_payload: Any, err_summary: str) -> Any:
            """Inject a compact retry hint without changing existing payload semantics."""
            try:
                hint = (
                    "RETRY: previous response failed to produce valid JSON for the Analyze Plan "
                    f"schema: {err_summary}. Output ONLY valid JSON conforming to the analyze_plan "
                    "schema. No extra text, no code fences."
                )
                if isinstance(base_payload, dict):
                    p2 = dict(base_payload)
                    p2["retry_hint"] = hint
                    return p2
                # If payload isn't a dict, keep it unchanged.
                return base_payload
            except Exception:
                return base_payload

        # Call the LLM with JSON schema enforcement, with a graceful fallback path.
        plan: Any = None
        res_meta: Any = None
        used_fallback_parse = False
        parse_error: Optional[ParseError] = None
        raw_response_text: Optional[str] = None
        try:
            plan, res_meta = self._chat_json(
                system_text,
                payload,
                schema=ANALYZE_PLAN_SCHEMA,
                temperature=0.0,
                phase="analyze",
                inject_brief=False,  # brief already in payload
                max_tokens=self._phase_max_tokens("analyze"),
            )
            _trace_model_usage("analyze", res_meta)
        except Exception as e:
            self._progress_error(
                "analyze_mode_chat_json",
                error=str(e),
                trace=traceback.format_exc(),
            )
            # Fallback: try a plain text call and parse via canonical helper
            try:
                raw = self._chat(
                    system_text,
                    payload,
                    phase="analyze_fallback",
                    inject_brief=False,
                    max_tokens=self._phase_max_tokens("analyze"),
                )
                raw_response_text = raw if isinstance(raw, str) else str(raw)
                try:
                    # Use the canonical parser which raises ParseError with deterministic fields
                    plan = parse_analyze_plan_text(raw_response_text)
                    parse_error = None
                except ParseError as pe:
                    # Deterministic parse error with message/snippet/suggestion fields
                    plan = None
                    parse_error = pe
                used_fallback_parse = True
            except Exception as e2:
                self._progress_error(
                    "analyze_mode_chat_fallback",
                    error=str(e2),
                    trace=traceback.format_exc(),
                )
                # Emit explicit completion event for error state.
                try:
                    self._emit_analysis_result(
                        focus=focus,
                        analysis="Analyze mode failed while calling LLM.",
                        cards=cards_payload,
                        plan=None,
                        status="error",
                        validation_errors=["llm_call_failed"],
                    )
                except Exception:
                    logging.debug("Failed to emit analyze error result", exc_info=True)
                return False, "Analyze mode failed while calling LLM."

        # Validate the plan against the centralized schema helper (preferred),
        # with a defensive fallback to local jsonschema validation.
        if plan is not None:
            jsonable_plan = self._to_jsonable(plan)
        else:
            # Keep a best-effort representation for UI/tracing when parse failed.
            jsonable_plan = {"raw": raw_response_text} if raw_response_text is not None else None

        ok, validation_errors, validation_exc = self._validate_analyze_plan_payload(
            jsonable_plan
        )
        if validation_exc is not None:
            # Validation infrastructure error (e.g., jsonschema missing).
            self._progress_error(
                "analyze_mode_validate_plan",
                error=str(validation_exc),
                trace=traceback.format_exc(),
            )
            try:
                self._emit_analysis_result(
                    focus=focus,
                    analysis="Analyze validation error.",
                    cards=cards_payload,
                    plan=jsonable_plan,
                    status="error",
                    validation_errors=validation_errors or ["validation_infrastructure_error"],
                )
                self.st.trace.write(
                    "ANALYSIS",
                    "validation_error",
                    {
                        "focus": focus,
                        "plan": jsonable_plan,
                        "validation_errors": validation_errors
                        or ["validation_infrastructure_error"],
                    },
                )
            except Exception:
                logging.debug("Failed to emit/trace analyze validation error", exc_info=True)
            return False, "Analyze validation error"

        # If the initial attempt produced a parse error (detected by parse_analyze_plan_text)
        # or failed schema validation, do at most one bounded retry with a compact hint.
        did_retry = False
        if (not ok) or (parse_error is not None):
            did_retry = True

            # Build deterministic validation diagnostics.
            if parse_error is not None:
                # Use ParseError fields for deterministic diagnostics exposed to the user/UI.
                errs: List[str] = []
                msg = getattr(parse_error, "message", None) or str(parse_error)
                if msg:
                    errs.append(f"json_parse_error: {msg}")
                snippet = getattr(parse_error, "snippet", None)
                if snippet:
                    errs.append(f"snippet: {snippet}")
                suggestion = getattr(parse_error, "suggestion", None)
                if suggestion:
                    errs.append(f"suggestion: {suggestion}")
                validation_errors = errs or ["json_parse_error: response was not valid JSON"]
            elif not validation_errors:
                validation_errors = ["schema_validation_failed"]

            # Build a compact retry hint derived from ParseError or schema summary.
            if parse_error is not None:
                parts = []
                if getattr(parse_error, "message", None):
                    parts.append(str(getattr(parse_error, "message")))
                if getattr(parse_error, "snippet", None):
                    parts.append(f"snippet: {getattr(parse_error, 'snippet')}")
                if getattr(parse_error, "suggestion", None):
                    parts.append(f"suggestion: {getattr(parse_error, 'suggestion')}")
                err_summary = " ; ".join(parts) if parts else "parse_error"
            else:
                err_summary = " ; ".join([str(x) for x in (validation_errors or [])[:3]]) or "parse_error"

            # Inject hint into the system prompt so the model will see it (not as an unused JSON key).
            retry_hint = (
                "RETRY HINT: The previous response was not accepted because it failed to produce valid JSON "
                f"for the Analyze Plan schema ({err_summary}). Please respond with ONLY valid JSON that conforms "
                "to the analyze_plan schema. Do not include any explanatory text, markdown, or code fences."
            )
            system_text_retry = f"{system_text}\n\n{retry_hint}"

            retry_plan: Any = None
            retry_res: Any = None
            try:
                retry_plan, retry_res = self._chat_json(
                    system_text_retry,
                    payload,
                    schema=ANALYZE_PLAN_SCHEMA,
                    temperature=0.0,
                    phase="analyze",
                    inject_brief=False,
                    max_tokens=self._phase_max_tokens("analyze"),
                )
                _trace_model_usage("analyze_retry", retry_res)
            except Exception as e:
                # Retry call failed: treat as invalid with deterministic error.
                self._progress_error(
                    "analyze_mode_retry_chat_json",
                    error=str(e),
                    trace=traceback.format_exc(),
                )
                retry_plan = None

            if retry_plan is not None:
                jsonable_retry_plan = self._to_jsonable(retry_plan)
                ok2, validation_errors2, validation_exc2 = self._validate_analyze_plan_payload(
                    jsonable_retry_plan
                )
                if validation_exc2 is not None:
                    # Infrastructure error on retry validation.
                    self._progress_error(
                        "analyze_mode_retry_validate_plan",
                        error=str(validation_exc2),
                        trace=traceback.format_exc(),
                    )
                    try:
                        self._emit_analysis_result(
                            focus=focus,
                            analysis="Analyze validation error.",
                            cards=cards_payload,
                            plan=jsonable_retry_plan,
                            status="error",
                            validation_errors=validation_errors2
                            or ["validation_infrastructure_error"],
                        )
                        self.st.trace.write(
                            "ANALYSIS",
                            "validation_error",
                            {
                                "focus": focus,
                                "plan": jsonable_retry_plan,
                                "validation_errors": validation_errors2
                                or ["validation_infrastructure_error"],
                                "retry": True,
                            },
                        )
                    except Exception:
                        logging.debug(
                            "Failed to emit/trace analyze retry validation error",
                            exc_info=True,
                        )
                    return False, "Analyze validation error"

                if ok2:
                    # Retry succeeded: proceed with the existing success path unchanged.
                    plan = retry_plan
                    jsonable_plan = jsonable_retry_plan
                    ok = True
                    validation_errors = []
                    parse_error = None
                else:
                    # Retry still invalid: structured failure with deterministic diagnostics.
                    errs2: List[str] = []
                    if validation_errors2:
                        errs2 = validation_errors2
                    else:
                        errs2 = ["schema_validation_failed"]

                    self._progress_error(
                        "analyze_mode_invalid_plan",
                        error="Schema validation failed",
                        validation_errors=errs2,
                    )
                    summary_text = self._summarize_analyze_plan(
                        retry_plan, fallback_focus=focus
                    )
                    try:
                        self._emit_analysis_result(
                            focus=focus,
                            analysis=summary_text,
                            cards=cards_payload,
                            plan=jsonable_retry_plan,
                            status="invalid",
                            validation_errors=errs2,
                        )
                        self.st.trace.write(
                            "ANALYSIS",
                            "validation_failed",
                            {
                                "focus": focus,
                                "plan": jsonable_retry_plan,
                                "summary": summary_text,
                                "validation_errors": errs2,
                                "cards_used": cards_payload,
                                "retry": True,
                            },
                        )
                    except Exception:
                        logging.debug(
                            "Failed to emit/trace analyze invalid result", exc_info=True
                        )

                    return (
                        False,
                        "Analyze produced invalid plan: schema validation failed",
                    )
            else:
                # Retry did not return a plan.
                self._progress_error(
                    "analyze_mode_invalid_plan",
                    error="Schema validation failed",
                    validation_errors=validation_errors,
                )
                summary_text = self._summarize_analyze_plan(plan, fallback_focus=focus)
                try:
                    self._emit_analysis_result(
                        focus=focus,
                        analysis=summary_text,
                        cards=cards_payload,
                        plan=jsonable_plan,
                        status="invalid",
                        validation_errors=validation_errors,
                    )
                    self.st.trace.write(
                        "ANALYSIS",
                        "validation_failed",
                        {
                            "focus": focus,
                            "plan": jsonable_plan,
                            "summary": summary_text,
                            "validation_errors": validation_errors,
                            "cards_used": cards_payload,
                            "retry": True,
                        },
                    )
                except Exception:
                    logging.debug(
                        "Failed to emit/trace analyze invalid result", exc_info=True
                    )
                return False, "Analyze produced invalid plan: schema validation failed"

        if not ok:
            # Schema-invalid output: do not silently accept as success.
            self._progress_error(
                "analyze_mode_invalid_plan",
                error="Schema validation failed",
                validation_errors=validation_errors,
            )

            # Best-effort human-readable summary for the UI (backwards compatible).
            # Keep this for display, but still return failure and emit invalid status.
            summary_text = self._summarize_analyze_plan(plan, fallback_focus=focus)
            # If we got here via fallback parsing with a ParseError, ensure deterministic diagnostics.
            if parse_error is not None:
                errs: List[str] = []
                msg = getattr(parse_error, "message", None) or str(parse_error)
                if msg:
                    errs.append(f"json_parse_error: {msg}")
                snippet = getattr(parse_error, "snippet", None)
                if snippet:
                    errs.append(f"snippet: {snippet}")
                suggestion = getattr(parse_error, "suggestion", None)
                if suggestion:
                    errs.append(f"suggestion: {suggestion}")
                validation_errors = errs or ["json_parse_error: response was not valid JSON"]
            elif not validation_errors:
                validation_errors = ["schema_validation_failed"]
            try:
                self._emit_analysis_result(
                    focus=focus,
                    analysis=summary_text,
                    cards=cards_payload,
                    plan=jsonable_plan,
                    status="invalid",
                    validation_errors=validation_errors,
                )
                self.st.trace.write(
                    "ANALYSIS",
                    "validation_failed",
                    {
                        "focus": focus,
                        "plan": jsonable_plan,
                        "summary": summary_text,
                        "validation_errors": validation_errors,
                        "cards_used": cards_payload,
                        "retry": did_retry,
                    },
                )
            except Exception:
                logging.debug("Failed to emit/trace analyze invalid result", exc_info=True)

            return False, "Analyze produced invalid plan: schema validation failed"

        # Best-effort human-readable summary for the UI (backwards compatible).
        summary_text = self._summarize_analyze_plan(plan, fallback_focus=focus)

        # NOTE: deep research is now run pre-LLM (when gated on) and only a bounded
        # digest is attached to the prompt payload. We keep the previous behavior of
        # exposing research in the final emitted result, but do NOT append large
        # research contents into the human-readable summary.

        # Emit structured result + summary.
        try:
            self._emit_analysis_result(
                focus=focus,
                analysis=summary_text,
                cards=cards_payload,
                plan=jsonable_plan,
                research_brief=research_brief,
                research_summary=None,
                status="success",
            )
            trace_payload: Dict[str, Any] = {
                "focus": focus,
                "plan": jsonable_plan,
                "summary": summary_text,
                "cards_used": cards_payload,
                "status": "success",
            }
            if research_brief is not None:
                # Trace stores jsonable brief; still avoid raw file contents in emitted events.
                trace_payload["research_brief"] = research_brief
            if did_retry:
                trace_payload["retry"] = True
            self.st.trace.write(
                "ANALYSIS",
                "result",
                trace_payload,
            )
        except Exception:
            logging.debug("Failed to emit or trace analysis result", exc_info=True)

        self._progress(
            "analyze_mode_done",
            focus=focus,
            status="success",
        )
        self._job_update(
            stage="analyze",
            message="Analysis completed.",
            progress_pct=100,
        )

        return True, "Analysis completed."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_analyze_plan_payload(
        self, plan: Any
    ) -> Tuple[bool, List[str], Optional[BaseException]]:
        """Validate analyze plan payload.

        Returns (ok, errors, exc):
        - ok: True if valid
        - errors: list of human-readable validation errors (empty if ok)
        - exc: exception if validation infrastructure failed (None otherwise)

        This prefers a centralized helper from aidev.schemas, but falls back to
        local jsonschema validation using ANALYZE_PLAN_SCHEMA.
        """
        errors: List[str] = []

        # 1) Prefer centralized helper(s) if available.
        try:
            validate_analyze_plan = None
            validate_schema = None
            validate_payload = None
            validate = None
            try:
                from .. import schemas as _schemas  # type: ignore

                validate_analyze_plan = getattr(_schemas, "validate_analyze_plan", None)
                validate_schema = getattr(_schemas, "validate_schema", None)
                validate_payload = getattr(_schemas, "validate_payload", None)
                validate = getattr(_schemas, "validate", None)
            except Exception:
                validate_analyze_plan = None
                validate_schema = None
                validate_payload = None
                validate = None

            helper = None
            helper_kind = ""
            if callable(validate_analyze_plan):
                helper = validate_analyze_plan
                helper_kind = "validate_analyze_plan"
            elif callable(validate_schema):
                helper = validate_schema
                helper_kind = "validate_schema"
            elif callable(validate_payload):
                helper = validate_payload
                helper_kind = "validate_payload"
            elif callable(validate):
                helper = validate
                helper_kind = "validate"

            if callable(helper):
                try:
                    if helper_kind == "validate_analyze_plan":
                        res = helper(plan)
                    elif helper_kind in ("validate_schema", "validate_payload"):
                        # Common signature: (name_or_schema, payload)
                        res = helper("analyze_plan", plan)
                    else:
                        # Unknown signature; try (name, payload) first.
                        try:
                            res = helper("analyze_plan", plan)
                        except TypeError:
                            res = helper(plan)

                    # Normalize plausible return shapes.
                    if isinstance(res, tuple) and len(res) >= 2:
                        ok = bool(res[0])
                        raw_errs = res[1]
                        if isinstance(raw_errs, list):
                            errors = [str(x) for x in raw_errs if str(x).strip()]
                        elif raw_errs:
                            errors = [str(raw_errs)]
                        return ok, errors, None

                    if isinstance(res, dict):
                        ok = bool(res.get("valid") or res.get("ok"))
                        raw_errs = res.get("errors") or res.get("validation_errors") or []
                        if isinstance(raw_errs, list):
                            errors = [str(x) for x in raw_errs if str(x).strip()]
                        elif raw_errs:
                            errors = [str(raw_errs)]
                        return ok, errors, None

                    if isinstance(res, bool):
                        return res, [], None

                    # If helper returns something unexpected, fall through to local validation.
                except Exception as e:
                    # Central helper exists but failed: treat as validation infrastructure error.
                    return False, [str(e)], e
        except Exception as e:
            return False, [str(e)], e

        # 2) Local fallback: validate with jsonschema if available.
        try:
            import jsonschema  # type: ignore

            jsonschema.validate(instance=plan, schema=ANALYZE_PLAN_SCHEMA)
            return True, [], None
        except Exception as e:
            # Distinguish schema-invalid vs infrastructure error.
            try:
                import jsonschema  # type: ignore

                if isinstance(e, jsonschema.ValidationError):
                    msg = getattr(e, "message", None) or str(e)
                    path = ""
                    try:
                        if getattr(e, "path", None):
                            path = "/".join([str(p) for p in list(e.path)])
                    except Exception:
                        path = ""
                    if path:
                        errors = [f"{path}: {msg}"]
                    else:
                        errors = [msg]
                    return False, errors, None
            except Exception:
                # If jsonschema import itself fails, treat as infrastructure error.
                pass

            return False, [str(e)], e

    def _best_effort_parse_plan(self, raw: str) -> Any:
        """
        Fallback parser for analyze plan when schema-enforced _chat_json fails.

        - If the model returns valid JSON, return the parsed object.
        - Otherwise, wrap the raw text in a simple object so the UI still has
          something structured to display.
        """
        if not isinstance(raw, str):
            return {"raw": raw}

        text = raw.strip()
        if not text:
            return {"raw": ""}

        # Very lightweight JSON detection to avoid pulling in extra helpers here.
        if text.startswith("{") or text.startswith("["):
            try:
                import json

                return json.loads(text)
            except Exception:
                # Fall back to raw text container.
                pass

        return {"overview": text}

    def _summarize_analyze_plan(self, plan: Any, fallback_focus: str) -> str:
        """
        Convert the structured Analyze Plan into a compact, human-friendly summary
        string for UIs that only know how to display `analysis` text.

        If `plan` is not a dict, we fall back to str(plan). If it's a list, we try
        to treat it as a list of theme-like objects.
        """
        # Handle list-shaped plans (e.g., themes-only array).
        if isinstance(plan, list):
            # Wrap as themes list, then reuse dict logic below.
            plan = {"themes": plan}

        if not isinstance(plan, dict):
            return str(plan)

        parts: List[str] = []

        overview = (plan.get("overview") or "").strip()
        if overview:
            parts.append(overview)

        themes = plan.get("themes")
        if isinstance(themes, list) and themes:
            parts.append("")
            parts.append("Key themes:")
            for theme in themes[:4]:
                if not isinstance(theme, dict):
                    continue
                title = (theme.get("title") or theme.get("id") or "").strip()
                if not title:
                    continue
                impact = (theme.get("impact") or "").strip()
                effort = (theme.get("effort") or "").strip()
                summary = (theme.get("summary") or theme.get("description") or "").strip()
                meta_bits: List[str] = []
                if impact:
                    meta_bits.append(f"impact: {impact}")
                if effort:
                    meta_bits.append(f"effort: {effort}")
                meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
                if summary:
                    parts.append(f"- {title}{meta}: {summary}")
                else:
                    parts.append(f"- {title}{meta}")

        text = "\n".join(parts).strip()
        if not text:
            return f"Analyze plan for focus '{fallback_focus}': {plan!r}"
        return text

    def _emit_analysis_result(
        self,
        *,
        focus: str,
        analysis: str,
        cards: List[Dict[str, Any]],
        plan: Any = None,
        research_brief: Any = None,
        research_summary: Optional[str] = None,
        status: str = "success",
        validation_errors: Optional[List[str]] = None,
    ) -> None:
        """
        Helper to emit a structured analyze completion event.

        Behavior:
        - Prefer the new canonical emit_analyze_result() in the events module.
          Call it with explicit keywords (plan, status, validation_errors, etc.)
          so the completion event is typed as 'analyze_result' with structured
          diagnostics.
        - For backward compatibility, also attempt to emit the legacy
          analyze_result(...) if present, and finally fall back to the
          generic status(...) event.

        The payload always includes `status` (one of: 'success', 'invalid', 'error')
        and structured `validation_errors` when present so UIs can deterministically
        display validation failures.
        """
        if validation_errors is None:
            validation_errors = []

        payload: Dict[str, Any] = {
            "focus": focus,
            "analysis": analysis,
            "cards": cards,
            "plan": plan,
            "status": status,
            "validation_errors": validation_errors,
            "session_id": getattr(self, "_session_id", None),
            "job_id": getattr(self, "job_id", None),
        }
        # Only include research fields when present/successful to avoid changing
        # existing behavior/shape on failures or when feature is absent.
        if research_brief is not None:
            payload["research_brief"] = research_brief
        if research_summary:
            payload["research_summary"] = research_summary

        try:
            # 1) Prefer the canonical new emitter
            fn_emit = getattr(_events, "emit_analyze_result", None)
            if callable(fn_emit):
                try:
                    # Try the obvious keyword-arg form first (plan as a named kw).
                    fn_emit(**payload)
                except TypeError:
                    try:
                        # Some implementations may expect a single dict arg.
                        fn_emit(payload)
                    except Exception:
                        try:
                            # Last resort: try common alternate signature
                            # (plan, status, validation_errors)
                            fn_emit(
                                payload.get("plan"),
                                payload.get("status"),
                                payload.get("validation_errors"),
                            )
                        except Exception:
                            logging.debug(
                                "emit_analyze_result call failed with unexpected signature",
                                exc_info=True,
                            )
                except Exception:
                    logging.debug("emit_analyze_result raised", exc_info=True)

                # Also attempt to emit the legacy analyze_result for backward compatibility
                try:
                    fn_legacy = getattr(_events, "analyze_result", None)
                    if callable(fn_legacy):
                        try:
                            fn_legacy(**payload)
                        except Exception:
                            try:
                                fn_legacy(payload)
                            except Exception:
                                logging.debug(
                                    "legacy analyze_result emission failed", exc_info=True
                                )
                except Exception:
                    logging.debug(
                        "Failed while trying to emit legacy analyze_result", exc_info=True
                    )

                return

            # 2) If canonical emitter missing, try the legacy analyze_result name
            fn_legacy = getattr(_events, "analyze_result", None)
            if callable(fn_legacy):
                try:
                    fn_legacy(**payload)
                    return
                except Exception:
                    try:
                        fn_legacy(payload)
                        return
                    except Exception:
                        logging.debug("legacy analyze_result emission failed", exc_info=True)

            # 3) Fallback to generic status event to ensure something is emitted.
            _events.status(
                analysis or "Analyze mode completed.",
                where="analysis_result",
                **payload,
            )
        except Exception:
            logging.debug("Failed to emit analyze_result event", exc_info=True)
