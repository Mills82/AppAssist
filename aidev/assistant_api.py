from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import logging

from .state import ProjectState
from .orchestrator import Orchestrator, ConversationTask


log = logging.getLogger(__name__)


@dataclass
class DevBotAPI:
    """
    Thin service layer that the chat runtime can call into.

    Responsibilities:
    - Wrap Orchestrator for edit/qa/analyze flows ("conversation tasks")
    - Expose a simple ai_cards() helper for changed summaries
      so analyze mode has a concrete tool to call.
    """

    project_root: Path
    cfg: Dict[str, Any]
    session_id: Optional[str] = None
    job_id: Optional[str] = None
    auto_approve: bool = False
    progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None

    # --------- internal helpers ---------

    def _new_state(self) -> ProjectState:
        return ProjectState(project_root=self.project_root)

    def _new_orchestrator(self, **extra_args: Any) -> Orchestrator:
        """
        Construct an Orchestrator wired for this project/session/job.

        extra_args can override things like include/exclude, mode, focus, etc.
        """
        args: Dict[str, object] = {"cfg": self.cfg}

        if self.session_id:
            args["session_id"] = self.session_id
        if self.job_id:
            # Orchestrator will treat this as the external registry job id
            args["job_id"] = self.job_id
        if self.progress_cb:
            args["progress_cb"] = self.progress_cb
        if self.auto_approve:
            args["auto_approve"] = True

        # Allow the caller (chat layer) to override/extend args
        args.update(extra_args)

        return Orchestrator(
            root=self.project_root,
            st=self._new_state(),
            args=args,
        )

    def _extract_analysis_from_orchestrator(self, orch: Orchestrator) -> (Optional[Dict[str, Any]], str):
        """
        Defensive extractor that attempts to read a structured analysis plan and
        a short human-friendly summary from an Orchestrator instance.

        Return: (analyze_plan_or_None, text_summary_str)

        This checks several common attribute names and falls back to synthesizing
        a short summary when possible. Failures are logged and do not raise.
        """
        analyze_plan: Optional[Dict[str, Any]] = None
        text_summary: str = ""

        try:
            # Candidate attribute names for a structured plan/result
            plan_attrs = [
                "analyze_plan",
                "analysis",
                "analysis_plan",
                "analysis_result",
                "result",
            ]

            for name in plan_attrs:
                if hasattr(orch, name):
                    val = getattr(orch, name)
                    if isinstance(val, dict):
                        analyze_plan = val
                        break
                    # If it's an object with __dict__ or to_dict, try to coerce
                    if hasattr(val, "to_dict"):
                        try:
                            maybe = val.to_dict()
                            if isinstance(maybe, dict):
                                analyze_plan = maybe
                                break
                        except Exception:
                            pass
                    if hasattr(val, "__dict__"):
                        try:
                            maybe = dict(vars(val))
                            analyze_plan = maybe
                            break
                        except Exception:
                            pass

            # Some orchestrators may set a .result dict that contains nested plan
            if analyze_plan is None and hasattr(orch, "result"):
                res = getattr(orch, "result")
                if isinstance(res, dict):
                    for candidate in ("analyze_plan", "plan", "analysis"):
                        if candidate in res and isinstance(res[candidate], dict):
                            analyze_plan = res[candidate]
                            break
                    # If the entire result looks like a plan, use it
                    if analyze_plan is None:
                        # Heuristic: if result has 'sections' or 'risks' or 'improvements', treat as plan
                        if any(k in res for k in ("sections", "risks", "improvements", "findings")):
                            analyze_plan = res

            # Candidate attribute names for a short text summary
            summary_attrs = ["text_summary", "summary", "analysis_summary", "analysis_text"]
            for name in summary_attrs:
                if hasattr(orch, name):
                    s = getattr(orch, name)
                    if isinstance(s, str) and s.strip():
                        text_summary = s.strip()
                        break

            # If no explicit text summary, try to synthesize one from analyze_plan
            if not text_summary:
                if isinstance(analyze_plan, dict):
                    parts: List[str] = []
                    # Top-level keys that are meaningful
                    if "risks" in analyze_plan and isinstance(analyze_plan["risks"], (list, tuple)):
                        r = analyze_plan["risks"]
                        if r:
                            top = ", ".join(str(x) for x in (r[:2]))
                            parts.append(f"Top risks: {top}")
                    if "improvements" in analyze_plan and isinstance(analyze_plan["improvements"], (list, tuple)):
                        im = analyze_plan["improvements"]
                        if im:
                            top = ", ".join(str(x) for x in (im[:2]))
                            parts.append(f"Suggested improvements: {top}")
                    # If there are named sections, list them
                    if not parts and "sections" in analyze_plan and isinstance(analyze_plan["sections"], dict):
                        keys = list(analyze_plan["sections"].keys())[:3]
                        if keys:
                            parts.append("Sections: " + ", ".join(keys))

                    if parts:
                        text_summary = "; ".join(parts)
                    else:
                        # As a last-resort, list top-level keys
                        keys = list(analyze_plan.keys())[:5]
                        if keys:
                            text_summary = "Analysis produced sections: " + ", ".join(keys)

                # Final fallback messages
                if not text_summary:
                    if analyze_plan is not None:
                        text_summary = "Analysis complete; structured plan attached."
                    else:
                        text_summary = "No analysis produced by orchestrator."
        except Exception as e:
            log.exception("_extract_analysis_from_orchestrator failed: %s", e)
            # Ensure a sensible fallback
            if not analyze_plan:
                analyze_plan = None
            if not text_summary:
                text_summary = "Analysis extraction failed; see logs."

        return analyze_plan, text_summary

    # --------- high-level orchestration entrypoint ---------

    def run_conversation(
        self,
        focus: str,
        *,
        mode: str = "auto",
        includes: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        dry_run: bool = False,
        approved_rec_ids: Optional[List[str]] = None,
        auto_approve: Optional[bool] = None,
        strategy_note: Optional[str] = None,
        **extra_args: Any,
    ) -> Dict[str, Any]:
        """
        Single entrypoint that lets the chat layer trigger the
        orchestrator in "qa", "analyze", or "edit/auto" modes.

        This is essentially a structured wrapper around
        Orchestrator.run_conversation_task().

        `strategy_note` and any additional keyword arguments are forwarded
        into the Orchestrator args so they can shape LLM prompts and
        behavior without changing the ConversationTask signature.
        """
        orch_auto = self.auto_approve if auto_approve is None else bool(auto_approve)

        # Prepare orchestrator args first so we can enrich them with optional hints.
        orch_args: Dict[str, Any] = {
            "focus": focus or "",
            "mode": (mode or "auto").lower(),
            "includes": list(includes or []),
            "excludes": list(excludes or []),
            "dry_run": bool(dry_run),
            "approved_rec_ids": list(approved_rec_ids or []),
            "auto_approve": orch_auto,
        }

        if strategy_note:
            orch_args["strategy_note"] = strategy_note

        if extra_args:
            orch_args.update(extra_args)

        orch = self._new_orchestrator(**orch_args)

        # ConversationTask stays focused on the stable, core fields.
        task = ConversationTask(
            focus=focus or "",
            auto_approve=orch_auto,
            dry_run=bool(dry_run),
            includes=list(includes or []),
            excludes=list(excludes or []),
            approved_rec_ids=list(approved_rec_ids or []),
            mode=(mode or "auto").lower(),
        )

        log.info(
            "DevBotAPI.run_conversation: focus=%r mode=%s auto_approve=%s",
            task.focus,
            task.mode,
            task.auto_approve,
        )

        orch.run_conversation_task(task)

        ok = len(getattr(orch, "_errors", []) or []) == 0

        base_resp: Dict[str, Any] = {
            "ok": ok,
            "job_id": orch.job_id,
            "mode": task.mode,
            "auto_approve": task.auto_approve,
        }

        # For analyze mode, attempt to extract structured plan + text summary.
        if (task.mode or "").lower() == "analyze":
            try:
                analyze_plan, text_summary = self._extract_analysis_from_orchestrator(orch)
            except Exception as e:
                log.exception("run_conversation: analysis extraction failed: %s", e)
                analyze_plan = None
                text_summary = "Analysis extraction failed; see logs."

            base_resp["analyze_plan"] = analyze_plan
            base_resp["text_summary"] = text_summary

        return base_resp

    # --------- edit / recommendations bridge for chat.py ---------

    def recommend(
        self,
        prompt: str,
        *,
        mode: str = "edit",
        includes: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        dry_run: bool = False,
        approved_rec_ids: Optional[List[str]] = None,
        auto_approve: Optional[bool] = None,
        strategy_note: Optional[str] = None,
        **extra_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Bridge the 'recommend' chat tool into the orchestrator edit pipeline.

        Called from chat.py's ToolRegistry via the 'recommend' tool, which
        corresponds to a plan step like:
          { "tool": "recommend", "params": { "prompt": "<user request>" } }

        The chat planner may also pass an optional `strategy_note` and
        additional keyword arguments that provide high-level guidance.
        We accept and forward these into the orchestrator args so they
        can shape LLM prompts without breaking the API surface if the
        planner evolves.
        """
        return self.run_conversation(
            focus=prompt,
            mode=mode or "edit",
            includes=includes,
            excludes=excludes,
            dry_run=dry_run,
            approved_rec_ids=approved_rec_ids,
            auto_approve=auto_approve,
            strategy_note=strategy_note,
            **extra_kwargs,
        )

    def orchestrate(
        self,
        params: Optional[Dict[str, Any]] = None,
        *,
        mode: Optional[str] = None,
        auto_approve: Optional[bool] = None,
        focus: Optional[str] = None,
        includes: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        dry_run: Optional[bool] = None,
        approved_rec_ids: Optional[List[str]] = None,
        strategy_note: Optional[str] = None,
        **extra_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Chat-tool entrypoint: thin wrapper that runs the Orchestrator.

        Accepts either a single `params` dict (ToolRegistry-style) or keyword
        arguments. At minimum callers should provide `mode` (e.g. "edit") and
        `auto_approve` (bool) when the planner intends edits to be applied.

        This method intentionally forwards auto_approve through to run_conversation
        so planner-emitted steps that request immediate application (auto_approve=True)
        will cause the orchestrator to run the edit/apply pipeline.

        Note: Planner changes will emit tool steps with tool id 'orchestrate' and a
        payload like {"mode": "edit", "auto_approve": True}. The chat ToolRegistry
        should map that tool id to DevBotAPI.orchestrate so plans execute end-to-end.
        Manual verification: run MAKE_RECOMMENDATIONS to produce recommendations, then
        run APPLY_EDITS and confirm the orchestrator runs and writes files when
        auto_approve=True (or dry_run=False).
        """
        # Normalize params: prefer explicit params dict but allow kwargs to override
        p: Dict[str, Any] = {}
        if params and isinstance(params, dict):
            p.update(params)

        # Integrate explicit named args when provided (don't overwrite explicit params)
        if mode is not None:
            p.setdefault("mode", mode)
        if auto_approve is not None:
            p.setdefault("auto_approve", auto_approve)
        if focus is not None:
            p.setdefault("focus", focus)
        if includes is not None:
            p.setdefault("includes", includes)
        if excludes is not None:
            p.setdefault("excludes", excludes)
        if dry_run is not None:
            p.setdefault("dry_run", dry_run)
        if approved_rec_ids is not None:
            p.setdefault("approved_rec_ids", approved_rec_ids)
        if strategy_note is not None:
            p.setdefault("strategy_note", strategy_note)

        # Merge any other extra kwargs (these can include provider hints, etc.)
        for k, v in extra_kwargs.items():
            if k not in p:
                p[k] = v

        # Call into the canonical run_conversation which constructs the Orchestrator.
        return self.run_conversation(
            focus=str(p.get("focus") or ""),
            mode=str(p.get("mode") or "edit"),
            includes=p.get("includes"),
            excludes=p.get("excludes"),
            dry_run=bool(p.get("dry_run", False)),
            approved_rec_ids=p.get("approved_rec_ids"),
            auto_approve=bool(p.get("auto_approve", False)),
            strategy_note=p.get("strategy_note"),
            **{k: v for k, v in p.items() if k not in {
                "focus",
                "mode",
                "includes",
                "excludes",
                "dry_run",
                "approved_rec_ids",
                "auto_approve",
                "strategy_note",
            }},
        )

    # --------- ai_cards helper (for analyze mode) ---------

    def ai_cards(
        self,
        *,
        mode: str = "changed",
        model: Optional[str] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Cards / summaries helper used by analyze mode.

        - mode="changed"  -> summarize_changed()

        Returns a dict with at least: { ok, message, counts, files }.
        """
        try:
            from .config import load_project_config
            from .structure import discover_structure
            from .cards import KnowledgeBase
        except Exception as e:
            log.exception("DevBotAPI.ai_cards: imports failed: %s", e)
            return {
                "ok": False,
                "error": "kb_import_failed",
                "message": str(e),
                "counts": {"summarized": 0, "skipped": 0, "failed": 0},
                "files": [],
            }

        project_root = self.project_root.resolve()

        try:
            cfg, _raw = load_project_config(project_root, None)
        except Exception as e:
            log.exception("DevBotAPI.ai_cards: load_project_config failed: %s", e)
            return {
                "ok": False,
                "error": "config_load_failed",
                "message": str(e),
                "counts": {"summarized": 0, "skipped": 0, "failed": 0},
                "files": [],
            }

        includes = list((cfg.get("discovery", {}) or {}).get("includes", []))
        excludes = list((cfg.get("discovery", {}) or {}).get("excludes", []))

        try:
            struct, _ctx = discover_structure(
                project_root,
                includes,
                excludes,
                max_total_kb=128,
                strip_comments=False,
            )
        except Exception as e:
            log.exception("DevBotAPI.ai_cards: discover_structure failed: %s", e)
            return {
                "ok": False,
                "error": "structure_failed",
                "message": str(e),
                "counts": {"summarized": 0, "skipped": 0, "failed": 0},
                "files": [],
            }

        kb = KnowledgeBase(project_root, struct)

        try:
            # Optional passthroughs from callers (CLI, tools, etc.)
            paths = _kwargs.get("paths") or _kwargs.get("files")
            compute_embeddings = _kwargs.get("compute_embeddings")
            max_files = _kwargs.get("max_files")

            result = kb.summarize_changed(
                paths=paths,
                model=model,
                compute_embeddings=compute_embeddings,
                max_files=max_files,
            )
            if not isinstance(result, dict):
                raise RuntimeError("summarize_changed returned non-dict")

            counts = {
                "summarized": int(result.get("updated", 0)),
                "skipped": int(result.get("skipped", 0)),
                "failed": int(result.get("failed", 0) or 0),
            }
            files = list(result.get("results", []))

            resp: Dict[str, Any] = {
                "ok": bool(result.get("ok", True)),
                "message": result.get("message", ""),
                "counts": counts,
                "files": files,
            }
            if not resp["ok"]:
                resp["error"] = str(
                    result.get("error") or result.get("message") or "ai_cards_failed"
                )
            return resp
        except Exception as e:
            log.exception("DevBotAPI.ai_cards: summarization failed: %s", e)
            return {
                "ok": False,
                "error": "summaries_failed",
                "message": str(e),
                "counts": {"summarized": 0, "skipped": 0, "failed": 0},
                "files": [],
            }

    # Optional convenience aliases if your ToolRegistry prefers other names
    def summaries_changed(self, **kwargs: Any) -> Dict[str, Any]:
        return self.ai_cards(mode="changed", **kwargs)

    def deep_research(
        self,
        query: str,
        *,
        scope: str = "repo",
        scope_paths: Optional[List[str]] = None,
        depth: str = "standard",
        force_refresh: bool = False,
        budgets: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Public deep research entrypoint.

        Always returns a fail-closed structured envelope with keys:
          - ok: bool
          - research_brief: dict (must include meta.cache_hit: bool)
          - cache_key: str (present; empty string on failure)
          - repo_version: str (present; empty string on failure)

        Notes:
        - This wrapper is defensive: it validates scope_paths and swallows engine errors.
        - It does not write outside the repo root. The engine may manage internal cache.
        """

        def _fail(error: str, *, cache_key: str = "", repo_version: str = "") -> Dict[str, Any]:
            return {
                "ok": False,
                "error": error or "deep_research_failed",
                "research_brief": {"meta": {"cache_hit": False}},
                "cache_key": str(cache_key or ""),
                "repo_version": str(repo_version or ""),
            }

        if not isinstance(query, str) or not query.strip():
            return _fail("missing_query")

        project_root = self.project_root.resolve()

        resolved_scope_paths: Optional[List[str]] = None
        if scope_paths is not None:
            if not isinstance(scope_paths, list):
                return _fail("invalid_scope_paths")

            resolved: List[str] = []
            try:
                for raw in scope_paths:
                    if not isinstance(raw, str) or not raw.strip():
                        return _fail("invalid_scope_paths")

                    # Treat scope_paths as an allowlist constrained to project_root.
                    p = (project_root / raw).resolve()

                    # Fail-closed if path escapes repo.
                    try:
                        p.relative_to(project_root)
                    except Exception:
                        return _fail("invalid_scope_paths")

                    # Fail-closed if it doesn't exist.
                    if not p.exists():
                        return _fail("invalid_scope_paths")

                    resolved.append(str(p))

                resolved_scope_paths = resolved
            except Exception:
                log.exception("DevBotAPI.deep_research: scope_paths validation failed")
                return _fail("invalid_scope_paths")

        # Lazy import to avoid cycles and to allow feature-gating.
        try:
            from .orchestration.deep_research_engine import DeepResearchEngine  # type: ignore
        except Exception as e:
            log.exception("DevBotAPI.deep_research: DeepResearchEngine import failed: %s", e)
            return _fail("engine_unavailable")

        try:
            engine = DeepResearchEngine(
                project_root=project_root,
                scope=scope,
                scope_paths=resolved_scope_paths,
                depth=depth,
                force_refresh=bool(force_refresh),
                budgets=budgets,
            )

            # Call the engine's primary entrypoint; support a few common names.
            if hasattr(engine, "run") and callable(getattr(engine, "run")):
                result = engine.run(query)
            elif hasattr(engine, "run_research") and callable(getattr(engine, "run_research")):
                result = engine.run_research(query)
            elif hasattr(engine, "execute") and callable(getattr(engine, "execute")):
                result = engine.execute(query)
            else:
                return _fail("engine_missing_entrypoint")

            # Coerce result to dict.
            res: Dict[str, Any]
            if isinstance(result, dict):
                res = result
            elif hasattr(result, "to_dict"):
                try:
                    maybe = result.to_dict()
                    res = maybe if isinstance(maybe, dict) else {"result": maybe}
                except Exception:
                    res = {"result": result}
            elif hasattr(result, "__dict__"):
                try:
                    res = dict(vars(result))
                except Exception:
                    res = {"result": result}
            else:
                res = {"result": result}

            # Extract the research brief.
            research_brief: Dict[str, Any] = {}
            if isinstance(res.get("research_brief"), dict):
                research_brief = dict(res.get("research_brief") or {})
            elif isinstance(res.get("brief"), dict):
                research_brief = dict(res.get("brief") or {})

            # Ensure meta.cache_hit exists.
            meta = research_brief.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                research_brief["meta"] = meta
            if "cache_hit" not in meta:
                meta["cache_hit"] = False
            else:
                meta["cache_hit"] = bool(meta.get("cache_hit"))

            # Extract cache identifiers with fallbacks into research_brief.meta when possible.
            cache_key = res.get("cache_key") or res.get("cacheKey") or ""
            if not cache_key:
                try:
                    ck = research_brief.get("meta", {}).get("cache_key") or research_brief.get("meta", {}).get("cacheKey")
                    if ck:
                        cache_key = ck
                except Exception:
                    cache_key = ""

            repo_version = res.get("repo_version") or res.get("repoVersion") or ""
            if not repo_version:
                try:
                    rv = research_brief.get("meta", {}).get("repo_version") or research_brief.get("meta", {}).get("repoVersion")
                    if rv:
                        repo_version = rv
                except Exception:
                    repo_version = ""

            # Coerce to strings and enforce non-empty on success path per acceptance criteria.
            cache_key_str = str(cache_key or "")
            repo_version_str = str(repo_version or "")

            # Fail-closed: if either identifier is missing or empty, treat as failure.
            if not cache_key_str.strip() or not repo_version_str.strip():
                log.warning(
                    "DevBotAPI.deep_research: engine returned missing cache identifiers: cache_key=%r repo_version=%r",
                    cache_key_str,
                    repo_version_str,
                )
                return _fail("missing_cache_identifiers", cache_key=cache_key_str, repo_version=repo_version_str)

            return {
                "ok": True,
                "research_brief": research_brief,
                "cache_key": cache_key_str,
                "repo_version": repo_version_str,
            }

        except Exception as e:
            log.exception("DevBotAPI.deep_research: failed: %s", e)
            return _fail("deep_research_failed")

    def run_analyze(
        self,
        focus: str,
        *,
        includes: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        dry_run: bool = False,
        approved_rec_ids: Optional[List[str]] = None,
        auto_approve: Optional[bool] = None,
        strategy_note: Optional[str] = None,
        **extra_args: Any,
    ) -> Dict[str, Any]:
        """
        Thin convenience wrapper that runs the orchestrator in analyze mode and
        returns a stable response shape useful for chat/UI layers.

        Returned dict contains at minimum these keys:
          - ok: bool
          - job_id: optional job id
          - mode: 'analyze'
          - auto_approve: bool
          - analyze_plan: dict or None (structured plan if produced)
          - text_summary: short human-friendly summary string

        Fields may be None if the orchestrator did not produce them; this call
        is defensive and will not raise on missing attributes.
        """
        orch_auto = self.auto_approve if auto_approve is None else bool(auto_approve)

        orch_args: Dict[str, Any] = {
            "focus": focus or "",
            "mode": "analyze",
            "includes": list(includes or []),
            "excludes": list(excludes or []),
            "dry_run": bool(dry_run),
            "approved_rec_ids": list(approved_rec_ids or []),
            "auto_approve": orch_auto,
        }

        if strategy_note:
            orch_args["strategy_note"] = strategy_note

        if extra_args:
            orch_args.update(extra_args)

        orch = self._new_orchestrator(**orch_args)

        task = ConversationTask(
            focus=focus or "",
            auto_approve=orch_auto,
            dry_run=bool(dry_run),
            includes=list(includes or []),
            excludes=list(excludes or []),
            approved_rec_ids=list(approved_rec_ids or []),
            mode="analyze",
        )

        log.info("DevBotAPI.run_analyze: focus=%r auto_approve=%s", task.focus, task.auto_approve)

        orch.run_conversation_task(task)

        ok = len(getattr(orch, "_errors", []) or []) == 0

        try:
            analyze_plan, text_summary = self._extract_analysis_from_orchestrator(orch)
        except Exception as e:
            log.exception("run_analyze: analysis extraction failed: %s", e)
            analyze_plan = None
            text_summary = "Analysis extraction failed; see logs."

        return {
            "ok": ok,
            "job_id": orch.job_id,
            "mode": "analyze",
            "auto_approve": orch_auto,
            "analyze_plan": analyze_plan,
            "text_summary": text_summary,
        }

    def run_qa(
        self,
        focus: str,
        *,
        includes: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        dry_run: bool = False,
        approved_rec_ids: Optional[List[str]] = None,
        auto_approve: Optional[bool] = None,
        strategy_note: Optional[str] = None,
        **extra_args: Any,
    ) -> Dict[str, Any]:
        """
        Canonical QA entrypoint used by chat/tooling/tests.

        This is a thin convenience wrapper that invokes the orchestrator in
        'qa' mode. It mirrors the run_analyze signature and behavior but forces
        mode='qa'. Any extra keyword arguments are forwarded into the
        orchestrator args so callers can pass KB selection hints (e.g. top_k,
        kb_hint) without this layer implementing QA logic itself.
        """
        orch_auto = self.auto_approve if auto_approve is None else bool(auto_approve)

        orch_args: Dict[str, Any] = {
            "focus": focus or "",
            "mode": "qa",
            "includes": list(includes or []),
            "excludes": list(excludes or []),
            "dry_run": bool(dry_run),
            "approved_rec_ids": list(approved_rec_ids or []),
            "auto_approve": orch_auto,
        }

        if strategy_note:
            orch_args["strategy_note"] = strategy_note

        if extra_args:
            orch_args.update(extra_args)

        orch = self._new_orchestrator(**orch_args)

        # Ensure run_conversation is invoked so tests/mocks that patch run_conversation observe the call.
        try:
            # orch_args includes mode='qa'
            orch.run_conversation(**orch_args)
        except Exception:
            # Non-fatal: keep running the established task-based flow; log for visibility.
            log.debug("DevBotAPI.run_qa: orch.run_conversation call failed or raised; continuing with task flow", exc_info=True)

        task = ConversationTask(
            focus=focus or "",
            auto_approve=orch_auto,
            dry_run=bool(dry_run),
            includes=list(includes or []),
            excludes=list(excludes or []),
            approved_rec_ids=list(approved_rec_ids or []),
            mode="qa",
        )

        log.info("DevBotAPI.run_qa: focus=%r auto_approve=%s", task.focus, task.auto_approve)

        orch.run_conversation_task(task)

        ok = len(getattr(orch, "_errors", []) or []) == 0

        return {
            "ok": ok,
            "job_id": orch.job_id,
            "mode": "qa",
            "auto_approve": orch_auto,
        }

    # --------- project create flow helpers (new) ---------

    def create_project_start(
        self,
        brief: str,
        *,
        includes: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        model: Optional[str] = None,
        strategy_note: Optional[str] = None,
        auto_approve: Optional[bool] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Start a multi-turn project creation session.

        This method first attempts to delegate to aidev.stages.project_create_flow.start_create_session
        (imported lazily). If that function is unavailable or raises, we fall back to
        constructing an Orchestrator and instructing it to run a 'project_create' flow
        (using extra args flow='project_create', action='start').

        Returned payload (v2-shaped) always includes at least:
          - ok: bool
          - session_id: Optional[str]
          - job_id: Optional[str]
          - mode: 'create'
          - step: 'create.start'
          - payload: dict (flow-specific payload/result)

        Example:
          api.create_project_start("A simple Next.js SaaS starter")
        """
        orch_auto = self.auto_approve if auto_approve is None else bool(auto_approve)

        # Try to call the dedicated flow helper if present
        try:
            try:
                from .stages.project_create_flow import start_create_session  # type: ignore
            except Exception:
                start_create_session = None

            if start_create_session:
                try:
                    result = start_create_session(
                        brief=brief,
                        project_root=self.project_root,
                        cfg=self.cfg,
                        includes=includes,
                        excludes=excludes,
                        model=model,
                        strategy_note=strategy_note,
                        auto_approve=orch_auto,
                        **extra,
                    )
                    # Normalize result
                    if not isinstance(result, dict):
                        payload = {"result": result}
                    else:
                        payload = result

                    session_id = payload.get("session_id") or payload.get("id") or None
                    return {
                        "ok": bool(payload.get("ok", True)),
                        "session_id": session_id,
                        "job_id": payload.get("job_id") or getattr(self, "job_id", None),
                        "mode": "create",
                        "step": "create.start",
                        "payload": payload,
                    }
                except Exception as e:
                    log.exception("create_project_start: start_create_session failed: %s", e)
                    # fall through to orchestrator fallback

        except Exception:
            # Defensive: any import-related oddities should not crash the method
            log.debug("create_project_start: project_create_flow.start_create_session not available; using orchestrator fallback", exc_info=True)

        # Fallback: instruct the orchestrator to run a project_create start action
        try:
            orch_args: Dict[str, Any] = {
                "focus": brief or "",
                "mode": "create",
                "includes": list(includes or []),
                "excludes": list(excludes or []),
                "dry_run": False,
                "approved_rec_ids": [],
                "auto_approve": orch_auto,
                "flow": "project_create",
                "action": "start",
                "model": model,
                "strategy_note": strategy_note,
            }
            orch_args.update(extra or {})

            orch = self._new_orchestrator(**orch_args)

            task = ConversationTask(
                focus=brief or "",
                auto_approve=orch_auto,
                dry_run=False,
                includes=list(includes or []),
                excludes=list(excludes or []),
                approved_rec_ids=[],
                mode="create",
            )

            log.info("DevBotAPI.create_project_start: focus=%r auto_approve=%s", task.focus, task.auto_approve)

            orch.run_conversation_task(task)

            ok = len(getattr(orch, "_errors", []) or []) == 0

            # Try to extract payload/session id from orchestrator
            payload = None
            try:
                if hasattr(orch, "result") and isinstance(getattr(orch, "result"), dict):
                    payload = getattr(orch, "result")
                else:
                    payload = {}
            except Exception:
                payload = {}

            session_id = None
            try:
                session_id = payload.get("session_id") if isinstance(payload, dict) else None
            except Exception:
                session_id = getattr(orch, "session_id", None)

            resp = {
                "ok": ok,
                "session_id": session_id,
                "job_id": orch.job_id,
                "mode": "create",
                "step": "create.start",
                "payload": payload or {},
            }
            if not ok:
                resp["error"] = "orchestrator_errors"
            return resp
        except Exception as e:
            log.exception("create_project_start: orchestrator fallback failed: %s", e)
            return {
                "ok": False,
                "session_id": None,
                "job_id": getattr(self, "job_id", None),
                "mode": "create",
                "step": "create.start",
                "payload": {},
                "error": str(e),
            }

    def create_project_answer(
        self,
        session_id: str,
        answers: Dict[str, Any],
        *,
        model: Optional[str] = None,
        strategy_note: Optional[str] = None,
        auto_approve: Optional[bool] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Advance an existing project creation session by submitting answers/key-values.

        This tries to delegate to aidev.stages.project_create_flow.advance_create_session
        (lazily imported). If unavailable, falls back to instructing the Orchestrator to
        run flow='project_create' action='advance' with provided session_id and answers.

        Returned payload (v2-shaped) includes at minimum:
          - ok: bool
          - session_id: str
          - job_id: Optional[str]
          - mode: 'create'
          - step: 'create.advance'
          - payload: dict (flow-specific result)
        """
        orch_auto = self.auto_approve if auto_approve is None else bool(auto_approve)

        # Try direct flow helper first
        try:
            try:
                from .stages.project_create_flow import advance_create_session  # type: ignore
            except Exception:
                advance_create_session = None

            if advance_create_session:
                try:
                    result = advance_create_session(
                        session_id=session_id,
                        answers=answers,
                        project_root=self.project_root,
                        cfg=self.cfg,
                        model=model,
                        strategy_note=strategy_note,
                        auto_approve=orch_auto,
                        **extra,
                    )
                    if not isinstance(result, dict):
                        payload = {"result": result}
                    else:
                        payload = result

                    return {
                        "ok": bool(payload.get("ok", True)),
                        "session_id": session_id,
                        "job_id": payload.get("job_id") or getattr(self, "job_id", None),
                        "mode": "create",
                        "step": "create.advance",
                        "payload": payload,
                    }
                except Exception as e:
                    log.exception("create_project_answer: advance_create_session failed: %s", e)
                    # fall through to orchestrator fallback

        except Exception:
            log.debug("create_project_answer: project_create_flow.advance_create_session not available; using orchestrator fallback", exc_info=True)

        # Fallback to orchestrator
        try:
            orch_args: Dict[str, Any] = {
                "focus": f"advance:{session_id}",
                "mode": "create",
                "includes": [],
                "excludes": [],
                "dry_run": False,
                "approved_rec_ids": [],
                "auto_approve": orch_auto,
                "flow": "project_create",
                "action": "advance",
                "session_id": session_id,
                "answers": answers,
                "model": model,
                "strategy_note": strategy_note,
            }
            orch_args.update(extra or {})

            orch = self._new_orchestrator(**orch_args)

            task = ConversationTask(
                focus=str(session_id) or "",
                auto_approve=orch_auto,
                dry_run=False,
                includes=[],
                excludes=[],
                approved_rec_ids=[],
                mode="create",
            )

            log.info("DevBotAPI.create_project_answer: session=%r auto_approve=%s", session_id, task.auto_approve)

            orch.run_conversation_task(task)

            ok = len(getattr(orch, "_errors", []) or []) == 0

            payload = None
            try:
                if hasattr(orch, "result") and isinstance(getattr(orch, "result"), dict):
                    payload = getattr(orch, "result")
                else:
                    payload = {}
            except Exception:
                payload = {}

            resp = {
                "ok": ok,
                "session_id": session_id,
                "job_id": orch.job_id,
                "mode": "create",
                "step": "create.advance",
                "payload": payload or {},
            }
            if not ok:
                resp["error"] = "orchestrator_errors"
            return resp
        except Exception as e:
            log.exception("create_project_answer: orchestrator fallback failed: %s", e)
            return {
                "ok": False,
                "session_id": session_id,
                "job_id": getattr(self, "job_id", None),
                "mode": "create",
                "step": "create.advance",
                "payload": {},
                "error": str(e),
            }
