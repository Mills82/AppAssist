# aidev/orchestrator.py
from __future__ import annotations

import asyncio
import concurrent.futures as _cf
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import queue as _queue

from .cards import Card, KnowledgeBase
from .io_utils import apply_edits_transactionally
from .llm_client import ChatGPT
from .llm_utils import parse_json_array, parse_json_object
from .runtime import run_preapply_checks
from .state import ProjectState
from .structure import compact_structure, discover_structure

try:
    # v2 repo map builder (preferred)
    from .repo_map import build_project_map as _build_repo_project_map  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency / back-compat
    _build_repo_project_map = None

from . import events as _events
from .orchestration.edit_mixin import OrchestratorEditMixin, _TIMEOUT
from .orchestration.qa_mixin import OrchestratorQAMixin
from .orchestration.analyze_mixin import OrchestratorAnalyzeMixin
from .orchestration.registry import get_job_registry

from .schemas import file_edit_schema

JOBS = get_job_registry()


# ----------------------------- Errors ---------------------------------


class OrchestratorError(Exception):
    """Base class for orchestrator errors."""


class LLMFailure(OrchestratorError):
    """LLM call failed or returned unusable output."""


class IOFailure(OrchestratorError):
    """Filesystem or patching failure."""


# ------------------ Private run index (for cancel) ---------------------


class _RunIndex:
    """
    Process-local index of live Orchestrator instances so HTTP cancel endpoints
    can flip a flag on specific runs.

    NOTE: the true durability / async view of jobs is in orchestration.registry;
    this is just for in-process cancellation.
    """

    _lock = threading.Lock()
    _by_job: Dict[str, "Orchestrator"] = {}
    _by_session: Dict[str, List[str]] = {}

    @classmethod
    def register(cls, orch: "Orchestrator") -> None:
        with cls._lock:
            cls._by_job[orch.job_id] = orch
            if orch._session_id:
                cls._by_session.setdefault(orch._session_id, []).append(orch.job_id)

    @classmethod
    def finish(cls, job_id: str) -> None:
        with cls._lock:
            orch = cls._by_job.pop(job_id, None)
            if orch and orch._session_id:
                jids = cls._by_session.get(orch._session_id, [])
                if job_id in jids:
                    jids.remove(job_id)
                if not jids:
                    try:
                        cls._by_session.pop(orch._session_id, None)
                    except Exception:
                        pass

    @classmethod
    def cancel(cls, job_id: str) -> bool:
        with cls._lock:
            orch = cls._by_job.get(job_id)
            if not orch:
                return False
            orch._mark_cancelled(reason="cancel_api_job")
            return True

    @classmethod
    def cancel_by_session(cls, session_id: str) -> int:
        with cls._lock:
            jids = list(cls._by_session.get(session_id, []))
            for jid in jids:
                orch = cls._by_job.get(jid)
                if orch:
                    orch._mark_cancelled(reason="cancel_api_session")
            return len(jids)


# ----------------------------- Schemas --------------------------------
# Canonical JSON Schemas are loaded via aidev.schemas, which is the single
# source of truth for file edits, targets, and other contracts.

try:
    EDIT_SCHEMA: Dict[str, Any] = file_edit_schema()
except Exception as e:
    logging.warning(
        "Failed to load EDIT_SCHEMA via aidev.schemas; falling back to empty schema: %s",
        e,
    )
    EDIT_SCHEMA = {}


def _index_errors_by_path(errors: Sequence[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a mapping: repo-relative path -> list of error payloads that mention that path.

    Each value entry is a trimmed error record with the most useful fields for
    targeted repairs.
    """
    by_path: Dict[str, List[Dict[str, Any]]] = {}

    for err in errors or []:
        if not isinstance(err, dict):
            continue

        files = err.get("files")
        targets: List[str] = []
        if isinstance(files, (list, tuple)):
            targets = [f for f in files if isinstance(f, str)]
        else:
            # Fallback: some callers may use a single 'path' instead of 'files'
            path = err.get("path")
            if isinstance(path, str):
                targets = [path]

        if not targets:
            continue

        payload = {
            "check_id": err.get("check_id"),
            "tool": err.get("tool"),
            "kind": err.get("kind"),
            "message": err.get("message"),
            "severity": err.get("severity", "error"),
        }

        for rel in targets:
            rel = rel.strip()
            if not rel:
                continue
            by_path.setdefault(rel, []).append(payload)

    return by_path


@dataclass
class ValidationResult:
    ok: bool
    details: Dict[str, Any]

    @property
    def errors(self) -> List[Dict[str, Any]]:
        return list(self.details.get("errors") or [])

    @property
    def errors_by_path(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Per-file view of validation errors, keyed by repo-relative path.

        Prefers a precomputed mapping in details["errors_by_path"], but will
        lazily derive it from self.errors when not present.
        """
        ebp = self.details.get("errors_by_path")
        if isinstance(ebp, dict):
            return ebp
        return _index_errors_by_path(self.errors)


# ----------------------------- Orchestrator ---------------------------


@dataclass
class ConversationTask:
    focus: str = ""
    auto_approve: bool = False
    dry_run: bool = False
    includes: List[str] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)
    approved_rec_ids: Optional[List[str]] = None
    mode: str = "auto"  # "qa", "analyze", "edit"


@dataclass
class Orchestrator(OrchestratorEditMixin, OrchestratorQAMixin, OrchestratorAnalyzeMixin):
    root: Path
    st: ProjectState
    args: Dict[str, object] = field(default_factory=dict)

    # Global auto-approval for this run
    auto_approve: bool = False

    # Summary counters
    _files_created: int = 0
    _files_modified: int = 0
    _files_skipped: int = 0

    # Cleanup prefs
    _cleanup_remove_empty: bool = False
    _cleanup_collapse_blank: bool = False

    # Run-time
    _file_change_summaries: List[Dict[str, object]] = field(default_factory=list, init=False)
    _writes_by_rec: Dict[str, List[str]] = field(default_factory=dict, init=False)
    _apply_results: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _llm: Optional[ChatGPT] = field(default=None, init=False)

    # Aggregated cross-file notes per recommendation
    _cross_file_notes_by_rec: Dict[str, Dict[str, List[str]]] = field(
        default_factory=dict,
        init=False,
    )

    # Deep research (run-scoped, sanitized digest only; no raw file contents)
    _research_brief: Optional[Dict[str, Any]] = field(default=None, init=False)
    _research_brief_cache_key: Optional[str] = field(default=None, init=False)
    _research_brief_meta: Optional[Dict[str, Any]] = field(default=None, init=False)

    # Store the last produced analyze plan (if any) so it can be emitted/surfaced
    _last_analyze_plan: Optional[Dict[str, Any]] = None

    _approval_cb: Optional[Callable[[List[Dict[str, Any]]], bool]] = field(default=None, init=False)
    _progress_cb: Optional[Callable[[str, Dict[str, Any]], Any]] = field(default=None, init=False)

    _llm_max_tokens: Optional[int] = field(default=None, init=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _errors: List[Dict[str, Any]] = field(default_factory=list, init=False)

    # Deterministic per-file ai_summary/card-summarize failures (non-blocking)
    _ai_summary_failures: List[Dict[str, str]] = field(default_factory=list, init=False)

    # Project brief cache (per run)
    _project_brief_text: Optional[str] = field(default=None, init=False)
    _project_brief_hash: Optional[str] = field(default=None, init=False)

    # Stage record-ids for smooth progress bars
    _rid_plan: Optional[str] = field(default=None, init=False)
    _rid_recs: Optional[str] = field(default=None, init=False)
    _rid_targets: Optional[str] = field(default=None, init=False)
    _rid_edits: Optional[str] = field(default=None, init=False)
    _rid_approval: Optional[str] = field(default=None, init=False)
    _rid_checks: Optional[str] = field(default=None, init=False)

    # Timeouts/limits
    _timeout_targets: float = field(default=330.0, init=False)
    _timeout_edit: float = field(default=420.0, init=False)
    _targets_fallback_n: int = field(default=5, init=False)

    # Job identity
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex, init=False)

    # External job registry mapping
    _registry_job_id: Optional[str] = field(default=None, init=False, repr=False)
    _job_finished: bool = field(default=False, init=False, repr=False)

    # ---------------- Lifecycle ----------------

    def __post_init__(self) -> None:
        self.root = self.root.resolve()

        # Cleanups
        if bool(self.args.get("fix_all")):
            self._cleanup_collapse_blank = True

        # Callbacks
        cb = self.args.get("approval_cb")
        if callable(cb):
            self._approval_cb = cb  # type: ignore[assignment]

        pcb = self.args.get("progress_cb")
        if callable(pcb):
            self._progress_cb = pcb  # type: ignore[assignment]

        # LLM config
        cfg = self.args.get("cfg") or {}
        if not isinstance(cfg, dict):
            cfg = {}

        llm_cfg = cfg.get("llm", {}) if isinstance(cfg.get("llm"), dict) else {}
        api_key_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env) or None

        # Base LLM timeout: from config if present, else from env, else a sane default
        timeout = float(llm_cfg.get("timeout_sec") or os.getenv("AIDEV_TIMEOUT_SEC", "1800"))

        model = llm_cfg.get("model")
        base_url = llm_cfg.get("base_url")

        self._llm = ChatGPT(model=model, base_url=base_url, api_key=api_key, timeout=timeout)

        try:
            self._llm_max_tokens = int(llm_cfg.get("max_output_tokens") or 0) or None
        except Exception:
            self._llm_max_tokens = None

        # Fallback to env default if not provided in cfg
        if self._llm_max_tokens is None:
            try:
                env_default = int(os.getenv("AIDEV_DEFAULT_MAX_OUTPUT_TOKENS", "0")) or None
            except Exception:
                env_default = None
            self._llm_max_tokens = env_default

        # Auto-approve
        try:
            self.auto_approve = bool(self.args.get("auto_approve") or False)
        except Exception:
            self.auto_approve = False

        # Timeouts
        def _env_float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, str(default)))
            except Exception:
                return default

        # Env vars are treated as “extra seconds beyond the LLM timeout”.
        targets_delta = _env_float("AIDEV_TIMEOUT_TARGETS", 30.0)  # extra seconds
        edit_delta = _env_float("AIDEV_TIMEOUT_EDIT", 120.0)       # extra seconds

        self._timeout_targets = float(
            self.args.get("llm_timeout_targets") or (timeout + targets_delta)
        )
        self._timeout_edit = float(
            self.args.get("llm_timeout_edit") or (timeout + edit_delta)
        )

        try:
            self._targets_fallback_n = int(
                self.args.get("targets_fallback_n") or os.getenv("AIDEV_TARGETS_FALLBACK_N", "5")
            )
        except Exception:
            self._targets_fallback_n = 5

        # SSE metrics: count only this session
        try:
            def _on_emit(payload: Dict[str, Any]) -> None:
                sid = payload.get("session_id")
                if sid and self._session_id and str(sid) != str(self._session_id):
                    return
                try:
                    incr = getattr(self.st.trace, "accumulate_sse_emitted", None)
                    if callable(incr):
                        incr(1)
                except Exception:
                    pass

            _events.register_emit_observer(_on_emit)
        except Exception:
            pass

        # Optional override job_id (server may pass the external JobRegistry id)
        job_id_arg: Optional[object] = None
        try:
            job_id_arg = self.args.get("job_id")
            if isinstance(job_id_arg, (str, bytes)) and job_id_arg:
                # HTTP/server path: job_id is the external JobRegistry id
                self.job_id = str(job_id_arg)
        except Exception:
            job_id_arg = None

        # Register in process-local run index
        _RunIndex.register(self)

        # Clear any stale cancel flag
        try:
            self._cancel_event.clear()
        except Exception:
            pass

        # Attach to or create external JobRegistry entry
        try:
            if job_id_arg:
                # Server/HTTP path: the job already exists in the registry.
                # Use the same id for all registry updates & approvals.
                self._registry_job_id = self.job_id
            else:
                # CLI/standalone path: create a new registry job for this run.
                job = self._arun(
                    JOBS.create(
                        self._session_id or "session-unknown",
                        kind="orchestrate",
                        meta={"root": str(self.root), "job_id": self.job_id},
                    )
                )
                self._registry_job_id = getattr(job, "id", None) or self.job_id
        except Exception:
            # Fall back to using the local job_id; this keeps approvals
            # consistent even if the registry is unavailable.
            self._registry_job_id = self.job_id

    # ---------------- Convenience ----------------

    @property
    def _session_id(self) -> Optional[str]:
        sid = self.args.get("session_id")
        return str(sid) if isinstance(sid, (str, bytes)) else None

    @property
    def _approval_job_id(self) -> str:
        """
        Job identifier used for ApprovalInbox and HTTP approval routes.

        Prefer the external JobRegistry id (used by /jobs/* APIs), falling
        back to the internal orchestrator job_id if registry creation failed.
        """
        return str(self._registry_job_id or self.job_id)

    # ---------------- Project root safety ----------------

    _ERR_NO_PROJECT_SELECTED = (
        "No project selected: set project_root in session or request; refusing to apply edits "
        "to the AI Dev Bot repo."
    )

    def _aidev_repo_root(self) -> Path:
        """Return the repository root of the running AI Dev Bot code.

        This is used to prevent accidental writes/checks against the bot repo
        when the UI/session did not provide a target project_root.
        """
        try:
            # aidev/orchestrator.py -> aidev/ -> repo root
            return Path(__file__).resolve().parent.parent
        except Exception:
            # Extremely defensive fallback; shouldn't happen.
            return self.root.resolve()

    def _is_aidev_repo_root(self, p: Path) -> bool:
        """True when p resolves to the AI Dev Bot repo root directory."""
        try:
            return Path(p).resolve() == self._aidev_repo_root().resolve()
        except Exception:
            return False

    def _allow_aidev_root_override(self) -> bool:
        """Allow maintainers to explicitly opt-in to running against the bot repo."""
        try:
            v = self.args.get("allow_aidev_root")
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", errors="ignore")
            if isinstance(v, str) and v.strip():
                return v.strip().lower() in {"1", "true", "yes", "y", "on"}
        except Exception:
            pass

        try:
            env = os.getenv("AIDEV_ALLOW_AIDEV_ROOT")
            if env is None:
                return False
            return str(env).strip().lower() in {"1", "true", "yes", "y", "on"}
        except Exception:
            return False

    def _assert_safe_project_root(self, *, where: str, action: str) -> None:
        """Refuse write/apply validations when the root is missing/invalid.

        Requirements enforced:
          - A root must be present.
          - Root must not equal the AI Dev Bot repo root unless override is enabled.

        Always emits a progress_error before raising so UI can surface the issue.
        """
        try:
            root = Path(self.root).resolve()
        except Exception:
            root = self.root

        # Treat missing/unset root as invalid (should not happen in normal construction,
        # but callers may pass placeholders).
        if not root:
            msg = self._ERR_NO_PROJECT_SELECTED
            self._progress_error(where, error=msg, job_id=self.job_id, action=action)
            raise OrchestratorError(msg)

        if not self._allow_aidev_root_override() and self._is_aidev_repo_root(root):
            msg = (
                "Refusing to apply/validate against the AI Dev Bot repository root. "
                "Select a project in the UI or provide a project_root in the request."
            )
            # Emit a clear error event before raising
            self._progress_error(where, error=msg, job_id=self.job_id, action=action)
            raise OrchestratorError(msg)

    def _mark_cancelled(self, *, reason: str = "cancel_api") -> None:
        try:
            self._cancel_event.set()
            _events.status(
                f"Cancelled: {reason}",
                stage="cancel",
                session_id=self._session_id,
                job_id=self.job_id,
            )
            self._job_update(stage="cancel", message=f"Cancelled: {reason}")
        except Exception:
            pass

    # Async bridge runner (safe even if already inside an event loop)
    def _arun(self, coro):
        try:
            asyncio.get_running_loop()
            # Already in a loop: run in a helper thread
            q: _queue.Queue = _queue.Queue(maxsize=1)

            def _runner():
                try:
                    q.put((True, asyncio.run(coro)))
                except BaseException as e:
                    q.put((False, e))

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            ok, val = q.get()
            t.join()
            if ok:
                return val
            raise val  # type: ignore[misc]
        except RuntimeError:
            # No running loop: run directly
            return asyncio.run(coro)

    def _job_update(
        self,
        *,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        progress_pct: Optional[float] = None,
    ) -> None:
        try:
            if not self._registry_job_id:
                return
            self._arun(
                JOBS.update(
                    self._registry_job_id,
                    stage=stage,
                    message=message,
                    progress_pct=progress_pct,
                )
            )
        except Exception:
            pass

    def _finish_job_registry(
        self,
        ok: bool,
        summary: Optional[str] = None,
        artifacts: Optional[List[str]] = None,
    ) -> None:
        if self._job_finished:
            return
        try:
            if self._registry_job_id:
                self._arun(
                    JOBS.finish(
                        self._registry_job_id,
                        ok=ok,
                        summary=summary,
                        artifacts=list(artifacts or []),
                    )
                )
            self._job_finished = True
        except Exception:
            pass

    # ---------------- Deep research helpers ----------------

    def _emit_deep_research_event(self, name: str, payload: Dict[str, Any]) -> None:
        """Emit deep_research.* events via events.py if available; fall back safely.

        This must never raise and must never include raw file contents.
        Prefer first-class helpers in aidev/events.py to ensure stable schemas
        and deterministic ordering (deep_research_start/done/attached_to_payload/cache_hit).
        """
        safe_payload = dict(payload or {})
        # Always include job/session IDs for correlation
        safe_payload.setdefault("session_id", self._session_id)
        safe_payload.setdefault("job_id", self.job_id)

        # Strip any potentially large or raw content keys to be safe
        for bad_key in (
            "files",
            "raw_files",
            "file_contents",
            "contents",
            "blob",
            "artifacts",
            "context",
            "ctx",
            "raw",
            "snippets",
        ):
            if bad_key in safe_payload:
                safe_payload.pop(bad_key, None)

        try:
            # Normalize incoming name (allow either 'start' or 'deep_research.start')
            token = name.split(".", 1)[1] if name.startswith("deep_research.") else name
            token = str(token or "").strip()

            helper_map = {
                "start": "deep_research_start",
                "done": "deep_research_done",
                "attached_to_payload": "deep_research_attached_to_payload",
                "attached": "deep_research_attached_to_payload",
                "cache_hit": "deep_research_cache_hit",
                "cache": "deep_research_cache_hit",
            }

            helper_name = helper_map.get(token) or f"deep_research_{token}"

            fn = getattr(_events, helper_name, None)
            if callable(fn):
                try:
                    # Prefer keyword args so event helpers get structured payload
                    fn(**safe_payload)
                    return
                except TypeError:
                    # Some older helpers may expect positional args; fall through to generic emit
                    pass

            # Otherwise use a generic emitter if present (pass event name and payload)
            generic_emitter = getattr(_events, "emit", None)
            if callable(generic_emitter):
                try:
                    generic_emitter(f"deep_research.{token}", safe_payload, session_id=self._session_id, job_id=self.job_id)
                    return
                except Exception:
                    # swallow and fall back
                    pass

            # Last resort: status event
            try:
                _events.status(
                    f"deep_research.{token}",
                    where=f"deep_research.{token}",
                    **safe_payload,
                )
            except Exception:
                pass
        except Exception:
            # Never fail orchestrator because of event emission
            pass

    def _deep_research_enabled(self) -> bool:
        """Gate deep research based on args/env.

        Disabled by default.
        """
        try:
            v = self.args.get("enable_deep_research")
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", errors="ignore")
            if isinstance(v, str) and v.strip():
                return v.strip().lower() in {"1", "true", "yes", "y", "on"}
        except Exception:
            pass

        try:
            env = os.getenv("AIDEV_ENABLE_DEEP_RESEARCH")
            if env is None:
                return False
            return str(env).strip().lower() in {"1", "true", "yes", "y", "on"}
        except Exception:
            return False

    def _build_deep_research_cache_key(
        self,
        *,
        mode: str,
        focus: str,
        includes: List[str],
        excludes: List[str],
        budget: Optional[object],
    ) -> str:
        """Deterministic cache key derived from non-sensitive run inputs.

        Must not include raw file contents or absolute paths.

        Prefer a canonical compute_cache_key exposed by aidev.orchestration.deep_research_cache
        when available so analyze_mixin and orchestrator share the same key algorithm.
        """
        brief_hash = (self._project_brief_hash or "").strip() or "no-brief"
        inc = sorted([str(x).strip() for x in (includes or []) if str(x).strip()])
        exc = sorted([str(x).strip() for x in (excludes or []) if str(x).strip()])
        focus_s = (focus or "").strip()
        if len(focus_s) > 240:
            focus_s = focus_s[:240]

        # Budget is included only as a scalar string; do not embed large objects.
        budget_s = None
        try:
            if budget is not None:
                budget_s = str(budget)
        except Exception:
            budget_s = None

        # Try to delegate to deep_research_cache.compute_cache_key when present.
        try:
            from .orchestration import deep_research_cache as _dr_cache  # type: ignore

            compute_fn = getattr(_dr_cache, "compute_cache_key", None)
            if callable(compute_fn):
                try:
                    # Prefer keyword-friendly call
                    key = compute_fn(
                        focus=focus_s,
                        includes=inc,
                        excludes=exc,
                        brief_hash=brief_hash,
                        mode=(mode or "").strip().lower(),
                        budget=budget_s,
                    )
                    if isinstance(key, str) and key:
                        return key
                except TypeError:
                    # Fallback to common positional signatures (best-effort)
                    try:
                        key = compute_fn(focus_s, inc, exc, budget_s)
                        if isinstance(key, str) and key:
                            return key
                    except Exception:
                        try:
                            key = compute_fn(focus_s, inc, exc)
                            if isinstance(key, str) and key:
                                return key
                        except Exception:
                            pass
                except Exception:
                    # If compute fails, fall through to local algorithm
                    pass
        except Exception:
            # deep_research_cache not available; fall back
            pass

        base = {
            "v": 1,
            "brief_hash": brief_hash,
            "mode": (mode or "").strip().lower(),
            "focus": focus_s,
            "includes": inc,
            "excludes": exc,
            "budget": budget_s,
        }
        raw = json.dumps(base, ensure_ascii=False, sort_keys=True)
        # Short, stable key
        return "dr_" + uuid.uuid5(uuid.NAMESPACE_URL, raw).hex

    def _sanitize_research_brief(self, obj: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Convert a deep research result into a small digest safe for caching/emitting.

        Returns (digest_or_none, meta) where meta includes counts/truncation.
        """
        meta: Dict[str, Any] = {
            "evidence_items": 0,
            "findings": 0,
            "truncated": False,
        }

        if not isinstance(obj, dict):
            return None, meta

        # Prefer a nested digest if the engine already returns one
        digest: Dict[str, Any] = {}

        # IDs/hashes only
        for k in (
            "id",
            "brief_id",
            "hash",
            "brief_hash",
            "cache_key",
            "query",
            "topic",
            "summary",
            "tl;dr",
            "high_level_summary",
        ):
            if k in obj and isinstance(obj.get(k), (str, int, float)):
                digest[k] = obj.get(k)

        # Findings: store only small textual bullets/titles; avoid large blobs
        findings = obj.get("findings")
        if isinstance(findings, list):
            safe_findings: List[Dict[str, Any]] = []
            for f in findings[:50]:
                if isinstance(f, dict):
                    sf: Dict[str, Any] = {}
                    for kk in ("title", "summary", "claim", "confidence", "source_count"):
                        if kk in f and isinstance(f.get(kk), (str, int, float, bool)):
                            sf[kk] = f.get(kk)
                    if sf:
                        safe_findings.append(sf)
                elif isinstance(f, str):
                    safe_findings.append({"summary": f[:400]})
            if safe_findings:
                digest["findings"] = safe_findings
            meta["findings"] = len(findings)
        elif isinstance(findings, dict):
            # Some engines return a dict of sections
            try:
                meta["findings"] = len(findings)
            except Exception:
                meta["findings"] = 0

        evidence = obj.get("evidence") or obj.get("evidence_items")
        if isinstance(evidence, list):
            # Keep evidence count and only lightweight citation metadata
            safe_evidence: List[Dict[str, Any]] = []
            for ev in evidence[:80]:
                if isinstance(ev, dict):
                    sev: Dict[str, Any] = {}
                    for kk in (
                        "title",
                        "url",
                        "uri",
                        "path",
                        "rel_path",
                        "kind",
                        "type",
                        "line_start",
                        "line_end",
                        "snippet_hash",
                    ):
                        if kk in ev and isinstance(ev.get(kk), (str, int, float, bool)):
                            # NOTE: paths should be repo-relative; keep as-is.
                            sev[kk] = ev.get(kk)
                    if sev:
                        safe_evidence.append(sev)
                elif isinstance(ev, str):
                    safe_evidence.append({"title": ev[:200]})
            if safe_evidence:
                digest["evidence"] = safe_evidence
            meta["evidence_items"] = len(evidence)

        # Truncation hint
        for k in ("truncated", "was_truncated", "is_truncated"):
            if k in obj:
                try:
                    meta["truncated"] = bool(obj.get(k))
                except Exception:
                    pass

        # If engine provided explicit counts, trust them
        counts = obj.get("counts")
        if isinstance(counts, dict):
            try:
                if "evidence_items" in counts:
                    meta["evidence_items"] = int(counts.get("evidence_items") or 0)
                if "findings" in counts:
                    meta["findings"] = int(counts.get("findings") or 0)
            except Exception:
                pass

        # Keep the meta alongside digest
        digest["counts"] = {
            "evidence_items": int(meta.get("evidence_items") or 0),
            "findings": int(meta.get("findings") or 0),
        }
        digest["truncated"] = bool(meta.get("truncated"))

        # Guard: never store raw contents if present
        for bad_key in (
            "files",
            "raw_files",
            "file_contents",
            "contents",
            "blob",
            "artifacts",
            "context",
            "ctx",
            "raw",
            "snippets",
        ):
            if bad_key in digest:
                digest.pop(bad_key, None)

        return digest if digest else None, meta

    def _run_deep_research_if_needed(
        self,
        *,
        kb: KnowledgeBase,
        meta: Dict[str, Any],
        ctx_blob: Any,
        mode: str,
        focus: str,
        includes: List[str],
        excludes: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Decide, run/cache, and attach a deep research digest; emit events.

        This is a no-op when gating is disabled.
        """
        # Only for analyze and edit/auto (recommendations). Never for QA.
        mode_norm = (mode or "").strip().lower()
        if mode_norm not in {"analyze", "auto", "edit"}:
            return None

        if not self._deep_research_enabled():
            return None

        # Ensure we have a brief hash; needed for deterministic cache key.
        if not self._project_brief_hash:
            try:
                _, brief_hash = self._init_project_brief()
                self._project_brief_hash = brief_hash
            except Exception:
                # still proceed with "no-brief" cache key component
                pass

        budget = self.args.get("deep_research_budget")
        cache_key = self._build_deep_research_cache_key(
            mode=mode_norm,
            focus=focus,
            includes=includes,
            excludes=excludes,
            budget=budget,
        )
        self._research_brief_cache_key = cache_key

        # Attempt cache read
        cache_obj = None
        try:
            from .orchestration import deep_research_cache as _dr_cache  # type: ignore

            get_fn = getattr(_dr_cache, "get", None) or getattr(_dr_cache, "load", None)
            if callable(get_fn):
                cache_obj = get_fn(cache_key)
            elif hasattr(_dr_cache, "DeepResearchCache"):
                # Some implementations may provide a cache class
                try:
                    inst = _dr_cache.DeepResearchCache(self.root)
                    cache_obj = getattr(inst, "get", lambda _k: None)(cache_key)
                except Exception:
                    cache_obj = None
        except Exception:
            cache_obj = None

        if cache_obj is not None:
            digest, meta_counts = self._sanitize_research_brief(cache_obj)
            if digest is None:
                # Cache contained something unexpected; treat as miss.
                cache_obj = None
            else:
                self._research_brief = digest
                self._research_brief_meta = {
                    "cache_key": cache_key,
                    "source": "cache",
                    **meta_counts,
                }

                # Emit canonical cache_hit and attachment events via first-class helpers
                self._emit_deep_research_event(
                    "cache_hit",
                    {
                        "mode": mode_norm,
                        "cache_key": cache_key,
                        "evidence_items": int(meta_counts.get("evidence_items") or 0),
                        "findings": int(meta_counts.get("findings") or 0),
                        "truncated": bool(meta_counts.get("truncated")),
                        "brief_hash": self._project_brief_hash or "",
                        "source": "cache",
                    },
                )
                self._emit_deep_research_event(
                    "attached_to_payload",
                    {
                        "mode": mode_norm,
                        "cache_key": cache_key,
                        "evidence_items": int(meta_counts.get("evidence_items") or 0),
                        "findings": int(meta_counts.get("findings") or 0),
                        "truncated": bool(meta_counts.get("truncated")),
                        "attached": True,
                        "source": "cache",
                    },
                )

                return digest

        # Cache miss: run engine
        self._emit_deep_research_event(
            "start",
            {
                "mode": mode_norm,
                "cache_key": cache_key,
                "brief_hash": self._project_brief_hash or "",
            },
        )

        engine_res = None
        try:
            from .orchestration import deep_research_engine as _dr_engine  # type: ignore

            # Try common entry points
            run_fn = (
                getattr(_dr_engine, "run", None)
                or getattr(_dr_engine, "execute", None)
                or getattr(_dr_engine, "search", None)
                or getattr(_dr_engine, "build", None)
            )

            if callable(run_fn):
                # Only pass safe inputs; engine can consult KB for file reads internally.
                try:
                    engine_res = run_fn(
                        kb=kb,
                        focus=focus,
                        budget=budget,
                        includes=list(includes or []),
                        excludes=list(excludes or []),
                        meta=meta,
                    )
                except TypeError:
                    # Back-compat: older signatures
                    try:
                        engine_res = run_fn(kb, focus, budget)
                    except Exception:
                        engine_res = run_fn(kb=kb, focus=focus)
            elif hasattr(_dr_engine, "DeepResearchEngine"):
                try:
                    inst = _dr_engine.DeepResearchEngine(root=self.root, kb=kb)
                    engine_res = getattr(inst, "run")(focus=focus, budget=budget)
                except Exception:
                    engine_res = None
        except Exception as e:
            # Engine is optional; do not break run
            self._emit_deep_research_event(
                "done",
                {
                    "mode": mode_norm,
                    "cache_key": cache_key,
                    "ok": False,
                    "error": str(e),
                },
            )
            return None

        digest, meta_counts = self._sanitize_research_brief(engine_res)
        if digest is None:
            self._emit_deep_research_event(
                "done",
                {
                    "mode": mode_norm,
                    "cache_key": cache_key,
                    "ok": False,
                    "error": "deep_research_engine returned no usable digest",
                },
            )
            return None

        # Persist to cache (best-effort)
        try:
            from .orchestration import deep_research_cache as _dr_cache  # type: ignore

            set_fn = getattr(_dr_cache, "set", None) or getattr(_dr_cache, "save", None)
            if callable(set_fn):
                try:
                    set_fn(cache_key, digest)
                except TypeError:
                    # Some cache APIs are (key, value, ttl)
                    set_fn(cache_key, digest, None)
            elif hasattr(_dr_cache, "DeepResearchCache"):
                try:
                    inst = _dr_cache.DeepResearchCache(self.root)
                    getattr(inst, "set", lambda _k, _v: None)(cache_key, digest)
                except Exception:
                    pass
        except Exception:
            pass

        self._research_brief = digest
        self._research_brief_meta = {
            "cache_key": cache_key,
            "source": "engine",
            **meta_counts,
        }

        self._emit_deep_research_event(
            "done",
            {
                "mode": mode_norm,
                "cache_key": cache_key,
                "ok": True,
                "evidence_items": int(meta_counts.get("evidence_items") or 0),
                "findings": int(meta_counts.get("findings") or 0),
                "truncated": bool(meta_counts.get("truncated")),
            },
        )

        self._emit_deep_research_event(
            "attached_to_payload",
            {
                "mode": mode_norm,
                "cache_key": cache_key,
                "evidence_items": int(meta_counts.get("evidence_items") or 0),
                "findings": int(meta_counts.get("findings") or 0),
                "truncated": bool(meta_counts.get("truncated")),
                "attached": True,
                "source": "engine",
            },
        )

        return digest

    # ---------------- Public API ----------------

    def run(self) -> None:
        try:
            self._progress("start", root=str(self.root), job_id=self.job_id)
            self._job_update(stage="start", message="orchestrate: start", progress_pct=0)

            includes = self._coerce_str_list(self.args.get("include"))
            excludes = self._coerce_str_list(self.args.get("exclude"))
            max_kb = int(self.args.get("max_context_kb") or 1024)
            strip_comments = bool(self.args.get("strip_comments") or False)

            # Discover structure
            try:
                struct, ctx_blob = discover_structure(
                    self.root,
                    includes,
                    excludes,
                    max_total_kb=max_kb,
                    strip_comments=strip_comments,
                )
            except Exception as e:
                self._progress_error(
                    "discover_structure",
                    error=str(e),
                    trace=traceback.format_exc(),
                    job_id=self.job_id,
                )
                raise IOFailure(str(e))

            kb = KnowledgeBase(self.root, struct)
            meta = compact_structure(struct)

            # Enrich meta with a full v2 project_map so downstream stages see ALL files.
            try:
                meta = self._build_and_attach_project_map(
                    kb=kb,
                    meta=meta,
                    out_path=self.root / ".aidev" / "project_map.targets.json",
                    trace_source="initial",
                )
            except Exception as e:
                logging.debug(
                    "Failed to enrich meta with project_map: %s",
                    e,
                    extra=self._log_extra(phase="plan"),
                )

            # logging.info("Structure: %s", meta)
            self.st.trace.write("STRUCTURE", "discover", {"meta": meta})
            self._progress("structure", meta=meta, job_id=self.job_id)

            brief_text, brief_hash = self._init_project_brief()
            self._project_brief_text = brief_text
            self._project_brief_hash = brief_hash
            self._progress(
                "brief_ready",
                hash=brief_hash,
                bytes=len(brief_text.encode("utf-8")),
                job_id=self.job_id,
            )

            # Resolve mode once up front so we can skip focus-card work for Q&A/analyze.
            mode = self._get_mode()

            # Defensive write-root validation (only for apply/edit flows and apply_jsonl).
            # This must happen before any file writes.
            try:
                if mode in {"auto", "edit"} or bool(self.args.get("apply_jsonl")):
                    self._assert_safe_project_root(where="project_root_invalid", action="run")
            except OrchestratorError as e:
                # Ensure a clear terminal result is emitted for UI consumers.
                self._emit_result_and_done(ok=False, summary=str(e))
                return

            if self._should_cancel():
                self._emit_result_and_done(ok=False, summary="Run cancelled before planning.")
                return

            # PLAN (lightweight prep; LLM recs are next stage)
            self._rid_plan = _events.progress_start(
                "plan",
                detail="Preparing plan (brief, structure, cards)…",
                session_id=self._session_id,
                job_id=self.job_id,
            )
            self._job_update(
                stage="plan",
                message="Preparing plan (brief, structure, cards)…",
                progress_pct=5,
            )

            # Card index refresh
            try:
                idx = kb.load_card_index() or {}
                nodes = (idx or {}).get("nodes", []) or []
                changed_only_flag = False if bool(self.args.get("cards_force")) or not nodes else True
                kb.update_cards(
                    force=bool(self.args.get("cards_force")),
                    changed_only=changed_only_flag,
                )

                # Best-effort: surface per-file ai_summary refresh failures in a
                # stable/deterministic order without impacting run success.
                try:
                    self._ai_summary_failures = self._gather_ai_summary_failures(kb)
                    if self._ai_summary_failures:
                        logging.debug(
                            "ai_summary_failures found: %d",
                            len(self._ai_summary_failures),
                            extra=self._log_extra(phase="plan"),
                        )

                    # Normalize/dedupe/sort failures and attach to meta so downstream
                    # stages see a deterministic list regardless of KB internals.
                    try:
                        failures: List[Dict[str, str]] = []
                        seen: set = set()
                        for item in list(self._ai_summary_failures or []):
                            if not isinstance(item, dict):
                                continue
                            p = item.get("path") or item.get("file") or item.get("rel_path")
                            if not p:
                                continue
                            p = str(p).strip()
                            if not p or p in seen:
                                continue
                            seen.add(p)
                            err_msg = item.get("error") or item.get("message") or item.get("err") or ""
                            failures.append({"path": p, "error": str(err_msg)})

                        failures.sort(key=lambda d: d.get("path", ""))
                        self._ai_summary_failures = failures

                        # Attach to meta for downstream consumers
                        try:
                            meta = dict(meta)
                            meta["ai_summary_failures"] = list(self._ai_summary_failures)
                        except Exception:
                            pass

                        # Emit a stable progress event so UIs/tests can observe failures
                        try:
                            self._progress(
                                "ai_summary_failures",
                                ai_summary_failures=list(self._ai_summary_failures),
                                job_id=self.job_id,
                            )
                        except Exception:
                            pass

                    except Exception:
                        # Never fail the run due to failure-normalization problems.
                        logging.debug("Failed to normalize ai_summary_failures", exc_info=True)

                except Exception:
                    # Never fail the run due to failure-report extraction.
                    pass

                try:
                    # Emit a trace event to indicate a heuristic-only card refresh
                    self.st.trace.write("heuristic_card_refresh", "cards", {"changed_only": changed_only_flag})
                except Exception:
                    pass
                self._progress(
                    "cards_indexed",
                    changed_only=changed_only_flag,
                    baseline_exists=bool(nodes),
                    job_id=self.job_id,
                )
            except Exception as e:
                logging.debug("kb.update_cards() failed: %s", e, extra=self._log_extra(phase="plan"))
                self._progress_error(
                    "kb.update_cards",
                    error=str(e),
                    trace=traceback.format_exc(),
                    job_id=self.job_id,
                )

            # Optional export project map
            export_arg = self.args.get("export_project_map")
            if export_arg is not None:
                if isinstance(export_arg, str) and export_arg.strip():
                    out_path = Path(export_arg)
                else:
                    out_path = self.root / ".aidev" / "project_map.json"

                try:
                    meta = self._build_and_attach_project_map(
                        kb=kb,
                        meta=meta,
                        out_path=out_path,
                        trace_source="export",
                    )
                    logging.info(
                        "Project map written: %s",
                        out_path,
                        extra=self._log_extra(phase="project_map"),
                    )
                    self._progress(
                        "project_map_saved",
                        path=str(out_path),
                        job_id=self.job_id,
                    )
                except Exception as e:
                    logging.exception(
                        "Failed to write project map: %s",
                        e,
                        extra=self._log_extra(phase="project_map"),
                    )
                    self._progress_error(
                        "project_map",
                        error=str(e),
                        trace=traceback.format_exc(),
                        job_id=self.job_id,
                    )

                if bool(self.args.get("project_map_only")):
                    _events.progress_finish(
                        "plan",
                        ok=True,
                        recId=self._rid_plan,
                        session_id=self._session_id,
                        job_id=self.job_id,
                    )
                    self._emit_result_and_done(ok=True, summary="Project map exported.")
                    return

            # Raw focus string (user goal) — used directly in recommendations
            focus = (self.args.get("focus") or "").strip()

            _events.progress_finish(
                "plan",
                ok=True,
                recId=self._rid_plan,
                session_id=self._session_id,
                job_id=self.job_id,
            )

            if self._should_cancel():
                self._emit_result_and_done(ok=False, summary="Run cancelled after planning.")
                return

            # Targets-only mode
            if bool(self.args.get("targets_only")):
                try:
                    cfg = (self.args.get("cfg") or {})
                    if not isinstance(cfg, dict):
                        cfg = {}
                    cards_cfg = cfg.get("cards", {}) if isinstance(cfg.get("cards"), dict) else {}
                    top_k = int(
                        self.args.get("cards_top_k")
                        or cards_cfg.get("default_top_k", 20)
                    )

                    hits = kb.select_cards(focus or "project overview", top_k=top_k)
                    self._progress(
                        "targets_only",
                        top=[h[0] for h in hits],
                        job_id=self.job_id,
                    )
                except Exception as e:
                    self._progress_error(
                        "targets_only",
                        error=str(e),
                        trace=traceback.format_exc(),
                        job_id=self.job_id,
                    )
                self._emit_result_and_done(ok=True, summary="Targets-only listing complete.")
                return

            # Apply JSONL edits (bypass LLM planning)
            jsonl_path = self.args.get("apply_jsonl")
            if jsonl_path:
                try:
                    from .llm_io import fetch_code_jsonl  # local import to avoid cycle

                    edits = []
                    for e in fetch_code_jsonl(Path(jsonl_path)):
                        if getattr(e, "patch_unified", None):
                            edits.append(
                                {
                                    "path": e.path,
                                    "patch_unified": e.patch_unified,
                                    "rec_id": e.rec_id or "jsonl",
                                }
                            )
                        else:
                            edits.append(
                                {
                                    "path": e.path,
                                    "content": e.content,
                                    "rec_id": e.rec_id or "jsonl",
                                }
                            )
                    self._progress(
                        "apply_jsonl",
                        edits=len(edits),
                        job_id=self.job_id,
                    )
                    apply_edits_transactionally(
                        self.root,
                        edits,
                        dry_run=bool(self.args.get("dry_run")),
                        stats=self,
                        st=self.st,
                    )

                    # After applying edits, refresh project_map and KB card index so
                    # subsequent operations see the freshly written files.
                    try:
                        # collect changed paths from edits
                        changed_paths = [e.get("path") for e in edits if isinstance(e.get("path"), str)]
                        changed_paths = [p for p in dict.fromkeys(changed_paths) if p]
                        if changed_paths:
                            try:
                                meta = self._build_and_attach_project_map(
                                    kb=kb,
                                    meta=meta,
                                    out_path=self.root / ".aidev" / "project_map.targets.json",
                                    recent_changed_files=changed_paths,
                                    trace_source="apply_jsonl",
                                )
                            except Exception as e:
                                logging.debug(
                                    "Failed to refresh project_map after apply_jsonl: %s",
                                    e,
                                    extra=self._log_extra(phase="apply_jsonl"),
                                )

                            # Reload the KB card index
                            try:
                                kb.load_card_index()
                                try:
                                    self.st.trace.write("heuristic_card_refresh", "cards", {"force": False, "changed_only": True})
                                except Exception:
                                    pass
                            except Exception:
                                try:
                                    kb.update_cards(force=True, changed_only=True)
                                    try:
                                        self.st.trace.write("heuristic_card_refresh", "cards", {"force": True, "changed_only": True})
                                    except Exception:
                                        pass
                                except Exception:
                                    logging.debug("Failed to refresh KB card index after apply_jsonl", extra=self._log_extra(phase="apply_jsonl"))

                            # Emit a progress event so downstream listeners know about recent changes
                            self._progress("post_apply_refresh", recent_changed_files=changed_paths, job_id=self.job_id)
                    except Exception:
                        logging.debug("post-apply refresh failed for jsonl path", extra=self._log_extra(phase="apply_jsonl"))

                    self._emit_result_and_done(
                        ok=True,
                        summary=f"Applied {len(edits)} JSONL edits.",
                    )
                    return
                except Exception as e:
                    self._progress_error(
                        "apply_jsonl",
                        error=str(e),
                        trace=traceback.format_exc(),
                        job_id=self.job_id,
                    )
                    raise IOFailure(str(e))

            # Full staged pipeline or conversational modes
            if focus or not jsonl_path:
                if self._should_cancel():
                    self._emit_result_and_done(
                        ok=False,
                        summary="Run cancelled before LLM pipeline.",
                    )
                    return

                # ---------------- QA MODE (conversational, no edits) ----------------
                if mode == "qa":
                    ok, summary = self._run_qa_pipeline(kb=kb, meta=meta, ctx_blob=ctx_blob)
                    self._emit_result_and_done(ok=ok, summary=summary)
                    return

                # -------------- ANALYZE MODE (multi-file analysis, no edits) -------
                if mode == "analyze":
                    # Ensure deep research events (if enabled) occur before the analyze LLM call.
                    try:
                        self._run_deep_research_if_needed(
                            kb=kb,
                            meta=meta,
                            ctx_blob=ctx_blob,
                            mode=mode,
                            focus=focus,
                            includes=includes,
                            excludes=excludes,
                        )
                    except Exception:
                        # Deep research is best-effort and must never break analyze.
                        logging.debug("deep research pre-analyze failed", exc_info=True)

                    # The analyze mixin may return different shapes:
                    # - (ok: bool, summary: str)
                    # - (ok: bool, summary: str, plan: dict)
                    # - plan: dict (direct)
                    analyze_plan: Optional[Dict[str, Any]] = None
                    ok: bool = False
                    summary: Optional[str] = None

                    try:
                        res = self._run_analyze_mode(
                            kb=kb,
                            meta=meta,
                            ctx_blob=ctx_blob,
                        )
                    except Exception as e:
                        # Ensure we emit a progress error and finish gracefully
                        self._progress_error("analyze", error=str(e), trace=traceback.format_exc(), job_id=self.job_id)
                        # propagate as non-ok result
                        self._emit_result_and_done(ok=False, summary=f"Analyze failed: {e}")
                        return None

                    # Normalize the various possible return shapes
                    try:
                        if isinstance(res, tuple) or isinstance(res, list):
                            if len(res) >= 3 and isinstance(res[2], dict):
                                ok = bool(res[0])
                                summary = str(res[1]) if res[1] is not None else None
                                analyze_plan = res[2]
                            elif len(res) >= 2:
                                ok = bool(res[0])
                                summary = str(res[1]) if res[1] is not None else None
                            else:
                                # Unexpected tuple shape; best-effort
                                try:
                                    ok, summary = res
                                except Exception:
                                    ok = True
                                    summary = "Analyze completed"
                        elif isinstance(res, dict):
                            # direct plan dict returned
                            analyze_plan = res
                            ok = True
                            summary = summary or "Analyze plan produced"
                        elif isinstance(res, bool):
                            ok = res
                            summary = summary or ("Analyze completed" if ok else "Analyze failed")
                        else:
                            # Fallback: try to unpack
                            try:
                                ok, summary = res
                            except Exception:
                                ok = True
                                summary = "Analyze completed"
                    except Exception:
                        ok = True
                        summary = "Analyze completed (normalization fallback)"

                    # If we have a plan, persist and emit an analyze_plan SSE event so API/UI can observe it
                    if isinstance(analyze_plan, dict) and analyze_plan:
                        self._last_analyze_plan = analyze_plan
                        try:
                            # First, try the generic emit API which tests may patch/mock
                            try:
                                generic_emitter = getattr(_events, "emit", None)
                                if callable(generic_emitter):
                                    try:
                                        generic_emitter("analyze_plan", analyze_plan, session_id=self._session_id, job_id=self.job_id)
                                    except Exception:
                                        # swallow emitter errors; fall back to other emitters below
                                        pass
                            except Exception:
                                pass

                            # Back-compat: call a potential analyze_plan function if present
                            emitter = getattr(_events, "analyze_plan", None)
                            if callable(emitter):
                                try:
                                    emitter(plan=analyze_plan, session_id=self._session_id, job_id=self.job_id)
                                except TypeError:
                                    # Some emitters may expect positional args or different signature
                                    try:
                                        emitter(analyze_plan, self._session_id, self.job_id)
                                    except Exception:
                                        # final fallback to status
                                        _events.status(
                                            "Analyze plan ready",
                                            where="analyze_plan",
                                            analyze_plan=analyze_plan,
                                            session_id=self._session_id,
                                            job_id=self.job_id,
                                        )
                            else:
                                _events.status(
                                    "Analyze plan ready",
                                    where="analyze_plan",
                                    analyze_plan=analyze_plan,
                                    session_id=self._session_id,
                                    job_id=self.job_id,
                                )
                            # Also emit a progress event for listeners/tests
                            try:
                                self._progress("analyze_plan_ready", plan_keys=list(analyze_plan.keys()), job_id=self.job_id)
                            except Exception:
                                pass
                        except Exception:
                            logging.debug("Failed to emit analyze_plan event", exc_info=True)

                    # Emit final result & return the plan to the caller (if any)
                    self._emit_result_and_done(ok=bool(ok), summary=summary)
                    return analyze_plan

                # -------------- EDIT / AUTO MODE (full recommendations pipeline) ----
                # Ensure deep research events (if enabled) occur before any downstream LLM calls.
                try:
                    self._run_deep_research_if_needed(
                        kb=kb,
                        meta=meta,
                        ctx_blob=ctx_blob,
                        mode=mode,
                        focus=focus,
                        includes=includes,
                        excludes=excludes,
                    )
                except Exception:
                    logging.debug("deep research pre-recommendations failed", exc_info=True)

                self._progress("pipeline", stage="recommendations", job_id=self.job_id)
                self._job_update(
                    stage="recommendations",
                    message="Generating recommendations…",
                    progress_pct=10,
                )
                try:
                    self._run_llm_pipeline(
                        kb=kb,
                        meta=meta,
                        ctx_blob=ctx_blob,
                        includes=includes,
                        excludes=excludes,
                        focus=focus,
                    )
                except OrchestratorError:
                    pass
                except Exception as e:
                    self._progress_error(
                        "llm_pipeline",
                        error=str(e),
                        trace=traceback.format_exc(),
                        job_id=self.job_id,
                    )
                    self._emit_result_and_done(ok=False, summary=f"LLM pipeline failed: {e}")
                    return

                # If any apply results were produced by the pipeline, ensure the
                # KB/card index and project_map are refreshed before subsequent
                # recommendation steps so the LLM can reference newly changed files.
                try:
                    changed_paths: List[str] = []
                    for r in getattr(self, "_apply_results", []) or []:
                        if isinstance(r, dict):
                            cp = r.get("changed_paths") or []
                            try:
                                changed_paths.extend([p for p in cp if isinstance(p, str)])
                            except Exception:
                                pass
                        else:
                            cp = getattr(r, "changed_paths", None)
                            if cp:
                                try:
                                    changed_paths.extend([p for p in list(cp) if isinstance(p, str)])
                                except Exception:
                                    pass
                    # dedupe while preserving order
                    if changed_paths:
                        changed_paths = [p for p in dict.fromkeys(changed_paths) if p]
                        try:
                            meta = self._build_and_attach_project_map(
                                kb=kb,
                                meta=meta,
                                out_path=self.root / ".aidev" / "project_map.targets.json",
                                recent_changed_files=changed_paths,
                                trace_source="apply",
                            )
                            # Also surface the changed paths in meta for other consumers.
                            meta = dict(meta)
                            meta["recent_changed_files"] = changed_paths
                            self._progress(
                                "post_apply_refresh",
                                recent_changed_files=changed_paths,
                                job_id=self.job_id,
                            )
                        except Exception:
                            logging.debug(
                                "post-apply refresh failed",
                                extra=self._log_extra(phase="apply"),
                            )

                            try:
                                kb.load_card_index()
                                try:
                                    self.st.trace.write("heuristic_card_refresh", "cards", {"force": False, "changed_only": True})
                                except Exception:
                                    pass
                            except Exception:
                                try:
                                    kb.update_cards(force=True, changed_only=True)
                                    try:
                                        self.st.trace.write("heuristic_card_refresh", "cards", {"force": True, "changed_only": True})
                                    except Exception:
                                        pass
                                except Exception:
                                    logging.debug("Failed to refresh KB card index after apply", extra=self._log_extra(phase="apply"))

                            # Surface the changed paths as progress/state so the
                            # next recommendation generation can include them in
                            # its planner/LLM payload (this at least leaves a
                            # trace and makes files discoverable in meta).
                            meta = dict(meta)
                            meta["recent_changed_files"] = changed_paths
                            self._progress("post_apply_refresh", recent_changed_files=changed_paths, job_id=self.job_id)
                except Exception:
                    logging.debug("post-apply collection failed", extra=self._log_extra(phase="apply"))

            self._emit_result_and_done(ok=(len(self._errors) == 0))
        finally:
            try:
                self._print_summary_and_trace()
            finally:
                _RunIndex.finish(self.job_id)
                if not self._job_finished:
                    self._finish_job_registry(ok=(len(self._errors) == 0))

    def run_conversation_task(self, task: ConversationTask) -> None:
        self.args["focus"] = task.focus or (self.args.get("focus") or "")
        if task.includes:
            self.args["include"] = list(task.includes)
        if task.excludes:
            self.args["exclude"] = list(task.excludes)
        if task.dry_run:
            self.args["dry_run"] = True
        if task.auto_approve:
            self.args["auto_approve"] = True
            self.auto_approve = True
            self._approval_cb = lambda _proposed: True
        if task.approved_rec_ids:
            self.args["approved_rec_ids"] = list(task.approved_rec_ids)
        self.args["mode"] = task.mode
        self.run()

    # ---------- Public preapply_checks for /checks/preapply ----------

    def preapply_checks(self, patches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Public API used by the /checks/preapply endpoint.

        `patches` is an array like:
          [
            { "path": "relative/file.py", "diff": "<unified diff>" },
            { "path": "relative/file.py", "content": "<full file text>" },
            ...
          ]

        When both `content` and a diff are present, we prefer `content` so
        checks run against the exact post-edit file text.
        """
        # Defensive root validation: preapply checks create a temporary workspace
        # and must not run against the AI Dev Bot repo root unless explicitly allowed.
        try:
            self._assert_safe_project_root(where="project_root_invalid", action="preapply_checks")
        except OrchestratorError as e:
            # Emit a terminal result so API/UI can show the refusal.
            self._emit_result_and_done(ok=False, summary=str(e))
            raise

        self._rid_checks = _events.progress_start(
            "preapply_checks",
            detail="Running build/test checks in a temporary workspace…",
            session_id=self._session_id,
            job_id=self.job_id,
        )
        _events.checks_started(
            total=None,
            session_id=self._session_id,
            job_id=self.job_id,
        )
        self._job_update(
            stage="preapply_checks",
            message="Running preapply checks…",
            progress_pct=60,
        )

        edits: List[Dict[str, Any]] = []
        for p in (patches or []):
            try:
                rel = (p.get("path") or "").strip()
                # Prefer full content if present (including preview_content)
                content = p.get("content") or p.get("preview_content")
                # Fallback: any diff/patch we were given
                diff = (
                    p.get("diff")
                    or p.get("patch_unified")
                    or p.get("patch")
                    or ""
                )

                if rel and isinstance(content, str) and content.strip():
                    edits.append(
                        {
                            "path": rel,
                            "content": content,
                            "rec_id": p.get("rec_id", "preapply"),
                        }
                    )
                elif rel and isinstance(diff, str) and diff.strip():
                    edits.append(
                        {
                            "path": rel,
                            "patch_unified": str(diff),
                            "rec_id": p.get("rec_id", "preapply"),
                        }
                    )
            except Exception:
                # Be defensive: ignore malformed entries instead of failing
                continue

        ok, details = run_preapply_checks(Path(self.root), edits)

        _events.progress_finish(
            "preapply_checks",
            ok=ok,
            recId=self._rid_checks,
            ok_flag=ok,
            session_id=self._session_id,
            job_id=self.job_id,
        )
        _events.checks_result(
            ok=ok,
            results=details,
            session_id=self._session_id,
            job_id=self.job_id,
        )
        self._job_update(
            stage="preapply_checks",
            message="Preapply checks finished",
            progress_pct=65,
        )

        return {"ok": bool(ok), "details": details}

    def validate_proposed_edits(
        self,
        rec_id: str,
        proposed: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Run pre-apply validation for a single recommendation's proposed edits.

        - Writes projected file contents into a temp workspace via run_preapply_checks
        - Runs cheap validators (syntax / lint / basic tests) as configured
        - Emits SSE events for UI (status + results)

        Returns ValidationResult(ok=..., details=...).

        The returned details are normalized to a dict with at least an
        'errors' list, to support per-file feedback extraction.
        """
        # Defensive root validation: validation creates a temporary workspace
        # and must not run against the AI Dev Bot repo root unless explicitly allowed.
        try:
            self._assert_safe_project_root(where="project_root_invalid", action="validate_proposed_edits")
        except OrchestratorError as e:
            self._emit_result_and_done(ok=False, summary=str(e))
            raise

        preapply_edits: List[Dict[str, Any]] = []

        for p in proposed or []:
            rel = (p.get("path") or "").strip()
            if not rel:
                continue

            # Prefer the final preview text if we have it; this is the most
            # faithful representation of what will be written to disk.
            preview = p.get("preview_content") or p.get("content")

            # Fallback to any diff/patch if no preview is available
            diff_text = (
                p.get("diff")
                or p.get("patch_unified")
                or p.get("patch")
                or ""
            )

            if isinstance(preview, str) and preview.strip():
                preapply_edits.append(
                    {
                        "path": rel,
                        "content": preview,
                        "rec_id": rec_id,
                    }
                )
            elif isinstance(diff_text, str) and diff_text.strip():
                preapply_edits.append(
                    {
                        "path": rel,
                        "patch_unified": diff_text,
                        "rec_id": rec_id,
                    }
                )

        if not preapply_edits:
            self._progress(
                "preapply_checks_skipped",
                rec_id=rec_id,
                reason="no_edits",
                job_id=self.job_id,
            )
            return ValidationResult(
                ok=True,
                details={"skipped": True, "reason": "no_edits", "errors": []},
            )

        self._rid_checks = _events.progress_start(
            "preapply_checks",
            detail=(
                f"Running pre-apply checks for recommendation {rec_id} "
                "in a temporary workspace…"
            ),
            session_id=self._session_id,
            job_id=self.job_id,
        )
        _events.checks_started(
            total=None,
            session_id=self._session_id,
            job_id=self.job_id,
        )

        ok, details = run_preapply_checks(Path(self.root), preapply_edits)

        _events.progress_finish(
            "preapply_checks",
            ok=bool(ok),
            recId=self._rid_checks,
            ok_flag=bool(ok),
            session_id=self._session_id,
            job_id=self.job_id,
        )
        _events.checks_result(
            ok=bool(ok),
            results=details,
            session_id=self._session_id,
            job_id=self.job_id,
        )

        # Normalize details into a dict with at least an 'errors' list so the
        # feedback helpers have a predictable shape.
        if isinstance(details, dict):
            normalized = dict(details)
            # Be generous: allow existing "errors" to be a single value or list.
            errs = normalized.get("errors")
            if isinstance(errs, list):
                pass
            elif errs is None:
                normalized["errors"] = []
            else:
                normalized["errors"] = [errs]
        elif isinstance(details, list):
            normalized = {"errors": list(details)}
        elif details is None:
            normalized = {"errors": []}
        else:
            normalized = {"errors": [details]}

        # -------- classify failures: blocking vs soft/advisory --------
        blocking_tools = {
            "pytest",
            "unittest",
            "npm_test",
            "yarn_test",
            "dart_test",
            "go_test",
            "mypy",
            "tsc",
            "build",
            "compile",
        }
        soft_tools = {
            "black",
            "isort",
            "prettier",
            "gofmt",
            "ruff_format",
            "eslint_format",
        }

        has_blocking = False
        has_soft = False

        def _mark_tool(tool: str) -> None:
            nonlocal has_blocking, has_soft
            t = (tool or "").lower()
            if not t:
                return
            # Very simple heuristic: anything clearly test/build/compile is blocking.
            if t in blocking_tools or "test" in t or "build" in t or "compile" in t:
                has_blocking = True
            elif t in soft_tools or "format" in t or "formatter" in t:
                has_soft = True

        # Look at normalized["errors"] for tool/check_id hints.
        for e in normalized.get("errors") or []:
            if isinstance(e, dict):
                tool = e.get("check_id") or e.get("tool") or ""
                _mark_tool(tool)

        # Also consider a structured "checks" section if present (run_preapply_checks
        # can put per-tool results here).
        for chk in normalized.get("checks") or []:
            if not isinstance(chk, dict) or chk.get("ok"):
                continue
            tool = chk.get("tool") or chk.get("id") or chk.get("label") or ""
            _mark_tool(tool)

        normalized["has_blocking"] = has_blocking
        normalized["has_soft"] = has_soft
        if has_blocking:
            normalized["severity"] = "blocking"
        elif has_soft:
            normalized["severity"] = "soft_only"
        else:
            normalized["severity"] = "unknown"

        # Optionally soften purely "soft" failures when AIDEV_PREAPPLY_MODE!=strict.
        mode = os.getenv("AIDEV_PREAPPLY_MODE", "strict").strip().lower()
        if not ok and mode != "strict" and not has_blocking:
            # Treat purely-soft failures as advisory: keep warnings in details,
            # but do not block the recommendation.
            ok = True

        # Build a per-file error index for targeted repairs.
        try:
            normalized["errors_by_path"] = _index_errors_by_path(
                normalized.get("errors") or []
            )
        except Exception:
            # Fail open: if indexing fails for some reason, just omit it.
            normalized.setdefault("errors_by_path", {})

        if not ok:
            self._progress_error(
                "preapply_checks",
                rec_id=rec_id,
                reason="preapply_failed",
                details=normalized,
                job_id=self.job_id,
            )
        elif not has_blocking and has_soft:
            # Soft failure only (formatting, etc.) – log as a warning for the UI.
            _events.status(
                "Pre-apply checks reported only soft/style issues; continuing.",
                stage="checks",
                session_id=self._session_id,
                job_id=self.job_id,
                rec_id=rec_id,
                severity="soft_only",
            )

        return ValidationResult(ok=bool(ok), details=normalized)

    # ---------------- Cancellation API (server calls) ----------------

    @classmethod
    def cancel_current_apply(
        cls,
        session_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> None:
        """
        Back-compat entry point used by server. Cancels either a specific job_id
        or all jobs associated with a session_id.
        """
        if job_id:
            _RunIndex.cancel(job_id)
            return
        if session_id:
            _RunIndex.cancel_by_session(session_id)
            return

    # ---------------- Internals ----------------

    def _should_cancel(self) -> bool:
        try:
            return self._cancel_event.is_set()
        except Exception:
            return False

    def _phase_max_tokens(self, phase: str) -> Optional[int]:
        """
        Optional per-phase output cap from env, falling back to global default.

        - recommendations / plan phase uses AIDEV_PLAN_MAX_OUTPUT_TOKENS if present
        """
        try:
            if phase in {"recommendations", "plan"}:
                v = os.getenv("AIDEV_PLAN_MAX_OUTPUT_TOKENS")
                if v is not None:
                    n = int(v)
                    if n > 0:
                        return n
        except Exception:
            pass
        return self._llm_max_tokens

    def _get_mode(self) -> str:
        """
        Resolve run mode from args.

        Supported:
          - "auto" (default): full recommendations + edits pipeline.
          - "edit": same as auto (explicit).
          - "qa": conversational Q&A about the project, no edits.
          - "analyze": multi-file analysis + suggestions, no edits.
        """
        raw = self.args.get("mode")
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8", errors="ignore")
            except Exception:
                raw = str(raw)
        if isinstance(raw, str):
            mode = raw.strip().lower()
        else:
            mode = "auto"
        if mode not in {"auto", "edit", "qa", "analyze"}:
            mode = "auto"
        return mode

    # ---------- LLM wrappers with brief injection ----------

    def _chat_json(
        self,
        system_text: str,
        user_payload: Any,
        *,
        schema: Optional[Dict[str, Any]],
        temperature: float = 0.0,
        phase: Optional[str] = None,
        inject_brief: bool = True,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        if not self._llm:
            raise RuntimeError("LLM client is not initialized")

        phase_label = phase or "llm"
        t0 = time.time()

        try:
            combined_system = (
                self._combine_with_brief(system_text) if inject_brief else (system_text or "")
            )
            mt = max_tokens if max_tokens is not None else self._llm_max_tokens

            # Local request id only for your own logs / trace correlation
            request_id = f"aidev-{phase_label}-{uuid.uuid4().hex[:8]}"

            # -------------------------
            # Collect extra metadata first
            # -------------------------
            extra = dict(kwargs.pop("extra", {}) or {})
            for key in ("phase", "llm_phase", "job_id", "rec_id", "service_tier"):
                if key in kwargs and key not in extra:
                    extra[key] = kwargs.pop(key)

            # -------------------------
            # Long-call hint + heuristic
            # -------------------------
            long_call_hint: bool = False
            total_chars = 0
            threshold_chars = 0

            # Check for explicit hint from caller (e.g. self_review forcing normal-call)
            explicit_hint = extra.get("long_call_hint", None)

            if isinstance(explicit_hint, bool):
                # Respect explicit True/False and log accordingly.
                long_call_hint = explicit_hint
                if explicit_hint:
                    logging.info(
                        "[chat_json] using long-call mode (forced by hint; phase=%s)",
                        phase_label,
                        extra=self._log_extra(phase=phase_label),
                    )
                else:
                    logging.info(
                        "[chat_json] using normal-call mode (forced by hint; phase=%s)",
                        phase_label,
                        extra=self._log_extra(phase=phase_label),
                    )
            else:
                # No explicit boolean hint -> use size heuristic
                try:
                    threshold_chars = int(
                        getattr(self._llm, "longcall_char_threshold", 0) or 0
                    )
                except Exception:
                    threshold_chars = 0

                if threshold_chars > 0:
                    try:
                        # Safely JSON-encode the payload for sizing; use the same
                        # jsonability guard rails as other helpers.
                        try:
                            jsonable = self._to_jsonable(user_payload)
                            payload_str = json.dumps(jsonable, ensure_ascii=False)
                        except Exception:
                            payload_str = str(user_payload)

                        total_chars = len(combined_system or "") + len(payload_str)

                        if total_chars >= threshold_chars:
                            long_call_hint = True

                            # Emit a status/progress event so the UI can show
                            # "this might take a while" messaging.
                            self._progress(
                                "llm_long_call",
                                stage="llm_long_call",
                                phase=phase_label,
                                chars=total_chars,
                                threshold=threshold_chars,
                                job_id=self.job_id,
                            )
                            logging.info(
                                "[chat_json] using long-call mode (chars=%d > threshold=%d phase=%s)",
                                total_chars,
                                threshold_chars,
                                phase_label,
                                extra=self._log_extra(phase=phase_label),
                            )
                        else:
                            logging.info(
                                "[chat_json] using normal-call mode (chars=%d < threshold=%d phase=%s)",
                                total_chars,
                                threshold_chars,
                                phase_label,
                                extra=self._log_extra(phase=phase_label),
                            )

                    except Exception:
                        # Heuristic must never break the call; just skip the hint.
                        long_call_hint = False

            # Make sure the inferred hint is present for the LLM client if caller
            # did not explicitly specify one.
            if "long_call_hint" not in extra and long_call_hint:
                extra["long_call_hint"] = True

            call_kwargs = dict(kwargs)
            if extra:
                call_kwargs["extra"] = extra

            # Forward a meaningful stage label to the llm client so it can
            # select an environment-configured model for that logical stage.
            # Be conservative: only forward when it's not the generic 'llm'.
            try:
                if phase_label and phase_label != "llm" and "stage" not in call_kwargs:
                    call_kwargs["stage"] = phase_label
            except Exception:
                pass

            data, res = self._llm.chat_json(
                [{"role": "user", "content": user_payload}],
                schema=schema,
                temperature=temperature,
                system=combined_system,
                max_tokens=mt,
                **call_kwargs,
            )

            # Measure latency and usage once we have a response
            dt_ms = int((time.time() - t0) * 1000)
            tin, tout = self._extract_token_usage(res)

            # Pull Response API metadata if present
            resp_id = getattr(res, "id", None) or getattr(res, "response_id", None)
            status = getattr(res, "status", None)
            model = getattr(res, "model", None)

            # High-level success log
            logging.info(
                "[orchestrator.chat_json] success phase=%s request_id=%s response_id=%s "
                "status=%s model=%s latency_ms=%d tokens_in=%s tokens_out=%s",
                phase_label,
                request_id,
                resp_id,
                status,
                model,
                dt_ms,
                tin,
                tout,
                extra=self._log_extra(phase=phase_label),
            )

            # Truncation warning
            try:
                if mt and tout and tout >= int(mt) - 8:
                    self._progress(
                        "warn_truncation",
                        where=phase_label,
                        message="LLM likely truncated by max_tokens; consider increasing.",
                        job_id=self.job_id,
                    )
            except Exception:
                pass

            # Trace entry
            try:
                self.st.trace.write_llm(
                    event="LLM_JSON",
                    model=model or getattr(self._llm, "model", None),
                    tokens_in=tin,
                    tokens_out=tout,
                    latency_ms=dt_ms,
                    phase=phase_label,
                    kind="chat_json",
                )
            except Exception:
                pass

            return data, res

        except Exception as e:
            extra = self._log_extra(phase=phase_label)

            logging.error(
                "[chat_json] exception phase=%s type=%s msg=%s",
                phase_label,
                type(e).__name__,
                str(e),
                extra=extra,
            )

            # Try to expose HTTP status + body when present (OpenAI v1 httpx errors)
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            resp = getattr(e, "response", None)

            if status is not None:
                logging.error("[chat_json] HTTP status=%s", status, extra=extra)

            if resp is not None:
                try:
                    body_text = None
                    if hasattr(resp, "text"):
                        body_text = resp.text if isinstance(resp.text, str) else str(resp.text)
                    elif hasattr(resp, "content"):
                        body_text = str(resp.content)

                    if body_text:
                        logging.error(
                            "[chat_json] HTTP body (truncated): %s",
                            body_text[:1000],
                            extra=extra,
                        )
                except Exception:
                    logging.debug(
                        "[chat_json] failed to log response body",
                        exc_info=True,
                        extra=extra,
                    )

            # Push error into your SSE + trace
            self._progress_error(
                "chat_json",
                error=str(e),
                system_excerpt=system_text[:160],
                trace=traceback.format_exc(),
                job_id=self.job_id,
            )

            # Keep your rec_id/reasoning-aware debug metadata
            rec_id = kwargs.get("rec_id")
            reasoning_meta = kwargs.get("reasoning") or kwargs.get("reason")

            # STRICT RULE:
            # If this stage is schema-bound, do NOT fall back to unstructured chat+parse.
            if schema is not None:
                logging.error(
                    "chat_json failed (%s); schema provided -> refusing unstructured fallback; raising.",
                    e,
                    extra=self._log_extra(
                        phase=phase_label,
                        rec_id=str(rec_id) if rec_id is not None else None,
                        reasoning=reasoning_meta,
                    ),
                )
                raise

            logging.debug(
                "chat_json failed (%s); no schema provided -> falling back to parse.",
                e,
                extra=self._log_extra(
                    phase=phase_label,
                    rec_id=str(rec_id) if rec_id is not None else None,
                    reasoning=reasoning_meta,
                ),
            )

            # Fallback path (ONLY when schema is None): call _chat() and try to salvage JSON
            text = self._chat(
                system_text,
                user_payload,
                phase=phase,
                inject_brief=inject_brief,
                max_tokens=max_tokens,
            )

            arr = parse_json_array(text)
            obj = parse_json_object(text)

            if obj is not None:
                return obj, None
            if arr is not None:
                return arr, None
            return {}, None

    def _chat(
        self,
        system_text: str,
        user_payload: Any,
        *,
        phase: Optional[str] = None,
        inject_brief: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self._llm:
            raise RuntimeError("LLM client is not initialized")

        phase_label = phase or "llm"
        t0 = time.time()

        try:
            mt = max_tokens if max_tokens is not None else self._llm_max_tokens

            # Safely JSON-encode the user payload; if it contains non-serializable
            # objects (e.g. builtin_function_or_method), fall back to stringifying.
            try:
                user_content = json.dumps(user_payload, ensure_ascii=False)
            except TypeError:
                logging.debug(
                    "user_payload not JSON-serializable in _chat; coercing types",
                    exc_info=True,
                    extra=self._log_extra(phase=phase_label),
                )

                def _fallback(o: Any):
                    try:
                        return str(o)
                    except Exception:
                        return f"<<unserializable:{type(o).__name__}>>"

                user_content = json.dumps(
                    user_payload,
                    ensure_ascii=False,
                    default=_fallback,
                )

            # Build kwargs for the underlying llm client and forward a meaningful
            # stage label when available so the client can resolve a stage-specific model.
            chat_kwargs: Dict[str, Any] = {"max_tokens": mt}
            try:
                if phase_label and phase_label != "llm":
                    chat_kwargs["stage"] = phase_label
            except Exception:
                pass

            out = self._llm.chat(
                [
                    {
                        "role": "system",
                        "content": self._combine_with_brief(system_text)
                        if inject_brief
                        else (system_text or ""),
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                **chat_kwargs,
            )

            dt_ms = int((time.time() - t0) * 1000)
            tin, tout = self._extract_token_usage(out)

            resp_id = getattr(out, "id", None) or getattr(out, "response_id", None)
            status = getattr(out, "status", None)
            model = getattr(out, "model", None)

            logging.info(
                "[chat] success phase=%s response_id=%s status=%s model=%s "
                "latency_ms=%d tokens_in=%s tokens_out=%s",
                phase_label,
                resp_id,
                status,
                model,
                dt_ms,
                tin,
                tout,
                extra=self._log_extra(phase=phase_label),
            )

            try:
                if mt and tout and tout >= int(mt) - 8:
                    self._progress(
                        "warn_truncation",
                        where=phase_label,
                        message="LLM likely truncated by max_tokens; consider increasing.",
                        job_id=self.job_id,
                    )
            except Exception:
                pass

            try:
                self.st.trace.write_llm(
                    event="LLM_TEXT",
                    model=model or getattr(self._llm, "model", None),
                    tokens_in=tin,
                    tokens_out=tout,
                    latency_ms=dt_ms,
                    phase=phase_label,
                    kind="chat",
                )
            except Exception:
                pass

            text = getattr(out, "text", None)
            if isinstance(text, str):
                return text
            if isinstance(out, dict) and "content" in out:
                return str(out["content"])
            return str(out)

        except Exception as e:
            self._progress_error(
                "chat",
                error=str(e),
                system_excerpt=system_text[:160],
                trace=traceback.format_exc(),
                job_id=self.job_id,
            )

            logging.exception(
                "LLM chat failed: %s",
                e,
                extra=self._log_extra(phase=phase_label),
            )
            return "[]"

    # ---------- PROJECT BRIEF helpers ----------

    def _init_project_brief(self) -> Tuple[str, str]:
        """
        Initialize the per-run PROJECT_BRIEF and its hash.

        Delegates to aidev.context.brief.get_or_build, which returns:
          { "text": <markdown>, "hash": <stable_hash>, ... }
        """
        from .context.brief import get_or_build

        force = bool(self.args.get("brief_refresh") or os.getenv("AIDEV_BRIEF_REFRESH"))
        ttl_arg = self.args.get("brief_ttl_hours")
        ttl_env = os.getenv("AIDEV_BRIEF_TTL_HOURS")
        ttl_raw = ttl_arg if ttl_arg is not None else ttl_env
        try:
            ttl_hours = float(ttl_raw) if ttl_raw is not None else None
        except Exception:
            ttl_hours = None

        try:
            model_name = getattr(self._llm, "model", None)
            res = get_or_build(
                self.root,
                model=model_name,
                force=force,
                ttl_hours=ttl_hours,
            )
            brief_text = (res or {}).get("text", "") if isinstance(res, dict) else ""
            brief_hash = (res or {}).get("hash", "") if isinstance(res, dict) else ""
            if not brief_text.strip() or not brief_hash:
                raise RuntimeError("Empty brief text or hash from brief.get_or_build()")
        except Exception as e:
            logging.debug("brief cache build failed; using fallback: %s", e, extra=self._log_extra(phase="brief"))
            brief_text = self._fallback_brief_from_files() or (
                "AI Dev Bot brief: (fallback) No project docs found."
            )
            brief_hash = "fallback"

        try:
            self.st.trace.write(
                "BRIEF",
                "cache",
                {
                    "hash": brief_hash,
                    "bytes": len(brief_text.encode("utf-8")),
                    "force": force,
                    "ttl_hours": ttl_hours,
                },
            )
        except Exception:
            pass

        return brief_text, brief_hash

    def _combine_with_brief(self, system_text: str) -> str:
        """
        Combine the given system prompt with the current PROJECT_BRIEF.
        """
        base = (system_text or "").strip()
        brief = self._project_brief_text
        if not brief:
            try:
                brief, brief_hash = self._init_project_brief()
                self._project_brief_text = brief
                self._project_brief_hash = brief_hash
            except Exception:
                brief = ""

        brief = (brief or "").strip()
        if not brief:
            return base

        try:
            max_bytes = int(os.getenv("AIDEV_BRIEF_MAX_BYTES", "12000"))
        except Exception:
            max_bytes = 12000

        if max_bytes > 0:
            try:
                enc = brief.encode("utf-8", errors="ignore")
                if len(enc) > max_bytes:
                    brief = enc[-max_bytes:].decode("utf-8", errors="ignore")
            except Exception:
                pass

        if base:
            return (
                f"{base}\n\n"
                "-----\n"
                "PROJECT_BRIEF (markdown, truncated if long):\n"
                f"{brief}\n"
                "-----"
            )
        else:
            return (
                "PROJECT_BRIEF (markdown, truncated if long):\n"
                f"{brief}\n"
                "-----"
            )

    def _fallback_brief_from_files(self) -> str:
        """
        Deterministic fallback if the brief compiler fails.
        """
        candidates = [
            self.root / ".aidev" / "project_description.md",
            self.root / "project_description.md",
            self.root / "app_descrip.txt",
            self.root / "README.md",
            self.root / "README",
        ]
        chunks: List[str] = []
        for p in candidates:
            try:
                if p.is_file():
                    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
                    if txt:
                        chunks.append(f"# {p.name}\n\n{txt}")
            except Exception:
                continue
        return "\n\n\n".join(chunks).strip()

    def _serialize_apply_results_for_event(self) -> List[Dict[str, Any]]:
        """
        Convert per-recommendation apply results (e.g. ApplyRecResult instances)
        into plain dicts suitable for SSE / JSON.

        Expected fields (when present):
          - rec_id
          - title
          - applied
          - skipped
          - reason
          - changed_paths
          - details

        Additionally, if any cross_file_notes were recorded for a rec_id,
        they are attached under details['cross_file_notes'].
        """
        raw = getattr(self, "_apply_results", None) or []
        out: List[Dict[str, Any]] = []

        for res in raw:
            # If it's already a dict, assume it's serializable and keep it.
            if isinstance(res, dict):
                rec_dict = dict(res)
            else:
                rec_dict: Dict[str, Any] = {}
                try:
                    rec_id = getattr(res, "rec_id", None)
                    if rec_id is not None:
                        rec_dict["rec_id"] = str(rec_id)

                    title = getattr(res, "title", None)
                    if title:
                        rec_dict["title"] = str(title)

                    applied = getattr(res, "applied", None)
                    if applied is not None:
                        rec_dict["applied"] = bool(applied)

                    skipped = getattr(res, "skipped", None)
                    if skipped is not None:
                        rec_dict["skipped"] = bool(skipped)

                    reason = getattr(res, "reason", None)
                    if reason is not None:
                        rec_dict["reason"] = str(reason)

                    changed_paths = getattr(res, "changed_paths", None)
                    if changed_paths is not None:
                        try:
                            rec_dict["changed_paths"] = list(changed_paths)
                        except TypeError:
                            rec_dict["changed_paths"] = []

                    details = getattr(res, "details", None)
                    if isinstance(details, dict):
                        rec_dict["details"] = details
                except Exception:
                    # Don't let a bad result blow up the whole summary; just skip it.
                    pass

            # attach aggregated cross_file_notes if we have any
            try:
                rec_id_str = rec_dict.get("rec_id")
                if rec_id_str and rec_id_str in self._cross_file_notes_by_rec:
                    notes = self._cross_file_notes_by_rec[rec_id_str]
                    if notes:
                        details = rec_dict.setdefault("details", {})
                        # Don't overwrite if caller already stashed some notes
                        existing = details.get("cross_file_notes")
                        if not existing:
                            details["cross_file_notes"] = notes
            except Exception:
                pass

            out.append(rec_dict)

        return out

    # ---------- ai_summary/card-summarize failures (non-blocking) ----------

    def _gather_ai_summary_failures(self, kb: KnowledgeBase) -> List[Dict[str, str]]:
        """Gather, normalize, dedupe, and sort per-file ai_summary failures.

        This is best-effort and must never raise; callers should wrap it.

        Sources are probed in this order:
          1) KnowledgeBase attributes (public or private variants)
          2) KnowledgeBase card index fields
        """

        def _normalize(raw: Any) -> List[Dict[str, str]]:
            normalized: List[Dict[str, str]] = []
            for item in (raw or []):
                p: Optional[Any] = None
                e: Optional[Any] = None

                if isinstance(item, dict):
                    p = item.get("path") or item.get("file") or item.get("rel_path")
                    e = item.get("error") or item.get("message") or item.get("err")
                elif isinstance(item, (list, tuple)):
                    if len(item) >= 1:
                        p = item[0]
                    if len(item) >= 2:
                        e = item[1]
                    elif len(item) == 1:
                        e = str(item[0])
                else:
                    # Unknown shape: cannot infer path reliably.
                    continue

                if p is None:
                    continue
                ps = str(p).strip()
                if not ps:
                    continue
                es = "" if e is None else str(e)
                normalized.append({"path": ps, "error": es})
            return normalized

        raw: Any = None
        # 1) Attribute probe (most direct)
        for attr in (
            "ai_summary_failures",
            "summarize_failures",
            "summary_failures",
            "_ai_summary_failures",
            "_summarize_failures",
            "_summary_failures",
        ):
            try:
                v = getattr(kb, attr, None)
                if v:
                    raw = v
                    break
            except Exception:
                continue

        # 2) Index probe (persisted)
        if raw is None:
            try:
                idx = kb.load_card_index() or {}
                if isinstance(idx, dict):
                    for key in (
                        "ai_summary_failures",
                        "summary_failures",
                        "summarize_failures",
                    ):
                        v = idx.get(key)
                        if v:
                            raw = v
                            break
            except Exception:
                raw = None

        normalized = _normalize(raw)

        # Dedupe by path (first occurrence wins), then sort by path for determinism.
        dedup: Dict[str, Dict[str, str]] = {}
        for rec in normalized:
            p = rec.get("path")
            if not isinstance(p, str) or not p.strip():
                continue
            if p not in dedup:
                dedup[p] = {"path": p, "error": str(rec.get("error", ""))}

        out = list(dedup.values())
        out.sort(key=lambda d: d.get("path", ""))
        return out

    # ---------- Post-apply summary / utils ----------

    def _emit_result_and_done(self, ok: bool, summary: Optional[str] = None) -> None:
        created, modified, skipped = (
            self._files_created,
            self._files_modified,
            self._files_skipped,
        )
        changed = created + modified
        summary = summary or (
            f"Changed {changed} file(s): {created} created, {modified} modified; {skipped} skipped."
        )

        final_ok = ok and (len(self._errors) == 0)

        # Serialize per-rec apply results for the UI
        rec_results = self._serialize_apply_results_for_event()

        # Build the result payload, optionally attaching analyze_plan when present
        result_kwargs: Dict[str, Any] = {
            "ok": final_ok,
            "summary": summary,
            "artifacts": [],
            "where": "pipeline",
            "session_id": self._session_id,
            "job_id": self.job_id,
            "rec_results": rec_results,
            "writes_by_rec": self._writes_by_rec,
            "errors": list(self._errors),
            "ai_summary_failures": list(self._ai_summary_failures or []),
        }

        if isinstance(self._last_analyze_plan, dict) and self._last_analyze_plan:
            # Attach the plan so API callers and listeners can observe it
            result_kwargs["analyze_plan"] = self._last_analyze_plan

        _events.result(**result_kwargs)
        _events.done(
            ok=final_ok,
            where="pipeline",
            session_id=self._session_id,
            job_id=self.job_id,
        )
        self._finish_job_registry(
            ok=final_ok,
            summary=summary,
            artifacts=[],
        )

    # ---------- Project map (v2 / LLM-facing) helpers ----------

    def _build_and_attach_project_map(
        self,
        *,
        kb: KnowledgeBase,
        meta: Dict[str, Any],
        out_path: Optional[Path] = None,
        force: bool = False,
        recent_changed_files: Optional[Sequence[str]] = None,
        trace_source: str = "initial",
    ) -> Dict[str, Any]:
        """
        Build a lean, LLM-facing project_map and attach it to meta["project_map"].

        Preferred:
          - Use repo_map.build_project_map(self.root, force=force) to get the
            canonical .aidev/project_map.json schema.

        Fallback:
          - If repo_map isn't available, use KnowledgeBase.save_project_map(...)
            and reload it, but only attach if it exposes a 'files' list.

        Returns a *new* meta dict with project_map attached when successful.
        """
        if out_path is None:
            out_path = self.root / ".aidev" / "project_map.json"

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        pm_obj: Optional[Dict[str, Any]] = None

        # Preferred: repo_map.build_project_map (lean, LLM-facing)
        if _build_repo_project_map is not None:
            try:
                pm_obj = _build_repo_project_map(self.root, force=force)
                # Ensure it's also written to disk (repo_map already does this,
                # but keep out_path as the canonical location for callers).
                try:
                    out_path.write_text(
                        json.dumps(pm_obj, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            except Exception as e:
                logging.debug(
                    "repo_map.build_project_map failed: %s",
                    e,
                    extra=self._log_extra(phase="project_map"),
                )
                pm_obj = None

        # Fallback: legacy KnowledgeBase.save_project_map
        if pm_obj is None:
            save_fn = getattr(kb, "save_project_map", None)
            if callable(save_fn):
                try:
                    pm_path = save_fn(
                        out_path,
                        project_meta=meta,
                        include_tree=False,
                        include_files=True,
                        prefer_ai_summaries=True,
                        compact_tree=True,
                        pretty=False,
                    )
                    from pathlib import Path as _Path

                    if isinstance(pm_path, (str, _Path)):
                        pm_path = _Path(pm_path)
                    else:
                        pm_path = out_path

                    try:
                        pm_obj = json.loads(pm_path.read_text(encoding="utf-8"))
                    except Exception:
                        pm_obj = None
                except Exception as e:
                    logging.debug(
                        "save_project_map fallback failed: %s",
                        e,
                        extra=self._log_extra(phase="project_map"),
                    )

        new_meta = dict(meta)
        if isinstance(pm_obj, dict) and isinstance(pm_obj.get("files"), list):
            new_meta["project_map"] = pm_obj
            try:
                self.st.trace.write(
                    "project_map_refresh",
                    trace_source,
                    {
                        "path": str(out_path),
                        "total_files": pm_obj.get("total_files")
                        or pm_obj.get("file_count"),
                        "recent_changed_files": list(recent_changed_files or []),
                    },
                )
            except Exception:
                pass

        return new_meta

    def _with_timeout(self, fn: Callable[[], Any], timeout_sec: float, desc: str) -> Any:
        """
        Soft timeout wrapper.

        - Calls fn() synchronously.
        - Logs success with elapsed time.
        - If elapsed > timeout_sec, logs a 'soft timeout' warning and emits a
        'slow_call' progress event, but DOES NOT discard the result.
        - On exceptions, logs and pushes a progress error, then returns the
        global timeout sentinel (_TIMEOUT) so callers can detect failure.
        """
        start = time.time()
        try:
            result = fn()
            elapsed = time.time() - start

            if elapsed <= timeout_sec:
                logging.info(
                    "[with_timeout] %s completed in %.2fs",
                    desc,
                    elapsed,
                    extra=self._log_extra(phase=desc),
                )
            else:
                logging.warning(
                    "[with_timeout] %s exceeded soft timeout: elapsed=%.2fs timeout=%.2fs",
                    desc,
                    elapsed,
                    timeout_sec,
                    extra=self._log_extra(phase=desc),
                )
                # Emit a 'slow_call' style event instead of a hard timeout
                try:
                    self._progress(
                        "slow_call",
                        where=desc,
                        seconds=elapsed,
                        timeout=timeout_sec,
                        job_id=self.job_id,
                    )
                except Exception:
                    logging.debug(
                        "[with_timeout] failed to emit slow_call progress for %s",
                        desc,
                        exc_info=True,
                    )

            return result

        except Exception as e:
            elapsed = time.time() - start
            logging.exception(
                "Error in %s after %.2fs: %s",
                desc,
                elapsed,
                e,
                extra=self._log_extra(phase=desc),
            )
            try:
                self._progress_error(
                    desc,
                    error=str(e),
                    trace=traceback.format_exc(),
                    job_id=self.job_id,
                )
            except Exception:
                logging.debug(
                    "[with_timeout] failed to emit progress_error for %s",
                    desc,
                    exc_info=True,
                )
            # IMPORTANT: return the sentinel so callers that passed timeout_sentinel=_TIMEOUT
            # can detect this as a runner-level failure.
            return _TIMEOUT

    def _progress(self, event: str, **payload: Any) -> None:
        try:
            if self._progress_cb:
                self._progress_cb(event, payload)
                return
        except Exception as e:
            logging.debug("progress_cb failed for %s: %s", event, e)

        msg = payload.get("message") or payload.get("msg") or event
        where = payload.get("where") or event

        # Merge job_id into payload safely
        merged = dict(payload)
        merged.setdefault("job_id", self.job_id)

        _events.status(
            str(msg),
            where=where,
            session_id=self._session_id,
            **merged,
        )

    def _progress_error(self, where: str, **payload: Any) -> None:
        rec = {"where": where, "ts": time.time(), **payload}
        self._errors.append(
            {
                "where": where,
                **{k: v for k, v in payload.items() if k in ("error", "reason")},
            }
        )
        try:
            self._progress("error", **rec)
        except Exception:
            pass
        try:
            self.st.trace.write("ERROR", where, rec)
        except Exception:
            pass

        # job-aware, phase-aware logging extras
        extra = self._log_extra(
            phase=where,
            rec_id=str(payload.get("rec_id")) if payload.get("rec_id") is not None else None,
            reasoning=payload.get("reasoning") or payload.get("reason"),
        )

        if "error" in payload:
            logging.error("[%s] %s", where, payload["error"], extra=extra)
        elif "reason" in payload:
            logging.warning("[%s] %s", where, payload["reason"], extra=extra)

    def _log_extra(
        self,
        *,
        phase: Optional[str] = None,
        rec_id: Optional[str] = None,
        reasoning: Optional[str] = None,
        **more: Any,
    ) -> Dict[str, Any]:
        """
        Build a consistent logging.extra payload so all log records for this
        orchestrator carry job-scoped metadata.

        Always includes:
          - job_id   (str)
          - phase    (str)  : logical phase or 'orchestrator'
          - rec_id   (str)  : recommendation id, when available
          - reasoning(str)  : optional high-level explanation, when provided

        Additional keys from **more are merged in without overwriting the core
        ones above.
        """
        base: Dict[str, Any] = {
            "job_id": getattr(self, "job_id", None),
            "phase": phase or more.get("phase") or "orchestrator",
            "rec_id": rec_id or more.get("rec_id") or "",
            "reasoning": reasoning or more.get("reasoning") or "",
        }

        for k, v in more.items():
            if k not in base:
                base[k] = v

        return base

    def _is_safe_path(self, rel_path: str) -> bool:
        try:
            if os.path.isabs(rel_path):
                return False
            p = (self.root / rel_path).resolve()
            root = self.root.resolve()
            if p == root:
                return False
            return str(p).startswith(str(root) + os.sep)
        except Exception:
            return False

    @staticmethod
    def _tail(s: str, nbytes: int) -> str:
        if not s:
            return ""
        enc = s.encode("utf-8", errors="ignore")
        if len(enc) <= nbytes:
            return s
        return enc[-nbytes:].decode("utf-8", errors="ignore")

    def _print_summary_and_trace(self) -> None:
        created, modified, skipped = (
            self._files_created,
            self._files_modified,
            self._files_skipped,
        )
        changed = created + modified
        logging.info(
            "Summary: %d changed (%d created, %d modified), %d skipped",
            changed,
            created,
            modified,
            skipped,
            extra=self._log_extra(phase="summary"),
        )
        try:
            self.st.trace.write(
                "SUMMARY",
                "orchestrator",
                {
                    "changed": changed,
                    "created": created,
                    "modified": modified,
                    "skipped": skipped,
                    "files": self._file_change_summaries,
                    "writes_by_rec": self._writes_by_rec,
                    "brief_hash": self._project_brief_hash,
                    "job_id": self.job_id,
                    "errors": self._errors,
                    "ai_summary_failures": list(self._ai_summary_failures or []),
                },
            )
        except Exception:
            pass

    # ---------- Token usage extraction ----------

    def _extract_token_usage(self, res: Any) -> Tuple[int, int]:
        """
        Best-effort extraction of (input_tokens, output_tokens) from various
        OpenAI/SDK response shapes. Falls back to (0, 0) if absent.
        """
        try:
            if res is None:
                return 0, 0
            usage = getattr(res, "usage", None)
            if usage is None and isinstance(res, dict):
                usage = res.get("usage") or (res.get("response") or {}).get("usage")

            if usage is None:
                return 0, 0

            def _get(u: Any, key: str):
                if isinstance(u, dict):
                    return u.get(key)
                return getattr(u, key, None)

            tin = _get(usage, "input_tokens")
            tout = _get(usage, "output_tokens")

            if tin is None:
                tin = _get(usage, "prompt_tokens")
            if tout is None:
                tout = _get(usage, "completion_tokens")

            if tin is None:
                tin = _get(usage, "input_token_count") or _get(
                    usage, "prompt_token_count"
                )
            if tout is None:
                tout = _get(usage, "output_token_count") or _get(
                    usage, "completion_token_count"
                )

            return int(tin or 0), int(tout or 0)
        except Exception:
            return 0, 0

    # ---- Small arg coercion helpers ----

    def _to_jsonable(self, obj: Any, _depth: int = 0) -> Any:
        """
        Best-effort conversion of arbitrary Python objects into something that
        json.dumps can handle.

        - Primitives pass through unchanged.
        - dict / list / tuple / set are walked recursively (with depth guard).
        - Path is stringified.
        - Anything else that isn't JSON-serializable is converted to str(obj).
        """
        if _depth > 6:
            # Avoid pathological recursion; just stringify.
            return str(obj)

        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        if isinstance(obj, dict):
            return {
                str(k): self._to_jsonable(v, _depth=_depth + 1)
                for k, v in obj.items()
            }

        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(v, _depth=_depth + 1) for v in obj]

        if isinstance(obj, Path):
            return str(obj)

        # Last resort: if json can handle it already, keep it; otherwise stringify.
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    def _coerce_str_list(self, v: object) -> List[str]:
        """
        Accepts None | str | list|tuple|set -> List[str].

        - If str is JSON like '["a","b"]', parse it.
        - Otherwise split on commas/semicolons/newlines and whitespace.
        """
        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="ignore")
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            try:
                data = json.loads(s)
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except Exception:
                pass
            delim_parts = re.split(r"[,\n;]+", s)
            if len(delim_parts) > 1:
                parts = delim_parts
            else:
                parts = s.split()
            return [p.strip() for p in parts if p.strip()]
        try:
            return [str(v).strip()] if str(v).strip() else []
        except Exception:
            return []
