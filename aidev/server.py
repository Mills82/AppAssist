# aidev/server.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from uuid import UUID

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

# App routers
from .routes.workspaces import router as workspaces_router
from .routes.session import router as session_router
from .routes.projects import router as projects_router
from .routes.targets import router as targets_router
from .api.conversation import router as conversation_router
# Optional frontend router for new Next.js UI (non-fatal if missing)
try:
    from .routes.frontend import router as frontend_router  # type: ignore
except Exception:
    frontend_router = None  # type: ignore[assignment]

from .runtime import run_preapply_checks
from .assistant_api import DevBotAPI

# Internal modules
from .validators import safe_json
from .session_store import SESSIONS
from .state import ProjectState
from .orchestrator import Orchestrator
from .orchestration.approval_inbox import get_approval_inbox
from .recommendations_io import load_recommendations
from . import events as _events  # emit utils (no routes)
from .chat import run_chat_conversation  # chat-first entrypoint for jobs

# Optional: only mount if present
try:
    from .api.llm import router as llm_router  # type: ignore
except Exception:
    llm_router = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

log = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("AIDEV_LOGLEVEL", "INFO"))

APPROVALS = get_approval_inbox()

HEARTBEAT_SECS = 15
SSE_PREFIX = "/jobs/"  # single source of truth for SSE paths

# Header used to pass a selected project root from the UI to the server.
HEADER_NAME = "X-AIDEV-PROJECT"

TAGS_METADATA = [
    {"name": "chat", "description": "Chat & orchestration"},
    {"name": "projects", "description": "Project lifecycle (list/create/select)"},
    {"name": "summaries", "description": "Codebase summaries (changed/deep)"},
    {"name": "checks", "description": "Pre-apply and post-apply checks"},
    {"name": "events", "description": "Job-scoped SSE streams (/jobs/stream)"},
    {"name": "approval", "description": "Approval gate decisions"},
]


# ---------- Helpers: SSE header middleware ----------


class SSEHeaderMiddleware:
    """Force proper SSE headers on /jobs/* endpoints."""

    def __init__(self, app: ASGIApp, sse_prefix: str = SSE_PREFIX) -> None:
        self.app = app
        self.sse_prefix = sse_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(
            self.sse_prefix
        ):
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))

                def _set(k: str, v: str):
                    lower = k.lower().encode("latin-1")
                    for i, (hk, _) in enumerate(headers):
                        if hk.lower() == lower:
                            headers[i] = (k.encode("latin-1"), v.encode("latin-1"))
                            break
                    else:
                        headers.append((k.encode("latin-1"), v.encode("latin-1")))

                _set("Content-Type", "text/event-stream")
                _set("Cache-Control", "no-cache")
                _set("Connection", "keep-alive")
                _set("X-Accel-Buffering", "no")
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ---------- Helpers: Project resolution middleware ----------


class ProjectResolveMiddleware:
    """ASGI middleware that resolves the active project root for each HTTP request.

    Resolution order:
    1) session.project_root or session.root (if SessionMiddleware populated scope['session'])
    2) X-AIDEV-PROJECT header value


    The resolved absolute path (string) is set on scope['state'].selected_project or None if not resolvable.
    This middleware does not mutate session state and never raises on bad input.
    """

    def __init__(self, app: ASGIApp, header_name: str = HEADER_NAME) -> None:
        self.app = app
        self.header_name = header_name.lower()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only act on HTTP requests; pass through otherwise.
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        selected: Optional[str] = None
        try:
            # 1) Session-based project root (SessionMiddleware populates scope['session'])
            session = scope.get("session")
            if session:
                try:
                    # session may be a dict-like or object; try both
                    if isinstance(session, dict):
                        pr = session.get("project_root") or session.get("root")
                    else:
                        pr = getattr(session, "project_root", None) or getattr(
                            session, "root", None
                        )
                    if pr:
                        selected = str(pr)
                except Exception:
                    selected = None

            # 2) Header fallback if session provided nothing
            if not selected:
                headers = scope.get("headers") or []
                target = self.header_name.encode("latin-1")
                for (hk, hv) in headers:
                    try:
                        if hk.lower() == target:
                            hv_str = hv.decode("utf-8", errors="ignore").strip()
                            if hv_str:
                                selected = hv_str
                                break
                    except Exception:
                        continue

            # Normalize/resolve selected into an absolute path string if possible.
            resolved: Optional[str] = None
            if selected:
                try:
                    p = Path(selected)
                    # Resolve may raise on some OS-specific inputs; swallow any errors.
                    rp = p.resolve()
                    resolved = str(rp)
                except Exception:
                    resolved = None

            # Ensure scope has a state object and set attribute.
            state_obj = scope.get("state")
            if state_obj is None:
                # Minimal lightweight state holder if Starlette didn't create one.
                class _State:  # type: ignore
                    pass

                state_obj = _State()
                scope["state"] = state_obj

            setattr(state_obj, "selected_project", resolved)
        except Exception:
            # Never fail the request due to middleware issues; leave selected_project unset/None.
            try:
                state_obj = scope.get("state")
                if state_obj is None:

                    class _State2:  # type: ignore
                        pass

                    state_obj = _State2()
                    scope["state"] = state_obj
                setattr(state_obj, "selected_project", None)
            except Exception:
                pass

        await self.app(scope, receive, send)


def get_selected_project_from_scope(scope: Scope) -> Optional[str]:
    """Helper: return the middleware-populated selected project path (absolute) or None.

    Useful for handlers/tests that need a shared resolution API.
    """

    try:
        state = scope.get("state")
        if not state:
            return None
        return getattr(state, "selected_project", None)
    except Exception:
        return None


def _project_root_from_session_like(session: Any) -> Optional[str]:
    """Extract project_root/project_path/root from a cookie-session-like object (dict-like or attr-like)."""
    if not session:
        return None
    try:
        if isinstance(session, dict):
            pr = (
                session.get("project_root")
                or session.get("project_path")
                or session.get("root")
            )
        else:
            pr = (
                getattr(session, "project_root", None)
                or getattr(session, "project_path", None)
                or getattr(session, "root", None)
            )
        pr_s = str(pr).strip() if pr else ""
        return pr_s or None
    except Exception:
        return None


def _resolve_project_root(
    *,
    request: Optional[Request] = None,
    session_obj: Any = None,
    payload_project_root: Any = None,
    allow_fallback_cwd: bool = True,
) -> tuple[Optional[str], str]:
    """Resolve project_root as an absolute string and return (project_root, source).

    Resolution order (session-first for safety/consistency):
      1) server-side session object (SESSIONS.ensure/get)
      2) explicit payload.project_root (also accepts legacy project_path/root)
      3) middleware-selected project (from cookie session/header)
      4) fallback to cwd (optional; for legacy, non-edit flows)

    source is one of: session | payload | header | cwd | none
    """

    # 1) session (already handles project_root/project_path/root if you updated it)
    pr = _project_root_from_session_like(session_obj)
    if pr:
        try:
            return str(Path(pr).resolve()), "session"
        except Exception:
            pass

    # 2) payload (string or dict-like)
    if payload_project_root:
        try:
            pr2: Optional[str] = None

            if isinstance(payload_project_root, dict):
                # Accept canonical + legacy keys
                pr2 = (
                    payload_project_root.get("project_root")
                    or payload_project_root.get("project_path")
                    or payload_project_root.get("root")
                )
            else:
                pr2 = str(payload_project_root).strip()

            pr2_s = str(pr2).strip() if pr2 else ""
            if pr2_s:
                return str(Path(pr2_s).resolve()), "payload"
        except Exception:
            pass

    # 3) header (middleware-populated)
    if request is not None:
        try:
            selected = get_selected_project_from_scope(request.scope)
            if selected:
                return str(Path(selected).resolve()), "header"
        except Exception:
            pass

    # 4) fallback
    if allow_fallback_cwd:
        try:
            return str(Path.cwd().resolve()), "cwd"
        except Exception:
            return None, "none"

    return None, "none"


# ---------- In-memory Job Registry ----------


@dataclass
class JobRecord:
    job_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_ts: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    project_root: Optional[str] = None
    auto_approve: bool = False
    status: str = "pending"  # "pending" | "running" | "done" | "error" | "canceled"
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None
    cancel_cb: Optional[Callable[[], None]] = None


class JobRegistry:
    _jobs: Dict[str, JobRecord] = {}

    @classmethod
    def create(
        cls,
        session_id: Optional[str],
        project_root: Optional[str],
        auto_approve: bool = False,
    ) -> JobRecord:
        job_id = f"job_{int(time.time()*1000)}_{os.urandom(3).hex()}"
        rec = JobRecord(
            job_id=job_id,
            session_id=session_id,
            project_root=project_root,
            auto_approve=auto_approve,
        )
        cls._jobs[job_id] = rec
        return rec

    @classmethod
    def get(cls, job_id: str) -> Optional[JobRecord]:
        return cls._jobs.get(job_id)

    @classmethod
    def publish(cls, job_id: str, ev_type: str, payload: Dict[str, Any]) -> None:
        rec = cls.get(job_id)
        if not rec:
            return
        msg = {"type": ev_type, "payload": payload or {}, "ts": time.time()}
        try:
            rec.queue.put_nowait(msg)
        except Exception as e:
            log.warning("publish failed for %s: %s", job_id, e)

    @classmethod
    def complete(cls, job_id: str, ok: bool, error: Optional[str] = None) -> None:
        rec = cls.get(job_id)
        if not rec:
            return
        rec.status = "done" if ok else "error"
        rec.error = error
        # final event
        payload = {"ok": ok}
        if error:
            payload["error"] = error
        cls.publish(job_id, "result", payload)

    @classmethod
    def cancel(cls, job_id: str) -> bool:
        rec = cls.get(job_id)
        if not rec:
            return False
        if rec.status in ("done", "error", "canceled"):
            return True
        rec.status = "canceled"
        if rec.cancel_cb:
            try:
                rec.cancel_cb()
            except Exception:
                pass
        if rec.task:
            rec.task.cancel()
        cls.publish(job_id, "status", {"stage": "canceled"})
        cls.publish(job_id, "result", {"ok": False, "error": "canceled"})
        return True


# ---------- SSE packing ----------


def _sse_pack(event: Dict[str, Any], ev_type: Optional[str] = None) -> str:
    # Canonical SSE line format
    typ = ev_type or event.get("type") or "message"
    data = json.dumps(event, ensure_ascii=False)
    return f"event: {typ}\ndata: {data}\n\n"


async def _job_event_stream(job: JobRecord) -> AsyncGenerator[bytes, None]:
    """Stream job events with heartbeats."""

    last = time.time()
    try:
        # initial hello
        yield _sse_pack(
            {"type": "hello", "payload": {"job_id": job.job_id}, "ts": time.time()}
        ).encode("utf-8")
        while True:
            try:
                msg = await asyncio.wait_for(job.queue.get(), timeout=HEARTBEAT_SECS)
                yield _sse_pack(msg).encode("utf-8")
                last = time.time()
                if msg.get("type") == "result":
                    break
            except asyncio.TimeoutError:
                # heartbeat
                hb = {"type": "ping", "payload": {"ts": time.time()}, "ts": time.time()}
                yield _sse_pack(hb, ev_type="ping").encode("utf-8")
                if (
                    time.time() - last > 5 * HEARTBEAT_SECS
                    and job.status in ("done", "error", "canceled")
                ):
                    break
    except asyncio.CancelledError:
        # client disconnected
        return
    except Exception as e:
        err = {"type": "error", "payload": {"message": str(e)}, "ts": time.time()}
        yield _sse_pack(err, ev_type="error").encode("utf-8")


# ---------- Session helpers ----------


def _session_secret() -> str:
    return os.getenv("AIDEV_SESSION_SECRET", "dev-insecure-aidev-session-secret")


def _parse_cors_origins() -> tuple[List[str], bool]:
    """
    Returns (allow_origins, allow_credentials).
    Precedence:
      1) FRONTEND_ORIGIN env var (single origin override for local dev)
      2) AIDEV_CORS_ORIGINS env var (same behavior as before)
      3) Default to http://localhost:3000 for convenient Next.js dev use.

    If AIDEV_CORS_ORIGINS="*", we disable credentials for RFC/CORS compliance.
    Otherwise we allow credentials for specific origins.
    """

    frontend = os.getenv("FRONTEND_ORIGIN", "").strip()
    if frontend:
        try:
            log.debug("FRONTEND_ORIGIN override active: %s", frontend)
        except Exception:
            pass
        return [frontend], True

    raw = os.getenv("AIDEV_CORS_ORIGINS", "").strip()
    if not raw:
        # No explicit CORS configured: default to Next.js dev origin for local development
        return ["http://localhost:3000"], True
    if raw == "*":
        return ["*"], False
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return (origins or ["http://localhost:3000"]), True


def _progress_cb_for_job(job_id: str) -> Callable[[str, Dict[str, Any]], None]:
    """Return a callback that publishes canonical events into the job queue."""

    def _cb(ev_type: str, payload: Dict[str, Any]) -> None:
        try:
            JobRegistry.publish(job_id, ev_type, payload or {})
        except Exception as e:
            log.debug("progress publish failed: %s", e)

    return _cb


# ----- Bridge: mirror global _events emissions to a specific job stream -----


def _attach_events_bridge(job: JobRecord) -> Optional[Callable[[], None]]:
    def _observer(event: Dict[str, Any]) -> None:
        try:
            # Support both {"payload": {...}} and {"data": {...}} event shapes
            payload = event.get("payload") or event.get("data") or {}
            sid = payload.get("session_id")
            if sid and job.session_id and str(sid) == str(job.session_id):
                ev_type = event.get("type") or "status"
                JobRegistry.publish(job.job_id, ev_type, payload)
                # Keep in-memory state in sync if orchestrator already emitted a terminal result
                if ev_type == "result":
                    ok = bool(payload.get("ok", True))
                    job.status = "done" if ok else "error"
                    job.error = None if ok else str(
                        payload.get("error") or payload.get("summary") or ""
                    )
        except Exception:
            pass

    # Attach to the global events emitter and return a detach callable.
    detach: Optional[Callable[[], None]] = None
    try:
        # Common API #1: add_observer/remove_observer
        if hasattr(_events, "add_observer"):
            _events.add_observer(_observer)  # type: ignore[attr-defined]

            def _detach():
                try:
                    if hasattr(_events, "remove_observer"):
                        _events.remove_observer(_observer)  # type: ignore[attr-defined]
                except Exception:
                    pass

            detach = _detach
        # Common API #2: subscribe/unsubscribe (token-based)
        elif hasattr(_events, "subscribe"):
            token = _events.subscribe(_observer)  # type: ignore[attr-defined]

            def _detach():
                try:
                    if hasattr(_events, "unsubscribe"):
                        _events.unsubscribe(token)  # type: ignore[attr-defined]
                except Exception:
                    pass

            detach = _detach
        else:
            log.debug("events bridge: no subscribe API on _events")
    except Exception as e:
        log.debug("events bridge attach failed: %s", e)
        detach = None

    return detach


# ---------- Summaries helpers (Phase 3) ----------


def _coerce_file_item(x: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize various shapes from llm_client responses into:
      {path, ok, summary_len, error}
    """

    if isinstance(x, str):
        return {"path": x, "ok": True, "summary_len": 0, "error": ""}
    if not isinstance(x, dict):
        return None

    path = str(x.get("path") or x.get("rel") or x.get("file") or x.get("name") or "").strip()

    ok = x.get("ok")
    if ok is None:
        ok = not bool(x.get("error"))

    s = x.get("ai_summary") or x.get("summary") or x.get("text") or ""
    if isinstance(s, dict):
        s = s.get("text", "")
    summary_len = int(x.get("summary_len", len(str(s))))

    return {
        "path": path,
        "ok": bool(ok),
        "summary_len": summary_len,
        "error": str(x.get("error") or ""),
    }


def _gather_file_list(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Any] = []
    for key in ("files", "results", "per_file", "details", "items", "entries", "summaries"):
        v = result.get(key)
        if isinstance(v, list):
            candidates = v
            break

    out: List[Dict[str, Any]] = []
    for item in candidates or []:
        norm = _coerce_file_item(item)
        if norm and norm.get("path"):
            out.append(norm)

    errs = result.get("errors")
    if isinstance(errs, dict):
        for p, msg in errs.items():
            out.append({"path": str(p), "ok": False, "summary_len": 0, "error": str(msg)})

    uniq: Dict[str, Dict[str, Any]] = {}
    for rec in out:
        key = rec["path"]
        prev = uniq.get(key)
        if prev is None:
            uniq[key] = rec
        else:
            if (not prev["ok"]) or (not rec["ok"]):
                uniq[key] = rec if not rec["ok"] else prev
            elif rec["summary_len"] > prev["summary_len"]:
                uniq[key] = rec

    return list(uniq.values())


def _counts_from_files(files: List[Dict[str, Any]], fallback: Dict[str, int] | None = None) -> Dict[str, int]:
    summarized = sum(1 for f in files if f.get("ok"))
    failed = sum(1 for f in files if not f.get("ok"))
    skipped = (fallback or {}).get("skipped", 0)
    return {
        "summarized": summarized if summarized else (fallback or {}).get("summarized", 0),
        "skipped": skipped,
        "failed": failed if failed else (fallback or {}).get("failed", 0),
    }


# ---------- App factory ----------


def create_app(static_ui_dir: Optional[Path] = None, cfg: Optional[Dict[str, Any]] = None) -> FastAPI:
    app = FastAPI(title="AI-Dev-Bot Server", version="0.3.0", openapi_tags=TAGS_METADATA)

    SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"

    if SCHEMAS_DIR.exists():
        app.mount(
            "/schemas",
            StaticFiles(directory=str(SCHEMAS_DIR)),
            name="schemas",
        )

    # CORS (safe defaults; can be narrowed via AIDEV_CORS_ORIGINS or FRONTEND_ORIGIN)
    allow_origins, allow_credentials = _parse_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Sessions (used by some flows; cookie name configurable)
    app.add_middleware(
        SessionMiddleware,
        secret_key=_session_secret(),
        session_cookie=os.getenv("AIDEV_SESSION_COOKIE", "aidev_session"),
        max_age=60 * 60 * 24 * 30,
        same_site="lax",
        https_only=False,
    )

    # Ensure SSE headers for /jobs/*
    app.add_middleware(SSEHeaderMiddleware, sse_prefix=SSE_PREFIX)

    # Resolve selected project for each request (from session or header)
    app.add_middleware(ProjectResolveMiddleware, header_name=HEADER_NAME)

    # Routers (no legacy /events router)
    app.include_router(workspaces_router, prefix="", tags=["projects"])
    if llm_router:
        app.include_router(llm_router)
    app.include_router(session_router, prefix="", tags=["session"])
    app.include_router(projects_router, prefix="", tags=["projects"])
    # Mount /api/select-targets (router has prefix="/api" inside)
    app.include_router(targets_router, prefix="", tags=["projects"])
    app.include_router(conversation_router)

    # Optionally include the frontend router (provides /api/v1/* mocks for the new Next.js UI)
    if frontend_router:
        try:
            app.include_router(frontend_router, prefix="/api/v1")
            log.info("Mounted frontend router at /api/v1")
        except Exception:
            try:
                # Router may define its own prefix internally; try including without prefix as a fallback
                app.include_router(frontend_router)
                log.info("Mounted frontend router (router provided its own prefix)")
            except Exception as e:
                log.exception("Failed to include frontend router: %s", e)
    else:
        log.debug("No frontend router found; skipping /api/v1 UI mocks")

    # ---------- Jobs API (job-scoped streaming) ----------

    @app.post("/jobs/start", tags=["chat"])
    async def jobs_start(request: Request, payload: Dict[str, Any]):
        """
        Starts a chat-first orchestration run. Returns immediately with {job_id}.
        Body:
          - message: str (required)
          - session_id: str (optional)
          - project_root: str (optional; defaults to CWD or session project)
          - args: dict (optional; additional metadata: focus, etc.)
          - auto_approve: bool (optional; default False)
          - mode: "auto" | "qa" | "analyze" | "edit" (optional; chat mode hint/override)
        """

        message = (payload or {}).get("message", "")
        if not isinstance(message, str) or not message.strip():
            return JSONResponse(
                status_code=400, content={"ok": False, "error": "message required"}
            )

        session_id = (payload or {}).get("session_id") or None
        payload_project_root = (payload or {}).get("project_root") or None
        args = (payload or {}).get("args") or {}

        # Optional chat mode; can be provided either at the top level
        # (payload.mode) or inside args.mode. Top-level wins if both are present.
        mode: Optional[str] = None
        if isinstance(args, dict) and args.get("mode"):
            mode = str(args["mode"]).strip().lower()

        raw_mode = (payload or {}).get("mode")
        if raw_mode:
            mode = str(raw_mode).strip().lower()

        # Coerce auto_approve to a bool with sane defaults
        raw_auto = (payload or {}).get("auto_approve", False)

        if isinstance(raw_auto, str):
            auto_approve = raw_auto.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(raw_auto, (int, float)):
            auto_approve = bool(raw_auto)
        else:
            auto_approve = bool(raw_auto)

        # Resolve session once so we can use it for project_root and focus
        session = None
        # Allow explicit session_id in payload OR a cookie named 'session_id' (UI selects)
        if not session_id:
            try:
                cookie_sid = request.cookies.get("session_id")
                if cookie_sid:
                    session_id = cookie_sid
            except Exception:
                pass

        if session_id:
            try:
                # prefer ensure for create-if-missing semantics like other code paths
                session = await SESSIONS.ensure(session_id)
            except Exception:
                try:
                    session = await SESSIONS.get(session_id)
                except Exception:
                    session = None

        # Resolve project_root with session-first precedence.
        allow_fallback_cwd = (mode or "").lower() != "edit"
        project_root, pr_source = _resolve_project_root(
            request=request,
            session_obj=session,
            payload_project_root=payload_project_root,
            allow_fallback_cwd=allow_fallback_cwd,
        )

        # For edit/apply workflow: project must be selected (no implicit cwd fallback).
        if (mode or "").lower() == "edit" and not project_root:
            log.debug(
                "jobs_start rejected: mode=edit but no project_root resolved (source=%s)",
                pr_source,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "project_required",
                    "message": "No project selected. Select a project in the UI before applying edits.",
                },
            )

        # Persist project_root + focus back to the session for continuity
        # IMPORTANT: do not overwrite with a fallback cwd.
        if session is not None and project_root:
            try:
                setattr(session, "project_root", project_root)
            except Exception:
                pass

        # Derive focus:
        # 1) explicit args.focus
        # 2) session.focus (if your UI sets it)
        # 3) fallback to the chat message
        focus = ""
        if isinstance(args, dict):
            focus = str(args.get("focus", "") or "").strip()
        if not focus and session is not None:
            focus = str(getattr(session, "focus", "") or "").strip()
        if not focus:
            focus = message.strip()

        # Ensure focus is present in args; mode here is only a hint for chat mode
        if isinstance(args, dict):
            merged: Dict[str, Any] = {**args}
            merged.setdefault("focus", focus)
            args = merged
        else:
            args = {"focus": focus}

        if session is not None:
            try:
                if focus:
                    setattr(session, "focus", focus)
            except Exception:
                pass

        job = JobRegistry.create(
            session_id=session_id,
            project_root=project_root,
            auto_approve=auto_approve,
        )
        progress_cb = _progress_cb_for_job(job.job_id)

        # Emit an immediate status event describing the approval mode so the UI can show it
        mode_label = "on" if auto_approve else "off"
        JobRegistry.publish(
            job.job_id,
            "status",
            {
                "stage": "job.started",
                "auto_approve": auto_approve,
                "detail": f"Auto-approve: {mode_label}",
                "project_root_source": pr_source,
                "project_root": project_root,
            },
        )

        async def runner():
            job.status = "running"
            detach_bridge: Optional[Callable[[], None]] = None
            try:
                # Mirror global events (assistant/status/etc.) into this job stream
                try:
                    detach_bridge = _attach_events_bridge(job)
                except Exception:
                    detach_bridge = None

                if not project_root:
                    raise RuntimeError(
                        "project_root is required to run this job (missing after resolution)"
                    )

                # Build a DevBotAPI for this job/session so chat tools can do real work.
                api = DevBotAPI(
                    project_root=Path(project_root),
                    cfg=dict(cfg or {}),
                    session_id=session_id,
                    job_id=job.job_id,
                    auto_approve=auto_approve,
                    progress_cb=progress_cb,
                )

                # Initial status
                progress_cb(
                    "status",
                    {"stage": "start", "message": "chat conversation start"},
                )

                # Run the chat conversation in a *worker thread* so we don't
                # block the event loop and can return /jobs/start immediately.
                loop = asyncio.get_running_loop()

                def _run_sync_chat():
                    return run_chat_conversation(
                        session=session,
                        message=message,
                        explicit_mode=mode,
                        registry=None,  # let chat.py build default ToolRegistry(api)
                        api=api,
                    )

                _ = await loop.run_in_executor(None, _run_sync_chat)

                # Final status; UI mainly relies on streamed assistant/status messages
                progress_cb(
                    "status",
                    {"stage": "end", "message": "chat conversation complete"},
                )

                # Complete job. We keep the result payload minimal here; the rich
                # content is carried by SSE events from the chat layer.
                JobRegistry.complete(job.job_id, ok=True)
            except asyncio.CancelledError:
                JobRegistry.cancel(job.job_id)
            except Exception as e:
                log.exception("chat orchestration failed: %s", e)
                progress_cb("error", {"error": "chat_failed", "message": str(e)})
                JobRegistry.complete(job.job_id, ok=False, error=str(e))
            finally:
                if callable(detach_bridge):
                    try:
                        detach_bridge()
                    except Exception:
                        pass

        job.task = asyncio.create_task(runner())
        return JSONResponse(content={"ok": True, "job_id": job.job_id})

    @app.post("/jobs/cancel", tags=["chat"])
    async def jobs_cancel(job_id: str = Query(..., description="Job to cancel")):
        ok = JobRegistry.cancel(job_id)
        return JSONResponse(content={"ok": ok, "job_id": job_id})

    @app.get("/jobs/stream", tags=["events"])
    async def jobs_stream(
        job_id: Optional[str] = Query(default=None),
        session_id: Optional[str] = Query(default=None),
    ):
        """
        Primary: stream by job_id.
        Note: session_id is not supported here (was legacy); pass job_id only.
        """

        if not job_id:
            if session_id:
                return JSONResponse(status_code=400, content={"error": "stream requires job_id"})
            return JSONResponse(status_code=400, content={"error": "job_id required"})

        job = JobRegistry.get(job_id or "")
        if not job:
            return JSONResponse(
                status_code=404, content={"error": "unknown_job", "job_id": job_id}
            )

        return StreamingResponse(_job_event_stream(job), media_type="text/event-stream")

    # ---------- Approval (job-scoped, with session fallback) ----------

    # ---------- Approval helpers (job + optional recommendation) ----------

    def _extract_rec_id(payload: Dict[str, Any]) -> Optional[str]:
        """
        Accept both 'recommendation_id' and 'rec_id' from the client.
        Returns a normalized rec_id or None.
        """

        rid = str(payload.get("recommendation_id") or payload.get("rec_id") or "").strip()
        return rid or None

    async def _decide_approval(job_id: str, approved: bool, rec_id: Optional[str]):
        """
        Core approval handler shared by /jobs/approve and /jobs/reject

        For now we support job-scoped decisions via ApprovalInbox.decide_job.
        Per-recommendation decisions can be added later via decide_rec.
        """

        job = JobRegistry.get(job_id)
        if not job:
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": "unknown_job", "job_id": job_id},
            )

        try:
            # Future: if you add per-rec decisions, implement APPROVALS.decide_rec(...)
            if rec_id and hasattr(APPROVALS, "decide_rec"):
                # type: ignore[attr-defined]
                await APPROVALS.decide_rec(job_id, rec_id, approved)  # pragma: no cover
            elif hasattr(APPROVALS, "decide_job"):
                # type: ignore[attr-defined]
                await APPROVALS.decide_job(job_id, approved)
            else:
                # No supported decision API: fail clearly
                return JSONResponse(
                    status_code=500,
                    content={
                        "ok": False,
                        "error": "approval_not_supported",
                        "message": "ApprovalInbox does not implement decide_job/decide_rec",
                    },
                )

            # Publish a status event so the UI can reflect the decision
            JobRegistry.publish(
                job_id,
                "status",
                {
                    "stage": "approval",
                    "approved": approved,
                    "rec_id": rec_id,
                },
            )
            return JSONResponse(
                content={
                    "ok": True,
                    "job_id": job_id,
                    "approved": approved,
                    "rec_id": rec_id,
                }
            )
        except Exception as e:
            log.exception("approval decision failed: %s", e)
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": "internal_error",
                    "message": str(e),
                },
            )

    @app.post("/jobs/approve", tags=["approval"])
    async def jobs_approve(payload: Dict[str, Any]):
        """
        Body: {
          job_id: str,
          recommendation_id?: str | rec_id?: str
        }
        Signals that the current approval gate for this job (and optional
        recommendation) has been approved.
        """

        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "job_id required"},
            )

        rec_id = _extract_rec_id(payload)
        return await _decide_approval(job_id, approved=True, rec_id=rec_id)

    @app.post("/jobs/reject", tags=["approval"])
    async def jobs_reject(payload: Dict[str, Any]):
        """
        Body: {
          job_id: str,
          recommendation_id?: str | rec_id?: str
        }
        Signals that the current approval gate for this job (and optional
        recommendation) has been rejected.
        """

        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            return JSONResponse(
                status_code=400, content={"ok": False, "error": "job_id required"}
            )

        rec_id = _extract_rec_id(payload)
        return await _decide_approval(job_id, approved=False, rec_id=rec_id)

    # ---------- Run log JSON API ----------

    TRACE_PATH = Path(".aidev") / "trace.jsonl"

    @app.get("/api/run-log", tags=["projects"])
    async def api_run_log(
        project_id: Optional[str] = Query(
            default=None,
            description="Project UUID to filter trace records",
        )
    ):
        """
        Return newline-delimited JSON records from .aidev/trace.jsonl filtered by project_id.

        Behavior:
        - 400 + {"error": "..."} if project_id is missing or invalid.
        - 200 + [] if trace file is missing or no matching records.
        - 200 + JSON array of objects for matching records.
        """

        # Validate presence
        if not project_id:
            return JSONResponse(
                status_code=400,
                content={"error": "project_id required"},
            )

        # Validate UUID format
        try:
            project_uuid = str(UUID(project_id))
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid project_id"},
            )

        records: List[Dict[str, Any]] = []

        if not TRACE_PATH.exists():
            # No trace yet: empty array is success
            return JSONResponse(content=records)

        try:
            with TRACE_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning(
                            "[api_run-log] skipping malformed trace line: %r",
                            line[:200],
                        )
                        continue

                    if str(obj.get("project_id")) == project_uuid:
                        records.append(obj)
        except Exception as exc:
            log.exception("[api_run-log] failed reading trace file: %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": "failed to read trace log"},
            )

        # Always return a JSON array (possibly empty)
        return JSONResponse(content=records)

    # ---------- Projects convenience ----------

    def _discover_projects(base: Path, depth: int = 6) -> List[Dict[str, Any]]:
        markers = [
            "package.json",
            "pyproject.toml",
            "requirements.txt",
            "setup.py",
            "pubspec.yaml",
            "Cargo.toml",
            "manage.py",
            "Pipfile",
            "go.mod",
            ".git",
        ]
        results: List[Dict[str, Any]] = []
        base = base.resolve()
        max_parts = len(base.parts) + depth
        for p in base.rglob("*"):
            try:
                if p.is_dir() and len(p.parts) <= max_parts:
                    found = [m for m in markers if (p / m).exists()]
                    if found:
                        kind = (
                            "python"
                            if any(
                                (p / m).exists()
                                for m in [
                                    "pyproject.toml",
                                    "requirements.txt",
                                    "setup.py",
                                ]
                            )
                            else "node"
                            if (p / "package.json").exists()
                            else "flutter"
                            if (p / "pubspec.yaml").exists()
                            else "repo"
                        )
                        results.append(
                            {
                                "path": str(p),
                                "kind": kind,
                                "markers": found,
                                "children_count": 0,
                            }
                        )
            except Exception:
                continue
        seen = set()
        uniq = []
        for r in results:
            if r["path"] in seen:
                continue
            seen.add(r["path"])
            uniq.append(r)
        return uniq

    def _read_project_descriptions(root: Path) -> tuple[str, str]:
        """
        Read human-written and compiled descriptions for a project.

        Returns:
            (app_description, compiled_md)
        """

        app_description = ""
        compiled_md = ""
        try:
            # Human source-of-truth: .aidev/app_descrip.txt
            candidates_app = [
                root / ".aidev" / "app_descrip.txt",
            ]
            for p in candidates_app:
                if p.exists():
                    app_description = p.read_text(encoding="utf-8").strip()
                    break

            # LLM-compiled markdown: .aidev/project_description.md
            candidates_compiled = [
                root / ".aidev" / "project_description.md",
            ]
            for p in candidates_compiled:
                if p.exists():
                    compiled_md = p.read_text(encoding="utf-8").strip()
                    break
        except Exception as e:
            log.warning("failed to read project descriptions for %s: %s", root, e)

        return app_description, compiled_md

    @app.post("/projects/list", tags=["projects"])
    async def projects_list(payload: Dict[str, Any]):
        root = payload.get("workspace_root") or os.getcwd()
        depth = int(payload.get("depth") or 6)
        try:
            items = _discover_projects(Path(root), depth=depth)
            return JSONResponse(
                content={
                    "ok": True,
                    "workspace_root": str(Path(root).resolve()),
                    "candidates": items,
                }
            )
        except Exception as e:
            log.exception("projects_list failed: %s", e)
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": "internal_error", "message": str(e)},
            )

    @app.post("/projects/create", tags=["projects"])
    async def projects_create(payload: Dict[str, Any]):
        brief = (payload or {}).get("brief", "").strip()
        if not brief:
            return JSONResponse(
                status_code=400, content={"ok": False, "error": "brief required"}
            )
        name = (payload or {}).get("project_name") or f"aidev_{int(time.time())}"
        base_dir = Path((payload or {}).get("base_dir") or (Path.cwd() / "projects")).resolve()
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            proj = base_dir / name
            proj.mkdir(parents=True, exist_ok=True)

            aidev_dir = proj / ".aidev"
            aidev_dir.mkdir(exist_ok=True)

            # Human source-of-truth description
            app_descrip_path = aidev_dir / "app_descrip.txt"
            app_descrip_path.write_text(brief + "\n", encoding="utf-8")

            # LLM-compiled markdown brief (initially derived from brief)
            project_description_md = f"# {name}\n\n{brief}\n"
            compiled_path = aidev_dir / "project_description.md"
            compiled_path.write_text(project_description_md, encoding="utf-8")

            # Return with description fields so the UI can hydrate immediately
            return JSONResponse(
                content={
                    "ok": True,
                    "project": {
                        "path": str(proj),
                        "project_id": str(proj),
                        "app_description": brief,
                        "project_description_md": project_description_md,
                    },
                }
            )
        except Exception as e:
            log.exception("projects_create failed: %s", e)
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": "internal_error", "message": str(e)},
            )

    @app.post("/projects/select", tags=["projects"])
    async def projects_select(payload: Dict[str, Any]):
        """
        Body: { path?: str, project_root?: str, project_path?: str, session_id?: str }
        Stores selected root into the session (if provided) for UI convenience
        and returns description metadata so the UI can populate app_descrip +
        compiled brief panes.
        """

        # Accept canonical + legacy names
        project_path = str(
            payload.get("path")
            or payload.get("project_root")
            or payload.get("project_path")
            or ""
        ).strip()

        if not project_path:
            return JSONResponse(
                status_code=400, content={"ok": False, "error": "project_path required"}
            )

        session_id = (payload or {}).get("session_id") or None
        root = Path(project_path).resolve()
        if not root.exists():
            return JSONResponse(
                status_code=404, content={"ok": False, "error": "path_not_found"}
            )

        if session_id:
            try:
                session = await SESSIONS.ensure(session_id)
                if session:
                    # Write both fields for compatibility across code paths
                    setattr(session, "project_root", str(root))
                    setattr(session, "project_path", str(root))
                    # Also set "root" for ultra-legacy callers if your code uses it
                    try:
                        setattr(session, "root", str(root))
                    except Exception:
                        pass
            except Exception:
                pass

        app_description, compiled_md = _read_project_descriptions(root)

        return JSONResponse(
            content={
                "ok": True,
                "root": str(root),
                "project": {
                    "path": str(root),
                    "project_id": str(root),
                    "app_description": app_description,
                    "project_description_md": compiled_md,
                },
            }
        )

    # Compatibility endpoint: also accept /workspaces/select with identical semantics
    app.post("/workspaces/select", tags=["projects"])(projects_select)

    # ---------- Summaries ----------

    @app.post("/summaries/changed", tags=["summaries"])
    async def summaries_changed(request: Request, payload: Dict[str, Any]):
        """
        Summarize changed files and return per-file statuses.

        Body may include:
          - project_root?: str
          - session_id?: str         # used to resolve project_root if project_root omitted
          - model?: str              # optional LLM model override
          - paths?: list[str]        # optional explicit subset of files to summarize
          - files?: list[str]        # alias for paths
          - compute_embeddings?: bool
          - max_files?: int          # optional cap on files per run

        Returns always-200 JSON:
          {
            ok: bool,
            message: str,
            counts: { summarized: int, skipped: int, failed: int },
            files: [ { path, ok, summary_len, error }, ... ].
          }.
        """

        # Use the KnowledgeBase path so discovery/skip rules match the UI flows
        try:
            from .cards import KnowledgeBase
            from .config import load_project_config
            from .structure import discover_structure
        except Exception:
            return JSONResponse(
                content={
                    "ok": False,
                    "error": "kb_import_failed",
                    "counts": {"summarized": 0, "skipped": 0, "failed": 0},
                    "files": [],
                }
            )

        try:
            log.info("summaries_changed: start payload=%s", safe_json(payload or {}))

            # Resolve project_root so we can later read .aidev/cards/*.card.json
            session = None
            sid = (payload or {}).get("session_id")
            if sid:
                try:
                    session = await SESSIONS.get(sid)
                except Exception:
                    session = None

            pr_str, pr_source = _resolve_project_root(
                request=request,
                session_obj=session,
                payload_project_root=(payload or {}).get("project_root"),
                allow_fallback_cwd=True,
            )

            project_root = Path(pr_str).resolve() if pr_str else Path.cwd().resolve()
            if pr_source == "cwd":
                # Avoid persisting a fallback cwd into session.
                pass

            # Summarize via KnowledgeBase (writes .aidev/cards/*.card.json)
            cfg, _ = load_project_config(project_root, None)
            includes = list((cfg.get("discovery", {}) or {}).get("includes", []))
            excludes = list((cfg.get("discovery", {}) or {}).get("excludes", []))
            struct, _ = discover_structure(
                project_root,
                includes,
                excludes,
                max_total_kb=128,
                strip_comments=False,
            )
            kb = KnowledgeBase(project_root, struct)

            paths = (payload or {}).get("paths") or (payload or {}).get("files")
            compute_embeddings = (payload or {}).get("compute_embeddings")
            max_files = (payload or {}).get("max_files")

            result = kb.summarize_changed(
                paths=paths,
                model=(payload or {}).get("model"),
                compute_embeddings=compute_embeddings,
                max_files=max_files,
            )
            if not isinstance(result, dict):
                resp = {
                    "ok": False,
                    "error": "bad_llm_response",
                    "counts": {"summarized": 0, "skipped": 0, "failed": 0},
                    "files": [],
                }
                log.warning("summaries_changed: %s", resp["error"])
                return JSONResponse(content=resp)

            # Build counts directly from KnowledgeBase keys
            counts = {
                "summarized": int(result.get("updated", 0)),
                "skipped": int(result.get("skipped", 0)),
                "failed": int(result.get("failed", 0) or 0),
            }

            files: List[Dict[str, Any]] = list(result.get("results", []))

            resp = {
                "ok": bool(result.get("ok", True)),
                "message": result.get("message", ""),
                "counts": counts,
                "files": files,
            }
            if not resp["ok"]:
                resp["error"] = str(
                    result.get("error")
                    or result.get("message")
                    or "summaries_changed_failed"
                )

            log.info("summaries_changed: done ok=%s counts=%s", resp["ok"], resp["counts"])
            return JSONResponse(content=resp)
        except Exception as e:
            log.exception("summaries_changed: exception")
            return JSONResponse(
                content={
                    "ok": False,
                    "error": str(e),
                    "counts": {"summarized": 0, "skipped": 0, "failed": 0},
                    "files": [],
                }
            )

    # ---------- Pre-apply checks (safe, never throws to client) ----------

    @app.post("/checks/preapply", tags=["checks"])
    async def checks_preapply(request: Request, payload: Dict[str, Any]):
        """
        Body: {
          job_id?: str,
          project_root?: str,
          patches?: [{path:str, diff:str}]   # unified diffs
        }
        Returns: { passed: bool, details: [{runtime, ok, logs_tail, ...}] }
        """

        try:
            job_id = (payload or {}).get("job_id")
            patches = (payload or {}).get("patches") or []

            # Resolve project_root in a consistent way.
            session = None
            sid = (payload or {}).get("session_id")
            # If client didn't include explicit session_id, accept the UI cookie 'session_id' as a
            # secondary/session-selection hint (workspaces.select_project sets this cookie).
            if not sid:
                try:
                    sid = request.cookies.get("session_id")
                except Exception:
                    sid = None

            if sid:
                try:
                    session = await SESSIONS.get(sid)
                except Exception:
                    session = None

            # For preapply: if patches are present this is part of an apply/edit flow and must
            # not fall back to CWD. Enforce allow_fallback_cwd=False in that case.
            allow_fallback_cwd = not bool(patches)

            pr_str, pr_source = _resolve_project_root(
                request=request,
                session_obj=session,
                payload_project_root=(payload or {}).get("project_root"),
                allow_fallback_cwd=allow_fallback_cwd,
            )

            # If this check is associated with a job and no explicit project_root was
            # provided via session/payload/header, fall back to the job record.
            if job_id and pr_source in ("none", "cwd"):
                job = JobRegistry.get(job_id)
                if job and job.project_root:
                    pr_str = job.project_root

            # If patches were provided and we still don't have a resolved project root,
            # reject the request. This prevents accidental writes into the bot repo.
            if patches and not pr_str:
                return JSONResponse(
                    status_code=400,
                    content={
                        "passed": False,
                        "error": "project_required",
                        "message": "No project selected. Select a project in the UI before applying edits.",
                    },
                )

            if not pr_str:
                pr_str = str(Path.cwd().resolve())

            root_path = Path(pr_str).resolve()

            passed: bool = True
            details: List[Dict[str, Any]] = []

            try:
                # Delegate to the shared runtime helper so Orchestrator and
                # the HTTP endpoint stay in sync.
                res = await asyncio.to_thread(
                    run_preapply_checks,
                    root_path,
                    patches,
                )

                # Normal case: runtime.run_preapply_checks returns (ok, details)
                if isinstance(res, tuple) and len(res) == 2:
                    passed = bool(res[0])
                    raw_details = res[1] or []
                    details = list(raw_details) if isinstance(raw_details, list) else [raw_details]

                # Future-proof: accept dict shape {ok, details} if we ever switch
                elif isinstance(res, dict):
                    passed = bool(res.get("ok", res.get("passed", True)))
                    raw_details = res.get("details") or res.get("runs") or []
                    if isinstance(raw_details, list):
                        details = raw_details
                    elif raw_details:
                        details = [raw_details]

                else:
                    # Fallback: simple truthiness
                    passed = bool(res)

            except Exception as e:
                passed = False
                details.append(
                    {
                        "runtime": "generic",
                        "ok": False,
                        "logs_tail": str(e),
                    }
                )

            # If this check is associated with a job, emit an SSE event so
            # the UI can show the result in the job stream.
            if job_id:
                JobRegistry.publish(
                    job_id,
                    "preapply",
                    {
                        "passed": passed,
                        "details": details,
                        "project_root": str(root_path),
                    },
                )

            return JSONResponse(content={"passed": passed, "details": details})
        except Exception as e:
            # Never throw to client; always return a structured result
            return JSONResponse(
                content={
                    "passed": False,
                    "details": [
                        {
                            "runtime": "internal",
                            "ok": False,
                            "logs_tail": str(e),
                        }
                    ],
                }
            )

    # ---------- Recommendations pass-through ----------

    @app.get("/api/recommendations", tags=["projects"])
    async def api_recommendations():
        try:
            items = load_recommendations()
            return JSONResponse(content={"items": items, "count": len(items)})
        except Exception as e:
            log.exception("api_recommendations failed: %s", e)
            return JSONResponse(
                status_code=500, content={"error": "internal_error", "message": str(e)}
            )

    # ---------- Static UI ----------

    ui_root = static_ui_dir or Path(__file__).parent / "ui"
    if ui_root.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_root), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(ui_root / "index.html"))

    # ---------- Project map convenience ----------

    @app.get("/api/project-map", tags=["projects"])
    async def project_map(
        root: Optional[str] = Query(default=None),
        include: Optional[List[str]] = Query(default=None),
        exclude: Optional[List[str]] = Query(default=None),
        download: bool = Query(default=False),
        force: bool = Query(default=False),
        path: Optional[str] = Query(default=None),
        session_id: Optional[str] = Query(default=None),
    ):
        cfg_local = dict(cfg or {})
        project_root = Path(root).resolve() if root else Path.cwd().resolve()
        st = ProjectState(project_root=project_root)

        progress_cb = None
        if session_id:
            try:
                session = await SESSIONS.ensure(session_id)
                if session:
                    q = session.queue  # type: ignore[attr-defined]
                    loop = asyncio.get_running_loop()

                    def cb(ev: str, payload: Dict[str, Any]):
                        loop.call_soon_threadsafe(
                            q.put_nowait, {"type": ev, **(payload or {})}
                        )

                    progress_cb = cb
            except Exception as e:
                log.warning("Unable to attach session progress: %s", e)

        orch = Orchestrator(
            root=project_root,
            st=st,
            args={
                "include": include or [],
                "exclude": exclude or [],
                "export_project_map": (path or ".aidev/project_map.json"),
                "project_map_only": True,
                "cards_force": bool(force),
                "cfg": cfg_local,
                "progress_cb": progress_cb,
                "session_id": session_id,
            },
        )

        try:
            if progress_cb:
                progress_cb("project_map_start", {"root": str(project_root)})
            await asyncio.to_thread(orch.run)
            if progress_cb:
                progress_cb("project_map_end", {"root": str(project_root)})
        except Exception as e:
            err = {
                "error": "project_map_failed",
                "message": str(e),
                "trace": traceback.format_exc(),
            }
            if progress_cb:
                progress_cb("error", {"where": "project_map", **err})
            return JSONResponse(status_code=500, content=err)

        out = project_root / (path or ".aidev/project_map.json")
        if not out.exists():
            msg = {
                "error": "project_map_missing",
                "message": "project_map.json not generated",
            }
            if progress_cb:
                progress_cb("error", {"where": "project_map", **msg})
            return JSONResponse(status_code=500, content=msg)

        if download:
            return FileResponse(
                str(out), filename="project_map.json", media_type="application/json"
            )
        try:
            return JSONResponse(content=json.loads(out.read_text(encoding="utf-8")))
        except Exception:
            return FileResponse(
                str(out), filename="project_map.json", media_type="application/json"
            )

    return app


app = create_app()


def run(args_ns, cfg) -> int:
    import uvicorn

    _app = create_app(cfg=cfg)
    host = os.getenv("AIDEV_HOST", "127.0.0.1")
    port = int(os.getenv("AIDEV_PORT", "8080"))
    uvicorn.run(
        _app, host=host, port=port, log_level=os.getenv("AIDEV_LOGLEVEL", "info")
    )
    return 0
