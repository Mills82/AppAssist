"""
aidev/routes/frontend.py

Small FastAPI router that provides a lightweight frontend API shim for local/dev
use. Exposes health, session, auth-mock, conversation and llm endpoints.

Endpoints are safe/stubbed by default and will attempt a best-effort proxy to
existing aidev.api.conversation and aidev.api.llm handlers when available.

This file was extended to add a small UI config endpoint, a /session/new alias
and an EventSource-compatible /events SSE endpoint to support the clients/web
Next.js scaffold during local development. Mount this router under the '/api/v1'
prefix in aidev/server.py so the effective runtime endpoints become
/api/v1/ui/config, /api/v1/session/new, etc. See clients/web/API_CONTRACT.md for
the documented API contract (to be added in the clients/web directory).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Minimal additional imports to support SSE and JSON payloads for the UI
import json
import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

logger = logging.getLogger("aidev.routes.frontend")
logger.addHandler(logging.NullHandler())

router = APIRouter()
frontend_router = router

# -----------------------------
# Pydantic models (API contract)
# -----------------------------

class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str
    expires_in: int  # seconds
    metadata: Dict[str, Any] = {}


class AuthMockRequest(BaseModel):
    provider: Optional[str] = "mock"
    user_id: Optional[str] = None


class AuthMockAPIResponse(BaseModel):
    token: str
    user: Dict[str, Any]


class AuthMockResponse(BaseModel):
    id: str
    name: str
    email: str
    roles: List[str] = []
    metadata: Dict[str, Any] = {}


class ConversationRequest(BaseModel):
    # Accept both older 'input' field and newer 'text' field for compatibility
    input: Optional[str] = None
    text: Optional[str] = None
    session_id: Optional[str] = None
    mode: Optional[str] = "auto"  # e.g., auto, manual
    metadata: Dict[str, Any] = {}


class ConversationResponse(BaseModel):
    id: str
    session_id: Optional[str] = None
    status: str
    response: str
    proxied: bool = False
    meta: Dict[str, Any] = {}


class ConversationAPIResponse(BaseModel):
    conversation_id: str
    messages: List[Dict[str, Any]]
    debug: Dict[str, Any] = {}


class LLMRequest(BaseModel):
    prompt: str
    model: Optional[str] = "local-stub"
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 256
    metadata: Dict[str, Any] = {}


class LLMResponse(BaseModel):
    text: str
    model: str
    usage: Dict[str, Any] = {}
    proxied: bool = False


class LLMAPIRequest(BaseModel):
    session_id: Optional[str] = None
    prompt: str
    options: Dict[str, Any] = {}


class LLMAPIResponse(BaseModel):
    result: str
    type: str
    meta: Dict[str, Any] = {}


# -----------------------------
# Helpers
# -----------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _maybe_await(result: Any) -> Any:
    """Await if result is awaitable/coroutine, otherwise return directly."""
    if inspect.isawaitable(result):
        return await result
    return result


def _try_import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        logger.debug("optional import failed: %s -> %s", name, e)
        return None


async def _attempt_proxy(module_name: str, candidate_funcs: List[str], payload: Any) -> Tuple[Optional[Any], Optional[str]]:
    """
    Try to import module_name and call the first available function from
    candidate_funcs with payload. Returns (result, function_name) or
    (None, None) if proxy not available.
    """
    module = _try_import_module(module_name)
    if module is None:
        return None, None

    for fname in candidate_funcs:
        fn = getattr(module, fname, None)
        if fn and callable(fn):
            try:
                # Attempt to call. If fn expects a dict, pass payload.dict();
                # if it expects a pydantic model or raw payload, pass as-is.
                call_arg = None
                try:
                    sig = inspect.signature(fn)
                    # simple heuristic: if function expects a single parameter,
                    # pass payload (use dict() for pydantic models)
                    if len(sig.parameters) == 0:
                        call_arg = None
                    else:
                        # prefer dict for BaseModel
                        if hasattr(payload, "dict"):
                            call_arg = payload.dict()
                        else:
                            call_arg = payload
                except Exception:
                    # fallback
                    call_arg = payload.dict() if hasattr(payload, "dict") else payload

                if call_arg is None:
                    result = fn()
                else:
                    result = fn(call_arg)

                result = await _maybe_await(result)
                return result, fname
            except Exception as e:
                # If proxy function exists but errors, raise to let handler decide.
                logger.exception("proxy function %s.%s raised: %s", module_name, fname, e)
                raise
    return None, None


# -----------------------------
# New UI-supporting endpoints
# -----------------------------

# Note: The router should be mounted under '/api/v1' in aidev/server.py so the
# effective endpoints become /api/v1/ui/config, /api/v1/session/new, etc.
# See clients/web/API_CONTRACT.md (to be added) for the documented contract.

@router.get("/ui/config")
async def ui_config() -> Dict[str, Any]:
    """Return a small UI configuration describing stable endpoint paths and
    a local-dev note about CORS/origin. Clients can read these values at
    runtime to discover the backend surface for local development.
    """
    prefix = "/api/v1"  # recommended mounting prefix in aidev/server.py
    cfg = {
        "prefix": prefix,
        "endpoints": {
            "ui_config": f"{prefix}/ui/config",
            "session_new": f"{prefix}/session/new",
            "session": f"{prefix}/session",
            "auth_mock": f"{prefix}/auth/mock",
            "conversation": f"{prefix}/conversation",
            "llm": f"{prefix}/llm",
            "events": f"{prefix}/events",
        },
        "notes": "Mount this router under '/api/v1' and add a permissive local-dev CORS origin (http://localhost:3000).",
    }
    return cfg


@router.post("/session/new", response_model=SessionCreateResponse)
async def create_session_new() -> SessionCreateResponse:
    """Alias for /session to support UI expectations (returns same shape).

    Delegates to the existing create_session() implementation to avoid
    duplicating logic and to preserve behavior.
    """
    return await create_session()


async def _sse_generator(request: Request):
    """Async generator that yields an initial SSE event and then periodic
    keepalive comments until the client disconnects. Events are formatted
    per the SSE spec (id/event/data) and separated by double newlines.
    """
    # initial sample event
    payload = {"type": "init", "time": _now_iso(), "msg": "sample event"}
    try:
        yield f"id: {uuid.uuid4().hex}\nevent: message\ndata: {json.dumps(payload)}\n\n"
        # keepalive loop
        while True:
            if await request.is_disconnected():
                break
            # SSE comment as keepalive
            yield ": keepalive\n\n"
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        return


@router.get("/events")
async def events(request: Request):
    """EventSource-compatible endpoint for local/dev. Clients can open an
    EventSource to this endpoint to receive server-sent events. This is a
    stubbed stream intended for UI development and testing.
    """
    return StreamingResponse(_sse_generator(request), media_type="text/event-stream")


# -----------------------------
# Endpoints (existing)
# -----------------------------

@router.get("/health")
async def health() -> Dict[str, Any]:
    """Basic health check for the frontend-proxy service."""
    return {"ok": True, "service": "frontend-proxy", "time": _now_iso()}


@router.post("/session", response_model=SessionCreateResponse)
async def create_session() -> SessionCreateResponse:
    """Create a lightweight session id for the UI (no persistence).

    This is intentionally simple for local/dev flows.
    """
    sid = uuid.uuid4().hex
    created = _now_iso()
    resp = SessionCreateResponse(session_id=sid, created_at=created, expires_in=60 * 60 * 24, metadata={})
    logger.debug("created session: %s", sid)
    return resp


@router.post("/auth/mock", response_model=AuthMockAPIResponse)
async def auth_mock(req: AuthMockRequest) -> AuthMockAPIResponse:
    """Return a mocked authenticated user object for local/dev UI.

    This endpoint accepts a small JSON payload like {"provider":"mock","user_id":"u1"}
    and returns a token and a minimal user object. Intended for local-dev only.
    """
    uid = req.user_id or ("user_" + uuid.uuid4().hex[:8])
    token = f"mock-token-{uid}"
    user = {"id": uid, "email": f"{uid}@example.local"}
    logger.debug("auth mock for %s via provider=%s", uid, req.provider)
    return AuthMockAPIResponse(token=token, user=user)


@router.post("/conversation", response_model=ConversationAPIResponse)
async def conversation_endpoint(req: ConversationRequest) -> ConversationAPIResponse:
    """Handle a conversation request. Try proxying to aidev.api.conversation when
    available; otherwise return a deterministic stub response that matches the
    UI contract: {conversation_id, messages[], debug}.
    """
    # Normalize text content from either 'text' or legacy 'input'.
    content = (req.text or req.input or "").strip()
    logger.debug("conversation request: session=%s mode=%s text_len=%d", req.session_id, req.mode, len(content))

    # Candidate function names to try in the conversation module
    candidates = ["handle_conversation", "process_conversation", "create_conversation", "run_conversation", "handle_request"]

    try:
        proxied_result, used_fn = await _attempt_proxy("aidev.api.conversation", candidates, req)
    except Exception as e:
        # Proxy exists but raised an error during execution — surface a 500
        logger.exception("proxy conversation handler failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})

    if proxied_result is None:
        # No proxy available — return a stubbed response shape.
        cid = "conv_" + uuid.uuid4().hex
        messages = [
            {"role": "assistant", "text": ("[stub] This is a local stubbed conversation response. "
                                               "Connect to aidev.api.conversation to enable real behavior.")}
        ]
        debug = {"mode": req.mode, "proxied": False}
        logger.debug("conversation stubbed: %s", cid)
        return ConversationAPIResponse(conversation_id=cid, messages=messages, debug=debug)

    # If proxied_result is available, attempt to normalize it into the UI contract.
    try:
        # If the proxied_result is a dict-like, try to map common keys.
        if isinstance(proxied_result, dict):
            conv_id = proxied_result.get("conversation_id") or proxied_result.get("id") or ("conv_" + uuid.uuid4().hex)
            msgs = proxied_result.get("messages") or [{"role": "assistant", "text": str(proxied_result.get("response", proxied_result.get("text", "")))}]
            debug = {"via": used_fn, **proxied_result.get("debug", {})}
            return ConversationAPIResponse(conversation_id=conv_id, messages=msgs, debug=debug)

        # If the proxied_result is an object with attributes, try to read them.
        if hasattr(proxied_result, "id") or hasattr(proxied_result, "response"):
            conv_id = getattr(proxied_result, "id", "conv_" + uuid.uuid4().hex)
            resp_text = getattr(proxied_result, "response", getattr(proxied_result, "text", ""))
            msgs = [{"role": "assistant", "text": str(resp_text)}]
            debug = {"via": used_fn, "meta": getattr(proxied_result, "meta", {})}
            return ConversationAPIResponse(conversation_id=conv_id, messages=msgs, debug=debug)

        # Fallback: stringified result
        cid = "conv_" + uuid.uuid4().hex
        return ConversationAPIResponse(conversation_id=cid, messages=[{"role": "assistant", "text": str(proxied_result)}], debug={"via": used_fn})
    except Exception as e:
        logger.exception("failed to normalize proxied conversation response: %s", e)
        raise HTTPException(status_code=500, detail={"error": "proxy normalization failed", "msg": str(e)})


@router.post("/llm", response_model=LLMAPIResponse)
async def llm_endpoint(req: LLMAPIRequest) -> LLMAPIResponse:
    """Proxy a simple LLM request to aidev.api.llm when available, otherwise
    return a deterministic stubbed summary. Request shape: {session_id, prompt, options}.
    Response shape: {result, type: 'summary'}.
    """
    logger.debug("llm request: session=%s prompt_len=%d", req.session_id, len(req.prompt or ""))

    candidates = ["call_llm", "invoke_llm", "llm_request", "run_llm"]

    try:
        proxied_result, used_fn = await _attempt_proxy("aidev.api.llm", candidates, req)
    except Exception as e:
        # Proxy exists but errored while executing
        logger.exception("proxy llm handler failed: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})

    if proxied_result is None:
        # Return a simple, safe stubbed summary (no external calls)
        result_text = f"[stubbed summary] {req.prompt[:200]}"
        logger.debug("llm stubbed (summary)")
        return LLMAPIResponse(result=result_text, type="summary", meta={"proxied": False})

    # Normalize proxied result into the simple contract.
    try:
        if isinstance(proxied_result, dict):
            # Prefer explicit keys if present
            res_text = str(proxied_result.get("result") or proxied_result.get("text") or proxied_result.get("output") or "")
            res_type = proxied_result.get("type") or "proxied"
            return LLMAPIResponse(result=res_text, type=res_type, meta={"via": used_fn, **proxied_result.get("meta", {})})

        if hasattr(proxied_result, "text"):
            return LLMAPIResponse(result=str(getattr(proxied_result, "text")), type=getattr(proxied_result, "type", "proxied"), meta={"via": used_fn})

        # Fallback
        return LLMAPIResponse(result=str(proxied_result), type="proxied", meta={"via": used_fn})
    except Exception as e:
        logger.exception("failed to normalize proxied llm response: %s", e)
        raise HTTPException(status_code=500, detail={"error": "proxy normalization failed", "msg": str(e)})


# -----------------------------
# Minimal local runner for quick testing
# -----------------------------
if __name__ == "__main__":
    # This block allows quick manual testing of the router by running this file.
    # It will attempt to start a Uvicorn server on port 8001. If uvicorn is not
    # installed, it will print a short instruction instead.
    try:
        from fastapi import FastAPI
        import uvicorn

        app = FastAPI(title="aidev-frontend-proxy (dev)")
        app.include_router(router, prefix="/api/frontend")

        print("Starting local dev server at http://127.0.0.1:8001/api/frontend/health")
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
    except Exception as e:
        print("Unable to run local server. To test, import 'router' into your FastAPI app or install uvicorn.")
        print("Exception:", e)
