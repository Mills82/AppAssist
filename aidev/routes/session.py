# aidev/routes/session.py
from __future__ import annotations

from typing import Any, MutableMapping, Optional

from fastapi import APIRouter, Depends
from starlette.requests import Request

# Import the SSE SessionStore dependency under a different name
from ..session_store import get_session_store as get_store_dependency, SessionStore

router = APIRouter()


class _NullSession(dict):
    """
    Fallback dict when no Request/SessionMiddleware is available.
    Behaves like a mutable mapping so callers don't crash, but
    changes won't persist anywhere.
    """
    pass


def get_session_store(req: Optional[Request] = None) -> MutableMapping[str, Any]:
    """
    Cookie-session helper expected by other routes.

    - If a Request is provided (normal case), return req.session
      (requires Starlette SessionMiddleware to be installed).
    - If no Request (or SessionMiddleware missing), return a throwaway
      dict to avoid hard crashes. Callers that need persistence should
      pass the actual Request.
    """
    if req is not None:
        try:
            # Starlette attaches a mutable dict here when SessionMiddleware is present
            return req.session  # type: ignore[return-value]
        except AssertionError:
            # SessionMiddleware not installed; fall through to null session
            pass
    return _NullSession()


@router.post("/session/new")
async def new_session(sessions: SessionStore = Depends(get_store_dependency)):
    """
    Create a new SSE session (used by the EventSource stream).
    This is separate from the cookie session used for per-request state.
    """
    s = await sessions.create()
    return {"session_id": s.id}
