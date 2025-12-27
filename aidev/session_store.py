# aidev/session_store.py
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Session:
    id: str
    queue: "asyncio.Queue[dict]" = field(default_factory=asyncio.Queue)
    last_intent: Optional[dict] = None
    last_steps: Optional[List[dict]] = None
    focus: str = ""
    project_path: Optional[str] = None
    # Any state you want across calls (e.g., proposed edits summaries)
    meta: Dict[str, Any] = field(default_factory=dict)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Session:
        async with self._lock:
            sid = str(uuid.uuid4())
            s = Session(id=sid)
            self._sessions[sid] = s
            return s

    async def _create_with_id(self, sid: str) -> Session:
        async with self._lock:
            s = self._sessions.get(sid)
            if s:
                return s
            s = Session(id=sid)
            self._sessions[sid] = s
            return s

    async def get(self, sid: str) -> Session:
        async with self._lock:
            s = self._sessions.get(sid)
            if not s:
                raise KeyError(sid)
            return s

    async def ensure(self, sid: Optional[str]) -> Session:
        if sid:
            try:
                return await self.get(sid)
            except KeyError:
                # Resilience after reload: recreate a fresh session object with same id
                return await self._create_with_id(sid)
        return await self.create()

SESSIONS = SessionStore()

def get_session_store() -> SessionStore:
    return SESSIONS
