# aidev/orchestration/approval_inbox.py
from __future__ import annotations

import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import events


@dataclass
class ApprovalRequest:
    token: str
    session_id: str
    job_id: Optional[str]
    summary: str                # human summary shown to user
    risk: str                   # "low"|"medium"|"high" (UI hint)
    files: List[Dict[str, Any]] # [{path, added, removed, why?}, ...]
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None
    decision: Optional[str] = None     # "approved"|"rejected"|None
    reason: Optional[str] = None       # rejection reason (optional)
    meta: Dict[str, Any] = field(default_factory=dict)


class ApprovalInbox:
    """
    Small approval mailbox keyed by a token.

    NOTE: Internally this is thread-based (threading.Lock/Event) so it works
    correctly across:
      - Orchestrator worker threads (using asyncio.run to call request/wait)
      - The FastAPI/uvicorn event loop (calling decide_job)

    API remains async for compatibility:

        token = await APPROVALS.request(...)
        req   = await APPROVALS.wait(token, timeout=3600)

        await APPROVALS.decide_job(job_id, approved=True)
    """

    def __init__(self) -> None:
        # token -> ApprovalRequest
        self._items: Dict[str, ApprovalRequest] = {}
        # token -> threading.Event (signals that a decision was made)
        self._events: Dict[str, threading.Event] = {}
        # Protects _items and _events
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def request(
        self,
        session_id: str,
        *,
        job_id: Optional[str],
        summary: str,
        risk: str,
        files: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new approval request and emit UI events.

        Returns:
            token (str): used later with wait(...) and internal routing.
        """
        token = uuid.uuid4().hex
        req = ApprovalRequest(
            token=token,
            session_id=session_id,
            job_id=job_id,
            summary=summary,
            risk=risk,
            files=list(files or []),
            meta=dict(meta or {}),
        )

        ev = threading.Event()
        with self._lock:
            self._items[token] = req
            self._events[token] = ev

        # Fire UI events (compatible with your existing handlers)
        events.approval_summary(summary, risk, files, session_id=session_id)
        events.need_plan_approval(
            [f.get("path") for f in files if isinstance(f, dict) and f.get("path")],
            session_id=session_id,
        )

        # Also put a status line in the timeline
        events.emit_status(
            "awaiting approval",
            stage="approval",
            session_id=session_id,
            recId=job_id or token,
        )

        return token

    async def wait(self, token: str, timeout: Optional[float] = None) -> ApprovalRequest:
        """
        Block until this token has a decision or timeout expires.

        Returns the ApprovalRequest (decision may still be None on timeout).
        """
        with self._lock:
            req = self._items.get(token)
            ev = self._events.get(token)

        if req is None or ev is None:
            raise KeyError(token)

        # Wait using a plain threading.Event; this is safe to call from a
        # worker thread even inside asyncio.run(...)
        if timeout is not None:
            ev.wait(timeout)
        else:
            ev.wait()

        # On timeout, decision will still be None; caller decides how to handle.
        return req

    async def approve(self, token: str) -> ApprovalRequest:
        return await self._decide(token, "approved")

    async def reject(self, token: str, reason: Optional[str] = None) -> ApprovalRequest:
        return await self._decide(token, "rejected", reason=reason)

    async def _decide(
        self,
        token: str,
        decision: str,
        *,
        reason: Optional[str] = None,
    ) -> ApprovalRequest:
        """
        Internal helper: mark the request as approved/rejected and wake waiters.
        """
        with self._lock:
            req = self._items[token]
            req.decision = decision
            req.reason = reason
            req.decided_at = time.time()
            ev = self._events[token]
            ev.set()
            session_id = req.session_id
            rec_id = req.job_id or token

        # Emit a succinct update (outside the lock)
        detail = "approved" if decision == "approved" else f"rejected: {reason or ''}".strip()
        events.emit_status(
            "approval result",
            stage="approval",
            detail=detail,
            session_id=session_id,
            recId=rec_id,
        )
        return req

    async def get(self, token: str) -> ApprovalRequest:
        with self._lock:
            return self._items[token]

    async def decide_job(
        self,
        job_id: str,
        approved: bool,
        reason: Optional[str] = None,
    ) -> ApprovalRequest:
        """
        Decide the latest approval request associated with this job_id.

        - If there are pending (undecided) requests for the job, choose the most recent.
        - Otherwise, fall back to the most recent request for that job_id (if any).
        """
        with self._lock:
            candidates = [r for r in self._items.values() if r.job_id == job_id]
            if not candidates:
                raise KeyError(f"no approval request for job_id={job_id!r}")

            pending = [r for r in candidates if r.decision is None]
            target = max(pending or candidates, key=lambda r: r.created_at)
            token = target.token

        if approved:
            return await self.approve(token)
        else:
            return await self.reject(token, reason=reason)


APPROVALS = ApprovalInbox()


def get_approval_inbox() -> ApprovalInbox:
    return APPROVALS
