# aidev/orchestration/registry.py
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import events  # uses emit_status/result/done, etc.


@dataclass
class Job:
    id: str
    session_id: str
    kind: str
    created_at: float = field(default_factory=time.time)
    stage: Optional[str] = None       # e.g., "plan", "checks", "apply"
    message: Optional[str] = None
    progress_pct: Optional[float] = None
    ok: Optional[bool] = None
    artifacts: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class JobRegistry:
    """
    Process-local in-memory job registry.

    Typical flow:
      job = await JOBS.create(session_id, kind="orchestrate", meta={...})
      await JOBS.update(job.id, stage="plan", progress_pct=10, message="planning…")
      await JOBS.finish(job.id, ok=True, summary="All good", artifacts=[".aidev/patch.diff"])
    """
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._by_session: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    async def create(self, session_id: str, *, kind: str, meta: Optional[Dict[str, Any]] = None) -> Job:
        jid = uuid.uuid4().hex
        job = Job(id=jid, session_id=session_id, kind=kind, meta=dict(meta or {}))
        async with self._lock:
            self._jobs[jid] = job
            self._by_session.setdefault(session_id, []).append(jid)

        # Let the UI know the job queued
        events.emit_status(f"{kind}: queued", stage="queued", progress_pct=0, session_id=session_id, recId=jid)
        return job

    async def get(self, job_id: str) -> Job:
        async with self._lock:
            return self._jobs[job_id]

    async def list_for_session(self, session_id: str) -> List[Job]:
        async with self._lock:
            return [self._jobs[jid] for jid in self._by_session.get(session_id, [])]

    async def update(
        self,
        job_id: str,
        *,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        progress_pct: Optional[float] = None,
    ) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            if stage is not None:
                job.stage = stage
            if message is not None:
                job.message = message
            if progress_pct is not None:
                job.progress_pct = progress_pct
            session_id = job.session_id
        # emit after releasing the lock
        events.emit_status(
            message or (stage or "progress"),
            stage=stage,
            progress_pct=progress_pct,
            session_id=session_id,
            recId=job_id,
        )

    async def add_artifacts(self, job_id: str, artifacts: List[str]) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.artifacts.extend(artifacts or [])

    async def finish(
        self,
        job_id: str,
        ok: bool,
        *,
        summary: Optional[str] = None,
        artifacts: Optional[List[str]] = None,
    ) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            job.ok = ok
            if artifacts:
                job.artifacts.extend(artifacts)
            session_id = job.session_id
            where = job.stage

        events.emit_result(ok, summary=summary, artifacts=artifacts, where=where, session_id=session_id)
        events.done(ok=ok, where=where, session_id=session_id)


# A simple process-local singleton (mirrors your SessionStore pattern)
JOBS = JobRegistry()

def get_job_registry() -> JobRegistry:
    return JOBS