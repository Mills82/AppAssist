# aidev/stages/approval_gate.py
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Approval Inbox (async bridge) - keep lazy/optional to avoid importing heavy
# orchestration modules at import time (which can parse many files).
get_approval_inbox = None  # type: ignore[assignment]

# Events for UI / status updates (lazy import below where used)
_events = None  # type: ignore[assignment]

__all__ = [
    "ApprovalContext",
    "build_approval_payload",
    "evaluate_approval",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diff_stats_from_unified(diff: str) -> Tuple[int, int]:
    """
    Return (added, removed) for a unified diff string.

    We count lines starting with '+' and '-' inside hunks, ignoring headers like
    '+++', '---', 'diff ', 'index ', and '@@'.
    """
    if not diff:
        return 0, 0
    add = rem = 0
    for line in diff.splitlines():
        if not line:
            continue
        # Ignore file headers / metadata
        if line.startswith(("+++", "---", "diff ", "index ", "@@")):
            continue
        if line.startswith("+"):
            add += 1
        elif line.startswith("-"):
            rem += 1
    return add, rem


def _infer_risk_level(files: List[Dict[str, Any]], total_added: int, total_removed: int) -> str:
    """
    Heuristic risk estimate:
      - HIGH: very large edits or touching build/config entrypoints
      - MEDIUM: moderate churn
      - LOW: small localized changes
    """
    risky_paths = (
        "pubspec.yaml",
        "package.json",
        "android/app/build.gradle",
        "ios/Runner.xcodeproj",
        "ios/Runner/Info.plist",
        "build.gradle",
        "settings.gradle",
        "app/build.gradle",
    )

    if total_added + total_removed >= 500:
        return "high"

    for f in files:
        p = str(f.get("path", "")).replace("\\", "/").lower()
        if any(p.endswith(rp) for rp in risky_paths):
            return "high"

    if total_added + total_removed >= 120 or len(files) >= 8:
        return "medium"

    return "low"


def _brief_summary(files: List[Dict[str, Any]], total_added: int, total_removed: int) -> str:
    changed = len(files)
    return f"{changed} file(s) changed, +{total_added}/-{total_removed} lines"


# ---------------------------------------------------------------------------
# Public payload builder
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> List[Any]:
    """
    Normalize an arbitrary value into a list.

    - None -> []
    - list/tuple -> list(value)
    - anything else -> [value]
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_approval_payload(
    proposed: List[Dict[str, Any]],
    *,
    prompts: Optional[List[str]] = None,
    rec_id: Optional[str] = None,
    rec_title: Optional[str] = None,
    self_review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a rich approval payload that the UI can render:

    {
      "summary": str,
      "risk": "low" | "medium" | "high",
      "files": [
        {"path": str, "added": int, "removed": int, "why": str, "bytes": int}
      ],
      "references": {"prompts": [str, ...]},
      "rec_id": str | null,
      "rec_title": str | null,
      "self_review": {
        "overall_status": str | null,
        "warning_count": int,
        "file_update_request_count": int,
        "warnings": [...],
        "file_update_requests": [...],
        "raw": {...}
      }
    }

    In the per-rec flow, `proposed` should already be filtered to a single
    recommendation's edits; `rec_id` and `rec_title` annotate which rec
    this payload corresponds to.
    """
    files: List[Dict[str, Any]] = []
    total_added = total_removed = 0

    for item in (proposed or []):
        path = item.get("path", "")
        diff = item.get("diff", "") or ""
        why = item.get("why", "") or ""
        added, removed = _diff_stats_from_unified(diff)
        total_added += added
        total_removed += removed
        files.append(
            {
                "path": path,
                "added": added,
                "removed": removed,
                "why": why,
                "bytes": int(item.get("preview_bytes") or 0),
            }
        )

    # Derive references (rationales / prompts used)
    uniq_whys: List[str] = []
    for item in (proposed or []):
        w = (item.get("why") or "").strip()
        if w and w not in uniq_whys:
            uniq_whys.append(w)

    if prompts:
        for p in prompts:
            p = (p or "").strip()
            if p and p not in uniq_whys:
                uniq_whys.append(p)

    risk = _infer_risk_level(files, total_added, total_removed)
    summary = _brief_summary(files, total_added, total_removed)

    # Normalize self-review payload so the UI has a stable shape.
    sr = self_review if isinstance(self_review, dict) else None
    if sr is not None:
        warnings = _as_list(sr.get("warnings"))
        file_update_requests = _as_list(sr.get("file_update_requests"))
        overall_status = sr.get("overall_status") or sr.get("status") or None
        self_review_payload = {
            "overall_status": overall_status,
            "warning_count": len(warnings),
            "file_update_request_count": len(file_update_requests),
            "warnings": warnings,
            "file_update_requests": file_update_requests,
            # Keep the original object around so advanced UIs / callbacks can
            # inspect additional fields without us having to promote them all.
            "raw": sr,
        }
    else:
        self_review_payload = None

    return {
        "summary": summary,
        "risk": risk,
        "files": files,
        "references": {"prompts": uniq_whys},
        "rec_id": rec_id,
        "rec_title": rec_title,
        "self_review": self_review_payload,
    }


# ---------------------------------------------------------------------------
# Approval context and async bridge
# ---------------------------------------------------------------------------


@dataclass
class ApprovalContext:
    """
    Optional context the orchestrator can pass so this stage handles all
    default behaviors (Approval Inbox, timeout, etc.) without the orchestrator
    having to wire anything special.

    If `use_inbox` is True (default) and no `approval_cb` is provided,
    we will post to the Approval Inbox and wait up to `timeout_sec`.

    New fields for per-rec + auto-approval:
      - auto_approve: when True, we auto-approve this rec's payload but still
        emit events so the UI can see what would have been approved.
      - rec_id / rec_title: identify the current recommendation.
      - auto_apply_followups: hint flag the UI / approval callback can use to
        auto-apply self-review file_update_requests.
    """
    session_id: Optional[str] = None
    job_id: Optional[str] = None  # Prefer registry job id if you have one
    timeout_sec: Optional[float] = None  # defaults from env
    use_inbox: bool = True

    # Per-rec + auto-approval
    auto_approve: bool = False
    rec_id: Optional[str] = None
    rec_title: Optional[str] = None

    # Hint for self-review follow-ups; orchestrator/UI can set this.
    auto_apply_followups: bool = False


def _env_timeout(default: float = 3600.0) -> float:
    try:
        return float(os.getenv("AIDEV_APPROVAL_TIMEOUT_SEC", str(default)))
    except Exception:
        return default


def _arun(coro):
    """
    Run an async coroutine from sync code.
    If already in an event loop, run the coroutine in a helper thread.
    """
    try:
        # If there's a running loop in this thread, get_running_loop() will
        # succeed; in that case we must not call asyncio.run in this thread,
        # so execute the coroutine via asyncio.run inside a new thread.
        asyncio.get_running_loop()
        out: Dict[str, Any] = {}

        def runner() -> None:
            try:
                out["val"] = asyncio.run(coro)
            except BaseException as e:  # propagate across the thread
                out["err"] = e

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join()
        if "err" in out:
            raise out["err"]  # type: ignore[misc]
        return out.get("val")
    except RuntimeError:
        # No running event loop in this thread: safe to call asyncio.run
        return asyncio.run(coro)


def _default_inbox_decision(
    *,
    payload: Dict[str, Any],
    context: Optional[ApprovalContext],
) -> Optional[bool]:
    """
    If Approval Inbox is available and enabled, post the request and wait.
    Returns:
        True/False for decision, or None if not available / failed.
    """
    if not context or not context.use_inbox:
        return None

    # Lazy import the approval inbox to avoid importing heavy modules at
    # module-import time (which can surface syntax errors in other files).
    try:
        from ..orchestration.approval_inbox import get_approval_inbox as _gi
    except Exception:
        return None

    try:
        inbox = _gi()
    except Exception:
        return None

    try:
        session = context.session_id or "session-unknown"
        job_id = context.job_id or "job-unknown"
        timeout = float(
            context.timeout_sec if context.timeout_sec is not None else _env_timeout()
        )

        async def _request_and_wait():
            token = await inbox.request(
                session_id=session,
                job_id=job_id,
                summary=payload.get("summary"),
                risk=payload.get("risk"),
                files=payload.get("files", []),
            )
            return await inbox.wait(token, timeout=timeout)

        req = _arun(_request_and_wait())
        decision = getattr(req, "decision", None)
        return bool(decision == "approved")
    except Exception:
        # Any failure (network, timeout, etc.) -> None (caller decides fallback)
        return None


# ---------------------------------------------------------------------------
# Main approval gate
# ---------------------------------------------------------------------------

def evaluate_approval(
    *,
    proposed: List[Dict[str, Any]],
    approval_cb: Optional[Callable[..., bool]],
    dry_run: bool = False,
    context: Optional[ApprovalContext] = None,
    self_review: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Central approval gate used by Orchestrator.

    Behavior:
      - Always build a rich per-rec payload (summary, risk, file stats, refs,
        rec_id, rec_title, self_review) for UI / inbox.
      - If context.auto_approve is True:
          * Emit an "auto-approve" event (for UI timeline)
          * Return True so the orchestrator proceeds without blocking.
      - Else if dry_run:
          * Return False (no writes), but the caller may still use the payload
            for display/logging.
      - Else if approval_cb provided:
          * Prefer modern signature: approval_cb(payload) -> bool
          * Fallback to legacy signature: approval_cb(proposed) -> bool
      - Else:
          * If context.use_inbox is True: send to Approval Inbox and wait.
            - On explicit decision: return True/False accordingly.
            - On failure/timeout: return False (safe-by-default).
          * If context is None or use_inbox is False: allow by default (True).

    Returns:
      bool: whether to apply edits.
    """
    rec_id = getattr(context, "rec_id", None) if context else None
    rec_title = getattr(context, "rec_title", None) if context else None
    auto_approve = bool(getattr(context, "auto_approve", False)) if context else False
    auto_apply_followups = bool(
        getattr(context, "auto_apply_followups", False)
    ) if context else False

    # Build payload first so we always have per-rec metadata ready for UI / logs
    payload = build_approval_payload(
        proposed,
        rec_id=rec_id,
        rec_title=rec_title,
        self_review=self_review,
    )
    # Surface the follow-up hint (if any) so the UI / callback can see it.
    if auto_apply_followups:
        payload["auto_apply_followups"] = True

    # Auto-approval short-circuit: still emit an event for the UI timeline
    if auto_approve:
        # Lazy import events to avoid import-time coupling
        try:
            from .. import events as _events_local
        except Exception:
            _events_local = None

        if _events_local is not None:
            try:
                _events_local.status(
                    "approval.auto",
                    where="approval_gate",
                    stage="approval",
                    detail=f"Auto-approving {rec_id or 'current changes'}",
                    rec_id=rec_id,
                    rec_title=rec_title,
                    summary=payload.get("summary"),
                    risk=payload.get("risk"),
                    session_id=(getattr(context, "session_id", None) if context else None),
                    job_id=(getattr(context, "job_id", None) if context else None),
                )
            except Exception:
                # Best-effort only; do not fail approval on event errors
                pass
        return True

    # Dry-run: never actually approve writes
    if dry_run:
        return False

    if approval_cb:
        # Prefer modern signature: approval_cb(payload) -> bool
        try:
            rv = approval_cb(payload)
            return bool(rv)
        except TypeError:
            # Fall back to legacy: approval_cb(proposed) -> bool
            try:
                return bool(approval_cb(proposed))  # type: ignore[misc]
            except Exception:
                return False
        except Exception:
            # If the UI callback errors, be safe and do not apply changes
            return False

    # No callback: try Approval Inbox if configured
    decision = _default_inbox_decision(payload=payload, context=context)
    if decision is not None:
        return bool(decision)

    # Fallback when no context / no inbox: default allow (CLI / tests)
    return True
