"""aidev.concurrency

Small asyncio helper(s) to run per-item async work concurrently with a configurable cap,
while returning results in stable input order and recording deterministic per-item
failures.

This module intentionally uses only Python stdlib (asyncio, typing, traceback) and
exposes a single public function:

    run_concurrently_ordered(items, worker, max_concurrency=None, *, key_fn=None)

By default, when callers do not pass an explicit max_concurrency, this module uses
the shared CARD_SUMMARIZE_CONCURRENCY value from aidev.config (so callers may still
override by passing max_concurrency explicitly). This centralizes the runtime cap
for common flows such as per-file card_summarize callers.
"""

from __future__ import annotations

import asyncio
import inspect
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from aidev.config import CARD_SUMMARIZE_CONCURRENCY

T = TypeVar("T")
R = TypeVar("R")


def _format_error(exc: BaseException) -> str:
    """Return a compact, reproducible error string for deterministic failure records."""
    # Prefer exception-only formatting to avoid embedding full tracebacks.
    lines = traceback.format_exception_only(type(exc), exc)
    return "".join(lines).strip() or repr(exc)


async def run_concurrently_ordered(
    items: Sequence[T],
    worker: Callable[[T], Union[R, Awaitable[R]]],
    max_concurrency: Optional[int] = None,
    *,
    key_fn: Optional[Callable[[T], Any]] = None,
) -> Tuple[List[Optional[R]], List[Dict[str, Any]]]:
    """Run `worker(item)` for each item concurrently, capped by `max_concurrency`.

    - Results are returned in the same order as `items` regardless of completion timing.
    - Failures are recorded per-item and do not raise.

    By default (when max_concurrency is None) this function uses the shared
    CARD_SUMMARIZE_CONCURRENCY from aidev.config so callers participating in the
    card_summarize flow share a single source-of-truth concurrency cap. Callers may
    still override by passing an explicit max_concurrency.

    Args:
        items: Input items to process.
        worker: Async callable (or callable returning an awaitable) invoked per item.
        max_concurrency: If provided, limits the number of in-flight worker calls.
        key_fn: Optional function to extract a stable key (e.g., path) for failure records.

    Returns:
        (results, failures)

        results: list of length len(items); each entry is the worker result, or None on failure.
        failures: deterministic list of dicts with at least {index, key, error}.
    """

    # Use configured default cap when callers do not provide one.
    if max_concurrency is None:
        # Ensure the imported config is used as default; fall back to None semantics
        # only if the config value is not set (but config should provide an int).
        max_concurrency = CARD_SUMMARIZE_CONCURRENCY

    sem: Optional[asyncio.Semaphore]
    if max_concurrency is None:
        sem = None
    else:
        # Treat non-positive values as "no concurrency" (i.e., serialize).
        # This is safer than raising and still deterministic.
        sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    results: List[Optional[R]] = [None] * len(items)
    failures: List[Dict[str, Any]] = []

    async def _worker_wrapper(i: int, item: T) -> None:
        if sem is not None:
            await sem.acquire()
        try:
            maybe_awaitable = worker(item)
            # Use inspect.isawaitable to support any awaitable (coroutine/Future/Task/custom awaitable).
            results[i] = await maybe_awaitable if inspect.isawaitable(maybe_awaitable) else maybe_awaitable
        except Exception as e:  # noqa: BLE001 (intentional: record failures, don't raise)
            # Include structured fields (type/message) plus a formatted summary for deterministic consumers.
            failures.append(
                {
                    "index": i,
                    "key": key_fn(item) if key_fn is not None else None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error": _format_error(e),
                }
            )
        finally:
            if sem is not None:
                sem.release()

    tasks = [asyncio.create_task(_worker_wrapper(i, item)) for i, item in enumerate(items)]

    # Use gather with return_exceptions=True for robustness, although wrapper absorbs exceptions.
    await asyncio.gather(*tasks, return_exceptions=True)

    failures.sort(key=lambda f: f.get("index", -1))
    return results, failures


__all__ = ["run_concurrently_ordered"]