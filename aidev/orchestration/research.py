"""
aidev/orchestration/research.py

Helper to expand a KnowledgeBase-selected card set and produce an
enriched, bounded QA payload for a single retry when QA is low-confidence
or schema parsing fails.

This module is pure (no LLM calls), deterministic, and imposes hard caps
on total cards and per-snippet sizes to keep LLM payloads bounded.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging

# Conservative per-snippet limit to avoid oversized LLM payloads.
_PER_SNIPPET_CHAR_LIMIT = 2000

logger = logging.getLogger(__name__)


def _truncate(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_select_cards(question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Attempt to call a module-level aidev.cards.select_cards(question, top_k) if present.

    Returns an ordered list of card dicts with at least keys: id, path, summary, score (where available).

    NOTE: We intentionally avoid attempting to call KnowledgeBase.select_cards on the class object
    (i.e. unbound methods) from module scope. Calling an unbound instance method without an instance
    may fail or require internal state; callers that have a KnowledgeBase instance should call it
    directly or pass that instance into a helper. Prefer the simple module-level select_cards when
    available.
    """
    try:
        from aidev import cards as _cards_mod  # type: ignore
    except Exception:
        return []

    # Prefer a top-level function on the module. If present and callable, call it.
    module_fn = getattr(_cards_mod, "select_cards", None)
    if callable(module_fn):
        try:
            if top_k is None:
                return list(module_fn(question))
            return list(module_fn(question, top_k=top_k))
        except Exception:
            return []

    # No safe module-level selector available; do not attempt to call class-bound methods here.
    return []


def _get_cards_runner(orchestrator_or_client: Any):
    """Prefer orchestrator_or_client.cards_runner if present; otherwise try aidev.cards_runner."""
    if orchestrator_or_client is not None:
        cr = getattr(orchestrator_or_client, "cards_runner", None)
        if cr is not None:
            return cr
    try:
        from aidev import cards_runner as _cards_runner  # type: ignore

        return _cards_runner
    except Exception:
        return None


def _expand_neighbors(cards: Iterable[Dict[str, Any]], cards_runner: Any, max_cards: int) -> List[Dict[str, Any]]:
    """Given an iterable of card dicts, try to fetch neighbor cards/snippets via cards_runner.

    The function is defensive: it tries common runner methods (get_neighbors, neighbors, expand) and
    falls back to using whatever snippet/summary exists on the card itself. Deduplicates by id/path.
    Stops when max_cards reached.
    """
    out: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_paths = set()

    def _add_card(card: Dict[str, Any]):
        cid = card.get("id")
        path = card.get("path") or card.get("filepath") or card.get("file")
        if cid and cid in seen_ids:
            return False
        if path and path in seen_paths:
            return False
        if cid:
            seen_ids.add(cid)
        if path:
            seen_paths.add(path)
        out.append(card)
        return True

    for card in cards:
        if len(out) >= max_cards:
            break
        # Add the original card first (if it has at least id/path)
        _add_card(
            {
                "id": card.get("id"),
                "path": card.get("path"),
                "summary": card.get("summary") or card.get("title") or "",
                "snippet": _truncate(card.get("snippet") or card.get("summary") or "", _PER_SNIPPET_CHAR_LIMIT),
                "score": card.get("score"),
            }
        )

        if len(out) >= max_cards:
            break

        # Try to fetch neighbors using common runner APIs
        if cards_runner is None:
            continue

        neighbor_methods = [
            getattr(cards_runner, "get_neighbors", None),
            getattr(cards_runner, "neighbors", None),
            getattr(cards_runner, "expand", None),
            getattr(cards_runner, "fetch_neighbors", None),
        ]
        neighbor_fn = None
        for fn in neighbor_methods:
            if callable(fn):
                neighbor_fn = fn
                break

        if neighbor_fn is None:
            continue

        try:
            # Some APIs expect (card_id, top_k) or (path, k). Try both defensively.
            card_id = card.get("id")
            path = card.get("path")
            neighbors = []
            if card_id is not None:
                try:
                    neighbors = neighbor_fn(card_id, top_k=2)
                except TypeError:
                    try:
                        neighbors = neighbor_fn(card_id, 2)
                    except Exception:
                        neighbors = []
            elif path is not None:
                try:
                    neighbors = neighbor_fn(path, top_k=2)
                except TypeError:
                    try:
                        neighbors = neighbor_fn(path, 2)
                    except Exception:
                        neighbors = []

            # Normalize neighbor entries into card-like dicts
            for n in (neighbors or []):
                if len(out) >= max_cards:
                    break
                if not isinstance(n, dict):
                    # If neighbor is a (path, score) tuple, normalize
                    if isinstance(n, (list, tuple)) and len(n) >= 1:
                        npath = n[0]
                        nscore = n[1] if len(n) >= 2 else None
                        cand = {"id": None, "path": npath, "summary": "", "snippet": "", "score": nscore}
                        _add_card(cand)
                        continue
                    else:
                        continue
                snippet = _truncate(
                    n.get("snippet") or n.get("content") or n.get("summary") or "", _PER_SNIPPET_CHAR_LIMIT
                )
                cand = {
                    "id": n.get("id"),
                    "path": n.get("path") or n.get("filepath") or n.get("file"),
                    "summary": n.get("summary") or "",
                    "snippet": snippet,
                    "score": n.get("score"),
                }
                _add_card(cand)
        except Exception:
            # Be defensive: neighbor expansion failures should not abort the whole routine.
            logger.debug("neighbor expansion failed for card=%r", card, exc_info=True)
            continue

    # Ensure we don't exceed max_cards
    return out[:max_cards]


def build_bounded_evidence(
    cards: Iterable[Dict[str, Any]],
    orchestrator_or_client: Any = None,
    *,
    max_items: int = 100,
    max_total_chars: Optional[int] = None,
    per_item_char_limit: int = _PER_SNIPPET_CHAR_LIMIT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (evidence_list, budget_summary).

    This is a pure, deterministic helper for Deep Research Phase 2 (GATHER).

    Evidence is built by expanding the provided cards (and their neighbor cards, when available)
    using the runner returned by _get_cards_runner. Evidence ordering is deterministic:
    it preserves the ordering of the input iterable and the order produced by _expand_neighbors.

    Budgets:
      - max_items caps the number of evidence items.
      - per_item_char_limit caps the snippet size per item.
      - max_total_chars (if set) caps the total number of characters across all snippets; if the next
        item would exceed the cap, that item's snippet is truncated to fit and the list is ended.

    evidence_list: ordered list of
      {id, path, summary, snippet, source, truncated?: bool, truncated_reason?: str}

    budget_summary includes:
      {
        "requested": {"max_items": ..., "max_total_chars": ..., "per_item_char_limit": ...},
        "applied": {"max_items": ..., "max_total_chars": ..., "per_item_char_limit": ...},
        "counts": {"items_returned": ..., "total_chars": ...},
        "truncated": bool,
        "truncation_point": int | None
      }

    Provenance rules:
      - Uses only repo-relative path values provided in card dicts ("path") and card ids.
      - Does not introduce absolute filesystem paths.
    """

    def _classify_source(card: Dict[str, Any]) -> str:
        # Heuristic: treat items with an id as originating from a "card"; items without an id as
        # neighbor expansions (often normalized from tuples or minimal dicts).
        return "card" if card.get("id") is not None else "neighbor"

    cards_runner = _get_cards_runner(orchestrator_or_client)
    candidates = _expand_neighbors(cards, cards_runner, max_cards=max_items)

    evidence: List[Dict[str, Any]] = []
    total_chars = 0
    truncated_any = False
    truncation_point: Optional[int] = None

    for idx, c in enumerate(candidates):
        # Normalize and per-item truncate first.
        raw_snippet = c.get("snippet") or c.get("content") or c.get("summary") or ""
        snippet = _truncate(raw_snippet, per_item_char_limit)

        item: Dict[str, Any] = {
            "id": c.get("id"),
            "path": c.get("path"),
            "summary": c.get("summary") or "",
            "snippet": snippet,
            "source": _classify_source(c),
        }

        if max_total_chars is None:
            evidence.append(item)
            total_chars += len(snippet)
            continue

        remaining = max_total_chars - total_chars
        if remaining <= 0:
            truncated_any = True
            truncation_point = truncation_point if truncation_point is not None else idx
            break

        if len(snippet) <= remaining:
            evidence.append(item)
            total_chars += len(snippet)
            continue

        # This item would exceed the total budget. Truncate this snippet to fit and stop.
        fit_snippet = snippet[:remaining]
        item["snippet"] = fit_snippet
        item["truncated"] = True
        item["truncated_reason"] = "budget"
        evidence.append(item)
        total_chars += len(fit_snippet)
        truncated_any = True
        truncation_point = idx
        break

    budget_summary: Dict[str, Any] = {
        "requested": {
            "max_items": max_items,
            "max_total_chars": max_total_chars,
            "per_item_char_limit": per_item_char_limit,
        },
        "applied": {
            "max_items": max_items,
            "max_total_chars": max_total_chars,
            "per_item_char_limit": per_item_char_limit,
        },
        "counts": {"items_returned": len(evidence), "total_chars": total_chars},
        "truncated": truncated_any,
        "truncation_point": truncation_point,
    }

    return evidence, budget_summary


def research_and_retry(
    question: str,
    orchestrator_or_client: Any,
    original_cards: Iterable[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    max_cards: int = 10,
) -> Tuple[Dict[str, Any], bool]:
    """Build an enriched QA payload by expanding selected KB cards and their neighbors.

    Args:
        question: user question string.
        orchestrator_or_client: orchestrator object which may expose cards_runner attribute.
        original_cards: iterable of card dicts previously selected for QA (ordered). Each card is expected
            to have at least keys like 'id', 'path', 'summary'.
        top_k: optional integer passed to KnowledgeBase.select_cards to get more candidates.
        max_cards: hard cap on the total number of cards/snippets returned.

    Returns:
        (enriched_payload, did_retry)
        enriched_payload: dict with keys: question, cards (list of {id,path,summary,snippet}), metadata dict.
        did_retry: True iff this call produced expanded context beyond the original_cards (or when original
                   cards were empty but we attempted an expansion). Caller is responsible for enforcing
                   at-most-one retry.

    Notes:
        - This function does not call any LLM.
        - It is idempotent and stateless: same inputs -> same outputs.
        - Ordering from KnowledgeBase.select_cards is preserved for initial candidates.
    """
    # Normalise original_cards into a list of dicts
    orig_list = [c if isinstance(c, dict) else {"id": None, "path": str(c), "summary": ""} for c in original_cards]
    orig_count = len(orig_list)

    # Try to get additional candidates from KnowledgeBase.select_cards
    selector_top_k = top_k if top_k is not None else max(5, orig_count * 2 if orig_count > 0 else 5)
    try:
        selected = _safe_select_cards(question, top_k=selector_top_k)
    except Exception:
        selected = []

    # If select_cards returned empty and there were no original cards, we still signal a retry attempt
    # by setting did_attempt_expand=True (caller can decide to retry exactly once).
    did_attempt_expand = False

    # Merge original_list and selected while preserving ordering and deduping by id/path
    merged_iterable: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_paths = set()

    def _push(card: Dict[str, Any]):
        cid = card.get("id")
        path = card.get("path")
        if cid and cid in seen_ids:
            return
        if path and path in seen_paths:
            return
        if cid:
            seen_ids.add(cid)
        if path:
            seen_paths.add(path)
        merged_iterable.append(card)

    for c in orig_list:
        _push(c)

    for c in (selected or []):
        _push(c)

    # If selected added any new cards beyond originals, we consider this an expansion attempt
    selected_ids = {c.get("id") for c in (selected or []) if c}
    orig_ids = {c.get("id") for c in orig_list if c}
    added_by_selection = len(selected_ids - orig_ids) > 0
    if added_by_selection or (not selected and not orig_list):
        did_attempt_expand = True

    # Use cards_runner to fetch neighbor cards/snippets
    cards_runner = _get_cards_runner(orchestrator_or_client)

    # Expand neighbors but cap total cards
    expanded_cards = _expand_neighbors(merged_iterable, cards_runner, max_cards=max_cards)

    # Decide whether this produced additional useful context beyond original_cards
    expanded_ids = {c.get("id") for c in expanded_cards if c}
    orig_ids = {c.get("id") for c in orig_list if c}

    did_retry = False
    if len(expanded_cards) > orig_count:
        # We have more items than before -> indicate retry-worthy expansion
        did_retry = True
    elif not orig_list and (expanded_cards or selected):
        # No original cards but we attempted an expansion (even if empty), allow one retry
        did_retry = True
    elif added_by_selection and (expanded_cards or selected):
        did_retry = True

    # Build final payload in the minimal shape expected by qa_mixin: question, cards[], metadata
    normalized_cards = []
    for c in expanded_cards:
        normalized_cards.append(
            {
                "id": c.get("id"),
                "path": c.get("path"),
                "summary": c.get("summary") or "",
                "snippet": _truncate(c.get("snippet") or c.get("summary") or "", _PER_SNIPPET_CHAR_LIMIT),
                "score": c.get("score"),
            }
        )

    payload: Dict[str, Any] = {
        "question": question,
        "cards": normalized_cards,
        "metadata": {
            "orig_count": orig_count,
            "expanded_count": len(normalized_cards),
            "guard": "research_v1",
            "selected_added_new": added_by_selection,
            "did_attempt_expand": did_attempt_expand,
            "expanded_ids_count": len(expanded_ids),
        },
    }

    return payload, did_retry
