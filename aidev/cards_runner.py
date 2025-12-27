# aidev/cards_runner.py
from __future__ import annotations

from typing import Dict, Generator, List, Tuple, Any
from pathlib import Path
import logging

from .cards import Card, KnowledgeBase


def auto_refresh_cards(
    kb: KnowledgeBase,
    *,
    force: bool = False,
    changed_only: bool = True,
) -> Dict[str, Any]:
    """
    Refresh the heuristic cards index on disk/in-memory.

    - Computes/updates file_sha for each tracked file.
    - Preserves any existing ai_summary fields in the index.
    - When changed_only=True (default), only reindexes files that changed.
      Set force=True to rebuild all entries.

    Returns the (possibly updated) cards index dictionary for callers
    that want to read the latest summaries/metadata.
    """
    try:
        kb.update_cards(force=force, changed_only=changed_only)
    except Exception as e:
        # Non-fatal: keep behavior consistent with background refresh
        logging.debug("auto_refresh_cards: kb.update_cards failed: %s", e)

    # Best-effort: return whatever index we have
    try:
        return kb.load_card_index()
    except Exception as e:
        logging.debug("auto_refresh_cards: load_card_index failed: %s", e)
        return {}


def _auto_run_cards_generator(
    kb: KnowledgeBase,
    cards: List[Card],
    *,
    force: bool = False,
    changed_only: bool = True,
):
    """
    Background auto-refresh / selection helper.

    First ensures the heuristic cards index is up-to-date (computes file_sha,
    preserves ai_summary), then yields (card_id, target_paths) for each Card.

    This maintains backward compatibility with previous callers that iterate
    over (card_id, targets) while making sure the index stays fresh.

    Note: This does NOT invoke AI summarization; that is handled by explicit,
    costed endpoints (e.g., /workspaces/ai_summarize_*).
    """
    # Keep the heuristic summaries/index current before selecting targets
    auto_refresh_cards(kb, force=force, changed_only=changed_only)

    for c in cards:
        try:
            # Preserve existing behavior: select candidate paths for each card
            targets = kb.select(c.includes, c.excludes)
            yield (c.id, targets[: getattr(c, "top_k", len(targets))])
        except Exception as e:
            logging.debug("cards_generator: selection failed for card %s: %s", getattr(c, "id", "?"), e)
            yield (getattr(c, "id", "unknown"), [])
