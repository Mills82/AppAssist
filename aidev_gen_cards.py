# aidev_gen_cards.py
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

# -------------------- Logging --------------------

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

# -------------------- Args helpers --------------------

def _resolve_ttl_days(arg: Optional[int]) -> int:
    """
    CLI --ttl-days overrides env AIDEV_CARDS_TTL_DAYS.
    -1  => never force-refresh by TTL (only changed files update)
     0  => default; treat as "no TTL forcing" (changed-only unless enrich scope demands)
    >0  => files with ai_summary_ts older than N days will be re-enriched
    """
    if arg is not None:
        return int(arg)
    env = os.getenv("AIDEV_CARDS_TTL_DAYS", "").strip()
    if not env:
        return 0
    try:
        return max(-1, int(env))
    except Exception:
        return 0

# -------------------- Structure discovery --------------------

_KIND_BY_EXT = {
    # source
    ".py": "source", ".ts": "source", ".tsx": "source", ".js": "source", ".jsx": "source",
    ".java": "source", ".go": "source", ".rs": "source", ".cpp": "source", ".cc": "source",
    ".c": "source", ".h": "source", ".hpp": "source", ".cs": "source", ".php": "source",
    ".rb": "source", ".swift": "source", ".kt": "source",
    # config
    ".json": "config", ".toml": "config", ".ini": "config", ".cfg": "config",
    ".yml": "config", ".yaml": "config",
    # docs
    ".md": "doc", ".mdx": "doc", ".rst": "doc", ".adoc": "doc", ".txt": "doc",
    # styles
    ".css": "styles", ".scss": "styles", ".sass": "styles", ".less": "styles",
    # markup
    ".html": "html", ".xhtml": "html", ".xml": "html",
}

_SKIP_DIRS = {".git", ".aidev", "node_modules", ".venv", "__pycache__", "dist", "build", ".mypy_cache", ".ruff_cache"}

def _simple_discover_structure(root: Path) -> Dict[str, str]:
    """Very small, dependency-free fallback structure discovery."""
    out: Dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_dir():
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        try:
            rel = p.relative_to(root).as_posix()
        except Exception:
            continue
        kind = _KIND_BY_EXT.get(p.suffix.lower(), "other")
        out[rel] = kind
    return out

def _discover_structure(root: Path) -> Dict[str, str]:
    """
    Try repo's richer discovery if available; otherwise use lightweight fallback.
    """
    try:
        # Optional: if your repo provides a richer discoverer (recommended)
        from aidev.structure import discover_structure  # type: ignore
        logging.debug("Using aidev.structure.discover_structure()")
        return discover_structure(root)
    except Exception:
        logging.debug("Falling back to simple structure discovery")
        return _simple_discover_structure(root)

# -------------------- TTL / selection helpers --------------------

def _parse_iso_z(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    s = ts.strip()
    # accept "....Z"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _filter_paths_by_ttl(idx: Dict[str, Dict[str, Any]], *, ttl_days: int) -> List[str]:
    """
    Returns files whose ai_summary_ts is older than ttl_days (or missing).
    ttl_days:
      -1 => never select by TTL (returns [])
       0 => treat as "no TTL forcing" (returns [])
      >0 => select stale/missing
    """
    if ttl_days <= 0:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    stale: List[str] = []
    for rel, meta in idx.items():
        ts = _parse_iso_z(str(meta.get("ai_summary_ts") or ""))
        if ts is None or ts < cutoff:
            stale.append(rel)
    return stale

# -------------------- Main --------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate/refresh AI Dev knowledge cards")

    root_group = p.add_mutually_exclusive_group(required=True)
    root_group.add_argument("--project-root", help="Path to a project root")
    root_group.add_argument("--root", help="Alias for --project-root (for compatibility)")

    p.add_argument("--focus", default=None, help="Optional focus text to guide enrichment & selection")
    p.add_argument(
        "--enrich",
        action="store_true",
        help="Enable LLM enrichment (AI summaries) for this run",
    )
    p.add_argument(
        "--enrich-top-k",
        type=int,
        default=0,
        help=(
            "If >0 and --enrich is set, LLM summarizes top-K focus-relevant files; "
            "if 0, summarizes ALL selected files (respecting TTL/changed logic)."
        ),
    )
    p.add_argument(
        "--no-protect-llm",
        action="store_true",
        help="If set, do not protect prior AI summaries (we will force-refresh targeted files).",
    )
    p.add_argument(
        "--ttl-days",
        type=int,
        default=None,
        help="Re-enrich files with AI summaries older than this many days (default: env AIDEV_CARDS_TTL_DAYS or 0).",
    )

    # Compatibility-only args (accepted, ignored)
    p.add_argument("--top-k", type=int, default=None, help="(Compatibility) Ignored here.")
    p.add_argument("--graph-hops", type=int, default=None, help="(Compatibility) Ignored here.")
    p.add_argument("--budget", type=int, default=None, help="(Compatibility) Ignored here.")

    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    configure_logging(args.verbose)

    root_arg = args.project_root or args.root
    root = Path(root_arg).expanduser().resolve()
    if not root.exists():
        logging.error("Project root does not exist: %s", root)
        sys.exit(2)

    # Keep env signal for downstream components/tools that might read it.
    if args.enrich:
        os.environ["AIDEV_CARD_ENRICH"] = "1"

    ttl_days = _resolve_ttl_days(args.ttl_days)

    # Make project importable, then use the consolidated aidev.cards API.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from aidev.cards import KnowledgeBase  # type: ignore
    except Exception as e:
        logging.error("Failed to import aidev.cards: %s", e)
        sys.exit(2)

    # ----- Discover structure & create KB
    structure = _discover_structure(root)
    kb = KnowledgeBase(root=root, structure=structure)

    # ----- Pass 1: update heuristic cards & graph index (does NOT overwrite ai_* fields)
    logging.info("Updating heuristic cards and graph index…")
    kb.update_cards(force=False, changed_only=True, write_graph_index=True)

    # ----- Optional enrichment (AI summaries)
    if args.enrich:
        try:
            # Load current index to decide targets
            idx = kb.load_card_index()
            all_paths = sorted(structure.keys())

            # Base target set:
            #   - If --focus given and --enrich-top-k>0: select top-K by focus
            #   - Else: target all files
            if args.focus and args.enrich_top_k and args.enrich_top_k > 0:
                logging.info("Selecting top-%d files by focus scoring…", args.enrich_top_k)
                try:
                    # Late import of select_cards to avoid extra imports unless needed
                    from aidev.cards import KnowledgeBase as _KB  # reuse type
                    ranked = kb.select_cards(args.focus, top_k=args.enrich_top_k)
                    targets = [rel for (rel, _s) in ranked]
                except Exception:
                    # Fallback: all files if selection fails
                    targets = all_paths[: args.enrich_top_k]
            else:
                targets = all_paths

            # TTL filter: include files missing/old AI summaries when ttl_days>0
            ttl_targets = set(_filter_paths_by_ttl(idx, ttl_days=ttl_days)) if ttl_days != 0 else set()

            # If protecting LLM content is OFF, we force-refresh the 'targets' set.
            # If protecting is ON (default), we only refresh TTL-stale/missing + changed files,
            # but generate_ai_summaries(changed_only=True) already covers changed files.
            # So here we explicitly add TTL-stale targets and call with changed_only=False for those.
            force_paths: List[str] = []
            if not args.no_protect_llm:
                force_paths = sorted(ttl_targets.intersection(set(targets)))
            else:
                # no-protect => force-refresh exactly 'targets'
                force_paths = sorted(set(targets))

            logging.info(
                "Enrichment plan | focus=%r | top_k=%s | ttl_days=%d | force_count=%d",
                args.focus, (args.enrich_top_k or "ALL"), ttl_days, len(force_paths)
            )

            if force_paths:
                # Force-refresh for selected paths (ignoring changed_only)
                kb.generate_ai_summaries(changed_only=False, paths=force_paths)
            else:
                # No forced set; still refresh changed files to keep things current
                kb.generate_ai_summaries(changed_only=True, paths=None)

            logging.info("AI enrichment complete.")
        except Exception as e:
            logging.error("AI enrichment failed: %s", e)
            sys.exit(1)

    # Note about compatibility flags
    if args.top_k is not None or args.graph_hops is not None or args.budget is not None:
        logging.debug(
            "Note: selection-only flags provided (top_k=%s, graph_hops=%s, budget=%s) are ignored by the card generator.",
            args.top_k, args.graph_hops, args.budget
        )

if __name__ == "__main__":
    main()
