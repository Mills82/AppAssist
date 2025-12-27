# aidev/cards_cli.py
"""
DEPRECATED: developer-only helper.

This CLI is kept temporarily for dev workflows, but end users should use:
  - `aidev serve`  (web UI / REST)
  - `aidev chat`   (conversational TUI)
  - or run the orchestrator via `aidev` flags directly.

What this does now:
  - forwards to the orchestrator to refresh the cards cache
    (.aidev/cards/index.json) and then exits.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .state import ProjectState
from .orchestrator import Orchestrator

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cards-cli (deprecated)")
    p.add_argument("--project-root", default="", help="Project root (defaults to CWD)")
    p.add_argument("--include", nargs="*", default=[], help="Glob(s) to include")
    p.add_argument("--exclude", nargs="*", default=[], help="Glob(s) to exclude")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    # Back-compat command name; no-op but accepted to avoid script breakage
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("refresh-cards", help="(deprecated) Refresh cards by delegating to orchestrator")
    return p

def _deprecation_banner() -> None:
    logging.warning(
        "cards_cli.py is deprecated. Use `aidev serve` (web) or `aidev chat` (TUI). "
        "This command now forwards to the orchestrator to refresh the cards cache."
    )

def _refresh_via_orchestrator(project_root: Path, includes: List[str], excludes: List[str]) -> int:
    st = ProjectState(project_root=project_root)
    # We only want structure scan + kb.update_cards() and to stop BEFORE the LLM pipeline.
    # Orchestrator.run() calls kb.update_cards() early and returns if targets_only=True.
    orch = Orchestrator(
        root=project_root,
        st=st,
        args={
            "include": includes,
            "exclude": excludes,
            "targets_only": True,   # ensures we stop after structure/cards refresh
            "max_context_kb": 1024,
            "strip_comments": False,
            "fix_all": False,
            "dry_run": True,        # belt-and-suspenders; we’re not applying edits here
        },
    )
    try:
        orch.run()
        logging.info("Cards cache refreshed at .aidev/cards/index.json")
        return 0
    except Exception as e:
        logging.exception("Failed to refresh cards via orchestrator: %s", e)
        return 1

def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    _deprecation_banner()

    root = Path(args.project_root).resolve() if args.project_root else Path.cwd().resolve()
    includes = list(args.include or [])
    excludes = list(args.exclude or [])

    return _refresh_via_orchestrator(root, includes, excludes)

if __name__ == "__main__":
    raise SystemExit(main())
