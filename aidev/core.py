# aidev/core.py
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .discovery import interactive_pick_projects_if_needed
from .state import ProjectState
from .orchestrator import Orchestrator
from .config import load_env_variables, load_project_config, save_default_config


def _parse_env_bool(val: str | None) -> Optional[bool]:
    if val is None:
        return None
    t = val.strip().lower()
    if t in {"1", "true", "yes", "on"}:
        return True
    if t in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_env_float(val: str | None) -> Optional[float]:
    if val is None or not val.strip():
        return None
    try:
        return float(val.strip())
    except Exception:
        return None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("aidev", add_help=True)

    # Config
    p.add_argument("--config", default="", help="Path to .aidev/config.json (optional)")
    p.add_argument("--init-config", action="store_true", help="Write default .aidev/config.json and exit")

    # Common flags (kept stable so scripts/CI continue to work)
    p.add_argument("--deployment", default=os.getenv("OPENAI_DEPLOYMENT", ""), help="LLM deployment/model (optional)")
    p.add_argument("--project-root", default="", help="Project root (defaults to CWD if empty)")
    p.add_argument("--include", nargs="*", default=[], help="Glob(s) to include")
    p.add_argument("--exclude", nargs="*", default=[], help="Glob(s) to exclude")
    p.add_argument("--select", nargs="*", default=[], help="(Reserved) explicit target files")
    p.add_argument("--max-context-kb", type=int, default=1024, help="Context sampling budget (KB)")
    p.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    p.add_argument("--focus", default="", help="Natural language focus/task")
    p.add_argument("--fix-errors", action="store_true", help="(Reserved) try to auto-fix")
    p.add_argument("--fix-all", action="store_true", help="Apply general line cleanups (collapse blank lines)")
    p.add_argument("--git-snapshot", action="store_true", help="Make a git commit after apply")
    p.add_argument("--no-project-prompt", action="store_true", help="Skip interactive project selection")
    p.add_argument("--project-scan-depth", type=int, default=2, help="Depth for project discovery when none provided")
    p.add_argument("--yes", action="store_true", help="Assume yes for prompts")
    p.add_argument("--no-chat-fallback", action="store_true", help="(Reserved)")
    p.add_argument("--targets-only", action="store_true", help="List targets and exit")
    p.add_argument("--allow-extras", action="store_true", help="(Reserved)")
    p.add_argument("--quiet-http", action="store_true", help="(Reserved)")
    p.add_argument("--cards-top-k", dest="cards_top_k", type=int, default=20, help="Max files per card")
    p.add_argument("--cards-graph-hops", dest="cards_graph_hops", type=int, default=2, help="(Reserved)")
    p.add_argument("--cards-budget", dest="cards_budget", type=int, default=4096, help="LLM budget per card")
    p.add_argument("--enrich", action="store_true", help="(Reserved)")
    p.add_argument("--enrich-top-k", type=int, default=40, help="(Reserved)")
    p.add_argument("--git-push", action="store_true", help="Push after snapshot (if snapshot enabled)")
    p.add_argument("--strip-comments", action="store_true", help="Strip comments in context sampling only")
    p.add_argument("--apply-jsonl", default="", help="Apply code edits from JSONL ({path,content,rec_id?})")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    # QoL mode switches (keeps env-based mode working too)
    p.add_argument("--serve", action="store_true", help="Run the web UI server")
    p.add_argument("--chat", action="store_true", help="Interactive terminal chat mode")
        # Orchestration mode: lets power-users force QA / Analyze / Edit
    p.add_argument(
        "--mode",
        choices=["auto", "qa", "analyze", "edit"],
        default=None,
        help=(
            "Orchestration mode: "
            "'qa' (Q&A only), 'analyze' (plan only, no edits), "
            "'edit' (full recommendations + edits), or 'auto' to let the bot decide. "
            "If omitted, the orchestrator auto-detects per request."
        ),
    )

    # Project map output (structure + per-file summaries)
    p.add_argument(
        "--emit-project-map", "--export-project-map",
        dest="export_project_map",
        nargs="?",
        const=".aidev/project_map.json",
        default=None,
        help="Write structure+summaries to JSON (default: .aidev/project_map.json).",
    )
    p.add_argument(
        "--project-map-only",
        dest="project_map_only",
        action="store_true",
        help="Only emit the project map and exit.",
    )
    p.add_argument(
        "--cards-force",
        action="store_true",
        help="Rebuild card summaries even if hashes unchanged.",
    )

    # LLM stage tunables
    p.add_argument("--llm-timeout-targets", type=float, default=None,
                   help="Seconds to wait during target selection (overrides env).")
    p.add_argument("--llm-timeout-edit", type=float, default=None,
                   help="Seconds to wait during edit generation (overrides env).")
    p.add_argument("--targets-fallback-n", type=int, default=None,
                   help="If target selection times out, take top-N candidates.")

    # Brief cache controls (also supported via env; see aidev_cli shim)
    p.add_argument("--brief-refresh", action="store_true",
                   help="Force-refresh the cached project brief before this run.")
    p.add_argument("--brief-ttl-hours", type=float, default=None,
                   help="Reuse cached brief if newer than this TTL (in hours).")

    return p


def _resolve_roots(args_ns: argparse.Namespace, *, cfg: Dict[str, Any], reuse_project_root: Optional[Path] = None) -> List[Path]:
    """Resolve one or more project roots using discovery and config."""
    base_root = Path(args_ns.project_root).resolve() if args_ns.project_root else Path.cwd().resolve()
    if reuse_project_root:
        return [reuse_project_root]

    # Prefer config discovery.depth unless overridden by CLI
    scan_depth = int(cfg.get("discovery", {}).get("scan_depth", 2))
    if args_ns.project_scan_depth is not None:
        scan_depth = int(args_ns.project_scan_depth)

    roots = interactive_pick_projects_if_needed(
        base_root if args_ns.project_root else None,
        scan_start=base_root,
        scan_depth=scan_depth,
        assume_yes=bool(args_ns.yes or args_ns.no_project_prompt),
    )
    return roots


def _build_orchestrator_args(args_ns: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compose orchestrator args with config as the baseline; CLI overrides when provided."""
    # Discovery globs: CLI wins if specified; otherwise from cfg
    includes = list(args_ns.include) if args_ns.include else list(cfg.get("discovery", {}).get("includes", []))
    excludes = list(args_ns.exclude) if args_ns.exclude else list(cfg.get("discovery", {}).get("excludes", []))

    # Cards budgets: CLI wins if specified
    cards_budget = int(args_ns.cards_budget) if args_ns.cards_budget is not None else int(cfg.get("cards", {}).get("default_budget", 4096))
    cards_top_k = int(args_ns.cards_top_k) if args_ns.cards_top_k is not None else int(cfg.get("cards", {}).get("default_top_k", 20))
    cards_graph_hops = int(args_ns.cards_graph_hops) if args_ns.cards_graph_hops is not None else int(cfg.get("cards", {}).get("graph_hops", 2))

    # Optional: allow CLI deployment override to set LLM model
    if args_ns.deployment:
        cfg = dict(cfg)  # shallow clone so we don't mutate shared dict
        llm = dict(cfg.get("llm", {}))
        # Set for compatibility with "model"
        llm["model"] = str(args_ns.deployment)
        llm["deployment"] = str(args_ns.deployment)
        cfg["llm"] = llm

    # Brief cache: prefer CLI values; fall back to env set by aidev_cli shim
    brief_refresh: Optional[bool] = args_ns.brief_refresh or _parse_env_bool(os.getenv("AIDEV_BRIEF_REFRESH")) or False
    brief_ttl_hours: Optional[float] = (
        args_ns.brief_ttl_hours
        if args_ns.brief_ttl_hours is not None
        else _parse_env_float(os.getenv("AIDEV_BRIEF_TTL_HOURS"))
    )

    orch_args: Dict[str, Any] = {
        "include": includes,
        "exclude": excludes,
        "max_context_kb": args_ns.max_context_kb,
        "dry_run": bool(args_ns.dry_run),
        "focus": args_ns.focus,
        "fix_all": bool(args_ns.fix_all),
        "targets_only": bool(args_ns.targets_only),
        "cards_top_k": cards_top_k,
        "cards_budget": cards_budget,
        "cards_graph_hops": cards_graph_hops,
        "strip_comments": bool(args_ns.strip_comments),
        "apply_jsonl": args_ns.apply_jsonl,
        "git_snapshot": bool(args_ns.git_snapshot),
        "git_push": bool(args_ns.git_push),
        "cards_force": bool(args_ns.cards_force),
        # Enrichment toggles (were parsed but not forwarded)
        "enrich": bool(args_ns.enrich),
        "enrich_top_k": int(args_ns.enrich_top_k),
        # Project map
        "export_project_map": args_ns.export_project_map,
        "project_map_only": bool(args_ns.project_map_only),
        # Stage tunables
        "llm_timeout_targets": args_ns.llm_timeout_targets,
        "llm_timeout_edit": args_ns.llm_timeout_edit,
        "targets_fallback_n": args_ns.targets_fallback_n,
        # Brief cache controls (consumed by Orchestrator._init_project_brief)
        "brief_refresh": bool(brief_refresh),
        "brief_ttl_hours": brief_ttl_hours,
        # Pass the merged config downstream
        "cfg": cfg,
        # (Optional) future: approval/progress callbacks can be injected by UI
    }
    
    # Optional orchestration mode (qa / analyze / edit / auto).
    # If None, Orchestrator will auto-detect from the message.
    if getattr(args_ns, "mode", None):
        orch_args["mode"] = args_ns.mode

    return orch_args


def _run_pipeline_for_roots(roots: List[Path], args_ns: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    """Run the orchestrator-based pipeline for one or more roots."""
    orch_args: Dict[str, Any] = _build_orchestrator_args(args_ns, cfg)

    status = 0
    for root in roots:
        st = ProjectState(project_root=root)
        st.trace.write("start", "run", {"root": str(root)})
        orch = Orchestrator(root=root, st=st, args=orch_args)
        try:
            orch.run()
        except SystemExit as e:
            status = status or int(e.code or 0)
        except Exception as e:
            logging.exception("Unhandled error")
            st.trace.write("error", "exception", {"root": str(root), "error": str(e)})
            status = status or 2
        else:
            st.trace.write("end", "run", {"root": str(root)})
    return status


def _run_chat_mode(args_ns: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    """
    Minimal interactive terminal chat:
    - Finds/asks for a project (once), then loops asking the user for a natural-language goal.
    - Each turn sets args_ns.focus and runs the normal pipeline.
    - Type 'exit' or 'quit' to leave.
    """
    print("💬  AI-Dev-Bot chat mode — no flags required. Type 'exit' to quit.")
    roots = _resolve_roots(args_ns, cfg=cfg)
    if not roots:
        print("No project found. You can re-run after creating one (or use 'serve' for a GUI).")
        return 1
    root = roots[0]
    print(f"📁 Using project: {root}")

    status = 0
    while True:
        try:
            user_msg = input("\n🤖 What would you like me to do? ")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting chat.")
            break
        if not user_msg:
            continue
        if user_msg.strip().lower() in {"exit", "quit", "q"}:
            print("👋 Bye!")
            break

        args_ns.focus = user_msg
        status = _run_pipeline_for_roots([root], args_ns, cfg)

    return status


def _run_serve_mode(args_ns: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    """
    Hand off to a web server if available. This keeps core the single entrypoint.
    """
    try:
        # Expect a module that exposes run(args_ns, cfg) or run(args_ns)
        from .server import run as run_server  # type: ignore[attr-defined]
    except Exception:
        logging.error(
            "The web server entrypoint is not available yet.\n"
            "Add a module 'aidev/server.py' with a `run(args_ns, cfg)` function "
            "that starts your FastAPI (or similar) app."
        )
        return 2

    try:
        # Prefer the two-arg form, fallback to one-arg for early scaffolds
        try:
            return int(run_server(args_ns, cfg)) or 0  # type: ignore[misc]
        except TypeError:
            return int(run_server(args_ns)) or 0  # type: ignore[misc]
    except SystemExit as e:
        return int(e.code or 0)


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args_ns = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args_ns.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    # Load .env (non-destructive)
    load_env_variables()

    # Resolve project root early (for config path placement)
    project_root = Path(args_ns.project_root).resolve() if args_ns.project_root else Path.cwd().resolve()

    # Load (or initialize) project config
    cfg, cfg_path = load_project_config(project_root, args_ns.config or None)
    if args_ns.init_config:
        save_default_config(cfg_path)
        logging.info("Wrote default config to %s", cfg_path)
        return 0

    # Mode switches via env (thin CLI; set by aidev_cli subcommands)
    mode = os.getenv("AIDEV_MODE")
    if mode == "chat":
        return _run_chat_mode(args_ns, cfg)
    if mode == "serve":
        return _run_serve_mode(args_ns, cfg)
    # Also honor CLI flags (keeps old env-based behavior intact)
    if getattr(args_ns, "chat", False):
        return _run_chat_mode(args_ns, cfg)
    if getattr(args_ns, "serve", False):
        return _run_serve_mode(args_ns, cfg)

    # Default: legacy single-run CLI pipeline (unchanged UX, now powered by config)
    roots = _resolve_roots(args_ns, cfg=cfg)
    if not roots:
        logging.error("No project root selected or discovered.")
        return 1
    return _run_pipeline_for_roots(roots, args_ns, cfg)