# aidev_cli.py
from __future__ import annotations

"""
Thin CLI shim for AI-Dev-Bot.

- Adds `serve` and `chat` subcommands for non-technical users.
- Keeps existing flags for power users (delegated to core).
- Routes *all* behavior through a single `aidev.core.main()` API.

Core + CLI contract:
- This shim may set convenience env vars (e.g., AIDEV_MODE, AIDEV_BRIEF_REFRESH,
  AIDEV_BRIEF_TTL_HOURS) and strip the corresponding flags from argv.
- `aidev.core.main()` also understands `--brief-refresh` and `--brief-ttl-hours`
  directly when invoked without this shim, so older entrypoints remain
  backwards compatible.

Extra conveniences handled here (stripped from argv and set as env vars):
  --brief-refresh[=true|false|1|0]     -> AIDEV_BRIEF_REFRESH
  --brief-ttl-hours[=<float>] <float>  -> AIDEV_BRIEF_TTL_HOURS

Examples:
  $ python -m aidev_cli serve
  $ python -m aidev_cli chat
  $ python -m aidev_cli --project-root /path --focus "Polish UI"
  $ python -m aidev_cli serve --brief-refresh --brief-ttl-hours 12
"""

import os
import sys
from typing import Optional

# Allowed conversational entrypoints.
_SUBCOMMANDS = {"serve", "chat"}


def _extract_mode(argv: list[str]) -> Optional[str]:
    """
    If argv[1] is a known subcommand, return it (and remove it from argv).
    Otherwise return None and leave argv as-is.
    """
    if len(argv) > 1:
        cmd = argv[1].strip().lower()
        if cmd in _SUBCOMMANDS:
            # Remove the subcommand token so core argparse is not affected.
            del argv[1]
            return cmd
    return None


def _pop_flag_and_optional_value(argv: list[str], i: int) -> None:
    """
    Remove argv[i] and (if present and not another flag) argv[i+1].
    Used for flags that may accept a separate value (e.g., --foo 123).
    """
    del argv[i]
    # If the next token exists and is not another flag, remove it (it's the value we consumed).
    if i < len(argv) and not argv[i].startswith("-"):
        del argv[i]


def _parse_bool_token(tok: Optional[str]) -> bool:
    if tok is None:
        return True
    t = tok.strip().lower()
    if t in {"1", "true", "yes", "on"}:
        return True
    if t in {"0", "false", "no", "off"}:
        return False
    # If it's something else (like another flag), treat as True when used as --flag=value form without clear bool.
    return True


def _extract_passthrough_env(argv: list[str]) -> None:
    """
    Convert a couple of convenience flags into environment variables and strip them
    so core's argparse doesn't need to know about them (non-breaking).

    Supported:
      --brief-refresh[=BOOL]           -> AIDEV_BRIEF_REFRESH = "1" or "0"
      --brief-ttl-hours[=FLOAT] VALUE  -> AIDEV_BRIEF_TTL_HOURS = "<float>"
    """
    i = 1  # start scanning after program name
    while i < len(argv):
        arg = argv[i]

        # --brief-refresh[=...]
        if arg == "--brief-refresh" or arg.startswith("--brief-refresh="):
            val: Optional[str] = None
            if "=" in arg:
                _, val = arg.split("=", 1)
                # Remove just this token
                del argv[i]
            else:
                # Peek next as value if present and not another flag
                val = None
                if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    val = argv[i + 1]
                    _pop_flag_and_optional_value(argv, i)
                else:
                    # bare flag (truthy)
                    del argv[i]

            truthy = _parse_bool_token(val)
            os.environ["AIDEV_BRIEF_REFRESH"] = "1" if truthy else "0"
            # do not increment i; we already removed current token (and maybe its value)

            continue

        # --brief-ttl-hours[=FLOAT] (either --brief-ttl-hours=12 or --brief-ttl-hours 12)
        if arg == "--brief-ttl-hours" or arg.startswith("--brief-ttl-hours="):
            ttl_val: Optional[str] = None
            if "=" in arg:
                _, ttl_val = arg.split("=", 1)
                del argv[i]
            else:
                # separate value required (if present)
                if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    ttl_val = argv[i + 1]
                    _pop_flag_and_optional_value(argv, i)
                else:
                    # no value provided; remove flag and ignore silently
                    del argv[i]
                    ttl_val = None

            if ttl_val is not None and ttl_val.strip():
                os.environ["AIDEV_BRIEF_TTL_HOURS"] = ttl_val.strip()
            # already removed the tokens; continue scanning at same i
            continue

        # Not a convenience flag we handle; move on
        i += 1


def _delegate_to_core() -> int:
    # Defer all logic to core.main()
    from aidev.core import main as core_main  # local import to keep CLI lightweight
    return core_main()


def main() -> int:
    # Detect optional subcommand (serve/chat) and set an env knob for core.
    mode = _extract_mode(sys.argv)
    if mode:
        os.environ["AIDEV_MODE"] = mode  # core.main() should consult this

    # Convert convenience flags into env vars and strip them from argv (non-breaking).
    _extract_passthrough_env(sys.argv)

    # Hand off to core (no other logic here).
    return _delegate_to_core()


if __name__ == "__main__":
    raise SystemExit(main())
