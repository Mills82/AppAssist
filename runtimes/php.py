# runtimes/php.py
"""
PHP runtime helpers that provide safe, structured subprocess invocation for PHP
commands.

This module now delegates process execution to the canonical runtimes.runner.run_command
to avoid duplicate-runner drift and Windows decoding crashes.

Important for tests:
- tests may monkeypatch the module-level `subprocess` symbol.
  We pass `subprocess_module=subprocess` into the canonical runner to preserve that.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shlex
import subprocess
import traceback
from typing import Any, Dict, List, Optional, Sequence, Union

from runtimes.runner import run_command as _run

logger = logging.getLogger("aidev.runtimes.php")


def _emit_structured_log(
    level: str,
    msg: str,
    context: Optional[Dict[str, Any]] = None,
    path: str = "./app.log",
) -> None:
    try:
        # If a central structured logger exists, try to use it.
        try:
            import runtimes.logger as rl  # type: ignore

            if hasattr(rl, "AppLogger"):
                try:
                    app = rl.AppLogger()  # type: ignore
                    if hasattr(app, "log"):
                        app.log(level, msg, context=context)  # type: ignore
                        return
                    if hasattr(app, "write"):
                        app.write(  # type: ignore
                            {
                                "ts": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
                                "level": level,
                                "msg": msg,
                                "context": context,
                            }
                        )
                        return
                except Exception:
                    pass

            if hasattr(rl, "log"):
                try:
                    rl.log(level, msg, context=context)  # type: ignore
                    return
                except Exception:
                    pass

            if hasattr(rl, "write"):
                try:
                    rl.write(  # type: ignore
                        {
                            "ts": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
                            "level": level,
                            "msg": msg,
                            "context": context,
                        }
                    )
                    return
                except Exception:
                    pass
        except Exception:
            pass

        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)

        entry: Dict[str, Any] = {
            "ts": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
            "level": level,
            "msg": msg,
        }
        if context:
            entry["context"] = context

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception:
        logger.debug("failed to write structured app log: %s", traceback.format_exc())


def _tool_name(cmd: Union[str, Sequence[str]]) -> str:
    try:
        if isinstance(cmd, str):
            parts = shlex.split(cmd)
            return parts[0] if parts else cmd
        return str(list(cmd)[0])
    except Exception:
        return str(cmd)


def run_command(
    cmd: Union[str, Sequence[str]],
    timeout: Optional[float] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    text: bool = True,
) -> Dict[str, Any]:
    """
    Run a command and return the structured result from runtimes.runner.run_command.

    This function will NOT raise FileNotFoundError/OSError; instead it returns a structured
    dict and emits a structured log entry for missing executables / unexpected OS errors.
    """
    tool = _tool_name(cmd)
    res = _run(
        cmd,
        timeout=timeout,
        cwd=cwd,
        env=env,
        capture_output=capture_output,
        text=text,
        shell=False if not isinstance(cmd, str) else False,
        raise_on_missing=False,
        subprocess_module=subprocess,  # preserve test monkeypatching
    )

    if res.get("missing_executable"):
        msg = res.get("error") or f"executable not found: {tool}"
        _emit_structured_log("WARN", msg, {"command": tool, "cwd": cwd, "resolved_path": res.get("resolved_path")})
    elif res.get("error") and not res.get("timed_out"):
        _emit_structured_log("ERROR", str(res.get("error")), {"command": tool, "cwd": cwd})

    return res


def php_syntax_check(file_path: str, timeout: Optional[float] = 10.0) -> Dict[str, Any]:
    """Run `php -l <file_path>` to perform a syntax check and return the structured result."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    return run_command(["php", "-l", file_path], timeout=timeout)


def run_php_script(
    args: List[str],
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Run `php` with arbitrary arguments and return the structured result."""
    return run_command(["php", *args], timeout=timeout, env=env, cwd=cwd)


__all__ = ["run_command", "php_syntax_check", "run_php_script"]
