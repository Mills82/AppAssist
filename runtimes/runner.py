# runtimes/runner.py
"""
Canonical subprocess runner for the runtimes package.

Goals:
- One implementation used everywhere (avoid "duplicate runner" drift).
- Cross-platform binary resolution (when shell=False).
- Safe decoding on Windows (avoid UnicodeDecodeError from cp1252) by using:
    encoding="utf-8", errors="replace"
  whenever text=True.
- Timeouts return a structured result (no uncaught TimeoutExpired).
- Missing executables return a structured result by default (optionally raise).

Return shape (always a dict):
  {
    "returncode": int | None,
    "stdout": str,
    "stderr": str,
    "timed_out": bool,
    "missing_executable": bool,
    "resolved_path": str | None,
    "elapsed_sec": float,
    "ok": bool,
    "error": str | None,
  }
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union


class UnresolvedCommandError(FileNotFoundError):
    """Raised when the requested command binary cannot be resolved."""

    def __init__(self, command: Union[str, Sequence[str]]):
        cmd_display = command if isinstance(command, str) else " ".join(map(str, command))
        super().__init__(f"command not found: {cmd_display}")


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _to_argv(cmd: Union[str, Sequence[str]], *, shell: bool) -> Union[str, List[str]]:
    if shell:
        if isinstance(cmd, (list, tuple)):
            return " ".join(map(str, cmd))
        return str(cmd)

    if isinstance(cmd, (list, tuple)):
        return [str(x) for x in cmd]

    # shell=False and cmd is a string: split into argv
    posix = not _is_windows()
    return shlex.split(str(cmd), posix=posix)


def _resolve_binary(argv: List[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve argv[0] to an absolute path if possible.

    Returns (resolved_path, error_msg). If resolved_path is None, error_msg is set.
    """
    if not argv:
        return None, "empty command provided"

    exe = argv[0]
    is_win = _is_windows()

    # If it looks like a path, prefer it; otherwise use PATH resolution.
    looks_like_path = (os.path.sep in exe) or (is_win and ":" in exe)
    resolved = None

    if looks_like_path:
        abs_path = os.path.abspath(exe)
        if os.path.exists(abs_path):
            resolved = abs_path
        else:
            # Might still be resolvable (e.g., "npm.cmd" without full path)
            resolved = shutil.which(exe)
    else:
        resolved = shutil.which(exe)

    if not resolved:
        return None, f"executable not found: {exe}"

    return os.path.abspath(resolved), None


def _normalize_output(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    return str(val)


def run_command(
    cmd: Union[str, Sequence[str]],
    *,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    shell: bool = False,
    capture_output: bool = True,
    text: bool = True,
    raise_on_missing: bool = False,
    subprocess_module=None,
) -> Dict[str, Any]:
    """
    Run a command and return a structured result. By default this function does NOT raise
    for missing executables; set raise_on_missing=True to raise UnresolvedCommandError.

    `subprocess_module` is injectable for tests (defaults to stdlib subprocess).
    """
    if subprocess_module is None:
        import subprocess as subprocess_module  # type: ignore

    t0 = time.time()

    result: Dict[str, Any] = {
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "missing_executable": False,
        "resolved_path": None,
        "elapsed_sec": 0.0,
        "ok": False,
        "error": None,
    }

    # Build env
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    try:
        proc_args = _to_argv(cmd, shell=shell)

        # Resolve binary when shell=False
        if not shell:
            if not isinstance(proc_args, list):
                # defensive; _to_argv returns list for shell=False
                proc_args = _to_argv(str(proc_args), shell=False)  # type: ignore[assignment]
            argv: List[str] = proc_args  # type: ignore[assignment]
            if not argv:
                result["error"] = "empty command provided"
                result["elapsed_sec"] = round(time.time() - t0, 6)
                return result

            resolved, err = _resolve_binary(argv)
            if not resolved:
                if raise_on_missing:
                    raise UnresolvedCommandError(argv[0])
                result["missing_executable"] = True
                result["error"] = err
                result["elapsed_sec"] = round(time.time() - t0, 6)
                return result

            result["resolved_path"] = resolved
            argv[0] = resolved
            proc_args = argv

        popen_kwargs: Dict[str, Any] = {
            "cwd": cwd,
            "env": proc_env,
            "shell": shell,
        }

        if capture_output:
            popen_kwargs["stdout"] = subprocess_module.PIPE
            popen_kwargs["stderr"] = subprocess_module.PIPE
        else:
            popen_kwargs["stdout"] = None
            popen_kwargs["stderr"] = None

        popen_kwargs["text"] = bool(text)

        # Critical: avoid cp1252 decode crashes on Windows when text=True
        if text:
            popen_kwargs["encoding"] = "utf-8"
            popen_kwargs["errors"] = "replace"

        # Some Python builds / older versions may not accept encoding/errors with text=True
        try:
            proc = subprocess_module.Popen(proc_args, **popen_kwargs)
        except TypeError:
            popen_kwargs.pop("encoding", None)
            popen_kwargs.pop("errors", None)
            proc = subprocess_module.Popen(proc_args, **popen_kwargs)

        try:
            out, err = proc.communicate(timeout=timeout)
            result["returncode"] = proc.returncode
            if capture_output:
                result["stdout"] = _normalize_output(out)
                result["stderr"] = _normalize_output(err)
        except subprocess_module.TimeoutExpired:
            result["timed_out"] = True
            try:
                proc.kill()
            except Exception:
                pass
            try:
                out, err = proc.communicate(timeout=1)
                if capture_output:
                    result["stdout"] = _normalize_output(out)
                    result["stderr"] = _normalize_output(err)
            except Exception:
                # best-effort; keep whatever we have
                pass
            result["returncode"] = None
            result["error"] = "timeout"
        finally:
            result["elapsed_sec"] = round(time.time() - t0, 6)

        # ok status
        result["ok"] = (result["returncode"] == 0) and (not bool(result["timed_out"])) and (not bool(result["missing_executable"]))
        return result

    except FileNotFoundError as e:
        # Can happen if Popen is asked to exec a path that disappears, etc.
        result["missing_executable"] = True
        result["error"] = str(e)
        result["elapsed_sec"] = round(time.time() - t0, 6)
        result["ok"] = False
        return result
    except OSError as e:
        result["error"] = str(e)
        result["elapsed_sec"] = round(time.time() - t0, 6)
        result["ok"] = False
        return result
    except Exception as e:
        result["error"] = str(e)
        result["elapsed_sec"] = round(time.time() - t0, 6)
        result["ok"] = False
        return result
