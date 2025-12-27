# runtimes/__init__.py
"""
runtimes package public surface.

This module provides a cross-platform helper to run external commands in a
consistent, testable way. The helper resolves command binaries with
shutil.which (and raises a clear UnresolvedCommandError when a binary cannot
be found), normalizes commands for Windows, supports timeouts, captures
stdout/stderr, and returns a consistent structured result.

The helper is intentionally small and dependency-free so it can be imported
from tests and other runtime modules without creating import cycles.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

# Prefer to use the structured project logger when available. Importing
# runtimes.logger is safe: it is a sibling module and only performs
# lightweight setup when imported.
try:
    import runtimes.logger as _logger  # type: ignore[import]
except Exception:
    _logger = None

__all__ = [
    "run_command",
    "UnresolvedCommandError",
]


class UnresolvedCommandError(FileNotFoundError):
    """Raised when the requested command binary cannot be resolved.

    This error is intentionally a subclass of FileNotFoundError so calling
    code can catch it in the same way it would catch missing executables,
    but the message is clearer and deterministic (we resolve with
    shutil.which instead of letting a raw subprocess raise WinError / OSError).
    """

    def __init__(self, command: Union[str, Sequence[str]]):
        cmd_display = command if isinstance(command, str) else " ".join(map(str, command))
        super().__init__(f"command not found: {cmd_display}")


def _to_sequence(cmd: Union[str, Sequence[str]], posix: bool) -> List[str]:
    if isinstance(cmd, (list, tuple)):
        return [str(x) for x in cmd]
    return shlex.split(str(cmd), posix=posix)


def _log_error(
    msg: str,
    ctx: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
) -> None:
    """Write a human-friendly error to the project app.log when possible.

    We use runtimes.logger if available; otherwise fall back to a minimal
    stderr write so errors are still discoverable in environments where the
    logger module failed to import.

    The logger module exposes level-oriented helpers (info/warn/error) or a
    log(...) entry point. When the error is a FileNotFoundError (missing
    executable) we prefer to emit a WARN-level record so tests can assert on
    warnings; other unexpected exceptions are emitted as ERROR.
    """
    try:
        if _logger is not None:
            # Choose an appropriate level method on the logger. Try warn/warning first
            # for missing executables, otherwise prefer error.
            try:
                if isinstance(exc, FileNotFoundError) or msg.lower().startswith("executable not found"):
                    # prefer warn/warning -> fallback to info -> fallback to log
                    if hasattr(_logger, "warn"):
                        _logger.warn(msg, ctx=ctx, exc=exc)  # type: ignore[arg-type]
                    elif hasattr(_logger, "warning"):
                        _logger.warning(msg, ctx=ctx, exc=exc)  # type: ignore[arg-type]
                    elif hasattr(_logger, "info"):
                        _logger.info(msg, ctx=ctx, exc=exc)  # type: ignore[arg-type]
                    elif hasattr(_logger, "log"):
                        _logger.log("WARN", msg, ctx=ctx, exc=exc)  # type: ignore[arg-type]
                    else:
                        # Best-effort: try calling as function
                        _logger(msg, ctx=ctx, exc=exc)  # type: ignore[call-arg]
                else:
                    # Non-missing-executable issues should be treated as errors
                    if hasattr(_logger, "error"):
                        _logger.error(msg, ctx=ctx, exc=exc)  # type: ignore[arg-type]
                    elif hasattr(_logger, "err"):
                        _logger.err(msg, ctx=ctx, exc=exc)  # type: ignore[arg-type]
                    elif hasattr(_logger, "log"):
                        _logger.log("ERROR", msg, ctx=ctx, exc=exc)  # type: ignore[arg-type]
                    else:
                        _logger(msg, ctx=ctx, exc=exc)  # type: ignore[call-arg]
                return
            except TypeError:
                # Some logger implementations may have a different signature;
                # fall through to the simple fallback below.
                pass
    except Exception:
        # If the logger itself raises, fall back to stderr output below.
        pass

    try:
        # Best-effort fallback to stderr
        ctx_str = f" ctx={ctx}" if ctx else ""
        exc_str = f" exc={exc!r}" if exc else ""
        print(f"ERROR: {msg}{ctx_str}{exc_str}", file=sys.stderr)
    except Exception:
        # Nothing sensible we can do
        pass


def run_command(
    cmd: Union[str, Sequence[str]],
    *,
    timeout: Optional[float] = None,
    shell: bool = False,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    text: bool = True,
) -> Dict[str, Any]:
    """Run an external command with cross-platform normalization and timeouts.

    Args:
      cmd: Command to run. Can be a sequence (recommended) or a string. If a
        string, and shell=False, the string will be split into arguments.
      timeout: Seconds to wait before timing out the command. None means wait
        forever.
      shell: If True, run the command in the shell. When shell=True the
        helper will not attempt to resolve the binary; resolution is only
        performed for direct exec calls (shell=False).
      cwd: Working directory for the child process.
      env: Optional environment overrides for the child process.
      capture_output: If True capture stdout/stderr and return them as strings.
      text: If True decode stdout/stderr to str (universal_newlines).

    Returns:
      A dict containing:
        - returncode: int or None (None when the process was killed due to timeout)
        - stdout: str
        - stderr: str
        - timed_out: bool
        - resolved_path: the resolved absolute path of the binary (if resolved), else None
        - error: optional human-readable error message when an error occurred

    Raises:
      UnresolvedCommandError: when shell=False and the requested binary cannot
        be resolved via shutil.which. This avoids platform-specific raw
        OSError/WinError being raised by subprocess and gives a deterministic
        error type to catch in tests.
    """
    is_windows = sys.platform.startswith("win")
    posix = not is_windows

    resolved_path: Optional[str] = None
    timed_out = False
    stdout_data = ""
    stderr_data = ""

    proc_args: Union[List[str], str]
    if shell:
        # When shell=True we accept a string or join a sequence; shell handles resolution.
        if isinstance(cmd, (list, tuple)):
            proc_args = " ".join(map(str, cmd))
        else:
            proc_args = str(cmd)
        resolved_path = None
    else:
        parts = _to_sequence(cmd, posix=posix)
        if not parts:
            raise ValueError("empty command provided")
        first = parts[0]

        # If the first token looks like a path, prefer that; otherwise try which()
        if os.path.sep in first or (is_windows and ":" in first):
            resolved = os.path.abspath(first)
            if not os.path.exists(resolved):
                resolved = shutil.which(first)
        else:
            resolved = shutil.which(first)

        if not resolved:
            # Deterministic error type for missing binaries
            raise UnresolvedCommandError(first)

        resolved_path = os.path.abspath(resolved)
        parts[0] = resolved_path
        proc_args = parts

    popen_kwargs: Dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "shell": shell,
    }
    if capture_output:
        popen_kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})
    else:
        popen_kwargs.update({"stdout": None, "stderr": None})

    popen_kwargs["text"] = bool(text)

    # Try to start and communicate with the subprocess. We explicitly handle
    # FileNotFoundError/OSError so callers (and tests) see a structured result
    # instead of an uncaught exception that crashes the process.
    proc = None
    try:
        proc = subprocess.Popen(proc_args, **popen_kwargs)
        try:
            out, err = proc.communicate(timeout=timeout)
            stdout_data = out if out is not None else ""
            stderr_data = err if err is not None else ""
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                proc.kill()
            except Exception:
                pass
            try:
                out, err = proc.communicate(timeout=1)
                stdout_data = out if out is not None else ""
                stderr_data = err if err is not None else ""
            except Exception:
                # Best-effort: keep what we have
                stdout_data = stdout_data or ""
                stderr_data = stderr_data or ""
            returncode = None

        return {
            "returncode": returncode,
            "stdout": stdout_data,
            "stderr": stderr_data,
            "timed_out": timed_out,
            "resolved_path": resolved_path,
        }

    except FileNotFoundError as e:
        # This can happen on some platforms when Popen is asked to exec a path
        # that doesn't exist or when a lower-level OS resolution fails. Return a
        # structured error and log a readable message to app.log.
        msg = f"executable not found: {proc_args if not isinstance(cmd, str) else cmd}"
        _log_error(msg, ctx={"command": proc_args}, exc=e)
        return {
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "resolved_path": resolved_path,
            "error": str(e),
        }

    except OSError as e:
        # Catch other OS-level invocation errors (permission denied, etc.)
        msg = f"os error invoking executable: {proc_args if not isinstance(cmd, str) else cmd}: {e}"
        _log_error(msg, ctx={"command": proc_args}, exc=e)
        return {
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "resolved_path": resolved_path,
            "error": str(e),
        }

    except Exception as e:
        # Unanticipated exceptions: log and return a structured error so callers
        # can continue operating without the process crashing.
        msg = f"unexpected error invoking command: {proc_args if not isinstance(cmd, str) else cmd}: {e}"
        _log_error(msg, ctx={"command": proc_args}, exc=e)
        return {
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "resolved_path": resolved_path,
            "error": str(e),
        }
