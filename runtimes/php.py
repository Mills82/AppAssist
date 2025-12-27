# runtimes/php.py
"""
PHP runtime helpers that provide safe, structured subprocess invocation for PHP
commands. Missing executables (WinError 2 / ENOENT) are handled gracefully:
instead of letting FileNotFoundError/OSError propagate uncaught, callers get a
structured dict result and an informative log entry.

Behavior summary:
- Exposes run_command(...) that returns a dict with keys: returncode, stdout,
  stderr, timed_out, missing_executable, error, resolved_path.
- Prefers a central runtimes.base.run_command when available; otherwise falls
  back to a local implementation based on subprocess.run.
- Catches FileNotFoundError/OSError and returns a non-exceptional result with
  missing_executable=True and a human-friendly error message (also logged).
- Provides helpers: php_syntax_check(file_path, timeout) and run_php_script(args, ...).

Note: tests monkeypatch module-level `subprocess` and expect our functions to
exercise error-handling paths. We therefore use the module-level `subprocess`
symbol instead of re-importing it inside functions.
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


logger = logging.getLogger("aidev.runtimes.php")


# Try to reuse central runtimes.base.run_command if available. If not present
# we fall back to a local implementation that mirrors the same shape.
_base_run_command = None
try:
    import runtimes.base as _base  # type: ignore

    _base_run_command = getattr(_base, "run_command", None)
except Exception:
    _base_run_command = None


# Structured app.log helper: try to delegate to runtimes.logger when present;
# otherwise append one JSON object per line to ./app.log.
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

            # Try common shapes: AppLogger class, or log()/write() helper.
            if hasattr(rl, "AppLogger"):
                try:
                    # AppLogger might accept a path or use default; prefer default.
                    app = rl.AppLogger()  # type: ignore
                    if hasattr(app, "log"):
                        app.log(level, msg, context=context)  # type: ignore
                        return
                    if hasattr(app, "write"):
                        app.write(  # type: ignore
                            {
                                "ts": datetime.datetime.utcnow()
                                .replace(tzinfo=datetime.timezone.utc)
                                .isoformat(),
                                "level": level,
                                "msg": msg,
                                "context": context,
                            }
                        )
                        return
                except Exception:
                    # Fall back to other approaches below.
                    pass
            # try a module-level helper
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
                            "ts": datetime.datetime.utcnow()
                            .replace(tzinfo=datetime.timezone.utc)
                            .isoformat(),
                            "level": level,
                            "msg": msg,
                            "context": context,
                        }
                    )
                    return
                except Exception:
                    pass
        except Exception:
            # No runtimes.logger available or it failed: fall through to file write.
            pass

        # Ensure directory exists.
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)

        entry: Dict[str, Any] = {
            "ts": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
            "level": level,
            "msg": msg,
        }
        if context:
            entry["context"] = context

        # Conservative append; flush and try to fsync for durability (best-effort).
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                # Best-effort durability; don't raise.
                pass
    except Exception:
        # Avoid letting logging failures propagate; keep original behavior.
        logger.debug("failed to write structured app log: %s", traceback.format_exc())


def _tool_name(cmd: Union[str, Sequence[str]]) -> str:
    try:
        if isinstance(cmd, str):
            parts = shlex.split(cmd)
            return parts[0] if parts else cmd
        else:
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
    """Run a command and return a structured result.

    Returns a dict with at least the following keys:
      - returncode: int | None
      - stdout: str
      - stderr: str
      - timed_out: bool
      - missing_executable: bool
      - error: Optional[str]
      - resolved_path: Optional[str]

    This function will NOT raise FileNotFoundError/OSError; instead it returns
    an error-shaped dict and logs a message. Timeouts are normalized to a
    result with timed_out=True.
    """
    result: Dict[str, Any] = {
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "missing_executable": False,
        "error": None,
        "resolved_path": None,
    }

    tool = _tool_name(cmd)

    # If a central helper is available, prefer it and translate its output.
    if _base_run_command is not None:
        try:
            # base.run_command convention often accepts similar args.
            # We call it and then normalize the returned mapping.
            base_res = _base_run_command(
                cmd, timeout=timeout, cwd=cwd, env=env, capture_output=capture_output, text=text
            )

            # base_res may be a mapping-like object
            if isinstance(base_res, dict):
                result["returncode"] = base_res.get("returncode")
                result["stdout"] = str(base_res.get("stdout") or "")
                result["stderr"] = str(base_res.get("stderr") or "")
                result["timed_out"] = bool(base_res.get("timed_out") or base_res.get("timedout") or False)
                result["resolved_path"] = base_res.get("resolved_path")
                # some helpers use different keys for missing executable cases
                if base_res.get("resolved_path") is None and base_res.get("returncode") is None:
                    # best-effort: if returncode is None and no resolved_path, mark missing
                    result["missing_executable"] = True
                    result["error"] = base_res.get("error") or f"executable not found: {tool}"
                    _emit_structured_log(
                        "WARN",
                        result["error"],
                        {"command": tool, "resolved_path": base_res.get("resolved_path")},
                    )
                return result

            # If base_res is a CompletedProcess-like
            if hasattr(base_res, "returncode"):
                result["returncode"] = getattr(base_res, "returncode")
                stdout = getattr(base_res, "stdout", "")
                stderr = getattr(base_res, "stderr", "")
                result["stdout"] = stdout.decode() if isinstance(stdout, (bytes, bytearray)) else str(stdout or "")
                result["stderr"] = stderr.decode() if isinstance(stderr, (bytes, bytearray)) else str(stderr or "")
                return result

            # Fallback: stringify
            result["stdout"] = str(base_res)
            return result

        except FileNotFoundError as fnf:
            msg = f"executable not found: {tool} ({fnf})"
            logger.error(msg)
            _emit_structured_log("ERROR", msg, {"command": tool, "exception": str(fnf)})
            result["missing_executable"] = True
            result["error"] = msg
            return result
        except OSError as ose:
            msg = f"os error invoking executable: {tool}: {ose}"
            logger.error(msg)
            _emit_structured_log("ERROR", msg, {"command": tool, "exception": str(ose)})
            result["error"] = msg
            return result
        except subprocess.TimeoutExpired as te:
            # Some base helpers raise TimeoutExpired; normalize to timed_out
            out = getattr(te, "stdout", "")
            err = getattr(te, "stderr", "")
            result["timed_out"] = True
            result["stdout"] = out.decode() if isinstance(out, (bytes, bytearray)) else str(out or "")
            result["stderr"] = err.decode() if isinstance(err, (bytes, bytearray)) else str(err or "")
            result["error"] = f"timeout while running: {tool}"
            logger.warning("timeout invoking %s: %s", tool, te)
            _emit_structured_log("WARN", result["error"], {"command": tool, "exception": str(te)})
            return result
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"unexpected error invoking {tool}: {exc}"
            logger.exception(msg)
            _emit_structured_log("ERROR", msg, {"command": tool, "exception": str(exc)})
            result["error"] = msg
            return result

    # Fallback local implementation using subprocess.run and normalization.
    run_kwargs: Dict[str, Any] = {"cwd": cwd, "timeout": timeout}
    if env is not None:
        run_kwargs["env"] = env
    if text:
        run_kwargs["text"] = True
    if capture_output:
        run_kwargs["capture_output"] = True

    try:
        completed = subprocess.run(cmd, **run_kwargs)

        result["returncode"] = getattr(completed, "returncode", None)
        stdout = getattr(completed, "stdout", "")
        stderr = getattr(completed, "stderr", "")
        result["stdout"] = stdout.decode() if isinstance(stdout, (bytes, bytearray)) else str(stdout or "")
        result["stderr"] = stderr.decode() if isinstance(stderr, (bytes, bytearray)) else str(stderr or "")
        return result

    except FileNotFoundError as fnf:
        msg = f"executable not found: {tool} ({fnf})"
        logger.error(msg)
        _emit_structured_log("ERROR", msg, {"command": tool, "exception": str(fnf)})
        result["missing_executable"] = True
        result["error"] = msg
        return result
    except OSError as ose:
        msg = f"os error invoking executable: {tool}: {ose}"
        logger.error(msg)
        _emit_structured_log("ERROR", msg, {"command": tool, "exception": str(ose)})
        result["error"] = msg
        return result
    except subprocess.TimeoutExpired as te:
        # Normalize timeout into result
        out = getattr(te, "stdout", "")
        err = getattr(te, "stderr", "")
        result["timed_out"] = True
        result["stdout"] = out.decode() if isinstance(out, (bytes, bytearray)) else str(out or "")
        result["stderr"] = err.decode() if isinstance(err, (bytes, bytearray)) else str(err or "")
        result["error"] = f"timeout while running: {tool}"
        logger.warning("timeout invoking %s: %s", tool, te)
        _emit_structured_log("WARN", result["error"], {"command": tool, "exception": str(te)})
        return result
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"unexpected error invoking {tool}: {exc}"
        logger.exception(msg)
        _emit_structured_log("ERROR", msg, {"command": tool, "exception": str(exc)})
        result["error"] = msg
        return result


def php_syntax_check(file_path: str, timeout: Optional[float] = 10.0) -> Dict[str, Any]:
    """Run `php -l <file_path>` to perform a syntax check and return the structured result.

    If the file does not exist a FileNotFoundError is raised to match normal
    Python semantics for bad paths (this is an input validation error rather
    than a missing-executable issue).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    return run_command(["php", "-l", file_path], timeout=timeout)


def run_php_script(
    args: List[str],
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Run `php` with arbitrary arguments and return the structured result.

    Args is the sequence of arguments after the php executable, e.g. ['-v'] or ['script.php'].
    """
    return run_command(["php", *args], timeout=timeout, env=env, cwd=cwd)


__all__ = ["run_command", "php_syntax_check", "run_php_script"]
