# runtimes/base.py
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _resolve_exe(exe: str) -> Optional[str]:
    if os.path.isabs(exe) or os.path.dirname(exe):
        return exe if os.path.exists(exe) else None
    return shutil.which(exe)


def run_command(
    cmd: Union[str, Sequence[str]],
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    env: Optional[dict] = None,
    use_shell: bool = False,
) -> Dict[str, object]:
    """
    Normalized process runner:
      - resolves the binary via PATH
      - returns a compact dict with stdout/stderr/returncode
      - sets 'missing_executable' when the binary can't be resolved
    """
    if isinstance(cmd, str):
        if use_shell:
            argv = cmd
            resolved = None
        else:
            argv = shlex.split(cmd, posix=(os.name != "nt"))
            if not argv:
                return _err("empty command", resolved=None)
            exe = _resolve_exe(argv[0])
            if not exe:
                return _missing(argv[0])
            argv[0] = exe
            resolved = exe
    else:
        argv = list(cmd)
        if not argv:
            return _err("empty command", resolved=None)
        exe = _resolve_exe(argv[0])
        if not exe:
            return _missing(argv[0])
        argv[0] = exe
        resolved = exe

    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    started = time.time()
    try:
        p = subprocess.Popen(
            argv,
            cwd=cwd,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False if isinstance(argv, list) else use_shell,
        )
        try:
            out, err = p.communicate(timeout=timeout)
            rc = p.returncode
            dt = time.time() - started
            return {
                "returncode": rc,
                "stdout": out or "",
                "stderr": err or "",
                "timed_out": False,
                "missing_executable": False,
                "resolved_path": resolved,
                "elapsed_sec": dt,
                "ok": rc == 0,
            }
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass
            try:
                out, err = p.communicate(timeout=5)
            except Exception:
                out, err = "", ""
            dt = time.time() - started
            return {
                "returncode": None,
                "stdout": out or "",
                "stderr": err or "",
                "timed_out": True,
                "missing_executable": False,
                "resolved_path": resolved,
                "elapsed_sec": dt,
                "ok": False,
            }
    except FileNotFoundError:
        return _missing(str(argv[0]) if isinstance(argv, list) else str(argv))
    except Exception as e:
        return _err(str(e), resolved=resolved)


def _missing(binary: str) -> Dict[str, object]:
    return {
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "missing_executable": True,
        "resolved_path": None,
        "elapsed_sec": 0.0,
        "ok": False,
        "error": f"executable not found: {binary}",
    }


def _err(msg: str, *, resolved: Optional[str]) -> Dict[str, object]:
    return {
        "returncode": None,
        "stdout": "",
        "stderr": msg,
        "timed_out": False,
        "missing_executable": False,
        "resolved_path": resolved,
        "elapsed_sec": 0.0,
        "ok": False,
        "error": msg,
    }


class BaseRuntime:
    """
    Minimal base runtime with optional step methods.
    A concrete runtime may implement:
      - format(self) -> dict
      - lint(self)   -> dict
      - test(self)   -> dict
      - run_checks(self, project_root: Path) -> dict   (preferred)
    """
    name: str = "base"

    # Default run_checks stitches format/lint/test if subclass didn't override it.
    def run_checks(self, project_root: Path) -> Dict[str, object]:
        t0 = time.time()
        logs: List[str] = []
        ok = True

        def call(name: str) -> Dict[str, object]:
            fn = getattr(self, name, None)
            if callable(fn):
                return fn()
            return {"ok": True, "stdout": "", "stderr": "", "note": "unsupported"}

        for step in ("format", "lint", "test"):
            res = call(step)
            ok = ok and bool(res.get("ok", False))
            out = (res.get("stdout") or "").strip()
            err = (res.get("stderr") or "").strip()
            logs.append(f"== {step} ==")
            if out:
                logs.append(out)
            if err:
                logs.append("\n-- stderr --\n" + err)

        return {
            "ok": ok,
            "duration_sec": round(time.time() - t0, 3),
            "logs": "\n".join(logs).strip(),
        }


__all__ = ["BaseRuntime", "run_command"]
