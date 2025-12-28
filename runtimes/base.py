# runtimes/base.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

from runtimes.runner import run_command as _run_command


def run_command(
    cmd,
    cwd=None,
    timeout=None,
    env=None,
    use_shell: bool = False,
) -> Dict[str, object]:
    """
    Back-compat wrapper around the canonical runner.

    Older callers used:
      run_command(cmd, cwd=..., timeout=..., env=..., use_shell=...)
    """
    res = _run_command(
        cmd,
        cwd=cwd,
        timeout=timeout,
        env=env,
        shell=use_shell,
        capture_output=True,
        text=True,
        raise_on_missing=False,
    )
    # Preserve older shape keys that some callers may rely on.
    return {
        "returncode": res.get("returncode"),
        "stdout": res.get("stdout", ""),
        "stderr": res.get("stderr", ""),
        "timed_out": bool(res.get("timed_out")),
        "missing_executable": bool(res.get("missing_executable")),
        "resolved_path": res.get("resolved_path"),
        "elapsed_sec": float(res.get("elapsed_sec") or 0.0),
        "ok": bool(res.get("ok")),
        "error": res.get("error"),
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
