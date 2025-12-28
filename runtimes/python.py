# runtimes/python.py
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from runtimes.runner import run_command as _rc

_TAIL = int(os.getenv("AIDEV_PY_LOG_TAIL", "2000"))


def _tail(s: str, n: int = _TAIL) -> str:
    s = s or ""
    return s if len(s) <= n else s[-n:]


class PythonTools:
    """
    Python runtime checks & helpers.

    Contract:
      run_checks(project_root) -> {
        "ok": bool,
        "duration_sec": float,
        "steps": [{...}],
        "logs": str
      }
    """
    name = "python"

    def __init__(self, project_root: Union[str, Path]):
        self.root = Path(project_root).resolve()
        self.py_exe = self._pick_python()

    @staticmethod
    def detect(project_root: Union[str, Path]) -> bool:
        root = Path(project_root)
        if any((root / f).exists() for f in ("pyproject.toml", "setup.cfg", "requirements.txt")):
            return True
        for p in root.rglob("*.py"):
            if any(seg in {"venv", ".venv", "__pycache__", ".git", "node_modules"} for seg in p.parts):
                continue
            return True
        return False

    def run_checks(self, project_root: Optional[Union[str, Path]] = None) -> Dict[str, object]:
        if project_root:
            self.root = Path(project_root).resolve()
            self.py_exe = self._pick_python()

        t0 = time.time()
        steps: List[Dict[str, object]] = []
        logs: List[str] = []

        lint_cmd, lint_name = self._pick_linter()
        if lint_cmd is None:
            steps.append({"name": "lint", "ok": True, "rc": 0, "stdout": "", "stderr": "", "skipped": True})
            logs.append("## python: lint — skipped (ruff/flake8 not found)")
        else:
            res = _rc(lint_cmd, cwd=str(self.root), timeout=float(os.getenv("AIDEV_PY_LINT_TIMEOUT", "600")))
            ok = (res.get("returncode") == 0) and not bool(res.get("timed_out"))
            step = {
                "name": "lint", "ok": ok, "rc": res.get("returncode", 1),
                "stdout": _tail(res.get("stdout") or ""), "stderr": _tail(res.get("stderr") or ""),
                "skipped": False, "tool": lint_name
            }
            steps.append(step)
            logs.append(f"## python: lint ({lint_name})\n$ rc={step['rc']}\n{step['stdout']}\n{step['stderr']}")

        type_cmd = self._pick_mypy()
        if type_cmd is None:
            steps.append({"name": "typecheck", "ok": True, "rc": 0, "stdout": "", "stderr": "", "skipped": True})
            logs.append("## python: typecheck — skipped (mypy not found)")
        else:
            res = _rc(type_cmd, cwd=str(self.root), timeout=float(os.getenv("AIDEV_PY_MYPY_TIMEOUT", "900")))
            ok = (res.get("returncode") == 0) and not bool(res.get("timed_out"))
            step = {
                "name": "typecheck", "ok": ok, "rc": res.get("returncode", 1),
                "stdout": _tail(res.get("stdout") or ""), "stderr": _tail(res.get("stderr") or ""),
                "skipped": False, "tool": "mypy"
            }
            steps.append(step)
            logs.append(f"## python: typecheck (mypy)\n$ rc={step['rc']}\n{step['stdout']}\n{step['stderr']}")

        test_cmd = self._pick_pytest()
        if test_cmd is None:
            steps.append({"name": "tests", "ok": True, "rc": 0, "stdout": "", "stderr": "", "skipped": True})
            logs.append("## python: tests — skipped (pytest not detected)")
        else:
            res = _rc(test_cmd, cwd=str(self.root), timeout=float(os.getenv("AIDEV_PY_TEST_TIMEOUT", "1800")))
            rc = int(res.get("returncode") or 0)
            ok = (rc == 0) or (rc == 5)  # 5 = no tests collected
            step = {
                "name": "tests", "ok": ok, "rc": rc,
                "stdout": _tail(res.get("stdout") or ""), "stderr": _tail(res.get("stderr") or ""),
                "skipped": False, "tool": "pytest"
            }
            steps.append(step)
            logs.append(f"## python: tests (pytest -q)\n$ rc={rc}\n{step['stdout']}\n{step['stderr']}")

        total_ok = all(s["ok"] for s in steps if not s.get("skipped"))
        return {
            "ok": bool(total_ok),
            "duration_sec": round(time.time() - t0, 3),
            "steps": steps,
            "logs": ("\n".join(logs).rstrip() + "\n"),
        }

    # --- helpers ---
    def _pick_python(self) -> str:
        for d in (".venv", "venv"):
            py = self._venv_python(self.root / d)
            if py:
                return py
        return os.getenv("PYTHON", "") or os.getenv("PYTHON_EXE", "") or shutil.which("python") or shutil.which("python3") or "python"

    @staticmethod
    def _venv_python(venv_root: Path) -> Optional[str]:
        candidates = [venv_root / "bin" / "python", venv_root / "Scripts" / "python.exe"]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _module_cmd(self, mod: str, *args: str) -> List[str]:
        return [self.py_exe, "-m", mod, *args]

    def _pick_linter(self) -> (Optional[List[str]], Optional[str]):
        if shutil.which("ruff"):
            return ["ruff", "check", "."], "ruff"
        if shutil.which("flake8"):
            return ["flake8", "."], "flake8"
        return (self._module_cmd("ruff", "check", "."), "ruff") if self._importable("ruff") else \
               ((self._module_cmd("flake8", "."), "flake8") if self._importable("flake8") else (None, None))

    def _pick_mypy(self) -> Optional[List[str]]:
        cfg_exists = any((self.root / f).exists() for f in ("mypy.ini", ".mypy.ini", "pyproject.toml", "setup.cfg"))
        if cfg_exists or shutil.which("mypy") or self._importable("mypy"):
            return ["mypy", "--no-error-summary", "--hide-error-context", "."]
        return None

    def _pick_pytest(self) -> Optional[List[str]]:
        tests_exist = any((self.root / d).exists() for d in ("tests", "test"))
        if tests_exist or shutil.which("pytest") or self._importable("pytest"):
            return ["pytest", "-q"]
        return None

    def _importable(self, mod: str) -> bool:
        try:
            res = _rc([self._pick_python(), "-c", f"import {mod}; print({mod!r})"], cwd=str(self.root), timeout=15)
            return res.get("returncode") == 0
        except Exception:
            return False


__all__ = ["PythonTools"]
