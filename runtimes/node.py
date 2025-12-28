# runtimes/node.py
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, List

from runtimes.runner import run_command as _rc

_TAIL = int(os.getenv("AIDEV_NODE_LOG_TAIL", "2000"))


def _tail(s: str, n: int = _TAIL) -> str:
    s = s or ""
    return s if len(s) <= n else s[-n:]


class NodeTools:
    """
    Node runtime checks & helpers.

    Contract:
      run_checks(project_root) -> {
        "ok": bool,
        "duration_sec": float,
        "steps": [...],
        "logs": str
      }
    """
    name = "node"

    def __init__(self, project_root: Union[str, Path]):
        self.root = Path(project_root).resolve()
        self.pm = self._pick_pm()

    @staticmethod
    def detect(project_root: Union[str, Path]) -> bool:
        return (Path(project_root) / "package.json").exists()

    def run_checks(self, project_root: Optional[Union[str, Path]] = None) -> Dict[str, object]:
        if project_root:
            self.root = Path(project_root).resolve()
            self.pm = self._pick_pm()

        t0 = time.time()
        steps: List[Dict[str, object]] = []
        logs: List[str] = []

        pkg = self._read_pkg()
        scripts = (pkg.get("scripts") or {}) if pkg else {}

        # Install (only when a lockfile exists)
        install_cmd = self._install_cmd()
        if install_cmd is None:
            steps.append({"name": "install", "ok": True, "rc": 0, "stdout": "", "stderr": "", "skipped": True})
            logs.append("## node: install — skipped (no lockfile)")
        else:
            res = _rc(install_cmd, cwd=str(self.root), timeout=float(os.getenv("AIDEV_NODE_INSTALL_TIMEOUT", "1200")))
            ok = (res.get("returncode") == 0) and not bool(res.get("timed_out"))
            step = {
                "name": "install", "ok": ok, "rc": res.get("returncode", 1),
                "stdout": _tail(res.get("stdout") or ""), "stderr": _tail(res.get("stderr") or ""),
                "skipped": False, "tool": self.pm
            }
            steps.append(step)
            logs.append(f"## node: install ({self.pm})\n$ rc={step['rc']}\n{step['stdout']}\n{step['stderr']}")

        # ESLint
        if self._has_eslint(pkg):
            res = _rc(self._eslint_cmd(), cwd=str(self.root), timeout=float(os.getenv("AIDEV_NODE_LINT_TIMEOUT", "900")))
            ok = (res.get("returncode") == 0) and not bool(res.get("timed_out"))
            step = {
                "name": "lint", "ok": ok, "rc": res.get("returncode", 1),
                "stdout": _tail(res.get("stdout") or ""), "stderr": _tail(res.get("stderr") or ""),
                "skipped": False, "tool": "eslint"
            }
            steps.append(step)
            logs.append(f"## node: lint (eslint)\n$ rc={step['rc']}\n{step['stdout']}\n{step['stderr']}")
        else:
            steps.append({"name": "lint", "ok": True, "rc": 0, "stdout": "", "stderr": "", "skipped": True})
            logs.append("## node: lint — skipped (eslint not detected)")

        # TypeScript
        if self._has_typescript(pkg):
            res = _rc(self._tsc_cmd(), cwd=str(self.root), timeout=float(os.getenv("AIDEV_NODE_TSC_TIMEOUT", "1200")))
            ok = (res.get("returncode") == 0) and not bool(res.get("timed_out"))
            step = {
                "name": "typecheck", "ok": ok, "rc": res.get("returncode", 1),
                "stdout": _tail(res.get("stdout") or ""), "stderr": _tail(res.get("stderr") or ""),
                "skipped": False, "tool": "tsc --noEmit"
            }
            steps.append(step)
            logs.append(f"## node: typecheck (tsc --noEmit)\n$ rc={step['rc']}\n{step['stdout']}\n{step['stderr']}")
        else:
            steps.append({"name": "typecheck", "ok": True, "rc": 0, "stdout": "", "stderr": "", "skipped": True})
            logs.append("## node: typecheck — skipped (TypeScript not detected)")

        # Build (if script exists)
        if "build" in scripts:
            res = _rc(self._run_script_cmd("build"), cwd=str(self.root), timeout=float(os.getenv("AIDEV_NODE_BUILD_TIMEOUT", "1800")))
            ok = (res.get("returncode") == 0) and not bool(res.get("timed_out"))
            step = {
                "name": "build", "ok": ok, "rc": res.get("returncode", 1),
                "stdout": _tail(res.get("stdout") or ""), "stderr": _tail(res.get("stderr") or ""),
                "skipped": False, "tool": f"{self.pm} run build"
            }
            steps.append(step)
            logs.append(f"## node: build ({self.pm} run build)\n$ rc={step['rc']}\n{step['stdout']}\n{step['stderr']}")
        else:
            steps.append({"name": "build", "ok": True, "rc": 0, "stdout": "", "stderr": "", "skipped": True})
            logs.append("## node: build — skipped (no build script)")

        # Tests (if script meaningful)
        test_script = scripts.get("test")
        if isinstance(test_script, str) and "no test specified" not in test_script:
            res = _rc(self._run_script_cmd("test"), cwd=str(self.root), timeout=float(os.getenv("AIDEV_NODE_TEST_TIMEOUT", "2400")))
            ok = (res.get("returncode") == 0) and not bool(res.get("timed_out"))
            step = {
                "name": "tests", "ok": ok, "rc": res.get("returncode", 1),
                "stdout": _tail(res.get("stdout") or ""), "stderr": _tail(res.get("stderr") or ""),
                "skipped": False, "tool": f"{self.pm} test"
            }
            steps.append(step)
            logs.append(f"## node: tests ({self.pm} test)\n$ rc={step['rc']}\n{step['stdout']}\n{step['stderr']}")
        else:
            steps.append({"name": "tests", "ok": True, "rc": 0, "stdout": "", "stderr": "", "skipped": True})
            logs.append("## node: tests — skipped (no test script)")

        total_ok = all(s["ok"] for s in steps if not s.get("skipped"))
        return {
            "ok": bool(total_ok),
            "duration_sec": round(time.time() - t0, 3),
            "steps": steps,
            "logs": ("\n".join(logs).rstrip() + "\n"),
        }

    # --- helpers ---
    def _read_pkg(self) -> Dict[str, object]:
        p = self.root / "package.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _pick_pm(self) -> str:
        if (self.root / "pnpm-lock.yaml").exists() and shutil.which("pnpm"):
            return "pnpm"
        if (self.root / "yarn.lock").exists() and shutil.which("yarn"):
            return "yarn"
        return "npm"

    def _install_cmd(self) -> Optional[List[str]]:
        if self.pm == "pnpm" and (self.root / "pnpm-lock.yaml").exists():
            return ["pnpm", "install", "--frozen-lockfile"]
        if self.pm == "yarn" and (self.root / "yarn.lock").exists():
            return ["yarn", "install", "--frozen-lockfile"]
        if self.pm == "npm" and (self.root / "package-lock.json").exists():
            return ["npm", "ci"]
        return None

    def _run_script_cmd(self, script: str) -> List[str]:
        if self.pm == "yarn":
            return ["yarn", script]
        if self.pm == "pnpm":
            return ["pnpm", "run", script]
        return ["npm", "run", script, "--if-present", "--silent"]

    def _eslint_cmd(self) -> List[str]:
        return ["npx", "--yes", "eslint", ".", "-f", "unix"]

    def _tsc_cmd(self) -> List[str]:
        return ["npx", "--yes", "tsc", "--noEmit", "--pretty", "false"]

    def _has_eslint(self, pkg: Dict[str, object]) -> bool:
        if (self.root / ".eslintrc").exists() or any((self.root.glob(".eslintrc.*"))):
            return True
        deps = {**(pkg.get("devDependencies") or {}), **(pkg.get("dependencies") or {})} if pkg else {}
        return "eslint" in deps or shutil.which("eslint") is not None

    def _has_typescript(self, pkg: Dict[str, object]) -> bool:
        if (self.root / "tsconfig.json").exists():
            return True
        deps = {**(pkg.get("devDependencies") or {}), **(pkg.get("dependencies") or {})} if pkg else {}
        return "typescript" in deps or shutil.which("tsc") is not None


__all__ = ["NodeTools"]
