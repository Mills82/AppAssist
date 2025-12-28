# runtimes/flutter.py
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict

from runtimes.base import BaseRuntime
from runtimes.runner import run_command

_TAIL = int(os.getenv("AIDEV_FLUTTER_LOG_TAIL", "2000"))


def _tail(s: str, n: int = _TAIL) -> str:
    s = s or ""
    return s if len(s) <= n else s[-n:]


class FlutterTools:
    """
    Opportunistic Flutter/Dart quality gates:
      - format: dart format -o write .
      - lint:   flutter analyze --no-pub
      - test:   flutter test   (or skip if no tests)
    """
    name = "flutter"

    def __init__(self, project_root: str):
        self.root = Path(project_root).resolve()

    @staticmethod
    def detect(project_root: str) -> bool:
        return (Path(project_root) / "pubspec.yaml").exists()

    def _has(self, exe: str) -> bool:
        return shutil.which(exe) is not None

    def format(self) -> dict:
        if self._has("dart"):
            res = run_command(
                ["dart", "format", "-o", "write", "."],
                cwd=str(self.root),
                timeout=float(os.getenv("AIDEV_FLUTTER_FORMAT_TIMEOUT", "120")),
            )
        elif self._has("flutter"):
            res = run_command(
                ["flutter", "format", "."],
                cwd=str(self.root),
                timeout=float(os.getenv("AIDEV_FLUTTER_FORMAT_TIMEOUT", "120")),
            )
        else:
            return {"step": "format", "tool": "dart/flutter", "present": False, "ran": False, "ok": True, "note": "missing"}

        return {
            "step": "format",
            "tool": "dart/flutter",
            "present": True,
            "ran": True,
            "ok": res.get("returncode") == 0 and not bool(res.get("timed_out")),
            "stdout": _tail(res.get("stdout") or ""),
            "stderr": _tail(res.get("stderr") or ""),
        }

    def lint(self) -> dict:
        if not self._has("flutter"):
            return {"step": "lint", "tool": "flutter", "present": False, "ran": False, "ok": True, "note": "missing"}
        res = run_command(
            ["flutter", "analyze", "--no-pub"],
            cwd=str(self.root),
            timeout=float(os.getenv("AIDEV_FLUTTER_ANALYZE_TIMEOUT", "180")),
        )
        return {
            "step": "lint",
            "tool": "flutter",
            "present": True,
            "ran": True,
            "ok": res.get("returncode") == 0 and not bool(res.get("timed_out")),
            "stdout": _tail(res.get("stdout") or ""),
            "stderr": _tail(res.get("stderr") or ""),
        }

    def test(self) -> dict:
        if not self._has("flutter"):
            return {"step": "test", "tool": "flutter", "present": False, "ran": False, "ok": True, "note": "missing"}
        has_tests = (self.root / "test").exists() and any((self.root / "test").rglob("*_test.dart"))
        if not has_tests:
            return {"step": "test", "tool": "flutter", "present": True, "ran": False, "ok": True, "note": "no tests"}
        res = run_command(
            ["flutter", "test"],
            cwd=str(self.root),
            timeout=float(os.getenv("AIDEV_FLUTTER_TEST_TIMEOUT", "600")),
        )
        return {
            "step": "test",
            "tool": "flutter",
            "present": True,
            "ran": True,
            "ok": res.get("returncode") == 0 and not bool(res.get("timed_out")),
            "stdout": _tail(res.get("stdout") or ""),
            "stderr": _tail(res.get("stderr") or ""),
        }


class FlutterRuntime(BaseRuntime):
    """
    Full-suite check for Flutter projects:
      1) flutter pub get
      2) flutter analyze --no-pub
      3) flutter test  (or skip if no tests)
    """
    name = "flutter"

    def __init__(self, project_root: str):
        self.root = Path(project_root).resolve()

    def run_checks(self, project_root: Path) -> dict:
        self.root = Path(project_root).resolve()
        if not (self.root / "pubspec.yaml").exists():
            return {"ok": True, "logs": "Not a Flutter project (no pubspec.yaml); skipping.", "duration_sec": 0.0}

        if shutil.which("flutter") is None:
            return {"ok": False, "logs": "flutter not found on PATH.", "duration_sec": 0.0}

        t0 = time.time()
        logs: List[str] = []
        ok_all = True

        def add(title: str, res: Dict[str, object]) -> None:
            nonlocal ok_all
            rc = res.get("returncode")
            tout = bool(res.get("timed_out"))
            ok = (rc == 0) and (not tout)
            ok_all = ok_all and ok
            logs.append(
                f"\n=== {title} (rc={rc}, timeout={tout}) ===\n"
                f"{_tail(res.get('stdout') or '')}\n"
                f"{_tail(res.get('stderr') or '')}"
            )

        # 1) pub get
        res = run_command(
            ["flutter", "pub", "get"],
            cwd=str(self.root),
            timeout=float(os.getenv("AIDEV_FLUTTER_PUB_TIMEOUT", "180")),
        )
        add("flutter pub get", res)

        # 2) analyze
        res = run_command(
            ["flutter", "analyze", "--no-pub"],
            cwd=str(self.root),
            timeout=float(os.getenv("AIDEV_FLUTTER_ANALYZE_TIMEOUT", "240")),
        )
        add("flutter analyze --no-pub", res)

        # 3) tests (skip if none)
        has_tests = (self.root / "test").exists() and any((self.root / "test").rglob("*_test.dart"))
        if has_tests:
            res = run_command(
                ["flutter", "test"],
                cwd=str(self.root),
                timeout=float(os.getenv("AIDEV_FLUTTER_TEST_TIMEOUT", "600")),
            )
            add("flutter test", res)
        else:
            logs.append("\n=== flutter test ===\n(no tests; skipped)")

        return {"ok": bool(ok_all), "logs": "".join(logs).strip(), "duration_sec": round(time.time() - t0, 3)}


__all__ = ["FlutterTools", "FlutterRuntime"]
