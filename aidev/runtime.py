# aidev/runtime.py
from __future__ import annotations

import os
import subprocess
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .io_utils import apply_unified_patch  # reuse your patch helper

# -------------------------
# Back-compat (single-file helpers)
# -------------------------

@dataclass
class RuntimeProto:
    name: str

    def applies_to(self, path: Path) -> bool: ...
    def format_file(self, path: Path, text: str) -> str: ...


class _SimpleRuntime(RuntimeProto):
    """
    Legacy, single-file formatter. We keep this so any older call-sites
    that rely on per-file normalization still work. The orchestrator,
    however, uses detect_runtimes() objects that expose format/lint/test.
    """
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def applies_to(self, path: Path) -> bool:
        s = path.suffix.lower()
        if self.name == "python":
            return s == ".py"
        if self.name == "javascript":
            return s in {".js", ".cjs", ".mjs", ".jsx", ".ts", ".tsx"}
        if self.name == "dart":
            return s == ".dart"
        if self.name == "php":
            return s == ".php"
        return False

    def format_file(self, path: Path, text: str) -> str:
        # Lightweight normalization only; real formatters run via runtimes
        if self.name in {"python", "php", "dart", "javascript"}:
            lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            lines = [ln.rstrip() for ln in lines]
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        return text


# -------------------------
# Runtime discovery
# -------------------------

def _safe_import(modname: str) -> Any | None:
    try:
        return __import__(modname, fromlist=["*"])
    except Exception:
        return None


class _Adapter:
    """
    Wrap a runtime tools instance and provide a stable surface area:
      - format() / run_format()
      - lint() / run_lint()
      - test() / run_tests()
      - run_checks(project_root) if the underlying tool supports it
    If a tool doesn't implement a method, we return a benign record.
    """
    def __init__(self, name: str, tool: Any) -> None:
        self.name = name
        self._tool = tool

    # Optional full-suite check
    def run_checks(self, project_root: Path) -> dict:
        fn = getattr(self._tool, "run_checks", None)
        if callable(fn):
            return fn(project_root)
        # No full-suite; let callers fall back to format/lint/test.
        raise NotImplementedError("run_checks unsupported for this runtime")

    # Orchestrator/fallback calls one of these variants:
    def format(self) -> dict:
        fn = getattr(self._tool, "format", None)
        if callable(fn):
            return fn()
        return {
            "step": "format",
            "tool": self.name,
            "present": False,
            "ran": False,
            "ok": True,
            "note": "unsupported",
        }

    def run_format(self) -> dict:  # alias
        return self.format()

    def lint(self) -> dict:
        fn = getattr(self._tool, "lint", None)
        if callable(fn):
            return fn()
        return {
            "step": "lint",
            "tool": self.name,
            "present": False,
            "ran": False,
            "ok": True,
            "note": "unsupported",
        }

    def run_lint(self) -> dict:  # alias
        return self.lint()

    def test(self) -> dict:
        fn = getattr(self._tool, "test", None)
        if callable(fn):
            return fn()
        return {
            "step": "test",
            "tool": self.name,
            "present": False,
            "ran": False,
            "ok": True,
            "note": "unsupported",
        }

    def run_tests(self) -> dict:  # alias
        return self.test()


def _looks_like_python(root: Path) -> bool:
    if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists() or (root / "setup.cfg").exists():
        return True
    # Avoid materializing the entire tree; bail on first hit
    for _ in root.rglob("*.py"):
        return True
    return False


def _looks_like_node(root: Path) -> bool:
    if (root / "package.json").exists():
        return True
    patterns = ("*.js", "*.cjs", "*.mjs", "*.jsx", "*.ts", "*.tsx")
    for pat in patterns:
        for _ in root.rglob(pat):
            return True
    return False


def _looks_like_flutter(root: Path) -> bool:
    if (root / "pubspec.yaml").exists():
        return True
    for _ in root.rglob("*.dart"):
        return True
    return False


def _looks_like_php(root: Path) -> bool:
    if (root / "composer.json").exists():
        return True
    for _ in root.rglob("*.php"):
        return True
    return False


def detect_runtimes(project_root: Path) -> List[Any]:
    """
    Return a list of runtime tool adapters for the given project root.
    Each adapter exposes format/lint/test (+ run_format/run_lint/run_tests aliases).
    If the underlying tool implements run_checks(project_root), the adapter exposes that too.
    """
    root = Path(project_root).resolve()
    runtimes: List[Any] = []

    # Python
    if _looks_like_python(root):
        py_mod = _safe_import("runtimes.python")
        if py_mod and hasattr(py_mod, "PythonTools"):
            runtimes.append(_Adapter("python", py_mod.PythonTools(str(root))))
        else:
            runtimes.append(_Adapter("python", object()))

    # Node / JS / TS
    if _looks_like_node(root):
        node_mod = _safe_import("runtimes.node")
        if node_mod and hasattr(node_mod, "NodeTools"):
            runtimes.append(_Adapter("node", node_mod.NodeTools(str(root))))
        else:
            runtimes.append(_Adapter("node", object()))

    # Flutter / Dart — prefer a class with run_checks if present
    if _looks_like_flutter(root):
        fl_mod = _safe_import("runtimes.flutter")
        tool = None
        if fl_mod:
            tool = getattr(fl_mod, "FlutterRuntime", None) or getattr(fl_mod, "FlutterTools", None)
        if tool:
            runtimes.append(_Adapter("flutter", tool(str(root))))
        else:
            runtimes.append(_Adapter("flutter", object()))

    # PHP (optional; opportunistic)
    if _looks_like_php(root):
        php_mod = _safe_import("runtimes.php")
        tool_cls = None
        if php_mod:
            tool_cls = getattr(php_mod, "PHPTools", None) or getattr(php_mod, "PHPRuntime", None)
        if tool_cls:
            runtimes.append(_Adapter("php", tool_cls(str(root))))  # type: ignore[arg-type]
        else:
            runtimes.append(_Adapter("php", object()))

    return runtimes


def probe_runtime_for_path(p: Path) -> Optional[str]:
    s = p.suffix.lower()
    if s == ".py":
        return "python"
    if s in {".js", ".cjs", ".mjs", ".jsx", ".ts", ".tsx"}:
        return "javascript"
    if s == ".dart":
        return "dart"
    if s == ".php":
        return "php"
    return None


# -------------------------
# Shared pre-apply checks helper
# -------------------------

def build_file_overlays_from_edits(
    project_root: Path,
    edits: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Build {rel_path: final_content} by applying edits in order on top of disk baseline.

    Supports:
      - {path, content}
      - {path, patch_unified|patch|diff}
      - ignores sentinel entries like {"file_overlays": {...}}

    Returns:
      (overlays, errors)
    """
    root = Path(project_root).resolve()
    overlays: Dict[str, str] = {}
    errors: List[Dict[str, Any]] = []

    for e in edits or []:
        if not isinstance(e, dict):
            continue

        # ignore overlay-only sentinel
        if "file_overlays" in e and not e.get("path"):
            continue

        rel = e.get("path")
        if not isinstance(rel, str) or not rel.strip():
            continue
        rel = rel.strip()

        # base is: current overlay if present else disk baseline
        if rel in overlays:
            base = overlays[rel]
        else:
            try:
                p = (root / rel).resolve()
                # safety: ensure in root
                if p != root and root not in p.parents:
                    errors.append({"path": rel, "error": "unsafe_path_outside_root", "kind": "path"})
                    continue
                base = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
            except Exception as ex:
                errors.append({"path": rel, "error": f"read_baseline_failed: {ex}", "kind": "read"})
                continue

        # full content wins
        content = e.get("content")
        if isinstance(content, str):
            overlays[rel] = content
            continue

        # patch forms
        patch_text = None
        for key in ("patch_unified", "patch", "diff"):
            val = e.get(key)
            if isinstance(val, str) and val.strip():
                patch_text = val
                break
        if not patch_text:
            continue

        try:
            base_norm = base.replace("\r\n", "\n").replace("\r", "\n")
            patched = apply_unified_patch(base_norm, patch_text)
            if not isinstance(patched, str):
                raise TypeError("apply_unified_patch did not return str")
            overlays[rel] = patched
        except Exception as ex:
            errors.append(
                {
                    "path": rel,
                    "kind": "patch_apply_failed",
                    "error": str(ex),
                    "patch_tail": _tail_bytes(patch_text, 300),
                    "base_tail": _tail_bytes(base, 300),
                }
            )
            # Important: do NOT silently “keep base” in overlays; better to omit it
            # so callers can decide to fail-fast or warn.
            if rel in overlays:
                del overlays[rel]

    return overlays, errors


def _default_copy_ignore() -> Iterable[str]:
    return (
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        "__pycache__",
        "*.pyc",
        "node_modules",
        "dist",
        "build",
        "out",
        ".dart_tool",
        ".gradle",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "target",
    )


def _tail_bytes(s: str, nbytes: int) -> str:
    if not s:
        return ""
    enc = s.encode("utf-8", errors="ignore")
    return enc[-nbytes:].decode("utf-8", errors="ignore") if len(enc) > nbytes else s


def _is_pytest_no_tests(returncode: int, stdout: str, stderr: str) -> bool:
    """
    Heuristically detect the 'no tests collected / no tests ran' case for pytest.

    - pytest uses exit code 5 for 'no tests collected'.
    - Some environments may still show helpful text in stdout/stderr.

    We treat this as a WARNING (non-fatal) in preapply checks.
    """
    if returncode == 5:
        return True

    text = f"{stdout or ''}\n{stderr or ''}".lower()
    markers = (
        "collected 0 items",
        "no tests ran",
        "no tests collected",
        "did not collect any tests",
    )
    return any(m in text for m in markers)


def _overlay_edits(tmp_root: Path, edits: Sequence[Dict[str, Any]]) -> None:
    """
    Apply `edits` into `tmp_root`.

    Each item may be one of:
      - {path, content}
      - {path, patch_unified}
      - {path, patch}
      - {path, diff}   # alias for unified diff (useful for HTTP payloads)

    Additionally, callers may include a special entry with a `file_overlays` dict
    mapping relative paths -> full file content. This helps when patches were
    generated against an *updated* version of files (for example, after a
    repair pass) so we prefer using the provided overlay content as the base
    when applying unified patches. This implements the recommendation to run
    repairs using updated file content instead of falling back to the on-disk
    file when patches do not apply cleanly.
    """
    # Collect any provided overlays: a mapping rel_path -> content
    overlays: Dict[str, str] = {}
    for e in edits or []:
        if isinstance(e, dict) and "file_overlays" in e and isinstance(e["file_overlays"], dict):
            for k, v in e["file_overlays"].items():
                try:
                    overlays[str(k)] = str(v)
                except Exception:
                    # ignore non-stringable keys/values
                    continue

    for e in edits or []:
        # Skip overlay-only entries
        if isinstance(e, dict) and "file_overlays" in e and not e.get("path"):
            continue

        rel = e.get("path")
        if not isinstance(rel, str) or not rel:
            continue

        dest = (tmp_root / rel).resolve()

        # Safety: ensure inside tmp_root
        if dest != tmp_root and tmp_root not in dest.parents:
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        old_text = ""
        if dest.exists():
            old_text = dest.read_text(encoding="utf-8", errors="ignore")

        if "content" in e and isinstance(e["content"], str):
            # Preferred path: caller already computed the final file contents.
            new_text = e["content"]
            dest.write_text(new_text, encoding="utf-8")
            continue

        # Prefer explicit unified diff fields, but accept 'diff' as an alias.
        patch_text = None
        for key in ("patch_unified", "patch", "diff"):
            val = e.get(key)
            if isinstance(val, str):
                patch_text = val
                break

        if not isinstance(patch_text, str) or not patch_text.strip():
            # Nothing to apply for this entry
            continue

        # Try applying the patch against multiple candidate bases.
        # Prefer an overlay base (if provided) so repairs computed against
        # updated content can be applied.
        base_candidates: List[str] = []
        if rel in overlays:
            base_candidates.append(overlays[rel])

        # legacy/primary base: current on-disk content
        base_candidates.append(old_text)

        # allow callers to supply explicit base content keys
        for candidate_key in ("base_content", "base", "orig_content", "original_content"):
            val = e.get(candidate_key)
            if isinstance(val, str):
                # avoid duplicate entries
                if val not in base_candidates:
                    base_candidates.append(val)

        new_text: Optional[str] = None
        for base in base_candidates:
            try:
                base_norm = base.replace("\r\n", "\n").replace("\r", "\n")
                attempted = apply_unified_patch(base_norm, patch_text)
                # If apply_unified_patch returned something, accept it.
                if isinstance(attempted, str):
                    new_text = attempted
                    break
            except Exception:
                # Try the next candidate base if the patch doesn't apply cleanly.
                continue

        if new_text is None:
            # Emit a structured trace so callers/telemetry can distinguish
            # a fallback due to inability to apply the patch from other
            # recovery behaviors. This is best-effort: if the optional
            # `st` tracing helper isn't available we silently continue.
            try:
                import st  # type: ignore
            except Exception:
                st = None

            attempted_bases = []
            try:
                for b in base_candidates:
                    attempted_bases.append({
                        "len": len(b),
                        "tail": _tail_bytes(b, 200),
                    })
            except Exception:
                attempted_bases = ["<could not summarize bases>"]

            trace_payload = {
                "event": "preapply_recovery",
                "reason": "fallback_due_to_patch_apply_failure",
                "path": rel,
                "attempted_bases": attempted_bases,
                "patch_preview": _tail_bytes(patch_text or "", 200),
            }

            try:
                if st and hasattr(st, "trace") and hasattr(st.trace, "write"):
                    # Best-effort; do not error if tracing fails.
                    st.trace.write(trace_payload)
            except Exception:
                pass

            # Nothing we could apply; skip this entry.
            continue

        dest.write_text(new_text, encoding="utf-8")


def run_preapply_checks(
    project_root: Path,
    edits: Sequence[Dict[str, Any]],
    *,
    tail_limit_bytes: int = 4000,
    copy_ignore: Optional[Iterable[str]] = None,
    file_overlays: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Make a temp copy of the project, overlay `edits`, and run cheap validation
    commands (syntax / lint / light tests) in a sandbox.

    Returns:
        (ok, details) where details is a dict:

        {
          "changed_files": {
              "python": [...],
              "javascript": [...],
              "dart": [...],
              "other": [...],
          },
          "checks": [
            {
              "id": "python_syntax",
              "label": "Python syntax (py_compile)",
              "kind": "syntax",
              "tool": "python",
              "ok": bool,
              "ran": bool,
              "skipped": bool,
              "reason": str | None,
              "duration_sec": float,
              "returncode": int | None,
              "stdout_tail": str,
              "stderr_tail": str,
              "files": [...],
            },
            ...
          ],
          "errors": [
            {
              "check_id": str,
              "tool": str,
              "kind": str,
              "message": str,
              "files": [...],
              "severity": "error" | "warning",
            },
            ...
          ],
        }

    Notes:
    - Missing tools (ruff, eslint, dart, etc.) are treated as "skipped"
      and do NOT cause failure.
    - Black failures are recorded as WARNINGS but do not block apply.
    - pytest "no tests collected / no tests ran" is recorded as a WARNING
      and does NOT cause failure.
    - If no checks run at all, ok=True with an empty checks list.

    Additional keyword-only parameter:
    - file_overlays: Optional[Dict[str, str]] -- mapping of relative file paths
      to full file content to be used as preferred bases when applying unified
      patches. If provided, these overlays will be appended as a sentinel
      entry to the edits passed to _overlay_edits so patches can be applied
      against updated file snapshots.
    """
    root = Path(project_root).resolve()
    ignore_patterns = shutil.ignore_patterns(*(copy_ignore or _default_copy_ignore()))

    # Collect changed files by language based on edit paths
    changed_paths: List[str] = []
    py_files: List[str] = []
    js_files: List[str] = []
    ts_files: List[str] = []
    dart_files: List[str] = []
    other_files: List[str] = []

    for e in edits or []:
        rel = e.get("path")
        if not isinstance(rel, str) or not rel.strip():
            continue
        rel = rel.strip()
        changed_paths.append(rel)
        suffix = Path(rel).suffix.lower()
        if suffix == ".py":
            py_files.append(rel)
        elif suffix in {".js", ".cjs", ".mjs", ".jsx"}:
            js_files.append(rel)
        elif suffix in {".ts", ".tsx"}:
            ts_files.append(rel)
        elif suffix == ".dart":
            dart_files.append(rel)
        else:
            other_files.append(rel)

    # If overlays were provided, include them in the changed files list as well
    # so that subsequent checks consider files that were supplied as overlays.
    if file_overlays:
        for k in file_overlays.keys():
            try:
                rel = str(k).strip()
            except Exception:
                continue
            if not rel:
                continue
            if rel in changed_paths:
                continue
            changed_paths.append(rel)
            suffix = Path(rel).suffix.lower()
            if suffix == ".py":
                py_files.append(rel)
            elif suffix in {".js", ".cjs", ".mjs", ".jsx"}:
                js_files.append(rel)
            elif suffix in {".ts", ".tsx"}:
                ts_files.append(rel)
            elif suffix == ".dart":
                dart_files.append(rel)
            else:
                other_files.append(rel)

    # Deduplicate + sort
    def _uniq_sorted(items: List[str]) -> List[str]:
        return sorted({p for p in items})

    py_files = _uniq_sorted(py_files)
    js_files = _uniq_sorted(js_files)
    ts_files = _uniq_sorted(ts_files)
    dart_files = _uniq_sorted(dart_files)
    other_files = _uniq_sorted(other_files)

    details: Dict[str, Any] = {
        "changed_files": {
            "python": py_files,
            "javascript": js_files + ts_files,
            "dart": dart_files,
            "other": other_files,
        },
        "checks": [],
        "errors": [],
    }

    # Shortcut: if there are no edits at all, treat as ok.
    if not changed_paths:
        return True, details

    with tempfile.TemporaryDirectory(prefix="aidev-preapply-") as tmpdir:
        tmp_root = Path(tmpdir) / "repo"
        shutil.copytree(root, tmp_root, ignore=ignore_patterns)

        # Apply overlay edits into temp repo.
        # If file_overlays is provided, append a sentinel entry so the
        # overlay logic in _overlay_edits can prefer those bases when
        # attempting to apply unified patches.
        edits_to_apply: List[Dict[str, Any]] = list(edits) if edits else []
        if file_overlays:
            edits_to_apply = edits_to_apply + [{"file_overlays": file_overlays}]

        _overlay_edits(tmp_root, edits_to_apply)

        any_fail = False

        def _run_check(
            check_id: str,
            *,
            label: str,
            kind: str,
            tool: str,
            cmd: Sequence[str],
            files: Optional[Sequence[str]] = None,
        ) -> None:
            nonlocal any_fail

            files = list(files or [])
            record: Dict[str, Any] = {
                "id": check_id,
                "label": label,
                "kind": kind,
                "tool": tool,
                "files": files,
                "ok": True,
                "ran": False,
                "skipped": False,
                "reason": None,
                "duration_sec": 0.0,
                "returncode": None,
                "stdout_tail": "",
                "stderr_tail": "",
            }

            if not cmd:
                record["skipped"] = True
                record["reason"] = "no_command"
                details["checks"].append(record)
                return

            try:
                t0 = time.time()
                proc = subprocess.run(
                    list(cmd),
                    cwd=str(tmp_root),
                    capture_output=True,
                    text=True,
                )
                dt = time.time() - t0

                record["ran"] = True
                record["duration_sec"] = float(dt)
                record["returncode"] = int(proc.returncode)
                record["stdout_tail"] = _tail_bytes(proc.stdout or "", tail_limit_bytes)
                record["stderr_tail"] = _tail_bytes(proc.stderr or "", tail_limit_bytes)

                ok = proc.returncode == 0
                record["ok"] = bool(ok)

                if not ok:
                    is_pytest = check_id == "python_pytest"
                    # Special-case pytest "no tests collected / no tests ran" → warning, not failure.
                    if is_pytest and _is_pytest_no_tests(
                        proc.returncode,
                        record["stdout_tail"],
                        record["stderr_tail"],
                    ):
                        record["ok"] = True
                        record["skipped"] = True
                        record["reason"] = "no_tests_collected"
                        details["errors"].append(
                            {
                                "check_id": check_id,
                                "tool": tool,
                                "kind": kind,
                                "message": "pytest did not find any tests to run; treating as a warning, not a failure.",
                                "files": files,
                                "severity": "warning",
                            }
                        )
                    else:
                        # Black failures are advisory: warn but don't flip any_fail.
                        severity = "warning" if check_id == "python_black" else "error"
                        message = (
                            record["stderr_tail"]
                            or record["stdout_tail"]
                            or f"{label} failed with exit code {proc.returncode}"
                        )
                        details["errors"].append(
                            {
                                "check_id": check_id,
                                "tool": tool,
                                "kind": kind,
                                "message": message,
                                "files": files,
                                "severity": severity,
                            }
                        )
                        if severity == "error":
                            any_fail = True

            except FileNotFoundError:
                # Tool not installed → skip but do not fail the run.
                record["ok"] = True
                record["ran"] = False
                record["skipped"] = True
                record["reason"] = "command_not_found"
            except Exception as e:
                # Unexpected failure executing the command → mark as failed.
                record["ok"] = False
                record["ran"] = False
                record["skipped"] = True
                record["reason"] = f"exception: {type(e).__name__}: {e}"
                any_fail = True
                details["errors"].append(
                    {
                        "check_id": check_id,
                        "tool": tool,
                        "kind": kind,
                        "message": record["reason"],
                        "files": files,
                        "severity": "error",
                    }
                )

            details["checks"].append(record)

        # -----------------------------
        # Python checks
        # -----------------------------
        if py_files:
            # Syntax: python -m py_compile
            _run_check(
                "python_syntax",
                label="Python syntax (py_compile)",
                kind="syntax",
                tool="python",
                cmd=["python", "-m", "py_compile", *py_files],
                files=py_files,
            )

            # Linter: ruff (optional)
            _run_check(
                "python_ruff",
                label="Python linter (ruff)",
                kind="lint",
                tool="python",
                cmd=["ruff", "check", *py_files],
                files=py_files,
            )

            # Formatter: black --check (optional, advisory)
            _run_check(
                "python_black",
                label="Python formatter (black --check)",
                kind="format",
                tool="python",
                cmd=["black", "--check", *py_files],
                files=py_files,
            )

            # Optional smoke tests: pytest, if tests/ exists
            tests_dir = (tmp_root / "tests")
            if tests_dir.exists() and tests_dir.is_dir():
                _run_check(
                    "python_pytest",
                    label="Python tests (pytest -q)",
                    kind="test",
                    tool="python",
                    cmd=["pytest", "-q"],
                    files=py_files,
                )

        # -----------------------------
        # JS / TS checks
        # -----------------------------
        js_ts_files = js_files + ts_files
        if js_ts_files:
            # Syntax: node --check
            _run_check(
                "node_syntax",
                label="Node syntax (node --check)",
                kind="syntax",
                tool="node",
                cmd=["node", "--check", *js_ts_files],
                files=js_ts_files,
            )

            # Linter: eslint (optional)
            _run_check(
                "node_eslint",
                label="ESLint",
                kind="lint",
                tool="node",
                cmd=["eslint", *js_ts_files],
                files=js_ts_files,
            )

            # Formatter: prettier --check (optional)
            _run_check(
                "node_prettier",
                label="Prettier (--check)",
                kind="format",
                tool="node",
                cmd=["prettier", "--check", *js_ts_files],
                files=js_ts_files,
            )

        # TypeScript project-wide check if tsconfig.json is present
        if ts_files and (tmp_root / "tsconfig.json").exists():
            _run_check(
                "tsc_no_emit",
                label="TypeScript compile (tsc --noEmit)",
                kind="syntax",
                tool="node",
                cmd=["tsc", "--noEmit"],
                files=ts_files,
            )

        # -----------------------------
        # Dart / Flutter checks
        # -----------------------------
        if dart_files:
            # dart analyze (project-wide)
            _run_check(
                "dart_analyze",
                label="Dart analyze",
                kind="analyze",
                tool="dart",
                cmd=["dart", "analyze"],
                files=dart_files,
            )

            # dart format as a quick formatting sanity check
            _run_check(
                "dart_format",
                label="Dart format (--output=none)",
                kind="format",
                tool="dart",
                cmd=["dart", "format", "--output=none", *dart_files],
                files=dart_files,
            )

        # If nothing actually ran (all skipped / no commands), still treat as ok
        if not details["checks"]:
            return True, details

        return (not any_fail), details
