# aidev/discovery.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# --------------------------------------------------------------------------------------
# Configuration / Heuristics
# --------------------------------------------------------------------------------------

# Directories we never descend into
IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode",
    "node_modules", "__pycache__", ".venv", "venv", ".tox",
    "dist", "build", "out", "target"
}

# Files that usually indicate a LEAF "project" lives here
PROJECT_MARKERS = {
    # JS/TS
    "package.json",
    # Python
    "pyproject.toml", "setup.py", "requirements.txt",
    # Rust
    "Cargo.toml",
    # Go
    "go.mod",
    # Java/Kotlin
    "pom.xml", "build.gradle", "build.gradle.kts",
    # .NET
    "*.csproj",
    # Your original custom markers
    ".aidev",
    ".git",  # keep as a weak signal so single-repo roots still show up
}

# Files that indicate a MONOREPO ROOT
MONOREPO_ROOT_MARKERS = {
    "pnpm-workspace.yaml",
    "nx.json",
    "turbo.json",
    "rush.json",
    "lerna.json",
    # Sentinels you might drop at root
    ".aidevroot",
    "app_descrip.txt",
}

# Common “container” dirs inside monorepos
MONOREPO_CONTAINER_DIRS = {"packages", "apps", "services", "clients", "examples"}


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------

def _has_glob(path: Path, pattern: str) -> bool:
    """Return True if 'pattern' (literal or glob) exists under path."""
    if "*" in pattern or "?" in pattern or "[" in pattern:
        return any(path.glob(pattern))
    return (path / pattern).exists()


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_within(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Detection helpers
# --------------------------------------------------------------------------------------

def _is_project_dir(p: Path) -> bool:
    """Backward-compatible: any of the classic markers qualifies."""
    markers = [".git", "pyproject.toml", "package.json", ".aidev"]
    return any((p / m).exists() for m in markers)


def _any_project_marker(d: Path) -> bool:
    return any(_has_glob(d, m) for m in PROJECT_MARKERS)


def _detect_project_kind_and_markers(d: Path) -> Tuple[str, List[str]]:
    """Infer a coarse 'kind' plus the markers found."""
    marks: List[str] = []
    kind = "generic"

    if (d / "package.json").exists():
        kind = "node"; marks.append("package.json")

    if (d / "pyproject.toml").exists() or (d / "setup.py").exists():
        kind = "python"
        if (d / "pyproject.toml").exists(): marks.append("pyproject.toml")
        if (d / "setup.py").exists(): marks.append("setup.py")

    if (d / "Cargo.toml").exists():
        kind = "rust"; marks.append("Cargo.toml")

    if (d / "go.mod").exists():
        kind = "go"; marks.append("go.mod")

    if (d / "pom.xml").exists() or _has_glob(d, "build.gradle*"):
        kind = "java"
        if (d / "pom.xml").exists(): marks.append("pom.xml")
        if _has_glob(d, "build.gradle*"): marks.append("build.gradle*")

    # Fallback: record the first generic marker we see
    if kind == "generic":
        for m in PROJECT_MARKERS:
            if _has_glob(d, m):
                marks.append(m)
                break

    return kind, marks


def _is_monorepo_root_dir(d: Path) -> Tuple[bool, List[str]]:
    """Detect whether 'd' looks like a monorepo root, and return matching markers."""
    markers: List[str] = []

    # obvious monorepo root files
    for name in MONOREPO_ROOT_MARKERS:
        if _has_glob(d, name):
            markers.append(name)

    # package.json with "workspaces"
    pj = d / "package.json"
    if pj.exists():
        data = _read_json(pj)
        if isinstance(data, dict) and ("workspaces" in data):
            markers.append('package.json["workspaces"]')

    # git-toplevel heuristic: .git + >=2 child projects
    if (d / ".git").exists():
        child_projects = 0
        for c in d.iterdir():
            if not c.is_dir() or c.name in IGNORE_DIRS:
                continue
            if _any_project_marker(c):
                child_projects += 1
                if child_projects >= 2:
                    break
        if child_projects >= 2:
            markers.append(".git (multi-project)")

    return (len(markers) > 0, markers)


# --------------------------------------------------------------------------------------
# Scanning / grouping
# --------------------------------------------------------------------------------------

def _collect_leaf_candidates(workspace_root: Path, max_depth: int = 5) -> List[Path]:
    """Find directories that look like LEAF projects somewhere under workspace_root."""
    root = workspace_root.resolve()
    out: List[Path] = []

    def walk(dir_path: Path, depth: int):
        if depth > max_depth:
            return

        # If this folder looks like a project, keep it.
        if _any_project_marker(dir_path):
            out.append(dir_path)

        try:
            for entry in dir_path.iterdir():
                if not entry.is_dir():
                    continue
                if entry.name in IGNORE_DIRS:
                    continue
                walk(entry, depth + 1)
        except Exception:
            # best-effort
            return

    walk(root, 0)
    # unique + sorted
    uniq = sorted({p.resolve() for p in out})
    return uniq


def _ascend_to_monorepo_root(path: Path, stop: Path) -> Path:
    """Walk up from 'path' to find a monorepo root, stopping at 'stop' if none."""
    cur = path
    last_good = cur
    while True:
        is_root, _ = _is_monorepo_root_dir(cur)
        if is_root:
            return cur
        parent = cur.parent
        if parent == cur or parent == stop or not str(parent).startswith(str(stop)):
            # no explicit root; return the last project-ish dir we saw
            return last_good
        last_good = cur
        cur = parent


def _count_child_projects(root: Path) -> int:
    """Count child projects under a root using common monorepo containers and root children."""
    cnt = 0

    # look both at root and common container folders
    search_bases = [root]
    search_bases += [root / k for k in MONOREPO_CONTAINER_DIRS if (root / k).exists()]

    for base in search_bases:
        try:
            for entry in base.iterdir():
                if not entry.is_dir() or entry.name in IGNORE_DIRS:
                    continue
                if _any_project_marker(entry):
                    cnt += 1
        except Exception:
            pass

    return cnt


def scan_workspace_projects(
    workspace_root: str | Path,
    *,
    group: bool = True,
    max_depth: int = 5
) -> List[Dict[str, object]]:
    """
    Return project candidates under 'workspace_root'.

    If group=True, collapse monorepo subprojects under a detected root and return
    a single entry per monorepo root with:
        {"path": <root>, "kind": "monorepo", "markers": [...], "children_count": N}

    Otherwise return a flat list of leaf projects.
    """
    ws = Path(workspace_root).resolve()
    leaves = _collect_leaf_candidates(ws, max_depth=max_depth)

    if not group:
        # Flat leaf list
        result: List[Dict[str, object]] = []
        for p in leaves:
            kind, markers = _detect_project_kind_and_markers(p)
            result.append({"path": str(p), "kind": kind, "markers": markers})
        return sorted(result, key=lambda x: str(x["path"]).lower())

    # Group leaves under their monorepo roots
    by_root: Dict[Path, List[Path]] = {}
    for leaf in leaves:
        root = _ascend_to_monorepo_root(leaf, ws)
        by_root.setdefault(root, []).append(leaf)

    result: List[Dict[str, object]] = []
    for root, children in by_root.items():
        is_root, root_markers = _is_monorepo_root_dir(root)

        # If no explicit root markers but multiple children, infer a monorepo
        if not is_root and len(children) >= 2:
            root_markers = ["(inferred monorepo root: >=2 children)"]
            is_root = True

        if is_root:
            kind, _ = _detect_project_kind_and_markers(root)
            result.append({
                "path": str(root),
                "kind": "monorepo" if len(children) >= 2 else (kind or "generic"),
                "markers": list(dict.fromkeys(root_markers)),  # uniq-ish while preserving order
                "children_count": len(children),
            })
        else:
            # Single leaf that didn't collapse upward
            kind, markers = _detect_project_kind_and_markers(root)
            result.append({
                "path": str(root),
                "kind": kind,
                "markers": markers,
                "children_count": 1
            })

    # De-dup by path and stable sort
    uniq: Dict[str, Dict[str, object]] = {r["path"]: r for r in result}
    return sorted(uniq.values(), key=lambda x: str(x["path"]).lower())


# --------------------------------------------------------------------------------------
# Back-compat scanning APIs used by CLI / earlier code
# --------------------------------------------------------------------------------------

def _scan_candidates(start: Path, depth: int = 2) -> List[Path]:
    """
    OLD behavior: return Paths to project-like directories under 'start'.

    We now reuse the richer collector but return the leaf-ish list for compatibility.
    """
    start = start.resolve()
    # Prefer the newer collector (wider set of markers, ignores).
    leaves = _collect_leaf_candidates(start, max_depth=depth)
    # If nothing was found, still allow 'start' itself if it looks like a project.
    if not leaves and _any_project_marker(start):
        leaves = [start]
    # Unique + sorted
    return sorted(set(leaves))


def _dedupe_root_candidates(paths: List[Path]) -> List[Path]:
    """
    OLD behavior: dedupe by removing subdirectories once a parent is selected.
    Still used by interactive picker fallback.
    """
    out: List[Path] = []
    for p in sorted(paths, key=lambda x: len(x.as_posix())):
        if not any(str(p).startswith(str(q) + os.sep) for q in out):
            out.append(p)
    return out


def _parse_index_selection(s: str, n: int) -> List[int]:
    s = s.strip()
    if not s:
        return []
    idxs: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            a_i = max(0, min(n - 1, int(a)))
            b_i = max(0, min(n - 1, int(b)))
            idxs.extend(list(range(min(a_i, b_i), max(a_i, b_i) + 1)))
        else:
            i = max(0, min(n - 1, int(tok)))
            idxs.append(i)
    return sorted(set(idxs))


def interactive_pick_projects_if_needed(
    project_root: Path | None,
    *,
    scan_start: Path,
    scan_depth: int = 2,
    assume_yes: bool = True,
) -> List[Path]:
    """
    Back-compat interactive picker used by CLI.

    - If project_root is provided, return it.
    - Else, scan and (by default) GROUP subprojects under a single monorepo root.
    - If assume_yes: pick the first candidate.
    """
    if project_root:
        return [Path(project_root).resolve()]

    ws = scan_start.resolve()
    grouped = scan_workspace_projects(ws, group=True, max_depth=scan_depth)

    if not grouped:
        # nothing found; fallback to start itself
        return [ws]

    # choose first candidate (same behavior as before with assume_yes)
    if assume_yes:
        return [Path(grouped[0]["path"]).resolve()]

    # If later you want to build a TUI/interactive selector, wire it here.
    return [Path(grouped[0]["path"]).resolve()]
