# aidev/workspace.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ----- Candidate model ---------------------------------------------------------

@dataclass
class ProjectCandidate:
    path: Path
    score: int
    kind: Optional[str]
    markers: List[str]

    # convenience when API returns dicts
    @classmethod
    def from_parts(cls, path: Path, score: int, kind: Optional[str], markers: Iterable[str]) -> "ProjectCandidate":
        return cls(path=path, score=score, kind=kind, markers=list(markers))


# ----- Heuristics -------------------------------------------------------------

_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode",
    "node_modules", ".venv", "venv", "__pycache__",
    "target", "build", "dist", ".next", ".turbo", ".dart_tool",
}

def _looks_python(p: Path) -> Tuple[int, List[str]]:
    marks: List[str] = []
    if (p / "pyproject.toml").exists(): marks.append("pyproject.toml")
    if (p / "requirements.txt").exists(): marks.append("requirements.txt")
    if (p / "setup.cfg").exists() or (p / "setup.py").exists(): marks.append("setup")
    if (p / "manage.py").exists(): marks.append("manage.py")
    score = (3 if "pyproject.toml" in marks else 0) + (2 if "requirements.txt" in marks else 0) + len(marks)
    return score, marks

def _looks_node(p: Path) -> Tuple[int, List[str]]:
    marks: List[str] = []
    if (p / "package.json").exists(): marks.append("package.json")
    if (p / "next.config.js").exists() or (p / "next.config.mjs").exists(): marks.append("next")
    if (p / "vite.config.ts").exists() or (p / "vite.config.js").exists(): marks.append("vite")
    score = (3 if "package.json" in marks else 0) + len(marks)
    return score, marks

def _looks_flutter(p: Path) -> Tuple[int, List[str]]:
    marks: List[str] = []
    if (p / "pubspec.yaml").exists(): marks.append("pubspec.yaml")
    if (p / "android").exists(): marks.append("android/")
    if (p / "ios").exists(): marks.append("ios/")
    score = (3 if "pubspec.yaml" in marks else 0) + len(marks)
    return score, marks

def _looks_rust(p: Path) -> Tuple[int, List[str]]:
    marks: List[str] = []
    if (p / "Cargo.toml").exists(): marks.append("Cargo.toml")
    return (3 if marks else 0), marks

def _looks_go(p: Path) -> Tuple[int, List[str]]:
    marks: List[str] = []
    if (p / "go.mod").exists(): marks.append("go.mod")
    return (3 if marks else 0), marks

def _score_dir(p: Path) -> Optional[ProjectCandidate]:
    kind = None
    best_score = 0
    best_marks: List[str] = []

    for k, f in (
        ("python", _looks_python),
        ("node", _looks_node),
        ("flutter", _looks_flutter),
        ("rust", _looks_rust),
        ("go", _looks_go),
    ):
        s, m = f(p)
        if s > best_score:
            best_score, best_marks, kind = s, m, k

    if best_score > 0:
        return ProjectCandidate.from_parts(p.resolve(), best_score, kind, best_marks)
    return None


# ----- Discovery --------------------------------------------------------------

def _iter_dirs(root: Path, max_depth: int) -> Iterable[Path]:
    """Depth-limited directory walk (breadth-first), skipping heavy/hidden folders."""
    root = root.resolve()
    queue: List[Tuple[Path, int]] = [(root, 0)]
    seen: set[Path] = set()

    while queue:
        cur, depth = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        yield cur
        if depth >= max_depth:
            continue
        try:
            for child in cur.iterdir():
                if not child.is_dir():
                    continue
                name = child.name
                if name in _SKIP_DIRS or name.startswith(".") and name not in {".git"}:
                    continue
                queue.append((child, depth + 1))
        except Exception:
            # Ignore permission errors or transient issues
            continue


def find_projects(root: Path, max_depth: int = 3, max_projects: int = 200) -> List[ProjectCandidate]:
    """
    Scan 'root' and subdirectories up to 'max_depth' for recognizable projects.
    Returns candidates sorted by score desc, then by path.
    """
    root = Path(root).expanduser().resolve()
    cands: List[ProjectCandidate] = []
    for d in _iter_dirs(root, max_depth=max_depth):
        cand = _score_dir(d)
        if cand:
            cands.append(cand)
            if len(cands) >= max_projects:
                break

    cands.sort(key=lambda c: (-c.score, str(c.path).lower()))
    return cands


# Legacy helper used by chat route (keep simple best-pick if needed)
def confirm_project_with_user(candidates: List[ProjectCandidate]) -> Optional[ProjectCandidate]:
    if not candidates:
        return None
    # choose best scoring
    return max(candidates, key=lambda c: c.score)
