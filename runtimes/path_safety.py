# runtimes/path_safety.py
"""runtimes.path_safety

Helpers to safely resolve and validate filesystem paths against a selected
project_root.

Key APIs:
- resolve_within_root(project_root, rel_path) -> pathlib.Path:
  Preferred strict API for write entrypoints. It *rejects* absolute paths and any
  path containing ".." traversal parts, resolves under project_root (following
  symlinks), and guarantees the result is an absolute Path within project_root.

- resolve_safe_path(target_path, project_root) -> pathlib.Path:
  Backwards-compatible resolver that accepts absolute or relative paths. If the
  input is relative, it is interpreted as relative to project_root. The resolved
  path must be within project_root.

Note: resolution uses Path.resolve() where possible, which follows symlinks.
Containment checks are performed on the resolved path to prevent escapes via
symlinks.

For callers that still expect a string path, resolve_safe_path_str(...) is
provided, returning the resolved path as a string. The resolve_to_repo_path alias
continues to exist for backward compatibility.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Union


def _to_path(p: Union[str, Path]) -> Path:
    if isinstance(p, Path):
        return p
    return Path(str(p))


def validate_within_root(target_path: Union[str, Path], project_root: Union[str, Path]) -> bool:
    """
    Return True if `target_path` is the same as or located beneath `project_root`.

    `target_path` may be absolute or relative. If relative, it is interpreted
    as relative to `project_root`.
    """
    root = _to_path(project_root).resolve()
    tp = _to_path(target_path)
    # If target is relative, interpret relative to the project root
    candidate = (root / tp) if not tp.is_absolute() else tp
    try:
        candidate = candidate.resolve()
    except Exception:
        # Fall back to absolute path normalization
        candidate = candidate.absolute()
    try:
        candidate.relative_to(root)
        return True
    except Exception:
        return False


def resolve_within_root(project_root: Path, rel_path: str) -> Path:
    """Resolve a repo-relative path within project_root.

    This is the preferred strict API for write entrypoints: it rejects absolute
    paths and any path containing '..' traversal parts, resolves under
    project_root (following symlinks), and guarantees the returned Path is
    absolute and contained within project_root.

    Raises ValueError with deterministic messages on rejection.
    """
    root = Path(project_root).resolve()
    p = Path(rel_path)

    if p.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {rel_path}")
    if any(part == ".." for part in p.parts):
        raise ValueError(f"Path traversal '..' is not allowed: {rel_path}")

    candidate = root / p
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate.absolute()

    try:
        resolved.relative_to(root)
    except Exception:
        raise ValueError(
            f"Refusing path outside project root: attempted='{resolved}' project_root='{root}'"
        )

    return resolved


def resolve_safe_path(target_path: Union[str, Path], project_root: Union[str, Path]) -> Path:
    """
    Resolve `target_path` to an absolute pathlib.Path that is guaranteed to be within
    `project_root`. If the resolved path would be outside the root, raise ValueError.

    Returns the resolved absolute Path on success.
    """
    root = _to_path(project_root).resolve()

    if not root.is_absolute():
        raise ValueError(
            f"project_root must be a real absolute root; got {project_root!r}"
        )
    tp = _to_path(target_path)
    candidate = (root / tp) if not tp.is_absolute() else tp
    try:
        resolved = candidate.resolve()
    except Exception:
        # Best-effort fallback
        resolved = candidate.absolute()

    try:
        resolved.relative_to(root)
    except Exception:
        raise ValueError(
            f"Refusing path outside project root: attempted='{resolved}' project_root='{root}'"
        )
    return resolved


# Backwards/transition alias so callers can import either name.
# Prefer resolve_within_root(...) for strict write entrypoints.
resolve_to_repo_path = resolve_safe_path


def resolve_safe_path_str(target_path: Union[str, Path], project_root: Union[str, Path]) -> str:
    """
    Compatibility wrapper for callers that expect a string path.

    Calls resolve_safe_path(...) and returns the resolved absolute path as a
    string. This preserves the previous string-based API for older callers while
    encouraging new code to use the Path-returning resolve_safe_path.
    """
    resolved_path = resolve_safe_path(target_path, project_root)
    return str(resolved_path)


def resolve_glob_within_root(project_root: Union[str, Path], pattern: str) -> List[str]:
    """
    Expand a single target `pattern` (which may be a plain path or a glob
    pattern containing '*', '?', or '[') into a deterministic sorted list of
    repo-relative paths (POSIX style) that are guaranteed to live within
    `project_root`.

    Rules and guarantees:
    - If `pattern` contains no glob characters, it is treated as a single-file
      spec and validated via resolve_safe_path; the returned list contains one
      repo-relative path (as POSIX string).
    - Absolute patterns are rejected for globs. Patterns containing '..' are
      rejected to avoid root escape attempts.
    - Glob expansion is performed relative to the resolved project_root. Each
      match is resolved (symlinks followed) and must remain within project_root;
      if any matched path resolves outside the root, a ValueError is raised
      listing offending matches.
    - The returned list is deterministic: sorted, unique, and uses POSIX
      separators (Path.relative_to(root).as_posix()).

    Raises ValueError on attempts to escape the root or when matched paths
    resolve outside the root.
    """
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a str")

    root = _to_path(project_root).resolve()
    if not root.is_absolute():
        raise ValueError(f"project_root must be an absolute path; got {project_root!r}")

    glob_chars = set("*?[")
    contains_glob = any((c in pattern) for c in glob_chars)

    # Helper to normalize a resolved Path to repo-relative POSIX string
    def _rel_as_posix(p: Path) -> str:
        return p.relative_to(root).as_posix()

    # Non-glob single-file path: validate containment and return single entry
    if not contains_glob:
        # Use strict resolver for deterministic rejection of absolute/traversal inputs.
        abs_path = resolve_within_root(root, pattern)
        rel = _rel_as_posix(abs_path)
        return [rel]

    # For glob patterns: disallow absolute patterns or patterns that contain '..'
    tp = Path(pattern)
    if tp.is_absolute():
        raise ValueError(f"Absolute glob patterns are not allowed: {pattern}")
    if any(part == ".." for part in tp.parts):
        raise ValueError(f"Glob patterns containing '..' are not allowed: {pattern}")

    # Expand the pattern relative to the root. Use glob.glob to support '**' with recursive=True.
    full_pattern = str(root / pattern)
    try:
        matches = glob.glob(full_pattern, recursive=True)
    except Exception as e:
        raise ValueError(f"Failed to expand glob pattern {pattern!r}: {e}")

    resolved_relatives: List[str] = []
    offending: List[str] = []
    seen = set()

    for m in matches:
        p = Path(m)
        try:
            p_resolved = p.resolve()
        except Exception:
            p_resolved = p.absolute()

        # Ensure the resolved match is within the project root
        try:
            p_resolved.relative_to(root)
        except Exception:
            offending.append(str(p_resolved))
            continue

        rel = _rel_as_posix(p_resolved)
        if rel not in seen:
            seen.add(rel)
            resolved_relatives.append(rel)

    if offending:
        # Report offending resolved matches that escape the root
        offending_list = ", ".join(offending)
        raise ValueError(f"Matched path resolves outside project root: {offending_list}")

    # Deterministic ordering
    resolved_relatives.sort()
    return resolved_relatives
