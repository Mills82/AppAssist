"""
runtimes/patcher.py

Utilities to generate and apply unified diffs with normalized LF-only line endings.

This module delegates diff generation and patch application to `aidev.io_utils`
to keep a single source of truth, while preserving the public API and exceptions.

IMPORTANT: All filesystem reads/writes in this module are root-locked to an
explicit `project_root` via `runtimes.path_safety.resolve_within_root`.
This provides containment guarantees (rejecting absolute paths and traversal)
so patches cannot write outside the selected project.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from aidev.io_utils import (
    apply_unified_patch as _apply_unified_patch,
    generate_unified_diff as _gen_udiff,
)
from runtimes.path_safety import resolve_within_root


class PatchError(Exception):
    """Base class for patch-related errors."""


class PatchApplyError(PatchError):
    """Raised when a unified diff cannot be applied cleanly."""


def generate_unified_diff(from_path: str, to_path: str, original_text: str, modified_text: str) -> str:
    """Back-compat wrapper; delegates to aidev.io_utils.generate_unified_diff."""
    return _gen_udiff(from_path, to_path, original_text, modified_text)


def read_text_normalized(path: str, encoding: str = "utf-8") -> str:
    """Read a file in universal-newline mode and return a string with '\n' separators."""
    with open(path, "r", encoding=encoding, newline=None) as fh:
        return fh.read()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_text_lf(
    path: str,
    text: str,
    project_root: str | Path,
    encoding: str = "utf-8",
) -> None:
    """Atomically write text to path (relative to project_root), normalizing to LF-only endings.

    Args:
        path: Repo-relative path to write within the selected project.
        text: File contents.
        project_root: Root directory that all writes must be contained within.

    Raises:
        PatchApplyError: If path cannot be safely resolved within project_root.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # Containment guarantee lives in runtimes/path_safety.py; do not re-implement
    # path traversal/symlink-escape logic here.
    try:
        resolved = resolve_within_root(project_root, path)
    except Exception as e:
        raise PatchApplyError(str(e)) from e

    resolved_str = os.fspath(resolved)
    _ensure_parent_dir(resolved_str)

    dirpath = os.path.dirname(resolved_str) or "."
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp.patch.", dir=dirpath)
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fh.write(normalized)
            fd = None
        os.replace(tmp_path, resolved_str)
        tmp_path = None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def apply_unified_diff(
    patch_text: str,
    target_path: str,
    project_root: str | Path,
    encoding: str = "utf-8",
) -> bool:
    """Apply a unified diff to a target file under project_root.

    This resolves `target_path` with `resolve_within_root(project_root, target_path)`
    and performs all reads/writes against that resolved absolute path.

    Returns True if the file changed, False if the patch made no changes.
    Raises PatchApplyError if the patch cannot be applied cleanly or if the
    target path is not safely contained within project_root.
    """
    try:
        resolved = resolve_within_root(project_root, target_path)
    except Exception as e:
        raise PatchApplyError(str(e)) from e

    resolved_str = os.fspath(resolved)
    if not os.path.exists(resolved_str):
        raise PatchApplyError(f"Target file does not exist: {resolved_str}")

    original = read_text_normalized(resolved_str, encoding=encoding)
    try:
        new_text = _apply_unified_patch(original, patch_text)
    except Exception as e:
        raise PatchApplyError(str(e)) from e

    if new_text == original:
        return False

    # Write back to the same resolved path, still root-locked.
    write_text_lf(target_path, new_text, project_root=project_root, encoding=encoding)
    return True


__all__ = [
    "generate_unified_diff",
    "read_text_normalized",
    "write_text_lf",
    "apply_unified_diff",
    "PatchError",
    "PatchApplyError",
]
