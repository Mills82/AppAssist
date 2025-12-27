"""
runtimes/patcher.py

Utilities to generate and apply unified diffs with normalized LF-only line endings.

This module now delegates diff generation and patch application to `aidev.io_utils`
to keep a single source of truth, while preserving the public API and exceptions.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

from aidev.io_utils import (
    generate_unified_diff as _gen_udiff,
    apply_unified_patch as _apply_unified_patch,
)


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


def write_text_lf(path: str, text: str, encoding: str = "utf-8") -> None:
    """Atomically write text to path, normalizing to LF-only endings."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    _ensure_parent_dir(path)
    dirpath = os.path.dirname(path) or "."
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp.patch.", dir=dirpath)
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fh.write(normalized)
            fd = None
        os.replace(tmp_path, path)
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


def apply_unified_diff(patch_text: str, target_path: str, encoding: str = "utf-8") -> bool:
    """
    Apply a unified diff to the file at target_path by delegating to aidev.io_utils.

    Returns True if the file changed, False if the patch made no changes.
    Raises PatchApplyError if the patch cannot be applied cleanly.
    """
    if not os.path.exists(target_path):
        raise PatchApplyError(f"Target file does not exist: {target_path}")

    original = read_text_normalized(target_path, encoding=encoding)
    try:
        new_text = _apply_unified_patch(original, patch_text)
    except Exception as e:
        raise PatchApplyError(str(e)) from e

    if new_text == original:
        return False

    write_text_lf(target_path, new_text, encoding=encoding)
    return True


__all__ = [
    "generate_unified_diff",
    "read_text_normalized",
    "write_text_lf",
    "apply_unified_diff",
    "PatchError",
    "PatchApplyError",
]
