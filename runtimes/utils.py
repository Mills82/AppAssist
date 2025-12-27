"""Utility helpers for runtimes package.

Small, well-tested helpers that are useful across multiple runtime
modules. Keep this module intentionally tiny to avoid import cycles.
"""

from __future__ import annotations

from typing import Optional


def normalize_line_endings(text: Optional[str], to: str = "\n") -> str:
    """Normalize line endings in the provided text.

    Replaces CRLF ("\r\n") and lone CR ("\r") with the requested
    newline sequence (defaults to "\n"). If ``text`` is None, returns
    the empty string.

    This is intentionally simple and avoids touching other characters.
    """
    if text is None:
        return ""
    # First collapse CRLF to LF, then remaining CR to LF. Finally, if a
    # different newline was requested, map LF to the requested sequence.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if to != "\n":
        normalized = normalized.replace("\n", to)
    return normalized
