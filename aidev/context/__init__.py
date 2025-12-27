# aidev/context/__init__.py
from __future__ import annotations

"""
Context utilities for AI Dev Bot.

This package currently exposes:

- Project brief helpers (aidev.context.brief)
    * get_or_build(...)        -> {"text": ..., "hash": ..., "path": ...}
    * canonicalize_brief(...)
    * compute_brief_hash(...)

- Edit context helpers (aidev.context.edit_context)
    * build_context_for_edit(...)
    * build_context_bundle(...)  (alias, if provided by edit_context)

The intent is that higher-level stages (e.g. generate_edits) import from
this package rather than reaching into submodules directly, keeping the
surface area small and stable:

    from aidev.context import get_or_build, build_context_for_edit
"""

# Re-export brief helpers
from .brief import (  # noqa: F401
    get_or_build,
    canonicalize_brief,
    compute_brief_hash,
)

# Re-export edit-context helpers if present
try:
    from .edit_context import (  # type: ignore  # noqa: F401
        build_context_for_edit,
        build_context_bundle,
        ContextFile,
    )
except Exception:  # pragma: no cover - optional surface
    # In case edit_context is not present or partially defined (older installs),
    # we avoid hard import failures here to keep the package importable.
    build_context_for_edit = None  # type: ignore[assignment]
    build_context_bundle = None  # type: ignore[assignment]

    class ContextFile:  # type: ignore[no-redef]
        """
        Lightweight placeholder so type checkers / callers don't explode if
        edit_context isn't available. At runtime, callers should treat this as
        "no edit context" if build_context_for_edit is None.
        """

        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
            raise RuntimeError(
                "aidev.context.ContextFile is unavailable because "
                "aidev.context.edit_context could not be imported."
            )


__all__ = [
    # brief helpers
    "get_or_build",
    "canonicalize_brief",
    "compute_brief_hash",
    # edit-context helpers
    "build_context_for_edit",
    "build_context_bundle",
    "ContextFile",
]
