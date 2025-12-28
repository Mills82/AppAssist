# runtimes/__init__.py
"""
runtimes package public surface.

This package provides cross-platform helpers to run external commands in a
consistent, testable way, and lightweight runtime "tools" for Node/Python/Flutter/etc.

Canonical command runner lives in runtimes.runner to avoid duplicate-runner drift.
"""

from __future__ import annotations

from .runner import run_command, UnresolvedCommandError

__all__ = [
    "run_command",
    "UnresolvedCommandError",
]
