'''aidev.stages package initializer

Keep the package import lightweight. Heavy stage implementations (analyze_file, rec_apply,
apply_and_refresh, etc.) are imported lazily via attribute access to avoid import-time
side-effects and cycles. When a stage module is missing, raise a clear ImportError with
an actionable remediation hint.
'''

from __future__ import annotations

import importlib
import types
from typing import List

# Publicly documented/known submodules under aidev.stages that consumers may import.
# Keep this list small and only names of modules (not objects) to enable lazy imports.
known_submodules: List[str] = [
    "analyze_file",
    "analyze_stage_driver",
    "apply_and_refresh",
    "approval_gate",
    "consistency_checks",
    "generate_edits",
    "rec_apply",
]

__all__ = tuple(known_submodules)


def _import_stage(name: str) -> types.ModuleType:
    """Lazily import a stage module by name.

    Raises ImportError with an actionable message if the module cannot be imported.
    Chains the original ImportError to preserve traceback.
    """
    if name not in known_submodules:
        raise ImportError(
            f"Unknown stage module '{name}'. Known stages: {', '.join(known_submodules)}"
        )

    fullname = f"aidev.stages.{name}"
    try:
        module = importlib.import_module(fullname)
    except ImportError as e:
        # Fail-fast with an actionable remediation hint and preserve the original exception.
        hint = (
            "Ensure the module is present and installed (e.g. `pip install -e .`),\n"
            "verify your PYTHONPATH/project layout, or enable the optional plugin that provides it."
        )
        raise ImportError(
            f"Failed to import '{fullname}': {e}. Remediation: {hint}"
        ) from e

    return module


def __getattr__(name: str) -> types.ModuleType:
    """PEP 562 package attribute access hook for lazy importing known submodules.

    Accessing attributes like `aidev.stages.rec_apply` will import the module on demand.
    """
    if name in known_submodules:
        return _import_stage(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    # Include known submodules to improve tab-completion and tooling discoverability.
    result = list(globals().keys())
    for mod in known_submodules:
        if mod not in result:
            result.append(mod)
    return sorted(result)


if __name__ == "__main__":
    # Minimal smoke test / usage example demonstrating lazy import behavior.
    import sys

    print("aidev.stages known submodules:", ", ".join(known_submodules))

    # Attempt to lazily import a known stage. If it is missing this will fail loudly
    # with a helpful ImportError message (no silent fallback).
    try:
        print("Attempting lazy import of 'rec_apply'...")
        mod = _import_stage("rec_apply")
        print("Successfully imported:", getattr(mod, "__name__", str(mod)))
    except ImportError as exc:
        print("ERROR:", exc, file=sys.stderr)
        # Exit non-zero to signal the smoke run failed due to a missing/invalid module.
        sys.exit(1)

    print("Smoke import succeeded.")
