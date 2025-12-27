"""aidev.orchestration package public surface.

This module intentionally performs a guarded import of DeepResearchEngine so that
importing the package does not raise at import time while the implementation is
being added or iterated on.
"""

# Guarded import: keep package importable even if deep_research_engine.py is
# missing or partially implemented. Use a broad except to avoid import-time
# crashes from transient issues; the real implementation should replace this.
try:
    from .deep_research_engine import DeepResearchEngine  # type: ignore
except Exception:
    DeepResearchEngine = None  # type: ignore

__all__ = ["DeepResearchEngine"]
