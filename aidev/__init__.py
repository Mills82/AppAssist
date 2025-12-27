# aidev/__init__.py
from __future__ import annotations

__all__ = ["main", "__version__"]
__version__ = "0.1.0"

def main(*args, **kwargs):
    # Lazy import avoids import-time cycles
    from .core import main as _main
    return _main(*args, **kwargs)
