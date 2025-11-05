"""
Futures Roll Analysis Framework.

This package provides utilities to ingest minute-level futures data, aggregate it to
daily or variable-granularity bucket panels, identify front/next contracts, compute
calendar spreads, and detect roll events used in institutional roll pattern research.
"""

from importlib import metadata as _metadata


try:
    __version__ = _metadata.version("futures-roll-analysis")
except _metadata.PackageNotFoundError:  # pragma: no cover - fallback for local usage
    __version__ = "0.0.0"


__all__ = ["__version__"]
