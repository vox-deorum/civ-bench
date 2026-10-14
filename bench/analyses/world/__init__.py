"""World outcome analyses (``world.*``, benchmark.md §6.2)."""

from __future__ import annotations

from .conflict import WorldConflict
from .progress import WorldProgress

__all__ = [
    "WorldConflict",
    "WorldProgress",
]
