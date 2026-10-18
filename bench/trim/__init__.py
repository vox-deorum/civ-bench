"""``civ-bench trim``: remove extra games from repeated controlled slots.

Public surface: the :func:`run_trim` orchestrator and its result types.
"""

from __future__ import annotations

from .errors import TrimError
from .runner import TrimOutcome, TrimResult, run_trim

__all__ = [
    "TrimError",
    "TrimOutcome",
    "TrimResult",
    "run_trim",
]
