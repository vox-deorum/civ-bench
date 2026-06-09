"""Adjust layer — derived-table producers (stage 3).

An ``adjust`` stage turns an estimator's per-turn ``predicted_win_probability``
into a per-player-game **strength panel** (``adjusted_strength``) and registers it
as a named table that ``ratings.*`` (and some ``performance.*``) analyses consume
via ``uses.tables``. Today there is one module, ``strength``.

Pulls statsmodels (the civ OLS) on import of :mod:`strength`, so import lazily
from the CLI run path — never from the import-light config/pipeline layers.
"""

from __future__ import annotations

from .errors import AdjustError
from .registry import ADJUST_REGISTRY, get_adjuster, list_adjusters
from .runner import AdjustResult, run_adjust
from .strength import StrengthArtifacts, build_strength_panel

__all__ = [
    "ADJUST_REGISTRY",
    "AdjustError",
    "AdjustResult",
    "StrengthArtifacts",
    "build_strength_panel",
    "get_adjuster",
    "list_adjusters",
    "run_adjust",
]
