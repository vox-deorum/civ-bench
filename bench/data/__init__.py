"""Data layer: canonical CSV filter helpers (ported from
``shared/data_loading.py``). Pulls pandas only; catalog-driven."""

from __future__ import annotations

from .loading import (
    apply_filter_spec,
    drop_problem_games,
)

__all__ = [
    "apply_filter_spec",
    "drop_problem_games",
]
