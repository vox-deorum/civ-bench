"""Plotting layer: shared styles + coefficient/forest helpers.

Stage 0 ports ``plot_styles.py`` (catalog-parameterized) and the coefficient /
forest-plot helpers that :mod:`bench.stats` depends on. The remaining
notebook-only helpers from the old ``plot_utilities.py`` are ported per-analysis
as later stages need them (see plans/stage0.md). Imports here pull matplotlib, so
this package is NOT imported on the dry-run path.
"""

from __future__ import annotations

from .coefficients import (
    clean_variable_name,
    deviation_coefficients,
    log_odds_to_prob_change,
    plot_forest_plot,
    prepare_coefficient_data,
    pvalue_to_stars,
)
from .styles import (
    get_all_player_styles,
    get_player_alpha,
    get_player_color,
    get_player_hatch,
    get_player_linestyle,
    get_player_marker,
    sort_player_types,
)

__all__ = [
    "clean_variable_name",
    "deviation_coefficients",
    "log_odds_to_prob_change",
    "plot_forest_plot",
    "prepare_coefficient_data",
    "pvalue_to_stars",
    "get_all_player_styles",
    "get_player_alpha",
    "get_player_color",
    "get_player_hatch",
    "get_player_linestyle",
    "get_player_marker",
    "sort_player_types",
]
