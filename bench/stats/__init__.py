"""Statistics layer: OLS/logistic regression wrappers, clustered/weighted fits,
coefficient/odds-ratio heatmaps (ported from ``shared/regression_utilities.py``).

Imported by ``performance.score_ratio``, ``ratings.matchups``, and
``adjust/strength.py``. Pulls statsmodels/matplotlib — NOT on the dry-run path.
"""

from __future__ import annotations

from .regression import (
    LogisticHeatmapData,
    RegressionHeatmapData,
    RegressionResult,
    build_logistic_heatmap_data,
    build_regression_heatmap_data,
    fit_logistic_regression,
    fit_regression,
    plot_logistic_odds_ratio_heatmap,
    plot_regression_coefficient_heatmap,
    run_regression_suite,
)

__all__ = [
    "LogisticHeatmapData",
    "RegressionHeatmapData",
    "RegressionResult",
    "build_logistic_heatmap_data",
    "build_regression_heatmap_data",
    "fit_logistic_regression",
    "fit_regression",
    "plot_logistic_odds_ratio_heatmap",
    "plot_regression_coefficient_heatmap",
    "run_regression_suite",
]
