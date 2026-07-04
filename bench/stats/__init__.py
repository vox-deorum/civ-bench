"""Statistics layer: the OLS ``fit_regression`` wrapper + logit transforms
(ported from ``shared/regression_utilities.py``).

Imported by ``performance.score_ratio``, ``ratings.matchups``, and
``adjust/strength.py``. Pulls statsmodels — NOT on the dry-run path.
"""

from __future__ import annotations

from .regression import (
    RegressionResult,
    fit_regression,
)
from .transforms import LOGIT_EPS, inv_logit, logit

__all__ = [
    "LOGIT_EPS",
    "RegressionResult",
    "fit_regression",
    "inv_logit",
    "logit",
]
