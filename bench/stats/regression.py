"""Statistics layer: OLS regression wrapper with clustered/weighted fits.

Ported from ``shared/regression_utilities.py`` (the old paper repo). This is real
analysis code — imported by ``performance.score_ratio``, ``ratings.matchups``
(``validate_ols``), and ``adjust/strength.py``'s civ-adjustment — so it lives under
``stats/``, not ``plotting/``. The source's regression-suite / coefficient-heatmap
and logistic helpers were unused in this pipeline and have been removed; only the
``fit_regression`` wrapper and its :class:`RegressionResult` remain.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols


@dataclass
class RegressionResult:
    """Uniform wrapper around an OLS fit for downstream consumption."""

    fit: object
    formula: str
    clustered: bool
    weighted: bool = False

    @property
    def params(self) -> pd.Series:
        return self.fit.params

    @property
    def conf_int(self) -> pd.DataFrame:
        return self.fit.conf_int()

    @property
    def pvalues(self) -> pd.Series:
        return self.fit.pvalues

    @property
    def bse(self) -> pd.Series:
        """Coefficient standard errors, including robust SEs when fitted."""
        return self.fit.bse

    @property
    def nobs(self) -> int:
        return int(self.fit.nobs)

    @property
    def rsquared(self) -> float:
        return self.fit.rsquared

    @property
    def rsquared_adj(self) -> float:
        return self.fit.rsquared_adj

    def summary_line(self) -> str:
        parts = [
            f'R² = {self.rsquared:.4f}',
            f'Adj R² = {self.rsquared_adj:.4f}',
            f'n = {self.nobs}',
        ]
        tags = []
        if self.weighted:
            tags.append('weighted')
        if self.clustered:
            tags.append('cluster-robust SEs')
        if tags:
            parts.append(f'({"; ".join(tags)})')
        return ', '.join(parts)


def _resolve_group_col(
    data: pd.DataFrame,
    group_col: str | None,
    group_cols: list[str] | None,
) -> str | None:
    """Return a single group column name, creating a composite if needed."""
    if group_col is not None:
        return group_col
    if group_cols is not None:
        col = '_cluster_group'
        data[col] = data[group_cols].astype(str).agg('_'.join, axis=1)
        return col
    return None


def fit_regression(
    formula: str,
    data: pd.DataFrame,
    outcome_col: str | None = None,
    group_col: str | None = None,
    group_cols: list[str] | None = None,
    weight_col: str | None = None,
) -> RegressionResult:
    """Fit a linear regression, optionally weighted and/or cluster-robust."""
    data = data.copy()
    if outcome_col is not None:
        data = data.dropna(subset=[outcome_col])
    if weight_col is not None:
        data = data.dropna(subset=[weight_col])

    gcol = _resolve_group_col(data, group_col, group_cols)
    if weight_col is None:
        model = ols(formula, data=data)
    else:
        model = smf.wls(formula=formula, data=data, weights=data[weight_col])

    if gcol is not None:
        fit = model.fit(cov_type='cluster', cov_kwds={'groups': data[gcol]})
    else:
        fit = model.fit()

    return RegressionResult(
        fit=fit,
        formula=formula,
        clustered=gcol is not None,
        weighted=weight_col is not None,
    )
