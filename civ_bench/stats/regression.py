"""Statistics layer: OLS/logistic regression wrappers, clustered/weighted fits,
and coefficient/odds-ratio heatmaps.

Ported from ``shared/regression_utilities.py`` (the old paper repo). This is real
analysis code — imported by ``performance.score_ratio``, ``ratings.matchups``
(``validate_ols``), and ``adjust/strength.py``'s civ-adjustment — so it lives
under ``stats/``, not ``plotting/``. The only change from the source is routing
the coefficient/forest helpers through :mod:`civ_bench.plotting`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import logit as smf_logit
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from ..plotting.coefficients import (
    prepare_coefficient_data,
    plot_forest_plot,
    pvalue_to_stars,
)


MODEL_VAR_MARKER = 'replay_model_canonical'


@dataclass
class RegressionResult:
    """Uniform wrapper around an OLS fit for downstream consumption."""

    fit: object
    formula: str
    clustered: bool
    weighted: bool = False
    fixed_effect_names: list[str] | None = None

    def _is_nuisance(self, v: str) -> bool:
        """True for intercept, model-categorical, or fixed-effect dummies."""
        if v == 'Intercept' or MODEL_VAR_MARKER in v:
            return True
        if self.fixed_effect_names and _FE_GROUP_COL in v:
            return True
        return False

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

    @property
    def condition_vars(self) -> list[str]:
        return [v for v in self.params.index if not self._is_nuisance(v)]

    @property
    def model_vars(self) -> list[str]:
        return [v for v in self.params.index if MODEL_VAR_MARKER in v]

    @property
    def interaction_vars(self) -> list[str]:
        return [v for v in self.params.index
                if ':' in v and not self._is_nuisance(v)]

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
        if self.fixed_effect_names:
            tags.append(f'FE: {", ".join(self.fixed_effect_names)}')
        if tags:
            parts.append(f'({"; ".join(tags)})')
        return ', '.join(parts)


@dataclass
class RegressionHeatmapData:
    """Coefficient, uncertainty, and fit-stat frames for heatmap rendering."""

    coefficients: pd.DataFrame
    standard_errors: pd.DataFrame
    pvalues: pd.DataFrame
    r2: pd.Series
    nobs: pd.Series
    results: dict[str, RegressionResult]


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


def _default_predictor_formula(outcome: str, predictors: list[str]) -> str:
    if not predictors:
        raise ValueError('predictors must contain at least one term')
    return f'{outcome} ~ {" + ".join(predictors)}'


def build_regression_heatmap_data(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    group_cols: list[str] | None = None,
    model_col: str = 'replay_model_canonical',
    model_order: list[str] | None = None,
    formula: str | None = None,
    include_overall: bool = True,
    overall_label: str = '[Overall]',
    min_predictor_count: int | None = None,
    weight_col: str | None = None,
) -> RegressionHeatmapData:
    """Fit overall and per-model regressions for coefficient heatmaps."""
    fit_rows: list[tuple[str, pd.DataFrame]] = []

    subset = data.dropna(subset=[outcome]).copy()
    if include_overall:
        fit_rows.append((overall_label, subset))

    if model_order is None:
        model_order = sorted(subset[model_col].dropna().unique())
    for model_id in model_order:
        model_data = subset[subset[model_col] == model_id]
        if not model_data.empty:
            fit_rows.append((model_id, model_data))

    coef_rows = []
    se_rows = []
    pval_rows = []
    r2_rows = []
    n_rows = []
    labels = []
    results: dict[str, RegressionResult] = {}

    for label, fit_data in fit_rows:
        active_predictors = list(predictors)
        if min_predictor_count is not None:
            active_predictors = [
                predictor for predictor in predictors
                if predictor in fit_data.columns
                and pd.to_numeric(fit_data[predictor], errors='coerce').fillna(0).sum() >= min_predictor_count
            ]

        if min_predictor_count is not None:
            row_formula = (
                _default_predictor_formula(outcome, active_predictors)
                if active_predictors else f'{outcome} ~ 1'
            )
        else:
            row_formula = formula or _default_predictor_formula(outcome, predictors)

        result = fit_regression(
            row_formula,
            fit_data,
            outcome_col=outcome,
            group_cols=group_cols,
            weight_col=weight_col,
        )
        labels.append(label)
        results[label] = result
        coef_rows.append({v: result.params.get(v, np.nan) for v in predictors})
        se_rows.append({v: result.bse.get(v, np.nan) for v in predictors})
        pval_rows.append({v: result.pvalues.get(v, np.nan) for v in predictors})
        r2_rows.append(result.rsquared)
        n_rows.append(int(fit_data[weight_col].sum()) if weight_col is not None else result.nobs)

    return RegressionHeatmapData(
        coefficients=pd.DataFrame(coef_rows, index=labels, columns=predictors),
        standard_errors=pd.DataFrame(se_rows, index=labels, columns=predictors),
        pvalues=pd.DataFrame(pval_rows, index=labels, columns=predictors),
        r2=pd.Series(r2_rows, index=labels, name='R²'),
        nobs=pd.Series(n_rows, index=labels, name='n'),
        results=results,
    )


def _coefficient_heatmap_annotations(
    coefficients: pd.DataFrame,
    standard_errors: pd.DataFrame,
    pvalues: pd.DataFrame,
    coef_fmt: str = '{:.2f}',
    se_fmt: str = '{:.2f}',
) -> np.ndarray:
    annot = np.empty(coefficients.shape, dtype=object)
    for i in range(coefficients.shape[0]):
        for j in range(coefficients.shape[1]):
            coef = coefficients.iloc[i, j]
            se = standard_errors.iloc[i, j]
            if pd.isna(coef):
                annot[i, j] = ''
                continue
            se_text = 'NA' if pd.isna(se) else se_fmt.format(se)
            stars = pvalue_to_stars(pvalues.iloc[i, j])
            annot[i, j] = f'{coef_fmt.format(coef)}\n± {se_text}{stars}'
    return annot


def _pretty_predictor_labels(
    predictors: list[str],
    predictor_labels: dict[str, str] | None = None,
) -> list[str]:
    labels = predictor_labels or {}
    return [
        labels.get(v, v.replace('_', ' ').replace(':', ' × ').title())
        for v in predictors
    ]


def plot_regression_coefficient_heatmap(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    group_cols: list[str] | None = None,
    model_col: str = 'replay_model_canonical',
    model_order: list[str] | None = None,
    formula: str | None = None,
    include_overall: bool = True,
    title: str | None = None,
    coefficient_title: str = 'Coefficients',
    predictor_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = 'RdBu_r',
    cbar_label: str = 'OLS coefficient (β)',
    display_fig: bool = True,
    min_predictor_count: int | None = None,
    weight_col: str | None = None,
) -> tuple[object, tuple[object, object], RegressionHeatmapData]:
    """Render a coefficient heatmap plus adjacent R²/n panel."""
    heatmap_data = build_regression_heatmap_data(
        data=data,
        outcome=outcome,
        predictors=predictors,
        group_cols=group_cols,
        model_col=model_col,
        model_order=model_order,
        formula=formula,
        include_overall=include_overall,
        min_predictor_count=min_predictor_count,
        weight_col=weight_col,
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    coefficients = heatmap_data.coefficients
    finite = coefficients.to_numpy(dtype=float)
    if np.isfinite(finite).any():
        vabs = max(float(np.nanmax(np.abs(finite))), 0.01)
    else:
        vabs = 0.01
    if weight_col is not None and cbar_label == 'OLS coefficient (β)':
        cbar_label = 'WLS coefficient (β)'

    coef_annot = _coefficient_heatmap_annotations(
        coefficients,
        heatmap_data.standard_errors,
        heatmap_data.pvalues,
    )
    r2_df = heatmap_data.r2.to_frame()
    r2_annot = np.array(
        [f'{r:.4f}\nn={int(n)}' for r, n in zip(heatmap_data.r2, heatmap_data.nobs)],
        dtype=object,
    ).reshape(-1, 1)

    n_vars = max(len(predictors), 1)
    if figsize is None:
        figsize = (max(8, n_vars * 1.7 + 2), max(3.5, len(coefficients) * 0.48 + 1.8))

    fig, (ax_coef, ax_r2) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={'width_ratios': [n_vars, 1.2], 'wspace': 0.05},
        sharey=True,
    )

    sns.heatmap(
        coefficients,
        annot=coef_annot,
        fmt='',
        cmap=cmap,
        center=0,
        vmin=-vabs,
        vmax=vabs,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': cbar_label, 'shrink': 0.8},
        ax=ax_coef,
    )
    ax_coef.set_ylabel('')
    ax_coef.set_xlabel('')
    ax_coef.set_title(coefficient_title, fontsize=11)
    ax_coef.set_xticklabels(
        _pretty_predictor_labels(predictors, predictor_labels),
        rotation=30,
        ha='right',
    )

    r2_vmax = max(float(heatmap_data.r2.max()) * 1.2, 0.05)
    sns.heatmap(
        r2_df,
        annot=r2_annot,
        fmt='',
        cmap='YlOrRd',
        vmin=0,
        vmax=r2_vmax,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'R²', 'shrink': 0.8},
        ax=ax_r2,
    )
    ax_r2.set_ylabel('')
    ax_r2.set_xlabel('')
    ax_r2.set_title('R²', fontsize=11)
    ax_r2.set_xticklabels(['R²'], rotation=0, ha='center')

    if include_overall and len(coefficients) > 1:
        ax_coef.axhline(1, color='black', linewidth=1.2)
        ax_r2.axhline(1, color='black', linewidth=1.2)

    if title is None:
        title = f'Per-Model Regression Coefficients for Δ {outcome}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    fig.text(
        0.02, -0.02,
        'Cells: coefficient ± SE. Significance: * p<0.05, ** p<0.01, *** p<0.001'
        + (' (weighted)' if weight_col else '')
        + (' (cluster-robust)' if group_cols else ''),
        fontsize=9,
        style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    )
    plt.tight_layout()

    if display_fig:
        try:
            from IPython.display import display
            display(fig)
            plt.close(fig)
        except ImportError:
            plt.show()

    return fig, (ax_coef, ax_r2), heatmap_data


def _build_model_cat(model_var: str, reference_model: str) -> str:
    return f'C({model_var}, Treatment(reference="{reference_model}"))'


_FE_GROUP_COL = '_fe_group'


def _build_formulas(
    outcome: str,
    condition_factors: list[str],
    model_var: str,
    reference_model: str,
    fixed_effects: bool = False,
) -> dict[str, str]:
    """Build the three formula strings for a regression suite."""
    cat = _build_model_cat(model_var, reference_model)
    factors_str = ' + '.join(condition_factors)

    fe_str = ''
    if fixed_effects:
        fe_str = f' + C({_FE_GROUP_COL})'

    main = f'{outcome} ~ {factors_str} + {cat}{fe_str}'

    pairs = ' + '.join(
        f'{a} * {b}' for a, b in combinations(condition_factors, 2)
    )
    interactions = f'{outcome} ~ {pairs} + {cat}{fe_str}'

    model_x_cond = f'{outcome} ~ ({factors_str}) * {cat}{fe_str}'

    return {
        'main': main,
        'interactions': interactions,
        'model_x_condition': model_x_cond,
    }


def run_regression_suite(
    data: pd.DataFrame,
    outcome: str,
    condition_factors: list[str],
    model_var: str = 'replay_model_canonical',
    reference_model: str = 'GPT-OSS-120B',
    group_cols: list[str] | None = None,
    weight_col: str | None = None,
    plot: bool = True,
    plot_kwargs: dict | None = None,
) -> dict[str, RegressionResult]:
    """Run main-effects, condition-interaction, and model x condition regressions."""
    pkw = plot_kwargs or {}
    has_fe = group_cols is not None
    formulas = _build_formulas(
        outcome, condition_factors, model_var, reference_model,
        fixed_effects=has_fe,
    )

    subset = data.dropna(subset=[outcome]).copy()
    if group_cols is not None:
        subset[_FE_GROUP_COL] = (
            subset[group_cols].astype(str).agg('_'.join, axis=1)
        )
    fe_names = group_cols or None
    results: dict[str, RegressionResult] = {}
    for key, formula in formulas.items():
        print(f'[{key}] {formula}')
        r = fit_regression(
            formula, subset,
            group_col=None, group_cols=group_cols, weight_col=weight_col,
        )
        r.fixed_effect_names = fe_names
        results[key] = r

    baseline_desc = (
        f'[baseline: {reference_model}, '
        + ', '.join(f'no {f}' for f in condition_factors)
        + ']'
    )

    # --- Main effects ---
    r_main = results['main']
    if plot:
        cond_df = prepare_coefficient_data(
            r_main.params, r_main.conf_int, r_main.pvalues,
            r_main.condition_vars, var_type='condition',
        )
        model_df = prepare_coefficient_data(
            r_main.params, r_main.conf_int, r_main.pvalues,
            r_main.model_vars, var_type='condition',
        )
        coef_df = pd.concat(
            [cond_df.sort_values('Name'), model_df.sort_values('Name')],
            ignore_index=True,
        )
        plot_forest_plot(
            coef_df,
            title=f'Main Effects: Factor Contributions to Δ {outcome}\n{baseline_desc}',
            xlabel=f'Effect on {outcome} (OLS coefficient)',
            use_prob_scale=False,
            sort_alphabetically=True,
            **pkw,
        )
    print(f'=== Main Effects ({outcome}) ===')
    print(r_main.fit.summary().tables[0])
    print(r_main.summary_line())

    # --- Condition interactions ---
    r_inter = results['interactions']
    if plot:
        inter_df = prepare_coefficient_data(
            r_inter.params, r_inter.conf_int, r_inter.pvalues,
            r_inter.condition_vars, var_type='condition',
        )
        model_df_i = prepare_coefficient_data(
            r_inter.params, r_inter.conf_int, r_inter.pvalues,
            r_inter.model_vars, var_type='condition',
        )
        coef_df_i = pd.concat(
            [inter_df.sort_values('Name'), model_df_i.sort_values('Name')],
            ignore_index=True,
        )
        plot_forest_plot(
            coef_df_i,
            title=f'With Interactions: Factor Contributions to Δ {outcome}\n{baseline_desc}',
            xlabel=f'Effect on {outcome} (OLS coefficient)',
            use_prob_scale=False,
            sort_alphabetically=True,
            **pkw,
        )
    print(f'\n=== Condition Interactions ({outcome}) ===')
    print(r_inter.fit.summary().tables[0])
    print(r_inter.summary_line())

    # --- Model x condition interactions ---
    r_mx = results['model_x_condition']
    if plot:
        ix_terms = r_mx.interaction_vars
        if ix_terms:
            ix_df = prepare_coefficient_data(
                r_mx.params, r_mx.conf_int, r_mx.pvalues,
                ix_terms, var_type='condition',
            )
            plot_forest_plot(
                ix_df,
                title=(
                    f'Model × Condition Interaction Terms\n'
                    f'(deviation from {reference_model}\'s condition response)'
                ),
                xlabel=f'Interaction effect on {outcome}',
                use_prob_scale=False,
                sort_alphabetically=True,
                **pkw,
            )
    print(f'\n=== Model x Condition ({outcome}) ===')
    print(r_mx.fit.summary().tables[0])
    print(r_mx.summary_line())

    # --- Nested model comparisons (F-tests) ---
    print(f'\n=== F-tests ({outcome}) ===')
    print('Main vs Condition Interactions:')
    print(anova_lm(r_main.fit, r_inter.fit))
    print('Main vs Model x Condition:')
    print(anova_lm(r_main.fit, r_mx.fit))
    print('Condition Interactions vs Model x Condition:')
    print(anova_lm(r_inter.fit, r_mx.fit))

    return results


# =====================================================
# LOGISTIC REGRESSION UTILITIES
# =====================================================


@dataclass
class LogisticHeatmapData:
    """Log-odds, odds ratio, uncertainty, and fit-stat frames for logit heatmaps."""

    log_odds: pd.DataFrame
    odds_ratios: pd.DataFrame
    standard_errors: pd.DataFrame
    pvalues: pd.DataFrame
    pseudo_r2: pd.Series
    nobs: pd.Series
    baseline_prevalence: pd.Series


def fit_logistic_regression(
    formula: str,
    data: pd.DataFrame,
    outcome_col: str | None = None,
    group_col: str | None = None,
    group_cols: list[str] | None = None,
    weight_col: str | None = None,
) -> object | None:
    """Fit a logistic regression, optionally with weights and cluster-robust SEs."""
    data = data.copy()
    if outcome_col is not None:
        data = data.dropna(subset=[outcome_col])
    if weight_col is not None:
        data = data.dropna(subset=[weight_col])

    gcol = _resolve_group_col(data, group_col, group_cols)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            if weight_col is not None:
                model = smf.glm(
                    formula=formula,
                    data=data,
                    family=sm.families.Binomial(),
                    freq_weights=data[weight_col],
                )
                fit_kwargs = {}
            else:
                model = smf_logit(formula, data=data)
                fit_kwargs = {'method': 'bfgs', 'disp': 0, 'maxiter': 300}

            if gcol is not None:
                fit = model.fit(
                    cov_type='cluster',
                    cov_kwds={'groups': data[gcol]},
                    **fit_kwargs,
                )
            else:
                fit = model.fit(**fit_kwargs)
            return fit
        except Exception:
            return None


def build_logistic_heatmap_data(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    group_cols: list[str] | None = None,
    model_col: str = 'replay_model_canonical',
    model_order: list[str] | None = None,
    formula: str | None = None,
    include_overall: bool = True,
    overall_label: str = '[Overall]',
    baseline_condition: str = 'original',
    weight_col: str | None = None,
) -> LogisticHeatmapData:
    """Fit overall and per-model logistic regressions for OR heatmaps."""
    if formula is None:
        formula = _default_predictor_formula(outcome, predictors)

    fit_rows: list[tuple[str, pd.DataFrame]] = []
    subset = data.dropna(subset=[outcome]).copy()

    if include_overall:
        fit_rows.append((overall_label, subset))

    if model_order is None:
        model_order = sorted(subset[model_col].dropna().unique())
    for model_id in model_order:
        model_data = subset[subset[model_col] == model_id]
        if not model_data.empty:
            fit_rows.append((model_id, model_data))

    lo_rows: list[dict] = []
    se_rows: list[dict] = []
    pval_rows: list[dict] = []
    pr2_rows: list[float] = []
    n_rows: list[int] = []
    prev_rows: list[float] = []
    labels: list[str] = []

    nan_row = {v: np.nan for v in predictors}

    for label, fit_data in fit_rows:
        labels.append(label)
        outcome_vals = fit_data[outcome].dropna()
        if outcome_vals.nunique() < 2:
            fit = None
        else:
            fit = fit_logistic_regression(
                formula,
                fit_data,
                outcome_col=outcome,
                group_cols=group_cols,
                weight_col=weight_col,
            )
        if fit is None:
            lo_rows.append(nan_row)
            se_rows.append(nan_row)
            pval_rows.append(nan_row)
            pr2_rows.append(np.nan)
        else:
            lo_rows.append({v: fit.params.get(v, np.nan) for v in predictors})
            se_rows.append({v: fit.bse.get(v, np.nan) for v in predictors})
            pval_rows.append({v: fit.pvalues.get(v, np.nan) for v in predictors})
            pr2_rows.append(getattr(fit, 'prsquared', np.nan))
        n_rows.append(int(fit_data[weight_col].sum()) if weight_col is not None else len(fit_data))

        baseline = fit_data[fit_data['condition'] == baseline_condition]
        if len(baseline) == 0:
            prev_rows.append(np.nan)
        elif weight_col is None:
            prev_rows.append(baseline[outcome].mean())
        else:
            denominator = baseline[weight_col].sum()
            prev_rows.append(
                baseline[outcome].mul(baseline[weight_col]).sum() / denominator
                if denominator else np.nan
            )

    log_odds = pd.DataFrame(lo_rows, index=labels, columns=predictors)

    return LogisticHeatmapData(
        log_odds=log_odds,
        odds_ratios=np.exp(log_odds),
        standard_errors=pd.DataFrame(se_rows, index=labels, columns=predictors),
        pvalues=pd.DataFrame(pval_rows, index=labels, columns=predictors),
        pseudo_r2=pd.Series(pr2_rows, index=labels, name='pseudo_R²'),
        nobs=pd.Series(n_rows, index=labels, name='n'),
        baseline_prevalence=pd.Series(prev_rows, index=labels, name='P(tag)'),
    )


def _odds_ratio_annotations(
    odds_ratios: pd.DataFrame,
    pvalues: pd.DataFrame,
    or_fmt: str = '{:.2f}',
) -> np.ndarray:
    """Build annotation array showing OR with significance stars."""
    annot = np.empty(odds_ratios.shape, dtype=object)
    for i in range(odds_ratios.shape[0]):
        for j in range(odds_ratios.shape[1]):
            or_val = odds_ratios.iloc[i, j]
            if pd.isna(or_val):
                annot[i, j] = ''
                continue
            stars = pvalue_to_stars(pvalues.iloc[i, j])
            annot[i, j] = f'{or_fmt.format(or_val)}{stars}'
    return annot


def plot_logistic_odds_ratio_heatmap(
    data: pd.DataFrame,
    outcome: str,
    outcome_label: str,
    predictors: list[str],
    main_predictors: list[str],
    interaction_predictors: list[str],
    group_cols: list[str] | None = None,
    model_col: str = 'replay_model_canonical',
    model_order: list[str] | None = None,
    include_overall: bool = True,
    title: str | None = None,
    predictor_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = 'RdBu_r',
    vabs_cap: float | None = None,
    display_fig: bool = True,
    baseline_condition: str = 'original',
    weight_col: str | None = None,
) -> tuple[object, tuple[object, object, object], LogisticHeatmapData]:
    """Render a logistic regression odds-ratio heatmap."""
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.pyplot as plt
    import seaborn as sns

    heatmap_data = build_logistic_heatmap_data(
        data=data,
        outcome=outcome,
        predictors=predictors,
        group_cols=group_cols,
        model_col=model_col,
        model_order=model_order,
        include_overall=include_overall,
        baseline_condition=baseline_condition,
        weight_col=weight_col,
    )

    all_preds = main_predictors + interaction_predictors
    lo_all = heatmap_data.log_odds[all_preds]
    or_all = heatmap_data.odds_ratios[all_preds]
    pv_all = heatmap_data.pvalues[all_preds]

    finite = lo_all.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) > 0:
        if vabs_cap is not None:
            vabs = vabs_cap
        else:
            vabs = max(float(np.percentile(np.abs(finite), 95)), 0.1)
    else:
        vabs = 0.1
    or_norm = TwoSlopeNorm(
        vmin=float(np.exp(-vabs)),
        vcenter=1.0,
        vmax=float(np.exp(vabs)),
    )

    or_annot = _odds_ratio_annotations(or_all, pv_all)

    base_df = heatmap_data.baseline_prevalence.to_frame()
    base_annot = np.array([
        f'{p * 100:.1f}%\nn={int(n)}'
        if not pd.isna(p) else ''
        for p, n in zip(
            heatmap_data.baseline_prevalence,
            heatmap_data.nobs,
        )
    ], dtype=object).reshape(-1, 1)

    r2_df = heatmap_data.pseudo_r2.to_frame()
    r2_annot = np.array([
        f'{r:.4f}' if not pd.isna(r) else '—'
        for r in heatmap_data.pseudo_r2
    ], dtype=object).reshape(-1, 1)

    n_preds = len(all_preds)
    if figsize is None:
        figsize = (max(8, n_preds * 1.7 + 5),
                   max(3.5, len(lo_all) * 0.48 + 1.8))

    fig, (ax_base, ax_r2, ax_coef) = plt.subplots(
        1, 3, figsize=figsize,
        gridspec_kw={
            'width_ratios': [1.2, 0.8, n_preds],
            'wspace': 0.05,
        },
        sharey=True,
    )

    plabels = predictor_labels or {}

    bp = heatmap_data.baseline_prevalence
    base_vmax = max(float(bp.max()) * 1.2, 0.05) if not bp.isna().all() else 0.05
    sns.heatmap(
        base_df,
        annot=base_annot,
        fmt='',
        cmap='YlOrRd',
        vmin=0,
        vmax=base_vmax,
        linewidths=0.5,
        linecolor='gray',
        cbar=False,
        ax=ax_base,
    )
    ax_base.set_ylabel('')
    ax_base.set_xlabel('')
    ax_base.set_title('Baseline', fontsize=11)
    ax_base.set_xticklabels([f'P(tag | {baseline_condition})'], rotation=30, ha='right')

    r2_vals = heatmap_data.pseudo_r2.values.astype(float)
    r2_vmax = max(float(np.nanmax(r2_vals)) * 1.2, 0.01) if np.any(np.isfinite(r2_vals)) else 0.01
    sns.heatmap(
        r2_df,
        annot=r2_annot,
        fmt='',
        cmap='Greens',
        vmin=0,
        vmax=r2_vmax,
        linewidths=0.5,
        linecolor='gray',
        cbar=False,
        ax=ax_r2,
    )
    ax_r2.set_ylabel('')
    ax_r2.set_xlabel('')
    ax_r2.set_title('Fit', fontsize=11)
    ax_r2.set_xticklabels(['R²'], rotation=30, ha='right')

    sns.heatmap(
        or_all,
        annot=or_annot,
        fmt='',
        cmap=cmap,
        norm=or_norm,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Odds ratio', 'shrink': 0.8},
        ax=ax_coef,
    )
    ax_coef.set_ylabel('')
    ax_coef.set_xlabel('')
    ax_coef.set_title('Odds Ratios', fontsize=11)

    col_labels = []
    for v in all_preds:
        label = plabels.get(v, v.replace('_', ' ').replace(':', ' × ').title())
        col_labels.append(label)
    ax_coef.set_xticklabels(col_labels, rotation=30, ha='right')

    n_main = len(main_predictors)
    ax_coef.axvline(n_main, color='black', linewidth=1.2)

    if include_overall and len(lo_all) > 1:
        for ax in (ax_base, ax_r2, ax_coef):
            ax.axhline(1, color='black', linewidth=1.2)

    if title is None:
        title = f'{outcome_label} — Condition Effects on Tag Presence'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    fig.text(
        0.02, -0.02,
        'Cells: odds ratio. '
        '* p<0.05, ** p<0.01, *** p<0.001'
        + (' (cluster-robust)' if group_cols else ''),
        fontsize=9,
        style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    )
    plt.tight_layout()

    if display_fig:
        try:
            from IPython.display import display
            display(fig)
            plt.close(fig)
        except ImportError:
            plt.show()

    return fig, (ax_base, ax_r2, ax_coef), heatmap_data
