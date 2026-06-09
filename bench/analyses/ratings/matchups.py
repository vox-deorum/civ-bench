"""``ratings.matchups`` — empirical head-to-head matrices + OLS validation.

Ported from ``ratings/matchups.py``: for every pair of player types that met in a
game, report either the win rate ``P(A stronger than B)`` (``mode:"winrate"``) or
the mean strength difference ``mean(A − B)`` (``mode:"mean"``, default), with a
per-cell significance test (ANOVA / one-sample t-test). With ``validate_ols`` it
also fits ``adjusted_strength ~ C(player_type, Treatment(ref))`` and surfaces the
per-type deviation effects as a cross-check on the matrix ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError

_STRENGTH_COL = "adjusted_strength"


def create_matchup_matrix(strength_df: pd.DataFrame):
    """Empirical P(A has higher adjusted strength than B); ANOVA p-values."""
    from scipy.stats import f_oneway

    player_types = sorted(strength_df["player_type"].unique())
    n = len(player_types)
    idx = {p: i for i, p in enumerate(player_types)}
    win = np.zeros((n, n))
    count = np.zeros((n, n))
    a_vals = [[[] for _ in range(n)] for _ in range(n)]
    b_vals = [[[] for _ in range(n)] for _ in range(n)]
    for _, game in strength_df.groupby("game_id"):
        recs = list(game[["player_type", _STRENGTH_COL]].itertuples(index=False, name=None))
        for pta, sa in recs:
            for ptb, sb in recs:
                if pta == ptb:
                    continue
                i, j = idx[pta], idx[ptb]
                a_vals[i][j].append(sa)
                b_vals[i][j].append(sb)
                if sa > sb:
                    win[i, j] += 1
                count[i, j] += 1
    prob = np.full((n, n), np.nan)
    pval = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j or count[i, j] == 0:
                continue
            prob[i, j] = win[i, j] / count[i, j]
            if len(a_vals[i][j]) > 1 and len(b_vals[i][j]) > 1:
                pval[i, j] = f_oneway(a_vals[i][j], b_vals[i][j]).pvalue
    return (
        pd.DataFrame(prob, index=player_types, columns=player_types),
        pd.DataFrame(count, index=player_types, columns=player_types),
        pd.DataFrame(pval, index=player_types, columns=player_types),
    )


def create_mean_matchup_matrix(strength_df: pd.DataFrame):
    """Mean(A − B) adjusted-strength difference; one-sample t-test p-values."""
    from scipy.stats import ttest_1samp

    player_types = sorted(strength_df["player_type"].unique())
    n = len(player_types)
    idx = {p: i for i, p in enumerate(player_types)}
    diffs = [[[] for _ in range(n)] for _ in range(n)]
    count = np.zeros((n, n))
    for _, game in strength_df.groupby("game_id"):
        recs = list(game[["player_type", _STRENGTH_COL]].itertuples(index=False, name=None))
        for pta, sa in recs:
            for ptb, sb in recs:
                if pta == ptb:
                    continue
                i, j = idx[pta], idx[ptb]
                diffs[i][j].append(sa - sb)
                count[i, j] += 1
    mean = np.full((n, n), np.nan)
    pval = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            d = diffs[i][j]
            if len(d) > 1:
                mean[i, j] = float(np.mean(d))
                pval[i, j] = ttest_1samp(d, 0).pvalue
            elif len(d) == 1:
                mean[i, j] = d[0]
    return (
        pd.DataFrame(mean, index=player_types, columns=player_types),
        pd.DataFrame(count, index=player_types, columns=player_types),
        pd.DataFrame(pval, index=player_types, columns=player_types),
    )


class RatingsMatchups(Analysis):
    module = "ratings.matchups"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        mode = self.params.get("mode", "mean")
        if mode not in ("mean", "winrate"):
            raise AnalysisError(
                f"ratings.matchups '{self.stage_id}': mode must be 'mean' or 'winrate'."
            )
        table_id = next((t for t in ctx.uses_tables()
                         if t == "strength" or any(s.id == t for s in ctx.config.adjust)), "strength")
        panel = ctx.apply_filter(ctx.load_table(table_id))
        if _STRENGTH_COL not in panel.columns or panel.empty:
            raise AnalysisError(
                f"ratings.matchups '{self.stage_id}': need a non-empty strength table "
                f"with '{_STRENGTH_COL}'."
            )

        if mode == "winrate":
            matrix, count, pval = create_matchup_matrix(panel)
            center, vmin, vmax, label, cmap = 0.5, 0.0, 1.0, "P(row stronger than col)", "RdBu_r"
        else:
            matrix, count, pval = create_mean_matchup_matrix(panel)
            center, vmin, vmax, label, cmap = 0.0, None, None, "mean(row − col) strength", "RdBu_r"

        tables = {
            "matchup": matrix.reset_index(names="player_type"),
            "counts": count.reset_index(names="player_type"),
            "pvalues": pval.reset_index(names="player_type"),
        }
        figures = {"matchup": self._plot(matrix, center, vmin, vmax, label, cmap, mode)}

        if bool(self.params.get("validate_ols", False)):
            tables["ols_validation"] = self._ols_validation(panel, ctx.catalog.vanilla_label)

        summary = f"Matchup matrix ({mode}) over {len(matrix)} player types, {panel['game_id'].nunique()} games."
        return AnalysisResult(tables=tables, figures=figures, summary=summary, metadata={"mode": mode})

    def _ols_validation(self, panel: pd.DataFrame, vanilla: str) -> pd.DataFrame:
        from ...plotting.coefficients import deviation_coefficients
        from ...stats.regression import fit_regression

        formula = f'{_STRENGTH_COL} ~ C(player_type, Treatment(reference="{vanilla}"))'
        result = fit_regression(formula, panel, outcome_col=_STRENGTH_COL)
        return deviation_coefficients(result.fit, vanilla)

    def _plot(self, matrix, center, vmin, vmax, label, cmap, mode):
        import matplotlib.pyplot as plt
        import seaborn as sns

        n = len(matrix)
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 3), max(5, 0.6 * n + 2)))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, center=center,
                    vmin=vmin, vmax=vmax, square=True, linewidths=0.3, linecolor="lightgray",
                    cbar_kws={"label": label}, annot_kws={"fontsize": 7}, ax=ax)
        ax.set_title(f"Head-to-head matchups ({mode})", fontsize=12, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        return fig
