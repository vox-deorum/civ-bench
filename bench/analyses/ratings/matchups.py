"""``ratings.matchups`` — empirical head-to-head matrices + OLS validation.

Ported from ``ratings/matchups.py``: for every pair of player types that met in a
game, report the win rate ``P(A stronger than B)`` with ties split 0.5/0.5
(``mode:"winrate"``), the mean strength difference ``mean(A - B)`` (``mode:"mean"``),
or both (``mode:"both"``, default), with per-cell **paired** t-tests on the aligned
within-game strength pairs (``ttest_rel``). With
``validate_ols`` it also fits ``adjusted_strength ~ C(player_type, Treatment(ref))``
and surfaces the per-type deviation effects as a cross-check on the matrix
ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError

_STRENGTH_COL = "adjusted_strength"


def create_matchup_matrix(strength_df: pd.DataFrame):
    """Empirical P(A has higher adjusted strength than B), ties counted as 0.5.

    p-values are a **paired** t-test on the within-game (A, B) strength pairs
    (``ttest_rel``, equivalent to ``ttest_1samp(A - B, 0)`` used by the mean matrix)
    — the samples are aligned per game, not two independent groups, so ``f_oneway``
    was the wrong test. Counting exact ties as 0.5 (rather than a loss for both)
    restores ``P(i,j) + P(j,i) = 1``; the strength stage's ``enforce_winner`` can
    manufacture exact 1.0 ties, so ties are not merely a rounding artefact.
    """
    from scipy.stats import ttest_rel

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
                    win[i, j] += 1.0
                elif sa == sb:
                    win[i, j] += 0.5  # split ties → P(i,j)+P(j,i)=1
                count[i, j] += 1
    prob = np.full((n, n), np.nan)
    pval = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j or count[i, j] == 0:
                continue
            prob[i, j] = win[i, j] / count[i, j]
            if len(a_vals[i][j]) > 1 and len(b_vals[i][j]) > 1:
                pval[i, j] = ttest_rel(a_vals[i][j], b_vals[i][j]).pvalue
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
    friendly_name = "Adjusted-strength matchups"
    description = (
        "Compares every pair of player types using model-adjusted strength, "
        "including mean differences and win rates."
    )
    report_defaults = {
        "tables": [],
        "figures": ["matchup", "strength_mean", "strength_winrate"],
    }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        mode = self.params.get("mode", "both")
        if mode not in ("mean", "winrate", "both"):
            raise AnalysisError(
                f"ratings.matchups '{self.stage_id}': mode must be 'mean', 'winrate', or 'both'."
            )
        table_id = ctx.strength_table_id()
        panel = ctx.apply_filter(ctx.load_table(table_id))
        if _STRENGTH_COL not in panel.columns or panel.empty:
            raise AnalysisError(
                f"ratings.matchups '{self.stage_id}': need a non-empty strength table "
                f"with '{_STRENGTH_COL}'."
            )

        metadata = {"mode": mode, **ctx.strength_provenance(table_id, panel)}
        tables = {}
        figures = {}
        count_for_summary = None
        mean = mean_pval = winrate = winrate_pval = None
        display = ctx.matchup_display()
        reference = ctx.catalog.vanilla_label
        reference_available = reference in set(panel["player_type"].astype(str))
        use_vs_reference = display == "vs_reference" and reference_available

        if mode in ("mean", "both"):
            mean, count, pval = create_mean_matchup_matrix(panel)
            mean_pval = pval
            count_for_summary = count
            table_name = "matchup" if mode == "mean" else "strength_mean"
            pval_name = "pvalues" if mode == "mean" else "pvalues_mean"
            figure_name = "matchup" if mode == "mean" else "strength_mean"
            tables[table_name] = mean.reset_index(names="player_type")
            tables[pval_name] = pval.reset_index(names="player_type")
            if not use_vs_reference:
                figures[figure_name] = self._plot(
                    mean, 0.0, None, None, "mean(row - col) adjusted strength",
                    "RdBu_r", "strength mean", metadata,
                )

        if mode in ("winrate", "both"):
            winrate, count, pval = create_matchup_matrix(panel)
            winrate_pval = pval
            count_for_summary = count if count_for_summary is None else count_for_summary
            table_name = "matchup" if mode == "winrate" else "strength_winrate"
            pval_name = "pvalues" if mode == "winrate" else "pvalues_winrate"
            figure_name = "matchup" if mode == "winrate" else "strength_winrate"
            tables[table_name] = winrate.reset_index(names="player_type")
            tables[pval_name] = pval.reset_index(names="player_type")
            if not use_vs_reference:
                figures[figure_name] = self._plot(
                    winrate, 0.5, 0.0, 1.0, "P(row stronger than col)",
                    "RdBu_r", "strength win rate", metadata, percent=True, counts=count,
                )

        if count_for_summary is not None:
            tables["counts"] = count_for_summary.reset_index(names="player_type")

        if bool(self.params.get("validate_ols", False)):
            tables["ols_validation"] = self._ols_validation(panel, ctx.catalog.vanilla_label)

        if use_vs_reference:
            vs = self._vs_reference_table(
                ctx, reference, count_for_summary, mean, mean_pval, winrate, winrate_pval
            )
            tables["vs_reference"] = vs
            plot_vs = vs.copy()
            plot_vs["n_label"] = plot_vs["n"].map(
                lambda n: f"n={int(n)}" if pd.notna(n) else ""
            )
            pairing = ctx.condition_pairing()
            from ...plotting.pairing import PairingSpec, plot_paired_rows

            plot_spec = pairing or PairingSpec((), "base", "Identity")
            note = self._provenance_text(metadata)
            if mode in ("mean", "both"):
                slug = "matchup" if mode == "mean" else "strength_mean"
                figures[slug] = plot_paired_rows(
                    plot_vs,
                    catalog=ctx.catalog,
                    spec=plot_spec,
                    value_col="mean_diff_vs_ref",
                    identity_col="player_type",
                    ref_line=0,
                    annotate_col="n_label",
                    ascending=False,
                    xlabel=f"Mean adjusted-strength difference vs {reference}",
                    title=f"Adjusted strength vs {reference}",
                    provenance_note=note,
                )
            if mode in ("winrate", "both"):
                slug = "matchup" if mode == "winrate" else "strength_winrate"
                figures[slug] = plot_paired_rows(
                    plot_vs,
                    catalog=ctx.catalog,
                    spec=plot_spec,
                    value_col="winrate_vs_ref",
                    identity_col="player_type",
                    ref_line=0.5,
                    annotate_col="n_label",
                    ascending=False,
                    xlabel=f"P(row stronger than {reference})",
                    title=f"Adjusted-strength win rate vs {reference}",
                    provenance_note=note,
                )

        summary = (
            f"Model-adjusted strength compares {panel['player_type'].nunique()} player "
            f"types across {panel['game_id'].nunique()} games in a "
            f"{'reference view' if use_vs_reference else 'pairwise matrix'} ({mode})"
        )
        if display == "vs_reference" and not reference_available:
            summary += f"; reference '{reference}' is absent, so the report rendered matrix figures instead"
        summary += "."
        return AnalysisResult(tables=tables, figures=figures, summary=summary, metadata=metadata)

    @staticmethod
    def _vs_reference_table(ctx, reference, counts, mean, mean_pval, winrate, winrate_pval):
        from ...plotting.pairing import PairingSpec, attach_pair_columns

        identities = sorted(
            set().union(
                *(set(matrix.index.astype(str)) for matrix in (mean, winrate) if matrix is not None)
            ) - {reference}
        )
        rows = []
        for identity in identities:
            row = {"player_type": identity}
            if mean is not None:
                row["mean_diff_vs_ref"] = mean.loc[identity, reference]
                row["p_value_mean"] = mean_pval.loc[identity, reference]
            if winrate is not None:
                row["winrate_vs_ref"] = winrate.loc[identity, reference]
                row["p_value_winrate"] = winrate_pval.loc[identity, reference]
            n = counts.loc[identity, reference] if counts is not None else np.nan
            row["n"] = n
            rows.append(row)
        out = pd.DataFrame(rows)
        pairing = ctx.condition_pairing()
        spec = pairing or PairingSpec((), "base", "Identity")
        return attach_pair_columns(out, ctx.catalog, spec, "player_type").drop(
            columns="is_baseline"
        )

    def _ols_validation(self, panel: pd.DataFrame, vanilla: str) -> pd.DataFrame:
        from ...plotting.coefficients import deviation_coefficients
        from ...stats.regression import fit_regression

        formula = f'{_STRENGTH_COL} ~ C(player_type, Treatment(reference="{vanilla}"))'
        result = fit_regression(formula, panel, outcome_col=_STRENGTH_COL)
        return deviation_coefficients(result.fit, vanilla)

    def _plot(self, matrix, center, vmin, vmax, label, cmap, mode, metadata, percent=False, counts=None):
        import matplotlib.pyplot as plt
        import seaborn as sns

        n = len(matrix)
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 3), max(5, 0.6 * n + 2)))
        annot = True
        fmt = ".2f"
        if percent:
            annot = matrix.copy().astype(object)
            for row in matrix.index:
                for col in matrix.columns:
                    value = matrix.loc[row, col]
                    if pd.isna(value):
                        annot.loc[row, col] = ""
                        continue
                    n_txt = ""
                    if counts is not None and pd.notna(counts.loc[row, col]):
                        n_txt = f"\nn={int(counts.loc[row, col])}"
                    annot.loc[row, col] = f"{100 * value:.0f}%{n_txt}"
            fmt = ""
        sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap, center=center,
                    vmin=vmin, vmax=vmax, square=True, linewidths=0.3, linecolor="lightgray",
                    cbar_kws={"label": label}, annot_kws={"fontsize": 7}, ax=ax)
        ax.set_title(f"Head-to-head matchups ({mode})", fontsize=12, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        self._add_provenance_note(fig, metadata)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig

    @staticmethod
    def _provenance_text(metadata: dict) -> str:
        est = metadata.get("strength_estimator")
        model = metadata.get("estimator_model")
        block = metadata.get("adjust_block")
        bits = []
        if est:
            bits.append(f"strength estimator: {est}" + (f" ({model})" if model else ""))
        if block:
            bits.append(f"adjust block: {block}")
        return "; ".join(bits)

    @classmethod
    def _add_provenance_note(cls, fig, metadata: dict) -> None:
        note = cls._provenance_text(metadata)
        if note:
            fig.text(0.01, 0.01, note, ha="left", va="bottom",
                     fontsize=8, color="#666666")
