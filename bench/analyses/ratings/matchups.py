"""``ratings.matchups``: empirical head-to-head matrices + OLS validation.

Ported from ``ratings/matchups.py``: for every pair of player types that met in a
game, report the win rate ``P(A stronger than B)`` with ties split 0.5/0.5
(``mode:"winrate"``), the mean strength difference ``mean(A - B)`` (``mode:"mean"``),
or both (``mode:"both"``, default), with per-cell **paired** t-tests on the aligned
within-game strength pairs (``ttest_rel``). With
``validate_ols`` it also fits ``adjusted_strength ~ C(player_type, Treatment(ref))``
and surfaces the per-type deviation effects as a cross-check on the matrix
ordering.

With ``presentation.matchup_display: "matrix"`` each matrix shows as an HTML
heatmap table (``<name>_heatmap``); with ``"vs_reference"`` each shows as an
interactive forest plot against Vanilla. With ``mode:"both"`` the two metrics sit
behind the ``strength_winrate`` and ``strength_mean`` view tabs.
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
    (``ttest_rel``, equivalent to ``ttest_1samp(A - B, 0)`` used by the mean
    matrix). The samples are aligned per game, not two independent groups, so
    ``f_oneway`` was the wrong test. Counting exact ties as 0.5 (rather than a
    loss for both) restores ``P(i,j) + P(j,i) = 1``; the strength stage's
    ``enforce_winner`` can manufacture exact 1.0 ties, so ties are not merely a
    rounding artefact.
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
        "tables": ["matchup_heatmap", "strength_winrate_heatmap", "strength_mean_heatmap"],
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
        heatmaps = {}
        slugs = {}
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
            slug = slugs["mean"] = "matchup" if mode == "mean" else "strength_mean"
            tables[slug] = mean.reset_index(names="player_type")
            pval_name = "pvalues" if mode == "mean" else "pvalues_mean"
            tables[pval_name] = pval.reset_index(names="player_type")
            if not use_vs_reference:
                table, spec = self._mean_heatmap(ctx, mean, count, pval)
                tables[f"{slug}_heatmap"], heatmaps[f"{slug}_heatmap"] = table, spec

        if mode in ("winrate", "both"):
            winrate, count, pval = create_matchup_matrix(panel)
            winrate_pval = pval
            count_for_summary = count if count_for_summary is None else count_for_summary
            slug = slugs["winrate"] = "matchup" if mode == "winrate" else "strength_winrate"
            tables[slug] = winrate.reset_index(names="player_type")
            pval_name = "pvalues" if mode == "winrate" else "pvalues_winrate"
            tables[pval_name] = pval.reset_index(names="player_type")
            if not use_vs_reference:
                table, spec = self._winrate_heatmap(ctx, winrate, count, pval)
                tables[f"{slug}_heatmap"], heatmaps[f"{slug}_heatmap"] = table, spec

        if count_for_summary is not None:
            tables["counts"] = count_for_summary.reset_index(names="player_type")

        if bool(self.params.get("validate_ols", False)):
            tables["ols_validation"] = self._ols_validation(panel, ctx.catalog.vanilla_label)

        if use_vs_reference:
            vs = self._vs_reference_table(
                ctx, reference, count_for_summary, mean, mean_pval, winrate, winrate_pval
            )
            tables["vs_reference"] = vs
            figures.update(self._vs_reference_plots(ctx, vs, reference, slugs, metadata))

        if heatmaps:
            metadata["heatmaps"] = heatmaps
        if mode == "both":
            metadata["views"] = {
                "strength_winrate": {
                    "label": "Win rate",
                    "tip": "How often the row player type had higher adjusted strength.",
                    "tables": ["strength_winrate_heatmap"],
                    "figures": ["strength_winrate"],
                },
                "strength_mean": {
                    "label": "Mean difference",
                    "tip": "The average adjusted-strength gap, row minus column.",
                    "tables": ["strength_mean_heatmap"],
                    "figures": ["strength_mean"],
                },
            }
        metadata["display"] = "vs_reference" if use_vs_reference else "matrix"
        if display == "vs_reference" and not reference_available:
            metadata["warning"] = f"reference '{reference}' is absent"
        matrix = winrate if winrate is not None else mean
        values = matrix.where(np.isfinite(matrix)).copy()
        np.fill_diagonal(values.values, np.nan)
        if use_vs_reference:
            values = values[[reference]]
        pairs = values.stack().sort_values(ascending=False, kind="stable")
        if pairs.empty:
            summary = "No shared games are available to compare adjusted strength."
        else:
            identity, opponent = pairs.index[0]
            value = pairs.iloc[0]
            n = int(count_for_summary.loc[identity, opponent])
            if winrate is not None:
                summary = (
                    f"{identity} has the highest observed strength win rate: **{value:.1%}** "
                    f"against {opponent}, across **{n}** player comparisons."
                )
            else:
                summary = (
                    f"{identity} has the largest mean adjusted-strength advantage: "
                    f"**{value:+.3f}** against {opponent}, across **{n}** player comparisons."
                )
        return AnalysisResult(tables=tables, figures=figures, summary=summary, metadata=metadata)

    # ── matrix display ───────────────────────────────────────────────────────────
    _TIP_ROWS = [
        {"column": "n", "label": "Comparisons", "decimals": 0},
        {"column": "p_text", "label": "Paired t-test p", "text": True},
    ]

    def _winrate_heatmap(self, ctx, winrate, counts, pvalues):
        from .heatmaps import diverging_positions, matrix_cells, matrix_heatmap, p_text

        cells = matrix_cells(winrate * 100, n=counts, p_value=pvalues)
        cells["color_position"] = diverging_positions(cells["value"], 50.0, 50.0)
        cells["cell_text"] = cells["value"].map(lambda v: f"{v:.0f}%")
        cells["p_text"] = cells["p_value"].map(p_text)
        return matrix_heatmap(
            ctx, cells, winrate.index,
            title="Adjusted-strength win rate",
            help_text=(
                "Each cell is how often the row player type had higher adjusted strength "
                "than the column player type in the games they shared, with ties counted "
                "as half. Colors run from red (0%) through yellow (50%, even) to blue "
                "(100%). Hover a cell for the number of comparisons and the paired "
                "t-test p-value."
            ),
            value_label="Win rate", decimals=0, unit="%", tip_rows=self._TIP_ROWS,
            legend=[[0.0, "0%"], [0.25, "25%"], [0.5, "50% (even)"], [0.75, "75%"], [1.0, "100%"]],
        )

    def _mean_heatmap(self, ctx, mean, counts, pvalues):
        from .heatmaps import (
            diverging_positions, matrix_cells, matrix_heatmap, p_text, signed_legend, spread,
        )

        cells = matrix_cells(mean, n=counts, p_value=pvalues)
        half = spread(cells["value"], 0.0, 1e-9)
        cells["color_position"] = diverging_positions(cells["value"], 0.0, half)
        cells["cell_text"] = cells["value"].map(lambda v: f"{v:+.2f}")
        cells["p_text"] = cells["p_value"].map(p_text)
        return matrix_heatmap(
            ctx, cells, mean.index,
            title="Mean adjusted-strength difference",
            help_text=(
                "Each cell is the average adjusted-strength difference, row minus column, "
                "over the games the two player types shared. Blue means the row player "
                "type was stronger, red weaker; full color is the largest gap in the "
                "table. Hover a cell for the number of comparisons and the paired "
                "t-test p-value."
            ),
            value_label="Mean difference", decimals=2, signed=True, tip_rows=self._TIP_ROWS,
            legend=signed_legend(half, 2, "row weaker", "row stronger", "0 (even)"),
        )

    # ── vs-reference display ─────────────────────────────────────────────────────
    def _vs_reference_plots(self, ctx, vs, reference, slugs, metadata) -> dict:
        from ...plotting.pairing import PairingSpec, plot_paired_rows_interactive
        from .heatmaps import count_text, p_text

        spec = ctx.condition_pairing() or PairingSpec((), "base", "Identity")
        note = self._provenance_text(metadata) or None
        plot_vs = vs.copy()
        plot_vs["tip_n"] = plot_vs["n"].map(count_text)
        figures = {}
        if "mean" in slugs:
            plot_vs["tip_value"] = plot_vs["mean_diff_vs_ref"].map(
                lambda v: f"{v:+.3f}" if pd.notna(v) else ""
            )
            plot_vs["tip_p"] = plot_vs["p_value_mean"].map(p_text)
            figures[slugs["mean"]] = plot_paired_rows_interactive(
                plot_vs,
                catalog=ctx.catalog,
                spec=spec,
                value_col="mean_diff_vs_ref",
                identity_col="player_type",
                ref_line=0,
                ref_label="Even (0)",
                tooltip=[("Mean difference", "tip_value"), ("Paired t-test p", "tip_p"),
                         ("Comparisons", "tip_n")],
                ascending=False,
                xlabel=f"Mean adjusted-strength difference vs {reference}",
                title=f"Adjusted strength vs {reference}",
                provenance_note=note,
            )
        if "winrate" in slugs:
            plot_vs["tip_value"] = plot_vs["winrate_vs_ref"].map(
                lambda v: f"{v:.1%}" if pd.notna(v) else ""
            )
            plot_vs["tip_p"] = plot_vs["p_value_winrate"].map(p_text)
            figures[slugs["winrate"]] = plot_paired_rows_interactive(
                plot_vs,
                catalog=ctx.catalog,
                spec=spec,
                value_col="winrate_vs_ref",
                identity_col="player_type",
                ref_line=0.5,
                ref_label="Even (50%)",
                tooltip=[("Win rate", "tip_value"), ("Paired t-test p", "tip_p"),
                         ("Comparisons", "tip_n")],
                ascending=False,
                percent_x=True,
                xlabel=f"P(row stronger than {reference})",
                title=f"Adjusted-strength win rate vs {reference}",
                provenance_note=note,
            )
        return {name: fig for name, fig in figures.items() if fig is not None}

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
