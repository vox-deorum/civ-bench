"""``ratings.outcome_matchups`` - observed win-rate and score-ratio matchups.

With ``presentation.matchup_display: "matrix"`` each matrix shows as an HTML
heatmap table (``win_rate_heatmap``, ``score_ratio_margin_heatmap``); with
``"vs_reference"`` each shows as an interactive forest plot against Vanilla. The
victory rate and score margin sit behind the ``victory_rate`` and
``score_margin`` view tabs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bench.analyses.base import Analysis, AnalysisContext, AnalysisResult
from bench.analyses.errors import AnalysisError


def create_outcome_matchup_matrices(panel: pd.DataFrame, include_score_ratio: bool = True):
    """Observed row-vs-column outcomes, deduping repeated opponent identities.

    Each row player contributes at most one observation per opponent player_type in
    a game. If a game has several opponents of the same player_type, the win-rate
    denominator is still one for that row player and the score-ratio comparator is
    that opponent identity's in-game mean score_ratio.
    """
    player_types = sorted(panel["player_type"].dropna().astype(str).unique())
    n = len(player_types)
    idx = {p: i for i, p in enumerate(player_types)}
    wins = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=float)
    score_diffs = [[[] for _ in range(n)] for _ in range(n)]

    df = panel.copy()
    df["player_type"] = df["player_type"].astype(str)
    df["is_winner"] = pd.to_numeric(df["is_winner"], errors="coerce").fillna(0.0)
    if include_score_ratio:
        df["score_ratio"] = pd.to_numeric(df["score_ratio"], errors="coerce")

    for _, game in df.groupby("game_id", sort=False):
        score_by_type = {}
        if include_score_ratio:
            score_by_type = game.groupby("player_type")["score_ratio"].mean().to_dict()
        types_in_game = set(game["player_type"])
        for row in game.itertuples(index=False):
            row_type = str(getattr(row, "player_type"))
            row_idx = idx[row_type]
            for opp_type in sorted(types_in_game - {row_type}):
                col_idx = idx[opp_type]
                counts[row_idx, col_idx] += 1.0
                wins[row_idx, col_idx] += float(getattr(row, "is_winner"))
                if include_score_ratio:
                    row_score = getattr(row, "score_ratio")
                    opp_score = score_by_type.get(opp_type, np.nan)
                    if pd.notna(row_score) and pd.notna(opp_score):
                        score_diffs[row_idx][col_idx].append(float(row_score) - float(opp_score))

    win_rate = np.full((n, n), np.nan)
    score_margin = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j or counts[i, j] == 0:
                continue
            win_rate[i, j] = wins[i, j] / counts[i, j]
            if include_score_ratio and score_diffs[i][j]:
                score_margin[i, j] = float(np.mean(score_diffs[i][j]))

    index = pd.Index(player_types, name="player_type")
    return (
        pd.DataFrame(win_rate, index=index, columns=player_types),
        pd.DataFrame(score_margin, index=index, columns=player_types),
        pd.DataFrame(counts, index=index, columns=player_types),
    )


class RatingsOutcomeMatchups(Analysis):
    module = "ratings.outcome_matchups"
    friendly_name = "Victory matchups"
    description = (
        "Shows wins per player appearance in shared games and final-score margins. "
        "The equal-chance victory rate is 1 divided by the number of players "
        "in each game (12.5% for eight players)."
    )
    report_defaults = {
        "tables": ["win_rate_heatmap", "score_ratio_margin_heatmap"],
        "figures": ["win_rate", "score_ratio_margin"],
    }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        table_id = next((t for t in ctx.uses_tables() if t == "panel"), "panel")
        full_panel = ctx.load_table(table_id)
        panel = ctx.apply_filter(full_panel)
        include_score_ratio = bool(self.params.get("include_score_ratio", True))

        required = {"game_id", "player_type", "is_winner"}
        if include_score_ratio:
            required.add("score_ratio")
        missing = sorted(required - set(panel.columns))
        if missing or panel.empty:
            raise AnalysisError(
                f"ratings.outcome_matchups '{self.stage_id}': need a non-empty "
                f"panel table with columns {sorted(required)}"
                + (f" (missing {missing})." if missing else ".")
            )

        win_rate, score_margin, counts = create_outcome_matchup_matrices(
            panel, include_score_ratio=include_score_ratio
        )
        game_sizes = full_panel.groupby("game_id").size()
        expected_panel = panel.assign(is_winner=1.0 / panel["game_id"].map(game_sizes))
        expected_win_rate, _, _ = create_outcome_matchup_matrices(
            expected_panel, include_score_ratio=False
        )
        metadata = {
            "table": table_id,
            "include_score_ratio": include_score_ratio,
            "victory_rate_unit": "wins per player",
            "expected_win_rate": (
                "Equal chance: 1 / full game player count, averaged over player appearances."
            ),
            "p_value_win_rate": (
                "Binomial test against equal chance, available for fixed game sizes "
                "with one appearance of the player type per game."
            ),
        }
        display = ctx.matchup_display()
        reference = ctx.catalog.vanilla_label
        reference_available = reference in win_rate.columns
        use_vs_reference = display == "vs_reference" and reference_available
        tables = {
            "win_rate": win_rate.reset_index(),
            "counts": counts.reset_index(),
            "expected_win_rate": expected_win_rate.reset_index(),
        }
        figures = {}
        heatmaps = {}
        if not use_vs_reference:
            table, spec = self._win_rate_heatmap(ctx, win_rate, counts, expected_win_rate)
            tables["win_rate_heatmap"], heatmaps["win_rate_heatmap"] = table, spec
        if include_score_ratio:
            tables["score_ratio_margin"] = score_margin.reset_index()
            if not use_vs_reference:
                table, spec = self._margin_heatmap(ctx, score_margin, counts)
                tables["score_ratio_margin_heatmap"] = table
                heatmaps["score_ratio_margin_heatmap"] = spec

        if use_vs_reference:
            vs = self._vs_reference_table(
                ctx, panel, reference, win_rate, score_margin, counts,
                include_score_ratio, expected_win_rate, game_sizes,
            )
            tables["vs_reference"] = vs
            figures.update(self._vs_reference_plots(ctx, vs, reference, include_score_ratio))

        if heatmaps:
            metadata["heatmaps"] = heatmaps
        if include_score_ratio:
            metadata["views"] = {
                "victory_rate": {
                    "label": "Victory rate",
                    "tip": "Wins per player appearance in shared games.",
                    "tables": ["win_rate_heatmap"],
                    "figures": ["win_rate"],
                },
                "score_margin": {
                    "label": "Score margin",
                    "tip": "The average final score-ratio gap, row minus column.",
                    "tables": ["score_ratio_margin_heatmap"],
                    "figures": ["score_ratio_margin"],
                },
            }

        metadata["display"] = "vs_reference" if use_vs_reference else "matrix"
        if include_score_ratio:
            metadata["score_ratio_margin"] = "row minus column"
        if display == "vs_reference" and not reference_available:
            metadata["warning"] = f"reference '{reference}' is absent"
        values = win_rate[[reference]] if use_vs_reference else win_rate
        pairs = values.where(np.isfinite(values)).stack().sort_values(ascending=False, kind="stable")
        if pairs.empty:
            summary = "No shared games are available to compare victory rates."
        else:
            identity, opponent = pairs.index[0]
            n = int(counts.loc[identity, opponent])
            wins = int(round(float(pairs.iloc[0]) * n))
            expected = expected_win_rate.loc[identity, opponent]
            shared_games = panel.loc[panel["player_type"].astype(str) == opponent, "game_id"]
            identity_games = panel.loc[panel["player_type"].astype(str) == identity, "game_id"]
            sizes = game_sizes.loc[
                game_sizes.index.isin(shared_games) & game_sizes.index.isin(identity_games)
            ].unique()
            baseline_context = (
                f"in {int(sizes[0])}-player games" if len(sizes) == 1
                else "averaged over these player appearances"
            )
            summary = (
                f"{identity} has the highest per-player matchup victory rate: "
                f"**{pairs.iloc[0]:.1%}** (**{wins}/{n}**; expected "
                f"**{expected:.1%}** {baseline_context}.)"
            )
        return AnalysisResult(tables=tables, figures=figures, summary=summary, metadata=metadata)

    @staticmethod
    def _vs_reference_table(
        ctx,
        panel,
        reference,
        win_rate,
        score_margin,
        counts,
        include_score_ratio,
        expected_win_rate,
        game_sizes,
    ):
        from scipy.stats import binomtest, ttest_1samp

        from bench.plotting.pairing import PairingSpec, attach_pair_columns

        score_samples: dict[str, list[float]] = {}
        if include_score_ratio:
            for _, game in panel.groupby("game_id", sort=False):
                ref_rows = game[game["player_type"].astype(str) == reference]
                ref_mean = pd.to_numeric(ref_rows["score_ratio"], errors="coerce").mean()
                if pd.isna(ref_mean):
                    continue
                for row in game[game["player_type"].astype(str) != reference].itertuples(index=False):
                    value = pd.to_numeric(pd.Series([getattr(row, "score_ratio")]), errors="coerce").iloc[0]
                    if pd.notna(value):
                        score_samples.setdefault(str(getattr(row, "player_type")), []).append(
                            float(value) - float(ref_mean)
                        )

        rows = []
        reference_games = panel.loc[panel["player_type"].astype(str) == reference, "game_id"]
        for identity in sorted(set(win_rate.index.astype(str)) - {reference}):
            value = win_rate.loc[identity, reference]
            n = counts.loc[identity, reference]
            wins = int(round(float(value) * float(n))) if pd.notna(value) and n else 0
            expected = expected_win_rate.loc[identity, reference]
            appearances = panel.loc[
                (panel["player_type"].astype(str) == identity)
                & panel["game_id"].isin(reference_games), "game_id"
            ]
            binomial_valid = (
                n > 0 and not appearances.duplicated().any()
                and appearances.map(game_sizes).nunique() == 1
            )
            row = {
                "player_type": identity,
                "win_rate_vs_ref": value,
                "wins": wins,
                "n": n,
                "expected_win_rate": expected,
                "p_value_win_rate": (
                    binomtest(wins, int(n), p=expected).pvalue if binomial_valid else np.nan
                ),
            }
            if include_score_ratio:
                samples = score_samples.get(identity, [])
                row["score_ratio_margin_vs_ref"] = score_margin.loc[identity, reference]
                row["p_value_score_ratio"] = (
                    ttest_1samp(samples, 0).pvalue if len(samples) > 1 else np.nan
                )
            rows.append(row)
        out = pd.DataFrame(rows)
        pairing = ctx.condition_pairing()
        spec = pairing or PairingSpec((), "base", "Identity")
        return attach_pair_columns(out, ctx.catalog, spec, "player_type").drop(
            columns="is_baseline"
        )

    # ── matrix display ───────────────────────────────────────────────────────────
    def _win_rate_heatmap(self, ctx, win_rate, counts, expected):
        """Victory rates colored against each cell's equal-chance rate.

        Yellow is the equal-chance rate, red runs down to zero wins, and blue
        reaches full color at twice the equal-chance rate, so a strong result in
        an eight-player game (25% against 12.5%) reads as strongly as it is.
        """
        from .heatmaps import matrix_cells, matrix_heatmap

        cells = matrix_cells(win_rate * 100, n=counts, expected=expected * 100)
        rate, chance = cells["value"], pd.to_numeric(cells["expected"], errors="coerce")
        position = pd.Series(np.nan, index=cells.index)
        valid = chance > 0
        position[valid] = (0.5 + 0.5 * (rate[valid] - chance[valid]) / chance[valid]).clip(0.0, 1.0)
        cells["color_position"] = position
        cells["cell_text"] = rate.map(lambda v: f"{v:.1f}%")
        cells["wins_text"] = [
            f"{int(round(v * n / 100))}/{int(n)}" for v, n in zip(rate, cells["n"])
        ]
        return matrix_heatmap(
            ctx, cells, win_rate.index,
            title="Per-player matchup victory rates",
            help_text=(
                "Each cell is the share of the row player type's appearances, in games "
                "with the column player type, that ended in a win. The equal-chance rate "
                "is 1 divided by the number of players in the game (12.5% for eight "
                "players). Yellow is equal chance, blue is twice that or more, and red "
                "runs toward zero wins. Hover a cell for wins over appearances and the "
                "equal-chance rate."
            ),
            value_label="Victory rate", decimals=1, unit="%",
            tip_rows=[
                {"column": "wins_text", "label": "Wins", "text": True},
                {"column": "expected", "label": "Equal chance", "decimals": 1, "unit": "%"},
            ],
            legend=[[0.0, "no wins"], [0.25, ""], [0.5, "equal chance"], [0.75, ""],
                    [1.0, "2x equal chance or more"]],
        )

    def _margin_heatmap(self, ctx, margin, counts):
        from .heatmaps import (
            diverging_positions, matrix_cells, matrix_heatmap, signed_legend, spread,
        )

        cells = matrix_cells(margin, n=counts)
        half = spread(cells["value"], 0.0, 1e-9)
        cells["color_position"] = diverging_positions(cells["value"], 0.0, half)
        cells["cell_text"] = cells["value"].map(lambda v: f"{v:+.2f}")
        return matrix_heatmap(
            ctx, cells, margin.index,
            title="Matchup score-ratio margins",
            help_text=(
                "Each cell is the row player's final score ratio minus the column player "
                "type's, averaged over the games they shared. Blue means the row player "
                "type finished ahead, red behind; full color is the largest gap in the "
                "table."
            ),
            value_label="Score-ratio margin", decimals=2, signed=True,
            tip_rows=[{"column": "n", "label": "Appearances", "decimals": 0}],
            legend=signed_legend(half, 2, "row behind", "row ahead", "0 (even)"),
        )

    # ── vs-reference display ─────────────────────────────────────────────────────
    def _vs_reference_plots(self, ctx, vs, reference, include_score_ratio) -> dict:
        from bench.analyses.ratings.heatmaps import count_text, p_text
        from bench.plotting.pairing import PairingSpec, plot_paired_rows_interactive

        spec = ctx.condition_pairing() or PairingSpec((), "base", "Identity")
        plot_vs = vs.copy()
        expected_rates = vs["expected_win_rate"].dropna().unique()
        common_expected = float(expected_rates[0]) if len(expected_rates) == 1 else None
        plot_vs["tip_rate"] = plot_vs["win_rate_vs_ref"].map(
            lambda v: f"{v:.1%}" if pd.notna(v) else ""
        )
        plot_vs["tip_wins"] = [
            f"{int(w)}/{int(n)}" if n > 0 else "" for w, n in zip(plot_vs["wins"], plot_vs["n"])
        ]
        plot_vs["tip_expected"] = plot_vs["expected_win_rate"].map(
            lambda v: f"{v:.1%}" if pd.notna(v) else ""
        )
        plot_vs["tip_p"] = plot_vs["p_value_win_rate"].map(p_text)
        figures = {
            "win_rate": plot_paired_rows_interactive(
                plot_vs,
                catalog=ctx.catalog,
                spec=spec,
                value_col="win_rate_vs_ref",
                identity_col="player_type",
                ref_line=common_expected,
                ref_label=f"Expected ({common_expected:.1%})" if common_expected is not None else None,
                tooltip=[("Victory rate", "tip_rate"), ("Wins", "tip_wins"),
                         ("Equal chance", "tip_expected"), ("Binomial p", "tip_p")],
                ascending=False,
                percent_x=True,
                xlabel="Victory rate per player",
                title="Per-player victory rates",
                provenance_note=(
                    "Wins count per player appearance. The expected rate assumes each player "
                    "has an equal chance to win."
                ),
            )
        }
        if include_score_ratio:
            plot_vs["tip_margin"] = plot_vs["score_ratio_margin_vs_ref"].map(
                lambda v: f"{v:+.3f}" if pd.notna(v) else ""
            )
            plot_vs["tip_margin_p"] = plot_vs["p_value_score_ratio"].map(p_text)
            plot_vs["tip_n"] = plot_vs["n"].map(count_text)
            figures["score_ratio_margin"] = plot_paired_rows_interactive(
                plot_vs,
                catalog=ctx.catalog,
                spec=spec,
                value_col="score_ratio_margin_vs_ref",
                identity_col="player_type",
                ref_line=0,
                ref_label="Even (0)",
                tooltip=[("Score-ratio margin", "tip_margin"), ("t-test p", "tip_margin_p"),
                         ("Appearances", "tip_n")],
                ascending=False,
                xlabel=f"Mean score-ratio margin vs {reference}",
                title=f"Score-ratio margins vs {reference}",
            )
        return {name: fig for name, fig in figures.items() if fig is not None}
