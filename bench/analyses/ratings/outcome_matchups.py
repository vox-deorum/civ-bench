"""``ratings.outcome_matchups`` - observed win-rate and score-ratio matchups."""

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
        "tables": [],
        "figures": ["win_rate"],
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
        if not use_vs_reference:
            figures["win_rate"] = self._plot_win_rate(win_rate, counts, expected_win_rate)
        if include_score_ratio:
            tables["score_ratio_margin"] = score_margin.reset_index()
            if not use_vs_reference:
                figures["score_ratio_margin"] = self._plot_margin(score_margin)

        if use_vs_reference:
            vs = self._vs_reference_table(
                ctx, panel, reference, win_rate, score_margin, counts,
                include_score_ratio, expected_win_rate, game_sizes,
            )
            tables["vs_reference"] = vs
            plot_vs = vs.copy()
            expected_rates = vs["expected_win_rate"].dropna().unique()
            common_expected = float(expected_rates[0]) if len(expected_rates) == 1 else None
            plot_vs["n_label"] = plot_vs.apply(
                lambda row: (
                    f"{int(row['wins'])}/{int(row['n'])}"
                    + (f"; expected {row['expected_win_rate']:.1%}" if common_expected is None else "")
                ) if row["n"] > 0 else "",
                axis=1,
            )
            from bench.plotting.pairing import PairingSpec, plot_paired_rows

            pairing = ctx.condition_pairing()
            plot_spec = pairing or PairingSpec((), "base", "Identity")
            figures["win_rate"] = plot_paired_rows(
                plot_vs,
                catalog=ctx.catalog,
                spec=plot_spec,
                value_col="win_rate_vs_ref",
                identity_col="player_type",
                ref_line=common_expected,
                ref_label=f"Expected ({common_expected:.1%})" if common_expected is not None else None,
                annotate_col="n_label",
                ascending=False,
                xlabel="Victory rate per player",
                title="Per-player victory rates",
                provenance_note=(
                    "Counts show wins / player appearances. Expected rate assumes each player "
                    "has an equal chance to win."
                ),
            )
            from matplotlib.ticker import PercentFormatter

            figures["win_rate"].axes[0].xaxis.set_major_formatter(PercentFormatter(1))
            if include_score_ratio:
                plot_vs["n_label"] = plot_vs["n"].map(lambda n: f"n={int(n)}")
                figures["score_ratio_margin"] = plot_paired_rows(
                    plot_vs,
                    catalog=ctx.catalog,
                    spec=plot_spec,
                    value_col="score_ratio_margin_vs_ref",
                    identity_col="player_type",
                    ref_line=0,
                    annotate_col="n_label",
                    ascending=False,
                    xlabel=f"Mean score-ratio margin vs {reference}",
                    title=f"Score-ratio margins vs {reference}",
                )

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
                f"{identity} has the highest observed per-player matchup victory rate: "
                f"**{pairs.iloc[0]:.1%}** (**{wins}/{n}** wins per player appearance). "
                f"The expected per-player victory rate with equal chances is "
                f"**{expected:.1%}** {baseline_context}."
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

    def _plot_win_rate(
        self, matrix: pd.DataFrame, counts: pd.DataFrame, expected: pd.DataFrame
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns

        annot = matrix.copy().astype(object)
        for row in matrix.index:
            for col in matrix.columns:
                value = matrix.loc[row, col]
                if pd.isna(value):
                    annot.loc[row, col] = ""
                else:
                    n_appearances = int(counts.loc[row, col])
                    wins = int(round(float(value) * n_appearances))
                    annot.loc[row, col] = (
                        f"{100 * value:.1f}%\nexpected {expected.loc[row, col]:.1%}"
                        f"\n{wins}/{n_appearances}"
                    )

        n = len(matrix)
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * n + 3), max(5, 0.8 * n + 2)))
        sns.heatmap(
            matrix,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            square=True,
            linewidths=0.3,
            linecolor="lightgray",
            cbar_kws={"label": "Victory rate per player appearance"},
            annot_kws={"fontsize": 7},
            ax=ax,
        )
        ax.set_title("Per-player matchup victory rates", fontsize=12, fontweight="bold")
        ax.set_xlabel("Opponent player type\nCounts = wins / player appearances; expected = equal chance")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        return fig

    def _plot_margin(self, matrix: pd.DataFrame):
        import matplotlib.pyplot as plt
        import seaborn as sns

        n = len(matrix)
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 3), max(5, 0.6 * n + 2)))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0.0,
            square=True,
            linewidths=0.3,
            linecolor="lightgray",
            cbar_kws={"label": "mean(row - col) score_ratio"},
            annot_kws={"fontsize": 7},
            ax=ax,
        )
        ax.set_title("Matchup score-ratio margins", fontsize=12, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        return fig
