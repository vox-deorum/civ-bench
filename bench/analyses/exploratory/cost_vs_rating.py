"""``exploratory.cost_vs_rating`` — average token cost versus Elo rating."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .model_token_costs import compute_game_costs, summarize_game_costs


class ExploratoryCostVsRating(Analysis):
    module = "exploratory.cost_vs_rating"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        upstream = ctx.uses_analyses()
        if not upstream:
            raise AnalysisError(
                f"exploratory.cost_vs_rating '{self.stage_id}' requires the ratings "
                "stage in uses.analyses."
            )
        rating_stage = upstream[0]
        ratings = ctx.load_analysis_table(rating_stage, "ratings")
        required = {"player_type", "elo"}
        missing = sorted(required - set(ratings.columns))
        if missing:
            raise AnalysisError(
                f"exploratory.cost_vs_rating '{self.stage_id}': analysis "
                f"'{rating_stage}' ratings table is missing {missing}."
            )
        if ratings["player_type"].duplicated().any():
            raise AnalysisError(
                f"exploratory.cost_vs_rating '{self.stage_id}': analysis "
                f"'{rating_stage}' must rate one row per player_type."
            )

        tokens = ctx.apply_filter(ctx.load_table("tokens"))
        game_costs = compute_game_costs(tokens, ctx.catalog)
        costs = summarize_game_costs(game_costs, ["player_type"])

        baselines = {ctx.catalog.vanilla_label, ctx.catalog.null_label}
        baseline_mask = costs["player_type"].astype(str).isin(baselines)
        dropped_baselines = int(baseline_mask.sum())
        nonbaseline = costs[~baseline_mask].copy()
        unpriced_mask = nonbaseline["avg_cost_per_game"].isna()
        dropped_unpriced = int(unpriced_mask.sum())
        priced = nonbaseline[~unpriced_mask].copy()
        rating_ids = set(ratings["player_type"].astype(str))
        dropped_unrated = int((~priced["player_type"].astype(str).isin(rating_ids)).sum())

        rating_cols = ["player_type", "elo"]
        for column in ("se_elo", "ci_lower", "ci_upper"):
            if column not in ratings.columns:
                ratings[column] = np.nan
            rating_cols.append(column)
        joined = priced.merge(ratings[rating_cols], on="player_type", how="inner")

        from ...plotting.pairing import PairingSpec, attach_pair_columns

        pairing = ctx.condition_pairing()
        plot_spec = pairing or PairingSpec((), "base", "Base")
        joined = attach_pair_columns(joined, ctx.catalog, plot_spec, "player_type")
        columns = [
            "player_type", "base_identity", "condition", "avg_cost_per_game",
            "complete_games", "na_games", "elo", "se_elo", "ci_lower", "ci_upper",
        ]
        table = joined.reindex(columns=columns).sort_values(
            ["base_identity", "condition"], kind="stable"
        ).reset_index(drop=True)

        currency = str(self.params.get("currency", "usd"))
        log_x = bool(self.params.get("log_x", True))
        annotate = bool(self.params.get("annotate", True))
        fig = self._plot(table, ctx, plot_spec, currency, log_x, annotate)
        summary = (
            f"Average cost versus Elo for {len(table)} identities from rating stage "
            f"'{rating_stage}'; excluded {dropped_baselines} baseline, "
            f"{dropped_unpriced} unpriced, and {dropped_unrated} unrated identities."
        )
        metadata = {
            "currency": currency,
            "log_x": log_x,
            "ratings_stage": rating_stage,
            "dropped_baselines": dropped_baselines,
            "dropped_unpriced": dropped_unpriced,
            "dropped_unrated": dropped_unrated,
        }
        return AnalysisResult(
            tables={"cost_vs_rating": table},
            figures={"cost_vs_rating": fig} if fig is not None else {},
            summary=summary,
            metadata=metadata,
        )

    @staticmethod
    def _plot(table, ctx, spec, currency, log_x, annotate):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        from ...plotting.pairing import condition_marker
        from ...plotting.styles import get_player_color

        plot = table.dropna(subset=["avg_cost_per_game", "elo"]).copy()
        if log_x:
            plot = plot[plot["avg_cost_per_game"] > 0]
        if plot.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 6.5))
        for _, group in plot.groupby("base_identity", sort=False):
            if len(group) > 1:
                ordered = group.sort_values("condition", kind="stable")
                ax.plot(
                    ordered["avg_cost_per_game"], ordered["elo"],
                    color="#999999", linewidth=0.8, alpha=0.65, zorder=1,
                )

        present_conditions: set[str] = set()
        for row in plot.itertuples(index=False):
            condition = str(row.condition)
            color = get_player_color(ctx.catalog, str(row.base_identity))
            marker = condition_marker(spec, condition)
            face = color if condition == "base" else "none"
            lo = getattr(row, "ci_lower")
            hi = getattr(row, "ci_upper")
            if pd.notna(lo) and pd.notna(hi):
                yerr = [[max(0.0, row.elo - lo)], [max(0.0, hi - row.elo)]]
            elif pd.notna(getattr(row, "se_elo")):
                yerr = abs(float(row.se_elo))
            else:
                yerr = None
            ax.errorbar(
                row.avg_cost_per_game, row.elo, yerr=yerr, fmt=marker,
                color=color, ecolor=color, markerfacecolor=face,
                markeredgecolor=color, markersize=8, capsize=3, zorder=3,
            )
            present_conditions.add(condition)

        if annotate:
            for identity, group in plot.groupby("base_identity", sort=False):
                row = group.loc[group["avg_cost_per_game"].idxmax()]
                ax.annotate(
                    str(identity), (row["avg_cost_per_game"], row["elo"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8,
                    color=get_player_color(ctx.catalog, str(identity)),
                )

        if log_x:
            ax.set_xscale("log")
        ax.axhline(1500, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Average cost per complete game ({currency.upper()})")
        ax.set_ylabel("Elo rating (error bars: CI or +/-1 SE)")
        ax.set_title("Token cost versus strategist rating", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)
        handles = []
        for condition in ("base", *spec.suffixes):
            if condition not in present_conditions:
                continue
            handles.append(Line2D(
                [0], [0], marker=condition_marker(spec, condition), linestyle="none",
                markerfacecolor="#555555" if condition == "base" else "none",
                markeredgecolor="#555555",
                label=spec.base_label if condition == "base" else condition.lstrip("-"),
            ))
        handles.append(Line2D([0], [0], color="gray", linestyle="--", label="Reference (1500)"))
        ax.legend(handles=handles, fontsize=9)
        fig.tight_layout()
        return fig

