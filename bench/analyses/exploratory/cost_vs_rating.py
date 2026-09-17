"""``exploratory.cost_vs_rating``: average token cost versus Elo rating."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bench.analyses.base import Analysis, AnalysisContext, AnalysisResult
from bench.analyses.errors import AnalysisError
from bench.analyses.exploratory.model_token_costs import (
    COST_NOTE, compute_game_costs, summarize_game_costs,
)


class ExploratoryCostVsRating(Analysis):
    module = "exploratory.cost_vs_rating"
    friendly_name = "Cost versus skill"
    description = (
        "Compares average cost per player per game with each identity's "
        "estimated skill rating."
    )
    report_defaults = {"tables": [], "figures": ["cost_vs_rating"]}

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
        unpriced_mask = nonbaseline["avg_cost_per_player_game"].isna()
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

        from bench.plotting.pairing import PairingSpec, attach_pair_columns

        pairing = ctx.condition_pairing()
        plot_spec = pairing or PairingSpec((), "base", "Base")
        joined = attach_pair_columns(joined, ctx.catalog, plot_spec, "player_type")
        columns = [
            "player_type", "base_identity", "condition", "avg_cost_per_game",
            "complete_games", "na_games", "elo", "se_elo", "ci_lower", "ci_upper",
            "avg_cost_per_player_game", "avg_input", "avg_output",
            "player_games", "complete_player_games", "na_player_games",
        ]
        table = joined.reindex(columns=columns).sort_values(
            ["base_identity", "condition"], kind="stable"
        ).reset_index(drop=True)
        positive_cost = table["avg_cost_per_player_game"].where(
            table["avg_cost_per_player_game"] > 0
        )
        table["relative_strength_per_cost"] = (
            np.power(10.0, (table["elo"] - 1500) / 400) / positive_cost
        )

        currency = str(self.params.get("currency", "usd"))
        log_x = bool(self.params.get("log_x", True))
        annotate = bool(self.params.get("annotate", False))
        fig = self._plot(table, ctx, plot_spec, currency, log_x, annotate)
        comparable = table[
            np.isfinite(table["elo"]) & np.isfinite(table["avg_cost_per_player_game"])
        ]
        if comparable.empty:
            summary = "No identities have both a skill rating and a known player-game cost."
        else:
            efficient = comparable[np.isfinite(comparable["relative_strength_per_cost"])]
            if efficient.empty:
                summary = "No rated identities have a positive cost for comparing efficiency."
            else:
                best = efficient.loc[efficient["relative_strength_per_cost"].idxmax()]
                worst = efficient.loc[efficient["relative_strength_per_cost"].idxmin()]
                summary = (
                    f"Most cost-efficient: **{best['player_type']}** "
                    f"({best['elo']:.0f} Elo, **{best['avg_cost_per_player_game']:.4g} "
                    f"{currency.upper()}** per player per game). "
                    f"Least cost-efficient: **{worst['player_type']}** "
                    f"({worst['elo']:.0f} Elo, **{worst['avg_cost_per_player_game']:.4g} "
                    f"{currency.upper()}** per player per game)."
                )
        summary += (
            " Efficiency is Elo-derived relative strength per unit cost: "
            "10^((Elo - 1500) / 400) / average player-game cost; higher is better. "
            "Zero-cost identities are excluded from efficiency rankings. " + COST_NOTE
        )
        metadata = {
            "currency": currency,
            "log_x": log_x,
            "ratings_stage": rating_stage,
            "dropped_baselines": dropped_baselines,
            "dropped_unpriced": dropped_unpriced,
            "dropped_unrated": dropped_unrated,
            "cost_basis": "per player per complete game",
            "cached_tokens_accounted_for": False,
            "efficiency_metric": "10^((elo - 1500) / 400) / avg_cost_per_player_game",
        }
        return AnalysisResult(
            tables={"cost_vs_rating": table},
            figures={"cost_vs_rating": fig} if fig is not None else {},
            summary=summary,
            metadata=metadata,
        )

    @staticmethod
    def _plot(table, ctx, spec, currency, log_x, annotate):
        from html import escape

        import plotly.graph_objects as go

        from bench.plotting.styles import get_player_color

        plot = table[
            np.isfinite(table["avg_cost_per_player_game"]) & np.isfinite(table["elo"])
        ].copy()
        if log_x:
            plot = plot[plot["avg_cost_per_player_game"] > 0]
        if plot.empty:
            return None

        fig = go.Figure()
        symbols = ["circle", "diamond-open", "square-open", "triangle-up-open", "cross"]
        condition_symbols = {
            condition: symbols[index % len(symbols)]
            for index, condition in enumerate(("base", *spec.suffixes))
        }
        for identity, group in plot.groupby("base_identity", sort=False):
            group = group.sort_values("condition", kind="stable")
            details = [
                [escape(str(row.player_type)),
                 escape(spec.base_label if row.condition == "base" else str(row.condition).lstrip("-")),
                 row.avg_input, row.avg_output, row.complete_player_games,
                 row.na_player_games, row.relative_strength_per_cost]
                for row in group.itertuples(index=False)
            ]
            fig.add_trace(go.Scatter(
                x=group["avg_cost_per_player_game"].tolist(),
                y=group["elo"].tolist(),
                name=escape(str(identity)),
                mode="markers",
                marker={
                    "color": get_player_color(ctx.catalog, str(identity)),
                    "size": 11,
                    "symbol": [condition_symbols.get(str(c), "circle") for c in group["condition"]],
                    "line": {"width": 1.5},
                },
                customdata=details,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Condition: %{customdata[1]}"
                    "<br>Elo: %{y:.0f}"
                    f"<br>Average cost per player per game: %{{x:.4g}} {escape(currency.upper())}"
                    "<br>Average input tokens: %{customdata[2]:,.0f}"
                    "<br>Average output tokens (including reasoning): %{customdata[3]:,.0f}"
                    "<br>Complete player-games: %{customdata[4]}"
                    "<br>Incomplete player-games: %{customdata[5]}"
                    "<br>Relative strength per unit cost: %{customdata[6]:.3g}"
                    "<extra></extra>"
                ),
            ))

        # Optional labels cover only efficiency extremes to keep crowded plots readable.
        if annotate:
            ranked = plot[np.isfinite(plot["relative_strength_per_cost"])]
            if not ranked.empty:
                extremes = dict.fromkeys([
                    ranked["relative_strength_per_cost"].idxmax(),
                    ranked["relative_strength_per_cost"].idxmin(),
                ])
                for index in extremes:
                    row = ranked.loc[index]
                    x = float(row["avg_cost_per_player_game"])
                    fig.add_annotation(
                        x=float(np.log10(x)) if log_x else x,
                        y=float(row["elo"]), text=escape(str(row["player_type"])),
                        showarrow=True, ax=35, ay=-30,
                    )

        fig.add_hline(y=1500, line_dash="dash", line_color="#888888")
        fig.update_layout(
            template="plotly_white",
            title="Cost versus skill",
            xaxis={
                "title": f"Average cost per player per game ({escape(currency.upper())})",
                "type": "log" if log_x else "linear",
            },
            yaxis={"title": "Elo rating"},
            legend={"title": {"text": "Models (click to filter)"}},
            hovermode="closest",
            height=650,
            margin={"b": 110},
        )
        fig.add_annotation(
            x=0, y=-0.18, xref="paper", yref="paper", xanchor="left",
            text="Hover for token averages. Click legend entries to filter. Drag to zoom.<br>"
                 "Costs do not account for cached tokens or cache discounts.",
            showarrow=False, align="left", font={"size": 11},
        )
        return fig
