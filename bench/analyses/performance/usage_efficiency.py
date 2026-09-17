"""Per-player usage and skill relative to fitted resource requirements."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

from bench.analyses.base import Analysis, AnalysisContext, AnalysisResult
from bench.analyses.errors import AnalysisError
from bench.analyses.performance.usage import (
    COST_NOTE, compute_game_costs, plot_usage_figures, summarize_game_costs,
)


METRICS = {
    "cost": ("avg_cost_per_player_game", "Cost"),
    "input": ("avg_input", "Input tokens"),
    "output": ("avg_output", "Output tokens"),
}


def fit_usage_skill(table: pd.DataFrame, metric: str) -> dict | None:
    """Fit equally weighted identities and store their Elo residuals in place."""
    column = METRICS[metric][0]
    expected_col = f"expected_elo_{metric}"
    residual_col = f"efficiency_elo_{metric}"
    table[expected_col] = np.nan
    table[residual_col] = np.nan
    valid = np.isfinite(table["elo"]) & np.isfinite(table[column]) & (table[column] > 0)
    values = table.loc[valid, column]
    # Two observations fit exactly and cannot support a residual comparison.
    if len(values) < 3 or values.nunique() < 2:
        return None
    x = np.log10(values.to_numpy())
    y = table.loc[valid, "elo"].to_numpy()
    centered = x - x.mean()
    slope = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
    intercept = float(y.mean() - slope * x.mean())
    expected = intercept + slope * x
    residual = y - expected
    total_variation = float(np.dot(y - y.mean(), y - y.mean()))
    r_squared = (
        float(1 - np.dot(residual, residual) / total_variation)
        if total_variation > 0 else None
    )
    residual[np.isclose(residual, 0, atol=1e-9)] = 0
    table.loc[valid, expected_col] = expected
    table.loc[valid, residual_col] = residual
    return {"intercept": intercept, "slope": slope, "n": len(values), "r_squared": r_squared}


class PerformanceUsageEfficiency(Analysis):
    module = "performance.usage_efficiency"
    friendly_name = "Usage, cost, and skill"
    description = (
        "Compares cost and token use per player per game with skill, "
        "and measures Elo above or below the fitted usage-skill curve."
    )
    report_defaults = {
        "tables": [], "figures": ["usage_vs_rating"],
    }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        upstream = ctx.uses_analyses()
        if not upstream:
            raise AnalysisError(
                f"{self.module} '{self.stage_id}' requires the ratings stage in uses.analyses."
            )
        rating_stage = upstream[0]
        ratings = ctx.load_analysis_table(rating_stage, "ratings")
        missing = sorted({"player_type", "elo"} - set(ratings.columns))
        if missing:
            raise AnalysisError(
                f"{self.module} '{self.stage_id}': analysis '{rating_stage}' "
                f"ratings table is missing {missing}."
            )
        if ratings["player_type"].duplicated().any():
            raise AnalysisError(
                f"{self.module} '{self.stage_id}': analysis '{rating_stage}' "
                "must rate one row per player_type."
            )

        tokens = ctx.apply_filter(ctx.load_table("tokens"))
        usage = summarize_game_costs(compute_game_costs(tokens, ctx.catalog), ["player_type"])
        baselines = {ctx.catalog.vanilla_label, ctx.catalog.null_label}
        baseline_mask = usage["player_type"].astype(str).isin(baselines)
        nonbaseline = usage[~baseline_mask].copy()
        rating_cols = [column for column in ("player_type", "elo", "se_elo", "ci_lower", "ci_upper")
                       if column in ratings.columns]
        joined = nonbaseline.merge(ratings[rating_cols], on="player_type", how="inner")

        from bench.plotting.pairing import PairingSpec, attach_pair_columns

        spec = ctx.condition_pairing() or PairingSpec((), "base", "Base")
        table = attach_pair_columns(joined, ctx.catalog, spec, "player_type").sort_values(
            ["base_identity", "condition"], kind="stable"
        ).reset_index(drop=True)
        fits = {metric: fit_usage_skill(table, metric) for metric in METRICS}
        baseline_rows = ratings[
            (ratings["player_type"] == ctx.catalog.vanilla_label) & np.isfinite(ratings["elo"])
        ]
        baseline_elo = float(baseline_rows.iloc[0]["elo"]) if not baseline_rows.empty else 1500.0
        baseline_name = ctx.catalog.vanilla_label if not baseline_rows.empty else "Elo reference"
        null_rows = ratings[
            (ratings["player_type"] == ctx.catalog.null_label) & np.isfinite(ratings["elo"])
        ]
        null_baseline_elo = float(null_rows.iloc[0]["elo"]) if not null_rows.empty else None
        currency = str(self.params.get("currency", "usd"))
        log_x = bool(self.params.get("log_x", True))
        annotate = bool(self.params.get("annotate", False))
        figures = plot_usage_figures(nonbaseline, ctx, ratings, currency=currency)
        fig = self._plot(
            table, ctx, spec, currency, log_x, annotate, fits,
            baseline_elo, baseline_name, null_baseline_elo,
        )
        if fig is not None:
            figures["usage_vs_rating"] = fig

        efficient = table[np.isfinite(table["efficiency_elo_cost"])]
        if efficient.empty:
            summary = (
                "Cost efficiency needs at least three rated identities with positive "
                "costs and at least two distinct costs."
            )
        elif np.allclose(efficient["efficiency_elo_cost"], 0, atol=1e-9):
            summary = "All rated identities lie on the fitted cost-skill curve."
        else:
            best = efficient.loc[efficient["efficiency_elo_cost"].idxmax()]
            worst = efficient.loc[efficient["efficiency_elo_cost"].idxmin()]
            summary = (
                f"Most cost-efficient: **{best['player_type']}** "
                f"(**{best['efficiency_elo_cost']:+.0f} Elo** vs the fitted curve). "
                f"Least cost-efficient: **{worst['player_type']}** "
                f"(**{worst['efficiency_elo_cost']:+.0f} Elo** vs the fitted curve)."
            )
        metadata = {
            "currency": currency, "log_x": log_x, "ratings_stage": rating_stage,
            "dropped_baselines": int(baseline_mask.sum()),
            "unpriced_identities": int(nonbaseline["avg_cost_per_player_game"].isna().sum()),
            "unrated_identities": int((~nonbaseline["player_type"].isin(ratings["player_type"])).sum()),
            "cost_basis": "per player per complete game", "cached_tokens_accounted_for": False,
            "efficiency_metric": "elo - expected_elo",
            "usage_skill_equation": "expected_elo = intercept + slope * log10(average_usage)",
            "usage_skill_fits": fits, "baseline_elo": baseline_elo, "baseline_name": baseline_name,
            "null_baseline_elo": null_baseline_elo,
        }
        return AnalysisResult(
            tables={"usage": usage, "usage_vs_rating": table}, figures=figures,
            summary=summary, metadata=metadata,
        )

    @staticmethod
    def _plot(
        table, ctx, spec, currency, log_x, annotate, fits,
        baseline_elo, baseline_name, null_baseline_elo=None,
    ):
        import plotly.graph_objects as go

        from bench.plotting.styles import get_player_color

        if table.empty:
            return None
        has_null_baseline = null_baseline_elo is not None and np.isfinite(null_baseline_elo)
        baseline_label = "VPAI" if baseline_name == ctx.catalog.vanilla_label else baseline_name
        fig = go.Figure()
        symbols = ["circle", "diamond-open", "square-open", "triangle-up-open", "cross"]
        condition_symbols = {
            condition: symbols[index % len(symbols)]
            for index, condition in enumerate(("base", *spec.suffixes))
        }
        trace_metrics = []
        metric_annotations = {}
        axis_titles = {}
        for metric, (column, label) in METRICS.items():
            plot = table[np.isfinite(table[column]) & np.isfinite(table["elo"])].copy()
            if log_x:
                plot = plot[plot[column] > 0]
            unit = f" ({escape(currency.upper())})" if metric == "cost" else ""
            axis_titles[metric] = f"Average {label.lower()} per player per game{unit}"
            annotations = [{
                "x": 0, "y": -0.19, "xref": "paper", "yref": "paper", "xanchor": "left",
                "text": "Hover for details. Select a resource above. Click legend entries to filter.<br>"
                        "Costs exclude cache discounts. Output includes reasoning tokens.",
                "showarrow": False, "align": "left", "font": {"size": 11},
            }, {
                "x": 1, "y": baseline_elo, "xref": "paper", "yref": "y", "xanchor": "right",
                "yanchor": "top", "text": f"<b>{escape(baseline_label)} baseline: {baseline_elo:,.0f} Elo</b>",
                "showarrow": False, "bgcolor": "white", "font": {"color": "#334155"},
            }]
            if has_null_baseline:
                annotations.append({
                    "x": 1, "y": null_baseline_elo, "xref": "paper", "yref": "y", "xanchor": "right",
                    "yanchor": "bottom",
                    "text": f"<b>Null baseline: {null_baseline_elo:,.0f} Elo (lower bound)</b>",
                    "showarrow": False, "bgcolor": "white", "font": {"color": "#64748b"},
                })
            fit = fits[metric]
            equation = "Fit needs 3 rated identities with positive values and 2 distinct values."
            if fit is not None:
                values = plot.loc[plot[column] > 0, column]
                curve_x = np.geomspace(values.min(), values.max(), 100)
                equation_variable = f"cost in {escape(currency.upper())}" if metric == "cost" else label.lower()
                r_squared_label = f"{fit['r_squared']:.3f}" if fit["r_squared"] is not None else "N/A"
                equation = (
                    f"Fitted Elo = {fit['intercept']:.1f} {fit['slope']:+.1f} "
                    f"log₁₀({equation_variable}); R² = {r_squared_label}"
                )
                fig.add_trace(go.Scatter(
                    x=curve_x.tolist(), y=(fit["intercept"] + fit["slope"] * np.log10(curve_x)).tolist(),
                    mode="lines", name=f"Fitted {label.lower()}-skill curve", showlegend=False,
                    line={"color": "#475569", "width": 2, "dash": "dot"}, hoverinfo="skip",
                    visible=metric == "cost",
                ))
                trace_metrics.append(metric)
            annotations.append({
                "x": 0, "y": 1.02, "xref": "paper", "yref": "paper", "xanchor": "left",
                "text": equation, "showarrow": False, "font": {"size": 12, "color": "#475569"},
            })
            for identity, group in plot.groupby("base_identity", sort=False):
                group = group.sort_values("condition", kind="stable")
                details = []
                for row in group.to_dict("records"):
                    def number(value, format_spec=",.0f"):
                        return format(value, format_spec) if pd.notna(value) and np.isfinite(value) else "N/A"

                    token_metric = metric != "cost"
                    complete = row["token_complete_player_games"] if token_metric else row["complete_player_games"]
                    incomplete = row["token_na_player_games"] if token_metric else row["na_player_games"]
                    details.append([
                        str(row["player_type"]),
                        spec.base_label if row["condition"] == "base" else str(row["condition"]).lstrip("-"),
                        number(row["elo"]), f"{baseline_elo:,.0f} ({baseline_label})",
                        number(row[f"efficiency_elo_{metric}"], "+,.0f"),
                        number(row[f"expected_elo_{metric}"]),
                        f"{number(row['avg_cost_per_player_game'], ',.4g')} {currency.upper()}",
                        number(row["avg_input"]), number(row["avg_output"]),
                        number(complete), number(incomplete), label,
                    ])
                fig.add_trace(go.Scatter(
                    x=group[column].tolist(), y=group["elo"].tolist(), name=escape(str(identity)),
                    legendgroup=str(identity), mode="markers", visible=metric == "cost",
                    marker={
                        "color": get_player_color(ctx.catalog, str(identity)), "size": 11,
                        "symbol": [condition_symbols.get(str(c), "circle") for c in group["condition"]],
                        "line": {"width": 1.5},
                    },
                    customdata=details, meta={"table_tooltip": True}, hoverinfo="none",
                ))
                trace_metrics.append(metric)

            ranked = plot[np.isfinite(plot[f"efficiency_elo_{metric}"])]
            if annotate and not ranked.empty and not np.allclose(ranked[f"efficiency_elo_{metric}"], 0, atol=1e-9):
                for index in dict.fromkeys([
                    ranked[f"efficiency_elo_{metric}"].idxmax(), ranked[f"efficiency_elo_{metric}"].idxmin(),
                ]):
                    row = ranked.loc[index]
                    x = float(row[column])
                    annotations.append({
                        "x": float(np.log10(x)) if log_x else x, "y": float(row["elo"]),
                        "text": escape(str(row["player_type"])), "showarrow": True, "ax": 35, "ay": -30,
                    })
            metric_annotations[metric] = annotations

        # A paper-width reference is retained even when the default cost view is empty.
        fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=baseline_elo, y1=baseline_elo,
                      line={"dash": "dash", "color": "#334155", "width": 2})
        if has_null_baseline:
            fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                          y0=null_baseline_elo, y1=null_baseline_elo,
                          line={"dash": "dashdot", "color": "#64748b", "width": 2})
        buttons = [{
            "label": label, "method": "update",
            "args": [{"visible": [value == metric for value in trace_metrics]}, {
                "xaxis.title.text": axis_titles[metric], "xaxis.autorange": True,
                "yaxis.autorange": True, "annotations": metric_annotations[metric],
            }],
        } for metric, (_, label) in METRICS.items()]
        fig.update_layout(
            template="plotly_white",
            title={"text": "Usage versus skill", "x": 0.05, "y": 0.98, "yanchor": "top"},
            xaxis={"title": axis_titles["cost"], "type": "log" if log_x else "linear"},
            yaxis={"title": "Elo rating"},
            legend={"title": {"text": "Models (click to filter)"}}, hovermode="closest",
            height=650, margin={"b": 110, "t": 135}, annotations=metric_annotations["cost"],
            updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.14,
                          "xanchor": "left", "yanchor": "top", "active": 0}],
            meta={"table_tooltip": {
                "title_index": 0, "subtitle_index": 1,
                "note": "Averages per player per game. Counts are player-games for the selected resource. Output includes reasoning.",
                "rows": [
                    {"label": "Elo", "index": 2, "emphasis": True},
                    {"label": "Elo above fit", "index": 4, "emphasis": True},
                    {"label": "Avg. cost", "index": 6},
                    {"label": "Avg. input tokens", "index": 7},
                    {"label": "Avg. output tokens", "index": 8},
                    {"label": "Missing data", "index": 10},
                ],
            }},
        )
        return fig
