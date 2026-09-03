"""``exploratory.model_token_costs`` — token usage × pricing per model.

Ported from ``exploratory/model_token_costs.ipynb``: from ``model_token_usage``,
compute per-row cost = ``input_tokens/1e6 * input_price + (reasoning+output)/1e6
* output_price`` using the catalog's per-model pricing (``pricing_per_million``),
aggregate to per-game totals (a game with any missing token field is counted but
excluded from the averages, mirroring the legacy ``complete_group_sum``), then
summarize costs by player type + model by default, while also preserving the
legacy per-model aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult


def _complete_sum(s: pd.Series) -> float:
    """Sum that is NaN if any element is NaN (a partial-telemetry game is unusable)."""
    return float(s.sum()) if s.notna().all() else float("nan")


def compute_game_costs(tokens_df: pd.DataFrame, catalog) -> pd.DataFrame:
    """Price token rows and aggregate them to complete per-player-game costs."""
    pricing = catalog.pricing_per_million()
    df = tokens_df.copy()
    model_names = df["model_name"].where(df["model_name"].notna(), "Unattributed")
    df["model"] = model_names.astype(str).apply(catalog.canonicalize_model_name)

    def numeric_column(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    df["combined_output_tokens"] = (
        numeric_column("reasoning_tokens").fillna(0)
        + numeric_column("output_tokens").fillna(0)
    )
    df["input_tokens"] = numeric_column("input_tokens")
    df["input_per_million"] = df["model"].map(
        lambda model: pricing.get(model, {}).get("input_per_million")
    )
    df["output_per_million"] = df["model"].map(
        lambda model: pricing.get(model, {}).get("output_per_million")
    )
    df["row_cost"] = (
        df["input_tokens"] / 1_000_000 * df["input_per_million"]
        + df["combined_output_tokens"] / 1_000_000 * df["output_per_million"]
    )
    identity_cols = ["game_id"]
    if "player_type" in df.columns:
        identity_cols.append("player_type")
    identity_cols.append("model")
    per_game = df.groupby(identity_cols, as_index=False).agg(
        input_tokens=("input_tokens", _complete_sum),
        combined_output_tokens=("combined_output_tokens", _complete_sum),
        total_cost=("row_cost", _complete_sum),
        input_per_million=("input_per_million", "first"),
        output_per_million=("output_per_million", "first"),
    )
    per_game["available"] = per_game[
        ["input_tokens", "combined_output_tokens", "total_cost"]
    ].notna().all(axis=1)
    return per_game


def summarize_game_costs(per_identity_game: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize per-game costs with explicit complete-game denominators."""
    per_game = per_identity_game.groupby(["game_id", *group_cols], as_index=False).agg(
        input_tokens=("input_tokens", _complete_sum),
        combined_output_tokens=("combined_output_tokens", _complete_sum),
        total_cost=("total_cost", _complete_sum),
    )
    per_game["available"] = per_game[
        ["input_tokens", "combined_output_tokens", "total_cost"]
    ].notna().all(axis=1)
    valid = per_game[per_game["available"]]
    games = per_game.groupby(group_cols, as_index=False).agg(
        games=("game_id", "nunique"),
        na_games=("available", lambda values: int((~values).sum())),
    )
    averages = valid.groupby(group_cols, as_index=False).agg(
        avg_input=("input_tokens", "mean"),
        avg_output=("combined_output_tokens", "mean"),
        total_cost=("total_cost", "sum"),
    )
    summary_tbl = games.merge(averages, on=group_cols, how="left")
    summary_tbl["complete_games"] = summary_tbl["games"] - summary_tbl["na_games"]
    summary_tbl["avg_cost_per_game"] = (
        summary_tbl["total_cost"] / summary_tbl["complete_games"].replace(0, np.nan)
    )
    if "model" in group_cols:
        prices = per_identity_game.drop_duplicates("model").set_index("model")[
            ["input_per_million", "output_per_million"]
        ]
        summary_tbl = summary_tbl.merge(
            prices, left_on="model", right_index=True, how="left"
        )
    return summary_tbl.sort_values(
        "total_cost", ascending=False, na_position="last"
    ).reset_index(drop=True)


class ExploratoryModelTokenCosts(Analysis):
    module = "exploratory.model_token_costs"
    friendly_name = "Model usage and cost"
    description = (
        "Summarizes token use and estimated US-dollar cost by model and player "
        "type."
    )
    report_defaults = {"tables": [], "figures": ["token_costs"]}

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        currency = self.params.get("currency", "usd")
        by_player_type = bool(
            self.params.get("by_player_type", self.params.get("by_strategist", True))
        )

        tokens = ctx.load_table("tokens")
        tokens = ctx.apply_filter(tokens)
        catalog = ctx.catalog
        game_costs = compute_game_costs(tokens, catalog)

        model_tbl = self._summarize(game_costs, ["model"])
        tables = {}
        plot_tbl = model_tbl
        if by_player_type:
            by_player_tbl = self._summarize(game_costs, ["player_type", "model"])
            tables["token_costs_by_player_type"] = by_player_tbl
            plot_tbl = by_player_tbl
        tables["token_costs"] = model_tbl

        warning = ""
        pairing = ctx.condition_pairing() if by_player_type else None
        if pairing is None:
            fig = self._plot(plot_tbl, currency, catalog)
        else:
            fig, warning = self._plot_paired(plot_tbl, currency, ctx, pairing)
        total = model_tbl["total_cost"].sum(skipna=True)
        breakdown = " across player types" if by_player_type else ""
        summary = (
            f"The {len(model_tbl)} model(s){breakdown} cost {total:.2f} "
            f"{currency.upper()} across {int(model_tbl['games'].sum())} games."
        )
        if warning:
            summary = summary[:-1] + "; " + warning.rstrip(".") + "."
        return AnalysisResult(
            tables=tables,
            figures={"token_costs": fig} if fig is not None else {},
            summary=summary,
            metadata={"currency": currency, "by_player_type": by_player_type},
        )

    @staticmethod
    def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        return summarize_game_costs(df, group_cols)

    def _plot_paired(self, tbl, currency, ctx, pairing):
        from ...plotting.pairing import (
            attach_pair_columns,
            paired_sort_order,
            plot_paired_rows,
        )

        work = attach_pair_columns(tbl, ctx.catalog, pairing, "player_type")
        cost_order = paired_sort_order(work, pairing, "avg_cost_per_game", ascending=False)
        warning = ""
        upstream = ctx.uses_analyses()
        if upstream:
            ratings = ctx.load_analysis_table(upstream[0], "ratings")
            if "player_type" not in ratings.columns or "elo" not in ratings.columns:
                from ..errors import AnalysisError

                raise AnalysisError(
                    f"exploratory.model_token_costs '{self.stage_id}': analysis "
                    f"'{upstream[0]}' ratings table needs player_type and elo columns."
                )
            rated = attach_pair_columns(ratings, ctx.catalog, pairing, "player_type")
            rating_order = paired_sort_order(rated, pairing, "elo", ascending=False)
            available = set(work["base_identity"].astype(str))
            priced = set(
                work.loc[work["avg_cost_per_game"].notna(), "base_identity"].astype(str)
            )
            row_order = [
                identity for identity in rating_order
                if identity in available and identity in priced
            ]
            row_order.extend(identity for identity in cost_order if identity not in row_order)
        else:
            row_order = cost_order
            warning = (
                "No uses.analyses rating stage was declared; paired costs are "
                "ordered by cost instead of Elo"
            )

        plot_tbl = tbl.copy()
        plot_tbl["n_label"] = plot_tbl["complete_games"].map(
            lambda value: f"n={int(value)}" if pd.notna(value) else ""
        )
        fig = plot_paired_rows(
            plot_tbl,
            catalog=ctx.catalog,
            spec=pairing,
            value_col="avg_cost_per_game",
            identity_col="player_type",
            annotate_col="n_label",
            row_order=row_order,
            ascending=False,
            xlabel=f"Average cost per complete game ({currency.upper()})",
            title="Estimated token cost per game by player type",
        )
        return fig, warning

    def _plot(self, tbl: pd.DataFrame, currency: str, catalog):
        import matplotlib.pyplot as plt

        from ...plotting.styles import get_player_color

        priced = tbl.dropna(subset=["total_cost"])
        if priced.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(priced) + 1.5)))
        label_col = "player_type" if "player_type" in priced.columns else "model"
        labels = priced[label_col] if label_col == "model" else (
            priced["player_type"].astype(str) + " / " + priced["model"].astype(str)
        )
        colors = [get_player_color(catalog, m) for m in priced["model"]]
        ax.barh(labels[::-1], priced["total_cost"][::-1], color=colors[::-1])
        ax.set_xlabel(f"Total cost ({currency.upper()})")
        title = "Estimated token cost by player type" if "player_type" in priced.columns else (
            "Estimated token cost by model"
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig
