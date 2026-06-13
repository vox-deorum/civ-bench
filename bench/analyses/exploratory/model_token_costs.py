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


class ExploratoryModelTokenCosts(Analysis):
    module = "exploratory.model_token_costs"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        currency = self.params.get("currency", "usd")
        by_player_type = bool(
            self.params.get("by_player_type", self.params.get("by_strategist", True))
        )

        tokens = ctx.load_table("tokens")
        tokens = ctx.apply_filter(tokens)
        catalog = ctx.catalog
        pricing = catalog.pricing_per_million()

        df = tokens.copy()
        df["model"] = df["model_name"].apply(catalog.canonicalize_model_name)
        df["combined_output_tokens"] = (
            df.get("reasoning_tokens", 0).fillna(0) + df.get("output_tokens", 0).fillna(0)
        )
        df["input_per_million"] = df["model"].map(
            lambda m: pricing.get(m, {}).get("input_per_million")
        )
        df["output_per_million"] = df["model"].map(
            lambda m: pricing.get(m, {}).get("output_per_million")
        )
        df["row_cost"] = (
            df["input_tokens"] / 1_000_000 * df["input_per_million"]
            + df["combined_output_tokens"] / 1_000_000 * df["output_per_million"]
        )

        model_tbl = self._summarize(df, ["model"])
        tables = {}
        plot_tbl = model_tbl
        if by_player_type:
            by_player_tbl = self._summarize(df, ["player_type", "model"])
            tables["token_costs_by_player_type"] = by_player_tbl
            plot_tbl = by_player_tbl
        tables["token_costs"] = model_tbl

        fig = self._plot(plot_tbl, currency, catalog)
        total = model_tbl["total_cost"].sum(skipna=True)
        breakdown = " across player types" if by_player_type else ""
        summary = (
            f"Token costs for {len(model_tbl)} model(s){breakdown}; total = "
            f"{total:.2f} {currency.upper()} over {int(model_tbl['games'].sum())} games."
        )
        return AnalysisResult(
            tables=tables,
            figures={"token_costs": fig} if fig is not None else {},
            summary=summary,
            metadata={"currency": currency, "by_player_type": by_player_type},
        )

    @staticmethod
    def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        per_game = (
            df.groupby(["game_id", *group_cols], as_index=False)
            .agg(
                input_tokens=("input_tokens", _complete_sum),
                combined_output_tokens=("combined_output_tokens", _complete_sum),
                total_cost=("row_cost", _complete_sum),
            )
        )
        per_game["available"] = per_game[
            ["input_tokens", "combined_output_tokens", "total_cost"]
        ].notna().all(axis=1)
        valid = per_game[per_game["available"]]

        games = per_game.groupby(group_cols, as_index=False).agg(
            games=("game_id", "nunique"),
            na_games=("available", lambda s: int((~s).sum())),
        )
        averages = valid.groupby(group_cols, as_index=False).agg(
            avg_input=("input_tokens", "mean"),
            avg_output=("combined_output_tokens", "mean"),
            total_cost=("total_cost", "sum"),
        )
        summary_tbl = games.merge(averages, on=group_cols, how="left")
        prices = df.drop_duplicates("model").set_index("model")[
            ["input_per_million", "output_per_million"]
        ]
        summary_tbl = summary_tbl.merge(prices, left_on="model", right_index=True, how="left")
        summary_tbl = summary_tbl.sort_values("total_cost", ascending=False, na_position="last")
        summary_tbl = summary_tbl.reset_index(drop=True)
        return summary_tbl

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
