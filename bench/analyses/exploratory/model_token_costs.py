"""``exploratory.model_token_costs`` — token usage × pricing per model.

Ported from ``exploratory/model_token_costs.ipynb``: from ``model_token_usage``,
compute per-row cost = ``input_tokens/1e6 * input_price + (reasoning+output)/1e6
* output_price`` using the catalog's per-model pricing (``pricing_per_million``),
aggregate to per-game totals (a game with any missing token field is counted but
excluded from the averages, mirroring the legacy ``complete_group_sum``), then
summarize per model: games, N/A games, average input/output tokens, total cost.
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
        group_by_strategist = bool(self.params.get("by_strategist", False))

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

        group_cols = ["player_type", "model"] if group_by_strategist else ["model"]
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

        fig = self._plot(summary_tbl, currency)
        total = summary_tbl["total_cost"].sum(skipna=True)
        summary = (
            f"Token costs for {len(summary_tbl)} model(s); total = "
            f"{total:.2f} {currency.upper()} over {int(summary_tbl['games'].sum())} games."
        )
        return AnalysisResult(
            tables={"token_costs": summary_tbl},
            figures={"token_costs": fig} if fig is not None else {},
            summary=summary, metadata={"currency": currency},
        )

    def _plot(self, tbl: pd.DataFrame, currency: str):
        import matplotlib.pyplot as plt

        priced = tbl.dropna(subset=["total_cost"])
        if priced.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(priced) + 1.5)))
        label_col = "player_type" if "player_type" in priced.columns else "model"
        labels = priced[label_col] if label_col == "model" else (
            priced["player_type"].astype(str) + " / " + priced["model"].astype(str)
        )
        ax.barh(labels[::-1], priced["total_cost"][::-1], color="#4C72B0")
        ax.set_xlabel(f"Total cost ({currency.upper()})")
        ax.set_title("Estimated token cost by model", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig
