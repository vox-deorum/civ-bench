"""``performance.turn_predicted`` — per-identity predicted win-probability over time.

Ported from ``performance/turn_predicted.ipynb`` (Part A): aggregate an
estimator's per-turn ``predicted_win_probability`` per identity (``by``, default
``player_type``). ``player_type`` is not in ``predictions.csv``, so it is joined
from ``panel_data`` by ``(game_id, player_id)``. Reports both the per-identity
mean and the mean trajectory over ``turn_progress`` (the "victory probability
over time" curve).
"""

from __future__ import annotations

import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError


def _strength_table_id(ctx: AnalysisContext) -> str:
    """The adjust stage id referenced via uses.tables (default first adjust / 'strength')."""
    for tbl in ctx.uses_tables():
        if any(s.id == tbl for s in ctx.config.adjust):
            return tbl
    if ctx.config.adjust:
        return ctx.config.adjust[0].id
    return "strength"


class PerformanceTurnPredicted(Analysis):
    module = "performance.turn_predicted"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        estimators = ctx.uses_estimators()
        if not estimators:
            raise AnalysisError(
                f"performance.turn_predicted '{self.stage_id}': requires uses.estimators."
            )
        est = estimators[0]
        by = self.params.get("by", "player_type")
        aggregate = self.params.get("aggregate", "mean")

        pred = ctx.load_predictions(est)
        panel = ctx.load_table("panel")[["game_id", "player_id", by]].drop_duplicates(
            ["game_id", "player_id"]
        )
        df = pred.merge(panel, on=["game_id", "player_id"], how="left")
        df[by] = df[by].fillna("Player " + df["player_id"].astype(str))
        df = ctx.apply_filter(df)
        if df.empty:
            raise AnalysisError(
                f"performance.turn_predicted '{self.stage_id}': no rows after filtering."
            )

        agg = "mean" if aggregate not in {"mean", "median"} else aggregate
        by_identity = (
            df.groupby(by)
            .agg(
                mean_predicted=("predicted_win_probability", agg),
                n_rows=("predicted_win_probability", "size"),
                n_games=("game_id", "nunique"),
            )
            .reset_index()
            .sort_values("mean_predicted", ascending=False)
        )

        if "turn_progress" not in df.columns:
            df["turn_progress"] = (df["turn"] / df["max_turn"]).round(2)
        else:
            df["turn_progress"] = df["turn_progress"].round(2)
        over_progress = (
            df.groupby([by, "turn_progress"])["predicted_win_probability"]
            .mean()
            .reset_index(name="mean_predicted")
        )

        fig = self._plot(over_progress, by, est, ctx)
        summary = (
            f"Per-{by} predicted win-probability ({agg}) from estimator '{est}' over "
            f"{by_identity['n_games'].sum()} identity-games."
        )
        return AnalysisResult(
            tables={"by_identity": by_identity, "over_progress": over_progress},
            figures={"over_progress": fig} if fig is not None else {},
            summary=summary, metadata={"estimator": est, "by": by, "aggregate": agg},
        )

    def _plot(self, over_progress: pd.DataFrame, by: str, est: str, ctx: AnalysisContext):
        import matplotlib.pyplot as plt

        from ...plotting.styles import get_player_color, sort_player_types

        fig, ax = plt.subplots(figsize=(11, 6))
        idents = sort_player_types(over_progress[by].unique()) if by == "player_type" \
            else sorted(over_progress[by].unique())
        for ident in idents:
            grp = over_progress[over_progress[by] == ident].sort_values("turn_progress")
            color = get_player_color(ctx.catalog, str(ident)) if by == "player_type" else None
            ax.plot(grp["turn_progress"], grp["mean_predicted"], marker="o", markersize=3,
                    color=color, label=str(ident))
        ax.set_xlabel("Turn progress")
        ax.set_ylabel("Predicted win probability")
        ax.set_title(f"Predicted victory probability over time by {by} (est: {est})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
