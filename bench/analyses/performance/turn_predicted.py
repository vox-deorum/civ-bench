"""``performance.turn_predicted`` — per-identity predicted win-probability over time.

Ported from ``performance/turn_predicted.ipynb`` (Part A): aggregate an
estimator's per-turn ``predicted_win_probability`` per identity (``by``, default
``player_type``). When ``uses.estimators`` is omitted, every enabled estimator is
included. ``player_type`` is not in ``predictions.csv``, so it is joined from
``panel_data`` by ``(game_id, player_id)``. Reports both the per-identity mean
and the mean trajectory over ``turn_progress`` (the "victory probability over
time" curve).
"""

from __future__ import annotations

import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError


class PerformanceTurnPredicted(Analysis):
    module = "performance.turn_predicted"
    friendly_name = "Win-probability trends"
    description = (
        "Shows how each player identity's predicted chance of winning changes "
        "from the opening turns through the end of the game."
    )
    report_defaults = {"tables": [], "figures": ["over_progress"]}
    default_all_estimators = True

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        estimators = ctx.uses_estimators()
        if not estimators:
            raise AnalysisError(
                f"performance.turn_predicted '{self.stage_id}': requires at least one "
                f"enabled estimator or uses.estimators override."
            )
        by = self.params.get("by", "player_type")
        aggregate = self.params.get("aggregate", "mean")

        panel = ctx.load_table("panel")[["game_id", "player_id", by]].drop_duplicates(
            ["game_id", "player_id"]
        )
        frames = []
        for est in estimators:
            pred = ctx.load_predictions(est)
            # Inner join: both inputs already exclude flagged problem games, so a
            # prediction row with no panel identity is a genuine gap — drop it rather
            # than resurrect a fake ``Player <id>`` player_type that would pollute the
            # per-identity aggregation.
            df = pred.merge(panel, on=["game_id", "player_id"], how="inner")
            df = ctx.apply_filter(df)
            if df.empty:
                continue
            df.insert(0, "model", est)
            frames.append(df)
        if not frames:
            raise AnalysisError(
                f"performance.turn_predicted '{self.stage_id}': no rows after filtering."
            )
        df = pd.concat(frames, ignore_index=True)

        agg = "mean" if aggregate not in {"mean", "median"} else aggregate
        by_identity = (
            df.groupby(["model", by])
            .agg(
                mean_predicted=("predicted_win_probability", agg),
                n_rows=("predicted_win_probability", "size"),
                n_games=("game_id", "nunique"),
            )
            .reset_index()
            .sort_values(["model", "mean_predicted"], ascending=[True, False])
        )

        if "turn_progress" not in df.columns:
            df["turn_progress"] = (df["turn"] / df["max_turn"]).round(2)
        else:
            df["turn_progress"] = df["turn_progress"].round(2)
        over_progress = (
            df.groupby(["model", by, "turn_progress"])["predicted_win_probability"]
            .mean()
            .reset_index(name="mean_predicted")
        )

        fig = self._plot(over_progress, by, estimators, ctx)
        n_models = int(df["model"].nunique())
        summary = (
            f"Win-probability trends summarize {agg} predictions from {n_models} "
            f"estimator(s) for each {by} across "
            f"{by_identity['n_games'].sum()} identity-game(s)."
        )
        return AnalysisResult(
            tables={"by_identity": by_identity, "over_progress": over_progress},
            figures={"over_progress": fig} if fig is not None else {},
            summary=summary, metadata={"estimators": estimators, "by": by, "aggregate": agg},
        )

    def _plot(self, over_progress: pd.DataFrame, by: str, estimators: list[str], ctx: AnalysisContext):
        import matplotlib.pyplot as plt

        from ...plotting.styles import get_player_color, sort_player_types

        fig, axes = plt.subplots(
            1, len(estimators), figsize=(max(7, 4.8 * len(estimators)), 6),
            squeeze=False, sharey=True,
        )
        legend_ax = None
        for ax, est in zip(axes[0], estimators):
            sub = over_progress[over_progress["model"] == est]
            idents = sort_player_types(sub[by].unique()) if by == "player_type" \
                else sorted(sub[by].unique())
            for ident in idents:
                grp = sub[sub[by] == ident].sort_values("turn_progress")
                color = get_player_color(ctx.catalog, str(ident)) if by == "player_type" else None
                ax.plot(grp["turn_progress"], grp["mean_predicted"], marker="o", markersize=3,
                        color=color, label=str(ident))
            ax.set_xlabel("Turn progress")
            ax.set_title(str(est), fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
            if legend_ax is None and idents:
                legend_ax = ax
        axes[0][0].set_ylabel("Predicted win probability")
        if legend_ax is not None:
            legend_ax.legend(fontsize=8, loc="upper left", ncol=2)
        fig.suptitle(f"Predicted victory probability over time by {by}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig
