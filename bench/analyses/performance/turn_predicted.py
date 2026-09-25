"""``performance.turn_predicted``: predicted win probability over the game.

The pooled counterpart of the Matched Maps seat chart. It reads one estimator's
per-turn ``predicted_win_probability``: ``uses.estimators`` when it names one,
otherwise the estimator of the strength stage (the one whose P(win) defines
adjusted strength). ``player_type`` is not in ``predictions.csv``, so the
identity (``by``, default ``player_type``) is joined from ``panel`` by
``(game_id, player_id)``.

With ``by = player_type`` and condition pairing enabled, each identity splits
into a strategist and a condition exactly as on the Matched Maps pages, and the
strength stage's ``baseline_experiment`` games form the VPAI reference curve
(without one, every Vanilla seat does). Every run is interpolated onto the
shared 101-point progress grid (:mod:`bench.analyses.performance.curves`) and
each grid point averages the runs that cover it. The result declares the
curve table through ``metadata["curve_chart"]``, which the report renders as
the same interactive chart the Matched Maps pages use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...plotting.pairing import condition_display, order_conditions, order_strategists
from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .curves import (
    GRID_POINTS, clean_curve_predictions, curve_records, interpolate_curves,
    progress_grid, split_identities,
)

CURVE_KEYS = ["strategist", "condition"]
CURVE_COLUMNS = [*CURVE_KEYS, "turn_progress", "mean_predicted_win_probability", "n_runs"]

# Line colors for identities that are not player types (``by`` other than
# ``player_type``), which have no catalog color.
_FALLBACK_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

_CHART_HELP = (
    "Each curve is the mean of every run's interpolated victory probability on "
    "the fixed 0 to 1 progress grid, pooled over all games. The VPAI reference "
    "curve is drawn thicker when present. The vertical axis fits the visible "
    "curves. Hover the chart to compare every checked curve at one progress point."
)


class PerformanceTurnPredicted(Analysis):
    module = "performance.turn_predicted"
    friendly_name = "Win-probability trends"
    description = (
        "Shows how each player identity's predicted chance of winning changes "
        "from the opening turns through the end of the game."
    )
    report_defaults = {"tables": [], "figures": []}

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        where = f"performance.turn_predicted '{self.stage_id}'"
        table_id = ctx.strength_table_id()
        explicit = ctx.uses_estimators()
        estimator_id = explicit[0] if explicit else ctx.strength_estimator_id(table_id)
        if not estimator_id:
            raise AnalysisError(
                f"{where}: no estimator to read. Set uses.estimators, or list the "
                f"estimator in the uses.estimators of strength stage '{table_id}'."
            )
        by = self.params.get("by", "player_type")
        aggregate = self.params.get("aggregate", "mean")
        agg = aggregate if aggregate in {"mean", "median"} else "mean"
        vanilla = ctx.catalog.vanilla_label

        points = self._load_points(ctx, estimator_id, by, where)
        spec = ctx.condition_pairing() if by == "player_type" else None
        baseline = self._baseline_experiment(ctx, table_id) if by == "player_type" else None
        if by != "player_type":
            reference = pd.Series(False, index=points.index)
        elif baseline and "experiment" in points.columns:
            reference = points["experiment"].astype(str) == baseline
        else:
            reference = points["player_type"].astype(str) == vanilla
        strategists, keys = split_identities(
            ctx.catalog, spec, points[by].astype(str), vanilla, reference
        )
        points["strategist"] = strategists
        points["condition"] = [_condition(spec, key, vanilla) for key in keys]

        over_progress = pd.DataFrame(
            curve_records(interpolate_curves(points, progress_grid(), CURVE_KEYS), CURVE_KEYS),
            columns=CURVE_COLUMNS,
        ).sort_values([*CURVE_KEYS, "turn_progress"], kind="mergesort").reset_index(drop=True)

        by_identity = (
            points.groupby([by, *CURVE_KEYS], sort=True)
            .agg(
                mean_predicted=("predicted_win_probability", agg),
                n_rows=("predicted_win_probability", "size"),
                n_games=("game_id", "nunique"),
            )
            .reset_index()
            .sort_values(
                ["mean_predicted", by, *CURVE_KEYS],
                ascending=[False, True, True, True], kind="mergesort",
            )
            .reset_index(drop=True)
        )

        observed = [s for s in over_progress["strategist"].unique() if s != vanilla]
        if by == "player_type":
            strategist_order = order_strategists(ctx.catalog, observed)
            colors = self._strategist_colors(ctx, strategist_order, vanilla)
        else:
            strategist_order = sorted(observed)
            colors = {
                name: _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
                for i, name in enumerate(strategist_order)
            }
        condition_order = (
            order_conditions(spec, {k for k in keys if k}, vanilla) if spec is not None else []
        )

        metadata = {
            "estimator": estimator_id,
            "strength_table": table_id,
            "by": by,
            "aggregate": agg,
            "games": int(points["game_id"].nunique()),
            "curve_chart": {
                "table": "over_progress",
                "vanilla_label": vanilla,
                "strategist_order": strategist_order,
                "condition_order": condition_order,
                "strategist_colors": colors,
                "grid_points": GRID_POINTS,
                "help": _CHART_HELP,
            },
        }
        if baseline:
            metadata["baseline_experiment"] = baseline
        return AnalysisResult(
            tables={"by_identity": by_identity, "over_progress": over_progress},
            summary=_summary(by_identity, by, agg, estimator_id, vanilla),
            metadata=metadata,
        )

    # ── input preparation ──────────────────────────────────────────────────────
    def _load_points(
        self, ctx: AnalysisContext, estimator_id: str, by: str, where: str
    ) -> pd.DataFrame:
        """Filtered prediction points of one estimator, each with its identity."""
        panel = ctx.load_table("panel")
        if by not in panel.columns:
            raise AnalysisError(f"{where}: panel has no '{by}' column to group by.")
        panel = panel[["game_id", "player_id", by]].drop_duplicates(["game_id", "player_id"])
        panel["game_id"] = panel["game_id"].astype(str)
        pred = ctx.load_predictions(estimator_id)
        pred["game_id"] = pred["game_id"].astype(str)
        if "turn_progress" not in pred.columns and {"turn", "max_turn"} <= set(pred.columns):
            pred["turn_progress"] = pred["turn"] / pred["max_turn"]
        # Inner join: both inputs already exclude flagged problem games, so a
        # prediction row with no panel identity is a genuine gap, so drop it rather
        # than resurrect a fake ``Player <id>`` player_type that would pollute the
        # per-identity aggregation.
        df = ctx.apply_filter(pred.merge(panel, on=["game_id", "player_id"], how="inner"))
        keep = tuple(dict.fromkeys(c for c in (by, "experiment") if c in df.columns))
        points = clean_curve_predictions(df, estimator_id, where, keep)
        if points.empty:
            raise AnalysisError(f"{where}: no rows after filtering.")
        return points.reset_index(drop=True)

    def _baseline_experiment(self, ctx: AnalysisContext, table_id: str):
        stage = next((s for s in ctx.config.adjust if s.id == table_id), None)
        value = ((stage.raw.get("params") or {}).get("baseline_experiment")
                 if stage is not None else None)
        return str(value) if value else None

    def _strategist_colors(
        self, ctx: AnalysisContext, strategist_order: list[str], vanilla_label: str
    ) -> dict[str, str]:
        from ...plotting.styles import get_player_color

        colors = {vanilla_label: get_player_color(ctx.catalog, vanilla_label)}
        for name in strategist_order:
            colors[name] = get_player_color(ctx.catalog, name)
        return colors


def _condition(spec, key: str, vanilla_label: str) -> str:
    """The display condition; identities without pairing have none."""
    if spec is not None:
        return condition_display(spec, key, vanilla_label)
    return vanilla_label if key == "vanilla" else ""


def _identity_label(row: dict, vanilla_label: str) -> str:
    strategist, condition = str(row["strategist"]), str(row["condition"])
    name = "VPAI" if strategist == vanilla_label else strategist
    if condition and condition != vanilla_label:
        return f"{name} | {condition}"
    return name


def _summary(by_identity: pd.DataFrame, by: str, agg: str, estimator_id: str,
             vanilla_label: str) -> str:
    finite = by_identity[np.isfinite(by_identity["mean_predicted"])]
    if finite.empty:
        return f"No finite {agg} predicted win probabilities are available."
    highest = finite.iloc[0].to_dict()
    lowest = finite.iloc[-1].to_dict()
    lead = (
        f"Highest {agg} predicted win probability from **{estimator_id}**: "
        f"**{_identity_label(highest, vanilla_label)}** at "
        f"**{highest['mean_predicted'] * 100:.1f}%**."
    )
    if len(finite) == 1:
        return lead
    return (
        f"{lead} {agg.capitalize()}s range from **{lowest['mean_predicted'] * 100:.1f}%** "
        f"to **{highest['mean_predicted'] * 100:.1f}%**."
    )
