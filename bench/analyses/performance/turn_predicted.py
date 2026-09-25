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

On a controlled design the stage also emits ``over_progress_relative``: each
controlled run's probability at every progress point goes through the
strength stage's own matched-cell adjustment, so the curve shows adjusted
strength over the game (0.5 is level with matched VPAI self-play). The chart
then offers Relative and Absolute views.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...plotting.pairing import condition_display, order_conditions, order_strategists
from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .curves import (
    GRID_POINTS, adjusted_curves, clean_curve_predictions, curve_records,
    interpolate_curves, progress_grid, split_identities,
)

CURVE_KEYS = ["strategist", "condition"]
CURVE_COLUMNS = [*CURVE_KEYS, "turn_progress", "mean_predicted_win_probability", "n_runs"]
RELATIVE_VALUE = "mean_adjusted_strength"
RELATIVE_COLUMNS = [*CURVE_KEYS, "turn_progress", RELATIVE_VALUE, "n_runs"]

# Why the Absolute curves average raw probabilities (the report's seat pages
# carry the same sentence in bench.reports.curves).
LINEAR_MEAN_NOTE = (
    "Probabilities are averaged linearly to compare with their expectations "
    "(12.5% in 8-player games, for example)."
)
ABSOLUTE_TIP = "Mean predicted win probability, averaged linearly"

# Line colors for identities that are not player types (``by`` other than
# ``player_type``), which have no catalog color.
_FALLBACK_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

_CHART_HELP = (
    "Each curve is the mean of every run's interpolated victory probability on "
    "the fixed 0 to 1 progress grid, pooled over all games. " + LINEAR_MEAN_NOTE
    + " The VPAI reference "
    "curve is drawn thicker when present. The vertical axis fits the visible "
    "curves. Only VPAI and the best and worst strategists by mean curve value "
    "start checked; All and Best and worst switch the selection. Hover a "
    "checkbox to preview or highlight that strategist's curves, or a legend "
    "entry to highlight its curve. Hover the chart to compare every visible "
    "curve at one progress point."
)

RELATIVE_HELP = (
    "Adjusted strength over the game, the scale the ratings use. At each "
    "progress point, every run's victory probability goes through the "
    "strength stage's adjustment against the mean VPAI self-play value on the "
    "same map and seat, so 0.5 means level with VPAI from the same start. A "
    "strategist can sit above 0.5 here with a low absolute probability when it "
    "drew hard seats. Only controlled games count, and winner enforcement is "
    "off so the final outcome does not lift a whole curve. Checkbox choices "
    "apply to both views."
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

        points, pool = self._load_points(ctx, estimator_id, by, where)
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

        grid = progress_grid()
        over_progress = pd.DataFrame(
            curve_records(interpolate_curves(points, grid, CURVE_KEYS), CURVE_KEYS),
            columns=CURVE_COLUMNS,
        ).sort_values([*CURVE_KEYS, "turn_progress"], kind="mergesort").reset_index(drop=True)
        relative, relative_note = _relative_table(ctx, points, pool, grid, table_id)

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

        chart = {
            "table": "over_progress",
            "vanilla_label": vanilla,
            "strategist_order": strategist_order,
            "condition_order": condition_order,
            "strategist_colors": colors,
            "grid_points": GRID_POINTS,
            "help": _CHART_HELP,
        }
        tables = {"by_identity": by_identity, "over_progress": over_progress}
        metadata = {
            "estimator": estimator_id,
            "strength_table": table_id,
            "by": by,
            "aggregate": agg,
            "games": int(points["game_id"].nunique()),
            "curve_chart": chart,
        }
        if relative is not None:
            tables["over_progress_relative"] = relative
            # The relative view comes first, as on the behavior pages.
            chart["views"] = [
                {"name": "relative", "label": "Relative",
                 "tip": "Adjusted strength against matched VPAI self-play",
                 "table": "over_progress_relative", "value": RELATIVE_VALUE,
                 "relative": True, "help": RELATIVE_HELP},
                {"name": "absolute", "label": "Absolute",
                 "tip": ABSOLUTE_TIP,
                 "table": "over_progress", "help": _CHART_HELP},
            ]
        else:
            metadata["relative_view"] = relative_note
        if baseline:
            metadata["baseline_experiment"] = baseline
        return AnalysisResult(
            tables=tables,
            summary=_summary(by_identity, by, agg, estimator_id, vanilla),
            metadata=metadata,
        )

    # ── input preparation ──────────────────────────────────────────────────────
    def _load_points(
        self, ctx: AnalysisContext, estimator_id: str, by: str, where: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filtered prediction points of one estimator, each with its identity,
        plus the unfiltered points that can serve as the matched VPAI baseline.

        Both carry the game's ``seed`` and ``seating_rotation`` (``-1`` when
        uncontrolled). The pool is unfiltered, as the strength stage's input
        is, so a player filter such as ``only_llm`` cannot remove the baseline.
        """
        panel = ctx.load_table("panel")
        if by not in panel.columns:
            raise AnalysisError(f"{where}: panel has no '{by}' column to group by.")
        # player_type and model let the strength adjustment find the VPAI seats.
        identity = [
            c for c in dict.fromkeys([by, "player_type", "model"]) if c in panel.columns
        ]
        panel = panel[["game_id", "player_id", *identity]].drop_duplicates(
            ["game_id", "player_id"]
        )
        panel["game_id"] = panel["game_id"].astype(str)
        pred = ctx.load_predictions(estimator_id)
        pred["game_id"] = pred["game_id"].astype(str)
        if "turn_progress" not in pred.columns and {"turn", "max_turn"} <= set(pred.columns):
            pred["turn_progress"] = pred["turn"] / pred["max_turn"]
        # Inner join: both inputs already exclude flagged problem games, so a
        # prediction row with no panel identity is a genuine gap, so drop it rather
        # than resurrect a fake ``Player <id>`` player_type that would pollute the
        # per-identity aggregation.
        joined = pred.merge(panel, on=["game_id", "player_id"], how="inner")
        df = ctx.apply_filter(joined)
        keep = tuple(dict.fromkeys(c for c in (*identity, "experiment") if c in df.columns))
        points = clean_curve_predictions(df, estimator_id, where, keep)
        if points.empty:
            raise AnalysisError(f"{where}: no rows after filtering.")
        pool = clean_curve_predictions(joined, estimator_id, where, keep)
        design = _design(ctx)
        return (
            points.merge(design, on="game_id", how="left").reset_index(drop=True),
            pool.merge(design, on="game_id", how="left").reset_index(drop=True),
        )

    def _baseline_experiment(self, ctx: AnalysisContext, table_id: str):
        value = _strength_params(ctx, table_id).get("baseline_experiment")
        return str(value) if value else None

    def _strategist_colors(
        self, ctx: AnalysisContext, strategist_order: list[str], vanilla_label: str
    ) -> dict[str, str]:
        from ...plotting.styles import get_player_color

        colors = {vanilla_label: get_player_color(ctx.catalog, vanilla_label)}
        for name in strategist_order:
            colors[name] = get_player_color(ctx.catalog, name)
        return colors


def _design(ctx: AnalysisContext) -> pd.DataFrame:
    """Each game's ``seed`` and ``seating_rotation``, ``-1`` when unknown."""
    games = ctx.load_table("games")
    out = pd.DataFrame({"game_id": games["game_id"].astype(str)})
    for column in ("seed", "seating_rotation"):
        values = games[column] if column in games.columns else pd.Series(-1, index=games.index)
        out[column] = pd.to_numeric(values, errors="coerce").fillna(-1).astype(int)
    return out.drop_duplicates("game_id")


def _relative_table(ctx: AnalysisContext, points: pd.DataFrame, pool: pd.DataFrame,
                    grid, table_id: str):
    """The controlled runs' mean adjusted-strength curves (:func:`adjusted_curves`).

    Returns ``(table, None)`` or ``(None, reason)``.
    """
    curves = adjusted_curves(
        points, pool, grid, CURVE_KEYS, ctx.catalog, _strength_params(ctx, table_id)
    )
    if curves is None:
        return None, (
            "no controlled games, or the strength stage's block setting turns the "
            "matched-cell adjustment off"
        )
    if not curves:
        return None, "no controlled run has a matched VPAI baseline"
    table = pd.DataFrame(
        curve_records(curves, CURVE_KEYS, RELATIVE_VALUE), columns=RELATIVE_COLUMNS,
    ).sort_values([*CURVE_KEYS, "turn_progress"], kind="mergesort").reset_index(drop=True)
    return table, None


def _strength_params(ctx: AnalysisContext, table_id: str) -> dict:
    stage = next((s for s in ctx.config.adjust if s.id == table_id), None)
    return dict((stage.raw.get("params") or {}) if stage is not None else {})


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
