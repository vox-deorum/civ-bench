"""``behavior.flavors``: how strategists set the in-game AI's flavors.

A flavor is a 0 to 100 preference the strategist hands the in-game AI (50 is
balanced, the AI's own setting; the Null strategist never moves off it). From
the behavior table's ``flavor_<name>_avg`` (each player's turn-weighted average
over the turns it held a setting), per player type:

* **relative** (default): the player's average minus the mean of every
  completed experiment's strategists at the same ``(seed, player_id)`` cell.
  The in-game AI records no flavors, so the field average is the reference;
  completed experiments are part of their own baseline.
* **absolute**: the average setting, colored on the fixed 0 to 100 scale.

Both views are HTML heatmaps: one column per flavor under its short name, in
four groups, with the full name and the tool's description in the header
tooltip (:data:`bench.config.schema.FLAVOR_INFO`).
"""

from __future__ import annotations

from ...config import schema as S
from ...config.behavior import snake_case
from ..base import AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from . import common as C
from .base import BehaviorAnalysis, MetricViews

FLAVOR_RANGE = (0.0, 100.0)
BALANCED = 50


def flavor_column(name: str, stat: str = "avg") -> str:
    return f"flavor_{snake_case(name)}_{stat}"


class BehaviorFlavors(BehaviorAnalysis):
    module = "behavior.flavors"
    friendly_name = "Flavor settings"
    description = (
        "Shows how each strategist sets the in-game AI's flavors (0 to 100, 50 is "
        "balanced), against the average of completed experiments on the same map "
        "and seat and in absolute terms."
    )
    default_baseline = C.COMPLETED
    report_defaults = {
        "tables": ["flavors_relative", "flavors_absolute"],
        "figures": [],
    }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        requested = list(self.params.get("flavors") or S.FLAVOR_INFO)
        unknown = [f for f in requested if f not in S.FLAVOR_INFO]
        if unknown:
            raise AnalysisError(f"behavior.flavors '{self.stage_id}': unknown flavor(s) {unknown}.")
        # Group order, so each group's columns sit together under one header.
        requested = [f for f in S.FLAVOR_INFO if f in set(requested)]
        stats = {flavor_column(f, stat) for f in requested for stat in ("min", "max")}
        rows, pool, _rate_scaled, baseline = self.load_behavior(
            ctx, [flavor_column(f) for f in requested] + sorted(stats), allow_missing=True,
        )
        flavors = [f for f in requested if flavor_column(f) in rows.columns]
        if not flavors:
            raise AnalysisError(
                f"behavior.flavors '{self.stage_id}': the behavior table has no flavor "
                "averages; re-run extract with data.extract.behavior.flavor selected and "
                "'avg' in data.extract.behavior.stats."
            )
        if rows.empty:
            return AnalysisResult(summary="No behavior rows remain after filtering.")

        metrics = [flavor_column(f) for f in flavors]
        views = C.build_views(rows, pool, metrics, C.controlled_cells(ctx), baseline)
        names = {flavor_column(f): f for f in flavors}
        ranges = {
            flavor_column(f): (flavor_column(f, "min"), flavor_column(f, "max"))
            for f in flavors
            if flavor_column(f, "min") in rows.columns and flavor_column(f, "max") in rows.columns
        }
        baseline_label = baseline.label if baseline is not None else "baseline"
        out = MetricViews()
        self.add_heatmap_views(
            ctx, views, out, "flavors",
            titles={
                C.RELATIVE: f"Flavor setting against the {baseline_label}",
                C.ABSOLUTE: "Average flavor setting",
            },
            help_texts={
                C.RELATIVE: (
                    "Each cell is the mean, over players, of the player's turn-weighted "
                    f"average flavor minus the {baseline_label} at the same map and seat. "
                    "Colors run from 2 SDs of the baseline players below (red) to 2 SDs "
                    "above (blue). Completed experiments are part of their own baseline."
                ),
                C.ABSOLUTE: (
                    "Each cell is the mean, over players, of the player's turn-weighted "
                    "average flavor while it held a setting. Colors are fixed from 0 "
                    "(red) through 50, balanced and the in-game AI's own value (yellow), "
                    "to 100 (blue)."
                ),
            },
            names=names,
            short_names={flavor_column(f): S.FLAVOR_INFO[f][0] for f in flavors},
            column_tips={flavor_column(f): f"{f}: {S.FLAVOR_INFO[f][2]}" for f in flavors},
            groups={flavor_column(f): S.FLAVOR_INFO[f][1] for f in flavors},
            fixed={m: FLAVOR_RANGE for m in metrics},
            legends={
                C.RELATIVE: [[0.0, "2 SD below"], [0.25, "1 SD below"], [0.5, "same as the baseline"],
                             [0.75, "1 SD above"], [1.0, "2 SD above"]],
                C.ABSOLUTE: [[0.0, "0 forbid"], [0.3, "30 enough"], [0.5, "50 balanced (in-game AI)"],
                             [0.7, "70 prioritize"], [1.0, "100 emergency focus"]],
            },
            value_labels={C.RELATIVE: f"vs the {baseline_label}", C.ABSOLUTE: "average setting"},
            decimals=0,
            row_tips={ctx.catalog.null_label: (
                f"Never changes flavors, so it holds the in-game AI's balanced value of {BALANCED}."
            )},
            ranges=ranges,
            range_label="In-game range (mean of each player's min and max)",
        )
        metadata = {
            **views.metadata(),
            "flavors": flavors,
            "views": C.views_metadata(out.declared, baseline),
            "heatmaps": out.heatmaps,
        }
        missing = [f for f in requested if f not in flavors]
        if missing:
            metadata["flavors_not_extracted"] = missing
        return AnalysisResult(
            tables=out.tables,
            summary=self.heatmap_headline(ctx, out, views, "flavors", names, decimals=0),
            metadata=metadata,
        )
