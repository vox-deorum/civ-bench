"""``behavior.profiles``: a configurable fingerprint of behavior metrics.

For the ``behavior`` columns named in ``params.metrics``, shows each player
type's mean with a game-level bootstrap CI, relative to the matched in-game AI
(default view) and in absolute terms. Themes such as military or economic
profiles are separate stages with different metric lists.
"""

from __future__ import annotations

from ..base import AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from . import common as C
from .base import BehaviorAnalysis


class BehaviorProfiles(BehaviorAnalysis):
    module = "behavior.profiles"
    friendly_name = "Behavior profiles"
    description = (
        "Compares how each player type behaves on a chosen set of metrics, against "
        "the in-game AI on the same map and seat and in absolute terms."
    )
    report_defaults = {
        "tables": [],
        "figures": ["profile_relative", "profile_absolute"],
    }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        metrics = list(self.params.get("metrics") or [])
        if not metrics:
            raise AnalysisError(f"behavior.profiles '{self.stage_id}': params.metrics is required.")
        rows, pool, rate_scaled, baseline = self.load_behavior(ctx, metrics)
        if rows.empty:
            return AnalysisResult(summary="No behavior rows remain after filtering.")
        views = C.build_views(rows, pool, metrics, C.controlled_cells(ctx), baseline)
        labels = {m: C.metric_label(m, rate_scaled) for m in metrics}
        out = self.metric_views(ctx, views, "profile", "Behavior profile", labels)
        metadata = {
            **views.metadata(),
            "metrics": metrics,
            "rate": self.rate,
            "views": C.views_metadata(out.declared),
        }
        return AnalysisResult(
            tables=out.tables, figures=out.figures,
            summary=self.headline(out, views, "profile", labels), metadata=metadata,
        )
