"""``performance.experiment_completeness`` — controlled game-grid completeness.

Reports compact experiment-level completeness against the controlled
``seed × seating_rotation`` grid, plus actionable duplicate game ids for cleanup.
"""

from __future__ import annotations

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .strength_panel import build_experiment_completeness
from .turn_predicted import _strength_table_id


class PerformanceExperimentCompleteness(Analysis):
    module = "performance.experiment_completeness"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        table_id = _strength_table_id(ctx)
        panel = ctx.apply_filter(ctx.load_table(table_id))
        games = self._load_games(ctx)
        baseline_experiment = ctx.strength_provenance(table_id, panel).get(
            "adjust_baseline_experiment"
        )

        tables = build_experiment_completeness(panel, games, baseline_experiment)
        if not tables:
            return AnalysisResult(
                summary="No controlled experiment rows available for completeness reporting.",
            )

        out = {}
        for name, table in tables.items():
            if name in {"experiment_completeness", "repeated_games"} or not table.empty:
                out[name] = table

        comp = tables["experiment_completeness"]
        required_games = int(comp["required_games"].sum())
        present_games = int(comp["present_games"].sum())
        missing_games = int(comp["missing_games"].sum())
        repeated_slots = int(comp["repeated_slots"].sum())
        warning_experiments = int((comp["warning"].fillna("ok").astype(str) != "ok").sum())
        summary = (
            f"Experiment completeness: {present_games}/{required_games} games present, "
            f"{missing_games} missing slots, {repeated_slots} repeated slots, "
            f"{warning_experiments} experiment(s) with warnings."
        )
        return AnalysisResult(
            tables=out,
            summary=summary,
            metadata={"strength_table": table_id},
        )

    def _load_games(self, ctx: AnalysisContext):
        try:
            return ctx.load_table("games")
        except AnalysisError:
            return None
