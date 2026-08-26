"""``performance.experiment_completeness`` — controlled game-grid completeness.

Reports compact experiment-level completeness against the controlled
``seed × seating_rotation`` grid, plus actionable duplicate game ids for cleanup.
"""

from __future__ import annotations

import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .seating import SEATING_INDEX_COLUMNS, generate_seating_files
from .strength_panel import build_experiment_completeness


class PerformanceExperimentCompleteness(Analysis):
    module = "performance.experiment_completeness"
    friendly_name = "Experiment coverage"
    description = (
        "Reports completed, missing, and repeated games across the planned map, "
        "seat, and condition combinations."
    )
    report_defaults = {"tables": ["experiment_completeness"], "figures": []}

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        table_id = ctx.strength_table_id()
        panel = ctx.apply_filter(ctx.load_table(table_id))
        games = self._load_games(ctx)
        baseline_experiment = ctx.strength_provenance(table_id, panel).get(
            "adjust_baseline_experiment"
        )

        tables = build_experiment_completeness(panel, games, baseline_experiment)
        if not tables:
            return AnalysisResult(
                summary="Experiment coverage is unavailable because the run has no controlled experiment rows.",
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
            f"The run contains {present_games} of {required_games} planned games, "
            f"with {missing_games} missing slot(s), {repeated_slots} repeated slot(s), "
            f"and {warning_experiments} experiment(s) with warnings"
        )

        artifacts: dict[str, str] = {}
        if bool(self.params.get("emit_seating", True)):
            artifacts, index_rows, warnings = generate_seating_files(
                panel, games, baseline_experiment
            )
            if index_rows:
                out["seating_index"] = pd.DataFrame(index_rows, columns=SEATING_INDEX_COLUMNS)
                open_total = int(sum(r["open_cells"] for r in index_rows))
                summary += (
                    f"; generated {len(index_rows)} seating.json file(s) with "
                    f"{open_total} open cell(s)"
                )
            if warnings:
                notes = "; ".join(
                    note.rstrip(".").replace(f" {chr(8212)} ", ", ").replace(". ", ", ")
                    for note in warnings
                )
                summary += "; seating notes: " + notes
        summary += "."

        return AnalysisResult(
            tables=out,
            artifacts=artifacts,
            summary=summary,
            metadata={"strength_table": table_id},
        )

    def _load_games(self, ctx: AnalysisContext):
        try:
            return ctx.load_table("games")
        except AnalysisError:
            return None
