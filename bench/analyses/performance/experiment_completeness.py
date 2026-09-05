"""``performance.experiment_completeness``: controlled game-grid completeness.

Reports compact experiment-level completeness against the controlled
``seed × seating_rotation`` grid, actionable duplicate game ids for cleanup, and
failed decision-turn telemetry from token extraction.
"""

from __future__ import annotations

import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .seating import SEATING_INDEX_COLUMNS, generate_seating_files
from .strength_panel import build_experiment_completeness


DECISION_TURN_FAILURE_COLUMNS = [
    "experiment", "game_id", "player_id", "player_type", "valid_turn_count",
    "failed_turn_count", "failure_pct", "failed_turns",
]


def _failed_turn_list(values: pd.Series) -> str:
    turns: set[int] = set()
    for value in values.dropna().astype(str):
        for item in value.split(","):
            item = item.strip()
            if item:
                turns.add(int(item))
    return ",".join(str(turn) for turn in sorted(turns))


def attach_decision_turn_failures(
    completeness: pd.DataFrame,
    tokens: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add experiment failure rates and return actionable failed player traces.

    Token rows are per player and model, so a player trace that used multiple
    models repeats its turn counters. Collapse those rows before aggregating to
    avoid counting the same failed turn more than once.
    """
    out = completeness.copy()
    required = {
        "experiment", "game_id", "player_id", "valid_turn_count",
        "failed_turn_count",
    }
    if tokens is None or tokens.empty or not required <= set(tokens.columns):
        out["failed_turn_count"] = float("nan")
        out["avg_failure_count"] = float("nan")
        out["failure_pct"] = float("nan")
        out["warning"] = out["warning"].map(
            lambda warning: _append_warning(
                warning, "decision-turn failure telemetry unavailable"
            )
        )
        return out, pd.DataFrame(columns=DECISION_TURN_FAILURE_COLUMNS)

    source = tokens.copy()
    source["valid_turn_count"] = pd.to_numeric(
        source["valid_turn_count"], errors="coerce"
    ).fillna(0).astype(int)
    source["failed_turn_count"] = pd.to_numeric(
        source["failed_turn_count"], errors="coerce"
    ).fillna(0).astype(int)
    if "player_type" not in source.columns:
        source["player_type"] = "N/A"
    if "failed_turns" not in source.columns:
        source["failed_turns"] = ""

    per_player = source.groupby(
        ["experiment", "game_id", "player_id"], as_index=False, dropna=False
    ).agg(
        player_type=("player_type", "first"),
        valid_turn_count=("valid_turn_count", "max"),
        failed_turn_count=("failed_turn_count", "max"),
        failed_turns=("failed_turns", _failed_turn_list),
    )
    per_player["failure_pct"] = (
        per_player["failed_turn_count"]
        / per_player["valid_turn_count"].replace(0, pd.NA)
    ).astype(float).round(4)

    experiment = per_player.groupby("experiment", as_index=False).agg(
        failed_turn_count=("failed_turn_count", "sum"),
        avg_failure_count=("failed_turn_count", "mean"),
        valid_turn_count=("valid_turn_count", "sum"),
    )
    experiment["avg_failure_count"] = experiment["avg_failure_count"].round(4)
    experiment["failure_pct"] = (
        experiment["failed_turn_count"]
        / experiment["valid_turn_count"].replace(0, pd.NA)
    ).astype(float).round(4)
    experiment = experiment.drop(columns="valid_turn_count")

    out = out.merge(experiment, on="experiment", how="left")
    for index, row in out.iterrows():
        failed = row["failed_turn_count"]
        if pd.isna(failed):
            out.at[index, "warning"] = _append_warning(
                row["warning"], "decision-turn failure telemetry unavailable"
            )
        elif int(failed) > 0:
            out.at[index, "warning"] = _append_warning(
                row["warning"], f"{int(failed)} failed decision turn(s)"
            )

    details = per_player[per_player["failed_turn_count"] > 0].copy()
    details = details[DECISION_TURN_FAILURE_COLUMNS].sort_values(
        ["experiment", "game_id", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    summary_columns = [
        column for column in out.columns if column != "warning"
    ] + ["warning"]
    return out[summary_columns], details


def _append_warning(existing: object, warning: str) -> str:
    text = str(existing) if pd.notna(existing) else ""
    return warning if not text or text == "ok" else f"{text}; {warning}"


class PerformanceExperimentCompleteness(Analysis):
    module = "performance.experiment_completeness"
    friendly_name = "Experiment coverage"
    description = (
        "Reports completed, missing, and repeated games across the planned map, "
        "seat, and condition combinations, including decision-turn failures."
    )
    report_defaults = {
        "tables": ["experiment_completeness", "decision_turn_failures"],
        "figures": [],
    }

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

        tokens = self._load_tokens(ctx)
        tables["experiment_completeness"], decision_failures = (
            attach_decision_turn_failures(tables["experiment_completeness"], tokens)
        )
        tables["decision_turn_failures"] = decision_failures

        out = {}
        for name, table in tables.items():
            if name in {
                "experiment_completeness", "repeated_games", "decision_turn_failures"
            } or not table.empty:
                out[name] = table

        comp = tables["experiment_completeness"]
        required_games = int(comp["required_games"].sum())
        present_games = int(comp["present_games"].sum())
        missing_games = int(comp["missing_games"].sum())
        repeated_slots = int(comp["repeated_slots"].sum())
        warning_experiments = int((comp["warning"].fillna("ok").astype(str) != "ok").sum())
        failed_turns = int(comp["failed_turn_count"].fillna(0).sum())
        summary = (
            f"The run contains {present_games} of {required_games} planned games, "
            f"with {missing_games} missing slot(s), {repeated_slots} repeated slot(s), "
            f"{failed_turns} failed decision turn(s), and "
            f"{warning_experiments} experiment(s) with warnings"
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

    def _load_tokens(self, ctx: AnalysisContext):
        try:
            return ctx.apply_filter(ctx.load_table("tokens"))
        except AnalysisError:
            return None
