"""``performance.experiment_completeness``: controlled game-grid completeness.

Reports compact experiment-level completeness against the controlled
``seed × seating_rotation`` grid, actionable duplicate game ids for cleanup, and
failed decision-turn telemetry from token extraction.
"""

from __future__ import annotations

import pandas as pd

from bench.data.loading import drop_problem_games
from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .seating import SEATING_INDEX_COLUMNS, generate_seating_files
from .strength_panel import (
    _controlled_panel,
    _reference_slots,
    build_experiment_completeness,
)


DECISION_TURN_FAILURE_COLUMNS = [
    "experiment", "game_id", "player_id", "player_type", "valid_turn_count",
    "failed_turn_count", "failure_pct", "failed_turns",
]
CONDITION_PROGRESS_COLUMNS = [
    "experiment", "player_type", "completed_games", "required_games",
    "completed_at",
]


def build_condition_progress(
    panel: pd.DataFrame,
    games: pd.DataFrame | None,
    completeness: pd.DataFrame,
    baseline_experiment: str | None,
    excluded_game_ids,
    vanilla_label: str,
    null_label: str,
) -> pd.DataFrame:
    """Return controlled condition progress and completion timestamps.

    Game counts deliberately come from the experiment completeness table. A
    repeated game can occupy only one reference slot, so ``present_games`` is
    not a completion count. Completion time is experiment-wide: it uses the
    earliest valid accepted game in each reference slot, which keeps later
    reruns from moving it when player identities vary by slot.
    """
    controlled = _controlled_panel(panel)
    required_columns = {"experiment", "player_type"}
    if controlled.empty or not required_columns <= set(controlled.columns):
        return pd.DataFrame(columns=CONDITION_PROGRESS_COLUMNS)
    if completeness.empty or not {
        "experiment", "required_games", "missing_games",
    } <= set(completeness.columns):
        return pd.DataFrame(columns=CONDITION_PROGRESS_COLUMNS)

    reference_slots = _reference_slots(controlled, baseline_experiment)
    if not reference_slots:
        return pd.DataFrame(columns=CONDITION_PROGRESS_COLUMNS)

    timestamps: dict[str, pd.Timestamp] = {}
    if games is not None and {"game_id", "timestamp"} <= set(games.columns):
        time_rows = games[["game_id", "timestamp"]].copy()
        time_rows["game_id"] = time_rows["game_id"].astype(str)
        timestamp_ms = pd.to_numeric(time_rows["timestamp"], errors="coerce")
        timestamp_ms = timestamp_ms.where(
            timestamp_ms.between(-9_223_372_036_854, 9_223_372_036_854)
        )
        time_rows["timestamp"] = pd.to_datetime(
            timestamp_ms,
            unit="ms",
            utc=True,
            errors="coerce",
        )
        timestamps = (
            time_rows.dropna(subset=["timestamp"])
            .groupby("game_id", sort=False)["timestamp"]
            .min()
            .to_dict()
        )

    comp_by_experiment = completeness.set_index("experiment")
    excluded = {str(game_id) for game_id in (excluded_game_ids or ())}
    baseline_labels = {str(vanilla_label), str(null_label)}
    rows: list[dict] = []
    condition_rows = controlled[
        ~controlled["player_type"].astype(str).isin(baseline_labels)
    ].copy()

    for (experiment, player_type), _condition in condition_rows.groupby(
        ["experiment", "player_type"], sort=True
    ):
        if experiment not in comp_by_experiment.index:
            continue
        comp = comp_by_experiment.loc[experiment]
        required_games = int(comp["required_games"])
        completed_games = required_games - int(comp["missing_games"])
        completed_at = ""

        if completed_games == required_games:
            experiment_rows = controlled[controlled["experiment"] == experiment]
            accepted = drop_problem_games(experiment_rows, excluded)
            slot_times: list[pd.Timestamp] = []
            for seed, rotation in reference_slots:
                slot_ids = accepted.loc[
                    (accepted["seed"] == seed)
                    & (accepted["seating_rotation"] == rotation),
                    "game_id",
                ].astype(str).unique()
                valid_times = [timestamps[game_id] for game_id in slot_ids if game_id in timestamps]
                if not valid_times:
                    break
                slot_times.append(min(valid_times))
            if len(slot_times) == len(reference_slots):
                completed_at = max(slot_times).isoformat().replace("+00:00", "Z")

        rows.append({
            "experiment": experiment,
            "player_type": player_type,
            "completed_games": int(completed_games),
            "required_games": int(required_games),
            "completed_at": completed_at,
        })

    return pd.DataFrame(rows, columns=CONDITION_PROGRESS_COLUMNS)


def _failed_turn_list(values: pd.Series) -> str:
    turns: set[int] = set()
    for value in values.dropna().astype(str):
        for item in value.split(","):
            item = item.strip()
            if item:
                # CSV inference represents a lone turn as a float when other
                # traces have blank failed-turn lists.
                turns.add(int(float(item)))
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
    )
    source["failed_turn_count"] = pd.to_numeric(
        source["failed_turn_count"], errors="coerce"
    )
    source = source.dropna(subset=["valid_turn_count", "failed_turn_count"])
    source = source[source["valid_turn_count"] > 0]
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
        / per_player["valid_turn_count"].replace(0, float("nan"))
    ).astype(float).round(4)

    experiment = per_player.groupby("experiment", as_index=False).agg(
        failed_turn_count=("failed_turn_count", "sum"),
        avg_failure_count=("failed_turn_count", "mean"),
        valid_turn_count=("valid_turn_count", "sum"),
    )
    experiment["avg_failure_count"] = experiment["avg_failure_count"].round(4)
    experiment["failure_pct"] = (
        experiment["failed_turn_count"]
        / experiment["valid_turn_count"].replace(0, float("nan"))
    ).astype(float).round(4)
    experiment = experiment.drop(columns="valid_turn_count")

    out = out.merge(experiment, on="experiment", how="left")
    for index, row in out.iterrows():
        failed = row["failed_turn_count"]
        if pd.isna(failed) or pd.isna(row["failure_pct"]):
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
        strength = ctx.load_table(table_id)
        games = self._load_games(ctx)
        baseline_experiment = ctx.strength_provenance(table_id, strength).get(
            "adjust_baseline_experiment"
        )
        # Canonical rows retain the design and configured seats even when quality
        # filters removed an entire experiment upstream of the strength panel.
        try:
            panel = ctx.load_table("panel")
        except AnalysisError:
            panel = strength
        if games is not None and {"game_id", "seed", "seating_rotation"} <= set(games.columns):
            panel = panel.drop(columns=["seed", "seating_rotation"], errors="ignore").merge(
                games[["game_id", "seed", "seating_rotation"]].drop_duplicates("game_id"),
                on="game_id", how="left", validate="many_to_one",
            )
        panel = ctx.apply_filter(panel, include_quality=False)
        excluded_ids = ctx.decision_failure_ids()

        tables = build_experiment_completeness(panel, games, baseline_experiment, excluded_ids)
        if not tables:
            return AnalysisResult(
                summary="No controlled experiment coverage is available.",
                metadata={"controlled_rows_available": False},
            )
        tables["condition_progress"] = build_condition_progress(
            panel,
            games,
            tables["experiment_completeness"],
            baseline_experiment,
            excluded_ids,
            ctx.catalog.vanilla_label,
            ctx.catalog.null_label,
        )

        tokens = self._load_tokens(ctx)
        if tokens is not None:
            tokens = tokens[tokens["game_id"].isin(panel["game_id"])]
        tables["experiment_completeness"], decision_failures = (
            attach_decision_turn_failures(tables["experiment_completeness"], tokens)
        )
        tables["decision_turn_failures"] = decision_failures
        decision_failures["excluded_game"] = decision_failures["game_id"].astype(str).isin(excluded_ids)

        out = {}
        for name, table in tables.items():
            if name in {
                "experiment_completeness", "repeated_games", "decision_turn_failures",
                "condition_progress",
            } or not table.empty:
                out[name] = table

        comp = tables["experiment_completeness"]
        required_games = int(comp["required_games"].sum())
        present_games = int(comp["present_games"].sum())
        missing_games = int(comp["missing_games"].sum())
        repeated_slots = int(comp["repeated_slots"].sum())
        warning_experiments = int((comp["warning"].fillna("ok").astype(str) != "ok").sum())
        failed_turns = int(comp["failed_turn_count"].fillna(0).sum())
        complete_experiments = int((comp["missing_games"].fillna(0) == 0).sum())
        n_experiments = int(len(comp))
        coverage_pct = (100.0 * present_games / required_games) if required_games else 0.0
        summary = (
            f"**{present_games}/{required_games}** planned games "
            f"(**{coverage_pct:.1f}%**) are present across **{n_experiments}** experiment(s). "
            f"**{complete_experiments}/{n_experiments}** experiment(s) have every planned game."
        )

        artifacts: dict[str, str] = {}
        if bool(self.params.get("emit_seating", True)):
            artifacts, index_rows, warnings = generate_seating_files(
                panel, games, baseline_experiment, excluded_ids,
            )
            if index_rows:
                out["seating_index"] = pd.DataFrame(index_rows, columns=SEATING_INDEX_COLUMNS)
                open_total = int(sum(r["open_cells"] for r in index_rows))
            else:
                index_rows = []
                open_total = 0
        else:
            warnings = []
            index_rows = []
            open_total = 0

        return AnalysisResult(
            tables=out,
            artifacts=artifacts,
            summary=summary,
            metadata={
                "strength_table": table_id,
                "coverage": {
                    "missing_slots": missing_games,
                    "repeated_slots": repeated_slots,
                    "failed_decision_turns": failed_turns,
                    "excluded_games": int(comp["excluded_games"].sum()),
                    "experiments_with_warnings": warning_experiments,
                },
                "seating": {
                    "files_generated": int(len(index_rows)),
                    "open_cells": int(open_total),
                    "warnings": warnings,
                },
            },
        )

    def _load_games(self, ctx: AnalysisContext):
        try:
            return ctx.load_table("games")
        except AnalysisError:
            return None

    def _load_tokens(self, ctx: AnalysisContext):
        try:
            # Every player trace can reject the game, including players omitted
            # from the analysis selection. Keep those traces in the explanation.
            return ctx.load_table("tokens")
        except AnalysisError:
            return None
