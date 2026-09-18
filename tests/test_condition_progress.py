"""Synthetic coverage for saved condition-progress data."""

from __future__ import annotations

import pandas as pd

from bench.analyses.performance.experiment_completeness import (
    build_condition_progress,
)
from bench.analyses.performance.strength_panel import build_experiment_completeness


def _rows(experiment: str, game_id: str, seed: int, rotation: int, player_types: list[str]):
    return [
        {
            "experiment": experiment,
            "game_id": game_id,
            "seed": seed,
            "seating_rotation": rotation,
            "player_id": index,
            "player_type": player_type,
            "controlled": True,
        }
        for index, player_type in enumerate(player_types)
    ]


def test_condition_progress_uses_slots_accepted_games_and_valid_dates():
    rows = []
    rows += _rows("baseline", "base-0", 1, 0, ["Vanilla"])
    rows += _rows("baseline", "base-1", 1, 1, ["Vanilla"])
    rows += _rows("complete", "complete-old", 1, 0, ["Vanilla", "GPT-OSS (fast)"])
    rows += _rows("complete", "complete-rerun", 1, 0, ["Vanilla", "GPT-OSS (fast)"])
    rows += _rows("complete", "complete-rejected", 1, 0, ["Vanilla", "GPT-OSS (fast)"])
    rows += _rows("complete", "complete-1", 1, 1, ["Vanilla", "GPT-OSS (fast)"])
    rows += _rows("unknown-date", "unknown-0", 1, 0, ["Vanilla", "Kimi (reasoning)"])
    rows += _rows("unknown-date", "unknown-1", 1, 1, ["Vanilla", "Kimi (reasoning)"])
    rows += _rows("partial", "partial-0", 1, 0, ["Vanilla", "Claude (brief)"])
    panel = pd.DataFrame(rows)
    games = pd.DataFrame([
        {"game_id": "base-0", "timestamp": 100},
        {"game_id": "base-1", "timestamp": 200},
        {"game_id": "complete-old", "timestamp": 1_000},
        {"game_id": "complete-rerun", "timestamp": 9_000},
        {"game_id": "complete-rejected", "timestamp": 500},
        {"game_id": "complete-1", "timestamp": 3_000},
        {"game_id": "unknown-0", "timestamp": 4_000},
        {"game_id": "unknown-1", "timestamp": "999999999999999999999"},
        {"game_id": "partial-0", "timestamp": 5_000},
    ])
    completeness = build_experiment_completeness(
        panel,
        games,
        baseline_experiment="baseline",
        excluded_game_ids={"complete-rejected"},
    )["experiment_completeness"]

    progress = build_condition_progress(
        panel,
        games,
        completeness,
        "baseline",
        {"complete-rejected"},
        "Vanilla",
        "Null",
    ).set_index(["experiment", "player_type"])

    assert list(progress.columns) == ["completed_games", "required_games", "completed_at"]
    assert ("complete", "GPT-OSS (fast)") in progress.index
    assert ("complete", "Vanilla") not in progress.index
    complete = progress.loc[("complete", "GPT-OSS (fast)")]
    assert complete["completed_games"] == 2
    assert complete["required_games"] == 2
    assert complete["completed_at"] == "1970-01-01T00:00:03Z"
    assert progress.loc[("unknown-date", "Kimi (reasoning)"), "completed_at"] == ""
    partial = progress.loc[("partial", "Claude (brief)")]
    assert partial["completed_games"] == 1
    assert partial["required_games"] == 2
    assert partial["completed_at"] == ""


def test_condition_progress_shares_experiment_time_and_keeps_zero_progress_rows():
    rows = []
    rows += _rows("baseline", "base-0", 1, 0, ["Custom VPAI"])
    rows += _rows("baseline", "base-1", 1, 1, ["Custom VPAI"])
    rows += _rows("mixed", "mixed-0", 1, 0, ["Custom VPAI", "Alpha (fast)"])
    rows += _rows("mixed", "mixed-1", 1, 1, ["No Agent", "Beta (slow)"])
    rows += _rows("zero", "zero-0", 1, 0, ["Custom VPAI", "Gamma (trial)"])
    rows += _rows("zero", "zero-1", 1, 1, ["No Agent", "Gamma (trial)"])
    panel = pd.DataFrame(rows)
    games = pd.DataFrame([
        {"game_id": "base-0", "timestamp": 100},
        {"game_id": "base-1", "timestamp": 200},
        {"game_id": "mixed-0", "timestamp": 1_000},
        {"game_id": "mixed-1", "timestamp": 4_000},
        {"game_id": "zero-0", "timestamp": 5_000},
        {"game_id": "zero-1", "timestamp": 6_000},
    ])
    excluded = {"zero-0", "zero-1"}
    completeness = build_experiment_completeness(
        panel,
        games,
        baseline_experiment="baseline",
        excluded_game_ids=excluded,
    )["experiment_completeness"]

    progress = build_condition_progress(
        panel,
        games,
        completeness,
        "baseline",
        excluded,
        "Custom VPAI",
        "No Agent",
    ).set_index(["experiment", "player_type"])

    assert ("mixed", "Custom VPAI") not in progress.index
    assert ("mixed", "No Agent") not in progress.index
    assert progress.loc[("mixed", "Alpha (fast)"), "completed_at"] == "1970-01-01T00:00:04Z"
    assert progress.loc[("mixed", "Beta (slow)"), "completed_at"] == "1970-01-01T00:00:04Z"
    zero = progress.loc[("zero", "Gamma (trial)")]
    assert zero["completed_games"] == 0
    assert zero["required_games"] == 2
    assert zero["completed_at"] == ""
