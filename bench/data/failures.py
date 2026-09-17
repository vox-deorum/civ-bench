"""Decision failure rates from canonical per-player token telemetry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bench.config.filters import DEFAULT_MAX_DECISION_FAILURE_PCT


def decision_failure_games(tokens: pd.DataFrame | None) -> pd.DataFrame:
    """Count each player trace once and record the worst failure rate per game.

    Model fallback rows repeat a trace's counters. Missing counters and traces
    with no selected turn roots do not establish a failure rate.
    """
    columns = [
        "experiment", "game_id", "valid_turn_count", "failed_turn_count",
        "failure_pct", "max_player_failure_pct",
    ]
    required = set(columns[:4]) | {"player_id"}
    if tokens is None or tokens.empty or not required <= set(tokens.columns):
        return pd.DataFrame(columns=columns)
    source = tokens.copy()
    for column in ("valid_turn_count", "failed_turn_count"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=["valid_turn_count", "failed_turn_count"])
    source = source[source["valid_turn_count"] > 0]
    players = source.groupby(
        ["experiment", "game_id", "player_id"], as_index=False, dropna=False,
    )[["valid_turn_count", "failed_turn_count"]].max()
    players["failure_pct"] = players["failed_turn_count"] / players["valid_turn_count"]
    games = players.groupby(["experiment", "game_id"], as_index=False, dropna=False).agg(
        valid_turn_count=("valid_turn_count", "sum"),
        failed_turn_count=("failed_turn_count", "sum"),
        max_player_failure_pct=("failure_pct", "max"),
    )
    games["failure_pct"] = games["failed_turn_count"] / games["valid_turn_count"]
    return games[columns]


def failed_game_ids(tokens: pd.DataFrame | None, resolved_filter: dict) -> set[str]:
    """Games at or above the configured cutoff, using unrounded rates."""
    threshold = resolved_filter.get("max_decision_failure_pct", DEFAULT_MAX_DECISION_FAILURE_PCT)
    if threshold is None:
        return set()
    games = decision_failure_games(tokens)
    return set(games.loc[games["max_player_failure_pct"] >= threshold, "game_id"].astype(str))


def failed_game_ids_from_tokens(tokens_path, resolved_filter: dict) -> set[str]:
    """Read available telemetry; unknown failure rates do not exclude games."""
    if resolved_filter.get("max_decision_failure_pct", DEFAULT_MAX_DECISION_FAILURE_PCT) is None:
        return set()
    if not tokens_path or not Path(tokens_path).exists():
        return set()
    try:
        tokens = pd.read_csv(tokens_path)
    except pd.errors.EmptyDataError:
        return set()
    return failed_game_ids(tokens, resolved_filter)
