"""Controlled-slot helpers shared by the coverage analyses and ``civ-bench trim``.

A controlled experiment needs one game per ``(seed, seating_rotation)`` slot. When
reruns leave several games in one slot, the earliest by timestamp is the keep
candidate and the rest are extras. Keeping that rule in one place means the
``repeated_games`` table and the ``trim`` command always agree on which game stays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REPEATED_GAMES_COLUMNS = [
    "experiment", "seed", "seating_rotation", "n_games", "game_ids",
    "keep_candidate_game_id", "extra_game_ids",
]


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def controlled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of controlled games, with slot columns cast to their canonical types.

    Works on the per-player panel and on the one-row-per-game ``games`` table;
    ``player_id`` is cast only when present.
    """
    required = {"experiment", "game_id", "seed", "seating_rotation"}
    if not required <= set(frame.columns):
        return pd.DataFrame(columns=list(frame.columns))
    if "controlled" in frame.columns:
        mask = _truthy(frame["controlled"])
    else:
        mask = (frame["seed"].fillna(-1).astype(int) != -1) & (
            frame["seating_rotation"].fillna(-1).astype(int) != -1
        )
    out = frame[mask].copy()
    if out.empty:
        return out
    out["seed"] = out["seed"].astype(int)
    out["seating_rotation"] = out["seating_rotation"].astype(int)
    if "player_id" in out.columns:
        out["player_id"] = out["player_id"].astype(int)
    out["game_id"] = out["game_id"].astype(str)
    out["experiment"] = out["experiment"].astype(str)
    return out


def _game_sort_frame(games: pd.DataFrame | None) -> pd.DataFrame:
    if games is None or games.empty or "game_id" not in games.columns:
        return pd.DataFrame(columns=["game_id", "_timestamp_num", "_timestamp_str"])
    cols = ["game_id"] + (["timestamp"] if "timestamp" in games.columns else [])
    out = games[cols].drop_duplicates("game_id").copy()
    out["game_id"] = out["game_id"].astype(str)
    if "timestamp" in out.columns:
        out["_timestamp_num"] = pd.to_numeric(out["timestamp"], errors="coerce")
        out["_timestamp_str"] = out["timestamp"].astype(str)
    else:
        out["_timestamp_num"] = np.nan
        out["_timestamp_str"] = ""
    return out[["game_id", "_timestamp_num", "_timestamp_str"]]


def ordered_game_ids(ids: list[str], games: pd.DataFrame | None) -> list[str]:
    """Order game ids earliest first; ids without a timestamp go last, then by id."""
    base = pd.DataFrame({"game_id": sorted(set(map(str, ids)))})
    if base.empty:
        return []
    order = base.merge(_game_sort_frame(games), on="game_id", how="left")
    order["_timestamp_missing"] = order["_timestamp_num"].isna()
    order["_timestamp_str"] = order["_timestamp_str"].fillna("")
    order = order.sort_values(
        ["_timestamp_missing", "_timestamp_num", "_timestamp_str", "game_id"],
        kind="mergesort",
    )
    return order["game_id"].tolist()


def repeated_slots(controlled: pd.DataFrame, games: pd.DataFrame | None) -> list[dict]:
    """One row per slot holding more than one game, in ``REPEATED_GAMES_COLUMNS``.

    ``controlled`` comes from :func:`controlled_rows`, with excluded games already
    dropped, so an excluded game is never the keep candidate.
    """
    rows: list[dict] = []
    if controlled.empty:
        return rows
    slots = controlled.groupby(["experiment", "seed", "seating_rotation"], sort=True)["game_id"]
    for (experiment, seed, rot), ids in slots:
        ordered = ordered_game_ids(ids.dropna().astype(str).unique().tolist(), games)
        if len(ordered) > 1:
            rows.append({
                "experiment": experiment,
                "seed": int(seed),
                "seating_rotation": int(rot),
                "n_games": int(len(ordered)),
                "game_ids": ",".join(ordered),
                "keep_candidate_game_id": ordered[0],
                "extra_game_ids": ",".join(ordered[1:]),
            })
    return rows
