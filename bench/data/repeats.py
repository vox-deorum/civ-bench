"""Controlled-slot helpers shared by the coverage analyses and ``civ-bench trim``.

A controlled experiment needs one game per ``(seed, seating_rotation)`` slot. When
reruns leave several games in one slot, the earliest by timestamp is the keep
candidate and the rest are extras. Keeping that rule in one place means the
``repeated_games`` table and the ``trim`` command always agree on which game stays.

A recorded rotation only names seats relative to the runner's ``basePerm``: for
rotation ``r``, the i-th sorted config slot sits at ``basePerm[(r + i) % N]``. A
seating file with a different ``basePerm`` (such as one generated from an older
civ-bench) makes the same rotation label mean different seats, so slots are read
from the seats a game was actually played in. :func:`align_rotations` infers each
experiment's ``basePerm`` from those seats and relabels every game by it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REPEATED_GAMES_COLUMNS = [
    "experiment", "seed", "seating_rotation", "n_games", "game_ids",
    "keep_candidate_game_id", "extra_game_ids",
]
OFF_CYCLE_GAMES_COLUMNS = [
    "experiment", "game_id", "seed", "recorded_rotation", "seats",
]


@dataclass(frozen=True)
class SeatingCycle:
    """One experiment's seat cycle, in the runner's terms.

    For rotation ``r``, the i-th entry of ``config_slots`` sits at
    ``base_perm[(r + i) % len(base_perm)]``.
    """

    config_slots: tuple[int, ...]
    base_perm: tuple[int, ...]

    @property
    def total_seats(self) -> int:
        return len(self.base_perm)

    def rotation_of(self, seats: dict[int, int]) -> int | None:
        """The rotation whose seats equal ``seats`` (config slot -> player id), or ``None``."""
        if not seats:
            return None
        n = self.total_seats
        slot, seat = next(iter(seats.items()))
        if seat not in self.base_perm or slot not in self.config_slots:
            return None
        rotation = (self.base_perm.index(seat) - self.config_slots.index(slot)) % n
        placed = self.placements(rotation, seats)
        if placed is None or any(self.base_perm[pos] != p for pos, p in placed.items()):
            return None
        return rotation

    def placements(self, rotation: int, seats: dict[int, int]) -> dict[int, int] | None:
        """``basePerm`` position -> seat implied by playing ``seats`` at ``rotation``."""
        n = self.total_seats
        if not all(s in self.config_slots and 0 <= p < n for s, p in seats.items()):
            return None
        return {(rotation + self.config_slots.index(s)) % n: p for s, p in seats.items()}


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


def _game_seats(seats: pd.DataFrame | None) -> dict[str, dict[int, int]]:
    """``game_id -> {config_slot: player_id}`` for the configured (treatment) seats.

    A game whose config slot maps to more than one player is not a runner seating,
    so it is left out and keeps its recorded rotation.
    """
    if seats is None or not {"game_id", "config_slot", "player_id"} <= set(seats.columns):
        return {}
    rows = seats[["game_id", "config_slot", "player_id"]].copy()
    rows["config_slot"] = pd.to_numeric(rows["config_slot"], errors="coerce")
    rows["player_id"] = pd.to_numeric(rows["player_id"], errors="coerce")
    rows = rows.dropna()
    rows = rows[rows["config_slot"] >= 0].drop_duplicates()
    invalid = rows.duplicated(["game_id", "config_slot"], keep=False)
    rows = rows[~rows["game_id"].isin(rows.loc[invalid, "game_id"])]
    out: dict[str, dict[int, int]] = {}
    for game_id, slot, player_id in rows.itertuples(index=False):
        out.setdefault(str(game_id), {})[int(slot)] = int(player_id)
    return out


def _fits(perm: dict[int, int], placed: dict[int, int]) -> bool:
    """True when ``placed`` extends the partial permutation ``perm`` without conflict."""
    position_of = {seat: pos for pos, seat in perm.items()}
    return all(
        perm.get(pos, seat) == seat and position_of.get(seat, pos) == pos
        for pos, seat in placed.items()
    )


def infer_cycle(
    config_slots: list[int], games: list[tuple[int, int, dict[int, int]]],
) -> SeatingCycle:
    """Reconstruct the ``basePerm`` that the most games were played under.

    ``games`` holds ``(seed, recorded_rotation, seats)`` earliest first. A candidate
    ``basePerm`` is built from games that agree at their recorded rotations: one per
    group, where each game joins the first group it agrees with, and one per game,
    taking that game and then each later and earlier game that agrees. Each candidate
    also gets a copy where every game left over joins at the first other rotation that
    agrees, since a game seated by another ``basePerm`` can still fit this one under
    a different label. Positions a
    candidate leaves unpinned get the remaining seats in ascending order; any
    completion is a valid cycle, because no game was played there. The completed
    ``basePerm`` under which all games fill the most distinct ``(seed, rotation)``
    slots wins (the earliest group on a tie), so the fewest games fall off the cycle
    and need a rerun.
    """
    slots = tuple(sorted(config_slots))
    n = max(slots) + 1
    probe = SeatingCycle(slots, tuple(range(n)))
    observed = [
        (rotation, seats) for _seed, rotation, seats in games
        if 0 <= rotation < n and probe.placements(rotation, seats)
    ]

    def join(perm: dict[int, int], order, other_rotations: bool) -> dict[int, int]:
        for rotation, seats in order:
            shifts = [s for s in range(n) if s != rotation] if other_rotations else [rotation]
            for shift in shifts:
                placed = probe.placements(shift, seats)
                if _fits(perm, placed):
                    perm.update(placed)
                    break
        return perm

    groups: list[dict[int, int]] = []
    for rotation, seats in observed:
        placed = probe.placements(rotation, seats)
        for perm in groups:
            if _fits(perm, placed):
                perm.update(placed)
                break
        else:
            groups.append(dict(placed))
    starts = [
        join({}, observed[i:] + observed[:i], other_rotations=False)
        for i in range(len(observed))
    ]
    groups += starts
    groups += [join(dict(perm), observed, other_rotations=True) for perm in groups]

    def complete(perm: dict[int, int]) -> SeatingCycle:
        free = iter(sorted(set(range(n)) - set(perm.values())))
        return SeatingCycle(slots, tuple(perm[pos] if pos in perm else next(free) for pos in range(n)))

    def filled(cycle: SeatingCycle) -> int:
        cells = {(seed, cycle.rotation_of(seats)) for seed, _rotation, seats in games}
        return len({cell for cell in cells if cell[1] is not None})

    candidates = [complete(perm) for perm in groups] or [probe]
    return max(candidates, key=filled)


def align_rotations(
    controlled: pd.DataFrame,
    seats: pd.DataFrame | None,
    games: pd.DataFrame | None,
    excluded_game_ids=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, SeatingCycle]]:
    """Relabel each game's ``seating_rotation`` by the seats it was played in.

    ``controlled`` comes from :func:`controlled_rows`; ``seats`` is any frame with
    ``game_id``, ``config_slot`` and ``player_id`` (the panel). Every game in
    ``controlled`` informs the cycle, since an excluded game was still seated by
    it, but only accepted games are returned. Returns ``(aligned, off_cycle, cycles)``:

    * ``aligned``: the accepted rows of ``controlled`` with ``seating_rotation`` set
      to the rotation of the game's seats under its experiment's inferred cycle,
      without off-cycle games;
    * ``off_cycle``: one row per game whose seats match no rotation of that cycle,
      in :data:`OFF_CYCLE_GAMES_COLUMNS`. Such a game unbalances the seats, so it
      fills no slot and ``trim`` removes it;
    * ``cycles``: experiment -> :class:`SeatingCycle`.

    A game without configured seats in ``seats`` keeps its recorded rotation.
    """
    excluded = {str(g) for g in (excluded_game_ids or ())}
    if controlled.empty:
        return controlled, pd.DataFrame(columns=OFF_CYCLE_GAMES_COLUMNS), {}
    game_seats = _game_seats(seats)
    labels = controlled[["experiment", "game_id", "seed", "seating_rotation"]].drop_duplicates("game_id")
    rotation_by_game: dict[str, int] = {}
    off_rows: list[dict] = []
    cycles: dict[str, SeatingCycle] = {}
    for experiment, exp_labels in labels.groupby("experiment", sort=True):
        by_id = {str(r.game_id): r for r in exp_labels.itertuples(index=False)}
        seated = [g for g in ordered_game_ids(list(by_id), games) if g in game_seats]
        slots = sorted({slot for g in seated for slot in game_seats[g]})
        if not slots:
            continue
        cycle = infer_cycle(slots, [
            (int(by_id[g].seed), int(by_id[g].seating_rotation), game_seats[g]) for g in seated
        ])
        cycles[str(experiment)] = cycle
        for game_id in (g for g in seated if g not in excluded):
            rotation = cycle.rotation_of(game_seats[game_id])
            if rotation is not None:
                rotation_by_game[game_id] = rotation
                continue
            label = by_id[game_id]
            off_rows.append({
                "experiment": experiment,
                "game_id": game_id,
                "seed": int(label.seed),
                "recorded_rotation": int(label.seating_rotation),
                "seats": ",".join(f"{s}:{p}" for s, p in sorted(game_seats[game_id].items())),
            })
    off_cycle = pd.DataFrame(off_rows, columns=OFF_CYCLE_GAMES_COLUMNS)
    dropped = excluded | set(off_cycle["game_id"])
    aligned = controlled[~controlled["game_id"].astype(str).isin(dropped)].copy()
    recorded = aligned["seating_rotation"]
    aligned["seating_rotation"] = (
        aligned["game_id"].astype(str).map(rotation_by_game).fillna(recorded).astype(int)
    )
    return aligned, off_cycle, cycles
