"""Per-game facts shared by every behavior family, read once per DB."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from ...config import schema as S
from ...config.behavior import snake_case
from ..utilities import is_schema_mismatch

__all__ = ["GameContext", "build_game_context", "snake_case"]


@dataclass
class GameContext:
    major_players: list
    # player_id → team id (a major's team usually equals its player id).
    team_by_player: dict = field(default_factory=dict)
    # player_id → last turn with a PlayerSummaries row.
    survival_turn: dict = field(default_factory=dict)
    # Gated flavor → {player_id: first turn its gate is cleared}. A gated flavor
    # with no entry for a player (or a DB without the gate's column) is blank.
    flavor_gates: dict = field(default_factory=dict)
    flavor_access: dict = field(default_factory=dict)

    def players_on_team(self, team_id) -> list:
        return [pid for pid in self.major_players if self.team_by_player.get(pid) == team_id]


def build_game_context(cursor, major_players, flavor_gates=None) -> GameContext:
    placeholders = ",".join(["?"] * len(major_players))

    try:
        cursor.execute(
            f"SELECT Key, TeamID FROM PlayerInformations WHERE Key IN ({placeholders})",
            major_players,
        )
        team_by_player = {pid: team for pid, team in cursor.fetchall() if team is not None}
    except sqlite3.DatabaseError as exc:
        # Older DBs may lack TeamID; without teams, each player is its own team.
        if not is_schema_mismatch(exc):
            raise
        team_by_player = {}
    for pid in major_players:
        team_by_player.setdefault(pid, pid)

    cursor.execute(
        f"SELECT Key, MAX(Turn) FROM PlayerSummaries WHERE Key IN ({placeholders}) GROUP BY Key",
        major_players,
    )
    survival_turn = {pid: turn for pid, turn in cursor.fetchall() if turn is not None}

    gates = dict(flavor_gates or {})
    access = _flavor_access(cursor, major_players, gates) if gates else {}
    return GameContext(list(major_players), team_by_player, survival_turn, gates, access)


def _flavor_access(cursor, major_players, gates: dict) -> dict:
    """First turn each player clears each flavor gate (see ``S.DEFAULT_FLAVOR_GATES``)."""
    placeholders = ",".join(["?"] * len(major_players))
    access: dict = {}
    by_techs: dict = {}
    for flavor, gate in gates.items():
        if "techs" in gate:
            by_techs.setdefault(tuple(gate["techs"]), []).append(flavor)
    for techs, flavors in by_techs.items():
        likes = " OR ".join(["CurrentResearch LIKE ? || '%'"] * len(techs))
        try:
            cursor.execute(
                f"SELECT Key, MIN(Turn) FROM PlayerSummaries WHERE Key IN ({placeholders}) "
                f"AND ({likes}) GROUP BY Key",
                [*major_players, *techs],
            )
            turns = {pid: turn for pid, turn in cursor.fetchall() if turn is not None}
        except sqlite3.DatabaseError as exc:
            if not is_schema_mismatch(exc):
                raise
            turns = {}
        for flavor in flavors:
            access[flavor] = dict(turns)

    eras = {flavor: gate["era"] for flavor, gate in gates.items() if "era" in gate}
    if eras:
        try:
            cursor.execute(
                f"SELECT Key, Turn, Era FROM PlayerSummaries WHERE Key IN ({placeholders}) "
                "AND Era IS NOT NULL ORDER BY Key, Turn",
                major_players,
            )
            rows = cursor.fetchall()
        except sqlite3.DatabaseError as exc:
            if not is_schema_mismatch(exc):
                raise
            rows = []
        rank = {name.lower(): i for i, name in enumerate(S.ERA_ORDER)}
        # player → {era rank: first turn at that rank or later}
        reached: dict = {}
        for pid, turn, era in rows:
            words = str(era).split()
            level = rank.get(words[0].lower()) if words else None
            if level is None:
                continue
            first = reached.setdefault(pid, {})
            for lower in range(level + 1):
                first.setdefault(lower, turn)
        for flavor, era in eras.items():
            level = rank[era.lower()]
            access[flavor] = {pid: first[level] for pid, first in reached.items() if level in first}
    return access
