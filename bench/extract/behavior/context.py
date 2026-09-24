"""Per-game facts shared by every behavior family, read once per DB."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

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

    def players_on_team(self, team_id) -> list:
        return [pid for pid in self.major_players if self.team_by_player.get(pid) == team_id]


def build_game_context(cursor, major_players) -> GameContext:
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

    return GameContext(list(major_players), team_by_player, survival_turn)
