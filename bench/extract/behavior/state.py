"""State families: min / avg / max of values a player sets over the game.

``FlavorChanges`` and ``PersonaChanges`` share one shape: each row is the full
state after a change, several rows can share a turn, and a state holds until the
next row. The state in effect on a turn is that turn's last row by ``ID``.
"""

from __future__ import annotations

import sqlite3

from ..utilities import is_schema_mismatch
from .context import GameContext, snake_case


class StateFamily:
    def __init__(self, table: str, prefix: str):
        self.table = table
        self.prefix = prefix

    def extract(self, cursor, ctx: GameContext, selection, stats) -> dict:
        if not selection:
            return {}
        cursor.execute(f"PRAGMA table_info({self.table})")
        available = {row[1] for row in cursor.fetchall()}
        selected = [name for name in selection if name in available]
        if not selected:
            return {}
        placeholders = ",".join(["?"] * len(ctx.major_players))
        # Column names come from the schema allowlist, never from free text.
        try:
            cursor.execute(
                f"SELECT Key, Turn, {', '.join(selected)} FROM {self.table} "
                f"WHERE Key IN ({placeholders}) ORDER BY Key, Turn, ID",
                ctx.major_players,
            )
            rows = cursor.fetchall()
        except sqlite3.DatabaseError as exc:
            # An older DB may lack a structural column; corruption must surface.
            if is_schema_mismatch(exc):
                return {}
            raise

        # player → {turn: values}; a later row in the same turn replaces an earlier one.
        states: dict = {}
        for key, turn, *values in rows:
            states.setdefault(key, {})[turn] = values

        result = {}
        for pid, by_turn in states.items():
            turns = sorted(by_turn)
            end = max(ctx.survival_turn.get(pid, turns[-1]), turns[-1])
            weights = turn_weights(turns, end)
            record = {}
            for idx, name in enumerate(selected):
                series = [(by_turn[t][idx], w) for t, w in zip(turns, weights) if by_turn[t][idx] is not None]
                for stat in stats:
                    record[f"{self.prefix}_{snake_case(name)}_{stat}"] = _summarize(series, stat)
            result[pid] = record
        return result


def turn_weights(turns, end) -> list[int]:
    """Turns each state holds: until the next state's turn, the last one through ``end``.

    ``turns`` is sorted. A state is clipped at ``end`` (inclusive), so a state that
    starts after ``end`` weighs 0.
    """
    bounds = list(turns[1:]) + [end + 1]
    return [max(0, min(nxt, end + 1) - cur) for cur, nxt in zip(turns, bounds)]


def _summarize(series, stat):
    if not series:
        return ""
    values = [v for v, _ in series]
    if stat == "min":
        return min(values)
    if stat == "max":
        return max(values)
    total = sum(w for _, w in series)
    return round(sum(v * w for v, w in series) / total, 4)
