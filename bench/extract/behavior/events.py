"""Event family: per-player counts of selected ``GameEvents``.

Each counter is data: an event type, the payload field naming the actor, and
optional payload filters. All selected types are read in one pass per DB.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from ..utilities import is_schema_mismatch
from .context import GameContext

# ``GameEvents`` has no index led by ``Type``, so a filtered read scans the whole
# (payload-heavy) table. Every per-player index carries ``Type``, so scanning one
# of them for rowids and then fetching those rows is far cheaper on large DBs.
_TYPE_INDEX = "idx_gameevents_player0"
_CHUNK = 500


@dataclass(frozen=True)
class EventCounter:
    type: str
    actor: str                   # dotted payload path to the acting player (or team)
    match: tuple = ()            # ((path, value), ...) that must all hold
    require: str | None = None   # dotted payload path that must be present
    actor_is_team: bool = False  # actor is a team id; credit every major on it

    def accepts(self, payload: dict) -> bool:
        if any(_get(payload, path) != value for path, value in self.match):
            return False
        return self.require is None or _get(payload, self.require) not in (None, "")


EVENT_COUNTERS = {
    # The aggressor's own declarations, including the automatic ones on the
    # target's defensive-pact partners; a vassal following its master is not.
    "wars_declared": EventCounter("DeclareWar", actor="OriginatingPlayerID",
                                  match=(("IsAggressor", True),)),
    "wars_received": EventCounter("DeclareWar", actor="TargetTeamID", actor_is_team=True),
    "cities_nuked": EventCounter("NuclearDetonation", actor="PlayerID", require="Plot.City"),
    "cities_razed": EventCounter("CityRazed", actor="PlayerID"),
}


class EventFamily:
    def columns(self, selection, stats) -> list[str]:
        return list(selection)

    def extract(self, cursor, ctx: GameContext, selection, stats) -> dict:
        if not selection:
            return {}
        counters = {name: EVENT_COUNTERS[name] for name in selection}
        counts = {pid: dict.fromkeys(selection, 0) for pid in ctx.major_players}

        seen = set()
        for event in _fetch_events(cursor, sorted({c.type for c in counters.values()})):
            # Replayed turns can log the same event twice; count it once.
            if event in seen:
                continue
            seen.add(event)
            event_type, _turn, payload_json = event
            try:
                payload = json.loads(payload_json)
            except (json.JSONDecodeError, TypeError):
                continue
            for name, counter in counters.items():
                if counter.type != event_type or not counter.accepts(payload):
                    continue
                actor = _get(payload, counter.actor)
                players = ctx.players_on_team(actor) if counter.actor_is_team else [actor]
                for pid in players:
                    if pid in counts:
                        counts[pid][name] += 1
        return counts


def _get(payload, path: str):
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _fetch_events(cursor, types) -> list:
    """``(Type, Turn, Payload)`` rows of the given types, in rowid order."""
    placeholders = ",".join(["?"] * len(types))
    try:
        cursor.execute(
            f"SELECT rowid FROM GameEvents INDEXED BY {_TYPE_INDEX} WHERE Type IN ({placeholders})",
            types,
        )
        rowids = [r[0] for r in cursor.fetchall()]
    except sqlite3.DatabaseError as exc:
        if is_schema_mismatch(exc):
            return []
        if "no such index" not in str(exc):
            raise
        cursor.execute(
            f"SELECT Type, Turn, Payload FROM GameEvents WHERE Type IN ({placeholders}) ORDER BY rowid",
            types,
        )
        return cursor.fetchall()

    rows = []
    for start in range(0, len(rowids), _CHUNK):
        chunk = rowids[start:start + _CHUNK]
        cursor.execute(
            f"SELECT Type, Turn, Payload FROM GameEvents "
            f"WHERE rowid IN ({','.join(['?'] * len(chunk))}) ORDER BY rowid",
            chunk,
        )
        rows.extend(cursor.fetchall())
    return rows
