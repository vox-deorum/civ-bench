"""Shared readers for the ``GameEvents`` table.

The behavior events family and the ``world`` table both count game events, so
they read them the same way: one indexed pass per DB, payloads parsed once, and
replayed duplicates dropped.
"""

from __future__ import annotations

import json
import sqlite3

from .utilities import is_schema_mismatch

# ``GameEvents`` has no index led by ``Type``, so a filtered read scans the whole
# (payload-heavy) table. Every per-player index carries ``Type``, so scanning one
# of them for rowids and then fetching those rows is far cheaper on large DBs.
_TYPE_INDEX = "idx_gameevents_player0"
_CHUNK = 500


def fetch_events(cursor, types) -> list:
    """``(Type, Turn, Payload)`` rows of the given types, in rowid order."""
    types = list(types)
    if not types:
        return []
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


def unique_events(rows) -> list:
    """``(Type, Turn, payload dict)`` in order, each logged event once.

    Replayed turns can log the same event twice, so an exact repeat of
    ``(Type, Turn, Payload)`` counts once. Rows whose payload is not a JSON
    object are skipped.
    """
    seen = set()
    out = []
    for event in rows:
        if event in seen:
            continue
        seen.add(event)
        event_type, turn, payload_json = event
        try:
            payload = json.loads(payload_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            out.append((event_type, turn, payload))
    return out


def payload_get(payload, path: str):
    """The value at a dotted ``path`` such as ``Plot.City``, or ``None``."""
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
