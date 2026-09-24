"""Helpers for counting state-change rows that record a real mutation."""

from __future__ import annotations

import json


def has_real_changes(changes_json) -> bool:
    """True when ``Changes`` contains at least one non-rationale field."""
    if changes_json is None or not isinstance(changes_json, str):
        return False
    changes_json = changes_json.strip()
    if not changes_json:
        return False

    try:
        changes = json.loads(changes_json)
    except (json.JSONDecodeError, TypeError):
        return changes_json not in ("[]", '["Rationale"]')

    if not isinstance(changes, list):
        return False
    return any(change != "Rationale" for change in changes)


def count_real_change_rows(cursor, table_name: str, player_id: int) -> int:
    """Count rows whose ``Changes`` JSON records a real field mutation."""
    cursor.execute(f"SELECT Changes FROM {table_name} WHERE Key = ?", (player_id,))
    return sum(1 for (changes_json,) in cursor.fetchall() if has_real_changes(changes_json))
