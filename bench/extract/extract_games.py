"""Per-game ``game_data`` extraction (rename of the old ``game_timestamps``).

One row per game: ``game_id, timestamp, experiment, seed, seating_rotation``. The
``seed`` / ``seating_rotation`` carry the ``-1`` uncontrolled sentinel (rule 14);
per-player ``config_slot`` is **not** here (it lives in ``panel_data``). Unlike the
source, which read only filenames, this opens each game DB to pull the controlled
seeds/rotation from ``GameMetadata``.
"""

from __future__ import annotations

import sqlite3

from .export_common import run_table_export
from .issues import record_db_failure
from .utilities import (
    extract_seeding_fields,
    get_experiment_from_path,
    get_game_id_from_path,
    get_timestamp_from_path,
    open_database_readonly,
    read_game_metadata,
)


GAME_FIELDNAMES = ["game_id", "timestamp", "experiment", "seed", "seating_rotation"]


def extract_game_row(db_path, issues=None) -> dict | None:
    """Build the ``game_data`` row for one game DB.

    Returns ``None`` (skipping the game) when the DB cannot be opened or its
    ``GameMetadata`` cannot be read, recording an :class:`ImportIssue` instead of
    writing a misleading ``seed=-1`` row or crashing the run.
    """
    game_id = get_game_id_from_path(db_path)
    if not game_id:
        return None

    timestamp = get_timestamp_from_path(db_path)
    experiment = get_experiment_from_path(db_path)

    conn, cursor = open_database_readonly(db_path)
    if conn is None:
        if issues is not None:
            issues.record(stage="games", db_path=db_path, message="could not open database")
        return None

    try:
        metadata = read_game_metadata(cursor)
        seeding = extract_seeding_fields(metadata, where=f"game {game_id}")
        seed = seeding.seed
        seating_rotation = seeding.seating_rotation
    except sqlite3.DatabaseError as exc:
        # Malformed/locked image: record and skip. A controlled-seed mismatch
        # (ExtractError) is a hard policy abort, not a DB fault: let it propagate.
        record_db_failure(issues, stage="games", db_path=db_path, exc=exc)
        return None
    finally:
        conn.close()

    return {
        "game_id": game_id,
        "timestamp": timestamp if timestamp is not None else "N/A",
        "experiment": experiment,
        "seed": seed,
        "seating_rotation": seating_rotation,
    }


def export_game_data(db_files, available_game_ids, output_file, prune_only=False, issues=None) -> int:
    """Export per-game rows to ``output_file``; returns the count of new rows."""
    def _extract_rows(db_file, issues):
        row = extract_game_row(db_file, issues=issues)
        return [row] if row is not None else []

    # games is 1 row per DB, so there is nothing to dedupe (dedupe_key=None).
    return run_table_export(
        db_files, available_game_ids, output_file,
        stage="games", fieldnames=GAME_FIELDNAMES, extract_rows=_extract_rows,
        dedupe_key=None, noun="game rows", prune_only=prune_only, issues=issues,
    )
