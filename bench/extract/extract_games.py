"""Per-game ``game_data`` extraction (rename of the old ``game_timestamps``).

One row per game: ``game_id, timestamp, experiment, seed, seating_rotation``. The
``seed`` / ``seating_rotation`` carry the ``-1`` uncontrolled sentinel (rule 14);
per-player ``config_slot`` is **not** here — it lives in ``panel_data``. Unlike the
source, which read only filenames, this opens each game DB to pull the controlled
seeds/rotation from ``GameMetadata``.
"""

from __future__ import annotations

import sqlite3

from .issues import record_db_failure
from .utilities import (
    append_csv_file,
    extract_seeding_fields,
    filter_existing_data,
    get_experiment_from_path,
    get_game_id_from_path,
    get_timestamp_from_path,
    open_database_readonly,
    read_existing_csv,
    read_game_metadata,
    should_skip_game,
    write_csv_file,
)


GAME_FIELDNAMES = ["game_id", "timestamp", "experiment", "seed", "seating_rotation"]


def extract_game_row(db_path, issues=None) -> dict | None:
    """Build the ``game_data`` row for one game DB.

    Returns ``None`` (skipping the game) when the DB cannot be opened or its
    ``GameMetadata`` cannot be read — recording an :class:`ImportIssue` instead of
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
        # Malformed/locked image — record and skip. A controlled-seed mismatch
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
    expected_fieldnames = GAME_FIELDNAMES
    existing_data, existing_game_ids, structure_matches = read_existing_csv(
        output_file, expected_fieldnames
    )
    pruned_rows = 0

    if not structure_matches:
        print("Discarding existing game data due to structure mismatch...")
        existing_data = []
        existing_game_ids = set()
    else:
        existing_data, existing_game_ids, pruned_rows, pruned_game_ids = filter_existing_data(
            existing_data, available_game_ids
        )
        if pruned_rows > 0:
            print(f"  Filtered out {pruned_rows} rows from {len(pruned_game_ids)} games without database files")
        print(f"Found {len(existing_data)} existing game rows from {len(existing_game_ids)} games")

    new_rows = []
    skipped_count = 0
    print("\nExtracting game data...")
    if prune_only:
        print("Prune-only mode: skipping extraction of new game rows.")
    else:
        for db_file in db_files:
            game_id = get_game_id_from_path(db_file)
            if game_id and should_skip_game(game_id, existing_game_ids):
                skipped_count += 1
                continue
            if issues is not None:
                issues.mark_evaluated("games", game_id)
            row = extract_game_row(db_file, issues=issues)
            if row is not None:
                new_rows.append(row)

    print(f"Skipped {skipped_count} games that were already exported")

    all_rows = existing_data + new_rows
    needs_rewrite = pruned_rows > 0 or not structure_matches
    if needs_rewrite and (new_rows or pruned_rows > 0):
        if write_csv_file(output_file, expected_fieldnames, all_rows):
            print(f"\nRewrote {len(all_rows)} game rows to {output_file}")
    elif new_rows:
        if append_csv_file(output_file, expected_fieldnames, new_rows):
            print(f"\nAppended {len(new_rows)} new game rows to {output_file}")
    else:
        print(f"\nNo new game data to export. Existing file contains {len(existing_data)} rows.")

    return len(new_rows)
