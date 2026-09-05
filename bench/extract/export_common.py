"""Shared export driver for the four canonical extract tables.

``export_games`` / ``export_panel`` / ``export_turns`` / ``export_model_tokens``
were four near-identical copies of the same incremental-export skeleton (read the
existing CSV → prune rows whose DB is gone → dedupe → skip already-exported games →
extract the rest → write or append). Consolidating them here removes that
duplication and fixes two latent bugs that lived in every copy:

* **A**: a failed ``write_csv_file`` / ``append_csv_file`` (both return ``False`` on
  error) was ignored, so a full disk / permission error silently dropped rows while
  the run still reported success. It now raises :class:`ExtractError`.
* **B**: a structure-mismatch printed "Discarding existing … data" but then only
  rewrote when there were new rows or prunes, so a mismatch with zero new rows left
  the old (wrong-schema) file untouched; the message lied. A normal run now always
  full-rewrites on mismatch (so the file matches the current schema); a prune-only
  run (which inspects no DBs and would never rewrite) skips the "Discarding" message
  and instead warns that the file was left untouched and how to rebuild it.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from .errors import ExtractError
from .utilities import (
    append_csv_file,
    filter_existing_data,
    get_game_id_from_path,
    read_existing_csv,
    should_skip_game,
    write_csv_file,
)


def run_table_export(
    db_files,
    available_game_ids,
    output_file,
    *,
    stage: str,
    fieldnames: list[str],
    extract_rows: Callable[[str, Optional[object]], list],
    dedupe_key: Optional[Callable[[dict], tuple]] = None,
    describe_db: Optional[Callable[[str, list], str]] = None,
    noun: str = "rows",
    prune_only: bool = False,
    issues=None,
) -> int:
    """Incrementally export one canonical table; return the count of NEW rows.

    ``extract_rows(db_file, issues) -> list[dict]`` pulls a game's rows (empty when
    the game is skipped/failed). ``dedupe_key(row) -> tuple`` (``None`` for the
    one-row-per-game ``games`` table) collapses duplicate existing rows.
    ``describe_db(db_file, rows) -> str`` supplies the optional per-DB progress line.
    """
    existing_data, existing_game_ids, structure_matches = read_existing_csv(
        output_file, fieldnames
    )
    pruned_rows = 0

    if not structure_matches:
        if prune_only:
            # Prune-only inspects no DBs and never rewrites; the honest thing is to
            # leave the mismatched file untouched and say so (Bug B), not print
            # "Discarding" and then silently keep the old file.
            print(
                f"Existing {noun} file has an out-of-date structure; prune-only left "
                f"it untouched. Re-run extract without prune_missing to rebuild "
                f"{output_file}."
            )
            return 0
        print(f"Discarding existing {noun} due to structure mismatch (full rewrite)...")
        existing_data = []
        existing_game_ids = set()
    else:
        existing_data, existing_game_ids, pruned_rows, pruned_game_ids = filter_existing_data(
            existing_data, available_game_ids
        )
        if pruned_rows > 0:
            print(
                f"  Filtered out {pruned_rows} rows from {len(pruned_game_ids)} games "
                f"without database files"
            )
        if dedupe_key is not None:
            seen: dict = {}
            for i, row in enumerate(existing_data):
                seen[dedupe_key(row)] = i
            if len(seen) < len(existing_data):
                deduped_count = len(existing_data) - len(seen)
                existing_data = [existing_data[i] for i in sorted(seen.values())]
                existing_game_ids = {
                    r["game_id"] for r in existing_data
                    if r.get("game_id") and r["game_id"] != "N/A"
                }
                pruned_rows += deduped_count
                print(f"  Removed {deduped_count} duplicate {noun}")
        print(f"Found {len(existing_data)} existing {noun} from {len(existing_game_ids)} games")

    new_rows: list = []
    skipped_count = 0
    print(f"\nExtracting {noun}...")
    if prune_only:
        print(f"Prune-only mode: skipping extraction of new {noun}.")
    else:
        for db_file in db_files:
            game_id = get_game_id_from_path(db_file)
            if game_id and should_skip_game(game_id, existing_game_ids):
                skipped_count += 1
                continue
            if issues is not None:
                issues.mark_evaluated(stage, game_id)
            rows = extract_rows(db_file, issues)
            if rows:
                new_rows.extend(rows)
                if describe_db is not None:
                    print(f"Processed: {os.path.basename(db_file)} ({describe_db(db_file, rows)})")

    print(f"Skipped {skipped_count} databases that were already exported")

    all_rows = existing_data + new_rows
    # needs_rewrite ⇒ pruned rows and/or a discarded mismatch: rewrite the whole file
    # (unconditionally, so a mismatch with zero new rows still lands the new schema).
    needs_rewrite = pruned_rows > 0 or not structure_matches
    if needs_rewrite:
        if not write_csv_file(output_file, fieldnames, all_rows):
            raise ExtractError(f"failed to write {noun} to {output_file}")
        print(f"\nRewrote {len(all_rows)} {noun} to {output_file}")
    elif new_rows:
        if not append_csv_file(output_file, fieldnames, new_rows):
            raise ExtractError(f"failed to append {noun} to {output_file}")
        print(f"\nAppended {len(new_rows)} new {noun} to {output_file}")
    else:
        print(f"\nNo new {noun} to export. Existing file contains {len(existing_data)} rows.")

    return len(new_rows)
