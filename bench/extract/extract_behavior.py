"""Per-player-per-game ``behavior`` extraction (benchmark.md §3.0).

One row per major player: identity keys, then the columns of each behavior
family in ``data.extract.behavior``. Every family reads its table once per DB.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional

from ..catalog import Catalog
from ..config.behavior import family_columns, resolve_behavior_spec
from .behavior import FAMILIES, build_game_context
from .errors import ExtractError
from .export_common import run_table_export
from .identity import compose_identities
from .issues import record_db_failure
from .utilities import (
    extract_seeding_fields,
    get_experiment_from_path, get_game_id_from_path, get_player_info_cache,
    open_database_readonly, read_game_metadata,
)

BEHAVIOR_KEY_FIELDS = [
    "experiment", "game_id", "player_id", "player_type", "civilization", "survival_turn",
]


def behavior_fieldnames(spec: Optional[dict] = None) -> list[str]:
    spec = resolve_behavior_spec(spec)
    columns = list(BEHAVIOR_KEY_FIELDS)
    for name in FAMILIES:
        columns.extend(column for column, _kind in family_columns(name, spec[name], spec["stats"]))
    return columns


def extract_game_behavior_data(db_path, spec: dict, catalog: Optional[Catalog] = None, issues=None):
    """Extract behavior rows for one game DB; a malformed DB is recorded and skipped."""
    conn, cursor = open_database_readonly(db_path)
    if not conn:
        if issues is not None:
            issues.record(stage="behavior", db_path=db_path, message="could not open database")
        return []

    metadata: dict = {}
    try:
        experiment = get_experiment_from_path(db_path)
        metadata = read_game_metadata(cursor)
        game_id = metadata.get("gameId", get_game_id_from_path(db_path))

        player_info_cache = get_player_info_cache(cursor)
        major_players = [pid for pid, info in player_info_cache.items() if info["is_major"]]
        if not major_players:
            return []

        seeding = extract_seeding_fields(metadata, where=f"game {game_id}")
        identities = compose_identities(metadata, major_players, experiment, catalog, seeding)
        ctx = build_game_context(cursor, major_players)

        values_by_family = {
            name: family.extract(cursor, ctx, spec[name], spec["stats"])
            for name, family in FAMILIES.items()
        }
        conn.close()

        rows = []
        for pid in major_players:
            row = {
                "experiment": experiment,
                "game_id": game_id,
                "player_id": pid,
                "player_type": identities.get(pid, {}).get("player_type"),
                "civilization": player_info_cache[pid]["civilization"],
                "survival_turn": ctx.survival_turn.get(pid, ""),
            }
            for name in FAMILIES:
                values = values_by_family[name].get(pid, {})
                for column, _kind in family_columns(name, spec[name], spec["stats"]):
                    row[column] = values.get(column, "")
            rows.append(row)
        return rows
    except ExtractError:
        conn.close()
        raise
    except sqlite3.DatabaseError as exc:
        record_db_failure(issues, stage="behavior", db_path=db_path, exc=exc, metadata=metadata)
        conn.close()
        return []
    except Exception as exc:
        print(f"Error processing {os.path.basename(db_path)}: {exc}")
        conn.close()
        return []


def export_behavior_data(db_files, available_game_ids, output_file, spec: Optional[dict] = None,
                         catalog: Optional[Catalog] = None, prune_only=False, issues=None) -> int:
    """Export behavior rows to ``output_file``; returns the count of new rows."""
    spec = resolve_behavior_spec(spec)

    def _extract_rows(db_file, issues):
        return extract_game_behavior_data(db_file, spec, catalog=catalog, issues=issues)

    return run_table_export(
        db_files, available_game_ids, output_file,
        stage="behavior", fieldnames=behavior_fieldnames(spec),
        extract_rows=_extract_rows,
        dedupe_key=lambda r: (r.get("game_id"), r.get("player_id")),
        describe_db=lambda db_file, rows: f"{len(rows)} player rows",
        noun="behavior rows", prune_only=prune_only, issues=issues,
    )
