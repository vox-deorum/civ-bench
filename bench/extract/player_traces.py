"""Shared readers for the per-player telemetry DBs (``<game_id>-player-<n>.db``).

Each strategist writes its OpenTelemetry spans to one SQLite file per player next
to the game DB. The token and agentic extractors both locate those files, pick the
latest ``strategist.turn.*`` trace per turn, and label players from the game DB.
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import List, Optional, Tuple

from ..catalog import Catalog
from .identity import compose_identities
from .issues import record_db_failure
from .utilities import (
    extract_seeding_fields,
    get_experiment_from_path, get_game_id_from_path, get_player_info_cache,
    open_database_readonly, read_game_metadata,
)

OTEL_ERROR_STATUS_CODE = 2

PLAYER_TRACE_PATTERN = re.compile(
    r"^(?P<game_id>[0-9a-f-]+)-player-(?P<player_id>\d+)\.db$",
    re.IGNORECASE,
)


def is_top_level_agent_span(span_name: str) -> bool:
    return span_name.startswith("agent.") and ".step." not in span_name


def parse_player_trace_path(trace_db_path: str) -> Tuple[Optional[str], Optional[int]]:
    match = PLAYER_TRACE_PATTERN.match(os.path.basename(trace_db_path))
    if not match:
        return None, None
    return match.group("game_id"), int(match.group("player_id"))


def find_player_trace_databases(game_db_path: str) -> List[Tuple[int, str]]:
    game_id = get_game_id_from_path(game_db_path)
    if not game_id:
        return []

    directory = os.path.dirname(game_db_path)
    trace_dbs = []
    for filename in os.listdir(directory):
        if not filename.startswith(f"{game_id}-player-") or not filename.endswith(".db"):
            continue
        if filename.endswith(".telepathist.db"):
            continue
        _, player_id = parse_player_trace_path(filename)
        if player_id is None:
            continue
        trace_dbs.append((player_id, os.path.join(directory, filename)))
    return sorted(trace_dbs, key=lambda item: item[0])


def fetch_valid_traces(cursor) -> List[Tuple[int, str, int]]:
    """``(turn, traceId, statusCode)`` of the latest ``strategist.turn.*`` root per turn.

    A retried turn leaves several roots; only the last one counts, so retries are
    not double-counted.
    """
    cursor.execute(
        """
        SELECT turn, traceId, statusCode
        FROM (
            SELECT turn, traceId, statusCode, startTime, id,
                   ROW_NUMBER() OVER (
                       PARTITION BY turn
                       ORDER BY startTime DESC, id DESC
                   ) AS rn
            FROM spans
            WHERE name LIKE 'strategist.turn.%'
        )
        WHERE rn = 1
        ORDER BY turn
        """
    )
    return [
        (int(turn), trace_id, int(status_code))
        for turn, trace_id, status_code in cursor.fetchall()
    ]


def game_player_types(game_db_path: str, catalog: Optional[Catalog], issues=None, stage: str = "tokens"):
    """Compose ``({player_id: player_type}, metadata)`` from the game DB (§3.3).

    The extracted values come from the separate ``*-player-*.db`` trace files, so a
    corrupt game DB only costs us the ``player_type`` labels: record the issue and
    degrade to ``({}, {})`` (labels become ``N/A``) rather than crashing or dropping
    rows. The ``metadata`` is returned so a *trace* failure on an otherwise-readable
    game can still report the game's real seed/rotation instead of "unknown".
    """
    conn, cursor = open_database_readonly(game_db_path)
    if not conn:
        if issues is not None:
            issues.record(stage=stage, db_path=game_db_path, message="could not open database")
        return {}, {}
    try:
        metadata = read_game_metadata(cursor)
        experiment = get_experiment_from_path(game_db_path)
        game_id = metadata.get("gameId", get_game_id_from_path(game_db_path))
        player_info_cache = get_player_info_cache(cursor)
        major_players = [pid for pid, info in player_info_cache.items() if info["is_major"]]
        seeding = extract_seeding_fields(metadata, where=f"game {game_id}")
        identities = compose_identities(metadata, major_players, experiment, catalog, seeding)
    except sqlite3.DatabaseError as exc:
        # Malformed/locked image: record and degrade labels to N/A. A
        # controlled-seed mismatch (ExtractError) stays a hard abort.
        record_db_failure(issues, stage=stage, db_path=game_db_path, exc=exc)
        return {}, {}
    finally:
        conn.close()
    return {pid: ident.get("player_type") for pid, ident in identities.items()}, metadata
