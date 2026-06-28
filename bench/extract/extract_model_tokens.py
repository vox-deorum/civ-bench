"""Per-game telemetry ``model_token_usage`` extraction (ported from the analysis repo).

Departure from the source (``plans/stage1.md``): ``player_type`` is no longer
re-derived from experiment-config JSON. It is the **orthodox identity** composed
from the game DB's ``GameMetadata`` (``model-{id}`` / ``strategist-{id}``, §3.3) —
the same single source of truth that feeds ``panel_data`` — and looked up by
``player_id`` for each player-trace DB. Model-name normalization routes through the
supplied :class:`Catalog` instead of the old ``shared.model_catalog`` globals.
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

from ..catalog import Catalog
from .identity import compose_identities
from .issues import record_db_failure
from .utilities import (
    append_csv_file, extract_seeding_fields, filter_existing_data,
    get_experiment_from_path, get_game_id_from_path, get_player_info_cache,
    open_database_readonly, read_existing_csv, read_game_metadata,
    should_skip_game, write_csv_file,
)


MODEL_TOKEN_FIELDNAMES = [
    "experiment",
    "game_id",
    "player_id",
    "player_type",
    "model_name",
    "model_base",
    "model_variants",
    "agent_names",
    "input_tokens",
    "reasoning_tokens",
    "output_tokens",
    "total_tokens",
    "tool_count",
    "focus_briefer_count",
    "valid_turn_count",
]

VANILLA_MODEL_NAME = "Vanilla"
VANILLA_MODEL_BASE = "vanilla"

PLAYER_TRACE_PATTERN = re.compile(
    r"^(?P<game_id>[0-9a-f-]+)-player-(?P<player_id>\d+)\.db$",
    re.IGNORECASE,
)


def _extract_agent_name(span_name: str, agent_name: Optional[str]) -> str:
    if agent_name:
        return agent_name
    if span_name.startswith("agent."):
        stripped = span_name[len("agent.") :]
        return stripped.split(".step.", 1)[0]
    return span_name


def _is_top_level_agent_span(span_name: str) -> bool:
    return span_name.startswith("agent.") and ".step." not in span_name


def _is_tool_span(span_name: str) -> bool:
    return span_name.startswith("mcp-tool.") or span_name.startswith("simple-tool.")


def _parse_player_trace_path(trace_db_path: str) -> Tuple[Optional[str], Optional[int]]:
    match = PLAYER_TRACE_PATTERN.match(os.path.basename(trace_db_path))
    if not match:
        return None, None
    return match.group("game_id"), int(match.group("player_id"))


def _find_player_trace_databases(game_db_path: str) -> List[Tuple[int, str]]:
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
        _, player_id = _parse_player_trace_path(filename)
        if player_id is None:
            continue
        trace_dbs.append((player_id, os.path.join(directory, filename)))
    return sorted(trace_dbs, key=lambda item: item[0])


def _fetch_valid_traces(cursor) -> List[Tuple[int, str]]:
    cursor.execute(
        """
        SELECT turn, traceId
        FROM (
            SELECT turn, traceId, startTime, id,
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
    return [(int(turn), trace_id) for turn, trace_id in cursor.fetchall()]


def _fetch_relevant_spans(cursor, trace_ids: List[str]) -> List[dict]:
    if not trace_ids:
        return []
    placeholders = ",".join(["?"] * len(trace_ids))
    cursor.execute(
        f"""
        SELECT
            id, traceId, spanId, parentSpanId, turn, name,
            json_extract(attributes, '$."agent.name"') AS agent_name,
            json_extract(attributes, '$.model') AS model,
            CAST(COALESCE(json_extract(attributes, '$."tokens.input"'), 0) AS INTEGER) AS input_tokens,
            CAST(COALESCE(json_extract(attributes, '$."tokens.reasoning"'), 0) AS INTEGER) AS reasoning_tokens,
            CAST(COALESCE(json_extract(attributes, '$."tokens.output"'), 0) AS INTEGER) AS output_tokens
        FROM spans
        WHERE traceId IN ({placeholders})
          AND (
              name LIKE 'agent.%'
              OR name LIKE 'mcp-tool.%'
              OR name LIKE 'simple-tool.%'
              OR name LIKE 'strategist.turn.%'
          )
        ORDER BY traceId, id
        """,
        trace_ids,
    )
    spans = []
    for row in cursor.fetchall():
        spans.append({
            "id": row[0],
            "trace_id": row[1],
            "span_id": row[2],
            "parent_span_id": row[3],
            "turn": row[4],
            "name": row[5],
            "agent_name": row[6],
            "model_raw": row[7],
            "input_tokens": int(row[8] or 0),
            "reasoning_tokens": int(row[9] or 0),
            "output_tokens": int(row[10] or 0),
        })
    return spans


def _get_or_create_model_row(
    rows_by_model: Dict[str, dict],
    catalog: Catalog,
    experiment: str,
    game_id: str,
    player_id: int,
    player_type: str,
    model_raw: str,
    valid_turn_count: int,
) -> dict:
    model_base = catalog.normalize_model_base(model_raw)
    if not model_base:
        raise ValueError("model_raw must normalize to a non-empty model_base")

    if model_base not in rows_by_model:
        rows_by_model[model_base] = {
            "experiment": experiment,
            "game_id": game_id,
            "player_id": player_id,
            "player_type": player_type,
            "model_name": catalog.canonicalize_model_name(model_base),
            "model_base": model_base,
            "input_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "tool_count": 0,
            "focus_briefer_count": 0,
            "valid_turn_count": valid_turn_count,
            "_model_variants": set(),
            "_agent_names": set(),
        }

    rows_by_model[model_base]["_model_variants"].add(model_raw)
    return rows_by_model[model_base]


def _find_owner_span(tool_span: dict, span_lookup: Dict[Tuple[str, str], dict]) -> Optional[dict]:
    current_parent_span_id = tool_span["parent_span_id"]
    trace_id = tool_span["trace_id"]
    while current_parent_span_id:
        parent = span_lookup.get((trace_id, current_parent_span_id))
        if not parent:
            return None
        if _is_top_level_agent_span(parent["name"]) and parent["model_raw"]:
            return parent
        current_parent_span_id = parent["parent_span_id"]
    return None


def _build_zero_token_row(experiment, game_id, player_id, player_type, valid_turn_count) -> dict:
    return {
        "experiment": experiment,
        "game_id": game_id,
        "player_id": player_id,
        "player_type": player_type,
        "model_name": VANILLA_MODEL_NAME,
        "model_base": VANILLA_MODEL_BASE,
        "model_variants": VANILLA_MODEL_BASE,
        "agent_names": "",
        "input_tokens": 0,
        "reasoning_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "tool_count": 0,
        "focus_briefer_count": 0,
        "valid_turn_count": valid_turn_count,
    }


def extract_player_model_token_rows(
    trace_db_path: str,
    game_id: str,
    experiment: str,
    player_type: str,
    catalog: Catalog,
    issues=None,
    metadata=None,
) -> List[dict]:
    """Aggregate one row per (player, model) from a player-trace DB.

    ``metadata`` is the owning game's already-read ``GameMetadata`` (or ``{}`` if it
    was unreadable); passing it lets a trace failure report the game's real seed/
    rotation rather than "unknown".
    """
    _, player_id = _parse_player_trace_path(trace_db_path)
    if player_id is None:
        return []

    conn, _ = open_database_readonly(trace_db_path)
    if not conn:
        if issues is not None:
            issues.record(
                stage="tokens", db_path=trace_db_path, game_id=game_id,
                experiment=experiment, player_id=player_id, metadata=metadata,
                message="could not open trace database",
            )
        return []

    try:
        cursor = conn.cursor()
        valid_turns = _fetch_valid_traces(cursor)
        if not valid_turns:
            conn.close()
            return []
        valid_turn_count = len(valid_turns)
        trace_ids = [trace_id for _, trace_id in valid_turns]
        spans = _fetch_relevant_spans(cursor, trace_ids)
        conn.close()
    except sqlite3.DatabaseError as exc:
        record_db_failure(
            issues, stage="tokens", db_path=trace_db_path, exc=exc,
            game_id=game_id, experiment=experiment, player_id=player_id, metadata=metadata,
        )
        conn.close()
        return []
    except Exception as exc:
        print(f"Error processing {os.path.basename(trace_db_path)}: {exc}")
        conn.close()
        return []

    if not spans:
        return []

    rows_by_model: Dict[str, dict] = {}
    span_lookup = {(span["trace_id"], span["span_id"]): span for span in spans}

    for span in spans:
        if not _is_top_level_agent_span(span["name"]) or not span["model_raw"]:
            continue
        row = _get_or_create_model_row(
            rows_by_model, catalog, experiment, game_id, player_id,
            player_type, span["model_raw"], valid_turn_count,
        )
        row["input_tokens"] += span["input_tokens"]
        row["reasoning_tokens"] += span["reasoning_tokens"]
        row["output_tokens"] += span["output_tokens"]
        row["_agent_names"].add(_extract_agent_name(span["name"], span["agent_name"]))

    for span in spans:
        if not _is_tool_span(span["name"]):
            continue
        owner_span = _find_owner_span(span, span_lookup)
        if not owner_span or not owner_span["model_raw"]:
            continue
        row = _get_or_create_model_row(
            rows_by_model, catalog, experiment, game_id, player_id,
            player_type, owner_span["model_raw"], valid_turn_count,
        )
        row["tool_count"] += 1
        if span["name"] == "simple-tool.focus-briefer":
            row["focus_briefer_count"] += 1
        row["_agent_names"].add(_extract_agent_name(owner_span["name"], owner_span["agent_name"]))

    if not rows_by_model and player_type == VANILLA_MODEL_NAME and valid_turn_count > 0:
        return [_build_zero_token_row(experiment, game_id, player_id, player_type, valid_turn_count)]

    final_rows = []
    for row in sorted(rows_by_model.values(), key=lambda item: item["model_base"]):
        final_rows.append({
            "experiment": row["experiment"],
            "game_id": row["game_id"],
            "player_id": row["player_id"],
            "player_type": row["player_type"],
            "model_name": row["model_name"],
            "model_base": row["model_base"],
            "model_variants": "|".join(sorted(row["_model_variants"])),
            "agent_names": "|".join(sorted(row["_agent_names"])),
            "input_tokens": row["input_tokens"],
            "reasoning_tokens": row["reasoning_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["input_tokens"] + row["reasoning_tokens"] + row["output_tokens"],
            "tool_count": row["tool_count"],
            "focus_briefer_count": row["focus_briefer_count"],
            "valid_turn_count": row["valid_turn_count"],
        })
    return final_rows


def _game_player_types(game_db_path: str, catalog: Optional[Catalog], issues=None):
    """Compose ``({player_id: player_type}, metadata)`` from the game DB (§3.3).

    Token counts come from the separate ``*-player-*.db`` trace files, so a corrupt
    game DB only costs us the ``player_type`` labels — record the issue and degrade
    to ``({}, {})`` (labels become ``N/A``) rather than crashing or dropping tokens.
    The ``metadata`` is returned so a *trace* failure on an otherwise-readable game
    can still report the game's real seed/rotation instead of "unknown".
    """
    conn, cursor = open_database_readonly(game_db_path)
    if not conn:
        if issues is not None:
            issues.record(stage="tokens", db_path=game_db_path, message="could not open database")
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
        # Malformed/locked image — record and degrade labels to N/A. A
        # controlled-seed mismatch (ExtractError) stays a hard abort.
        record_db_failure(issues, stage="tokens", db_path=game_db_path, exc=exc)
        return {}, {}
    finally:
        conn.close()
    return {pid: ident.get("player_type") for pid, ident in identities.items()}, metadata


def extract_game_model_token_data(game_db_path: str, catalog: Catalog, issues=None) -> List[dict]:
    """Aggregate per-(player, model) token rows for one game's trace DBs."""
    experiment = get_experiment_from_path(game_db_path)
    game_id = get_game_id_from_path(game_db_path)
    if not game_id:
        return []

    player_types, metadata = _game_player_types(game_db_path, catalog, issues=issues)

    all_rows = []
    for player_id, trace_db_path in _find_player_trace_databases(game_db_path):
        player_type = player_types.get(player_id) or "N/A"
        all_rows.extend(
            extract_player_model_token_rows(
                trace_db_path=trace_db_path,
                game_id=game_id,
                experiment=experiment,
                player_type=player_type,
                metadata=metadata,
                catalog=catalog,
                issues=issues,
            )
        )
    return all_rows


def export_model_token_data(db_files, available_game_ids, output_file, catalog: Catalog, prune_only=False, issues=None) -> int:
    """Export model-token rows to ``output_file``; returns the count of new rows."""
    expected_fieldnames = MODEL_TOKEN_FIELDNAMES
    existing_data, existing_game_ids, structure_matches = read_existing_csv(output_file, expected_fieldnames)
    pruned_rows = 0

    if not structure_matches:
        print("Discarding existing model token usage data due to structure mismatch...")
        existing_data = []
        existing_game_ids = set()
    else:
        existing_data, existing_game_ids, pruned_rows, pruned_game_ids = filter_existing_data(
            existing_data, available_game_ids
        )
        if pruned_rows > 0:
            print(f"  Filtered out {pruned_rows} model-token rows from {len(pruned_game_ids)} games without database files")

        seen = {}
        for index, row in enumerate(existing_data):
            seen[(row.get("game_id"), row.get("player_id"), row.get("model_base"))] = index
        if len(seen) < len(existing_data):
            deduped_count = len(existing_data) - len(seen)
            existing_data = [existing_data[index] for index in sorted(seen.values())]
            existing_game_ids = {
                row["game_id"] for row in existing_data
                if row.get("game_id") and row.get("game_id") != "N/A"
            }
            pruned_rows += deduped_count
            print(f"  Removed {deduped_count} duplicate model-token rows")

        print(f"Found {len(existing_data)} existing model-token rows from {len(existing_game_ids)} games")

    print("\nExtracting model token usage data...")
    new_rows = []
    skipped_count = 0
    processed_count = 0

    if prune_only:
        print("Prune-only mode: skipping extraction of new model token rows.")
    else:
        for db_file in db_files:
            game_id = get_game_id_from_path(db_file)
            if game_id and should_skip_game(game_id, existing_game_ids):
                skipped_count += 1
                continue
            if issues is not None:
                issues.mark_evaluated("tokens", game_id)
            model_rows = extract_game_model_token_data(db_file, catalog, issues=issues)
            if not model_rows:
                continue
            new_rows.extend(model_rows)
            processed_count += 1
            unique_players = len({row["player_id"] for row in model_rows})
            unique_models = len({(row["player_id"], row["model_base"]) for row in model_rows})
            print(f"Processed: {os.path.basename(db_file)} ({unique_players} players, {unique_models} player-model rows)")

    print(f"\nProcessed {processed_count} new databases")
    print(f"Skipped {skipped_count} databases that were already exported")

    all_rows = existing_data + new_rows
    needs_rewrite = pruned_rows > 0 or not structure_matches
    if needs_rewrite and (new_rows or pruned_rows > 0):
        if write_csv_file(output_file, expected_fieldnames, all_rows):
            print(f"\nRewrote {len(all_rows)} model-token rows to {output_file}")
    elif new_rows:
        if append_csv_file(output_file, expected_fieldnames, new_rows):
            print(f"\nAppended {len(new_rows)} new model-token rows to {output_file}")
    else:
        print(f"\nNo new model token usage data to export. Existing file contains {len(existing_data)} rows.")

    return len(new_rows)
