"""Per-player-per-game ``panel_data`` extraction (ported from the analysis repo).

Departure from the source (``plans/stage1.md``): each row now carries the
**orthodox identity** composed at extract, ``player_type`` (benchmark.md §3.3),
plus the raw ``model`` / ``strategist`` it was composed from and the controlled
``config_slot``. Behavioral summaries and token telemetry are exported by their
own canonical tables.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional

from ..catalog import Catalog
from .change_counts import count_real_change_rows, has_real_changes as _has_real_changes
from .errors import ExtractError
from .export_common import run_table_export
from .identity import compose_identities
from .issues import record_db_failure
from .utilities import (
    STRATEGY_MAPPINGS, CHANGE_FIELDS, PLAYER_CORE_FIELDS,
    extract_seeding_fields,
    get_experiment_from_path, get_game_id_from_path, get_player_info_cache,
    open_database_readonly,
    read_game_metadata,
)

# Combine all player-specific fields for error handling
ALL_PLAYER_FIELDS = (
    PLAYER_CORE_FIELDS
    + CHANGE_FIELDS
    + ["decisions"]
    + list(STRATEGY_MAPPINGS.values())
)

# Identity fields composed at extract (benchmark.md §3.3), inserted per player.
IDENTITY_FIELDS = ["player_type", "model", "strategist", "config_slot"]

MUTABLE_KNOWLEDGE_METADATA_COLUMNS = {
    "ID",
    "Turn",
    "Key",
    "OwnerID",
    "KnownByIDs",
    "Payload",
    "IsLatest",
    "CreatedAt",
    "Version",
    "Changes",
    "Rationale",
}

# Define field mappings for panel data structure
PANEL_FIELD_MAPPINGS = {
    # Game-level fields
    "experiment": None,  # Special handling: use folder name
    "game_id": "gameId",
    "turn": "turn",
    "map_type": "mapType",
    "map_size": "mapSize",
    "difficulty": "difficulty",
    "game_speed": "gameSpeed",
    "victory_type": "victoryType",
    "victory_player_id": "victoryPlayerID",

    # Player-specific fields
    "player_id": None,  # Player ID (0, 1, 2, etc.)
    "player_type": None,  # Orthodox identity composed at extract (§3.3)
    "model": None,  # Raw model-{id} metadata the player_type was composed from
    "strategist": None,  # Raw strategist-{id} metadata
    "config_slot": None,  # Controlled-seating config slot (-1 for non-treatment seats)
    "civilization": None,  # Player's civilization name
    "score": None,
    "score_rank": None,
    "score_ratio": None,
    "survival_turn": None,
    "is_winner": None,
    "strategy_changes": None,  # turns with an *actual* flavor-number change
    "decisions": None,         # turns the strategist acted, incl. status-quo+rationale
    "persona_changes": None,
    "research_changes": None,
    "domination_ratio": None,
    "culture_ratio": None,
    "diplomatic_ratio": None,
    "science_ratio": None,
}


def calculate_score_ranks(cursor):
    """Rank major players by their highest in-game score. Returns ``{player_id: rank}``."""
    cursor.execute("""
        SELECT ps.Key, MAX(ps.Score) as MaxScore
        FROM PlayerSummaries ps
        INNER JOIN PlayerInformations pi ON ps.Key = pi.Key
        WHERE pi.IsMajor = 1
        GROUP BY ps.Key
        HAVING MaxScore IS NOT NULL
        ORDER BY MaxScore DESC
    """)
    final_scores = cursor.fetchall()
    return {player_key: rank for rank, (player_key, _score) in enumerate(final_scores, start=1)}


def _table_columns(cursor, table_name: str) -> list[str]:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def count_strategist_persona_state_changes(cursor, player_id: int) -> int:
    """Count changes in the strategist-authored persona state.

    ``PersonaChanges`` also records in-game-AI refresh rows. Those can alternate
    with a null strategist reapplying the same baseline and make raw DB mutations
    look like hundreds of persona changes, so this compares only strategist rows.
    """
    columns = _table_columns(cursor, "PersonaChanges")
    if not columns:
        return 0

    persona_columns = [c for c in columns if c not in MUTABLE_KNOWLEDGE_METADATA_COLUMNS]
    if not persona_columns:
        return count_real_change_rows(cursor, "PersonaChanges", player_id)

    order_column = next((c for c in ("Version", "ID", "Turn") if c in columns), "rowid")
    select_columns = list(persona_columns)
    if "Rationale" in columns:
        select_columns.append("Rationale")

    cursor.execute(
        f"""
        SELECT {", ".join(select_columns)}
        FROM PersonaChanges
        WHERE Key = ?
        ORDER BY {order_column}
        """,
        (player_id,),
    )

    count = 0
    previous_state = None
    for row in cursor.fetchall():
        data = dict(zip(select_columns, row))
        rationale = data.get("Rationale")
        if isinstance(rationale, str) and rationale.startswith("Tweaked by In-Game AI"):
            continue

        state = tuple(data.get(column) for column in persona_columns)
        if previous_state is None or state != previous_state:
            count += 1
            previous_state = state

    return count


def perform_strategy_sanity_checks(cursor, player_id, player_data, db_name, experiment_name):
    """Emit warnings on suspicious strategy-change gaps (ported diagnostics)."""
    if not experiment_name or experiment_name.startswith("none-strategist"):
        return

    try:
        cursor.execute("""
            SELECT Turn, Changes FROM FlavorChanges WHERE Key = ? ORDER BY Turn
        """, (player_id,))
        changes = cursor.fetchall()
        has_flavor_changes = True
    except Exception:
        cursor.execute("""
            SELECT Turn, Changes FROM StrategyChanges WHERE Key = ? ORDER BY Turn
        """, (player_id,))
        changes = cursor.fetchall()
        has_flavor_changes = False

    strategy_turns = [row[0] for row in changes]

    if len(strategy_turns) > 1:
        max_gap = 0
        gap_start_turn = 0
        for i in range(1, len(strategy_turns)):
            gap = strategy_turns[i] - strategy_turns[i - 1]
            if gap > max_gap:
                max_gap = gap
                gap_start_turn = strategy_turns[i - 1]
        if max_gap > 15:
            print(f"  WARNING: Player {player_id} in {db_name} ({experiment_name}) has max strategy change gap of {max_gap} turns starting from turn {gap_start_turn}")

    if strategy_turns and player_data["survival_turn"] != "N/A":
        last_strategy_turn = strategy_turns[-1]
        survival_turn = player_data["survival_turn"]
        final_gap = survival_turn - last_strategy_turn
        if final_gap > 15:
            print(f"  WARNING: Player {player_id} in {db_name} ({experiment_name}) has {final_gap} turns between last strategy change (turn {last_strategy_turn}) and survival (turn {survival_turn})")

    if not has_flavor_changes:
        cursor.execute("""
            SELECT COUNT(*) FROM StrategyChanges
            WHERE Key = ? AND Rationale = 'Tweaked by In-Game AI(Unknown)'
        """, (player_id,))
        in_game_ai_count = cursor.fetchone()[0]
        if in_game_ai_count > 3:
            print(f"  WARNING: Player {player_id} in {db_name} ({experiment_name}) has {in_game_ai_count} strategy changes with 'In-Game AI' rationale")


def extract_player_data(cursor, player_id, player_info_cache, highest_score, victory_player_id, db_name, experiment_name):
    """Extract per-player outcome and strategy fields."""
    player_data = {}

    try:
        if player_id in player_info_cache:
            player_data["civilization"] = player_info_cache[player_id]["civilization"]
        else:
            player_data["civilization"] = "N/A"

        cursor.execute("SELECT MAX(Score) as MaxScore FROM PlayerSummaries WHERE Key = ?", (player_id,))
        max_score_result = cursor.fetchone()
        player_data["score"] = max_score_result[0] if (max_score_result and max_score_result[0] is not None) else "N/A"

        cursor.execute("SELECT Turn FROM PlayerSummaries WHERE Key = ? AND IsLatest = 1", (player_id,))
        survival_result = cursor.fetchone()
        player_data["survival_turn"] = survival_result[0] if (survival_result and survival_result[0] is not None) else "N/A"

        if player_data["score"] != "N/A":
            if highest_score > 0:
                player_data["score_ratio"] = round(player_data["score"] / highest_score, 4)
            else:
                player_data["score_ratio"] = 0 if player_data["score"] == 0 else 1
        else:
            player_data["score_ratio"] = "N/A"

        player_data["is_winner"] = 1 if player_id == victory_player_id else 0

        cursor.execute("""
            SELECT Turn, GrandStrategy, Changes FROM StrategyChanges WHERE Key = ? ORDER BY Turn
        """, (player_id,))
        all_strategy_changes = cursor.fetchall()

        # strategy_changes: turns with an *actual* flavor-number change, i.e. a
        # decision that touched a field other than the rationale (excludes both
        # the status-quo '["Rationale"]' and the truly empty '[]'/null rows).
        cursor.execute("""
            SELECT COUNT(*) FROM FlavorChanges
            WHERE Key = ? AND Changes IS NOT NULL AND Changes NOT IN ('[]', '["Rationale"]')
        """, (player_id,))
        flavor_changes = cursor.fetchone()[0] or 0
        if flavor_changes != 0:
            player_data["strategy_changes"] = flavor_changes
        else:
            player_data["strategy_changes"] = sum(
                1 for row in all_strategy_changes if row[2] not in (None, "[]", '["Rationale"]')
            )

        # decisions: every turn the strategist acted, including a status-quo turn
        # ('["Rationale"]') where it kept everything the same but gave a rationale.
        # Only a truly empty row ('[]'/null) is not a decision.
        cursor.execute("""
            SELECT COUNT(*) FROM FlavorChanges
            WHERE Key = ? AND Changes IS NOT NULL AND Changes != '[]'
        """, (player_id,))
        flavor_decisions = cursor.fetchone()[0] or 0
        if flavor_decisions != 0:
            player_data["decisions"] = flavor_decisions
        else:
            player_data["decisions"] = sum(
                1 for row in all_strategy_changes if row[2] not in (None, "[]")
            )

        strategy_turns = {strategy: 0 for strategy in STRATEGY_MAPPINGS.keys()}
        if all_strategy_changes:
            for i, (turn, grand_strategy, _changes) in enumerate(all_strategy_changes):
                if grand_strategy and grand_strategy in strategy_turns:
                    if i < len(all_strategy_changes) - 1:
                        duration = all_strategy_changes[i + 1][0] - turn
                    else:
                        if player_data["survival_turn"] != "N/A":
                            duration = player_data["survival_turn"] - turn + 1
                        else:
                            duration = 1
                    strategy_turns[grand_strategy] += duration

        total_turns = sum(strategy_turns.values())
        if total_turns > 0:
            for strategy, ratio_field in STRATEGY_MAPPINGS.items():
                player_data[ratio_field] = round(strategy_turns[strategy] / total_turns, 4)
        else:
            for ratio_field in STRATEGY_MAPPINGS.values():
                player_data[ratio_field] = 0

        if player_id in [2, 3]:
            perform_strategy_sanity_checks(cursor, player_id, player_data, db_name, experiment_name)

        player_data["persona_changes"] = count_strategist_persona_state_changes(cursor, player_id)
        player_data["research_changes"] = count_real_change_rows(cursor, "ResearchChanges", player_id)
    except sqlite3.DatabaseError:
        # A corrupt/locked image is a game-level fact, not a per-player one: let
        # it bubble to the game handler so the whole game is recorded once and
        # skipped (rather than emitting an all-N/A row per player).
        raise
    except Exception as exc:
        print(f"Error extracting data for player {player_id}: {exc}")
        for key in ALL_PLAYER_FIELDS:
            if key not in player_data:
                player_data[key] = "N/A"

    return player_data


def extract_game_panel_data(db_path, catalog: Optional[Catalog] = None, issues=None):
    """Extract panel rows (one per major player) for a single game DB.

    A malformed/locked DB is recorded as a single :class:`ImportIssue` and the
    game is skipped (returns ``[]``); no all-N/A rows.
    """
    panel_rows = []
    conn, cursor = open_database_readonly(db_path)
    if not conn:
        if issues is not None:
            issues.record(stage="panel", db_path=db_path, message="could not open database")
        return []

    # Pre-bound so the except handler can report seed/rotation even when the
    # failure happens before (or during) the metadata read.
    metadata: dict = {}
    try:
        metadata = read_game_metadata(cursor)
        experiment = get_experiment_from_path(db_path)
        game_id = metadata.get("gameId", get_game_id_from_path(db_path))

        player_info_cache = get_player_info_cache(cursor)
        major_players = [pid for pid, info in player_info_cache.items() if info["is_major"]]

        # Orthodox identity composed once per (game, player) (§3.3).
        seeding = extract_seeding_fields(metadata, where=f"game {game_id}")
        identities = compose_identities(metadata, major_players, experiment, catalog, seeding)

        cursor.execute("""
            SELECT MAX(ps.Score)
            FROM PlayerSummaries ps
            INNER JOIN PlayerInformations pi ON ps.Key = pi.Key
            WHERE pi.IsMajor = 1
        """)
        highest_score = cursor.fetchone()[0] or 0

        rank_map = calculate_score_ranks(cursor)

        victory_player_id_raw = metadata.get("victoryPlayerID", "N/A")
        try:
            victory_player_id = int(float(victory_player_id_raw))
        except (ValueError, TypeError):
            victory_player_id = -1

        victory_type = metadata.get("victoryType", "N/A")
        turn = metadata.get("turn", "N/A")
        if victory_type == "Cultural" and turn != "N/A":
            try:
                if int(turn) <= 300:
                    victory_type = "Survival"
                    print(f"  Note: Cultural victory at turn {int(turn)} marked as Survival in {os.path.basename(db_path)}")
            except (ValueError, TypeError):
                pass

        for player_id in major_players:
            row = {
                "experiment": experiment,
                # game_id is the dedup/prune key; keep it equal to the filename
                # uuid (what skip/prune use) when metadata omits gameId.
                "game_id": game_id,
                "turn": metadata.get("turn", "N/A"),
                "map_type": metadata.get("mapType", "N/A"),
                "map_size": metadata.get("mapSize", "N/A"),
                "difficulty": metadata.get("difficulty", "N/A"),
                "game_speed": metadata.get("gameSpeed", "N/A"),
                "victory_type": victory_type,
                "victory_player_id": victory_player_id if victory_player_id != -1 else "N/A",
                "player_id": player_id,
            }

            identity = identities.get(player_id, {})
            row["player_type"] = identity.get("player_type")
            row["model"] = identity.get("model", "N/A")
            row["strategist"] = identity.get("strategist", "N/A")
            # Non-treatment seats get the -1 sentinel, never the player_id: the
            # identity always carries config_slot, so this default only guards a
            # truly identity-less row (and must still not resurrect seat=player_id).
            row["config_slot"] = identity.get("config_slot", -1)

            player_data = extract_player_data(
                cursor, player_id, player_info_cache, highest_score,
                victory_player_id, os.path.basename(db_path), experiment,
            )
            row.update(player_data)
            row["score_rank"] = rank_map.get(player_id, "N/A")
            panel_rows.append(row)

        conn.close()

        if "experiment" in metadata and not experiment.startswith(metadata["experiment"]):
            print(f"WARNING: Folder name '{experiment}' does not start with metadata experiment '{metadata['experiment']}' in {os.path.basename(db_path)}")

        return panel_rows
    except ExtractError:
        # A controlled-seed mismatch (rule 14) is a hard policy abort: it must
        # propagate even on a panel-only run, never be swallowed into [].
        conn.close()
        raise
    except sqlite3.DatabaseError as exc:
        # Malformed/locked image: record once and skip the game (no N/A rows).
        record_db_failure(issues, stage="panel", db_path=db_path, exc=exc, metadata=metadata)
        conn.close()
        return []
    except Exception as exc:
        # Other non-DB faults keep the prior skip-and-continue behavior; they are
        # not malformed-DB issues.
        print(f"Error processing {os.path.basename(db_path)}: {exc}")
        conn.close()
        return []


def export_panel_data(db_files, available_game_ids, output_file, catalog: Optional[Catalog] = None, prune_only=False, issues=None) -> int:
    """Export panel rows to ``output_file``; returns the count of new rows."""
    def _extract_rows(db_file, issues):
        return extract_game_panel_data(db_file, catalog=catalog, issues=issues)

    return run_table_export(
        db_files, available_game_ids, output_file,
        stage="panel", fieldnames=list(PANEL_FIELD_MAPPINGS.keys()),
        extract_rows=_extract_rows,
        dedupe_key=lambda r: (r.get("game_id"), r.get("player_id")),
        describe_db=lambda db_file, rows: f"{len(rows)} player rows",
        noun="panel rows", prune_only=prune_only, issues=issues,
    )
