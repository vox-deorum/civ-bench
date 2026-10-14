"""Per-game ``world`` extraction (benchmark.md §3.0).

One row per game with whole-world totals: how many wars, peace deals, combat
losses, conquests, razes, nukes, and eliminations the game saw among its civs
(major players). The counters reuse the behavior events family's rules, so a
world total matches the sum of its civs' own counts, except where one event
involves two civs (a war or a peace deal counts once).
"""

from __future__ import annotations

import os
import sqlite3

from .behavior import build_game_context
from .behavior.events import EVENT_COUNTERS, EventCounter, count_game_events
from .errors import ExtractError
from .export_common import run_table_export
from .game_events import fetch_events, payload_get, unique_events
from .issues import record_db_failure
from .utilities import (
    get_experiment_from_path, get_game_id_from_path, get_player_info_cache,
    open_database_readonly, read_game_metadata,
)

# World column → the counter whose events it counts (once per event).
WORLD_COUNTERS = {
    # Wars between civs: an aggressor's declaration on another civ's team.
    "wars_declared": EventCounter("DeclareWar", actor="OriginatingPlayerID",
                                  match=(("IsAggressor", True),),
                                  target="TargetTeamID", target_is_team=True,
                                  target_kind="major"),
    **{name: EVENT_COUNTERS[name] for name in (
        "wars_on_city_states", "peace_deals", "units_lost", "cities_captured",
        "capitals_captured", "city_state_cities_captured", "cities_razed",
        "nukes_detonated", "cities_nuked",
    )},
}
_WAR_STATE_TYPES = ("DeclareWar", "MakePeace")

WORLD_FIELDNAMES = [
    "experiment", "game_id", "final_turn", "n_civs",
    "wars_declared", "wars_on_city_states", "peace_deals", "peak_wars",
    "units_lost", "cities_captured", "capitals_captured", "city_state_cities_captured",
    "civs_eliminated", "cities_razed", "nukes_detonated", "cities_nuked",
]


def peak_wars(events, ctx, last_alive: dict) -> int:
    """The most wars between civ teams open at once.

    Replays every declaration and peace deal between two civ teams in turn
    order. A war ends at a peace deal, or once every civ on one side is gone
    (its last turn with a positive score in ``last_alive`` has passed).
    """
    def team_gone(team, turn) -> bool:
        players = ctx.players_on_team(team)
        return bool(players) and all(last_alive.get(pid, -1) < turn for pid in players)

    open_pairs: set = set()
    peak = 0
    ordered = sorted(
        (e for e in events if e[0] in _WAR_STATE_TYPES),
        key=lambda event: event[1] if event[1] is not None else -1,
    )
    for event_type, turn, payload in ordered:
        origin = payload_get(payload, "OriginatingPlayerID")
        pair = frozenset((ctx.team_of.get(origin, origin), payload_get(payload, "TargetTeamID")))
        if len(pair) != 2 or any(ctx.team_kind.get(team) != "major" for team in pair):
            continue
        current = turn if turn is not None else -1
        open_pairs = {p for p in open_pairs if not any(team_gone(t, current) for t in p)}
        if event_type == "DeclareWar":
            open_pairs.add(pair)
            peak = max(peak, len(open_pairs))
        else:
            open_pairs.discard(pair)
    return peak


def extract_game_world_data(db_path, issues=None) -> list[dict]:
    """The world row for one game DB, as a one-item list (empty when skipped)."""
    conn, cursor = open_database_readonly(db_path)
    if not conn:
        if issues is not None:
            issues.record(stage="world", db_path=db_path, message="could not open database")
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
        ctx = build_game_context(cursor, major_players)

        cursor.execute("SELECT MAX(Turn) FROM PlayerSummaries")
        final_turn = cursor.fetchone()[0]
        # Each player's last turn alive, reading only the latest summary of
        # each player and turn (the highest ID), as the turns table does: a
        # replayed turn's positive score does not count once a later row of
        # that turn has a score of 0.
        placeholders = ",".join(["?"] * len(major_players))
        cursor.execute(
            f"""
            SELECT ps.Key, MAX(ps.Turn)
            FROM PlayerSummaries ps
            JOIN (
                SELECT MAX(ID) AS MaxID FROM PlayerSummaries
                WHERE Key IN ({placeholders}) GROUP BY Key, Turn
            ) latest ON ps.ID = latest.MaxID
            WHERE ps.Score > 0
            GROUP BY ps.Key
            """,
            major_players,
        )
        last_alive = {pid: turn for pid, turn in cursor.fetchall() if turn is not None}
        # A civ is eliminated when its final-turn summary is missing or has a
        # score of 0, as in the turns table.
        eliminated = sum(1 for pid in major_players if last_alive.get(pid) != final_turn)

        types = {c.type for c in WORLD_COUNTERS.values()} | set(_WAR_STATE_TYPES)
        events = unique_events(fetch_events(cursor, sorted(types)))
        counts = count_game_events(events, WORLD_COUNTERS, ctx)
        conn.close()

        row = {
            "experiment": experiment,
            "game_id": game_id,
            "final_turn": final_turn if final_turn is not None else "",
            "n_civs": len(major_players),
            **counts,
            "peak_wars": peak_wars(events, ctx, last_alive),
            "civs_eliminated": eliminated if final_turn is not None else "",
        }
        return [{name: row[name] for name in WORLD_FIELDNAMES}]
    except ExtractError:
        conn.close()
        raise
    except sqlite3.DatabaseError as exc:
        record_db_failure(issues, stage="world", db_path=db_path, exc=exc, metadata=metadata)
        conn.close()
        return []
    except Exception as exc:
        print(f"Error processing {os.path.basename(db_path)}: {exc}")
        conn.close()
        return []


def export_world_data(db_files, available_game_ids, output_file, prune_only=False,
                      issues=None, prune_gate=None) -> int:
    """Export world rows to ``output_file``; returns the count of new rows."""
    return run_table_export(
        db_files, available_game_ids, output_file,
        stage="world", fieldnames=WORLD_FIELDNAMES,
        extract_rows=lambda db_file, issues: extract_game_world_data(db_file, issues=issues),
        dedupe_key=lambda r: (r.get("game_id"),),
        describe_db=lambda db_file, rows: "1 game row" if rows else "no game row",
        noun="world rows", prune_only=prune_only, issues=issues,
        prune_gate=prune_gate,
    )
