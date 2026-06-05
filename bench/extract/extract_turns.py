"""Per-player-per-turn ``turn_data`` extraction (ported from the analysis repo).

Departure from the source (``plans/stage1.md``): each row carries the orthodox
``player_type`` (composed once per (game, player) and broadcast across that
player's turns, §3.3). It stores **no** ``seed`` and no seating columns — the
strength stage joins ``seed`` from ``game_data`` by ``game_id`` where it needs the
start-cell. Everything else (the carry-forward flavor/strategy logic) is verbatim.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from ..catalog import Catalog
from .identity import compose_identities
from .utilities import (
    append_csv_file, extract_seeding_fields, filter_existing_data,
    get_experiment_from_path, get_game_id_from_path, get_player_info_cache,
    open_database_readonly, read_existing_csv, read_game_metadata,
    should_skip_game, write_csv_file,
)

# Mapping from FlavorChanges DB column (PascalCase) to CSV column (snake_case)
FLAVOR_COLUMNS = [
    ("Offense", "flavor_offense"),
    ("Defense", "flavor_defense"),
    ("Mobilization", "flavor_mobilization"),
    ("CityDefense", "flavor_city_defense"),
    ("MilitaryTraining", "flavor_military_training"),
    ("Recon", "flavor_recon"),
    ("Ranged", "flavor_ranged"),
    ("Mobile", "flavor_mobile"),
    ("Nuke", "flavor_nuke"),
    ("UseNuke", "flavor_use_nuke"),
    ("Naval", "flavor_naval"),
    ("NavalRecon", "flavor_naval_recon"),
    ("NavalGrowth", "flavor_naval_growth"),
    ("NavalTileImprovement", "flavor_naval_tile_improvement"),
    ("Air", "flavor_air"),
    ("AirCarrier", "flavor_air_carrier"),
    ("Antiair", "flavor_antiair"),
    ("Airlift", "flavor_airlift"),
    ("Expansion", "flavor_expansion"),
    ("Growth", "flavor_growth"),
    ("TileImprovement", "flavor_tile_improvement"),
    ("Infrastructure", "flavor_infrastructure"),
    ("Production", "flavor_production"),
    ("WaterConnection", "flavor_water_connection"),
    ("Gold", "flavor_gold"),
    ("Science", "flavor_science"),
    ("Culture", "flavor_culture"),
    ("Happiness", "flavor_happiness"),
    ("GreatPeople", "flavor_great_people"),
    ("Wonder", "flavor_wonder"),
    ("Religion", "flavor_religion"),
    ("Diplomacy", "flavor_diplomacy"),
    ("Spaceship", "flavor_spaceship"),
    ("Espionage", "flavor_espionage"),
]

# Define field mappings for turn-based data
TURN_FIELD_MAPPINGS = {
    "experiment": None,
    "game_id": None,
    "player_id": None,
    "player_type": None,       # Orthodox identity (§3.3), broadcast per (game, player)
    "civilization": None,
    "turn": None,
    "max_turn": None,
    "score": None,
    "rank": None,
    "max_score": None,
    "cities": None,
    "population": None,
    "territory": None,
    "technologies": None,
    "military_strength": None,
    "military_units": None,
    "military_supply": None,
    "gold": None,
    "gold_per_turn": None,
    "production_per_turn": None,
    "food_per_turn": None,
    "happiness_percentage": None,
    "culture_per_turn": None,
    "science_per_turn": None,
    "tourism_per_turn": None,
    "faith_per_turn": None,
    "policies": None,
    "votes": None,
    "religion_percentage": None,
    "minor_allies": None,
    "highest_war_weariness": None,
    "active_wars": None,
    "truces": None,
    "friendships": None,
    "defensive_pacts": None,
    "is_winner": None,
    "is_changed": None,
    **{csv_col: None for _, csv_col in FLAVOR_COLUMNS},
    "grand_strategy": None,
    "rationale": None,
}


def _fetch_flavor_events(cursor, major_players):
    """Pre-fetch deduplicated FlavorChanges events per player (carry-forward source)."""
    placeholders = ",".join(["?"] * len(major_players))
    db_cols = ", ".join(f"fc.{db_col}" for db_col, _ in FLAVOR_COLUMNS)

    try:
        cursor.execute(f"""
            SELECT fc.Key, fc.Turn, {db_cols},
                   fc.GrandStrategy, fc.Rationale, fc.Changes
            FROM FlavorChanges fc
            WHERE fc.Key IN ({placeholders})
            ORDER BY fc.Key, fc.Turn, fc.ID
        """, major_players)
    except Exception:
        return {}

    latest = {}
    for row in cursor.fetchall():
        player_id = row[0]
        turn = row[1]
        flavor_values = row[2:2 + len(FLAVOR_COLUMNS)]
        grand_strategy = row[2 + len(FLAVOR_COLUMNS)]
        rationale = row[3 + len(FLAVOR_COLUMNS)]
        changes_json = row[4 + len(FLAVOR_COLUMNS)]
        is_changed = 0 if changes_json in ("[]", '["Rationale"]') else 1

        key = (player_id, turn)
        existing = latest.get(key)
        if existing is None:
            latest[key] = (flavor_values, grand_strategy, rationale, changes_json, is_changed)
        elif existing[4] == 1 and is_changed == 0:
            continue
        else:
            displaced = existing
            latest[key] = (flavor_values, grand_strategy, rationale, changes_json, is_changed)
            cascade_turn = turn - 1
            while displaced is not None:
                prev_key = (player_id, cascade_turn)
                prev_existing = latest.get(prev_key)
                if prev_existing is None:
                    latest[prev_key] = displaced
                    break
                elif prev_existing[4] == 1 and displaced[4] == 0:
                    break
                else:
                    latest[prev_key] = displaced
                    displaced = prev_existing
                    cascade_turn -= 1

    result = {}
    for (player_id, turn), entry in sorted(latest.items()):
        result.setdefault(player_id, []).append((turn, *entry))
    return result


def _build_flavor_lookup(flavor_events, major_players, all_turns):
    """Carry-forward flavor lookup ``{(player_id, turn): (...)}`` from event lists."""
    lookup = {}
    for player_id in major_players:
        events = flavor_events.get(player_id, [])
        if not events:
            continue
        event_idx = 0
        n_events = len(events)
        for turn in all_turns:
            while event_idx < n_events and events[event_idx][0] <= turn:
                event_idx += 1
            if event_idx == 0:
                continue
            ev_turn, ev_flavors, ev_gs, ev_rationale, ev_changes, ev_is_changed = events[event_idx - 1]
            if ev_turn == turn:
                lookup[(player_id, turn)] = (ev_flavors, ev_gs, ev_rationale, ev_changes, ev_is_changed)
            else:
                lookup[(player_id, turn)] = (ev_flavors, ev_gs, "", "", 0)
    return lookup


def _fetch_strategy_grand_strategy(cursor, major_players):
    """GrandStrategy events from StrategyChanges, for players with no FlavorChanges."""
    placeholders = ",".join(["?"] * len(major_players))
    try:
        cursor.execute(f"""
            SELECT Key, Turn, GrandStrategy
            FROM StrategyChanges
            WHERE Key IN ({placeholders})
            ORDER BY Key, Turn, ID
        """, major_players)
    except Exception:
        return {}

    latest = {}
    for player_id, turn, gs in cursor.fetchall():
        latest[(player_id, turn)] = gs

    result = {}
    for (player_id, turn), gs in sorted(latest.items()):
        result.setdefault(player_id, []).append((turn, gs))
    return result


def _build_strategy_lookup(strategy_events, players_needing_fallback, all_turns):
    """Carry-forward grand-strategy-only lookup for fallback players."""
    lookup = {}
    for player_id in players_needing_fallback:
        events = strategy_events.get(player_id, [])
        if not events:
            continue
        event_idx = 0
        n_events = len(events)
        for turn in all_turns:
            while event_idx < n_events and events[event_idx][0] <= turn:
                event_idx += 1
            if event_idx == 0:
                continue
            ev_turn, ev_gs = events[event_idx - 1]
            if ev_gs and ev_gs != "None":
                lookup[(player_id, turn)] = ev_gs
    return lookup


def extract_game_turn_data(db_path, catalog: Optional[Catalog] = None):
    """Extract per-player-per-turn rows for a single game DB."""
    turn_data = []
    conn, cursor = open_database_readonly(db_path)
    if not conn:
        return []

    try:
        experiment_name = get_experiment_from_path(db_path)
        metadata = read_game_metadata(cursor)
        game_id = metadata.get("gameId", get_game_id_from_path(db_path))

        player_info_cache = get_player_info_cache(cursor)
        major_players = [pid for pid, info in player_info_cache.items() if info["is_major"]]
        if not major_players:
            print(f"  No major players found in {os.path.basename(db_path)}")
            return []

        # Orthodox identity composed once per (game, player), then broadcast (§3.3).
        seeding = extract_seeding_fields(metadata, where=f"game {game_id}")
        identities = compose_identities(metadata, major_players, experiment_name, catalog, seeding)
        player_type_by_id = {pid: ident.get("player_type") for pid, ident in identities.items()}

        cursor.execute("SELECT MAX(Turn) FROM PlayerSummaries")
        max_turn = cursor.fetchone()[0]
        if max_turn is None:
            print(f"  No turn data found in {os.path.basename(db_path)}")
            return []

        victory_player_id_raw = metadata.get("victoryPlayerID", "N/A")
        try:
            victory_player_id = int(float(victory_player_id_raw))
        except (ValueError, TypeError):
            victory_player_id = -1

        cursor.execute("""
            SELECT Turn, MajorityReligion, COUNT(*) as city_count
            FROM CityInformations
            WHERE MajorityReligion IS NOT NULL
            GROUP BY Turn, MajorityReligion
        """)
        city_religion_by_turn = {}
        for turn, religion, count in cursor.fetchall():
            city_religion_by_turn.setdefault(turn, {})[religion] = count

        cursor.execute("SELECT Turn, COUNT(*) as total_cities FROM CityInformations GROUP BY Turn")
        total_cities_by_turn = dict(cursor.fetchall())

        cursor.execute("""
            SELECT ps.Turn, ps.MajorAlly, COUNT(*) as ally_count
            FROM PlayerSummaries ps
            INNER JOIN PlayerInformations pi ON ps.Key = pi.Key
            WHERE pi.IsMajor = 0 AND ps.MajorAlly IS NOT NULL
            GROUP BY ps.Turn, ps.MajorAlly
        """)
        allies_by_turn = {}
        for turn, civ, count in cursor.fetchall():
            allies_by_turn.setdefault(turn, {})[civ] = count

        placeholders = ",".join(["?"] * len(major_players))
        cursor.execute("""
            SELECT ci.Turn, pi.Key,
                   SUM(ci.ProductionPerTurn) as total_production,
                   SUM(ci.FoodPerTurn) as total_food
            FROM CityInformations ci
            INNER JOIN PlayerInformations pi ON ci.Owner = pi.Civilization
            WHERE pi.Key IN ({})
            GROUP BY ci.Turn, pi.Key
        """.format(placeholders), major_players)
        city_yields_by_turn = {}
        for turn, player_id, production, food in cursor.fetchall():
            city_yields_by_turn.setdefault(turn, {})[player_id] = {
                "production": production if production is not None else 0,
                "food": food if food is not None else 0,
            }

        cursor.execute("SELECT DISTINCT Turn FROM PlayerSummaries ORDER BY Turn")
        all_turns = [row[0] for row in cursor.fetchall()]

        flavor_events = _fetch_flavor_events(cursor, major_players)
        flavor_by_player_turn = _build_flavor_lookup(flavor_events, major_players, all_turns)

        players_needing_fallback = [p for p in major_players if p not in flavor_events]
        if players_needing_fallback:
            strategy_events = _fetch_strategy_grand_strategy(cursor, players_needing_fallback)
            strategy_gs_lookup = _build_strategy_lookup(strategy_events, players_needing_fallback, all_turns)
        else:
            strategy_gs_lookup = {}

        cursor.execute(f"""
            WITH all_turns AS (
                SELECT DISTINCT Turn FROM PlayerSummaries
            ),
            all_player_turns AS (
                SELECT t.Turn, p.Key, p.Civilization
                FROM all_turns t
                CROSS JOIN (
                    SELECT Key, Civilization
                    FROM PlayerInformations
                    WHERE Key IN ({placeholders})
                ) p
            ),
            latest_summaries AS (
                SELECT Turn, Key, MAX(ID) as MaxID
                FROM PlayerSummaries
                WHERE Key IN ({placeholders})
                GROUP BY Turn, Key
            )
            SELECT
                apt.Turn,
                apt.Key,
                COALESCE(ps.Score, 0) as Score,
                COALESCE(ps.Cities, 0) as Cities,
                COALESCE(ps.Population, 0) as Population,
                COALESCE(ps.Territory, 0) as Territory,
                COALESCE(ps.Technologies, 0) as Technologies,
                COALESCE(ps.MilitaryStrength, 0) as MilitaryStrength,
                COALESCE(ps.MilitaryUnits, 0) as MilitaryUnits,
                COALESCE(ps.MilitarySupply, 0) as MilitarySupply,
                COALESCE(ps.Gold, 0) as Gold,
                COALESCE(ps.GoldPerTurn, 0) as GoldPerTurn,
                COALESCE(ps.HappinessPercentage, 0) as HappinessPercentage,
                COALESCE(ps.CulturePerTurn, 0) as CulturePerTurn,
                COALESCE(ps.SciencePerTurn, 0) as SciencePerTurn,
                COALESCE(ps.TourismPerTurn, 0) as TourismPerTurn,
                COALESCE(ps.FaithPerTurn, 0) as FaithPerTurn,
                ps.PolicyBranches,
                COALESCE(ps.Votes, 0) as Votes,
                ps.FoundedReligion,
                apt.Civilization,
                ps.Relationships
            FROM all_player_turns apt
            LEFT JOIN latest_summaries ls
                ON apt.Turn = ls.Turn AND apt.Key = ls.Key
            LEFT JOIN PlayerSummaries ps
                ON ls.Turn = ps.Turn AND ls.Key = ps.Key AND ls.MaxID = ps.ID
            ORDER BY apt.Turn, ps.Score DESC
        """, major_players + major_players)

        current_turn = None
        turn_players = []

        for row in cursor.fetchall():
            (turn_num, player_id, score, cities, pop, territory, tech, military,
             military_units, military_supply,
             gold, gold_per_turn, happiness_percentage, culture_per_turn, science_per_turn,
             tourism_per_turn, faith_per_turn, policy_branches_json, votes, founded_religion,
             civilization, relationships_json) = row

            if current_turn != turn_num:
                if turn_players:
                    process_turn_group(
                        turn_players, turn_data, experiment_name, game_id,
                        max_turn, player_info_cache, victory_player_id,
                        city_religion_by_turn, total_cities_by_turn, allies_by_turn,
                        city_yields_by_turn, flavor_by_player_turn, strategy_gs_lookup,
                        player_type_by_id,
                    )
                turn_players = []
                current_turn = turn_num

            policies_count = 0
            if policy_branches_json:
                try:
                    policy_branches = json.loads(policy_branches_json)
                    for branch_policies in policy_branches.values():
                        if isinstance(branch_policies, list):
                            policies_count += len(branch_policies)
                except (json.JSONDecodeError, ValueError):
                    policies_count = 0

            highest_war_weariness = 0
            active_wars = 0
            truces = 0
            friendships = 0
            defensive_pacts = 0
            if relationships_json:
                try:
                    relationships = json.loads(relationships_json)
                    for _civ, items in relationships.items():
                        if not isinstance(items, list):
                            continue
                        for item in items:
                            if item.startswith("War "):
                                active_wars += 1
                                m = re.search(r"War Weariness: (\d+)%", item)
                                if m:
                                    ww = int(m.group(1))
                                    if ww > highest_war_weariness:
                                        highest_war_weariness = ww
                            elif item == "Peace Treaty":
                                truces += 1
                            elif item == "Declaration of Friendship":
                                friendships += 1
                            elif item == "Defensive Pact":
                                defensive_pacts += 1
                except (json.JSONDecodeError, ValueError):
                    pass

            turn_players.append({
                "turn": turn_num,
                "player_id": player_id,
                "score": score if score is not None else 0,
                "cities": cities if cities is not None else 0,
                "population": pop if pop is not None else 0,
                "territory": territory if territory is not None else 0,
                "technologies": tech if tech is not None else 0,
                "military_strength": military if military is not None else 0,
                "military_units": military_units if military_units is not None else 0,
                "military_supply": military_supply if military_supply is not None else 0,
                "gold": gold if gold is not None else 0,
                "gold_per_turn": gold_per_turn if gold_per_turn is not None else 0,
                "happiness_percentage": happiness_percentage if happiness_percentage is not None else 0,
                "culture_per_turn": culture_per_turn if culture_per_turn is not None else 0,
                "science_per_turn": science_per_turn if science_per_turn is not None else 0,
                "tourism_per_turn": tourism_per_turn if tourism_per_turn is not None else 0,
                "faith_per_turn": faith_per_turn if faith_per_turn is not None else 0,
                "policies": policies_count,
                "votes": votes if votes is not None else 0,
                "religion_percentage": founded_religion,
                "civilization": civilization,
                "highest_war_weariness": highest_war_weariness,
                "active_wars": active_wars,
                "truces": truces,
                "friendships": friendships,
                "defensive_pacts": defensive_pacts,
            })

        if turn_players:
            process_turn_group(
                turn_players, turn_data, experiment_name, game_id,
                max_turn, player_info_cache, victory_player_id,
                city_religion_by_turn, total_cities_by_turn, allies_by_turn,
                city_yields_by_turn, flavor_by_player_turn, strategy_gs_lookup,
                player_type_by_id,
            )

        conn.close()
        return turn_data
    except Exception as exc:
        print(f"Error processing {os.path.basename(db_path)}: {exc}")
        conn.close()
        return []


def process_turn_group(turn_players, turn_data, experiment_name, game_id, max_turn, player_info_cache, victory_player_id,
                       city_religion_by_turn, total_cities_by_turn, allies_by_turn, city_yields_by_turn,
                       flavor_by_player_turn, strategy_gs_lookup=None, player_type_by_id=None):
    """Rank a single turn's players and append their full records to ``turn_data``."""
    if not turn_players:
        return

    player_type_by_id = player_type_by_id or {}

    alive_players = [p for p in turn_players if p["score"] > 0]
    eliminated_players = [p for p in turn_players if p["score"] == 0]
    max_score = alive_players[0]["score"] if alive_players else 0

    current_rank = 1
    last_score = None
    players_with_rank = []
    for i, player_data in enumerate(alive_players):
        if last_score is not None and player_data["score"] != last_score:
            current_rank = i + 1
        players_with_rank.append({**player_data, "rank": current_rank})
        last_score = player_data["score"]

    worst_rank = len(turn_players)
    for player_data in eliminated_players:
        players_with_rank.append({**player_data, "rank": worst_rank})

    for player_info in players_with_rank:
        player_id = player_info["player_id"]
        turn_num = player_info["turn"]
        civilization = player_info["civilization"]
        founded_religion = player_info["religion_percentage"]

        religion_percentage = 0
        if founded_religion and founded_religion != "Pantheon (Religion Possible)":
            total_cities = total_cities_by_turn.get(turn_num, 0)
            if total_cities > 0:
                cities_with_religion = city_religion_by_turn.get(turn_num, {}).get(founded_religion, 0)
                religion_percentage = round((cities_with_religion / total_cities) * 100, 2)

        minor_allies_count = allies_by_turn.get(turn_num, {}).get(civilization, 0)
        city_yields = city_yields_by_turn.get(turn_num, {}).get(player_id, {"production": 0, "food": 0})
        production_per_turn = city_yields.get("production", 0)
        food_per_turn = city_yields.get("food", 0)

        flavor_entry = flavor_by_player_turn.get((player_id, turn_num))
        if flavor_entry is not None:
            flavor_values, grand_strategy, rationale, flavor_changes, is_changed = flavor_entry
        else:
            flavor_values = (None,) * len(FLAVOR_COLUMNS)
            grand_strategy = (strategy_gs_lookup or {}).get((player_id, turn_num))
            rationale = None
            flavor_changes = None
            is_changed = 0

        record = {
            "experiment": experiment_name,
            "game_id": game_id,
            "player_id": player_id,
            "player_type": player_type_by_id.get(player_id),
            "civilization": civilization,
            "turn": turn_num,
            "max_turn": max_turn,
            "score": player_info["score"],
            "rank": player_info["rank"],
            "max_score": max_score,
            "cities": player_info["cities"],
            "population": player_info["population"],
            "territory": player_info["territory"],
            "technologies": player_info["technologies"],
            "military_strength": player_info["military_strength"],
            "military_units": player_info["military_units"],
            "military_supply": player_info["military_supply"],
            "gold": player_info["gold"],
            "gold_per_turn": player_info["gold_per_turn"],
            "production_per_turn": production_per_turn,
            "food_per_turn": food_per_turn,
            "happiness_percentage": player_info["happiness_percentage"],
            "culture_per_turn": player_info["culture_per_turn"],
            "science_per_turn": player_info["science_per_turn"],
            "tourism_per_turn": player_info["tourism_per_turn"],
            "faith_per_turn": player_info["faith_per_turn"],
            "policies": player_info["policies"],
            "votes": player_info["votes"],
            "religion_percentage": religion_percentage,
            "minor_allies": minor_allies_count,
            "highest_war_weariness": player_info["highest_war_weariness"],
            "active_wars": player_info["active_wars"],
            "truces": player_info["truces"],
            "friendships": player_info["friendships"],
            "defensive_pacts": player_info["defensive_pacts"],
            "is_winner": 1 if player_id == victory_player_id else 0,
            "is_changed": is_changed,
        }

        for (_, csv_col), value in zip(FLAVOR_COLUMNS, flavor_values):
            record[csv_col] = value if value is not None else ""

        record["grand_strategy"] = grand_strategy if grand_strategy is not None else ""
        record["rationale"] = rationale if rationale is not None else ""

        turn_data.append(record)


def export_turn_data(db_files, available_game_ids, output_file, catalog: Optional[Catalog] = None, prune_only=False) -> int:
    """Export turn rows to ``output_file``; returns the count of new rows."""
    expected_fieldnames = list(TURN_FIELD_MAPPINGS.keys())
    existing_data, existing_game_ids, structure_matches = read_existing_csv(output_file, expected_fieldnames)
    pruned_rows = 0

    if not structure_matches:
        print("Discarding existing turn data due to structure mismatch...")
        existing_data = []
        existing_game_ids = set()
    else:
        existing_data, existing_game_ids, pruned_rows, pruned_game_ids = filter_existing_data(
            existing_data, available_game_ids
        )
        if pruned_rows > 0:
            print(f"  Filtered out {pruned_rows} turn records from {len(pruned_game_ids)} games without database files")

        seen = {}
        for i, row in enumerate(existing_data):
            seen[(row.get("game_id"), row.get("player_id"), row.get("turn"))] = i
        if len(seen) < len(existing_data):
            deduped_count = len(existing_data) - len(seen)
            existing_data = [existing_data[i] for i in sorted(seen.values())]
            existing_game_ids = {r["game_id"] for r in existing_data if r.get("game_id") and r["game_id"] != "N/A"}
            pruned_rows += deduped_count
            print(f"  Removed {deduped_count} duplicate turn records")

        print(f"Found {len(existing_data)} existing turn records from {len(existing_game_ids)} games")

    new_turn_data = []
    skipped_count = 0
    processed_count = 0
    print("\nExtracting turn-based data...")
    if prune_only:
        print("Prune-only mode: skipping extraction of new turn rows.")
    else:
        for db_file in db_files:
            game_id = get_game_id_from_path(db_file)
            if game_id and should_skip_game(game_id, existing_game_ids):
                skipped_count += 1
                continue
            turn_rows = extract_game_turn_data(db_file, catalog=catalog)
            if turn_rows:
                new_turn_data.extend(turn_rows)
                processed_count += 1
                unique_turns = len(set(row["turn"] for row in turn_rows))
                num_players = len(set(row["player_id"] for row in turn_rows))
                changed_rows = sum(1 for row in turn_rows if row["is_changed"])
                print(f"Processed: {os.path.basename(db_file)} ({num_players} players × {unique_turns} turns = {len(turn_rows)} records, {changed_rows} with flavor changes)")

    print(f"\nProcessed {processed_count} new databases")
    print(f"Skipped {skipped_count} databases that were already exported")

    all_turn_data = existing_data + new_turn_data
    needs_rewrite = pruned_rows > 0 or not structure_matches
    if needs_rewrite and (new_turn_data or pruned_rows > 0):
        if write_csv_file(output_file, expected_fieldnames, all_turn_data):
            print(f"\nRewrote {len(all_turn_data)} turn records to {output_file}")
    elif new_turn_data:
        if append_csv_file(output_file, expected_fieldnames, new_turn_data):
            print(f"\nAppended {len(new_turn_data)} new turn records to {output_file}")
    else:
        print(f"\nNo new turn data to export. Existing file contains {len(existing_data)} records.")

    return len(new_turn_data)
