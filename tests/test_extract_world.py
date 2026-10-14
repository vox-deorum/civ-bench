"""World table extraction (benchmark.md §3.0) on small synthetic game DBs."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from bench.config import schema as S
from bench.extract.extract_world import (
    WORLD_FIELDNAMES,
    export_world_data,
    extract_game_world_data,
    peak_wars,
)


def _event(event_type, turn, payload):
    return (turn, event_type, json.dumps(payload, sort_keys=True))


def _war(turn, origin, target, aggressor=True):
    return _event("DeclareWar", turn, {"OriginatingPlayerID": origin, "TargetTeamID": target,
                                       "IsAggressor": aggressor})


def _peace(turn, origin, target):
    return _event("MakePeace", turn, {"OriginatingPlayerID": origin, "TargetTeamID": target})


def _capture(turn, old, new, *, conquest=True, capital=False, city=1):
    return _event("CityCaptureComplete", turn, {
        "OldOwnerID": old, "NewOwnerID": new, "IsConquest": conquest, "IsCapital": capital,
        "City": {"CityID": city},
    })


def _make_db(path: Path, events, summaries) -> Path:
    """Majors 0, 1, and 2 on their own teams; 22 is a city-state, 63 the barbarians.

    ``summaries`` are ``(key, turn, score)`` rows of ``PlayerSummaries``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE GameMetadata (Key TEXT, Value TEXT)")
    cur.execute("INSERT INTO GameMetadata VALUES ('gameId', ?)", (path.stem.split("_")[0],))
    cur.execute("CREATE TABLE PlayerInformations (Key INTEGER, Civilization TEXT, TeamID INTEGER, IsMajor INTEGER)")
    cur.executemany("INSERT INTO PlayerInformations VALUES (?, ?, ?, ?)", [
        (0, "Rome", 0, 1), (1, "Greece", 1, 1), (2, "Egypt", 2, 1),
        (22, "Ife", 22, 0), (63, "Barbarians", 63, 0),
    ])
    cur.execute("CREATE TABLE PlayerSummaries (ID INTEGER, Key INTEGER, Turn INTEGER, Score INTEGER)")
    cur.executemany("INSERT INTO PlayerSummaries VALUES (?, ?, ?, ?)",
                    [(i, *row) for i, row in enumerate(summaries, start=1)])
    cur.execute("CREATE TABLE GameEvents (ID INTEGER PRIMARY KEY, Turn INTEGER, Type TEXT, Payload TEXT, Player0 INTEGER)")
    cur.execute('CREATE INDEX "idx_gameevents_player0" ON "GameEvents" ("ID", "Turn", "Type", "Player0")')
    cur.executemany("INSERT INTO GameEvents (Turn, Type, Payload, Player0) VALUES (?, ?, ?, 0)", events)
    conn.commit()
    conn.close()
    return path


# Egypt (2) has score 0 on the final turn; Greece (1) has no final-turn row.
SUMMARIES = [
    (0, 10, 100), (0, 30, 300),
    (1, 10, 90), (1, 20, 80),
    (2, 10, 50), (2, 30, 0),
    (22, 30, 10),
]

EVENTS = [
    _war(2, 0, 1),
    _war(2, 0, 1),                      # exact replay duplicate: once
    _war(3, 0, 2),                      # two wars open at once
    _war(3, 2, 1, aggressor=False),     # a vassal following: a war, not a declaration
    _war(4, 1, 22),                     # on a city-state
    _peace(5, 0, 1), _peace(5, 1, 0), _peace(5, 0, 1),   # one deal, logged three times
    _peace(5, 1, 22),                   # with a city-state: not a deal between civs
    _peace(5, 22, 0),                   # logged by the city-state: not one either
    _war(6, 0, 1),                      # Rome and Greece fight again
    _event("UnitKilledInCombat", 6, {"KillerPlayerID": 0, "KilledPlayerID": 1, "KilledUnitType": "Warrior"}),
    _event("UnitKilledInCombat", 6, {"KillerPlayerID": 1, "KilledPlayerID": 63, "KilledUnitType": "Brute"}),
    _capture(7, 1, 0, capital=True, city=1),
    _capture(7, 22, 0, city=2),
    _capture(8, 63, 2, city=3),         # retaken from barbarians: neither kind
    _capture(8, 1, 63, city=4),         # taken by barbarians: no civ conquered it
    _capture(9, 0, 2, conquest=False, city=5),
    _event("CityRazed", 9, {"PlayerID": 0}),
    _event("NuclearDetonation", 9, {"PlayerID": 0, "Plot": {"City": "Athens"}}),
    _event("NuclearDetonation", 9, {"PlayerID": 0, "Plot": {"PlotType": "Land"}}),
]


def test_world_row_counts_every_event_once(tmp_path):
    db = _make_db(tmp_path / "exp" / "g_1.db", EVENTS, SUMMARIES)
    rows = extract_game_world_data(str(db))
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == WORLD_FIELDNAMES
    assert row == {
        "experiment": "exp", "game_id": "g", "final_turn": 30, "n_civs": 3,
        "wars_declared": 3, "wars_on_city_states": 1, "peace_deals": 1,
        # Rome-Greece, Rome-Egypt, and the Egypt-Greece vassal war, all at turn 3.
        "peak_wars": 3,
        "units_lost": 1, "cities_captured": 1, "capitals_captured": 1,
        "city_state_cities_captured": 1, "civs_eliminated": 2,
        "cities_razed": 1, "nukes_detonated": 2, "cities_nuked": 1,
    }


def test_world_reads_only_the_latest_summary_of_a_replayed_turn(tmp_path):
    # Egypt's turn-30 row with a positive score is replayed with a score of 0
    # (a higher ID), as the turns table reads it: Egypt is gone after turn 10.
    summaries = [(0, 10, 100), (0, 30, 300), (1, 10, 90), (1, 30, 90),
                 (2, 10, 50), (2, 30, 60), (2, 30, 0)]
    events = [_war(3, 0, 2), _war(20, 0, 1)]
    db = _make_db(tmp_path / "exp" / "g_1.db", events, summaries)
    row = extract_game_world_data(str(db))[0]
    assert row["civs_eliminated"] == 1
    # The war on Egypt closed when Egypt fell, before Rome declared on Greece.
    assert row["peak_wars"] == 1


class _Ctx:
    """The parts of a ``GameContext`` that :func:`peak_wars` reads."""

    def __init__(self, majors):
        self.major_players = list(majors)
        self.team_of = {pid: pid for pid in [*majors, 22]}
        self.team_kind = {**{pid: "major" for pid in majors}, 22: "minor"}

    def players_on_team(self, team):
        return [pid for pid in self.major_players if pid == team]


def _replay(events):
    return [(t, turn, json.loads(p)) for turn, t, p in events]


def test_peak_wars_closes_wars_at_peace_and_elimination():
    ctx = _Ctx([0, 1, 2])
    alive = {0: 30, 1: 30, 2: 30}
    # Declared, closed, and declared again: never two at once.
    events = _replay([_war(1, 0, 1), _peace(2, 1, 0), _war(3, 0, 2), _war(3, 0, 2)])
    assert peak_wars(events, ctx, alive) == 1
    # A war on a civ that is gone ends with it, though no peace is signed.
    events = _replay([_war(1, 0, 1), _war(8, 0, 2)])
    assert peak_wars(events, ctx, {**alive, 1: 5}) == 1
    assert peak_wars(events, ctx, alive) == 2
    # Wars with city-states do not count.
    assert peak_wars(_replay([_war(1, 0, 22), _war(1, 1, 22)]), ctx, alive) == 0


def test_world_export_is_byte_stable_and_one_row_per_game(tmp_path):
    first_db = _make_db(tmp_path / "exp" / "g_1.db", EVENTS, SUMMARIES)
    second_db = _make_db(tmp_path / "exp" / "h_1.db", [], SUMMARIES)
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    for out in (first, second):
        export_world_data([str(first_db), str(second_db)], {"g", "h"}, str(out))
    assert first.read_bytes() == second.read_bytes()
    rows = list(csv.DictReader(first.open(encoding="utf-8")))
    assert [r["game_id"] for r in rows] == ["g", "h"]
    assert rows[1]["wars_declared"] == "0"
    assert rows[1]["peak_wars"] == "0"
    assert rows[1]["civs_eliminated"] == "2"


def test_world_is_a_canonical_table():
    from bench.extract.runner import DEFAULT_TABLE_PATHS

    assert "world" in S.TABLE_NAMES
    assert DEFAULT_TABLE_PATHS["world"] == "runs/world_data.csv"
