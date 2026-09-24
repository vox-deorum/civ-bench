"""Behavior table extraction (benchmark.md §3.0) on small synthetic game DBs."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from bench.catalog import Catalog
from bench.config import schema as S
from bench.config.behavior import behavior_columns, resolve_behavior_spec
from bench.config.errors import ConfigError
from bench.config.models import OutputConfig, RunConfig
from bench.extract import run_extract
from bench.extract.behavior import EVENT_COUNTERS, FAMILIES, snake_case
from bench.extract.extract_behavior import (
    behavior_fieldnames,
    export_behavior_data,
    extract_game_behavior_data,
)
from bench.extract.extract_panel import PANEL_FIELD_MAPPINGS

SPEC = {
    "stats": ["min", "avg", "max"],
    "flavor": ["Offense", "UseNuke"],
    "persona": ["Boldness"],
    "events": list(S.BEHAVIOR_EVENTS),
    "policies": list(S.BEHAVIOR_POLICIES),
    "relationships": list(S.BEHAVIOR_RELATIONSHIPS),
}


def _event(event_type, turn, payload):
    return (turn, event_type, json.dumps(payload, sort_keys=True))


def _make_db(path: Path, *, flavor=None, persona=None, policy_rows=(), events=(), with_index=True,
             with_team=True, relationships=None, summaries=None) -> Path:
    """Players 0 and 1 are majors on teams 0 and 1; player 22 is a city-state.

    ``summaries`` overrides the ``PlayerSummaries`` rows as ``(key, last_turn)``
    pairs; a key other than 0, 1, or 22 becomes an extra major.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE GameMetadata (Key TEXT, Value TEXT)")
    cur.execute("INSERT INTO GameMetadata VALUES ('gameId', ?)", (path.stem.split("_")[0],))
    if with_team:
        cur.execute("CREATE TABLE PlayerInformations (Key INTEGER, Civilization TEXT, TeamID INTEGER, IsMajor INTEGER)")
        cur.executemany("INSERT INTO PlayerInformations VALUES (?, ?, ?, ?)",
                        [(0, "Rome", 0, 1), (1, "Greece", 1, 1), (22, "Ife", 22, 0)])
    else:
        cur.execute("CREATE TABLE PlayerInformations (Key INTEGER, Civilization TEXT, IsMajor INTEGER)")
        cur.executemany("INSERT INTO PlayerInformations VALUES (?, ?, ?)",
                        [(0, "Rome", 1), (1, "Greece", 1), (22, "Ife", 0)])
    summary_rows = [(1, 0, 0), (2, 0, 9), (3, 1, 0), (4, 1, 5)]
    if summaries is not None:
        summary_rows = [(i, key, turn) for i, (key, turn) in enumerate(summaries, start=1)]
        extra = sorted({key for key, _ in summaries} - {0, 1, 22})
        if with_team:
            cur.executemany("INSERT INTO PlayerInformations VALUES (?, ?, ?, 1)",
                            [(key, f"Civ{key}", key) for key in extra])
        else:
            cur.executemany("INSERT INTO PlayerInformations VALUES (?, ?, 1)",
                            [(key, f"Civ{key}") for key in extra])
    cur.execute("CREATE TABLE PlayerSummaries (ID INTEGER, Key INTEGER, Turn INTEGER)")
    cur.executemany("INSERT INTO PlayerSummaries VALUES (?, ?, ?)", summary_rows)
    if relationships is not None:
        cur.execute("CREATE TABLE RelationshipChanges (ID INTEGER PRIMARY KEY, Turn INTEGER, PlayerID INTEGER, "
                    "TargetID INTEGER, PublicValue INTEGER, PrivateValue INTEGER, Rationale TEXT)")
        cur.executemany("INSERT INTO RelationshipChanges (Turn, PlayerID, TargetID, PublicValue, PrivateValue, "
                        "Rationale) VALUES (?, ?, ?, ?, ?, 'why')", relationships)
    if flavor is not None:
        cur.execute("CREATE TABLE FlavorChanges (ID INTEGER, Key INTEGER, Turn INTEGER, Offense INTEGER, UseNuke INTEGER)")
        cur.executemany("INSERT INTO FlavorChanges VALUES (?, ?, ?, ?, ?)", flavor)
    if persona is not None:
        cur.execute("CREATE TABLE PersonaChanges (ID INTEGER, Key INTEGER, Turn INTEGER, Boldness INTEGER)")
        cur.executemany("INSERT INTO PersonaChanges VALUES (?, ?, ?, ?)", persona)
    cur.execute("CREATE TABLE PolicyChanges (ID INTEGER, Key INTEGER, Changes TEXT)")
    cur.executemany("INSERT INTO PolicyChanges VALUES (?, ?, ?)", policy_rows)
    cur.execute("CREATE TABLE GameEvents (ID INTEGER PRIMARY KEY, Turn INTEGER, Type TEXT, Payload TEXT, Player0 INTEGER)")
    if with_index:
        cur.execute('CREATE INDEX "idx_gameevents_player0" ON "GameEvents" ("ID", "Turn", "Type", "Player0")')
    cur.executemany("INSERT INTO GameEvents (Turn, Type, Payload, Player0) VALUES (?, ?, ?, 0)", events)
    conn.commit()
    conn.close()
    return path


def _rows_by_player(db: Path, spec=SPEC) -> dict:
    return {r["player_id"]: r for r in extract_game_behavior_data(str(db), resolve_behavior_spec(spec))}


# ── config ───────────────────────────────────────────────────────────────────
def test_spec_defaults_fill_missing_keys():
    spec = resolve_behavior_spec({"flavor": ["Offense"]})
    assert spec["flavor"] == ["Offense"]
    assert spec["persona"] == S.BEHAVIOR_DEFAULTS["persona"]
    assert spec["stats"] == ["min", "avg", "max"]
    assert spec["policies"] == S.BEHAVIOR_DEFAULTS["policies"]
    assert resolve_behavior_spec(None) == resolve_behavior_spec({})


@pytest.mark.parametrize("raw, match", [
    ({"mood": []}, "unknown key"),
    ({"flavor": ["Offence"]}, "unknown name"),
    ({"persona": ["Offense"]}, "unknown name"),
    ({"events": ["wars_started"]}, "unknown name"),
    ({"stats": ["median"]}, "unknown name"),
    ({"flavor": ["Nuke", "Nuke"]}, "duplicate"),
    ({"flavor": "Nuke"}, "list of strings"),
    ([], "expected an object"),
])
def test_spec_rejects_bad_selection(raw, match):
    with pytest.raises(ConfigError, match=match):
        resolve_behavior_spec(raw)


def test_registries_match_schema():
    assert list(FAMILIES) == list(S.BEHAVIOR_FAMILIES)
    assert set(EVENT_COUNTERS) == set(S.BEHAVIOR_EVENTS)


def test_fieldnames_follow_selection():
    names = behavior_fieldnames(SPEC)
    assert names[:6] == ["experiment", "game_id", "player_id", "player_type", "civilization", "survival_turn"]
    assert names[6:] == [
        "flavor_offense_min", "flavor_offense_avg", "flavor_offense_max",
        "flavor_use_nuke_min", "flavor_use_nuke_avg", "flavor_use_nuke_max",
        "persona_boldness_min", "persona_boldness_avg", "persona_boldness_max",
        "wars_declared", "wars_received", "cities_nuked", "cities_razed",
        *S.BEHAVIOR_POLICIES,
        *S.BEHAVIOR_RELATIONSHIPS,
    ]
    assert snake_case("MinorCivWarBias") == "minor_civ_war_bias"


def test_behavior_columns_carry_kinds():
    kinds = behavior_columns(SPEC)
    assert list(kinds) == behavior_fieldnames(SPEC)[6:]
    assert kinds["flavor_offense_avg"] == "level"
    assert kinds["wars_declared"] == "count"
    assert kinds["policy_changes"] == "count"
    assert kinds["tradition"] == "turn"
    assert kinds["stance_masked_hostility_share"] == "share"
    assert set(kinds.values()) <= set(S.BEHAVIOR_COLUMN_KINDS)
    assert "persona_friendliness_avg" in behavior_columns({})


def test_panel_omits_behavior_and_token_columns():
    assert "nuke" not in PANEL_FIELD_MAPPINGS
    assert "use_nuke" not in PANEL_FIELD_MAPPINGS
    assert "input_tokens" not in PANEL_FIELD_MAPPINGS
    assert "reasoning_tokens" not in PANEL_FIELD_MAPPINGS
    assert "output_tokens" not in PANEL_FIELD_MAPPINGS
    assert not set(S.BEHAVIOR_POLICIES) & set(PANEL_FIELD_MAPPINGS)


def test_policy_counts_and_first_adoption_turns_live_in_behavior(tmp_path):
    db = _make_db(
        tmp_path / "exp" / "g_1.db",
        policy_rows=[
            (1, 0, '["Policy"]'),
            (2, 0, '["Rationale"]'),
            (3, 0, "[]"),
        ],
        events=[
            _event("PlayerAdoptPolicyBranch", 8, {"PlayerID": 0, "BranchType": "Tradition"}),
            _event("PlayerAdoptPolicyBranch", 3, {"PlayerID": 0, "BranchType": "Tradition"}),
            _event("IdeologyAdopted", 7, {"PlayerID": 1, "BranchType": "Freedom"}),
        ],
    )

    rows = _rows_by_player(db)
    assert rows[0]["policy_changes"] == 1
    assert rows[0]["tradition"] == 3
    assert rows[0]["authority"] == "N/A"
    assert rows[1]["policy_changes"] == 0
    assert rows[1]["freedom"] == 7
    assert rows[1]["tradition"] == "N/A"


# ── state families ───────────────────────────────────────────────────────────
def test_state_stats_use_last_row_per_turn_and_weight_by_turns(tmp_path):
    db = _make_db(tmp_path / "exp" / "g_1.db", flavor=[
        # Turn 0 has two rows: the later one (Offense 30) is the state in effect.
        (1, 0, 0, 10, 50),
        (2, 0, 0, 30, 50),
        (3, 0, 4, 60, 80),   # holds turns 4..9 (survival turn 9)
    ], persona=[(1, 0, 0, 5), (2, 1, 0, 7)])
    row = _rows_by_player(db)[0]
    # Offense: 30 for turns 0-3 (4 turns), 60 for turns 4-9 (6 turns).
    assert row["flavor_offense_min"] == 30
    assert row["flavor_offense_max"] == 60
    assert row["flavor_offense_avg"] == pytest.approx((30 * 4 + 60 * 6) / 10)
    assert row["flavor_use_nuke_avg"] == pytest.approx((50 * 4 + 80 * 6) / 10)
    assert row["persona_boldness_min"] == row["persona_boldness_max"] == 5
    assert row["survival_turn"] == 9


def test_state_blank_without_rows_or_table(tmp_path):
    db = _make_db(tmp_path / "exp" / "g_1.db", flavor=[(1, 0, 0, 10, 50)], persona=None)
    rows = _rows_by_player(db)
    assert rows[1]["flavor_offense_min"] == ""      # no FlavorChanges rows for player 1
    assert rows[0]["persona_boldness_avg"] == ""    # no PersonaChanges table at all
    assert 22 not in rows                            # city-states get no row


def test_state_missing_column_preserves_other_selected_values(tmp_path):
    db = _make_db(tmp_path / "exp" / "g_1.db", persona=[(1, 0, 0, 7)])
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE FlavorChanges (ID INTEGER, Key INTEGER, Turn INTEGER, Offense INTEGER)"
        )
        conn.execute("INSERT INTO FlavorChanges VALUES (1, 0, 0, 30)")
        conn.execute("INSERT INTO FlavorChanges VALUES (2, 0, 4, 60)")

    spec = {
        "stats": ["min", "avg", "max"],
        "flavor": ["UseNuke", "Offense"],
        "persona": ["WarBias", "Boldness"],
        "events": [],
    }
    row = _rows_by_player(db, spec)[0]
    assert row["flavor_offense_avg"] == pytest.approx((30 * 4 + 60 * 6) / 10)
    assert row["flavor_use_nuke_avg"] == ""
    assert row["persona_boldness_avg"] == 7
    assert row["persona_war_bias_avg"] == ""


# ── event family ─────────────────────────────────────────────────────────────
def _war(turn, origin, target, aggressor=True):
    return _event("DeclareWar", turn, {"OriginatingPlayerID": origin, "TargetTeamID": target,
                                       "IsAggressor": aggressor})


EVENTS = [
    _war(3, 0, 1),
    _war(3, 0, 1),                      # exact duplicate: counted once
    _war(4, 0, 22),                     # on a city-state: still Rome's declaration
    _war(5, 1, 0, aggressor=False),     # joined as a vassal: not a declaration
    _war(6, 22, 1, aggressor=False),    # city-state follows its ally: Greece receives it
    _event("NuclearDetonation", 7, {"PlayerID": 0, "Plot": {"City": "Athens", "CityID": 9}}),
    _event("NuclearDetonation", 7, {"PlayerID": 0, "Plot": {"PlotType": "Land"}}),   # no city
    _event("CityRazed", 8, {"PlayerID": 1, "City": {"CityID": 3}}),
    _event("CityRazed", 8, {"PlayerID": 22, "City": {"CityID": 4}}),                # minor
    _event("UnitMoved", 8, {"PlayerID": 0}),
]


@pytest.mark.parametrize("with_index", [True, False])
def test_event_counts(tmp_path, with_index):
    db = _make_db(tmp_path / "exp" / "g_1.db", events=EVENTS, with_index=with_index)
    rows = _rows_by_player(db)
    assert {k: rows[0][k] for k in S.BEHAVIOR_EVENTS} == {
        "wars_declared": 2, "wars_received": 1, "cities_nuked": 1, "cities_razed": 0,
    }
    assert {k: rows[1][k] for k in S.BEHAVIOR_EVENTS} == {
        "wars_declared": 0, "wars_received": 2, "cities_nuked": 0, "cities_razed": 1,
    }


def test_wars_received_without_team_column_uses_player_id(tmp_path):
    db = _make_db(tmp_path / "exp" / "g_1.db", events=EVENTS, with_team=False)
    assert _rows_by_player(db)[1]["wars_received"] == 2


def test_empty_family_selection_adds_no_columns(tmp_path):
    spec = {"flavor": [], "persona": [], "events": ["cities_razed"], "policies": [], "relationships": []}
    assert behavior_fieldnames(spec)[6:] == ["cities_razed"]
    db = _make_db(tmp_path / "exp" / "g_1.db", events=EVENTS)
    assert _rows_by_player(db, spec)[1]["cities_razed"] == 1


# ── relationship family ──────────────────────────────────────────────────────
def test_relationship_worked_example_weights_pair_turns_since_first_set(tmp_path):
    # Player 0 lives to turn 150; public +20 on player 1 from turn 50, -10 on player 2 from turn 140.
    db = _make_db(
        tmp_path / "exp" / "g_1.db",
        summaries=[(0, 150), (1, 150), (2, 150)],
        relationships=[(50, 0, 1, 20, 0), (140, 0, 2, -10, 0)],
    )
    row = _rows_by_player(db)[0]
    # Survival turns are inclusive: 101 pair-turns at +20 (50-150) and 11 at -10 (140-150).
    assert row["stance_public_avg"] == pytest.approx(round((101 * 20 + 11 * -10) / 112, 4))
    assert row["stance_public_avg"] == pytest.approx(17.0536)
    assert row["stance_private_avg"] == 0
    assert row["relationship_changes"] == 2
    assert row["relationship_targets"] == 2


def test_relationship_window_resets_and_target_survival(tmp_path):
    # Player 1 dies at turn 5, so a stance on it counts only through turn 5.
    db = _make_db(
        tmp_path / "exp" / "g_1.db",
        relationships=[
            (1, 0, 1, 30, -20),   # masked hostility, turns 1-2
            (3, 0, 1, 30, -20),   # same values again: not a change
            (3, 0, 1, 0, 0),      # later row in turn 3 wins: reset, turns 3-5 still count
            (2, 1, 0, -5, 10),    # player 1: masked goodwill, turns 2-5
            (4, 0, 22, 50, 50),   # city-state target: ignored
        ],
    )
    rows = _rows_by_player(db)
    first = rows[0]
    assert first["relationship_changes"] == 2
    assert first["relationship_targets"] == 1
    assert first["stance_public_avg"] == pytest.approx(round(30 * 2 / 5, 4))
    assert first["stance_net_avg"] == pytest.approx(round(10 * 2 / 5, 4))
    assert first["stance_masked_hostility_share"] == pytest.approx(0.4)
    assert first["stance_masked_goodwill_share"] == 0
    second = rows[1]
    assert second["stance_masked_goodwill_share"] == 1
    assert second["stance_private_avg"] == 10


def test_relationship_counts_zero_and_averages_blank_without_rows(tmp_path):
    db = _make_db(tmp_path / "exp" / "g_1.db", relationships=[(1, 0, 1, 10, 10)])
    row = _rows_by_player(db)[1]
    assert row["relationship_changes"] == 0
    assert row["relationship_targets"] == 0
    assert row["stance_public_avg"] == ""
    assert row["stance_masked_hostility_share"] == ""


def test_relationship_blank_without_table(tmp_path):
    row = _rows_by_player(_make_db(tmp_path / "exp" / "g_1.db"))[0]
    assert row["relationship_changes"] == ""
    assert row["stance_net_avg"] == ""


# ── export + runner ──────────────────────────────────────────────────────────
def test_export_is_byte_stable(tmp_path):
    db = _make_db(tmp_path / "exp" / "g_1.db", flavor=[(1, 0, 0, 10, 50)], events=EVENTS)
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    export_behavior_data([str(db)], {"g"}, str(first), SPEC)
    export_behavior_data([str(db)], {"g"}, str(second), SPEC)
    assert first.read_bytes() == second.read_bytes()
    rows = list(csv.DictReader(first.open(encoding="utf-8")))
    assert [r["player_id"] for r in rows] == ["0", "1"]


def _run_config(tmp_path: Path, behavior: dict) -> RunConfig:
    extract = {
        "runs_dir": str(tmp_path / "runs"),
        "outputs": ["behavior"],
        "issues_path": str(tmp_path / "import_issues.csv"),
        "behavior": behavior,
    }
    return RunConfig(
        name="test", seed=1, config_path=tmp_path / "benchmark.json", raw={},
        output=OutputConfig(),
        data={"extract": extract, "tables": {"behavior": str(tmp_path / "behavior.csv")}},
    )


def test_changed_selection_forces_rebuild(tmp_path, configs_dir):
    catalog = Catalog.from_paths(configs_dir / "models.json", configs_dir / "experiments.json")
    _make_db(tmp_path / "runs" / "exp" / "g_1.db", events=EVENTS)
    first = run_extract(_run_config(tmp_path, {"events": ["cities_razed"]}), catalog=catalog)
    assert not first.skipped
    again = run_extract(_run_config(tmp_path, {"events": ["cities_razed"]}), catalog=catalog)
    assert again.skipped

    changed = run_extract(_run_config(tmp_path, {"events": ["cities_razed", "cities_nuked"]}), catalog=catalog)
    assert not changed.skipped
    header = (tmp_path / "behavior.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "cities_razed,cities_nuked," in header
