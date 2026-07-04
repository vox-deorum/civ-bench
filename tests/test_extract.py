"""Stage 1 — extract: controlled-seed policy, orthodox identity, skip-if-newer.

The heavy per-turn/panel SQL is ported verbatim from the analysis repo; these
tests target the **new** behavior (benchmark.md §3, §3.3, rule 14): the seeding
policy, the orthodox ``player_type`` composition (incl. seat rotation + legacy
fallback), the skip-if-newer shortcut, and end-to-end ``game_data`` / ``panel_data``
extraction from synthetic game DBs.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

from bench.catalog import Catalog
from bench.config.models import OutputConfig, RunConfig
from bench.extract import (
    ExtractError,
    compose_identities,
    extract_seeding_fields,
    invert_seating_map,
    outputs_are_fresh,
    run_extract,
)
from bench.extract.extract_games import export_game_data
from bench.extract.extract_panel import _has_real_changes
from bench.extract.utilities import UNCONTROLLED


# ── fixtures / helpers ───────────────────────────────────────────────────────
@pytest.fixture
def catalog(configs_dir) -> Catalog:
    return Catalog.from_paths(configs_dir / "models.json", configs_dir / "experiments.json")


def _make_game_db(
    path: Path,
    metadata: dict,
    players=None,
    summaries=None,
    flavor_rows=None,
    persona_rows=None,
    research_rows=None,
    policy_rows=None,
) -> None:
    """Create a minimal game DB with a GameMetadata Key→Value table (+ optional rows).

    ``flavor_rows`` is a list of ``(ID, Key, Turn, Changes, GrandStrategy, Rationale,
    Nuke, UseNuke)`` tuples inserted into ``FlavorChanges`` (for decision counting).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE GameMetadata (Key TEXT, Value TEXT)")
    for key, value in metadata.items():
        cur.execute("INSERT INTO GameMetadata VALUES (?, ?)", (key, str(value)))

    if players is not None:
        cur.execute("CREATE TABLE PlayerInformations (Key INTEGER, Civilization TEXT, IsMajor INTEGER)")
        cur.executemany("INSERT INTO PlayerInformations VALUES (?, ?, ?)", players)

    if summaries is not None:
        cur.execute(
            "CREATE TABLE PlayerSummaries "
            "(ID INTEGER, Key INTEGER, Turn INTEGER, Score INTEGER, IsLatest INTEGER, CurrentResearch TEXT)"
        )
        cur.executemany("INSERT INTO PlayerSummaries VALUES (?, ?, ?, ?, ?, ?)", summaries)
        # Empty change/event tables so the verbatim per-player queries take the
        # normal (non-exception) path instead of the catch-all N/A fallback.
        cur.execute("CREATE TABLE FlavorChanges (ID INTEGER, Key INTEGER, Turn INTEGER, Changes TEXT, GrandStrategy TEXT, Rationale TEXT, Nuke INTEGER, UseNuke INTEGER)")
        if flavor_rows:
            cur.executemany("INSERT INTO FlavorChanges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", flavor_rows)
        cur.execute("CREATE TABLE StrategyChanges (ID INTEGER, Key INTEGER, Turn INTEGER, GrandStrategy TEXT, Changes TEXT, Rationale TEXT)")
        cur.execute(
            "CREATE TABLE PersonaChanges "
            "(ID INTEGER, Key INTEGER, Turn INTEGER, Version INTEGER, Changes TEXT, "
            "Rationale TEXT, DiplomaticBalance INTEGER, Boldness INTEGER)"
        )
        if persona_rows:
            cur.executemany("INSERT INTO PersonaChanges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", persona_rows)
        cur.execute("CREATE TABLE ResearchChanges (Key INTEGER, Changes TEXT)")
        if research_rows:
            cur.executemany("INSERT INTO ResearchChanges VALUES (?, ?)", research_rows)
        cur.execute("CREATE TABLE PolicyChanges (Key INTEGER, Changes TEXT)")
        if policy_rows:
            cur.executemany("INSERT INTO PolicyChanges VALUES (?, ?)", policy_rows)
        cur.execute("CREATE TABLE GameEvents (Turn INTEGER, Type TEXT, Payload TEXT, Player0 INTEGER, Player1 INTEGER)")

    conn.commit()
    conn.close()


def _run_config(tmp_path: Path, extract: dict, tables: dict) -> RunConfig:
    # Default the issue report into tmp so tests never write the repo's runs/ and
    # the freshness check (which now counts issues_path) is deterministic; a test
    # may still override issues_path via its own ``extract`` dict.
    extract = {"issues_path": str(tmp_path / "import_issues.csv"), **extract}
    return RunConfig(
        name="test",
        seed=1,
        config_path=tmp_path / "benchmark.json",
        raw={},
        output=OutputConfig(),
        data={"extract": extract, "tables": tables},
    )


# ── seeding policy (rule 14) ─────────────────────────────────────────────────
def test_seeding_matched_seeds_are_controlled():
    info = extract_seeding_fields({
        "configuredSyncRandSeed": "12345",
        "configuredMapRandSeed": "12345",
        "seatingRotation": "3",
    })
    assert info.seed == 12345
    assert info.seating_rotation == 3
    assert info.controlled is True


def test_seeding_mismatch_aborts():
    with pytest.raises(ExtractError, match="mismatched seeds"):
        extract_seeding_fields({
            "configuredSyncRandSeed": "1",
            "configuredMapRandSeed": "2",
        })


def test_seeding_zero_is_uncontrolled():
    # 0 is Civ's "pick random" → uncontrolled sentinel, not a controlled seed.
    info = extract_seeding_fields({"configuredSyncRandSeed": "0", "configuredMapRandSeed": "0"})
    assert info.seed == UNCONTROLLED


def test_seeding_absent_uses_sentinels():
    info = extract_seeding_fields({})
    assert info.seed == UNCONTROLLED
    assert info.seating_rotation == UNCONTROLLED
    assert info.config_slots == {}
    # a seat absent from the seatingMap is not part of the controlled seating → -1
    assert info.config_slot(2) == UNCONTROLLED


def test_seeding_rotation_zero_is_valid():
    info = extract_seeding_fields({"seatingRotation": "0"})
    assert info.seating_rotation == 0


def test_invert_seating_map():
    # {config_slot: player_id} → {player_id: config_slot}
    assert invert_seating_map('{"0": 2, "1": 5}') == {2: 0, 5: 1}
    assert invert_seating_map(None) == {}
    with pytest.raises(ExtractError):
        invert_seating_map("{not json}")


def test_seeding_config_slot_from_seating_map():
    info = extract_seeding_fields({"seatingMap": '{"0": 1, "1": 0}'})
    assert info.config_slot(1) == 0
    assert info.config_slot(0) == 1


# ── orthodox identity (§3.3) ─────────────────────────────────────────────────
def test_identity_follows_player_through_rotation(catalog):
    # Sonnet sits in seat 1 this game (seatingMap puts config slot 0 → player 1);
    # identity travels with the player, not the seat.
    metadata = {
        "model-1": "claude-sonnet-4-5",
        "strategist-1": "simple-strategist-briefed",
        "model-0": "VPAI",
        "strategist-0": "none-strategist",
        "seatingMap": '{"0": 1, "1": 0}',
        "seatingRotation": "1",
        "configuredSyncRandSeed": "777",
        "configuredMapRandSeed": "777",
    }
    seeding = extract_seeding_fields(metadata)
    ids = compose_identities(metadata, [0, 1], "2026-staff-standard", catalog, seeding)

    assert ids[1]["player_type"] == "Sonnet-4.5-Briefed"
    assert ids[1]["config_slot"] == 0           # seat 1 holds config slot 0
    assert ids[0]["player_type"] == "Vanilla"   # VPAI baseline
    assert ids[0]["config_slot"] == 1


def test_identity_unmarked_seat_defaults_to_vpai(catalog):
    # No model-{id} metadata and no legacy seat map → default VPAI → Vanilla.
    ids = compose_identities({}, [0], "gemma-4-standard-fixed", catalog, extract_seeding_fields({}))
    assert ids[0]["player_type"] == "Vanilla"
    assert ids[0]["model"] == "VPAI"
    assert ids[0]["config_slot"] == UNCONTROLLED


def test_identity_legacy_condition_uses_static_seat_map(catalog):
    # A condition present in condition_player_mapping keeps the legacy seat label
    # for an unmarked seat (games that predate the per-player metadata).
    mapping = catalog.condition_player_mapping()
    legacy_cond = next(iter(mapping))
    ids = compose_identities({}, [0], legacy_cond, catalog, extract_seeding_fields({}))
    assert ids[0]["player_type"] == mapping[legacy_cond][0]


# ── skip-if-newer (§3) ───────────────────────────────────────────────────────
def test_outputs_are_fresh(tmp_path):
    db = tmp_path / "exp" / "game_1.db"
    out = tmp_path / "out.csv"

    db.parent.mkdir(parents=True)
    db.write_text("x", encoding="utf-8")
    # No output yet → not fresh (must build).
    assert outputs_are_fresh([str(out)], [str(db)]) is False

    # Output written after the DB → fresh.
    time.sleep(0.01)
    out.write_text("y", encoding="utf-8")
    assert outputs_are_fresh([str(out)], [str(db)]) is True

    # DB touched after the output → stale.
    time.sleep(0.01)
    os.utime(str(db), None)
    assert outputs_are_fresh([str(out)], [str(db)]) is False

    # No DBs at all → nothing to (re)build.
    assert outputs_are_fresh([str(out)], []) is True


# ── game_data end-to-end ─────────────────────────────────────────────────────
def test_export_game_data_writes_seed_and_rotation(tmp_path):
    db = tmp_path / "2026-staff" / "abc_1700000000000.db"
    _make_game_db(db, {
        "gameId": "abc",
        "configuredSyncRandSeed": "42",
        "configuredMapRandSeed": "42",
        "seatingRotation": "5",
    })
    out = tmp_path / "game_data.csv"

    new_rows = export_game_data([str(db)], {"abc"}, str(out))
    assert new_rows == 1

    import csv
    row = list(csv.DictReader(out.open(encoding="utf-8")))[0]
    assert row["game_id"] == "abc"
    assert row["experiment"] == "2026-staff"
    assert row["seed"] == "42"
    assert row["seating_rotation"] == "5"
    assert row["timestamp"] == "1700000000000"


def test_export_game_data_uncontrolled_sentinels(tmp_path):
    db = tmp_path / "exp" / "g_100.db"
    _make_game_db(db, {"gameId": "g"})
    out = tmp_path / "game_data.csv"
    export_game_data([str(db)], {"g"}, str(out))

    import csv
    row = list(csv.DictReader(out.open(encoding="utf-8")))[0]
    assert row["seed"] == "-1"
    assert row["seating_rotation"] == "-1"


def test_export_game_data_mismatch_aborts(tmp_path):
    db = tmp_path / "exp" / "bad_100.db"
    _make_game_db(db, {"gameId": "bad", "configuredSyncRandSeed": "1", "configuredMapRandSeed": "2"})
    out = tmp_path / "game_data.csv"
    with pytest.raises(ExtractError, match="mismatched seeds"):
        export_game_data([str(db)], {"bad"}, str(out))


# ── WS7: export driver — swallowed-failure + structure-mismatch bugs ──────────
def test_export_append_failure_raises(tmp_path, monkeypatch):
    from bench.extract import export_common

    db = tmp_path / "exp" / "g_100.db"
    _make_game_db(db, {"gameId": "g"})
    out = tmp_path / "game_data.csv"  # fresh file → append path
    monkeypatch.setattr(export_common, "append_csv_file", lambda *a, **k: False)
    with pytest.raises(ExtractError, match="failed to append"):
        export_game_data([str(db)], {"g"}, str(out))


def test_export_write_failure_raises(tmp_path, monkeypatch):
    from bench.extract import export_common

    db = tmp_path / "exp" / "g_100.db"
    _make_game_db(db, {"gameId": "g"})
    out = tmp_path / "game_data.csv"
    out.write_text("game_id,old_col\nx,1\n", encoding="utf-8")  # mismatch → rewrite path
    monkeypatch.setattr(export_common, "write_csv_file", lambda *a, **k: False)
    with pytest.raises(ExtractError, match="failed to write"):
        export_game_data([str(db)], {"g"}, str(out))


def test_export_structure_mismatch_rewrites_on_normal_run(tmp_path):
    from bench.extract.extract_games import GAME_FIELDNAMES

    db = tmp_path / "exp" / "g_100.db"
    _make_game_db(db, {"gameId": "g"})
    out = tmp_path / "game_data.csv"
    out.write_text("game_id,old_col\nx,1\n", encoding="utf-8")  # wrong schema

    export_game_data([str(db)], {"g"}, str(out))

    import csv
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    # the file now carries the current schema, not the stale one it was "discarded" for
    assert list(rows[0].keys()) == GAME_FIELDNAMES
    assert rows[0]["game_id"] == "g"


def test_export_structure_mismatch_prune_only_leaves_file_untouched(tmp_path, capsys):
    db = tmp_path / "exp" / "g_100.db"
    _make_game_db(db, {"gameId": "g"})
    out = tmp_path / "game_data.csv"
    original = "game_id,old_col\nx,1\n"
    out.write_text(original, encoding="utf-8")

    n = export_game_data([str(db)], {"g"}, str(out), prune_only=True)

    assert n == 0
    assert out.read_text(encoding="utf-8") == original  # not silently discarded
    assert "untouched" in capsys.readouterr().out       # …and it says so honestly


# ── run_extract orchestration ────────────────────────────────────────────────
def test_run_extract_disabled_is_skipped(tmp_path, catalog):
    cfg = _run_config(tmp_path, {"enabled": False}, {})
    result = run_extract(cfg, catalog=catalog)
    assert result.skipped is True
    assert "enabled is false" in result.reason


def test_run_extract_skips_when_fresh(tmp_path, catalog):
    runs = tmp_path / "runs"
    db = runs / "exp" / "g_1.db"
    _make_game_db(db, {"gameId": "g"})
    out = tmp_path / "game_data.csv"
    out.write_text("game_id,timestamp,experiment,seed,seating_rotation\n", encoding="utf-8")
    # The issue report is one of the outputs the freshness check requires; a fresh
    # run only skips when it too exists and is newer than the DBs.
    issues = tmp_path / "import_issues.csv"
    issues.write_text("game_id,experiment,seed,seating_rotation,stages,players,db_name,message\n", encoding="utf-8")
    time.sleep(0.01)
    os.utime(str(out), None)  # outputs newer than the DB
    os.utime(str(issues), None)

    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["games"], "issues_path": str(issues)},
        {"games": str(out)},
    )
    result = run_extract(cfg, catalog=catalog)
    assert result.skipped is True
    assert "newer than the source" in result.reason


def test_run_extract_carries_forward_prior_issue_for_unreexamined_stage(tmp_path, catalog):
    # Reconcile, don't clobber: a prior tokens-stage issue survives a games-only run
    # because the tokens stage never re-examined the game (the report is durable).
    runs = tmp_path / "runs"
    db = runs / "exp" / "oldg_100.db"
    _make_game_db(db, {"gameId": "oldg", "configuredSyncRandSeed": "7",
                       "configuredMapRandSeed": "7", "seatingRotation": "1"})
    issues = tmp_path / "import_issues.csv"
    issues.write_text(
        "game_id,experiment,seed,seating_rotation,stages,players,db_name,message\n"
        "oldg,exp,7,1,tokens,2,oldg-player-2.db,trace boom\n",
        encoding="utf-8",
    )
    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["games"], "issues_path": str(issues)},
        {"games": str(tmp_path / "game_data.csv")},
    )
    result = run_extract(cfg, catalog=catalog)

    assert result.skipped is False
    assert len(result.issues) == 1
    issue = result.issues.issues()[0]
    assert issue.game_id == "oldg" and issue.stages == {"tokens"}
    assert issue.seed == 7 and issue.seating_rotation == 1
    import csv
    persisted = list(csv.DictReader(issues.open(encoding="utf-8")))
    assert persisted and persisted[0]["game_id"] == "oldg" and persisted[0]["stages"] == "tokens"


def test_run_extract_prune_only_reconciles_report_without_clobber(tmp_path, catalog):
    # Prune-only inspects no DBs: it must keep issues for present games and drop
    # those whose DB is gone — never overwrite the report with a clean header.
    runs = tmp_path / "runs"
    _make_game_db(runs / "exp" / "keep_100.db", {"gameId": "keep"})
    issues = tmp_path / "import_issues.csv"
    issues.write_text(
        "game_id,experiment,seed,seating_rotation,stages,players,db_name,message\n"
        "keep,exp,-1,-1,panel,,keep_100.db,boom\n"
        "gone,exp,-1,-1,panel,,gone_100.db,boom\n",
        encoding="utf-8",
    )
    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["games"],
         "prune_missing": True, "issues_path": str(issues)},
        {"games": str(tmp_path / "game_data.csv")},
    )
    result = run_extract(cfg, catalog=catalog)

    assert result.skipped is False
    assert {i.game_id for i in result.issues.issues()} == {"keep"}
    import csv
    persisted = {r["game_id"] for r in csv.DictReader(issues.open(encoding="utf-8"))}
    assert persisted == {"keep"}


def test_run_extract_does_not_skip_when_issue_report_missing(tmp_path, catalog):
    # Outputs fresh but the issue report has never been written → must (re)build so
    # the report is not stranded behind already-fresh tables.
    runs = tmp_path / "runs"
    db = runs / "exp" / "g_1.db"
    _make_game_db(db, {"gameId": "g"})
    out = tmp_path / "game_data.csv"
    out.write_text("game_id,timestamp,experiment,seed,seating_rotation\n", encoding="utf-8")
    time.sleep(0.01)
    os.utime(str(out), None)

    issues = tmp_path / "import_issues.csv"  # deliberately absent
    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["games"], "issues_path": str(issues)},
        {"games": str(out)},
    )
    result = run_extract(cfg, catalog=catalog)
    assert result.skipped is False
    assert issues.exists()  # the (re)build created the report


def test_run_extract_force_rebuild_runs_anyway(tmp_path, catalog):
    runs = tmp_path / "runs"
    db = runs / "exp" / "g_1.db"
    _make_game_db(db, {"gameId": "g", "configuredSyncRandSeed": "9", "configuredMapRandSeed": "9"})
    out = tmp_path / "game_data.csv"
    out.write_text("game_id,timestamp,experiment,seed,seating_rotation\n", encoding="utf-8")
    time.sleep(0.01)
    os.utime(str(out), None)

    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["games"], "force_rebuild": True},
        {"games": str(out)},
    )
    result = run_extract(cfg, catalog=catalog)
    assert result.skipped is False
    assert result.new_rows["games"] == 1


def test_run_extract_force_rebuild_param_overrides_config(tmp_path, catalog):
    # CLI --force-rebuild (the param) overrides data.extract.force_rebuild=false.
    runs = tmp_path / "runs"
    db = runs / "exp" / "g_1.db"
    _make_game_db(db, {"gameId": "g", "configuredSyncRandSeed": "9", "configuredMapRandSeed": "9"})
    out = tmp_path / "game_data.csv"
    out.write_text("game_id,timestamp,experiment,seed,seating_rotation\n", encoding="utf-8")
    time.sleep(0.01)
    os.utime(str(out), None)  # output newer than the DB → would normally skip

    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["games"], "force_rebuild": False},
        {"games": str(out)},
    )
    result = run_extract(cfg, catalog=catalog, force_rebuild=True)
    assert result.skipped is False
    assert result.new_rows["games"] == 1


def test_run_extract_panel_writes_player_type(tmp_path, catalog):
    runs = tmp_path / "runs"
    db = runs / "gemma-4-standard-fixed" / "g1_100.db"
    _make_game_db(
        db,
        metadata={
            "gameId": "g1",
            # one marked treatment seat at config slot 0; the other seat is an
            # unmarked in-game-AI opponent (no model-{id} metadata).
            "model-0": "claude-sonnet-4-5",
            "strategist-0": "simple-strategist-briefed",
            "seatingMap": '{"0": 0}',
            "configuredSyncRandSeed": "5",
            "configuredMapRandSeed": "5",
            "seatingRotation": "0",
        },
        players=[(0, "America", 1), (1, "Rome", 1)],
        summaries=[
            (1, 0, 10, 500, 1, "Pottery"),
            (2, 1, 10, 400, 1, "Pottery"),
        ],
    )
    panel_out = tmp_path / "panel_data.csv"
    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["panel"]},
        {"panel": str(panel_out)},
    )
    result = run_extract(cfg, catalog=catalog)
    assert result.skipped is False

    import csv
    rows = {int(r["player_id"]): r for r in csv.DictReader(panel_out.open(encoding="utf-8"))}
    # marked treatment seat
    assert rows[0]["player_type"] == "Sonnet-4.5-Briefed"
    assert rows[0]["model"] == "claude-sonnet-4-5"
    assert rows[0]["config_slot"] == "0"
    # unmarked seat → VPAI default (→ Vanilla), config_slot -1
    assert rows[1]["player_type"] == "Vanilla"
    assert rows[1]["model"] == "VPAI"
    assert rows[1]["config_slot"] == "-1"


def test_is_decision_changes_predicate():
    from bench.extract.utilities import is_decision_changes
    assert is_decision_changes('["Rationale"]') is True            # status quo + rationale
    assert is_decision_changes('["Rationale","Offense"]') is True  # actual change
    assert is_decision_changes("[]") is False                      # empty
    assert is_decision_changes("") is False
    assert is_decision_changes(None) is False                      # carry-forward


def test_has_real_changes_predicate():
    assert _has_real_changes('["Rationale","Policy"]') is True
    assert _has_real_changes('["Policy"]') is True
    assert _has_real_changes('["Rationale"]') is False
    assert _has_real_changes("[]") is False
    assert _has_real_changes("") is False
    assert _has_real_changes(None) is False


def test_panel_decisions_count_includes_status_quo(tmp_path, catalog):
    runs = tmp_path / "runs"
    db = runs / "gemma-4-standard-fixed" / "g2_100.db"
    _make_game_db(
        db,
        metadata={
            "gameId": "g2",
            "model-0": "claude-sonnet-4-5",
            "strategist-0": "simple-strategist-briefed",
            "seatingMap": '{"0": 0}',
            "configuredSyncRandSeed": "5",
            "configuredMapRandSeed": "5",
            "seatingRotation": "0",
        },
        players=[(0, "America", 1)],
        summaries=[(1, 0, 10, 500, 1, "Pottery")],
        # 2 status-quo decisions + 1 actual change for player 0.
        flavor_rows=[
            (1, 0, 2, '["Rationale"]', "Conquest", "keep status quo", 50, 50),
            (2, 0, 3, '["Rationale"]', "Conquest", "still optimal", 50, 50),
            (3, 0, 4, '["Rationale","Offense"]', "Conquest", "ramp offense", 60, 50),
        ],
    )
    panel_out = tmp_path / "panel_data.csv"
    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["panel"]},
        {"panel": str(panel_out)},
    )
    run_extract(cfg, catalog=catalog)

    import csv
    row = list(csv.DictReader(panel_out.open(encoding="utf-8")))[0]
    assert row["decisions"] == "3"        # all three turns are decisions
    assert row["strategy_changes"] == "1"  # only the one with an actual flavor change


def test_panel_persona_changes_ignore_null_strategist_reapply_loop(tmp_path, catalog):
    runs = tmp_path / "runs"
    db = runs / "null-standard-fixed" / "g3_100.db"
    _make_game_db(
        db,
        metadata={
            "gameId": "g3",
            "model-0": "VPAI",
            "strategist-0": "null-strategist",
            "seatingMap": '{"0": 0}',
            "configuredSyncRandSeed": "5",
            "configuredMapRandSeed": "5",
            "seatingRotation": "0",
        },
        players=[(0, "America", 1)],
        summaries=[(1, 0, 10, 500, 1, "Pottery")],
        persona_rows=[
            (1, 0, 0, 1, '["DiplomaticBalance","Boldness"]', "Null agent baseline", 5, 5),
            (2, 0, 1, 2, '["DiplomaticBalance"]', "Tweaked by In-Game AI (Null agent baseline)", 6, 5),
            (3, 0, 1, 3, '["DiplomaticBalance","Rationale"]', "Null agent baseline", 5, 5),
            (4, 0, 2, 4, '["DiplomaticBalance"]', "Tweaked by In-Game AI (Null agent baseline)", 6, 5),
            (5, 0, 2, 5, '["DiplomaticBalance","Rationale"]', "Null agent baseline", 5, 5),
        ],
    )
    panel_out = tmp_path / "panel_data.csv"
    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["panel"]},
        {"panel": str(panel_out)},
    )
    run_extract(cfg, catalog=catalog)

    import csv
    row = list(csv.DictReader(panel_out.open(encoding="utf-8")))[0]
    assert row["player_type"] == "Null"
    assert row["persona_changes"] == "1"


def test_panel_persona_changes_count_distinct_strategist_authored_states(tmp_path, catalog):
    runs = tmp_path / "runs"
    db = runs / "gemma-4-standard-fixed" / "g4_100.db"
    _make_game_db(
        db,
        metadata={
            "gameId": "g4",
            "model-0": "claude-sonnet-4-5",
            "strategist-0": "simple-strategist-briefed",
            "seatingMap": '{"0": 0}',
            "configuredSyncRandSeed": "5",
            "configuredMapRandSeed": "5",
            "seatingRotation": "0",
        },
        players=[(0, "America", 1)],
        summaries=[(1, 0, 10, 500, 1, "Pottery")],
        persona_rows=[
            (1, 0, 0, 1, '["DiplomaticBalance","Boldness"]', "Opening persona", 5, 5),
            (2, 0, 1, 2, '["DiplomaticBalance"]', "Tweaked by In-Game AI (Opening persona)", 6, 5),
            (3, 0, 1, 3, '["DiplomaticBalance","Rationale"]', "Opening persona", 5, 5),
            (4, 0, 2, 4, '["Boldness"]', "More aggressive diplomacy", 5, 7),
        ],
    )
    panel_out = tmp_path / "panel_data.csv"
    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["panel"]},
        {"panel": str(panel_out)},
    )
    run_extract(cfg, catalog=catalog)

    import csv
    row = list(csv.DictReader(panel_out.open(encoding="utf-8")))[0]
    assert row["persona_changes"] == "2"
