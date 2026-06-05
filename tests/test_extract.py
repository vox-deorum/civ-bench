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
from bench.extract.utilities import UNCONTROLLED


# ── fixtures / helpers ───────────────────────────────────────────────────────
@pytest.fixture
def catalog(configs_dir) -> Catalog:
    return Catalog.from_paths(configs_dir / "models.json", configs_dir / "experiments.json")


def _make_game_db(path: Path, metadata: dict, players=None, summaries=None) -> None:
    """Create a minimal game DB with a GameMetadata Key→Value table (+ optional rows)."""
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
        cur.execute("CREATE TABLE StrategyChanges (ID INTEGER, Key INTEGER, Turn INTEGER, GrandStrategy TEXT, Changes TEXT, Rationale TEXT)")
        cur.execute("CREATE TABLE PersonaChanges (Key INTEGER, Changes TEXT)")
        cur.execute("CREATE TABLE ResearchChanges (Key INTEGER, Changes TEXT)")
        cur.execute("CREATE TABLE PolicyChanges (Key INTEGER, Changes TEXT)")
        cur.execute("CREATE TABLE GameEvents (Turn INTEGER, Type TEXT, Payload TEXT, Player0 INTEGER, Player1 INTEGER)")

    conn.commit()
    conn.close()


def _run_config(tmp_path: Path, extract: dict, tables: dict) -> RunConfig:
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
    # an absent player still gets identity config_slot
    assert info.config_slot(2) == 2


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


def test_identity_legacy_game_falls_back_to_seat_map(catalog):
    # No model-{id}/strategist-{id} metadata → static (condition, slot) fallback.
    ids = compose_identities({}, [0], "unknown-condition", catalog, extract_seeding_fields({}))
    assert ids[0]["player_type"] == "Player 0"
    assert ids[0]["model"] == "N/A"


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
    time.sleep(0.01)
    os.utime(str(out), None)  # output newer than the DB

    cfg = _run_config(
        tmp_path,
        {"enabled": True, "runs_dir": str(runs), "outputs": ["games"]},
        {"games": str(out)},
    )
    result = run_extract(cfg, catalog=catalog)
    assert result.skipped is True
    assert "newer than the source" in result.reason


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


def test_run_extract_panel_writes_player_type(tmp_path, catalog):
    runs = tmp_path / "runs"
    db = runs / "2026-staff-standard" / "g1_100.db"
    _make_game_db(
        db,
        metadata={
            "gameId": "g1",
            "model-0": "claude-sonnet-4-5",
            "strategist-0": "simple-strategist-briefed",
            "model-1": "VPAI",
            "strategist-1": "none-strategist",
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
    assert rows[0]["player_type"] == "Sonnet-4.5-Briefed"
    assert rows[0]["model"] == "claude-sonnet-4-5"
    assert rows[1]["player_type"] == "Vanilla"
    assert rows[0]["config_slot"] == "0"
