"""Stage 1 — malformed-DB import-issue recording tests.

A corrupt game DB must no longer (a) crash the run, (b) emit all-N/A panel rows,
or (c) leave a print that names neither the game nor its experiment/seed/rotation.
Each failure site records a single :class:`ImportIssue` (deduped by game_id across
stages); the log prints a grouped summary and persists ``import_issues.csv``.

Fixtures are tiny and synthetic (no machine data roots, per AGENTS.md): a corrupt
DB is just a file with a non-SQLite header — its first query raises
``sqlite3.DatabaseError`` ("file is not a database"), the same path a
"database disk image is malformed" image takes.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from bench.extract.errors import ExtractError
from bench.extract.extract_games import extract_game_row
from bench.extract.extract_model_tokens import _game_player_types
from bench.extract.extract_panel import extract_game_panel_data
from bench.extract.extract_turns import _fetch_flavor_events, extract_game_turn_data
from bench.extract.issues import UNKNOWN, ImportIssueLog
from bench.extract.utilities import (
    UNCONTROLLED,
    is_schema_mismatch,
    open_database_readonly,
)


GAME_ID = "abc-def-0123"
EXPERIMENT = "2026-staff-sonnet"


def _corrupt_db(tmp_path, game_id=GAME_ID, experiment=EXPERIMENT, ts=1700000000000):
    """Write a non-SQLite file at ``…/<experiment>/<game_id>_<ts>.db``."""
    exp_dir = tmp_path / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    db = exp_dir / f"{game_id}_{ts}.db"
    db.write_bytes(b"this is not a sqlite database\x00" * 8)
    return str(db)


def _valid_db(tmp_path, metadata, game_id=GAME_ID, experiment=EXPERIMENT, ts=1700000000000):
    """A minimal *readable* game DB (GameMetadata + one major player, no Flavor table)."""
    exp_dir = tmp_path / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    db = exp_dir / f"{game_id}_{ts}.db"
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("CREATE TABLE GameMetadata (Key TEXT, Value TEXT)")
    for key, value in metadata.items():
        cur.execute("INSERT INTO GameMetadata VALUES (?, ?)", (key, str(value)))
    cur.execute("CREATE TABLE PlayerInformations (Key INTEGER, Civilization TEXT, IsMajor INTEGER)")
    cur.execute("INSERT INTO PlayerInformations VALUES (0, 'Rome', 1)")
    conn.commit()
    conn.close()
    return str(db)


# ── DB-level wiring: a corrupt DB records, returns nothing, never crashes ─────
def test_panel_skips_corrupt_game_and_records_once(tmp_path):
    db = _corrupt_db(tmp_path)
    log = ImportIssueLog()

    rows = extract_game_panel_data(db, issues=log)

    assert rows == []                       # skipped, no all-N/A rows
    assert len(log) == 1
    issue = log.issues()[0]
    assert issue.game_id == GAME_ID
    assert issue.experiment == EXPERIMENT
    assert issue.stages == {"panel"}
    assert issue.seed == UNKNOWN            # metadata unreadable on a corrupt DB
    assert issue.seating_rotation == UNKNOWN


def test_one_merged_issue_across_all_four_stages(tmp_path):
    db = _corrupt_db(tmp_path)
    log = ImportIssueLog()

    assert extract_game_row(db, issues=log) is None
    assert extract_game_panel_data(db, issues=log) == []
    assert extract_game_turn_data(db, issues=log) == []
    assert _game_player_types(db, catalog=None, issues=log) == ({}, {})

    # Deduped by game_id → one entry whose stages list every affected exporter.
    assert len(log) == 1
    issue = log.issues()[0]
    assert issue.game_id == GAME_ID
    assert issue.stages == {"games", "panel", "turns", "tokens"}


# ── ImportIssueLog merge / upgrade / CSV semantics ───────────────────────────
def test_record_reads_seed_and_rotation_from_metadata(tmp_path):
    db = _corrupt_db(tmp_path)
    log = ImportIssueLog()

    log.record(
        stage="panel", db_path=db, message="boom",
        metadata={"configuredSyncRandSeed": "5", "seatingRotation": "2"},
    )

    issue = log.issues()[0]
    assert issue.seed == 5
    assert issue.seating_rotation == 2


def test_uncontrolled_metadata_is_not_unknown(tmp_path):
    db = _corrupt_db(tmp_path)
    log = ImportIssueLog()
    # Readable metadata with no controlled seeding → the -1 sentinel, not UNKNOWN.
    log.record(stage="games", db_path=db, message="x", metadata={})
    # metadata={} is falsy → treated as unreadable → UNKNOWN; use a non-empty dict.
    log.record(stage="panel", db_path=db, message="x", metadata={"turn": "100"})
    issue = log.issues()[0]
    assert issue.seed == UNCONTROLLED
    assert issue.seating_rotation == UNCONTROLLED


def test_unknown_seed_is_upgraded_when_a_later_stage_reads_metadata(tmp_path):
    db = _corrupt_db(tmp_path)
    log = ImportIssueLog()

    # tokens hits first with no metadata, then panel recovers the real seeding.
    log.record(stage="tokens", db_path=db, message="trace boom", player_id=3)
    log.record(
        stage="panel", db_path=db, message="game boom",
        metadata={"configuredSyncRandSeed": "7", "seatingRotation": "0"},
    )

    assert len(log) == 1
    issue = log.issues()[0]
    assert issue.seed == 7
    assert issue.seating_rotation == 0
    assert issue.stages == {"tokens", "panel"}
    assert issue.players == {3}


def test_explicit_ids_override_path_derivation(tmp_path):
    # A player-trace path ({uuid}-player-{n}.db) has no "_" game pattern, so the
    # tokens stage must pass game_id/experiment explicitly.
    trace = tmp_path / EXPERIMENT / f"{GAME_ID}-player-2.db"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_bytes(b"nope")
    log = ImportIssueLog()

    log.record(
        stage="tokens", db_path=str(trace), game_id=GAME_ID,
        experiment=EXPERIMENT, player_id=2, message="trace boom",
    )

    issue = log.issues()[0]
    assert issue.game_id == GAME_ID
    assert issue.experiment == EXPERIMENT
    assert issue.players == {2}


def test_write_csv_emits_header_and_rows(tmp_path):
    db = _corrupt_db(tmp_path)
    log = ImportIssueLog()
    log.record(stage="panel", db_path=db, message="database disk image is malformed", player_id=0)
    log.record(stage="turns", db_path=db, message="database disk image is malformed", player_id=1)

    out = tmp_path / "import_issues.csv"
    assert log.write_csv(str(out)) is True

    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["game_id"] == GAME_ID
    assert row["experiment"] == EXPERIMENT
    assert row["stages"] == "panel|turns"   # sorted, pipe-joined
    assert row["players"] == "0|1"


def test_write_csv_is_header_only_when_clean(tmp_path):
    log = ImportIssueLog()
    assert not log
    out = tmp_path / "import_issues.csv"
    assert log.write_csv(str(out)) is True

    with open(out, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rest = list(reader)

    assert header[0] == "game_id"
    assert rest == []


# ── P1.1: a controlled-seed mismatch aborts, even on panel/turns-only runs ────
@pytest.mark.parametrize("extract_fn", [extract_game_panel_data, extract_game_turn_data])
def test_seed_mismatch_aborts_not_recorded(tmp_path, extract_fn):
    db = _valid_db(tmp_path, {
        "gameId": GAME_ID,
        "configuredSyncRandSeed": "1",
        "configuredMapRandSeed": "2",
    })
    log = ImportIssueLog()
    # The mismatch must propagate as a hard ExtractError (rule 14), never be
    # swallowed into [] nor demoted to a recorded "import issue".
    with pytest.raises(ExtractError, match="mismatched seeds"):
        extract_fn(db, issues=log)
    assert len(log) == 0


# ── P1.2: localized schema gaps tolerated, corruption never silently swallowed ─
def test_is_schema_mismatch_distinguishes_missing_table_from_corruption(tmp_path):
    db = _valid_db(tmp_path, {"gameId": GAME_ID})
    conn, cursor = open_database_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError) as missing:
            cursor.execute("SELECT * FROM FlavorChanges")
    finally:
        conn.close()
    # A real "no such table" is tolerable; corruption/locking is not.
    assert is_schema_mismatch(missing.value) is True
    assert is_schema_mismatch(sqlite3.DatabaseError("database disk image is malformed")) is False
    assert is_schema_mismatch(sqlite3.OperationalError("database is locked")) is False


def test_fetch_flavor_events_tolerates_missing_table(tmp_path):
    # An older DB without FlavorChanges falls back to {} (not an error/issue).
    db = _valid_db(tmp_path, {"gameId": GAME_ID})
    conn, cursor = open_database_readonly(db)
    try:
        assert _fetch_flavor_events(cursor, [0]) == {}
    finally:
        conn.close()


# ── P1: the durable report is reconciled across runs, not clobbered ───────────
def _game_path(tmp_path, game_id=GAME_ID, experiment=EXPERIMENT):
    return str(tmp_path / experiment / f"{game_id}_1700000000000.db")


def _written_report(tmp_path, **record_kwargs):
    """Write a one-issue report to disk and return its path."""
    log = ImportIssueLog()
    log.record(db_path=_game_path(tmp_path), **record_kwargs)
    report = tmp_path / "import_issues.csv"
    assert log.write_csv(str(report))
    return str(report)


def test_prior_issue_carried_forward_when_not_reexamined(tmp_path):
    # A trace issue on a game still in the DB set but not re-examined this run must
    # survive (the original bug erased it on the next incremental run).
    report = _written_report(
        tmp_path, stage="tokens", player_id=2, message="trace boom",
        metadata={"configuredSyncRandSeed": "7", "seatingRotation": "1"},
    )
    log = ImportIssueLog()
    log.load(report)
    log.reconcile(available_game_ids={GAME_ID})

    assert len(log) == 1
    issue = log.issues()[0]
    assert issue.stages == {"tokens"}
    assert issue.players == {2}
    assert issue.seed == 7 and issue.seating_rotation == 1  # round-tripped, not lost


def test_prior_issue_dropped_when_db_gone(tmp_path):
    report = _written_report(tmp_path, stage="panel", message="boom")
    log = ImportIssueLog()
    log.load(report)
    log.reconcile(available_game_ids=set())  # the game's DB no longer exists
    assert len(log) == 0


def test_reexamined_healthy_stage_clears_prior(tmp_path):
    report = _written_report(tmp_path, stage="panel", message="boom")
    log = ImportIssueLog()
    log.load(report)
    log.mark_evaluated("panel", GAME_ID)  # re-read this run and it was fine now
    log.reconcile(available_game_ids={GAME_ID})
    assert len(log) == 0


def test_reconcile_keeps_unreexamined_stage_drops_reexamined(tmp_path):
    # Prior issue spans panel+tokens. This run re-examines panel (still fails) but
    # skips tokens → the merged issue keeps tokens (carried) and panel (fresh).
    log0 = ImportIssueLog()
    p = _game_path(tmp_path)
    log0.record(stage="panel", db_path=p, message="panel boom")
    log0.record(stage="tokens", db_path=p, player_id=2, message="trace boom")
    report = tmp_path / "import_issues.csv"
    assert log0.write_csv(str(report))

    log = ImportIssueLog()
    log.load(str(report))
    log.mark_evaluated("panel", GAME_ID)
    log.record(stage="panel", db_path=p, message="panel boom again")
    log.reconcile(available_game_ids={GAME_ID})

    assert len(log) == 1
    assert log.issues()[0].stages == {"panel", "tokens"}


# ── P2: a trace failure reports the (readable) game's seed/rotation ───────────
def test_trace_failure_reports_game_seed_rotation(tmp_path, configs_dir):
    from bench.catalog import Catalog
    from bench.extract.extract_model_tokens import extract_game_model_token_data

    game_db = _valid_db(tmp_path, {
        "gameId": GAME_ID,
        "configuredSyncRandSeed": "7",
        "configuredMapRandSeed": "7",
        "seatingRotation": "1",
    })
    # A corrupt player-trace beside the (readable) game DB.
    trace = Path(game_db).parent / f"{GAME_ID}-player-2.db"
    trace.write_bytes(b"not a sqlite trace db\x00" * 4)

    catalog = Catalog.from_paths(configs_dir / "models.json", configs_dir / "experiments.json")
    log = ImportIssueLog()
    extract_game_model_token_data(game_db, catalog, issues=log)

    assert len(log) == 1
    issue = log.issues()[0]
    assert issue.stages == {"tokens"}
    assert issue.players == {2}
    # The game metadata was readable → real seed/rotation, not "unknown".
    assert issue.seed == 7
    assert issue.seating_rotation == 1
