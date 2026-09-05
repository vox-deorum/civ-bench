"""``civ-bench fix``: best-effort recovery of malformed game SQLite DBs.

Covers the recovery core (:func:`repair_database`) across its layered strategies, the
orchestrator (:func:`run_fix`) including the atomic original→``.bak`` swap and its
refusals (existing backup, not-found, ambiguous), and a CLI smoke test proving the
``fix`` subcommand short-circuits before the generic ``--dry-run`` → print-DAG path.
"""

from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path

import pytest

from bench.config.models import OutputConfig, RunConfig
from bench.extract.issues import ISSUE_FIELDNAMES
from bench.fix import FixError, FixResult, repair_database, run_fix


# ── fixtures / helpers ───────────────────────────────────────────────────────
def _make_db(path: Path, n_rows: int = 50, with_index: bool = True, wide: bool = False) -> None:
    """Create a healthy two-table DB (``items`` + ``meta``), optionally indexed/large."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, val INTEGER)")
    pad = ("x" * 400) if wide else ""  # widen rows so the table spans many pages
    cur.executemany(
        "INSERT INTO items (id, name, val) VALUES (?, ?, ?)",
        [(i, f"name-{i}-{pad}", i * i) for i in range(1, n_rows + 1)],
    )
    cur.execute("CREATE TABLE meta (k TEXT, v TEXT)")
    cur.executemany("INSERT INTO meta VALUES (?, ?)", [("a", "1"), ("b", "2")])
    if with_index:
        cur.execute("CREATE INDEX ix_items_val ON items(val)")
    conn.commit()
    conn.close()


def _make_corrupt_db(path: Path, n_rows: int = 40) -> None:
    """Create a DB whose index points at a bogus rootpage → fails quick_check.

    The table data is intact, so recovery rebuilds the index clean: a faithful stand-in
    for the real "malformed disk image" games (whose corruption is index-tree damage).
    """
    _make_db(path, n_rows=n_rows, with_index=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute("UPDATE sqlite_master SET rootpage=2 WHERE name='ix_items_val'")
    conn.commit()
    conn.close()


def _write_issues(path: Path, rows: list[dict]) -> None:
    """Write a minimal ``import_issues.csv`` (only db_name/game_id/experiment matter)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ISSUE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ISSUE_FIELDNAMES})


def _cfg(tmp_path: Path, runs_dir: Path, issues_path: Path) -> RunConfig:
    return RunConfig(
        name="test",
        seed=1,
        config_path=tmp_path / "benchmark.json",
        raw={},
        output=OutputConfig(),
        data={
            "extract": {"runs_dir": str(runs_dir), "issues_path": str(issues_path)},
            "tables": {},
        },
    )


def _reads_cleanly(path: Path) -> bool:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        conn.execute("SELECT * FROM items").fetchall()
        conn.close()
        return True
    except sqlite3.DatabaseError:
        return False


def _row_count(path: Path, table: str = "items") -> int:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _quick_check_ok(path: Path) -> bool:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        conn.close()


# ── repair_database core ─────────────────────────────────────────────────────
def test_repair_healthy_db_uses_iterdump(tmp_path):
    src = tmp_path / "g.db"
    _make_db(src, n_rows=30)
    dst = tmp_path / "out.db"

    report = repair_database(str(src), str(dst))

    assert report.success
    assert report.strategy == "iterdump"
    assert set(report.tables_recovered) == {"items", "meta"}
    assert report.rows_recovered == 32  # 30 items + 2 meta
    assert report.integrity == "ok"
    assert _quick_check_ok(dst) and _row_count(dst) == 30


def test_repair_tolerant_rebuild_recovers_everything(monkeypatch, tmp_path):
    # Force strategy A off to deterministically exercise the tolerant-rebuild path
    # (create tables → bulk-copy rows → rebuild indexes → quick_check) on a clean DB.
    src = tmp_path / "g.db"
    _make_db(src, n_rows=40)
    monkeypatch.setattr("bench.fix.repair._try_iterdump", lambda *a, **k: False)
    dst = tmp_path / "out.db"

    report = repair_database(str(src), str(dst))

    assert report.success
    assert report.strategy == "tolerant-rebuild"
    assert set(report.tables_recovered) == {"items", "meta"}
    assert report.rows_recovered == 42
    assert report.rows_skipped == 0
    assert _quick_check_ok(dst) and _row_count(dst) == 40


def test_repair_rowid_range_scan_path(monkeypatch, tmp_path):
    # Force iterdump off and make the bulk fast-path raise, so _copy_table falls into
    # the rowid range walk, which must still recover every row on a healthy DB.
    import bench.fix.repair as repair_mod

    src = tmp_path / "g.db"
    _make_db(src, n_rows=2500, with_index=False)
    monkeypatch.setattr(repair_mod, "_try_iterdump", lambda *a, **k: False)

    def _boom(*_a, **_k):
        raise sqlite3.DatabaseError("simulated corrupt page")

    monkeypatch.setattr(repair_mod, "_bulk_copy", _boom)
    dst = tmp_path / "out.db"

    report = repair_database(str(src), str(dst))

    assert report.success and report.strategy == "tolerant-rebuild"
    assert report.rows_recovered == 2502  # all rows recovered via the range scan
    assert report.rows_skipped == 0
    assert _row_count(dst) == 2500


def test_repair_corrupt_index_is_recovered(tmp_path):
    src = tmp_path / "g.db"
    _make_db(src, n_rows=60)
    # Point the index at a bogus rootpage → integrity_check fails, but the table data
    # pages are intact, so recovery rebuilds the index clean from the recovered rows.
    conn = sqlite3.connect(str(src))
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute("UPDATE sqlite_master SET rootpage=2 WHERE name='ix_items_val'")
    conn.commit()
    conn.close()
    assert not _quick_check_ok(src)  # the source is genuinely malformed

    dst = tmp_path / "out.db"
    report = repair_database(str(src), str(dst))

    assert report.success
    assert _quick_check_ok(dst) and _row_count(dst) == 60


def test_repair_garbage_file_fails_gracefully(tmp_path):
    src = tmp_path / "g.db"
    src.write_bytes(b"this is definitely not a sqlite database" * 32)
    dst = tmp_path / "out.db"

    report = repair_database(str(src), str(dst))

    assert not report.success
    assert report.strategy == "failed"
    assert not dst.exists()  # the failed attempt leaves no output


def test_repair_byte_flip_recovers_unaffected_rows(tmp_path):
    """Soft test: a mid-file page flip should still recover most rows via range scan."""
    src = tmp_path / "g.db"
    _make_db(src, n_rows=4000, with_index=False, wide=True)
    data = bytearray(src.read_bytes())
    page_size = 4096
    offset = 12 * page_size + 100  # mid-file leaf page, well past the header/page-1
    if offset >= len(data):
        pytest.skip("DB smaller than expected; cannot place a mid-file flip")
    for i in range(offset, offset + 256):  # smear a chunk to force a malformed page
        data[i] ^= 0xFF
    src.write_bytes(data)
    if _reads_cleanly(src):
        pytest.skip("byte-flip did not corrupt this SQLite build")

    dst = tmp_path / "out.db"
    report = repair_database(str(src), str(dst))

    assert report.success
    assert _quick_check_ok(dst)
    assert _row_count(dst) > 0  # recovered rows on at least one side of the bad page


# ── run_fix orchestration ────────────────────────────────────────────────────
def test_run_fix_repairs_and_swaps(tmp_path):
    runs = tmp_path / "runs"
    db = runs / "exp" / "abc_111.db"
    _make_corrupt_db(db, n_rows=25)
    original = db.read_bytes()
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": "abc_111.db", "game_id": "abc", "experiment": "exp"}])

    result = run_fix(_cfg(tmp_path, runs, issues))

    assert isinstance(result, FixResult)
    assert [o.status for o in result.outcomes] == ["repaired"]
    backup = db.with_suffix(".db.bak")
    assert backup.exists() and backup.read_bytes() == original  # corrupt original preserved
    assert _quick_check_ok(db) and _row_count(db) == 25       # fixed DB at the canonical path


def test_run_fix_healthy_db_is_left_untouched(tmp_path):
    # A flagged game whose DB is actually fine: examined, found healthy, left as-is.
    # No rewrite, no backup ("if nothing changed, delete the bak").
    runs = tmp_path / "runs"
    db = runs / "exp" / "abc_111.db"
    _make_db(db, n_rows=20)
    original = db.read_bytes()
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": "abc_111.db", "game_id": "abc"}])

    result = run_fix(_cfg(tmp_path, runs, issues))

    assert [o.status for o in result.outcomes] == ["healthy"]
    assert not db.with_suffix(".db.bak").exists()
    assert db.read_bytes() == original  # untouched, byte for byte


def test_run_fix_repairs_game_and_skips_healthy_traces(tmp_path):
    # The headline case: one flagged game → fix the corrupt game DB AND examine its
    # player-trace DBs; healthy traces are left alone with no backup, non-DB siblings
    # (.Civ5Save) are ignored entirely.
    uuid = "11111111-2222-3333-4444-555555555555"
    exp = tmp_path / "runs" / "exp"
    main = exp / f"{uuid}_111.db"
    trace1 = exp / f"{uuid}-player-1.db"
    trace7 = exp / f"{uuid}-player-7.db"
    _make_corrupt_db(main, n_rows=30)
    _make_db(trace1, n_rows=12)
    _make_db(trace7, n_rows=12)
    save = exp / f"{uuid}_111.Civ5Save"
    save.write_bytes(b"not a sqlite save file")
    trace1_bytes, trace7_bytes, save_bytes = trace1.read_bytes(), trace7.read_bytes(), save.read_bytes()

    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": f"{uuid}_111.db", "game_id": uuid}])

    result = run_fix(_cfg(tmp_path, tmp_path / "runs", issues))

    by_name = {o.db_name: o.status for o in result.outcomes}
    assert by_name == {
        f"{uuid}_111.db": "repaired",
        f"{uuid}-player-1.db": "healthy",
        f"{uuid}-player-7.db": "healthy",
    }
    assert main.with_suffix(".db.bak").exists() and _quick_check_ok(main)  # game DB fixed + backed up
    assert not trace1.with_suffix(".db.bak").exists() and trace1.read_bytes() == trace1_bytes
    assert not trace7.with_suffix(".db.bak").exists() and trace7.read_bytes() == trace7_bytes
    assert save.read_bytes() == save_bytes  # non-DB sibling never touched


def test_run_fix_repairs_corrupt_trace(tmp_path):
    # A corrupt telemetry DB is repaired even though the game DB itself is healthy.
    uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    exp = tmp_path / "runs" / "exp"
    main = exp / f"{uuid}_111.db"
    trace = exp / f"{uuid}-player-3.db"
    _make_db(main, n_rows=10)
    _make_corrupt_db(trace, n_rows=20)
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": f"{uuid}_111.db", "game_id": uuid}])

    result = run_fix(_cfg(tmp_path, tmp_path / "runs", issues))

    by_name = {o.db_name: o.status for o in result.outcomes}
    assert by_name == {f"{uuid}_111.db": "healthy", f"{uuid}-player-3.db": "repaired"}
    assert not main.with_suffix(".db.bak").exists()              # healthy game DB → no backup
    assert trace.with_suffix(".db.bak").exists() and _quick_check_ok(trace)


def test_run_fix_dry_run_writes_nothing(tmp_path):
    runs = tmp_path / "runs"
    db = runs / "exp" / "abc_111.db"
    _make_corrupt_db(db, n_rows=10)
    original = db.read_bytes()
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": "abc_111.db", "game_id": "abc"}])

    result = run_fix(_cfg(tmp_path, runs, issues), dry_run=True)

    assert [o.status for o in result.outcomes] == ["dry-run"]
    assert not db.with_suffix(".db.bak").exists()
    assert db.read_bytes() == original


def test_run_fix_refuses_to_clobber_existing_bak(tmp_path):
    runs = tmp_path / "runs"
    db = runs / "exp" / "abc_111.db"
    _make_corrupt_db(db, n_rows=10)
    backup = db.with_suffix(".db.bak")
    backup.write_bytes(b"precious original backup")
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": "abc_111.db", "game_id": "abc"}])

    # Without --force: skipped, backup untouched.
    result = run_fix(_cfg(tmp_path, runs, issues))
    assert [o.status for o in result.outcomes] == ["skipped-bak-exists"]
    assert backup.read_bytes() == b"precious original backup"

    # With --force: repaired, backup replaced by the (now-current) original.
    result = run_fix(_cfg(tmp_path, runs, issues), force=True)
    assert [o.status for o in result.outcomes] == ["repaired"]
    assert _quick_check_ok(db)


def test_run_fix_not_found_and_ambiguous(tmp_path):
    runs = tmp_path / "runs"
    _make_db(runs / "expA" / "dup_1.db", n_rows=5)
    _make_db(runs / "expB" / "dup_1.db", n_rows=5)  # same basename in two folders
    issues = tmp_path / "import_issues.csv"
    _write_issues(
        issues,
        [
            {"db_name": "missing_9.db", "game_id": "miss"},
            {"db_name": "dup_1.db", "game_id": "dup"},
        ],
    )

    result = run_fix(_cfg(tmp_path, runs, issues))

    by_name = {o.db_name: o.status for o in result.outcomes}
    assert by_name == {"missing_9.db": "not-found", "dup_1.db": "ambiguous"}
    # Neither ambiguous file was touched.
    assert not (runs / "expA" / "dup_1.db.bak").exists()
    assert not (runs / "expB" / "dup_1.db.bak").exists()


def test_run_fix_no_issues_file_is_noop(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    result = run_fix(_cfg(tmp_path, runs, tmp_path / "absent.csv"))
    assert result.outcomes == []


def test_run_fix_is_idempotent(tmp_path):
    runs = tmp_path / "runs"
    db = runs / "exp" / "abc_111.db"
    _make_corrupt_db(db, n_rows=15)
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": "abc_111.db", "game_id": "abc"}])

    first = run_fix(_cfg(tmp_path, runs, issues))
    assert [o.status for o in first.outcomes] == ["repaired"]
    fixed_bytes = db.read_bytes()

    # Second run: the DB is now healthy, so it is recognised as such and left untouched.
    second = run_fix(_cfg(tmp_path, runs, issues))
    assert [o.status for o in second.outcomes] == ["healthy"]
    assert db.read_bytes() == fixed_bytes


def test_run_fix_failed_db_leaves_original_and_no_temp(tmp_path):
    runs = tmp_path / "runs"
    db = runs / "exp" / "abc_111.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    garbage = b"not a database at all" * 16
    db.write_bytes(garbage)
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": "abc_111.db", "game_id": "abc"}])

    result = run_fix(_cfg(tmp_path, runs, issues))

    assert [o.status for o in result.outcomes] == ["failed"]
    assert db.read_bytes() == garbage                      # original untouched
    assert not db.with_suffix(".db.bak").exists()          # no backup made
    leftover = [p.name for p in db.parent.iterdir() if ".fix-" in p.name]
    assert leftover == []                                  # temp cleaned up


def test_run_fix_missing_runs_dir_raises(tmp_path):
    # An operational failure (wrong runs_dir) must be loud, not a silent zero-match run.
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": "abc_111.db", "game_id": "abc"}])
    cfg = _cfg(tmp_path, tmp_path / "nonexistent_runs", issues)
    with pytest.raises(FixError, match="runs_dir"):
        run_fix(cfg)


def test_run_fix_swap_failure_restores_original(monkeypatch, tmp_path):
    import bench.fix.runner as runner_mod

    runs = tmp_path / "runs"
    db = runs / "exp" / "abc_111.db"
    _make_corrupt_db(db, n_rows=12)
    original = db.read_bytes()
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": "abc_111.db", "game_id": "abc"}])

    real_replace = os.replace

    def flaky_replace(src, dst):
        if str(src).endswith(".tmp"):  # the second move (tmp → canonical) fails
            raise OSError("simulated rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(runner_mod.os, "replace", flaky_replace)

    result = run_fix(_cfg(tmp_path, runs, issues))

    assert [o.status for o in result.outcomes] == ["failed"]
    assert db.exists() and db.read_bytes() == original   # original rolled back into place
    assert not db.with_suffix(".db.bak").exists()         # backup undone by the rollback
    leftover = [p.name for p in db.parent.iterdir() if ".fix-" in p.name]
    assert leftover == []                                 # temp cleaned up


def test_repair_recovers_tail_when_bounds_unreadable(monkeypatch, tmp_path):
    # Regression: a corrupt page that blocks the bulk copy AND hides MIN/MAX(rowid) used
    # to fall back to a plain scan that truncated the table's tail (losing everything past
    # the bad page). The rowid walk must now recover the whole table regardless: a rowid
    # table can always be range-walked; unreadable bounds are no reason to drop the tail.
    import bench.fix.repair as repair_mod

    src = tmp_path / "g.db"
    _make_db(src, n_rows=1500, with_index=False)
    monkeypatch.setattr(repair_mod, "_try_iterdump", lambda *a, **k: False)

    def _boom(*_a, **_k):
        raise sqlite3.DatabaseError("simulated corrupt page")

    monkeypatch.setattr(repair_mod, "_bulk_copy", _boom)
    monkeypatch.setattr(repair_mod, "_rowid_bounds", lambda *a, **k: (None, None))
    dst = tmp_path / "out.db"

    report = repair_database(str(src), str(dst))

    assert report.success and report.strategy == "tolerant-rebuild"
    assert report.rows_recovered == 1502    # 1500 items + 2 meta, whole tail recovered
    assert report.rows_skipped == 0
    assert report.tables_partial == []      # nothing lost → no partial-table note
    assert _row_count(dst) == 1500


def test_copy_by_rowid_gives_up_after_skip_limit(monkeypatch):
    # Safety guard: if the rowid b-tree is unreadable (every window errors), the walk must
    # not grind toward the open upper bound; it stops after _SKIP_LIMIT consecutive
    # unreadable rowids. Because no real rows ever followed those skips, they are *not*
    # reported as lost rows (they were end-probing, not data); only `truncated` is set.
    import bench.fix.repair as repair_mod

    monkeypatch.setattr(repair_mod, "_SKIP_LIMIT", 5)
    monkeypatch.setattr(repair_mod, "_rowid_bounds", lambda *a, **k: (None, None))

    class _DeadSrc:  # every range read raises → the b-tree cannot be walked at all
        def execute(self, _sql, _params=None):
            raise sqlite3.DatabaseError("b-tree unreadable")

    dst = sqlite3.connect(":memory:")
    dst.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")

    copied, skipped, truncated = repair_mod._copy_by_rowid(
        _DeadSrc(), dst, "t", "INSERT INTO t VALUES (?, ?)"
    )
    dst.close()

    assert copied == 0
    assert skipped == 0       # the give-up probes are speculative, never counted as lost
    assert truncated is True


def test_partial_note_reads_as_kept_not_dropped():
    # The user-facing fix: partial row loss in a recovered table must never read as a
    # dropped table. A clean table produces no note at all.
    from bench.fix.repair import _partial_note

    assert _partial_note("GameEvents", copied=5, skipped=0, truncated=False) is None

    skipped_note = _partial_note("GameEvents", copied=739223, skipped=9, truncated=False)
    assert skipped_note == "GameEvents: recovered 739223 row(s); 9 on a corrupt page were unreadable"
    assert "dropped" not in skipped_note

    trunc_note = _partial_note("GameEvents", copied=10, skipped=2, truncated=True)
    assert "recovered 10 row(s)" in trunc_note and "corrupt region" in trunc_note
    assert "dropped" not in trunc_note


def test_count_rows_on_disk_is_accurate(tmp_path):
    # The honest denominator: a raw page scan must count exactly the table rows physically
    # present (user tables + sqlite_master's schema rows), independent of the b-tree.
    from bench.fix.repair import _count_rows_on_disk

    db = tmp_path / "g.db"
    _make_db(db, n_rows=300, with_index=True, wide=True)  # spans many leaf pages + an interior page

    conn = sqlite3.connect(str(db))
    expected = sum(
        conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("items", "meta", "sqlite_master")
    )
    conn.close()

    assert _count_rows_on_disk(str(db)) == expected
    # A non-SQLite file is reported as unknown, not zero.
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not a sqlite database")
    assert _count_rows_on_disk(str(bad)) is None


def test_run_fix_clears_stale_sidecars(tmp_path):
    runs = tmp_path / "runs"
    db = runs / "exp" / "abc_111.db"
    _make_corrupt_db(db, n_rows=8)
    wal = Path(str(db) + "-wal")
    shm = Path(str(db) + "-shm")
    wal.write_bytes(b"stale wal")
    shm.write_bytes(b"stale shm")
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": "abc_111.db", "game_id": "abc"}])

    result = run_fix(_cfg(tmp_path, runs, issues))

    assert [o.status for o in result.outcomes] == ["repaired"]
    assert not wal.exists() and not shm.exists()  # stale sidecars cleared next to fixed DB
    assert _quick_check_ok(db)


def test_run_fix_repairs_every_corrupt_related_db(tmp_path):
    # When the same fault corrupts both the game DB and a player trace, one flagged game
    # row drives repair of *all* of them (the game-expansion that this command adds).
    uuid = "12121212-3434-5656-7878-909090909090"
    exp = tmp_path / "runs" / "exp"
    main = exp / f"{uuid}_111.db"
    trace = exp / f"{uuid}-player-5.db"
    _make_corrupt_db(main, n_rows=18)
    _make_corrupt_db(trace, n_rows=18)
    issues = tmp_path / "import_issues.csv"
    _write_issues(issues, [{"db_name": f"{uuid}_111.db", "game_id": uuid}])

    result = run_fix(_cfg(tmp_path, tmp_path / "runs", issues))

    assert sorted(o.status for o in result.outcomes) == ["repaired", "repaired"]
    assert main.with_suffix(".db.bak").exists() and _quick_check_ok(main)
    assert trace.with_suffix(".db.bak").exists() and _quick_check_ok(trace)


# ── CLI wiring ───────────────────────────────────────────────────────────────
def test_cli_fix_short_circuits_before_dag(tmp_path, dev_spec, write_spec, capsys):
    # Point extract at empty tmp paths so `fix` finds nothing to do, then prove it
    # ran the fix path (not the generic --dry-run → print-DAG path).
    dev_spec["data"]["extract"]["runs_dir"] = str(tmp_path / "runs")
    dev_spec["data"]["extract"]["issues_path"] = str(tmp_path / "absent.csv")
    spec_path = write_spec(dev_spec)

    from bench.cli import main

    rc = main(["fix", "--config", str(spec_path), "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "civ-bench fix" in out
    assert "Dry run: config loaded and validated" not in out  # generic path NOT taken
