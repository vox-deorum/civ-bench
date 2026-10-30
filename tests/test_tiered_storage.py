"""Two-tier extraction uses synthetic sources and an independent cached dataset."""

import csv
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from bench.catalog import Catalog
from bench.config.models import OutputConfig, RunConfig
from bench.extract.errors import ExtractError
from bench.extract.issues import ISSUE_FIELDNAMES, ImportIssueLog
from bench.extract.prune import PruneGate
from bench.extract.runner import DEFAULT_TABLE_PATHS, expected_fieldnames, run_extract
from bench.config.behavior import resolve_behavior_spec
from bench.extract.sources import SourceState, csv_game_ids, scan_sources


@pytest.fixture
def storage(tmp_path, configs_dir):
    hot, cold = tmp_path / "hot", tmp_path / "cold"
    hot.mkdir()
    cold.mkdir()
    tables = {key: str(tmp_path / "cache" / f"{key}.csv") for key in DEFAULT_TABLE_PATHS}
    cfg = RunConfig(
        name="tier-test", seed=1, config_path=configs_dir / "benchmark.json", raw={},
        output=OutputConfig(), data={"extract": {
            "runs_dir": str(hot), "cold_runs_dir": str(cold), "outputs": ["games"],
            "issues_path": str(tmp_path / "cache" / "issues.csv"),
        }, "tables": tables},
    )
    catalog = Catalog.from_paths(configs_dir / "models.json", configs_dir / "experiments.json")
    return cfg, catalog, hot, cold


def database(root, gid, timestamp=100, experiment="condition", seed=1):
    path = root / experiment / f"{gid}_{timestamp}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE GameMetadata (Key TEXT, Value TEXT)")
        connection.executemany("INSERT INTO GameMetadata VALUES (?, ?)", [
            ("gameId", gid), ("configuredSyncRandSeed", str(seed)),
            ("configuredMapRandSeed", str(seed)), ("seatingRotation", "0"),
        ])
        connection.commit()
    return path


def write_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def issues_for(cfg, *gids):
    write_rows(cfg.data["extract"]["issues_path"], ISSUE_FIELDNAMES, [
        {"game_id": gid, "experiment": "condition", "stages": "tokens", "message": "bad trace"}
        for gid in gids
    ])


def run(storage, **kwargs):
    cfg, catalog, _, _ = storage
    return run_extract(cfg, catalog=catalog, **kwargs)


@pytest.mark.parametrize("mode", [{}, {"force_rebuild": True}, {"prune_missing": True}])
def test_offline_cold_retains_cached_rows_and_issues_but_hot_can_be_removed(storage, mode):
    cfg, _, hot, cold = storage
    hot_db = database(hot, "active")
    database(cold, "archived")
    run(storage)
    issues_for(cfg, "active", "archived")
    hot_db.unlink()
    cold.rename(cold.with_name("disconnected"))
    gate = PruneGate()
    # Even an explicit caller approval must not defeat source protection.
    gate.approve({"active", "archived"})
    result = run(storage, prune_gate=gate, **mode)
    assert result.pruned_game_ids == {"active"}
    assert result.protected_game_ids == {"archived"}
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"archived"}
    assert csv_game_ids(cfg.data["extract"]["issues_path"]) == {"archived"}
    assert result.tier_availability == {"hot": True, "cold": False}


def test_move_observed_on_skipped_run_establishes_cold_ownership(storage):
    cfg, _, hot, cold = storage
    source = database(hot, "moving")
    run(storage)
    assert run(storage).skipped
    destination = cold / source.parent.name / source.name
    destination.parent.mkdir()
    source.rename(destination)
    run(storage)
    cold.rename(cold.with_name("offline"))
    assert run(storage, prune_missing=True).protected_game_ids == {"moving"}
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"moving"}


def test_declined_hot_removal_keeps_issues_and_never_prompts_for_offline_cold(storage):
    cfg, _, hot, cold = storage
    active = database(hot, "active")
    database(cold, "archived")
    run(storage)
    issues_for(cfg, "active", "archived")
    active.unlink()
    cold.rename(cold.with_name("offline"))
    prompts = []
    result = run(storage, prune_missing=True, prune_gate=PruneGate(lambda rows: prompts.extend(rows) or False))
    assert [row["game_id"] for row in prompts] == ["active"]
    assert result.kept_missing_game_ids == {"active"}
    assert result.protected_game_ids == {"archived"}
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"active", "archived"}
    assert csv_game_ids(cfg.data["extract"]["issues_path"]) == {"active", "archived"}


def test_selected_hot_game_never_uses_cold_player_traces(storage):
    from bench.extract.player_traces import find_player_trace_databases

    cfg, _, hot, cold = storage
    hot_db = database(hot, "abc", 10)
    cold_db = database(cold, "abc", 999)
    trace = cold_db.parent / "abc-player-1.db"
    trace.touch()
    inventory = scan_sources(cfg.data["extract"])
    assert inventory.db_files == [str(hot_db)]
    assert find_player_trace_databases(inventory.db_files[0]) == []
    local_trace = hot_db.parent / trace.name
    local_trace.touch()
    assert find_player_trace_databases(inventory.db_files[0]) == [(1, str(local_trace))]


def test_prune_only_does_not_open_databases(storage, monkeypatch):
    cfg, _, hot, cold = storage
    database(hot, "active")
    database(cold, "archived")

    def no_sql(*args, **kwargs):
        pytest.fail("prune-only opened a source database")

    monkeypatch.setattr(sqlite3, "connect", no_sql)
    run(storage, prune_missing=True)
    assert not csv_game_ids(cfg.data["tables"]["games"])


@pytest.mark.parametrize("dual_owned", [True, False])
def test_trim_uses_real_ownership_to_defer_offline_copies_or_trim_hot_only(storage, dual_owned):
    from bench.trim.runner import run_trim

    cfg, _, hot, cold = storage
    keeper = database(hot, "aaa", 100)
    extra = database(hot, "bbb", 200)
    if dual_owned:
        database(cold, "bbb", 300)
    run(storage)
    cold.rename(cold.with_name("offline"))
    index = Path(cfg.data["tables"]["games"] + ".sources.json")
    before = index.read_bytes()
    if dual_owned:
        result = run_trim(cfg)
        assert [outcome.status for outcome in result.outcomes] == ["skipped-source-unavailable"]
        assert index.read_bytes() == before
        assert extra.exists()
    else:
        result = run_trim(cfg, dry_run=True)
        assert [outcome.status for outcome in result.outcomes] == ["would-trim"]
        assert index.read_bytes() == before
        result = run_trim(cfg)
        assert [outcome.game_id for outcome in result.trimmed] == ["bbb"]
        assert not extra.exists()
        assert csv_game_ids(cfg.data["tables"]["games"]) == {"aaa"}
    assert keeper.exists()


@pytest.mark.parametrize("dual_owned", [True, False])
def test_fix_only_reports_offline_copies_for_cold_or_unknown_ownership(storage, dual_owned):
    from bench.fix.runner import run_fix

    cfg, _, hot, cold = storage
    database(hot, "abc")
    if dual_owned:
        database(cold, "abc")
    run(storage)
    issues_for(cfg, "abc")
    cold.rename(cold.with_name("offline"))
    index = Path(cfg.data["tables"]["games"] + ".sources.json")
    before = index.read_bytes()
    result = run_fix(cfg)
    expected = ["healthy", "unresolved"] if dual_owned else ["healthy"]
    assert [outcome.status for outcome in result.outcomes] == expected
    assert index.read_bytes() == before


@pytest.mark.parametrize("command", ["fix", "trim"])
def test_maintenance_requires_hot_storage_even_when_cold_is_available(storage, command):
    from bench.fix import FixError, run_fix
    from bench.trim import TrimError, run_trim

    cfg, _, hot, cold = storage
    database(cold, "aaa", 100)
    database(cold, "bbb", 200)
    run(storage)
    hot.rename(hot.with_name("offline"))
    if command == "fix":
        issues_for(cfg, "aaa")
        with pytest.raises(FixError, match="runs_dir"):
            run_fix(cfg)
    else:
        with pytest.raises(TrimError, match="cannot scan runs_dir"):
            run_trim(cfg)


@pytest.mark.parametrize("command", ["fix", "trim"])
def test_maintenance_rejects_corrupt_ownership_before_source_changes(storage, command):
    from bench.fix import FixError, run_fix
    from bench.trim import TrimError, run_trim

    cfg, _, hot, cold = storage
    paths = [database(hot, "aaa", 100), database(hot, "bbb", 200)]
    run(storage)
    index = Path(cfg.data["tables"]["games"] + ".sources.json")
    index.write_text("{broken", encoding="utf-8")
    original = {path: path.read_bytes() for path in paths}
    if command == "fix":
        issues_for(cfg, "aaa")
        cold.rename(cold.with_name("offline"))
        with pytest.raises(FixError, match="ownership index"):
            run_fix(cfg)
    else:
        with pytest.raises(TrimError, match="ownership index"):
            run_trim(cfg)
    assert original == {path: path.read_bytes() for path in paths}
    assert not list(hot.rglob("*.bak"))


def test_partial_trim_defers_offline_pending_removal_until_cold_returns(storage, monkeypatch):
    from bench.extract import tiered
    from bench.trim import TrimError, run_trim

    cfg, _, hot, cold = storage
    database(hot, "aaa", 100)
    database(hot, "bbb", 200)
    database(cold, "bbb", 300)
    run(storage)
    panel = cfg.data["tables"]["panel"]
    write_rows(panel, expected_fieldnames("panel", {}), [{"game_id": gid} for gid in ("aaa", "bbb")])
    prune_csv = tiered.prune_csv

    def fail_games(path, removed):
        if path == cfg.data["tables"]["games"]:
            raise OSError("simulated locked games table")
        prune_csv(path, removed)

    monkeypatch.setattr(tiered, "prune_csv", fail_games)
    with pytest.raises(TrimError, match="pruning the trimmed games failed"):
        run_trim(cfg)
    monkeypatch.setattr(tiered, "prune_csv", prune_csv)
    offline = cold.with_name("offline")
    cold.rename(offline)
    before = Path(cfg.data["tables"]["games"]).read_bytes()
    assert [outcome.status for outcome in run_trim(cfg).outcomes] == ["skipped-source-unavailable"]
    assert Path(cfg.data["tables"]["games"]).read_bytes() == before
    assert csv_game_ids(panel) == {"aaa"}
    offline.rename(cold)
    run_trim(cfg)
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"aaa"}


def test_offline_unknown_copies_make_fix_preview_unresolved_without_writes(storage, dev_spec, write_spec):
    from bench.cli import main

    cfg, _, hot, cold = storage
    database(hot, "aaa")
    issues_for(cfg, "aaa")
    cold.rmdir()
    dev_spec["data"] = cfg.data
    assert main(["fix", "--config", str(write_spec(dev_spec)), "--dry-run"]) == 2
    assert not list(hot.rglob("*.bak"))
    assert not Path(cfg.data["tables"]["games"] + ".sources.json").exists()


def test_unknown_cache_is_protected_offline_then_empty_cold_allows_confirmed_drop(storage):
    cfg, _, _, cold = storage
    fields = expected_fieldnames("games", {})
    write_rows(cfg.data["tables"]["games"], fields, [{"game_id": "legacy", "experiment": "old"}])
    offline = cold.with_name("offline")
    cold.rename(offline)
    assert run(storage, prune_missing=True).protected_game_ids == {"legacy"}
    offline.rename(cold)
    seen = []
    declined = run(storage, prune_missing=True, prune_gate=PruneGate(lambda rows: seen.extend(rows) or False))
    assert declined.kept_missing_game_ids == {"legacy"}
    assert seen[0]["experiment"] == "old"
    assert run(storage, prune_missing=True).pruned_game_ids == {"legacy"}


def test_unknown_cached_hot_game_stays_unknown_until_cold_can_be_observed(storage):
    cfg, _, hot, cold = storage
    source = database(hot, "legacy")
    write_rows(cfg.data["tables"]["games"], expected_fieldnames("games", {}), [
        {"game_id": "legacy", "experiment": "condition", "seed": "1", "seating_rotation": "0"},
    ])
    cold.rename(cold.with_name("offline"))
    run(storage)
    source.unlink()
    result = run(storage, prune_missing=True)
    assert result.protected_game_ids == {"legacy"}
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"legacy"}
    cold.with_name("offline").rename(cold)
    assert run(storage, prune_missing=True).pruned_game_ids == {"legacy"}


def test_hot_wins_newer_cold_and_force_keeps_existing_per_game_rows(storage):
    cfg, _, hot, cold = storage
    database(hot, "duplicate", 10, seed=3)
    database(cold, "duplicate", 999, seed=8)
    run(storage)
    path = cfg.data["tables"]["games"]
    with open(path, newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert (row["timestamp"], row["seed"]) == ("10", "3")
    original = Path(path).read_bytes()
    database(hot, "duplicate", 20, seed=5)
    result = run(storage, force_rebuild=True)
    assert result.new_rows["games"] == 0
    assert Path(path).read_bytes() == original


def test_returning_old_cold_sources_reconcile_each_output_independently(storage):
    cfg, _, hot, cold = storage
    database(hot, "active")
    archived = database(cold, "archived")
    os.utime(archived, (1, 1))
    offline = cold.with_name("offline")
    cold.rename(offline)
    run(storage)
    # Reconcile a different output first after reconnecting. Its checkpoint must
    # not mark the games output complete for this new inventory.
    offline.rename(cold)
    cfg.data["extract"]["outputs"] = ["world"]
    run(storage)
    cfg.data["extract"]["outputs"] = ["games"]
    result = run(storage)
    assert not result.skipped
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"active", "archived"}


def test_max_dbs_does_not_limit_ownership_or_mark_outputs_complete(storage):
    cfg, _, hot, cold = storage
    database(hot, "active")
    database(cold, "archived")
    cfg.data["extract"]["max_dbs"] = 1
    run(storage)
    cold.rename(cold.with_name("offline"))
    state = SourceState.load(cfg.data["tables"]["games"], scan_sources(cfg.data["extract"]))
    assert state.games["archived"]["tiers"] == ["cold"]
    assert state.checkpoints == {}
    cold.with_name("offline").rename(cold)
    cfg.data["extract"].pop("max_dbs")
    assert not run(storage).skipped
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"active", "archived"}


def test_capped_recreation_invalidates_an_older_complete_checkpoint(storage):
    cfg, _, hot, cold = storage
    database(hot, "active")
    database(cold, "archived")
    run(storage)
    Path(cfg.data["tables"]["games"]).unlink()
    cfg.data["extract"]["max_dbs"] = 1
    run(storage)
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"active"}
    cfg.data["extract"].pop("max_dbs")
    assert not run(storage).skipped
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"active", "archived"}


@pytest.mark.parametrize("restored_output", ["games", "panel"])
def test_prune_then_restore_old_source_reconciles_even_omitted_outputs(storage, restored_output):
    cfg, _, _, cold = storage
    source = database(cold, "archived")
    panel = cfg.data["tables"]["panel"]
    write_rows(panel, expected_fieldnames("panel", {}), [{"game_id": "archived", "player_id": "0"}])
    cfg.data["extract"]["outputs"] = ["games", "panel"]
    run(storage)
    backup = cold.parent / "archived.db.backup"
    source.rename(backup)
    cfg.data["extract"]["outputs"] = ["games"]
    run(storage, prune_missing=True)
    assert not csv_game_ids(panel)
    backup.rename(source)
    cfg.data["extract"]["outputs"] = [restored_output]
    result = run(storage)
    assert not result.skipped
    if restored_output == "games":
        assert csv_game_ids(cfg.data["tables"]["games"]) == {"archived"}


def test_approved_deletion_covers_omitted_tables_and_issues_only_games(storage):
    cfg, _, hot, cold = storage
    db = database(cold, "gone", experiment="removed-condition")
    run(storage)
    spec = resolve_behavior_spec(None)
    for key, path in cfg.data["tables"].items():
        write_rows(path, expected_fieldnames(key, spec), [{"game_id": "gone"}])
    issues_for(cfg, "gone", "issues-only")
    db.unlink()
    result = run(storage, prune_missing=True)
    assert result.pruned_game_ids == {"gone", "issues-only"}
    assert all(not csv_game_ids(path) for path in cfg.data["tables"].values())
    assert not csv_game_ids(cfg.data["extract"]["issues_path"])


def test_inventory_changes_reconcile_issues_even_when_no_tables_are_selected(storage):
    cfg, _, _, cold = storage
    source = database(cold, "gone")
    run(storage)
    issues_for(cfg, "gone")
    source.unlink()
    cfg.data["extract"]["outputs"] = []
    result = run(storage)
    assert not result.skipped
    assert result.pruned_game_ids == {"gone"}
    assert not csv_game_ids(cfg.data["extract"]["issues_path"])
    assert not csv_game_ids(cfg.data["tables"]["games"])


def test_unsafe_schema_rewrite_fails_before_any_cached_file_or_index_changes(storage):
    cfg, _, hot, cold = storage
    database(cold, "archived")
    run(storage)
    cold.rename(cold.with_name("offline"))
    cfg.data["extract"]["outputs"] = ["world", "games"]
    Path(cfg.data["tables"]["games"]).write_text("game_id,old_column\narchived,valuable\n", encoding="utf-8")
    files = list(Path(cfg.data["tables"]["games"]).parent.iterdir())
    original = {path: path.read_bytes() for path in files}
    with pytest.raises(ExtractError, match="unavailable or unselected"):
        run(storage, force_rebuild=True)
    assert {path: path.read_bytes() for path in files} == original
    assert not Path(cfg.data["tables"]["world"]).exists()


def test_partial_prune_retries_durable_approval_without_prompt(storage, monkeypatch):
    from bench.extract import tiered

    cfg, _, _, cold = storage
    db = database(cold, "gone")
    run(storage)
    write_rows(cfg.data["tables"]["world"], expected_fieldnames("world", {}), [{"game_id": "gone"}])
    issues_for(cfg, "gone")
    db.unlink()
    original = tiered.prune_csv

    def fail_games(path, removed):
        if path == cfg.data["tables"]["games"]:
            raise OSError("simulated interruption")
        original(path, removed)

    monkeypatch.setattr(tiered, "prune_csv", fail_games)
    with pytest.raises(OSError, match="simulated interruption"):
        run(storage, prune_missing=True)
    assert not csv_game_ids(cfg.data["tables"]["world"])
    monkeypatch.setattr(tiered, "prune_csv", original)
    prompts = []
    result = run(storage, prune_missing=True, prune_gate=PruneGate(lambda rows: prompts.append(rows) or False))
    assert result.pruned_game_ids == {"gone"}
    assert prompts == []
    assert not csv_game_ids(cfg.data["tables"]["games"])
    assert not csv_game_ids(cfg.data["extract"]["issues_path"])


def test_deferred_pending_removal_can_skip_offline_and_resume_after_reconnect(storage, monkeypatch):
    cfg, _, _, cold = storage
    source = database(cold, "archived")
    run(storage)
    source.unlink()
    original = ImportIssueLog.write_csv
    monkeypatch.setattr(ImportIssueLog, "write_csv", lambda *args: False)
    with pytest.raises(ExtractError, match="failed to write"):
        run(storage, prune_missing=True)
    monkeypatch.setattr(ImportIssueLog, "write_csv", original)
    offline = cold.with_name("offline")
    cold.rename(offline)
    assert run(storage).protected_game_ids == {"archived"}
    assert run(storage).skipped
    offline.rename(cold)
    prompts = []
    result = run(storage, prune_missing=True, prune_gate=PruneGate(lambda rows: prompts.extend(rows) or False))
    assert result.pruned_game_ids == {"archived"}
    assert prompts == []


@pytest.mark.parametrize("reappear", [True, False])
def test_pending_approval_rechecked_for_reappearance_or_offline_protection(storage, reappear):
    cfg, _, hot, cold = storage
    db = database(cold, "archived")
    run(storage)
    inventory = scan_sources(cfg.data["extract"])
    state = SourceState.load(cfg.data["tables"]["games"], inventory)
    state.pending_removals.add("archived")
    state.save()
    if not reappear:
        cold.rename(cold.with_name("offline"))
    result = run(storage, prune_missing=True)
    assert not result.pruned_game_ids
    assert csv_game_ids(cfg.data["tables"]["games"]) == {"archived"}
    if reappear:
        state = SourceState.load(cfg.data["tables"]["games"], inventory)
        assert not state.pending_removals


def test_changed_roots_invalidate_ownership_and_corrupt_index_fails(storage):
    cfg, _, hot, cold = storage
    database(hot, "active")
    run(storage)
    replacement = cold.with_name("replacement")
    cfg.data["extract"]["cold_runs_dir"] = str(replacement)
    for path in hot.rglob("*.db"):
        path.unlink()
    assert run(storage, prune_missing=True).protected_game_ids == {"active"}
    index = Path(cfg.data["tables"]["games"] + ".sources.json")
    index.write_text("{broken", encoding="utf-8")
    before = Path(cfg.data["tables"]["games"]).read_bytes()
    with pytest.raises(ExtractError, match="ownership index"):
        run(storage, prune_missing=True)
    assert Path(cfg.data["tables"]["games"]).read_bytes() == before


def test_partial_cold_walk_is_discarded_and_hot_scan_failure_aborts(storage, monkeypatch):
    from bench.extract import sources

    cfg, _, hot, cold = storage
    database(cold, "archived")
    run(storage)
    walk = os.walk

    def interrupted(root, onerror=None):
        yield from walk(root)
        if str(root) == str(cold.resolve()):
            onerror(PermissionError("subdirectory denied"))

    monkeypatch.setattr(sources.os, "walk", interrupted)
    result = run(storage, prune_missing=True)
    assert result.protected_game_ids == {"archived"}
    assert scan_sources(cfg.data["extract"]).tiers["cold"].files == []
    hot.rmdir()
    before = Path(cfg.data["tables"]["games"]).read_bytes()
    with pytest.raises(ExtractError, match="cannot scan runs_dir"):
        run(storage, force_rebuild=True)
    assert Path(cfg.data["tables"]["games"]).read_bytes() == before
