"""``civ-bench trim``: remove extra games from repeated controlled slots.

Source files are empty placeholders: the rename path never opens them, and the
prune-only extract that follows only needs ``{uuid}_{ts}.db`` names to exist.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from bench.config.models import OutputConfig, RunConfig
from bench.extract import export_common
from bench.extract.errors import ExtractError
from bench.extract.extract_panel import PANEL_FIELD_MAPPINGS
from bench.extract.issues import ISSUE_FIELDNAMES
from bench.extract.runner import DEFAULT_TABLE_PATHS
from bench.trim import TrimError, run_trim

GAME_HEADER = "game_id,timestamp,experiment,seed,seating_rotation\n"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _game_files(runs: Path, game_id: str, ts: int, exp: str = "exp") -> list[Path]:
    """A game DB, a player trace DB, a replay, and a save for one game."""
    folder = runs / exp
    return [
        _touch(folder / f"{game_id}_{ts}.db"),
        _touch(folder / f"{game_id}-player-1.db"),
        _touch(folder / f"{game_id}_{ts}.Civ5Replay"),
        _touch(folder / f"{game_id}.Civ5Save"),
    ]


def _write_games(path: Path, rows: list[str]) -> None:
    path.write_text(GAME_HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")


def _write_panel(path: Path, game_ids: list[str]) -> None:
    """A panel table outside ``data.extract.outputs``, one blank row per game."""
    fields = list(PANEL_FIELD_MAPPINGS)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({f: "" for f in fields} | {"game_id": g} for g in game_ids)


def _game_ids(path: Path) -> set[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["game_id"] for row in csv.DictReader(fh)}


@pytest.fixture
def env(tmp_path, configs_dir):
    """A runs_dir plus a games table, with catalogs resolving to the tracked configs.

    Every table points into ``tmp_path``: trim prunes each table that exists, and the
    defaults would reach the real tables under ``runs/``.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    tables = {key: str(tmp_path / f"{key}.csv") for key in DEFAULT_TABLE_PATHS}
    games = Path(tables["games"])
    tokens = Path(tables["tokens"])
    panel = Path(tables["panel"])
    issues = tmp_path / "import_issues.csv"

    def make_cfg(enabled: bool = True, cold_runs_dir: Path | None = None) -> RunConfig:
        extract = {
            "enabled": enabled, "runs_dir": str(runs), "outputs": ["games"],
            "issues_path": str(issues),
        }
        if cold_runs_dir is not None:
            extract["cold_runs_dir"] = str(cold_runs_dir)
        return RunConfig(
            name="test",
            seed=1,
            config_path=configs_dir / "benchmark.json",
            raw={},
            output=OutputConfig(),
            data={"extract": extract, "tables": tables},
        )

    return {
        "runs": runs, "games": games, "tokens": tokens, "panel": panel, "issues": issues,
        "cfg": make_cfg,
    }


def _repeated_slot(env) -> dict[str, list[Path]]:
    """Slot (seed 3, rotation 1) holds a, b, c; slot (3, 2) holds only d."""
    files = {
        "a": _game_files(env["runs"], "a", 100),
        "b": _game_files(env["runs"], "b", 200),
        "c": _game_files(env["runs"], "c", 300),
        "d": _game_files(env["runs"], "d", 400),
    }
    _write_games(env["games"], [
        "c,300,exp,3,1",
        "a,100,exp,3,1",
        "b,200,exp,3,1",
        "d,400,exp,3,2",
    ])
    return files


def _baks(files: list[Path]) -> bool:
    return all(not p.exists() and Path(f"{p}.bak").exists() for p in files)


def _untouched(files: list[Path]) -> bool:
    return all(p.exists() and not Path(f"{p}.bak").exists() for p in files)


def test_keeps_earliest_game_and_trims_the_rest(env):
    files = _repeated_slot(env)
    old_copy = _touch(env["runs"] / "exp" / "b_150.db")  # an older timestamped copy

    result = run_trim(env["cfg"]())

    assert [s["keep_candidate_game_id"] for s in result.slots] == ["a"]
    assert sorted(o.game_id for o in result.trimmed) == ["b", "c"]
    assert _untouched(files["a"]) and _untouched(files["d"])
    assert _baks(files["b"] + files["c"] + [old_copy])
    # The prune-only extract dropped the trimmed games' rows without asking.
    assert _game_ids(env["games"]) == {"a", "d"}


def test_excluded_games_are_neither_kept_nor_trimmed(env):
    files = _repeated_slot(env)
    # `a` (the earliest) failed most of its decisions; `b` is a malformed DB.
    env["tokens"].write_text(
        "experiment,game_id,player_id,valid_turn_count,failed_turn_count\n"
        "exp,a,1,10,10\n",
        encoding="utf-8",
    )
    with open(env["issues"], "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ISSUE_FIELDNAMES)
        writer.writeheader()
        writer.writerow({f: "" for f in ISSUE_FIELDNAMES} | {"game_id": "b", "db_name": "b_200.db"})

    result = run_trim(env["cfg"](), prune_confirm=lambda games: False)

    assert result.slots == []  # only c is accepted in the slot
    assert _untouched(files["a"] + files["b"] + files["c"])


def _write_seats(path: Path, seats: dict[str, dict[int, int]]) -> None:
    """A panel table with one row per configured seat: ``game_id -> {config_slot: player_id}``."""
    fields = list(PANEL_FIELD_MAPPINGS)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for game_id, slots in seats.items():
            for slot, player_id in slots.items():
                writer.writerow({f: "" for f in fields} | {
                    "game_id": game_id, "player_id": player_id, "config_slot": slot,
                })


def _mixed_seats(env) -> dict[str, list[Path]]:
    """Config slots 0 and 3 make a 4-seat cycle, played as the identity by ``a``, ``c``
    and ``e``. ``b`` repeats ``a``'s seats under another recorded rotation, and ``d``'s
    seats fit no rotation of that cycle."""
    stamps = [("a", 100), ("c", 110), ("e", 120), ("b", 200), ("d", 300)]
    files = {g: _game_files(env["runs"], g, ts) for g, ts in stamps}
    _write_games(env["games"], [
        "a,100,exp,1,0", "c,110,exp,1,1", "e,120,exp,1,2", "b,200,exp,1,1", "d,300,exp,1,3",
    ])
    _write_seats(env["panel"], {
        "a": {0: 0, 3: 1}, "c": {0: 1, 3: 2}, "e": {0: 2, 3: 3},
        "b": {0: 0, 3: 1}, "d": {0: 3, 3: 1},
    })
    return files


def test_slots_follow_the_seats_each_game_was_played_in(env):
    files = _mixed_seats(env)

    result = run_trim(env["cfg"]())

    assert [(s["keep_candidate_game_id"], s["extra_game_ids"]) for s in result.slots] == [("a", "b")]
    assert [g["game_id"] for g in result.off_cycle] == ["d"]
    assert sorted(o.game_id for o in result.trimmed) == ["b", "d"]
    assert _untouched(files["a"] + files["c"] + files["e"]) and _baks(files["b"] + files["d"])
    assert _game_ids(env["games"]) == {"a", "c", "e"}


def test_malformed_games_do_not_choose_the_cycle(env):
    """Flagged games ``f1``..``f3`` were seated by the identity cycle and ``g`` by
    [1, 0, 3, 2]. Only ``g`` is accepted, so its own cycle holds and nothing is trimmed,
    matching the report, which never loads flagged games."""
    stamps = [("f1", 100), ("f2", 110), ("f3", 120), ("g", 200)]
    files = {gid: _game_files(env["runs"], gid, ts) for gid, ts in stamps}
    _write_games(env["games"], ["f1,100,exp,1,0", "f2,110,exp,1,1", "f3,120,exp,1,2", "g,200,exp,1,0"])
    _write_seats(env["panel"], {
        "f1": {0: 0, 3: 1}, "f2": {0: 1, 3: 2}, "f3": {0: 2, 3: 3}, "g": {0: 1, 3: 0},
    })
    with open(env["issues"], "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ISSUE_FIELDNAMES)
        writer.writeheader()
        for gid, ts in stamps[:3]:
            writer.writerow({f: "" for f in ISSUE_FIELDNAMES} | {"game_id": gid, "db_name": f"{gid}_{ts}.db"})

    result = run_trim(env["cfg"]())

    assert result.slots == [] and result.off_cycle == []
    assert all(_untouched(f) for f in files.values())


def test_uncontrolled_games_are_ignored(env):
    files = _game_files(env["runs"], "a", 100) + _game_files(env["runs"], "b", 200)
    _write_games(env["games"], ["a,100,exp,-1,-1", "b,200,exp,-1,-1"])

    result = run_trim(env["cfg"]())

    assert result.slots == []
    assert _untouched(files)


def test_dry_run_and_declined_prompt_change_nothing(env):
    files = _repeated_slot(env)
    before = env["games"].read_text(encoding="utf-8")

    dry = run_trim(env["cfg"](), dry_run=True)
    asked = []
    declined = run_trim(env["cfg"](), confirm=lambda slots: asked.append(slots) or False)

    assert sorted(o.status for o in dry.outcomes) == ["would-trim", "would-trim"]
    assert declined.declined and not declined.outcomes
    assert len(asked) == 1 and asked[0][0]["extra_game_ids"] == "b,c"
    assert all(_untouched(f) for f in files.values())
    assert env["games"].read_text(encoding="utf-8") == before


def test_existing_backup_skips_the_whole_game(env):
    files = _repeated_slot(env)
    _touch(Path(f"{files['b'][0]}.bak"))  # e.g. left behind by `civ-bench fix`

    result = run_trim(env["cfg"]())

    status = {o.game_id: o.status for o in result.outcomes}
    assert status == {"b": "skipped-bak-exists", "c": "trimmed"}
    assert all(p.exists() for p in files["b"])
    assert _game_ids(env["games"]) == {"a", "b", "d"}


def test_failed_rename_rolls_back_that_game(env, monkeypatch):
    files = _repeated_slot(env)
    real_rename = os.rename

    def flaky(src, dst):
        if str(src).endswith("b.Civ5Save"):
            raise OSError("locked")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", flaky)
    result = run_trim(env["cfg"]())

    status = {o.game_id: o.status for o in result.outcomes}
    assert status == {"b": "failed", "c": "trimmed"}
    assert _untouched(files["b"])
    assert _baks(files["c"])


def test_failed_rollback_reports_the_files_left_renamed(env, monkeypatch):
    files = _repeated_slot(env)
    trace = files["b"][1]  # b-player-1.db sorts first, so it is renamed before the save
    real_rename = os.rename

    def flaky(src, dst):
        if str(src).endswith("b.Civ5Save") or str(src) == f"{trace}.bak":
            raise OSError("locked")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", flaky)
    result = run_trim(env["cfg"]())

    failed = {o.game_id: o for o in result.failed}["b"]
    assert failed.stranded == [f"{trace}.bak"]
    assert f"could not restore {trace}" in failed.detail
    assert Path(f"{trace}.bak").exists() and not trace.exists()


def test_slot_whose_keeper_db_is_missing_is_left_alone(env):
    files = _repeated_slot(env)
    files["a"][0].unlink()  # the keeper's game DB

    result = run_trim(env["cfg"](), prune_confirm=lambda games: False)

    status = {o.game_id: o.status for o in result.outcomes}
    assert status == {"b": "skipped-keeper-missing", "c": "skipped-keeper-missing"}
    assert _untouched(files["b"] + files["c"])
    assert _game_ids(env["games"]) == {"a", "b", "c", "d"}


def test_prune_covers_tables_missing_from_outputs(env):
    _repeated_slot(env)
    _write_panel(env["panel"], ["a", "b", "c", "d"])  # outputs lists only games

    run_trim(env["cfg"]())

    assert _game_ids(env["games"]) == {"a", "d"}
    assert _game_ids(env["panel"]) == {"a", "d"}


def test_table_with_stale_header_stops_trim_before_renaming(env):
    files = _repeated_slot(env)
    env["panel"].write_text("game_id\nb\n", encoding="utf-8")

    with pytest.raises(TrimError, match="out-of-date header"):
        run_trim(env["cfg"]())

    assert all(_untouched(f) for f in files.values())


def test_retry_finishes_a_prune_that_failed_partway(env, monkeypatch):
    files = _repeated_slot(env)
    _write_panel(env["panel"], ["a", "b", "c", "d"])
    real_write = export_common.write_csv_file

    def locked_panel(path, fieldnames, rows):
        return False if path == str(env["panel"]) else real_write(path, fieldnames, rows)

    monkeypatch.setattr(export_common, "write_csv_file", locked_panel)
    with pytest.raises(TrimError, match="run `civ-bench trim` again"):
        run_trim(env["cfg"]())
    assert _baks(files["b"] + files["c"])
    assert _game_ids(env["panel"]) == {"a", "b", "c", "d"}

    monkeypatch.setattr(export_common, "write_csv_file", real_write)
    result = run_trim(env["cfg"]())

    assert sorted(o.status for o in result.outcomes) == ["not-found", "not-found"]
    assert _game_ids(env["games"]) == {"a", "d"}
    assert _game_ids(env["panel"]) == {"a", "d"}


def test_retry_finishes_the_prune_after_the_seats_were_pruned(env, monkeypatch):
    """The prune removed ``b``'s and ``d``'s panel rows, then failed on ``games``. The
    retry has no seats left for them, yet still finishes their removal."""
    files = _mixed_seats(env)
    real_write = export_common.write_csv_file

    def locked_games(path, fieldnames, rows):
        return False if path == str(env["games"]) else real_write(path, fieldnames, rows)

    monkeypatch.setattr(export_common, "write_csv_file", locked_games)
    with pytest.raises(TrimError, match="run `civ-bench trim` again"):
        run_trim(env["cfg"]())
    assert _game_ids(env["panel"]) == {"a", "c", "e"}
    assert _game_ids(env["games"]) == {"a", "b", "c", "d", "e"}

    monkeypatch.setattr(export_common, "write_csv_file", real_write)
    result = run_trim(env["cfg"]())

    assert sorted((o.game_id, o.status) for o in result.outcomes) == [
        ("b", "not-found"), ("d", "not-found"),
    ]
    assert _game_ids(env["games"]) == {"a", "c", "e"}
    assert _untouched(files["a"] + files["c"] + files["e"])


def test_other_missing_games_still_go_through_the_prune_prompt(env):
    _repeated_slot(env)
    with open(env["games"], "a", encoding="utf-8") as fh:
        fh.write("gone,500,exp,-1,-1\n")  # its DB was removed by hand earlier
    asked = []

    run_trim(env["cfg"](), prune_confirm=lambda games: asked.append(games) or False)

    assert [[g["game_id"] for g in batch] for batch in asked] == [["gone"]]
    assert _game_ids(env["games"]) == {"a", "d", "gone"}


def test_disabled_extract_renames_without_pruning(env):
    _repeated_slot(env)

    result = run_trim(env["cfg"](enabled=False))

    assert result.prune is None
    assert len(result.trimmed) == 2
    assert _game_ids(env["games"]) == {"a", "b", "c", "d"}


def test_missing_games_table_is_an_error(env):
    with pytest.raises(TrimError, match="games table not found"):
        run_trim(env["cfg"]())


def test_trim_renames_matching_files_in_both_tiers(env):
    files = _repeated_slot(env)
    cold = env["runs"].parent / "cold"
    cold_b = _touch(cold / "exp" / "b_175.db")
    cold_b_trace = _touch(cold / "exp" / "b-player-2.db")
    cold_c_save = _touch(cold / "exp" / "c.Civ5Save")

    result = run_trim(env["cfg"](enabled=False, cold_runs_dir=cold))

    assert sorted(o.game_id for o in result.trimmed) == ["b", "c"]
    assert _baks(files["b"] + files["c"] + [cold_b, cold_b_trace, cold_c_save])
    assert _untouched(files["a"] + files["d"])


def test_trim_keeper_may_exist_only_in_cold(env):
    files = _repeated_slot(env)
    files["a"][0].unlink()  # the keeper's game DB lives only in the cold tier
    cold = env["runs"].parent / "cold"
    _touch(cold / "exp" / "a_100.db")

    result = run_trim(env["cfg"](enabled=False, cold_runs_dir=cold))

    assert sorted(o.game_id for o in result.trimmed) == ["b", "c"]
    assert _baks(files["b"] + files["c"])
    assert _untouched(files["d"])


def test_trim_scanner_failure_is_a_trim_error(env, monkeypatch):
    files = _repeated_slot(env)
    import bench.trim.runner as trim_runner

    def boom(_extract_cfg, require_hot=True):
        raise ExtractError("hot source is unavailable")

    monkeypatch.setattr(trim_runner, "scan_sources", boom)

    with pytest.raises(TrimError, match="could not scan"):
        run_trim(env["cfg"]())

    assert all(_untouched(f) for f in files.values())


def test_legacy_trim_with_missing_root_keeps_cached_games(env):
    _repeated_slot(env)
    original = env["games"].read_bytes()
    env["runs"].rename(env["runs"].with_name("unavailable"))
    result = run_trim(env["cfg"]())
    assert {o.status for o in result.outcomes} == {"skipped-keeper-missing"}
    assert env["games"].read_bytes() == original


def test_cli_trim_short_circuits_before_dag(tmp_path, dev_spec, write_spec, capsys):
    games = tmp_path / "game_data.csv"
    _write_games(games, [])
    dev_spec["data"]["extract"]["runs_dir"] = str(tmp_path / "runs")
    dev_spec["data"]["extract"]["issues_path"] = str(tmp_path / "absent.csv")
    dev_spec["data"]["tables"]["games"] = str(games)
    dev_spec["data"]["tables"]["tokens"] = str(tmp_path / "absent_tokens.csv")
    spec_path = write_spec(dev_spec)

    from bench.cli import main

    rc = main(["trim", "--config", str(spec_path), "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "nothing to trim" in out
    assert "Dry run: config loaded and validated" not in out
