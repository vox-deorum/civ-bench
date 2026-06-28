"""Problem-game exclusion: the malformed-DB issues report is *consumed*.

The extract stage records malformed games in ``import_issues.csv`` but their stale,
identity-less rows can linger in ``panel_data.csv`` (the DB file still exists, just
corrupt, so it is neither pruned nor re-read). Left in, those rows collapse to
``Player <id>`` fallbacks that form a disconnected clique and make the Plackett-Luce
information matrix singular (``vcov`` Cholesky aborts the whole run). Every analysis
input now drops games named in the report via ``AnalysisContext.load_table``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from bench.analyses.base import AnalysisContext
from bench.extract.issues import read_problem_game_ids


# ── read_problem_game_ids ────────────────────────────────────────────────────
def test_read_problem_game_ids_missing_file_is_empty(tmp_path):
    assert read_problem_game_ids(str(tmp_path / "nope.csv")) == set()
    assert read_problem_game_ids("") == set()


def test_read_problem_game_ids_returns_game_id_set(tmp_path):
    report = tmp_path / "import_issues.csv"
    pd.DataFrame(
        {
            "game_id": ["g1", "g2", " g3 "],  # whitespace is stripped
            "experiment": ["e", "e", "e"],
            "message": ["malformed", "malformed", "malformed"],
        }
    ).to_csv(report, index=False)

    assert read_problem_game_ids(str(report)) == {"g1", "g2", "g3"}


def test_read_problem_game_ids_tolerates_garbage(tmp_path):
    report = tmp_path / "import_issues.csv"
    report.write_bytes(b"\x00\x01 not,a,valid\ncsv at all")
    # A corrupt report must not abort analyses — just yields nothing.
    assert isinstance(read_problem_game_ids(str(report)), set)


# ── AnalysisContext.load_table exclusion ─────────────────────────────────────
def _ctx(tables: dict, issues_path: str | None) -> AnalysisContext:
    """A minimal context: load_table on a *canonical* table only reads config.data."""
    extract = {"issues_path": issues_path} if issues_path is not None else {}
    config = SimpleNamespace(data={"tables": tables, "extract": extract})
    return AnalysisContext(
        config=config, catalog=None, stage_id="t", stage_raw={}, out_dir=Path(".")
    )


def _write_table(path: Path, game_ids: list[str]) -> None:
    pd.DataFrame({"game_id": game_ids, "value": range(len(game_ids))}).to_csv(path, index=False)


def test_load_table_drops_flagged_games(tmp_path):
    panel = tmp_path / "panel.csv"
    _write_table(panel, ["keep1", "bad1", "keep2", "bad2", "keep1"])
    report = tmp_path / "import_issues.csv"
    pd.DataFrame({"game_id": ["bad1", "bad2"]}).to_csv(report, index=False)

    ctx = _ctx({"panel": str(panel)}, str(report))
    out = ctx.load_table("panel")

    assert set(out["game_id"]) == {"keep1", "keep2"}
    assert len(out) == 3  # the duplicate keep1 row is retained; only bad* dropped


def test_load_table_no_issues_file_keeps_everything(tmp_path):
    panel = tmp_path / "panel.csv"
    _write_table(panel, ["g1", "g2", "g3"])

    ctx = _ctx({"panel": str(panel)}, str(tmp_path / "absent.csv"))
    out = ctx.load_table("panel")

    assert set(out["game_id"]) == {"g1", "g2", "g3"}


def test_load_table_without_game_id_column_is_untouched(tmp_path):
    tbl = tmp_path / "tokens.csv"
    pd.DataFrame({"model": ["a", "b"], "cost": [1, 2]}).to_csv(tbl, index=False)
    report = tmp_path / "import_issues.csv"
    pd.DataFrame({"game_id": ["bad1"]}).to_csv(report, index=False)

    ctx = _ctx({"tokens": str(tbl)}, str(report))
    out = ctx.load_table("tokens")

    assert len(out) == 2  # no game_id column → nothing to exclude, left as-is
