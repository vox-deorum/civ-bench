"""Auto-fix orchestration: ``extract → fix → re-import``, on by default.

``civ-bench run``/``extract`` repair the malformed DBs recorded in
``import_issues.csv`` and re-import so the ledger reflects the fixed state, unless
disabled via ``data.extract.auto_fix`` / ``--no-fix``. These tests target the
orchestrator [`cli._extract_with_autofix`] — the order of the three steps and every
short-circuit guard — plus the config/flag toggle wiring, with fakes for the
(separately-tested) ``run_extract`` and ``run_fix`` so no SQLite corruption is needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bench import cli
from bench.config.models import OutputConfig, RunConfig
from bench.extract.issues import ISSUE_FIELDNAMES, ImportIssueLog
from bench.extract.runner import ExtractResult
from bench.fix import FixError, FixOutcome, FixResult


# ── fakes / helpers ──────────────────────────────────────────────────────────
def _cfg(tmp_path: Path, **extract) -> RunConfig:
    # Default the issue report into tmp so the suite never touches the repo's runs/.
    issues_path = str(tmp_path / "import_issues.csv")
    return RunConfig(
        name="t",
        seed=1,
        config_path=tmp_path / "benchmark.json",
        raw={},
        output=OutputConfig(),
        data={
            "extract": {
                "runs_dir": "runs/",
                "issues_path": issues_path,
                **extract,
            },
            "tables": {},
        },
    )


def _issues_log() -> ImportIssueLog:
    """A truthy issue log (one recorded malformed game)."""
    log = ImportIssueLog()
    log.record(
        stage="games",
        db_path="runs/exp/uuid_1700000000000.db",
        message="database disk image is malformed",
    )
    return log


def _carried_forward_log(tmp_path: Path) -> ImportIssueLog:
    """A truthy log with NO fresh issues — one game carried over from a prior run.

    Built the way the extract runner builds it: load the prior ledger and reconcile
    (the game's DB still exists), but record nothing this run, so ``_fresh`` is empty.
    """
    report = tmp_path / "prior_issues.csv"
    with open(report, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ISSUE_FIELDNAMES)
        w.writeheader()
        w.writerow({
            "game_id": "old_game", "experiment": "e", "seed": "-1",
            "seating_rotation": "-1", "stages": "games", "players": "",
            "db_name": "old_game_123.db", "message": "malformed",
        })
    log = ImportIssueLog()
    log.load(str(report))
    log.reconcile({"old_game"})  # DB still present → carried forward, not dropped
    return log


def _extract_result(*, skipped: bool = False, issues: ImportIssueLog | None = None) -> ExtractResult:
    return ExtractResult(
        skipped=skipped,
        reason="fresh outputs",
        issues=issues or ImportIssueLog(),
        issues_path="runs/import_issues.csv",
    )


def _fix_result(*, repaired: bool = True) -> FixResult:
    status = "repaired" if repaired else "healthy"
    return FixResult(outcomes=[FixOutcome("uuid_1700000000000.db", "uuid", status)])


def _install(monkeypatch, extract_results, *, fix_result=None, fix_error=None) -> dict:
    """Patch ``cli.run_extract`` and ``bench.fix.run_fix`` with recording fakes.

    ``extract_results`` is returned one-per-call (the last entry repeats). Returns a
    state dict: ``extract_calls`` (the ``force_rebuild`` of each call) and ``fix_calls``.
    """
    state = {"extract_calls": [], "extract_prune": [], "fix_calls": 0, "fix_only_game_ids": []}
    seq = list(extract_results)

    def fake_extract(cfg, catalog=None, force_rebuild=False, prune_missing=None):
        state["extract_calls"].append(force_rebuild)
        state["extract_prune"].append(prune_missing)
        return seq[min(len(state["extract_calls"]) - 1, len(seq) - 1)]

    def fake_fix(cfg, dry_run=False, force=False, only_game_ids=None):
        state["fix_calls"] += 1
        state["fix_only_game_ids"].append(only_game_ids)
        if fix_error is not None:
            raise fix_error
        return fix_result

    monkeypatch.setattr(cli, "run_extract", fake_extract)
    # run_fix is imported inside the helper via ``from .fix import run_fix`` — patch the
    # package attribute so the runtime lookup resolves to the fake.
    monkeypatch.setattr("bench.fix.run_fix", fake_fix)
    return state


# ── _extract_with_autofix: order + guards ────────────────────────────────────
def test_repairs_then_reimports(monkeypatch, tmp_path):
    st = _install(
        monkeypatch,
        [_extract_result(issues=_issues_log()), _extract_result()],
        fix_result=_fix_result(repaired=True),
    )
    cli._extract_with_autofix(_cfg(tmp_path), object(), force_rebuild=False, auto_fix=True)

    assert st["fix_calls"] == 1
    assert st["extract_calls"] == [False, True]  # initial extract, then a forced re-import
    # the re-import re-inspects (prune_missing=False), never prune-only
    assert st["extract_prune"] == [None, False]
    # fix is scoped to exactly the games that failed THIS run
    assert st["fix_only_game_ids"] == [_issues_log().fresh_game_ids]


def test_no_issues_no_fix(monkeypatch, tmp_path):
    st = _install(monkeypatch, [_extract_result()], fix_result=_fix_result())
    cli._extract_with_autofix(_cfg(tmp_path), object(), force_rebuild=False, auto_fix=True)

    assert st["fix_calls"] == 0
    assert st["extract_calls"] == [False]


def test_auto_fix_disabled_skips_fix(monkeypatch, tmp_path):
    st = _install(monkeypatch, [_extract_result(issues=_issues_log())], fix_result=_fix_result())
    cli._extract_with_autofix(_cfg(tmp_path), object(), force_rebuild=False, auto_fix=False)

    assert st["fix_calls"] == 0
    assert st["extract_calls"] == [False]


def test_skipped_extract_skips_fix(monkeypatch, tmp_path):
    # Fresh outputs → extract skipped → nothing new to fix (even with a stale ledger).
    st = _install(monkeypatch, [_extract_result(skipped=True)], fix_result=_fix_result())
    cli._extract_with_autofix(_cfg(tmp_path), object(), force_rebuild=False, auto_fix=True)

    assert st["fix_calls"] == 0
    assert st["extract_calls"] == [False]


def test_nothing_repaired_skips_reimport(monkeypatch, tmp_path):
    st = _install(
        monkeypatch,
        [_extract_result(issues=_issues_log())],
        fix_result=_fix_result(repaired=False),
    )
    cli._extract_with_autofix(_cfg(tmp_path), object(), force_rebuild=False, auto_fix=True)

    assert st["fix_calls"] == 1          # fix ran…
    assert st["extract_calls"] == [False]  # …but recovered nothing → no re-import


def test_carried_forward_only_ledger_does_not_trigger_fix(monkeypatch, tmp_path):
    # A ledger that still LISTS a prior game (truthy) but recorded nothing fresh this
    # run must not re-attempt the already-unrecoverable game every invocation.
    carried = _carried_forward_log(tmp_path)
    assert carried                       # truthy: the game is still in the ledger
    assert not carried.has_fresh_issues  # …but nothing failed *this* run
    st = _install(monkeypatch, [_extract_result(issues=carried)], fix_result=_fix_result())
    cli._extract_with_autofix(_cfg(tmp_path), object(), force_rebuild=False, auto_fix=True)

    assert st["fix_calls"] == 0
    assert st["extract_calls"] == [False]


def test_fix_error_is_swallowed(monkeypatch, tmp_path):
    st = _install(
        monkeypatch,
        [_extract_result(issues=_issues_log())],
        fix_error=FixError("runs_dir vanished"),
    )
    result = cli._extract_with_autofix(_cfg(tmp_path), object(), force_rebuild=False, auto_fix=True)

    assert st["fix_calls"] == 1
    assert st["extract_calls"] == [False]  # a fix failure never triggers a re-import…
    assert result.issues                    # …and returns the original (unrepaired) result


# ── CLI wiring: config default + --no-fix override reach the helper ──────────
def test_cli_extract_autofix_on_by_default(monkeypatch, dev_spec, write_spec):
    st = _install(
        monkeypatch,
        [_extract_result(issues=_issues_log())],
        fix_result=_fix_result(repaired=False),
    )
    rc = cli.main(["extract", "--config", str(write_spec(dev_spec))])

    assert rc == 0
    assert st["fix_calls"] == 1


def test_cli_extract_no_fix_flag_disables(monkeypatch, dev_spec, write_spec):
    st = _install(
        monkeypatch,
        [_extract_result(issues=_issues_log())],
        fix_result=_fix_result(repaired=False),
    )
    rc = cli.main(["extract", "--config", str(write_spec(dev_spec)), "--no-fix"])

    assert rc == 0
    assert st["fix_calls"] == 0


def test_cli_extract_auto_fix_config_false_disables(monkeypatch, dev_spec, write_spec):
    dev_spec["data"]["extract"]["auto_fix"] = False
    st = _install(
        monkeypatch,
        [_extract_result(issues=_issues_log())],
        fix_result=_fix_result(repaired=False),
    )
    rc = cli.main(["extract", "--config", str(write_spec(dev_spec))])

    assert rc == 0
    assert st["fix_calls"] == 0


def test_cli_run_autofixes_then_reimports(monkeypatch, dev_spec, write_spec):
    # `run --only extract` exercises the same helper from inside the pipeline.
    monkeypatch.setattr(cli.Catalog, "from_run_config", staticmethod(lambda cfg: object()))
    st = _install(
        monkeypatch,
        [_extract_result(issues=_issues_log()), _extract_result()],
        fix_result=_fix_result(repaired=True),
    )
    rc = cli.main(["run", "--config", str(write_spec(dev_spec)), "--only", "extract"])

    assert rc == 0
    assert st["fix_calls"] == 1
    assert st["extract_calls"] == [False, True]
