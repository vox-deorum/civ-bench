"""``civ-bench fix`` orchestrator: repair the malformed DBs in ``import_issues.csv``.

Reads the malformed-DB ledger written by the extract stage. A flagged game is rarely
a single corrupt file: the same interrupted write that broke the main game DB can
also break the player-level telemetry DBs saved alongside it. So for each flagged
game ``fix`` examines **every related ``.db``**: the game DB (``{uuid}_{ts}.db``) and
its trace exports (``{uuid}-player-*.db``), matched by the game's uuid prefix (the
non-SQLite ``.Civ5Save``/``.Civ5Replay`` siblings are excluded by extension).

Each related DB is checked with ``PRAGMA quick_check``:

* **already healthy** → left exactly as found, with no backup (nothing changed, so
  there is no ``.bak`` to keep);
* **corrupt** → recovered into a fresh file, then swapped in: the original is kept as
  ``<name>.db.bak`` and the recovered DB written as ``<name>.db``;
* **unrecoverable** → left exactly as found and reported.

A failure on one file never aborts the rest of the batch. Scope is deliberately
narrow: ``fix`` only repairs files; it does **not** rewrite ``import_issues.csv``
(only ``extract`` owns that ledger). After fixing, re-run
``civ-bench extract --force-rebuild`` to re-read the repaired DBs and clear the report.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..config import RunConfig
from ..extract.issues import resolve_issues_path
from .errors import FixError
from .repair import SIDECAR_SUFFIXES, RepairReport, _remove, repair_database, source_quick_check

Status = Literal[
    "repaired", "healthy", "failed", "not-found", "ambiguous",
    "skipped-bak-exists", "dry-run",
]
# Statuses that mean a recorded game's DB was not resolved (drives the CLI exit code).
_UNRESOLVED = {"failed", "not-found", "ambiguous"}


@dataclass
class FixOutcome:
    """The result of examining one related DB file."""

    db_name: str
    game_id: str
    status: Status
    path: str = ""
    report: RepairReport | None = None
    detail: str = ""


@dataclass
class FixResult:
    runs_dir: str = ""
    issues_path: str = ""
    outcomes: list[FixOutcome] = field(default_factory=list)

    @property
    def repaired(self) -> list[FixOutcome]:
        return [o for o in self.outcomes if o.status == "repaired"]

    @property
    def failed(self) -> list[FixOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    @property
    def unresolved(self) -> list[FixOutcome]:
        """Recorded DBs left unresolved (failed / not-found / ambiguous)."""
        return [o for o in self.outcomes if o.status in _UNRESOLVED]


def _read_issue_rows(path: str) -> list[dict]:
    """Issue rows (``db_name``/``game_id``) from ``import_issues.csv``.

    A missing/empty/unreadable report yields ``[]`` (with a warning) rather than an
    error; ``fix`` simply has nothing to do. Rows are deduped by ``game_id`` (the
    ledger's own key): all of a game's related DBs are gathered from one row.
    """
    if not path or not Path(path).exists():
        return []
    rows: list[dict] = []
    seen: set[str] = set()
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("db_name") or "").strip()
                gid = (row.get("game_id") or "").strip()
                key = gid or name
                if not key or key in seen:
                    continue
                seen.add(key)
                rows.append({"db_name": name, "game_id": gid})
    except Exception as exc:  # a corrupt report must not abort the tool
        print(f"WARNING: could not read import-issues report {path}: {exc}")
        return []
    return rows


def _index_db_basenames(runs_dir: str) -> dict[str, list[str]]:
    """Map every ``*.db`` basename under ``runs_dir`` to its full path(s).

    Keeps *every* ``.db`` (no latest-per-game collapse, no ``-player-`` filtering): we
    need the game DB and its trace DBs alike, and the exact files to repair.
    """
    index: dict[str, list[str]] = {}
    for root, _dirs, files in os.walk(runs_dir):
        for file in files:
            if file.endswith(".db"):
                index.setdefault(file, []).append(os.path.join(root, file))
    return index


def _related_basenames(index: dict[str, list[str]], game_id: str, db_name: str) -> list[str]:
    """Basenames of every related DB: those sharing the game's uuid prefix.

    Falls back to the exact ``db_name`` when the row carries no usable game id.
    """
    if game_id:
        return sorted(b for b in index if b.startswith(game_id))
    return [db_name] if db_name in index else []


def _clear_sidecars(db_path: str) -> None:
    """Delete stale ``-wal``/``-shm``/``-journal`` next to ``db_path``."""
    for suffix in SIDECAR_SUFFIXES:
        sidecar = db_path + suffix
        try:
            if os.path.exists(sidecar):
                os.remove(sidecar)
        except OSError:
            pass


def _swap_in(tmp_path: str, full_path: str, backup_path: str) -> None:
    """Replace ``full_path`` with the repaired ``tmp_path``, keeping the original as backup.

    Both renames are atomic individually; ordering them original→backup then tmp→canonical
    means the original is safe in ``backup_path`` throughout. If the second rename fails the
    first is rolled back, so the canonical path is never left empty.
    """
    os.replace(full_path, backup_path)
    try:
        os.replace(tmp_path, full_path)
    except OSError:
        os.replace(backup_path, full_path)  # restore the original
        raise
    _clear_sidecars(full_path)


def _process_one(
    full_path: str, db_name: str, game_id: str, *, dry_run: bool, force: bool
) -> FixOutcome:
    """Examine one related DB; repair + swap it only if it is actually corrupt. Never raises."""
    if dry_run:
        # Preview only: skip the (potentially large) quick_check scans; the real run
        # classifies each file as healthy or repairable.
        return FixOutcome(db_name, game_id, "dry-run", path=full_path)

    health = source_quick_check(full_path)
    if health == "ok":
        # Nothing changed → leave it untouched and keep no backup.
        return FixOutcome(db_name, game_id, "healthy", path=full_path)

    backup_path = full_path + ".bak"
    if os.path.exists(backup_path) and not force:
        return FixOutcome(
            db_name, game_id, "skipped-bak-exists", path=full_path,
            detail=f"{os.path.basename(backup_path)} already exists (use --force to overwrite)",
        )

    tmp_path = os.path.join(
        os.path.dirname(full_path), f".{db_name}.fix-{os.getpid()}.tmp"
    )
    try:
        _remove(tmp_path)
        report = repair_database(full_path, tmp_path)
        if not report.success:
            _remove(tmp_path)
            return FixOutcome(
                db_name, game_id, "failed", path=full_path, report=report,
                detail=report.error or "unrecoverable",
            )
        _swap_in(tmp_path, full_path, backup_path)
        return FixOutcome(db_name, game_id, "repaired", path=full_path, report=report)
    except Exception as exc:  # a swap/IO failure on one DB must not abort the batch
        _remove(tmp_path)
        if not os.path.exists(full_path) and os.path.exists(backup_path):
            detail = (f"swap failed ({exc}); original preserved at "
                      f"{os.path.basename(backup_path)}")
        else:
            detail = f"swap failed: {exc}"
        return FixOutcome(db_name, game_id, "failed", path=full_path, detail=detail)


def run_fix(
    cfg: RunConfig,
    dry_run: bool = False,
    force: bool = False,
    only_game_ids: set[str] | None = None,
) -> FixResult:
    """Repair the malformed DBs (and their related trace DBs) named in ``import_issues.csv``.

    Raises :class:`FixError` for an operational failure (a ``runs_dir`` that does not
    exist while there are DBs to repair), distinct from a per-file ``failed`` outcome.

    ``only_game_ids`` (``None`` = repair everything recorded, the standalone
    ``civ-bench fix`` behaviour) narrows the batch to the ledger rows whose dedupe key
    (``game_id`` or, absent that, ``db_name``) is in the set, used by auto-fix to
    repair only the games that failed *this* run, not the whole carried-forward ledger.
    """
    extract_cfg = cfg.data.get("extract", {}) or {}
    runs_dir = extract_cfg.get("runs_dir", "runs/")
    issues_path = resolve_issues_path(cfg.data)

    print("=" * 60)
    print("civ-bench fix" + (" (dry-run)" if dry_run else ""))
    print("=" * 60)
    print(f"runs_dir:    {runs_dir}")
    print(f"issues:      {issues_path}")

    result = FixResult(runs_dir=runs_dir, issues_path=issues_path)
    rows = _read_issue_rows(issues_path)
    if only_game_ids is not None:
        rows = [r for r in rows if (r["game_id"] or r["db_name"]) in only_game_ids]
    if not rows:
        print("No problem databases recorded: nothing to fix.")
        return result

    if not os.path.isdir(runs_dir):
        raise FixError(
            f"runs_dir does not exist or is not a directory: {runs_dir!r} "
            f"(cannot locate the {len(rows)} recorded problem game(s))."
        )

    print(f"Problem games recorded: {len(rows)}")
    index = _index_db_basenames(runs_dir)

    for row in rows:
        game_id, db_name = row["game_id"], row["db_name"]
        related = _related_basenames(index, game_id, db_name)
        if not related:
            outcome = FixOutcome(db_name, game_id, "not-found")
            result.outcomes.append(outcome)
            _print_outcome(outcome)
            continue
        for basename in related:
            paths = index[basename]
            if len(paths) > 1:
                outcome = FixOutcome(
                    basename, game_id, "ambiguous",
                    detail=f"{len(paths)} files share this name under runs_dir",
                )
            else:
                outcome = _process_one(paths[0], basename, game_id, dry_run=dry_run, force=force)
            result.outcomes.append(outcome)
            _print_outcome(outcome)

    _print_summary(result)
    return result


def _print_outcome(outcome: FixOutcome) -> None:
    """One human-readable line per DB, in the extract-runner style."""
    rep = outcome.report
    if outcome.status == "repaired" and rep is not None:
        # Lead with the on-disk ground truth (page scan): it catches rows the walk's skip
        # counter silently misses. Fall back to the skip count only when no scan was made.
        if rep.rows_on_disk and rep.rows_on_disk > rep.rows_recovered:
            missing = rep.rows_on_disk - rep.rows_recovered
            pct = 100.0 * rep.rows_recovered / rep.rows_on_disk
            rows = (
                f"{rep.rows_recovered:,} of {rep.rows_on_disk:,} row(s) recovered ({pct:.1f}%; "
                f"{missing:,} lost to corrupt pages)"
            )
        elif rep.rows_skipped:
            rows = f"{rep.rows_recovered:,} row(s) recovered ({rep.rows_skipped:,} lost to corrupt pages)"
        else:
            rows = f"{rep.rows_recovered:,} row(s) recovered"
        dropped = f", {len(rep.objects_skipped)} object(s) dropped" if rep.objects_skipped else ""
        print(
            f"  [repaired] {outcome.db_name} ({rep.strategy}): "
            f"{len(rep.tables_recovered)} table(s), {rows}{dropped}"
        )
        for note in rep.tables_partial:   # tables kept, with some row loss
            print(f"             - {note}")
        for note in rep.objects_skipped:  # objects dropped entirely
            print(f"             - dropped {note}")
    elif outcome.status == "healthy":
        print(f"  [healthy]  {outcome.db_name}: already valid, left as-is")
    elif outcome.status == "dry-run":
        print(f"  [would examine] {outcome.db_name} → {outcome.path}")
    else:
        suffix = f": {outcome.detail}" if outcome.detail else ""
        print(f"  [{outcome.status}] {outcome.db_name}{suffix}")


def _print_summary(result: FixResult) -> None:
    counts = Counter(o.status for o in result.outcomes)

    print("\n" + "=" * 60)
    print("FIX COMPLETE")
    print("=" * 60)
    for status in (
        "repaired", "healthy", "dry-run", "failed",
        "not-found", "ambiguous", "skipped-bak-exists",
    ):
        if counts[status]:
            print(f"  {status}: {counts[status]}")

    if result.repaired:
        print(
            "\nRepaired DBs saved as <name>.db (originals kept as <name>.db.bak). "
            "Re-run `civ-bench extract --force-rebuild` to refresh the CSVs and clear "
            "import_issues.csv."
        )
    for outcome in result.failed:
        print(f"civ-bench: fix: could not repair {outcome.db_name}: {outcome.detail}",
              file=sys.stderr)
