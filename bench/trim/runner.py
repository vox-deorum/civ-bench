"""``civ-bench trim`` orchestrator: drop extra games from repeated controlled slots.

A controlled experiment needs one game per ``(seed, seating_rotation)`` slot. When a
slot holds several accepted games, the earliest by timestamp stays and the rest are
trimmed, using the same rule as the ``repeated_games`` coverage table. Games excluded
for decision failures or recorded in ``import_issues.csv`` are left alone: they are
never counted, kept, or trimmed.

A slot whose keep candidate has no game DB in any available source tier is skipped,
since trimming it would leave the slot empty once the prune drops the keeper's rows
too. A slot is also deferred when an extra may still be owned by an offline cold
tier: neither its hot copies nor its extras are renamed, because the missing source
might hold the only remaining copy.

Trimming a game renames every file in the available hot and cold roots whose name
starts with its uuid (game DBs of any timestamp, player trace DBs, saves, replays)
to ``<name>.bak``. A
prune-only extract then drops the trimmed games' rows from every canonical table on
disk, not only ``data.extract.outputs``. The ``games`` table is pruned last: if
another table fails, the slot still shows as repeated, and re-running trim finds the
already-renamed extras as ``not-found`` and finishes the prune.
If an affected tier is offline, the retry waits for it to return, even when the
earlier removal was approved. A two-tier trim requires a complete hot scan.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal, Optional

import pandas as pd

from ..config import RunConfig
from bench.config.schema import DEFAULT_RUNS_DIR
from ..config.behavior import resolve_behavior_spec
from ..config.filters import resolve_filter_spec
from ..data.failures import failed_game_ids_from_tokens
from ..data.loading import drop_problem_games
from ..data.repeats import controlled_rows, repeated_slots
from ..extract import ExtractResult, run_extract
from ..extract.errors import ExtractError
from ..extract.issues import read_problem_game_ids, resolve_issues_path
from ..extract.prune import PruneConfirm, PruneGate
from ..extract.runner import DEFAULT_TABLE_PATHS, expected_fieldnames, table_path
from bench.extract.sources import Inventory, SourceState, TierScan, scan_sources
from ..extract.utilities import csv_header_matches
from .errors import TrimError

Status = Literal[
    "trimmed", "not-found", "skipped-bak-exists", "skipped-keeper-missing", "failed",
    "would-trim", "skipped-source-unavailable",
]
# Receives the repeated-slot rows about to be trimmed and returns True to go ahead.
TrimConfirm = Callable[[list[dict]], bool]


@dataclass
class TrimOutcome:
    """The result of trimming one extra game."""

    game_id: str
    status: Status
    files: list[str] = field(default_factory=list)
    detail: str = ""
    # ``.bak`` paths a failed rename could not restore; empty after a clean rollback.
    stranded: list[str] = field(default_factory=list)


@dataclass
class TrimResult:
    slots: list[dict] = field(default_factory=list)
    outcomes: list[TrimOutcome] = field(default_factory=list)
    declined: bool = False
    prune: Optional[ExtractResult] = None

    @property
    def trimmed(self) -> list[TrimOutcome]:
        return [o for o in self.outcomes if o.status == "trimmed"]

    @property
    def failed(self) -> list[TrimOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]


def _excluded_game_ids(cfg: RunConfig) -> set[str]:
    """Malformed-DB games plus games over the global decision-failure cutoff."""
    resolved = resolve_filter_spec(cfg.data.get("filter"), cfg.filters, "data.filter")
    return read_problem_game_ids(resolve_issues_path(cfg.data)) | failed_game_ids_from_tokens(
        table_path(cfg, "tokens"), resolved
    )


def _index_files(inventory: Inventory) -> list[str]:
    """Every file in the available source tiers except backups, in stable order.

    Renames follow the game prefix across both roots, so duplicate snapshots,
    traces, replays, and saves in either tier are trimmed together.
    """
    paths = {
        path
        for tier in inventory.tiers.values()
        if tier.available
        for path in tier.files
        if not os.path.basename(path).endswith(".bak")
    }
    return sorted(paths)


def _files_for(paths: list[str], game_id: str) -> list[str]:
    return [p for p in paths if os.path.basename(p).startswith(game_id)]


def _rename_game(game_id: str, files: list[str]) -> TrimOutcome:
    """Rename all of one game's files to ``.bak``, or none of them. Never raises."""
    if not files:
        return TrimOutcome(game_id, "not-found", detail="no source files under runs_dir")
    taken = [p + ".bak" for p in files if os.path.exists(p + ".bak")]
    if taken:
        names = ", ".join(os.path.basename(p) for p in taken)
        return TrimOutcome(game_id, "skipped-bak-exists", files, detail=f"{names} already exists")
    done: list[str] = []
    try:
        for path in files:
            os.rename(path, path + ".bak")
            done.append(path)
    except OSError as exc:
        details = [str(exc)]
        stranded: list[str] = []
        for path in reversed(done):  # put the game back the way it was
            try:
                os.rename(path + ".bak", path)
            except OSError as undo_exc:
                stranded.append(path + ".bak")
                details.append(f"could not restore {path}: {undo_exc}")
        stranded.sort()
        if stranded:
            details.append("still renamed: " + ", ".join(stranded))
        return TrimOutcome(game_id, "failed", files, "; ".join(details), stranded)
    return TrimOutcome(game_id, "trimmed", files)


def _prune_tables(cfg: RunConfig) -> list[str]:
    """Canonical tables on disk that the prune must cover, with ``games`` last.

    Analyses read every table at its ``data.tables`` path, so the prune covers each
    one that exists, whatever ``data.extract.outputs`` lists. Pruning ``games`` last
    keeps a failed prune retryable. Raises :class:`TrimError` when a table's header
    is out of date, since a prune-only extract leaves such a file untouched.
    """
    behavior_spec = resolve_behavior_spec((cfg.data.get("extract", {}) or {}).get("behavior"))
    keys = [k for k in DEFAULT_TABLE_PATHS if k != "games"] + ["games"]
    tables = [k for k in keys if Path(table_path(cfg, k)).exists()]
    stale = [
        f"{k} ({table_path(cfg, k)})" for k in tables
        if not csv_header_matches(table_path(cfg, k), expected_fieldnames(k, behavior_spec))
    ]
    if stale:
        raise TrimError(
            "these tables have an out-of-date header, so the trimmed games' rows could "
            f"not be pruned from them: {', '.join(stale)}. Rebuild them with "
            "`civ-bench extract --force-rebuild` (or move them away), then run trim again."
        )
    return tables


def run_trim(
    cfg: RunConfig,
    dry_run: bool = False,
    confirm: Optional[TrimConfirm] = None,
    prune_confirm: Optional[PruneConfirm] = None,
) -> TrimResult:
    """Trim the extra games in repeated controlled slots and prune their rows.

    ``confirm`` sees the repeated slots before anything is renamed; ``None`` goes
    ahead without asking. ``prune_confirm`` is the prune gate's callback for any
    *other* game whose DB is missing; the trimmed games are already approved.
    Raises :class:`TrimError` before renaming when the ``games`` table is missing or
    a table could not be pruned, and after renaming when the prune fails.
    """
    extract_cfg = cfg.data.get("extract", {}) or {}
    runs_dir = extract_cfg.get("runs_dir", DEFAULT_RUNS_DIR)
    games_path = table_path(cfg, "games")

    print("=" * 60)
    print("civ-bench trim" + (" (dry-run)" if dry_run else ""))
    print("=" * 60)
    print(f"runs_dir: {runs_dir}")
    print(f"games:    {games_path}")

    if not Path(games_path).exists():
        raise TrimError(
            f"games table not found at {games_path!r}; run `civ-bench extract` first."
        )
    games = pd.read_csv(games_path, dtype={"game_id": str})
    accepted = drop_problem_games(controlled_rows(games), _excluded_game_ids(cfg))

    result = TrimResult(slots=repeated_slots(accepted, games))
    if not result.slots:
        print("No repeated controlled slots: nothing to trim.")
        return result

    tables = _prune_tables(cfg) if cfg.extract_enabled else []
    try:
        tiered = extract_cfg.get("cold_runs_dir") is not None
        if not tiered and not os.path.isdir(runs_dir):
            # A legacy cached-only trim reports a missing keeper without writing.
            inventory = Inventory({"hot": TierScan(str(Path(runs_dir).resolve()), False)})
        else:
            inventory = scan_sources(extract_cfg)
        state = SourceState.load(games_path, inventory) if tiered else None
    except (ExtractError, OSError) as exc:
        raise TrimError(f"could not scan the game data sources or load ownership: {exc}") from exc
    # Reading the state sidecar validates it but never writes. Dry runs take the
    # same path, so neither a dry run nor a deferred slot changes ownership.
    paths = _index_files(inventory)
    plan: list[tuple[str, list[str]]] = []
    active_slots: list[dict] = []
    for slot in result.slots:
        keeper = slot["keep_candidate_game_id"]
        extras = slot["extra_game_ids"].split(",")
        print(f"\n{slot['experiment']}  seed={slot['seed']}  "
              f"rotation={slot['seating_rotation']}  keep {keeper}")
        if keeper not in inventory.game_ids:
            print(f"  skip: {keeper} has no game DB in any available source tier, so "
                  f"trimming would empty the slot")
            result.outcomes += [
                TrimOutcome(g, "skipped-keeper-missing", detail=f"keeper {keeper} has no game DB")
                for g in extras
            ]
            continue
        extra_ids = set(extras)
        blocked = sorted(state.unavailable_copies(extra_ids, inventory)) if state else []
        if blocked:
            detail = (
                "cold source is offline and owns or may own cached copies for "
                f"{', '.join(blocked)}; the whole slot was deferred"
            )
            print(f"  skip: {detail}")
            result.outcomes += [
                TrimOutcome(g, "skipped-source-unavailable", detail=detail)
                for g in extras
            ]
            continue
        active_slots.append(slot)
        for game_id in extras:
            files = _files_for(paths, game_id)
            plan.append((game_id, files))
            note = f"{len(files)} file(s)" if files else "no files left, rows only"
            print(f"  trim {game_id} ({note})")
            for path in files:
                print(f"    {path}")

    if dry_run:
        result.outcomes += [TrimOutcome(g, "would-trim", files) for g, files in plan]
        _print_summary(result)
        return result
    if not plan:
        _print_summary(result)
        return result
    if confirm is not None and not confirm(active_slots):
        result.declined = True
        print("Trim declined: nothing was renamed.")
        return result

    for game_id, files in plan:
        outcome = _rename_game(game_id, files)
        result.outcomes.append(outcome)
        if outcome.detail:
            print(f"  [{outcome.status}] {game_id}: {outcome.detail}")
    _print_summary(result)

    # Games with no files left on disk are gone either way, so their rows go too.
    gone = {o.game_id for o in result.outcomes if o.status in ("trimmed", "not-found")}
    if gone and cfg.extract_enabled:
        gate = PruneGate(prune_confirm)
        gate.approve(gone)
        prune_cfg = replace(cfg, data={**cfg.data, "extract": {**extract_cfg, "outputs": tables}})
        try:
            result.prune = run_extract(prune_cfg, prune_missing=True, prune_gate=gate)
        except (ExtractError, OSError) as exc:
            raise TrimError(
                f"pruning the trimmed games failed: {exc}. Their files stay renamed; fix "
                "the cause and run `civ-bench trim` again to finish the prune."
            ) from exc
    elif gone:
        print("data.extract.enabled is false: remove the trimmed games from the tables yourself.")
    return result


def _print_summary(result: TrimResult) -> None:
    counts = Counter(o.status for o in result.outcomes)
    print("\n" + "=" * 60)
    print("TRIM COMPLETE")
    print("=" * 60)
    print(f"  repeated slots: {len(result.slots)}")
    for status in ("trimmed", "would-trim", "not-found", "skipped-bak-exists",
                   "skipped-keeper-missing", "skipped-source-unavailable", "failed"):
        if counts[status]:
            print(f"  {status}: {counts[status]}")
