"""Retention and removal preflight for extraction with cold storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bench.extract.errors import ExtractError
from bench.extract.prune import PruneGate
from bench.extract.sources import Inventory, SourceState, csv_game_ids, prune_csv
from bench.extract.utilities import csv_header_matches, get_game_id_from_path


@dataclass
class TieredExtraction:
    """Two-tier bookkeeping for one extract run.

    ``load`` reads only the ownership sidecar, which is enough for the skip check.
    ``read_tables`` reads every cached game id and must run before the review,
    preflight and write steps.
    """

    inventory: Inventory
    state: SourceState
    cached: dict[str, set[str]] = field(default_factory=dict)
    protected: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, inventory: Inventory, games_path: str) -> TieredExtraction:
        return cls(inventory, SourceState.load(games_path, inventory))

    def split_absent(self, game_ids: set[str]) -> tuple[set[str], set[str]]:
        """Split games with no selected source into (protected, missing)."""
        absent = game_ids - self.inventory.game_ids
        protected = self.state.protected(absent, self.inventory)
        return protected, absent - protected

    def is_current(self, checkpoint_paths: dict[str, str]) -> bool:
        """Every output was completed against this inventory and no removal is due.

        A matching checkpoint means the last run already observed this exact
        inventory, so a skipped run has nothing new to record.
        """
        _, due = self.split_absent(self.state.pending_removals)
        return not due and all(
            self.state.checkpoints.get(str(Path(path).resolve())) == self.inventory.signature
            for path in checkpoint_paths.values()
        )

    def read_tables(self, all_paths: dict[str, str], issues_path: str) -> None:
        self.cached = {path: csv_game_ids(path) for path in [*all_paths.values(), issues_path]}
        all_ids = set().union(*self.cached.values(), self.state.pending_removals)
        self.state.observe(self.inventory, unknown_cached_ids=all_ids - set(self.state.games))
        self.protected, self.missing = self.split_absent(all_ids)

    def preflight(self, output_paths: dict[str, str], all_paths: dict[str, str], fields: dict,
                  selected_files: list[str], prune_only: bool, removed: set[str]) -> None:
        selected_ids = {get_game_id_from_path(path) for path in selected_files}
        for key, path in all_paths.items():
            if not Path(path).exists() or csv_header_matches(path, fields[key]):
                continue
            retained = self.cached[path] - removed
            affected = bool(self.cached[path] & removed)
            if key not in output_paths and not affected:
                continue
            if prune_only or key not in output_paths:
                raise ExtractError(f"cannot prune table with an out-of-date header: {path}")
            if retained - selected_ids:
                raise ExtractError(
                    f"cannot rebuild {path}: cached games have unavailable or unselected sources; "
                    "restore the sources and remove max_dbs before changing the table schema"
                )

    def review(self, gate: PruneGate) -> set[str]:
        gate.protect(self.protected | self.inventory.game_ids)
        gate.approve(self.state.pending_removals & self.missing)
        return gate.review(self.missing)

    def journal(self, removed: set[str], all_paths: dict[str, str], checkpoint_paths: dict[str, str]) -> None:
        """Save approved removals and drop the checkpoints this run will invalidate.

        This runs before the first table write. A failed run can finish without a
        second confirmation, but a reappeared source cancels its old approval.
        """
        self.state.pending_removals.update(removed)
        changed_paths = {
            path for path in all_paths.values() if self.cached[path] & removed
        } | set(checkpoint_paths.values())
        for path in changed_paths:
            self.state.checkpoints.pop(str(Path(path).resolve()), None)
        self.state.save()

    def prune(self, removed: set[str], all_paths: dict[str, str]) -> None:
        """Drop removed games from every table on disk, ``games`` last."""
        keys = [key for key in all_paths if key != "games"] + ["games"]
        for key in keys:
            path = all_paths[key]
            if self.cached[path] & removed:
                prune_csv(path, removed)

    def finish(self, removed: set[str], checkpoint_paths: dict[str, str], complete: bool) -> None:
        for gid in removed:
            self.state.games.pop(gid, None)
        self.state.pending_removals.difference_update(removed)
        if complete:
            for path in checkpoint_paths.values():
                self.state.checkpoints[str(Path(path).resolve())] = self.inventory.signature
        self.state.save()
