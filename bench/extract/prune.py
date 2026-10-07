"""Confirmation gate for dropping table rows whose game DB is gone.

Any extract run that is not skipped drops rows for games with no DB file under
``runs_dir`` (a prune-only run does nothing else). A :class:`PruneGate` sits in
front of that step: it hands the missing games to a ``confirm`` callback and
remembers the answer per game, so a game is asked about once even when several
tables (or the auto-fix re-import) see it. Without a callback every missing game
is pruned, which keeps library callers and tests on the old behavior.
"""

from __future__ import annotations

import csv
import os
from typing import Callable, Iterable, Optional

# Receives one dict per missing game (its ``game_data`` row when known, else just
# ``{"game_id": ...}``) and returns True to prune them all.
PruneConfirm = Callable[[list[dict]], bool]


class PruneGate:
    """Decides which missing games may have their rows dropped."""

    def __init__(self, confirm: Optional[PruneConfirm] = None) -> None:
        self._confirm = confirm
        self._details: dict = {}
        self.approved: set = set()
        self.declined: set = set()

    def add_details(self, rows: dict) -> None:
        """Remember ``game_id → game_data row`` so the prompt can describe each game."""
        for game_id, row in rows.items():
            self._details.setdefault(game_id, row)

    def review(self, game_ids: Iterable[str]) -> set:
        """Return the subset of ``game_ids`` approved for pruning, asking about new ones."""
        game_ids = set(game_ids)
        pending = sorted(game_ids - self.approved - self.declined)
        if pending:
            if self._confirm is None or self._confirm(
                [self._details.get(g, {"game_id": g}) for g in pending]
            ):
                self.approved.update(pending)
            else:
                self.declined.update(pending)
        return game_ids & self.approved


def read_game_rows(path: str) -> dict:
    """Read ``game_data`` rows keyed by ``game_id`` (empty when the file is absent)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return {
            row["game_id"]: row
            for row in csv.DictReader(fh)
            if row.get("game_id") and row["game_id"] != "N/A"
        }
