"""Policy change counts and first-adoption turns for behavior rows."""

from __future__ import annotations

import json
import sqlite3

from ..change_counts import count_real_change_rows
from ..utilities import is_schema_mismatch
from .context import GameContext
from .events import _fetch_events

_POLICY_CHANGE_COUNT = "policy_changes"
_ADOPTION_EVENTS = ("PlayerAdoptPolicyBranch", "IdeologyAdopted")


class PolicyFamily:
    def columns(self, selection, stats) -> list[str]:
        return list(selection)

    def extract(self, cursor, ctx: GameContext, selection, stats) -> dict:
        if not selection:
            return {}

        result = {pid: {} for pid in ctx.major_players}
        if _POLICY_CHANGE_COUNT in selection:
            for pid in ctx.major_players:
                try:
                    result[pid][_POLICY_CHANGE_COUNT] = count_real_change_rows(
                        cursor, "PolicyChanges", pid,
                    )
                except sqlite3.DatabaseError as exc:
                    if not is_schema_mismatch(exc):
                        raise
                    result[pid][_POLICY_CHANGE_COUNT] = "N/A"

        adoption_branches = [name for name in selection if name != _POLICY_CHANGE_COUNT]
        for pid in ctx.major_players:
            for branch in adoption_branches:
                result[pid][branch] = "N/A"

        if not adoption_branches:
            return result

        selected = set(adoption_branches)
        adopted = set()
        events = sorted(_fetch_events(cursor, _ADOPTION_EVENTS), key=lambda event: event[1])
        for _event_type, turn, payload_json in events:
            try:
                payload = json.loads(payload_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            player_id = payload.get("PlayerID", -1)
            branch_type = payload.get("BranchType", "")
            branch = branch_type.lower() if isinstance(branch_type, str) else ""
            if player_id not in result or branch not in selected:
                continue
            key = (player_id, branch)
            if key in adopted:
                continue
            result[player_id][branch] = turn
            adopted.add(key)

        return result
