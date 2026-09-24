"""Relationship family: how a player sets its diplomatic modifiers toward rivals.

``RelationshipChanges`` rows record the public and private modifiers a
strategist sets on one target. The two values add up in the game's diplomacy
calculation, so the family reports each side, their sum, and how often the two
point in opposite directions.

A pair's state follows the ``StateFamily`` rule: the last row of a turn wins and
holds until the next row. Averages and shares weigh pair-turns, counted from the
pair's first row through the earlier of the two players' survival turns. Turns
after a reset to 0 still count; targets the player never set are left out.
"""

from __future__ import annotations

import sqlite3

from ..utilities import is_schema_mismatch
from .context import GameContext
from .state import turn_weights


class RelationshipFamily:
    def extract(self, cursor, ctx: GameContext, selection, stats) -> dict:
        if not selection:
            return {}
        majors = set(ctx.major_players)
        placeholders = ",".join(["?"] * len(ctx.major_players))
        try:
            cursor.execute(
                f"SELECT PlayerID, TargetID, Turn, PublicValue, PrivateValue FROM RelationshipChanges "
                f"WHERE PlayerID IN ({placeholders}) ORDER BY PlayerID, TargetID, Turn, ID",
                ctx.major_players,
            )
            rows = cursor.fetchall()
        except sqlite3.DatabaseError as exc:
            if is_schema_mismatch(exc):
                return {}
            raise

        # player → target → {turn: (public, private)}; a later row in a turn replaces an earlier one.
        states: dict = {}
        changes = dict.fromkeys(ctx.major_players, 0)
        previous: dict = {}
        for pid, target, turn, public, private in rows:
            if target not in majors or target == pid or public is None or private is None:
                continue
            values = (public, private)
            # A re-set to the values already in effect is not a change.
            if previous.get((pid, target), (0, 0)) != values:
                changes[pid] += 1
            previous[(pid, target)] = values
            states.setdefault(pid, {}).setdefault(target, {})[turn] = values

        result = {}
        for pid in ctx.major_players:
            by_target = states.get(pid, {})
            totals = {"weight": 0, "public": 0, "private": 0, "hostility": 0, "goodwill": 0}
            for target, by_turn in by_target.items():
                turns = sorted(by_turn)
                end = min(
                    ctx.survival_turn.get(pid, turns[-1]),
                    ctx.survival_turn.get(target, turns[-1]),
                )
                for turn, weight in zip(turns, turn_weights(turns, end)):
                    public, private = by_turn[turn]
                    totals["weight"] += weight
                    totals["public"] += public * weight
                    totals["private"] += private * weight
                    if public > 0 > private:
                        totals["hostility"] += weight
                    elif public < 0 < private:
                        totals["goodwill"] += weight

            weight = totals["weight"]
            values = {
                "relationship_changes": changes[pid],
                "relationship_targets": len(by_target),
                "stance_public_avg": _ratio(totals["public"], weight),
                "stance_private_avg": _ratio(totals["private"], weight),
                "stance_net_avg": _ratio(totals["public"] + totals["private"], weight),
                "stance_masked_hostility_share": _ratio(totals["hostility"], weight),
                "stance_masked_goodwill_share": _ratio(totals["goodwill"], weight),
            }
            result[pid] = {name: values[name] for name in selection}
        return result


def _ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else ""
