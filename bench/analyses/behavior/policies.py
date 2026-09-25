"""``behavior.policies``: which policy branches players adopt, and when.

From the ``policies`` family of the behavior table (first-adoption turn per
branch, ``N/A`` for never), per player per game:

* ``adopted_<branch>``: 100 when the branch was adopted, else 0, so a group
  mean reads as the percent of players who adopted it;
* ``adoption_turn_<branch>``: the first-adoption turn, among adopters only;
* ``first_<branch>``: 100 when the branch was the player's first pick in its
  tier (:data:`bench.config.schema.POLICY_INFO`), so the Ancient tier's shares
  are the opening branches and the Ideology tier's the first ideology.

Both views are HTML heatmaps with one column per branch grouped by tier:
**Branch adoption** (fixed 0 to 100%, the tooltip adding the first-adoption
turn and the first-pick share) and **First-adoption turn** (a z-score per
branch in the absolute view). The relative view subtracts the matched baseline,
by default the in-game AI, pinned as the top row; against the in-game AI,
Vanilla players leave the tables. In controlled runs the module writes
``policies_by_seed`` / ``policies_by_seat`` for the Matched Maps Policies tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import schema as S
from ..base import AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from . import common as C
from .base import BehaviorAnalysis, MetricViews

NONE_LABEL = "none"
SHARE_RANGE = (0.0, 100.0)
SHARE_LEGEND = [[0.0, "0% never"], [0.5, "50%"], [1.0, "100% always"]]
RELATIVE_LEGEND = [[0.0, "2 SD below"], [0.25, "1 SD below"], [0.5, "same as the baseline"],
                   [0.75, "1 SD above"], [1.0, "2 SD above"]]
TURN_LEGEND = [[0.0, "2 SD earlier than the average player"], [0.25, "1 SD earlier"],
               [0.5, "average"], [0.75, "1 SD later"], [1.0, "2 SD later"]]
RELATIVE_TURN_LEGEND = [[0.0, "2 SD earlier than the baseline"], [0.25, "1 SD earlier"],
                        [0.5, "same as the baseline"], [0.75, "1 SD later"], [1.0, "2 SD later"]]
ADOPTION_TIPS = [
    {"column": "turn", "label": "Average first-adoption turn", "decimals": 0},
    {"column": "first", "label": "First pick in its tier", "unit": "%", "decimals": 0},
]
TURN_TIPS = [{"column": "share", "label": "Players who adopted it", "unit": "%", "decimals": 0}]


def adopted(branch: str) -> str:
    return f"adopted_{branch}"


def adoption_turn(branch: str) -> str:
    return f"adoption_turn_{branch}"


def first_pick(branch: str) -> str:
    return f"first_{branch}"


def _info(branch: str) -> tuple[str, str, str, str]:
    return S.POLICY_INFO.get(branch, (branch.title(), branch.title(), "Other", ""))


class BehaviorPolicies(BehaviorAnalysis):
    module = "behavior.policies"
    friendly_name = "Policy paths"
    description = (
        "Shows which policy branches and ideologies each player type adopts, which it "
        "picks first in each tier, and how early, against the in-game AI on the same "
        "map and seat."
    )
    report_defaults = {
        "tables": ["adoption_relative", "adoption_turn_relative",
                   "adoption_absolute", "adoption_turn_absolute"],
        "figures": [],
    }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        kinds = self.behavior_kinds(ctx)
        branches = [c for c, kind in kinds.items() if kind == "turn"]
        requested = self.params.get("branches")
        if requested:
            missing = [b for b in requested if b not in branches]
            if missing:
                raise AnalysisError(
                    f"behavior.policies '{self.stage_id}': branch(es) {missing} are not "
                    "selected in data.extract.behavior.policies."
                )
            branches = list(requested)
        if not branches:
            raise AnalysisError(
                f"behavior.policies '{self.stage_id}': the behavior extract selects no policy branch."
            )
        # Tier order, so each tier's columns sit together under one header.
        branches = [b for b in S.POLICY_INFO if b in set(branches)] + \
            [b for b in branches if b not in S.POLICY_INFO]

        rows, pool, _rate_scaled, baseline = self.load_behavior(ctx, branches)
        in_game_ai = baseline is not None and baseline.in_game_ai
        if in_game_ai:
            rows = rows[rows["player_type"].astype(str) != ctx.catalog.vanilla_label]
        if rows.empty:
            return AnalysisResult(summary="No behavior rows remain after filtering.")
        rows, pool = self._derive(rows, branches), self._derive(pool, branches)
        shares = [adopted(b) for b in branches]
        turns = [adoption_turn(b) for b in branches]
        cells = C.controlled_cells(ctx)
        views = C.build_views(rows, pool, shares + turns, cells, baseline)

        out = MetricViews()
        label = baseline.label if baseline is not None else "baseline"
        note = ""
        if views.relative is not None and views.baseline_summary:
            note = f" The top row is the {label} itself, in absolute terms."
            if in_game_ai:
                note += " In-game AI players are left out; the top row stands for them."
        common = dict(
            short_names={m: _info(b)[1] for b, m in zip(branches * 2, shares + turns)},
            column_tips={**{adopted(b): f"{_info(b)[0]}\n{_info(b)[3]}" for b in branches},
                         **{adoption_turn(b): f"{_info(b)[0]}\n{_info(b)[3]} Turn counted "
                            "among players who adopted it." for b in branches}},
            groups={m: _info(b)[2] for b, m in zip(branches * 2, shares + turns)},
            decimals=0,
            baseline_row=True,
        )
        self.add_heatmap_views(
            ctx, views, out, "adoption",
            titles={C.RELATIVE: "Relative branch adoption", C.ABSOLUTE: "Branch adoption"},
            help_texts={
                C.RELATIVE: (
                    "Each cell is the percent of players who adopted the branch minus the "
                    f"{label} at the same map and seat, in percentage points. Colors run from "
                    "2 SDs of the baseline players below (red) to 2 SDs above (blue). Columns "
                    f"are grouped by the tier in which the branch opens.{note}"
                ),
                C.ABSOLUTE: (
                    "Each cell is the percent of players who adopted the branch at any point. "
                    "Colors are fixed from 0% (red) through 50% (yellow) to 100% (blue). Hover "
                    "a cell for the average first-adoption turn and how often the branch was "
                    f"the first pick in its tier.{note}"
                ),
            },
            names={adopted(b): _info(b)[0] for b in branches},
            fixed={m: SHARE_RANGE for m in shares},
            legends={C.RELATIVE: RELATIVE_LEGEND, C.ABSOLUTE: SHARE_LEGEND},
            value_labels={C.RELATIVE: "Difference", C.ABSOLUTE: "Adopted by"},
            column_units={C.ABSOLUTE: {m: "%" for m in shares}, C.RELATIVE: {m: "pts" for m in shares}},
            companions={C.ABSOLUTE: {adopted(b): {"turn": adoption_turn(b), "first": first_pick(b)}
                                     for b in branches}},
            tip_rows={C.ABSOLUTE: ADOPTION_TIPS},
            **common,
        )
        self.add_heatmap_views(
            ctx, views, out, "adoption_turn",
            titles={C.RELATIVE: "Relative first-adoption turn", C.ABSOLUTE: "First-adoption turn"},
            help_texts={
                C.RELATIVE: (
                    "Each cell is the player's first-adoption turn minus the average of the "
                    f"{label} players who adopted it at the same map and seat, among players who "
                    "adopted the branch. Negative is earlier. Colors run from 2 SDs earlier (red) "
                    f"to 2 SDs later (blue).{note}"
                ),
                C.ABSOLUTE: (
                    "Each cell is the average turn on which players first adopted the branch, "
                    "among players who adopted it. Colors show each cell as a z-score within its "
                    f"branch, red earlier and blue later, full color at 2 SDs.{note}"
                ),
            },
            names={adoption_turn(b): _info(b)[0] for b in branches},
            legends={C.RELATIVE: RELATIVE_TURN_LEGEND, C.ABSOLUTE: TURN_LEGEND},
            value_labels={C.RELATIVE: "Difference", C.ABSOLUTE: "Average turn"},
            column_units={C.RELATIVE: {m: "turns" for m in turns}},
            companions={C.ABSOLUTE: {adoption_turn(b): {"share": adopted(b)} for b in branches}},
            tip_rows={C.ABSOLUTE: TURN_TIPS},
            **common,
        )
        matched_maps = self.add_matched_map_tables(
            ctx, views, pool, cells, out, "policies",
            label="Policies",
            tip="Policy paths: share of players adopting each branch",
            title="Branch adoption",
            help_text=(
                "Each cell is the percent of players on this map (and seat) who adopted the "
                f"branch. The top row is the {label} on the same map (and seat); the tooltip "
                "shows each cell's difference from it, matched seat by seat, the average "
                "first-adoption turn, and how often the branch was the first pick in its tier. "
                "Colors are fixed from 0% (red) through 50% (yellow) to 100% (blue)."
            ),
            names={adopted(b): _info(b)[0] for b in branches},
            short_names=common["short_names"],
            column_tips=common["column_tips"],
            groups=common["groups"],
            fixed={m: SHARE_RANGE for m in shares},
            legend=SHARE_LEGEND,
            decimals=0,
            column_units={m: "%" for m in shares},
            companions={adopted(b): {"turn": adoption_turn(b), "first": first_pick(b)}
                        for b in branches},
            tip_rows=ADOPTION_TIPS,
            value_label="Adopted by",
        )

        metadata = {
            **views.metadata(),
            "branches": branches,
            "views": C.views_metadata(out.declared, views.baseline),
            "heatmaps": out.heatmaps,
        }
        if matched_maps is not None:
            metadata["matched_maps"] = matched_maps
        names = {adopted(b): f"{_info(b)[0]} adoption (%)" for b in branches}
        return AnalysisResult(
            tables=out.tables,
            summary=self.heatmap_headline(ctx, out, views, "adoption", names, decimals=0),
            metadata=metadata,
        )

    def _derive(self, df: pd.DataFrame, branches: list[str]) -> pd.DataFrame:
        df = df.copy()
        for branch in branches:
            df[adopted(branch)] = df[branch].notna().astype(float) * 100.0
            df[adoption_turn(branch)] = df[branch]
        tiers: dict[str, list[str]] = {}
        for branch in branches:
            tiers.setdefault(_info(branch)[2], []).append(branch)
        for tier_branches in tiers.values():
            first = _earliest(df, tier_branches)
            for branch in tier_branches:
                df[first_pick(branch)] = (first == branch).astype(float) * 100.0
        return df


def _earliest(df: pd.DataFrame, branches: list[str]) -> pd.Series:
    """The branch with the smallest adoption turn per row (schema order breaks ties)."""
    if not branches or df.empty:
        return pd.Series(NONE_LABEL, index=df.index)
    turns = df[branches].to_numpy(dtype=float)
    filled = np.where(np.isnan(turns), np.inf, turns)
    first = np.array(branches, dtype=object)[filled.argmin(axis=1)]
    return pd.Series(np.where(np.isinf(filled.min(axis=1)), NONE_LABEL, first), index=df.index)
