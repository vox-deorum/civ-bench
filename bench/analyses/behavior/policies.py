"""``behavior.policies``: which policy branches players adopt, and when.

From the ``policies`` family of the behavior table (first-adoption turn per
branch, ``N/A`` for never), per player per game:

* ``adopted_<branch>``: 1 when the branch was adopted, else 0.
* ``adoption_turn_<branch>``: the first-adoption turn, among adopters only.

The absolute view adds the opening branch (the earliest non-ideology branch)
and the ideology (the first ideology adopted), as shares of players.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import schema as S
from ..base import AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from . import common as C
from .base import BehaviorAnalysis

NONE_LABEL = "none"


class BehaviorPolicies(BehaviorAnalysis):
    module = "behavior.policies"
    friendly_name = "Policy paths"
    description = (
        "Shows which policy branches and ideologies each player type adopts and "
        "how early, against the in-game AI on the same map and seat."
    )
    report_defaults = {
        "tables": [],
        "figures": [
            "adoption_relative", "adoption_turn_relative",
            "adoption_absolute", "adoption_turn_absolute",
            "openings_absolute", "ideologies_absolute",
        ],
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

        rows, pool, _rate_scaled, baseline = self.load_behavior(ctx, branches)
        if rows.empty:
            return AnalysisResult(summary="No behavior rows remain after filtering.")
        rows, pool = self._derive(rows, branches), self._derive(pool, branches)
        adopted = [f"adopted_{b}" for b in branches]
        turns = [f"adoption_turn_{b}" for b in branches]
        views = C.build_views(rows, pool, adopted + turns, C.controlled_cells(ctx), baseline)
        labels = {
            **{f"adopted_{b}": b.title() for b in branches},
            **{f"adoption_turn_{b}": b.title() for b in branches},
        }
        out = self.metric_views(ctx, views, "adoption", "Branch adoption share", labels, metrics=adopted)
        self.add_metric_views(ctx, views, out, "adoption_turn", "First-adoption turn", labels, metrics=turns)

        for slug, column, categories, title in (
            ("openings", "opening_branch", self._openings(branches), "Opening branch"),
            ("ideologies", "ideology", self._ideologies(branches), "Ideology"),
        ):
            if len(categories) <= 1:
                continue
            table = C.share_table(views.absolute, self.by, column, categories)
            out.add(C.ABSOLUTE, f"{slug}_absolute", table=table)
            out.add(C.ABSOLUTE, f"{slug}_absolute", figure=C.stacked_shares_figure(
                table.set_index(self.by)[categories], title=title,
            ))

        metadata = {
            **views.metadata(),
            "branches": branches,
            "views": C.views_metadata(out.declared),
        }
        return AnalysisResult(
            tables=out.tables, figures=out.figures,
            summary=self.headline(out, views, "adoption", {m: f"{labels[m]} adoption" for m in adopted}),
            metadata=metadata,
        )

    def _openings(self, branches: list[str]) -> list[str]:
        return [b for b in branches if b not in S.BEHAVIOR_IDEOLOGIES] + [NONE_LABEL]

    def _ideologies(self, branches: list[str]) -> list[str]:
        return [b for b in branches if b in S.BEHAVIOR_IDEOLOGIES] + [NONE_LABEL]

    def _derive(self, df: pd.DataFrame, branches: list[str]) -> pd.DataFrame:
        df = df.copy()
        for branch in branches:
            df[f"adopted_{branch}"] = df[branch].notna().astype(float)
            df[f"adoption_turn_{branch}"] = df[branch]
        df["opening_branch"] = _earliest(df, [b for b in branches if b not in S.BEHAVIOR_IDEOLOGIES])
        df["ideology"] = _earliest(df, [b for b in branches if b in S.BEHAVIOR_IDEOLOGIES])
        return df


def _earliest(df: pd.DataFrame, branches: list[str]) -> pd.Series:
    """The branch with the smallest adoption turn per row (schema order breaks ties)."""
    if not branches or df.empty:
        return pd.Series(NONE_LABEL, index=df.index)
    turns = df[branches].to_numpy(dtype=float)
    filled = np.where(np.isnan(turns), np.inf, turns)
    first = np.array(branches, dtype=object)[filled.argmin(axis=1)]
    return pd.Series(np.where(np.isinf(filled.min(axis=1)), NONE_LABEL, first), index=df.index)
