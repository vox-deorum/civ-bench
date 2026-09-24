"""``behavior.diplomacy``: diplomatic persona and relationship stance.

Combines the diplomatic persona traits (turn-weighted averages) with the
``relationships`` family: how often the strategist changes its modifiers, the
public, private, and net stance it holds, and how often public and private
point in opposite directions. The in-game AI never sets relationship
modifiers, so against an in-game AI baseline the stance metrics appear only in
the absolute view.
"""

from __future__ import annotations

import pandas as pd

from ...config import schema as S
from ...config.behavior import snake_case
from ..base import AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from . import common as C
from .base import BehaviorAnalysis

STANCE_COLUMNS = [
    "relationship_changes", "relationship_targets",
    "stance_public_avg", "stance_private_avg", "stance_net_avg",
    "stance_masked_hostility_share", "stance_masked_goodwill_share",
]
MIX_LABELS = {
    "stance_masked_hostility_share": "masked hostility",
    "stance_masked_goodwill_share": "masked goodwill",
    "aligned": "aligned",
}


def trait_column(trait: str) -> str:
    return f"persona_{snake_case(trait)}_avg"


class BehaviorDiplomacy(BehaviorAnalysis):
    module = "behavior.diplomacy"
    friendly_name = "Diplomatic behavior"
    description = (
        "Describes diplomatic persona traits and the public and private stances "
        "strategists set toward rivals, including how often the two conflict."
    )
    report_defaults = {
        "tables": [],
        "figures": ["diplomacy_relative", "diplomacy_absolute", "stance_mix_absolute"],
    }

    @property
    def traits(self) -> list[str]:
        return list(self.params.get("traits") or S.BEHAVIOR_DIPLOMACY_TRAITS)

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        kinds = self.behavior_kinds(ctx)
        traits = [trait_column(t) for t in self.traits]
        metrics = [m for m in traits + STANCE_COLUMNS if m in kinds]
        if not metrics:
            raise AnalysisError(
                f"behavior.diplomacy '{self.stage_id}': the behavior extract selects none of "
                "the diplomatic persona averages or relationship columns."
            )
        rows, pool, rate_scaled, baseline = self.load_behavior(ctx, metrics)
        if rows.empty:
            return AnalysisResult(summary="No behavior rows remain after filtering.")
        views = C.build_views(rows, pool, metrics, C.controlled_cells(ctx), baseline)
        labels = {m: C.metric_label(m, rate_scaled) for m in metrics}
        for trait, column in zip(self.traits, traits):
            labels[column] = f"{trait} (persona)"
        out = self.metric_views(ctx, views, "diplomacy", "Diplomatic behavior", labels)

        mix = self._stance_mix(views.absolute)
        mix_note = ""
        if mix is not None:
            out.add(C.ABSOLUTE, "stance_mix_absolute", table=mix)
            shares = mix.set_index(self.by)[list(MIX_LABELS.values())]
            out.add(C.ABSOLUTE, "stance_mix_absolute", figure=C.stacked_shares_figure(
                shares, title="Stance mix over pair-turns", xlabel="share of pair-turns (player mean)",
            ))
            top = mix.sort_values("masked hostility", ascending=False).iloc[0]
            mix_note = (
                f" **{top[self.by]}** holds the most masked hostility (public warm, private "
                f"hostile) on **{top['masked hostility']:.0%}** of pair-turns."
            )

        metadata = {
            **views.metadata(),
            "traits": self.traits,
            "rate": self.rate,
            "views": C.views_metadata(out.declared, views.baseline),
        }
        return AnalysisResult(
            tables=out.tables, figures=out.figures,
            summary=self.headline(out, views, "diplomacy", labels) + mix_note,
            metadata=metadata,
        )

    def _stance_mix(self, rows: pd.DataFrame):
        shares = ["stance_masked_hostility_share", "stance_masked_goodwill_share"]
        if any(s not in rows.columns for s in shares):
            return None
        stance = rows.dropna(subset=shares)
        if stance.empty:
            return None
        grouped = stance.groupby(self.by)[shares].mean()
        grouped["aligned"] = (1.0 - grouped[shares].sum(axis=1)).clip(lower=0.0)
        grouped = grouped.rename(columns=MIX_LABELS)
        grouped.insert(0, "n_players", stance.groupby(self.by).size())
        out = grouped.reset_index()
        order = C.order_groups(out.assign(metric=""), self.by)[self.by]
        return out.set_index(self.by).loc[list(order)].reset_index()
