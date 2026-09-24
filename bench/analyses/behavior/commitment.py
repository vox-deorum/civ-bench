"""``behavior.commitment``: how consistently strategists hold a plan.

From the per-turn table, per player per game:

* ``decision_rate`` / ``change_rate``: share of turns the strategist acted
  (``is_decision``) or changed a flavor value (``is_changed``).
* ``grand_strategy_switches``: grand-strategy changes between consecutive
  recorded turns, per 100 turns.
* ``main_strategy_share``: share of turns spent in the player's most common
  grand strategy; ``first_pivot_turn``: first turn off the opening strategy.
* ``grand_strategy_share_<name>``: share of turns in each grand strategy.

The panel adds the strategist's real change counts (``strategy_changes``,
``persona_changes``, ``research_changes``), per 100 turns by default.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..base import AnalysisContext, AnalysisResult
from . import common as C
from .base import BehaviorAnalysis

TURN_COLUMNS = ["experiment", "game_id", "player_id", "player_type", "turn",
                "is_decision", "is_changed", "grand_strategy"]
PANEL_COUNTS = ["strategy_changes", "persona_changes", "research_changes"]
KEY = ["game_id", "player_id"]
LABELS = {
    "decision_rate": "decision rate",
    "change_rate": "change rate",
    "grand_strategy_switches": "grand-strategy switches /100 turns",
    "main_strategy_share": "main-strategy share",
    "first_pivot_turn": "first pivot turn",
}


def strategy_column(name: str) -> str:
    return "grand_strategy_share_" + re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def turn_metrics(turns: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Per ``(game_id, player_id)`` commitment metrics from per-turn rows."""
    turns = turns.sort_values(KEY + ["turn"], kind="stable")
    grouped = turns.groupby(KEY, sort=True)
    out = pd.DataFrame({
        "n_turns": grouped.size(),
        "decision_rate": grouped["is_decision"].mean(),
        "change_rate": grouped["is_changed"].mean(),
    })

    gs = turns[turns["grand_strategy"].notna() & (turns["grand_strategy"].astype(str) != "")]
    if gs.empty:
        for column in ["grand_strategy_switches", "main_strategy_share", "first_pivot_turn",
                       *[strategy_column(s) for s in strategies]]:
            out[column] = np.nan
        return out.reset_index()

    same_player = (gs["game_id"] == gs["game_id"].shift()) & (gs["player_id"] == gs["player_id"].shift())
    switched = same_player & (gs["grand_strategy"] != gs["grand_strategy"].shift())
    gs_grouped = gs.assign(_switch=switched.astype(int)).groupby(KEY, sort=True)
    n_gs = gs_grouped.size()
    out["grand_strategy_switches"] = gs_grouped["_switch"].sum() / n_gs * 100.0

    counts = gs.groupby(KEY + ["grand_strategy"], sort=True).size().unstack(fill_value=0)
    out["main_strategy_share"] = counts.max(axis=1) / n_gs
    for strategy in strategies:
        column = counts[strategy] if strategy in counts.columns else 0
        out[strategy_column(strategy)] = column / n_gs

    opening = gs_grouped["grand_strategy"].first().rename("_opening")
    off = gs.join(opening, on=KEY)
    off = off[off["grand_strategy"] != off["_opening"]]
    out["first_pivot_turn"] = off.groupby(KEY, sort=True)["turn"].min()
    return out.reset_index()


class BehaviorCommitment(BehaviorAnalysis):
    module = "behavior.commitment"
    friendly_name = "Strategic commitment"
    description = (
        "Shows how often strategists act, change course, and switch grand "
        "strategy, and how much of the game they spend in their main strategy."
    )
    report_defaults = {
        "tables": [],
        "figures": [
            "commitment_relative", "strategy_shares_relative",
            "commitment_absolute", "strategy_mix_absolute",
        ],
    }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        turn_range = ctx.resolved_filter().get("turn_range")
        full_turns = self._turns(ctx, turn_range)
        full_panel = C.prepare_keys(ctx.load_table("panel"), self.stage_id, "panel")
        C.check_by(full_panel, self.by, self.stage_id)
        counts = [c for c in PANEL_COUNTS if c in full_panel.columns]
        full_panel = C.numeric(full_panel, counts)
        rate_scaled = set(counts) if self.rate == "per_100_turns" else set()
        if rate_scaled:
            full_panel = C.per_100_turns(full_panel, sorted(rate_scaled))

        strategies = sorted(
            str(s) for s in full_turns["grand_strategy"].dropna().unique() if str(s) != ""
        )
        metrics = [*LABELS, *[strategy_column(s) for s in strategies], *counts]
        labels = {**LABELS, **{strategy_column(s): f"{s} share" for s in strategies},
                  **{c: C.metric_label(c, rate_scaled) for c in counts}}

        rows_turns = ctx.apply_filter(full_turns)
        rows_panel = C.prepare_keys(ctx.apply_filter(full_panel), self.stage_id, "panel")
        rows = self._join(rows_panel, turn_metrics(rows_turns, strategies), counts)
        if rows.empty:
            return AnalysisResult(summary="No players remain after filtering.")

        baseline = self.baseline(ctx)
        pool = pd.DataFrame(columns=rows.columns)
        if baseline:
            pool_turns = C.baseline_pool(ctx, full_turns, baseline)
            pool_panel = C.baseline_pool(ctx, full_panel, baseline)
            pool = self._join(pool_panel, turn_metrics(pool_turns, strategies), counts)
        views = C.build_views(rows, pool, metrics, C.controlled_cells(ctx), baseline)

        core = [m for m in metrics if not m.startswith("grand_strategy_share_")]
        out = self.metric_views(ctx, views, "commitment", "Strategic commitment", labels, metrics=core)
        shares = [strategy_column(s) for s in strategies]
        if shares:
            self.add_metric_views(ctx, views, out, "strategy_shares", "Grand-strategy mix",
                                  labels, metrics=shares)
            mix = self._mix(views.absolute, strategies)
            out.add(C.ABSOLUTE, "strategy_mix_absolute", table=mix)
            out.add(C.ABSOLUTE, "strategy_mix_absolute", figure=C.stacked_shares_figure(
                mix.set_index(self.by)[strategies], title="Grand-strategy mix",
                xlabel="share of turns (player mean)",
            ))

        metadata = {
            **views.metadata(),
            "grand_strategies": strategies,
            "rate": self.rate,
            "views": C.views_metadata(out.declared, views.baseline),
        }
        if turn_range is not None:
            metadata["turn_range"] = list(turn_range)
        return AnalysisResult(
            tables=out.tables, figures=out.figures,
            summary=self.headline(out, views, "commitment", labels), metadata=metadata,
        )

    def _turns(self, ctx: AnalysisContext, turn_range) -> pd.DataFrame:
        turns = C.prepare_keys(
            ctx.load_table("turns", usecols=TURN_COLUMNS), self.stage_id, "turns", per_player=False,
        )
        turns = C.numeric(turns, ["turn", "is_decision", "is_changed"])
        if turn_range is not None:
            lo, hi = turn_range
            if lo is not None:
                turns = turns[turns["turn"] >= lo]
            if hi is not None:
                turns = turns[turns["turn"] <= hi]
        return turns

    def _join(self, panel: pd.DataFrame, metrics: pd.DataFrame, counts: list[str]) -> pd.DataFrame:
        keep = list(dict.fromkeys([*C.KEY_COLUMNS, self.by, "survival_turn", *counts]))
        keep = [c for c in keep if c in panel.columns]
        return panel[keep].merge(metrics, on=KEY, how="inner")

    def _mix(self, rows: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
        columns = [strategy_column(s) for s in strategies]
        valid = rows.dropna(subset=columns, how="all")
        grouped = valid.groupby(self.by)[columns].mean()
        grouped.columns = strategies
        grouped.insert(0, "n_players", valid.groupby(self.by).size())
        out = grouped.reset_index()
        order = C.order_groups(out.assign(metric=""), self.by)[self.by]
        return out.set_index(self.by).loc[list(order)].reset_index()
