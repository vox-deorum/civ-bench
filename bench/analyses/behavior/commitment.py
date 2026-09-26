"""``behavior.commitment``: how a strategist steers, and which grand strategy it holds.

From the per-turn table, per player per game:

* ``acts_pct``: share of turns the strategist made a decision (``is_decision``);
* ``revise_pct``: share of decisions that changed a flavor value (``is_changed``);
* ``flavors_touched`` / ``step_size`` / ``flavor_shift``: per revision (a turn
  whose flavor values differ from the previous turn's), how many flavors moved,
  how far one moved on average, and the total movement;
* ``net_direction_pct``: the distance from each flavor's first to its last
  setting as a share of all movement (100 means every change pushed one way);
* ``grand_strategy_share_<name>``, ``main_strategy_share``,
  ``grand_strategy_switches`` (per 100 recorded turns), and
  ``first_pivot_turn`` (the first turn off the opening strategy).

The panel adds real persona changes, per 100 turns alive by default (``rate``).

The Relative and Absolute views are HTML heatmaps of the steering columns
against the completed-experiment average at the same map and seat, pinned as
the top row; the in-game AI and the Null strategist have no strategist, so they
are left out. The **Grand strategy** view is laid out like the Matched Maps
Focus tab: the main grand strategy in its victory color at share intensity, the
share of each strategy, switches, and the first pivot. Its reference row is the
in-game AI in its own games (the first strength stage's ``baseline_experiment``).

In controlled runs the module offers two Matched Maps tabs: Commitment
(``commitment_by_seed`` / ``commitment_by_seat``) and Grand strategy
(``grand_strategy_by_seed``, one Main cell per seat, and ``grand_strategy_by_seat``).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ...config import schema as S
from ...config.behavior import snake_case
from ...plotting.styles import VICTORY_COLORS
from ..base import AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from . import common as C
from .base import BehaviorAnalysis, MetricViews

TURN_COLUMNS = ["experiment", "game_id", "player_id", "player_type", "turn",
                "is_decision", "is_changed", "grand_strategy"]
FLAVOR_COLUMNS = [f"flavor_{snake_case(name)}" for name in S.FLAVOR_NAMES]
KEY = ["game_id", "player_id"]
GRAND_STRATEGY_VIEW = "grand_strategy"

# Steering column → (full name, short header, description, decimals, unit).
STEERING = {
    "acts_pct": (
        "Decision turns", "Acts %",
        "Share of turns on which the strategist made a decision, including decisions that "
        "kept every setting. Mostly set by how often the strategist is consulted (every "
        "turn or every 5 turns).", 0, "%"),
    "revise_pct": (
        "Decisions that change flavors", "Revises %",
        "Share of the strategist's decisions that changed at least one flavor value. The "
        "rest kept the current settings.", 0, "%"),
    "flavors_touched": (
        "Flavors changed per revision", "Touched",
        "How many flavors one revision changes, on average.", 1, "flavors"),
    "step_size": (
        "Size of one flavor change", "Step",
        "How far one changed flavor moves, on average, in points on the 0 to 100 flavor "
        "scale.", 1, "points"),
    "net_direction_pct": (
        "Changes that add up", "Net %",
        "The distance from each flavor's first to its last setting, as a share of all "
        "flavor movement. 100% means every change pushed the same way; a low value means "
        "later changes undid earlier ones.", 0, "%"),
    "persona_changes": (
        "Persona changes", "Persona",
        "How often the strategist changed its diplomatic persona, {per}.", 1, ""),
}
Z_LEGEND = [[0.0, "2 SD below the average strategist"], [0.25, "1 SD below"], [0.5, "average"],
            [0.75, "1 SD above"], [1.0, "2 SD above"]]
RELATIVE_LEGEND = [[0.0, "2 SD below"], [0.25, "1 SD below"], [0.5, "same as the baseline"],
                   [0.75, "1 SD above"], [1.0, "2 SD above"]]
SHIFT_TIP = {"column": "shift", "label": "Total change per revision", "unit": "points", "decimals": 1}


def strategy_column(name: str) -> str:
    return "grand_strategy_share_" + re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def strategy_order(names) -> list[str]:
    """Known grand strategies in schema order, then any others alphabetically."""
    names = {str(n) for n in names if str(n) != ""}
    return [s for s in S.GRAND_STRATEGY_INFO if s in names] + sorted(names - set(S.GRAND_STRATEGY_INFO))


def turn_metrics(turns: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Per ``(game_id, player_id)`` steering and grand-strategy metrics from per-turn rows."""
    turns = turns.sort_values(KEY + ["turn"], kind="stable").reset_index(drop=True)
    grouped = turns.groupby(KEY, sort=True)
    decisions = grouped["is_decision"].sum()
    out = pd.DataFrame({
        "n_turns": grouped.size(),
        "acts_pct": grouped["is_decision"].mean() * 100.0,
        "revise_pct": (grouped["is_changed"].sum() / decisions.where(decisions > 0)) * 100.0,
    })
    out = out.join(_change_sizes(turns))
    out = out.join(_grand_strategy(turns, strategies))
    return out.reset_index()


def _change_sizes(turns: pd.DataFrame) -> pd.DataFrame:
    """Touched, step, shift, and net direction from consecutive flavor settings."""
    flavors = [c for c in FLAVOR_COLUMNS if c in turns.columns]
    index = pd.MultiIndex.from_frame(turns[KEY].drop_duplicates())
    empty = pd.DataFrame(np.nan, index=index,
                         columns=["flavors_touched", "step_size", "flavor_shift", "net_direction_pct"])
    if not flavors or turns.empty:
        return empty
    values = turns[flavors].to_numpy(dtype=np.float32)
    same = ((turns["game_id"].to_numpy() == np.roll(turns["game_id"].to_numpy(), 1))
            & (turns["player_id"].to_numpy() == np.roll(turns["player_id"].to_numpy(), 1)))
    same[0] = False
    delta = np.abs(values - np.roll(values, 1, axis=0))
    delta[~same] = np.nan
    moved = np.nan_to_num(delta) > 0
    frame = turns[KEY].copy()
    frame["_touched"] = moved.sum(axis=1)
    frame["_moved"] = np.where(moved, delta, 0.0).sum(axis=1)
    events = frame[frame["_touched"] > 0].groupby(KEY, sort=True)
    n_events = events.size()
    touched = events["_touched"].sum()
    moved_total = events["_moved"].sum()
    by_player = turns.groupby(KEY, sort=True)[flavors]
    net = (by_player.last() - by_player.first()).abs().sum(axis=1)
    result = pd.DataFrame({
        "flavors_touched": touched / n_events,
        "step_size": moved_total / touched,
        "flavor_shift": moved_total / n_events,
        "net_direction_pct": net.reindex(moved_total.index) / moved_total * 100.0,
    })
    return empty[[]].join(result)


def _grand_strategy(turns: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Per-strategy shares, the main share, switches, and the first pivot, in percent."""
    columns = ["grand_strategy_switches", "main_strategy_share", "first_pivot_turn",
               *[strategy_column(s) for s in strategies]]
    gs = turns[turns["grand_strategy"].notna() & (turns["grand_strategy"].astype(str) != "")]
    index = pd.MultiIndex.from_frame(turns[KEY].drop_duplicates())
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    if gs.empty:
        return out
    same_player = (gs["game_id"] == gs["game_id"].shift()) & (gs["player_id"] == gs["player_id"].shift())
    switched = same_player & (gs["grand_strategy"] != gs["grand_strategy"].shift())
    gs_grouped = gs.assign(_switch=switched.astype(int)).groupby(KEY, sort=True)
    n_gs = gs_grouped.size()
    found = pd.DataFrame(index=n_gs.index)
    found["grand_strategy_switches"] = gs_grouped["_switch"].sum() / n_gs * 100.0
    counts = gs.groupby(KEY + ["grand_strategy"], sort=True).size().unstack(fill_value=0)
    found["main_strategy_share"] = counts.max(axis=1) / n_gs * 100.0
    for strategy in strategies:
        column = counts[strategy] if strategy in counts.columns else 0
        found[strategy_column(strategy)] = column / n_gs * 100.0
    opening = gs_grouped["grand_strategy"].first().rename("_opening")
    off = gs.join(opening, on=KEY)
    off = off[off["grand_strategy"] != off["_opening"]]
    found["first_pivot_turn"] = off.groupby(KEY, sort=True)["turn"].min()
    return out[[]].join(found[columns])


class BehaviorCommitment(BehaviorAnalysis):
    module = "behavior.commitment"
    friendly_name = "Strategic commitment"
    description = (
        "Shows how often strategists act and revise their settings, how large and how "
        "lasting their changes are, and which grand strategy they hold."
    )
    default_baseline = C.COMPLETED
    report_defaults = {
        "tables": ["commitment_relative", "commitment_absolute", "grand_strategy"],
        "figures": [],
    }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        turn_range = ctx.resolved_filter().get("turn_range")
        full_turns = self._turns(ctx, turn_range)
        full_panel = C.prepare_keys(ctx.load_table("panel"), self.stage_id, "panel")
        C.check_by(full_panel, self.by, self.stage_id)
        counts = [c for c in ["persona_changes"] if c in full_panel.columns]
        full_panel = C.numeric(full_panel, counts)
        rate_scaled = set(counts) if self.rate == "per_100_turns" else set()
        if rate_scaled:
            full_panel = C.per_100_turns(full_panel, sorted(rate_scaled))

        strategies = strategy_order(full_turns["grand_strategy"].dropna().unique())
        rows = self._join(C.prepare_keys(ctx.apply_filter(full_panel), self.stage_id, "panel"),
                          turn_metrics(ctx.apply_filter(full_turns), strategies), counts)
        if rows.empty:
            return AnalysisResult(summary="No players remain after filtering.")

        baseline = self.baseline(ctx)
        pool = pd.DataFrame(columns=rows.columns)
        if baseline:
            pool = self._join(C.baseline_pool(ctx, full_panel, baseline),
                              turn_metrics(C.baseline_pool(ctx, full_turns, baseline), strategies),
                              counts)
        cells = C.controlled_cells(ctx)
        out = MetricViews()
        steering = [m for m in STEERING if m in rows.columns]
        skip = {ctx.catalog.null_label, ctx.catalog.vanilla_label}
        strategists = rows[~rows["player_type"].astype(str).isin(skip)]
        views = C.build_views(strategists, pool, steering, cells, baseline)
        per = "per 100 turns alive" if rate_scaled else "per game"
        info = {m: (full, short, text.format(per=per), places, unit)
                for m, (full, short, text, places, unit) in STEERING.items() if m in steering}
        tabs = []
        if not views.absolute.empty:
            self._steering_views(ctx, views, out, info)
            tab = self._steering_tab(ctx, views, pool, cells, out, info)
            if tab is not None:
                tabs.append(tab)

        reference = self._reference_rows(ctx, rows)
        tab = self._grand_strategy(ctx, reference, strategies, cells, out)
        if tab is not None:
            tabs.append(tab)

        names = {m: v[0] for m, v in info.items()}
        metadata = {
            **views.metadata(),
            "grand_strategies": strategies,
            "rate": self.rate,
            "views": C.views_metadata(out.declared, views.baseline, extra={
                GRAND_STRATEGY_VIEW: ("Grand strategy", "Main grand strategy and its share of turns"),
            }),
            "heatmaps": out.heatmaps,
        }
        if tabs:
            metadata["matched_maps"] = tabs
        if turn_range is not None:
            metadata["turn_range"] = list(turn_range)
        victory_metrics = {
            S.GRAND_STRATEGY_INFO[s][1]: strategy_column(s)
            for s in strategies if s in S.GRAND_STRATEGY_INFO
        }
        summary = self.top_condition_summary(
            out.tables.get("grand_strategy"), victory_metrics,
            {ctx.catalog.null_label, ctx.catalog.vanilla_label},
        )
        return AnalysisResult(
            tables=out.tables,
            summary=summary or "No strategist had grand-strategy values.",
            metadata=metadata,
        )

    # ── inputs ────────────────────────────────────────────────────────────────
    def _turns(self, ctx: AnalysisContext, turn_range) -> pd.DataFrame:
        wanted = set(TURN_COLUMNS) | set(FLAVOR_COLUMNS)
        # The flavor columns are wide; float32 halves their memory.
        turns = ctx.load_table("turns", usecols=lambda c: c in wanted,
                               dtype={c: "float32" for c in FLAVOR_COLUMNS})
        missing = [c for c in TURN_COLUMNS if c not in turns.columns]
        if missing:
            raise AnalysisError(f"behavior.commitment '{self.stage_id}': turns table lacks {missing}.")
        turns = C.prepare_keys(turns, self.stage_id, "turns", per_player=False)
        flavors = [c for c in FLAVOR_COLUMNS if c in turns.columns]
        turns = C.numeric(turns, ["turn", "is_decision", "is_changed", *flavors])
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

    def _reference_rows(self, ctx: AnalysisContext, rows: pd.DataFrame) -> pd.DataFrame:
        """Rows for the grand-strategy tables: the in-game AI only in its own games."""
        own = C.strength_baseline_experiment(ctx)
        if own is None:
            return rows
        vanilla = rows["player_type"].astype(str) == ctx.catalog.vanilla_label
        return rows[~vanilla | (rows["experiment"] == own)]

    # ── steering heatmaps ─────────────────────────────────────────────────────
    def _steering_views(self, ctx, views, out, info) -> None:
        label = views.baseline.label if views.baseline is not None else "baseline"
        note = (" The in-game AI and the Null strategist have no strategist, so they are "
                "left out.")
        if views.relative is not None and views.baseline_summary:
            note += f" The top row is the {label} itself, in absolute terms."
        units = {m: v[4] for m, v in info.items()}
        self.add_heatmap_views(
            ctx, views, out, "commitment",
            titles={C.RELATIVE: "Relative commitment", C.ABSOLUTE: "Commitment"},
            help_texts={
                C.RELATIVE: (
                    "Each cell is the mean, over players, of the player's value minus the "
                    f"{label} at the same map and seat. Percent columns differ in percentage "
                    "points. Colors run from 2 SDs of the baseline players below (red) to 2 SDs "
                    "above (blue). Completed experiments are part of their own baseline. Hover a "
                    f"column heading for its definition.{note}"
                ),
                C.ABSOLUTE: (
                    "Each cell is the mean, over players, of the player's value over the game. "
                    "A revision is a turn whose flavor settings differ from the previous turn's. "
                    "Colors show each cell as a z-score across strategists, full color at 2 SDs. "
                    f"Hover a column heading for its definition.{note}"
                ),
            },
            names={m: v[0] for m, v in info.items()},
            short_names={m: v[1] for m, v in info.items()},
            column_tips={m: f"{v[0]}\n{v[2]}" for m, v in info.items()},
            legends={C.RELATIVE: RELATIVE_LEGEND, C.ABSOLUTE: Z_LEGEND},
            value_labels={C.RELATIVE: "Difference", C.ABSOLUTE: "Average"},
            column_decimals={m: v[3] for m, v in info.items()},
            column_units={
                C.ABSOLUTE: units,
                C.RELATIVE: {m: ("pts" if u == "%" else u) for m, u in units.items()},
            },
            companions={C.ABSOLUTE: {m: {"shift": "flavor_shift"}
                                     for m in ("flavors_touched", "step_size") if m in info}},
            tip_rows={C.ABSOLUTE: [SHIFT_TIP]},
            baseline_row=True,
        )

    def _steering_tab(self, ctx, views, pool, cells, out, info):
        label = views.baseline.label if views.baseline is not None else "baseline"
        return self.add_matched_map_tables(
            ctx, views, pool, cells, out, "commitment",
            label="Commitment",
            tip="Strategic commitment: how often and how much strategists change their settings",
            title="Commitment",
            help_text=(
                "Each cell is the mean, over players on this map (and seat), of the player's "
                f"value over the game. The top row is the {label} on the same map (and seat); "
                "the tooltip shows each cell's difference from it, matched seat by seat. Colors "
                "show each cell as a z-score across strategists, full color at 2 SDs. Hover a "
                "column heading for its definition."
            ),
            names={m: v[0] for m, v in info.items()},
            short_names={m: v[1] for m, v in info.items()},
            column_tips={m: f"{v[0]}\n{v[2]}" for m, v in info.items()},
            legend=Z_LEGEND,
            column_decimals={m: v[3] for m, v in info.items()},
            column_units={m: v[4] for m, v in info.items()},
            companions={m: {"shift": "flavor_shift"}
                        for m in ("flavors_touched", "step_size") if m in info},
            tip_rows=[SHIFT_TIP],
            key="commitment",
        )

    # ── grand strategy ────────────────────────────────────────────────────────
    def _grand_strategy(self, ctx, rows, strategies, cells, out):
        """Write the Grand strategy view and, in controlled runs, its tab tables."""
        if not strategies:
            return None
        columns = self._gs_columns(strategies)
        own = C.strength_baseline_experiment(ctx)
        row_tips = {}
        if own is not None:
            row_tips[ctx.catalog.vanilla_label] = (
                f"The in-game AI in its own games ('{own}'), not as an opponent of a strategist."
            )
        table, order = self._gs_table(ctx, rows, strategies, [], list(columns))
        if table.empty:
            return None
        out.add(GRAND_STRATEGY_VIEW, "grand_strategy", table=table)
        out.heatmaps["grand_strategy"] = self._gs_spec(
            ctx, table, order, columns, strategies, row_tips,
            title="Grand strategy",
            help_text=(
                "Main is the grand strategy with the largest average share of turns, in its "
                "victory color at share intensity. Each share column is colored the same way. "
                "Switches and Pivot are plain numbers. Hover a column heading for its definition."
            ),
        )
        if cells is None or cells.empty:
            return None
        placed = rows.merge(cells, on="game_id", how="inner")
        seed_table, seed_order = self._gs_table(ctx, placed, strategies, ["seed", "player_id"], ["gs_main"])
        seat_table, seat_order = self._gs_table(ctx, placed, strategies, ["seed", "player_id"],
                                                list(columns))
        if seed_table.empty or seat_table.empty:
            return None
        seed_table.insert(seed_table.columns.get_loc("metric"), "seat",
                          seed_table["player_id"].astype(int).astype(str))
        out.tables["grand_strategy_by_seed"] = seed_table
        out.tables["grand_strategy_by_seat"] = seat_table
        out.heatmaps["grand_strategy_by_seed"] = {
            **self._gs_spec(
                ctx, seed_table, seed_order, {"gs_main": columns["gs_main"]}, strategies, row_tips,
                title="Main grand strategy",
                help_text=(
                    "The grand strategy with the largest average share of turns for each "
                    "strategist and seat, in its victory color at share intensity. Hover a cell "
                    "for switches and the first pivot."
                ),
            ),
            "column": "seat",
            "seat_columns": True,
            "unit": "%",
        }
        out.heatmaps["grand_strategy_by_seat"] = self._gs_spec(
            ctx, seat_table, seat_order, columns, strategies, row_tips,
            title="Grand strategy",
            help_text=(
                "Main is the grand strategy with the largest average share of turns on this "
                "seat, in its victory color at share intensity; each share column is colored "
                "the same way. Hover a column heading for its definition."
            ),
        )
        return {"label": "Grand strategy", "tip": "Main grand strategy and its share of turns",
                "key": "grand-strategy", "seed_table": "grand_strategy_by_seed",
                "seat_table": "grand_strategy_by_seat"}

    def _gs_columns(self, strategies) -> dict:
        """Grand-strategy column → (full name, short header, description, decimals, unit)."""
        columns = {"gs_main": ("Main grand strategy", "Main",
                               "The grand strategy with the largest average share of turns.", 0, "%")}
        for s in strategies:
            name = S.GRAND_STRATEGY_INFO.get(s, (s, ""))[0]
            short = "UN" if s == "UnitedNations" else name
            columns[strategy_column(s)] = (f"{name} share", f"{short} %",
                                           f"Share of turns spent in the {name} grand strategy.", 0, "%")
        columns["gs_switches"] = ("Grand-strategy switches", "Switches",
                                  "How often the grand strategy changes, per 100 recorded turns.",
                                  2, "per 100 turns")
        columns["gs_pivot"] = ("First pivot turn", "Pivot",
                               "The first turn the grand strategy differs from the opening one, "
                               "averaged over players who ever change it.", 0, "")
        return columns

    def _gs_table(self, ctx, rows, strategies, keys, metrics) -> tuple[pd.DataFrame, list[str]]:
        """One long table of grand-strategy cells per ``keys`` cell and group."""
        by = self.by
        shares = [strategy_column(s) for s in strategies]
        records = []
        for values, grp in rows.groupby(keys + [by], sort=True):
            values = values if isinstance(values, tuple) else (values,)
            grp = grp.dropna(subset=shares, how="all")
            if grp.empty:
                continue
            base = {**dict(zip(keys, values[:-1])), by: values[-1],
                    "n_players": int(len(grp)), "n_games": int(grp["game_id"].nunique())}
            mean = grp[shares].mean()
            best = max(range(len(strategies)), key=lambda i: (mean.iloc[i], -i))
            main, main_share = strategies[best], float(mean.iloc[best])
            name = S.GRAND_STRATEGY_INFO.get(main, (main, ""))[0]
            extras = {"held": float(grp["main_strategy_share"].mean()),
                      "switches": float(grp["grand_strategy_switches"].mean()),
                      "pivot": float(grp["first_pivot_turn"].mean())}
            for metric in metrics:
                record = {**base, "metric": metric, "category": "", "strategy": "", "text": "",
                          "color_position": float("nan")}
                if metric == "gs_main":
                    record.update(mean=main_share, category=main, strategy=name,
                                  text=f"{name} {main_share:.0f}%",
                                  color_position=main_share / 100.0, **extras)
                elif metric == "gs_switches":
                    record["mean"] = extras["switches"]
                elif metric == "gs_pivot":
                    record["mean"] = extras["pivot"]
                else:
                    strategy = strategies[shares.index(metric)]
                    share = float(mean[metric])
                    record.update(mean=share, category=strategy, color_position=share / 100.0)
                records.append(record)
        if not records:
            return pd.DataFrame(), []
        table, order = C.heatmap_rows(ctx, pd.DataFrame(records), by)
        table.insert(0, "row_kind", "group")
        rank = {label: i for i, label in enumerate(order)}
        metric_rank = {m: i for i, m in enumerate(metrics)}
        table = table.assign(
            _row=table["row_label"].map(rank), _metric=table["metric"].map(metric_rank),
        ).sort_values(keys + ["_row", "_metric"], kind="mergesort").drop(columns=["_row", "_metric"])
        leading = ["row_kind", *keys, by, "strategist", "condition", "row_label"]
        table = table[leading + [c for c in table.columns if c not in leading]]
        return table.reset_index(drop=True), order

    def _gs_spec(self, ctx, table, order, columns, strategies, row_tips, *, title, help_text) -> dict:
        present = set(table["row_label"])
        reference = [r for r in (ctx.catalog.vanilla_label, ctx.catalog.null_label) if r in present]
        spec = C.heatmap_spec(
            view=C.ABSOLUTE,
            title=title,
            help_text=help_text,
            row_order=order,
            reference_rows=reference,
            row_tips={k: v for k, v in row_tips.items() if k in present},
            column_order=list(columns),
            column_names={m: v[0] for m, v in columns.items()},
            column_labels={m: v[1] for m, v in columns.items()},
            column_tips={m: f"{v[0]}\n{v[2]}" for m, v in columns.items()},
            grouped=False,
            decimals=0,
            value_label="Share of turns",
            ci_level=self.ci_level,
            legend=[[VICTORY_COLORS[S.GRAND_STRATEGY_INFO[s][1]], S.GRAND_STRATEGY_INFO[s][0]]
                    for s in strategies if s in S.GRAND_STRATEGY_INFO],
            column_decimals={m: v[3] for m, v in columns.items()},
            column_units={m: v[4] for m, v in columns.items()},
            tip_rows=[
                {"column": "strategy", "label": "Strategy", "text": True},
                {"column": "held", "label": "Time in each player's most-used strategy",
                 "unit": "%", "decimals": 0},
                {"column": "switches", "label": "Switches", "unit": "per 100 turns", "decimals": 2},
                {"column": "pivot", "label": "First pivot turn", "decimals": 0},
            ],
        )
        spec.update({
            "text_column": "text",
            "category_column": "category",
            "category_colors": {s: VICTORY_COLORS[S.GRAND_STRATEGY_INFO[s][1]]
                                for s in strategies if s in S.GRAND_STRATEGY_INFO},
            "column_value_labels": {"gs_switches": "Switches", "gs_pivot": "Turn"},
        })
        return spec
