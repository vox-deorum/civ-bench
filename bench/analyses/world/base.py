"""Common scaffolding for the ``world.*`` modules (benchmark.md §6.2).

A world module answers two questions side by side for every metric:

* **Player**: what the strategist's own civ did, against the baseline in the
  same seat on the same map;
* **World**: what happened in the whole game, against the baseline's games on
  the same map (for the in-game AI baseline, the AI-only world).

Every row of the analysis frame is one player in one game. Player values sit in
``p__<metric>`` columns and the game's World values in ``w__<metric>``, copied
onto every player of that game. :func:`bench.analyses.behavior.common.build_views`
matches a ``p__`` column against the baseline pool's ``(seed, player_id)`` cell
mean, seat by seat, and a ``w__`` column (passed as ``per_game``) against the
mean over the pool's distinct games on the same seed, map by map. The
completed-experiment pool holds only strategist seats, so a seat-level match
would compare World values with whichever games put a strategist in that
seat. Each metric becomes one column group with a Player and a World
sub-column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..base import AnalysisContext
from ..behavior import common as C
from ..behavior.base import BehaviorAnalysis, MetricViews
from ..errors import AnalysisError

PLAYER = "p__"
WORLD = "w__"
SCOPES = ((PLAYER, "Player"), (WORLD, "World"))
# The pinned row's name against the in-game AI baseline.
AI_ONLY_WORLD = "AI-only world"

# How a relative cell reads, by the catalog's unit, and the absolute unit.
RELATIVE_UNITS = {"diff": "", "pts": "pts", "pct": "%"}
ABSOLUTE_UNITS = {"diff": "", "pts": "%", "pct": ""}
# Whole numbers, whatever the unit.
WHOLE_METRICS = {"turns", "game_length"}

RELATIVE_LEGEND = [[0.0, "2 SD below"], [0.25, "1 SD below"], [0.5, "same as the baseline"],
                   [0.75, "1 SD above"], [1.0, "2 SD above"]]
ABSOLUTE_LEGEND = [[0.0, "2 SD below the average row"], [0.25, "1 SD below"], [0.5, "average"],
                   [0.75, "1 SD above"], [1.0, "2 SD above"]]


def metric_id(scope: str, metric: str, suffix: str = "") -> str:
    return f"{scope}{metric}{suffix}"


@dataclass
class Columns:
    """Heatmap column settings for a set of paired metric columns."""

    names: dict = field(default_factory=dict)
    short_names: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    column_tips: dict = field(default_factory=dict)
    units: dict = field(default_factory=lambda: {C.RELATIVE: {}, C.ABSOLUTE: {}})
    decimals: dict = field(default_factory=dict)
    percent: set = field(default_factory=set)

    def subset(self, metrics) -> "Columns":
        keep = [m for m in self.names if m in set(metrics)]
        return Columns(
            names={m: self.names[m] for m in keep},
            short_names={m: self.short_names[m] for m in keep},
            groups={m: self.groups[m] for m in keep},
            column_tips={m: self.column_tips[m] for m in keep},
            units={view: {m: u for m, u in units.items() if m in keep}
                   for view, units in self.units.items()},
            decimals={m: self.decimals[m] for m in keep},
            percent={m for m in self.percent if m in keep},
        )


def paired_columns(catalog: dict, metrics, *, suffix: str = "", label: str = "") -> Columns:
    """Player and World columns for each metric in ``metrics``, in that order.

    ``catalog`` maps a metric to ``(name, short, player tip, world tip, player
    unit, world unit)`` (:data:`bench.config.schema.WORLD_CONFLICT_METRICS`).
    ``suffix`` ends each column id (such as ``@200``) and ``label`` each group
    header and name (such as ``T200``).
    """
    out = Columns()
    for metric in metrics:
        name, short, player_tip, world_tip, player_unit, world_unit = catalog[metric]
        group = f"{short} · {label}" if label else short
        full = f"{name} · {label}" if label else name
        for (scope, scope_name), tip, unit in zip(SCOPES, (player_tip, world_tip),
                                                  (player_unit, world_unit)):
            column = metric_id(scope, metric, suffix)
            out.names[column] = f"{full} ({scope_name.lower()})"
            out.short_names[column] = scope_name
            out.groups[column] = group
            out.column_tips[column] = f"{full}, {scope_name.lower()}\n{tip}"
            out.units[C.RELATIVE][column] = RELATIVE_UNITS[unit]
            out.units[C.ABSOLUTE][column] = ABSOLUTE_UNITS[unit]
            out.decimals[column] = 0 if unit != "diff" or metric in WHOLE_METRICS else 1
            if unit == "pct":
                out.percent.add(column)
    return out


def world_columns(columns: Columns) -> frozenset:
    """The World column ids of ``columns``, for ``build_views(per_game=...)``."""
    return frozenset(m for m in columns.names if m.startswith(WORLD))


def merge_columns(*parts: Columns) -> Columns:
    out = Columns()
    for part in parts:
        out.names.update(part.names)
        out.short_names.update(part.short_names)
        out.groups.update(part.groups)
        out.column_tips.update(part.column_tips)
        for view, units in part.units.items():
            out.units.setdefault(view, {}).update(units)
        out.decimals.update(part.decimals)
        out.percent |= part.percent
    return out


class WorldAnalysis(BehaviorAnalysis):
    """Settings and helpers shared by the world modules; subclasses implement ``run``."""

    def table(self, ctx: AnalysisContext, name: str, wanted=None) -> pd.DataFrame:
        """A canonical table, failing with a hint when the run did not extract it."""
        path = (ctx.config.data.get("tables") or {}).get(name)
        if not path:
            raise AnalysisError(
                f"{self.module} '{self.stage_id}': no '{name}' table is configured; add "
                f"\"{name}\" to data.extract.outputs (or data.tables) and re-run extract."
            )
        if wanted is None:
            return ctx.load_table(name)
        wanted = set(wanted)
        return ctx.load_table(name, usecols=lambda c: c in wanted)

    def players(self, ctx: AnalysisContext, extra=()) -> pd.DataFrame:
        """Every civ of every game from the panel, unfiltered: keys, ``by``, and ``extra``."""
        panel = ctx.load_table("panel")
        panel = C.prepare_keys(panel, self.stage_id, "panel")
        C.check_by(panel, self.by, self.stage_id)
        keep = list(dict.fromkeys([*C.KEY_COLUMNS, self.by, *extra]))
        missing = [c for c in keep if c not in panel.columns]
        if missing:
            raise AnalysisError(f"{self.module} '{self.stage_id}': panel lacks column(s) {missing}.")
        return panel[keep]

    def turns(self, ctx: AnalysisContext, columns=()) -> pd.DataFrame:
        """The turns table's keys, ``turn``, ``score``, and ``columns``, as numbers."""
        wanted = [*C.KEY_COLUMNS, "turn", "score", *columns]
        turns = ctx.load_table("turns", usecols=lambda c: c in set(wanted))
        missing = [c for c in wanted if c not in turns.columns]
        if missing:
            raise AnalysisError(f"{self.module} '{self.stage_id}': turns table lacks {missing}.")
        turns = C.prepare_keys(turns, self.stage_id, "turns", per_player=False)
        return C.numeric(turns, ["turn", "score", *columns])

    def frame(self, ctx: AnalysisContext, players: pd.DataFrame, player_values: pd.DataFrame,
              world_values: pd.DataFrame, metrics: list[str]):
        """``(rows, pool, baseline)``: players joined with their own and their game's values.

        ``player_values`` is keyed by ``(game_id, player_id)`` and ``world_values``
        by ``game_id``. Against the in-game AI, Vanilla players leave the rows
        (the pinned row stands for them); the Null strategist stays.
        """
        full = players.merge(player_values, on=["game_id", "player_id"], how="left")
        full = full.merge(world_values, on="game_id", how="left")
        for column in metrics:
            if column not in full.columns:
                full[column] = float("nan")
        full = C.numeric(full, metrics)
        baseline = self.baseline(ctx)
        rows = C.prepare_keys(ctx.apply_filter(full), self.stage_id, "panel")
        if baseline is not None and baseline.in_game_ai:
            rows = rows[rows["player_type"].astype(str) != ctx.catalog.vanilla_label]
        pool = C.baseline_pool(ctx, full, baseline)
        return rows, pool, baseline

    def pinned_label(self, views: C.BehaviorViews) -> str:
        if views.baseline is None:
            return ""
        return AI_ONLY_WORLD if views.baseline.in_game_ai else views.baseline.label

    def baseline_words(self, views: C.BehaviorViews) -> tuple[str, str]:
        """How help texts name the Player and World baselines."""
        if views.baseline is not None and views.baseline.in_game_ai:
            return ("the in-game AI in the same seat on the same map",
                    "the AI-only games on the same map")
        label = views.baseline.label if views.baseline is not None else "baseline"
        return (f"the {label} in the same seat on the same map",
                f"the {label} games on the same map")

    def versus_words(self, views: C.BehaviorViews) -> str:
        """How titles and help texts name the baseline: "the in-game AI"."""
        if views.baseline is not None and views.baseline.in_game_ai:
            return "the in-game AI"
        return f"the {views.baseline.label}" if views.baseline is not None else "the baseline"

    # ── heatmaps ──────────────────────────────────────────────────────────────
    def add_paired_views(self, ctx: AnalysisContext, views: C.BehaviorViews, out: MetricViews,
                         slug: str, columns: Columns, *, titles: dict, help_texts: dict,
                         emit=None, view_names=None) -> None:
        """Relative and absolute heatmaps of paired columns, with units in every cell."""
        companions = {m: {"value_mean": f"{m}__value", "baseline_mean": f"{m}__baseline"}
                      for m in columns.names}
        absolute_units = columns.units[C.ABSOLUTE]
        pinned = self.pinned_label(views)
        # The heatmap capitalizes the pinned row's label.
        pinned_row = pinned[:1].upper() + pinned[1:]
        world_base = self.baseline_words(views)[1]
        self.add_heatmap_views(
            ctx, views, out, slug,
            titles=titles,
            help_texts=help_texts,
            names=columns.names,
            short_names=columns.short_names,
            column_tips=columns.column_tips,
            groups=columns.groups,
            legends={C.RELATIVE: RELATIVE_LEGEND, C.ABSOLUTE: ABSOLUTE_LEGEND},
            value_labels={C.RELATIVE: "Difference", C.ABSOLUTE: "Average"},
            decimals=1,
            row_tips={pinned_row: f"{world_base[:1].upper()}{world_base[1:]}. "
                                  "Actual values, not differences."} if pinned else None,
            baseline_row=True,
            view_names=view_names,
            column_decimals=columns.decimals,
            column_units=columns.units,
            companions={C.RELATIVE: companions},
            tip_rows={C.RELATIVE: [
                {"column": "value_mean", "label": "Strategist average", "units": absolute_units},
                {"column": "baseline_mean", "label": "Baseline average", "units": absolute_units},
            ]},
            emit=emit,
            cell_text=True,
            pinned_label=pinned,
        )

    def add_paired_matched_maps(self, ctx: AnalysisContext, views: C.BehaviorViews,
                                pool: pd.DataFrame, cells, out: MetricViews, slug: str,
                                columns: Columns, *, label: str, tip: str, title: str,
                                help_text: str):
        return self.add_matched_map_tables(
            ctx, views, pool, cells, out, slug,
            label=label, tip=tip, title=title, help_text=help_text,
            names=columns.names,
            short_names=columns.short_names,
            column_tips=columns.column_tips,
            groups=columns.groups,
            legend=ABSOLUTE_LEGEND,
            decimals=1,
            column_decimals=columns.decimals,
            column_units=columns.units[C.ABSOLUTE],
            cell_text=True,
            pinned_label=self.pinned_label(views),
        )

    def world_headline(self, ctx: AnalysisContext, out: MetricViews, views: C.BehaviorViews,
                       slugs, columns: Columns) -> str:
        """The strongest World cell and the strongest Player cell, leaving out Null and Vanilla.

        Cells from fewer than 3 games count only when no other cell has more.
        """
        skip = {ctx.catalog.null_label, ctx.catalog.vanilla_label}
        for view in (C.RELATIVE, C.ABSOLUTE):
            tables = [out.summaries.get((slug, view)) for slug in slugs]
            tables = [t for t in tables if t is not None and not t.empty]
            if not tables:
                continue
            table = pd.concat(tables, ignore_index=True)
            table = table[(table["row_kind"] != "baseline")
                          & ~table[self.by].astype(str).isin(skip)
                          & table["metric"].isin(set(columns.names))]
            table = table.dropna(subset=["mean"])
            strong = table[table["n_games"] >= 3]
            table = strong if not strong.empty else table
            if table.empty:
                continue
            parts = []
            for scope, phrase in ((WORLD, "the whole world"), (PLAYER, "its own civ")):
                scoped = table[table["metric"].str.startswith(scope)]
                if scoped.empty:
                    continue
                best = scoped.loc[(scoped["color_position"] - 0.5).abs().idxmax()]
                name = columns.names[best["metric"]].rsplit(" (", 1)[0]
                text = best.get("cell_text") or C.format_cell(best["mean"], 1)
                parts.append(f"**{best['row_label']}** on **{name[:1].lower() + name[1:]}** for {phrase} (**{text}**)")
            if not parts:
                continue
            if view == C.RELATIVE:
                return (f"Against the {self.pinned_label(views)}, the largest shifts are "
                        + " and ".join(parts) + ".")
            return "The most distinctive values are " + " and ".join(parts) + "."
        return "No game had values for the selected world metrics."
