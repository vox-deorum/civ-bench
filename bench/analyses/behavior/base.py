"""Common scaffolding for the ``behavior.*`` modules.

A module computes one row per player per game with its metric columns, then
summarizes both views with :meth:`BehaviorAnalysis.add_heatmap_views`, which
writes one long table per view plus the ``metadata["heatmaps"]`` layout the
report draws as an HTML heatmap. :meth:`BehaviorAnalysis.add_matched_map_tables`
writes the per-seed and per-seat tables behind a Matched Maps tab.

The declared views render behind the relative/absolute toggle; a module may
declare more views (the diplomacy page's Stance view) through ``view_names``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ...config.behavior import behavior_columns
from ..base import Analysis, AnalysisContext
from ..errors import AnalysisError
from . import common as C


@dataclass
class MetricViews:
    tables: dict = field(default_factory=dict)
    declared: dict = field(default_factory=lambda: {C.RELATIVE: {"tables": [], "figures": []},
                                                    C.ABSOLUTE: {"tables": [], "figures": []}})
    summaries: dict = field(default_factory=dict)
    heatmaps: dict = field(default_factory=dict)

    def add(self, view: str, name: str, table=None) -> None:
        self.declared.setdefault(view, {"tables": [], "figures": []})
        if table is not None:
            self.tables[name] = table
            self.declared[view]["tables"].append(name)


class BehaviorAnalysis(Analysis):
    """Settings shared by every behavior module; subclasses implement ``run``."""

    # ``params.baseline`` when the stage sets none: ``None`` is the in-game AI
    # (the first strength stage's baseline experiment), or ``"completed"``.
    default_baseline: str | None = None

    @property
    def by(self) -> str:
        return str(self.params.get("by", "player_type"))

    @property
    def rate(self) -> str:
        return str(self.params.get("rate", C.DEFAULT_RATE))

    @property
    def bootstrap_n(self) -> int:
        return int(self.params.get("bootstrap_n", C.DEFAULT_BOOTSTRAP_N))

    @property
    def ci_level(self) -> float:
        return float(self.params.get("ci_level", C.DEFAULT_CI_LEVEL))

    # ── inputs ────────────────────────────────────────────────────────────────
    def behavior_kinds(self, ctx: AnalysisContext) -> dict:
        extract = (ctx.config.data.get("extract") or {})
        return behavior_columns(extract.get("behavior"))

    def baseline(self, ctx: AnalysisContext) -> C.Baseline | None:
        return C.resolve_baseline(ctx, self.params, self.default_baseline)

    def load_behavior(self, ctx: AnalysisContext, metrics: list[str], allow_missing: bool = False):
        """Filtered rows and the baseline pool of the behavior table, rates applied.

        Returns ``(rows, pool, rate_scaled, baseline)`` where ``rate_scaled`` names
        the count columns divided by turns alive. With ``allow_missing``, columns
        an older extract lacks are skipped instead of failing the run.
        """
        kinds = self.behavior_kinds(ctx)
        full = ctx.load_table("behavior")
        missing = [m for m in metrics if m not in full.columns]
        if allow_missing:
            metrics, missing = [m for m in metrics if m in full.columns], []
        if missing:
            raise AnalysisError(
                f"analysis '{self.stage_id}': behavior table lacks column(s) {missing}; "
                "re-run extract with the matching data.extract.behavior selection."
            )
        full = C.prepare_keys(full, self.stage_id, "behavior table")
        C.check_by(full, self.by, self.stage_id)
        full = C.numeric(full, metrics)
        rate_scaled = {m for m in metrics if kinds.get(m) == "count"} if self.rate == "per_100_turns" else set()
        if rate_scaled:
            full = C.per_100_turns(full, sorted(rate_scaled))
        rows = C.prepare_keys(ctx.apply_filter(full), self.stage_id, "behavior table")
        baseline = self.baseline(ctx)
        pool = C.baseline_pool(ctx, full, baseline)
        return rows, pool, rate_scaled, baseline

    # ── HTML heatmap views ────────────────────────────────────────────────────
    def add_heatmap_views(
        self,
        ctx: AnalysisContext,
        views: C.BehaviorViews,
        out: MetricViews,
        slug: str,
        *,
        titles: dict,
        help_texts: dict,
        names: dict,
        short_names: dict | None = None,
        column_tips: dict | None = None,
        groups: dict | None = None,
        fixed: dict | None = None,
        legends: dict,
        value_labels: dict,
        decimals: int = 1,
        row_tips: dict | None = None,
        ranges: dict | None = None,
        range_label: str = "",
        baseline_row: bool = False,
        view_names: dict | None = None,
        column_decimals: dict | None = None,
        column_units: dict | None = None,
        companions: dict | None = None,
        tip_rows: dict | None = None,
    ) -> None:
        """Write ``<slug>_relative`` / ``<slug>_absolute`` and their heatmap specs.

        ``names`` maps each metric column to its full name and fixes the column
        order; ``short_names`` and ``column_tips`` give the header text and
        tooltip; ``groups`` names a spanning header per metric. ``fixed`` maps a
        metric to its absolute ``(low, high)`` color range. ``ranges`` maps a
        metric to its per-player ``(min, max)`` columns, averaged into
        ``mean_min`` / ``mean_max`` for the tooltip; in the relative view they
        are differences from the baseline when :func:`C.build_views` shifted
        them (its ``companions``). The per-view dictionaries (``titles``,
        ``help_texts``, ``legends``, ``value_labels``) are keyed by
        :data:`C.RELATIVE` / :data:`C.ABSOLUTE`. ``view_names`` maps a view to
        the page view that shows its table (by default the same name), so a
        table can sit in a view of its own.

        With ``baseline_row``, both views pin a row with the baseline pool's absolute
        mean per metric (``row_kind == "baseline"``), colored like the absolute
        view; it needs a relative view, so an uncontrolled run has none.

        ``column_decimals`` sets a metric's number format. ``column_units``,
        ``companions``, and ``tip_rows`` are keyed by view: a metric's tooltip
        unit, ``{metric: {out_column: source_column}}`` group means of other
        per-player columns (the pinned row takes the pool's), and the tooltip
        lines that show them.
        """
        metrics = list(names)
        per_view = [(C.ABSOLUTE, views.absolute, metrics)]
        if views.relative is not None:
            per_view.insert(0, (C.RELATIVE, views.relative,
                                [m for m in metrics if m in views.relative_metrics]))
        reference = [ctx.catalog.null_label, ctx.catalog.vanilla_label]
        pinned = ""
        if baseline_row and views.relative is not None and views.baseline_summary:
            label = views.baseline.label
            pinned = label[:1].upper() + label[1:]
            reference.insert(0, pinned)
        for view, frame, view_metrics in per_view:
            if not view_metrics:
                continue
            summary = C.summarize(ctx, frame, view_metrics, self.by, view,
                                  self.bootstrap_n, self.ci_level)
            if summary.empty:
                continue
            summary.insert(summary.columns.get_loc("metric") + 1, "metric_group",
                           summary["metric"].map(lambda m: (groups or {}).get(m, "")))
            # A relative range needs min and max shifted by the baseline too.
            view_ranges = ranges if view == C.ABSOLUTE else {
                m: pair for m, pair in (ranges or {}).items() if set(pair) <= views.shifted
            }
            if view_ranges:
                summary = _attach_means(summary, frame, self.by, {
                    m: {"mean_min": low, "mean_max": high} for m, (low, high) in view_ranges.items()
                })
            view_companions = (companions or {}).get(view) or {}
            if view_companions:
                summary = _attach_means(summary, frame, self.by, view_companions)
            summary["color_position"] = C.color_positions(summary, view, views, fixed)
            table, order = C.heatmap_rows(ctx, summary, self.by)
            table.insert(0, "row_kind", "group")
            if pinned:
                table = _with_baseline_row(table, views, view_metrics, pinned, self.by, groups, fixed,
                                           view_companions)
                order = [pinned] + order
            leading = ["row_kind", self.by, "strategist", "condition", "row_label"]
            table = table[leading + [c for c in table.columns if c not in leading]]
            name = f"{slug}_{view}"
            out.add((view_names or {}).get(view, view), name, table=table)
            out.summaries[(slug, view)] = table
            present = set(table["row_label"])
            out.heatmaps[name] = C.heatmap_spec(
                view=view,
                title=titles[view],
                help_text=help_texts[view],
                row_order=order,
                reference_rows=[r for r in reference if r in present],
                row_tips={k: v for k, v in (row_tips or {}).items() if k in present},
                column_order=view_metrics,
                column_names={m: names[m] for m in view_metrics},
                column_labels={m: (short_names or names)[m] for m in view_metrics},
                column_tips={m: (column_tips or {}).get(m, names[m]) for m in view_metrics},
                grouped=bool(groups),
                decimals=decimals,
                value_label=value_labels[view],
                ci_level=self.ci_level,
                legend=legends[view],
                range_label=range_label if view_ranges else "",
                range_note="mean min to max, vs. baseline" if view == C.RELATIVE else "",
                baseline_rows=[pinned] if pinned else None,
                column_decimals=column_decimals,
                column_units=(column_units or {}).get(view),
                tip_rows=(tip_rows or {}).get(view),
                baseline_units=(column_units or {}).get(C.ABSOLUTE, {}) if column_units else None,
            )

    def add_matched_map_tables(
        self,
        ctx: AnalysisContext,
        views: C.BehaviorViews,
        pool: pd.DataFrame,
        cells: pd.DataFrame | None,
        out: MetricViews,
        slug: str,
        *,
        label: str,
        tip: str,
        title: str,
        help_text: str,
        names: dict,
        short_names: dict | None = None,
        column_tips: dict | None = None,
        groups: dict | None = None,
        fixed: dict | None = None,
        legend: list,
        decimals: int = 1,
        column_decimals: dict | None = None,
        column_units: dict | None = None,
        companions: dict | None = None,
        tip_rows: list | None = None,
        key: str = "",
        value_label: str = "Average",
    ) -> dict | None:
        """Write ``<slug>_by_seed`` / ``<slug>_by_seat`` for a Matched Maps tab.

        Both are absolute heatmap tables over controlled games, one keyed by
        ``seed`` and one by ``seed`` and ``player_id``, each pinning the
        baseline pool's average for that seed or seat as its top row. Vanilla
        players are left out: the chapter's pinned row is the reference. A cell's
        ``difference`` is the mean relative value there (the player minus the
        baseline at its own seed and seat), shown in the tooltip. No bootstrap:
        the Matched Maps tables report plain means. ``companions`` and
        ``tip_rows`` add per-cell means of other columns to the tooltip, as in
        :meth:`add_heatmap_views`; ``key`` names the tab when a module offers
        more than one. Returns the ``metadata["matched_maps"]`` entry, or
        ``None`` without a relative view.
        """
        if views.relative is None or cells is None or cells.empty or not views.baseline_summary:
            return None
        metrics = [m for m in names if m in views.relative_metrics]
        if not metrics:
            return None
        label_text = views.baseline.label
        pinned = label_text[:1].upper() + label_text[1:]
        rows = views.absolute[views.absolute["player_type"].astype(str) != ctx.catalog.vanilla_label]
        rows = rows.merge(cells, on="game_id", how="inner")
        base = pool.merge(cells, on="game_id", how="inner")
        declared = {"label": label, "tip": tip, **({"key": key} if key else {})}
        for keys, name, entry in ((["seed"], f"{slug}_by_seed", "seed_table"),
                                  (["seed", "player_id"], f"{slug}_by_seat", "seat_table")):
            table, order = self._matched_map_table(ctx, rows, base, views, metrics, keys,
                                                   pinned, groups, fixed, companions or {})
            if table.empty:
                return None
            out.tables[name] = table
            out.heatmaps[name] = C.heatmap_spec(
                view=C.ABSOLUTE,
                title=title,
                help_text=help_text,
                row_order=[pinned] + order,
                reference_rows=[pinned],
                row_tips={},
                column_order=metrics,
                column_names={m: names[m] for m in metrics},
                column_labels={m: (short_names or names)[m] for m in metrics},
                column_tips={m: (column_tips or {}).get(m, names[m]) for m in metrics},
                grouped=bool(groups),
                decimals=decimals,
                value_label=value_label,
                ci_level=self.ci_level,
                legend=legend,
                baseline_rows=[pinned],
                column_decimals=column_decimals,
                column_units=column_units,
                # A difference of percents is in percentage points.
                tip_rows=[{"column": "difference", "label": "Vs. baseline", "signed": True,
                           "units": {m: "pts" if u == "%" else u
                                     for m, u in (column_units or {}).items()}},
                          *(tip_rows or [])],
            )
            declared[entry] = name
        return declared

    def _matched_map_table(self, ctx, rows, base, views, metrics, keys, pinned, groups, fixed,
                           companions):
        """One long absolute table per ``keys`` cell, the baseline row first."""
        by = self.by
        difference = views.relative.groupby(keys + [by])[metrics].mean()
        records = []
        for values, grp in rows.groupby(keys + [by], sort=True):
            cell, group = tuple(values[:-1]), values[-1]
            for metric in metrics:
                col = grp[["game_id", metric]].dropna()
                if col.empty:
                    continue
                diff = difference[metric].get(tuple(values), float("nan"))
                records.append({**dict(zip(keys, cell)), by: group, "metric": metric,
                                "mean": float(col[metric].mean()), "difference": float(diff),
                                **_companion_means(grp, companions.get(metric)),
                                "n_players": int(len(col)), "n_games": int(col["game_id"].nunique())})
        if not records:
            return pd.DataFrame(), []
        summary = pd.DataFrame(records)
        summary, order = C.heatmap_rows(ctx, summary, by)
        summary.insert(0, "row_kind", "group")
        baseline = []
        for values, grp in base.groupby(keys, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            for metric in metrics:
                col = grp[["game_id", metric]].dropna()
                if col.empty:
                    continue
                baseline.append({"row_kind": "baseline", **dict(zip(keys, values)), by: pinned,
                                 "strategist": pinned, "condition": "", "row_label": pinned,
                                 "metric": metric, "mean": float(col[metric].mean()),
                                 "difference": float("nan"),
                                 **_companion_means(grp, companions.get(metric)),
                                 "n_players": int(len(col)),
                                 "n_games": int(col["game_id"].nunique())})
        table = pd.concat([pd.DataFrame(baseline), summary], ignore_index=True) if baseline else summary
        table["metric_group"] = table["metric"].map(lambda m: (groups or {}).get(m, ""))
        table["color_position"] = C.color_positions(table, C.ABSOLUTE, views, fixed)
        # Deterministic order: cell keys, baseline row first, rows, then metrics.
        rank = {label: i for i, label in enumerate([pinned] + order)}
        metric_rank = {m: i for i, m in enumerate(metrics)}
        table = table.assign(
            _row=table["row_label"].map(rank), _metric=table["metric"].map(metric_rank),
        ).sort_values(keys + ["_row", "_metric"], kind="mergesort").drop(columns=["_row", "_metric"])
        leading = ["row_kind", *keys, by, "strategist", "condition", "row_label"]
        table = table[leading + [c for c in table.columns if c not in leading]]
        return table.reset_index(drop=True), order

    def heatmap_headline(self, ctx: AnalysisContext, out: MetricViews, views: C.BehaviorViews,
                         slug: str, names: dict, decimals: int = 1,
                         column_decimals: dict | None = None) -> str:
        """One sentence on the strongest-colored cell, leaving out Null and Vanilla rows."""
        skip = {ctx.catalog.null_label, ctx.catalog.vanilla_label}
        for view in (C.RELATIVE, C.ABSOLUTE):
            table = out.summaries.get((slug, view))
            if table is None or table.empty:
                continue
            table = table[~table[self.by].astype(str).isin(skip)]
            if "row_kind" in table.columns:
                table = table[table["row_kind"] != "baseline"]
            strong = table[table["n_players"] >= 3]
            table = strong if not strong.empty else table
            if table.empty:
                continue
            score = (table["color_position"] - 0.5).abs()
            best = table.loc[score.idxmax()]
            row, name = best["row_label"], names[best["metric"]]
            places = int((column_decimals or {}).get(best["metric"], decimals))
            if view == C.RELATIVE:
                return (
                    f"Against the {views.baseline.label} on the same map and seat, the largest "
                    f"departure is **{row}** on {name} (**{best['mean']:+.{places}f}**), across "
                    f"**{len(views.relative)}** controlled players."
                )
            return (
                f"The most distinctive value is **{row}** on {name} "
                f"(**{best['mean']:.{places}f}**), across **{len(views.absolute)}** players in "
                f"**{views.absolute['game_id'].nunique()}** games."
            )
        return "No player had values for the selected behavior metrics."


def _with_baseline_row(table: pd.DataFrame, views: C.BehaviorViews, metrics: list[str],
                       label: str, by: str, groups: dict | None, fixed: dict | None,
                       companions: dict | None = None) -> pd.DataFrame:
    """``table`` with the baseline pool's absolute means prepended as one row."""
    pool = views.pool if views.pool is not None else pd.DataFrame()
    records = []
    for metric in metrics:
        stats = views.baseline_summary.get(metric)
        if stats is None:
            continue
        records.append({
            "row_kind": "baseline", by: label, "strategist": label, "condition": "",
            "row_label": label, "metric": metric, "metric_group": (groups or {}).get(metric, ""),
            "mean": stats["mean"], "sd": stats["sd"],
            **_companion_means(pool, (companions or {}).get(metric)),
            "n_players": stats["n_players"], "n_games": stats["n_games"],
        })
    if not records:
        return table
    rows = pd.DataFrame(records)
    rows["color_position"] = C.color_positions(rows, C.ABSOLUTE, views, fixed)
    return pd.concat([rows, table], ignore_index=True)


def _attach_means(summary: pd.DataFrame, rows: pd.DataFrame, by: str, companions: dict) -> pd.DataFrame:
    """Per-group means of other per-player columns, keyed ``{metric: {out: source}}``.

    Each metric's cells get its ``out`` columns; other cells read NaN there.
    """
    columns = sorted({column for pairs in companions.values() for column in pairs})
    groups = rows[by].astype(str)
    means = {}
    for metric, pairs in companions.items():
        sources = sorted({src for src in pairs.values() if src in rows.columns})
        if sources:
            means[metric] = rows.groupby(groups)[sources].mean()
    values = {column: [] for column in columns}
    for rec in summary.itertuples(index=False):
        frame = means.get(rec.metric)
        pairs = companions.get(rec.metric, {})
        group = str(getattr(rec, by))
        for column in columns:
            src = pairs.get(column)
            found = frame is not None and src in frame.columns and group in frame.index
            values[column].append(float(frame.loc[group, src]) if found else float("nan"))
    out = summary.copy()
    for column in columns:
        out[column] = values[column]
    return out


def _companion_means(rows: pd.DataFrame, pairs: dict | None) -> dict:
    """``{out: mean of rows[source]}`` for one cell; NaN for a missing column."""
    return {
        column: float(rows[src].mean()) if src in rows.columns and not rows.empty else float("nan")
        for column, src in (pairs or {}).items()
    }
