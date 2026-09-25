"""Common scaffolding for the ``behavior.*`` modules.

A module computes one row per player per game with its metric columns, then
summarizes both views through one of two paths:

* :meth:`BehaviorAnalysis.add_heatmap_views` writes one long table per view
  plus the ``metadata["heatmaps"]`` layout the report draws as an HTML heatmap
  (the flavors page);
* :meth:`BehaviorAnalysis.metric_views` draws matplotlib heatmaps (the pages
  not yet moved to HTML tables).

Either way the declared views render behind the relative/absolute toggle; a
module may declare more views (the diplomacy page's Stance view) through
``view_names``.
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
    figures: dict = field(default_factory=dict)
    declared: dict = field(default_factory=lambda: {C.RELATIVE: {"tables": [], "figures": []},
                                                    C.ABSOLUTE: {"tables": [], "figures": []}})
    summaries: dict = field(default_factory=dict)
    colors: dict = field(default_factory=dict)
    heatmaps: dict = field(default_factory=dict)

    def add(self, view: str, name: str, table=None, figure=None) -> None:
        self.declared.setdefault(view, {"tables": [], "figures": []})
        if table is not None:
            self.tables[name] = table
            self.declared[view]["tables"].append(name)
        if figure is not None:
            self.figures[name] = figure
            self.declared[view]["figures"].append(name)


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
                summary = _attach_ranges(summary, frame, self.by, view_ranges)
            summary["color_position"] = C.color_positions(summary, view, views, fixed)
            table, order = C.heatmap_rows(ctx, summary, self.by)
            table.insert(0, "row_kind", "group")
            if pinned:
                table = _with_baseline_row(table, views, view_metrics, pinned, self.by, groups, fixed)
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
    ) -> dict | None:
        """Write ``<slug>_by_seed`` / ``<slug>_by_seat`` for a Matched Maps tab.

        Both are absolute heatmap tables over controlled games, one keyed by
        ``seed`` and one by ``seed`` and ``player_id``, each pinning the
        baseline pool's average for that seed or seat as its top row. Vanilla
        players are left out: the chapter's pinned row is the reference. A cell's
        ``difference`` is the mean relative value there (the player minus the
        baseline at its own seed and seat), shown in the tooltip. No bootstrap:
        the Matched Maps tables report plain means. Returns the
        ``metadata["matched_maps"]`` entry, or ``None`` without a relative view.
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
        declared = {"label": label, "tip": tip}
        for keys, name, key in ((["seed"], f"{slug}_by_seed", "seed_table"),
                                (["seed", "player_id"], f"{slug}_by_seat", "seat_table")):
            table, order = self._matched_map_table(ctx, rows, base, views, metrics, keys,
                                                   pinned, groups, fixed)
            if table.empty:
                return None
            out.tables[name] = table
            spec = C.heatmap_spec(
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
                value_label="Average",
                ci_level=self.ci_level,
                legend=legend,
                baseline_rows=[pinned],
            )
            spec["tip_rows"] = [{"column": "difference", "label": "Vs. baseline",
                                 "signed": True, "decimals": decimals + 1}]
            out.heatmaps[name] = spec
            declared[key] = name
        return declared

    def _matched_map_table(self, ctx, rows, base, views, metrics, keys, pinned, groups, fixed):
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
                                 "difference": float("nan"), "n_players": int(len(col)),
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
                         slug: str, names: dict, decimals: int = 1) -> str:
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
            if view == C.RELATIVE:
                return (
                    f"Against the {views.baseline.label} on the same map and seat, the largest "
                    f"departure is **{row}** on {name} (**{best['mean']:+.{decimals}f}**), across "
                    f"**{len(views.relative)}** controlled players."
                )
            return (
                f"The most distinctive value is **{row}** on {name} "
                f"(**{best['mean']:.{decimals}f}**), across **{len(views.absolute)}** players in "
                f"**{views.absolute['game_id'].nunique()}** games."
            )
        return "No player had values for the selected behavior metrics."

    # ── outputs ───────────────────────────────────────────────────────────────
    def metric_views(
        self,
        ctx: AnalysisContext,
        views: C.BehaviorViews,
        slug: str,
        title: str,
        labels: dict,
        metrics: list[str] | None = None,
        figure_slug: str | None = None,
    ) -> MetricViews:
        """Summary tables and heatmaps for ``metrics`` in both views."""
        out = MetricViews()
        self.add_metric_views(ctx, views, out, slug, title, labels, metrics, figure_slug)
        return out

    def add_metric_views(
        self,
        ctx: AnalysisContext,
        views: C.BehaviorViews,
        out: MetricViews,
        slug: str,
        title: str,
        labels: dict,
        metrics: list[str] | None = None,
        figure_slug: str | None = None,
    ) -> None:
        by = self.by
        figure_slug = figure_slug or slug
        metrics = list(metrics if metrics is not None else views.metrics)
        if views.relative is not None:
            rel_metrics = [m for m in metrics if m in views.relative_metrics]
            if rel_metrics:
                rel = C.summarize(ctx, views.relative, rel_metrics, by, C.RELATIVE,
                                  self.bootstrap_n, self.ci_level)
                colors = C.relative_colors(rel, views.baseline_sd, by)
                fig = C.summary_heatmap(
                    rel, by, rel_metrics, colors, labels=labels, signed=True,
                    title=f"{title}: difference from the matched in-game AI",
                    cbar_label="difference / baseline SD",
                ) if not rel.empty else None
                out.add(C.RELATIVE, f"{slug}_relative", table=rel)
                if fig is not None:
                    out.add(C.RELATIVE, f"{figure_slug}_relative", figure=fig)
                out.summaries[(slug, C.RELATIVE)] = rel
                out.colors[(slug, C.RELATIVE)] = colors
        absolute = C.summarize(ctx, views.absolute, metrics, by, C.ABSOLUTE,
                               self.bootstrap_n, self.ci_level)
        colors = C.absolute_colors(absolute, views.absolute, by)
        fig = C.summary_heatmap(
            absolute, by, metrics, colors, labels=labels, signed=False,
            title=f"{title}: absolute values", cbar_label="z-score across players",
        ) if not absolute.empty else None
        out.add(C.ABSOLUTE, f"{slug}_absolute", table=absolute)
        if fig is not None:
            out.add(C.ABSOLUTE, f"{figure_slug}_absolute", figure=fig)
        out.summaries[(slug, C.ABSOLUTE)] = absolute
        out.colors[(slug, C.ABSOLUTE)] = colors

    def headline(self, out: MetricViews, views: C.BehaviorViews, slug: str, labels: dict) -> str:
        """One sentence leading with the relative finding when there is one."""
        rel = out.summaries.get((slug, C.RELATIVE))
        if rel is not None and not rel.empty:
            pick = _pick(rel, out.colors[(slug, C.RELATIVE)], self.by)
            if pick is not None:
                group, metric, mean = pick
                return (
                    f"Against the {views.baseline.label}, the largest departure is **{group}** on "
                    f"{labels[metric]} (**{mean:+.2f}**), across **{len(views.relative)}** "
                    "controlled players."
                )
        absolute = out.summaries.get((slug, C.ABSOLUTE))
        if absolute is None or absolute.empty:
            return "No player had values for the selected behavior metrics."
        pick = _pick(absolute, out.colors[(slug, C.ABSOLUTE)], self.by)
        n_games = views.absolute["game_id"].nunique()
        if pick is None:
            return f"Behavior covers **{len(views.absolute)}** players in **{n_games}** games."
        group, metric, mean = pick
        overall = float(views.absolute[metric].mean())
        return (
            f"The most distinctive value is **{group}** on {labels[metric]} (**{mean:.2f}** vs "
            f"**{overall:.2f}** overall), across **{len(views.absolute)}** players in "
            f"**{n_games}** games."
        )


def _with_baseline_row(table: pd.DataFrame, views: C.BehaviorViews, metrics: list[str],
                       label: str, by: str, groups: dict | None, fixed: dict | None) -> pd.DataFrame:
    """``table`` with the baseline pool's absolute means prepended as one row."""
    records = []
    for metric in metrics:
        stats = views.baseline_summary.get(metric)
        if stats is None:
            continue
        records.append({
            "row_kind": "baseline", by: label, "strategist": label, "condition": "",
            "row_label": label, "metric": metric, "metric_group": (groups or {}).get(metric, ""),
            "mean": stats["mean"], "sd": stats["sd"],
            "n_players": stats["n_players"], "n_games": stats["n_games"],
        })
    if not records:
        return table
    rows = pd.DataFrame(records)
    rows["color_position"] = C.color_positions(rows, C.ABSOLUTE, views, fixed)
    return pd.concat([rows, table], ignore_index=True)


def _attach_ranges(summary: pd.DataFrame, rows: pd.DataFrame, by: str, ranges: dict) -> pd.DataFrame:
    """Per-group means of each metric's per-player ``(min, max)`` columns."""
    lows, highs = [], []
    means = {}
    for metric, (low, high) in ranges.items():
        if low in rows.columns and high in rows.columns:
            means[metric] = rows.groupby(rows[by].astype(str))[[low, high]].mean()
    for rec in summary.itertuples(index=False):
        frame = means.get(rec.metric)
        group = str(getattr(rec, by))
        if frame is None or group not in frame.index:
            lows.append(float("nan"))
            highs.append(float("nan"))
            continue
        low, high = ranges[rec.metric]
        lows.append(float(frame.loc[group, low]))
        highs.append(float(frame.loc[group, high]))
    out = summary.copy()
    out["mean_min"], out["mean_max"] = lows, highs
    return out


def _pick(summary: pd.DataFrame, colors: dict, by: str):
    return C.largest_departure(summary, colors, by, min_players=3) or C.largest_departure(summary, colors, by)
