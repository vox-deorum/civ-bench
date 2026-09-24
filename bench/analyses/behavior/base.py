"""Common scaffolding for the ``behavior.*`` modules.

A module computes one row per player per game with its metric columns, then
:meth:`BehaviorAnalysis.metric_views` summarizes both views, draws their
heatmaps, and returns the tables, figures, and view declarations the report
renders behind the relative/absolute toggle.
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

    def add(self, view: str, name: str, table=None, figure=None) -> None:
        if table is not None:
            self.tables[name] = table
            self.declared[view]["tables"].append(name)
        if figure is not None:
            self.figures[name] = figure
            self.declared[view]["figures"].append(name)


class BehaviorAnalysis(Analysis):
    """Settings shared by every behavior module; subclasses implement ``run``."""

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

    def load_behavior(self, ctx: AnalysisContext, metrics: list[str]):
        """Filtered rows and the baseline pool of the behavior table, rates applied.

        Returns ``(rows, pool, rate_scaled)`` where ``rate_scaled`` names the count
        columns divided by turns alive.
        """
        kinds = self.behavior_kinds(ctx)
        full = ctx.load_table("behavior")
        missing = [m for m in metrics if m not in full.columns]
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
        baseline = C.resolve_baseline_experiment(ctx, self.params)
        pool = C.baseline_pool(ctx, full, baseline)
        return rows, pool, rate_scaled, baseline

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
                    f"Against the matched in-game AI, the largest departure is **{group}** on "
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


def _pick(summary: pd.DataFrame, colors: dict, by: str):
    return C.largest_departure(summary, colors, by, min_players=3) or C.largest_departure(summary, colors, by)
