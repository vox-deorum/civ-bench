"""``agentic.engagement``: how much strategists use the workspace, and when.

Per player per game, from the ``agentic`` table:

* ``use_share``: percent of runs with at least one workspace call;
* ``calls_per_run``: workspace calls per run;
* ``look_first_share``: percent of runs with an action in which a workspace
  call came in an earlier step than the first action;
* ``steps_per_run``: model replies per run, from the run's ``run_steps``,
  counting replies that made no call and runs that made none at all;
* ``gather_steps_per_run``: steps with workspace calls and no action, per run;
* ``kb_per_run``: thousands of output characters the workspace printed, per run;
* ``mixed_share``: percent of workspace calls made in a step that also acted.

The ``progress`` table and figure follow ``use_share`` and ``calls_per_run``
over the game, in ``progress_bins`` equal slices of each player's turns, each
with a game-level bootstrap CI.
"""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

from ..base import AnalysisContext, AnalysisResult
from ..behavior import common as C
from ..behavior.base import MetricViews
from .base import ACTION, MIXED, GATHER, WORKSPACE, AgenticAnalysis, ratio, share

DEFAULT_PROGRESS_BINS = 10

NAMES = {
    "use_share": "Runs using the workspace",
    "calls_per_run": "Workspace calls per run",
    "look_first_share": "Looked before acting",
    "steps_per_run": "Steps per run",
    "gather_steps_per_run": "Look-up steps per run",
    "kb_per_run": "Output read per run (thousand characters)",
    "mixed_share": "Calls made while acting",
}
SHORT = {
    "use_share": "Use", "calls_per_run": "Calls", "look_first_share": "Look first",
    "steps_per_run": "Steps", "gather_steps_per_run": "Look-up steps",
    "kb_per_run": "K chars read", "mixed_share": "With action",
}
TIPS = {
    "use_share": "Percent of decision runs with at least one workspace command.",
    "calls_per_run": "Workspace commands per decision run, counting runs without any.",
    "look_first_share": ("Among runs that took a game action: percent in which a workspace "
                         "command came in an earlier step than the first action."),
    "steps_per_run": ("Model replies per decision run, counting replies that made no "
                      "workspace call and runs that made none at all."),
    "gather_steps_per_run": "Steps that ran workspace commands and took no action, per run.",
    "kb_per_run": "Characters the workspace printed, in thousands, per run.",
    "mixed_share": ("Percent of workspace commands sent in the same reply as a game action, "
                    "so their output could not inform that action."),
}
SHARES = {"use_share", "look_first_share", "mixed_share"}
DECIMALS = {"use_share": 0, "look_first_share": 0, "mixed_share": 0, "calls_per_run": 1,
            "steps_per_run": 1, "gather_steps_per_run": 1, "kb_per_run": 1}
DASHES = ("solid", "dash", "dot", "dashdot")
# The progress figure's metrics: subplot title, tooltip unit, and tooltip decimals.
PROGRESS = {
    "use_share": ("Runs using the workspace", "%", 1),
    "calls_per_run": ("Workspace calls per run", "", 2),
}


def engagement_metrics(grp: pd.DataFrame) -> dict:
    """The engagement metrics of one player in one game."""
    runs = int(grp["run"].nunique())
    ws = grp[grp["kind"] == WORKSPACE]
    acts = grp[grp["kind"] == ACTION]
    first_action = acts.groupby("run")["step"].min()
    first_look = ws.groupby("run")["step"].min()
    looked = first_look.reindex(first_action.index)
    steps = grp.dropna(subset=["step"]).drop_duplicates(["run", "step"])
    run_steps = pd.to_numeric(grp.drop_duplicates("run")["run_steps"], errors="coerce")
    return {
        "runs": runs,
        "use_share": share(ws["run"].nunique(), runs),
        "calls_per_run": ratio(len(ws), runs),
        "look_first_share": share(int((looked < first_action).sum()), len(first_action)),
        "steps_per_run": float(run_steps.mean()),
        "gather_steps_per_run": ratio(int((steps["step_kind"] == GATHER).sum()), runs),
        "kb_per_run": ratio(ws["out_chars"].fillna(0).sum() / 1000.0, runs),
        "mixed_share": share(int((ws["step_kind"] == MIXED).sum()), len(ws)),
    }


def progress_tips(grp: pd.DataFrame, label: str, column: str) -> list[list[str]]:
    """Tooltip text per progress point: title, subtitle, average, CI, and players."""
    title, unit, places = PROGRESS[column]

    def number(value) -> str:
        return f"{value:.{places}f}{unit}" if np.isfinite(value) else ""

    tips = []
    for rec in grp.to_dict("records"):
        lo, hi = number(rec[f"{column}_ci_lower"]), number(rec[f"{column}_ci_upper"])
        games = int(rec["n_games"])
        tips.append([
            label,
            f"{title} · {rec['progress_start']:.0f}% to {rec['progress_end']:.0f}% of the game",
            number(rec[column]),
            f"{lo} to {hi}" if lo and hi else "N/A (needs 2 games)",
            f"{int(rec['n_players'])} in {games} game{'s' if games != 1 else ''}",
        ])
    return tips


class AgenticEngagement(AgenticAnalysis):
    module = "agentic.engagement"
    friendly_name = "Workspace use"
    description = (
        "Shows how often strategists with a workspace use it, how much they read, and "
        "whether they look things up before acting, over the whole game and by game stage."
    )
    columns = ("run_steps",)
    report_defaults = {"tables": ["engagement_absolute"], "figures": ["progress"]}

    @property
    def progress_bins(self) -> int:
        return int(self.params.get("progress_bins", DEFAULT_PROGRESS_BINS))

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        rows, hidden = self.load(ctx)
        if rows.empty:
            return self.empty_result(hidden)
        frame = self.player_games(rows, engagement_metrics)
        out = MetricViews()
        views = self.heatmap(
            ctx, frame, out, "engagement",
            title="Workspace use",
            help_text=("Each cell averages the player-games of a strategist and condition. "
                       "A run is one turn's decision; a step is one model reply."),
            names=NAMES, short_names=SHORT, column_tips=TIPS, shares=SHARES,
            column_decimals=DECIMALS,
        )
        progress = self._progress(ctx, rows)
        out.tables["progress"] = progress
        figures = {}
        fig = self._plot(ctx, progress)
        if fig is not None:
            figures["progress"] = fig
            out.declared[C.ABSOLUTE]["figures"].append("progress")
        summary = self.top_row_summary(
            out.summaries.get(("engagement", C.ABSOLUTE)),
            {"Most use": "use_share", "Most calls per run": "calls_per_run",
             "Most often looks first": "look_first_share"},
            self.skip_labels(ctx),
            formats={"Most calls per run": "{:.1f}"},
        ) or "No workspace calls were recorded."
        return self.result(ctx, out, views, hidden, summary, figures,
                           {"progress_bins": self.progress_bins})

    def _progress(self, ctx: AnalysisContext, rows: pd.DataFrame) -> pd.DataFrame:
        """Per group and game slice: mean ``use_share`` and ``calls_per_run`` over player-games."""
        bins = self.progress_bins
        keys = self.keys()
        runs = rows.groupby([*keys, "run"], sort=True).agg(
            progress=("progress", "min"),
            calls=("kind", lambda kinds: int((kinds == WORKSPACE).sum())),
        ).reset_index()
        # Rounding first keeps a run on a slice boundary (4 of 7 turns at 7
        # bins is 3.9999999999999996) in the slice it starts.
        runs["bin"] = np.minimum(np.floor((runs["progress"] * bins).round(9)).astype(int), bins - 1)
        runs["used"] = (runs["calls"] > 0).astype(float) * 100.0
        per_player = runs.groupby([*keys, "bin"], sort=True).agg(
            use_share=("used", "mean"), calls_per_run=("calls", "mean"),
        ).reset_index()
        records = []
        for b, grp in per_player.groupby("bin", sort=True):
            summary = C.summarize(ctx, grp, list(PROGRESS), self.by, f"progress_{b}",
                                  self.bootstrap_n, self.ci_level)
            for group, cells in summary.groupby(self.by, sort=False):
                record = {self.by: group, "bin": int(b)}
                for cell in cells.itertuples(index=False):
                    record[cell.metric] = cell.mean
                    record[f"{cell.metric}_ci_lower"] = cell.ci_lower
                    record[f"{cell.metric}_ci_upper"] = cell.ci_upper
                # Neither metric is ever blank, so both count the same players.
                record["n_players"] = int(cells["n_players"].iloc[0])
                record["n_games"] = int(cells["n_games"].iloc[0])
                records.append(record)
        table = pd.DataFrame(records).sort_values([self.by, "bin"], kind="mergesort")
        table["progress_start"] = table["bin"] / bins * 100.0
        table["progress_end"] = (table["bin"] + 1) / bins * 100.0
        table, order = C.heatmap_rows(ctx, table, self.by)
        rank = {label: i for i, label in enumerate(order)}
        table = table.assign(_row=table["row_label"].map(rank)).sort_values(
            ["_row", "bin"], kind="mergesort").drop(columns="_row")
        leading = [self.by, "strategist", "condition", "row_label", "bin",
                   "progress_start", "progress_end"]
        table = table[leading + [c for c in table.columns if c not in leading]]
        return table.reset_index(drop=True)

    def _plot(self, ctx: AnalysisContext, progress: pd.DataFrame):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from bench.plotting.styles import get_player_color

        if progress.empty:
            return None
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=[title for title, _, _ in PROGRESS.values()])
        seen: dict[str, int] = {}
        for label, grp in progress.groupby("row_label", sort=False):
            group = str(grp[self.by].iloc[0])
            color = get_player_color(ctx.catalog, group) if self.by == "player_type" else None
            dash = DASHES[seen.get(color, 0) % len(DASHES)]
            seen[color] = seen.get(color, 0) + 1
            x = ((grp["progress_start"] + grp["progress_end"]) / 2).tolist()
            for row, column in enumerate(PROGRESS, start=1):
                fig.add_trace(go.Scatter(
                    x=x, y=grp[column].round(4).tolist(), name=escape(str(label)),
                    legendgroup=str(label), showlegend=row == 1, mode="lines+markers",
                    line={"color": color, "dash": dash, "width": 2}, marker={"size": 6},
                    customdata=progress_tips(grp, str(label), column),
                    meta={"table_tooltip": True}, hoverinfo="none",
                ), row=row, col=1)
        fig.update_layout(
            template="plotly_white",
            title={"text": "Workspace use over the game", "x": 0.05},
            legend={"title": {"text": "Strategist | Condition"}},
            height=620, hovermode="closest",
            meta={"table_tooltip": {
                "title_index": 0, "subtitle_index": 1,
                "note": "Averages per player per game. The CI resamples whole games.",
                "rows": [
                    {"label": "Average", "index": 2, "emphasis": True},
                    {"label": f"{self.ci_level * 100:g}% CI", "index": 3},
                    {"label": "Players", "index": 4},
                ],
            }},
        )
        fig.update_yaxes(title_text="Runs (%)", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="Calls per run", rangemode="tozero", row=2, col=1)
        fig.update_xaxes(title_text="Game progress (% of the player's turns)", range=[0, 100],
                         row=2, col=1)
        return fig
