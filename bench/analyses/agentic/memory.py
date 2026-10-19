"""``agentic.memory``: whether strategists keep notes, and whether the notes survive.

Per player per game, from the workspace calls of the ``agentic`` table:

* ``writes_<dest>_per_100_runs``: paths written per 100 runs, by where they
  landed: kept in the notes folder (persisted), scratch, lost, or refused;
* ``persist_share``: percent of written paths that were kept (blank with no
  writes);
* ``guide_edits_per_100_runs``: commands that changed the guide, per 100 runs;
* ``note_reads_per_run``: commands reading the notes or guide, per run;
* ``note_missing_share``: percent of note or guide reads that hit a file that did not exist.
"""

from __future__ import annotations

import pandas as pd

from ..base import AnalysisContext, AnalysisResult
from ..behavior import common as C
from ..behavior.base import MetricViews
from .base import WORKSPACE, AgenticAnalysis, ratio, share
from .foraging import flagged

DESTINATIONS = {
    "persisted": ("Kept", "Paths written to the notes folder, kept all game."),
    "scratch": ("Scratch", "Paths written to /tmp, gone after the run."),
    "lost": ("Lost", "Paths written anywhere else, gone after the command."),
    "rejected": ("Refused", "Writes to a read-only folder, or failed writes to the notes folder."),
}


def writes(dest: str) -> str:
    return f"writes_{dest}"


def per_100(dest: str) -> str:
    return f"writes_{dest}_per_100_runs"


NAMES = {
    **{per_100(d): f"{name} writes per 100 runs" for d, (name, _) in DESTINATIONS.items()},
    "persist_share": "Writes kept",
    "guide_edits_per_100_runs": "Guide edits per 100 runs",
    "note_reads_per_run": "Note reads per run",
    "note_missing_share": "Note reads that found nothing",
}
SHORT = {
    **{per_100(d): name for d, (name, _) in DESTINATIONS.items()},
    "persist_share": "Kept share", "guide_edits_per_100_runs": "Guide edits",
    "note_reads_per_run": "Note reads", "note_missing_share": "Missing notes",
}
TIPS = {
    **{per_100(d): f"{tip} Counted per 100 decision runs." for d, (_, tip) in DESTINATIONS.items()},
    "persist_share": "Percent of written paths that were kept in the notes folder.",
    "guide_edits_per_100_runs": "Commands that changed the guide (AGENTS.md), per 100 runs.",
    "note_reads_per_run": "Workspace commands reading the notes or guide, per run.",
    "note_missing_share": "Percent of note or guide reads that asked for a file that did not exist.",
}
GROUPS = {
    **{per_100(d): "Writes per 100 runs" for d in DESTINATIONS},
    "persist_share": "Writes per 100 runs",
    "guide_edits_per_100_runs": "Guide and notes",
    "note_reads_per_run": "Guide and notes",
    "note_missing_share": "Guide and notes",
}
SHARES = {"persist_share", "note_missing_share"}
DECIMALS = {"persist_share": 0, "note_missing_share": 0, "note_reads_per_run": 2}


def memory_metrics(grp: pd.DataFrame) -> dict:
    """The memory metrics of one player in one game."""
    runs = int(grp["run"].nunique())
    ws = grp[grp["kind"] == WORKSPACE]
    totals = {d: float(ws[writes(d)].fillna(0).sum()) for d in DESTINATIONS}
    out = {"runs": runs, **{writes(d): totals[d] for d in DESTINATIONS}}
    for dest in DESTINATIONS:
        out[per_100(dest)] = 100.0 * ratio(totals[dest], runs)
    out["persist_share"] = share(totals["persisted"], sum(totals.values()))
    out["guide_edits_per_100_runs"] = 100.0 * ratio(int(flagged(ws, "guide_edit").sum()), runs)
    reads = ws[flagged(ws, "src_notes") | flagged(ws, "src_guide")]
    out["note_reads_per_run"] = ratio(len(reads), runs)
    out["note_missing_share"] = share(int(flagged(reads, "note_missing").sum()), len(reads))
    return out


class AgenticMemory(AgenticAnalysis):
    module = "agentic.memory"
    friendly_name = "Notes and memory"
    description = (
        "Shows whether strategists write notes that last the whole game, where their writes "
        "end up, and whether they read their notes and guide back."
    )
    report_defaults = {"tables": ["memory_absolute"], "figures": []}

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        rows, hidden = self.load(ctx)
        if rows.empty:
            return self.empty_result(hidden)
        frame = self.player_games(rows, memory_metrics)
        out = MetricViews()
        views = self.heatmap(
            ctx, frame, out, "memory", title="Notes and memory",
            help_text=("Each cell averages the player-games of a strategist and condition. Only "
                       "the notes folder lasts the whole game; scratch and lost writes vanish, "
                       "and refused writes never happened."),
            names=NAMES, short_names=SHORT, column_tips=TIPS, groups=GROUPS, shares=SHARES,
            column_decimals=DECIMALS,
        )
        return self.result(ctx, out, views, hidden, self._summary(ctx, out, frame))

    def _summary(self, ctx: AnalysisContext, out: MetricViews, frame: pd.DataFrame) -> str:
        """Lead with persistence: who kept notes, or who had writes refused."""
        table = out.summaries.get(("memory", C.ABSOLUTE))
        if table is None or table.empty:
            return "No workspace writes were recorded."
        labels = dict(zip(table[self.by].astype(str), table["row_label"].astype(str)))
        totals = frame.groupby(self.by)[[writes(d) for d in DESTINATIONS]].sum()
        totals = totals[~totals.index.isin(self.skip_labels(ctx))]
        kept = totals[writes("persisted")]
        if (kept > 0).any():
            return self.top_row_summary(table, {"Keeps most notes": "persist_share"},
                                        self.skip_labels(ctx))
        refused = totals[writes("rejected")]
        if (refused > 0).any():
            top = refused.max()
            names = " / ".join(labels.get(str(g), str(g)) for g in refused.index[refused == top])
            return f"No strategist kept a note; **{names}** had **{top:.0f}** writes refused."
        if totals.to_numpy().sum() > 0:
            return "No strategist kept a note; every write went to scratch or was lost."
        return "No strategist wrote to the workspace."
