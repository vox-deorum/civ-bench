"""``agentic.reliability``: how often workspace commands fail, and what follows.

Per player per game, from the workspace calls of the ``agentic`` table:

* ``error_share``: percent of commands whose outcome is neither ``ok`` nor
  ``no_match`` (a search that found nothing is not a failure);
* ``no_match_share``: percent of commands that searched and found nothing;
* ``truncated_share``: percent of commands whose output the host cut;
* ``recovery_share``: percent of failures followed by another workspace
  command later in the same run;
* ``error_<kind>_per_100_calls``: failures of each kind per 100 commands, so
  a player's kinds add up to its ``error_share``.
"""

from __future__ import annotations

import pandas as pd

from ..base import AnalysisContext, AnalysisResult
from ..behavior import common as C
from ..behavior.base import MetricViews
from .base import WORKSPACE, AgenticAnalysis, share

OK, NO_MATCH = "ok", "no_match"
ERROR_KINDS = {
    "read_only": ("Read-only folder", "A write to a folder the model can only read."),
    "missing_file": ("Missing file", "A path that did not exist."),
    "syntax": ("Shell syntax", "The shell could not parse the command."),
    "not_found": ("Unknown command", "A command the workspace does not have (exit 127)."),
    "other": ("Other", "Any other failure."),
}


def error_kind(kind: str) -> str:
    return f"error_{kind}_per_100_calls"


NAMES = {
    "error_share": "Commands that failed",
    "no_match_share": "Searches that found nothing",
    "truncated_share": "Output cut by the host",
    "recovery_share": "Failures followed by another command",
}
SHORT = {"error_share": "Failed", "no_match_share": "No match", "truncated_share": "Cut",
         "recovery_share": "Recovered"}
TIPS = {
    "error_share": "Percent of workspace commands that failed. Searches that found nothing do not count.",
    "no_match_share": "Percent of workspace commands that searched and found nothing (exit 1, no error).",
    "truncated_share": "Percent of workspace commands whose output the host cut short.",
    "recovery_share": ("Percent of failed commands followed by another workspace command later "
                       "in the same run."),
}


def reliability_metrics(grp: pd.DataFrame) -> dict:
    """The reliability metrics of one player in one game."""
    ws = grp[grp["kind"] == WORKSPACE]
    outcome = ws["outcome"].astype(str)
    failed = ws[ws["outcome"].notna() & ~outcome.isin({OK, NO_MATCH})]
    later = ws.groupby("run")["call"].max()
    recovered = int((failed["call"] < failed["run"].map(later)).sum())
    out = {
        "workspace_calls": int(len(ws)),
        "errors": int(len(failed)),
        "error_share": share(len(failed), len(ws)),
        "no_match_share": share(int((outcome == NO_MATCH).sum()), len(ws)),
        "truncated_share": share(int((ws["truncated_chars"].fillna(0) > 0).sum()), len(ws)),
        "recovery_share": share(recovered, len(failed)),
    }
    kinds = failed["outcome"].astype(str)
    for kind in ERROR_KINDS:
        out[error_kind(kind)] = share(int((kinds == kind).sum()), len(ws))
    return out


class AgenticReliability(AgenticAnalysis):
    module = "agentic.reliability"
    friendly_name = "Command failures"
    description = (
        "Shows how often strategists' workspace commands fail, why they fail, and whether "
        "the strategist tries again in the same turn."
    )
    report_defaults = {"tables": ["reliability_absolute", "error_kinds_absolute"], "figures": []}

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        rows, hidden = self.load(ctx)
        if rows.empty:
            return self.empty_result(hidden)
        frame = self.player_games(rows, reliability_metrics)
        out = MetricViews()
        common = "Each cell averages the player-games of a strategist and condition."
        views = self.heatmap(
            ctx, frame, out, "reliability", title="Command failures",
            help_text=common, names=NAMES, short_names=SHORT, column_tips=TIPS,
            shares=set(NAMES), decimals=0,
        )
        self.heatmap(
            ctx, frame, out, "error_kinds", title="Kinds of failure",
            help_text=(f"{common} A cell is the failed commands of each kind per 100 workspace "
                       "commands. A row adds up to the Failed column of Command failures."),
            names={error_kind(k): f"{name} failures per 100 commands"
                   for k, (name, _) in ERROR_KINDS.items()},
            short_names={error_kind(k): name for k, (name, _) in ERROR_KINDS.items()},
            column_tips={error_kind(k): f"{tip} Counted per 100 workspace commands."
                         for k, (_, tip) in ERROR_KINDS.items()},
            decimals=1,
        )
        summary = self.top_row_summary(
            out.summaries.get(("reliability", C.ABSOLUTE)),
            {"Fewest failures": "error_share", "Most failures": "error_share",
             "Most often tries again": "recovery_share"},
            self.skip_labels(ctx), lowest=("Fewest failures",),
            formats={"Fewest failures": "{:.1f}%", "Most failures": "{:.1f}%"},
        )
        return self.result(ctx, out, views, hidden,
                           summary or "No workspace commands were recorded.")
