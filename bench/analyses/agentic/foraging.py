"""``agentic.foraging``: what strategists read in the workspace.

Per player per game, from the workspace calls of the ``agentic`` table:

* **Sources**: percent of workspace calls reading each folder (records,
  reference, notes, the guide, scratch, other). A call can read several, so a
  row need not sum to 100;
* **Record sections**: percent of record-reading calls reading each section;
* **Reference**: percent of reference-reading calls reading each category,
  plus ``ref_run_share``, the percent of runs that read any reference;
* **Read style**: percent of workspace calls that print a whole file, part of
  one, only names, or nothing;
* **Lookback**: among calls that read a turn's records, the percent that
  reach an earlier turn, and the mean and 90th percentile of how far back;
* **Informed choices**: percent of successful ``set-research`` and
  ``set-policy`` choices whose option the model had looked up earlier in the
  same run (``grounded``, a lower bound).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisContext, AnalysisResult
from ..behavior import common as C
from ..behavior.base import MetricViews
from .base import AgenticAnalysis, share

SOURCES = {
    "src_records": ("Records", "Reports of past and current turns."),
    "src_reference": ("Reference", "The game rules: technologies, policies, buildings, units."),
    "src_notes": ("Notes", "The model's own notes, kept all game."),
    "src_guide": ("Guide", "The editable guide, AGENTS.md."),
    "src_tmp": ("Scratch", "Temporary files under /tmp."),
    "src_other": ("Other", "Any other path under /workspace."),
}
# Record sections, grouped for the spanning header.
SECTIONS = {
    "rec_events_all": ("All events", "Events"),
    "rec_events_economic": ("Economic", "Events"),
    "rec_events_military": ("Military", "Events"),
    "rec_events_diplomatic": ("Diplomatic", "Events"),
    "rec_events_others": ("Other events", "Events"),
    "rec_situation": ("Situation", "Turn state"),
    "rec_civilization": ("Civilization", "Turn state"),
    "rec_options": ("Options", "Turn state"),
    "rec_strategies": ("Strategies", "Turn state"),
    "rec_victory": ("Victory", "Turn state"),
    "rec_players": ("Players", "Turn state"),
    "rec_cities": ("Cities", "Turn state"),
    "rec_military": ("Military", "Turn state"),
    "rec_catalog": ("Catalog", "Catalog and other"),
    "rec_other": ("Other", "Catalog and other"),
}
REFERENCE = {
    "ref_technologies": "Technologies",
    "ref_policies": "Policies",
    "ref_buildings": "Buildings",
    "ref_units": "Units",
    "ref_other": "Other",
}
STYLES = {
    "full": ("Whole file", "A whole file reached the output."),
    "targeted": ("Part of a file", "Part of a file or a count, through grep, sed, head, wc, "
                                   "a piped filter, and similar."),
    "listing": ("Names only", "Only file names or whether a file exists."),
    "none": ("Nothing", "The command read no file."),
}
CHOICES = {
    "set-research": ("grounded_research_share", "Research choices"),
    "set-policy": ("grounded_policy_share", "Policy choices"),
}


def style_column(style: str) -> str:
    return f"style_{style}"


def flagged(frame: pd.DataFrame, column: str) -> pd.Series:
    """Rows whose 0/1 flag ``column`` is set (blank reads as unset)."""
    return frame[column].fillna(0) > 0


def foraging_metrics(grp: pd.DataFrame) -> dict:
    """The foraging metrics of one player in one game."""
    runs = int(grp["run"].nunique())
    ws = grp[grp["kind"] == "workspace"]
    out = {"runs": runs, "workspace_calls": int(len(ws))}
    for column in SOURCES:
        out[column] = share(int(flagged(ws, column).sum()), len(ws))
    records = ws[flagged(ws, "src_records")]
    for column in SECTIONS:
        out[column] = share(int(flagged(records, column).sum()), len(records))
    reference = ws[flagged(ws, "src_reference")]
    for column in REFERENCE:
        out[column] = share(int(flagged(reference, column).sum()), len(reference))
    out["ref_run_share"] = share(reference["run"].nunique(), runs)
    styles = ws["read_style"].astype(str)
    for style in STYLES:
        out[style_column(style)] = share(int((styles == style).sum()), len(ws))
    lookback = ws["lookback"].dropna()
    out["lookback_share"] = share(int((lookback > 0).sum()), len(lookback))
    out["lookback_mean"] = float(lookback.mean()) if len(lookback) else float("nan")
    out["lookback_p90"] = float(np.quantile(lookback, 0.9)) if len(lookback) else float("nan")
    for tool, (column, _label) in CHOICES.items():
        chosen = grp[(grp["kind"] == "action") & (grp["tool"].astype(str) == tool)
                     & flagged(grp, "action_ok")]
        out[column] = share(int(flagged(chosen, "grounded").sum()), len(chosen))
    return out


class AgenticForaging(AgenticAnalysis):
    module = "agentic.foraging"
    friendly_name = "What strategists read"
    description = (
        "Shows which parts of the workspace strategists read (records, rules, or notes), "
        "how they read them, how far back in the game they look, and how often a research "
        "or policy choice follows a look-up of that option."
    )
    columns = (*SECTIONS, *REFERENCE)
    report_defaults = {
        "tables": ["sources_absolute", "sections_absolute", "reference_absolute",
                   "read_style_absolute", "lookback_absolute", "choices_absolute"],
        "figures": [],
    }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        rows, hidden = self.load(ctx)
        if rows.empty:
            return self.empty_result(hidden)
        frame = self.player_games(rows, foraging_metrics)
        out = MetricViews()
        common = ("Each cell averages the player-games of a strategist and condition.")
        views = self.heatmap(
            ctx, frame, out, "sources", title="Where commands read",
            help_text=(f"{common} A cell is the percent of a player's workspace commands that "
                       "read the folder. A command can read several folders, so a row need not "
                       "add up to 100%."),
            names={c: name for c, (name, _) in SOURCES.items()},
            column_tips={c: tip for c, (_, tip) in SOURCES.items()},
            shares=set(SOURCES), decimals=0,
        )
        self.heatmap(
            ctx, frame, out, "sections", title="Record sections read",
            help_text=(f"{common} A cell is the percent of a player's record-reading commands "
                       "that read the section."),
            names={c: name for c, (name, _) in SECTIONS.items()},
            groups={c: group for c, (_, group) in SECTIONS.items()},
            shares=set(SECTIONS), decimals=0,
        )
        self.heatmap(
            ctx, frame, out, "reference", title="Rules read",
            help_text=(f"{common} The category cells are the percent of a player's "
                       "reference-reading commands that read the category. Any reference is "
                       "the percent of runs that read the rules at all."),
            names={"ref_run_share": "Any reference", **REFERENCE},
            column_tips={"ref_run_share": "Percent of decision runs that read any reference file."},
            shares={"ref_run_share", *REFERENCE}, decimals=0,
        )
        self.heatmap(
            ctx, frame, out, "read_style", title="How commands read",
            help_text=f"{common} A cell is the percent of a player's workspace commands in each style.",
            names={style_column(s): name for s, (name, _) in STYLES.items()},
            column_tips={style_column(s): tip for s, (_, tip) in STYLES.items()},
            shares={style_column(s) for s in STYLES}, decimals=0,
        )
        self.heatmap(
            ctx, frame, out, "lookback", title="Looking back",
            help_text=(f"{common} Among commands that read a turn's records: the percent that "
                       "reached an earlier turn than the one being decided, and how many turns "
                       "back the oldest one was."),
            names={"lookback_share": "Reads of earlier turns",
                   "lookback_mean": "Average turns back",
                   "lookback_p90": "90th percentile turns back"},
            short_names={"lookback_share": "Earlier turns", "lookback_mean": "Mean back",
                         "lookback_p90": "P90 back"},
            shares={"lookback_share"},
            column_decimals={"lookback_share": 0, "lookback_mean": 1, "lookback_p90": 1},
        )
        self.heatmap(
            ctx, frame, out, "choices", title="Informed choices",
            help_text=(f"{common} A cell is the percent of a player's successful choices whose "
                       "option the model had looked up (its heading or name appeared in a "
                       "workspace output) in an earlier step of the same run. This is a lower "
                       "bound: reading it in an earlier turn does not count."),
            names={column: label for column, label in CHOICES.values()},
            shares={column for column, _ in CHOICES.values()}, decimals=0,
        )
        summary = self.top_row_summary(
            out.summaries.get(("sources", C.ABSOLUTE)),
            {"Most reference reading": "src_reference", "Most note reading": "src_notes"},
            self.skip_labels(ctx),
        )
        informed = self.top_row_summary(
            out.summaries.get(("choices", C.ABSOLUTE)),
            {"Most informed research": "grounded_research_share"},
            self.skip_labels(ctx),
        )
        summary = " · ".join(part for part in (summary, informed) if part)
        return self.result(ctx, out, views, hidden,
                           summary or "No workspace reads were recorded.")
