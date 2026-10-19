"""Common scaffolding for the ``agentic.*`` modules.

Every module reads the ``agentic`` table (one row per tool call, plus a
``kind == "none"`` row for a run without calls), keeps the groups that used a
workspace at least once, computes one row per player per game with its metric
columns, and summarizes them with the behavior heatmap machinery
(:class:`bench.analyses.behavior.base.BehaviorAnalysis`). No baseline plays
with a workspace, so every table is absolute.

Denominators shared by the modules:

* **runs**: the distinct ``run`` values of a player, ``none`` rows included;
* **workspace calls**: rows with ``kind == "workspace"``;
* ``helper`` rows count as neither a workspace call nor an action.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from ..behavior import common as C
from ..behavior.base import BehaviorAnalysis, MetricViews

WORKSPACE = "workspace"
ACTION = "action"
NONE = "none"
GATHER, ACT, MIXED = "gather", "act", "mixed"
NO_WORKSPACE = "No condition used workspace tools."

SHARE_RANGE = (0.0, 100.0)
SHARE_LEGEND = [[0.0, "0%"], [0.5, "50%"], [1.0, "100%"]]
Z_LEGEND = [[0.0, "2 SD below the average player"], [0.25, "1 SD below"], [0.5, "average"],
            [0.75, "1 SD above"], [1.0, "2 SD above"]]
MIXED_LEGEND = [[0.0, "0% or 2 SD below"], [0.5, "50% or average"], [1.0, "100% or 2 SD above"]]

NUMERIC_COLUMNS = (
    "turn", "run", "step", "call", "action_ok", "grounded", "exit_code", "out_chars",
    "truncated_chars", "duration_ms", "lookback", "writes_persisted", "writes_scratch",
    "writes_lost", "writes_rejected", "guide_edit", "note_missing",
)
FLAG_PREFIXES = ("src_", "rec_", "ref_")
# Columns every module reads; a module adds its own through ``columns``.
REQUIRED_COLUMNS = (
    *NUMERIC_COLUMNS, "kind", "tool", "option", "outcome", "read_style",
    "src_records", "src_reference", "src_notes", "src_guide", "src_tmp", "src_other",
)


def share(part, whole) -> float:
    """``100 * part / whole``, NaN when ``whole`` is zero."""
    return 100.0 * float(part) / float(whole) if whole else float("nan")


def ratio(part, whole) -> float:
    return float(part) / float(whole) if whole else float("nan")


class AgenticAnalysis(BehaviorAnalysis):
    """Settings shared by every agentic module; subclasses implement ``run``."""

    # Columns this module reads beyond :data:`REQUIRED_COLUMNS`.
    columns: tuple = ()

    # ── inputs ────────────────────────────────────────────────────────────────
    def load(self, ctx: AnalysisContext) -> tuple[pd.DataFrame, list[str]]:
        """Filtered rows of the groups that used a workspace, and the hidden groups."""
        rows = C.prepare_keys(ctx.load_table("agentic"), self.stage_id, "agentic table",
                              per_player=False)
        C.check_by(rows, self.by, self.stage_id)
        missing = [c for c in (*REQUIRED_COLUMNS, *self.columns) if c not in rows.columns]
        if missing:
            raise AnalysisError(
                f"analysis '{self.stage_id}': agentic table lacks column(s) {missing}; "
                "re-run extract to rebuild it."
            )
        rows = C.prepare_keys(ctx.apply_filter(rows), self.stage_id, "agentic table",
                              per_player=False)
        flags = [c for c in rows.columns if c.startswith(FLAG_PREFIXES)]
        rows = C.numeric(rows, [c for c in (*NUMERIC_COLUMNS, *flags) if c in rows.columns])
        rows["kind"] = rows["kind"].astype(str)
        groups = rows[self.by].astype(str)
        used = set(groups[rows["kind"] == WORKSPACE])
        hidden = sorted(set(groups) - used)
        rows = self._derive(rows)
        return rows[groups.isin(used)].reset_index(drop=True), hidden

    def keys(self) -> list[str]:
        """Columns naming one player in one game (and its ``by`` group)."""
        return list(dict.fromkeys([*C.KEY_COLUMNS, self.by]))

    def _derive(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Add ``step_kind`` (per run step) and ``progress`` (turn over the player's last)."""
        if rows.empty:
            return rows.assign(step_kind=pd.Series(dtype=object), progress=pd.Series(dtype=float))
        keys = C.KEY_COLUMNS
        step_keys = [*keys, "run", "step"]
        flags = rows.assign(_ws=rows["kind"] == WORKSPACE, _act=rows["kind"] == ACTION)
        steps = flags.dropna(subset=["step"]).groupby(step_keys, sort=False)[["_ws", "_act"]].any()
        kinds = np.select([steps["_ws"] & steps["_act"], steps["_ws"], steps["_act"]],
                          [MIXED, GATHER, ACT], default="")
        steps = pd.Series(kinds, index=steps.index, name="step_kind").replace("", np.nan)
        rows = rows.join(steps, on=step_keys)
        last = rows.groupby(keys, sort=False)["turn"].transform("max")
        rows["progress"] = (rows["turn"] / last.where(last > 0)).fillna(0.0).clip(0.0, 1.0)
        return rows

    def player_games(self, rows: pd.DataFrame, metrics_fn) -> pd.DataFrame:
        """One row per player per game: its keys plus ``metrics_fn(group) -> dict``."""
        keys = self.keys()
        records = []
        for values, grp in rows.groupby(keys, sort=True):
            records.append({**dict(zip(keys, values)), **metrics_fn(grp)})
        return pd.DataFrame(records)

    @staticmethod
    def run_count(grp: pd.DataFrame) -> int:
        return int(grp["run"].nunique())

    @staticmethod
    def workspace(grp: pd.DataFrame) -> pd.DataFrame:
        return grp[grp["kind"] == WORKSPACE]

    # ── output ────────────────────────────────────────────────────────────────
    def heatmap(self, ctx: AnalysisContext, frame: pd.DataFrame, out: MetricViews, slug: str, *,
                title: str, help_text: str, names: dict, short_names: dict | None = None,
                column_tips: dict | None = None, groups: dict | None = None,
                shares: set | None = None, decimals: int = 1,
                column_decimals: dict | None = None, value_label: str = "Average") -> C.BehaviorViews:
        """Summarize ``frame`` (one row per player-game) as one absolute heatmap ``<slug>_absolute``.

        Metrics in ``shares`` are percents on a fixed 0 to 100 color scale; the
        others are colored by z-score across players.
        """
        metrics = [m for m in names if m in frame.columns]
        shares = set(shares or ()) & set(metrics)
        empty = frame.iloc[0:0]
        views = C.build_views(frame, empty, metrics, None, None)
        units = {m: "%" for m in shares}
        if shares == set(metrics):
            legend = SHARE_LEGEND
            scale = "Colors are fixed from 0% (red) through 50% (yellow) to 100% (blue)."
        elif shares:
            legend = MIXED_LEGEND
            scale = ("Percent columns are colored on a fixed 0% (red) to 100% (blue) scale; "
                     "other columns show each cell as a z-score within its column, full "
                     "color at 2 SDs.")
        else:
            legend = Z_LEGEND
            scale = "Colors show each cell as a z-score within its column, full color at 2 SDs."
        self.add_heatmap_views(
            ctx, views, out, slug,
            titles={C.ABSOLUTE: title},
            help_texts={C.ABSOLUTE: f"{help_text} {scale}"},
            names={m: names[m] for m in metrics},
            short_names={m: (short_names or names).get(m, names[m]) for m in metrics},
            column_tips=column_tips,
            groups=groups,
            fixed={m: SHARE_RANGE for m in shares},
            legends={C.ABSOLUTE: legend},
            value_labels={C.ABSOLUTE: value_label},
            decimals=decimals,
            column_decimals=column_decimals,
            column_units={C.ABSOLUTE: units},
        )
        return views

    def result(self, ctx: AnalysisContext, out: MetricViews, views: C.BehaviorViews | None,
               hidden: list[str], summary: str, figures: dict | None = None,
               extra: dict | None = None) -> AnalysisResult:
        # Every view is absolute, so the behavior note on why there is no
        # relative view would only mislead.
        shared = {k: v for k, v in (views.metadata() if views is not None else {}).items()
                  if k != "relative_view"}
        metadata = {
            **shared,
            "hidden": list(hidden),
            "views": C.views_metadata(out.declared),
            "heatmaps": out.heatmaps,
            **(extra or {}),
        }
        return AnalysisResult(tables=out.tables, figures=figures or {}, summary=summary,
                              metadata=metadata)

    def empty_result(self, hidden: list[str]) -> AnalysisResult:
        return AnalysisResult(summary=NO_WORKSPACE, metadata={"hidden": list(hidden)})

    def skip_labels(self, ctx: AnalysisContext) -> set[str]:
        return {ctx.catalog.null_label, ctx.catalog.vanilla_label}
