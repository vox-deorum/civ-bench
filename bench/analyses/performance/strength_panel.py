"""``performance.strength_panel``: per-identity strength summary + coverage report.

Consumes the adjust stage's ``strength`` table and reports, per identity (``by``,
default ``player_type``), the mean of ``metric`` (default ``adjusted_strength``)
with its per-identity ``n_games`` and a nonparametric bootstrap CI, flagging
identities below ``min_games_preliminary`` as **preliminary**. When that param is
omitted, the threshold is inherited from the enabled ``ratings.*`` stages'
``min_games`` (the rating-cutoff this run actually uses; max across them when
several exist), falling back to 5 when no ratings stage supplies one.

This module also **owns** the controlled-design cell-coverage report: it renders
``cell_coverage.csv`` (which ``(seed, player_id)`` cells of the entirety each
controlled experiment is missing) once, here, as its completeness table:
``calibration.cell_baseline`` only *consumes* that file to mark missing cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bench.data.loading import drop_problem_games
from bench.data.repeats import (
    REPEATED_GAMES_COLUMNS, SeatingCycle, align_rotations, controlled_rows, repeated_slots,
)
from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError


COMPLETENESS_COLUMNS = [
    "experiment", "required_games", "present_games", "missing_games",
    "completeness_pct", "repeated_slots", "off_cycle_games", "excluded_games", "warning",
]
COMPLETENESS_GAPS_COLUMNS = [
    "experiment", "seed", "missing_rotations", "n_missing",
]
CELL_REPEAT_ISSUES_COLUMNS = [
    "experiment", "seed", "expected_games_per_cell", "observed_counts",
    "affected_player_ids",
]


def _ratings_min_games_default(ctx: AnalysisContext) -> int:
    """Inherit the preliminary threshold from the run's ratings cutoff.

    Use the max ``min_games`` across enabled ``ratings.*`` analyses (the cutoff
    this run actually rates against); fall back to 5 when none supplies a
    positive cutoff, so the preliminary flag always carries a meaningful floor.
    """
    cutoffs = [
        int((s.raw.get("params") or {}).get("min_games", 0) or 0)
        for s in ctx.config.analyses
        if s.enabled and s.module and s.module.startswith("ratings.")
    ]
    best = max(cutoffs) if cutoffs else 0
    return best if best > 0 else 5


def _bootstrap_ci(values: np.ndarray, n: int, ci_level: float, rng: np.random.Generator):
    if len(values) < 2 or n < 1:
        return float("nan"), float("nan")
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _summarize_by(
    df: pd.DataFrame,
    by: str,
    value_col: str,
    min_games_prelim: int,
    boot_n: int,
    ci_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Per-identity mean of ``value_col`` with bootstrap CI and a preliminary flag.

    Shared by the main ``by_identity`` summary and the controlled-mode logit-advantage
    view so both report identical columns and use the same bootstrap machinery.
    """
    rows = []
    for ident, grp in df.groupby(by):
        vals = grp[value_col].dropna().to_numpy()
        n_games = int(grp["game_id"].nunique()) if "game_id" in grp.columns else int(len(grp))
        lo, hi = _bootstrap_ci(vals, boot_n, ci_level, rng)
        rows.append({
            by: ident,
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "n_rows": int(len(vals)),
            "n_games": n_games,
            "ci_lower": lo,
            "ci_upper": hi,
            "preliminary": bool(n_games < min_games_prelim),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def _reference_slots(controlled: pd.DataFrame, baseline_experiment: str | None) -> list[tuple[int, int]]:
    ref = controlled
    if baseline_experiment is not None:
        explicit = controlled[controlled["experiment"] == str(baseline_experiment)]
        if not explicit.empty:
            ref = explicit
    seeds = sorted(ref["seed"].dropna().astype(int).unique())
    rotations = sorted(ref["seating_rotation"].dropna().astype(int).unique())
    return [(int(seed), int(rot)) for seed in seeds for rot in rotations]


def build_experiment_completeness(
    panel: pd.DataFrame,
    games: pd.DataFrame | None = None,
    baseline_experiment: str | None = None,
    excluded_game_ids=None,
    seats: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compact controlled-design game completeness diagnostics.

    The summary counts distinct games, while completeness is based on occupied
    seed/rotation slots so repeated game ids in one slot cannot mask gaps. Slots
    come from the seats each game was played in (:func:`align_rotations`), so a
    game whose recorded rotation names other seats is placed where it belongs, or
    listed in ``off_cycle_games`` when it fits no rotation.
    ``seats`` is the frame configured seats are read from; it defaults to
    ``panel`` and should be the unfiltered panel, so a player filter cannot hide seats.
    """
    controlled = controlled_rows(panel)
    if controlled.empty:
        return {}

    reference_slots = _reference_slots(controlled, baseline_experiment)
    if not reference_slots:
        return {}
    reference_set = set(reference_slots)
    accepted_all = drop_problem_games(controlled, excluded_game_ids)
    aligned_all, off_cycle, cycles = align_rotations(
        controlled, panel if seats is None else seats, games, excluded_game_ids,
    )

    summary_rows: list[dict] = []
    repeated_rows: list[dict] = []
    gap_rows: list[dict] = []
    cell_issue_rows: list[dict] = []

    for experiment, exp_df in controlled.groupby("experiment", sort=True):
        accepted = accepted_all[accepted_all["experiment"] == experiment]
        excluded_games = exp_df["game_id"].nunique() - accepted["game_id"].nunique()
        aligned = aligned_all[aligned_all["experiment"] == experiment]
        n_off_cycle = int((off_cycle["experiment"] == experiment).sum())
        present_slots = {
            (int(seed), int(rot))
            for seed, rot in aligned[["seed", "seating_rotation"]].drop_duplicates().itertuples(index=False)
        }
        repeats = repeated_slots(aligned, games)
        repeated_rows.extend(repeats)
        n_repeated = len(repeats)

        missing_by_seed: dict[int, list[int]] = {}
        for seed, rot in sorted(reference_set - present_slots):
            missing_by_seed.setdefault(int(seed), []).append(int(rot))
        for seed, rotations in sorted(missing_by_seed.items()):
            gap_rows.append({
                "experiment": experiment,
                "seed": seed,
                "missing_rotations": ",".join(str(r) for r in sorted(rotations)),
                "n_missing": int(len(rotations)),
            })

        # A full cycle seats each config slot once in every seat per seed, so each
        # seat holds a treatment player in exactly len(config_slots) games.
        cycle = cycles.get(str(experiment))
        expected_per_cell = len(cycle.config_slots) if cycle else 0
        seat_issues = _seat_balance_issues(accepted, cycle) if cycle else []
        for seed, bad in seat_issues:
            cell_issue_rows.append({
                "experiment": experiment,
                "seed": int(seed),
                "expected_games_per_cell": int(expected_per_cell),
                "observed_counts": ", ".join(f"{pid}:{n}" for pid, n in bad.items()),
                "affected_player_ids": ",".join(str(pid) for pid in bad),
            })

        required = int(len(reference_slots))
        present_games = int(accepted["game_id"].nunique())
        missing_games = int(len(reference_set - present_slots))
        warnings = []
        if excluded_games:
            warnings.append(f"{excluded_games} game(s) excluded by decision failure cutoff")
        if missing_games:
            warnings.append(f"{missing_games} missing slot(s)")
        if n_repeated:
            warnings.append(f"{n_repeated} repeated slot(s)")
        if n_off_cycle:
            warnings.append(f"{n_off_cycle} off-cycle game(s)")
        if seat_issues:
            warnings.append(
                f"treatment seat counts differ from expected {expected_per_cell}"
            )
        summary_rows.append({
            "experiment": experiment,
            "required_games": required,
            "present_games": present_games,
            "missing_games": missing_games,
            "completeness_pct": round((required - missing_games) / required, 4)
            if required else float("nan"),
            "repeated_slots": int(n_repeated),
            "off_cycle_games": n_off_cycle,
            "excluded_games": int(excluded_games),
            "warning": "; ".join(warnings) if warnings else "ok",
        })

    return {
        "experiment_completeness": pd.DataFrame(summary_rows, columns=COMPLETENESS_COLUMNS),
        "repeated_games": pd.DataFrame(repeated_rows, columns=REPEATED_GAMES_COLUMNS),
        "off_cycle_games": off_cycle.reset_index(drop=True),
        "experiment_completeness_gaps": pd.DataFrame(
            gap_rows, columns=COMPLETENESS_GAPS_COLUMNS
        ),
        "cell_repeat_issues": pd.DataFrame(cell_issue_rows, columns=CELL_REPEAT_ISSUES_COLUMNS),
    }


def _seat_balance_issues(accepted: pd.DataFrame, cycle: SeatingCycle) -> list[tuple[int, dict]]:
    """Per seed, the seats whose treatment-game count is not ``len(config_slots)``.

    Counts every accepted game as played, including repeats and off-cycle games,
    so it shows the imbalance the analyses currently see. A seed is listed only
    when some seat is over-filled; seats that are merely short belong to slots
    the gap table already reports.
    """
    if "config_slot" not in accepted.columns:
        return []
    slots = pd.to_numeric(accepted["config_slot"], errors="coerce")
    treatment = accepted[slots >= 0]
    expected = len(cycle.config_slots)
    issues = []
    for seed, rows in treatment.groupby("seed", sort=True):
        counts = rows.groupby("player_id")["game_id"].nunique()
        counts = counts.reindex(range(cycle.total_seats), fill_value=0)
        bad = {int(pid): int(n) for pid, n in counts.items() if n != expected}
        if (counts > expected).any():
            issues.append((int(seed), bad))
    return issues


class PerformanceStrengthPanel(Analysis):
    module = "performance.strength_panel"
    friendly_name = "Gameplay strength"
    description = (
        "Summarizes model-adjusted strength, uncertainty, and experiment coverage "
        "for each player identity (bootstrap confidence intervals)."
    )
    report_defaults = {"tables": [], "figures": ["strength"]}

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        table_id = ctx.strength_table_id()
        metric = self.params.get("metric", "adjusted_strength")
        by = self.params.get("by", "player_type")
        min_games_prelim = int(self.params.get(
            "min_games_preliminary", _ratings_min_games_default(ctx)
        ))
        boot_n = int(self.params.get("bootstrap_n", 1000))
        ci_level = float(self.params.get("ci_level", 0.95))

        panel = ctx.load_table(table_id)
        panel = ctx.apply_filter(panel)
        if metric not in panel.columns or by not in panel.columns:
            raise AnalysisError(
                f"performance.strength_panel '{self.stage_id}': need columns "
                f"'{metric}' and '{by}' in the strength table."
            )

        rng = np.random.default_rng(ctx.config.seed)
        summary_tbl = _summarize_by(
            panel, by, metric, min_games_prelim, boot_n, ci_level, rng
        )

        tables = {"by_identity": summary_tbl}
        figures = {}
        fig = self._plot_bars(
            summary_tbl, by, ctx, f"mean {metric}",
            f"{metric} by {by} (bootstrap CI; * = preliminary)",
        )
        if fig is not None:
            figures["strength"] = fig

        # Controlled-mode alternative view: per-identity mean of the exact start-cell
        # advantage `cell_logit_advantage` (= logit_strength - cell_baseline, persisted
        # by the adjust stage before any post_cell_normalize), i.e. "your strength -
        # matched Vanilla VPAI baseline in this cell" on the logit scale. NaN for
        # non-cell rows, so this view only appears in controlled runs. Vanilla is NOT
        # forced to 0; it is summarized like every identity so the report shows where
        # it actually lands.
        n_advantage_rows = 0
        advantage_col = "cell_logit_advantage"
        if advantage_col in panel.columns and panel[advantage_col].notna().any():
            cell = panel[panel[advantage_col].notna()].copy()
            n_advantage_rows = int(len(cell))
            adv_tbl = _summarize_by(
                cell, by, advantage_col, min_games_prelim, boot_n, ci_level, rng
            )
            tables["by_identity_logit_advantage"] = adv_tbl
            adv_fig = self._plot_bars(
                adv_tbl, by, ctx, "logit advantage vs Vanilla cell baseline",
                f"logit advantage by {by} (0 = cell baseline; bootstrap CI; * = preliminary)",
                center_zero=True,
            )
            if adv_fig is not None:
                figures["logit_advantage"] = adv_fig
        coverage = self._load_coverage(ctx, table_id)
        if coverage is not None:
            tables["cell_coverage_summary"] = self._coverage_summary(coverage)
            tables["cell_coverage"] = coverage
        n_prelim = int(summary_tbl["preliminary"].sum())
        coverage_metadata = None
        if coverage is not None:
            missing = int(coverage["missing"].fillna(False).astype(bool).sum())
            no_baseline = int(
                (
                    (coverage["n_rows"].fillna(0) > 0)
                    & (coverage["n_vanilla"].fillna(0) == 0)
                ).sum()
            )
            coverage_metadata = {
                "cells": int(len(coverage)),
                "missing_cells": missing,
                "cells_without_vanilla_baseline": no_baseline,
            }
        metric_label = str(metric).replace("_", " ")
        finite = summary_tbl[np.isfinite(summary_tbl["mean"])]
        if finite.empty:
            summary = f"No finite mean {metric_label} values are available."
        else:
            high = finite.iloc[0]
            low = finite.iloc[-1]
            if len(finite) == 1:
                summary = f"Mean {metric_label}: **{high[by]}** at **{high['mean']:.3f}**."
            else:
                summary = (
                    f"Highest mean {metric_label}: **{high[by]}** at **{high['mean']:.3f}**. "
                    f"Identity means range from **{low['mean']:.3f}** to **{high['mean']:.3f}**."
                )
        return AnalysisResult(
            tables=tables, figures=figures,
            summary=summary,
            metadata={
                "by": by,
                "metric": metric,
                "preliminary_identities": n_prelim,
                "preliminary_minimum_games": min_games_prelim,
                "cell_coverage": coverage_metadata,
                "logit_advantage_rows": n_advantage_rows,
            },
        )

    def _load_coverage(self, ctx: AnalysisContext, table_id: str):
        path = Path(ctx.adjust_dir(table_id)) / "cell_coverage.csv"
        if not path.exists():
            return None
        cov = pd.read_csv(path)
        return cov if not cov.empty else None

    @staticmethod
    def _coverage_summary(cov: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for experiment, grp in cov.groupby("experiment", sort=True):
            missing = grp["missing"].fillna(False).astype(bool)
            n_rows = grp["n_rows"].fillna(0)
            n_vanilla = grp["n_vanilla"].fillna(0)
            has_baseline = grp["has_baseline"].fillna(False).astype(bool)
            n_cells = int(len(grp))
            present_cells = int((~missing).sum())
            rows.append({
                "experiment": experiment,
                "n_cells": n_cells,
                "present_cells": present_cells,
                "missing_cells": int(missing.sum()),
                "baseline_cells": int(has_baseline.sum()),
                "no_vanilla_baseline_cells": int(((n_rows > 0) & (n_vanilla == 0)).sum()),
                "coverage_pct": round(present_cells / n_cells, 4) if n_cells else float("nan"),
            })
        return pd.DataFrame(rows)

    def _plot_bars(
        self,
        tbl: pd.DataFrame,
        by: str,
        ctx: AnalysisContext,
        xlabel: str,
        title: str,
        *,
        center_zero: bool = False,
    ):
        """Horizontal per-identity bar chart with bootstrap-CI whiskers.

        Shared by the main metric view and the controlled-mode logit-advantage view;
        ``center_zero`` adds the dashed baseline line for the (signed) advantage plot.
        """
        import matplotlib.pyplot as plt

        from ...plotting.styles import get_player_color

        if tbl.empty:
            return None
        pairing = ctx.condition_pairing() if by == "player_type" else None
        if pairing is not None:
            from ...plotting.pairing import plot_paired_rows

            return plot_paired_rows(
                tbl,
                catalog=ctx.catalog,
                spec=pairing,
                value_col="mean",
                lo_col="ci_lower",
                hi_col="ci_upper",
                identity_col=by,
                ref_line=0 if center_zero else None,
                preliminary_col="preliminary",
                ascending=False,
                xlabel=xlabel,
                title=title,
            )
        n = len(tbl)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * n + 1.5)))
        order = tbl.iloc[::-1]  # highest at top
        y = range(len(order))
        colors = [get_player_color(ctx.catalog, str(v)) if by == "player_type" else "#4C72B0"
                  for v in order[by]]
        xerr = np.vstack([
            (order["mean"] - order["ci_lower"]).clip(lower=0).fillna(0).to_numpy(),
            (order["ci_upper"] - order["mean"]).clip(lower=0).fillna(0).to_numpy(),
        ])
        ax.barh(list(y), order["mean"], color=colors, alpha=0.85)
        ax.errorbar(order["mean"], list(y), xerr=xerr, fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=3)
        if center_zero:
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        for i, prelim in enumerate(order["preliminary"]):
            if prelim:
                ax.text(0, i, " *", va="center", ha="left", color="darkred", fontsize=10)
        ax.set_yticks(list(y))
        ax.set_yticklabels(order[by])
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig
