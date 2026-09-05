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

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError


COMPLETENESS_COLUMNS = [
    "experiment", "required_games", "present_games", "missing_games",
    "completeness_pct", "repeated_slots", "warning",
]
REPEATED_GAMES_COLUMNS = [
    "experiment", "seed", "seating_rotation", "n_games", "game_ids",
    "keep_candidate_game_id", "extra_game_ids",
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


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _controlled_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"experiment", "game_id", "seed", "seating_rotation", "player_id"}
    if not required <= set(panel.columns):
        return pd.DataFrame(columns=list(panel.columns))
    if "controlled" in panel.columns:
        mask = _truthy(panel["controlled"])
    else:
        mask = (panel["seed"].fillna(-1).astype(int) != -1) & (
            panel["seating_rotation"].fillna(-1).astype(int) != -1
        )
    out = panel[mask].copy()
    if out.empty:
        return out
    out["seed"] = out["seed"].astype(int)
    out["seating_rotation"] = out["seating_rotation"].astype(int)
    out["player_id"] = out["player_id"].astype(int)
    out["game_id"] = out["game_id"].astype(str)
    out["experiment"] = out["experiment"].astype(str)
    return out


def _game_sort_frame(games: pd.DataFrame | None) -> pd.DataFrame:
    if games is None or games.empty or "game_id" not in games.columns:
        return pd.DataFrame(columns=["game_id", "_timestamp_num", "_timestamp_str"])
    cols = ["game_id"] + (["timestamp"] if "timestamp" in games.columns else [])
    out = games[cols].drop_duplicates("game_id").copy()
    out["game_id"] = out["game_id"].astype(str)
    if "timestamp" in out.columns:
        out["_timestamp_num"] = pd.to_numeric(out["timestamp"], errors="coerce")
        out["_timestamp_str"] = out["timestamp"].astype(str)
    else:
        out["_timestamp_num"] = np.nan
        out["_timestamp_str"] = ""
    return out[["game_id", "_timestamp_num", "_timestamp_str"]]


def _ordered_game_ids(ids: list[str], games: pd.DataFrame | None) -> list[str]:
    base = pd.DataFrame({"game_id": sorted(set(map(str, ids)))})
    if base.empty:
        return []
    order = base.merge(_game_sort_frame(games), on="game_id", how="left")
    order["_timestamp_missing"] = order["_timestamp_num"].isna()
    order["_timestamp_str"] = order["_timestamp_str"].fillna("")
    order = order.sort_values(
        ["_timestamp_missing", "_timestamp_num", "_timestamp_str", "game_id"],
        kind="mergesort",
    )
    return order["game_id"].tolist()


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
) -> dict[str, pd.DataFrame]:
    """Compact controlled-design game completeness diagnostics.

    The summary counts distinct games, while completeness is based on occupied
    seed/rotation slots so repeated game ids in one slot cannot mask gaps.
    """
    controlled = _controlled_panel(panel)
    if controlled.empty:
        return {}

    reference_slots = _reference_slots(controlled, baseline_experiment)
    if not reference_slots:
        return {}
    reference_set = set(reference_slots)
    expected_per_cell = len({rot for _, rot in reference_slots})

    summary_rows: list[dict] = []
    repeated_rows: list[dict] = []
    gap_rows: list[dict] = []
    cell_issue_rows: list[dict] = []

    for experiment, exp_df in controlled.groupby("experiment", sort=True):
        slot_groups = exp_df.groupby(["seed", "seating_rotation"], sort=True)["game_id"]
        present_slots: set[tuple[int, int]] = set()
        repeated_slots = 0
        for (seed, rot), ids in slot_groups:
            key = (int(seed), int(rot))
            present_slots.add(key)
            ordered = _ordered_game_ids(ids.dropna().astype(str).unique().tolist(), games)
            if len(ordered) > 1:
                repeated_slots += 1
                repeated_rows.append({
                    "experiment": experiment,
                    "seed": int(seed),
                    "seating_rotation": int(rot),
                    "n_games": int(len(ordered)),
                    "game_ids": ",".join(ordered),
                    "keep_candidate_game_id": ordered[0],
                    "extra_game_ids": ",".join(ordered[1:]),
                })

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

        cell_counts = (
            exp_df.groupby(["seed", "player_id"], sort=True)["game_id"]
            .nunique()
            .reset_index(name="n_games")
        )
        for seed, seed_counts in cell_counts.groupby("seed", sort=True):
            bad = seed_counts[seed_counts["n_games"] != expected_per_cell]
            if bad.empty:
                continue
            observed = ", ".join(
                f"{int(r.player_id)}:{int(r.n_games)}"
                for r in bad.itertuples(index=False)
            )
            cell_issue_rows.append({
                "experiment": experiment,
                "seed": int(seed),
                "expected_games_per_cell": int(expected_per_cell),
                "observed_counts": observed,
                "affected_player_ids": ",".join(str(int(v)) for v in bad["player_id"]),
            })

        required = int(len(reference_slots))
        present_games = int(exp_df["game_id"].nunique())
        missing_games = int(len(reference_set - present_slots))
        cell_repeat_issue = bool(not cell_counts.empty and (
            cell_counts["n_games"] != expected_per_cell
        ).any())
        warnings = []
        if missing_games:
            warnings.append(f"{missing_games} missing slot(s)")
        if repeated_slots:
            warnings.append(f"{repeated_slots} repeated slot(s)")
        if cell_repeat_issue:
            warnings.append(
                f"cell repeat counts differ from expected {expected_per_cell}"
            )
        summary_rows.append({
            "experiment": experiment,
            "required_games": required,
            "present_games": present_games,
            "missing_games": missing_games,
            "completeness_pct": round((required - missing_games) / required, 4)
            if required else float("nan"),
            "repeated_slots": int(repeated_slots),
            "warning": "; ".join(warnings) if warnings else "ok",
        })

    return {
        "experiment_completeness": pd.DataFrame(summary_rows, columns=COMPLETENESS_COLUMNS),
        "repeated_games": pd.DataFrame(repeated_rows, columns=REPEATED_GAMES_COLUMNS),
        "experiment_completeness_gaps": pd.DataFrame(
            gap_rows, columns=COMPLETENESS_GAPS_COLUMNS
        ),
        "cell_repeat_issues": pd.DataFrame(cell_issue_rows, columns=CELL_REPEAT_ISSUES_COLUMNS),
    }


class PerformanceStrengthPanel(Analysis):
    module = "performance.strength_panel"
    friendly_name = "Adjusted-strength summary"
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
        adv_note = ""
        advantage_col = "cell_logit_advantage"
        if advantage_col in panel.columns and panel[advantage_col].notna().any():
            cell = panel[panel[advantage_col].notna()].copy()
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
            adv_note = f"logit-advantage view over {len(cell)} cell-baselined rows"
            if by == "player_type":
                vrow = adv_tbl[adv_tbl[by] == ctx.catalog.vanilla_label]
                if not vrow.empty:
                    adv_note += f"; Vanilla mean = {float(vrow['mean'].iloc[0]):+.3f}"

        coverage = self._load_coverage(ctx, table_id)
        if coverage is not None:
            tables["cell_coverage_summary"] = self._coverage_summary(coverage)
            tables["cell_coverage"] = coverage
        n_prelim = int(summary_tbl["preliminary"].sum())
        report_notes = []
        if coverage is not None:
            missing = int(coverage["missing"].fillna(False).astype(bool).sum())
            no_baseline = int(
                (
                    (coverage["n_rows"].fillna(0) > 0)
                    & (coverage["n_vanilla"].fillna(0) == 0)
                ).sum()
            )
            report_notes.append(
                f"cell-coverage report has {len(coverage)} cells, "
                f"{missing} missing, {no_baseline} without Vanilla baseline"
            )
        summary = (
            f"The strength panel covers {len(summary_tbl)} identities by {metric}, "
            f"including {n_prelim} preliminary result(s) with fewer than "
            f"{min_games_prelim} games"
        )
        if report_notes:
            summary += "; " + "; ".join(report_notes)
        if adv_note:
            summary += "; " + adv_note
        summary += "."
        return AnalysisResult(
            tables=tables, figures=figures,
            summary=summary, metadata={"by": by, "metric": metric},
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
