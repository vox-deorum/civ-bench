"""``performance.strength_panel`` — per-identity strength summary + coverage report.

Consumes the adjust stage's ``strength`` table and reports, per identity (``by``,
default ``player_type``), the mean of ``metric`` (default ``adjusted_strength``)
with its per-identity ``n_games`` and a nonparametric bootstrap CI, flagging
identities below ``min_games_preliminary`` as **preliminary**. When that param is
omitted, the threshold is inherited from the enabled ``ratings.*`` stages'
``min_games`` (the rating-cutoff this run actually uses; max across them when
several exist), falling back to 5 when no ratings stage supplies one.

This module also **owns** the controlled-design cell-coverage report: it renders
``cell_coverage.csv`` (which ``(seed, player_id)`` cells of the entirety each
controlled experiment is missing) once, here, as its completeness table —
``calibration.cell_baseline`` only *consumes* that file to mark missing cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .turn_predicted import _strength_table_id


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


class PerformanceStrengthPanel(Analysis):
    module = "performance.strength_panel"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        table_id = _strength_table_id(ctx)
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
        rows = []
        for ident, grp in panel.groupby(by):
            vals = grp[metric].dropna().to_numpy()
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
        summary_tbl = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)

        tables = {"by_identity": summary_tbl}
        coverage = self._load_coverage(ctx, table_id)
        if coverage is not None:
            tables["cell_coverage_summary"] = self._coverage_summary(coverage)
            tables["cell_coverage"] = coverage

        fig = self._plot(summary_tbl, by, metric, ctx)
        n_prelim = int(summary_tbl["preliminary"].sum())
        coverage_note = "."
        if coverage is not None:
            missing = int(coverage["missing"].fillna(False).astype(bool).sum())
            no_baseline = int(
                (
                    (coverage["n_rows"].fillna(0) > 0)
                    & (coverage["n_vanilla"].fillna(0) == 0)
                ).sum()
            )
            coverage_note = (
                f"; cell-coverage report has {len(coverage)} cells, "
                f"{missing} missing, {no_baseline} without Vanilla baseline."
            )
        summary = (
            f"{len(summary_tbl)} identities by {metric}; {n_prelim} preliminary "
            f"(< {min_games_prelim} games)"
            + coverage_note
        )
        return AnalysisResult(
            tables=tables, figures={"strength": fig} if fig is not None else {},
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

    def _plot(self, tbl: pd.DataFrame, by: str, metric: str, ctx: AnalysisContext):
        import matplotlib.pyplot as plt

        from ...plotting.styles import get_player_color

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
        for i, prelim in enumerate(order["preliminary"]):
            if prelim:
                ax.text(0, i, " *", va="center", ha="left", color="darkred", fontsize=10)
        ax.set_yticks(list(y))
        ax.set_yticklabels(order[by])
        ax.set_xlabel(f"mean {metric}")
        ax.set_title(f"{metric} by {by} (bootstrap CI; * = preliminary)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig
