"""``calibration.cell_baseline`` (controlled-design only) — implicit-vs-explicit
cell-baseline heatmap.

Consumes the adjust stage's ``cell_baseline.csv`` (+ ``cell_coverage.csv`` for
missing-cell marking) and renders, **per seed**, a diverging heatmap whose rows
are conditions (``experiment`` — the implicit per-experiment baselines) and whose
columns are that seed's ``(seed, player_id)`` cells, each labelled with the
seat-bound ``civilization``. Cell colour = ``cell_baseline`` on the logit scale,
with shared robust symmetric limits across the per-seed facets (percentile-clipped
so the enforced-winner sentinel ``logit(1-eps) ≈ 11.51`` does not saturate); each
cell is annotated with ``n_vanilla`` (the baseline's support), and missing cells
(``cell_coverage.missing`` / ``n_vanilla == 0``) are hatched.

The explicit pathway — the single shared baseline spanning the whole grid — is
pinned as a top reference row separated by a rule, so each implicit condition reads
against it directly. An implicit-only run (``baseline_experiment:null``) omits that
reference row. Returns an empty result (no figures) when there are no controlled
rows (``cell_baseline.csv`` empty / absent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...plotting.heatmap import plot_diverging_heatmap, robust_symmetric_limit
from ...stats.transforms import LOGIT_EPS, logit
from ..base import Analysis, AnalysisContext, AnalysisResult
from .civ_effects import _adjust_table_id

# The enforced-winner / clip sentinel on the logit scale (relative_strength == 1-eps).
_SENTINEL = float(logit(np.array([1.0 - LOGIT_EPS]))[0])
_SENTINEL_TOL = 0.5
_EXPLICIT_ROW = "(explicit baseline)"


class CalibrationCellBaseline(Analysis):
    module = "calibration.cell_baseline"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        adjust_dir = Path(ctx.adjust_dir(_adjust_table_id(ctx)))
        baseline_path = adjust_dir / "cell_baseline.csv"
        if not baseline_path.exists():
            return AnalysisResult(summary="No cell_baseline.csv (adjust stage not run).")
        cb = pd.read_csv(baseline_path)
        if cb.empty:
            return AnalysisResult(
                summary="cell_baseline.csv is empty (no controlled rows); nothing to render."
            )

        coverage = self._load_coverage(adjust_dir)
        # Shared, robust symmetric colour limit across every seed facet.
        vlim = robust_symmetric_limit(cb["cell_baseline"].to_numpy(), pct=98.0)

        figures = {}
        for seed in sorted(cb["seed"].unique()):
            fig = self._plot_seed(cb[cb["seed"] == seed], coverage, int(seed), vlim)
            if fig is not None:
                figures[f"cell_baseline_seed_{int(seed)}"] = fig

        n_explicit = int((cb["pathway"] == "explicit").sum())
        summary = (
            f"Cell-baseline heatmaps for {cb['seed'].nunique()} seed(s); "
            f"{cb['experiment'].nunique()} condition(s), "
            f"{'with' if n_explicit else 'without'} an explicit reference row."
        )
        return AnalysisResult(
            tables={"cell_baseline": cb},
            figures=figures,
            summary=summary,
            metadata={"n_seeds": int(cb["seed"].nunique()), "has_explicit": bool(n_explicit)},
        )

    def _load_coverage(self, adjust_dir: Path) -> Optional[pd.DataFrame]:
        path = adjust_dir / "cell_coverage.csv"
        if not path.exists():
            return None
        cov = pd.read_csv(path)
        return cov if not cov.empty else None

    def _plot_seed(self, seed_cb: pd.DataFrame, coverage, seed: int, vlim: float):
        player_ids = sorted(seed_cb["player_id"].unique())
        seat_civ = {
            pid: (seed_cb[seed_cb["player_id"] == pid]["civilization"].dropna().iloc[0]
                  if not seed_cb[seed_cb["player_id"] == pid]["civilization"].dropna().empty else "")
            for pid in player_ids
        }
        col_labels = [f"p{pid}\n{seat_civ[pid]}" for pid in player_ids]

        # Rows: explicit reference (pinned top, if present) then implicit conditions.
        implicit = seed_cb[seed_cb["pathway"] == "implicit"]
        explicit = seed_cb[seed_cb["pathway"] == "explicit"]
        row_specs: list[tuple[str, str, pd.DataFrame]] = []
        rule_after = None
        if not explicit.empty:
            row_specs.append((_EXPLICIT_ROW, "explicit", explicit))
            rule_after = 0
        for exp in sorted(implicit["experiment"].unique()):
            row_specs.append((exp, "implicit", implicit[implicit["experiment"] == exp]))
        if not row_specs:
            return None

        n_rows, n_cols = len(row_specs), len(player_ids)
        values = np.full((n_rows, n_cols), np.nan)
        annot = np.full((n_rows, n_cols), "", dtype=object)
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        sentinel = np.zeros((n_rows, n_cols), dtype=bool)
        for i, (_, pathway, rows) in enumerate(row_specs):
            by_pid = {int(r.player_id): r for r in rows.itertuples()}
            for j, pid in enumerate(player_ids):
                r = by_pid.get(pid)
                missing = r is None or int(getattr(r, "n_vanilla", 0)) == 0
                if pathway == "implicit" and coverage is not None:
                    missing = missing or self._coverage_missing(coverage, rows, seed, pid)
                if missing or r is None:
                    mask[i, j] = True
                    continue
                values[i, j] = r.cell_baseline
                annot[i, j] = f"{r.cell_baseline:.2f}\nn={int(r.n_vanilla)}"
                if abs(float(r.cell_baseline) - _SENTINEL) <= _SENTINEL_TOL:
                    sentinel[i, j] = True

        index = [name for name, _, _ in row_specs]
        fig, _ = plot_diverging_heatmap(
            pd.DataFrame(values, index=index, columns=col_labels),
            annot=pd.DataFrame(annot, index=index, columns=col_labels),
            vlim=vlim,
            mask=pd.DataFrame(mask, index=index, columns=col_labels),
            sentinel_mask=pd.DataFrame(sentinel, index=index, columns=col_labels),
            title=f"Cell baseline (logit scale) — seed {seed}",
            cbar_label="cell_baseline (logit)",
            xlabel="(seed, player_id) cell → civilization",
            ylabel="condition",
            rule_after_row=rule_after,
        )
        return fig

    @staticmethod
    def _coverage_missing(coverage: pd.DataFrame, rows: pd.DataFrame, seed: int, pid: int) -> bool:
        exp = rows["experiment"].iloc[0] if not rows.empty else None
        if exp is None:
            return True
        hit = coverage[
            (coverage["experiment"] == exp)
            & (coverage["seed"] == seed)
            & (coverage["player_id"] == pid)
        ]
        if hit.empty:
            return False
        return bool(hit["missing"].iloc[0]) or int(hit["n_vanilla"].iloc[0]) == 0
