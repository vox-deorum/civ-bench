"""``calibration.reliability`` — reliability diagram + ECE per estimator.

Ported from ``predict/visualize_calibration.ipynb``: bin predictions into
``n_bins`` equal-width probability bins, and per bin report mean predicted
probability vs observed win frequency (and the bin count). Adds the expected
calibration error (ECE) as a scalar summary — the count-weighted mean
|mean_pred − mean_true| over occupied bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from ..prediction.metrics import filtered_prediction_rows


def reliability_table(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    pred = df["predicted_win_probability"].to_numpy()
    true = df["is_winner"].to_numpy()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(pred, bin_edges[1:-1])
    rows = []
    for i in range(n_bins):
        sel = bin_indices == i
        count = int(sel.sum())
        rows.append({
            "bin": i,
            "bin_center": float(bin_centers[i]),
            "mean_pred": float(pred[sel].mean()) if count else float("nan"),
            "mean_true": float(true[sel].mean()) if count else float("nan"),
            "count": count,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    occ = table[table["count"] > 0]
    if occ.empty:
        return float("nan")
    w = occ["count"].to_numpy()
    gap = (occ["mean_pred"] - occ["mean_true"]).abs().to_numpy()
    return float((w * gap).sum() / w.sum())


class CalibrationReliability(Analysis):
    module = "calibration.reliability"
    friendly_name = "Reliability"
    description = "Reliability diagram and expected calibration error (ECE) of estimator win probabilities."
    report_defaults = {"tables": ["ece"], "figures": ["reliability"]}
    default_all_estimators = True

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        estimators = ctx.uses_estimators()
        if not estimators:
            raise AnalysisError(
                f"calibration.reliability '{self.stage_id}': requires at least one "
                f"enabled estimator or uses.estimators override."
            )
        n_bins = int(self.params.get("n_bins", 10))

        per_model = {}
        ece_rows = []
        for est in estimators:
            df = filtered_prediction_rows(ctx, ctx.load_predictions(est))
            tbl = reliability_table(df, n_bins)
            tbl.insert(0, "model", est)
            per_model[est] = tbl
            ece_rows.append({"model": est, "ece": expected_calibration_error(tbl), "n_rows": int(len(df))})

        reliability = pd.concat(per_model.values(), ignore_index=True)
        ece = pd.DataFrame(ece_rows)
        fig = self._plot(per_model, ctx)

        best = ece.sort_values("ece").iloc[0]
        summary = (
            f"Reliability over {len(estimators)} estimator(s), {n_bins} bins; "
            f"lowest ECE = {best['ece']:.4f} ({best['model']})."
        )
        return AnalysisResult(
            tables={"reliability": reliability, "ece": ece},
            figures={"reliability": fig} if fig is not None else {},
            summary=summary,
            metadata={"n_bins": n_bins},
        )

    def _plot(self, per_model: dict, ctx: AnalysisContext):
        import matplotlib.pyplot as plt

        colors = ctx.catalog.prediction_model_colors()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
        for est, tbl in per_model.items():
            occ = tbl[tbl["count"] > 0]
            ax.plot(occ["mean_pred"], occ["mean_true"], marker="o", markersize=4,
                    color=colors.get(est, None), label=est)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed win rate")
        ax.set_title("Reliability diagram", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
