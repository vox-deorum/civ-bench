"""``calibration.loss_by_progress`` — prediction loss binned by game progress.

Ported from ``predict/visualize_loss_by_progress.ipynb`` +
``models/model_evaluator.py``: bin predictions by ``turn_progress`` into
``n_bins`` equal-width bins and compute the configured loss metrics (default
Brier + log-loss) per bin per estimator. The legacy version aggregated across
CV folds; here we score the in-sample predictions directly per bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from ..prediction.metrics import compute_metric, filtered_prediction_rows

_DEFAULT_METRICS = ["brier_score", "log_loss"]


class CalibrationLossByProgress(Analysis):
    module = "calibration.loss_by_progress"
    friendly_name = "Prediction error over time"
    description = (
        "Tracks win-probability error from the opening turns through the end of "
        "the game (Brier score and log loss by game progress)."
    )
    report_defaults = {"tables": [], "figures": ["loss_by_progress"]}
    default_all_estimators = True

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        estimators = ctx.uses_estimators()
        if not estimators:
            raise AnalysisError(
                f"calibration.loss_by_progress '{self.stage_id}': requires at least one "
                f"enabled estimator or uses.estimators override."
            )
        n_bins = int(self.params.get("n_bins", 10))
        metrics = list(self.params.get("metrics") or _DEFAULT_METRICS)
        edges = np.linspace(0, 1, n_bins + 1)
        labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(n_bins)]

        rows = []
        for est in estimators:
            df = filtered_prediction_rows(ctx, ctx.load_predictions(est)).copy()
            df["bin"] = pd.cut(df["turn_progress"], bins=edges, labels=labels,
                               right=False, include_lowest=True)
            df.loc[df["turn_progress"] >= edges[-1], "bin"] = labels[-1]
            for label, grp in df.groupby("bin", observed=True):
                if len(grp) < 2:
                    continue
                rec = {"model": est, "turn_progress_bin": str(label), "n_samples": int(len(grp))}
                for m in metrics:
                    rec[m] = compute_metric(m, grp)
                rows.append(rec)

        table = pd.DataFrame(rows, columns=["model", "turn_progress_bin", "n_samples", *metrics])
        fig = self._plot(table, metrics, labels, ctx)
        summary = (
            f"Prediction error is tracked across {n_bins} stages of game progress for "
            f"{len(estimators)} estimator(s) using {', '.join(metrics)}."
        )
        return AnalysisResult(
            tables={"loss_by_progress": table},
            figures={"loss_by_progress": fig} if fig is not None else {},
            summary=summary,
            metadata={"n_bins": n_bins, "metrics": metrics},
        )

    def _plot(self, table: pd.DataFrame, metrics: list[str], labels: list[str], ctx: AnalysisContext):
        import matplotlib.pyplot as plt

        colors = ctx.catalog.prediction_model_colors()
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(max(5, 5 * n), 4.5), squeeze=False)
        for ax, m in zip(axes[0], metrics):
            for est, grp in table.groupby("model"):
                grp = grp.set_index("turn_progress_bin").reindex(labels).reset_index()
                ax.plot(grp["turn_progress_bin"], grp[m], marker="o",
                        color=colors.get(est, None), label=est)
            ax.set_xlabel("Turn progress")
            ax.set_ylabel(m)
            ax.set_title(f"{m} by turn progress", fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig
