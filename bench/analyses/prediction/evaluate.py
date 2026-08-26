"""``prediction.evaluate`` — multi-metric scoring table per estimator.

Ported from the comparison logic in ``predict/visualize_model_comparison.ipynb``
+ ``models/compare_models.py``: for each resolved estimator (``uses.estimators``
when supplied, otherwise every enabled estimator), load its ``predictions.csv``
and compute the configured metrics over every prediction row. One table row per
model; columns are the metrics plus ``n_rows`` / ``n_games``.
"""

from __future__ import annotations

import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .metrics import (
    DEFAULT_METRICS,
    LOWER_IS_BETTER,
    compute_metric,
    filtered_prediction_rows,
)


class PredictionEvaluate(Analysis):
    module = "prediction.evaluate"
    friendly_name = "Prediction quality"
    description = (
        "Measures how well each estimator identifies likely winners and matches "
        "observed outcomes (discrimination and calibration)."
    )
    report_defaults = {"tables": ["metrics"], "figures": []}
    default_all_estimators = True

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        estimators = ctx.uses_estimators()
        if not estimators:
            raise AnalysisError(
                f"prediction.evaluate '{self.stage_id}': requires at least one "
                f"enabled estimator or uses.estimators override."
            )
        metrics = list(self.params.get("metrics") or DEFAULT_METRICS)

        rows = []
        for est in estimators:
            df = filtered_prediction_rows(ctx, ctx.load_predictions(est))
            rec = {"model": est, "n_rows": int(len(df)), "n_games": int(df["game_id"].nunique())}
            for m in metrics:
                rec[m] = compute_metric(m, df)
            rows.append(rec)

        table = pd.DataFrame(rows, columns=["model", "n_rows", "n_games", *metrics])
        fig = self._plot(table, metrics, ctx)

        # Headline: best model on the first metric.
        first = metrics[0]
        ascending = first in LOWER_IS_BETTER
        ranked = table.sort_values(first, ascending=ascending)
        best = ranked.iloc[0]
        summary = (
            f"{best['model']} performs best on {first} at {best[first]:.4f} across "
            f"{len(estimators)} estimator(s) and {len(metrics)} quality metric(s)."
        )
        return AnalysisResult(
            tables={"metrics": table},
            figures={"metrics": fig} if fig is not None else {},
            summary=summary,
            metadata={"metrics": metrics, "n_models": len(estimators)},
        )

    def _plot(self, table: pd.DataFrame, metrics: list[str], ctx: AnalysisContext):
        import matplotlib.pyplot as plt

        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(max(4, 3.2 * n), 4.2), squeeze=False)
        colors = ctx.catalog.prediction_model_colors()
        for ax, m in zip(axes[0], metrics):
            sub = table.sort_values(m, ascending=(m not in LOWER_IS_BETTER))
            bar_colors = [colors.get(name, "#4C72B0") for name in sub["model"]]
            ax.barh(sub["model"], sub[m], color=bar_colors)
            ax.set_title(m, fontsize=10)
            ax.grid(True, axis="x", alpha=0.3)
        fig.suptitle("Prediction metrics by estimator", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig
