"""``prediction.compare`` — cross-estimator agreement.

Ported from ``predict/visualize_model_comparison.ipynb``: merge every referenced
estimator's predictions on ``(game_id, player_id, turn)`` and report how much the
models agree — pairwise probability correlation (R²) and within-decision rank
agreement (Spearman ρ on the per-``(game_id, turn)`` player ranking). No
parametric hypothesis test (the source computes correlations + heatmaps only).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from .metrics import filtered_prediction_rows


class PredictionCompare(Analysis):
    module = "prediction.compare"
    friendly_name = "Predictor comparison"
    description = "Compares the win-probability estimates of the enabled estimators side by side."
    report_defaults = {"tables": [], "figures": ["rank_agreement"]}
    default_all_estimators = True

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        estimators = ctx.uses_estimators()
        if len(estimators) < 2:
            raise AnalysisError(
                f"prediction.compare '{self.stage_id}': needs >= 2 resolved estimators "
                f"to compare (got {estimators})."
            )

        merged = None
        for est in estimators:
            df = filtered_prediction_rows(ctx, ctx.load_predictions(est))[
                ["game_id", "player_id", "turn", "predicted_win_probability"]
            ].rename(columns={"predicted_win_probability": f"pred_{est}"})
            merged = df if merged is None else merged.merge(
                df, on=["game_id", "player_id", "turn"], how="inner"
            )
        if merged is None or merged.empty:
            raise AnalysisError(
                f"prediction.compare '{self.stage_id}': no overlapping "
                f"(game_id, player_id, turn) rows across the estimators."
            )

        pred_cols = [f"pred_{e}" for e in estimators]

        # Within-decision player rank per model (1 = most likely winner).
        rank_cols = {}
        for est, col in zip(estimators, pred_cols):
            rc = f"rank_{est}"
            merged[rc] = merged.groupby(["game_id", "turn"])[col].rank(ascending=False)
            rank_cols[est] = rc

        prob_corr = pd.DataFrame(np.eye(len(estimators)), index=estimators, columns=estimators)
        rank_corr = pd.DataFrame(np.ones((len(estimators), len(estimators))), index=estimators, columns=estimators)
        pair_rows = []
        for a, b in combinations(estimators, 2):
            r = float(np.corrcoef(merged[f"pred_{a}"], merged[f"pred_{b}"])[0, 1])
            prob_corr.loc[a, b] = prob_corr.loc[b, a] = r ** 2
            r1 = merged[rank_cols[a]].to_numpy()
            r2 = merged[rank_cols[b]].to_numpy()
            mask = ~(np.isnan(r1) | np.isnan(r2))
            rho = float(spearmanr(r1[mask], r2[mask]).statistic) if mask.sum() > 1 else float("nan")
            rank_corr.loc[a, b] = rank_corr.loc[b, a] = rho
            pair_rows.append({"model_a": a, "model_b": b, "prob_r2": r ** 2, "rank_spearman": rho})

        pairs = pd.DataFrame(pair_rows)
        fig = self._plot(rank_corr)

        summary = (
            f"Compared {len(estimators)} estimators over {len(merged):,} shared rows; "
            f"mean within-decision rank agreement (Spearman ρ) = "
            f"{pairs['rank_spearman'].mean():.3f}."
        )
        return AnalysisResult(
            tables={
                "pairs": pairs,
                "prob_correlation": prob_corr.reset_index(names="model"),
                "rank_agreement": rank_corr.reset_index(names="model"),
            },
            figures={"rank_agreement": fig} if fig is not None else {},
            summary=summary,
            metadata={"n_models": len(estimators), "n_rows": int(len(merged))},
        )

    def _plot(self, rank_corr: pd.DataFrame):
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(rank_corr) + 3), max(4, 0.8 * len(rank_corr) + 2)))
        sns.heatmap(
            rank_corr, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, center=0.5,
            square=True, linewidths=0.5, linecolor="gray",
            cbar_kws={"label": "Spearman ρ"}, ax=ax,
        )
        ax.set_title("Within-decision rank agreement (Spearman ρ)", fontsize=12, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        return fig
