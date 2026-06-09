"""Prediction metric functions (ported from ``models/`` evaluator logic).

The exact sklearn calls the old ``compare_models.py`` / ``model_evaluator.py``
used, isolated here so ``prediction.evaluate`` and the calibration loss views
share one definition. ``balanced_accuracy`` reproduces the legacy **group-argmax**
hard prediction: within each ``(game_id, turn)`` group the highest-probability
player is the predicted winner (1), everyone else 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from ..errors import AnalysisError


def _group_argmax_preds(df: pd.DataFrame, prob_col: str) -> np.ndarray:
    """Hard 0/1 predictions: 1 for the per-(game_id, turn) argmax probability.

    Mirrors ``compare_models.py``: the predicted winner of each decision point is
    the player with the highest predicted probability.
    """
    groups = [df["game_id"], df["turn"]]
    p = pd.Series(df[prob_col].to_numpy(), index=df.index)
    preds = np.zeros(len(df), dtype=np.int64)
    winner_idx = p.groupby(groups, sort=False).idxmax()
    idx_to_pos = pd.Series(np.arange(len(df)), index=df.index)
    preds[idx_to_pos[winner_idx].to_numpy()] = 1
    return preds


def compute_metric(
    name: str,
    df: pd.DataFrame,
    y_col: str = "is_winner",
    prob_col: str = "predicted_win_probability",
) -> float:
    """Compute one metric over a predictions frame (NaN if undefined)."""
    y_true = df[y_col].to_numpy()
    y_prob = df[prob_col].to_numpy()
    if name == "roc_auc":
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    if name == "brier_score":
        return float(brier_score_loss(y_true, y_prob))
    if name == "log_loss":
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    if name == "balanced_accuracy":
        if "game_id" not in df.columns or "turn" not in df.columns:
            preds = (y_prob >= 0.5).astype(int)
        else:
            preds = _group_argmax_preds(df, prob_col)
        return float(balanced_accuracy_score(y_true, preds))
    if name == "accuracy":
        preds = (y_prob >= 0.5).astype(int)
        return float(accuracy_score(y_true, preds))
    raise ValueError(f"unknown prediction metric '{name}'")


# Metrics whose better direction is "lower" (losses). Everything else: higher better.
LOWER_IS_BETTER = {"brier_score", "log_loss"}

DEFAULT_METRICS = ["roc_auc", "brier_score", "log_loss", "balanced_accuracy"]


def filtered_prediction_rows(ctx, df: pd.DataFrame) -> pd.DataFrame:
    """Attach filter metadata when available, then apply the analysis filter."""
    out = df
    if "player_type" not in out.columns and {"game_id", "player_id"} <= set(out.columns):
        try:
            panel = ctx.load_table("panel")[
                ["game_id", "player_id", "player_type"]
            ].drop_duplicates(["game_id", "player_id"])
        except (AnalysisError, FileNotFoundError, KeyError, ValueError):
            panel = None
        if panel is not None:
            out = out.merge(panel, on=["game_id", "player_id"], how="left")
    return ctx.apply_filter(out)
