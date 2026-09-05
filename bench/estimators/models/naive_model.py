#!/usr/bin/env python3
"""Naive baseline: always predict the training-set win rate.

Ported verbatim from ``../vox-deorum-analysis/models/models/naive_model.py``.
The simplest possible baseline: a constant ``P(win) = positives / total``.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from .base_predictor import BasePredictor


class NaivePredictor(BasePredictor):
    """Constant-probability baseline (predicts the training win rate for all rows)."""

    DISABLE_RESAMPLING = True

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        super().__init__(include_features, exclude_features, random_state)
        self.win_rate_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> "NaivePredictor":
        y_array = y.values if isinstance(y, pd.Series) else y
        self.win_rate_ = float(np.mean(y_array))
        self.selected_features_ = []
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.win_rate_ is None:
            raise ValueError("Model must be fitted before making predictions")
        n_samples = len(X)
        probs_win = np.full(n_samples, self.win_rate_)
        probs_loss = 1 - probs_win
        return np.column_stack([probs_loss, probs_win])

    def get_feature_importance(self) -> None:
        return None

    def get_model_summary(self) -> dict:
        if self.win_rate_ is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            "model_type": "Naive (constant probability)",
            "win_rate": self.win_rate_,
            "n_features": 0,
            "description": f"Always predicts P(win) = {self.win_rate_:.4f}",
        }

    # ── Save / Load ─────────────────────────────────────────────────────────
    def _save_model_state(self, dir_path: Path) -> None:
        state = {"win_rate": self.win_rate_}
        with open(dir_path / "model.json", "w") as f:
            json.dump(state, f)

    @classmethod
    def _load_model_state(cls, dir_path: Path, metadata: dict) -> "NaivePredictor":
        instance = cls(
            include_features=metadata.get("include_features"),
            exclude_features=metadata.get("exclude_features"),
            random_state=metadata.get("random_state", 42),
        )
        with open(dir_path / "model.json", "r") as f:
            state = json.load(f)
        instance.win_rate_ = state["win_rate"]
        instance.selected_features_ = metadata.get("selected_features", [])
        return instance
