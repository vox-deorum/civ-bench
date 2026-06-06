#!/usr/bin/env python3
"""Score-based heuristic predictor: softmax of ``score_ratio`` within each group.

Ported verbatim from ``../vox-deorum-analysis/models/models/score_model.py``.
No training required — applies ``score_ratio**exponent`` normalized within each
``(game_id, turn)`` group.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Set

from .base_predictor import BasePredictor


class ScorePredictor(BasePredictor):
    """Naive baseline: P(win) = softmax(score_ratio**exponent) within each group."""

    DISABLE_RESAMPLING = True
    REQUIRES_ID_COLUMNS = ["game_id", "turn"]
    SUPPORTED_FEATURES: Optional[Set[str]] = {"score_ratio"}
    DEFAULT_FEATURES: Optional[List[str]] = ["score_ratio"]

    def __init__(
        self,
        exponent: float = 4.236,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        super().__init__(include_features, exclude_features, random_state)
        self.exponent = exponent

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> "ScorePredictor":
        self._filter_features(X)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.selected_features_ is None:
            raise ValueError("Model must be fitted before making predictions")

        scores = X["score_ratio"].values
        groups = X.groupby(["game_id", "turn"], sort=False)

        probs_win = np.empty(len(X), dtype=np.float64)
        for _, idx in groups.indices.items():
            s = scores[idx]
            s2 = s ** self.exponent
            probs_win[idx] = s2 / s2.sum()

        probs_loss = 1 - probs_win
        return np.column_stack([probs_loss, probs_win])

    def get_feature_importance(self) -> None:
        return None

    def get_model_summary(self) -> dict:
        return {
            "model_type": f"Score (score_ratio^{self.exponent})",
            "n_features": 1,
            "exponent": self.exponent,
            "description": f"P(win) = score_ratio^{self.exponent} / sum within each (game_id, turn) group",
        }

    # ── Save / Load ─────────────────────────────────────────────────────────
    def _get_hyperparams(self) -> dict:
        return {"exponent": self.exponent}

    def _save_model_state(self, dir_path: Path) -> None:
        state = {"exponent": self.exponent}
        with open(dir_path / "model.json", "w") as f:
            json.dump(state, f)

    @classmethod
    def _load_model_state(cls, dir_path: Path, metadata: dict) -> "ScorePredictor":
        hp = metadata.get("hyperparams", {})
        instance = cls(
            exponent=hp.get("exponent", 4.236),
            include_features=metadata.get("include_features"),
            exclude_features=metadata.get("exclude_features"),
            random_state=metadata.get("random_state", 42),
        )
        with open(dir_path / "model.json", "r") as f:
            state = json.load(f)
        instance.exponent = state["exponent"]
        instance.selected_features_ = metadata.get("selected_features", ["score_ratio"])
        return instance
