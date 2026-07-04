#!/usr/bin/env python3
"""Abstract base class for victory-prediction models.

Ported verbatim from ``../vox-deorum-analysis/models/models/base_predictor.py``.
Provides the unified train/predict/feature-filter interface plus the save/load
contract (``metadata.json`` + per-class state) the registry dispatches on.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set, Dict, Any
from pathlib import Path
import json
import pandas as pd
import numpy as np
import fnmatch


class BasePredictor(ABC):
    """
    Abstract base class for all victory prediction models.

    Subclasses must implement:
    - fit(X, y, clusters): Train the model
    - predict_proba(X): Return probability predictions [n_samples, 2]

    Optional methods:
    - get_feature_importance(): Return feature importance DataFrame
    - get_model_summary(): Return model metadata dictionary

    Class-level attributes to define (optional):
    - SUPPORTED_FEATURES: Set of all features this model can use (None = all)
    - DEFAULT_FEATURES: Default feature list if none specified (None = all)
    - REQUIRED_FEATURES: Features that must be included (None = none required)
    - DISABLE_RESAMPLING: If True, skip resampling even when requested (default: False)
    - REQUIRES_ID_COLUMNS: List of ID columns needed in X (e.g., ['game_id', 'turn', 'player_id'])
    """

    SUPPORTED_FEATURES: Optional[Set[str]] = None
    DEFAULT_FEATURES: Optional[List[str]] = None
    REQUIRED_FEATURES: Optional[Set[str]] = None
    DISABLE_RESAMPLING: bool = False
    REQUIRES_ID_COLUMNS: Optional[List[str]] = None
    FILTER_ZERO_SCORE: bool = True  # filter eliminated players before training

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.include_features = include_features
        self.exclude_features = exclude_features if exclude_features else []
        self.random_state = random_state
        self.selected_features_: Optional[List[str]] = None  # Set during fit

    def _expand_wildcards(self, patterns: List[str], available_features: List[str]) -> List[str]:
        """Expand include/exclude patterns to a deterministic, de-duplicated list.

        Literal patterns keep their declared order; wildcard patterns expand in
        ``available_features`` (data-column) order. First occurrence wins. Returning
        a list rather than a set is what makes the selected feature order — and hence
        the fitted column order and predictions — byte-stable across runs; a set's
        iteration order over strings is hash-randomized (PYTHONHASHSEED).
        """
        matched: List[str] = []
        seen: Set[str] = set()
        for pattern in patterns:
            if "*" in pattern or "?" in pattern:
                candidates = fnmatch.filter(available_features, pattern)
            else:
                candidates = [pattern]
            for feat in candidates:
                if feat not in seen:
                    seen.add(feat)
                    matched.append(feat)
        return matched

    def _filter_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply include/exclude logic to a feature matrix and store selection."""
        available_features = list(X.columns)

        if self.SUPPORTED_FEATURES is not None:
            available_features = [f for f in available_features if f in self.SUPPORTED_FEATURES]

        if self.include_features is not None:
            included = self._expand_wildcards(self.include_features, available_features)
            missing = set(included) - set(available_features)
            if missing:
                raise ValueError(f"Requested features not available in data: {missing}")
            selected = included
        elif self.DEFAULT_FEATURES is not None:
            missing = [f for f in self.DEFAULT_FEATURES if f not in available_features]
            if missing:
                raise ValueError(f"DEFAULT_FEATURES not found in data: {missing}")
            selected = list(self.DEFAULT_FEATURES)
        else:
            selected = available_features

        if self.exclude_features:
            excluded = set(self._expand_wildcards(self.exclude_features, selected))
            selected = [f for f in selected if f not in excluded]

        if self.REQUIRED_FEATURES is not None:
            missing_required = self.REQUIRED_FEATURES - set(selected)
            if missing_required:
                raise ValueError(f"Required features missing after filtering: {missing_required}")

        self.selected_features_ = selected
        return X[selected].copy()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> "BasePredictor":
        """Fit the model on training data. Returns self."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict victory probabilities, shape (n_samples, 2) = [P(loss), P(win)]."""
        ...

    def predict(self, X: pd.DataFrame, groups=None) -> np.ndarray:
        """Predict binary outcomes (argmax within groups, else 0.5 threshold)."""
        proba = self.predict_proba(X)[:, 1]
        if groups is None:
            return (proba >= 0.5).astype(np.int64)
        p = pd.Series(proba, index=X.index)
        preds = np.zeros(len(X), dtype=np.int64)
        winner_idx = p.groupby(groups, sort=False).idxmax()
        idx_to_pos = pd.Series(np.arange(len(X)), index=X.index)
        preds[idx_to_pos[winner_idx].values] = 1
        return preds

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return None

    def get_model_summary(self) -> Optional[Dict[str, Any]]:
        return None

    def get_parameter_count(self) -> Optional[int]:
        return None

    def get_selected_features(self) -> Optional[List[str]]:
        return self.selected_features_

    # ── Save / Load ─────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save a fitted model to a directory (metadata.json + per-class state)."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_class": self.__class__.__name__,
            "selected_features": self.selected_features_,
            "random_state": self.random_state,
            "include_features": self.include_features,
            "exclude_features": self.exclude_features,
            "hyperparams": self._get_hyperparams(),
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self._save_model_state(save_dir)

    def _get_hyperparams(self) -> dict:
        return {}

    def _save_model_state(self, dir_path: Path) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _save_model_state(). "
            f"Save/load is not supported for this model."
        )

    @classmethod
    def _load_model_state(cls, dir_path: Path, metadata: dict) -> "BasePredictor":
        raise NotImplementedError(
            f"{cls.__name__} does not implement _load_model_state(). "
            f"Save/load is not supported for this model."
        )
