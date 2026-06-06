#!/usr/bin/env python3
"""XGBoost predictor with probability calibration.

Ported from ``../vox-deorum-analysis/models/models/xgboost_model.py``. The source
guarded the imports behind a ``HAS_XGBOOST`` try/except; per AGENTS.md (all deps
are mandatory, no soft-fail) xgboost/sklearn are imported directly here.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from .base_predictor import BasePredictor


class XGBoostPredictor(BasePredictor):
    """XGBoost classifier with optional probability calibration."""

    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = None
    REQUIRED_FEATURES = None
    DISABLE_RESAMPLING = True

    @classmethod
    def optuna_default_params(cls):
        """Return current __init__ defaults as Optuna raw trial parameters."""
        import inspect
        sig = inspect.signature(cls.__init__)
        d = {k: v.default for k, v in sig.parameters.items()
             if v.default is not inspect.Parameter.empty}
        return {k: d[k] for k in [
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "gamma", "reg_lambda",
            "calibrate", "calibration_method",
        ]}

    @staticmethod
    def convert_optuna_params(raw_params):
        """Convert Optuna raw params to model __init__ kwargs."""
        return dict(raw_params)

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.189227,
        subsample: float = 0.909952,
        colsample_bytree: float = 0.821711,
        min_child_weight: int = 9,
        gamma: float = 2.50405e-05,
        reg_lambda: float = 16.8929,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        reg_alpha: float = 0,
        early_stopping_rounds: Optional[int] = 10,
        eval_fraction: float = 0.1,
    ):
        super().__init__(include_features, exclude_features, random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction = eval_fraction
        self.best_iteration_ = None
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> "XGBoostPredictor":
        X_filtered = self._filter_features(X)
        self.feature_names = list(X_filtered.columns)

        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        xgb_params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )

        if self.early_stopping_rounds is not None:
            X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
                X_filtered, y,
                test_size=self.eval_fraction,
                random_state=self.random_state,
                stratify=y,
            )

            es_model = xgb.XGBClassifier(
                early_stopping_rounds=self.early_stopping_rounds,
                **xgb_params,
            )
            es_model.fit(X_train_es, y_train_es, eval_set=[(X_val_es, y_val_es)], verbose=False)
            self.best_iteration_ = es_model.best_iteration

            if self.calibrate:
                optimal_params = xgb_params.copy()
                optimal_params["n_estimators"] = self.best_iteration_ + 1

                base_model = xgb.XGBClassifier(**optimal_params)
                self.model = CalibratedClassifierCV(
                    base_model, method=self.calibration_method, cv=5, n_jobs=-1,
                )
                self.model.fit(X_filtered, y)
            else:
                self.model = es_model
        else:
            self.best_iteration_ = None
            base_model = xgb.XGBClassifier(**xgb_params)
            if self.calibrate:
                self.model = CalibratedClassifierCV(
                    base_model, method=self.calibration_method, cv=5, n_jobs=-1,
                )
            else:
                self.model = base_model
            self.model.fit(X_filtered, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        if self.selected_features_ is None:
            raise ValueError("Model was not properly fitted (selected_features_ is None)")
        X_filtered = X[self.selected_features_]
        return self.model.predict_proba(X_filtered)

    def get_parameter_count(self) -> Optional[int]:
        if self.model is None:
            return None
        if self.calibrate:
            estimators = [cc.estimator for cc in self.model.calibrated_classifiers_]
        else:
            estimators = [self.model]
        total = 0
        for est in estimators:
            booster = est.get_booster()
            trees_df = booster.trees_to_dataframe()
            total += len(trees_df)
        return total

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        if self.calibrate:
            importances_list = []
            for calibrated_classifier in self.model.calibrated_classifiers_:
                base_estimator = calibrated_classifier.estimator
                importances_list.append(base_estimator.feature_importances_)
            importances = np.mean(importances_list, axis=0)
        else:
            importances = self.model.feature_importances_

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": importances,
            "abs_coefficient": np.abs(importances),
        })
        return importance_df.sort_values("abs_coefficient", ascending=False)

    def get_model_summary(self) -> dict:
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            "model_type": "XGBoostClassifier" + (" (Calibrated)" if self.calibrate else ""),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "calibrated": self.calibrate,
            "calibration_method": self.calibration_method if self.calibrate else None,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "early_stopping_rounds": self.early_stopping_rounds,
            "best_iteration": self.best_iteration_,
            "effective_n_estimators": (self.best_iteration_ + 1) if self.best_iteration_ is not None else self.n_estimators,
        }

    # ── Save / Load ─────────────────────────────────────────────────────────
    def _get_hyperparams(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "calibrate": self.calibrate,
            "calibration_method": self.calibration_method,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_fraction": self.eval_fraction,
        }

    def _save_model_state(self, dir_path: Path) -> None:
        state = {
            "model": self.model,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration_,
        }
        with open(dir_path / "model.pkl", "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def _load_model_state(cls, dir_path: Path, metadata: dict) -> "XGBoostPredictor":
        hp = metadata["hyperparams"]
        instance = cls(
            include_features=metadata.get("include_features"),
            exclude_features=metadata.get("exclude_features"),
            random_state=metadata.get("random_state", 42),
            **hp,
        )
        with open(dir_path / "model.pkl", "rb") as f:
            state = pickle.load(f)
        instance.model = state["model"]
        instance.feature_names = state["feature_names"]
        instance.best_iteration_ = state["best_iteration"]
        instance.selected_features_ = metadata["selected_features"]
        return instance
