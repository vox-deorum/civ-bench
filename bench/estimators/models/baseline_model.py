#!/usr/bin/env python3
"""Baseline logistic-regression predictor with cluster-robust SEs + calibration.

Ported from ``../vox-deorum-analysis/models/models/baseline_model.py``. The only
change is the feature-default import, which now resolves from
:mod:`bench.estimators.features` instead of the old ``utils.data_utils``.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as SklearnLR
import statsmodels.api as sm
from typing import Dict, Any, Optional, List

from .base_predictor import BasePredictor
from scipy.special import expit
from ..features import get_selected_feature_names


class _LogitSnapshot:
    """Lightweight stand-in for statsmodels LogitResults (prediction + reporting)."""

    def __init__(self, d: dict):
        self.params = d["params"]
        self.bse = d["bse"]
        self.tvalues = d["tvalues"]
        self.pvalues = d["pvalues"]
        self._conf_int = d["conf_int"]
        self.mle_retvals = d["mle_retvals"]
        self.llf = d["llf"]
        self.prsquared = d["prsquared"]
        self.aic = d["aic"]
        self.bic = d["bic"]

    def predict(self, X):
        return expit(X @ self.params)

    def conf_int(self):
        return self._conf_int


class BaselineVictoryPredictor(BasePredictor):
    """Logistic regression (statsmodels) with standardized features + calibration."""

    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = [f for f in get_selected_feature_names() if f != "turn_progress"]
    REQUIRED_FEATURES = None

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
    ):
        super().__init__(include_features, exclude_features, random_state)
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.scaler = StandardScaler()
        self.model_results = None
        self.feature_names = None
        self.calibrator_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> "BaselineVictoryPredictor":
        X_filtered = self._filter_features(X)
        self.feature_names = list(X_filtered.columns)

        X_scaled = self.scaler.fit_transform(X_filtered)
        X_scaled_const = sm.add_constant(X_scaled)
        y_array = y.values if isinstance(y, pd.Series) else y

        logit_model = sm.Logit(y_array, X_scaled_const)

        if clusters is not None:
            cluster_array = clusters.values if isinstance(clusters, pd.Series) else clusters
            try:
                self.model_results = logit_model.fit(
                    disp=False, method="bfgs", maxiter=1000,
                    cov_type="cluster",
                    cov_kwds={"groups": cluster_array, "use_correction": True},
                )
            except (np.linalg.LinAlgError, ValueError, AttributeError):
                print("Warning: Cluster-robust SE calculation failed. "
                      "Falling back to HC1 robust standard errors.")
                try:
                    self.model_results = logit_model.fit(
                        disp=False, method="bfgs", maxiter=1000, cov_type="HC1",
                    )
                except (np.linalg.LinAlgError, ValueError, AttributeError):
                    print("Warning: All robust SE methods failed. Using non-robust standard errors.")
                    self.model_results = logit_model.fit(disp=False, method="bfgs", maxiter=1000)
        else:
            self.model_results = logit_model.fit(disp=False, method="bfgs", maxiter=1000)

        if self.calibrate:
            self._fit_calibrator(X_filtered, y_array, clusters)

        return self

    def _fit_calibrator(self, X_filtered: pd.DataFrame, y_array: np.ndarray,
                        clusters: Optional[pd.Series]) -> None:
        n_splits = 5
        if clusters is not None:
            cluster_array = clusters.values if isinstance(clusters, pd.Series) else clusters
            kf = GroupKFold(n_splits=n_splits)
            split_iter = kf.split(X_filtered, y_array, groups=cluster_array)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_iter = kf.split(X_filtered, y_array)

        oof_predictions = np.full(len(y_array), np.nan)

        for train_idx, val_idx in split_iter:
            temp_scaler = StandardScaler()
            X_cal_scaled = temp_scaler.fit_transform(X_filtered.iloc[train_idx])
            X_cal_const = sm.add_constant(X_cal_scaled)

            temp_model = sm.Logit(y_array[train_idx], X_cal_const)
            temp_results = temp_model.fit(disp=False, method="bfgs", maxiter=1000)

            X_val_scaled = temp_scaler.transform(X_filtered.iloc[val_idx])
            X_val_const = sm.add_constant(X_val_scaled, has_constant="add")
            oof_predictions[val_idx] = temp_results.predict(X_val_const)

        if self.calibration_method == "isotonic":
            self.calibrator_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            self.calibrator_.fit(oof_predictions, y_array)
        elif self.calibration_method == "sigmoid":
            eps = 1e-15
            p = np.clip(oof_predictions, eps, 1 - eps)
            logit_preds = np.log(p / (1 - p))
            self.calibrator_ = SklearnLR(C=1e10, solver="lbfgs", max_iter=1000)
            self.calibrator_.fit(logit_preds.reshape(-1, 1), y_array)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_results is None:
            raise ValueError("Model must be fitted before making predictions")
        if self.selected_features_ is None:
            raise ValueError("Model was not properly fitted (selected_features_ is None)")

        X_filtered = X[self.selected_features_]
        X_scaled = self.scaler.transform(X_filtered)
        X_scaled_const = sm.add_constant(X_scaled, has_constant="add")

        probs_win = self.model_results.predict(X_scaled_const)

        if self.calibrator_ is not None:
            if self.calibration_method == "isotonic":
                probs_win = self.calibrator_.predict(probs_win)
            elif self.calibration_method == "sigmoid":
                eps = 1e-15
                p = np.clip(probs_win, eps, 1 - eps)
                logit_preds = np.log(p / (1 - p))
                probs_win = self.calibrator_.predict_proba(logit_preds.reshape(-1, 1))[:, 1]

        probs_loss = 1 - probs_win
        return np.column_stack([probs_loss, probs_win])

    def get_feature_importance(self, use_robust_se: bool = True) -> pd.DataFrame:
        if self.model_results is None:
            raise ValueError("Model must be fitted before getting feature importance")

        coefficients = self.model_results.params[1:]  # Skip intercept
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        })

        if use_robust_se:
            importance_df["robust_se"] = self.model_results.bse[1:]
            conf_int = self.model_results.conf_int()
            importance_df["robust_ci_lower"] = conf_int[1:, 0]
            importance_df["robust_ci_upper"] = conf_int[1:, 1]
            importance_df["z_statistic"] = self.model_results.tvalues[1:]
            importance_df["p_value"] = self.model_results.pvalues[1:]
            importance_df["significant_95"] = importance_df["p_value"] < 0.05

        return importance_df.sort_values("abs_coefficient", ascending=False)

    def get_model_summary(self) -> Dict[str, Any]:
        if self.model_results is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            "model_type": "Logit (statsmodels)",
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "intercept": float(self.model_results.params[0]),
            "converged": self.model_results.mle_retvals["converged"],
            "n_iterations": self.model_results.mle_retvals.get("iterations", "N/A"),
            "log_likelihood": float(self.model_results.llf),
            "pseudo_r_squared": float(self.model_results.prsquared),
            "aic": float(self.model_results.aic),
            "bic": float(self.model_results.bic),
            "calibrated": self.calibrate,
            "calibration_method": self.calibration_method if self.calibrate else None,
        }

    # ── Save / Load ─────────────────────────────────────────────────────────
    def _get_hyperparams(self) -> dict:
        return {
            "calibrate": self.calibrate,
            "calibration_method": self.calibration_method,
        }

    def _save_model_state(self, dir_path: Path) -> None:
        r = self.model_results
        conf_int = r.conf_int()
        model_snapshot = {
            "params": np.array(r.params),
            "bse": np.array(r.bse),
            "tvalues": np.array(r.tvalues),
            "pvalues": np.array(r.pvalues),
            "conf_int": np.array(conf_int),
            "mle_retvals": r.mle_retvals,
            "llf": float(r.llf),
            "prsquared": float(r.prsquared),
            "aic": float(r.aic),
            "bic": float(r.bic),
        }
        state = {
            "model_snapshot": model_snapshot,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "calibrator": self.calibrator_,
        }
        with open(dir_path / "model.pkl", "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def _load_model_state(cls, dir_path: Path, metadata: dict) -> "BaselineVictoryPredictor":
        hp = metadata.get("hyperparams", {})
        instance = cls(
            include_features=metadata.get("include_features"),
            exclude_features=metadata.get("exclude_features"),
            random_state=metadata.get("random_state", 42),
            calibrate=hp.get("calibrate", False),
            calibration_method=hp.get("calibration_method", "isotonic"),
        )
        with open(dir_path / "model.pkl", "rb") as f:
            state = pickle.load(f)
        if "model_snapshot" in state:
            instance.model_results = _LogitSnapshot(state["model_snapshot"])
        else:
            instance.model_results = state["model_results"]
        instance.scaler = state["scaler"]
        instance.feature_names = state["feature_names"]
        instance.calibrator_ = state.get("calibrator", None)
        instance.selected_features_ = metadata["selected_features"]
        return instance
