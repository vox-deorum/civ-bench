#!/usr/bin/env python3
"""Plain MLP predictor (per-sample binary cross-entropy).

Ported verbatim from ``../vox-deorum-analysis/models/models/mlp_model.py``. Uses
the same residual ``_UtilityNet`` as the grouped MLP but trains/predicts per-row
(sigmoid), not group-softmax.
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_torch_predictor import BaseTorchPredictor
from .grouped_mlp_model import (
    _UtilityNet,
    _layer_sizes_convert_optuna,
    _layer_sizes_optuna_defaults,
    _utilitynet_feature_importance,
)


class MLPPredictor(BaseTorchPredictor):
    """PyTorch MLP for victory prediction (per-sample BCE)."""

    SUPPORTED_FEATURES = None
    DEFAULT_FEATURES = [
        "science_share", "culture_share", "tourism_share", "gold_share",
        "food_share", "military_share",
        "cities_share", "population_share", "votes_share",
        "faith_raw_share", "production_raw_share",
        "highest_war_weariness", "active_wars", "truces", "defensive_pacts", "friendships",
        "minor_allies_share",
        "technologies_gap", "policies_gap",
        "happiness_percentage", "military_utilization", "religion_percentage",
        "turn_progress", "score_ratio",
    ]
    REQUIRED_FEATURES = None

    @classmethod
    def optuna_default_params(cls):
        return _layer_sizes_optuna_defaults(cls)

    @staticmethod
    def convert_optuna_params(raw_params):
        return _layer_sizes_convert_optuna(raw_params)

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        layer_sizes: tuple = (60,),
        dropout: float = 0.447875,
        lr: float = 0.00275382,
        weight_decay: float = 0.000280091,
        epochs: int = 27,
        loss_tp_alpha: float = 0.413903,
        batch_size: int = 32768,
        device: Optional[str] = None,
    ):
        super().__init__(
            include_features=include_features,
            exclude_features=exclude_features,
            random_state=random_state,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            loss_tp_alpha=loss_tp_alpha,
            device=device,
        )
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size

    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None) -> "MLPPredictor":
        self._seed_torch()
        X_filtered = self._filter_features(X)
        self.feature_names = list(X_filtered.columns)

        Xmat = X_filtered.to_numpy(dtype=np.float32)
        Xmat = self._standardize_fit(Xmat)
        ymat = y.to_numpy(dtype=np.float32)

        d = Xmat.shape[1]
        n = Xmat.shape[0]

        self.model = _UtilityNet(d_in=d, layer_sizes=self.layer_sizes, dropout=self.dropout).to(self.device)
        self._compile_model()
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._log_device("MLP")

        X_all = torch.tensor(Xmat, dtype=torch.float32, device=self.device)
        y_all = torch.tensor(ymat, dtype=torch.float32, device=self.device)
        if self.loss_tp_alpha != 0 and "turn_progress" in X.columns:
            tp_all = torch.tensor(X["turn_progress"].values, dtype=torch.float32, device=self.device)
        else:
            tp_all = None

        self.model.train()
        gen = self._make_generator()
        scaler = self._create_scaler()
        for epoch in range(self.epochs):
            idx = torch.randperm(n, generator=gen, device=self.device)
            total_loss_t = torch.zeros(1, device=self.device)

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]

                X_batch = X_all[batch_idx]
                y_batch = y_all[batch_idx]

                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=self._amp_enabled):
                    logits = self.model(X_batch)

                    if tp_all is not None:
                        tp_batch = tp_all[batch_idx]
                        weight = tp_batch ** self.loss_tp_alpha
                        raw_loss = F.binary_cross_entropy_with_logits(logits, y_batch, reduction="none")
                        loss = (raw_loss * weight).mean()
                    else:
                        loss = F.binary_cross_entropy_with_logits(logits, y_batch)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                self._xla_mark_step()
                total_loss_t += loss.detach() * len(batch_idx)

            total_loss = total_loss_t.item() / n
            print(f"[MLP] epoch={epoch} loss={total_loss:.4f} samples={n}")

            if epoch_callback is not None and not epoch_callback(epoch, total_loss):
                break

        self._restore_model()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        if self.selected_features_ is None:
            raise ValueError("Model was not properly fitted (selected_features_ is None)")

        X_filtered = X[self.selected_features_]
        Xmat = X_filtered.to_numpy(dtype=np.float32)
        Xmat = self._standardize_apply(Xmat)

        self.model.eval()
        X_t = torch.tensor(Xmat, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            p_win = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([1.0 - p_win, p_win])

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return _utilitynet_feature_importance(self.model, self.layer_sizes, self.feature_names)

    def get_model_summary(self) -> dict:
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            "model_type": "MLP (PyTorch)",
            "n_features": len(self.feature_names or []),
            "feature_names": self.feature_names,
            "layer_sizes": self.layer_sizes,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "loss_tp_alpha": self.loss_tp_alpha,
            "device": str(self.device),
        }

    # ── Save / Load ─────────────────────────────────────────────────────────
    def _get_hyperparams(self) -> dict:
        hp = super()._get_hyperparams()
        hp["layer_sizes"] = list(self.layer_sizes)
        hp["batch_size"] = self.batch_size
        return hp

    @classmethod
    def _load_model_state(cls, dir_path, metadata: dict) -> "MLPPredictor":
        hp = metadata["hyperparams"]
        instance = cls(
            include_features=metadata.get("include_features"),
            exclude_features=metadata.get("exclude_features"),
            random_state=metadata.get("random_state", 42),
            layer_sizes=tuple(hp.get("layer_sizes", (64,))),
            dropout=hp.get("dropout", 0.0),
            lr=hp.get("lr", 0.001),
            weight_decay=hp.get("weight_decay", 0.001),
            epochs=hp.get("epochs", 10),
            loss_tp_alpha=hp.get("loss_tp_alpha", 0.0),
            batch_size=hp.get("batch_size", 32768),
        )
        instance._restore_torch_state(dir_path, metadata)
        d_in = len(instance.selected_features_)
        instance.model = _UtilityNet(d_in=d_in, layer_sizes=instance.layer_sizes, dropout=instance.dropout).to(instance.device)
        instance._load_model_weights(dir_path)
        return instance
