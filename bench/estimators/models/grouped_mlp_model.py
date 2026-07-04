#!/usr/bin/env python3
"""Grouped MLP predictor for real-time winrate (group-softmax over (game_id, turn)).

Ported verbatim from ``../vox-deorum-analysis/models/models/grouped_mlp_model.py``.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from .base_torch_predictor import GroupedTorchPredictor, _BatchedGroups  # noqa: F401


class _UtilityNet(nn.Module):
    """Residual MLP mapping features → scalar utility."""

    def __init__(self, d_in: int, layer_sizes: Tuple[int, ...] = (64,), dropout: float = 0.0):
        super().__init__()
        if len(layer_sizes) == 0:
            self.net = nn.Linear(d_in, 1)
        elif len(layer_sizes) == 1:
            self.net = nn.Sequential(
                nn.Linear(d_in, layer_sizes[0]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0], 1),
            )
        else:
            hidden = layer_sizes[0]
            self.proj = nn.Linear(d_in, hidden)
            self.blocks = nn.ModuleList()
            for _ in layer_sizes:
                self.blocks.append(nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
            self.norm_out = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, 1)
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is not None:
            return self.net(x).squeeze(-1)
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.norm_out(x)
        return self.head(x).squeeze(-1)


# ── shared MLP helpers (deduped: MLPPredictor and GroupedMLPPredictor both use
#    a _UtilityNet with a `layer_sizes` hyperparameter, so these were byte-identical
#    copies in both modules). mlp_model already imports _UtilityNet from here.
def _layer_sizes_optuna_defaults(model_class) -> dict:
    """Optuna default-param dict derived from a layer_sizes-based ``__init__``."""
    import inspect
    sig = inspect.signature(model_class.__init__)
    d = {k: v.default for k, v in sig.parameters.items()
         if v.default is not inspect.Parameter.empty}
    layer_sizes = d["layer_sizes"]
    return {
        "n_layers": len(layer_sizes),
        "layer_size": layer_sizes[0] if layer_sizes else 64,
        "dropout": d["dropout"], "lr": d["lr"],
        "weight_decay": d["weight_decay"], "epochs": d["epochs"],
        "loss_tp_alpha": d["loss_tp_alpha"],
    }


def _layer_sizes_convert_optuna(raw_params) -> dict:
    """Fold optuna's flat (n_layers, layer_size) back into a ``layer_sizes`` tuple."""
    params = dict(raw_params)
    n_layers = params.pop("n_layers", None)
    layer_size = params.pop("layer_size", None)
    if n_layers is not None and layer_size is not None:
        params["layer_sizes"] = tuple([layer_size] * n_layers)
    return params


def _utilitynet_feature_importance(model, layer_sizes, feature_names) -> Optional[pd.DataFrame]:
    """|weight|-based feature importance for a fitted ``_UtilityNet`` predictor."""
    if model is None:
        raise ValueError("Model must be fitted before getting feature importance")

    if len(layer_sizes) == 0:
        weights = model.net.weight.detach().cpu().numpy()
        importances = np.abs(weights).flatten()
    elif len(layer_sizes) == 1:
        weights = model.net[0].weight.detach().cpu().numpy()
        importances = np.abs(weights).mean(axis=0)
    else:
        weights = model.proj.weight.detach().cpu().numpy()
        importances = np.abs(weights).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": importances,
        "abs_coefficient": np.abs(importances),
    })
    return importance_df.sort_values("abs_coefficient", ascending=False)


class GroupedMLPPredictor(GroupedTorchPredictor):
    """Scores each player with an MLP, then group-softmax over (game_id, turn)."""

    SUPPORTED_FEATURES = None
    FILTER_ZERO_SCORE = False
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
        group_cols: Tuple[str, str] = ("game_id", "turn"),
        id_cols: Tuple[str, ...] = ("experiment", "game_id", "player_id", "turn"),
        layer_sizes: Tuple[int, ...] = (222,),
        dropout: float = 0.211075,
        lr: float = 0.00777935,
        weight_decay: float = 0.00311362,
        epochs: int = 21,
        loss_tp_alpha: float = 0.398396,
        batch_size_groups: int = 4096,
        device: Optional[str] = None,
    ):
        super().__init__(
            include_features=include_features,
            exclude_features=exclude_features,
            random_state=random_state,
            group_cols=group_cols,
            id_cols=id_cols,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size_groups=batch_size_groups,
            loss_tp_alpha=loss_tp_alpha,
            device=device,
        )
        self.layer_sizes = layer_sizes

    def _model_display_name(self) -> str:
        return "GroupedMLP"

    def _build_model(self, d_in: int) -> nn.Module:
        return _UtilityNet(d_in=d_in, layer_sizes=self.layer_sizes, dropout=self.dropout).to(self.device)

    def _forward_train(self, X_batch: torch.Tensor, mask_batch: torch.Tensor) -> torch.Tensor:
        B, P, D = X_batch.shape
        flat = X_batch.reshape(B * P, D)
        return self.model(flat).reshape(B, P)

    def _forward_inference(self, X_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        N, P, D = X_t.shape
        flat = X_t.reshape(N * P, D)
        return self.model(flat).reshape(N, P)

    def get_model_summary(self) -> dict:
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            "model_type": "GroupedMLP (group-softmax)",
            "group_cols": self.group_cols,
            "n_features": len(self.feature_names or []),
            "feature_names": self.feature_names,
            "layer_sizes": self.layer_sizes,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "loss_tp_alpha": self.loss_tp_alpha,
            "device": self.device,
        }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return _utilitynet_feature_importance(self.model, self.layer_sizes, self.feature_names)

    # ── Save / Load ─────────────────────────────────────────────────────────
    def _get_hyperparams(self) -> dict:
        hp = super()._get_hyperparams()
        hp["layer_sizes"] = list(self.layer_sizes)
        return hp

    @classmethod
    def _load_model_state(cls, dir_path, metadata: dict) -> "GroupedMLPPredictor":
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
            batch_size_groups=hp.get("batch_size_groups", 4096),
        )
        instance._restore_torch_state(dir_path, metadata)
        d_in = len(instance.selected_features_)
        instance.model = instance._build_model(d_in)
        instance._load_model_weights(dir_path)
        return instance
