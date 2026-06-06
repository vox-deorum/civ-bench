#!/usr/bin/env python3
"""Interaction MLP predictor (DeepSets) for real-time winrate.

Ported verbatim from ``../vox-deorum-analysis/models/models/interaction_mlp_model.py``.
Encodes each player, pools across the group (masked mean+max), decodes to a
per-player logit, then group-softmax over (game_id, turn).
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from .base_torch_predictor import GroupedTorchPredictor


class _ResidualMLP(nn.Module):
    """Residual MLP block with configurable output dimension."""

    def __init__(self, d_in: int, d_out: int, layer_sizes: Tuple[int, ...] = (64,), dropout: float = 0.0):
        super().__init__()
        if len(layer_sizes) == 0:
            self.net = nn.Linear(d_in, d_out)
        elif len(layer_sizes) == 1:
            self.net = nn.Sequential(
                nn.Linear(d_in, layer_sizes[0]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0], d_out),
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
            self.head = nn.Linear(hidden, d_out)
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is not None:
            return self.net(x)
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.norm_out(x)
        return self.head(x)


class _DeepSetsNet(nn.Module):
    """DeepSets-style network for cross-player interaction."""

    POOL_MODES = ("mean", "max", "mean+max")

    def __init__(
        self,
        d_in: int,
        encoder_sizes: Tuple[int, ...] = (64,),
        decoder_sizes: Tuple[int, ...] = (64,),
        dropout: float = 0.0,
        pool_mode: str = "mean+max",
    ):
        super().__init__()
        if pool_mode not in self.POOL_MODES:
            raise ValueError(f"pool_mode must be one of {self.POOL_MODES}, got '{pool_mode}'")
        self.pool_mode = pool_mode
        self.embed_dim = encoder_sizes[0] if encoder_sizes else d_in
        n_pools = len(pool_mode.split("+"))

        self.encoder = _ResidualMLP(d_in, self.embed_dim, encoder_sizes, dropout)
        self.decoder = _ResidualMLP(self.embed_dim * (1 + n_pools), 1, decoder_sizes, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, P, D = x.shape

        flat = x.reshape(B * P, D)
        h = self.encoder(flat).reshape(B, P, self.embed_dim)

        mask_f = mask.unsqueeze(-1).float()
        pools = []
        if "mean" in self.pool_mode:
            h_masked = h * mask_f
            pools.append(h_masked.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1))
        if "max" in self.pool_mode:
            h_for_max = h.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pools.append(h_for_max.max(dim=1).values)

        parts = [h]
        for pool in pools:
            parts.append(pool.unsqueeze(1).expand(-1, P, -1))
        combined = torch.cat(parts, dim=-1)

        flat_combined = combined.reshape(B * P, -1)
        logits = self.decoder(flat_combined).reshape(B, P)
        return logits


class InteractionMLPPredictor(GroupedTorchPredictor):
    """DeepSets-based MLP: encode → pool → decode → group-softmax."""

    SUPPORTED_FEATURES = None
    FILTER_ZERO_SCORE = False
    DEFAULT_FEATURES = [
        "science_adj", "culture_adj", "tourism_adj", "gold_adj",
        "food_adj", "production_adj", "military_adj", "faith_adj",
        "population", "cities", "votes", "minor_allies",
        "highest_war_weariness", "active_wars", "truces", "defensive_pacts", "friendships",
        "technologies_gap", "policies_gap",
        "happiness_percentage", "religion_percentage",
        "military_utilization",
        "turn_progress", "score_ratio",
    ]
    REQUIRED_FEATURES = None

    @classmethod
    def optuna_default_params(cls):
        import inspect
        sig = inspect.signature(cls.__init__)
        d = {k: v.default for k, v in sig.parameters.items()
             if v.default is not inspect.Parameter.empty}
        return {
            "n_encoder_layers": len(d["encoder_sizes"]),
            "encoder_size": d["encoder_sizes"][0] if d["encoder_sizes"] else 64,
            "n_decoder_layers": len(d["decoder_sizes"]),
            "decoder_size": d["decoder_sizes"][0] if d["decoder_sizes"] else 64,
            "dropout": d["dropout"], "lr": d["lr"],
            "weight_decay": d["weight_decay"], "epochs": d["epochs"],
            "loss_tp_alpha": d["loss_tp_alpha"],
        }

    @staticmethod
    def convert_optuna_params(raw_params):
        params = dict(raw_params)
        n_enc = params.pop("n_encoder_layers", None)
        enc_size = params.pop("encoder_size", None)
        n_dec = params.pop("n_decoder_layers", None)
        dec_size = params.pop("decoder_size", None)
        if n_enc is not None and enc_size is not None:
            params["encoder_sizes"] = tuple([enc_size] * n_enc)
        if n_dec is not None and dec_size is not None:
            params["decoder_sizes"] = tuple([dec_size] * n_dec)
        return params

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        group_cols: Tuple[str, str] = ("game_id", "turn"),
        id_cols: Tuple[str, ...] = ("experiment", "game_id", "player_id", "turn"),
        pool_mode: str = "mean+max",
        encoder_sizes: Tuple[int, ...] = (80,),
        decoder_sizes: Tuple[int, ...] = (183,),
        dropout: float = 0.322111,
        lr: float = 0.000259737,
        weight_decay: float = 0.000738948,
        epochs: int = 29,
        loss_tp_alpha: float = 0.104261,
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
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.pool_mode = pool_mode

    def _model_display_name(self) -> str:
        return "InteractionMLP"

    def _build_model(self, d_in: int) -> nn.Module:
        return _DeepSetsNet(
            d_in=d_in,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            dropout=self.dropout,
            pool_mode=self.pool_mode,
        ).to(self.device)

    def _forward_train(self, X_batch: torch.Tensor, mask_batch: torch.Tensor) -> torch.Tensor:
        return self.model(X_batch, mask_batch)

    def _forward_inference(self, X_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        return self.model(X_t, mask_t)

    def _get_encoder_sizes(self) -> Tuple[int, ...]:
        return self.encoder_sizes

    def get_model_summary(self) -> dict:
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")
        return {
            "model_type": "InteractionMLP (DeepSets group-softmax)",
            "group_cols": self.group_cols,
            "n_features": len(self.feature_names or []),
            "feature_names": self.feature_names,
            "encoder_sizes": self.encoder_sizes,
            "decoder_sizes": self.decoder_sizes,
            "pool_mode": self.pool_mode,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "loss_tp_alpha": self.loss_tp_alpha,
            "device": self.device,
        }

    # ── Save / Load ─────────────────────────────────────────────────────────
    def _get_hyperparams(self) -> dict:
        hp = super()._get_hyperparams()
        hp["encoder_sizes"] = list(self.encoder_sizes)
        hp["decoder_sizes"] = list(self.decoder_sizes)
        hp["pool_mode"] = self.pool_mode
        return hp

    @classmethod
    def _load_model_state(cls, dir_path, metadata: dict) -> "InteractionMLPPredictor":
        hp = metadata["hyperparams"]
        instance = cls(
            include_features=metadata.get("include_features"),
            exclude_features=metadata.get("exclude_features"),
            random_state=metadata.get("random_state", 42),
            encoder_sizes=tuple(hp.get("encoder_sizes", (64,))),
            decoder_sizes=tuple(hp.get("decoder_sizes", (64,))),
            pool_mode=hp.get("pool_mode", "mean+max"),
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
