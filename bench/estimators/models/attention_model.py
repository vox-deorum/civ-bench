#!/usr/bin/env python3
"""Attention-based predictor for real-time winrate.

Ported verbatim from ``../vox-deorum-analysis/models/models/attention_model.py``.
Encodes each player, applies multi-head self-attention across the group, decodes
to a per-player logit, then group-softmax over (game_id, turn).
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from .base_torch_predictor import GroupedTorchPredictor
from .interaction_mlp_model import _ResidualMLP


class _AttentionBlock(nn.Module):
    """Pre-norm self-attention block with residual connection."""

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        return x + attn_out


class _AttentionNet(nn.Module):
    """Attention-based network for cross-player interaction."""

    def __init__(
        self,
        d_in: int,
        encoder_sizes: Tuple[int, ...] = (128,),
        decoder_sizes: Tuple[int, ...] = (128,),
        dropout: float = 0.0,
        num_heads: int = 4,
        n_attn_layers: int = 1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = encoder_sizes[0] if encoder_sizes else d_in

        self.encoder = _ResidualMLP(d_in, self.embed_dim, encoder_sizes, dropout)
        self.attn_layers = nn.ModuleList([
            _AttentionBlock(self.embed_dim, num_heads, attn_dropout)
            for _ in range(n_attn_layers)
        ])
        self.decoder = _ResidualMLP(self.embed_dim, 1, decoder_sizes, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, P, D = x.shape

        flat = x.reshape(B * P, D)
        h = self.encoder(flat).reshape(B, P, self.embed_dim)

        key_padding_mask = ~mask
        for layer in self.attn_layers:
            h = layer(h, key_padding_mask)

        flat_h = h.reshape(B * P, self.embed_dim)
        logits = self.decoder(flat_h).reshape(B, P)
        return logits


class AttentionMLPPredictor(GroupedTorchPredictor):
    """Self-attention group-softmax predictor."""

    SUPPORTED_FEATURES = None
    FILTER_ZERO_SCORE = False
    DEFAULT_FEATURES = [
        "science_adj", "culture_adj", "tourism_adj", "gold_adj",
        "food_adj", "production_adj", "military_adj", "faith_adj",
        "population", "cities", "votes", "minor_allies",
        "technologies_gap", "policies_gap",
        "highest_war_weariness", "active_wars", "truces", "defensive_pacts", "friendships",
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
        enc_size = d["encoder_sizes"][0] if d["encoder_sizes"] else 64
        return {
            "n_encoder_layers": len(d["encoder_sizes"]),
            "encoder_mult": enc_size // d["num_heads"],
            "num_heads": d["num_heads"],
            "n_decoder_layers": len(d["decoder_sizes"]),
            "decoder_size": d["decoder_sizes"][0] if d["decoder_sizes"] else 64,
            "n_attn_layers": d["n_attn_layers"],
            "attn_dropout": d["attn_dropout"],
            "dropout": d["dropout"], "lr": d["lr"],
            "weight_decay": d["weight_decay"], "epochs": d["epochs"],
            "loss_tp_alpha": d["loss_tp_alpha"],
        }

    @staticmethod
    def convert_optuna_params(raw_params):
        params = dict(raw_params)
        n_enc = params.pop("n_encoder_layers", None)
        enc_mult = params.pop("encoder_mult", None)
        num_heads = params.get("num_heads")
        n_dec = params.pop("n_decoder_layers", None)
        dec_size = params.pop("decoder_size", None)
        if n_enc is not None and enc_mult is not None and num_heads is not None:
            enc_size = enc_mult * num_heads
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
        encoder_sizes: Tuple[int, ...] = (30,),
        decoder_sizes: Tuple[int, ...] = (25,),
        num_heads: int = 3,
        n_attn_layers: int = 1,
        attn_dropout: float = 0.259432,
        dropout: float = 0.260767,
        lr: float = 0.00303038,
        weight_decay: float = 0.00450614,
        epochs: int = 13,
        loss_tp_alpha: float = 0.04768,
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
        self.num_heads = num_heads
        self.n_attn_layers = n_attn_layers
        self.attn_dropout = attn_dropout

    def _model_display_name(self) -> str:
        return "AttentionMLP"

    def _build_model(self, d_in: int) -> nn.Module:
        return _AttentionNet(
            d_in=d_in,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            dropout=self.dropout,
            num_heads=self.num_heads,
            n_attn_layers=self.n_attn_layers,
            attn_dropout=self.attn_dropout,
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
            "model_type": "AttentionMLP (self-attention group-softmax)",
            "group_cols": self.group_cols,
            "n_features": len(self.feature_names or []),
            "feature_names": self.feature_names,
            "encoder_sizes": self.encoder_sizes,
            "decoder_sizes": self.decoder_sizes,
            "num_heads": self.num_heads,
            "n_attn_layers": self.n_attn_layers,
            "attn_dropout": self.attn_dropout,
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
        hp["num_heads"] = self.num_heads
        hp["n_attn_layers"] = self.n_attn_layers
        hp["attn_dropout"] = self.attn_dropout
        return hp

    @classmethod
    def _load_model_state(cls, dir_path, metadata: dict) -> "AttentionMLPPredictor":
        hp = metadata["hyperparams"]
        instance = cls(
            include_features=metadata.get("include_features"),
            exclude_features=metadata.get("exclude_features"),
            random_state=metadata.get("random_state", 42),
            encoder_sizes=tuple(hp.get("encoder_sizes", (128,))),
            decoder_sizes=tuple(hp.get("decoder_sizes", (128,))),
            num_heads=hp.get("num_heads", 4),
            n_attn_layers=hp.get("n_attn_layers", 1),
            attn_dropout=hp.get("attn_dropout", 0.1),
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
