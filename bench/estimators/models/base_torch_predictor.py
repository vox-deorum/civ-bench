#!/usr/bin/env python3
"""Common base classes for PyTorch-based victory-prediction models.

Ported from ``../vox-deorum-analysis/models/models/base_torch_predictor.py``.

- :class:`BaseTorchPredictor`: device detection, standardization, compile, AMP,
  RNG, and the torch save/load state contract.
- :class:`GroupedTorchPredictor`: the grouped ``(game_id, turn)`` softmax training
  loop + group-winrate inference shared by GroupedMLP / InteractionMLP / AttentionMLP.

The torch/XLA imports are direct (no soft-fail) per AGENTS.md: torch is a
mandatory dependency. The ``torch_xla`` probe inside ``__init__`` is genuine TPU
device detection, not a dependency gate.
"""

from __future__ import annotations

import json
import sys
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_predictor import BasePredictor


@dataclass
class _BatchedGroups:
    """Pre-padded tensors for vectorized training/inference."""
    X: np.ndarray          # (n_groups, max_players, n_features)
    y_indices: np.ndarray  # (n_groups,) winner index per group
    mask: np.ndarray       # (n_groups, max_players) True for real players
    tp: np.ndarray         # (n_groups,) turn_progress per group
    n_groups: int


class BaseTorchPredictor(BasePredictor):
    """Base class for all PyTorch-based predictors."""

    DISABLE_RESAMPLING = True

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        epochs: int = 10,
        loss_tp_alpha: float = 0.0,
        device: Optional[str] = None,
    ):
        super().__init__(include_features, exclude_features, random_state)

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.loss_tp_alpha = loss_tp_alpha

        if device:
            self.device = device
        else:
            try:
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
            except (ImportError, RuntimeError):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._is_xla = "xla" in str(self.device)
        self._amp_enabled = (
            not self._is_xla
            and torch.cuda.is_available()
            and "cuda" in str(self.device)
        )

        self.model: Optional[nn.Module] = None
        self.feature_names: Optional[List[str]] = None
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    def get_parameter_count(self) -> Optional[int]:
        if self.model is None:
            return None
        return sum(p.numel() for p in self.model.parameters())

    # ── standardization ─────────────────────────────────────────────────────
    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        return (X - self._mu) / self._sigma

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None or self._sigma is None:
            return X
        return (X - self._mu) / self._sigma

    # ── torch.compile helpers ────────────────────────────────────────────────
    def _compile_model(self) -> None:
        self._uncompiled_model = self.model
        if self._is_xla or sys.platform == "win32":
            return
        try:
            torch.compiler.reset()
            torch.set_float32_matmul_precision("high")
            self.model = torch.compile(self.model)
        except Exception:
            self.model = self._uncompiled_model

    def _restore_model(self) -> None:
        self.model = self._uncompiled_model

    # ── device logging ────────────────────────────────────────────────────────
    def _log_device(self, name: str) -> None:
        print(f"[{name}] Training on device: {self.device}")
        if self._is_xla:
            print(f"[{name}] TPU available via torch_xla")
        elif torch.cuda.is_available():
            print(f"[{name}] GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[{name}] GPU not available, using CPU")

    # ── RNG generator ─────────────────────────────────────────────────────────
    def _make_generator(self) -> torch.Generator:
        gen_device = "cpu" if self._is_xla or not torch.cuda.is_available() else self.device
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(self.random_state)
        return gen

    def _seed_torch(self) -> None:
        """Seed the torch *global* RNGs at the start of each fit.

        ``_make_generator`` covers the explicit ``torch.randperm`` shuffle, but
        weight init (``nn.Linear`` etc.) and dropout draw from the process-global
        RNG; unseeded, they make every fit's predictions drift run-to-run despite
        the documented byte-stability guarantee. Seeding per fit (not once at
        import) keeps each CV fold reproducible independent of fold order. We do
        **not** enable ``torch.use_deterministic_algorithms`` (it can hard-error and
        needs a CUBLAS env var on CUDA); the guarantee is same-machine/same-device.
        """
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    # ── AMP ─────────────────────────────────────────────────────────────────
    def _create_scaler(self) -> Optional["torch.amp.GradScaler"]:
        if self._amp_enabled:
            return torch.amp.GradScaler("cuda")
        return None

    def _xla_mark_step(self) -> None:
        if self._is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

    # ── Save / Load ─────────────────────────────────────────────────────────
    def _get_hyperparams(self) -> dict:
        return {
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "loss_tp_alpha": self.loss_tp_alpha,
        }

    def _save_model_state(self, dir_path: Path) -> None:
        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        torch.save(self.model.state_dict(), dir_path / "model_state.pt")
        state = {
            "mu": self._mu.tolist() if self._mu is not None else None,
            "sigma": self._sigma.tolist() if self._sigma is not None else None,
            "feature_names": self.feature_names,
        }
        with open(dir_path / "torch_state.json", "w") as f:
            json.dump(state, f)

    def _restore_torch_state(self, dir_path: Path, metadata: dict) -> None:
        with open(dir_path / "torch_state.json", "r") as f:
            state = json.load(f)
        self._mu = np.array(state["mu"], dtype=np.float32) if state["mu"] is not None else None
        self._sigma = np.array(state["sigma"], dtype=np.float32) if state["sigma"] is not None else None
        self.feature_names = state["feature_names"]
        self.selected_features_ = metadata["selected_features"]

    def _load_model_weights(self, dir_path: Path) -> None:
        state_dict = torch.load(dir_path / "model_state.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()


class GroupedTorchPredictor(BaseTorchPredictor):
    """Base for grouped MLP models using ``(game_id, turn)`` group-softmax."""

    REQUIRES_ID_COLUMNS = ["game_id", "turn", "player_id"]

    def __init__(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        random_state: int = 42,
        group_cols: Tuple[str, str] = ("game_id", "turn"),
        id_cols: Tuple[str, ...] = ("experiment", "game_id", "player_id", "turn"),
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        epochs: int = 10,
        batch_size_groups: int = 4096,
        loss_tp_alpha: float = 0.0,
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
        self.group_cols = group_cols
        self.id_cols = id_cols
        self.batch_size_groups = batch_size_groups

    # ── abstract hooks ────────────────────────────────────────────────────────
    @abstractmethod
    def _build_model(self, d_in: int) -> nn.Module:
        ...

    @abstractmethod
    def _forward_train(self, X_batch: torch.Tensor, mask_batch: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def _forward_inference(self, X_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def _model_display_name(self) -> str:
        ...

    # ── group building ────────────────────────────────────────────────────────
    def _build_groups(self, df: pd.DataFrame, y: pd.Series, raw_tp: Optional[pd.Series] = None) -> _BatchedGroups:
        g_game, g_turn = self.group_cols

        winner_map = (
            df.loc[y == 1, ["game_id", "player_id"]]
            .drop_duplicates()
            .set_index("game_id")["player_id"]
        )
        df = df.copy()
        df["_winner_pid"] = df["game_id"].map(winner_map)
        df = df.dropna(subset=["_winner_pid"])

        grp = df.groupby([g_game, g_turn], sort=False)
        df["_gid"] = grp.ngroup()
        df["_pos"] = grp.cumcount()
        df["_is_winner"] = (df["player_id"] == df["_winner_pid"]).astype(int)

        all_gids = df["_gid"].unique()
        valid_gids = df.loc[df["_is_winner"] == 1, "_gid"].unique()
        n_dropped = len(all_gids) - len(valid_gids)
        if n_dropped > 0:
            warnings.warn(f"_build_groups: dropped {n_dropped} groups with no winner")
        df = df[df["_gid"].isin(valid_gids)]

        unique_gids = df["_gid"].unique()
        gid_remap = pd.Series(np.arange(len(unique_gids)), index=unique_gids)
        df["_gid"] = df["_gid"].map(gid_remap)

        n_groups = len(unique_gids)
        max_players = int(df["_pos"].max()) + 1
        n_features = len(self.selected_features_)

        gids = df["_gid"].values
        pos = df["_pos"].values

        X_padded = np.zeros((n_groups, max_players, n_features), dtype=np.float32)
        X_padded[gids, pos, :] = df[self.selected_features_].to_numpy(dtype=np.float32)

        mask = np.zeros((n_groups, max_players), dtype=bool)
        mask[gids, pos] = True

        y_indices = np.zeros(n_groups, dtype=np.int64)
        winner_df = df[df["_is_winner"] == 1]
        y_indices[winner_df["_gid"].values] = winner_df["_pos"].values

        tp = np.ones(n_groups, dtype=np.float32)
        if raw_tp is not None:
            tp_vals = raw_tp.loc[df.index].values.astype(np.float32)
            tp_per_row = pd.Series(tp_vals, index=df.index)
            tp_df = df[["_gid"]].assign(_tp=tp_per_row).groupby("_gid")["_tp"].first()
            tp[tp_df.index.values] = tp_df.values

        return _BatchedGroups(X=X_padded, y_indices=y_indices, mask=mask, tp=tp, n_groups=n_groups)

    # ── template fit() ────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: pd.Series, clusters: Optional[pd.Series] = None, epoch_callback=None):
        self._seed_torch()
        self._filter_features(X)

        missing = [c for c in self.REQUIRES_ID_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires columns {missing} in X. "
                f"These should be automatically injected by the runner. "
                f"If calling fit() directly, ensure X includes: {self.REQUIRES_ID_COLUMNS}"
            )

        self.selected_features_ = [c for c in self.selected_features_ if c not in self.id_cols]
        self.feature_names = list(self.selected_features_)

        Xmat = X[self.selected_features_].to_numpy(dtype=np.float32)
        Xmat = self._standardize_fit(Xmat)

        X_std = pd.DataFrame(Xmat, columns=self.selected_features_, index=X.index)
        X_std = pd.concat([X[["game_id", "turn", "player_id"]], X_std], axis=1)

        raw_tp = X["turn_progress"] if "turn_progress" in X.columns and self.loss_tp_alpha != 0 else None
        batched = self._build_groups(X_std, y, raw_tp=raw_tp)
        if batched.n_groups == 0:
            raise ValueError("No valid (game_id, turn) groups constructed. Check your data.")

        d = len(self.selected_features_)
        self.model = self._build_model(d)
        self._compile_model()

        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        name = self._model_display_name()
        self._log_device(name)

        X_all = torch.tensor(batched.X, dtype=torch.float32, device=self.device)
        y_all = torch.tensor(batched.y_indices, dtype=torch.long, device=self.device)
        mask_all = torch.tensor(batched.mask, device=self.device)
        tp_all = torch.tensor(batched.tp, dtype=torch.float32, device=self.device) if self.loss_tp_alpha != 0 else None

        self.model.train()
        gen = self._make_generator()
        scaler = self._create_scaler()
        n_batches = max(1, (batched.n_groups + self.batch_size_groups - 1) // self.batch_size_groups)

        for epoch in range(self.epochs):
            idx = torch.randperm(batched.n_groups, generator=gen, device=self.device)
            total_loss_t = torch.zeros(1, device=self.device)

            for start in range(0, batched.n_groups, self.batch_size_groups):
                batch_idx_t = idx[start:start + self.batch_size_groups]

                X_batch = X_all[batch_idx_t]
                y_batch = y_all[batch_idx_t]
                mask_batch = mask_all[batch_idx_t]

                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=self._amp_enabled):
                    logits = self._forward_train(X_batch, mask_batch)
                    logits = logits.masked_fill(~mask_batch, float("-inf"))

                    if tp_all is not None:
                        tp_batch = tp_all[batch_idx_t]
                        weight = tp_batch ** self.loss_tp_alpha
                        raw_loss = F.cross_entropy(logits, y_batch, reduction="none")
                        loss = (raw_loss * weight).mean()
                    else:
                        loss = F.cross_entropy(logits, y_batch)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                self._xla_mark_step()
                total_loss_t += loss.detach()

            total_loss = total_loss_t.item() / n_batches
            print(f"[{name}] epoch={epoch} loss={total_loss:.4f} groups={batched.n_groups}")

            if epoch_callback is not None and not epoch_callback(epoch, total_loss):
                break

        self._restore_model()
        return self

    # ── prediction ────────────────────────────────────────────────────────────
    def predict_group_winrate(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        if self.selected_features_ is None:
            raise ValueError("Model was not properly fitted (selected_features_ is None)")
        for c in ["game_id", "turn"]:
            if c not in X.columns:
                raise ValueError(f"predict_group_winrate requires column '{c}' in X")

        self.model.eval()

        if len(X) == 0:
            return np.array([], dtype=np.float64)

        Xmat = X[self.selected_features_].to_numpy(dtype=np.float32)
        Xmat = self._standardize_apply(Xmat)

        n_features = Xmat.shape[1]

        gb = X.groupby(list(self.group_cols), sort=False)
        gids = gb.ngroup().values
        pos = gb.cumcount().values
        n_groups = gids.max() + 1
        max_players = pos.max() + 1

        X_padded = np.zeros((n_groups, max_players, n_features), dtype=np.float32)
        X_padded[gids, pos, :] = Xmat
        mask = np.zeros((n_groups, max_players), dtype=bool)
        mask[gids, pos] = True

        X_t = torch.tensor(X_padded, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, device=self.device)

        with torch.no_grad():
            logits = self._forward_inference(X_t, mask_t)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            pg = torch.softmax(logits, dim=1).cpu().numpy()

        probs = pg[gids, pos].astype(np.float32)
        return pd.Series(probs, index=X.index, name="p_win_group")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_win = self.predict_group_winrate(X).to_numpy()
        return np.column_stack([1.0 - p_win, p_win])

    # ── feature importance (default: encoder-based) ───────────────────────────
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        encoder = self.model.encoder
        layer_sizes = self._get_encoder_sizes()

        if len(layer_sizes) == 0:
            weights = encoder.net.weight.detach().cpu().numpy()
            importances = np.abs(weights).mean(axis=0)
        elif len(layer_sizes) == 1:
            weights = encoder.net[0].weight.detach().cpu().numpy()
            importances = np.abs(weights).mean(axis=0)
        else:
            weights = encoder.proj.weight.detach().cpu().numpy()
            importances = np.abs(weights).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": importances,
            "abs_coefficient": np.abs(importances),
        })
        return importance_df.sort_values("abs_coefficient", ascending=False)

    def _get_encoder_sizes(self) -> Tuple[int, ...]:
        return getattr(self, "encoder_sizes", ())

    def _get_hyperparams(self) -> dict:
        hp = super()._get_hyperparams()
        hp["batch_size_groups"] = self.batch_size_groups
        return hp
