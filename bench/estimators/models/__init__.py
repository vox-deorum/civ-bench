"""Predictor classes (ported from ``vox-deorum-analysis/models/models/``).

Each predictor subclasses :class:`BasePredictor` and implements the save/load
contract the registry dispatches on. For the load-only path (stage 2) the live
methods are ``_load_model_state`` + ``predict_proba``; ``fit`` (training) is
exercised in stage 6.
"""

from __future__ import annotations

from .attention_model import AttentionMLPPredictor
from .base_predictor import BasePredictor
from .base_torch_predictor import BaseTorchPredictor, GroupedTorchPredictor
from .baseline_model import BaselineVictoryPredictor
from .grouped_mlp_model import GroupedMLPPredictor
from .interaction_mlp_model import InteractionMLPPredictor
from .mlp_model import MLPPredictor
from .naive_model import NaivePredictor
from .score_model import ScorePredictor
from .xgboost_model import XGBoostPredictor

__all__ = [
    "AttentionMLPPredictor",
    "BasePredictor",
    "BaseTorchPredictor",
    "BaselineVictoryPredictor",
    "GroupedMLPPredictor",
    "GroupedTorchPredictor",
    "InteractionMLPPredictor",
    "MLPPredictor",
    "NaivePredictor",
    "ScorePredictor",
    "XGBoostPredictor",
]
