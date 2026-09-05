"""Prediction-model registry (port of ``models/utils/model_registry.py``).

Maps a string ``model`` id (as used in ``configs/models.json`` and an estimator
entry's ``model`` field) to its predictor class, and dispatches :func:`load_model`
on a saved dir's ``metadata.model_class``. Unlike the source, xgboost is imported
directly (no ``HAS_XGBOOST`` soft-fail) per AGENTS.md: every dependency is
mandatory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Type

from .models.attention_model import AttentionMLPPredictor
from .models.base_predictor import BasePredictor
from .models.baseline_model import BaselineVictoryPredictor
from .models.grouped_mlp_model import GroupedMLPPredictor
from .models.interaction_mlp_model import InteractionMLPPredictor
from .models.mlp_model import MLPPredictor
from .models.naive_model import NaivePredictor
from .models.score_model import ScorePredictor
from .models.xgboost_model import XGBoostPredictor


# Registry mapping model ids → classes. Keys match the `prediction_models` ids in
# configs/models.json (so an estimator entry's `model` selects a class for fit:train).
MODEL_REGISTRY: Dict[str, Type[BasePredictor]] = {
    "naive": NaivePredictor,
    "score": ScorePredictor,
    "baseline": BaselineVictoryPredictor,
    "xgboost": XGBoostPredictor,
    "mlp": MLPPredictor,
    "grouped_mlp": GroupedMLPPredictor,
    "interaction_mlp": InteractionMLPPredictor,
    "attention_mlp": AttentionMLPPredictor,
}


def get_model(name: str, **kwargs) -> BasePredictor:
    """Instantiate a model by registry id."""
    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name_lower](**kwargs)


def list_models() -> List[str]:
    """All registered model ids, sorted."""
    return sorted(MODEL_REGISTRY.keys())


def register_model(name: str, model_class: Type[BasePredictor]) -> None:
    """Register a new model class (raises on duplicate / wrong base class)."""
    if not issubclass(model_class, BasePredictor):
        raise TypeError(f"{model_class.__name__} must be a subclass of BasePredictor")
    name_lower = name.lower()
    if name_lower in MODEL_REGISTRY:
        raise ValueError(f"Model name '{name}' is already registered")
    MODEL_REGISTRY[name_lower] = model_class


def load_model(path: str | Path) -> BasePredictor:
    """Load a saved model, dispatching on ``metadata.model_class``.

    Raises ``FileNotFoundError`` when ``metadata.json`` is missing and
    ``ValueError`` when its ``model_class`` is not a registered predictor.
    """
    load_dir = Path(path)
    metadata_path = load_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {load_dir}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_name = metadata.get("model_class")
    model_class = None
    for cls in MODEL_REGISTRY.values():
        if cls.__name__ == class_name:
            model_class = cls
            break

    if model_class is None:
        raise ValueError(
            f"Unknown model class: '{class_name}'. "
            f"Available: {[cls.__name__ for cls in MODEL_REGISTRY.values()]}"
        )

    return model_class._load_model_state(load_dir, metadata)
