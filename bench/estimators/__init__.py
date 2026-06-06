"""Estimators layer — prediction-model producers (tune / train / load / infer).

Stage 2 ships the **load-only** path: :func:`run_estimator` resolves a
``fit:"pretrained"`` entry, loads a saved ``model_dir`` via :func:`load_model`
(dispatched on ``metadata.model_class``), and re-infers on the canonical turns
table to emit ``predictions.csv``. Training/tuning land in stage 6.

This package pulls heavy deps (torch / xgboost) on import, so import it lazily
from the CLI run path — never from the import-light config/pipeline layers.
"""

from __future__ import annotations

from .errors import EstimatorError
from .features import build_feature_frame, needs_variant_columns, prepare_features
from .registry import MODEL_REGISTRY, get_model, list_models, load_model, register_model
from .runner import EstimatorResult, run_estimator

__all__ = [
    "EstimatorError",
    "EstimatorResult",
    "MODEL_REGISTRY",
    "build_feature_frame",
    "get_model",
    "list_models",
    "load_model",
    "needs_variant_columns",
    "prepare_features",
    "register_model",
    "run_estimator",
]
