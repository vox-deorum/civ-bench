"""Estimators layer — prediction-model producers (tune / train / load / infer).

Stage 2 ships the **load-only** path: :func:`run_estimator` resolves a
``fit:"pretrained"`` entry, loads a saved ``model_dir`` via :func:`load_model`
(dispatched on ``metadata.model_class``), and re-infers on the canonical turns
table to emit ``predictions.csv``. Training/tuning land in stage 6.

This package pulls heavy deps (torch / xgboost) on import, so import it lazily
from the CLI run path — never from the import-light config/pipeline layers.
"""

from __future__ import annotations

# This must run before registry/model imports pull in torch. On affected Windows
# ROCm wheels it selects the sole installed target and bypasses a broken path probe.
from .environment import configure_rocm_sdk_target

configure_rocm_sdk_target()

from .errors import EstimatorError
from .features import build_feature_frame, needs_variant_columns, prepare_features
from .registry import MODEL_REGISTRY, get_model, list_models, load_model, register_model
from .runner import EstimatorResult, run_estimator
from .training import TrainResult, run_cross_val, run_full_train

del configure_rocm_sdk_target

__all__ = [
    "EstimatorError",
    "EstimatorResult",
    "MODEL_REGISTRY",
    "TrainResult",
    "build_feature_frame",
    "get_model",
    "list_models",
    "load_model",
    "needs_variant_columns",
    "prepare_features",
    "register_model",
    "run_cross_val",
    "run_estimator",
    "run_full_train",
]
