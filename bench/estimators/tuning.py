"""Optuna hyperparameter tuning for victory-prediction estimators (stage 6).

Ported from ``../vox-deorum-analysis/models/tune_model.py`` — the search spaces,
the feature-variant reconstruction, and the CV objective — folded behind the
estimator ``tune`` config block (benchmark.md §4.3). No separate CLI / Colab
notebook (``tune_colab.ipynb`` is obsolete): a run sets ``tune`` and the runner
calls :func:`run_tune`, or skips the search entirely via ``load_params``.

``tune.search`` selects what the study optimizes:

- ``"hyperparameters"`` — model hyperparameters only (the model's coded
  ``DEFAULT_FEATURES`` are used). This is the template default.
- ``"features"`` — feature-variant selection only (model default hyperparams).
- ``"both"`` — both simultaneously.

The study optimizes a **single scalar** ``objective`` (``brier_score`` /
``log_loss`` minimized; ``roc_auc`` / ``balanced_accuracy`` maximized) with a
per-fold overfitting penalty, matching the source. The sampler is seeded from
the top-level ``seed`` so a fresh search is reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .features import FEATURE_GROUPS, prepare_features
from .registry import MODEL_REGISTRY
from .training import (
    _filter_zero_score,
    _strip_id_columns_if_not_needed,
    apply_resampling,
    get_kfold_splits,
)

MINIMIZE_METRICS = {"brier_score", "log_loss"}
SEARCH_TO_MODE = {"hyperparameters": "params", "features": "variables", "both": "both"}


# ── feature-variant tuning (port of tune_model FEATURE_FAMILIES etc.) ─────────
FEATURE_FAMILIES = {
    "science":    {"adj": "science_adj",    "share": "science_share",    "raw_share": "science_raw_share"},
    "culture":    {"adj": "culture_adj",    "share": "culture_share",    "raw_share": "culture_raw_share"},
    "tourism":    {"adj": "tourism_adj",    "share": "tourism_share",    "raw_share": "tourism_raw_share"},
    "gold":       {"adj": "gold_adj",       "share": "gold_share",       "raw_share": "gold_raw_share"},
    "faith":      {"adj": "faith_adj",      "share": "faith_share",      "raw_share": "faith_raw_share"},
    "production": {"adj": "production_adj", "share": "production_share", "raw_share": "production_raw_share"},
    "food":       {"adj": "food_adj",       "share": "food_share",       "raw_share": "food_raw_share"},
    "military":   {"adj": "military_adj", "share": "military_share"},
    "cities":       {"none": "", "raw": "cities",       "share": "cities_share"},
    "population":   {"raw": "population",   "share": "population_share"},
    "votes":        {"raw": "votes",        "share": "votes_share"},
    "minor_allies": {"raw": "minor_allies", "share": "minor_allies_share"},
}
FIXED_FEATURES = FEATURE_GROUPS["progress"] + FEATURE_GROUPS["gaps"]
TOGGLE_FEATURES = FEATURE_GROUPS["percentages"]


def suggest_feature_variants(trial) -> list:
    """Let Optuna pick one variant per family + toggle the optional features."""
    selected = list(FIXED_FEATURES)
    for family_name, variants in FEATURE_FAMILIES.items():
        chosen = trial.suggest_categorical(f"feat_{family_name}", list(variants.keys()))
        if chosen != "none":
            selected.append(variants[chosen])
    for feat_name in TOGGLE_FEATURES:
        if trial.suggest_categorical(f"feat_{feat_name}", [True, False]):
            selected.append(feat_name)
    return selected


def reconstruct_include_features(raw_params: dict) -> list:
    """Rebuild include_features from stored ``feat_*`` trial params."""
    selected = list(FIXED_FEATURES)
    for family_name, variants in FEATURE_FAMILIES.items():
        chosen = raw_params.get(f"feat_{family_name}", "share")
        if chosen != "none":
            selected.append(variants[chosen])
    for feat_name in TOGGLE_FEATURES:
        if raw_params.get(f"feat_{feat_name}", True):
            selected.append(feat_name)
    return selected


def _reverse_map_features(model_class) -> dict:
    """Reverse-map a model's DEFAULT_FEATURES → feat_* categorical params."""
    features = getattr(model_class, "DEFAULT_FEATURES", None)
    if not features:
        return {}
    feat_set = set(features)
    params = {}
    for family_name, variants in FEATURE_FAMILIES.items():
        chosen = None
        for variant_key, column_name in variants.items():
            if column_name and column_name in feat_set:
                chosen = variant_key
                break
        if chosen is None:
            chosen = "none" if "none" in variants else list(variants.keys())[0]
        params[f"feat_{family_name}"] = chosen
    for feat_name in TOGGLE_FEATURES:
        params[f"feat_{feat_name}"] = feat_name in feat_set
    return params


# ── search spaces (port of tune_model.suggest_*) ─────────────────────────────
def suggest_score_params(trial) -> Dict:
    return {"exponent": trial.suggest_float("exponent", 1.0, 8.0)}


def suggest_xgboost_params(trial) -> Dict:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 20.0, log=True),
    }
    calibrate = trial.suggest_categorical("calibrate", [True, False])
    params["calibrate"] = calibrate
    if calibrate:
        params["calibration_method"] = trial.suggest_categorical(
            "calibration_method", ["isotonic", "sigmoid"]
        )
    else:
        params["calibration_method"] = "sigmoid"
    return params


def suggest_mlp_params(trial) -> Dict:
    n_layers = trial.suggest_int("n_layers", 1, 16)
    layer_size = trial.suggest_int("layer_size", 16, 192)
    return {
        "layer_sizes": tuple([layer_size] * n_layers) if n_layers > 0 else (),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 5, 30),
        "loss_tp_alpha": trial.suggest_float("loss_tp_alpha", 0, 2.0),
    }


def suggest_grouped_mlp_params(trial) -> Dict:
    n_layers = trial.suggest_int("n_layers", 1, 8)
    layer_size = trial.suggest_int("layer_size", 16, 256)
    return {
        "layer_sizes": tuple([layer_size] * n_layers) if n_layers > 0 else (),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 5, 30),
        "loss_tp_alpha": trial.suggest_float("loss_tp_alpha", 0, 2.0),
    }


def suggest_interaction_mlp_params(trial) -> Dict:
    n_encoder_layers = trial.suggest_int("n_encoder_layers", 1, 8)
    encoder_size = trial.suggest_int("encoder_size", 16, 256)
    n_decoder_layers = trial.suggest_int("n_decoder_layers", 1, 8)
    decoder_size = trial.suggest_int("decoder_size", 16, 256)
    return {
        "encoder_sizes": tuple([encoder_size] * n_encoder_layers),
        "decoder_sizes": tuple([decoder_size] * n_decoder_layers),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 5, 30),
        "loss_tp_alpha": trial.suggest_float("loss_tp_alpha", 0, 2.0),
    }


def suggest_attention_mlp_params(trial) -> Dict:
    n_encoder_layers = trial.suggest_int("n_encoder_layers", 1, 6)
    num_heads = trial.suggest_int("num_heads", 2, 4)
    encoder_mult = trial.suggest_int("encoder_mult", 2, 128 // num_heads)
    encoder_size = encoder_mult * num_heads
    n_decoder_layers = trial.suggest_int("n_decoder_layers", 1, 6)
    decoder_size = trial.suggest_int("decoder_size", 16, 128)
    return {
        "encoder_sizes": tuple([encoder_size] * n_encoder_layers),
        "decoder_sizes": tuple([decoder_size] * n_decoder_layers),
        "num_heads": num_heads,
        "n_attn_layers": trial.suggest_int("n_attn_layers", 1, 2),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 5, 30),
        "loss_tp_alpha": trial.suggest_float("loss_tp_alpha", 0, 2.0),
    }


SEARCH_SPACES = {
    "score": suggest_score_params,
    "xgboost": suggest_xgboost_params,
    "mlp": suggest_mlp_params,
    "grouped_mlp": suggest_grouped_mlp_params,
    "interaction_mlp": suggest_interaction_mlp_params,
    "attention_mlp": suggest_attention_mlp_params,
}


def convert_best_params(model_name: str, best_params: Dict) -> Dict:
    """Convert raw Optuna params to model kwargs (feat_* → include_features)."""
    params = dict(best_params)
    if any(k.startswith("feat_") for k in params):
        params["include_features"] = reconstruct_include_features(params)
        for k in [k for k in params if k.startswith("feat_")]:
            del params[k]
    model_class = MODEL_REGISTRY[model_name]
    if hasattr(model_class, "convert_optuna_params"):
        params = model_class.convert_optuna_params(params)
    return params


# ── CV objective (port of tune_model.create_objective + evaluate_fold) ───────
def _evaluate_fold_metrics(
    model, X_train, y_train, X_val, y_val, clusters_train,
    resample_method, random_state,
) -> Dict[str, float]:
    """Train on a fold and return train+val metrics (slim port of evaluate_fold)."""
    X_train, y_train, clusters_train = _filter_zero_score(
        X_train, y_train, clusters_train, type(model)
    )
    if "_is_zero_score" in X_val.columns:
        X_val = X_val.drop(columns=["_is_zero_score"])

    val_groups = [X_val["game_id"], X_val["turn"]] if "game_id" in X_val.columns else None

    X_train = _strip_id_columns_if_not_needed(X_train, model)
    X_val = _strip_id_columns_if_not_needed(X_val, model)

    if resample_method is not None and not getattr(model, "DISABLE_RESAMPLING", False):
        X_train, y_train, clusters_train = apply_resampling(
            X_train, y_train, clusters_train,
            method=resample_method, random_state=random_state,
        )

    model.fit(X_train, y_train, clusters=clusters_train)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = model.predict(X_train)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = model.predict(X_val, groups=val_groups)

    return {
        "roc_auc": roc_auc_score(y_val, y_val_proba),
        "brier_score": brier_score_loss(y_val, y_val_proba),
        "log_loss": log_loss(y_val, y_val_proba),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
        "train_roc_auc": roc_auc_score(y_train, y_train_proba),
        "train_brier_score": brier_score_loss(y_train, y_train_proba),
        "train_log_loss": log_loss(y_train, y_train_proba),
        "train_balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
    }


def _make_objective(model_name, metric, random_state, resample_method, precomputed, mode):
    import optuna

    model_class = MODEL_REGISTRY[model_name]
    suggest_fn = SEARCH_SPACES[model_name]
    df, X, y, cv_splits = precomputed

    def objective(trial) -> float:
        if mode == "variables":
            params = {"include_features": suggest_feature_variants(trial)}
        elif mode == "params":
            params = suggest_fn(trial)
        else:  # both
            params = suggest_fn(trial)
            params["include_features"] = suggest_feature_variants(trial)

        fold_penalized = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            try:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                clusters_train = df.iloc[train_idx]["game_id"]
                model = model_class(random_state=random_state, **params)
                fold_metrics = _evaluate_fold_metrics(
                    model, X_train, y_train, X_val, y_val, clusters_train,
                    resample_method, random_state,
                )
            except optuna.TrialPruned:
                raise
            except Exception:
                return float("inf") if metric in MINIMIZE_METRICS else 0.0

            fold_val = fold_metrics[metric]
            fold_train = fold_metrics.get(f"train_{metric}")
            if fold_train is not None:
                gap = (fold_val - fold_train) if metric in MINIMIZE_METRICS else (fold_train - fold_val)
                penalty = gap ** 2 * 10
                fold_val = fold_val + penalty if metric in MINIMIZE_METRICS else fold_val - penalty
            fold_penalized.append(fold_val)

            trial.report(float(np.mean(fold_penalized)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_penalized))

    return objective


# ── tune entry points ────────────────────────────────────────────────────────
def load_best_params(path: str, model_name: str) -> Dict:
    """Load a saved ``best_params.json`` and convert it to model kwargs.

    Accepts the file :func:`run_tune` writes (``{"best_params": {...}}``) or a
    bare raw-params object. This is the *pre-trained hyperparameter set* path —
    no search runs.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("best_params", payload) if isinstance(payload, dict) else payload
    return convert_best_params(model_name, raw)


def run_tune(
    model_name: str,
    df: pd.DataFrame,
    *,
    search: str = "hyperparameters",
    n_trials: int = 100,
    objective: str = "brier_score",
    n_splits: int = 5,
    random_state: int = 42,
    resample_method: Optional[str] = None,
    storage: Optional[str] = None,
    use_variants: bool = False,
    save_params: Optional[str] = None,
) -> Dict:
    """Run an Optuna study over ``df`` and return the converted best-param kwargs.

    Writes the full best-params payload to ``save_params`` (if set) so a later
    run can ``load_params`` it. The returned dict is ready to splat into the
    model constructor.
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"No tuning search space for model '{model_name}'. "
            f"Tunable: {sorted(SEARCH_SPACES)}."
        )

    mode = SEARCH_TO_MODE.get(search)
    if mode is None:
        raise ValueError(
            f"unknown tune.search '{search}'. Expected one of {sorted(SEARCH_TO_MODE)}."
        )

    model_class = MODEL_REGISTRY[model_name]
    direction = "minimize" if objective in MINIMIZE_METRICS else "maximize"

    X, y = prepare_features(df, use_variant_columns=use_variants)
    cv_splits = get_kfold_splits(df, n_splits=n_splits, random_state=random_state)
    precomputed = (df, X, y, cv_splits)

    study_name = f"tune_{model_name}_{mode}"
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=10),
        storage=storage,
        load_if_exists=True,
    )

    # Enqueue the model's coded defaults as the first trial of a fresh study.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) == 0:
        default_params = {}
        if mode in ("params", "both") and hasattr(model_class, "optuna_default_params"):
            default_params.update(model_class.optuna_default_params())
        if mode in ("variables", "both"):
            default_params.update(_reverse_map_features(model_class))
        if default_params:
            study.enqueue_trial(default_params)

    obj = _make_objective(model_name, objective, random_state, resample_method, precomputed, mode)
    study.optimize(obj, n_trials=n_trials, n_jobs=1)

    converted = convert_best_params(model_name, study.best_params)
    features = converted.get("include_features")
    if features is None and getattr(model_class, "DEFAULT_FEATURES", None) is not None:
        features = list(model_class.DEFAULT_FEATURES)

    if save_params is not None:
        payload = {
            "model": model_name,
            "search": search,
            "mode": mode,
            "metric": objective,
            "direction": direction,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "resample_method": resample_method,
            "features": features,
        }
        Path(save_params).parent.mkdir(parents=True, exist_ok=True)
        with open(save_params, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    return converted
