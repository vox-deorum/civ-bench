"""Training pipeline for victory-prediction estimators (stage 6).

Ported from ``../vox-deorum-analysis/models/utils/model_evaluator.py`` (the
full-train + k-fold cross-val logic) and the resampling/split helpers from
``models/utils/data_utils.py``. The CLI/notebook ``print(...)``-as-output style
and the single-dataset path assumptions are stripped; everything returns values
and is driven by the estimator config block.

Two entry points mirror the estimator ``predict`` axis (benchmark.md §4.2):

- :func:`run_full_train` (``predict: in_sample``) — fit one model on the
  ``train_subset`` rows, predict on the ``predict_subset`` rows, optionally save
  the fitted model dir (a later run's ``pretrained.model_dir`` points here).
- :func:`run_cross_val` (``predict: cross_val``) — K held-out folds over the
  data, emitting **honest out-of-fold** predictions (each row scored by a model
  that never trained on its game) plus an aggregated feature-importance table.

Determinism: the single ``random_state`` (threaded from the top-level ``seed``)
seeds the GroupKFold split order, the resamplers (SMOTE/SMOTENC/RandomUnderSampler),
xgboost, and — as of the fit-time ``_seed_torch`` — each torch model's global RNG
(weight init, dropout, the ``randperm`` shuffle). Selected feature order is
deterministic (first-occurrence dedupe, no set iteration). Together these make an
identical config re-run **byte-stable on the same machine and device**; results are
not promised to match bit-for-bit across CPUs/GPUs/BLAS builds, and torch
non-deterministic kernels are not force-disabled (that can hard-error on CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .features import prepare_features

ResampleMethod = Optional[str]  # None | "oversample" | "undersample" | "combined"


# ── resampling (port of data_utils.apply_resampling) ─────────────────────────
def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    clusters: Optional[pd.Series] = None,
    method: ResampleMethod = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Resample training rows to address class imbalance (imbalanced-learn).

    Only ever applied to *training* data — and only after ID columns have been
    stripped, so ``X`` is a purely numeric feature matrix (SMOTE would crash on a
    string ``game_id``/``experiment`` column). ``None`` returns the inputs untouched.

    Cluster ids (``game_id``, consumed downstream for cluster-robust SEs) are encoded
    as a temporary integer column and declared **categorical** to SMOTENC. That way
    each synthetic row inherits a *real* neighbour's cluster instead of the rounded
    average of two arbitrary game ids the old numeric-SMOTE path produced. On the way
    out the column is decoded back to the **original** ``game_id`` values (the previous
    code leaked the internal encoder integers). Undersampling only ever keeps real rows.
    """
    if method is None:
        return X, y, clusters

    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.under_sampling import RandomUnderSampler

    has_clusters = clusters is not None
    if has_clusters:
        unique_clusters = clusters.unique()
        cluster_to_int = {cid: idx for idx, cid in enumerate(unique_clusters)}
        X_with_clusters = X.copy()
        X_with_clusters["__cluster_id__"] = clusters.map(cluster_to_int).values
        # SMOTENC keeps the encoded cluster categorical: synthetic rows copy a real
        # neighbour's category rather than interpolating (and rounding) two game ids.
        cat_idx = [X_with_clusters.columns.get_loc("__cluster_id__")]
        oversampler = SMOTENC(categorical_features=cat_idx, random_state=random_state)
    else:
        X_with_clusters = X.copy()
        oversampler = SMOTE(random_state=random_state)

    if method == "oversample":
        X_resampled, y_resampled = oversampler.fit_resample(X_with_clusters, y)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_with_clusters, y)
    elif method == "combined":
        X_temp, y_temp = oversampler.fit_resample(X_with_clusters, y)
        undersampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = undersampler.fit_resample(X_temp, y_temp)
    else:
        raise ValueError(
            f"Unknown resampling method: {method!r}. "
            f"Choose from 'oversample', 'undersample', 'combined' (or None)."
        )

    if has_clusters:
        # SMOTENC/undersampling both yield exact encoder integers here; map them back
        # through unique_clusters so callers get the real game_ids they started with.
        cluster_ints = np.asarray(X_resampled["__cluster_id__"]).astype(int)
        cluster_ints = cluster_ints.clip(0, len(unique_clusters) - 1)
        clusters_resampled = pd.Series(
            np.asarray(unique_clusters)[cluster_ints], name=clusters.name
        )
        X_resampled = X_resampled.drop(columns=["__cluster_id__"])
    else:
        clusters_resampled = None

    X_resampled = pd.DataFrame(X_resampled, columns=X.columns).reset_index(drop=True)
    y_resampled = pd.Series(np.asarray(y_resampled), name=y.name)
    return X_resampled, y_resampled, clusters_resampled


def get_kfold_splits(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Game-grouped k-fold splits — whole games stay together in train or val.

    ``GroupKFold`` is deterministic given the row order, so threading the same
    frame yields identical folds run-to-run (``random_state`` is accepted for a
    uniform signature; GroupKFold itself does not shuffle).
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["game_id"]
    return [(train_idx, val_idx) for train_idx, val_idx in gkf.split(df, groups=groups)]


def _strip_id_columns_if_not_needed(X: pd.DataFrame, model) -> pd.DataFrame:
    """Drop ID columns when the model doesn't require them (port of the evaluator)."""
    required_ids = getattr(model, "REQUIRES_ID_COLUMNS", None)
    if required_ids is None:
        id_cols = ["game_id", "turn", "player_id", "experiment"]
        return X[[c for c in X.columns if c not in id_cols]]
    return X


# ── feature importance (port of aggregate_feature_importance) ────────────────
def aggregate_feature_importance(
    models: List, use_robust_se: bool = True
) -> Optional[pd.DataFrame]:
    """Mean feature importance across folds, or ``None`` if unsupported."""
    importance_dfs = []
    for i, model in enumerate(models):
        imp_df = model.get_feature_importance()
        if imp_df is None:
            return None
        imp_df = imp_df.copy()
        imp_df["fold"] = i
        importance_dfs.append(imp_df)

    if not importance_dfs:
        return None

    all_importance = pd.concat(importance_dfs, ignore_index=True)
    agg_cols = {"coefficient": ["mean", "std"]}
    has_robust = use_robust_se and "robust_se" in all_importance.columns
    if has_robust:
        agg_cols["robust_se"] = ["mean", "std"]
        agg_cols["z_statistic"] = ["mean", "std"]
        agg_cols["significant_95"] = "sum"

    agg = all_importance.groupby("feature").agg(agg_cols).reset_index()
    if has_robust:
        agg.columns = [
            "feature", "coef_mean", "coef_std",
            "robust_se_mean", "robust_se_std",
            "z_stat_mean", "z_stat_std", "n_folds_significant",
        ]
    else:
        agg.columns = ["feature", "coef_mean", "coef_std"]

    return agg.sort_values("coef_mean", ascending=False, key=abs)


# ── results ──────────────────────────────────────────────────────────────────
@dataclass
class TrainResult:
    """Output of :func:`run_full_train` / :func:`run_cross_val`."""

    predictions: pd.DataFrame                 # df_pred metadata + predicted_win_probability
    model: Optional[object] = None            # fitted model (in_sample only)
    feature_importance: Optional[pd.DataFrame] = None  # cross_val only
    selected_features: Optional[List[str]] = None
    n_train_rows: int = 0
    n_train_games: int = 0


def _filter_zero_score(
    X: pd.DataFrame, y: pd.Series, clusters: pd.Series, model_class
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Drop eliminated-player rows before training when the model requests it."""
    filter_zero = getattr(model_class, "FILTER_ZERO_SCORE", True)
    if filter_zero and "_is_zero_score" in X.columns:
        keep = ~X["_is_zero_score"]
        X, y, clusters = X[keep], y[keep], clusters[keep]
    if "_is_zero_score" in X.columns:
        X = X.drop(columns=["_is_zero_score"])
    return X, y, clusters


# ── full-train (predict: in_sample) ──────────────────────────────────────────
def run_full_train(
    model_class: type,
    model_kwargs: dict,
    df_train: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    random_state: int = 42,
    resample_method: ResampleMethod = None,
    use_variants: bool = False,
) -> TrainResult:
    """Fit one model on ``df_train`` and predict ``df_pred`` (no cross-validation).

    Mirrors the source ``run_full_prediction``: prepare features, drop
    zero-score training rows per the model flag, optionally resample, strip ID
    columns the model doesn't need, fit, then re-infer on the (possibly larger)
    prediction frame. The returned ``predictions`` carries the ``df_pred``
    metadata plus ``predicted_win_probability``.
    """
    X_train, y_train = prepare_features(df_train, use_variant_columns=use_variants)
    clusters_train = df_train["game_id"]
    X_train, y_train, clusters_train = _filter_zero_score(
        X_train, y_train, clusters_train, model_class
    )

    model = model_class(random_state=random_state, **model_kwargs)

    # Strip ID columns BEFORE resampling so SMOTE sees a purely numeric matrix
    # (string game_id/experiment would crash it). Models that keep their ID
    # columns (REQUIRES_ID_COLUMNS) all set DISABLE_RESAMPLING, so the two paths
    # never collide. Mirrors tuning._evaluate_fold_metrics.
    X_train = _strip_id_columns_if_not_needed(X_train, model)
    if resample_method is not None and not getattr(model, "DISABLE_RESAMPLING", False):
        X_train, y_train, clusters_train = apply_resampling(
            X_train, y_train, clusters_train,
            method=resample_method, random_state=random_state,
        )

    model.fit(X_train, y_train, clusters=clusters_train)

    X_pred, _ = prepare_features(df_pred, use_variant_columns=use_variants)
    if "_is_zero_score" in X_pred.columns:
        X_pred = X_pred.drop(columns=["_is_zero_score"])
    X_pred = _strip_id_columns_if_not_needed(X_pred, model)

    y_pred = model.predict_proba(X_pred)[:, 1]
    predictions = df_pred.copy()
    predictions["predicted_win_probability"] = y_pred

    return TrainResult(
        predictions=predictions,
        model=model,
        selected_features=model.get_selected_features(),
        n_train_rows=len(X_train),
        n_train_games=int(clusters_train.nunique()) if clusters_train is not None else 0,
    )


# ── cross-val OOF (predict: cross_val) ───────────────────────────────────────
def run_cross_val(
    model_class: type,
    model_kwargs: dict,
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    resample_method: ResampleMethod = None,
    use_variants: bool = False,
    train_experiments: Optional[List[str]] = None,
) -> TrainResult:
    """K-fold out-of-fold predictions over ``df`` (honest, held-out per game).

    Each row is predicted exactly once, by the fold's held-out model. When
    ``train_experiments`` is given, each fold's *training* indices are narrowed
    to those experiments (the validation/OOF coverage stays the whole frame) —
    the source's ``train_non_llm_only`` generalization setup. Feature importance
    is aggregated across folds when the model supports it.
    """
    X, y = prepare_features(df, use_variant_columns=use_variants)
    cv_splits = get_kfold_splits(df, n_splits=n_splits, random_state=random_state)

    train_mask = None
    if train_experiments is not None:
        exp_col = "condition" if "condition" in df.columns else "experiment"
        train_mask = df[exp_col].isin(train_experiments).values

    fold_models: List = []
    val_predictions_all: List[pd.DataFrame] = []
    meta_cols = [
        c for c in [
            "experiment", "game_id", "player_id", "civilization",
            "turn", "max_turn", "is_winner", "turn_progress",
        ] if c in df.columns
    ]

    n_train_rows = 0
    for train_idx, val_idx in cv_splits:
        if train_mask is not None:
            train_idx = train_idx[train_mask[train_idx]]

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        clusters_train = df.iloc[train_idx]["game_id"]

        X_train, y_train, clusters_train = _filter_zero_score(
            X_train, y_train, clusters_train, model_class
        )
        if "_is_zero_score" in X_val.columns:
            X_val = X_val.drop(columns=["_is_zero_score"])

        model = model_class(random_state=random_state, **model_kwargs)
        # Strip ID columns before resampling (numeric matrix for SMOTE); see the
        # note in run_full_train. Matches tuning._evaluate_fold_metrics.
        X_train = _strip_id_columns_if_not_needed(X_train, model)
        X_val = _strip_id_columns_if_not_needed(X_val, model)
        if resample_method is not None and not getattr(model, "DISABLE_RESAMPLING", False):
            X_train, y_train, clusters_train = apply_resampling(
                X_train, y_train, clusters_train,
                method=resample_method, random_state=random_state,
            )

        model.fit(X_train, y_train, clusters=clusters_train)
        n_train_rows += len(X_train)

        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_preds = df.iloc[val_idx][meta_cols].copy().reset_index(drop=True)
        val_preds["predicted_win_probability"] = y_val_pred
        val_predictions_all.append(val_preds)
        fold_models.append(model)

    predictions = pd.concat(val_predictions_all, ignore_index=True)
    feature_importance = aggregate_feature_importance(fold_models, use_robust_se=True)

    return TrainResult(
        predictions=predictions,
        model=None,
        feature_importance=feature_importance,
        selected_features=fold_models[0].get_selected_features() if fold_models else None,
        n_train_rows=n_train_rows,
        n_train_games=int(df["game_id"].nunique()),
    )
