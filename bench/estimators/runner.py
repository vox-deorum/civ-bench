"""Estimator-stage orchestrator — load (stage 2) **and** train/tune (stage 6).

Resolves one estimator entry and emits its ``predictions.csv``:

- ``fit:"pretrained"`` (stage 2) — load a saved ``model_dir`` and re-infer on the
  canonical turns table. ``model_dir`` is an INPUT read as-authored (never
  re-rooted by ``output.suffix``).
- ``fit:"train"`` (stage 6) — fit a fresh model on this run's data, optionally
  preceded by an Optuna ``tune`` search. ``predict:"in_sample"`` deploys one
  model (saved to ``train.save_model``) and predicts ``predict_subset``;
  ``predict:"cross_val"`` emits honest out-of-fold predictions + an aggregated
  ``feature_importance.csv``.

The **cross variant** is an ordinary ``fit:"train"`` run with
``train.train_subset:"non_llm"`` (train on Vanilla/Null seats, predict everyone)
paired with ``output.suffix:"-cross"`` so its whole report lands in
``reports-cross/`` (benchmark.md §2.1, §4.4).

Hyperparameter precedence (highest wins): explicit ``params`` → ``tune.load_params``
→ a fresh ``tune`` search (``save_params``) → the model class's coded defaults.
Determinism is threaded from the top-level ``seed`` into splits, resampling, and
model init.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..catalog import Catalog
from ..config import RunConfig
from ..config.filters import resolve_filter_spec
from ..data import apply_filter_spec, incomplete_experiments_from_games
from .errors import EstimatorError
from .features import build_feature_frame, needs_variant_columns, prepare_features
from .registry import MODEL_REGISTRY, load_model


# Columns kept in a prediction output CSV (port of cli_utils.PREDICTION_OUTPUT_COLUMNS).
PREDICTION_OUTPUT_COLUMNS = [
    "experiment", "game_id", "player_id", "civilization",
    "turn", "max_turn", "is_winner",
    "predicted_win_probability", "turn_progress",
]


@dataclass
class EstimatorResult:
    id: str
    model: str
    predictions_path: str
    n_rows: int
    model_dir: Optional[str] = None        # pretrained source / in_sample save dir
    importance_path: Optional[str] = None  # cross_val feature_importance.csv


def _table_path(cfg: RunConfig, key: str) -> str:
    tables = cfg.data.get("tables", {}) or {}
    path = tables.get(key)
    if not path:
        raise EstimatorError(
            f"data.tables.{key} is not set; an estimator needs the canonical "
            f"'{key}' CSV to re-infer on."
        )
    return path


def _default_predictions_path(cfg: RunConfig, stage_id: str) -> str:
    return f"{cfg.output.root}/estimators/{stage_id}/predictions.csv"


def _exp_col(df: pd.DataFrame) -> str:
    return "experiment" if "experiment" in df.columns else "condition"


def _subset_experiment_list(
    df: pd.DataFrame, subset, catalog: Optional[Catalog]
) -> Optional[List[str]]:
    """Resolve a train/predict subset spec to a concrete experiment list.

    Returns ``None`` for ``"all"`` (no narrowing). ``"non_llm"`` is the cross
    split (Vanilla/Null experiments); ``"llm"`` is its complement; a
    ``{"experiments": [...]}`` object names experiments directly.
    """
    if subset in (None, "all"):
        return None
    col = _exp_col(df)

    if isinstance(subset, dict):
        experiments = subset.get("experiments")
        if not experiments:
            raise EstimatorError(
                f"subset object must carry a non-empty 'experiments' list; got {subset!r}."
            )
        return list(experiments)

    if subset in ("non_llm", "llm"):
        if catalog is None:
            raise EstimatorError(
                f"subset='{subset}' needs the model/experiment catalogs to resolve "
                f"the non-LLM experiment group."
            )
        non_llm = set(catalog.non_llm_experiments())
        present = set(df[col].unique())
        if subset == "non_llm":
            return sorted(present & non_llm)
        return sorted(present - non_llm)

    raise EstimatorError(
        f"unknown subset '{subset}'. Expected 'all' | 'non_llm' | 'llm' "
        f"| {{'experiments': [...]}}."
    )


def _narrow_to_subset(
    df: pd.DataFrame, subset, catalog: Optional[Catalog]
) -> pd.DataFrame:
    """Filter ``df`` to a train/predict subset (no-op for ``'all'``)."""
    exps = _subset_experiment_list(df, subset, catalog)
    if exps is None:
        return df
    return df[df[_exp_col(df)].isin(exps)]


def _model_kwargs_from_features(stage_raw: dict, base: dict) -> dict:
    """Apply an estimator ``features`` block onto already-resolved kwargs.

    ``features.include`` (when not null) overrides any include set by tuning;
    ``features.exclude`` sets the exclude list. Hyperparameters are untouched.
    """
    features = stage_raw.get("features")
    if not features:
        return base
    out = dict(base)
    include = features.get("include")
    exclude = features.get("exclude")
    if include is not None:
        out["include_features"] = include
    if exclude is not None:
        out["exclude_features"] = exclude
    return out


# ── pretrained (stage 2) ─────────────────────────────────────────────────────
def _run_pretrained(
    cfg: RunConfig, stage_raw: dict, catalog: Catalog
) -> EstimatorResult:
    stage_id = stage_raw["id"]
    model_name = stage_raw["model"]

    pretrained = stage_raw.get("pretrained") or {}
    model_dir = pretrained.get("model_dir")
    if not model_dir:
        raise EstimatorError(
            f"estimator '{stage_id}': fit='pretrained' requires pretrained.model_dir."
        )
    # model_dir is an INPUT read as-authored — NOT re-rooted by output.suffix.
    model = load_model(model_dir)
    model_class = type(model)

    use_variants = needs_variant_columns(model_class)
    df = _load_and_filter(cfg, stage_id, catalog, use_variants)

    df_pred = _narrow_to_subset(df, stage_raw.get("predict_subset", "all"), catalog)
    if df_pred.empty:
        raise EstimatorError(
            f"estimator '{stage_id}': predict_subset selected zero rows from the turns table."
        )

    from .training import _strip_id_columns_if_not_needed  # lazy: keeps dry-run import light

    X, _ = prepare_features(df_pred, use_variant_columns=use_variants)
    if "_is_zero_score" in X.columns:
        X = X.drop(columns=["_is_zero_score"])
    X = _strip_id_columns_if_not_needed(X, model)

    y_pred = model.predict_proba(X)[:, 1]
    df_out = df_pred.copy()
    df_out["predicted_win_probability"] = y_pred

    save_path = _write_predictions(cfg, stage_raw, stage_id, df_out)
    return EstimatorResult(
        id=stage_id, model=model_name, predictions_path=save_path,
        n_rows=_n_pred_rows(df_out), model_dir=str(model_dir),
    )


# ── train / tune (stage 6) ───────────────────────────────────────────────────
def _resolve_hyperparams(
    cfg: RunConfig, stage_raw: dict, model_name: str, df: pd.DataFrame, use_variants: bool
) -> dict:
    """Resolve model kwargs honouring the §4.3 precedence.

    Base layer is the tuning result (``load_params`` or a fresh search); explicit
    ``params`` override it; ``features`` then set include/exclude. Anything left
    unset falls through to the model class's coded defaults.
    """
    from .tuning import load_best_params, run_tune

    model_kwargs: dict = {}
    tune_block = stage_raw.get("tune")
    if tune_block and tune_block.get("enabled", True):
        load_params = tune_block.get("load_params")
        if load_params:
            # load_params is an INPUT (a pre-trained hyperparameter set) — as-authored.
            model_kwargs.update(load_best_params(load_params, model_name))
        else:
            model_kwargs.update(run_tune(
                model_name,
                df,
                search=tune_block.get("search", "hyperparameters"),
                n_trials=tune_block.get("n_trials", 100),
                objective=tune_block.get("objective", "brier_score"),
                n_splits=tune_block.get("n_splits", 5),
                random_state=cfg.seed,
                resample_method=_resample(tune_block.get("resample")),
                storage=cfg.output.resolve(tune_block.get("storage")),
                use_variants=use_variants,
                save_params=cfg.output.resolve(tune_block.get("save_params")),
            ))

    # Explicit params win over tuning.
    model_kwargs.update(stage_raw.get("params") or {})
    return _model_kwargs_from_features(stage_raw, model_kwargs)


def _run_train(cfg: RunConfig, stage_raw: dict, catalog: Catalog) -> EstimatorResult:
    from .training import run_cross_val, run_full_train

    stage_id = stage_raw["id"]
    model_name = stage_raw["model"]
    model_class = MODEL_REGISTRY[model_name.lower()]
    predict = stage_raw.get("predict", "in_sample")
    train_block = stage_raw.get("train") or {}
    train_subset = train_block.get("train_subset", "all")
    resample_method = _resample(train_block.get("resample"))

    tune_block = stage_raw.get("tune") or {}
    tune_on = bool(tune_block) and tune_block.get("enabled", True)
    tune_mode_uses_variants = (
        tune_on
        and not tune_block.get("load_params")
        and tune_block.get("search", "hyperparameters") in ("features", "both")
    )
    use_variants = needs_variant_columns(model_class) or tune_mode_uses_variants

    df = _load_and_filter(cfg, stage_id, catalog, use_variants)

    model_kwargs = _resolve_hyperparams(cfg, stage_raw, model_name, df, use_variants)

    if predict == "cross_val":
        train_experiments = _subset_experiment_list(df, train_subset, catalog)
        result = run_cross_val(
            model_class, model_kwargs, df,
            n_splits=train_block.get("n_splits", 5),
            random_state=cfg.seed,
            resample_method=resample_method,
            use_variants=use_variants,
            train_experiments=train_experiments,
        )
        save_path = _write_predictions(cfg, stage_raw, stage_id, result.predictions)
        importance_path = None
        if train_block.get("save_importance") and result.feature_importance is not None:
            importance_path = _write_importance(cfg, save_path, result.feature_importance)
        return EstimatorResult(
            id=stage_id, model=model_name, predictions_path=save_path,
            n_rows=_n_pred_rows(result.predictions), importance_path=importance_path,
        )

    # predict: in_sample — fit one model on train_subset, predict predict_subset.
    df_train = _narrow_to_subset(df, train_subset, catalog)
    if df_train.empty:
        raise EstimatorError(
            f"estimator '{stage_id}': train_subset selected zero rows to train on."
        )
    df_pred = _narrow_to_subset(df, stage_raw.get("predict_subset", "all"), catalog)
    if df_pred.empty:
        raise EstimatorError(
            f"estimator '{stage_id}': predict_subset selected zero rows to predict on."
        )

    result = run_full_train(
        model_class, model_kwargs, df_train, df_pred,
        random_state=cfg.seed,
        resample_method=resample_method,
        use_variants=use_variants,
    )

    save_dir = _save_model(cfg, stage_raw, model_name, result.model)
    save_path = _write_predictions(cfg, stage_raw, stage_id, result.predictions)
    return EstimatorResult(
        id=stage_id, model=model_name, predictions_path=save_path,
        n_rows=_n_pred_rows(result.predictions), model_dir=save_dir,
    )


# ── shared helpers ───────────────────────────────────────────────────────────
def _resample(value) -> Optional[str]:
    """Map a config resample value to apply_resampling's argument (``"none"``→None)."""
    if value in (None, "none"):
        return None
    return value


def _load_and_filter(
    cfg: RunConfig, stage_id: str, catalog: Catalog, use_variants: bool
) -> pd.DataFrame:
    """Build the engineered turns frame and apply the global ``data.filter``."""
    turns_csv = _table_path(cfg, "turns")
    if not Path(turns_csv).exists():
        raise EstimatorError(
            f"estimator '{stage_id}': turns table not found at '{turns_csv}'. "
            f"Run extract first (or point data.tables.turns at an existing CSV)."
        )
    df = build_feature_frame(turns_csv, use_variants=use_variants, filter_zero_score=False)
    filter_spec = cfg.data.get("filter")
    global_spec = resolve_filter_spec(filter_spec, cfg.filters, "data.filter")
    condition_incomplete = None
    if global_spec.get("min_condition_completeness") is not None:
        games_csv = _table_path(cfg, "games")
        if not Path(games_csv).exists():
            raise EstimatorError(
                f"estimator '{stage_id}': games table not found at '{games_csv}'. "
                f"data.filter.min_condition_completeness needs the controlled grid. "
                f"Run extract first."
            )
        condition_incomplete = incomplete_experiments_from_games(
            games_csv, global_spec
        )
    df = apply_filter_spec(
        df, catalog=catalog,
        filter_spec=filter_spec, presets=cfg.filters,
        condition_incomplete=condition_incomplete,
    )
    if df.empty:
        raise EstimatorError(
            f"estimator '{stage_id}': data.filter selected zero rows from the turns table."
        )
    return df


def _write_predictions(
    cfg: RunConfig, stage_raw: dict, stage_id: str, df_out: pd.DataFrame
) -> str:
    save_df = df_out[[c for c in PREDICTION_OUTPUT_COLUMNS if c in df_out.columns]]
    save_path = cfg.output.resolve(
        stage_raw.get("save_predictions") or _default_predictions_path(cfg, stage_id)
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(save_path, index=False)
    return save_path


def _write_importance(cfg: RunConfig, predictions_path: str, importance: pd.DataFrame) -> str:
    out = Path(predictions_path).parent / "feature_importance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out, index=False)
    return str(out)


def _save_model(cfg: RunConfig, stage_raw: dict, model_name: str, model) -> Optional[str]:
    """Save the fitted in_sample model dir (skipped for the trivial naive model)."""
    save_model = (stage_raw.get("train") or {}).get("save_model")
    if not save_model or model is None:
        return None
    if model_name.lower() == "naive":
        return None  # trivially retrainable; mirrors the source's save skip
    save_dir = cfg.output.resolve(save_model)
    model.save(save_dir)
    return save_dir


def _n_pred_rows(df_out: pd.DataFrame) -> int:
    return int(len(df_out))


def run_estimator(
    cfg: RunConfig,
    stage_raw: dict,
    catalog: Optional[Catalog] = None,
) -> EstimatorResult:
    """Run one estimator stage and write its ``predictions.csv``."""
    stage_id = stage_raw["id"]
    fit = stage_raw.get("fit")

    if catalog is None:
        catalog = Catalog.from_run_config(cfg)

    if fit == "train":
        return _run_train(cfg, stage_raw, catalog)
    if fit == "pretrained":
        if "tune" in stage_raw or "train" in stage_raw:
            raise EstimatorError(
                f"estimator '{stage_id}': fit='pretrained' must not carry train/tune blocks."
            )
        return _run_pretrained(cfg, stage_raw, catalog)

    raise EstimatorError(
        f"estimator '{stage_id}': unsupported fit='{fit}'. Expected 'train' | 'pretrained'."
    )
