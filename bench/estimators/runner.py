"""Estimator-stage orchestrator — the **load-only** path (stage 2).

Resolves a ``fit:"pretrained"`` estimator entry, loads the saved ``model_dir``,
re-runs inference on the canonical turns table (rebuilding the exact feature
matrix the saved ``metadata.selected_features`` expects), and writes a validated
``predictions.csv`` (input metadata rows + ``predicted_win_probability``).

``fit:"train"`` (and ``tune``) raise :class:`NotImplementedError` — they land in
stage 6. ``fit:"pretrained"`` is **model_dir-only**: weights are loaded and
re-inferred; no raw prediction CSV is ever read as an estimator artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..catalog import Catalog
from ..config import RunConfig
from ..data import apply_filter_spec
from .errors import EstimatorError
from .features import build_feature_frame, needs_variant_columns, prepare_features
from .registry import load_model


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
    model_dir: str


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


def _strip_id_columns_if_not_needed(X: pd.DataFrame, model) -> pd.DataFrame:
    """Drop ID columns when the model doesn't require them (port of the evaluator)."""
    required_ids = getattr(model, "REQUIRES_ID_COLUMNS", None)
    if required_ids is None:
        id_cols_to_strip = ["game_id", "turn", "player_id", "experiment"]
        cols_to_keep = [c for c in X.columns if c not in id_cols_to_strip]
        return X[cols_to_keep]
    return X


def _apply_predict_subset(df: pd.DataFrame, subset, catalog: Optional[Catalog]) -> pd.DataFrame:
    """Narrow the inference frame to the entry's ``predict_subset`` (benchmark.md §4.6).

    ``"all"`` (default) keeps everything; ``"non_llm"`` / ``"llm"`` use the
    catalog's non-LLM experiment groups; a ``{"experiments": [...]}`` object keeps
    only those experiments.
    """
    if subset in (None, "all"):
        return df

    col = "experiment" if "experiment" in df.columns else "condition"

    if isinstance(subset, dict):
        experiments = subset.get("experiments")
        if not experiments:
            raise EstimatorError(
                f"predict_subset object must carry a non-empty 'experiments' list; got {subset!r}."
            )
        return df[df[col].isin(experiments)]

    if subset in ("non_llm", "llm"):
        if catalog is None:
            raise EstimatorError(
                f"predict_subset='{subset}' needs the model/experiment catalogs to "
                f"resolve the non-LLM experiment group."
            )
        non_llm = set(catalog.non_llm_experiments())
        if subset == "non_llm":
            return df[df[col].isin(non_llm)]
        return df[~df[col].isin(non_llm)]

    raise EstimatorError(
        f"unknown predict_subset '{subset}'. Expected 'all' | 'non_llm' | 'llm' "
        f"| {{'experiments': [...]}}."
    )


def run_estimator(
    cfg: RunConfig,
    stage_raw: dict,
    catalog: Optional[Catalog] = None,
) -> EstimatorResult:
    """Run one estimator stage (load-only) and write its ``predictions.csv``."""
    stage_id = stage_raw["id"]
    model_name = stage_raw["model"]
    fit = stage_raw.get("fit")

    if fit == "train":
        raise NotImplementedError(
            f"estimator '{stage_id}': fit='train' is not implemented yet "
            f"(the train/tune pipeline lands in stage 6). Use fit='pretrained'."
        )
    if "tune" in stage_raw:
        raise NotImplementedError(
            f"estimator '{stage_id}': 'tune' is not implemented yet (stage 6)."
        )
    if fit != "pretrained":
        raise EstimatorError(
            f"estimator '{stage_id}': unsupported fit='{fit}' for the load-only path."
        )

    pretrained = stage_raw.get("pretrained") or {}
    model_dir = pretrained.get("model_dir")
    if not model_dir:
        raise EstimatorError(
            f"estimator '{stage_id}': fit='pretrained' requires pretrained.model_dir."
        )
    # model_dir is an INPUT read as-authored — it is NOT re-rooted by output.suffix
    # (only save-paths are; benchmark.md §2.1). So the tracked pretrained/<id>/ store
    # is reused across the default and -dev/-cross variants alike.

    # Load — load_model raises FileNotFoundError (missing metadata.json) / ValueError
    # (model_class not in registry); both surface loudly to the caller.
    model = load_model(model_dir)
    model_class = type(model)

    turns_csv = _table_path(cfg, "turns")
    if not Path(turns_csv).exists():
        raise EstimatorError(
            f"estimator '{stage_id}': turns table not found at '{turns_csv}'. "
            f"Run extract first (or point data.tables.turns at an existing CSV)."
        )

    use_variants = needs_variant_columns(model_class)
    df = build_feature_frame(turns_csv, use_variants=use_variants, filter_zero_score=False)

    if catalog is None:
        catalog = Catalog.from_run_config(cfg)
    df = apply_filter_spec(
        df,
        catalog=catalog,
        filter_spec=cfg.data.get("filter"),
        presets=cfg.filters,
    )
    if df.empty:
        raise EstimatorError(
            f"estimator '{stage_id}': data.filter selected zero rows from the turns table."
        )

    df_pred = _apply_predict_subset(df, stage_raw.get("predict_subset", "all"), catalog)
    if df_pred.empty:
        raise EstimatorError(
            f"estimator '{stage_id}': predict_subset selected zero rows from the turns table."
        )

    X, _ = prepare_features(df_pred, use_variant_columns=use_variants)
    if "_is_zero_score" in X.columns:
        X = X.drop(columns=["_is_zero_score"])
    X = _strip_id_columns_if_not_needed(X, model)

    y_pred = model.predict_proba(X)[:, 1]

    df_out = df_pred.copy()
    df_out["predicted_win_probability"] = y_pred
    save_df = df_out[[c for c in PREDICTION_OUTPUT_COLUMNS if c in df_out.columns]]

    save_path = cfg.output.resolve(
        stage_raw.get("save_predictions") or _default_predictions_path(cfg, stage_id)
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(save_path, index=False)

    return EstimatorResult(
        id=stage_id,
        model=model_name,
        predictions_path=save_path,
        n_rows=len(save_df),
        model_dir=str(model_dir),
    )
