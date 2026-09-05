"""Stage 2: estimators (load-only) tests.

Exercise the registry's ``load_model`` dispatch, the feature pipeline, and the
``run_estimator`` load→infer→write path end-to-end on a tiny **synthetic** turns
fixture (no machine-specific data roots, per AGENTS.md). The pre-trained model
used throughout is :class:`ScorePredictor`, a training-free heuristic we can
construct, save, and reload deterministically, so a saved-model re-inference must
reproduce a direct in-memory inference exactly (the stage-2 verification, in
miniature).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bench.config import load_config
from bench.estimators import EstimatorError, run_estimator
from bench.estimators.environment import _sole_default_target
from bench.estimators import features as feature_module
from bench.estimators.features import (
    build_feature_frame,
    needs_variant_columns,
    prepare_features,
)
from bench.estimators.models import ScorePredictor, AttentionMLPPredictor
from bench.estimators.models.base_predictor import BasePredictor
from bench.estimators.registry import MODEL_REGISTRY, get_model, list_models, load_model
from bench.estimators.training import apply_resampling


class _DummyPredictor(BasePredictor):
    """Minimal concrete predictor for exercising BasePredictor helpers (no fit)."""

    def fit(self, X, y, clusters=None, epoch_callback=None):
        return self

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


# ── synthetic turns fixture ──────────────────────────────────────────────────
# Every raw column the feature pipeline touches (apply_city_adjustments →
# add_relative_features → add_competitive_features → prepare_features).
_RAW_NUMERIC = [
    "cities", "population", "territory", "technologies", "policies",
    "military_strength", "military_units", "military_supply",
    "gold_per_turn", "production_per_turn", "food_per_turn",
    "culture_per_turn", "science_per_turn", "tourism_per_turn", "faith_per_turn",
    "votes", "minor_allies", "happiness_percentage", "religion_percentage",
    "highest_war_weariness", "active_wars", "truces", "friendships", "defensive_pacts",
]


def _make_turns_df(n_games: int = 2, n_players: int = 4, n_turns: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    civs = ["Rome", "Egypt", "Mongolia", "Spain", "Greece", "Japan"]
    for g in range(n_games):
        game_id = f"game-{g}"
        experiment = "exp-llm" if g == 0 else "exp-vanilla"
        winner = g % n_players
        for t in range(n_turns):
            scores = rng.uniform(50, 500, n_players)
            max_score = float(scores.max())
            order = np.argsort(-scores)
            rank_of = {int(p): int(r) + 1 for r, p in enumerate(order)}
            for p in range(n_players):
                row = {
                    "experiment": experiment,
                    "game_id": game_id,
                    "player_id": p,
                    "civilization": civs[(g * n_players + p) % len(civs)],
                    "turn": t,
                    "max_turn": n_turns - 1,
                    "score": float(scores[p]),
                    "max_score": max_score,
                    "rank": rank_of[p],
                    "is_winner": int(p == winner and t == n_turns - 1),
                }
                for c in _RAW_NUMERIC:
                    row[c] = float(rng.uniform(1, 100))
                row["cities"] = float(rng.integers(1, 6))
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def turns_csv(tmp_path):
    df = _make_turns_df()
    path = tmp_path / "turn_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def score_model_dir(tmp_path):
    """A saved ScorePredictor model dir (metadata.json + model.json)."""
    model = ScorePredictor()
    model.selected_features_ = ["score_ratio"]
    out = tmp_path / "score_model"
    model.save(str(out))
    return out


# ── registry ─────────────────────────────────────────────────────────────────
def test_registry_lists_all_eight_models():
    assert set(list_models()) == {
        "naive", "score", "baseline", "xgboost",
        "mlp", "grouped_mlp", "interaction_mlp", "attention_mlp",
    }
    assert set(MODEL_REGISTRY) == set(list_models())


def test_get_model_instantiates_by_id():
    assert isinstance(get_model("score", exponent=3.0), ScorePredictor)
    with pytest.raises(ValueError):
        get_model("nope")


def test_load_model_dispatches_on_metadata(score_model_dir):
    model = load_model(score_model_dir)
    assert isinstance(model, ScorePredictor)
    assert model.selected_features_ == ["score_ratio"]


def test_load_model_missing_metadata_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist")


def test_load_model_unknown_class_raises(tmp_path):
    bad = tmp_path / "bad_model"
    bad.mkdir()
    (bad / "metadata.json").write_text(json.dumps({"model_class": "NotARealPredictor"}))
    with pytest.raises(ValueError):
        load_model(bad)


# ── feature pipeline ─────────────────────────────────────────────────────────
def test_build_feature_frame_has_selected_features(turns_csv):
    df = build_feature_frame(str(turns_csv), use_variants=False, filter_zero_score=False)
    for col in ("score_ratio", "turn_progress", "science_share", "technologies_gap"):
        assert col in df.columns
    # turn_progress is the UNROUNDED feature (turn / max_turn), not loaders' rounded one.
    assert df["turn_progress"].between(0.0, 1.0).all()


def test_build_feature_frame_disables_chunked_dtype_inference(turns_csv, monkeypatch):
    original = feature_module.pd.read_csv
    calls = []

    def read_csv_spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_module.pd, "read_csv", read_csv_spy)
    build_feature_frame(str(turns_csv))
    assert calls[0]["low_memory"] is False


def test_rocm_default_override_requires_one_unambiguous_target():
    assert _sole_default_target("custom", ["custom"]) == "custom"
    assert _sole_default_target("gfx110X", ["gfx110X", "gfx120X"]) is None
    assert _sole_default_target("custom", ["gfx110X"]) is None


def test_needs_variant_columns():
    assert needs_variant_columns(ScorePredictor) is False
    # Attention defaults pull *_adj / raw columns outside SELECTED_FEATURES.
    assert needs_variant_columns(AttentionMLPPredictor) is True


# ── run_estimator: load-only end to end ──────────────────────────────────────
def _spec(turns_csv, model_dir, save_predictions, extra_estimator=None):
    estimators = [{
        "id": "score", "model": "score", "fit": "pretrained",
        "predict": "in_sample", "enabled": True, "predict_subset": "all",
        "save_predictions": str(save_predictions),
        "pretrained": {"model_dir": str(model_dir)},
    }]
    if extra_estimator:
        estimators.append(extra_estimator)
    return {
        "name": "test-estimators", "seed": 42,
        "output": {"root": "reports", "suffix": ""},
        "data": {
            "extract": {"enabled": False},
            "tables": {
                "turns": str(turns_csv),
                "panel": "runs/panel_data.csv",
                "games": "runs/game_data.csv",
                "tokens": "runs/model_token_usage.csv",
            },
        },
        "estimators": estimators,
        "analyses": [{
            "id": "pred_metrics", "module": "prediction.evaluate",
            "enabled": True, "uses": {"estimators": ["score"]},
            "params": {"metrics": ["roc_auc"]},
        }],
        "report": {"out_dir": "reports/", "formats": ["md"]},
    }


def test_run_estimator_writes_validated_predictions(turns_csv, score_model_dir, tmp_path, write_spec):
    save_pred = tmp_path / "predictions.csv"
    cfg = load_config(write_spec(_spec(turns_csv, score_model_dir, save_pred)))
    result = run_estimator(cfg, cfg.estimators[0].raw)

    assert result.predictions_path == str(save_pred)
    out = pd.read_csv(save_pred)
    expected_cols = [
        "experiment", "game_id", "player_id", "civilization",
        "turn", "max_turn", "is_winner",
        "predicted_win_probability", "turn_progress",
    ]
    assert list(out.columns) == expected_cols
    assert result.n_rows == len(out)
    # Probabilities are valid, and each (game_id, turn) group sums to ~1 (softmax).
    assert out["predicted_win_probability"].between(0.0, 1.0).all()
    grp = out.groupby(["game_id", "turn"])["predicted_win_probability"].sum()
    assert np.allclose(grp.values, 1.0)


def test_loaded_model_reproduces_direct_inference(turns_csv, score_model_dir, tmp_path, write_spec):
    """Saved-model re-inference == direct in-memory inference (stage-2 verification)."""
    save_pred = tmp_path / "predictions.csv"
    cfg = load_config(write_spec(_spec(turns_csv, score_model_dir, save_pred)))
    run_estimator(cfg, cfg.estimators[0].raw)
    got = pd.read_csv(save_pred)["predicted_win_probability"].to_numpy()

    # Reference: build the frame + predict directly with the same model.
    model = load_model(score_model_dir)
    df = build_feature_frame(str(turns_csv), use_variants=False, filter_zero_score=False)
    X, _ = prepare_features(df, use_variant_columns=False)
    ref = model.predict_proba(X)[:, 1]

    assert np.allclose(got, ref, atol=1e-12)


def test_predict_subset_non_llm_narrows_rows(turns_csv, score_model_dir, tmp_path, write_spec):
    save_pred = tmp_path / "predictions.csv"
    spec = _spec(turns_csv, score_model_dir, save_pred)
    spec["estimators"][0]["predict_subset"] = {"experiments": ["exp-vanilla"]}
    cfg = load_config(write_spec(spec))
    result = run_estimator(cfg, cfg.estimators[0].raw)
    out = pd.read_csv(save_pred)
    assert set(out["experiment"]) == {"exp-vanilla"}
    assert result.n_rows == len(out)


def test_data_filter_narrows_predictions(turns_csv, score_model_dir, tmp_path, write_spec):
    save_pred = tmp_path / "predictions.csv"
    spec = _spec(turns_csv, score_model_dir, save_pred)
    spec["data"]["filter"] = {
        "experiments": ["exp-llm"],
        "turn_range": [2, None],
    }
    cfg = load_config(write_spec(spec))
    result = run_estimator(cfg, cfg.estimators[0].raw)
    out = pd.read_csv(save_pred)

    assert set(out["experiment"]) == {"exp-llm"}
    assert out["turn"].min() >= 2
    assert result.n_rows == len(out)


def test_predict_subset_empty_raises(turns_csv, score_model_dir, tmp_path, write_spec):
    save_pred = tmp_path / "predictions.csv"
    spec = _spec(turns_csv, score_model_dir, save_pred)
    spec["estimators"][0]["predict_subset"] = {"experiments": ["no-such-exp"]}
    cfg = load_config(write_spec(spec))
    with pytest.raises(EstimatorError):
        run_estimator(cfg, cfg.estimators[0].raw)


def test_missing_model_dir_fails_loudly(turns_csv, tmp_path, write_spec):
    save_pred = tmp_path / "predictions.csv"
    cfg = load_config(write_spec(_spec(turns_csv, tmp_path / "nope", save_pred)))
    with pytest.raises(FileNotFoundError):
        run_estimator(cfg, cfg.estimators[0].raw)


# ── train / tune (stage 6) ───────────────────────────────────────────────────
def _train_spec(turns_csv, save_predictions, *, model="score", predict="in_sample",
                predict_subset="all", train=None, tune=None, params=None):
    estimator = {
        "id": model, "model": model, "fit": "train", "predict": predict,
        "enabled": True, "predict_subset": predict_subset,
        "save_predictions": str(save_predictions),
        "train": train if train is not None else {"train_subset": "all", "resample": "none"},
    }
    if tune is not None:
        estimator["tune"] = tune
    if params is not None:
        estimator["params"] = params
    return {
        "name": "test-train", "seed": 42,
        "output": {"root": "reports", "suffix": ""},
        "data": {
            "extract": {"enabled": False},
            "tables": {
                "turns": str(turns_csv),
                "panel": "runs/panel_data.csv",
                "games": "runs/game_data.csv",
                "tokens": "runs/model_token_usage.csv",
            },
        },
        "estimators": [estimator],
        "analyses": [{
            "id": "pred_metrics", "module": "prediction.evaluate",
            "enabled": True, "uses": {"estimators": [model]},
            "params": {"metrics": ["roc_auc"]},
        }],
        "report": {"out_dir": "reports/", "formats": ["md"]},
    }


def test_train_in_sample_writes_and_saves_model(turns_csv, tmp_path, write_spec):
    save_pred = tmp_path / "predictions.csv"
    save_model = tmp_path / "score_model_out"
    spec = _train_spec(
        turns_csv, save_pred,
        train={"train_subset": "all", "resample": "none", "save_model": str(save_model)},
        params={"exponent": 3.0},
    )
    cfg = load_config(write_spec(spec))
    result = run_estimator(cfg, cfg.estimators[0].raw)

    out = pd.read_csv(save_pred)
    assert list(out.columns) == [
        "experiment", "game_id", "player_id", "civilization",
        "turn", "max_turn", "is_winner", "predicted_win_probability", "turn_progress",
    ]
    # A fitted model dir was saved and is reloadable.
    assert (save_model / "metadata.json").exists()
    assert result.model_dir == str(save_model)
    reloaded = load_model(save_model)
    assert isinstance(reloaded, ScorePredictor)
    assert reloaded.exponent == 3.0


def test_train_score_reproduces_pretrained_inference(turns_csv, tmp_path, write_spec):
    """Stage-6 verification (miniature): a fresh in-sample train of the deterministic
    score model reproduces a pretrained load of the same model within tolerance."""
    # Pretrained reference at exponent 3.0.
    ref_model = ScorePredictor(exponent=3.0)
    ref_model.selected_features_ = ["score_ratio"]
    ref_dir = tmp_path / "ref_score"
    ref_model.save(str(ref_dir))
    ref_pred = tmp_path / "ref_pred.csv"
    cfg_ref = load_config(write_spec(_spec(turns_csv, ref_dir, ref_pred)))
    run_estimator(cfg_ref, cfg_ref.estimators[0].raw)
    ref = pd.read_csv(ref_pred)["predicted_win_probability"].to_numpy()

    # Fresh in-sample train at the same exponent.
    train_pred = tmp_path / "train_pred.csv"
    cfg_train = load_config(write_spec(
        _train_spec(turns_csv, train_pred, params={"exponent": 3.0})
    ))
    run_estimator(cfg_train, cfg_train.estimators[0].raw)
    got = pd.read_csv(train_pred)["predicted_win_probability"].to_numpy()

    assert np.allclose(got, ref, atol=1e-12)


def test_train_is_byte_stable(turns_csv, tmp_path, write_spec):
    """Re-running an identical train config is byte-stable (determinism via seed)."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    cfg_a = load_config(write_spec(_train_spec(turns_csv, a, params={"exponent": 2.5})))
    run_estimator(cfg_a, cfg_a.estimators[0].raw)
    cfg_b = load_config(write_spec(_train_spec(turns_csv, b, params={"exponent": 2.5})))
    run_estimator(cfg_b, cfg_b.estimators[0].raw)
    assert a.read_bytes() == b.read_bytes()


def test_train_subset_narrows_training(turns_csv, tmp_path, write_spec):
    """train_subset trains on a subset but predict_subset='all' predicts everyone."""
    save_pred = tmp_path / "predictions.csv"
    spec = _train_spec(
        turns_csv, save_pred,
        train={"train_subset": {"experiments": ["exp-vanilla"]}, "resample": "none"},
        predict_subset="all",
    )
    cfg = load_config(write_spec(spec))
    run_estimator(cfg, cfg.estimators[0].raw)
    out = pd.read_csv(save_pred)
    # Predictions cover BOTH experiments even though training used only one.
    assert set(out["experiment"]) == {"exp-llm", "exp-vanilla"}


def test_train_subset_empty_raises(turns_csv, tmp_path, write_spec):
    spec = _train_spec(
        turns_csv, tmp_path / "p.csv",
        train={"train_subset": {"experiments": ["no-such-exp"]}, "resample": "none"},
    )
    cfg = load_config(write_spec(spec))
    with pytest.raises(EstimatorError):
        run_estimator(cfg, cfg.estimators[0].raw)


def test_cross_val_oof_covers_all_rows_and_importance(tmp_path, write_spec):
    # A larger fixture so each of the 3 folds has training games + held-out games.
    big_csv = tmp_path / "big_turns.csv"
    _make_turns_df(n_games=6, n_players=4, n_turns=6).to_csv(big_csv, index=False)

    save_pred = tmp_path / "oof.csv"
    spec = _train_spec(
        turns_csv=big_csv, save_predictions=save_pred,
        model="xgboost", predict="cross_val",
        train={"train_subset": "all", "resample": "none", "n_splits": 3,
               "save_importance": True},
        params={"calibrate": False, "n_estimators": 5, "max_depth": 2},
    )
    cfg = load_config(write_spec(spec))
    result = run_estimator(cfg, cfg.estimators[0].raw)

    out = pd.read_csv(save_pred)
    full = build_feature_frame(str(big_csv), use_variants=False, filter_zero_score=False)
    # Every row appears exactly once in the OOF predictions (held-out per game).
    assert len(out) == len(full)
    assert out["predicted_win_probability"].between(0.0, 1.0).all()
    # Feature importance was aggregated and written next to the predictions.
    assert result.importance_path is not None
    imp = pd.read_csv(result.importance_path)
    assert "feature" in imp.columns and len(imp) > 0


def test_tune_load_params_skips_search(turns_csv, tmp_path, write_spec):
    """load_params reuses a saved best_params.json instead of running a search."""
    best_params = tmp_path / "best_params.json"
    best_params.write_text(json.dumps({"model": "score", "best_params": {"exponent": 2.0}}))

    save_pred = tmp_path / "tuned.csv"
    spec = _train_spec(
        turns_csv, save_pred,
        tune={"enabled": True, "engine": "optuna", "search": "hyperparameters",
              "load_params": str(best_params)},
    )
    cfg = load_config(write_spec(spec))
    run_estimator(cfg, cfg.estimators[0].raw)
    got = pd.read_csv(save_pred)["predicted_win_probability"].to_numpy()

    # Reference: pretrained score at exponent 2.0 (what load_params should yield).
    ref_model = ScorePredictor(exponent=2.0)
    ref_model.selected_features_ = ["score_ratio"]
    ref_dir = tmp_path / "ref2"
    ref_model.save(str(ref_dir))
    ref_pred = tmp_path / "ref2.csv"
    cfg_ref = load_config(write_spec(_spec(turns_csv, ref_dir, ref_pred)))
    run_estimator(cfg_ref, cfg_ref.estimators[0].raw)
    ref = pd.read_csv(ref_pred)["predicted_win_probability"].to_numpy()

    assert np.allclose(got, ref, atol=1e-12)


def test_tune_search_runs_and_saves_params(tmp_path, write_spec):
    """A fresh Optuna search runs, writes best_params.json, and feeds training."""
    big_csv = tmp_path / "big_turns.csv"
    _make_turns_df(n_games=6, n_players=4, n_turns=6).to_csv(big_csv, index=False)

    best_params = tmp_path / "best_params.json"
    save_pred = tmp_path / "tuned.csv"
    spec = _train_spec(
        turns_csv=big_csv, save_predictions=save_pred,
        tune={"enabled": True, "engine": "optuna", "search": "hyperparameters",
              "n_trials": 4, "objective": "brier_score", "n_splits": 3,
              "save_params": str(best_params), "load_params": None},
    )
    cfg = load_config(write_spec(spec))
    run_estimator(cfg, cfg.estimators[0].raw)

    assert best_params.exists()
    payload = json.loads(best_params.read_text())
    assert payload["model"] == "score" and "exponent" in payload["best_params"]
    out = pd.read_csv(save_pred)
    assert out["predicted_win_probability"].between(0.0, 1.0).all()


def test_explicit_params_override_load_params(turns_csv, tmp_path, write_spec):
    """Hyperparameter precedence: explicit params beat load_params (§4.3)."""
    best_params = tmp_path / "best_params.json"
    best_params.write_text(json.dumps({"best_params": {"exponent": 2.0}}))
    save_pred = tmp_path / "p.csv"
    spec = _train_spec(
        turns_csv, save_pred,
        tune={"enabled": True, "search": "hyperparameters", "load_params": str(best_params)},
        params={"exponent": 5.0},
    )
    cfg = load_config(write_spec(spec))
    run_estimator(cfg, cfg.estimators[0].raw)

    save_model = tmp_path / "m"
    spec2 = _train_spec(
        turns_csv, tmp_path / "p2.csv",
        train={"train_subset": "all", "resample": "none", "save_model": str(save_model)},
        params={"exponent": 5.0},
    )
    cfg2 = load_config(write_spec(spec2))
    run_estimator(cfg2, cfg2.estimators[0].raw)
    assert load_model(save_model).exponent == 5.0


# ── determinism helpers (WS1/WS6, pure-unit: no model fits) ─────────────────
def test_expand_wildcards_is_order_preserving_and_deduped():
    m = _DummyPredictor()
    # literals keep declared order; a wildcard expands in data-column order; the
    # repeated literal is de-duplicated (first occurrence wins): a list, not a set.
    out = m._expand_wildcards(
        ["z_lit", "*_share", "z_lit"], ["b_share", "a_share", "c_other"]
    )
    assert out == ["z_lit", "b_share", "a_share"]


def test_filter_features_include_order_is_declared_not_hashed():
    m = _DummyPredictor()
    m.include_features = ["gamma", "alpha", "beta"]
    df = pd.DataFrame({"alpha": [1.0], "beta": [2.0], "gamma": [3.0], "delta": [4.0]})
    out = m._filter_features(df)
    # Declared include order preserved (previously ``list(set(...))`` → hash order).
    assert list(out.columns) == ["gamma", "alpha", "beta"]
    assert m.selected_features_ == ["gamma", "alpha", "beta"]


def test_apply_resampling_oversample_keeps_columns_and_real_clusters():
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    })
    # imbalanced target with enough minority rows for SMOTENC (k_neighbors=5)
    y = pd.Series([1] * 8 + [0] * 32, name="is_winner")
    clusters = pd.Series([f"game-{i % 6}" for i in range(n)], name="game_id")

    Xr, yr, cr = apply_resampling(X, y, clusters, method="oversample", random_state=42)

    assert list(Xr.columns) == list(X.columns)      # __cluster_id__ removed, order kept
    assert len(Xr) == len(yr) == len(cr)
    assert cr.name == "game_id"
    # Synthetic rows inherit a real neighbour's game (never a rounded/encoder value).
    assert set(cr.unique()) <= set(clusters.unique())
    assert int(yr.sum()) > int(y.sum())             # minority class was oversampled


def test_apply_resampling_none_is_identity():
    X = pd.DataFrame({"f1": [1.0, 2.0]})
    y = pd.Series([0, 1])
    clusters = pd.Series(["g0", "g1"])
    Xr, yr, cr = apply_resampling(X, y, clusters, method=None)
    assert Xr is X and yr is y and cr is clusters
