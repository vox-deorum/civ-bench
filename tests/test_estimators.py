"""Stage 2 — estimators (load-only) tests.

Exercise the registry's ``load_model`` dispatch, the feature pipeline, and the
``run_estimator`` load→infer→write path end-to-end on a tiny **synthetic** turns
fixture (no machine-specific data roots, per AGENTS.md). The pre-trained model
used throughout is :class:`ScorePredictor` — a training-free heuristic we can
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
from bench.estimators.features import (
    build_feature_frame,
    needs_variant_columns,
    prepare_features,
)
from bench.estimators.models import ScorePredictor, AttentionMLPPredictor
from bench.estimators.registry import MODEL_REGISTRY, get_model, list_models, load_model


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
        "report": {"template": "default", "out_dir": "reports/", "formats": ["md"]},
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


# ── train / tune deferred to stage 6 ─────────────────────────────────────────
def test_fit_train_raises_not_implemented(turns_csv, score_model_dir, tmp_path, write_spec):
    save_pred = tmp_path / "predictions.csv"
    cfg = load_config(write_spec(_spec(turns_csv, score_model_dir, save_pred)))
    stage_raw = {
        "id": "x", "model": "score", "fit": "train",
        "train": {"train_subset": "all"},
    }
    with pytest.raises(NotImplementedError):
        run_estimator(cfg, stage_raw)


def test_tune_block_raises_not_implemented(turns_csv, score_model_dir, tmp_path, write_spec):
    save_pred = tmp_path / "predictions.csv"
    cfg = load_config(write_spec(_spec(turns_csv, score_model_dir, save_pred)))
    stage_raw = {
        "id": "x", "model": "score", "fit": "pretrained",
        "tune": {"enabled": True},
        "pretrained": {"model_dir": str(score_model_dir)},
    }
    with pytest.raises(NotImplementedError):
        run_estimator(cfg, stage_raw)


def test_missing_model_dir_fails_loudly(turns_csv, tmp_path, write_spec):
    save_pred = tmp_path / "predictions.csv"
    cfg = load_config(write_spec(_spec(turns_csv, tmp_path / "nope", save_pred)))
    with pytest.raises(FileNotFoundError):
        run_estimator(cfg, cfg.estimators[0].raw)
