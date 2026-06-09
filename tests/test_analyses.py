"""Stage 4 — analyses tests.

Exercise the pluggable analysis layer on tiny synthetic fixtures (no machine data
roots, per AGENTS.md): the registry + context wiring, per-module param validation,
the prediction/calibration/performance/exploratory modules end-to-end, the
controlled-design civ/cell calibration views, the ratings input-prep + group_by
compositing + bootstrap resample/readjust (the R fit itself is monkeypatched so the
suite runs without Rscript), and the ``run_analysis`` persistence integration.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bench.analyses import AnalysisError, get_analysis, list_analyses, run_analysis
from bench.analyses.grouping import grouping_label
from bench.analyses.prediction.metrics import compute_metric
from bench.catalog import Catalog
from bench.config import ConfigError, load_config


# ── synthetic data ────────────────────────────────────────────────────────────
PLAYER_TYPES = ["Vanilla", "GPT-OSS-120B", "Kimi-K2.5"]
CIVS = ["Rome", "Greece", "Egypt"]
LATE = [(4, 0.4), (6, 0.6), (8, 0.8)]
MAX_TURN = 10


def _build_csvs(tmp_path, n_games=12, controlled=False):
    """Write predictions / panel / games / tokens + a strength panel + audit trails."""
    pred, panel, games, tokens, strength = [], [], [], [], []
    rng = np.random.default_rng(0)
    for g in range(n_games):
        gid = f"G{g}"
        exp = "llm-standard" if g % 3 else "observe-vanilla-standard"
        seed = (g % 3) + 1 if controlled else -1
        rot = (g % 2) if controlled else -1
        games.append({"game_id": gid, "timestamp": "2026-01-01", "experiment": exp,
                      "seed": seed, "seating_rotation": rot})
        # strength of each seat increases with player index (winner = last seat)
        base = {"Vanilla": 0.18, "GPT-OSS-120B": 0.22, "Kimi-K2.5": 0.30}
        for pid, pt in enumerate(PLAYER_TYPES):
            civ = CIVS[(pid + g) % len(CIVS)]
            p = float(np.clip(base[pt] + rng.normal(0, 0.02), 0.02, 0.95))
            win = int(pt == "Kimi-K2.5")
            model = "VPAI" if pt == "Vanilla" else ("gpt-oss-120b" if "GPT" in pt else "kimi-k2.5")
            ratios = {"domination_ratio": 0.4 if pt == "Kimi-K2.5" else 0.1,
                      "culture_ratio": 0.1, "diplomatic_ratio": 0.1,
                      "science_ratio": 0.4 if pt != "Kimi-K2.5" else 0.1}
            panel.append({"experiment": exp, "game_id": gid, "turn": MAX_TURN, "player_id": pid,
                          "player_type": pt, "model": model, "strategist": "s", "config_slot": pid,
                          "civilization": civ, "score_ratio": p, "is_winner": win, **ratios})
            tokens.append({"experiment": exp, "game_id": gid, "player_id": pid, "player_type": pt,
                           "model_name": model, "model_base": model,
                           "input_tokens": 1000 * (pid + 1), "reasoning_tokens": 100,
                           "output_tokens": 200 * (pid + 1), "total_tokens": 0,
                           "focus_briefer_count": 0, "valid_turn_count": 5})
            for (turn, tp) in LATE:
                pred.append({"experiment": exp, "game_id": gid, "player_id": pid, "civilization": civ,
                             "turn": turn, "max_turn": MAX_TURN,
                             "is_winner": int(win and turn == LATE[-1][0]),
                             "predicted_win_probability": p, "turn_progress": turn / MAX_TURN})
            logit = float(np.log(p / (1 - p)))
            strength.append({"experiment": exp, "game_id": gid, "player_id": pid, "player_type": pt,
                             "civilization": civ, "seed": seed, "seating_rotation": rot,
                             "config_slot": pid, "controlled": controlled, "is_winner": win,
                             "weighted_strength": p, "relative_strength": p, "logit_strength": logit,
                             "adjusted_strength": p,
                             "adjust_method": "cell" if controlled else "civ"})

    paths = {}
    for name, rows in (("turns", pred), ("panel", panel), ("games", games), ("tokens", tokens)):
        df = pd.DataFrame(rows)
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = str(p)
    # predictions for the estimator
    pd.DataFrame(pred).to_csv(tmp_path / "predictions.csv", index=False)
    # strength panel + audit trails (adjust outputs)
    adj = tmp_path / "adjust"
    adj.mkdir(exist_ok=True)
    pd.DataFrame(strength).to_csv(adj / "player_strength_panel.csv", index=False)
    paths["adjust_dir"] = str(adj)
    return paths, pd.DataFrame(strength)


def _write_audit_trails(adj_dir, controlled, strength_df):
    civ_eff = pd.DataFrame({
        "civilization": CIVS,
        "civ_effect": [0.5, -0.2, -0.3],
        "n_rows": [10, 10, 10],
    })
    civ_eff.to_csv(f"{adj_dir}/civ_effects.csv", index=False)
    if controlled:
        van = strength_df[(strength_df["player_type"] == "Vanilla") & strength_df["controlled"]]
        cb_rows, cov_rows = [], []
        for (exp, seed, pid), grp in strength_df[strength_df["controlled"]].groupby(
            ["experiment", "seed", "player_id"]
        ):
            vg = van[(van["seed"] == seed) & (van["player_id"] == pid)]
            if not vg.empty:
                cb_rows.append({"experiment": exp, "pathway": "implicit", "seed": seed,
                                "player_id": pid, "civilization": grp["civilization"].iloc[0],
                                "cell_baseline": float(vg["logit_strength"].mean()),
                                "n_vanilla": len(vg), "n_games": grp["game_id"].nunique(),
                                "n_models": 1, "has_vanilla_baseline": True, "vanilla_connected": True})
            cov_rows.append({"experiment": exp, "seed": seed, "player_id": pid,
                             "civilization": grp["civilization"].iloc[0], "in_entirety": True,
                             "n_rows": len(grp), "n_vanilla": len(vg),
                             "has_baseline": not vg.empty, "missing": False})
        pd.DataFrame(cb_rows, columns=["experiment", "pathway", "seed", "player_id", "civilization",
                                       "cell_baseline", "n_vanilla", "n_games", "n_models",
                                       "has_vanilla_baseline", "vanilla_connected"]).to_csv(
            f"{adj_dir}/cell_baseline.csv", index=False)
        pd.DataFrame(cov_rows).to_csv(f"{adj_dir}/cell_coverage.csv", index=False)
    else:
        for fn, cols in (("cell_baseline.csv",
                          ["experiment", "pathway", "seed", "player_id", "civilization",
                           "cell_baseline", "n_vanilla", "n_games", "n_models",
                           "has_vanilla_baseline", "vanilla_connected"]),
                         ("cell_coverage.csv",
                          ["experiment", "seed", "player_id", "civilization", "in_entirety",
                           "n_rows", "n_vanilla", "has_baseline", "missing"])):
            pd.DataFrame(columns=cols).to_csv(f"{adj_dir}/{fn}", index=False)


@pytest.fixture
def env(tmp_path, write_spec, dev_spec):
    """A loaded RunConfig + Catalog wired to synthetic tmp CSVs, plus a run() helper."""
    controlled = True
    paths, strength_df = _build_csvs(tmp_path, controlled=controlled)
    _write_audit_trails(paths["adjust_dir"], controlled, strength_df)

    spec = dev_spec
    spec["output"] = {"root": str(tmp_path / "out"), "suffix": ""}
    spec["data"]["extract"]["enabled"] = False
    spec["data"]["tables"] = {k: paths[k] for k in ("turns", "panel", "games", "tokens")}
    spec["data"]["filter"] = "staff_recent"
    spec["estimators"] = [{
        "id": "est", "model": "attention_mlp", "fit": "pretrained", "predict": "in_sample",
        "enabled": True, "predict_subset": "all",
        "save_predictions": str(tmp_path / "predictions.csv"),
        "pretrained": {"model_dir": "pretrained/attention_mlp/"},
    }]
    spec["adjust"] = [{
        "id": "strength", "module": "strength", "enabled": True,
        "uses": {"estimators": ["est"]},
        "save": str(paths["adjust_dir"] + "/player_strength_panel.csv"),
        "params": {"block": "auto", "civ_adjust": "ols_logit"},
    }]
    spec["analyses"] = [{"id": "placeholder", "module": "prediction.compare", "enabled": True,
                         "uses": {"estimators": ["est"]}, "params": {}}]

    cfg = load_config(write_spec(spec))
    catalog = Catalog.from_run_config(cfg)

    def run(module, params=None, uses=None, sid="t", stage_filter=None):
        raw = {"id": sid, "module": module, "enabled": True,
               "uses": uses or {}, "params": params or {}}
        if stage_filter is not None:
            raw["filter"] = stage_filter
        return run_analysis(cfg, raw, catalog=catalog)

    run.cfg = cfg
    run.catalog = catalog
    return run


# ── registry / wiring ──────────────────────────────────────────────────────────
def test_registry_has_all_core_modules():
    expected = {
        "ratings.bradley_terry", "ratings.plackett_luce", "ratings.matchups",
        "prediction.evaluate", "prediction.compare",
        "calibration.reliability", "calibration.loss_by_progress",
        "calibration.civ_effects", "calibration.cell_baseline",
        "performance.score_ratio", "performance.strength_panel", "performance.turn_predicted",
        "exploratory.model_token_costs",
    }
    assert expected <= set(list_analyses())


def test_get_analysis_reserved_module_is_loud():
    with pytest.raises(AnalysisError, match="registry-reserved"):
        get_analysis("ratings.ablation_bt")


def test_get_analysis_unknown_module_is_loud():
    with pytest.raises(AnalysisError, match="unknown analysis module"):
        get_analysis("ratings.nonsense")


# ── grouping + metrics ──────────────────────────────────────────────────────────
def test_grouping_label_argmax():
    df = pd.DataFrame({"a": [1, 0], "b": [0, 1]})
    g = {"kind": "argmax", "columns": ["a", "b"], "labels": ["A", "B"]}
    assert list(grouping_label(df, g, "x")) == ["A", "B"]


def test_grouping_missing_column_is_loud():
    with pytest.raises(AnalysisError, match="missing source column"):
        grouping_label(pd.DataFrame({"a": [1]}), {"kind": "argmax", "columns": ["a", "z"]}, "x")


def test_metrics_match_sklearn():
    df = pd.DataFrame({
        "is_winner": [0, 1, 0, 1], "predicted_win_probability": [0.1, 0.9, 0.2, 0.8],
        "game_id": ["g", "g", "g", "g"], "turn": [1, 1, 2, 2],
    })
    assert compute_metric("roc_auc", df) == pytest.approx(1.0)
    assert compute_metric("brier_score", df) == pytest.approx(np.mean([0.01, 0.01, 0.04, 0.04]))
    assert 0.0 <= compute_metric("balanced_accuracy", df) <= 1.0


# ── config param validation (§6) ────────────────────────────────────────────────
def _validate(dev_spec, write_spec, module, params):
    dev_spec["analyses"] = [{"id": "x", "module": module, "enabled": True,
                             "uses": {"tables": ["strength"]} if module.startswith("ratings.") else {},
                             "params": params}]
    return load_config(write_spec(dev_spec))


def test_unknown_analysis_param_rejected(dev_spec, write_spec):
    with pytest.raises(ConfigError, match="unknown key"):
        _validate(dev_spec, write_spec, "prediction.evaluate", {"bogus": 1})


def test_bad_metric_rejected(dev_spec, write_spec):
    with pytest.raises(ConfigError, match="unknown metric"):
        _validate(dev_spec, write_spec, "prediction.evaluate", {"metrics": ["roc_auc", "nope"]})


def test_bad_matchups_mode_rejected(dev_spec, write_spec):
    with pytest.raises(ConfigError, match="mode"):
        _validate(dev_spec, write_spec, "ratings.matchups", {"mode": "weird"})


def test_bad_aggregate_rejected(dev_spec, write_spec):
    with pytest.raises(ConfigError, match="aggregate"):
        _validate(dev_spec, write_spec, "performance.turn_predicted", {"aggregate": "sum"})


def test_bad_n_bins_rejected(dev_spec, write_spec):
    with pytest.raises(ConfigError, match="n_bins"):
        _validate(dev_spec, write_spec, "calibration.reliability", {"n_bins": 0})


# ── prediction ──────────────────────────────────────────────────────────────────
def test_prediction_evaluate(env):
    r = env("prediction.evaluate", {"metrics": ["roc_auc", "brier_score"]}, {"estimators": ["est"]})
    assert not r.empty and "metrics" in r.table_paths
    tbl = pd.read_csv(r.table_paths["metrics"])
    assert {"model", "roc_auc", "brier_score", "n_rows"} <= set(tbl.columns)


def test_prediction_evaluate_applies_stage_filter(env):
    r = env(
        "prediction.evaluate",
        {"metrics": ["brier_score"]},
        {"estimators": ["est"]},
        stage_filter={"turn_range": [8, None]},
    )
    tbl = pd.read_csv(r.table_paths["metrics"])
    assert tbl.loc[0, "n_rows"] == 12 * len(PLAYER_TYPES)


def test_prediction_compare_needs_two(env):
    with pytest.raises(AnalysisError, match=">= 2"):
        env("prediction.compare", {}, {"estimators": ["est"]})


# ── calibration ──────────────────────────────────────────────────────────────────
def test_calibration_reliability(env):
    r = env("calibration.reliability", {"n_bins": 5}, {"estimators": ["est"]})
    assert "reliability" in r.table_paths and "ece" in r.table_paths


def test_calibration_loss_by_progress(env):
    r = env("calibration.loss_by_progress", {"n_bins": 4}, {"estimators": ["est"]})
    assert "loss_by_progress" in r.table_paths


def test_calibration_civ_effects(env):
    r = env("calibration.civ_effects", {}, {"tables": ["strength"]})
    assert not r.empty and "civ_effects" in r.table_paths


def test_calibration_cell_baseline_controlled(env):
    r = env("calibration.cell_baseline", {}, {"tables": ["strength"]})
    assert not r.empty and r.figure_paths  # one heatmap per seed


# ── performance ──────────────────────────────────────────────────────────────────
def test_performance_score_ratio(env):
    r = env("performance.score_ratio", {"target": "score_ratio",
            "predictors": ["player_type", "civilization"]})
    assert "player_type_effects" in r.table_paths and "coefficients" in r.table_paths


def test_performance_strength_panel(env):
    r = env("performance.strength_panel", {"metric": "adjusted_strength", "by": "player_type",
            "bootstrap_n": 50}, {"tables": ["strength"]})
    tbl = pd.read_csv(r.table_paths["by_identity"])
    assert {"player_type", "mean", "n_games", "ci_lower", "ci_upper", "preliminary"} <= set(tbl.columns)
    assert "cell_coverage" in r.table_paths  # controlled run surfaces the coverage report


def test_performance_turn_predicted(env):
    r = env("performance.turn_predicted", {"by": "player_type"}, {"estimators": ["est"]})
    assert "by_identity" in r.table_paths and "over_progress" in r.table_paths


# ── exploratory ──────────────────────────────────────────────────────────────────
def test_exploratory_model_token_costs(env):
    r = env("exploratory.model_token_costs", {"currency": "usd"}, {"tables": ["tokens"]})
    tbl = pd.read_csv(r.table_paths["token_costs"])
    assert "total_cost" in tbl.columns and "games" in tbl.columns


# ── ratings (R fit monkeypatched) ─────────────────────────────────────────────────
def _fake_bt(strength_df, margin=None, reference="Vanilla"):
    means = strength_df.groupby("player_type")["adjusted_strength"].mean()
    ref_mean = means.get(reference, means.mean())
    rows = []
    for pt, m in means.items():
        log_worth = float(m - ref_mean)
        rows.append({"player_type": pt, "log_worth": log_worth, "worth": float(np.exp(log_worth)),
                     "se_log_worth": 0.1, "z_value": log_worth / 0.1,
                     "p_value": 0.5, "elo": 1500 + 400 * log_worth / np.log(10),
                     "se_elo": 40.0, "mu": log_worth, "sigma": 0.1})
    return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)


def test_ratings_player_type(env, monkeypatch):
    monkeypatch.setattr("bench.analyses.ratings.bradley_terry.calculate_ratings_bt", _fake_bt)
    r = env("ratings.bradley_terry", {"group_by": ["player_type"], "ref": "Vanilla"},
            {"tables": ["strength"]})
    tbl = pd.read_csv(r.table_paths["ratings"])
    assert {"player_type", "elo"} <= set(tbl.columns)
    assert set(tbl["player_type"]) >= set(PLAYER_TYPES)


def test_ratings_strategy_group_by(env, monkeypatch):
    monkeypatch.setattr("bench.analyses.ratings.bradley_terry.calculate_ratings_bt", _fake_bt)
    r = env("ratings.bradley_terry",
            {"group_by": ["player_type", "strategy"], "ref": "Vanilla", "min_games": 1},
            {"tables": ["strength"]})
    tbl = pd.read_csv(r.table_paths["ratings"])
    assert {"player_type", "strategy", "composite_type", "elo"} <= set(tbl.columns)
    # composite identity = player_type-strategy (the strategy grouping label)
    assert tbl["composite_type"].str.contains("-").all()


def test_ratings_bootstrap_blocked_with_strategy(env, monkeypatch):
    monkeypatch.setattr("bench.analyses.ratings.bradley_terry.calculate_ratings_bt", _fake_bt)
    with pytest.raises(AnalysisError, match="bootstrap with a multi-dimension"):
        env("ratings.bradley_terry",
            {"group_by": ["player_type", "strategy"], "bootstrap": {"n": 5}},
            {"tables": ["strength"]})


def test_ratings_matchups_no_r(env):
    r = env("ratings.matchups", {"mode": "mean", "validate_ols": True}, {"tables": ["strength"]})
    assert "matchup" in r.table_paths and "ols_validation" in r.table_paths


# ── bootstrap internals (no R) ────────────────────────────────────────────────────
def test_bootstrap_resample_and_readjust(env):
    from bench.analyses.ratings import bootstrap as boot

    panel = pd.read_csv(env.cfg.adjust[0].raw["save"])
    rng = np.random.default_rng(1)
    resampled = boot.resample_games(panel, rng, stratified=True)
    assert resampled["game_id"].nunique() == panel["game_id"].nunique()  # synthetic ids, same count
    readj = boot.readjust(resampled, {"civ_adjust": "ols_logit", "baseline_experiment": None},
                          env.catalog)
    assert "adjusted_strength" in readj.columns
    assert np.isfinite(readj["adjusted_strength"]).all()


def test_ratings_with_bootstrap_player_type(env, monkeypatch):
    monkeypatch.setattr("bench.analyses.ratings.bradley_terry.calculate_ratings_bt", _fake_bt)
    r = env("ratings.bradley_terry",
            {"group_by": ["player_type"], "ref": "Vanilla", "weighted": True,
             "bootstrap": {"n": 8, "ci_level": 0.9}},
            {"tables": ["strength"]})
    tbl = pd.read_csv(r.table_paths["ratings"])
    assert "ci_lower" in tbl.columns and "ci_upper" in tbl.columns
