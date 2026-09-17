"""Stage 4: analyses tests.

Exercise the pluggable analysis layer on tiny synthetic fixtures (no machine data
roots, per AGENTS.md): the registry + context wiring, per-module param validation,
the prediction/calibration/performance/exploratory modules end-to-end, the
controlled-design civ/cell calibration views, the ratings input-prep + group_by
compositing + bootstrap resample/readjust (the R fit itself is monkeypatched so the
suite runs without Rscript), and the ``run_analysis`` persistence integration.
"""

from __future__ import annotations

import json
from pathlib import Path

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
                           "focus_briefer_count": 0, "valid_turn_count": 5,
                           "failed_turn_count": 0, "failure_pct": 0.0,
                           "failed_turns": ""})
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
    sdf = pd.DataFrame(strength)
    # Mirror the adjust stage's persisted cell advantage (logit_strength - cell_baseline),
    # NaN on uncontrolled runs so the analysis only surfaces the view when controlled.
    sdf["cell_logit_advantage"] = np.nan
    if controlled:
        van = sdf[sdf["player_type"] == "Vanilla"]
        cb = van.groupby(["experiment", "seed"])["logit_strength"].mean()
        base = pd.Series([cb.get((e, s)) for e, s in zip(sdf["experiment"], sdf["seed"])],
                         index=sdf.index)
        cmask = sdf["controlled"].astype(bool)
        sdf.loc[cmask, "cell_logit_advantage"] = sdf.loc[cmask, "logit_strength"] - base[cmask]
    sdf.to_csv(adj / "player_strength_panel.csv", index=False)
    paths["adjust_dir"] = str(adj)
    return paths, sdf


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
                                "n_vanilla": len(vg), "win_rate": float(vg["is_winner"].mean()),
                                "n_games": grp["game_id"].nunique(),
                                "n_models": 1, "has_vanilla_baseline": True, "vanilla_connected": True})
            cov_rows.append({"experiment": exp, "seed": seed, "player_id": pid,
                             "civilization": grp["civilization"].iloc[0], "in_entirety": True,
                             "n_rows": len(grp), "n_vanilla": len(vg),
                             "has_baseline": not vg.empty, "missing": False})
        pd.DataFrame(cb_rows, columns=["experiment", "pathway", "seed", "player_id", "civilization",
                                       "cell_baseline", "n_vanilla", "win_rate", "n_games", "n_models",
                                       "has_vanilla_baseline", "vanilla_connected"]).to_csv(
            f"{adj_dir}/cell_baseline.csv", index=False)
        pd.DataFrame(cov_rows).to_csv(f"{adj_dir}/cell_coverage.csv", index=False)
    else:
        for fn, cols in (("cell_baseline.csv",
                          ["experiment", "pathway", "seed", "player_id", "civilization",
                           "cell_baseline", "n_vanilla", "win_rate", "n_games", "n_models",
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
        "ratings.outcome_matchups",
        "prediction.evaluate", "prediction.compare",
        "calibration.reliability", "calibration.loss_by_progress",
        "calibration.civ_effects", "calibration.cell_baseline",
        "performance.experiment_completeness",
        "performance.score_ratio", "performance.strength_panel", "performance.turn_predicted",
        "performance.usage_efficiency",
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


def test_matchups_mode_both_allowed(dev_spec, write_spec):
    cfg = _validate(dev_spec, write_spec, "ratings.matchups", {"mode": "both"})
    assert cfg.analyses[0].raw["params"]["mode"] == "both"


def test_outcome_matchups_config_accepts_panel(dev_spec, write_spec):
    dev_spec["analyses"] = [{
        "id": "outcomes", "module": "ratings.outcome_matchups", "enabled": True,
        "uses": {"tables": ["panel"]}, "params": {"include_score_ratio": "true"},
    }]
    cfg = load_config(write_spec(dev_spec))
    assert cfg.analyses[0].raw["params"]["include_score_ratio"] is True


def test_bad_aggregate_rejected(dev_spec, write_spec):
    with pytest.raises(ConfigError, match="aggregate"):
        _validate(dev_spec, write_spec, "performance.turn_predicted", {"aggregate": "sum"})


def test_bad_n_bins_rejected(dev_spec, write_spec):
    with pytest.raises(ConfigError, match="n_bins"):
        _validate(dev_spec, write_spec, "calibration.reliability", {"n_bins": 0})


# ── prediction ──────────────────────────────────────────────────────────────────
def test_prediction_evaluate(env):
    r = env("prediction.evaluate", {"metrics": ["roc_auc", "brier_score"]})
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
    r = env("calibration.reliability", {"n_bins": 5})
    assert "reliability" in r.table_paths and "ece" in r.table_paths


def test_calibration_loss_by_progress(env):
    r = env("calibration.loss_by_progress", {"n_bins": 4})
    assert "loss_by_progress" in r.table_paths


def test_calibration_loss_by_progress_keeps_turn_progress_one(env):
    pred_path = env.cfg.estimators[0].raw["save_predictions"]
    pred = pd.read_csv(pred_path)
    final = pred.copy()
    final["turn"] = final["max_turn"]
    final["turn_progress"] = 1.0
    pd.concat([pred, final], ignore_index=True).to_csv(pred_path, index=False)

    r = env("calibration.loss_by_progress", {"n_bins": 4})
    tbl = pd.read_csv(r.table_paths["loss_by_progress"])
    last = tbl[tbl["turn_progress_bin"] == "0.75-1.00"]
    assert not last.empty
    assert int(last["n_samples"].sum()) == 144


def test_calibration_civ_effects(env):
    r = env("calibration.civ_effects", {}, {"tables": ["strength"]})
    assert not r.empty and "civ_effects" in r.table_paths


def test_calibration_cell_baseline_controlled(env):
    r = env("calibration.cell_baseline", {}, {"tables": ["strength"]})
    assert not r.empty and r.figure_paths  # one heatmap per seed


def test_cell_baseline_excludes_explicit_conditions_implicit_row():
    from bench.analyses.calibration.cell_baseline import (
        CalibrationCellBaseline, _EXPLICIT_ROW,
    )

    # One seed, two player cells; experiment "base" is the explicit baseline source,
    # so it appears as both an explicit row and (redundantly) its own implicit row.
    rows = []
    for pathway, exp in (("explicit", "base"), ("implicit", "base"), ("implicit", "treat")):
        for pid in (0, 1):
            rows.append({"experiment": exp, "pathway": pathway, "seed": 7,
                         "player_id": pid, "civilization": CIVS[pid],
                         "cell_baseline": 0.3 + 0.1 * pid, "n_vanilla": 4,
                         "win_rate": 0.25 * (pid + 1)})
    seed_cb = pd.DataFrame(rows)

    fig = CalibrationCellBaseline("cb")._plot_seed(seed_cb, None, 7, vlim=2.0)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    # The explicit reference is pinned; "treat" reads against it; "base" implicit is dropped.
    assert _EXPLICIT_ROW in labels
    assert "treat" in labels
    assert "base" not in labels
    annotations = [t.get_text() for t in fig.axes[0].texts]
    assert any("win=25%" in text for text in annotations)


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
    assert "cell_coverage_summary" in r.table_paths
    assert "cell_coverage" in r.table_paths  # controlled run surfaces the coverage report
    assert "experiment_completeness" not in r.table_paths
    coverage = pd.read_csv(r.table_paths["cell_coverage_summary"])
    assert {"missing_cells", "no_vanilla_baseline_cells", "coverage_pct"} <= set(coverage.columns)

    # Controlled run also surfaces the logit-advantage alternative view.
    assert "by_identity_logit_advantage" in r.table_paths
    assert "logit_advantage" in r.figure_paths
    adv = pd.read_csv(r.table_paths["by_identity_logit_advantage"])
    assert {"player_type", "mean", "ci_lower", "ci_upper", "preliminary"} <= set(adv.columns)
    assert adv["mean"].notna().all()
    assert set(adv["player_type"]) == set(tbl["player_type"])  # same identities, just re-scaled


def test_paired_performance_player_type_figures_smoke(env):
    env.cfg.presentation = {"condition_pairing": {"enabled": True}}
    strength = env(
        "performance.strength_panel",
        {"metric": "adjusted_strength", "by": "player_type", "bootstrap_n": 20},
        {"tables": ["strength"]},
        sid="paired_strength",
    )
    score = env(
        "performance.score_ratio",
        {"target": "score_ratio", "predictors": ["player_type", "civilization"]},
        sid="paired_score",
    )
    assert "strength" in strength.figure_paths
    assert "logit_advantage" in strength.figure_paths
    assert "player_type_effects" in score.figure_paths


def test_performance_strength_panel_logit_advantage_uncontrolled_gate(env):
    # No cell-baselined rows (adjust_method != "cell") ⇒ the logit-advantage view is
    # omitted; only the base by_identity view renders.
    save = env.cfg.adjust[0].raw["save"]
    panel = pd.read_csv(save)
    panel["adjust_method"] = "civ"
    panel["controlled"] = False
    panel["cell_logit_advantage"] = np.nan
    panel.to_csv(save, index=False)

    r = env("performance.strength_panel", {"metric": "adjusted_strength", "by": "player_type",
            "bootstrap_n": 50}, {"tables": ["strength"]})
    assert "by_identity" in r.table_paths
    assert "by_identity_logit_advantage" not in r.table_paths
    assert "logit_advantage" not in r.figure_paths


def test_performance_experiment_completeness(env):
    r = env(
        "performance.experiment_completeness", {},
        {"tables": ["strength", "tokens"]},
    )
    assert "experiment_completeness" in r.table_paths
    assert "repeated_games" in r.table_paths
    completeness = pd.read_csv(r.table_paths["experiment_completeness"])
    assert {
        "required_games", "present_games", "missing_games", "repeated_slots",
        "avg_failure_count", "failure_pct", "warning",
    } <= set(completeness.columns)
    assert "repeat_warning" not in completeness.columns
    assert "decision_turn_failures" in r.table_paths


def test_experiment_completeness_reports_decision_turn_failures():
    from bench.analyses.performance.experiment_completeness import (
        attach_decision_turn_failures,
    )

    completeness = pd.DataFrame([
        {"experiment": "ctrl", "warning": "ok"},
        {"experiment": "clean", "warning": "1 missing slot(s)"},
    ])
    tokens = pd.DataFrame([
        {"experiment": "ctrl", "game_id": "g1", "player_id": 0,
         "player_type": "TestLLM", "model_base": "main", "valid_turn_count": 10,
         "failed_turn_count": 2, "failed_turns": "3,7"},
        # A second model row for the same trace must not double count its turns.
        {"experiment": "ctrl", "game_id": "g1", "player_id": 0,
         "player_type": "TestLLM", "model_base": "briefer", "valid_turn_count": 10,
         "failed_turn_count": 2, "failed_turns": "3,7"},
        {"experiment": "ctrl", "game_id": "g1", "player_id": 1,
         "player_type": "Vanilla", "model_base": "vanilla", "valid_turn_count": 10,
         "failed_turn_count": 0, "failed_turns": ""},
        {"experiment": "clean", "game_id": "g2", "player_id": 0,
         "player_type": "TestLLM", "model_base": "main", "valid_turn_count": 20,
         "failed_turn_count": 0, "failed_turns": ""},
    ])

    summary, failures = attach_decision_turn_failures(completeness, tokens)
    by_experiment = summary.set_index("experiment")
    assert by_experiment.loc["ctrl", "avg_failure_count"] == pytest.approx(1.0)
    assert by_experiment.loc["ctrl", "failure_pct"] == pytest.approx(0.1)
    assert "2 failed decision turn(s)" in by_experiment.loc["ctrl", "warning"]
    assert by_experiment.loc["clean", "avg_failure_count"] == 0
    assert by_experiment.loc["clean", "failure_pct"] == 0
    assert by_experiment.loc["clean", "warning"] == "1 missing slot(s)"
    assert len(failures) == 1
    assert failures.iloc[0]["failed_turns"] == "3,7"


def test_experiment_completeness_warns_for_legacy_token_csv():
    from bench.analyses.performance.experiment_completeness import (
        attach_decision_turn_failures,
    )

    completeness = pd.DataFrame([{"experiment": "ctrl", "warning": "ok"}])
    legacy_tokens = pd.DataFrame([
        {"experiment": "ctrl", "game_id": "g1", "player_id": 0,
         "valid_turn_count": 10},
    ])

    summary, failures = attach_decision_turn_failures(completeness, legacy_tokens)

    assert pd.isna(summary.iloc[0]["failed_turn_count"])
    assert pd.isna(summary.iloc[0]["avg_failure_count"])
    assert pd.isna(summary.iloc[0]["failure_pct"])
    assert summary.iloc[0]["warning"] == "decision-turn failure telemetry unavailable"
    assert failures.empty


def test_strength_panel_coverage_summary_counts_gaps():
    from bench.analyses.performance.strength_panel import PerformanceStrengthPanel

    cov = pd.DataFrame([
        {"experiment": "ctrl", "seed": 1, "player_id": 0, "n_rows": 4,
         "n_vanilla": 2, "has_baseline": True, "missing": False},
        {"experiment": "ctrl", "seed": 1, "player_id": 1, "n_rows": 3,
         "n_vanilla": 0, "has_baseline": False, "missing": False},
        {"experiment": "ctrl", "seed": 1, "player_id": 2, "n_rows": 0,
         "n_vanilla": 0, "has_baseline": False, "missing": True},
    ])

    summary = PerformanceStrengthPanel._coverage_summary(cov)

    row = summary.iloc[0]
    assert row["n_cells"] == 3
    assert row["present_cells"] == 2
    assert row["missing_cells"] == 1
    assert row["baseline_cells"] == 1
    assert row["no_vanilla_baseline_cells"] == 1
    assert row["coverage_pct"] == pytest.approx(2 / 3, abs=0.0001)


def _strength_rows(experiment, game_id, seed, rotation, players=(0, 1), controlled=True):
    return [
        {
            "experiment": experiment,
            "game_id": game_id,
            "player_id": pid,
            "player_type": "Vanilla" if pid == 0 else "TestLLM",
            "seed": seed,
            "seating_rotation": rotation,
            "controlled": controlled,
            "adjusted_strength": 0.5,
        }
        for pid in players
    ]


def test_experiment_completeness_complete_grid():
    from bench.analyses.performance.strength_panel import build_experiment_completeness

    rows = []
    for seed in (1, 2):
        for rot in (0, 1):
            rows.extend(_strength_rows("ctrl", f"g-s{seed}-r{rot}", seed, rot))
    out = build_experiment_completeness(pd.DataFrame(rows))

    comp = out["experiment_completeness"].iloc[0]
    assert comp["required_games"] == 4
    assert comp["present_games"] == 4
    assert comp["missing_games"] == 0
    assert comp["completeness_pct"] == pytest.approx(1.0)
    assert comp["repeated_slots"] == 0
    assert comp["warning"] == "ok"
    assert out["repeated_games"].empty
    assert out["experiment_completeness_gaps"].empty


def test_experiment_completeness_groups_missing_rotations():
    from bench.analyses.performance.strength_panel import build_experiment_completeness

    rows = []
    for seed in (1, 2):
        for rot in (0, 1):
            rows.extend(_strength_rows("full", f"full-s{seed}-r{rot}", seed, rot))
    rows.extend(_strength_rows("partial", "partial-s1-r0", 1, 0))

    out = build_experiment_completeness(pd.DataFrame(rows))
    comp = out["experiment_completeness"].set_index("experiment")
    assert comp.loc["partial", "required_games"] == 4
    assert comp.loc["partial", "present_games"] == 1
    assert comp.loc["partial", "missing_games"] == 3
    assert comp.loc["partial", "completeness_pct"] == pytest.approx(0.25)

    gaps = out["experiment_completeness_gaps"]
    p_gaps = gaps[gaps["experiment"] == "partial"].set_index("seed")
    assert p_gaps.loc[1, "missing_rotations"] == "1"
    assert p_gaps.loc[2, "missing_rotations"] == "0,1"


def test_emit_seating_param_allowed(dev_spec, write_spec):
    cfg = _validate(dev_spec, write_spec, "performance.experiment_completeness",
                    {"emit_seating": False})
    assert cfg.analyses[0].raw["params"]["emit_seating"] is False


def _seating_rows(experiment, game_id, seed, rotation, config_slots=(0, 1)):
    """Controlled strength rows carrying ``config_slot`` (needed for seating headers)."""
    return [
        {
            "experiment": experiment, "game_id": game_id, "player_id": pid,
            "player_type": "Vanilla" if pid == 0 else "TestLLM",
            "seed": seed, "seating_rotation": rotation, "config_slot": pid,
            "controlled": True, "adjusted_strength": 0.5,
        }
        for pid in config_slots
    ]


def test_generate_seating_opens_missing_cells():
    from bench.analyses.performance.seating import generate_seating_files

    # Design grid: configSlots {0,1} -> totalSeats 2; seeds {1,2} -> seedCount 2 = 4 cells.
    # Present (seed,rot): (1,0),(1,1),(2,0); missing (2,1) -> exactly one open cell.
    rows = []
    rows.extend(_seating_rows("ctrl", "g-s1-r0", 1, 0))
    rows.extend(_seating_rows("ctrl", "g-s1-r1", 1, 1))
    rows.extend(_seating_rows("ctrl", "g-s2-r0", 2, 0))

    artifacts, index_rows, warnings = generate_seating_files(pd.DataFrame(rows))

    assert "seating/ctrl.seating.json" in artifacts
    state = json.loads(artifacts["seating/ctrl.seating.json"])
    assert state["totalSeats"] == 2
    assert state["seedCount"] == 2
    assert state["configSlots"] == [0, 1]
    assert state["seatingSeed"] == 0
    assert state["basePerm"] == [0, 1]
    assert len(state["consumeOrder"]) == 4  # totalSeats * seedCount
    assert state["completedCycles"] == 0
    assert "_comment" in state
    # seed 1 -> seedIndex 0, seed 2 -> seedIndex 1.
    assert state["cells"]["0"]["0"] == {"status": "completed", "gameID": "g-s1-r0"}
    assert state["cells"]["1"]["0"]["gameID"] == "g-s1-r1"
    assert state["cells"]["0"]["1"]["gameID"] == "g-s2-r0"
    # Missing (rotation 1, seedIndex 1) is absent => open.
    assert "1" not in state["cells"].get("1", {})

    idx = index_rows[0]
    assert idx == {"experiment": "ctrl", "file": "seating/ctrl.seating.json",
                   "total_cells": 4, "completed_cells": 3, "open_cells": 1}
    assert warnings == []


def test_generate_seating_skips_complete_experiment():
    from bench.analyses.performance.seating import generate_seating_files

    rows = []
    for seed in (1, 2):
        for rot in (0, 1):
            rows.extend(_seating_rows("ctrl", f"g-s{seed}-r{rot}", seed, rot))

    artifacts, index_rows, _ = generate_seating_files(pd.DataFrame(rows))
    assert artifacts == {} and index_rows == []  # full grid -> nothing to open


def test_generated_state_passes_validator():
    from bench.analyses.performance.seating import generate_seating_files, validate_seating_state

    rows = []
    rows.extend(_seating_rows("ctrl", "g-s1-r0", 1, 0))
    rows.extend(_seating_rows("ctrl", "g-s2-r0", 2, 0))
    artifacts, _, _ = generate_seating_files(pd.DataFrame(rows))
    state = json.loads(artifacts["seating/ctrl.seating.json"])
    assert validate_seating_state(state) == []


def test_validator_flags_unconsumable_state():
    from bench.analyses.performance.seating import validate_seating_state

    good = {"totalSeats": 2, "seedCount": 2, "configSlots": [0, 1], "seatingSeed": 0,
            "basePerm": [0, 1],
            "consumeOrder": [{"rotation": r, "seedIndex": s} for r in range(2) for s in range(2)],
            "cells": {}, "completedCycles": 0}
    assert validate_seating_state(good) == []

    # A consumeOrder that drops a cell would strand that open game in the runner.
    bad = dict(good, consumeOrder=good["consumeOrder"][:-1])
    assert any("consumeOrder" in p for p in validate_seating_state(bad))
    # configSlots out of [0, totalSeats) drifts the header => runner rebuilds.
    assert any("configSlots" in p for p in
               validate_seating_state(dict(good, configSlots=[0, 5])))
    # A completed cell with no gameID is malformed.
    assert any("gameID" in p for p in validate_seating_state(
        dict(good, cells={"0": {"0": {"status": "completed"}}})))


def test_seating_open_cells_equal_completeness_missing():
    """The cells the seating file leaves open == the games the completeness report flags
    missing, so the runner (which plays open cells) plays exactly civ-bench's gaps."""
    from bench.analyses.performance.seating import generate_seating_files
    from bench.analyses.performance.strength_panel import build_experiment_completeness

    # configSlots {0,1} -> totalSeats 2 == #rotations; seeds {1,2}; (2,1) absent.
    rows = []
    for seed, rot in [(1, 0), (1, 1), (2, 0)]:
        rows.extend(_seating_rows("ctrl", f"g-s{seed}-r{rot}", seed, rot))
    panel = pd.DataFrame(rows)

    # civ-bench's missing set from the completeness report (seed, rotation).
    gaps = build_experiment_completeness(panel)["experiment_completeness_gaps"]
    missing = {
        (int(r.seed), int(rot))
        for r in gaps.itertuples(index=False)
        for rot in str(r.missing_rotations).split(",")
    }
    assert missing == {(2, 1)}

    # The seating file's open cells, mapped seedIndex -> seed via the sorted design seeds.
    design_seeds = sorted({int(s) for s in panel["seed"].unique()})
    state = json.loads(generate_seating_files(panel)[0]["seating/ctrl.seating.json"])
    completed = {(int(rk), int(sk)) for rk, inner in state["cells"].items() for sk in inner}
    open_games = {
        (design_seeds[s], r)
        for r in range(state["totalSeats"])
        for s in range(state["seedCount"])
        if (r, s) not in completed
    }
    assert open_games == missing


def test_experiment_completeness_emits_seating_artifact(env):
    r = env("performance.experiment_completeness", {"emit_seating": True}, {"tables": ["strength"]})
    seating = [rel for rel in r.artifact_paths if rel.startswith("seating/")]
    assert seating, "expected at least one generated seating.json for the incomplete grid"
    state = json.loads(Path(r.artifact_paths[seating[0]]).read_text(encoding="utf-8"))
    assert {"totalSeats", "seedCount", "configSlots", "consumeOrder", "cells"} <= set(state)
    assert state["seatingSeed"] == 0 and "_comment" in state
    assert len(state["consumeOrder"]) == state["totalSeats"] * state["seedCount"]


def test_experiment_completeness_emit_seating_off(env):
    r = env("performance.experiment_completeness", {"emit_seating": False}, {"tables": ["strength"]})
    assert not any(rel.startswith("seating/") for rel in r.artifact_paths)
    assert "seating_index" not in r.table_paths


def test_experiment_completeness_repeated_games_are_actionable():
    from bench.analyses.performance.strength_panel import build_experiment_completeness

    rows = []
    rows.extend(_strength_rows("ctrl", "newer", 1, 0))
    rows.extend(_strength_rows("ctrl", "older", 1, 0))
    games = pd.DataFrame([
        {"game_id": "newer", "timestamp": 200},
        {"game_id": "older", "timestamp": 100},
    ])

    out = build_experiment_completeness(pd.DataFrame(rows), games)
    comp = out["experiment_completeness"].iloc[0]
    assert comp["required_games"] == 1
    assert comp["present_games"] == 2
    assert comp["missing_games"] == 0
    assert comp["repeated_slots"] == 1
    assert "1 repeated slot(s)" in comp["warning"]

    repeated = out["repeated_games"].iloc[0]
    assert repeated["n_games"] == 2  # distinct game ids, not player rows
    assert repeated["game_ids"] == "older,newer"
    assert repeated["keep_candidate_game_id"] == "older"
    assert repeated["extra_game_ids"] == "newer"


def test_experiment_completeness_uncontrolled_only_is_empty():
    from bench.analyses.performance.strength_panel import build_experiment_completeness

    panel = pd.DataFrame(_strength_rows("free", "g", -1, -1, controlled=False))
    assert build_experiment_completeness(panel) == {}


def test_performance_turn_predicted(env):
    r = env("performance.turn_predicted", {"by": "player_type"})
    assert "by_identity" in r.table_paths and "over_progress" in r.table_paths
    tbl = pd.read_csv(r.table_paths["by_identity"])
    assert "model" in tbl.columns


# ── exploratory ──────────────────────────────────────────────────────────────────
def test_performance_usage_efficiency_outputs(env):
    _write_ratings_artifact(env, [
        {"player_type": "GPT-OSS-120B", "elo": 1600},
        {"player_type": "Kimi-K2.5", "elo": 1700},
    ])
    r = env("performance.usage_efficiency", {"currency": "usd"},
            {"tables": ["tokens"], "analyses": ["bt_main"]})
    by_player = pd.read_csv(r.table_paths["usage"])
    assert {
        "player_type", "total_cost", "games", "complete_games",
        "avg_cost_per_game",
    } <= set(by_player.columns)
    assert set(r.table_paths) == {"usage", "usage_vs_rating"}
    assert set(r.figure_paths) == {"cost", "input_tokens", "output_tokens", "usage_vs_rating"}
    assert all(Path(r.figure_paths[name]).suffix == ".png"
               for name in ("cost", "input_tokens", "output_tokens"))
    assert by_player["player_type"].is_unique
    assert r.metadata["cost_basis"] == "per player per complete game"
    assert r.metadata["cached_tokens_accounted_for"] is False


def test_token_costs_average_players_and_keep_complete_records(env):
    from bench.analyses.performance.usage import (
        compute_game_costs, summarize_game_costs,
    )

    tokens = pd.DataFrame([
        {"game_id": "G", "player_id": pid, "player_type": identity,
         "model_name": "gpt-oss-120b", "input_tokens": inputs,
         "output_tokens": outputs, "reasoning_tokens": reasoning}
        for pid, identity, inputs, outputs, reasoning in [
            (0, "GPT-OSS-120B", 1000, 100, 10),
            (0, "GPT-OSS-120B", 2000, 200, 20),
            (1, "GPT-OSS-120B", 6000, 600, 60),
            (2, "GPT-OSS-120B", np.nan, 100, 10),
            (3, "Other-strategy", 9000, 900, 90),
        ]
    ])
    records = compute_game_costs(tokens, env.catalog)
    assert len(records) == 4
    by_player = summarize_game_costs(records, ["player_type", "model"])
    row = by_player.set_index("player_type").loc["GPT-OSS-120B"]
    assert row["avg_input"] == 4500
    assert row["avg_output"] == 495
    assert row["player_games"] == 3
    assert row["complete_player_games"] == 2
    assert row["na_player_games"] == 1
    assert row["games"] == row["na_games"] == 1
    assert row["complete_games"] == 0
    assert row["avg_cost_per_player_game"] == pytest.approx(row["total_cost"] / 2)

    by_model = summarize_game_costs(records, ["model"]).iloc[0]
    assert by_model["avg_input"] == 6000
    assert by_model["avg_output"] == 660
    assert by_model["complete_player_games"] == 3
    assert by_model["avg_cost_per_player_game"] == pytest.approx(by_model["total_cost"] / 3)


def test_token_costs_combine_models_for_one_player(env):
    from bench.analyses.performance.usage import (
        compute_game_costs, summarize_game_costs,
    )

    tokens = pd.DataFrame([
        {"game_id": "G", "player_id": 0, "player_type": "Mixed",
         "model_name": model, "input_tokens": 1000,
         "output_tokens": 100, "reasoning_tokens": 20}
        for model in ["gpt-oss-120b", "kimi-k2.5"]
    ])
    records = compute_game_costs(tokens, env.catalog)
    row = summarize_game_costs(records, ["player_type"]).iloc[0]
    assert row["complete_player_games"] == 1
    assert row["avg_input"] == 2000
    assert row["avg_output"] == 240
    assert row["avg_cost_per_player_game"] == pytest.approx(records["total_cost"].sum())


def test_compute_game_costs_accepts_missing_model_name(env):
    from bench.analyses.performance.usage import compute_game_costs

    tokens = pd.DataFrame([{
        "game_id": "failed-game",
        "player_type": "TestLLM",
        "model_name": np.nan,
        "input_tokens": 0,
        "reasoning_tokens": 0,
        "output_tokens": 0,
    }])

    costs = compute_game_costs(tokens, env.catalog)

    assert len(costs) == 1
    assert costs.iloc[0]["model"] == "Unattributed"
    assert bool(costs.iloc[0]["available"]) is False


def _write_ratings_artifact(env, rows):
    from bench.analyses.base import analyses_out_dir

    out = analyses_out_dir(env.cfg, "bt_main")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "ratings.csv", index=False)


def test_paired_sort_order_uses_requested_condition_fallback_and_nan_last():
    from bench.plotting.pairing import PairingSpec, paired_sort_order

    spec = PairingSpec(("-Per-5",), "-Per-5", "Every-turn")
    frame = pd.DataFrame([
        {"base_identity": "A", "condition": "base", "value": 1.0},
        {"base_identity": "A", "condition": "-Per-5", "value": 3.0},
        {"base_identity": "B", "condition": "base", "value": 2.0},
        {"base_identity": "Vanilla", "condition": "base", "value": 2.5},
        {"base_identity": "C", "condition": "base", "value": np.nan},
    ])
    assert paired_sort_order(frame, spec, "value", ascending=False) == [
        "A", "Vanilla", "B", "C",
    ]


@pytest.mark.parametrize(
    "ascending, expected",
    [
        (False, ["B", "A", "Vanilla", "C"]),
        (True, ["A", "B", "Vanilla", "C"]),
    ],
)
def test_paired_sort_order_best_keys_off_best_condition_value(ascending, expected):
    from bench.plotting.pairing import PairingSpec, paired_sort_order

    spec = PairingSpec(("-Per-5",), "best", "Every-turn")
    frame = pd.DataFrame([
        {"base_identity": "A", "condition": "base", "value": 3.0},
        {"base_identity": "A", "condition": "-Per-5", "value": 1.0},
        {"base_identity": "B", "condition": "base", "value": 2.0},
        {"base_identity": "B", "condition": "-Per-5", "value": 5.0},
        {"base_identity": "Vanilla", "condition": "base", "value": 2.5},
        {"base_identity": "C", "condition": "base", "value": np.nan},
    ])
    assert paired_sort_order(frame, spec, "value", ascending) == expected


def test_paired_token_costs_follow_ratings_order(env, monkeypatch):
    env.cfg.presentation = {
        "condition_pairing": {"enabled": True, "sort_condition": "-Per-5"}
    }
    _write_ratings_artifact(env, [
        {"player_type": "GPT-OSS-120B", "elo": 1700},
        {"player_type": "Kimi-K2.5", "elo": 1600},
        {"player_type": "Vanilla", "elo": 1500},
    ])
    captured = {}

    def fake_plot(*args, **kwargs):
        import matplotlib.pyplot as plt

        captured["order"] = kwargs["row_order"]
        return plt.subplots()[0]

    monkeypatch.setattr("bench.plotting.pairing.plot_paired_rows", fake_plot)
    env(
        "performance.usage_efficiency",
        {"currency": "usd"},
        {"tables": ["tokens"], "analyses": ["bt_main"]},
    )
    assert captured["order"] == ["GPT-OSS-120B", "Kimi-K2.5"]


def test_paired_token_costs_declared_missing_rating_is_loud(env):
    env.cfg.presentation = {"condition_pairing": {"enabled": True}}
    with pytest.raises(AnalysisError, match="run stage 'bt_main' first"):
        env(
            "performance.usage_efficiency",
            {"currency": "usd"},
            {"tables": ["tokens"], "analyses": ["bt_main"]},
        )


def test_global_completeness_filter_drops_incomplete_conditions_from_tokens(env):
    from bench.analyses.base import AnalysisContext

    # Replace the controlled grid: llm-standard covers seeds {1, 2} × rotations
    # {0, 1} (complete); observe-vanilla-standard covers only rotation 0 (half the
    # grid, incomplete at a 1.0 threshold).
    games_path = env.cfg.data["tables"]["games"]
    pd.DataFrame([
        {"game_id": f"F{i}", "timestamp": "2026-01-01", "experiment": "llm-standard",
         "seed": seed, "seating_rotation": rot}
        for i, (seed, rot) in enumerate([(1, 0), (1, 1), (2, 0), (2, 1)])
    ] + [
        {"game_id": f"P{i}", "timestamp": "2026-01-01",
         "experiment": "observe-vanilla-standard",
         "seed": seed, "seating_rotation": rot}
        for i, (seed, rot) in enumerate([(1, 0), (2, 0)])
    ]).to_csv(games_path, index=False)
    env.cfg.data["filter"] = {"min_condition_completeness": 1.0}

    ctx = AnalysisContext(
        config=env.cfg, catalog=env.catalog, stage_id="t", stage_raw={},
        out_dir=Path(env.cfg.output.root),
    )
    filtered = ctx.apply_filter(ctx.load_table("tokens"))
    assert set(filtered["experiment"]) == {"llm-standard"}


def test_usage_vs_rating_outputs_and_excludes_baseline(env):
    env.cfg.presentation = {"condition_pairing": {"enabled": True}}
    _write_ratings_artifact(env, [
        {"player_type": "Vanilla", "elo": 1500, "se_elo": 20},
        {"player_type": "GPT-OSS-120B", "elo": 1600, "se_elo": 30},
        {"player_type": "Kimi-K2.5", "elo": 1700, "se_elo": 25},
    ])
    r = env(
        "performance.usage_efficiency",
        {"currency": "usd", "log_x": True, "annotate": True},
        {"tables": ["tokens"], "analyses": ["bt_main"]},
        sid="cost_rating",
    )
    assert "usage_vs_rating" in r.table_paths
    assert "usage_vs_rating" in r.figure_paths
    table = pd.read_csv(r.table_paths["usage_vs_rating"])
    assert "Vanilla" not in set(table["player_type"])
    assert {"base_identity", "condition", "avg_cost_per_game", "elo"} <= set(table.columns)
    assert {"avg_input", "avg_output", "avg_cost_per_player_game",
            "efficiency_elo_cost", "complete_player_games"} <= set(table.columns)
    assert table["efficiency_elo_cost"].isna().all()
    assert "at least three rated identities" in r.summary
    assert r.metadata["cached_tokens_accounted_for"] is False
    path = Path(r.figure_paths["usage_vs_rating"])
    assert path.suffix == ".html"
    html = path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "Input tokens" in html
    assert "Output tokens" in html
    assert "table_tooltip" in html


@pytest.mark.parametrize("log_x", [False, True])
def test_usage_vs_rating_zero_cost_has_no_efficiency_score(env, log_x):
    tokens_path = env.cfg.data["tables"]["tokens"]
    tokens = pd.read_csv(tokens_path)
    tokens.loc[tokens["player_type"] == "GPT-OSS-120B",
               ["input_tokens", "output_tokens", "reasoning_tokens"]] = 0
    tokens.to_csv(tokens_path, index=False)
    _write_ratings_artifact(env, [
        {"player_type": "GPT-OSS-120B", "elo": 1900},
        {"player_type": "Kimi-K2.5", "elo": 1500},
    ])
    result = env("performance.usage_efficiency", {"log_x": log_x},
                 {"tables": ["tokens"], "analyses": ["bt_main"]})
    table = pd.read_csv(result.table_paths["usage_vs_rating"]).set_index("player_type")
    assert pd.isna(table.loc["GPT-OSS-120B", "efficiency_elo_cost"])
    assert "at least three rated identities" in result.summary
    assert pd.isna(table.loc["GPT-OSS-120B", "expected_elo_cost"])


def test_usage_vs_rating_requires_analysis_dependency(env):
    with pytest.raises(AnalysisError, match="requires the ratings stage"):
        env("performance.usage_efficiency", {}, {"tables": ["tokens"]})


def test_usage_efficiency_summary_ranks_residuals_and_uses_actual_baseline(env):
    tokens_path = env.cfg.data["tables"]["tokens"]
    tokens = pd.read_csv(tokens_path)
    template = tokens[tokens["player_type"] == "GPT-OSS-120B"].iloc[0].to_dict()
    model = env.catalog.canonicalize_model_name(template["model_name"])
    price = env.catalog.pricing_per_million()[model]["input_per_million"]
    names = ["Cheap", "Good", "Mid", "High"]
    pd.DataFrame([
        {**template, "player_id": index, "player_type": name,
         "input_tokens": cost * 1e6 / price, "output_tokens": 0, "reasoning_tokens": 0}
        for index, (name, cost) in enumerate(zip(names, [0.01, 0.1, 1, 10]))
    ]).to_csv(tokens_path, index=False)
    _write_ratings_artifact(env, [
        {"player_type": name, "elo": elo}
        for name, elo in zip([*names, "Vanilla"], [1000, 1500, 1600, 1700, 1450])
    ])
    result = env("performance.usage_efficiency", {},
                 {"tables": ["tokens"], "analyses": ["bt_main"]})
    assert "Most cost-efficient: **Good** (**+160 Elo**" in result.summary
    assert "Least cost-efficient: **Cheap** (**-120 Elo**" in result.summary
    assert result.metadata["baseline_elo"] == 1450
    assert result.metadata["baseline_name"] == "Vanilla"
    assert result.metadata["usage_skill_fits"]["cost"]["n"] == 4
    table = pd.read_csv(result.table_paths["usage_vs_rating"]).set_index("player_type")
    assert table.loc["Good", "efficiency_elo_cost"] == pytest.approx(160)
    assert table["efficiency_elo_output"].isna().all()


@pytest.mark.parametrize("metric,column", [
    ("cost", "avg_cost_per_player_game"), ("input", "avg_input"), ("output", "avg_output"),
])
def test_usage_efficiency_measures_elo_above_fitted_log_usage(metric, column):
    from bench.analyses.performance.usage_efficiency import fit_usage_skill

    table = pd.DataFrame({column: [0.01, 0.1, 1, 10, 0, np.nan, np.inf, 20],
                          "elo": [1000, 1500, 1600, 1700, 2000, 1800, 1900, np.nan]})
    fit = fit_usage_skill(table, metric)
    assert fit == pytest.approx({"intercept": 1560, "slope": 220, "n": 4, "r_squared": 121 / 145})
    np.testing.assert_allclose(table.loc[:3, f"expected_elo_{metric}"], [1120, 1340, 1560, 1780])
    np.testing.assert_allclose(table.loc[:3, f"efficiency_elo_{metric}"], [-120, 160, 40, -80])
    assert table[f"efficiency_elo_{metric}"].idxmax() == 1
    assert table.loc[4:, f"efficiency_elo_{metric}"].isna().all()


@pytest.mark.parametrize("ratings,expected_r_squared", [
    ([1000, 1100, 1200], 1.0),
    ([1600, 1300, 1600], 0.0),
    ([1500, 1500, 1500], None),
])
def test_usage_fit_r_squared_handles_perfect_flat_and_constant_ratings(ratings, expected_r_squared):
    from bench.analyses.performance.usage_efficiency import fit_usage_skill

    table = pd.DataFrame({"avg_cost_per_player_game": [1, 10, 100], "elo": ratings})
    fit = fit_usage_skill(table, "cost")
    assert fit["r_squared"] == expected_r_squared


@pytest.mark.parametrize("costs", [[], [1], [1, 2], [2, 2, 2]])
def test_usage_efficiency_does_not_rank_underdetermined_fit(costs):
    from bench.analyses.performance.usage_efficiency import fit_usage_skill

    table = pd.DataFrame({"avg_cost_per_player_game": costs, "elo": [1500] * len(costs)})
    assert fit_usage_skill(table, "cost") is None
    assert table["efficiency_elo_cost"].isna().all()


@pytest.mark.parametrize("log_x", [False, True])
def test_usage_chart_switches_points_fits_and_tooltips(env, log_x):
    from bench.analyses.base import AnalysisContext
    from bench.analyses.performance.usage_efficiency import PerformanceUsageEfficiency, fit_usage_skill
    from bench.plotting.pairing import PairingSpec

    table = pd.DataFrame({
        "player_type": ["Cheap", "Good", "Mid", "High", "Free"],
        "base_identity": ["Cheap", "Good", "Mid", "High", "Free"],
        "condition": ["base"] * 5,
        "elo": [1000, 1500, 1600, 1700, 1400],
        "avg_cost_per_player_game": [0.01, 0.1, 1, 10, 0],
        "avg_input": [10, 100, 1000, 10000, 100000],
        "avg_output": [1000, 100, 10, 1, 0],
        "complete_player_games": [4] * 5, "na_player_games": [0] * 5,
        "token_complete_player_games": [4] * 5, "token_na_player_games": [0] * 5,
    })
    fits = {metric: fit_usage_skill(table, metric) for metric in ("cost", "input", "output")}
    ctx = AnalysisContext(config=env.cfg, catalog=env.catalog, stage_id="usage", stage_raw={},
                          out_dir=Path(env.cfg.output.root))
    figure = PerformanceUsageEfficiency._plot(
        table, ctx, PairingSpec((), "base", "Base"), "usd", log_x, True, fits, 1450, "Vanilla",
    )
    assert figure.layout.shapes[0].y0 == 1450
    assert figure.layout.xaxis.type == ("log" if log_x else "linear")
    buttons = figure.layout.updatemenus[0].buttons
    assert [button.label for button in buttons] == ["Cost", "Input tokens", "Output tokens"]
    for metric, button in zip(("cost", "input", "output"), buttons):
        visible = [trace for trace, shown in zip(figure.data, button.args[0]["visible"]) if shown]
        curve = next(trace for trace in visible if trace.mode == "lines")
        assert curve.line.dash == "dot"
        assert any(
            f"R² = {fits[metric]['r_squared']:.3f}" in annotation["text"]
            for annotation in button.args[1]["annotations"]
        )
        np.testing.assert_allclose(curve.y, fits[metric]["intercept"] + fits[metric]["slope"] * np.log10(curve.x))
        markers = [trace for trace in visible if trace.mode == "markers"]
        good = next(trace for trace in markers if trace.name == "Good")
        expected = table.loc[1, f"efficiency_elo_{metric}"]
        assert good.customdata[0][4] == f"{expected:+,.0f}"
        assert good.customdata[0][3] == "1,450 (VPAI)"
        assert all(trace.hoverinfo == "none" and trace.error_y.array is None for trace in markers)
        assert any("VPAI baseline: 1,450 Elo" in annotation["text"] for annotation in button.args[1]["annotations"])
        assert any(trace.name == "Free" for trace in markers) == (not log_x or metric == "input")
    assert fits["cost"]["slope"] != fits["input"]["slope"]


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


def test_paired_bt_forest_keeps_ratings_table_shape(env, monkeypatch):
    monkeypatch.setattr("bench.analyses.ratings.bradley_terry.calculate_ratings_bt", _fake_bt)
    env.cfg.presentation = {"condition_pairing": {"enabled": True}}
    r = env(
        "ratings.bradley_terry",
        {"group_by": ["player_type"], "ref": "Vanilla"},
        {"tables": ["strength"]},
    )
    table = pd.read_csv(r.table_paths["ratings"])
    assert "base_identity" not in table.columns
    assert "ratings" in r.figure_paths


def test_ratings_strategy_group_by(env, monkeypatch):
    monkeypatch.setattr("bench.analyses.ratings.bradley_terry.calculate_ratings_bt", _fake_bt)
    r = env("ratings.bradley_terry",
            {"group_by": ["player_type", "strategy"], "ref": "Vanilla", "min_games": 1},
            {"tables": ["strength"]})
    tbl = pd.read_csv(r.table_paths["ratings"])
    assert {"player_type", "strategy", "composite_type", "elo"} <= set(tbl.columns)
    # composite identity = player_type-strategy (the strategy grouping label)
    assert tbl["composite_type"].str.contains("-").all()
    assert "ratings" in r.figure_paths

    manifest = json.loads(Path(r.manifest_path).read_text(encoding="utf-8"))
    assert isinstance(manifest["module_name"], str) and manifest["module_name"]
    assert isinstance(manifest["module_description"], str) and manifest["module_description"]


@pytest.mark.parametrize(
    ("module", "params"),
    [
        (
            "ratings.bradley_terry",
            {"group_by": ["player_type"]},
        ),
        (
            "ratings.bradley_terry",
            {"group_by": ["player_type", "strategy"]},
        ),
        (
            "ratings.plackett_luce",
            {"group_by": ["player_type"]},
        ),
        (
            "ratings.plackett_luce",
            {"group_by": ["player_type", "strategy"]},
        ),
    ],
)
def test_rating_identity_is_resolved_from_grouping(module, params):
    analysis = get_analysis(module)("ratings", params)
    name, description = analysis.report_identity()
    assert isinstance(name, str) and name
    assert isinstance(description, str) and description


def test_ratings_bootstrap_blocked_with_strategy(env, monkeypatch):
    monkeypatch.setattr("bench.analyses.ratings.bradley_terry.calculate_ratings_bt", _fake_bt)
    with pytest.raises(AnalysisError, match="bootstrap with a multi-dimension"):
        env("ratings.bradley_terry",
            {"group_by": ["player_type", "strategy"], "bootstrap": {"n": 5}},
            {"tables": ["strength"]})


def test_ratings_matchups_no_r(env):
    r = env("ratings.matchups", {"mode": "both", "validate_ols": True}, {"tables": ["strength"]})
    assert {"strength_mean", "strength_winrate", "counts", "ols_validation"} <= set(r.table_paths)
    assert r.metadata["strength_estimator"] == "est"


def test_ratings_matchups_vs_reference_matches_vanilla_matrix(env):
    env.cfg.presentation = {"matchup_display": "vs_reference"}
    r = env("ratings.matchups", {"mode": "both"}, {"tables": ["strength"]})
    matrix = pd.read_csv(r.table_paths["strength_mean"]).set_index("player_type")
    vs = pd.read_csv(r.table_paths["vs_reference"]).set_index("player_type")
    for identity in vs.index:
        assert vs.loc[identity, "mean_diff_vs_ref"] == pytest.approx(
            matrix.loc[identity, "Vanilla"]
        )
    assert {"strength_mean", "strength_winrate"} <= set(r.figure_paths)


def test_matchups_vs_reference_falls_back_when_vanilla_absent(env):
    path = Path(env.cfg.adjust[0].raw["save"])
    panel = pd.read_csv(path)
    panel[panel["player_type"] != "Vanilla"].to_csv(path, index=False)
    env.cfg.presentation = {"matchup_display": "vs_reference"}
    r = env("ratings.matchups", {"mode": "both"}, {"tables": ["strength"]})
    assert "vs_reference" not in r.table_paths
    assert r.metadata["display"] == "matrix"
    assert r.metadata["warning"] == "reference 'Vanilla' is absent"
    assert "reference 'Vanilla' is absent" not in r.summary


def test_outcome_matchups_outputs(env):
    r = env("ratings.outcome_matchups", {"include_score_ratio": True}, {"tables": ["panel"]})
    assert {"win_rate", "score_ratio_margin", "counts"} <= set(r.table_paths)
    win = pd.read_csv(r.table_paths["win_rate"])
    margin = pd.read_csv(r.table_paths["score_ratio_margin"])
    assert set(PLAYER_TYPES) <= set(win["player_type"])
    assert set(PLAYER_TYPES) <= set(margin["player_type"])


def test_outcome_matchups_vs_reference_matches_vanilla_matrix(env):
    env.cfg.presentation = {"matchup_display": "vs_reference"}
    r = env("ratings.outcome_matchups", {"include_score_ratio": True}, {"tables": ["panel"]})
    matrix = pd.read_csv(r.table_paths["win_rate"]).set_index("player_type")
    vs = pd.read_csv(r.table_paths["vs_reference"]).set_index("player_type")
    for identity in vs.index:
        assert vs.loc[identity, "win_rate_vs_ref"] == pytest.approx(
            matrix.loc[identity, "Vanilla"]
        )


def test_outcome_matchups_dedupes_repeated_opponent_type():
    from bench.analyses.ratings.outcome_matchups import create_outcome_matchup_matrices

    panel = pd.DataFrame([
        {"game_id": "g1", "player_id": 0, "player_type": "A", "is_winner": 1, "score_ratio": 1.0},
        {"game_id": "g1", "player_id": 1, "player_type": "B", "is_winner": 0, "score_ratio": 0.4},
        {"game_id": "g1", "player_id": 2, "player_type": "B", "is_winner": 0, "score_ratio": 0.6},
    ])
    win, margin, counts = create_outcome_matchup_matrices(panel)

    assert counts.loc["A", "B"] == 1
    assert win.loc["A", "B"] == pytest.approx(1.0)
    assert margin.loc["A", "B"] == pytest.approx(0.5)
    assert counts.loc["B", "A"] == 2


def _outcome_panel(game_sizes, identity_seats=None, winners=None):
    """Build a small completed-game panel for observed-outcome tests."""
    identity_seats = identity_seats or {}
    winners = set(winners or ())
    rows = []
    for game_no, game_size in enumerate(game_sizes):
        game_id = f"outcome-{game_no}"
        seats = ["Vanilla"] + ["A"] * max(1, identity_seats.get(game_no, 0))
        seats += [f"P{n}" for n in range(len(seats), game_size)]
        for player_id, player_type in enumerate(seats[:game_size]):
            rows.append({
                "game_id": game_id,
                "player_id": player_id,
                "player_type": player_type,
                "is_winner": int((game_no, player_id) in winners),
                "score_ratio": 1.0,
            })
    return pd.DataFrame(rows)


class _OutcomeContext:
    def __init__(self, panel, catalog, display="matrix", included_types=None):
        self.catalog = catalog
        self._panel = panel
        self._display = display
        self._included_types = included_types

    def uses_tables(self):
        return ["panel"]

    def load_table(self, _table_id):
        return self._panel.copy()

    def apply_filter(self, panel):
        if self._included_types is None:
            return panel
        return panel[panel["player_type"].isin(self._included_types)].copy()

    def matchup_display(self):
        return self._display

    def condition_pairing(self):
        return None


def _figure_text(figure):
    return " ".join(text.get_text() for axis in figure.axes for text in axis.texts)


def test_outcome_matchups_reports_per_appearance_expected_rate_for_eight_players(env):
    from bench.analyses.ratings.outcome_matchups import RatingsOutcomeMatchups

    panel = _outcome_panel([8], identity_seats={0: 2}, winners={(0, 1)})
    result = RatingsOutcomeMatchups("outcomes", {"include_score_ratio": True}).run(
        _OutcomeContext(panel, env.catalog)
    )
    expected = result.tables["expected_win_rate"].set_index("player_type")
    win = result.tables["win_rate"].set_index("player_type")
    counts = result.tables["counts"].set_index("player_type")
    assert win.loc["A", "Vanilla"] == pytest.approx(0.5)
    assert win.loc["Vanilla", "A"] == pytest.approx(0.0)
    assert expected.loc["A", "Vanilla"] == pytest.approx(0.125)
    assert counts.loc["A", "Vanilla"] == 2
    assert counts.loc["Vanilla", "A"] == 1
    assert "per-player" in result.summary
    assert "**50.0%** (**1/2** wins per player appearance)" in result.summary
    assert "12.5%" in result.summary
    assert "8-player games" in result.summary
    assert "featuring" not in result.summary
    figure = result.figures["win_rate"]
    assert "expected 12.5%" in _figure_text(figure)
    assert "1/2" in _figure_text(figure)
    assert "0/1" in _figure_text(figure)
    assert "Counts = wins / player appearances" in figure.axes[0].get_xlabel()


def test_outcome_matchups_expected_rate_uses_full_game_size_after_filter(env):
    from bench.analyses.ratings.outcome_matchups import RatingsOutcomeMatchups

    panel = _outcome_panel([8])
    result = RatingsOutcomeMatchups("outcomes", {"include_score_ratio": False}).run(
        _OutcomeContext(panel, env.catalog, included_types={"A", "Vanilla"})
    )
    expected = result.tables["expected_win_rate"].set_index("player_type")
    assert expected.loc["A", "Vanilla"] == pytest.approx(0.125)


def test_outcome_matchups_expected_rate_weights_mixed_game_sizes(env):
    from bench.analyses.ratings.outcome_matchups import RatingsOutcomeMatchups

    panel = _outcome_panel([8, 4], identity_seats={0: 2})
    result = RatingsOutcomeMatchups("outcomes", {"include_score_ratio": False}).run(
        _OutcomeContext(panel, env.catalog, display="vs_reference")
    )
    expected = result.tables["expected_win_rate"].set_index("player_type")
    assert expected.loc["A", "Vanilla"] == pytest.approx((2 / 8 + 1 / 4) / 3)
    assert "averaged over these player appearances" in result.summary


def test_outcome_matchups_vs_reference_uses_expected_rate_and_binomial_validity(env):
    from scipy.stats import binomtest
    from bench.analyses.ratings.outcome_matchups import RatingsOutcomeMatchups

    fixed = _outcome_panel([8, 8], winners={(0, 1)})
    result = RatingsOutcomeMatchups("outcomes", {"include_score_ratio": False}).run(
        _OutcomeContext(fixed, env.catalog, display="vs_reference")
    )
    vs = result.tables["vs_reference"].set_index("player_type")
    assert "wins" in vs.columns
    assert vs.loc["A", "wins"] == 1
    assert vs.loc["A", "n"] == 2
    assert vs.loc["P2", "wins"] == 0
    assert vs.loc["P2", "win_rate_vs_ref"] == pytest.approx(0.0)
    assert vs.loc["A", "expected_win_rate"] == pytest.approx(0.125)
    assert vs.loc["A", "p_value_win_rate"] == pytest.approx(
        binomtest(1, 2, p=0.125).pvalue
    )
    axis = result.figures["win_rate"].axes[0]
    assert "Expected (12.5%)" in [text.get_text() for text in axis.get_legend().get_texts()]
    reference = next(line for line in axis.lines if line.get_linestyle() == "--")
    assert list(reference.get_xdata()) == [0.125, 0.125]
    assert "per player appearance" in axis.get_xlabel()
    assert "1/2" in _figure_text(result.figures["win_rate"])
    assert "0/2" in _figure_text(result.figures["win_rate"])
    assert "Counts show wins / player appearances" in " ".join(
        text.get_text() for text in result.figures["win_rate"].texts
    )
    assert "featuring" not in result.summary

    mixed = _outcome_panel([8, 4], winners={(0, 1)})
    mixed_result = RatingsOutcomeMatchups("outcomes", {"include_score_ratio": False}).run(
        _OutcomeContext(mixed, env.catalog, display="vs_reference")
    )
    assert pd.isna(mixed_result.tables["vs_reference"].set_index("player_type").loc["A", "p_value_win_rate"])
    assert "expected 18.8%" in _figure_text(mixed_result.figures["win_rate"])

    repeated = _outcome_panel([8], identity_seats={0: 2}, winners={(0, 1)})
    repeated_result = RatingsOutcomeMatchups("outcomes", {"include_score_ratio": False}).run(
        _OutcomeContext(repeated, env.catalog, display="vs_reference")
    )
    assert pd.isna(repeated_result.tables["vs_reference"].set_index("player_type").loc["A", "p_value_win_rate"])


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


def test_bootstrap_explicit_baseline_uses_fixed_reference(env):
    from bench.analyses.ratings import bootstrap as boot
    from bench.stats.transforms import inv_logit

    panel = pd.DataFrame([{
        "experiment": "ctrl",
        "game_id": "g1",
        "player_id": 0,
        "player_type": "TestLLM-Simple",
        "model": "TestLLM",
        "civilization": "Rome",
        "seed": 1,
        "controlled": True,
        "logit_strength": 2.0,
        "relative_strength": 0.25,
        "adjusted_strength": 0.25,
        "adjust_method": "cell",
    }])

    out = boot.readjust(
        panel,
        {"civ_adjust": "none", "baseline_experiment": "vp-self"},
        env.catalog,
        fixed_cell_baseline={(1, 0): 1.5},
    )

    assert out.loc[0, "adjusted_strength"] == pytest.approx(inv_logit(0.5))


def test_bootstrap_implicit_baseline_is_fixed_not_recomputed(env):
    # Option C: implicit cell baselines are held fixed (keyed experiment, seed,
    # player_id) from the persisted trail, not recomputed from the resample.
    from bench.analyses.ratings import bootstrap as boot
    from bench.stats.transforms import inv_logit

    panel = pd.DataFrame([{
        "experiment": "ctrl",
        "game_id": "g1",
        "player_id": 0,
        "player_type": "TestLLM-Simple",
        "model": "TestLLM",
        "civilization": "Rome",
        "seed": 1,
        "controlled": True,
        "logit_strength": 2.0,
        "relative_strength": 0.25,
        "adjusted_strength": 0.25,
        "adjust_method": "cell",
    }])

    out = boot.readjust(
        panel,
        {"civ_adjust": "none", "baseline_experiment": None},
        env.catalog,
        fixed_cell_baseline={("ctrl", 1, 0): 1.5},
    )
    assert out.loc[0, "adjusted_strength"] == pytest.approx(inv_logit(0.5))

    # A cell with no fixed baseline falls back (civ:none ⇒ relative), never crashes.
    out2 = boot.readjust(
        panel,
        {"civ_adjust": "none", "baseline_experiment": None},
        env.catalog,
        fixed_cell_baseline={},
    )
    assert out2.loc[0, "adjusted_strength"] == pytest.approx(0.25)


def test_ratings_with_bootstrap_player_type(env, monkeypatch):
    monkeypatch.setattr("bench.analyses.ratings.bradley_terry.calculate_ratings_bt", _fake_bt)
    r = env("ratings.bradley_terry",
            {"group_by": ["player_type"], "ref": "Vanilla", "weighted": True,
             "bootstrap": {"n": 8, "ci_level": 0.9}},
            {"tables": ["strength"]})
    tbl = pd.read_csv(r.table_paths["ratings"])
    assert "ci_lower" in tbl.columns and "ci_upper" in tbl.columns


# ── WS4: statistical fixes ────────────────────────────────────────────────────
def test_matchup_matrix_ties_count_half_and_are_symmetric():
    from bench.analyses.ratings.matchups import create_matchup_matrix

    df = pd.DataFrame([
        {"game_id": "g1", "player_type": "A", "adjusted_strength": 0.8},
        {"game_id": "g1", "player_type": "B", "adjusted_strength": 0.2},
        {"game_id": "g2", "player_type": "A", "adjusted_strength": 0.5},
        {"game_id": "g2", "player_type": "B", "adjusted_strength": 0.5},  # exact tie
    ])
    prob, count, pval = create_matchup_matrix(df)
    # A wins g1 (1.0) + ties g2 (0.5) over 2 games = 0.75; ties split ⇒ P+P=1
    assert prob.loc["A", "B"] == pytest.approx(0.75)
    assert prob.loc["B", "A"] == pytest.approx(0.25)
    assert prob.loc["A", "B"] + prob.loc["B", "A"] == pytest.approx(1.0)


def test_matchup_pvalue_is_paired_ttest():
    from scipy.stats import ttest_1samp

    from bench.analyses.ratings.matchups import create_matchup_matrix

    df = pd.DataFrame([
        {"game_id": f"g{k}", "player_type": pt, "adjusted_strength": v}
        for k in range(4)
        for pt, v in (("A", 0.6 + 0.02 * k), ("B", 0.3 + 0.01 * k))
    ])
    _, _, pval = create_matchup_matrix(df)
    # paired t-test on aligned (A, B) pairs == one-sample t-test on the diffs
    diffs = [0.6 + 0.02 * k - (0.3 + 0.01 * k) for k in range(4)]
    assert pval.loc["A", "B"] == pytest.approx(ttest_1samp(diffs, 0).pvalue)


def _boot_panel() -> pd.DataFrame:
    return pd.DataFrame([
        {"game_id": f"g{g}", "experiment": "e", "player_type": pt, "elo": 0.0}
        for g in range(8) for pt in ("A", "B")
    ])


def _boot_point() -> pd.DataFrame:
    return pd.DataFrame({"player_type": ["A", "B"], "elo": [1500.0, 1400.0]})


def test_run_bootstrap_all_failures_raise_not_silent_nan():
    from bench.analyses.ratings import bootstrap as boot

    def always_fail(df):
        raise ValueError("singular fit")

    with pytest.raises(AnalysisError, match="replicates failed"):
        boot.run_bootstrap(
            _boot_panel(), _boot_point(), always_fail,
            group_col="player_type", n=8, seed=1, refit_strength=False,
        )


def test_run_bootstrap_tolerates_minority_failures(capsys):
    from bench.analyses.ratings import bootstrap as boot

    calls = {"n": 0}

    def sometimes_fail(df):
        calls["n"] += 1
        if calls["n"] % 5 == 0:  # ~20% of replicates
            raise ValueError("occasional singular")
        return pd.DataFrame({"player_type": ["A", "B"], "elo": [1500.0, 1400.0]})

    out = boot.run_bootstrap(
        _boot_panel(), _boot_point(), sometimes_fail,
        group_col="player_type", n=10, seed=1, refit_strength=False,
        max_failure_rate=0.5,
    )
    assert "ci_lower" in out.columns and out["ci_lower"].notna().any()  # real CIs, not all-NaN
    assert "replicate(s) failed" in capsys.readouterr().err  # counted + warned


def test_run_bootstrap_narrow_dropping_reference_still_succeeds():
    # Regression for the vanished-Vanilla bug: narrowing (which can drop the
    # reference rows) happens per replicate and must not abort the whole bootstrap.
    from bench.analyses.ratings import bootstrap as boot

    def calc(df):
        pts = sorted(df["player_type"].unique())
        return pd.DataFrame({"player_type": pts, "elo": [1500.0] * len(pts)})

    def narrow(df):
        return df[df["player_type"] != "B"]  # drop the "reference" every replicate

    out = boot.run_bootstrap(
        _boot_panel(), _boot_point(), calc,
        group_col="player_type", n=6, seed=1, refit_strength=False, narrow=narrow,
    )
    assert out.loc[out["player_type"] == "A", "ci_lower"].notna().any()


def test_fit_civ_effects_is_shared_and_zero_sum(env):
    from bench.adjust import strength as strength_mod
    from bench.analyses.ratings import bootstrap as boot

    # The bootstrap refits civ effects with the SAME implementation as the adjust
    # stage; no diverged private copy (whose ≥2-omitted-civ handling was wrong).
    assert boot.fit_civ_effects is strength_mod.fit_civ_effects

    df = pd.read_csv(env.cfg.adjust[0].raw["save"])
    effects, eff_df = strength_mod.fit_civ_effects(df, env.catalog)
    # every civilization gets an effect; Sum-coding ⇒ the effects (incl. the omitted
    # level, computed once as −sum(rest)) sum to zero.
    assert set(effects) == set(df["civilization"].dropna().unique())
    assert sum(effects.values()) == pytest.approx(0.0, abs=1e-9)
