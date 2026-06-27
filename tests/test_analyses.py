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
    r = env("performance.experiment_completeness", {}, {"tables": ["strength"]})
    assert "experiment_completeness" in r.table_paths
    assert "repeated_games" in r.table_paths
    completeness = pd.read_csv(r.table_paths["experiment_completeness"])
    assert {
        "required_games", "present_games", "missing_games", "repeated_slots", "warning",
    } <= set(completeness.columns)
    assert "repeat_warning" not in completeness.columns


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
    missing — so the runner (which plays open cells) plays exactly civ-bench's gaps."""
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
def test_exploratory_model_token_costs(env):
    r = env("exploratory.model_token_costs", {"currency": "usd"}, {"tables": ["tokens"]})
    assert "token_costs_by_player_type" in r.table_paths
    by_player = pd.read_csv(r.table_paths["token_costs_by_player_type"])
    assert {"player_type", "model", "total_cost", "games"} <= set(by_player.columns)

    tbl = pd.read_csv(r.table_paths["token_costs"])
    assert "player_type" not in tbl.columns
    assert "total_cost" in tbl.columns and "games" in tbl.columns
    assert r.metadata["by_player_type"] is True


def test_exploratory_model_token_costs_can_use_model_only_view(env):
    r = env(
        "exploratory.model_token_costs",
        {"currency": "usd", "by_player_type": False},
        {"tables": ["tokens"]},
    )

    assert set(r.table_paths) == {"token_costs"}
    tbl = pd.read_csv(r.table_paths["token_costs"])
    assert "player_type" not in tbl.columns
    assert r.metadata["by_player_type"] is False


def test_exploratory_model_token_costs_accepts_by_strategist_alias(env):
    r = env(
        "exploratory.model_token_costs",
        {"currency": "usd", "by_strategist": True},
        {"tables": ["tokens"]},
    )

    assert "token_costs_by_player_type" in r.table_paths
    tbl = pd.read_csv(r.table_paths["token_costs_by_player_type"])
    assert {"player_type", "model"} <= set(tbl.columns)


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
    assert "ratings" in r.figure_paths


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


def test_outcome_matchups_outputs(env):
    r = env("ratings.outcome_matchups", {"include_score_ratio": True}, {"tables": ["panel"]})
    assert {"win_rate", "score_ratio_margin", "counts"} <= set(r.table_paths)
    win = pd.read_csv(r.table_paths["win_rate"])
    margin = pd.read_csv(r.table_paths["score_ratio_margin"])
    assert set(PLAYER_TYPES) <= set(win["player_type"])
    assert set(PLAYER_TYPES) <= set(margin["player_type"])


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
