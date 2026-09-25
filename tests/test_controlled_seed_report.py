"""Controlled-seed report tests (analysis module + report annex/renderer).

Everything runs on small synthetic controlled-design fixtures (no machine data
roots, per AGENTS.md): two seeds, two final player positions, two seating
rotations, a dedicated Vanilla baseline experiment, and treatment experiments
carrying the catalog's ``-Per-5`` condition suffix. The report tests fabricate
real artifacts by running the analysis once, then render through
``run_report``; the controlled-seed heatmap pages ride along automatically.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from bench.analyses import run_analysis
from bench.analyses.errors import AnalysisError
from bench.analyses.performance.controlled_seed_report import CURVE_KEYS
from bench.analyses.performance.curves import GRID_POINTS, interpolate_curves
from bench.catalog import Catalog
from bench.config import ConfigError, load_config
from bench.reports import run_report
from bench.reports.controlled_seed import render_controlled_seed_site
from bench.reports.model import ControlledSeedDocument, GameLogDocument, MatchedMapTab
from bench.reports.runner import report_dir

BASELINE_EXP = "vanilla-standard-fixed"
TREAT_EXP = "treat-2026"
KIMI_EXP = "treat-kimi"
UNCONTROLLED_EXP = "uncontrolled-exp"
SEEDS = [1, 2]
PLAYERS = [0, 1]
ROTATIONS = [0, 1]
CIVS = {0: "Rome", 1: "Greece"}
TIE_RATIOS = {
    "domination_ratio": 0.25, "culture_ratio": 0.25,
    "diplomatic_ratio": 0.25, "science_ratio": 0.25,
}
IDENTITY_VALUES = {
    "Vanilla": {"adjusted": 0.40, "weighted": 0.38, "ratios": TIE_RATIOS},
    "GPT-OSS-120B-Simple": {
        "adjusted": 0.60, "weighted": 0.55,
        "ratios": {"domination_ratio": 0.10, "culture_ratio": 0.10,
                   "diplomatic_ratio": 0.10, "science_ratio": 0.50},
    },
    "GPT-OSS-120B-Simple-Per-5": {
        "adjusted": 0.50, "weighted": 0.45,
        "ratios": {"domination_ratio": 0.60, "culture_ratio": 0.10,
                   "diplomatic_ratio": 0.10, "science_ratio": 0.10},
    },
    "Kimi-K2.5": {
        "adjusted": 0.70, "weighted": 0.65,
        "ratios": {"domination_ratio": 0.10, "culture_ratio": 0.70,
                   "diplomatic_ratio": 0.10, "science_ratio": 0.10},
    },
    "Kimi-K2.5-Per-5": {
        "adjusted": 0.45, "weighted": 0.42,
        "ratios": {"domination_ratio": 0.15, "culture_ratio": 0.15,
                   "diplomatic_ratio": 0.55, "science_ratio": 0.15},
    },
}


def _seats(experiment: str) -> dict[int, str]:
    if experiment == BASELINE_EXP:
        return {0: "Vanilla", 1: "Vanilla"}
    if experiment == TREAT_EXP:
        return {0: "GPT-OSS-120B-Simple", 1: "GPT-OSS-120B-Simple-Per-5"}
    if experiment == KIMI_EXP:
        return {0: "Kimi-K2.5", 1: "Kimi-K2.5-Per-5"}
    return {0: "Vanilla", 1: "Vanilla"}


def _game_rows(uncontrolled_only: bool = False) -> list[dict]:
    games: list[dict] = []
    if uncontrolled_only:
        games.append({"game_id": "free-1", "timestamp": "2026-01-05",
                      "experiment": TREAT_EXP, "seed": -1, "seating_rotation": -1})
        return games
    for seed in SEEDS:
        for rot in ROTATIONS:
            if seed == 1:
                games.append({"game_id": f"base-s{seed}-r{rot}", "timestamp": "2026-01-01",
                              "experiment": BASELINE_EXP, "seed": seed,
                              "seating_rotation": rot})
            games.append({"game_id": f"tr-s{seed}-r{rot}", "timestamp": "2026-01-02",
                          "experiment": TREAT_EXP, "seed": seed, "seating_rotation": rot})
    games.append({"game_id": "tr-s1-r0b", "timestamp": "2026-01-03",
                  "experiment": TREAT_EXP, "seed": 1, "seating_rotation": 0})
    games.append({"game_id": "ki-s1-r1", "timestamp": "2026-01-04",
                  "experiment": KIMI_EXP, "seed": 1, "seating_rotation": 1})
    games.append({"game_id": "free-1", "timestamp": "2026-01-05",
                  "experiment": UNCONTROLLED_EXP, "seed": -1, "seating_rotation": -1})
    return games


def _build_csvs(tmp_path, uncontrolled_only: bool = False) -> dict[str, str]:
    """Write synthetic games / panel / strength / turns / tokens / predictions CSVs."""
    games_rows = _game_rows(uncontrolled_only)
    panel_rows, strength_rows, pred_rows = [], [], []
    for game in games_rows:
        gid, experiment = game["game_id"], game["experiment"]
        seed, rot = game["seed"], game["seating_rotation"]
        for pid, player_type in _seats(experiment).items():
            values = IDENTITY_VALUES[player_type]
            # The kimi game's seat 0 uses a second civilization so one seed-player
            # pair carries a comparability warning.
            civ = "Egypt" if (gid == "ki-s1-r1" and pid == 0) else CIVS[pid]
            row = {"experiment": experiment, "game_id": gid, "player_id": pid,
                   "player_type": player_type, "civilization": civ,
                   "is_winner": int(pid == 1), "config_slot": pid}
            panel_rows.append({**row, **values["ratios"]})
            strength_rows.append({**row, "seed": seed, "seating_rotation": rot,
                                  "controlled": seed != -1,
                                  "weighted_strength": values["weighted"],
                                  "relative_strength": values["weighted"],
                                  "logit_strength": 0.0,
                                  "adjusted_strength": values["adjusted"],
                                  "adjust_method": "cell", "cell_logit_advantage": np.nan})
            for turn in (2, 4, 6, 8):
                pred_rows.append({"experiment": experiment, "game_id": gid,
                                  "player_id": pid, "civilization": civ, "turn": turn,
                                  "max_turn": 10, "is_winner": int(pid == 1),
                                  # y = x: the mean curve equals the grid itself.
                                  "predicted_win_probability": turn / 10,
                                  "turn_progress": turn / 10})
    paths: dict[str, str] = {}
    for name, rows in (("games", games_rows), ("panel", panel_rows),
                       ("tokens", [])):
        p = tmp_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths[name] = str(p)
    p = tmp_path / "strength_panel.csv"
    pd.DataFrame(strength_rows).to_csv(p, index=False)
    paths["strength"] = str(p)
    p = tmp_path / "predictions.csv"
    pd.DataFrame(pred_rows).to_csv(p, index=False)
    paths["predictions"] = str(p)
    paths["turns"] = paths["predictions"]
    return paths


def _make_spec(dev_spec, paths, tmp_path) -> dict:
    spec = dev_spec
    spec["output"] = {"root": str(tmp_path / "out"), "suffix": ""}
    spec["data"]["extract"]["enabled"] = False
    spec["data"]["tables"] = {
        "turns": paths["turns"], "panel": paths["panel"],
        "games": paths["games"], "tokens": paths["tokens"],
    }
    spec["data"]["filter"] = None
    spec["presentation"] = {
        "condition_pairing": {"enabled": True, "base_label": "Every-turn"},
    }
    spec["estimators"] = [{
        "id": "est", "model": "attention_mlp", "fit": "pretrained",
        "predict": "in_sample", "enabled": True, "predict_subset": "all",
        "save_predictions": paths["predictions"],
        "pretrained": {"model_dir": "pretrained/attention_mlp/"},
    }]
    spec["adjust"] = [{
        "id": "strength", "module": "strength", "enabled": True,
        "uses": {"estimators": ["est"]}, "save": paths["strength"],
        "params": {"block": "auto", "civ_adjust": "ols_logit",
                   "baseline_experiment": BASELINE_EXP},
    }]
    spec["analyses"] = [{
        "id": "controlled_seed", "module": "performance.controlled_seed_report",
        "enabled": True,
        "uses": {"estimators": ["est"], "tables": ["strength"]},
        "params": {},
    }]
    spec["report"] = {
        "out_dir": str(tmp_path / "out") + "/",
        "formats": ["md", "html"], "sections": None, "overview_sections": None,
        "section_overrides": {}, "title": None, "include_disabled": False,
    }
    return spec


def _stage_raw() -> dict:
    return {
        "id": "controlled_seed", "module": "performance.controlled_seed_report",
        "enabled": True,
        "uses": {"estimators": ["est"], "tables": ["strength"]},
        "params": {},
    }


@pytest.fixture
def env(tmp_path, write_spec, dev_spec):
    """A loaded RunConfig + Catalog wired to the synthetic controlled dataset."""
    paths = _build_csvs(tmp_path)
    spec = _make_spec(dev_spec, paths, tmp_path)
    cfg = load_config(write_spec(spec))
    catalog = Catalog.from_run_config(cfg)
    run = lambda: run_analysis(cfg, _stage_raw(), catalog=catalog)  # noqa: E731
    run.cfg = cfg
    run.catalog = catalog
    run.paths = paths
    return run


def _tables(result) -> dict[str, pd.DataFrame]:
    """Load the analysis's persisted tables (run_analysis returns CSV paths)."""
    return {name: pd.read_csv(path) for name, path in result.table_paths.items()}


def test_controlled_seed_report_excludes_failed_games_with_cached_strength(env):
    pd.DataFrame([
        {"experiment": TREAT_EXP, "game_id": "tr-s1-r0b", "player_id": 0,
         "valid_turn_count": 5, "failed_turn_count": 1},
    ]).to_csv(env.paths["tokens"], index=False)
    tables = _tables(env())
    row = _summary_row(tables, 1, 0, "GPT-OSS-120B-Simple", "Every-turn")
    assert row["run_count"] == 2


def _summary_row(tables, seed, player_id, strategist, condition):
    frame = tables["seed_player_summary"]
    match = frame[
        (frame["seed"] == seed) & (frame["player_id"] == player_id)
        & (frame["strategist"] == strategist) & (frame["condition"] == condition)
    ]
    assert len(match) == 1, f"expected exactly one row for {(seed, player_id, strategist, condition)}"
    return match.iloc[0]


# ── analysis: acceptance, exclusion, aggregation ───────────────────────────────
def test_fully_controlled_dataset_accepted(env):
    tables = _tables(env())
    assert set(tables) == {
        "seed_player_summary", "seed_player_probability", "seed_player_adjusted",
        "seed_player_index",
    }
    assert len(tables["seed_player_summary"]) == 8  # 6 seed-1 + 2 seed-2 combos


def test_uncontrolled_only_dataset_is_loud(tmp_path, write_spec, dev_spec):
    paths = _build_csvs(tmp_path, uncontrolled_only=True)
    spec = _make_spec(dev_spec, paths, tmp_path)
    cfg = load_config(write_spec(spec))
    catalog = Catalog.from_run_config(cfg)
    with pytest.raises(AnalysisError, match="no controlled rows"):
        run_analysis(cfg, _stage_raw(), catalog=catalog)


def test_uncontrolled_rows_excluded_from_mixed_data(env):
    tables = _tables(env())
    summary = tables["seed_player_summary"]
    assert not summary["experiment"].str.contains(UNCONTROLLED_EXP).any()
    index = tables["seed_player_index"]
    # The uncontrolled game would add runs to pair (0, ...) only via seed matching;
    # every pair keeps exactly its controlled game counts.
    counts = {(int(r.seed), int(r.player_id)): int(r.run_count)
              for r in index.itertuples(index=False)}
    assert counts == {(1, 0): 6, (1, 1): 6, (2, 0): 2, (2, 1): 2}


def test_rotations_and_repeats_average_equally_without_ci(env):
    tables = _tables(env())
    row = _summary_row(tables, 1, 0, "GPT-OSS-120B-Simple", "Every-turn")
    # Rotations (r0, r1) plus the genuine repeated run (r0b) all count once.
    assert int(row["run_count"]) == 3
    assert float(row["mean_adjusted_strength"]) == pytest.approx(0.60)
    assert float(row["mean_weighted_victory_probability"]) == pytest.approx(0.55)
    # No confidence-interval columns on any emitted table.
    for table in tables.values():
        assert not [c for c in table.columns if c.startswith("ci_")]


def test_duplicate_per_player_source_rows_are_loud(env):
    for source in ("panel", "strength"):
        frame = pd.read_csv(env.paths[source])
        pd.concat([frame, frame.iloc[[0]]]).to_csv(env.paths[source], index=False)
        with pytest.raises(AnalysisError, match="duplicate records per"):
            env()
        frame.to_csv(env.paths[source], index=False)  # restore for the next case


def test_focus_means_before_dominant_with_tie_order(env):
    tables = _tables(env())
    vanilla = _summary_row(tables, 1, 0, "Vanilla", "Vanilla")
    # All four focus means tie at 25%: the first in the listed order wins.
    assert str(vanilla["dominant_focus"]) == "Domination"
    assert float(vanilla["dominant_focus_pct"]) == pytest.approx(25.0)
    assert float(vanilla["science_focus_pct"]) == pytest.approx(25.0)
    science = _summary_row(tables, 1, 0, "GPT-OSS-120B-Simple", "Every-turn")
    assert str(science["dominant_focus"]) == "Science"
    assert float(science["dominant_focus_pct"]) == pytest.approx(50.0)


def test_vanilla_match_differences_and_missing_baseline(env):
    tables = _tables(env())
    simple = _summary_row(tables, 1, 0, "GPT-OSS-120B-Simple", "Every-turn")
    assert float(simple["matched_vanilla_adjusted_strength"]) == pytest.approx(0.40)
    assert float(simple["adjusted_strength_difference"]) == pytest.approx(0.20)
    assert bool(simple["has_matched_vanilla"])
    vanilla = _summary_row(tables, 1, 0, "Vanilla", "Vanilla")
    # The difference is blank for Vanilla itself.
    assert pd.isna(vanilla["adjusted_strength_difference"])
    assert pd.isna(vanilla["matched_vanilla_adjusted_strength"])
    # Seed 2 has no baseline at all: rows survive with blank differences.
    seed2 = _summary_row(tables, 2, 0, "GPT-OSS-120B-Simple", "Every-turn")
    assert pd.isna(seed2["adjusted_strength_difference"])
    assert not bool(seed2["has_matched_vanilla"])
    assert float(seed2["mean_adjusted_strength"]) == pytest.approx(0.60)


def test_treatment_and_vanilla_run_counts_are_independent(env):
    tables = _tables(env())
    treatment = _summary_row(tables, 1, 0, "GPT-OSS-120B-Simple", "Every-turn")
    vanilla = _summary_row(tables, 1, 0, "Vanilla", "Vanilla")
    assert int(treatment["run_count"]) == 3
    assert int(vanilla["run_count"]) == 2
    # The treatment row keeps its own count even though the matched mean came
    # from a different number of baseline runs.
    assert float(treatment["matched_vanilla_adjusted_strength"]) == pytest.approx(0.40)


def test_baseline_canonicalized_without_suffix_pairing(env):
    tables = _tables(env())
    vanilla = _summary_row(tables, 1, 1, "Vanilla", "Vanilla")
    assert str(vanilla["strategist"]) == "Vanilla"
    assert str(vanilla["condition"]) == "Vanilla"
    assert str(vanilla["player_type"]) == "Vanilla"
    # Suffix pairing still splits treatment identities.
    per5 = _summary_row(tables, 1, 1, "GPT-OSS-120B-Simple", "Per-5")
    assert str(per5["player_type"]) == "GPT-OSS-120B-Simple-Per-5"


def test_ordering_and_colors_stable(env):
    result = env()
    meta = result.metadata
    assert meta["strategist_order"] == ["GPT-OSS-120B-Simple", "Kimi-K2.5"]
    assert meta["condition_order"] == ["Every-turn", "Per-5"]
    assert meta["base_label"] == "Every-turn"
    assert meta["vanilla_label"] == "Vanilla"
    assert meta["strategist_colors"]["GPT-OSS-120B-Simple"] == "#FF7F00"
    kimi_color = env.catalog.strategist_model_colors()["Kimi-K2.5"]
    assert meta["strategist_colors"]["Kimi-K2.5"] == kimi_color
    assert meta["strategist_colors"]["Vanilla"] == "#555555"
    assert meta["grid_points"] == GRID_POINTS
    assert meta["baseline_experiment"] == BASELINE_EXP
    assert meta["has_baseline"] is True


def test_condition_pairing_required(tmp_path, write_spec, dev_spec):
    paths = _build_csvs(tmp_path)
    spec = _make_spec(dev_spec, paths, tmp_path)
    spec["presentation"]["condition_pairing"]["enabled"] = False
    cfg = load_config(write_spec(spec))
    catalog = Catalog.from_run_config(cfg)
    with pytest.raises(AnalysisError, match="condition pairing must be enabled"):
        run_analysis(cfg, _stage_raw(), catalog=catalog)


# ── analysis: probability curves ───────────────────────────────────────────────
def test_interpolation_grid_averages_covering_runs():
    grid = np.round(np.arange(GRID_POINTS) / (GRID_POINTS - 1), 10)
    runs = pd.DataFrame({
        "seed": [1, 1, 1, 1], "player_id": [0, 0, 0, 0],
        "strategist": ["A"] * 4, "condition": ["Every-turn"] * 4,
        "game_id": ["g1", "g1", "g2", "g2"],
        "turn_progress": [0.2, 0.6, 0.4, 1.0],
        "predicted_win_probability": [0.2, 0.6, 0.0, 1.0],
    })
    curves = interpolate_curves(runs, grid, CURVE_KEYS)
    points = curves[(1, 0, "A", "Every-turn")]
    by_progress = {round(p, 2): (v, n) for p, v, n in points}
    # Run 1 covers [0.2, 0.6] with y = x; run 2 covers [0.4, 1.0] with
    # y = (x - 0.4) / 0.6. Each point averages exactly the covering runs.
    assert round(by_progress[0.2][0], 6) == round(0.2, 6)
    assert by_progress[0.2][1] == 1
    assert round(by_progress[0.4][0], 6) == round((0.4 + 0.0) / 2, 6)
    assert by_progress[0.4][1] == 2
    assert round(by_progress[0.6][0], 6) == round((0.6 + (0.6 - 0.4) / 0.6) / 2, 6)
    assert round(by_progress[1.0][0], 6) == round(1.0, 6)
    assert by_progress[1.0][1] == 1
    # No extrapolation: nothing outside both runs' observed ranges.
    assert 0.0 not in by_progress and 0.1 not in by_progress
    assert 0.0 <= min(by_progress) and max(by_progress) <= 1.0
    assert all(0.0 <= v <= 1.0 for v, _ in by_progress.values())


def test_seats_of_one_game_are_separate_runs():
    # Two seats of the same strategist in one game, pooled without player_id in
    # the keys (as Win-probability trends does). Merging them into one run made
    # exact grid hits return only the last row's seat.
    grid = np.round(np.arange(GRID_POINTS) / (GRID_POINTS - 1), 10)
    runs = pd.DataFrame({
        "strategist": ["A"] * 6, "condition": [""] * 6, "game_id": ["g1"] * 6,
        "player_id": [1, 2, 1, 2, 1, 2],
        "turn_progress": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
        "predicted_win_probability": [0.4, 0.2, 0.6, 0.2, 1.0, 0.0],
    })
    points = interpolate_curves(runs, grid, ["strategist", "condition"])[("A", "")]
    by_progress = {round(p, 2): (v, n) for p, v, n in points}
    assert by_progress[0.5] == (0.4, 2)
    assert by_progress[1.0] == (0.5, 2)
    assert by_progress[0.25] == (round((0.5 + 0.2) / 2, 6), 2)


@pytest.mark.parametrize("params", [
    {"baseline_experiment": "base"},
    {"baseline_experiment": "base", "relative_to": "game_leader"},
    {},  # implicit: each experiment's own VPAI seats
])
def test_adjusted_curves_match_the_strength_adjustment(params):
    from types import SimpleNamespace

    from bench.adjust.strength import cell_adjust_rows
    from bench.analyses.performance.curves import adjusted_curves

    # Flat curves, so each grid point must equal the strength stage's
    # adjustment of that constant probability.
    seats = [
        # game, experiment, player_id, player_type, probability
        ("b1", "base", 0, "Vanilla", 0.2), ("b1", "base", 1, "Vanilla", 0.3),
        ("b2", "base", 0, "Vanilla", 0.4), ("b2", "base", 1, "Vanilla", 0.1),
        ("t1", "exp", 0, "A", 0.5), ("t1", "exp", 1, "Vanilla", 0.25),
    ]
    pool = pd.DataFrame([
        {"game_id": g, "experiment": e, "player_id": pid, "player_type": pt,
         "seed": 1, "seating_rotation": 0, "turn_progress": x,
         "predicted_win_probability": p}
        for g, e, pid, pt, p in seats for x in (0.0, 1.0)
    ])
    pool["strategist"] = pool["player_type"]
    pool["condition"] = ""
    catalog = SimpleNamespace(vanilla_label="Vanilla")
    grid = np.round(np.arange(GRID_POINTS) / (GRID_POINTS - 1), 10)
    shown = pool[pool["game_id"] == "t1"]
    curves = adjusted_curves(shown, pool, grid, ["strategist", "condition"], catalog, params)

    rows = pool.drop_duplicates(["game_id", "player_id"]).rename(
        columns={"predicted_win_probability": "weighted_strength"}
    )
    rows = rows.assign(controlled=True, is_winner=0)
    expected = cell_adjust_rows(rows, catalog, params, enforce_winner=False)
    expected = expected.set_index(["game_id", "player_id"])["adjusted_strength"]
    for strategist, player_id in (("A", 0), ("Vanilla", 1)):
        if not np.isfinite(expected[("t1", player_id)]):
            # Implicit mode: exp has no VPAI seat at player 0, so A has no
            # matched baseline and is left out, as in the strength panel.
            assert (strategist, "") not in curves
            continue
        points = curves[(strategist, "")]
        assert len(points) == GRID_POINTS
        assert {n for _x, _v, n in points} == {1}
        want = round(float(expected[("t1", player_id)]), 6)
        assert {v for _x, v, _n in points} == {want}


def test_single_point_run_contributes_nothing():
    grid = np.round(np.arange(GRID_POINTS) / (GRID_POINTS - 1), 10)
    runs = pd.DataFrame({
        "seed": [1], "player_id": [0], "strategist": ["A"],
        "condition": ["Every-turn"], "game_id": ["g1"],
        "turn_progress": [0.5], "predicted_win_probability": [0.5],
    })
    assert interpolate_curves(runs, grid, CURVE_KEYS) == {}


def test_curves_average_runs_on_the_grid(env):
    tables = _tables(env())
    probability = tables["seed_player_probability"]
    key = probability[
        (probability["seed"] == 1) & (probability["player_id"] == 0)
        & (probability["strategist"] == "GPT-OSS-120B-Simple")
        & (probability["condition"] == "Every-turn")
    ]
    # Every run covers [0.2, 0.8] with y = x, so the mean curve is the grid.
    progresses = sorted(key["turn_progress"].round(2))
    assert progresses == [round(i / 100, 2) for i in range(20, 81)]
    assert np.allclose(key.sort_values("turn_progress")["mean_predicted_win_probability"],
                       progresses, atol=1e-6)
    assert set(key["n_runs"]) == {3}
    # Seed-2 keys average their two runs.
    seed2 = probability[
        (probability["seed"] == 2) & (probability["strategist"] == "Vanilla")
    ]
    assert seed2.empty  # no baseline rows exist for seed 2


def test_conflicting_prediction_points_are_loud(env):
    pred = pd.read_csv(env.paths["predictions"])
    conflicting = pred.iloc[[0]].copy()
    conflicting["predicted_win_probability"] = 0.123
    pd.concat([pred, conflicting]).to_csv(env.paths["predictions"], index=False)
    with pytest.raises(AnalysisError, match="conflicting duplicate"):
        env()
    # Exact duplicate rows collapse silently instead.
    pd.concat([pred, pred.iloc[[0]]]).to_csv(env.paths["predictions"], index=False)
    assert env() is not None


def test_pair_without_predictions_keeps_scalar_summary(env):
    # Wipe every prediction for seed 2's treatment games; the scalar rows and
    # the page survive, and the pair is flagged as having no curve.
    pred = pd.read_csv(env.paths["predictions"])
    keep = ~pred["game_id"].isin(["tr-s2-r0", "tr-s2-r1"])
    pred[keep].to_csv(env.paths["predictions"], index=False)
    result = env()
    tables = _tables(result)
    row = _summary_row(tables, 2, 0, "GPT-OSS-120B-Simple", "Every-turn")
    assert float(row["mean_adjusted_strength"]) == pytest.approx(0.60)
    index = tables["seed_player_index"]
    pair = index[(index["seed"] == 2) & (index["player_id"] == 0)].iloc[0]
    assert not bool(pair["has_probability"])
    probability = tables["seed_player_probability"]
    assert probability[(probability["seed"] == 2)].empty
    assert any(
        "no usable prediction rows" in note
        for note in result.metadata["coverage"]["notes"]
    )
    assert "no usable prediction rows" not in result.summary.lower()


# ── config validation ──────────────────────────────────────────────────────────
def test_config_requires_exactly_one_estimator(tmp_path, write_spec, dev_spec):
    paths = _build_csvs(tmp_path)
    spec = _make_spec(dev_spec, paths, tmp_path)
    spec["estimators"].append({
        "id": "est2", "model": "score", "fit": "pretrained", "predict": "in_sample",
        "enabled": True, "predict_subset": "all",
        "save_predictions": paths["predictions"],
        "pretrained": {"model_dir": "pretrained/score/"},
    })
    spec["analyses"][0]["uses"]["estimators"] = ["est", "est2"]
    with pytest.raises(ConfigError, match="exactly one estimator"):
        load_config(write_spec(spec))


def test_config_requires_one_strength_table_reference(tmp_path, write_spec, dev_spec):
    paths = _build_csvs(tmp_path)
    spec = _make_spec(dev_spec, paths, tmp_path)
    spec["analyses"][0]["uses"]["tables"] = []
    with pytest.raises(ConfigError, match="exactly one strength table"):
        load_config(write_spec(spec))


def test_config_rejects_removed_template_key(tmp_path, write_spec, dev_spec):
    paths = _build_csvs(tmp_path)
    spec = _make_spec(dev_spec, paths, tmp_path)
    spec["report"]["template"] = "default"
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_spec(spec))


def test_config_rejects_two_enabled_controlled_seed_analyses(
    tmp_path, write_spec, dev_spec
):
    paths = _build_csvs(tmp_path)
    spec = _make_spec(dev_spec, paths, tmp_path)
    spec["analyses"].append({
        "id": "controlled_seed_again",
        "module": "performance.controlled_seed_report",
        "enabled": True,
        "uses": {"estimators": ["est"], "tables": ["strength"]},
        "params": {},
    })
    with pytest.raises(ConfigError, match="at most one"):
        load_config(write_spec(spec))


def test_config_tabs_must_offer_a_matched_maps_tab(tmp_path, write_spec, dev_spec):
    paths = _build_csvs(tmp_path)
    spec = _make_spec(dev_spec, paths, tmp_path)
    spec["analyses"].append({"id": "flavors", "module": "behavior.flavors", "enabled": True,
                             "params": {}})
    spec["analyses"].append({"id": "diplomacy", "module": "behavior.diplomacy", "enabled": True,
                             "params": {}})
    spec["analyses"].append({"id": "commitment", "module": "behavior.commitment", "enabled": True,
                             "params": {}})
    spec["analyses"].append({"id": "policies", "module": "behavior.policies", "enabled": True,
                             "params": {}})
    spec["analyses"].append({"id": "coverage", "module": "performance.experiment_completeness",
                             "enabled": True, "uses": {"tables": ["strength"]}, "params": {}})
    tabs = ["flavors", "diplomacy", "commitment", "policies"]
    spec["analyses"][0]["uses"]["analyses"] = tabs
    cfg = load_config(write_spec(spec))
    assert cfg.analyses[0].uses_analyses == tabs
    spec["analyses"][0]["uses"]["analyses"] = ["flavors", "coverage"]
    with pytest.raises(ConfigError, match="offer a Matched Maps tab"):
        load_config(write_spec(spec))


# ── report: rendering the controlled-seed chapter ─────────────────────────────
@pytest.fixture
def rendered(env):
    env()
    result = run_report(env.cfg)
    out = report_dir(env.cfg)
    return env, result, out


def _read(out, name):
    return (out / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("footer", [None, "Copyright **Example** [Site](https://example.com)", ""])
def test_footer_on_controlled_seed_pages(env, footer):
    env()
    env.cfg.report["footer"] = footer
    run_report(env.cfg)
    out = report_dir(env.cfg)
    for page in (out / "controlled-seed").glob("*.html"):
        html = page.read_text(encoding="utf-8")
        assert ("<footer" in html) == (footer != "")
        assert "@article{chen2026civbench," not in html
        if footer is None:
            assert 'href="https://arxiv.org/abs/2604.07733"' in html
        elif footer:
            assert 'Copyright <strong>Example</strong> <a href="https://example.com">Site</a>' in html


def test_controlled_chapter_rides_along_with_the_family_report(rendered):
    env, result, out = rendered
    expected = {
        "report.md", "index.html",
        "controlled-seed/index.html",
        "controlled-seed/seed-1-player-0.html", "controlled-seed/seed-1-player-1.html",
        "controlled-seed/seed-2-player-0.html", "controlled-seed/seed-2-player-1.html",
        "assets/report.css", "assets/report-common.js",
        "assets/report-help.js",
        "assets/plotly.min.js",
        "assets/curve-chart.js",
        "assets/controlled-seed-report.js",
    }
    written = {
        str(path.relative_to(out)).replace("\\", "/")
        for path in map(type(out), result.written)
    }
    assert written == expected
    assert all((out / rel).exists() for rel in expected)
    # The analysis's three tables are copied into the self-contained asset tree.
    for table in ("seed_player_summary", "seed_player_probability", "seed_player_index"):
        assert (out / "assets" / "controlled_seed" / f"{table}.csv").exists()
    assert result.formats == ["md", "html"]
    assert result.n_sections == 1
    # The only analysis is the chapter, so no family page exists and the
    # sidebar lists the chapter parallel to where families would sit.
    assert not (out / "performance.html").exists()
    markdown = _read(out, "report.md")
    assert "## " in markdown
    assert "## Overview" in markdown
    overview_page = _read(out, "index.html")
    assert '<a href="controlled-seed/index.html">' in overview_page


def test_chapter_pages_carry_the_site_sidebar(rendered):
    env, result, out = rendered
    index = _read(out, "controlled-seed/index.html")
    # Sidebar links are rebased for pages inside the chapter folder.
    assert '<a href="../index.html">Overview</a>' in index
    assert 'href="../controlled-seed/index.html#seed-1"' in index
    assert 'aria-current="page">' in index
    assert "<h1>" in index
    detail = _read(out, "controlled-seed/seed-1-player-0.html")
    assert '<a href="../index.html">Overview</a>' in detail
    assert '<link rel="stylesheet" href="../assets/report.css">' in detail
    # The shared util loads before the page script on every chapter page.
    assert '<script src="../assets/report-common.js" defer></script>' in detail
    assert '<script src="../assets/report-help.js" defer></script>' in detail
    assert '<script src="../assets/controlled-seed-report.js" defer></script>' in detail
    assert detail.index("report-help.js") < detail.index("controlled-seed-report.js")
    assert detail.index("report-common.js") < detail.index("controlled-seed-report.js")
    assert 'aria-describedby="chapter-help"' in index
    assert 'class="help-text" role="tooltip" id="chapter-help">' in index
    assert "Synthetic dev spec for the test suite" in index
    # The family report's sidebar points into the chapter folder.
    overview_page = _read(out, "index.html")
    assert 'href="controlled-seed/index.html#seed-1"' in overview_page
    assert 'href="controlled-seed/index.html#seed-2"' in overview_page


def test_two_heatmaps_per_seed_and_blank_cells(rendered):
    env, result, out = rendered
    index = _read(out, "controlled-seed/index.html")
    assert index.count('<figure class="heat-figure">') == 4  # 2 seeds x 2 heatmaps
    assert index.count("<h2") == 3  # seed 1, seed 2, source tables
    assert index.count('id="seed-1"') == 1 and index.count('id="seed-2"') == 1
    # The completed global grid leaves unobserved combinations blank (12 empty
    # cells per heatmap: 4 in seed 1, 8 in seed 2) and the strength heatmap's
    # Avg column adds 3 more in seed 2 (the two Kimi rows and the vanilla row
    # have no seed-2 cells to pool).
    assert index.count("heat-cell-empty") == 27
    # One isolated vanilla tbody per heatmap.
    assert index.count('<tbody class="vanilla-body">') == 4


def test_vanilla_is_separate_condition_row(rendered):
    env, result, out = rendered
    index = _read(out, "controlled-seed/index.html")
    assert '<tr class="vanilla-row"><th scope="row" class="row-label">' in index
    assert 'data-tip="VPAI self-play: all players use VPAI.">VPAI</span>' in index
    detail = _read(out, "controlled-seed/seed-1-player-0.html")
    assert '<tr class="vanilla-row">' in detail
    # The strength cell is colored like the overview heatmap: the baseline 0.40 sits
    # exactly on the RdYlBu 0.4 anchor.
    assert 'style="background-color:#fee090;color:#18202a" data-value="0.4"' in detail
    assert '<tbody class="vanilla-body"><tr class="vanilla-row">' in detail


def test_overview_columns_pair_player_with_civilization(rendered):
    env, result, out = rendered
    index = _read(out, "controlled-seed/index.html")
    assert re.search(r'<th scope="col" data-col="\d+">1: Greece</th>', index)
    assert re.search(r'<th scope="col" data-col="\d+">0: Rome</th>', index)  # seed 2 has only Rome
    # The kimi game gives seed-1 player 0 a second civilization, so the heading
    # lists both rather than hiding the comparability problem.
    assert re.search(r'<th scope="col" data-col="\d+">0: Egypt, Rome</th>', index)
    assert "Player 0" not in index and "Player 1" not in index


def test_cells_link_with_preselection_query(rendered):
    env, result, out = rendered
    index = _read(out, "controlled-seed/index.html")
    assert (
        'href="seed-1-player-0.html?strategist=GPT-OSS-120B-Simple&amp;'
        'condition=Every-turn"' in index
    )
    assert 'href="seed-1-player-1.html?strategist=Vanilla&amp;condition=Vanilla"' in index


def test_cell_tooltips_share_one_meaning(rendered):
    env, result, out = rendered
    index = _read(out, "controlled-seed/index.html")
    # Both heatmaps carry the same tooltip per cell: the row title and seat,
    # then strength, focus, and runs as tab-separated grid lines.
    expected = (
        'data-tip="GPT-OSS-120B-Simple | Every-turn\n'
        "P0 · Rome\n"
        "Strength\t0.600\n"
        "Focus\tScience\t50%\n"
        'Runs\t3"'
    )
    assert index.count(expected) == 2  # the strength and the focus heatmap
    # The one-run kimi game shows its own civilization on the seat line.
    assert (
        'data-tip="Kimi-K2.5 | Every-turn\nP0 · Egypt\n'
        'Strength\t0.700\nFocus\tCulture\t70%\nRuns\t1"'
    ) in index


def test_strength_scale_is_rdylbu(rendered):
    env, result, out = rendered
    index = _read(out, "controlled-seed/index.html")
    # Vanilla 0.40 lands exactly on the yellow-side anchor #fee090 and Kimi
    # 0.70 on #abd9e9: the scale runs red at 0, yellow at 0.5, blue at 1.
    assert "background-color:#fee090" in index
    assert "background-color:#abd9e9" in index
    assert "red 0, yellow 0.5, blue 1" in index


def test_strength_heatmap_leads_with_avg_column(rendered):
    env, result, out = rendered
    index = _read(out, "controlled-seed/index.html")
    # One Avg column per strength heatmap (one per seed), none on the focus
    # heatmaps.
    assert index.count('<th scope="col" class="col-divider" data-col="1">') == 2
    # The vanilla row pools both seats' runs; a one-seat row pools its own.
    assert ('data-tip="VPAI\nSeed average\nStrength\t0.400\nRuns\t4\tover 2 seats"'
            in index)
    assert 'Strength\t0.700\nRuns\t1\tover 1 seat"' in index
    # The avg cell is a colored summary, not a link into a detail page.
    assert (
        '<td class="heat-cell heat-cell-value col-divider" '
        'style="background-color:#fee090;color:#18202a" data-value="0.4" '
        'data-tip="VPAI\nSeed average\nStrength\t0.400\n'
        'Runs\t4\tover 2 seats">0.40</td>' in index
    )
    # Seed 2's uncovered rows leave the avg cell blank (Kimi x 2, vanilla).
    assert index.count('class="heat-cell heat-cell-empty col-divider"') == 3


def test_shared_color_util_and_adaptive_axis(rendered):
    env, result, out = rendered
    common = _read(out, "assets/report-common.js")
    assert common.startswith("/* civ-bench")
    assert "window.civBench" in common
    assert "distinguishColors" in common
    script = _read(out, "assets/curve-chart.js")
    assert "distinguishColors" in script  # same-family colors get spread
    assert "Plotly.restyle" in script
    assert '"yaxis.autorange": true' in script
    assert "svgTag" not in script
    assert "curve-data" not in script
    detail = _read(out, "controlled-seed/seed-1-player-0.html")
    # Relative (adjusted strength) and Absolute charts, the Plotly bundle once.
    assert 'id="plotly-matched-map-relative"' in detail
    assert 'id="plotly-matched-map-absolute"' in detail
    assert detail.count('src="../assets/plotly.min.js"') == 1
    assert '"hovermode":"x unified"' in detail


def test_detail_page_columns_and_run_counts(rendered):
    env, result, out = rendered
    detail = _read(out, "controlled-seed/seed-1-player-0.html")
    assert "Per-condition means over every unique run" in detail
    # The comparison is tabbed like the overview, each tab a shared heatmap.
    assert 'data-view="strength"' in detail and 'data-view="focus"' in detail
    assert 'class="comparison"' not in detail
    # Condensed headers carry the full wording as tooltips.
    assert 'data-tip="Unique runs averaged">Runs<' in detail
    assert 'data-tip="Mean weighted victory probability">Win prob<' in detail
    assert 'data-tip="Mean adjusted strength">Adj strength<' in detail
    assert 'data-tip="Dominant victory focus">Focus<' in detail
    assert 'data-tip="Domination focus %">Dom %<' in detail
    assert "rotations and repeated runs contribute equally" in detail
    # The difference column is gone; strength and focus cells are colored like
    # the overview heatmaps.
    assert "Difference" not in detail
    assert "<td>+0.2000</td>" not in detail
    assert ">Science 50%</td>" in detail
    assert "background-color:#93c293" in detail  # Science 50% focus cell
    # Prev/next/return links stay inside the chapter folder.
    assert 'href="index.html#seed-1"' in detail
    assert 'href="seed-1-player-1.html"' in detail


def test_strategist_checkboxes_replace_the_dropdown(rendered):
    env, result, out = rendered
    detail = _read(out, "controlled-seed/seed-1-player-0.html")
    assert "<select" not in detail
    assert 'class="curve-chart" data-query-select="true" data-sync="matched-map"' in detail
    assert 'id="matched-map-relative-filters"' in detail
    assert 'id="matched-map-absolute-filters"' in detail
    assert 'src="../assets/curve-chart.js"' in detail
    # VPAI leads and controls the self-play reference; with only two other
    # strategists everyone is checked and no preset buttons appear. Each of
    # the two views has its own checkbox group.
    assert detail.count('type="checkbox"') == 6
    assert detail.index('value="Vanilla" checked') < detail.index('value="GPT-OSS-120B-Simple"')
    assert 'data-tip="VPAI self-play: all players use VPAI."><input' in detail
    assert 'value="GPT-OSS-120B-Simple" checked' in detail
    assert 'value="Kimi-K2.5" checked' in detail
    assert "chart-preset" not in detail
    script = _read(out, "assets/curve-chart.js")
    # Plotly provides the shared hover comparison while the chart script updates
    # trace visibility, colors, and the adaptive axis.
    assert "Plotly.restyle" in script
    assert "Plotly.relayout" in script
    assert "mousemove" not in script


def test_vanilla_curve_emphasized_and_missing_baseline_noted(rendered):
    env, result, out = rendered
    seed1 = _read(out, "controlled-seed/seed-1-player-0.html")
    assert '"vanilla":true' in seed1
    assert '"width":3.5' in seed1
    seed2 = _read(out, "controlled-seed/seed-2-player-0.html")
    assert '"vanilla":true' not in seed2
    assert "The dedicated VPAI baseline is unavailable" in seed2
    assert '<tr class="vanilla-row">' not in seed2


def test_comparability_warning_for_multiple_civilizations(rendered):
    env, result, out = rendered
    seed1 = _read(out, "controlled-seed/seed-1-player-0.html")
    assert "Multiple civilizations occupy this seed-player pair" in seed1
    assert "Egypt, Rome" in seed1
    other = _read(out, "controlled-seed/seed-1-player-1.html")
    assert "Multiple civilizations" not in other


def test_controlled_re_render_is_byte_stable(rendered):
    env, result, out = rendered
    first = {
        str(p.relative_to(out)): p.read_bytes()
        for p in out.rglob("*") if p.is_file()
    }
    run_report(env.cfg)
    second = {
        str(p.relative_to(out)): p.read_bytes()
        for p in out.rglob("*") if p.is_file()
    }
    assert second == first


def test_controlled_pages_skipped_without_html_format(env):
    env.cfg.report["formats"] = ["md"]
    env()
    result = run_report(env.cfg)
    out = report_dir(env.cfg)
    assert result.formats == ["md"]
    assert (out / "report.md").exists()
    assert not (out / "controlled-seed").exists()
    # The pages are html-only, so the run warns and the chapter drops the link
    # instead of pointing at pages that were never written.
    assert any("HTML-only" in warning for warning in result.warnings)
    assert "controlled-seed/index.html" not in _read(out, "report.md")


def test_omitted_formats_default_to_md_and_html(env):
    env.cfg.report["formats"] = None
    env()
    result = run_report(env.cfg)
    out = report_dir(env.cfg)
    assert result.formats == ["md", "html"]
    assert (out / "report.md").exists()
    assert (out / "index.html").exists()
    assert (out / "controlled-seed" / "index.html").exists()


@pytest.mark.parametrize("sections, matched_first", [
    (None, True),   # the default family order puts Matched Maps before Prediction
    (["controlled_seed"], True),
    (["controlled_seed", "pred_compare"], True),
    (["pred_compare", "controlled_seed"], False),
])
def test_controlled_chapter_replaces_the_performance_section(
    tmp_path, write_spec, dev_spec, sections, matched_first
):
    paths = _build_csvs(tmp_path)
    spec = _make_spec(dev_spec, paths, tmp_path)
    spec["analyses"].append({
        "id": "pred_compare", "module": "prediction.compare", "enabled": True,
        "uses": {"estimators": ["est"]}, "params": {},
    })
    spec["report"]["sections"] = sections
    cfg = load_config(write_spec(spec))
    catalog = Catalog.from_run_config(cfg)
    run_analysis(cfg, _stage_raw(), catalog=catalog)
    _emit_empty_manifest(cfg, "pred_compare")
    result = run_report(cfg)
    out = report_dir(cfg)
    assert result.n_sections == 2
    assert (out / "controlled-seed" / "index.html").exists()
    assert (out / "prediction.html").exists()
    # The chapter claims the analysis; the performance family has no sections
    # left, so it gets no page at all.
    assert not (out / "performance.html").exists()
    prediction = _read(out, "prediction.html")
    assert "controlled_seed" not in prediction
    for page in ("index.html", "prediction.html", "controlled-seed/index.html"):
        html = _read(out, page)
        assert (html.index(">Matched Maps</a>") < html.index(">Prediction</a>")) == matched_first
    markdown = _read(out, "report.md")
    assert (markdown.index("## Matched Maps") < markdown.index("## Prediction")) == matched_first


def _emit_empty_manifest(cfg, sid):
    from bench.reports.runner import _analyses_dir

    d = _analyses_dir(cfg, sid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({
        "id": sid, "module": "prediction.compare", "summary": "", "metadata": {},
        "empty": True, "tables": [], "figures": [], "artifacts": [],
    }), encoding="utf-8")


# ── renderer unit: escaping and query encoding ────────────────────────────────
def _tiny_doc() -> ControlledSeedDocument:
    summary = pd.DataFrame([{
        "seed": 1, "player_id": 0, "strategist": 'Weird & <Model>',
        "condition": "Per 5", "player_type": 'Weird & <Model>-Per-5',
        "experiment": "exp", "civilization": 'Civ "X"', "run_count": 2,
        "mean_weighted_victory_probability": 0.5, "mean_adjusted_strength": 0.6,
        "matched_vanilla_adjusted_strength": 0.4,
        "adjusted_strength_difference": 0.2, "has_matched_vanilla": True,
        "dominant_focus": "Science", "dominant_focus_pct": 72.0,
        "domination_focus_pct": 10.0, "culture_focus_pct": 10.0,
        "diplomatic_focus_pct": 8.0, "science_focus_pct": 72.0,
    }, {
        "seed": 1, "player_id": 0, "strategist": "Vanilla", "condition": "Vanilla",
        "player_type": "Vanilla", "experiment": "base", "civilization": 'Civ "X"',
        "run_count": 1, "mean_weighted_victory_probability": 0.4,
        "mean_adjusted_strength": 0.4,
        "matched_vanilla_adjusted_strength": np.nan,
        "adjusted_strength_difference": np.nan, "has_matched_vanilla": True,
        "dominant_focus": "Domination", "dominant_focus_pct": 25.0,
        "domination_focus_pct": 25.0, "culture_focus_pct": 25.0,
        "diplomatic_focus_pct": 25.0, "science_focus_pct": 25.0,
    }])
    probability = pd.DataFrame([
        {"seed": 1, "player_id": 0, "strategist": 'Weird & <Model>',
         "condition": "Per 5", "turn_progress": 0.5,
         "mean_predicted_win_probability": 0.6, "n_runs": 2},
        {"seed": 1, "player_id": 0, "strategist": "Vanilla", "condition": "Vanilla",
         "turn_progress": 0.5, "mean_predicted_win_probability": 0.4, "n_runs": 1},
    ])
    index = pd.DataFrame([{
        "seed": 1, "player_id": 0, "civilization": 'Civ "X"', "n_civilizations": 1,
        "run_count": 3, "has_matched_vanilla": True, "has_probability": True,
    }])
    return ControlledSeedDocument(
        title='Report & <Summary>', run_name="r", seed=1, config_path="c.json",
        output_root="reports", section_id="controlled_seed", summary="ok",
        metadata={
            "vanilla_label": "Vanilla", "base_label": "Every turn",
            "strategist_order": ['Weird & <Model>'],
            "condition_order": ["Every turn", "Per 5"],
            "strategist_colors": {'Weird & <Model>': "#123456", "Vanilla": "#555555"},
        },
        summary_table=summary, probability_table=probability, index_table=index,
        downloads=[],
    )


def test_mixed_vpai_keeps_its_condition_and_distinct_baseline():
    doc = _tiny_doc()
    mixed = doc.summary_table.iloc[-1].copy()
    mixed["condition"] = "Every-turn"
    mixed["experiment"] = "mixed"
    mixed["mean_adjusted_strength"] = 0.3
    doc.summary_table = pd.concat([doc.summary_table, mixed.to_frame().T], ignore_index=True)
    curve = doc.probability_table.iloc[-1].copy()
    curve["condition"] = "Every-turn"
    doc.probability_table = pd.concat([doc.probability_table, curve.to_frame().T], ignore_index=True)

    pages = render_controlled_seed_site(doc)
    overview = pages["controlled-seed/index.html"]
    assert ">VPAI | Every-turn</span>" in overview
    assert "Vanilla | Every-turn" not in overview
    assert "VPAI self-play: all players use VPAI." in overview
    assert "VPAI in games with LLM players." in overview
    assert "LLMs can counter VPAI&#x27;s playstyle." in overview
    assert "strategist=Vanilla&amp;condition=Every-turn" in overview

    detail = pages["controlled-seed/seed-1-player-0.html"]
    # One pinned self-play row per tab (Strength, Focus); the mixed VPAI row is
    # an ordinary condition row.
    assert detail.count('<tr class="vanilla-row">') == 2
    assert ">VPAI | Every-turn</span></th>" in detail
    # One VPAI checkbox covers both VPAI curves, and its tooltip explains both.
    assert detail.count('value="Vanilla"') == 1
    assert 'value="Vanilla" checked data-default="true"> VPAI</label>' in detail
    assert "The other VPAI lines are VPAI in games with LLM players." in detail
    # Trace metadata keeps the baseline distinct from the mixed VPAI condition
    # without embedding a second chart-data payload.
    assert '"name":"VPAI"' in detail
    assert '"vanilla":true' in detail
    assert '"condition":"Every-turn"' in detail
    assert 'id="curve-data"' not in detail


def test_game_log_links_and_seat_rows_follow_detail_filters():
    doc = _tiny_doc()
    doc.game_log = GameLogDocument(
        title=doc.title,
        section_id="game_log",
        games=pd.DataFrame([{
            "game_id": "g-treatment", "seed": 1, "seating_rotation": 0,
            "date_utc": "2026-09-18", "victory_type": "Science",
            "winner_player_id": 0, "winner_civilization": "Rome",
            "winner_is_vanilla": False,
        }]),
        game_players=pd.DataFrame([
            {"game_id": "g-treatment", "player_id": 0,
             "strategist": "Weird & <Model>", "condition": "Per 5",
             "is_vanilla": False, "civilization": "Rome", "is_winner": True},
            {"game_id": "g-treatment", "player_id": 1,
             "strategist": "Vanilla", "condition": "Base",
             "is_vanilla": True, "civilization": "Greece", "is_winner": False},
        ]),
        metadata={"vanilla_label": "Vanilla"},
    )
    pages = render_controlled_seed_site(doc)
    overview = pages["controlled-seed/index.html"]
    detail = pages["controlled-seed/seed-1-player-0.html"]
    assert 'games.html?seed=1' in overview
    assert 'Browse these games in the Game Log' in detail
    assert 'games.html?seed=1&amp;player=0&amp;strategist=Weird%20%26%20%3CModel%3E&amp;condition=Per%205' in detail
    assert 'games.html?seed=1&amp;player=0&amp;strategist=Vanilla' in detail
    assert 'class="seat-games"' in detail
    assert 'data-strategist="Weird &amp; &lt;Model&gt;"' in detail
    assert 'data-vanilla="false"' in detail
    assert '<tr class="game-row"' in detail
    assert '<tr hidden class="game-row"' not in detail


def test_renderer_escapes_labels_and_query_parameters():
    doc = _tiny_doc()
    doc.summary = "**2** runs for *one* seed, `**raw**` & <Model>."
    pages = render_controlled_seed_site(doc)
    overview = pages["controlled-seed/index.html"]
    assert (
        "<p><strong>2</strong> runs for <em>one</em> seed, "
        "<code>**raw**</code> &amp; &lt;Model&gt;.</p>"
    ) in overview
    # Without navigation the chapter page renders standalone (centered layout).
    assert "controlled-content" in overview
    assert "<title>" in overview and "Report &amp; &lt;Summary&gt;" in overview
    assert "<h1>" in overview
    assert "Weird &amp; &lt;Model&gt; | Per 5" in overview
    assert "href=\"seed-1-player-0.html?strategist=Weird+%26+%3CModel%3E&amp;condition=Per+5\"" in overview
    # Attribute values escape embedded quotes; tip newlines and tabs stay literal.
    assert (
        'data-tip="Weird &amp; &lt;Model&gt; | Per 5\nP0 · Civ &quot;X&quot;\n'
        'Strength\t0.600\nFocus\tScience\t72%\nRuns\t2"'
    ) in overview
    assert (
        'data-tip="VPAI\nP0 · Civ &quot;X&quot;\n'
        'Strength\t0.400\nFocus\tDomination\t25%\nRuns\t1"'
    ) in overview
    assert (
        'data-tip="Weird &amp; &lt;Model&gt; | Per 5\nSeed average\n'
        'Strength\t0.600\nRuns\t2\tover 1 seat"'
    ) in overview
    detail = pages["controlled-seed/seed-1-player-0.html"]
    assert "Weird &amp; &lt;Model&gt;" in detail
    # Checkbox values escape the same labels.
    assert 'value="Weird &amp; &lt;Model&gt;" checked' in detail
    assert "<script" in detail and "../assets/controlled-seed-report.js" in detail
    assert 'href="index.html#seed-1"' in detail
    assert pages["assets/report-common.js"].startswith("/* civ-bench")
    assert "distinguishColors" in pages["assets/report-common.js"]
    assert pages["assets/controlled-seed-report.js"].startswith("/* civ-bench")


def test_each_seed_offers_strength_and_focus_tabs(rendered):
    env, result, out = rendered
    index = _read(out, "controlled-seed/index.html")
    # One tab group per seed; the same view names switch every seed together.
    assert index.count('<div class="view-group">') == 2
    assert index.count('data-view="strength" aria-controls=') == 2
    assert 'aria-controls="seed-2-view-focus"' in index
    assert 'data-tip="Mean adjusted strength">Strength</button>' in index
    # A Focus cell opens the seat page on its Focus tab.
    assert "condition=Every-turn&amp;view=focus" in index
    assert "condition=Every-turn&amp;view=strength" not in index
    script = _read(out, "assets/report-help.js")
    assert 'get("view")' in script


def _strategic_tab() -> MatchedMapTab:
    """A Strategy tab like behavior.flavors writes it, for seed 1 seat 0."""
    def rows(keys):
        base = [
            {"row_kind": "baseline", "row_label": "Completed-experiment average",
             "metric": "flavor_offense_avg", "metric_group": "Military", "mean": 50.0,
             "difference": np.nan, "n_players": 4, "n_games": 4, "color_position": 0.5},
            {"row_kind": "group", "row_label": "Weird & <Model> | Per 5",
             "metric": "flavor_offense_avg", "metric_group": "Military", "mean": 80.0,
             "difference": 30.0, "n_players": 2, "n_games": 2, "color_position": 0.8},
        ]
        return pd.DataFrame([{**keys, **row} for row in base])

    def spec(title):
        return {
            "title": title, "help": "Flavor help.", "row": "row_label", "column": "metric",
            "value": "mean", "row_heading": "Strategist | Condition",
            "row_order": ["Completed-experiment average", "Weird & <Model> | Per 5"],
            "reference_rows": ["Completed-experiment average"],
            "baseline_rows": ["Completed-experiment average"],
            "column_order": ["flavor_offense_avg"],
            "column_names": {"flavor_offense_avg": "Offense"},
            "column_labels": {"flavor_offense_avg": "Off"},
            "column_group": "metric_group", "decimals": 0, "value_label": "Average",
            "tip_rows": [{"column": "difference", "label": "Vs. baseline", "signed": True,
                          "decimals": 1}],
        }

    return MatchedMapTab(
        name="beh_flavors", label="Strategy", tip="Strategic settings",
        seed_table=rows({"seed": 1}), seat_table=rows({"seed": 1, "player_id": 0}),
        seed_spec=spec("Seed flavors"), seat_spec=spec("Seat flavors"),
    )


def test_strategic_tab_renders_on_overview_and_seat_pages():
    doc = _tiny_doc()
    doc.tabs = [_strategic_tab()]
    pages = render_controlled_seed_site(doc)
    overview = pages["controlled-seed/index.html"]
    detail = pages["controlled-seed/seed-1-player-0.html"]
    for page, title in ((overview, "Seed flavors"), (detail, "Seat flavors")):
        assert 'data-tip="Strategic settings">Strategy</button>' in page
        assert f"<figcaption>{title}" in page
        # The baseline row is pinned; the tooltip carries the difference.
        assert '<tbody class="vanilla-body"><tr class="vanilla-row"><th scope="row" ' \
               'class="row-label">Completed-experiment average</th>' in page
        assert "Vs. baseline\t+30.0" in page
        assert ">80</td>" in page
    assert 'aria-controls="seed-1-view-beh_flavors"' in overview
    assert 'aria-controls="seat-view-beh_flavors"' in detail


def test_strategic_tab_without_rows_for_a_seat_says_so():
    doc = _tiny_doc()
    tab = _strategic_tab()
    tab.seat_table = tab.seat_table.assign(player_id=5)
    doc.tabs = [tab]
    detail = render_controlled_seed_site(doc)["controlled-seed/seed-1-player-0.html"]
    assert "No strategy data for this seat." in detail


def test_strategy_and_diplomacy_tabs_follow_strength_and_focus():
    doc = _tiny_doc()
    diplomacy = _strategic_tab()
    diplomacy.name, diplomacy.label, diplomacy.tip = "beh_diplomacy", "Diplomacy", "Diplomatic persona"
    doc.tabs = [_strategic_tab(), diplomacy]
    pages = render_controlled_seed_site(doc)
    for page in (pages["controlled-seed/index.html"], pages["controlled-seed/seed-1-player-0.html"]):
        labels = [">Strength</button>", ">Focus</button>", ">Strategy</button>", ">Diplomacy</button>"]
        positions = [page.index(label) for label in labels]
        assert positions == sorted(positions)


def _grand_strategy_tab() -> MatchedMapTab:
    """A Focus-style Grand strategy tab like behavior.commitment writes it."""
    cell = {"row_kind": "group", "seed": 1, "player_id": 0, "row_label": "Weird & <Model> | Per 5",
            "mean": 62.0, "color_position": 0.62, "category": "Spaceship", "strategy": "Spaceship",
            "text": "Spaceship 62%", "switches": 0.5, "n_players": 2, "n_games": 2}
    spec = {
        "title": "Main grand strategy", "help": "Grand strategy help.", "row": "row_label",
        "column": "seat", "value": "mean", "row_order": ["Weird & <Model> | Per 5"],
        "seat_columns": True, "decimals": 0, "unit": "%", "value_label": "Share of turns",
        "text_column": "text", "category_column": "category",
        "category_colors": {"Spaceship": "#4e9b4e"},
        "tip_rows": [{"column": "strategy", "label": "Strategy", "text": True}],
    }
    seat_spec = {**spec, "column": "metric", "seat_columns": False,
                 "column_order": ["gs_main"], "column_names": {"gs_main": "Main grand strategy"}}
    return MatchedMapTab(
        name="beh_commitment-grand-strategy", label="Grand strategy", tip="Main grand strategy",
        seed_table=pd.DataFrame([{**cell, "seat": 0, "metric": "gs_main"}]),
        seat_table=pd.DataFrame([{**cell, "metric": "gs_main"}]),
        seed_spec=spec, seat_spec=seat_spec,
    )


def test_seat_column_tabs_lay_out_like_the_focus_tab():
    from bench.reports.heatmap import category_background

    doc = _tiny_doc()
    doc.tabs = [_grand_strategy_tab()]
    pages = render_controlled_seed_site(doc)
    overview = pages["controlled-seed/index.html"]
    # One column per seat of the seed, headed like Focus, in the strategy color.
    panel = overview[overview.index('id="seed-1-view-beh_commitment-grand-strategy"'):]
    assert '<th scope="col" data-col="1">0: Civ &quot;X&quot;</th>' in panel
    assert f"background-color:{category_background('#4e9b4e', 0.62)}" in panel
    assert ">Spaceship 62%</td>" in panel
    assert "Share of turns\t62.0%" in panel and "Strategy\tSpaceship" in panel
    detail = pages["controlled-seed/seed-1-player-0.html"]
    assert 'aria-controls="seat-view-beh_commitment-grand-strategy"' in detail
    assert "Main grand strategy\nShare of turns\t62.0%" in detail


def test_tabs_load_from_listed_sections_and_warn_on_unusable_ones(tmp_path):
    from bench.reports.context import ReportBuildContext
    from bench.reports.controlled_seed import _matched_map_tabs
    from bench.reports.model import Section

    tab = _strategic_tab()
    tab.seed_table.to_csv(tmp_path / "flavors_by_seed.csv", index=False)
    tab.seat_table.to_csv(tmp_path / "flavors_by_seat.csv", index=False)
    flavors = Section(id="beh_flavors", module="behavior.flavors", metadata={
        "matched_maps": {"label": "Strategy", "tip": "Strategic settings",
                         "seed_table": "flavors_by_seed", "seat_table": "flavors_by_seat"},
        "heatmaps": {"flavors_by_seed": tab.seed_spec, "flavors_by_seat": tab.seat_spec},
    })
    plain = Section(id="beh_other", module="behavior.flavors", metadata={})
    ctx = ReportBuildContext(meta={}, sections=[flavors, plain])
    for name in ("flavors_by_seed", "flavors_by_seat"):
        ctx.record_table("beh_flavors", name, tmp_path, f"{name}.csv")
    tabs = _matched_map_tabs(ctx, ["beh_flavors", "beh_other", "gone"])
    assert [t.name for t in tabs] == ["beh_flavors"]
    assert tabs[0].label == "Strategy" and tabs[0].seed_spec["title"] == "Seed flavors"
    assert list(tabs[0].seat_table["player_id"]) == [0, 0]
    assert ctx.warnings == [
        "Matched Maps: analysis 'beh_other' offers no Matched Maps tab; its tab is skipped.",
        "Matched Maps: analysis 'gone' is not among the report sections; its tab is skipped.",
    ]


def test_one_section_may_declare_several_tabs(tmp_path):
    from bench.reports.context import ReportBuildContext
    from bench.reports.controlled_seed import _matched_map_tabs
    from bench.reports.model import Section

    tab = _strategic_tab()
    for name in ("a_by_seed", "b_by_seed"):
        tab.seed_table.to_csv(tmp_path / f"{name}.csv", index=False)
    for name in ("a_by_seat", "b_by_seat"):
        tab.seat_table.to_csv(tmp_path / f"{name}.csv", index=False)
    section = Section(id="beh_commitment", module="behavior.commitment", metadata={
        "matched_maps": [
            {"label": "Commitment", "key": "commitment", "seed_table": "a_by_seed",
             "seat_table": "a_by_seat"},
            {"label": "Grand strategy", "key": "grand-strategy", "seed_table": "b_by_seed",
             "seat_table": "b_by_seat"},
        ],
        "heatmaps": {},
    })
    ctx = ReportBuildContext(meta={}, sections=[section])
    for name in ("a_by_seed", "a_by_seat", "b_by_seed", "b_by_seat"):
        ctx.record_table("beh_commitment", name, tmp_path, f"{name}.csv")
    tabs = _matched_map_tabs(ctx, ["beh_commitment"])
    assert [(t.name, t.label) for t in tabs] == [
        ("beh_commitment-commitment", "Commitment"),
        ("beh_commitment-grand-strategy", "Grand strategy"),
    ]
    assert not ctx.warnings
