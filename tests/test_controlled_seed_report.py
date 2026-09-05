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

import numpy as np
import pandas as pd
import pytest

from bench.analyses import run_analysis
from bench.analyses.errors import AnalysisError
from bench.analyses.performance.controlled_seed_report import (
    GRID_POINTS,
    _interpolate_curves,
)
from bench.catalog import Catalog
from bench.config import ConfigError, load_config
from bench.reports import run_report
from bench.reports.controlled_seed import render_controlled_seed_site
from bench.reports.model import ControlledSeedDocument
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
        "seed_player_summary", "seed_player_probability", "seed_player_index"
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
    curves = _interpolate_curves(runs, grid)
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


def test_single_point_run_contributes_nothing():
    grid = np.round(np.arange(GRID_POINTS) / (GRID_POINTS - 1), 10)
    runs = pd.DataFrame({
        "seed": [1], "player_id": [0], "strategist": ["A"],
        "condition": ["Every-turn"], "game_id": ["g1"],
        "turn_progress": [0.5], "predicted_win_probability": [0.5],
    })
    assert _interpolate_curves(runs, grid) == {}


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
    assert "no usable prediction rows" in result.summary.lower()


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


# ── report: rendering the controlled-seed pages ───────────────────────────────
@pytest.fixture
def rendered(env):
    env()
    result = run_report(env.cfg)
    out = report_dir(env.cfg)
    return env, result, out


def _read(out, name):
    return (out / name).read_text(encoding="utf-8")


def test_controlled_pages_ride_along_with_the_family_report(rendered):
    env, result, out = rendered
    expected = {
        "report.md", "report.html", "performance.html",
        "controlled-seed.html",
        "seed-1-player-0.html", "seed-1-player-1.html",
        "seed-2-player-0.html", "seed-2-player-1.html",
        "assets/report.css", "assets/controlled-seed-report.js",
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
    # The family report links to the heatmap pages from the section's downloads.
    markdown = _read(out, "report.md")
    assert "[Controlled-seed heatmap pages (HTML)](controlled-seed.html)" in markdown
    performance = _read(out, "performance.html")
    assert 'href="controlled-seed.html"' in performance


def test_two_heatmaps_per_seed_and_blank_cells(rendered):
    env, result, out = rendered
    overview = _read(out, "controlled-seed.html")
    assert overview.count('<figure class="heat-figure">') == 4  # 2 seeds x 2 heatmaps
    assert overview.count("<h2") == 3  # seed 1, seed 2, source tables
    assert overview.count('id="seed-1"') == 1 and overview.count('id="seed-2"') == 1
    # The completed global grid leaves unobserved combinations blank (12 empty
    # cells per heatmap: 4 in seed 1, 8 in seed 2).
    assert overview.count("heat-cell-empty") == 24
    # One isolated vanilla tbody per heatmap.
    assert overview.count('<tbody class="vanilla-body">') == 4
    # The annex links back to the family report overview.
    assert '<a href="report.html">← Full report</a>' in overview


def test_vanilla_is_separate_condition_row(rendered):
    env, result, out = rendered
    overview = _read(out, "controlled-seed.html")
    assert '<tr class="vanilla-row"><th scope="row" class="row-label">Vanilla</th>' in overview
    detail = _read(out, "seed-1-player-0.html")
    assert '<tr class="vanilla-row">' in detail
    assert '<td class="vanilla-value">0.4000</td>' in detail


def test_cells_link_with_preselection_query(rendered):
    env, result, out = rendered
    overview = _read(out, "controlled-seed.html")
    assert (
        'href="seed-1-player-0.html?strategist=GPT-OSS-120B-Simple&amp;'
        'condition=Every-turn"' in overview
    )
    assert 'href="seed-1-player-1.html?strategist=Vanilla&amp;condition=Vanilla"' in overview
    # Tooltip carries exact value, civilization, and run count.
    assert "data-tip=" in overview
    assert "Runs: 3" in overview and "Civilization: Rome" in overview


def test_detail_page_columns_and_run_counts(rendered):
    env, result, out = rendered
    detail = _read(out, "seed-1-player-0.html")
    assert "6 source run(s)" in detail
    assert "Run count" in detail
    assert "Rotations" not in detail and "rotations" not in detail
    assert "Mean weighted victory probability" in detail
    assert "Difference from matched Vanilla adjusted strength" in detail
    assert "<td>+0.2000</td>" in detail  # 0.60 - 0.40
    assert "<td>+0.3000</td>" in detail  # Kimi 0.70 - 0.40
    # The Vanilla row's difference cell is blank.
    vanilla_row = detail.split('<tr class="vanilla-row">')[1].split("</tr>")[0]
    assert "<td></td>" in vanilla_row
    # The return link targets the annex overview, not the family report page.
    assert 'href="controlled-seed.html#seed-1"' in detail


def test_vanilla_curve_emphasized_and_missing_baseline_noted(rendered):
    env, result, out = rendered
    seed1 = _read(out, "seed-1-player-0.html")
    assert '"vanilla":true' in seed1
    assert '"width":3.5' in seed1
    seed2 = _read(out, "seed-2-player-0.html")
    assert '"vanilla":true' not in seed2
    assert "The dedicated Vanilla baseline is unavailable" in seed2
    assert '<tr class="vanilla-row">' not in seed2


def test_comparability_warning_for_multiple_civilizations(rendered):
    env, result, out = rendered
    seed1 = _read(out, "seed-1-player-0.html")
    assert "Multiple civilizations occupy this seed-player pair" in seed1
    assert "Egypt, Rome" in seed1
    other = _read(out, "seed-1-player-1.html")
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
    assert not (out / "controlled-seed.html").exists()
    assert not any(out.glob("seed-*-player-*.html"))
    # The pages are html-only, so the run warns and the section drops the link
    # instead of pointing at a page that was never written.
    assert any("HTML-only" in warning for warning in result.warnings)
    assert "controlled-seed.html" not in _read(out, "report.md")


def test_omitted_formats_default_to_md_and_html(env):
    env.cfg.report["formats"] = None
    env()
    result = run_report(env.cfg)
    out = report_dir(env.cfg)
    assert result.formats == ["md", "html"]
    assert (out / "report.md").exists()
    assert (out / "report.html").exists()
    assert (out / "controlled-seed.html").exists()


def test_controlled_pages_render_alongside_other_sections(tmp_path, write_spec, dev_spec):
    paths = _build_csvs(tmp_path)
    spec = _make_spec(dev_spec, paths, tmp_path)
    spec["analyses"].append({
        "id": "pred_compare", "module": "prediction.compare", "enabled": True,
        "uses": {"estimators": ["est"]}, "params": {},
    })
    cfg = load_config(write_spec(spec))
    catalog = Catalog.from_run_config(cfg)
    run_analysis(cfg, _stage_raw(), catalog=catalog)
    _emit_empty_manifest(cfg, "pred_compare")
    result = run_report(cfg)
    out = report_dir(cfg)
    assert result.n_sections == 2
    assert (out / "controlled-seed.html").exists()
    assert (out / "prediction.html").exists()
    assert (out / "performance.html").exists()


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


def test_renderer_escapes_labels_and_query_parameters():
    pages = render_controlled_seed_site(_tiny_doc())
    overview = pages["controlled-seed.html"]
    assert "Weird &amp; &lt;Model&gt; | Per 5" in overview
    assert "Report &amp; &lt;Summary&gt;" in overview
    assert "href=\"seed-1-player-0.html?strategist=Weird+%26+%3CModel%3E&amp;condition=Per+5\"" in overview
    # Attribute values escape embedded quotes.
    assert 'data-tip="Dominant focus: Science (72.00%)\nCivilization: Civ &quot;X&quot;\nRuns: 2"' in overview
    detail = pages["seed-1-player-0.html"]
    assert "Weird &amp; &lt;Model&gt;" in detail
    assert "<script" in detail and "assets/controlled-seed-report.js" in detail
    assert 'href="controlled-seed.html#seed-1"' in detail
    assert pages["assets/controlled-seed-report.js"].startswith("/* civ-bench")
