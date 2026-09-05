"""Shared pytest fixtures for the civ-bench test suite.

Tests live at ``civ-bench/tests/`` and run with ``pytest`` from the repo root
(``pip install -e ".[test]"`` first). They exercise the stage-0 scaffold: config
load/validation, the DAG, and the orthodox player_type composition — no stage
execution and no machine-specific data roots.

The core config fixtures build a **synthetic** run-spec in-process
(:func:`_make_dev_spec`) that mirrors ``configs/benchmark.dev.json`` structurally
but is checked into the suite itself, so the tests run on a fresh clone that has
never had the gitignored dev config. ``write_spec`` writes specs under ``tmp_path``
(never the tracked ``configs/`` tree) and points catalog resolution at the tracked
sibling catalogs via an explicit ``catalogs`` block.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEV_CONFIG = CONFIGS_DIR / "benchmark.dev.json"
PRETRAINED_TEMPLATE = CONFIGS_DIR / "benchmark.pretrained.template.json"

# Tracked catalogs every synthetic spec resolves against (siblings of the real
# templates). Injected as an explicit ``catalogs`` block by ``write_spec`` so the
# spec need not live next to them.
_CATALOGS = {
    "models": str(CONFIGS_DIR / "models.json"),
    "experiments": str(CONFIGS_DIR / "experiments.json"),
    "paths": str(CONFIGS_DIR / "paths.json"),
}


def _make_dev_spec() -> dict:
    """A synthetic run-spec mirroring ``configs/benchmark.dev.json``'s structure.

    The mutation tests in ``test_config.py`` index into this by position and id, so
    the shape is load-bearing: 3 estimators (``score``, ``attention``, ``xgboost``)
    in order, one ``strength`` adjust stage, and 17 analyses in the dev order
    (``bt_main`` … ``explore_token_costs``). ``runs_dir`` is a neutral placeholder —
    ``load_config`` never touches the filesystem there.
    """
    return {
        "name": "civbench-dev",
        "description": "Synthetic dev spec for the test suite (see conftest._make_dev_spec).",
        "seed": 42,
        "output": {"root": "reports", "suffix": "-dev"},
        "filters": {
            "staff_recent": {},
            "late_game": {"turn_range": [200, None]},
        },
        "groupings": {
            "strategy": {
                "kind": "argmax",
                "columns": ["domination_ratio", "culture_ratio", "diplomatic_ratio", "science_ratio"],
                "labels": ["Domination", "Culture", "Diplomatic", "Science"],
            }
        },
        "data": {
            "extract": {
                "enabled": True,
                "runs_dir": "runs_source/",
                "outputs": ["turns", "panel", "games", "tokens"],
                "max_dbs": None,
                "prune_missing": False,
                "force_rebuild": False,
            },
            "tables": {
                "turns": "runs/turn_data.csv",
                "panel": "runs/panel_data.csv",
                "games": "runs/game_data.csv",
                "tokens": "runs/model_token_usage.csv",
            },
            "filter": "staff_recent",
        },
        "estimators": [
            {
                "id": "score",
                "model": "score",
                "fit": "pretrained",
                "predict": "in_sample",
                "enabled": True,
                "predict_subset": "all",
                "save_predictions": "reports/estimators/score/predictions.csv",
                "pretrained": {"model_dir": "pretrained/score/"},
            },
            {
                "id": "attention",
                "model": "attention_mlp",
                "fit": "pretrained",
                "predict": "in_sample",
                "enabled": True,
                "predict_subset": "all",
                "save_predictions": "reports/estimators/attention/predictions.csv",
                "pretrained": {"model_dir": "pretrained/attention_mlp/"},
            },
            {
                "id": "xgboost",
                "model": "xgboost",
                "fit": "pretrained",
                "predict": "in_sample",
                "enabled": True,
                "predict_subset": "all",
                "save_predictions": "reports/estimators/xgboost/predictions.csv",
                "pretrained": {"model_dir": "pretrained/xgboost/"},
            },
        ],
        "adjust": [
            {
                "id": "strength",
                "module": "strength",
                "enabled": True,
                "uses": {"estimators": ["attention"]},
                "save": "reports/adjust/player_strength_panel.csv",
                "params": {
                    "turn_progress_min": 0.2,
                    "weight": "turn_progress",
                    "enforce_winner": True,
                    "civ_adjust": "ols_logit",
                    "block": "auto",
                    "post_cell_normalize": "none",
                    "baseline_experiment": "vanilla-standard-fixed",
                },
            }
        ],
        "analyses": [
            {
                "id": "bt_main",
                "module": "ratings.bradley_terry",
                "enabled": True,
                "uses": {"tables": ["strength"]},
                "params": {"group_by": ["player_type"], "weighted": True, "ref": "Vanilla", "min_games": 5, "bootstrap": None},
            },
            {
                "id": "bt_strategy",
                "module": "ratings.bradley_terry",
                "enabled": True,
                "uses": {"tables": ["strength"]},
                "params": {"group_by": ["player_type", "strategy"], "weighted": True, "ref": "Vanilla", "min_games": 5, "bootstrap": None},
            },
            {
                "id": "pl_main",
                "module": "ratings.plackett_luce",
                "enabled": True,
                "uses": {"tables": ["strength"]},
                "params": {"group_by": ["player_type"], "ref": "Vanilla", "min_games": 5, "bootstrap": None},
            },
            {
                "id": "pl_strategy",
                "module": "ratings.plackett_luce",
                "enabled": True,
                "uses": {"tables": ["strength"]},
                "params": {"group_by": ["player_type", "strategy"], "ref": "Vanilla", "min_games": 5, "bootstrap": None},
            },
            {
                "id": "matchup_strength",
                "module": "ratings.matchups",
                "enabled": True,
                "uses": {"tables": ["strength"]},
                "params": {"mode": "both", "validate_ols": True},
            },
            {
                "id": "matchup_winrates",
                "module": "ratings.outcome_matchups",
                "enabled": True,
                "uses": {"tables": ["panel"]},
                "params": {"include_score_ratio": True},
            },
            {
                "id": "pred_metrics",
                "module": "prediction.evaluate",
                "enabled": True,
                "params": {"metrics": ["roc_auc", "brier_score", "log_loss", "balanced_accuracy"]},
            },
            {
                "id": "pred_compare",
                "module": "prediction.compare",
                "enabled": True,
                "params": {},
            },
            {
                "id": "cal_reliability",
                "module": "calibration.reliability",
                "enabled": True,
                "params": {"n_bins": 10},
            },
            {
                "id": "cal_loss_progress",
                "module": "calibration.loss_by_progress",
                "enabled": True,
                "params": {"n_bins": 20, "metrics": ["brier_score", "log_loss"]},
            },
            {
                "id": "cal_civ_effects",
                "module": "calibration.civ_effects",
                "enabled": True,
                "uses": {"tables": ["strength"]},
                "params": {},
            },
            {
                "id": "cal_cell_baseline",
                "module": "calibration.cell_baseline",
                "enabled": True,
                "uses": {"tables": ["strength"]},
                "params": {},
            },
            {
                "id": "perf_score_ratio",
                "module": "performance.score_ratio",
                "enabled": True,
                "params": {"target": "score_ratio", "predictors": ["player_type", "civilization"]},
            },
            {
                "id": "perf_strength",
                "module": "performance.strength_panel",
                "enabled": True,
                "uses": {"tables": ["strength"]},
                "params": {"metric": "adjusted_strength", "by": "player_type"},
            },
            {
                "id": "perf_experiment_completeness",
                "module": "performance.experiment_completeness",
                "enabled": True,
                "uses": {"tables": ["strength", "tokens"]},
                "params": {},
            },
            {
                "id": "perf_turn_predicted",
                "module": "performance.turn_predicted",
                "enabled": True,
                "params": {"aggregate": "mean", "by": "player_type"},
            },
            {
                "id": "explore_token_costs",
                "module": "exploratory.model_token_costs",
                "enabled": True,
                "uses": {"tables": ["tokens"]},
                "params": {"currency": "usd", "by_player_type": True},
            },
        ],
        "report": {
            "out_dir": "reports/",
            "formats": ["md", "html"],
            "sections": None,
            "title": None,
            "include_disabled": False,
        },
    }


@pytest.fixture
def configs_dir() -> Path:
    return CONFIGS_DIR


@pytest.fixture
def dev_spec() -> dict:
    """The synthetic dev run-spec as a plain dict (fresh copy per test)."""
    return copy.deepcopy(_make_dev_spec())


@pytest.fixture
def write_spec(tmp_path):
    """Write a spec dict under ``tmp_path`` and return its path.

    Returns a callable ``write(spec) -> Path``. The spec is written to
    ``tmp_path/spec_<n>.json`` (never the tracked ``configs/`` tree), and an
    explicit ``catalogs`` block pointing at the tracked sibling catalogs is
    injected (via ``setdefault``) so ``models.json`` / ``experiments.json`` /
    ``paths.json`` resolve without the spec living next to them.
    """
    counter = {"n": 0}

    def _write(spec: dict) -> Path:
        spec.setdefault("catalogs", copy.deepcopy(_CATALOGS))
        path = tmp_path / f"spec_{counter['n']}.json"
        counter["n"] += 1
        path.write_text(json.dumps(spec), encoding="utf-8")
        return path

    return _write
