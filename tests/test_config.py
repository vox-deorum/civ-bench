"""Config load + validation (benchmark.md §8)."""

from __future__ import annotations

import pytest

from bench.config import ConfigError, OutputConfig, load_config


def test_dev_config_loads(configs_dir):
    cfg = load_config(configs_dir / "benchmark.dev.json")
    assert cfg.name == "civbench-dev"
    assert cfg.seed == 42
    assert cfg.output.resolved_root == "reports-dev"
    assert len(cfg.estimators) == 3
    # 12 original core analyses + the two controlled-design calibration views
    # (calibration.civ_effects, calibration.cell_baseline) added in stage 4.
    assert len(cfg.analyses) == 14


def test_pretrained_template_loads(configs_dir):
    cfg = load_config(configs_dir / "benchmark.pretrained.template.json")
    assert cfg.output.resolved_root == "reports"
    assert cfg.extract_enabled is False


# ── output root resolution (§2.1) ───────────────────────────────────────────
def test_output_resolve_default_passthrough():
    out = OutputConfig(root="reports", suffix="")
    assert out.resolve("reports/estimators/x/predictions.csv") == "reports/estimators/x/predictions.csv"


def test_output_resolve_with_suffix():
    out = OutputConfig(root="reports", suffix="-cross")
    assert out.resolve("reports/adjust/p.csv") == "reports-cross/adjust/p.csv"
    assert out.resolve("reports") == "reports-cross"
    # paths not under the base root are untouched
    assert out.resolve("/abs/elsewhere.csv") == "/abs/elsewhere.csv"


# ── malformed configs must fail loudly (§8) ─────────────────────────────────
def _mutations():
    return {
        "unknown_top_key": lambda c: c.__setitem__("bogus", 1),
        "missing_required": lambda c: c.pop("seed"),
        "unknown_nested_key": lambda c: c["data"]["extract"].__setitem__("typo", 1),
        "dangling_needs": lambda c: c["analyses"][0].__setitem__("needs", ["nope"]),
        "estimator_dangling_needs": lambda c: c["estimators"][0].__setitem__("needs", ["ghost"]),
        # rule 3: an estimator may not `needs` a disabled stage (xgboost off, score needs it)
        "estimator_needs_disabled": lambda c: (
            c["estimators"][2].__setitem__("enabled", False),
            c["estimators"][0].__setitem__("needs", ["xgboost"]),
        ),
        # disabled-target reference via `uses`: pred_metrics uses the now-disabled score estimator
        "uses_disabled_estimator": lambda c: c["estimators"][0].__setitem__("enabled", False),
        "dangling_uses_estimator": lambda c: c["analyses"][4]["uses"]["estimators"].append("ghost"),
        "uses_estimator_string": lambda c: c["analyses"][4]["uses"].__setitem__("estimators", "score"),
        "needs_string": lambda c: c["analyses"][4].__setitem__("needs", "score"),
        "report_formats_string": lambda c: c["report"].__setitem__("formats", "html"),
        "unknown_analysis_module": lambda c: c["analyses"][0].__setitem__("module", "ratings.fake"),
        "unknown_adjust_module": lambda c: c["adjust"][0].__setitem__("module", "nope"),
        "bad_strength_enum": lambda c: c["adjust"][0]["params"].__setitem__("block", "wild"),
        "adjust_params_not_object": lambda c: c["adjust"][0].__setitem__("params", "bad"),
        "group_by_undefined": lambda c: c["analyses"][1]["params"].__setitem__("group_by", ["player_type", "nope"]),
        "bootstrap_n_zero": lambda c: c["analyses"][0]["params"].__setitem__("bootstrap", {"n": 0}),
        "unknown_model": lambda c: c["estimators"][0].__setitem__("model", "not_a_model"),
        "fit_block_mismatch": lambda c: c["estimators"][0].__setitem__("fit", "train"),
        "ratings_without_strength": lambda c: c["analyses"][0]["uses"].__setitem__("tables", ["panel"]),
        "duplicate_ids": lambda c: c["analyses"][1].__setitem__("id", "bt_main"),
        "reserved_id": lambda c: c["analyses"][0].__setitem__("id", "extract"),
        "turn_range_min_gt_max": lambda c: c["filters"].__setitem__("late_game", {"turn_range": [300, 100]}),
        # stage 1: data.extract scalar + data.tables path types fail loud
        "extract_enabled_not_bool": lambda c: c["data"]["extract"].__setitem__("enabled", "yes"),
        "extract_max_dbs_zero": lambda c: c["data"]["extract"].__setitem__("max_dbs", 0),
        "tables_path_not_string": lambda c: c["data"]["tables"].__setitem__("turns", 7),
        "stage_filter_widens_global": lambda c: (
            c["data"].__setitem__("filter", {"experiments": ["global_only"]}),
            c["analyses"][4].__setitem__("filter", {"experiments": ["other"]}),
        ),
    }


@pytest.mark.parametrize("name", list(_mutations().keys()))
def test_malformed_config_raises(name, dev_spec, write_spec):
    _mutations()[name](dev_spec)
    path = write_spec(dev_spec)
    with pytest.raises(ConfigError):
        load_config(path)


def test_estimator_needs_validated(dev_spec, write_spec):
    # Estimators carry no `uses`, so `needs` is their only edge — a dangling ref
    # must fail loudly at load (not be silently dropped at DAG build).
    dev_spec["estimators"][0]["needs"] = ["ghost"]
    path = write_spec(dev_spec)
    with pytest.raises(ConfigError, match="needs unknown id 'ghost'"):
        load_config(path)


def test_cycle_detected_at_load(dev_spec, write_spec):
    # bt_main(0) needs pl_main(2); pl_main(2) needs bt_main(0) → cycle
    dev_spec["analyses"][0]["needs"] = ["pl_main"]
    dev_spec["analyses"][2]["needs"] = ["bt_main"]
    path = write_spec(dev_spec)
    with pytest.raises(ConfigError, match="cycle"):
        load_config(path)


def test_stage_filter_list_can_narrow_global(dev_spec, write_spec):
    dev_spec["data"]["filter"] = {"turn_range": [100, None]}
    dev_spec["analyses"][4]["filter"] = ["late_game", {"players": ["Sonnet-4.5"]}]
    path = write_spec(dev_spec)
    assert load_config(path).analyses[4].raw["filter"][0] == "late_game"
