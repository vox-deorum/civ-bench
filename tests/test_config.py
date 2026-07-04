"""Config load + validation (benchmark.md §8)."""

from __future__ import annotations

import pytest

from bench.config import ConfigError, OutputConfig, load_config
from bench.config.analysis_metadata import analysis_defaults_to_all_estimators
from bench.pipeline import build_dag


def test_dev_config_loads(configs_dir):
    cfg = load_config(configs_dir / "benchmark.dev.json")
    assert cfg.name == "civbench-dev"
    assert cfg.seed == 42
    assert cfg.output.resolved_root == "reports-dev"
    assert len(cfg.estimators) == 3
    # 12 original core analyses + PL strategy ratings + observed matchup winrates
    # + two controlled-design calibration views + experiment completeness.
    assert len(cfg.analyses) == 17


def test_pretrained_template_loads(configs_dir):
    cfg = load_config(configs_dir / "benchmark.pretrained.template.json")
    assert cfg.output.resolved_root == "reports"
    assert cfg.extract_enabled is False


def test_default_train_template_loads(configs_dir):
    """The default (train-based) template validates — incl. features.include: null."""
    cfg = load_config(configs_dir / "benchmark.template.json")
    assert cfg.output.resolved_root == "reports"
    fits = {e.id: e.raw["fit"] for e in cfg.estimators}
    assert fits == {"score": "train", "attention": "train", "xgboost_cv": "train"}
    # attention carries features.include: null (use coded DEFAULT_FEATURES) + a tune block.
    attention = next(e for e in cfg.estimators if e.id == "attention")
    assert attention.raw["features"]["include"] is None
    assert attention.raw["tune"]["enabled"] is True


def test_cross_template_redirects_and_trains_non_llm(configs_dir):
    """The cross variant: suffix → reports-cross/ and every estimator trains non_llm."""
    cfg = load_config(configs_dir / "benchmark.cross.template.json")
    assert cfg.output.resolved_root == "reports-cross"
    subsets = {e.id: e.raw["train"]["train_subset"] for e in cfg.estimators}
    assert subsets == {"score": "non_llm", "attention": "non_llm", "xgboost_cv": "non_llm"}


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


def test_output_resolve_with_trailing_slash_root():
    out = OutputConfig(root="reports/", suffix="-dev")
    assert out.resolved_root == "reports-dev"
    assert out.resolve("reports/adjust/p.csv") == "reports-dev/adjust/p.csv"


# ── malformed configs must fail loudly (§8) ─────────────────────────────────
def _analysis(spec, stage_id):
    return next(a for a in spec["analyses"] if a["id"] == stage_id)


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
        # Explicit uses still fail on disabled/unknown ids; omitted uses defaults to enabled estimators.
        "uses_disabled_estimator": lambda c: (
            c["estimators"][0].__setitem__("enabled", False),
            _analysis(c, "pred_metrics").setdefault("uses", {}).__setitem__("estimators", ["score"]),
        ),
        "dangling_uses_estimator": lambda c: _analysis(c, "pred_metrics").setdefault("uses", {}).__setitem__(
            "estimators", ["ghost"]
        ),
        "uses_estimator_string": lambda c: _analysis(c, "pred_metrics").setdefault("uses", {}).__setitem__(
            "estimators", "score"
        ),
        "needs_string": lambda c: _analysis(c, "pred_metrics").__setitem__("needs", "score"),
        "report_formats_string": lambda c: c["report"].__setitem__("formats", "html"),
        "unknown_analysis_module": lambda c: c["analyses"][0].__setitem__("module", "ratings.fake"),
        "unknown_adjust_module": lambda c: c["adjust"][0].__setitem__("module", "nope"),
        "bad_strength_enum": lambda c: c["adjust"][0]["params"].__setitem__("block", "wild"),
        "bad_relative_to": lambda c: c["adjust"][0]["params"].__setitem__("relative_to", "wild"),
        "adjust_params_not_object": lambda c: c["adjust"][0].__setitem__("params", "bad"),
        "group_by_undefined": lambda c: c["analyses"][1]["params"].__setitem__("group_by", ["player_type", "nope"]),
        "bootstrap_n_zero": lambda c: c["analyses"][0]["params"].__setitem__("bootstrap", {"n": 0}),
        "unknown_model": lambda c: c["estimators"][0].__setitem__("model", "not_a_model"),
        "fit_block_mismatch": lambda c: c["estimators"][0].__setitem__("fit", "train"),
        "ratings_without_strength": lambda c: c["analyses"][0]["uses"].__setitem__("tables", ["panel"]),
        "duplicate_ids": lambda c: c["analyses"][1].__setitem__("id", "bt_main"),
        "reserved_id": lambda c: c["analyses"][0].__setitem__("id", "extract"),
        "turn_range_min_gt_max": lambda c: c["filters"].__setitem__("late_game", {"turn_range": [300, 100]}),
        "turn_range_bad_type": lambda c: c["filters"].__setitem__("late_game", {"turn_range": [100, "x"]}),
        "min_games_bad_type": lambda c: c["filters"].__setitem__("late_game", {"min_games": "lots"}),
        # stage 1: data.extract scalar + data.tables path types fail loud
        "extract_enabled_not_bool": lambda c: c["data"]["extract"].__setitem__("enabled", "yes"),
        "estimator_enabled_not_bool": lambda c: c["estimators"][0].__setitem__("enabled", "nope"),
        "adjust_enabled_not_bool": lambda c: c["adjust"][0].__setitem__("enabled", "nope"),
        "analysis_enabled_not_bool": lambda c: c["analyses"][0].__setitem__("enabled", "nope"),
        "report_include_disabled_not_bool": lambda c: c["report"].__setitem__("include_disabled", "nope"),
        "report_sections_not_list": lambda c: c["report"].__setitem__("sections", "bt_main"),
        "report_out_dir_not_string": lambda c: c["report"].__setitem__("out_dir", 7),
        "extract_max_dbs_zero": lambda c: c["data"]["extract"].__setitem__("max_dbs", 0),
        "extract_auto_fix_not_bool": lambda c: c["data"]["extract"].__setitem__("auto_fix", "yes"),
        "extract_issues_path_not_string": lambda c: c["data"]["extract"].__setitem__("issues_path", 7),
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


def test_model_token_costs_rejects_unknown_param(dev_spec, write_spec):
    _analysis(dev_spec, "explore_token_costs")["params"]["bogus"] = True
    path = write_spec(dev_spec)

    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_extract_issues_path_override_loads(dev_spec, write_spec):
    """data.extract.issues_path is an accepted (string) override under strict validation."""
    dev_spec["data"]["extract"]["issues_path"] = "runs/custom_issues.csv"
    cfg = load_config(write_spec(dev_spec))
    assert cfg.data["extract"]["issues_path"] == "runs/custom_issues.csv"


def test_extract_auto_fix_loads_and_coerces(dev_spec, write_spec):
    """data.extract.auto_fix is an accepted bool (coerced from a string like the siblings)."""
    dev_spec["data"]["extract"]["auto_fix"] = "FALSE"
    cfg = load_config(write_spec(dev_spec))
    assert cfg.data["extract"]["auto_fix"] is False


@pytest.mark.parametrize("value", ["none", None, "game_leader"])
def test_relative_to_none_and_null_load(dev_spec, write_spec, value):
    """relative_to accepts "none"/null (⇒ no leader normalization) and "game_leader"."""
    dev_spec["adjust"][0]["params"]["relative_to"] = value
    cfg = load_config(write_spec(dev_spec))
    assert cfg is not None


def test_boolean_strings_are_case_insensitive_and_normalized(dev_spec, write_spec):
    dev_spec["data"]["extract"]["enabled"] = "FALSE"
    dev_spec["estimators"][0]["enabled"] = "TrUe"
    dev_spec["adjust"][0]["enabled"] = "TRUE"
    dev_spec["adjust"][0]["params"]["enforce_winner"] = "false"
    dev_spec["analyses"][0]["enabled"] = "tRuE"
    dev_spec["analyses"][0]["params"]["weighted"] = "FALSE"
    dev_spec["analyses"][0]["params"]["only_llm"] = "True"
    dev_spec["analyses"][0]["params"]["bootstrap"] = {
        "n": 2,
        "stratified": "FALSE",
    }
    _analysis(dev_spec, "matchup_winrates")["params"]["include_score_ratio"] = "FALSE"
    _analysis(dev_spec, "explore_token_costs")["params"]["by_player_type"] = "FALSE"
    _analysis(dev_spec, "explore_token_costs")["params"]["by_strategist"] = "TRUE"
    dev_spec["data"]["filter"] = {"only_llm": "FaLsE"}
    dev_spec["report"]["include_disabled"] = "FALSE"

    cfg = load_config(write_spec(dev_spec))

    assert cfg.extract_enabled is False
    assert cfg.estimators[0].enabled is True
    assert cfg.adjust[0].enabled is True
    assert cfg.adjust[0].raw["params"]["enforce_winner"] is False
    assert cfg.analyses[0].enabled is True
    assert cfg.analyses[0].raw["params"]["weighted"] is False
    assert cfg.analyses[0].raw["params"]["only_llm"] is True
    assert cfg.analyses[0].raw["params"]["bootstrap"]["stratified"] is False
    winrates = next(a for a in cfg.analyses if a.id == "matchup_winrates")
    assert winrates.raw["params"]["include_score_ratio"] is False
    token_costs = next(a for a in cfg.analyses if a.id == "explore_token_costs")
    assert token_costs.raw["params"]["by_player_type"] is False
    assert token_costs.raw["params"]["by_strategist"] is True
    assert cfg.data["filter"]["only_llm"] is False
    assert cfg.report["include_disabled"] is False


def test_estimator_needs_validated(dev_spec, write_spec):
    # Estimators carry no `uses`, so `needs` is their only edge — a dangling ref
    # must fail loudly at load (not be silently dropped at DAG build).
    dev_spec["estimators"][0]["needs"] = ["ghost"]
    path = write_spec(dev_spec)
    with pytest.raises(ConfigError, match="needs unknown id 'ghost'"):
        load_config(path)


def test_estimator_consuming_analyses_default_to_all_enabled(dev_spec, write_spec):
    stage = _analysis(dev_spec, "pred_metrics")
    stage.pop("uses", None)
    path = write_spec(dev_spec)
    cfg = load_config(path)
    dag = build_dag(cfg)

    enabled_estimators = {s.id for s in cfg.estimators if s.enabled}
    assert dag.nodes["pred_metrics"].deps >= enabled_estimators


def test_empty_analysis_estimator_uses_defaults_to_all_enabled(dev_spec, write_spec):
    _analysis(dev_spec, "pred_metrics")["uses"] = {"estimators": []}
    path = write_spec(dev_spec)
    cfg = load_config(path)
    dag = build_dag(cfg)

    enabled_estimators = {s.id for s in cfg.estimators if s.enabled}
    assert dag.nodes["pred_metrics"].deps >= enabled_estimators


def test_analysis_default_all_estimator_metadata_lives_on_modules():
    assert analysis_defaults_to_all_estimators("prediction.evaluate")
    assert analysis_defaults_to_all_estimators("calibration.loss_by_progress")
    assert not analysis_defaults_to_all_estimators("performance.score_ratio")


def test_cycle_detected_at_load(dev_spec, write_spec):
    # bt_main(0) needs pl_main(2); pl_main(2) needs bt_main(0) → cycle
    dev_spec["analyses"][0]["needs"] = ["pl_main"]
    dev_spec["analyses"][2]["needs"] = ["bt_main"]
    path = write_spec(dev_spec)
    with pytest.raises(ConfigError, match="cycle"):
        load_config(path)


def test_strength_table_id_need_not_be_literally_strength(dev_spec, write_spec):
    # The strength stage id doubles as the table name (benchmark.md §5); a
    # ratings analysis must reference *that id*, not the literal "strength".
    strength = next(s for s in dev_spec["adjust"] if s.get("module") == "strength")
    old_id = strength["id"]
    strength["id"] = "strength_main"
    for a in dev_spec["analyses"]:
        tables = a.get("uses", {}).get("tables")
        if tables and old_id in tables:
            a["uses"]["tables"] = ["strength_main" if t == old_id else t for t in tables]

    cfg = load_config(write_spec(dev_spec))
    dag = build_dag(cfg)
    assert "strength_main" in dag.nodes
    # every enabled strength-based rating analysis now depends on the renamed stage
    for a in cfg.analyses:
        if a.enabled and a.module in {
            "ratings.bradley_terry", "ratings.plackett_luce", "ratings.matchups",
        }:
            assert "strength_main" in dag.nodes[a.id].deps


def test_ratings_referencing_wrong_table_id_still_fails(dev_spec, write_spec):
    # Renaming the strength stage without updating the ratings reference must fail.
    strength = next(s for s in dev_spec["adjust"] if s.get("module") == "strength")
    strength["id"] = "strength_main"
    path = write_spec(dev_spec)
    with pytest.raises(ConfigError):
        load_config(path)


def test_stage_filter_list_can_narrow_global(dev_spec, write_spec):
    dev_spec["data"]["filter"] = {"turn_range": [100, None]}
    dev_spec["analyses"][4]["filter"] = ["late_game", {"players": ["Sonnet-4.5"]}]
    path = write_spec(dev_spec)
    assert load_config(path).analyses[4].raw["filter"][0] == "late_game"
