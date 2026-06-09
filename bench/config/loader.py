"""Load + validate a benchmark run-spec into a :class:`RunConfig`.

Validation follows `configs/benchmark.md` §8. Unknown keys and missing required
fields are hard errors (invariant 1). Per-module analysis ``params`` schemas are
validated by each module as it lands (stages 3-5); stage 0 validates the common
envelope, the estimator/adjust wiring, filters, groupings, bootstrap, the output
root, and the strength params available before the stage implementation lands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .dependencies import resolve_stage_graph
from .errors import ConfigError
from .filters import (
    ensure_filter_narrows,
    resolve_filter_spec,
    validate_filter_object,
)
from .models import OutputConfig, RunConfig, Stage
from . import schema as S


# ── small helpers ───────────────────────────────────────────────────────────
def _require_mapping(obj: Any, where: str) -> dict:
    if not isinstance(obj, dict):
        raise ConfigError(f"{where}: expected an object, got {type(obj).__name__}.")
    return obj


def _check_keys(obj: dict, allowed: Iterable[str], where: str,
                required: Iterable[str] = ()) -> None:
    allowed = set(allowed)
    keys = set(obj.keys())
    unknown = sorted(keys - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}."
        )
    missing = sorted(set(required) - keys)
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}.")


def _check_type(value: Any, types: tuple, where: str) -> None:
    if not isinstance(value, types):
        names = "/".join(t.__name__ for t in types)
        raise ConfigError(f"{where}: expected {names}, got {type(value).__name__}.")


def _check_string_list(value: Any, where: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ConfigError(f"{where}: expected a list of strings.")
    if not allow_empty and not value:
        raise ConfigError(f"{where}: expected a non-empty list of strings.")
    return list(value)


def _check_domain(value: Any, domain: set, where: str) -> None:
    if value not in domain:
        raise ConfigError(f"{where}: must be one of {sorted(domain)}.")


def _read_catalog(cfg: RunConfig, which: str, needed_by: str) -> dict:
    """Lazily read a sibling/override catalog json; error only when needed."""
    path = cfg.catalog_path(which)
    if not path.exists():
        raise ConfigError(
            f"{needed_by} requires the '{which}' catalog, but no file was found "
            f"at {path}. Provide it via catalogs.{which} or place {which}.json "
            f"next to the run-spec."
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"catalog '{which}' at {path} is not valid JSON: {exc}.")


# ── output (§2.1) ───────────────────────────────────────────────────────────
def _parse_output(raw: dict) -> OutputConfig:
    out = raw.get("output")
    if out is None:
        return OutputConfig()
    _require_mapping(out, "output")
    _check_keys(out, S.OUTPUT_KEYS, "output")
    root = out.get("root", S.DEFAULT_OUTPUT_ROOT)
    suffix = out.get("suffix", S.DEFAULT_OUTPUT_SUFFIX)
    _check_type(root, (str,), "output.root")
    _check_type(suffix, (str,), "output.suffix")
    return OutputConfig(root=root, suffix=suffix)


# ── data (§3) ───────────────────────────────────────────────────────────────
def _validate_data(data: dict, presets: dict) -> dict:
    """Validate the `data` block and return the resolved global filter."""
    _require_mapping(data, "data")
    _check_keys(data, S.DATA_KEYS, "data")
    extract = data.get("extract")
    if extract is not None:
        _require_mapping(extract, "data.extract")
        _check_keys(extract, S.EXTRACT_KEYS, "data.extract")
        outputs = extract.get("outputs")
        if outputs is not None:
            outputs = _check_string_list(outputs, "data.extract.outputs")
            bad = [o for o in outputs if o not in S.TABLE_NAMES]
            if bad:
                raise ConfigError(
                    f"data.extract.outputs: unknown table name(s) {bad}. "
                    f"Allowed: {list(S.TABLE_NAMES)}."
                )
        for key in ("enabled", "prune_missing", "force_rebuild"):
            if key in extract:
                _check_type(extract[key], (bool,), f"data.extract.{key}")
        if "runs_dir" in extract:
            _check_type(extract["runs_dir"], (str,), "data.extract.runs_dir")
        max_dbs = extract.get("max_dbs")
        if max_dbs is not None:
            if isinstance(max_dbs, bool) or not isinstance(max_dbs, int) or max_dbs < 1:
                raise ConfigError(
                    "data.extract.max_dbs: must be null or an integer >= 1."
                )
    tables = data.get("tables")
    if tables is not None:
        _require_mapping(tables, "data.tables")
        _check_keys(tables, set(S.TABLE_NAMES), "data.tables")
        for key, value in tables.items():
            _check_type(value, (str,), f"data.tables.{key}")
    return resolve_filter_spec(data.get("filter"), presets, "data.filter")


# ── estimators (§4) ─────────────────────────────────────────────────────────
def _validate_estimator(entry: dict, idx: int, model_ids: set[str]) -> Stage:
    where = f"estimators[{idx}]"
    _require_mapping(entry, where)
    _check_keys(entry, S.ESTIMATOR_KEYS, where, required=("id", "model", "fit"))
    sid = entry["id"]
    where = f"estimators[{idx}] (id={sid!r})"

    fit = entry["fit"]
    _check_domain(fit, S.FIT_VALUES, f"{where}.fit")

    model = entry["model"]
    if model not in model_ids:
        raise ConfigError(
            f"{where}.model: '{model}' is not a prediction_models id in models.json. "
            f"Known: {sorted(model_ids)}."
        )

    predict = entry.get("predict", "in_sample")
    _check_domain(predict, S.PREDICT_VALUES, f"{where}.predict")

    if "needs" in entry:
        _check_string_list(entry["needs"], f"{where}.needs")

    has_train = "train" in entry
    has_pretrained = "pretrained" in entry
    has_tune = "tune" in entry

    # Rule 5: fit matches exactly the one sub-block present.
    if fit == "train":
        if has_pretrained:
            raise ConfigError(f"{where}: fit=train must not carry a 'pretrained' block.")
        if not has_train:
            raise ConfigError(f"{where}: fit=train requires a 'train' block.")
        _require_mapping(entry["train"], f"{where}.train")
        _check_keys(entry["train"], S.TRAIN_KEYS, f"{where}.train")
        if has_tune:
            _require_mapping(entry["tune"], f"{where}.tune")
            _check_keys(entry["tune"], S.TUNE_KEYS, f"{where}.tune")
    else:  # pretrained
        if has_train or has_tune:
            raise ConfigError(
                f"{where}: fit=pretrained must not carry 'train'/'tune' blocks."
            )
        if not has_pretrained:
            raise ConfigError(f"{where}: fit=pretrained requires a 'pretrained' block.")
        _require_mapping(entry["pretrained"], f"{where}.pretrained")
        _check_keys(entry["pretrained"], S.PRETRAINED_KEYS, f"{where}.pretrained",
                    required=("model_dir",))
        if predict == "cross_val":
            raise ConfigError(
                f"{where}: predict=cross_val is only valid with fit=train."
            )

    features = entry.get("features")
    if features is not None:
        _require_mapping(features, f"{where}.features")
        _check_keys(features, S.FEATURES_KEYS, f"{where}.features")
        for key in ("include", "exclude"):
            if key in features:
                _check_string_list(features[key], f"{where}.features.{key}")

    return Stage(id=sid, kind="estimators", enabled=entry.get("enabled", True), raw=entry)


# ── adjust (§5) ─────────────────────────────────────────────────────────────
def _validate_uses(uses: Any, where: str) -> None:
    if uses is None:
        return
    _require_mapping(uses, where)
    _check_keys(uses, S.USES_KEYS, where)
    for key in ("estimators", "tables"):
        if key in uses:
            _check_string_list(uses[key], f"{where}.{key}")


def _validate_strength_params(params: dict, where: str) -> None:
    _check_keys(params, S.STRENGTH_PARAM_KEYS, f"{where}.params")
    enum_checks = {
        "weight": S.STRENGTH_WEIGHT,
        "relative_to": S.STRENGTH_RELATIVE_TO,
        "civ_adjust": S.STRENGTH_CIV_ADJUST,
        "block": S.STRENGTH_BLOCK,
        "post_cell_normalize": S.STRENGTH_POST_CELL_NORMALIZE,
    }
    # `relative_to` is nullable: null (or omission) ⇒ "none" (no leader normalization).
    nullable = {"relative_to"}
    for key, domain in enum_checks.items():
        if key not in params:
            continue
        if params[key] is None and key in nullable:
            continue
        if params[key] not in domain:
            raise ConfigError(
                f"{where}.params.{key}: '{params[key]}' invalid; "
                f"must be one of {sorted(domain)}."
            )
    if "turn_progress_min" in params:
        value = params["turn_progress_min"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ConfigError(
                f"{where}.params.turn_progress_min: must be numeric in [0, 1]."
            )
    if "enforce_winner" in params:
        _check_type(params["enforce_winner"], (bool,), f"{where}.params.enforce_winner")
    # `baseline_experiment` (§5.1): null ⇒ implicit per-experiment path; a value names the
    # explicit baseline source. Experiment ids can be inferred from extracted data, so this is
    # only type-checked here rather than validated against the legacy experiments catalog.
    be = params.get("baseline_experiment")
    if be is not None:
        if not isinstance(be, str):
            raise ConfigError(
                f"{where}.params.baseline_experiment: must be a string experiment id or null."
            )


def _validate_adjust(entry: dict, idx: int) -> Stage:
    where = f"adjust[{idx}]"
    _require_mapping(entry, where)
    _check_keys(entry, S.ADJUST_KEYS, where, required=("id", "module"))
    sid = entry["id"]
    where = f"adjust[{idx}] (id={sid!r})"

    module = entry["module"]
    if module not in S.ADJUST_MODULES:
        raise ConfigError(
            f"{where}.module: '{module}' is not in the adjust registry "
            f"{sorted(S.ADJUST_MODULES)}."
        )

    uses = entry.get("uses")
    if "needs" in entry:
        _check_string_list(entry["needs"], f"{where}.needs")
    _validate_uses(uses, f"{where}.uses")
    est = (uses or {}).get("estimators") or []
    if len(est) != 1:
        raise ConfigError(
            f"{where}.uses.estimators: an adjust stage must declare exactly one "
            f"estimator (got {est})."
        )

    if module == "strength":
        params = entry.get("params")
        if params is not None:
            _require_mapping(params, f"{where}.params")
            _validate_strength_params(params, where)

    return Stage(id=sid, kind="adjust", enabled=entry.get("enabled", True), raw=entry)


# ── analyses (§6) ───────────────────────────────────────────────────────────
def _validate_analysis(
    entry: dict,
    idx: int,
    presets: dict,
    groupings: dict,
    global_filter: dict,
) -> Stage:
    where = f"analyses[{idx}]"
    _require_mapping(entry, where)
    _check_keys(entry, S.ANALYSIS_KEYS, where, required=("id", "module"))
    sid = entry["id"]
    where = f"analyses[{idx}] (id={sid!r})"

    module = entry["module"]
    if module not in S.ANALYSIS_MODULES:
        raise ConfigError(
            f"{where}.module: '{module}' is not in the analysis registry. "
            f"Known modules: {sorted(S.ANALYSIS_MODULES)}."
        )

    if "needs" in entry:
        _check_string_list(entry["needs"], f"{where}.needs")
    _validate_uses(entry.get("uses"), f"{where}.uses")
    stage_filter = resolve_filter_spec(entry.get("filter"), presets, f"{where}.filter")
    if entry.get("filter") is not None:
        ensure_filter_narrows(global_filter, stage_filter, f"{where}.filter")

    params = entry.get("params")
    if params is not None:
        _require_mapping(params, f"{where}.params")
        _validate_analysis_params(module, params, where)
        # Cross-cutting params on fitted ratings (group_by + bootstrap), §6.2.
        if module in ("ratings.bradley_terry", "ratings.plackett_luce"):
            _validate_group_by(params.get("group_by"), groupings, where)
            _validate_bootstrap(params.get("bootstrap"), where)

    return Stage(id=sid, kind="analyses", enabled=entry.get("enabled", True), raw=entry)


def _validate_analysis_params(module: str, params: dict, where: str) -> None:
    """Per-module param-key + enum validation (unknown keys are hard errors, §6)."""
    allowed = S.ANALYSIS_PARAM_KEYS.get(module)
    if allowed is None:
        return  # not a core module with a param schema yet (reserved/optional)
    _check_keys(params, allowed, f"{where}.params")
    # Enum / type checks for the params with a constrained domain.
    if "metrics" in params:
        metrics = _check_string_list(params["metrics"], f"{where}.params.metrics")
        bad = [m for m in metrics if m not in S.PREDICTION_METRICS]
        if bad:
            raise ConfigError(
                f"{where}.params.metrics: unknown metric(s) {bad}. "
                f"Allowed: {sorted(S.PREDICTION_METRICS)}."
            )
    if module == "ratings.matchups" and "mode" in params:
        _check_domain(params["mode"], S.MATCHUPS_MODE, f"{where}.params.mode")
    if module == "performance.turn_predicted" and "aggregate" in params:
        _check_domain(params["aggregate"], S.TURN_PREDICTED_AGGREGATE, f"{where}.params.aggregate")
    if "predictors" in params:
        _check_string_list(params["predictors"], f"{where}.params.predictors", allow_empty=False)
    for key in ("n_bins", "min_games", "min_games_preliminary", "bootstrap_n"):
        if key in params:
            v = params[key]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError(f"{where}.params.{key}: must be an integer >= 1.")
    if "ci_level" in params:
        v = params["ci_level"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1:
            raise ConfigError(f"{where}.params.ci_level: must be a number in (0, 1).")
    for key in ("weighted", "only_llm", "validate_ols", "by_strategist"):
        if key in params:
            _check_type(params[key], (bool,), f"{where}.params.{key}")


def _validate_group_by(group_by: Any, groupings: dict, where: str) -> None:
    if group_by is None:
        return
    _check_string_list(group_by, f"{where}.params.group_by", allow_empty=False)
    # group_by[0] is the base identity (typically player_type); extra dims must
    # name a grouping in top-level `groupings` (§3.2 / rule 9).
    for dim in group_by[1:]:
        if dim not in groupings:
            raise ConfigError(
                f"{where}.params.group_by: '{dim}' is not a defined grouping. "
                f"Defined groupings: {sorted(groupings)}."
            )


def _validate_bootstrap(bootstrap: Any, where: str) -> None:
    if bootstrap is None:
        return
    if not isinstance(bootstrap, dict):
        raise ConfigError(f"{where}.params.bootstrap: expected null or an object.")
    n = bootstrap.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ConfigError(
            f"{where}.params.bootstrap.n: required integer >= 1 (got {n!r})."
        )


# ── groupings (§3.2) ────────────────────────────────────────────────────────
def _validate_groupings(groupings: dict) -> None:
    _require_mapping(groupings, "groupings")
    for name, g in groupings.items():
        where = f"groupings.{name}"
        _require_mapping(g, where)
        _check_keys(g, S.GROUPING_KEYS, where, required=("kind",))
        kind = g["kind"]
        if kind in S.GROUPING_KINDS_RESERVED:
            raise ConfigError(
                f"{where}.kind: '{kind}' is reserved but not implemented yet "
                f"(only {sorted(S.GROUPING_KINDS_IMPLEMENTED)} work today)."
            )
        if kind not in S.GROUPING_KINDS_IMPLEMENTED:
            raise ConfigError(
                f"{where}.kind: '{kind}' is not a known grouping kind "
                f"({sorted(S.GROUPING_KINDS_IMPLEMENTED)})."
            )
        if kind == "argmax":
            cols = g.get("columns")
            _check_string_list(cols, f"{where}.columns", allow_empty=False)
            labels = g.get("labels")
            if labels is not None:
                _check_string_list(labels, f"{where}.labels")
            if labels is not None and len(labels) != len(cols):
                raise ConfigError(
                    f"{where}.labels: must be positional with columns "
                    f"({len(labels)} labels vs {len(cols)} columns)."
                )


# ── report (§7) ─────────────────────────────────────────────────────────────
def _validate_report(report: dict) -> None:
    _require_mapping(report, "report")
    _check_keys(report, S.REPORT_KEYS, "report")
    formats = report.get("formats")
    if formats is not None:
        formats = _check_string_list(formats, "report.formats")
        bad = [f for f in formats if f not in S.REPORT_FORMATS]
        if bad:
            raise ConfigError(
                f"report.formats: unknown format(s) {bad}. "
                f"Allowed: {sorted(S.REPORT_FORMATS)}."
            )


# ── top-level entry point ───────────────────────────────────────────────────
def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}.")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid JSON: {exc}.")

    _require_mapping(raw, "<root>")
    _check_keys(raw, S.TOP_LEVEL_KEYS, "<root>", required=S.TOP_LEVEL_REQUIRED)

    _check_type(raw["name"], (str,), "name")
    seed = raw["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"seed: expected an integer, got {seed!r}.")

    catalogs = raw.get("catalogs") or {}
    if catalogs:
        _require_mapping(catalogs, "catalogs")
        _check_keys(catalogs, S.CATALOG_KEYS, "catalogs")

    presets = raw.get("filters") or {}
    if presets:
        _require_mapping(presets, "filters")
        for name, f in presets.items():
            validate_filter_object(_require_mapping(f, f"filters.{name}"),
                                   f"filters.{name}")

    groupings = raw.get("groupings") or {}
    if groupings:
        _validate_groupings(groupings)

    output = _parse_output(raw)
    global_filter = _validate_data(raw["data"], presets)
    _validate_report(raw["report"])

    cfg = RunConfig(
        name=raw["name"],
        seed=seed,
        config_path=config_path,
        raw=raw,
        output=output,
        description=raw.get("description", ""),
        filters=presets,
        groupings=groupings,
        data=raw["data"],
        report=raw["report"],
        _catalog_overrides=catalogs,
    )

    # Estimators need models.json for membership checks (lazy: only if present).
    estimators_raw = raw.get("estimators")
    if estimators_raw is not None:
        _check_type(estimators_raw, (list,), "estimators")
        models_cat = _read_catalog(cfg, "models", "an 'estimators' stage")
        model_ids = {m["id"] for m in models_cat.get("prediction_models", [])}
        cfg.estimators = [
            _validate_estimator(e, i, model_ids) for i, e in enumerate(estimators_raw)
        ]

    adjust_raw = raw.get("adjust")
    if adjust_raw is not None:
        _check_type(adjust_raw, (list,), "adjust")
        cfg.adjust = [_validate_adjust(a, i) for i, a in enumerate(adjust_raw)]

    analyses_raw = raw["analyses"]
    _check_type(analyses_raw, (list,), "analyses")
    if not analyses_raw:
        raise ConfigError("analyses: at least one analysis is required.")
    cfg.analyses = [
        _validate_analysis(a, i, presets, groupings, global_filter)
        for i, a in enumerate(analyses_raw)
    ]

    cfg._resolved_graph = resolve_stage_graph(cfg)
    return cfg
