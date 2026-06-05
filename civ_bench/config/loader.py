"""Load + validate a benchmark run-spec into a :class:`RunConfig`.

Validation follows `configs/benchmark.md` §8. Unknown keys and missing required
fields are hard errors (invariant 1). Per-module analysis ``params`` schemas are
validated by each module as it lands (stages 3-5); stage 0 validates the common
envelope, the estimator/adjust wiring, filters, groupings, bootstrap, the output
root, and the strength controlled-design enums.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .errors import ConfigError
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


# ── filters (§3.1) ──────────────────────────────────────────────────────────
def _validate_filter_object(obj: dict, where: str) -> None:
    _check_keys(obj, S.FILTER_KEYS, where)
    tr = obj.get("turn_range")
    if tr is not None:
        if not isinstance(tr, (list, tuple)) or len(tr) != 2:
            raise ConfigError(
                f"{where}.turn_range: expected [min, max] (either bound nullable)."
            )
        lo, hi = tr
        if lo is not None and hi is not None and lo > hi:
            raise ConfigError(
                f"{where}.turn_range: min ({lo}) must be <= max ({hi})."
            )


def _validate_filter_ref(spec: Any, presets: dict, where: str) -> None:
    """A filter value may be an inline object, a preset name, or a list of both."""
    if spec is None:
        return
    if isinstance(spec, str):
        if spec not in presets:
            raise ConfigError(
                f"{where}: references undefined filter preset '{spec}'. "
                f"Defined presets: {sorted(presets)}."
            )
    elif isinstance(spec, dict):
        _validate_filter_object(spec, where)
    elif isinstance(spec, list):
        for i, item in enumerate(spec):
            _validate_filter_ref(item, presets, f"{where}[{i}]")
    else:
        raise ConfigError(
            f"{where}: a filter must be a preset name, an inline object, or a "
            f"list of those (got {type(spec).__name__})."
        )


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
def _validate_data(data: dict, presets: dict) -> None:
    _require_mapping(data, "data")
    _check_keys(data, S.DATA_KEYS, "data")
    extract = data.get("extract")
    if extract is not None:
        _require_mapping(extract, "data.extract")
        _check_keys(extract, S.EXTRACT_KEYS, "data.extract")
        outputs = extract.get("outputs")
        if outputs is not None:
            bad = [o for o in outputs if o not in S.TABLE_NAMES]
            if bad:
                raise ConfigError(
                    f"data.extract.outputs: unknown table name(s) {bad}. "
                    f"Allowed: {list(S.TABLE_NAMES)}."
                )
    tables = data.get("tables")
    if tables is not None:
        _require_mapping(tables, "data.tables")
        _check_keys(tables, set(S.TABLE_NAMES), "data.tables")
    _validate_filter_ref(data.get("filter"), presets, "data.filter")


# ── estimators (§4) ─────────────────────────────────────────────────────────
def _validate_estimator(entry: dict, idx: int, model_ids: set[str]) -> Stage:
    where = f"estimators[{idx}]"
    _require_mapping(entry, where)
    _check_keys(entry, S.ESTIMATOR_KEYS, where, required=("id", "model", "fit"))
    sid = entry["id"]
    where = f"estimators[{idx}] (id={sid!r})"

    fit = entry["fit"]
    if fit not in S.FIT_VALUES:
        raise ConfigError(f"{where}.fit: must be one of {sorted(S.FIT_VALUES)}.")

    model = entry["model"]
    if model not in model_ids:
        raise ConfigError(
            f"{where}.model: '{model}' is not a prediction_models id in models.json. "
            f"Known: {sorted(model_ids)}."
        )

    predict = entry.get("predict", "in_sample")
    if predict not in S.PREDICT_VALUES:
        raise ConfigError(
            f"{where}.predict: must be one of {sorted(S.PREDICT_VALUES)}."
        )

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

    return Stage(id=sid, kind="estimators", enabled=entry.get("enabled", True), raw=entry)


# ── adjust (§5) ─────────────────────────────────────────────────────────────
def _validate_uses(uses: Any, where: str) -> None:
    if uses is None:
        return
    _require_mapping(uses, where)
    _check_keys(uses, S.USES_KEYS, where)


def _validate_strength_params(params: dict, where: str) -> None:
    enum_checks = {
        "block": S.STRENGTH_BLOCK,
        "baseline_source": S.STRENGTH_BASELINE_SOURCE,
        "post_cell_normalize": S.STRENGTH_POST_CELL_NORMALIZE,
        "engine": S.STRENGTH_ENGINE,
    }
    for key, domain in enum_checks.items():
        if key in params and params[key] not in domain:
            raise ConfigError(
                f"{where}.params.{key}: '{params[key]}' invalid; "
                f"must be one of {sorted(domain)}."
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
def _validate_analysis(entry: dict, idx: int, presets: dict, groupings: dict) -> Stage:
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

    _validate_uses(entry.get("uses"), f"{where}.uses")
    _validate_filter_ref(entry.get("filter"), presets, f"{where}.filter")

    params = entry.get("params")
    if params is not None:
        _require_mapping(params, f"{where}.params")
        # Cross-cutting params on fitted ratings (group_by + bootstrap), §6.2.
        if module in ("ratings.bradley_terry", "ratings.plackett_luce"):
            _validate_group_by(params.get("group_by"), groupings, where)
            _validate_bootstrap(params.get("bootstrap"), where)

    return Stage(id=sid, kind="analyses", enabled=entry.get("enabled", True), raw=entry)


def _validate_group_by(group_by: Any, groupings: dict, where: str) -> None:
    if group_by is None:
        return
    if not isinstance(group_by, list) or not group_by:
        raise ConfigError(f"{where}.params.group_by: expected a non-empty list.")
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
            if not isinstance(cols, list) or not cols:
                raise ConfigError(f"{where}.columns: argmax requires a non-empty list.")
            labels = g.get("labels")
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
        bad = [f for f in formats if f not in S.REPORT_FORMATS]
        if bad:
            raise ConfigError(
                f"report.formats: unknown format(s) {bad}. "
                f"Allowed: {sorted(S.REPORT_FORMATS)}."
            )


# ── id graph rules (§8 rules 3 & 7) ─────────────────────────────────────────
def _validate_id_graph(cfg: RunConfig) -> None:
    estimator_ids = {s.id for s in cfg.estimators}
    adjust_ids = {s.id for s in cfg.adjust}
    analysis_ids = {s.id for s in cfg.analyses}

    # Rule 3: unique ids across estimators + adjust + analyses.
    seen: dict[str, str] = {}
    for stage in cfg.all_stages():
        if stage.id in seen:
            raise ConfigError(
                f"duplicate stage id '{stage.id}' (used by both {seen[stage.id]} "
                f"and {stage.kind})."
            )
        seen[stage.id] = stage.kind
    if "extract" in seen:
        raise ConfigError("stage id 'extract' is reserved for the extract stage.")
    if "report" in seen:
        raise ConfigError("stage id 'report' is reserved for the report stage.")

    enabled_ids = {s.id for s in cfg.all_stages() if s.enabled}
    enabled_adjust_ids = {s.id for s in cfg.adjust if s.enabled}
    table_keys = set(cfg.table_names)

    def _check_needs(stage: Stage) -> None:
        for dep in stage.needs:
            if dep == "extract":
                continue
            if dep not in seen:
                raise ConfigError(
                    f"stage '{stage.id}' needs unknown id '{dep}'."
                )
            if dep not in enabled_ids:
                raise ConfigError(
                    f"stage '{stage.id}' needs disabled stage '{dep}'."
                )

    # Adjust wiring (rule 7): estimator refs exist + enabled.
    for stage in cfg.adjust:
        _check_needs(stage)
        for est in stage.uses_estimators:
            if est not in estimator_ids:
                raise ConfigError(
                    f"adjust '{stage.id}' uses unknown estimator '{est}'."
                )
            if est not in enabled_ids:
                raise ConfigError(
                    f"adjust '{stage.id}' uses disabled estimator '{est}'."
                )

    has_strength_table = any(
        s.enabled and s.raw.get("module") == "strength" for s in cfg.adjust
    )

    for stage in cfg.analyses:
        _check_needs(stage)
        for est in stage.uses_estimators:
            if est not in estimator_ids:
                raise ConfigError(
                    f"analysis '{stage.id}' uses unknown estimator '{est}'."
                )
            if est not in enabled_ids:
                raise ConfigError(
                    f"analysis '{stage.id}' uses disabled estimator '{est}'."
                )
        for tbl in stage.uses_tables:
            if tbl in table_keys:
                continue
            if tbl in adjust_ids:
                if tbl not in enabled_adjust_ids:
                    raise ConfigError(
                        f"analysis '{stage.id}' uses table '{tbl}' from a disabled "
                        f"adjust stage."
                    )
                continue
            raise ConfigError(
                f"analysis '{stage.id}' uses table '{tbl}' which is neither a "
                f"canonical table {sorted(table_keys)} nor an adjust stage id "
                f"{sorted(adjust_ids)}."
            )
        # Rule 7: any ratings.* analysis must rate a `strength` table.
        if stage.enabled and stage.module and stage.module.startswith(S.RATINGS_PREFIX):
            if "strength" not in stage.uses_tables:
                raise ConfigError(
                    f"ratings analysis '{stage.id}' must reference a strength table "
                    f"via uses.tables (it rates adjusted_strength, not panel_data)."
                )
            if not has_strength_table:
                raise ConfigError(
                    f"ratings analysis '{stage.id}' requires an enabled adjust "
                    f"'strength' stage to produce the strength table; none found."
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
            _validate_filter_object(_require_mapping(f, f"filters.{name}"),
                                    f"filters.{name}")

    groupings = raw.get("groupings") or {}
    if groupings:
        _validate_groupings(groupings)

    output = _parse_output(raw)
    _validate_data(raw["data"], presets)
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
        _validate_analysis(a, i, presets, groupings) for i, a in enumerate(analyses_raw)
    ]

    _validate_id_graph(cfg)

    # Rule 4 (acyclic after edge resolution): resolving the DAG raises ConfigError
    # on a cycle. Lazy import keeps the config layer free of a pipeline dependency
    # at module-load time (pipeline imports config).
    from ..pipeline.dag import build_dag

    build_dag(cfg)
    return cfg
