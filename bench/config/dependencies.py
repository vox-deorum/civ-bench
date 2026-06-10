"""Resolve and validate pipeline dependencies from a :class:`RunConfig`."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import schema as S
from .analysis_metadata import analysis_defaults_to_all_estimators
from .errors import ConfigError
from .models import RunConfig, Stage


EXTRACT_ID = "extract"
REPORT_ID = "report"


@dataclass
class ResolvedNode:
    id: str
    kind: str
    deps: set[str] = field(default_factory=set)
    raw: dict = field(default_factory=dict)


@dataclass
class ResolvedGraph:
    nodes: dict[str, ResolvedNode]
    order: list[str]


def resolve_stage_graph(cfg: RunConfig) -> ResolvedGraph:
    """Validate stage refs and return enabled nodes with resolved dependencies."""
    estimator_ids = {s.id for s in cfg.estimators}
    adjust_ids = {s.id for s in cfg.adjust}
    table_keys = set(cfg.table_names)

    seen = _validate_unique_ids(cfg)
    enabled_ids = {s.id for s in cfg.all_stages() if s.enabled}
    enabled_adjust_ids = {s.id for s in cfg.adjust if s.enabled}

    for stage in cfg.all_stages():
        _check_needs(stage, seen, enabled_ids)

    for stage in cfg.adjust:
        for est in stage.uses_estimators:
            _check_estimator_ref(stage, est, estimator_ids, enabled_ids, "adjust")

    # Strength tables are named by the *id* of any enabled strength-module
    # adjust stage (the id doubles as the table name — benchmark.md §5), not
    # the literal string "strength". A ratings analysis must reference one of
    # these ids via uses.tables.
    strength_table_ids = {
        s.id for s in cfg.adjust
        if s.enabled and s.raw.get("module") == "strength"
    }
    for stage in cfg.analyses:
        for est in stage.uses_estimators:
            _check_estimator_ref(stage, est, estimator_ids, enabled_ids, "analysis")
        if (
            stage.enabled
            and analysis_defaults_to_all_estimators(stage.module)
            and not stage.uses_estimators
            and not any(s.enabled for s in cfg.estimators)
        ):
            raise ConfigError(
                f"analysis '{stage.id}' defaults to all enabled estimators, "
                f"but this run has none."
            )
        for tbl in stage.uses_tables:
            _check_table_ref(stage, tbl, table_keys, adjust_ids, enabled_adjust_ids)
        if stage.enabled and stage.module and stage.module.startswith(S.RATINGS_PREFIX):
            _check_ratings_strength_ref(stage, strength_table_ids)

    nodes = _build_enabled_nodes(cfg, table_keys)
    order = _topo_sort(nodes)
    return ResolvedGraph(nodes=nodes, order=order)


def _validate_unique_ids(cfg: RunConfig) -> dict[str, str]:
    seen: dict[str, str] = {}
    for stage in cfg.all_stages():
        if stage.id in seen:
            raise ConfigError(
                f"duplicate stage id '{stage.id}' (used by both {seen[stage.id]} "
                f"and {stage.kind})."
            )
        seen[stage.id] = stage.kind
    if EXTRACT_ID in seen:
        raise ConfigError("stage id 'extract' is reserved for the extract stage.")
    if REPORT_ID in seen:
        raise ConfigError("stage id 'report' is reserved for the report stage.")
    return seen


def _check_needs(stage: Stage, seen: dict[str, str], enabled_ids: set[str]) -> None:
    for dep in stage.needs:
        if dep == EXTRACT_ID:
            continue
        if dep not in seen:
            raise ConfigError(f"stage '{stage.id}' needs unknown id '{dep}'.")
        if dep not in enabled_ids:
            raise ConfigError(f"stage '{stage.id}' needs disabled stage '{dep}'.")


def _check_estimator_ref(
    stage: Stage,
    est: str,
    estimator_ids: set[str],
    enabled_ids: set[str],
    kind_label: str,
) -> None:
    if est not in estimator_ids:
        raise ConfigError(f"{kind_label} '{stage.id}' uses unknown estimator '{est}'.")
    if est not in enabled_ids:
        raise ConfigError(f"{kind_label} '{stage.id}' uses disabled estimator '{est}'.")


def _check_table_ref(
    stage: Stage,
    table: str,
    table_keys: set[str],
    adjust_ids: set[str],
    enabled_adjust_ids: set[str],
) -> None:
    if table in table_keys:
        return
    if table in adjust_ids:
        if table not in enabled_adjust_ids:
            raise ConfigError(
                f"analysis '{stage.id}' uses table '{table}' from a disabled "
                f"adjust stage."
            )
        return
    raise ConfigError(
        f"analysis '{stage.id}' uses table '{table}' which is neither a "
        f"canonical table {sorted(table_keys)} nor an adjust stage id "
        f"{sorted(adjust_ids)}."
    )


def _check_ratings_strength_ref(stage: Stage, strength_table_ids: set[str]) -> None:
    if not strength_table_ids:
        raise ConfigError(
            f"ratings analysis '{stage.id}' requires an enabled adjust "
            f"strength-module stage to produce a strength table; none found."
        )
    if not strength_table_ids.intersection(stage.uses_tables):
        raise ConfigError(
            f"ratings analysis '{stage.id}' must reference a strength table "
            f"via uses.tables (one of {sorted(strength_table_ids)}); it rates "
            f"adjusted_strength, not panel_data."
        )


def _build_enabled_nodes(cfg: RunConfig, table_keys: set[str]) -> dict[str, ResolvedNode]:
    nodes: dict[str, ResolvedNode] = {}
    extract_enabled = cfg.extract_enabled
    if extract_enabled:
        nodes[EXTRACT_ID] = ResolvedNode(
            id=EXTRACT_ID, kind="extract", raw=cfg.data.get("extract", {})
        )

    enabled_adjust_ids = {s.id for s in cfg.adjust if s.enabled}

    for stage in cfg.estimators:
        if not stage.enabled:
            continue
        node = ResolvedNode(id=stage.id, kind="estimators", raw=stage.raw)
        if extract_enabled:
            node.deps.add(EXTRACT_ID)
        node.deps.update(_resolve_needs(stage.needs, extract_enabled))
        nodes[stage.id] = node

    for stage in cfg.adjust:
        if not stage.enabled:
            continue
        node = ResolvedNode(id=stage.id, kind="adjust", raw=stage.raw)
        if extract_enabled:
            node.deps.add(EXTRACT_ID)
        node.deps.update(stage.uses_estimators)
        node.deps.update(_resolve_needs(stage.needs, extract_enabled))
        nodes[stage.id] = node

    for stage in cfg.analyses:
        if not stage.enabled:
            continue
        node = ResolvedNode(id=stage.id, kind="analyses", raw=stage.raw)
        node.deps.update(_analysis_estimator_deps(stage, cfg))
        for tbl in stage.uses_tables:
            if tbl in enabled_adjust_ids:
                node.deps.add(tbl)
            elif tbl in table_keys and extract_enabled:
                node.deps.add(EXTRACT_ID)
        node.deps.update(_resolve_needs(stage.needs, extract_enabled))
        if extract_enabled and not node.deps:
            node.deps.add(EXTRACT_ID)
        nodes[stage.id] = node

    report_node = ResolvedNode(id=REPORT_ID, kind="report", raw=cfg.report)
    report_node.deps.update(s.id for s in cfg.analyses if s.enabled)
    nodes[REPORT_ID] = report_node
    return nodes


def _analysis_estimator_deps(stage: Stage, cfg: RunConfig) -> list[str]:
    explicit = stage.uses_estimators
    if explicit:
        return explicit
    if analysis_defaults_to_all_estimators(stage.module):
        return [s.id for s in cfg.estimators if s.enabled]
    return []


def _resolve_needs(needs: list[str], extract_enabled: bool) -> set[str]:
    out: set[str] = set()
    for dep in needs:
        if dep == EXTRACT_ID and not extract_enabled:
            continue
        out.add(dep)
    return out


def _topo_sort(nodes: dict[str, ResolvedNode]) -> list[str]:
    indeg = {nid: len(node.deps) for nid, node in nodes.items()}
    dependents: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        for dep in node.deps:
            if dep not in dependents:
                # Validation upstream guarantees every dep is an enabled node;
                # if that invariant ever breaks, fail loud with a precise message
                # instead of a bare KeyError or a misleading "cycle" report.
                raise ConfigError(
                    f"stage '{nid}' depends on '{dep}', which is not an enabled "
                    f"stage in the resolved graph."
                )
            dependents[dep].append(nid)

    ready = sorted(nid for nid, degree in indeg.items() if degree == 0)
    order: list[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for dependent in sorted(dependents[nid]):
            indeg[dependent] -= 1
            if indeg[dependent] == 0:
                ready.append(dependent)
                ready.sort()

    ordered = set(order)
    if len(order) != len(nodes):
        cyclic = sorted(nid for nid in nodes if nid not in ordered)
        raise ConfigError(
            f"pipeline DAG has a cycle among stages: {cyclic}. "
            f"Check `needs`/`uses` references."
        )
    return order
