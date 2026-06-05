"""Build the stage DAG from a validated :class:`RunConfig`, topo-sort it, and
render it.

Edges come from three places (benchmark.md §1), all resolved into one
topological order before anything runs:

1. **Kind ordering (implicit)** — extract → estimators → adjust → analyses →
   report. Concretely: every estimator depends on ``extract`` (it re-infers on
   the canonical turns table); ``report`` depends on every enabled analysis.
2. **`needs` (explicit)** — extra ordering the harness cannot infer.
3. **`uses` (referential)** — an estimator id or a table name in a stage's
   ``uses`` block creates an edge automatically (to the producing estimator /
   adjust stage, or to extract for a canonical table).

Disabled stages (``enabled: false``) are dropped from the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import RunConfig
from ..config.errors import ConfigError


EXTRACT_ID = "extract"
REPORT_ID = "report"


@dataclass
class DagNode:
    id: str
    kind: str
    deps: set[str] = field(default_factory=set)  # ids this node must run after
    raw: dict = field(default_factory=dict)


@dataclass
class Dag:
    nodes: dict[str, DagNode]
    order: list[str]  # topologically sorted node ids

    def __iter__(self):
        for nid in self.order:
            yield self.nodes[nid]


def build_dag(cfg: RunConfig) -> Dag:
    """Resolve the run-spec into a topologically sorted DAG of enabled stages."""
    nodes: dict[str, DagNode] = {}

    extract_enabled = cfg.extract_enabled
    if extract_enabled:
        nodes[EXTRACT_ID] = DagNode(id=EXTRACT_ID, kind="extract",
                                    raw=cfg.data.get("extract", {}))

    enabled_estimators = [s for s in cfg.estimators if s.enabled]
    enabled_adjust = [s for s in cfg.adjust if s.enabled]
    enabled_analyses = [s for s in cfg.analyses if s.enabled]

    adjust_ids = {s.id for s in enabled_adjust}
    table_keys = set(cfg.table_names)

    # estimators
    for s in enabled_estimators:
        node = DagNode(id=s.id, kind="estimators", raw=s.raw)
        if extract_enabled:
            node.deps.add(EXTRACT_ID)
        node.deps.update(d for d in s.needs if d != EXTRACT_ID or extract_enabled)
        nodes[s.id] = node

    # adjust
    for s in enabled_adjust:
        node = DagNode(id=s.id, kind="adjust", raw=s.raw)
        if extract_enabled:
            node.deps.add(EXTRACT_ID)
        node.deps.update(s.uses_estimators)
        node.deps.update(_resolve_needs(s.needs, extract_enabled))
        nodes[s.id] = node

    # analyses
    for s in enabled_analyses:
        node = DagNode(id=s.id, kind="analyses", raw=s.raw)
        node.deps.update(s.uses_estimators)
        for tbl in s.uses_tables:
            if tbl in adjust_ids:
                node.deps.add(tbl)
            elif tbl in table_keys and extract_enabled:
                node.deps.add(EXTRACT_ID)
        node.deps.update(_resolve_needs(s.needs, extract_enabled))
        # An analysis with no explicit producer still reads canonical CSVs.
        if extract_enabled and not node.deps:
            node.deps.add(EXTRACT_ID)
        nodes[s.id] = node

    # report — depends on every enabled analysis (it walks their results)
    report_node = DagNode(id=REPORT_ID, kind="report", raw=cfg.report)
    report_node.deps.update(s.id for s in enabled_analyses)
    nodes[REPORT_ID] = report_node

    # Drop dangling deps that point at disabled/absent nodes only if they were
    # never validated away. (Validation already guarantees needs/uses target
    # enabled ids; extract may be absent when disabled.)
    for node in nodes.values():
        node.deps = {d for d in node.deps if d in nodes}

    order = _topo_sort(nodes)
    return Dag(nodes=nodes, order=order)


def _resolve_needs(needs: list[str], extract_enabled: bool) -> set[str]:
    out: set[str] = set()
    for d in needs:
        if d == EXTRACT_ID and not extract_enabled:
            continue
        out.add(d)
    return out


def _topo_sort(nodes: dict[str, DagNode]) -> list[str]:
    """Kahn's algorithm with deterministic ordering; raises on a cycle."""
    # indegree = number of unmet deps
    indeg = {nid: len(node.deps) for nid, node in nodes.items()}
    # dependents map
    dependents: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        for dep in node.deps:
            dependents[dep].append(nid)

    ready = sorted(nid for nid, d in indeg.items() if d == 0)
    order: list[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for dependent in sorted(dependents[nid]):
            indeg[dependent] -= 1
            if indeg[dependent] == 0:
                ready.append(dependent)
                ready.sort()

    if len(order) != len(nodes):
        cyclic = sorted(nid for nid in nodes if nid not in set(order))
        raise ConfigError(
            f"pipeline DAG has a cycle among stages: {cyclic}. "
            f"Check `needs`/`uses` references."
        )
    return order


# ── pretty printer (dry run) ────────────────────────────────────────────────
_KIND_ORDER = {"extract": 0, "estimators": 1, "adjust": 2, "analyses": 3, "report": 4}


def render_dag(dag: Dag, cfg: RunConfig) -> str:
    out = cfg.output
    lines: list[str] = []
    lines.append(f"Run:           {cfg.name}")
    if cfg.description:
        desc = cfg.description if len(cfg.description) <= 100 else cfg.description[:97] + "..."
        lines.append(f"Description:   {desc}")
    lines.append(f"Seed:          {cfg.seed}")
    lines.append(f"Config:        {cfg.config_path}")
    lines.append(
        f"Output root:   {out.resolved_root}/   "
        f"(root={out.root!r}, suffix={out.suffix!r})"
    )
    lines.append("")
    lines.append(f"Resolved DAG ({len(dag.order)} stages, topological order):")
    lines.append("")

    width = max((len(n) for n in dag.order), default=4)
    for i, nid in enumerate(dag.order, 1):
        node = dag.nodes[nid]
        module = node.raw.get("module")
        kind_label = node.kind if not module else f"{node.kind}:{module}"
        deps = sorted(node.deps)
        dep_str = ", ".join(deps) if deps else "—"
        lines.append(
            f"  {i:>2}. {nid:<{width}}  [{kind_label}]  <- {dep_str}"
        )
        save = _save_path(node)
        if save is not None:
            lines.append(f"      └ writes: {out.resolve(save)}")

    return "\n".join(lines)


def _save_path(node: DagNode) -> Optional[str]:
    raw = node.raw
    if node.kind == "estimators":
        return raw.get("save_predictions")
    if node.kind == "adjust":
        return raw.get("save")
    if node.kind == "report":
        return raw.get("out_dir")
    return None
