"""Project a validated :class:`RunConfig` into renderable DAG nodes.

Dependency semantics live in :mod:`bench.config.dependencies`, where the graph
is resolved and cached at config-load time. Edges come from three places
(benchmark.md §1), all resolved into one topological order before anything runs:

1. **Kind ordering (implicit)** — extract → estimators → adjust → analyses →
   report. Concretely: every estimator depends on ``extract`` (it re-infers on
   the canonical turns table); ``report`` depends on every enabled analysis.
2. **`needs` (explicit)** — extra ordering the harness cannot infer.
3. **`uses` (referential)** — an estimator id, table name, or analysis id in a
   stage's ``uses`` block creates an edge automatically (to the producing
   estimator / adjust / analysis stage, or to extract for a canonical table).
   Estimator-consuming analyses with omitted/empty ``uses.estimators`` depend
   on every enabled estimator.

Disabled stages (``enabled: false``) are dropped from the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import RunConfig
from ..config.dependencies import (
    ResolvedGraph,
    resolve_stage_graph,
)


@dataclass
class DagNode:
    id: str
    kind: str
    deps: set[str] = field(default_factory=set)  # ids this node must run after
    raw: dict = field(default_factory=dict)

    @property
    def save_path(self) -> Optional[str]:
        if self.kind == "estimators":
            return self.raw.get("save_predictions")
        if self.kind == "adjust":
            return self.raw.get("save")
        if self.kind == "report":
            return self.raw.get("out_dir")
        return None

    def resolved_save_path(self, cfg: RunConfig) -> Optional[str]:
        resolved = cfg.output.resolve(self.save_path)
        if self.kind == "report" and resolved is not None:
            # The report writes the per-run subdir <resolved-out_dir>/<name>/ (§7);
            # show that, not just the bare output root, so the dry-run is faithful.
            return f"{resolved.rstrip('/')}/{cfg.name}/"
        return resolved


@dataclass
class Dag:
    nodes: dict[str, DagNode]
    order: list[str]  # topologically sorted node ids

    def __iter__(self):
        for nid in self.order:
            yield self.nodes[nid]


def build_dag(cfg: RunConfig) -> Dag:
    """Resolve the run-spec into a topologically sorted DAG of enabled stages."""
    graph = _resolved_graph(cfg)
    nodes = {
        nid: DagNode(
            id=node.id,
            kind=node.kind,
            deps=set(node.deps),
            raw=node.raw,
        )
        for nid, node in graph.nodes.items()
    }
    return Dag(nodes=nodes, order=list(graph.order))


def _resolved_graph(cfg: RunConfig) -> ResolvedGraph:
    graph = cfg._resolved_graph
    if graph is None:
        graph = resolve_stage_graph(cfg)
        cfg._resolved_graph = graph
    return graph


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
        save = node.resolved_save_path(cfg)
        if save is not None:
            lines.append(f"      └ writes: {save}")

    return "\n".join(lines)
