"""DAG build + topological order (benchmark.md §1)."""

from __future__ import annotations

from bench.config import load_config
from bench.pipeline import build_dag, render_dag


def test_dag_is_topologically_ordered(configs_dir):
    cfg = load_config(configs_dir / "benchmark.dev.json")
    dag = build_dag(cfg)
    pos = {nid: i for i, nid in enumerate(dag.order)}
    # every dependency appears before its dependent
    for node in dag.nodes.values():
        for dep in node.deps:
            assert pos[dep] < pos[node.id], f"{dep} must precede {node.id}"


def test_extract_node_present_when_enabled(configs_dir):
    cfg = load_config(configs_dir / "benchmark.dev.json")
    dag = build_dag(cfg)
    assert "extract" in dag.nodes
    # estimators depend on extract
    assert "extract" in dag.nodes["attention"].deps


def test_extract_dropped_when_disabled(configs_dir):
    cfg = load_config(configs_dir / "benchmark.pretrained.template.json")
    dag = build_dag(cfg)
    assert "extract" not in dag.nodes
    assert dag.nodes["attention"].deps == set()


def test_report_depends_on_all_analyses(configs_dir):
    cfg = load_config(configs_dir / "benchmark.dev.json")
    dag = build_dag(cfg)
    analysis_ids = {s.id for s in cfg.analyses if s.enabled}
    assert analysis_ids <= dag.nodes["report"].deps


def test_render_dag_shows_resolved_root(configs_dir):
    cfg = load_config(configs_dir / "benchmark.dev.json")
    text = render_dag(build_dag(cfg), cfg)
    assert "reports-dev/" in text
    assert "Resolved DAG" in text
