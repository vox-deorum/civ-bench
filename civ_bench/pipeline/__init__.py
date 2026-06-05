"""Pipeline layer: build + topo-sort the stage DAG and render it.

Import-light by design (only the config layer) so dry runs never pull in stage
implementations or heavy deps.
"""

from __future__ import annotations

from .dag import Dag, DagNode, build_dag, render_dag

__all__ = ["Dag", "DagNode", "build_dag", "render_dag"]
