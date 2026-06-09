"""Analysis-stage orchestrator (stage 4).

Executes one ``analyses`` node from the resolved DAG: builds the
:class:`AnalysisContext` (output-root-aware table/estimator resolvers), runs the
registered module, then persists the returned :class:`AnalysisResult` — tables to
CSV and figures to PNG under ``<root>/analyses/<id>/`` — and returns a small
summary object for the CLI / report stage.

Imports matplotlib (figures), statsmodels (regressions), and optionally calls
``Rscript`` (ratings), so it lives off the import-light config/dry-run path and
is only imported from the CLI run dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # batch rendering: no display, deterministic file output
import matplotlib.pyplot as plt  # noqa: E402

from ..catalog import Catalog  # noqa: E402
from ..config import RunConfig  # noqa: E402
from .base import AnalysisContext, AnalysisResult  # noqa: E402
from .errors import AnalysisError  # noqa: E402
from .registry import get_analysis  # noqa: E402


@dataclass
class AnalysisRunResult:
    id: str
    module: str
    table_paths: dict[str, str] = field(default_factory=dict)
    figure_paths: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    empty: bool = False
    metadata: dict = field(default_factory=dict)


def _analyses_out_dir(cfg: RunConfig, stage_id: str) -> Path:
    authored = f"{cfg.output.root}/analyses/{stage_id}"
    return Path(cfg.output.resolve(authored))


def run_analysis(
    cfg: RunConfig,
    stage_raw: dict,
    catalog: Optional[Catalog] = None,
) -> AnalysisRunResult:
    """Run one analysis stage and persist its result."""
    stage_id = stage_raw["id"]
    module = stage_raw["module"]
    cls = get_analysis(module)  # AnalysisError on unknown / unimplemented module

    if catalog is None:
        catalog = Catalog.from_run_config(cfg)

    out_dir = _analyses_out_dir(cfg, stage_id)
    ctx = AnalysisContext(
        config=cfg,
        catalog=catalog,
        stage_id=stage_id,
        stage_raw=stage_raw,
        out_dir=out_dir,
    )

    analysis = cls(stage_id, stage_raw.get("params"))
    result: AnalysisResult = analysis.run(ctx)

    table_paths: dict[str, str] = {}
    figure_paths: dict[str, str] = {}
    if not result.is_empty():
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in result.tables.items():
            path = out_dir / f"{name}.csv"
            table.to_csv(path, index=False)
            table_paths[name] = str(path)
        for name, fig in result.figures.items():
            path = out_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figure_paths[name] = str(path)

    return AnalysisRunResult(
        id=stage_id,
        module=module,
        table_paths=table_paths,
        figure_paths=figure_paths,
        summary=result.summary,
        empty=result.is_empty(),
        metadata=result.metadata,
    )
