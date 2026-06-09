"""Adjust-stage orchestrator (stage 3).

Executes one ``adjust`` node from the resolved DAG: resolves its single estimator's
saved ``predictions.csv``, the canonical ``panel``/``games`` tables, and the
output-root-adjusted ``save`` path, runs the module builder (currently
``strength``), and writes the panel plus the always-on audit trails
(``civ_effects.csv``, ``cell_baseline.csv``, ``cell_coverage.csv``) next to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..catalog import Catalog
from ..config import RunConfig
from .errors import AdjustError
from .registry import get_adjuster


@dataclass
class AdjustResult:
    id: str
    module: str
    estimator_id: str
    table_path: str
    n_rows: int
    civ_effects_path: str
    cell_baseline_path: str
    cell_coverage_path: str
    warnings: list[str]


def _default_save_path(cfg: RunConfig, stage_id: str) -> str:
    return f"{cfg.output.root}/adjust/player_strength_panel.csv"


def _table_path(cfg: RunConfig, key: str) -> str:
    tables = cfg.data.get("tables", {}) or {}
    path = tables.get(key)
    if not path:
        raise AdjustError(
            f"data.tables.{key} is not set; the strength stage needs the canonical "
            f"'{key}' CSV."
        )
    return path


def _estimator_predictions_path(cfg: RunConfig, estimator_id: str) -> str:
    """Resolve the saved predictions CSV of the referenced estimator (output-rooted)."""
    stage = next((s for s in cfg.estimators if s.id == estimator_id), None)
    if stage is None:
        raise AdjustError(
            f"adjust stage references unknown estimator '{estimator_id}'."
        )
    raw = stage.raw
    authored = raw.get("save_predictions") or f"{cfg.output.root}/estimators/{estimator_id}/predictions.csv"
    return cfg.output.resolve(authored)


def run_adjust(
    cfg: RunConfig,
    stage_raw: dict,
    catalog: Optional[Catalog] = None,
) -> AdjustResult:
    """Run one adjust stage and write its named table + audit trails."""
    stage_id = stage_raw["id"]
    module = stage_raw["module"]
    builder = get_adjuster(module)  # ValueError on unknown module → fail loud

    uses = stage_raw.get("uses") or {}
    estimators = uses.get("estimators") or []
    if len(estimators) != 1:
        raise AdjustError(
            f"adjust '{stage_id}': exactly one uses.estimators is required (got {estimators})."
        )
    estimator_id = estimators[0]

    predictions_path = _estimator_predictions_path(cfg, estimator_id)
    if not Path(predictions_path).exists():
        raise AdjustError(
            f"adjust '{stage_id}': estimator '{estimator_id}' predictions not found at "
            f"'{predictions_path}'. Run the estimator stage first."
        )
    panel_path = _table_path(cfg, "panel")
    games_path = _table_path(cfg, "games")
    for label, path in (("panel", panel_path), ("games", games_path)):
        if not Path(path).exists():
            raise AdjustError(
                f"adjust '{stage_id}': {label} table not found at '{path}'. Run extract first."
            )

    if catalog is None:
        catalog = Catalog.from_run_config(cfg)

    artifacts = builder(
        predictions_path,
        panel_path,
        games_path,
        stage_raw.get("params"),
        catalog,
        estimator_id,
    )

    save_path = cfg.output.resolve(stage_raw.get("save") or _default_save_path(cfg, stage_id))
    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.panel.to_csv(save_path, index=False)

    civ_path = str(out_dir / "civ_effects.csv")
    cell_baseline_path = str(out_dir / "cell_baseline.csv")
    cell_coverage_path = str(out_dir / "cell_coverage.csv")
    artifacts.civ_effects.to_csv(civ_path, index=False)
    artifacts.cell_baseline.to_csv(cell_baseline_path, index=False)
    artifacts.cell_coverage.to_csv(cell_coverage_path, index=False)

    return AdjustResult(
        id=stage_id,
        module=module,
        estimator_id=estimator_id,
        table_path=save_path,
        n_rows=len(artifacts.panel),
        civ_effects_path=civ_path,
        cell_baseline_path=cell_baseline_path,
        cell_coverage_path=cell_coverage_path,
        warnings=list(artifacts.warnings),
    )
