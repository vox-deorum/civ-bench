"""Analysis interface, result container, and execution context (stage 4).

Every analysis module is a self-contained unit behind the :class:`Analysis`
interface; the JSON ``analyses[].module`` string selects it from the registry
(:mod:`bench.analyses.registry`). An analysis consumes its declared inputs via
the :class:`AnalysisContext` and returns an :class:`AnalysisResult` (tables +
figures + summary) — it never writes files itself (the runner persists the
result), so the same module is reusable from a notebook, a test, or the report.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..catalog import Catalog
from ..config import RunConfig
from .errors import AnalysisError


@dataclass
class AnalysisResult:
    """What an analysis returns: named tables, named figures, a text summary.

    ``tables`` and ``figures`` are keyed by a short slug (used for the persisted
    filename); ``figures`` values are matplotlib ``Figure`` objects. An analysis
    that legitimately produces nothing for the given inputs (e.g.
    ``calibration.cell_baseline`` on a fully uncontrolled run) returns an empty
    result — :meth:`is_empty` is true and the runner records it without error.

    ``artifacts`` is the escape hatch for files that are neither a tabular CSV
    nor a figure (e.g. the generated ``seating/*.seating.json`` files): it maps a
    path *relative to the analysis dir* (subdirs allowed) to the file's text
    content. The runner writes each verbatim and the report copies them into
    ``assets/`` as downloadable links.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    metadata: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.tables and not self.figures and not self.artifacts


class Analysis(ABC):
    """Base class for every analysis module.

    Subclasses set the ``module`` class attribute to their registry name and
    implement :meth:`run`. Construction is cheap (just stashes id + params); all
    real work happens in :meth:`run` so the registry can be imported without
    touching data.
    """

    module: str = ""
    default_all_estimators: bool = False

    def __init__(self, stage_id: str, params: Optional[dict] = None) -> None:
        self.stage_id = stage_id
        self.params = dict(params or {})

    @abstractmethod
    def run(self, ctx: "AnalysisContext") -> AnalysisResult:
        """Produce the analysis result from the resolved context."""
        raise NotImplementedError


@dataclass
class AnalysisContext:
    """Resolved inputs handed to an analysis: config, catalog, and table/estimator
    resolvers that honour the output root and the run's filters.

    The context centralizes path resolution (mirroring the adjust runner) so a
    module asks for ``ctx.load_table("strength")`` or
    ``ctx.load_predictions("attention")`` without re-deriving output-rooted
    paths. ``apply_filter`` applies the global ``data.filter`` narrowed by this
    stage's ``filter`` (the same preset/intersection semantics the config
    validates).
    """

    config: RunConfig
    catalog: Catalog
    stage_id: str
    stage_raw: dict
    out_dir: Path
    default_all_estimators: bool = False

    @property
    def params(self) -> dict:
        return self.stage_raw.get("params") or {}

    @property
    def stage_filter(self):
        return self.stage_raw.get("filter")

    # ── filter resolution ────────────────────────────────────────────────────
    def apply_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the global ``data.filter`` narrowed by this stage's ``filter``."""
        from ..data.loading import apply_filter_spec

        return apply_filter_spec(
            df,
            catalog=self.catalog,
            filter_spec=self.config.data.get("filter"),
            presets=self.config.filters,
            stage_filter=self.stage_filter,
        )

    # ── canonical / adjust table resolution ───────────────────────────────────
    def _canonical_path(self, name: str) -> Optional[str]:
        tables = self.config.data.get("tables") or {}
        return tables.get(name)

    def _adjust_stage(self, table_id: str):
        return next((s for s in self.config.adjust if s.id == table_id), None)

    def _adjust_default_save(self) -> str:
        return f"{self.config.output.root}/adjust/player_strength_panel.csv"

    def adjust_save_path(self, table_id: str = "strength") -> str:
        """Output-root-resolved save path of an adjust stage's table."""
        stage = self._adjust_stage(table_id)
        if stage is None:
            raise AnalysisError(
                f"analysis '{self.stage_id}': no adjust stage with id '{table_id}'."
            )
        authored = stage.raw.get("save") or self._adjust_default_save()
        return self.config.output.resolve(authored)

    def adjust_dir(self, table_id: str = "strength") -> Path:
        """Directory holding an adjust stage's panel + audit trails."""
        return Path(self.adjust_save_path(table_id)).parent

    def table_path(self, name: str) -> str:
        """Resolve a ``uses.tables`` entry: a canonical table key or an adjust id."""
        canonical = self._canonical_path(name)
        if canonical is not None:
            return canonical
        if self._adjust_stage(name) is not None:
            return self.adjust_save_path(name)
        raise AnalysisError(
            f"analysis '{self.stage_id}': table '{name}' is neither a canonical "
            f"table {sorted((self.config.data.get('tables') or {}).keys())} nor an "
            f"adjust stage id."
        )

    def load_table(self, name: str) -> pd.DataFrame:
        """Read a resolved table CSV (canonical or adjust output), unfiltered."""
        path = self.table_path(name)
        if not Path(path).exists():
            raise AnalysisError(
                f"analysis '{self.stage_id}': table '{name}' not found at '{path}'. "
                f"Run the upstream stage (extract / adjust) first."
            )
        return pd.read_csv(path)

    def strength_provenance(self, table_id: Optional[str] = None, panel: Optional[pd.DataFrame] = None) -> dict:
        """Compact provenance for analyses consuming a strength adjust table."""
        if table_id is None:
            table_id = next(
                (t for t in self.uses_tables() if self._adjust_stage(t) is not None),
                "strength",
            )
        stage = self._adjust_stage(table_id)
        if stage is None:
            return {"strength_table": table_id}

        params = dict(stage.raw.get("params") or {})
        estimators = list((stage.raw.get("uses") or {}).get("estimators") or [])
        estimator_id = estimators[0] if estimators else None
        estimator = next((s for s in self.config.estimators if s.id == estimator_id), None)
        estimator_raw = estimator.raw if estimator is not None else {}

        raw_block = params.get("block", "auto")
        effective_block = raw_block
        if raw_block == "auto" and panel is not None and "controlled" in panel.columns:
            effective_block = "start_cell" if bool(panel["controlled"].fillna(False).any()) else "none"
        block_label = (
            f"{raw_block}/{effective_block}" if effective_block != raw_block else str(raw_block)
        )

        out = {
            "strength_table": table_id,
            "adjust_stage": stage.id,
            "strength_estimator": estimator_id,
            "estimator_model": estimator_raw.get("model"),
            "estimator_fit": estimator_raw.get("fit"),
            "estimator_predict": estimator_raw.get("predict", "in_sample") if estimator_raw else None,
            "adjust_block": block_label,
        }
        for key in (
            "turn_progress_min",
            "weight",
            "relative_to",
            "enforce_winner",
            "civ_adjust",
            "baseline_experiment",
            "post_cell_normalize",
        ):
            if key in params and params[key] is not None:
                out[f"adjust_{key}"] = params[key]
        return {k: v for k, v in out.items() if v is not None}

    # ── estimator predictions ──────────────────────────────────────────────────
    def predictions_path(self, estimator_id: str) -> str:
        stage = next((s for s in self.config.estimators if s.id == estimator_id), None)
        if stage is None:
            raise AnalysisError(
                f"analysis '{self.stage_id}': references unknown estimator "
                f"'{estimator_id}'."
            )
        authored = stage.raw.get("save_predictions") or (
            f"{self.config.output.root}/estimators/{estimator_id}/predictions.csv"
        )
        return self.config.output.resolve(authored)

    def load_predictions(self, estimator_id: str, usecols=None) -> pd.DataFrame:
        path = self.predictions_path(estimator_id)
        if not Path(path).exists():
            raise AnalysisError(
                f"analysis '{self.stage_id}': estimator '{estimator_id}' predictions "
                f"not found at '{path}'. Run the estimator stage first."
            )
        return pd.read_csv(path, usecols=usecols)

    def uses_estimators(self) -> list[str]:
        explicit = list((self.stage_raw.get("uses") or {}).get("estimators") or [])
        if explicit:
            return explicit
        if self.default_all_estimators:
            return [s.id for s in self.config.estimators if s.enabled]
        return []

    def uses_tables(self) -> list[str]:
        return list((self.stage_raw.get("uses") or {}).get("tables") or [])
