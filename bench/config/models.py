"""Typed views over a validated run-spec.

These dataclasses are produced by :mod:`bench.config.loader` after validation.
They expose the run-spec in a structured way to every stage and resolve the
single output root (§2.1) that all save-paths are threaded through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .schema import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_SUFFIX,
    TABLE_NAMES,
)


@dataclass(frozen=True)
class OutputConfig:
    """The run output root + variant suffix (§2.1).

    Every stage save-path authored under ``<root>/...`` is re-rooted to
    ``<root><suffix>/...``. Default ``reports/``; ``suffix="-cross"`` →
    ``reports-cross/`` (the llm/non-llm variant).
    """

    root: str = DEFAULT_OUTPUT_ROOT
    suffix: str = DEFAULT_OUTPUT_SUFFIX

    @property
    def resolved_root(self) -> str:
        return f"{self.root.rstrip('/')}{self.suffix}"

    def resolve(self, path: Optional[str]) -> Optional[str]:
        """Re-root an authored save-path under the resolved output root.

        A path written as ``<root>/sub/...`` (e.g. ``reports/estimators/x``) is
        rewritten to ``<root><suffix>/sub/...``. Paths that do not start with the
        base ``<root>`` segment are returned unchanged (they are not output-root
        relative).
        """
        if path is None:
            return None
        if self.suffix == "":
            return path
        norm = path.replace("\\", "/")
        base = self.root.rstrip("/")
        if norm == base:
            return self.resolved_root
        prefix = base + "/"
        if norm.startswith(prefix):
            return self.resolved_root + "/" + norm[len(prefix):]
        return path


@dataclass
class Stage:
    """A node in the pipeline DAG."""

    id: str
    kind: str  # extract | estimators | adjust | analyses | report
    enabled: bool
    raw: dict = field(default_factory=dict)

    @property
    def module(self) -> Optional[str]:
        return self.raw.get("module")

    @property
    def uses_estimators(self) -> list[str]:
        return list(self.raw.get("uses", {}).get("estimators", []) or [])

    @property
    def uses_tables(self) -> list[str]:
        return list(self.raw.get("uses", {}).get("tables", []) or [])

    @property
    def uses_analyses(self) -> list[str]:
        return list(self.raw.get("uses", {}).get("analyses", []) or [])

    @property
    def needs(self) -> list[str]:
        return list(self.raw.get("needs", []) or [])


@dataclass
class RunConfig:
    """A validated benchmark run-spec."""

    name: str
    seed: int
    config_path: Path
    raw: dict
    output: OutputConfig
    friendly_name: str = ""
    description: str = ""
    presentation: dict = field(default_factory=dict)
    filters: dict = field(default_factory=dict)
    groupings: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    estimators: list[Stage] = field(default_factory=list)
    adjust: list[Stage] = field(default_factory=list)
    analyses: list[Stage] = field(default_factory=list)
    report: dict = field(default_factory=dict)
    _catalog_overrides: dict = field(default_factory=dict)
    _resolved_graph: Any = field(default=None, repr=False)

    # ── catalog resolution (lazy: a path need only exist when a stage needs it)
    def catalog_path(self, which: str) -> Path:
        """Resolve a catalog file path (``paths`` | ``models`` | ``experiments``).

        Uses an explicit override from ``catalogs`` if present, else the sibling
        file next to the run-spec (``<config_dir>/<which>.json``).
        """
        override = self._catalog_overrides.get(which)
        if override:
            p = Path(override)
            return p if p.is_absolute() else (self.config_path.parent / p)
        return self.config_path.parent / f"{which}.json"

    @property
    def table_names(self) -> tuple[str, ...]:
        return TABLE_NAMES

    @property
    def extract_enabled(self) -> bool:
        return bool(self.data.get("extract", {}).get("enabled", True))

    def all_stages(self) -> list[Stage]:
        return list(self.estimators) + list(self.adjust) + list(self.analyses)
