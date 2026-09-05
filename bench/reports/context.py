"""The report build context handed to document builders (stage 5).

A builder is a pure function over this context: it receives the run metadata,
the resolved sections, and a containment-checked loader for the *full* named CSV
artifacts of each selected analysis manifest. The loader is what lets a
specialized document (e.g. the controlled-seed annex) rebuild complete tables
from persisted artifacts without the report stage ever reading canonical
tables or estimator predictions directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .errors import ReportError
from .model import Section


@dataclass(frozen=True)
class _TableSource:
    """Where one named table artifact lives: its analysis dir + relative file."""

    analysis_dir: Path
    rel_file: str


@dataclass
class ReportBuildContext:
    """Run metadata, resolved sections, and the full-artifact table loader."""

    meta: dict
    sections: list[Section] = field(default_factory=list)
    _table_sources: dict[tuple[str, str], _TableSource] = field(default_factory=dict)

    def record_table(self, stage_id: str, name: str, analysis_dir: Path, rel_file: str) -> None:
        """Record one manifest table artifact as loadable (runner bookkeeping)."""
        self._table_sources[(stage_id, name)] = _TableSource(
            analysis_dir=analysis_dir, rel_file=rel_file
        )

    def section(self, stage_id: str) -> Section:
        for section in self.sections:
            if section.id == stage_id:
                return section
        raise ReportError(
            f"report section '{stage_id}' is not among the resolved sections "
            f"{[s.id for s in self.sections]}."
        )

    def load_table(self, stage_id: str, name: str) -> pd.DataFrame:
        """Load the full named CSV artifact of one selected analysis section.

        The recorded source is containment-checked against its analysis dir at
        load time (defense in depth on top of the copy-time check), so a
        manifest-supplied path can never read outside the analysis tree.
        """
        source: Optional[_TableSource] = self._table_sources.get((stage_id, name))
        if source is None:
            available = sorted(
                tbl for sid, tbl in self._table_sources if sid == stage_id
            )
            raise ReportError(
                f"report section '{stage_id}' has no table artifact '{name}' "
                f"(available: {available})."
            )
        root = source.analysis_dir.resolve()
        path = (root / source.rel_file).resolve()
        if path != root and root not in path.parents:
            raise ReportError(
                f"report section '{stage_id}': table artifact '{source.rel_file}' "
                "escapes the analysis tree; refusing to load it."
            )
        if not path.exists():
            raise ReportError(
                f"report section '{stage_id}': table artifact '{name}' is missing "
                f"at '{source.rel_file}'."
            )
        return pd.read_csv(path)
