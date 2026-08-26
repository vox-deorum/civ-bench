"""The renderer-agnostic report document model (stage 5).

A :class:`ReportDocument` is the intermediate representation a *template* builds
from the produced :class:`~bench.analyses.base.AnalysisResult` manifests, and the
md/html renderers consume. Keeping a structured model (rather than templating
strings directly) lets every output format render the *same* document faithfully
— a markdown table and an HTML ``<table>`` come from one :class:`Table`, not two
hand-written variants.

Nothing here imports matplotlib or reads data; figures are referenced by the
relative path the runner already copied into the report's ``assets/`` tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Figure:
    """A rendered figure, referenced by its report-relative path."""

    caption: str
    rel_path: str  # e.g. "assets/bt_main/ratings.png" — relative to the report dir


@dataclass
class Table:
    """A tabular artifact rendered inline (capped) with a link to the full CSV."""

    name: str
    frame: pd.DataFrame
    rel_csv: Optional[str] = None  # report-relative path to the full CSV
    n_total_rows: int = 0
    n_shown_rows: int = 0

    @property
    def truncated(self) -> bool:
        return self.n_total_rows > self.n_shown_rows


@dataclass
class Download:
    """A non-tabular, non-figure file artifact offered as a download link."""

    label: str
    rel_path: str  # e.g. "assets/<id>/seating/<exp>.seating.json" — relative to the report dir


@dataclass
class Section:
    """One analysis stage's contribution to the report."""

    id: str
    module: str
    display_name: str = ""  # friendly heading text (module default or stage override)
    description: str = ""  # one-line module description (module default or stage override)
    summary: str = ""
    metadata: dict = field(default_factory=dict)
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    downloads: list[Download] = field(default_factory=list)
    empty: bool = False

    @property
    def title(self) -> str:
        """The visible heading for this section: friendly name, else the stage id."""
        return self.display_name or self.id


@dataclass
class FamilyGroup:
    """Sections sharing a module family (``ratings.*`` → "Ratings"), the level-2
    headings that turn the old per-area notebooks into generated chapters."""

    key: str  # e.g. "ratings"
    title: str  # e.g. "Ratings"
    sections: list[Section] = field(default_factory=list)


@dataclass
class ReportDocument:
    """The full report: a title, run provenance, and grouped sections."""

    title: str
    run_name: str
    seed: int
    config_path: str
    output_root: str
    description: str = ""  # the run-spec's top-level description, shown on the report page
    groups: list[FamilyGroup] = field(default_factory=list)
    intro: str = ""
    overview_sections: list[Section] = field(default_factory=list)

    @property
    def n_sections(self) -> int:
        return sum(len(g.sections) for g in self.groups)
