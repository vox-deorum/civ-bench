"""The renderer-agnostic report document model (stage 5).

A :class:`ReportDocument` is the intermediate representation a *template* builds
from the produced :class:`~bench.analyses.base.AnalysisResult` manifests, and the
md/html renderers consume. Keeping a structured model (rather than templating
strings directly) lets every output format render the *same* document faithfully:
a markdown table and an HTML ``<table>`` come from one :class:`Table`, not two
hand-written variants.

Nothing here imports matplotlib or reads data; figures are referenced by the
relative path the runner already copied into the report's ``assets/`` tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from bench.config.schema import REPORT_DEFAULT_FOOTER


@dataclass
class Figure:
    """A rendered figure, referenced by its report-relative path."""

    caption: str
    rel_path: str  # e.g. "assets/bt_main/ratings.png", relative to the report dir

    @property
    def interactive(self) -> bool:
        return self.rel_path.lower().endswith(".html")


@dataclass
class Table:
    """A tabular artifact rendered inline (capped) with a link to the full CSV."""

    name: str
    frame: pd.DataFrame
    rel_csv: Optional[str] = None  # report-relative path to the full CSV
    n_total_rows: int = 0
    n_shown_rows: int = 0
    heatmap: Optional[dict] = None  # layout spec when the table renders as a heatmap

    @property
    def truncated(self) -> bool:
        return self.n_total_rows > self.n_shown_rows


@dataclass
class Download:
    """A non-tabular, non-figure file artifact offered as a download link."""

    label: str
    rel_path: str  # e.g. "assets/<id>/seating/<exp>.seating.json", report-relative


@dataclass
class View:
    """One of several alternative presentations of a section's results.

    An analysis declares views in ``metadata["views"]`` (an ordered mapping of
    view name to ``{"label", "tables", "figures"}``). The first view is the
    default; the HTML report shows one view at a time behind a toggle, and the
    Markdown report renders every view under its own subheading.
    """

    name: str
    label: str
    figures: list[Figure] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)


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
    views: list[View] = field(default_factory=list)  # alternative presentations, default first
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
    summary: str = ""


@dataclass
class Announcement:
    """A compact front-page update with inline Markdown and an optional ISO date."""

    kind: str
    title: str
    text: str
    date: str | None = None


@dataclass
class ReportDocument:
    """The full report: a title, run provenance, and grouped sections.

    When the resolved sections carry the controlled-seed analysis, its section
    becomes a chapter of its own (the last group) and ``controlled_seed`` holds
    the annex document behind the chapter's heatmap pages; the html site
    renders the chapter under ``controlled-seed/``, a markdown-only render
    keeps the chapter's section and skips the pages.
    """

    title: str
    run_name: str
    seed: int
    config_path: str
    output_root: str
    description: str = ""  # the run-spec's top-level description, shown on the report page
    groups: list[FamilyGroup] = field(default_factory=list)
    intro: str = ""
    overview_sections: list[Section] = field(default_factory=list)
    controlled_seed: Optional["ControlledSeedDocument"] = None
    footer: str = REPORT_DEFAULT_FOOTER
    benchmark_citation: dict[str, str] | None = None
    announcements: list[Announcement] = field(default_factory=list)
    game_log: Optional["GameLogDocument"] = None

    @property
    def n_sections(self) -> int:
        return sum(len(g.sections) for g in self.groups)


@dataclass
class ControlledSeedDocument:
    """The document behind the report's controlled-seed heatmap pages.

    Built by :func:`bench.reports.controlled_seed.controlled_seed_document`
    from one ``performance.controlled_seed_report`` section's persisted tables
    and carried on the :class:`ReportDocument`. It carries the three
    report-ready tables plus the ordering and color metadata the analysis
    recorded in its manifest, so the renderer needs no catalog or
    canonical-table access.
    """

    title: str
    run_name: str
    seed: int
    config_path: str
    output_root: str
    description: str = ""
    section_id: str = ""
    summary: str = ""
    metadata: dict = field(default_factory=dict)
    summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    probability_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    downloads: list[Download] = field(default_factory=list)
    footer: str = REPORT_DEFAULT_FOOTER
    game_log: Optional["GameLogDocument"] = None

    @property
    def vanilla_label(self) -> str:
        return str(self.metadata.get("vanilla_label", "Vanilla"))

    @property
    def base_label(self) -> str:
        return str(self.metadata.get("base_label", "Base"))

    @property
    def strategist_order(self) -> list[str]:
        return list(self.metadata.get("strategist_order") or [])

    @property
    def condition_order(self) -> list[str]:
        return list(self.metadata.get("condition_order") or [])

    @property
    def strategist_colors(self) -> dict[str, str]:
        return dict(self.metadata.get("strategist_colors") or {})


@dataclass
class ReplayOptions:
    """Published save locations and the viewer used to open them."""

    viewer_url: str
    base_url: str | None = None
    save_paths: dict[str, str] = field(default_factory=dict)
    latest_game: bool = True


@dataclass
class GameLogDocument:
    """Full saved game tables used by the log and its report links."""

    title: str = ""
    section_id: str = ""
    games: pd.DataFrame = field(default_factory=pd.DataFrame)
    game_players: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)
    downloads: list[Download] = field(default_factory=list)
    replay: ReplayOptions | None = None
    footer: str = REPORT_DEFAULT_FOOTER
