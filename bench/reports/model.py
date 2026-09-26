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
    height: Optional[int] = None  # an interactive figure's layout height in pixels

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
    tip: str = ""  # the long wording behind a short label, shown on hover


@dataclass
class CurveChart:
    """The data behind one interactive victory-probability curve chart.

    ``frame`` holds one row per curve point (``strategist``, ``condition``,
    ``turn_progress``, and the mean in ``value_column``). The ordering and
    color fields come from the producing analysis, so the renderer needs no
    catalog. A curve whose strategist and condition both equal
    ``vanilla_label`` is the VPAI reference curve. Rendered by
    :func:`bench.reports.curves.render_curve_views_html` on family pages and on
    the Matched Maps seat pages. ``view``, ``label``, and ``tip`` name the
    chart's tab when a section offers several views of its curves (Relative
    for adjusted strength, Absolute for probability); ``reference_line`` draws
    a dotted level line, such as 0.5 for adjusted strength.
    """

    frame: pd.DataFrame
    vanilla_label: str = "Vanilla"
    strategist_order: list[str] = field(default_factory=list)
    condition_order: list[str] = field(default_factory=list)
    strategist_colors: dict[str, str] = field(default_factory=dict)
    help: str = ""
    value_column: str = "mean_predicted_win_probability"
    y_title: str = "Mean predicted win probability"
    reference_line: Optional[float] = None
    view: str = ""
    label: str = ""
    tip: str = ""


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
    # declared by metadata["curve_chart"]; several are alternative views
    curve_charts: list[CurveChart] = field(default_factory=list)
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
class MatchedMapTab:
    """One extra Matched Maps tab from another analysis (``uses.analyses``).

    The analysis declares it through ``metadata["matched_maps"]`` (one entry
    or a list): a short label plus one long heatmap table keyed by ``seed``
    (the overview) and one keyed by ``seed`` and ``player_id`` (the seat
    pages), each with its layout spec in ``metadata["heatmaps"]``. The chapter
    slices both by seed or seat.
    """

    name: str  # the analysis stage id (plus "-<key>"), also the view name
    label: str
    tip: str = ""
    seed_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    seat_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    seed_spec: dict = field(default_factory=dict)
    seat_spec: dict = field(default_factory=dict)


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
    # Adjusted-strength curves (the seat pages' Relative view); empty when the
    # analysis emitted none.
    adjusted_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Each player's in-game rank by weighted victory probability; empty when
    # the analysis emitted none.
    rank_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    downloads: list[Download] = field(default_factory=list)
    footer: str = REPORT_DEFAULT_FOOTER
    game_log: Optional["GameLogDocument"] = None
    tabs: list[MatchedMapTab] = field(default_factory=list)

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
    # (game_id, player_id) -> in-game rank by weighted victory probability,
    # from the Matched Maps analysis; empty without it.
    ranks: dict[tuple[str, int], int] = field(default_factory=dict)
    footer: str = REPORT_DEFAULT_FOOTER
