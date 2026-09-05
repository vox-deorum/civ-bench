"""Reports layer: assemble produced analysis artifacts into a document (stage 5).

The report stage walks each enabled analysis's persisted ``result.json`` manifest
(tables + figures + summary) in dependency order, copies the artifacts into a
self-contained ``assets/`` tree, builds a renderer-agnostic document, and renders
it: the family report (:class:`~bench.reports.model.ReportDocument`, rendered as
Markdown plus a compact HTML overview and family pages) plus, when the report
carries the ``performance.controlled_seed_report`` analysis, the controlled-seed
heatmap annex (:class:`~bench.reports.model.ControlledSeedDocument`, an overview
with per-seed heatmaps and one HTML page per seed-player pair). No analysis
hardcodes its place in the document, and the whole site regenerates from disk, so
``civ-bench report`` reproduces it without re-running any analysis (invariant 3).

Import-light: needs pandas + the stdlib only (figures are already PNGs on disk), so
it pulls neither matplotlib nor R; safe to import from the CLI report path.
"""

from __future__ import annotations

from .assets import render_common_script
from .context import ReportBuildContext
from .controlled_seed import controlled_seed_document
from .errors import ReportError
from .model import (
    ControlledSeedDocument,
    Figure,
    FamilyGroup,
    ReportDocument,
    Section,
    Table,
)
from .render import render_html, render_html_site, render_markdown, render_stylesheet
from .runner import ReportRunResult, report_dir, run_report
from .templates import default_template

__all__ = [
    "ControlledSeedDocument",
    "Figure",
    "FamilyGroup",
    "ReportBuildContext",
    "ReportDocument",
    "ReportError",
    "ReportRunResult",
    "Section",
    "Table",
    "controlled_seed_document",
    "default_template",
    "render_common_script",
    "render_html",
    "render_html_site",
    "render_markdown",
    "render_stylesheet",
    "report_dir",
    "run_report",
]
