"""Reports layer — assemble produced analysis artifacts into a document (stage 5).

The report stage walks each enabled analysis's persisted ``result.json`` manifest
(tables + figures + summary) in dependency order, copies the artifacts into a
self-contained ``assets/`` tree, builds a renderer-agnostic document via the
selected template, and renders it. The ``default`` template produces the
:class:`~bench.reports.model.ReportDocument` (Markdown + a compact HTML overview
and family pages); the ``controlled_seed`` template produces the dedicated
:class:`~bench.reports.model.ControlledSeedDocument` (a controlled-seed overview
plus one page per seed-player pair, HTML only). No analysis hardcodes its place
in the document, and the whole site regenerates from disk, so
``civ-bench report`` reproduces it without re-running any analysis (invariant 3).

Import-light: needs pandas + the stdlib only (figures are already PNGs on disk), so
it pulls neither matplotlib nor R; safe to import from the CLI report path.
"""

from __future__ import annotations

from .context import ReportBuildContext
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
from .templates import TEMPLATES, get_template, template_formats

__all__ = [
    "ControlledSeedDocument",
    "Figure",
    "FamilyGroup",
    "ReportBuildContext",
    "ReportDocument",
    "ReportError",
    "ReportRunResult",
    "Section",
    "TEMPLATES",
    "Table",
    "get_template",
    "render_html",
    "render_html_site",
    "render_markdown",
    "render_stylesheet",
    "report_dir",
    "run_report",
    "template_formats",
]
