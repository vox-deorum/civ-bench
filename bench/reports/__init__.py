"""Reports layer — assemble produced analysis artifacts into a document (stage 5).

The report stage walks each enabled analysis's persisted ``result.json`` manifest
(tables + figures + summary) in dependency order, copies the artifacts into a
self-contained ``assets/`` tree, builds a renderer-agnostic
:class:`~bench.reports.model.ReportDocument` via the selected template, and renders
markdown + HTML. No analysis hardcodes its place in the document, and the whole
document regenerates from disk — so ``civ-bench report`` reproduces it without
re-running any analysis (invariant 3).

Import-light: needs pandas + the stdlib only (figures are already PNGs on disk), so
it pulls neither matplotlib nor R; safe to import from the CLI report path.
"""

from __future__ import annotations

from .errors import ReportError
from .model import Figure, FamilyGroup, ReportDocument, Section, Table
from .render import render_html, render_markdown
from .runner import ReportRunResult, report_dir, run_report
from .templates import TEMPLATES, get_template

__all__ = [
    "Figure",
    "FamilyGroup",
    "ReportDocument",
    "ReportError",
    "ReportRunResult",
    "Section",
    "TEMPLATES",
    "Table",
    "get_template",
    "render_html",
    "render_markdown",
    "report_dir",
    "run_report",
]
