"""Render a :class:`ReportDocument` to markdown and HTML (stage 5).

Both renderers walk the same document model so the two formats stay faithful to
each other: a :class:`Table` becomes a GitHub-flavoured pipe table in markdown and
a ``<table>`` in HTML from the *same* DataFrame, never two hand-written variants.
Every cell is formatted to a display string once (:func:`_display_frame`) before
either renderer touches it, so the two formats show byte-identical content;
tabulate and pandas otherwise diverge on float precision and missing-value text.
Output uses only saved dates, so a re-render of unchanged artifacts is
byte-stable (AGENTS.md determinism invariant).
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import replace
from datetime import date

from bench.plotting.interactive import plotly_javascript
from bench.reports.assets import FRONT_PAGE_JS, REPORT_COMMON_JS, REPORT_HELP_JS
from bench.reports.game_log import GAME_LOG_JS, game_log_markdown, latest_game_card_html, render_game_log_page
from bench.reports.content import (
    render_citation_html,
    render_citation_markdown,
    render_footer_html,
    render_summary_html,
    render_help_html,
    render_view_group,
    metadata_text,
    report_summary,
)

from .curves import CURVE_CHART_JS, render_curve_views_html
from .controlled_seed import (
    CONTROLLED_SEED_DIR,
    CONTROLLED_SEED_OVERVIEW,
    chapter_seeds,
    render_controlled_seed_site,
)
from .heatmap import render_heatmap_html, render_heatmap_md
from .front_page import in_sentence
from .front_svg import cost_svg, leaderboard_svgs, shape_icon, styles_svg
from .model import FrontChart, FrontPage, ReportDocument, Section, Table


def _annex_group(doc: ReportDocument):
    """The controlled-seed chapter group, or ``None`` when the report has none."""
    if doc.controlled_seed is None:
        return None
    for group in doc.groups:
        if group.key == CONTROLLED_SEED_DIR:
            return group
    return None


def _slug(text: str) -> str:
    """A GitHub-style anchor slug for TOC links."""
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _build_anchors(doc: ReportDocument) -> dict:
    """Assign a unique anchor to every group and section, keyed by object id.

    Group and section slugs share one HTML id namespace, so a family titled
    "Calibration" and a section id ``calibration`` would otherwise both emit
    ``id="calibration"`` (invalid HTML, mis-targeted TOC links). Namespacing the
    two and deduplicating collisions deterministically (``-2``, ``-3``) keeps
    every anchor unique; keying by object id also survives duplicate section ids.
    The TOC and the headings read the same map, so links always resolve.
    """
    seen: set[str] = set()
    anchors: dict = {}

    def uniq(base: str) -> str:
        base = base or "x"
        cand, i = base, 2
        while cand in seen:
            cand = f"{base}-{i}"
            i += 1
        seen.add(cand)
        return cand

    for group in doc.groups:
        anchors[id(group)] = uniq("family-" + _slug(group.title))
        for section in group.sections:
            anchors[id(section)] = uniq("section-" + _slug(section.id))
    return anchors


def _metadata_line(metadata: dict) -> str:
    """A compact ``key: value`` rendering of an analysis's metadata, or ``""``."""
    return metadata_text(metadata)


def _section_summary(section: Section) -> str:
    """Return the result sentence shown in overview and detailed views."""
    return report_summary(section.summary.strip(), section.metadata) or "No result summary was produced for this analysis."


def _section_details(section: Section) -> str:
    return "\n\n".join(filter(None, (
        section.description,
        f"Module: {section.module}",
        _metadata_line(section.metadata),
    )))


def _announcement_date(value: str) -> str:
    recorded = date.fromisoformat(value)
    return f"{recorded:%b} {recorded.day}, {recorded.year}"


def _render_announcements(doc: ReportDocument) -> list[str]:
    parts = []
    for announcement in doc.announcements:
        kind = _html.escape(announcement.kind, quote=True)
        parts.append(f'<aside class="announcement" data-announcement="{kind}">')
        parts.append(f'<p class="eyebrow">{_html.escape(announcement.title)}')
        if announcement.date:
            parts.append(
                f' &middot; <time datetime="{_html.escape(announcement.date, quote=True)}">'
                f'{_announcement_date(announcement.date)}</time>'
            )
        parts.append(f'</p><p>{_md_inline_to_html(announcement.text)}</p></aside>')
    return parts


# ── markdown ──────────────────────────────────────────────────────────────────
def render_markdown(doc: ReportDocument) -> str:
    anchors = _build_anchors(doc)
    lines: list[str] = []
    lines.append(f"# {doc.title}")
    lines.append("")
    if doc.description:
        lines.append(f"*{doc.description}*")
        lines.append("")
    if doc.intro:
        lines.append(doc.intro)
        lines.append("")
    if doc.front_page is not None:
        _render_front_page_md(doc, anchors, lines)

    for announcement in doc.announcements:
        heading = announcement.title
        if announcement.date:
            heading += f" · {_announcement_date(announcement.date)}"
        lines.extend([f"> **{heading}**", ">", f"> {announcement.text}", ""])

    if doc.overview_sections and doc.front_page is None:
        family_for = {
            id(section): group
            for group in doc.groups
            for section in group.sections
        }
        lines.append("## Overview")
        lines.append("")
        for section in doc.overview_sections:
            group = family_for[id(section)]
            lines.append(
                f"- **{section.title}** ({group.title}): {_section_summary(section)} "
                f"[Details](#{anchors[id(section)]})"
            )
        lines.append("")

    # Table of contents.
    lines.append("## Contents")
    lines.append("")
    for group in doc.groups:
        lines.append(f"- [{group.title}](#{anchors[id(group)]})")
        for section in group.sections:
            lines.append(f"  - [{section.title}](#{anchors[id(section)]})")
    lines.append("")

    for group in doc.groups:
        lines.append(f'<a id="{anchors[id(group)]}"></a>')
        lines.append(f"## {group.title}")
        lines.append("")
        if group.summary:
            lines.append(group.summary)
            lines.append("")
        for section in group.sections:
            lines.append(f'<a id="{anchors[id(section)]}"></a>')
            _render_section_md(section, lines)
    if doc.game_log is not None:
        lines.extend(game_log_markdown(doc.game_log))
    lines.extend([render_citation_markdown(doc.benchmark_citation), ""])
    if doc.footer.strip():
        lines.extend(["---", "", doc.footer.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _plain_mode(row: dict) -> str:
    return row["mode"] or "-"


def _render_front_page_md(doc: ReportDocument, anchors: dict, lines: list[str]) -> None:
    """The plain-language front page as Markdown: text, a top-10 table, links, and cards."""
    front = doc.front_page
    lines.extend([f"## {front.headline}", ""])
    for paragraph in front.body:
        lines.extend([paragraph, ""])
    if front.facts:
        lines.extend([" · ".join(f"**{fact.value}** {fact.label}" for fact in front.facts), ""])
    if front.leader:
        lines.extend([f"Leading right now: {front.leader}.", ""])
    board = front.chart("leaderboard")
    if board is not None:
        lines.extend([f"### {board.title}", ""])
        if board.text:
            lines.extend([board.text, ""])
        lines.extend(["| # | Player | Deciding | Rating | Likely range |", "| ---: | --- | --- | ---: | --- |"])
        for row in board.rows[: int(board.extra.get("top", 10))]:
            span = f"{row['low']:.0f} to {row['high']:.0f}" if row["high"] > row["low"] else "-"
            name = row["name"].replace("|", "\\|")
            lines.append(f"| {row['rank']} | {name} | {_plain_mode(row)} | {row['elo']:.0f} | {span} |")
        lines.append("")
    if front.projects or front.links:
        lines.extend([f"### {front.projects_title}", ""])
        if front.projects_text:
            lines.extend([front.projects_text, ""])
        for project in front.projects:
            lines.append(f"- **{project.name}**: {project.text} [{project.label}]({project.url})")
        for link in front.links:
            lines.append(f"- [{link.label}]({link.url})")
        lines.append("")
    if front.cards:
        lines.extend(["### Key findings", ""])
        for card in front.cards:
            lines.append(f"- **{card.label}**: {card.text} [Details](#{anchors[id(card.section)]})")
        lines.append("")


def _render_section_md(section: Section, lines: list[str]) -> None:
    lines.append(f"### {section.title}")
    lines.append("")
    lines.extend(["<details>", "<summary>Technical details</summary>", ""])
    lines.append(f"*Module: `{section.module}`*")
    if section.description:
        lines.append("")
        lines.append(f"*{section.description}*")
    meta = _metadata_line(section.metadata)
    if meta:
        lines.append("")
        lines.append(f"*{meta}*")
    lines.extend(["", "</details>"])
    lines.append("")
    lines.append(_section_summary(section))
    lines.append("")
    if section.empty:
        lines.append(
            "_This analysis produced no artifacts for the given inputs "
            "(e.g. a controlled-design view on an uncontrolled run)._"
        )
        lines.append("")
        return
    # Artifacts no view claims follow the views, so a section's tabs lead.
    for view in section.views:
        lines.append(f"**{view.label}**")
        lines.append("")
        _render_artifacts_md(view.figures, view.tables, lines)
    _render_artifacts_md(section.figures, section.tables, lines)
    if section.downloads:
        lines.append("**Downloads and supporting files**")
        lines.append("")
        for dl in section.downloads:
            lines.append(f"- [{dl.label}]({dl.rel_path})")
        lines.append("")


def _render_artifacts_md(figures, tables, lines: list[str]) -> None:
    for figure in figures:
        if figure.interactive:
            caption = figure.caption.replace("[", "(").replace("]", ")")
            lines.append(
                f"[Figure: {caption} (interactive HTML)]({figure.rel_path})"
            )
            lines.append("")
            continue
        # Bracket/paren in a caption would break the inline image link; the path
        # is artifact-controlled (assets/<id>/<name>.png) so only the caption needs it.
        alt = figure.caption.replace("[", "(").replace("]", ")")
        lines.append(f"![{alt}]({figure.rel_path})")
        lines.append("")
        lines.append(f"*Figure: {figure.caption}*")
        lines.append("")
    for table in tables:
        _render_table_md(table, lines)


def _escape_md_cells(frame):
    """Escape ``|`` in string cells/headers so a value can't break the pipe table
    (tabulate does not escape them); pandas' HTML path already escapes, so this
    keeps the two formats faithful."""
    out = frame.copy()
    out.columns = [str(c).replace("|", "\\|") for c in out.columns]
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda v: v.replace("|", "\\|") if isinstance(v, str) else v
            )
    return out


def _format_cell(value) -> str:
    """One display string per cell, shared by both renderers.

    Missing values render as ``""`` (never the literal ``nan`` tabulate prints for a
    float NaN in an object column), and floats use ``g``, matching tabulate's own
    default so the markdown output is unchanged while the HTML side stops diverging
    (pandas' ``to_html`` otherwise uses full-precision/scientific formatting).
    """
    if isinstance(value, float):
        if value != value:  # NaN: the only value not equal to itself
            return ""
        return f"{value:g}"
    if value is None:
        return ""
    if isinstance(value, str) and value == "Vanilla":
        return "VPAI"
    return str(value)


def _display_frame(frame):
    """Format every cell to a display string (see :func:`_format_cell`) so the
    markdown and HTML tables are byte-for-byte faithful. Full-precision values are
    always one click away via the linked ``full CSV``."""
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].map(_format_cell)
    out.columns = ["VPAI" if str(col) == "Vanilla" else col for col in out.columns]
    return out


def _render_table_md(table: Table, lines: list[str]) -> None:
    if table.heatmap is not None:
        lines.extend(render_heatmap_md(table.frame, table.heatmap))
    else:
        lines.append(f"**{table.name}**")
        lines.append("")
        lines.append(_escape_md_cells(_display_frame(table.frame)).to_markdown(index=False))
        lines.append("")
    note = []
    if table.truncated:
        note.append(
            f"Showing {table.n_shown_rows} of {table.n_total_rows} rows"
        )
    if table.rel_csv:
        link = f"[full CSV]({table.rel_csv})"
        note.append(link if not note else f"; {link}")
    if note:
        lines.append(f"_{' '.join(note)}._")
        lines.append("")


# ── html ──────────────────────────────────────────────────────────────────────
_HTML_STYLE = """
@import url("https://fonts.googleapis.com/css2?family=EB+Garamond:wght@500;600;700&display=swap");
:root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #18202a; background: #f7f8fa; }
* { box-sizing: border-box; }
body { margin: 0; line-height: 1.55; font-size: 14px; }
a { color: #175ca8; }
a:hover { color: #0b3d73; }
.skip-link { position: fixed; top: .5rem; left: .5rem; z-index: 20; padding: .5rem .75rem; background: white; transform: translateY(-180%); }
.skip-link:focus { transform: none; }
.sidebar { position: fixed; inset: 0 auto 0 0; width: 19rem; overflow-y: auto; padding: 1.5rem 1rem; color: #e9eef5; background: #172536; }
.site-title { display: block; margin: 0 0 1rem; color: white; font-family: "EB Garamond", Garamond, Georgia, serif; font-size: 1.3rem; font-weight: 600; line-height: 1.25; text-decoration: none; }
.sidebar ul { margin: 0; padding: 0; list-style: none; }
.sidebar ul ul { margin: .3rem 0 .8rem .75rem; border-left: 1px solid #4b5b6d; padding-left: .75rem; }
.sidebar a { display: block; border-radius: .3rem; padding: .25rem .45rem; color: #cbd7e5; text-decoration: none; }
.sidebar a:hover, .sidebar a[aria-current="page"] { color: white; background: #2b4159; }
.content { max-width: 78rem; margin-left: 19rem; padding: 2.25rem 3rem 4rem; }
h1, h2, h3 { font-family: "EB Garamond", Garamond, Georgia, serif; font-weight: 600; line-height: 1.2; text-wrap: balance; }
h1 { margin-top: 0; border-bottom: 2px solid #cfd6df; padding-bottom: .45rem; font-size: 2.3rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #dfe4ea; padding-bottom: .25rem; font-size: 1.75rem; }
h3 { margin-top: 1.75rem; font-size: 1.35rem; }
.eyebrow { margin-bottom: .35rem; color: #637083; font-size: .85rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 1rem; margin-top: 1.5rem; }
.overview-card { border: 1px solid #d9e0e8; border-radius: .55rem; padding: 1rem; background: white; box-shadow: 0 1px 2px rgb(0 0 0 / 6%); }
.overview-card h2 { margin: 0 0 .25rem; border: 0; padding: 0; font-size: 1.35rem; }
.overview-card p { margin: .45rem 0; }
.announcement { margin-top: 1rem; border-left: 4px solid #377ca8; border-radius: .3rem; padding: .65rem 1rem; background: #eef5fa; overflow-wrap: anywhere; }
.announcement[data-announcement="testing"] { border-left-color: #b08732; background: #faf6eb; }
.announcement p { margin: .25rem 0; }
.module { color: #445164; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.meta, .caption { color: #687486; font-size: .88rem; }
.report-help { display: inline-block; position: relative; margin-left: .35rem; vertical-align: middle; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: .85rem; font-weight: 400; line-height: 1.5; }
.help-toggle { display: inline-flex; align-items: center; justify-content: center; width: 1.25rem; height: 1.25rem; border: 1px solid #7a8798; border-radius: 50%; padding: 0; color: #445164; background: #fff; font: inherit; cursor: help; }
.help-toggle:focus-visible { outline: 2px solid #175ca8; outline-offset: 2px; }
.help-text { display: none; position: absolute; top: 100%; left: 0; z-index: 40; width: max-content; max-width: min(30rem, 80vw); max-height: 65vh; overflow: auto; padding: .7rem .85rem; border: 1px solid #a8b3c2; border-radius: .35rem; color: #18202a; background: #fffdf5; box-shadow: 0 2px 8px rgb(0 0 0 / 18%); white-space: pre-line; overflow-wrap: anywhere; text-align: left; font-style: normal; }
.report-help:not(.help-dismissed):hover .help-text, .report-help:not(.help-dismissed):focus-within .help-text, .report-help.help-open .help-text { display: block; }
.caption { font-style: italic; }
.report-footer { margin-top: 2rem; border-top: 1px solid #d8dee6; padding-top: 1rem; color: #687486; font-size: .88rem; }
pre { max-width: 100%; overflow-x: auto; padding: 1rem; background: #f0f3f7; }
.empty { color: #778294; font-style: italic; }
figure { margin: 1.25rem 0 2rem; }
img { max-width: 100%; height: auto; border: 1px solid #e2e6eb; background: white; }
.interactive-figure { width: 100%; min-height: 700px; border: 1px solid #e2e6eb; background: white; }
.interactive-figure-link { margin: .4rem 0 0; font-size: .88rem; }
.table-scroll { max-width: 100%; overflow-x: auto; border: 1px solid #d8dee6; border-radius: .35rem; background: white; }
table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: .9rem; }
th, td { border-bottom: 1px solid #e0e5eb; padding: .4rem .65rem; text-align: right; white-space: nowrap; }
th { background: #f0f3f7; }
td:first-child, th:first-child { text-align: left; }
.view-switch { display: none; gap: 0; margin: .75rem 0 .25rem; border: 1px solid #a8b3c2; border-radius: .4rem; overflow: hidden; }
.views-ready .view-switch { display: inline-flex; }
.view-button { border: 0; border-right: 1px solid #a8b3c2; padding: .35rem .8rem; color: #29384b; background: white; font: inherit; cursor: pointer; }
.view-button:last-child { border-right: 0; }
.view-button[aria-pressed="true"] { color: white; background: #29507a; }
.view-button:focus-visible { outline: 2px solid #175ca8; outline-offset: -2px; }
.views-ready .view-heading { display: none; }
details.downloads { margin: 1.25rem 0 2rem; border: 1px solid #d8dee6; border-radius: .4rem; padding: .65rem .8rem; background: #fbfcfd; }
details.downloads summary { cursor: pointer; font-weight: 650; }
details.downloads ul { margin-bottom: .25rem; }
/* controlled-seed report */
.sr-only { position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; border: 0; clip: rect(0 0 0 0); overflow: hidden; white-space: nowrap; }
.controlled-content { max-width: 92rem; margin-left: auto; margin-right: auto; padding: 2.25rem 2rem 4rem; }
.page-nav { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin-bottom: 1.25rem; }
.page-nav .return-link { font-weight: 650; }
.warning { margin: .6rem 0; border-left: 4px solid #c28e21; border-radius: .25rem; padding: .5rem .75rem; color: #6b5312; background: #fdf6e3; }
.heat-figure { margin: 1rem 0 1.75rem; }
.heat-figure figcaption { margin-bottom: .4rem; color: #445164; font-size: .92rem; font-weight: 650; }
table.heatmap { font-size: .85rem; }
table.heatmap th, table.heatmap td { text-align: center; white-space: nowrap; }
table.heatmap .row-label { text-align: left; white-space: normal; min-width: 13rem; }
.heat-cell { padding: 0; }
.heat-cell-link a { display: block; padding: .38rem .55rem; color: inherit; text-decoration: none; }
.heat-cell a:hover, .heat-cell a:focus-visible { outline: 2px solid #175ca8; outline-offset: -2px; }
.heat-cell-empty { background: transparent; }
.heat-cell-value { padding: .38rem .55rem; }
table.heatmap th.heat-group { border-left: 2px solid #b6c1cd; color: #445164; font-size: .8rem; letter-spacing: .03em; }
table.heatmap .row-label { position: sticky; left: 0; z-index: 1; background: #f0f3f7; }
table.heatmap tbody .row-label { background: #fbfcfd; }
table.heatmap .col-divider { border-right: 2px solid #b6c1cd; }
tbody.vanilla-body tr.vanilla-row th, tbody.vanilla-body tr.vanilla-row td { border-top: 3px double #8d99a9; border-bottom: 3px double #8d99a9; background: #f3f0e8; }
tbody.vanilla-body tr.vanilla-row .heat-cell { background-image: linear-gradient(rgba(255,255,255,.35), rgba(255,255,255,.35)); }
.heat-tooltip { position: absolute; z-index: 30; display: none; max-width: 22rem; border: 1px solid #445164; border-radius: .3rem; padding: .4rem .6rem; color: #18202a; background: #fffdf5; box-shadow: 0 2px 8px rgb(0 0 0 / 18%); font-size: .82rem; line-height: 1.35; pointer-events: none; }
.heat-tooltip > span { display: block; }
.heat-tooltip .tip-title { font-weight: 700; }
.heat-tooltip .tip-sub { color: #5b6778; }
.heat-tooltip .tip-grid { display: grid; grid-template-columns: auto auto; gap: .1rem .75rem; margin-top: .3rem; }
.heat-tooltip .tip-label { color: #5b6778; }
.heat-tooltip .tip-number { color: #6b3fa0; font-variant-numeric: tabular-nums; }
.heat-tooltip .tip-number-positive { color: #237a45; }
.heat-tooltip .tip-number-negative { color: #c0392b; }
.heat-tooltip .tip-number-unsigned { color: #6b3fa0; }
.heat-tooltip .tip-note { color: #5b6778; font-variant-numeric: tabular-nums; }
.heat-tooltip .tip-value > .tip-number + .tip-note { margin-left: .4rem; }
.heat-tooltip .tip-strong { font-weight: 700; }
.heat-tooltip .tip-list { margin: .25rem 0 0; padding-left: 1.1rem; }
.heat-tooltip .tip-list li { margin: .1rem 0; }
.heat-tooltip .tip-list li::marker { color: #8d99a9; }
.focus-legend { display: flex; flex-wrap: wrap; gap: .9rem; margin: .25rem 0 1.5rem; padding: 0; list-style: none; font-size: .85rem; }
.focus-legend .swatch { display: inline-block; width: .85rem; height: .85rem; margin-right: .3rem; border: 1px solid #44516433; border-radius: .15rem; vertical-align: -0.1em; }
.chart-controls { display: flex; flex-wrap: wrap; gap: .5rem 1.1rem; align-items: center; margin: .75rem 0; }
.chart-controls .controls-label { color: #445164; font-weight: 650; }
.chart-controls .strategist-check { display: inline-flex; align-items: center; gap: .3rem; }
.chart-controls .chart-presets { display: inline-flex; gap: .4rem; }
.chart-controls .chart-preset { padding: .1rem .55rem; border: 1px solid #a8b3c2; border-radius: .3rem; color: #18202a; background: #fff; font: inherit; font-size: .85rem; cursor: pointer; }
.chart-controls .chart-preset:hover { background: #eef2f6; }
.content.game-log { max-width: none; }
#game-log-table { width: 100%; }
#game-log-table th, #game-log-table td { vertical-align: top; }
#game-log-table th:nth-child(2), #game-log-table td:nth-child(2),
#game-log-table th:nth-child(6), #game-log-table td:nth-child(6),
#game-log-table th:nth-child(7), #game-log-table td:nth-child(7) {
  text-align: left; white-space: normal; overflow-wrap: anywhere;
}
.game-winner { display: block; color: #687486; font-size: .88em; }
.game-log-filters { display: flex; flex-wrap: wrap; gap: .7rem 1rem; align-items: center; margin: 1.2rem 0; }
.game-log-filters label { display: inline-flex; align-items: center; gap: .3rem; }
.game-log-filters select, .game-log-filters button { font: inherit; padding: .25rem; }
.game-log th button, th .sort-button { font: inherit; font-weight: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: pointer; text-align: inherit; }
th .sort-button:focus-visible { outline: 2px solid #175ca8; outline-offset: 2px; }
[aria-sort="descending"] > button::after { content: " ▾"; }
[aria-sort="ascending"] > button::after { content: " ▴"; }
.replay-link { white-space: nowrap; }
.muted { color: #64748b; }
.latest-game p:last-child { margin-bottom: 0; }
/* plain-language front page (report.intro) */
.hero h1 { margin: .2rem 0 1rem; border: 0; padding: 0; font-size: 2.75rem; line-height: 1.1; }
.hero .eyebrow { margin: 0; }
.lede { margin: 0 0 .8rem; font-size: 1.05rem; }
.section-lede { margin: 0 0 1rem; color: #445164; }
.term { border-bottom: 1.5px dotted #b08732; cursor: help; }
[data-tip]:focus-visible { outline: 2px solid #175ca8; outline-offset: 2px; }
.facts { display: flex; flex-wrap: wrap; gap: .6rem; margin: 1.4rem 0 0; padding: 0; list-style: none; }
.facts li { display: flex; flex-direction: column; min-width: 7.5rem; border: 1px solid #d9e0e8; border-radius: .45rem; padding: .55rem .9rem; background: white; cursor: help; }
.facts b { font-family: "EB Garamond", Garamond, Georgia, serif; font-size: 1.75rem; font-weight: 600; line-height: 1.1; font-variant-numeric: lining-nums tabular-nums; }
.facts span { color: #637083; font-size: .85rem; }
.leader { display: flex; flex-wrap: wrap; align-items: baseline; gap: .25rem .75rem; margin: 1.2rem 0 0; border-left: 4px solid #b08732; border-radius: .3rem; padding: .65rem 1rem; background: #faf6eb; }
.leader .eyebrow { margin: 0; color: #8a6a24; }
.front-section .chart-link { margin: .5rem 0 0; font-size: .9rem; }
.front-legend { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem 1.1rem; margin: 0 0 .75rem; padding: 0; list-style: none; color: #445164; font-size: .88rem; }
.front-legend li { display: inline-flex; align-items: center; gap: .4rem; }
.front-legend .sw { display: inline-block; width: .9rem; height: .9rem; border-radius: .15rem; }
.sw.up { background: #2c6aa8; }
.sw.down { background: #b4885a; }
.front-legend .sw.whisk { width: 1.1rem; height: 0; border-top: 2px solid #7a8798; border-radius: 0; }
.sw.dom, .seg.dom rect { background: #b5524a; fill: #b5524a; }
.sw.culture, .seg.culture rect { background: #7a55b0; fill: #7a55b0; }
.sw.diplo, .seg.diplo rect { background: #2f7fb8; fill: #2f7fb8; }
.sw.science, .seg.science rect { background: #2f8a6f; fill: #2f8a6f; }
.front-chart { overflow-x: auto; border: 1px solid #d8dee6; border-radius: .35rem; padding: 1rem 1rem .5rem; background: white; }
.front-chart svg { display: block; width: 100%; min-width: 720px; height: auto; font-size: 13px; }
.front-chart svg text { fill: #18202a; }
.front-chart .tick, .front-chart .axis, .front-chart .rank, .front-chart .sub, .front-chart .val, .front-chart .base-label, .front-chart .fit-label { fill: #687486; }
.front-chart .tick, .front-chart .val { font-size: 11.5px; font-variant-numeric: tabular-nums; }
.front-chart .base-label, .front-chart .fit-label { font-size: 12px; }
.front-chart .fit-label { font-style: italic; }
.front-chart .grid { stroke: #e9edf2; stroke-width: 1; }
.front-chart .base { stroke: #18202a; stroke-width: 1.5; stroke-dasharray: 4 3; }
.front-chart .fit { stroke: #687486; stroke-width: 1.5; stroke-dasharray: 6 4; }
.lb-row, .pt, .seg { outline: none; }
.lb-row .hit { fill: transparent; }
.lb-row:hover .hit, .lb-row:focus .hit { fill: #eef2f6; }
.lb-row.up .bar { fill: #2c6aa8; }
.lb-row.down .bar { fill: #b4885a; }
.lb-row.ref .dot { fill: #18202a; }
.lb-row.ref .name, .front-chart .refname { font-weight: 700; }
.lb-row .whisk { stroke: #18202a; stroke-opacity: .45; stroke-width: 1.5; }
.mode { cursor: help; }
.mode rect { fill: white; stroke: #c3ccd7; }
.mode.m1 rect, .mode.m2 rect, .mode.m3 rect { fill: #faf6eb; stroke: #b08732; }
.front-chart .mode text { fill: #445164; font-size: 11px; }
details.more-rows { margin-top: .5rem; }
details.more-rows summary { padding: .25rem 0; color: #175ca8; font-weight: 650; cursor: pointer; }
.pt circle, .pt rect, .pt path { fill: var(--c); stroke: white; stroke-width: 1.5; }
.pt:hover circle, .pt:hover rect, .pt:hover path, .pt:focus circle, .pt:focus rect, .pt:focus path { stroke: #18202a; }
.front-chart .pt-label { font-size: 12px; font-weight: 650; pointer-events: none; paint-order: stroke; stroke: white; stroke-width: 3px; stroke-linejoin: round; }
.pt-label.pick { display: none; }
.cost-chart.picking .pt:not(.on) { opacity: .15; }
.cost-chart.picking .pt-label.rest { display: none; }
.cost-chart.picking .pt-label.pick.on { display: inline; }
.series-key { display: inline-flex; align-items: center; gap: .4rem; border: 1px solid #c3ccd7; border-radius: 1rem; padding: .1rem .6rem; color: #18202a; background: white; font: inherit; font-size: .85rem; cursor: pointer; }
.series-key:hover { background: #eef2f6; }
.series-key[aria-pressed="true"] { border-color: #18202a; font-weight: 650; }
.series-key .sw { width: .75rem; height: .75rem; border-radius: 50%; }
.shape-key svg { width: .9rem; height: .9rem; fill: #687486; }
.front-chart .seg text { fill: white; font-size: 11.5px; font-weight: 650; font-variant-numeric: tabular-nums; pointer-events: none; }
.seg:hover rect, .seg:focus rect { stroke: #18202a; stroke-width: 2; }
.projects { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; }
.project { display: flex; flex-direction: column; gap: .35rem; min-width: 0; border-top: 3px solid #b08732; border-radius: 0 0 .45rem .45rem; padding: .9rem 1.1rem; background: white; box-shadow: 0 1px 2px rgb(0 0 0 / 6%); }
.project h3 { margin: 0; }
.project p { flex: 1; margin: 0; }
.also { margin: 1rem 0 0; color: #445164; }
.findings { display: grid; grid-template-columns: repeat(auto-fill, minmax(17rem, 1fr)); gap: 1rem; }
.finding { display: flex; flex-direction: column; min-width: 0; border: 1px solid #d9e0e8; border-radius: .55rem; padding: 1rem; background: white; box-shadow: 0 1px 2px rgb(0 0 0 / 6%); }
.finding p { flex: 1; margin: .5rem 0; }
.finding-head { display: flex; align-items: center; gap: .55rem; }
.finding-head h3 { flex: 1; margin: 0; font-size: 1.25rem; }
.finding-head .report-help { margin-left: 0; }
.finding .icon svg { display: block; width: 1.4rem; height: 1.4rem; fill: #b08732; stroke: #b08732; stroke-width: 1.2; stroke-linejoin: round; }
.finding .icon svg .cut { fill: none; stroke: white; stroke-width: 1.8; stroke-linecap: round; }
.finding .icon svg .open { fill: none; stroke-width: 2; stroke-linecap: round; }
.finding > a { font-size: .9rem; font-weight: 650; text-decoration: none; }
@media (max-width: 820px) {
  .sidebar { position: static; width: auto; max-height: none; }
  .sidebar ul ul { display: none; }
  .content { margin-left: 0; padding: 1.5rem 1rem 3rem; }
  .hero h1 { font-size: 2.1rem; }
}
""".strip() + "\n"


def render_stylesheet() -> str:
    """Return the shared, deterministic stylesheet for the HTML report site."""
    return _HTML_STYLE


def _family_filenames(doc: ReportDocument) -> dict[int, str]:
    used = {"report"}
    filenames: dict[int, str] = {}
    for group in doc.groups:
        if doc.controlled_seed is not None and group.key == CONTROLLED_SEED_DIR:
            # The chapter's page is the heatmap overview inside its own folder.
            filenames[id(group)] = CONTROLLED_SEED_OVERVIEW
            used.add(CONTROLLED_SEED_DIR)
            continue
        base = _slug(group.key) or "family"
        candidate, index = base, 2
        while candidate in used:
            candidate = f"{base}-{index}"
            index += 1
        used.add(candidate)
        filenames[id(group)] = f"{candidate}.html"
    return filenames


def _render_navigation(
    doc: ReportDocument,
    anchors: dict,
    filenames: dict[int, str],
    active: str,
    prefix: str = "",
) -> list[str]:
    """The site sidebar. ``prefix`` rebases the links for pages rendered inside
    a subfolder (the controlled-seed chapter)."""
    annex = _annex_group(doc)
    parts = ['<aside class="sidebar">']
    parts.append(
        f'<a class="site-title" href="{_html.escape(prefix)}index.html">'
        f"{_html.escape(doc.title)}</a>"
    )
    parts.append('<nav aria-label="Report navigation"><ul>')
    current = ' aria-current="page"' if active == "report" else ""
    parts.append(f'<li><a href="{_html.escape(prefix)}index.html"{current}>Overview</a></li>')
    for group in doc.groups:
        filename = filenames[id(group)]
        current = ' aria-current="page"' if active == group.key else ""
        parts.append(
            f'<li><a href="{_html.escape(prefix + filename)}"{current}>'
            f"{_html.escape(group.title)}</a><ul>"
        )
        if group is annex:
            # The chapter page is the section, so its sub-entries are the seeds.
            for seed in chapter_seeds(doc.controlled_seed):
                parts.append(
                    f'<li><a href="{_html.escape(prefix + filename)}#seed-{seed}">'
                    f"Seed {seed}</a></li>"
                )
        else:
            for section in group.sections:
                parts.append(
                    f'<li><a href="{_html.escape(prefix + filename)}#{anchors[id(section)]}">'
                    f"{_html.escape(section.title)}</a></li>"
                )
        parts.append("</ul></li>")
    parts.append("</ul></nav></aside>")
    return parts


def _page_start(doc: ReportDocument, page_title: str) -> list[str]:
    return [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_html.escape(page_title)}</title>",
        '<link rel="stylesheet" href="assets/report.css">',
        '<script src="assets/report-help.js" defer></script>',
        *(['<script src="assets/report-common.js" defer></script>'] if doc.game_log is not None else []),
        "</head><body>",
        '<a class="skip-link" href="#main-content">Skip to content</a>',
    ]


def _section_targets(doc: ReportDocument, anchors: dict, filenames: dict, annex) -> dict:
    """Each section's page link: its family page anchor, or the chapter page."""
    targets = {}
    for group in doc.groups:
        for section in group.sections:
            if group is annex:
                targets[section.id] = filenames[id(group)]
            else:
                targets[section.id] = f"{filenames[id(group)]}#{anchors[id(section)]}"
    return targets


# Small line icons for the finding cards, drawn in the accent color.
_CARD_ICONS = {
    "crown": '<path d="M3 18h18l-1.5-10-4.5 4-3-7-3 7-4.5-4z"/>',
    "flag": '<path d="M5 21V4h11l-1.5 4L16 12H7v9z"/>',
    "eye": '<path d="M12 5C6 5 2 12 2 12s4 7 10 7 10-7 10-7-4-7-10-7zm0 11a4 4 0 110-8 4 4 0 010 8z"/>',
    "coin": '<circle cx="12" cy="12" r="9"/><path class="cut" d="M12 7v10M9.5 9.5h4a1.5 1.5 0 010 3h-3a1.5 1.5 0 000 3h4"/>',
    "dial": ('<path class="open" d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/>'
             '<circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>'),
    "hand": '<path d="M2 12l5-5 4 2 4-2 7 5-6 6-3-2-2 2-4-3z"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle class="cut" cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="2"/>',
    "scroll": '<path d="M6 3h12v15a3 3 0 01-3 3H6a3 3 0 01-3-3v-2h3z"/><path class="cut" d="M9 8h6M9 12h6"/>',
    "dot": '<circle cx="12" cy="12" r="6"/>',
}

_RANGE_TIP = (
    "We are fairly sure (95%) the true rating lies inside this range. "
    "When two ranges overlap a lot, more games could swap their order."
)


def _term_html(text: str, tip: str) -> str:
    return f'<span class="term" tabindex="0" data-tip="{_html.escape(tip, quote=True)}">{text}</span>'


def _with_terms(html_text: str, glossary: dict[str, str]) -> str:
    """Underline the first use of each glossary phrase, outside tags and links."""
    pieces = re.split(r"(<[^>]+>)", html_text)
    marks: list[tuple[str, str]] = []  # marked spans, swapped in last so later phrases skip them
    in_link = False
    for index, piece in enumerate(pieces):
        if piece.startswith("<"):
            if re.match(r"<a[\s>]", piece):
                in_link = True
            elif piece.startswith("</a"):
                in_link = False
            continue
        if in_link:
            continue
        for phrase, tip in glossary.items():
            if any(phrase == used for used, _ in marks):
                continue
            pattern = r"(?<![\w\0])" + re.escape(_html.escape(phrase, quote=False)) + r"(?![\w\0])"
            match = re.search(pattern, piece, flags=re.IGNORECASE)
            if match is None:
                continue
            marks.append((phrase, _term_html(match.group(0), tip)))
            piece = f"{piece[: match.start()]}\0{len(marks) - 1}\0{piece[match.end():]}"
        pieces[index] = piece
    return re.sub("\0(\\d+)\0", lambda m: marks[int(m.group(1))][1], "".join(pieces))


def _prose_html(text: str, glossary: dict[str, str]) -> str:
    return _with_terms(_md_inline_to_html(text), glossary)


def _legend_item(swatch: str, label_html: str, tip: str = "") -> str:
    attrs = f' tabindex="0" data-tip="{_html.escape(tip, quote=True)}"' if tip else ""
    return f'<li{attrs}><span class="sw {swatch}"></span>{label_html}</li>'


def _chart_link(chart: FrontChart, targets: dict) -> str:
    target = targets.get(chart.stage)
    if not target:
        return ""
    return f'<p class="chart-link"><a href="{_html.escape(target)}">Full results →</a></p>'


def _render_front_chart(chart: FrontChart, front: FrontPage, targets: dict) -> list[str]:
    parts = [f'<section class="front-section" aria-labelledby="front-{chart.kind}">']
    parts.append(f'<h2 id="front-{chart.kind}">{_html.escape(chart.title)}</h2>')
    if chart.text:
        parts.append(f'<p class="section-lede">{_prose_html(chart.text, front.glossary)}</p>')
    if chart.kind == "leaderboard":
        reference = _html.escape(in_sentence(chart.extra.get("reference_name") or "reference"))
        parts.append('<ul class="front-legend">')
        parts.append(_legend_item("up", f"Stronger than the {reference}"))
        parts.append(_legend_item("down", "Weaker"))
        parts.append(_legend_item("whisk", '<span class="term">likely range</span>', _RANGE_TIP))
        parts.append("</ul>")
        top, rest = leaderboard_svgs(chart)
        parts.append(f'<div class="front-chart">{top}</div>')
        if rest:
            n_rest = len(chart.rows) - int(chart.extra.get("top", 10))
            parts.append(
                f'<details class="more-rows"><summary>Show the other {n_rest}</summary>'
                f'<div class="front-chart">{rest}</div></details>'
            )
    elif chart.kind == "cost":
        parts.append('<div class="cost-chart">')
        parts.append('<ul class="front-legend">')
        for series in chart.extra.get("series") or []:
            name = _html.escape(series["name"], quote=True)
            parts.append(
                f'<li><button type="button" class="series-key" data-series="{name}" aria-pressed="false">'
                f'<span class="sw" style="background:{_html.escape(series["color"], quote=True)}"></span>'
                f"{name}</button></li>"
            )
        for shape in chart.extra.get("shapes") or []:
            parts.append(
                f'<li class="shape-key">{shape_icon(shape["shape"])}{_html.escape(shape["label"])}</li>'
            )
        parts.append("</ul>")
        parts.append(f'<div class="front-chart">{cost_svg(chart)}</div>')
        parts.append("</div>")
    elif chart.kind == "styles":
        parts.append('<ul class="front-legend">')
        for goal in chart.extra.get("goals") or []:
            parts.append(_legend_item(goal["cls"], f'<span class="term">{_html.escape(goal["label"])}</span>', goal["tip"]))
        parts.append("</ul>")
        parts.append(f'<div class="front-chart">{styles_svg(chart)}</div>')
    parts.append(_chart_link(chart, targets))
    parts.append("</section>")
    return parts


def _render_front_page_html(doc: ReportDocument, targets: dict) -> list[str]:
    """The plain-language front page: story, charts, projects, findings, news."""
    front = doc.front_page
    glossary = front.glossary
    parts = ['<main class="content front" id="main-content">', '<header class="hero">']
    parts.append(f'<p class="eyebrow">{_html.escape(front.eyebrow or doc.title)}</p>')
    parts.append(f"<h1>{_html.escape(front.headline)}</h1>")
    for paragraph in front.body:
        parts.append(f'<p class="lede">{_prose_html(paragraph, glossary)}</p>')
    if front.facts:
        parts.append('<ul class="facts">')
        for fact in front.facts:
            attrs = f' tabindex="0" data-tip="{_html.escape(fact.tip, quote=True)}"' if fact.tip else ""
            parts.append(
                f"<li{attrs}><b>{_html.escape(fact.value)}</b><span>{_html.escape(fact.label)}</span></li>"
            )
        parts.append("</ul>")
    if front.leader:
        parts.append(
            '<p class="leader"><span class="eyebrow">Leading right now</span>'
            f"<span>{_prose_html(front.leader, glossary)}</span></p>"
        )
    parts.append("</header>")

    def chart(kind: str) -> None:
        found = front.chart(kind)
        if found is not None:
            parts.extend(_render_front_chart(found, front, targets))

    chart("leaderboard")
    if front.projects or front.links:
        parts.append('<section class="front-section" aria-labelledby="front-projects">')
        parts.append(f'<h2 id="front-projects">{_html.escape(front.projects_title)}</h2>')
        if front.projects_text:
            parts.append(f'<p class="section-lede">{_prose_html(front.projects_text, glossary)}</p>')
        if front.projects:
            parts.append('<div class="projects">')
            for project in front.projects:
                parts.append(
                    f'<article class="project"><h3>{_html.escape(project.name)}</h3>'
                    f"<p>{_md_inline_to_html(project.text)}</p>"
                    f'<a href="{_html.escape(project.url, quote=True)}">{_html.escape(project.label)} ↗</a></article>'
                )
            parts.append("</div>")
        if front.links:
            links = " · ".join(
                f'<a href="{_html.escape(link.url, quote=True)}">{_html.escape(link.label)} ↗</a>'
                for link in front.links
            )
            parts.append(f'<p class="also">{links}</p>')
        parts.append("</section>")
    chart("cost")
    chart("styles")

    if front.cards:
        parts.append('<section class="front-section" aria-labelledby="front-findings">')
        parts.append('<h2 id="front-findings">Key findings</h2>')
        parts.append(
            '<p class="section-lede">One sentence each. The <b>?</b> button shows the '
            "technical version, and the link opens the full results.</p>"
        )
        parts.append('<div class="findings">')
        for card in front.cards:
            section = card.section
            technical = "\n\n".join(filter(None, (
                f"{section.title}: {_section_summary(section).replace('**', '')}",
                _section_details(section),
            )))
            help_html = render_help_html(technical, f"help-front-{_slug(section.id)}", "Technical details")
            icon = _CARD_ICONS.get(card.icon, _CARD_ICONS["dot"])
            parts.append('<article class="finding">')
            parts.append(
                f'<div class="finding-head"><span class="icon"><svg viewBox="0 0 24 24" aria-hidden="true">{icon}</svg></span>'
                f"<h3>{_html.escape(card.label)}</h3>{help_html}</div>"
            )
            parts.append(f"<p>{_md_inline_to_html(card.text)}</p>")
            target = targets.get(section.id)
            if target:
                parts.append(f'<a href="{_html.escape(target)}">See details →</a>')
            parts.append("</article>")
        parts.append("</div></section>")

    news = _render_announcements(replace(doc, announcements=front.news))
    if doc.game_log is not None:
        news.append(latest_game_card_html(doc.game_log) or '<p><a href="games.html">Browse recent games</a></p>')
    if news:
        parts.append('<section class="front-section" aria-labelledby="front-news">')
        parts.append('<h2 id="front-news">Latest news</h2>')
        parts.extend(news)
        parts.append("</section>")
    parts.append(render_citation_html(doc.benchmark_citation))
    parts.append(render_footer_html(doc.footer))
    parts.append("</main>")
    return parts


def _render_overview_html(doc: ReportDocument, anchors: dict, filenames: dict, family_for: dict, annex) -> list[str]:
    """The classic overview page, used when ``report.intro`` is not configured."""
    parts: list[str] = []
    parts.append('<main class="content" id="main-content">')
    parts.append(f"<h1>{_html.escape(doc.title)}</h1>")
    if doc.description:
        parts.append(f'<p class="caption">{_html.escape(doc.description)}</p>')
    if doc.intro:
        parts.append(f"<p>{_md_inline_to_html(doc.intro)}</p>")
    parts.extend(_render_announcements(doc))
    if doc.game_log is not None:
        card = latest_game_card_html(doc.game_log)
        parts.append(card or '<p><a href="games.html">Browse recent games</a></p>')
    parts.append('<section aria-labelledby="overview-heading">')
    parts.append('<h2 id="overview-heading">Overview</h2>')
    parts.append('<div class="overview-grid">')
    for section in doc.overview_sections:
        group = family_for[id(section)]
        if group is annex:
            # The chapter page is the section's own page; no deep anchor exists.
            target = filenames[id(group)]
        else:
            target = f"{filenames[id(group)]}#{anchors[id(section)]}"
        parts.append('<article class="overview-card">')
        parts.append(f'<p class="eyebrow">{_html.escape(group.title)}</p>')
        help_html = render_help_html(_section_details(section), "help-" + anchors[id(section)] + "-overview")
        parts.append(f"<h2>{_html.escape(section.title)}{help_html}</h2>")
        parts.append(f"<p>{_md_inline_to_html(_section_summary(section))}</p>")
        parts.append(f'<p><a href="{_html.escape(target)}">View details</a></p>')
        parts.append("</article>")
    parts.append("</div></section>")
    parts.append(render_citation_html(doc.benchmark_citation))
    parts.append(render_footer_html(doc.footer))
    parts.append("</main></body></html>")
    return parts


def render_html_site(doc: ReportDocument) -> dict[str, str]:
    """Render the overview, one HTML page per represented analysis family, and
    (when the document carries it) the controlled-seed chapter: its heatmap
    overview plus one detail page per seed-player pair under ``controlled-seed/``."""
    anchors = _build_anchors(doc)
    filenames = _family_filenames(doc)
    annex = _annex_group(doc)
    family_for = {
        id(section): group
        for group in doc.groups
        for section in group.sections
    }
    pages: dict[str, str] = {"assets/report-help.js": REPORT_HELP_JS}

    parts = _page_start(doc, doc.title)
    parts.extend(_render_navigation(doc, anchors, filenames, "report"))
    if doc.front_page is not None:
        targets = _section_targets(doc, anchors, filenames, annex)
        parts.extend(_render_front_page_html(doc, targets))
        parts.append('<script src="assets/front-page.js" defer></script>')
        parts.append("</body></html>")
        pages["assets/front-page.js"] = FRONT_PAGE_JS
    else:
        parts.extend(_render_overview_html(doc, anchors, filenames, family_for, annex))
    pages["index.html"] = "\n".join(parts) + "\n"

    for group in doc.groups:
        if group is annex:
            continue  # the chapter's page is the heatmap overview, rendered below
        page_title = f"{group.title} | {doc.title}"
        parts = _page_start(doc, page_title)
        parts.extend(_render_navigation(doc, anchors, filenames, group.key))
        parts.append('<main class="content" id="main-content">')
        parts.append(f'<p class="eyebrow">{_html.escape(doc.title)}</p>')
        help_html = render_help_html(doc.description, "help-" + anchors[id(group)], "About this benchmark")
        parts.append(f'<h1 id="{anchors[id(group)]}">{_html.escape(group.title)}{help_html}</h1>')
        if group.summary:
            parts.append(f"<p>{_md_inline_to_html(group.summary)}</p>")
        for section in group.sections:
            _render_section_html(section, parts, anchors[id(section)], doc.model_tips)
        parts.append(render_footer_html(doc.footer))
        parts.append("</main>")
        if any(section.curve_charts for section in group.sections):
            # report-common.js is already in the head when the report has a game log.
            if doc.game_log is None:
                parts.append('<script src="assets/report-common.js" defer></script>')
            parts.append('<script src="assets/curve-chart.js" defer></script>')
            pages["assets/plotly.min.js"] = plotly_javascript()
            pages["assets/report-common.js"] = REPORT_COMMON_JS
            pages["assets/curve-chart.js"] = CURVE_CHART_JS
        parts.append("</body></html>")
        pages[filenames[id(group)]] = "\n".join(parts) + "\n"
    if doc.game_log is not None:
        pages["assets/report-common.js"] = REPORT_COMMON_JS
        pages["assets/game-log.js"] = GAME_LOG_JS
        pages["games.html"] = render_game_log_page(
            doc.game_log, _render_navigation(doc, anchors, filenames, "games")
        )
    if doc.controlled_seed is not None:
        navigation = _render_navigation(
            doc, anchors, filenames, CONTROLLED_SEED_DIR, prefix="../"
        )
        pages.update(render_controlled_seed_site(doc.controlled_seed, navigation))
    return pages


def render_html(doc: ReportDocument) -> str:
    """Compatibility wrapper returning the generated overview page."""
    return render_html_site(doc)["index.html"]


def _render_section_html(
    section: Section, parts: list[str], anchor: str, model_tips: dict[str, str] | None = None,
) -> None:
    parts.append(f'<section aria-labelledby="{anchor}">')
    help_html = render_help_html(_section_details(section), "help-" + anchor)
    parts.append(f'<h2 id="{anchor}">{_html.escape(section.title)}{help_html}</h2>')
    parts.append(f"<p>{_md_inline_to_html(_section_summary(section))}</p>")
    if section.empty:
        parts.append(
            '<p class="empty">This analysis produced no artifacts for the given '
            "inputs (e.g. a controlled-design view on an uncontrolled run).</p>"
        )
        parts.append("</section>")
        return
    if section.curve_charts:
        parts.append(render_curve_views_html(
            section.curve_charts, anchor, "assets/plotly.min.js", captions=True,
        ))
    # Artifacts no view claims follow the views, so a section's tabs lead.
    if section.views:
        _render_views_html(section, parts, anchor, model_tips)
    _render_artifacts_html(section.figures, section.tables, parts, anchor, model_tips)
    if section.downloads:
        parts.append('<details class="downloads">')
        parts.append(
            f"<summary>Downloads and supporting files ({len(section.downloads)})</summary>"
        )
        parts.append("<ul>")
        for dl in section.downloads:
            parts.append(
                f'<li><a href="{_html.escape(dl.rel_path)}">{_html.escape(dl.label)}</a></li>'
            )
        parts.append("</ul>")
        parts.append("</details>")
    parts.append("</section>")


def _render_artifacts_html(
    figures, tables, parts: list[str], anchor: str = "", model_tips: dict[str, str] | None = None,
) -> None:
    for figure in figures:
        if figure.interactive:
            escaped_path = _html.escape(figure.rel_path)
            escaped_caption = _html.escape(figure.caption)
            # A fixed-height chart gets a frame that fits it, so it never scrolls inside.
            style = f' style="height:{figure.height + 20}px;min-height:0"' if figure.height else ""
            parts.append(
                f'<figure><iframe class="interactive-figure"{style} src="{escaped_path}" '
                f'title="{escaped_caption}" loading="lazy"></iframe>'
                f'<figcaption class="caption">{escaped_caption}</figcaption>'
                f'<p class="interactive-figure-link"><a href="{escaped_path}">'
                "Open interactive figure (HTML)</a></p></figure>"
            )
            continue
        parts.append(
            f'<figure><img src="{_html.escape(figure.rel_path)}" '
            f'alt="{_html.escape(figure.caption)}">'
            f'<figcaption class="caption">{_html.escape(figure.caption)}</figcaption></figure>'
        )
    for table in tables:
        _render_table_html(table, parts, anchor, model_tips)


def _render_views_html(
    section: Section, parts: list[str], anchor: str, model_tips: dict[str, str] | None = None,
) -> None:
    """Render a section's views behind the shared tab switch (see render_view_group)."""
    views = []
    for view in section.views:
        body: list[str] = []
        _render_artifacts_html(view.figures, view.tables, body, anchor, model_tips)
        views.append((_slug(view.name), view.label, view.tip, "\n".join(body)))
    parts.append(render_view_group(anchor, views))


def _render_table_html(
    table: Table, parts: list[str], anchor: str = "", model_tips: dict[str, str] | None = None,
) -> None:
    if table.heatmap is not None:
        link = (
            f'<a href="{_html.escape(table.rel_csv)}">full CSV</a>' if table.rel_csv else ""
        )
        help_text = str(table.heatmap.get("help") or "")
        help_html = render_help_html(help_text, f"help-{anchor or 'heat'}-{_slug(table.name)}") if help_text else ""
        parts.append(render_heatmap_html(table.frame, table.heatmap, help_html, link, model_tips))
        return
    parts.append(f"<p><strong>{_html.escape(table.name)}</strong></p>")
    parts.append('<div class="table-scroll" role="region" tabindex="0">')
    parts.append(_display_frame(table.frame).to_html(index=False, border=0))
    parts.append("</div>")
    note = []
    if table.truncated:
        note.append(f"Showing {table.n_shown_rows} of {table.n_total_rows} rows")
    if table.rel_csv:
        link = f'<a href="{_html.escape(table.rel_csv)}">full CSV</a>'
        note.append(link if not note else f"; {link}")
    if note:
        parts.append(f'<p class="caption">{" ".join(note)}</p>')


def _md_inline_to_html(text: str) -> str:
    """Render inline Markdown consistently across report pages."""
    return render_summary_html(text)
