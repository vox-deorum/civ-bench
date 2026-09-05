"""Render a :class:`ReportDocument` to markdown and HTML (stage 5).

Both renderers walk the same document model so the two formats stay faithful to
each other: a :class:`Table` becomes a GitHub-flavoured pipe table in markdown and
a ``<table>`` in HTML from the *same* DataFrame, never two hand-written variants.
Every cell is formatted to a display string once (:func:`_display_frame`) before
either renderer touches it, so the two formats show byte-identical content —
tabulate and pandas otherwise diverge on float precision and missing-value text.
Output is deterministic — no timestamps — so a re-render of unchanged artifacts is
byte-stable (AGENTS.md determinism invariant).
"""

from __future__ import annotations

import html as _html

from .controlled_seed import render_controlled_seed_site
from .model import ReportDocument, Section, Table


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
    if not metadata:
        return ""
    parts = []
    for key, value in metadata.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


def _section_summary(section: Section) -> str:
    """Return the result sentence shown in overview and detailed views."""
    return section.summary.strip() or "No result summary was produced for this analysis."


# ── markdown ──────────────────────────────────────────────────────────────────
def render_markdown(doc: ReportDocument) -> str:
    anchors = _build_anchors(doc)
    lines: list[str] = []
    lines.append(f"# {doc.title}")
    lines.append("")
    if doc.description:
        lines.append(f"*{doc.description}*")
        lines.append("")
    lines.append(doc.intro)
    lines.append("")

    if doc.overview_sections:
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
    return "\n".join(lines).rstrip() + "\n"


def _render_section_md(section: Section, lines: list[str]) -> None:
    lines.append(f"### {section.title}")
    lines.append("")
    lines.append(f"*Module: `{section.module}`*")
    if section.description:
        lines.append("")
        lines.append(f"*{section.description}*")
    meta = _metadata_line(section.metadata)
    if meta:
        lines.append("")
        lines.append(f"*{meta}*")
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
    for figure in section.figures:
        # Bracket/paren in a caption would break the inline image link; the path
        # is artifact-controlled (assets/<id>/<name>.png) so only the caption needs it.
        alt = figure.caption.replace("[", "(").replace("]", ")")
        lines.append(f"![{alt}]({figure.rel_path})")
        lines.append("")
        lines.append(f"*Figure: {figure.caption}*")
        lines.append("")
    for table in section.tables:
        _render_table_md(table, lines)
    if section.downloads:
        lines.append("**Downloads and supporting files**")
        lines.append("")
        for dl in section.downloads:
            lines.append(f"- [{dl.label}]({dl.rel_path})")
        lines.append("")


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
    float NaN in an object column), and floats use ``g`` — matching tabulate's own
    default so the markdown output is unchanged while the HTML side stops diverging
    (pandas' ``to_html`` otherwise uses full-precision/scientific formatting).
    """
    if isinstance(value, float):
        if value != value:  # NaN — the only value not equal to itself
            return ""
        return f"{value:g}"
    if value is None:
        return ""
    return str(value)


def _display_frame(frame):
    """Format every cell to a display string (see :func:`_format_cell`) so the
    markdown and HTML tables are byte-for-byte faithful. Full-precision values are
    always one click away via the linked ``full CSV``."""
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].map(_format_cell)
    return out


def _render_table_md(table: Table, lines: list[str]) -> None:
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
        note.append(link if not note else f"— {link}")
    if note:
        lines.append(f"_{' '.join(note)}._")
        lines.append("")


# ── html ──────────────────────────────────────────────────────────────────────
_HTML_STYLE = """
:root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #18202a; background: #f7f8fa; }
* { box-sizing: border-box; }
body { margin: 0; line-height: 1.55; font-size: 14px; }
a { color: #175ca8; }
a:hover { color: #0b3d73; }
.skip-link { position: fixed; top: .5rem; left: .5rem; z-index: 20; padding: .5rem .75rem; background: white; transform: translateY(-180%); }
.skip-link:focus { transform: none; }
.sidebar { position: fixed; inset: 0 auto 0 0; width: 19rem; overflow-y: auto; padding: 1.5rem 1rem; color: #e9eef5; background: #172536; }
.site-title { display: block; margin: 0 0 1rem; color: white; font-size: 1.05rem; font-weight: 700; text-decoration: none; }
.sidebar ul { margin: 0; padding: 0; list-style: none; }
.sidebar ul ul { margin: .3rem 0 .8rem .75rem; border-left: 1px solid #4b5b6d; padding-left: .75rem; }
.sidebar a { display: block; border-radius: .3rem; padding: .25rem .45rem; color: #cbd7e5; text-decoration: none; }
.sidebar a:hover, .sidebar a[aria-current="page"] { color: white; background: #2b4159; }
.content { max-width: 78rem; margin-left: 19rem; padding: 2.25rem 3rem 4rem; }
h1 { margin-top: 0; border-bottom: 2px solid #cfd6df; padding-bottom: .45rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #dfe4ea; padding-bottom: .25rem; }
h3 { margin-top: 1.75rem; }
.eyebrow { margin-bottom: .35rem; color: #637083; font-size: .85rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); gap: 1rem; margin-top: 1.5rem; }
.overview-card { border: 1px solid #d9e0e8; border-radius: .55rem; padding: 1rem; background: white; box-shadow: 0 1px 2px rgb(0 0 0 / 6%); }
.overview-card h2 { margin: 0 0 .25rem; border: 0; padding: 0; font-size: 1.1rem; }
.overview-card p { margin: .45rem 0; }
.module { color: #445164; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.meta, .caption { color: #687486; font-size: .88rem; }
.caption { font-style: italic; }
.empty { color: #778294; font-style: italic; }
figure { margin: 1.25rem 0 2rem; }
img { max-width: 100%; height: auto; border: 1px solid #e2e6eb; background: white; }
.table-scroll { max-width: 100%; overflow-x: auto; border: 1px solid #d8dee6; border-radius: .35rem; background: white; }
table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: .9rem; }
th, td { border-bottom: 1px solid #e0e5eb; padding: .4rem .65rem; text-align: right; white-space: nowrap; }
th { background: #f0f3f7; }
td:first-child, th:first-child { text-align: left; }
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
.heat-cell a { display: block; padding: .38rem .55rem; color: inherit; text-decoration: none; }
.heat-cell a:hover, .heat-cell a:focus-visible { outline: 2px solid #175ca8; outline-offset: -2px; }
.heat-cell-empty { background: transparent; }
tbody.vanilla-body tr.vanilla-row th, tbody.vanilla-body tr.vanilla-row td { border-top: 3px double #8d99a9; border-bottom: 3px double #8d99a9; background: #f3f0e8; }
tbody.vanilla-body tr.vanilla-row .heat-cell { background-image: linear-gradient(rgba(255,255,255,.35), rgba(255,255,255,.35)); }
.heat-tooltip { position: absolute; z-index: 30; display: none; max-width: 22rem; border: 1px solid #445164; border-radius: .3rem; padding: .4rem .6rem; color: #18202a; background: #fffdf5; box-shadow: 0 2px 8px rgb(0 0 0 / 18%); white-space: pre-line; font-size: .82rem; pointer-events: none; }
.focus-legend { display: flex; flex-wrap: wrap; gap: .9rem; margin: .25rem 0 1.5rem; padding: 0; list-style: none; font-size: .85rem; }
.focus-legend .swatch { display: inline-block; width: .85rem; height: .85rem; margin-right: .3rem; border: 1px solid #44516433; border-radius: .15rem; vertical-align: -0.1em; }
.chart-controls { display: flex; gap: .55rem; align-items: center; margin: .75rem 0; }
.curve-chart svg { max-width: 100%; height: auto; }
.curve-legend { display: flex; flex-wrap: wrap; gap: .8rem; margin: .5rem 0; padding: 0; list-style: none; font-size: .85rem; }
.curve-legend .curve-swatch { display: inline-block; width: 1.6rem; margin-right: .3rem; vertical-align: middle; }
.curve-legend li.preselected { border: 1px solid #175ca8; border-radius: .3rem; padding: .05rem .4rem; font-weight: 650; background: #eef3fa; }
table.comparison td.vanilla-value { font-weight: 700; }
@media (max-width: 820px) {
  .sidebar { position: static; width: auto; max-height: none; }
  .sidebar ul ul { display: none; }
  .content { margin-left: 0; padding: 1.5rem 1rem 3rem; }
}
""".strip() + "\n"


def render_stylesheet() -> str:
    """Return the shared, deterministic stylesheet for the HTML report site."""
    return _HTML_STYLE


def _family_filenames(doc: ReportDocument) -> dict[int, str]:
    used = {"report"}
    filenames: dict[int, str] = {}
    for group in doc.groups:
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
) -> list[str]:
    parts = ['<aside class="sidebar">']
    parts.append(f'<a class="site-title" href="report.html">{_html.escape(doc.title)}</a>')
    parts.append('<nav aria-label="Report navigation"><ul>')
    current = ' aria-current="page"' if active == "report" else ""
    parts.append(f'<li><a href="report.html"{current}>Overview</a></li>')
    for group in doc.groups:
        filename = filenames[id(group)]
        current = ' aria-current="page"' if active == group.key else ""
        parts.append(
            f'<li><a href="{filename}"{current}>{_html.escape(group.title)}</a><ul>'
        )
        for section in group.sections:
            parts.append(
                f'<li><a href="{filename}#{anchors[id(section)]}">'
                f'{_html.escape(section.title)}</a></li>'
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
        "</head><body>",
        '<a class="skip-link" href="#main-content">Skip to content</a>',
    ]


def render_html_site(doc: ReportDocument) -> dict[str, str]:
    """Render the overview, one HTML page per represented analysis family, and
    (when the document carries it) the controlled-seed heatmap annex."""
    anchors = _build_anchors(doc)
    filenames = _family_filenames(doc)
    family_for = {
        id(section): group
        for group in doc.groups
        for section in group.sections
    }
    pages: dict[str, str] = {}

    parts = _page_start(doc, doc.title)
    parts.extend(_render_navigation(doc, anchors, filenames, "report"))
    parts.append('<main class="content" id="main-content">')
    parts.append(f"<h1>{_html.escape(doc.title)}</h1>")
    if doc.description:
        parts.append(f'<p class="caption">{_html.escape(doc.description)}</p>')
    parts.append(f"<p>{_md_inline_to_html(doc.intro)}</p>")
    parts.append('<section aria-labelledby="overview-heading">')
    parts.append('<h2 id="overview-heading">Overview</h2>')
    parts.append('<div class="overview-grid">')
    for section in doc.overview_sections:
        group = family_for[id(section)]
        target = f"{filenames[id(group)]}#{anchors[id(section)]}"
        parts.append('<article class="overview-card">')
        parts.append(f'<p class="eyebrow">{_html.escape(group.title)}</p>')
        parts.append(f"<h2>{_html.escape(section.title)}</h2>")
        parts.append(f"<p>{_md_inline_to_html(_section_summary(section))}</p>")
        parts.append(f'<p><a href="{target}">View details</a></p>')
        parts.append("</article>")
    parts.append("</div></section></main></body></html>")
    pages["report.html"] = "\n".join(parts) + "\n"

    for group in doc.groups:
        page_title = f"{group.title} | {doc.title}"
        parts = _page_start(doc, page_title)
        parts.extend(_render_navigation(doc, anchors, filenames, group.key))
        parts.append('<main class="content" id="main-content">')
        parts.append(f'<p class="eyebrow">{_html.escape(doc.title)}</p>')
        parts.append(f'<h1 id="{anchors[id(group)]}">{_html.escape(group.title)}</h1>')
        if group.summary:
            parts.append(f"<p>{_md_inline_to_html(group.summary)}</p>")
        for section in group.sections:
            _render_section_html(section, parts, anchors[id(section)])
        parts.append("</main></body></html>")
        pages[filenames[id(group)]] = "\n".join(parts) + "\n"
    if doc.controlled_seed is not None:
        pages.update(render_controlled_seed_site(doc.controlled_seed))
    return pages


def render_html(doc: ReportDocument) -> str:
    """Compatibility wrapper returning the generated overview page."""
    return render_html_site(doc)["report.html"]


def _render_section_html(section: Section, parts: list[str], anchor: str) -> None:
    parts.append(f'<section aria-labelledby="{anchor}">')
    parts.append(f'<h2 id="{anchor}">{_html.escape(section.title)}</h2>')
    parts.append(f'<p class="module">Module: {_html.escape(section.module)}</p>')
    if section.description:
        parts.append(f'<p class="caption">{_html.escape(section.description)}</p>')
    meta = _metadata_line(section.metadata)
    if meta:
        parts.append(f'<p class="meta">{_html.escape(meta)}</p>')
    parts.append(f"<p>{_md_inline_to_html(_section_summary(section))}</p>")
    if section.empty:
        parts.append(
            '<p class="empty">This analysis produced no artifacts for the given '
            "inputs (e.g. a controlled-design view on an uncontrolled run).</p>"
        )
        parts.append("</section>")
        return
    for figure in section.figures:
        parts.append(
            f'<figure><img src="{_html.escape(figure.rel_path)}" '
            f'alt="{_html.escape(figure.caption)}">'
            f'<figcaption class="caption">{_html.escape(figure.caption)}</figcaption></figure>'
        )
    for table in section.tables:
        _render_table_html(table, parts)
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


def _render_table_html(table: Table, parts: list[str]) -> None:
    parts.append(f"<p><strong>{_html.escape(table.name)}</strong></p>")
    parts.append('<div class="table-scroll" role="region" tabindex="0">')
    parts.append(_display_frame(table.frame).to_html(index=False, border=0))
    parts.append("</div>")
    note = []
    if table.truncated:
        note.append(f"Showing {table.n_shown_rows} of {table.n_total_rows} rows")
    if table.rel_csv:
        link = f'<a href="{_html.escape(table.rel_csv)}">full CSV</a>'
        note.append(link if not note else f"— {link}")
    if note:
        parts.append(f'<p class="caption">{" ".join(note)}</p>')


def _md_inline_to_html(text: str) -> str:
    """Escape text, then re-enable the tiny inline-markdown subset our summaries use
    (``**bold**``, `` `code` ``). Keeps the HTML faithful to the markdown without a
    full markdown parser dependency.

    Code spans are emitted first and their contents are left verbatim, so a ``**``
    inside a code span is never mistaken for bold (which would open a ``<strong>``
    across the span and make the HTML disagree with the markdown).
    """
    escaped = _html.escape(text)
    out: list[str] = []
    i, n = 0, len(escaped)
    while i < n:
        if escaped[i] == "`":
            close = escaped.find("`", i + 1)
            if close == -1:  # unbalanced tick — leave it (and the rest) literal
                out.append(_md_bold(escaped[i:]))
                break
            out.append(f"<code>{escaped[i + 1 : close]}</code>")
            i = close + 1
        else:
            nxt = escaped.find("`", i)
            out.append(_md_bold(escaped[i:] if nxt == -1 else escaped[i:nxt]))
            i = n if nxt == -1 else nxt
    return "".join(out)


def _md_bold(text: str) -> str:
    """Convert ``**bold**`` runs to ``<strong>`` in a code-free text segment."""
    out = text
    while "**" in out:
        first = out.find("**")
        second = out.find("**", first + 2)
        if second == -1:
            break
        out = out[:first] + f"<strong>{out[first + 2 : second]}</strong>" + out[second + 2 :]
    return out
