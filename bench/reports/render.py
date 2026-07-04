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

from .model import FamilyGroup, ReportDocument, Section, Table


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


# ── markdown ──────────────────────────────────────────────────────────────────
def render_markdown(doc: ReportDocument) -> str:
    anchors = _build_anchors(doc)
    lines: list[str] = []
    lines.append(f"# {doc.title}")
    lines.append("")
    lines.append(doc.intro)
    lines.append("")

    # Table of contents.
    lines.append("## Contents")
    lines.append("")
    for group in doc.groups:
        lines.append(f"- [{group.title}](#{anchors[id(group)]})")
        for section in group.sections:
            lines.append(f"  - [{section.id}](#{anchors[id(section)]})")
    lines.append("")

    for group in doc.groups:
        lines.append(f'<a id="{anchors[id(group)]}"></a>')
        lines.append(f"## {group.title}")
        lines.append("")
        for section in group.sections:
            lines.append(f'<a id="{anchors[id(section)]}"></a>')
            _render_section_md(section, lines)
    return "\n".join(lines).rstrip() + "\n"


def _render_section_md(section: Section, lines: list[str]) -> None:
    lines.append(f"### {section.id}")
    lines.append("")
    lines.append(f"*Module: `{section.module}`*")
    meta = _metadata_line(section.metadata)
    if meta:
        lines.append("")
        lines.append(f"*{meta}*")
    lines.append("")
    if section.summary:
        lines.append(section.summary)
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
        lines.append("**Generated files**")
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
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       max-width: 1000px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5;
       color: #1a1a1a; }
h1 { border-bottom: 2px solid #ddd; padding-bottom: .3rem; }
h2 { border-bottom: 1px solid #eee; padding-bottom: .2rem; margin-top: 2.5rem; }
h3 { margin-top: 2rem; }
table { border-collapse: collapse; margin: 1rem 0; font-size: .9rem; }
th, td { border: 1px solid #ccc; padding: .3rem .6rem; text-align: right; }
th { background: #f5f5f5; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; height: auto; border: 1px solid #eee; }
.meta { color: #666; font-size: .9rem; }
.module { color: #444; font-family: monospace; }
.caption { color: #666; font-size: .85rem; font-style: italic; }
.empty { color: #888; font-style: italic; }
nav ul { list-style: none; padding-left: 1rem; }
"""


def render_html(doc: ReportDocument) -> str:
    anchors = _build_anchors(doc)
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append(f"<title>{_html.escape(doc.title)}</title>")
    parts.append(f"<style>{_HTML_STYLE}</style>")
    parts.append("</head><body>")
    parts.append(f"<h1>{_html.escape(doc.title)}</h1>")
    parts.append(f"<p>{_md_inline_to_html(doc.intro)}</p>")

    # Table of contents.
    parts.append("<nav><h2>Contents</h2><ul>")
    for group in doc.groups:
        parts.append(f'<li><a href="#{anchors[id(group)]}">{_html.escape(group.title)}</a><ul>')
        for section in group.sections:
            parts.append(f'<li><a href="#{anchors[id(section)]}">{_html.escape(section.id)}</a></li>')
        parts.append("</ul></li>")
    parts.append("</ul></nav>")

    for group in doc.groups:
        parts.append(f'<h2 id="{anchors[id(group)]}">{_html.escape(group.title)}</h2>')
        for section in group.sections:
            _render_section_html(section, parts, anchors[id(section)])

    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _render_section_html(section: Section, parts: list[str], anchor: str) -> None:
    parts.append(f'<h3 id="{anchor}">{_html.escape(section.id)}</h3>')
    parts.append(f'<p class="module">Module: {_html.escape(section.module)}</p>')
    meta = _metadata_line(section.metadata)
    if meta:
        parts.append(f'<p class="meta">{_html.escape(meta)}</p>')
    if section.summary:
        parts.append(f"<p>{_md_inline_to_html(section.summary)}</p>")
    if section.empty:
        parts.append(
            '<p class="empty">This analysis produced no artifacts for the given '
            "inputs (e.g. a controlled-design view on an uncontrolled run).</p>"
        )
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
        parts.append("<p><strong>Generated files</strong></p>")
        parts.append("<ul>")
        for dl in section.downloads:
            parts.append(
                f'<li><a href="{_html.escape(dl.rel_path)}">{_html.escape(dl.label)}</a></li>'
            )
        parts.append("</ul>")


def _render_table_html(table: Table, parts: list[str]) -> None:
    parts.append(f"<p><strong>{_html.escape(table.name)}</strong></p>")
    parts.append(_display_frame(table.frame).to_html(index=False, border=0))
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
