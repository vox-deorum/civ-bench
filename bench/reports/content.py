"""Shared report summaries, help tooltips, citations, and footers."""

import html
import re

from markdown_it import MarkdownIt

from bench.config.schema import REPORT_DEFAULT_FOOTER


CITATION_BIBTEX = """@article{chen2026civbench,
  title={CivBench: Progress-Based Evaluation for LLMs' Strategic Decision-Making in Civilization V},
  author={Chen, John and Cheng, Sihan and Gurkan, Can and Lin, Mingyi},
  journal={Conference on Language Modeling (COLM)},
  year={2026}
}"""
CITATION_MARKDOWN = f"## Citation\n\n```bibtex\n{CITATION_BIBTEX}\n```"


def resolve_footer(value: str | None) -> str:
    """Use the default for null, preserving an explicitly empty footer."""
    return REPORT_DEFAULT_FOOTER if value is None else value


def render_summary_html(summary: str) -> str:
    """Render inline CommonMark with raw HTML escaped and unsafe links disabled."""
    return MarkdownIt("commonmark", {"html": False}).renderInline(summary)


def report_summary(summary: str, metadata: dict) -> str:
    """Keep baseline provenance in metadata and use its public report label."""
    baseline = metadata.get("baseline_experiment")
    if baseline:
        for quoted in (f"'{baseline}'", f'"{baseline}"', f"`{baseline}`"):
            summary = summary.replace(f": {quoted}", "").replace(f" {quoted}", "")
    summary = re.sub(r"\bVanilla\b", "VPAI", summary)
    # Older saved manifests carry unformatted coverage counts.
    if summary.startswith("Controlled seed comparison covers "):
        summary = summary.replace("Controlled seed comparison covers ", "The comparison covers ", 1)
        summary = re.sub(r"(?<!\*)\b\d+\b(?!\*)", r"**\g<0>**", summary)
    return summary


# Metadata keys that steer rendering rather than describe provenance.
LAYOUT_METADATA_KEYS = {"views", "heatmaps", "matched_maps", "tabs"}


def metadata_text(metadata: dict) -> str:
    """Format analysis provenance for a tooltip or Markdown disclosure."""
    parts = []
    for key, value in metadata.items():
        if key in LAYOUT_METADATA_KEYS:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


def render_help_html(text: str, tip_id: str, label: str = "Details") -> str:
    """An escaped help control shared by every HTML report page."""
    if not text.strip():
        return ""
    return (
        '<span class="report-help">'
        f'<button type="button" class="help-toggle" aria-label="{html.escape(label)}" '
        f'aria-describedby="{html.escape(tip_id)}" aria-expanded="false">?</button>'
        f'<span class="help-text" role="tooltip" id="{html.escape(tip_id)}">'
        f'{html.escape(text)}</span></span>'
    )


def render_view_group(anchor: str, views: list[tuple[str, str, str, str]]) -> str:
    """Alternative views of one block behind a tab switch.

    ``views`` holds ``(name, label, tip, body_html)`` in display order; the first
    is the default. The shared report script shows one view at a time, and a
    choice switches every group on the page that offers a view of the same
    name. ``?view=<name>`` in the page URL picks the initial view. Without
    JavaScript every view stays visible under its label. A ``tip`` (the long
    wording behind a short label) shows as the button's tooltip.
    """
    parts = ['<div class="view-group">']
    if len(views) > 1:
        parts.append('<div class="view-switch" role="group" aria-label="Choose a view">')
        for index, (name, label, tip, _body) in enumerate(views):
            tip_attr = f' data-tip="{html.escape(tip)}"' if tip else ""
            parts.append(
                f'<button type="button" class="view-button" data-view="{html.escape(name)}" '
                f'aria-controls="{html.escape(anchor)}-view-{html.escape(name)}" '
                f'aria-pressed="{"true" if index == 0 else "false"}"{tip_attr}>'
                f"{html.escape(label)}</button>"
            )
        parts.append("</div>")
    for name, label, _tip, body in views:
        parts.append(
            f'<div class="view-panel" id="{html.escape(anchor)}-view-{html.escape(name)}" '
            f'data-view="{html.escape(name)}">'
        )
        parts.append(f'<p class="view-heading"><strong>{html.escape(label)}</strong></p>')
        parts.append(body)
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def render_footer_html(footer: str) -> str:
    """Render CommonMark with raw HTML escaped and unsafe links disabled."""
    if not footer.strip():
        return ""
    content = MarkdownIt("commonmark", {"html": False}).render(footer)
    return f'<footer class="report-footer">\n{content}</footer>'


def render_citation_markdown(benchmark_citation: dict[str, str] | None = None) -> str:
    """Render the paper and optional benchmark results as BibTeX entries."""
    if benchmark_citation is None:
        return CITATION_MARKDOWN
    title = _bibtex_text(benchmark_citation["title"])
    url = _bibtex_text(benchmark_citation["url"])
    benchmark_bibtex = (
        "@misc{civbench_results,\n"
        f"  title={{{title}}},\n"
        f"  url={{{url}}}\n"
        "}"
    )
    return (
        f"## Citation\n\n### Paper\n\n```bibtex\n{CITATION_BIBTEX}\n```\n\n"
        f"### Benchmark results\n\n```bibtex\n{benchmark_bibtex}\n```"
    )


def _bibtex_text(value: str) -> str:
    """Keep configured text within its BibTeX field and Markdown fence."""
    value = " ".join(value.split())
    escapes = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}"}
    return "".join(escapes.get(char, char) for char in value)


def render_citation_html(benchmark_citation: dict[str, str] | None = None) -> str:
    """Render the citations as copyable BibTeX code blocks."""
    content = MarkdownIt("commonmark", {"html": False}).render(
        render_citation_markdown(benchmark_citation)
    )
    return f'<section aria-label="Citation">\n{content}</section>'
