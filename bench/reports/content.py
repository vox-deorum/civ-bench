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


def metadata_text(metadata: dict) -> str:
    """Format analysis provenance for a tooltip or Markdown disclosure."""
    parts = []
    for key, value in metadata.items():
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


def render_footer_html(footer: str) -> str:
    """Render CommonMark with raw HTML escaped and unsafe links disabled."""
    if not footer.strip():
        return ""
    content = MarkdownIt("commonmark", {"html": False}).render(footer)
    return f'<footer class="report-footer">\n{content}</footer>'


def render_citation_html() -> str:
    """Render the paper citation as a copyable BibTeX code block."""
    content = MarkdownIt("commonmark", {"html": False}).render(CITATION_MARKDOWN)
    return f'<section aria-label="Citation">\n{content}</section>'
