"""Shared Markdown rendering for report summaries, citations, and footers."""

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
