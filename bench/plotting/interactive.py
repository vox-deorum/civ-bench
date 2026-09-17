"""Shared offline HTML rendering for interactive report charts."""

from __future__ import annotations


def figure_html(
    figure,
    name: str,
    *,
    full_html: bool = True,
    include_plotlyjs: bool | str = True,
) -> str:
    """Render a figure with stable IDs and common interaction settings.

    Standalone figures embed JavaScript. Report pages can reference the site's
    local Plotly bundle to share it across charts.
    """
    from plotly.io import to_html

    return to_html(
        figure,
        div_id=f"plotly-{name}",
        full_html=full_html,
        include_plotlyjs=include_plotlyjs,
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
        default_height="650px",
    )


def plotly_javascript() -> str:
    """Return the installed Plotly runtime for the report's local asset tree."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()
