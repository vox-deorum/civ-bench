"""Shared offline HTML rendering for interactive report charts."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

# The report's one shared copy of the Plotly runtime, under its assets folder.
PLOTLY_ASSET = "plotly.min.js"


def _table_tooltip_post_script() -> str:
    """Return the opt-in table-tooltip behavior installed after Plotly renders."""
    return r"""
(function () {
    const plot = document.getElementById("{plot_id}");
    if (!plot || !plot.layout || !plot.layout.meta) {
        return;
    }
    const specification = plot.layout.meta.table_tooltip;
    if (!specification || !Array.isArray(specification.rows)) {
        return;
    }

    const tooltip = document.createElement("div");
    tooltip.setAttribute("role", "tooltip");
    tooltip.style.cssText = [
        "position:fixed", "z-index:10000", "display:none", "pointer-events:none",
        "max-width:360px", "padding:10px 12px", "border:1px solid #cbd5e1",
        "border-radius:6px", "background:#ffffff", "color:#1f2937",
        "box-shadow:0 4px 12px rgba(15, 23, 42, 0.18)",
        "font:13px/1.35 Arial, sans-serif", "white-space:normal"
    ].join(";");
    const title = document.createElement("div");
    title.style.cssText = "font-weight:700;font-size:14px;margin-bottom:2px";
    const subtitle = document.createElement("div");
    subtitle.style.cssText = "color:#475569;margin-bottom:7px";
    const table = document.createElement("table");
    table.style.cssText = "border-collapse:collapse;width:100%;font-size:12px";
    const note = document.createElement("div");
    note.style.cssText = "color:#64748b;font-size:11px;margin-top:7px";
    tooltip.append(title, subtitle, table, note);
    document.body.appendChild(tooltip);

    function valueAt(data, index) {
        if (!Array.isArray(data) || !Number.isInteger(index) || index < 0) {
            return "";
        }
        const value = data[index];
        return value === null || value === undefined ? "" : String(value);
    }

    function hide() {
        tooltip.style.display = "none";
    }

    function place(event) {
        const offset = 14;
        const padding = 8;
        const x = event && Number.isFinite(event.clientX) ? event.clientX + offset : padding;
        const y = event && Number.isFinite(event.clientY) ? event.clientY + offset : padding;
        const width = tooltip.offsetWidth;
        const height = tooltip.offsetHeight;
        tooltip.style.left = Math.max(padding, Math.min(x, window.innerWidth - width - padding)) + "px";
        tooltip.style.top = Math.max(padding, Math.min(y, window.innerHeight - height - padding)) + "px";
    }

    plot.on("plotly_hover", function (eventData) {
        const point = eventData && eventData.points && eventData.points[0];
        const trace = point && point.fullData;
        if (!point || !trace || !trace.meta || trace.meta.table_tooltip !== true) {
            hide();
            return;
        }
        const data = point.customdata;
        title.textContent = valueAt(data, specification.title_index);
        subtitle.textContent = valueAt(data, specification.subtitle_index);
        table.replaceChildren();
        specification.rows.forEach(function (row) {
            if (!row || !Number.isInteger(row.index)) {
                return;
            }
            const tableRow = document.createElement("tr");
            const label = document.createElement("th");
            label.scope = "row";
            label.textContent = row.label === null || row.label === undefined ? "" : String(row.label);
            label.style.cssText = "padding:2px 12px 2px 0;text-align:left;font-weight:inherit;white-space:nowrap";
            const value = document.createElement("td");
            value.textContent = valueAt(data, row.index);
            value.style.cssText = "padding:2px 0;text-align:right";
            if (row.emphasis === true) {
                tableRow.style.fontWeight = "700";
            }
            tableRow.append(label, value);
            table.appendChild(tableRow);
        });
        note.textContent = specification.note === null || specification.note === undefined
            ? "" : String(specification.note);
        note.style.display = note.textContent ? "block" : "none";
        tooltip.style.display = "block";
        place(eventData.event);
    });
    plot.on("plotly_unhover", hide);
    plot.on("plotly_relayout", hide);
    plot.on("plotly_restyle", hide);
    plot.on("plotly_update", hide);
    plot.on("plotly_legendclick", hide);
    plot.on("plotly_legenddoubleclick", hide);
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
}());
"""


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

    layout_meta = getattr(figure.layout, "meta", None)
    post_script = None
    if isinstance(layout_meta, dict) and layout_meta.get("table_tooltip"):
        post_script = _table_tooltip_post_script()

    return to_html(
        figure,
        div_id=f"plotly-{name}",
        full_html=full_html,
        include_plotlyjs=include_plotlyjs,
        config={"responsive": True, "displaylogo": False},
        post_script=post_script,
        default_width="100%",
        default_height="650px",
    )


@lru_cache(maxsize=1)
def plotly_javascript() -> str:
    """Return the installed Plotly runtime for the report's local asset tree."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def share_plotly(html: str, src: str) -> Optional[str]:
    """``html`` with its embedded Plotly runtime replaced by ``<script src=src>``.

    A standalone chart embeds the whole runtime (about 4.7 MB). A report that
    copies several charts links each one to a single shared copy instead.
    Returns ``None`` when ``html`` does not embed the installed runtime, for
    example a chart saved by another Plotly version, so it is left as it is.
    """
    bundle = plotly_javascript()
    start = html.find(bundle)
    if start < 0:
        return None
    open_tag = html.rfind("<script", 0, start)
    end = html.find("</script>", start + len(bundle))
    if open_tag < 0 or end < 0 or "</script>" in html[open_tag:start]:
        return None
    end += len("</script>")
    return f'{html[:open_tag]}<script charset="utf-8" src="{src}"></script>{html[end:]}'
