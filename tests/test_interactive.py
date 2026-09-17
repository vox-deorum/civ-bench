"""Tests for shared Plotly HTML rendering."""

from __future__ import annotations

import json
import shutil
import subprocess

import plotly.graph_objects as go
import pytest

from bench.plotting.interactive import _table_tooltip_post_script, figure_html


def test_table_tooltip_is_opt_in_and_uses_safe_text_nodes():
    figure = go.Figure(go.Scatter(
        x=[1],
        y=[2],
        customdata=[["<model>", "<condition>", "1,500"]],
        meta={"table_tooltip": True},
        hoverinfo="none",
    ))
    figure.update_layout(meta={
        "table_tooltip": {
            "title_index": 0,
            "subtitle_index": 1,
            "note": "Averages per player per game.",
            "rows": [{"label": "Elo", "index": 2, "emphasis": True}],
        },
    })

    html = figure_html(figure, "safe-tooltip", include_plotlyjs=False)

    assert "plotly_hover" in html
    assert "trace.meta.table_tooltip !== true" in html
    assert "textContent" in html
    assert "innerHTML" not in html
    assert "\\u003cmodel\\u003e" in html
    assert "fontWeight = \"700\"" in html
    assert "font-weight:inherit" in html
    assert "background:#ffffff" in html
    assert "plotly_restyle" in html
    assert "plotly_update" in html
    assert "window.addEventListener(\"scroll\", hide, true)" in html
    assert "window.addEventListener(\"resize\", hide)" in html


def test_table_tooltip_script_is_omitted_without_layout_configuration():
    figure = go.Figure(go.Scatter(x=[1], y=[2]))

    html = figure_html(figure, "ordinary-chart", include_plotlyjs=False)

    assert "plotly_hover" not in html
    assert "table_tooltip" not in html


def test_table_tooltip_behavior_handles_hoverinfo_none_and_lifecycle_events():
    """Run the generated browser script against a small DOM and Plotly event mock."""
    if shutil.which("node") is None:
        pytest.skip("Node.js is required to execute the tooltip behavior test.")

    tooltip_script = _table_tooltip_post_script().replace("{plot_id}", "plotly-test")
    javascript = f"""
class Element {{
    constructor() {{ this.children = []; this.style = {{}}; this.textContent = ""; }}
    append(...children) {{ this.children.push(...children); }}
    appendChild(child) {{ this.children.push(child); }}
    replaceChildren() {{ this.children = []; }}
    setAttribute() {{}}
    get offsetWidth() {{ return 100; }}
    get offsetHeight() {{ return 50; }}
}}
const plot = new Element();
plot.layout = {{meta: {{table_tooltip: {{
    title_index: 0, subtitle_index: 1, note: "Per game.",
    rows: [{{label: "Elo", index: 2, emphasis: true}}]
}}}}}};
plot.handlers = {{}};
plot.on = (name, handler) => {{ plot.handlers[name] = handler; }};
const document = {{
    body: new Element(),
    getElementById: () => plot,
    createElement: () => new Element(),
}};
const listeners = {{}};
const window = {{
    innerWidth: 300, innerHeight: 200,
    addEventListener: (name, handler, capture) => {{ listeners[`${{name}}:${{Boolean(capture)}}`] = handler; }},
}};
eval({json.dumps(tooltip_script)});
const tooltip = document.body.children[0];
plot.handlers.plotly_hover({{
    points: [{{
        fullData: {{meta: {{table_tooltip: true}}, hoverinfo: "none"}},
        customdata: ["<unsafe>", "Every-turn", "1,500"],
    }}],
    event: {{clientX: 290, clientY: 190}},
}});
if (tooltip.style.display !== "block") throw new Error("tooltip did not show");
if (tooltip.children[0].textContent !== "<unsafe>") throw new Error("title was not text");
if (tooltip.children[2].children[0].style.fontWeight !== "700") throw new Error("Elo was not emphasized");
if (tooltip.style.left !== "192px" || tooltip.style.top !== "142px") throw new Error("tooltip was not clamped");
plot.handlers.plotly_restyle();
if (tooltip.style.display !== "none") throw new Error("restyle did not hide tooltip");
plot.handlers.plotly_hover({{points: [{{fullData: {{meta: {{table_tooltip: true}}}}, customdata: ["a", "b", "c"]}}]}});
plot.handlers.plotly_update();
if (tooltip.style.display !== "none") throw new Error("update did not hide tooltip");
plot.handlers.plotly_hover({{points: [{{fullData: {{meta: {{table_tooltip: true}}}}, customdata: ["a", "b", "c"]}}]}});
listeners["scroll:true"]();
if (tooltip.style.display !== "none") throw new Error("scroll did not hide tooltip");
plot.handlers.plotly_hover({{points: [{{fullData: {{meta: {{table_tooltip: true}}}}, customdata: ["a", "b", "c"]}}]}});
listeners["resize:false"]();
if (tooltip.style.display !== "none") throw new Error("resize did not hide tooltip");
"""

    result = subprocess.run(
        ["node", "-"],
        input=javascript,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
