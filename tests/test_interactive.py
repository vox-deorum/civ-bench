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


# ── interactive forest plot and shared heatmap labels ─────────────────────────
def _pairing_setup(configs_dir):
    import bench.analyses  # noqa: F401  (loads the package in its normal order)
    from bench.catalog import Catalog
    from bench.plotting.pairing import PairingSpec

    catalog = Catalog.from_paths(configs_dir / "models.json", configs_dir / "experiments.json")
    return catalog, PairingSpec(("-Per-5",), "base", "Every-turn")


def test_interactive_forest_has_one_tooltip_trace_per_point(configs_dir):
    import pandas as pd

    from bench.plotting.pairing import plot_paired_rows_interactive

    catalog, spec = _pairing_setup(configs_dir)
    frame = pd.DataFrame([
        {"player_type": "GPT-OSS-120B-Simple", "elo": 1600.0, "lo": 1570.0, "hi": 1640.0, "tip": "1,600"},
        {"player_type": "GPT-OSS-120B-Simple-Per-5", "elo": 1580.0, "lo": 1550.0, "hi": 1600.0, "tip": "1,580"},
        {"player_type": "Opus-5.5-Simple", "elo": 1650.0, "lo": 1600.0, "hi": 1700.0, "tip": "1,650"},
        {"player_type": "Vanilla", "elo": 1500.0, "lo": None, "hi": None, "tip": "1,500"},
    ])
    figure = plot_paired_rows_interactive(
        frame, catalog=catalog, spec=spec, value_col="elo", identity_col="player_type",
        lo_col="lo", hi_col="hi", ref_line=1500, tooltip=[("Elo", "tip")], xlabel="Elo",
    )

    points = [trace for trace in figure.data if trace.meta and trace.meta.get("table_tooltip")]
    assert len(points) == 4
    by_name = {trace.name: trace for trace in points}
    assert list(by_name["GPT-OSS-120B-Simple"].customdata[0]) == ["GPT-OSS-120B-Simple", "Every-turn", "1,600"]
    assert by_name["GPT-OSS-120B-Simple-Per-5"].marker.symbol == "diamond-open"
    assert by_name["GPT-OSS-120B-Simple"].error_x.arrayminus == (30.0,)
    assert by_name["Vanilla"].error_x.type is None
    # One legend-only trace per plotted condition toggles its points.
    legend = [trace for trace in figure.data if trace.showlegend and trace.mode == "markers"]
    assert [(trace.name, trace.legendgroup) for trace in legend] == [
        ("Every-turn", "base"), ("Per-5", "-Per-5"),
    ]
    reference = [trace for trace in figure.data if trace.mode == "lines"]
    assert len(reference) == 1 and reference[0].x == (1500, 1500)
    # Rows sort by the base condition, highest first; a missing condition is noted.
    assert figure.layout.yaxis.ticktext[1:] == ("GPT-OSS-120B-Simple", "Vanilla")
    assert "(base only)" in figure.layout.yaxis.ticktext[0]
    assert figure.layout.meta["table_tooltip"]["rows"] == [{"label": "Elo", "index": 2, "emphasis": True}]
    assert figure.layout.height == 360
    assert "plotly_hover" in figure_html(figure, "forest", include_plotlyjs=False)


def test_interactive_forest_returns_none_without_rows(configs_dir):
    import pandas as pd

    from bench.plotting.pairing import plot_paired_rows_interactive

    catalog, spec = _pairing_setup(configs_dir)
    assert plot_paired_rows_interactive(
        pd.DataFrame(), catalog=catalog, spec=spec, value_col="elo", identity_col="player_type",
    ) is None


@pytest.mark.parametrize("paired, labels, order", [
    (True,
     {"GPT-OSS-120B-Simple-Per-5": "GPT-OSS-120B-Simple | Per-5", "Vanilla": "Vanilla",
      "GPT-OSS-120B-Simple": "GPT-OSS-120B-Simple | Every-turn", "Null": "Null"},
     ["Null", "Vanilla", "GPT-OSS-120B-Simple | Every-turn", "GPT-OSS-120B-Simple | Per-5"]),
    (False,
     {"GPT-OSS-120B-Simple-Per-5": "GPT-OSS-120B-Simple-Per-5", "Vanilla": "Vanilla",
      "GPT-OSS-120B-Simple": "GPT-OSS-120B-Simple", "Null": "Null"},
     ["Null", "Vanilla", "GPT-OSS-120B-Simple", "GPT-OSS-120B-Simple-Per-5"]),
])
def test_identity_labels_match_heatmap_rows(configs_dir, paired, labels, order):
    import pandas as pd

    from bench.analyses.heatmap_layout import heatmap_rows, identity_labels

    catalog, spec = _pairing_setup(configs_dir)

    class Context:
        def __init__(self):
            self.catalog = catalog

        def condition_pairing(self):
            return spec if paired else None

    identities = list(labels)
    assert identity_labels(Context(), identities) == (labels, order)
    rows, row_order = heatmap_rows(Context(), pd.DataFrame({"player_type": identities}), "player_type")
    assert row_order == order
    assert rows["row_label"].tolist() == [labels[i] for i in identities]
