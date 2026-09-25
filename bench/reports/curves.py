"""The shared interactive victory-probability curve chart.

One chart, two homes: each Matched Maps seat page (curves for one seed and
seat) and the family page of ``performance.turn_predicted`` (curves pooled over
every game). Both render a :class:`~bench.reports.model.CurveChart` through
:func:`render_curve_chart_html`: a strategist checkbox group above an offline
Plotly chart. Every value, color, and label is computed server-side, so
unchanged inputs re-render byte-identically. The static :data:`CURVE_CHART_JS`
asset only changes trace visibility, color, and emphasis, refits the Y axis to
the visible curves, and spreads same-family strategist colors through the
shared :js:func:`civBench.distinguishColors` util in :mod:`bench.reports.assets`.

Each change of the checkboxes dispatches a bubbling ``curvechart:change`` event
on the chart container whose ``detail`` carries ``checked`` (strategist to
bool) and ``filtered`` (whether the chart has checkboxes at all), so a page can
filter related content, such as the seat page's game rows.
"""

from __future__ import annotations

import html as _html
from typing import Optional

from bench.plotting.interactive import figure_html

from .model import CurveChart

# Plotly dash styles cycled by condition position, so one strategist's
# conditions stay distinguishable while sharing its color.
_DASH_PATTERNS = ("solid", "dash", "dot", "dashdot")
_FALLBACK_COLOR = "#555555"

# The checkbox tooltip of VPAI seats in games with LLM players (the separate
# self-play reference curve has no checkbox).
VPAI_MIXED_TIP = (
    "VPAI in games with LLM players. Performance may differ from self-play "
    "because LLMs can counter VPAI's playstyle."
)


def _esc(text) -> str:
    return _html.escape(str(text))


def strategist_label(chart: CurveChart, strategist: str) -> str:
    return "VPAI" if strategist == chart.vanilla_label else strategist


def curve_series(chart: CurveChart) -> list[dict]:
    """The chart series in deterministic display order.

    Strategists follow ``strategist_order`` and conditions ``condition_order``;
    the VPAI reference curve comes last so it draws on top. An empty condition
    labels the curve with its strategist alone.
    """
    vanilla = chart.vanilla_label
    frame = chart.frame
    if frame.empty:
        return []
    strategist_rank = {name: i for i, name in enumerate(chart.strategist_order)}
    condition_rank = {name: i for i, name in enumerate(chart.condition_order)}
    colors = chart.strategist_colors
    series: dict[tuple, dict] = {}
    for row in frame.to_dict("records"):
        condition = row["condition"]
        condition = "" if condition is None or condition != condition else str(condition)
        key = (str(row["strategist"]), condition)
        if key not in series:
            is_vanilla = key == (vanilla, vanilla)
            if is_vanilla:
                color = colors.get(vanilla, _FALLBACK_COLOR)
            else:
                color = colors.get(key[0], colors.get(vanilla, _FALLBACK_COLOR))
            name = strategist_label(chart, key[0])
            series[key] = {
                "strategist": key[0],
                "condition": key[1],
                "label": "VPAI" if is_vanilla else (f"{name} · {key[1]}" if key[1] else name),
                "vanilla": is_vanilla,
                "color": color,
                "dash": _DASH_PATTERNS[condition_rank.get(key[1], 0) % len(_DASH_PATTERNS)],
                "width": 3.5 if is_vanilla else 2.0,
                "points": [],
            }
        series[key]["points"].append(
            [round(float(row["turn_progress"]), 2), round(float(row["mean_predicted_win_probability"]), 6)]
        )
    ordered = sorted(
        series.values(),
        key=lambda s: (
            s["vanilla"],
            strategist_rank.get(s["strategist"], len(strategist_rank)),
            s["strategist"],
            condition_rank.get(s["condition"], len(condition_rank)),
            s["condition"],
        ),
    )
    for entry in ordered:
        entry["points"].sort(key=lambda point: point[0])
    return ordered


def curve_figure(series: list[dict]):
    """Build the probability chart with stable trace metadata."""
    import plotly.graph_objects as go

    figure = go.Figure()
    for entry in series:
        points = entry["points"]
        figure.add_trace(go.Scatter(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            mode="lines",
            name=_esc(entry["label"]),
            meta={
                "strategist": entry["strategist"],
                "condition": entry["condition"],
                "vanilla": entry["vanilla"],
                "base_color": entry["color"],
                "base_width": entry["width"],
            },
            line={"color": entry["color"], "dash": entry["dash"], "width": entry["width"]},
            hovertemplate="%{fullData.name}: %{y:.3f}<extra></extra>",
        ))
    figure.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 60, "r": 20, "t": 18, "b": 56},
        hovermode="x unified",
        legend={
            "orientation": "h", "y": -0.22,
            "itemclick": False, "itemdoubleclick": False,
        },
        xaxis={"title": "Turn progress", "range": [0, 1], "tickformat": ".2f"},
        yaxis={
            "title": "Mean predicted win probability", "autorange": True,
            "autorangeoptions": {"clipmin": 0, "clipmax": 1},
            "tickformat": ".2f",
        },
    )
    return figure


def render_curve_chart_html(
    chart: CurveChart,
    name: str,
    plotly_src: str,
    tips: Optional[dict[str, str]] = None,
    query_select: bool = False,
) -> str:
    """The strategist checkboxes plus the Plotly chart, in one container.

    ``name`` makes the element ids unique on the page (the chart div is
    ``plotly-<name>``); ``plotly_src`` is the page-relative path of the local
    Plotly bundle. ``tips`` maps a strategist to its checkbox tooltip (VPAI
    defaults to :data:`VPAI_MIXED_TIP`). With
    ``query_select``, the page's ``?strategist=`` and ``?condition=`` query
    preselects one strategist and emphasizes one condition.
    """
    series = curve_series(chart)
    strategists: list[str] = []
    for entry in series:
        if not entry["vanilla"] and entry["strategist"] not in strategists:
            strategists.append(entry["strategist"])
    tips = {chart.vanilla_label: VPAI_MIXED_TIP, **(tips or {})}
    select = "true" if query_select else "false"
    parts = [f'<div class="curve-chart" data-query-select="{select}">']
    if strategists:
        parts.append(
            f'<div class="chart-controls" id="{_esc(name)}-filters" role="group" '
            'aria-label="Strategists">'
        )
        parts.append('<span class="controls-label">Strategists</span>')
        for strategist in strategists:
            tooltip = tips.get(strategist, "")
            tip_attr = f' data-tip="{_esc(tooltip)}"' if tooltip else ""
            parts.append(
                f'<label class="strategist-check"{tip_attr}>'
                f'<input type="checkbox" value="{_esc(strategist)}" checked> '
                f"{_esc(strategist_label(chart, strategist))}</label>"
            )
        parts.append("</div>")
    parts.append(figure_html(
        curve_figure(series), name, full_html=False, include_plotlyjs=plotly_src,
    ))
    parts.append("</div>")
    return "\n".join(parts)


# ── the static browser script ─────────────────────────────────────────────────
CURVE_CHART_JS = """/* civ-bench victory-probability curve charts.
   Strategist checkboxes, same-family color spreading (the shared
   civBench.distinguishColors util in assets/report-common.js), condition
   emphasis, and the adaptive Y axis for every .curve-chart on the page. */
(function () {
  "use strict";

  function readQuery() {
    var params = {};
    var search = new URLSearchParams(window.location.search);
    search.forEach(function (value, key) { params[key] = value; });
    return params;
  }

  function initChart(container) {
    var chart = container.querySelector(".plotly-graph-div");
    if (!chart || !window.Plotly || !chart.data) {
      return;
    }
    var query = container.dataset.querySelect === "true" ? readQuery() : {};
    var highlightedCondition = query.condition || null;

    var boxes = Array.prototype.slice.call(
      container.querySelectorAll('.chart-controls input[type="checkbox"]')
    );
    // A query link focuses its strategist (a strategist with no checkbox, such
    // as the VPAI reference, keeps everyone checked); direct entry also keeps
    // everyone checked.
    if (query.strategist) {
      var known = boxes.some(function (box) {
        return box.value === query.strategist;
      });
      if (known) {
        boxes.forEach(function (box) {
          box.checked = box.value === query.strategist;
        });
      }
    }

    // Same-family strategists can share a catalog color; the shared util
    // spreads their hues so their curves stay distinguishable.
    var colorMap = null;
    if (window.civBench && window.civBench.distinguishColors) {
      colorMap = window.civBench.distinguishColors(
        chart.data.map(function (trace) {
          return { key: trace.meta.strategist, color: trace.meta.base_color };
        })
      );
    }

    function checkedMap() {
      var checked = {};
      boxes.forEach(function (box) { checked[box.value] = box.checked; });
      return checked;
    }

    function updateChart(checked) {
      var visible = [];
      var colors = [];
      var widths = [];
      chart.data.forEach(function (trace) {
        var meta = trace.meta;
        visible.push(meta.vanilla || !boxes.length || checked[meta.strategist]);
        colors.push((colorMap && colorMap[meta.strategist]) || meta.base_color);
        widths.push(meta.vanilla ? meta.base_width :
          (highlightedCondition === meta.condition ? 3 : meta.base_width));
      });
      window.Plotly.restyle(chart, {visible: visible, "line.color": colors,
        "line.width": widths}).then(function () {
          return window.Plotly.relayout(chart, {"yaxis.autorange": true});
        });
    }

    function update() {
      var checked = checkedMap();
      updateChart(checked);
      container.dispatchEvent(new CustomEvent("curvechart:change", {
        bubbles: true,
        detail: { checked: checked, filtered: boxes.length > 0 }
      }));
    }

    boxes.forEach(function (box) {
      box.addEventListener("change", update);
    });
    update();
  }

  document.addEventListener("DOMContentLoaded", function () {
    Array.prototype.forEach.call(
      document.querySelectorAll(".curve-chart"), initChart
    );
  });
})();
"""
