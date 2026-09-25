"""The shared interactive victory-probability curve chart.

One chart, two homes: each Matched Maps seat page (curves for one seed and
seat) and the family page of ``performance.turn_predicted`` (curves pooled over
every game). Both render :class:`~bench.reports.model.CurveChart` objects through
:func:`render_curve_views_html`: a strategist checkbox group above an offline
Plotly chart, or, with Relative (adjusted strength) and Absolute (probability)
charts, one such chart per tab with the checkboxes kept in step. Every value, color, label, and the default selection (VPAI
plus the best and worst strategists) is computed server-side, so unchanged
inputs re-render byte-identically. The static :data:`CURVE_CHART_JS` asset only
changes trace visibility, color, and emphasis, previews or highlights curves on
checkbox and legend hover, refits the Y axis to the visible curves, and spreads
same-family strategist colors through the shared
:js:func:`civBench.distinguishColors` util in :mod:`bench.reports.assets`.

Each change of the checkboxes dispatches a bubbling ``curvechart:change`` event
on the chart container whose ``detail`` carries ``checked`` (strategist to
bool) and ``filtered`` (whether the chart has checkboxes at all), so a page can
filter related content, such as the seat page's game rows.
"""

from __future__ import annotations

import html as _html
from typing import Optional

from bench.plotting.interactive import figure_html

from .content import render_view_group
from .model import CurveChart

# Plotly dash styles cycled by condition position, so one strategist's
# conditions stay distinguishable while sharing its color.
_DASH_PATTERNS = ("solid", "dash", "dot", "dashdot")
_FALLBACK_COLOR = "#555555"

# Why the Absolute curves average raw probabilities, for the chart help and tab
# (performance.turn_predicted carries the same sentence).
LINEAR_MEAN_NOTE = (
    "Probabilities are averaged linearly to compare with their expectations "
    "(12.5% in 8-player games, for example)."
)
ABSOLUTE_TIP = "Mean predicted win probability, averaged linearly"

# The tooltip of VPAI seats in games with LLM players.
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
    if frame.empty or chart.value_column not in frame.columns:
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
            [round(float(row["turn_progress"]), 2), round(float(row[chart.value_column]), 6)]
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


def curve_figure(series: list[dict], chart: Optional[CurveChart] = None):
    """Build the curve chart with stable trace metadata.

    ``chart`` supplies the vertical axis title and the optional dotted
    reference line (0.5 for adjusted strength).
    """
    import plotly.graph_objects as go

    y_title = chart.y_title if chart is not None else "Mean predicted win probability"
    reference = chart.reference_line if chart is not None else None
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
        # A vertical legend beside the plot scrolls when long instead of
        # taking height from the plot area.
        legend={
            "x": 1.02, "y": 1, "xanchor": "left", "yanchor": "top",
            "itemclick": False, "itemdoubleclick": False,
        },
        xaxis={"title": "Turn progress", "range": [0, 1], "tickformat": ".2f"},
        yaxis={
            "title": y_title, "autorange": True,
            "autorangeoptions": {"clipmin": 0, "clipmax": 1},
            "tickformat": ".2f",
        },
    )
    if reference is not None:
        figure.add_hline(
            y=reference, line={"color": "#8a96a8", "width": 1, "dash": "dot"},
        )
    return figure


def _vpai_tip(series: list[dict], vanilla: str) -> str:
    """The VPAI checkbox tooltip, covering whichever VPAI curves exist."""
    has_reference = any(entry["vanilla"] for entry in series)
    has_mixed = any(
        entry["strategist"] == vanilla and not entry["vanilla"] for entry in series
    )
    if has_reference and has_mixed:
        return (
            "VPAI self-play (the thick line): all players use VPAI. The other "
            "VPAI lines are " + VPAI_MIXED_TIP
        )
    if has_reference:
        return "VPAI self-play: all players use VPAI."
    return VPAI_MIXED_TIP


def default_strategists(series: list[dict], vanilla: str) -> list[str]:
    """The strategists checked by default: VPAI plus the best and the worst.

    Best and worst rank the non-VPAI strategists by the mean of all their
    curve points (every condition pooled); ties go to the earlier strategist
    in display order. With two or fewer such strategists, all are checked.
    """
    values: dict[str, list[float]] = {}
    for entry in series:
        if entry["strategist"] != vanilla:
            values.setdefault(entry["strategist"], []).extend(p[1] for p in entry["points"])
    chosen = [vanilla] if any(entry["strategist"] == vanilla for entry in series) else []
    if len(values) <= 2:
        return chosen + list(values)
    means = {name: sum(ys) / len(ys) for name, ys in values.items() if ys}
    order = list(means)
    best = max(order, key=lambda name: (means[name], -order.index(name)))
    worst = min(order, key=lambda name: (means[name], order.index(name)))
    return chosen + [best] + ([worst] if worst != best else [])


def render_curve_chart_html(
    chart: CurveChart,
    name: str,
    plotly_src: str,
    tips: Optional[dict[str, str]] = None,
    query_select: bool = False,
    defaults: Optional[list[str]] = None,
    sync: str = "",
    include_plotly: bool = True,
) -> str:
    """The strategist checkboxes plus the Plotly chart, in one container.

    ``name`` makes the element ids unique on the page (the chart div is
    ``plotly-<name>``); ``plotly_src`` is the page-relative path of the local
    Plotly bundle. ``tips`` maps a strategist to its checkbox tooltip (VPAI
    gets one describing its curves). The VPAI checkbox comes first and
    controls every VPAI curve, the self-play reference included. Only VPAI and
    the best and worst strategists (:func:`default_strategists`) start checked,
    so the chart stays readable; two buttons check all of them or restore
    that default. With ``query_select``, the page's ``?strategist=`` and
    ``?condition=`` query preselects one strategist (plus VPAI) and
    emphasizes one condition.

    ``defaults`` overrides the default selection, ``sync`` names a group of
    charts whose checkboxes change together, and ``include_plotly`` adds the
    Plotly script tag (only the first chart on a page needs it).
    """
    series = curve_series(chart)
    vanilla = chart.vanilla_label
    strategists: list[str] = []
    if any(entry["strategist"] == vanilla for entry in series):
        strategists.append(vanilla)
    for entry in series:
        if entry["strategist"] not in strategists:
            strategists.append(entry["strategist"])
    defaults = set(default_strategists(series, vanilla) if defaults is None else defaults)
    tips = {vanilla: _vpai_tip(series, vanilla), **(tips or {})}
    select = "true" if query_select else "false"
    sync_attr = f' data-sync="{_esc(sync)}"' if sync else ""
    parts = [f'<div class="curve-chart" data-query-select="{select}"{sync_attr}>']
    if strategists:
        parts.append(
            f'<div class="chart-controls" id="{_esc(name)}-filters" role="group" '
            'aria-label="Strategists">'
        )
        parts.append('<span class="controls-label">Strategists</span>')
        for strategist in strategists:
            tooltip = tips.get(strategist, "")
            tip_attr = f' data-tip="{_esc(tooltip)}"' if tooltip else ""
            state = ' checked data-default="true"' if strategist in defaults else ""
            parts.append(
                f'<label class="strategist-check"{tip_attr}>'
                f'<input type="checkbox" value="{_esc(strategist)}"{state}> '
                f"{_esc(strategist_label(chart, strategist))}</label>"
            )
        if len(defaults) < len(strategists):
            parts.append(
                '<span class="chart-presets">'
                '<button type="button" class="chart-preset" data-preset="all">All</button>'
                '<button type="button" class="chart-preset" data-preset="default" '
                'data-tip="VPAI plus the strategists with the highest and lowest '
                'mean curve value">Best and worst</button></span>'
            )
        parts.append("</div>")
    parts.append(figure_html(
        curve_figure(series, chart), name, full_html=False,
        include_plotlyjs=plotly_src if include_plotly else False,
    ))
    parts.append("</div>")
    return "\n".join(parts)


def render_curve_views_html(
    charts: list[CurveChart],
    name: str,
    plotly_src: str,
    tips: Optional[dict[str, str]] = None,
    query_select: bool = False,
    captions: bool = False,
) -> str:
    """One chart, or several alternative charts behind the shared view tabs.

    A single chart renders as :func:`render_curve_chart_html`. Several charts
    (such as Relative and Absolute) each get a tab from
    :func:`bench.reports.content.render_view_group`, chart ids
    ``plotly-<name>-<view>``, and one shared default selection taken from the
    first chart; their checkboxes stay in step, so switching tabs compares
    the same strategists. With ``captions``, each chart's ``help`` renders as a
    caption below it.
    """
    def caption(chart: CurveChart) -> str:
        if captions and chart.help:
            return f'\n<p class="caption">{_esc(chart.help)}</p>'
        return ""

    if len(charts) == 1:
        chart = charts[0]
        return render_curve_chart_html(
            chart, name, plotly_src, tips=tips, query_select=query_select,
        ) + caption(chart)
    first = curve_series(charts[0])
    defaults = default_strategists(first, charts[0].vanilla_label)
    views = []
    for index, chart in enumerate(charts):
        view = chart.view or f"view-{index + 1}"
        body = render_curve_chart_html(
            chart, f"{name}-{view}", plotly_src, tips=tips, query_select=query_select,
            defaults=defaults, sync=name, include_plotly=index == 0,
        ) + caption(chart)
        views.append((view, chart.label or view, chart.tip, body))
    return render_view_group(f"{name}-curves", views)


# ── the static browser script ─────────────────────────────────────────────────
CURVE_CHART_JS = """/* civ-bench victory-probability curve charts.
   Strategist checkboxes (VPAI's covers every VPAI curve), the All and Best
   and worst presets, hover previews from the checkboxes and the legend,
   same-family color spreading (the shared civBench.distinguishColors util in
   assets/report-common.js), condition emphasis, and the adaptive Y axis for
   every .curve-chart on the page. Charts sharing a data-sync name (the
   Relative and Absolute tabs of one chart) keep their checkboxes in step,
   and a chart redraws at full width when its hidden tab opens. */
(function () {
  "use strict";

  var DIMMED_OPACITY = 0.2;
  var FOCUS_EXTRA_WIDTH = 1.5;
  // Sync group name to the charts in it, each with an apply(checked) hook.
  var syncGroups = {};

  function readQuery() {
    var params = {};
    var search = new URLSearchParams(window.location.search);
    search.forEach(function (value, key) { params[key] = value; });
    return params;
  }

  // The trace index of a Plotly legend entry: its bound legend item first,
  // then its label text as a fallback.
  function legendIndex(chart, item) {
    var bound = item.__data__;
    if (bound && bound[0] && bound[0].trace && typeof bound[0].trace.index === "number") {
      return bound[0].trace.index;
    }
    var text = item.querySelector(".legendtext");
    if (!text) { return null; }
    for (var i = 0; i < chart.data.length; i++) {
      if (chart.data[i].name === text.textContent) { return i; }
    }
    return null;
  }

  function initChart(container) {
    var chart = container.querySelector(".plotly-graph-div");
    if (!chart || !window.Plotly || !chart.data) {
      return;
    }
    var query = container.dataset.querySelect === "true" ? readQuery() : {};
    var highlightedCondition = query.condition || null;
    // The hover preview: {strategist: name} from a checkbox, {index: i} from
    // the legend, or null.
    var focus = null;
    var lastVisible = null;

    var boxes = Array.prototype.slice.call(
      container.querySelectorAll('.chart-controls input[type="checkbox"]')
    );
    var vpaiBox = boxes.length ? boxes[0] : null;
    // A query link focuses its strategist next to VPAI (an unknown strategist
    // keeps the default selection).
    if (query.strategist) {
      var known = boxes.some(function (box) {
        return box.value === query.strategist;
      });
      if (known) {
        boxes.forEach(function (box) {
          box.checked = box.value === query.strategist || box === vpaiBox;
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

    function inFocus(meta, index) {
      if (!focus) { return false; }
      if (focus.index !== undefined) { return focus.index === index; }
      return focus.strategist === meta.strategist;
    }

    function updateChart() {
      var checked = checkedMap();
      var visible = [];
      var inLegend = [];
      var colors = [];
      var widths = [];
      var opacities = [];
      chart.data.forEach(function (trace, index) {
        var meta = trace.meta;
        var focused = inFocus(meta, index);
        // A hovered checkbox previews its unchecked curves. Previews stay out
        // of the legend so its entries do not reflow on every hover.
        var shown = !boxes.length || checked[meta.strategist] === true;
        visible.push(shown || (focused && focus.strategist !== undefined));
        inLegend.push(shown);
        colors.push((colorMap && colorMap[meta.strategist]) || meta.base_color);
        var width = !meta.vanilla && highlightedCondition === meta.condition ?
          3 : meta.base_width;
        widths.push(focused ? width + FOCUS_EXTRA_WIDTH : width);
        opacities.push(focus && !focused ? DIMMED_OPACITY : 1);
      });
      var key = visible.join(",");
      window.Plotly.restyle(chart, {visible: visible, showlegend: inLegend,
        "line.color": colors,
        "line.width": widths, opacity: opacities}).then(function () {
          // Refit the Y axis only when the visible curves change.
          if (key === lastVisible) { return null; }
          lastVisible = key;
          return window.Plotly.relayout(chart, {"yaxis.autorange": true});
        });
    }

    function setFocus(next) {
      var same = (focus === null && next === null) || (focus && next &&
        focus.index === next.index && focus.strategist === next.strategist);
      if (same) { return; }
      focus = next;
      updateChart();
    }

    function update() {
      updateChart();
      var checked = checkedMap();
      container.dispatchEvent(new CustomEvent("curvechart:change", {
        bubbles: true,
        detail: { checked: checked, filtered: boxes.length > 0 }
      }));
      return checked;
    }

    // A user change here updates the other charts of the sync group.
    var sync = container.dataset.sync || "";
    var self = {
      apply: function (checked) {
        boxes.forEach(function (box) {
          if (Object.prototype.hasOwnProperty.call(checked, box.value)) {
            box.checked = checked[box.value];
          }
        });
        updateChart();
      }
    };
    if (sync) {
      (syncGroups[sync] = syncGroups[sync] || []).push(self);
    }
    function userUpdate() {
      var checked = update();
      (syncGroups[sync] || []).forEach(function (other) {
        if (other !== self) { other.apply(checked); }
      });
    }

    // A chart drawn or resized inside a hidden tab keeps a stale width;
    // redraw it once its tab shows it at a new width.
    if (window.ResizeObserver) {
      var lastWidth = container.clientWidth;
      new window.ResizeObserver(function () {
        var width = container.clientWidth;
        if (width && width !== lastWidth) {
          lastWidth = width;
          window.Plotly.Plots.resize(chart);
        }
      }).observe(container);
    }

    boxes.forEach(function (box) {
      box.addEventListener("change", userUpdate);
      var label = box.closest("label") || box;
      label.addEventListener("mouseenter", function () {
        setFocus({strategist: box.value});
      });
      label.addEventListener("mouseleave", function () { setFocus(null); });
    });
    Array.prototype.forEach.call(
      container.querySelectorAll(".chart-preset"),
      function (button) {
        button.addEventListener("click", function () {
          var all = button.dataset.preset === "all";
          boxes.forEach(function (box) {
            box.checked = all || box.dataset.default === "true";
          });
          userUpdate();
        });
      }
    );

    // Legend hover highlights one curve. Plotly redraws the legend on every
    // restyle, so the listeners sit on the chart and find the entry.
    chart.addEventListener("mouseover", function (event) {
      var item = event.target.closest ? event.target.closest(".legend .traces") : null;
      var index = item ? legendIndex(chart, item) : null;
      if (index !== null) {
        setFocus({index: index});
      } else if (focus && focus.index !== undefined) {
        setFocus(null);
      }
    });
    chart.addEventListener("mouseleave", function () {
      if (focus && focus.index !== undefined) { setFocus(null); }
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
