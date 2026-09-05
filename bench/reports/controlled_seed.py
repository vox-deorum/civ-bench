"""Render a :class:`ControlledSeedDocument` to the controlled-seed HTML site (stage 5).

The site is self-contained and deterministic: one overview page with two heatmap
tables per controlled seed (mean adjusted strength on a fixed 0-1 scale, and the
dominant victory focus with a stable categorical color per strategy), one detail
page per ``(seed, player_id)`` pair, and one static vanilla-JavaScript asset for
tooltips, the strategist selector, and the probability-curve chart. Every value,
color, link, and query string is computed server-side, so unchanged inputs
re-render byte-identically; the browser script only reads embedded data.

Rows are strategist-condition combinations and columns are final ``player_id``
values (the orientation that keeps the dedicated Vanilla condition a separate,
visually isolated row). The renderer completes the global row and column grid
and leaves unobserved combinations blank.
"""

from __future__ import annotations

import html as _html
import json
from urllib.parse import urlencode

import numpy as np

from .model import ControlledSeedDocument

# Stable categorical colors for the four victory-focus strategies.
FOCUS_COLORS = {
    "Domination": "#b3452f",
    "Culture": "#c28e21",
    "Diplomatic": "#4f81bd",
    "Science": "#4e9b4e",
}

# The adjusted-strength heatmap scale is fixed from 0 to 1 across every seed so
# the panels compare directly.
_STRENGTH_SCALE = ("#f2f6fa", "#1f5fa8")
_TEXT_DARK = "#18202a"
_TEXT_LIGHT = "#ffffff"

# SVG dash patterns cycled by condition position, so one strategist's conditions
# stay distinguishable while sharing its color.
_DASH_PATTERNS = ("", "6 4", "2 3", "8 3 2 3")

_CHART_WIDTH = 760
_CHART_HEIGHT = 420
_CHART_MARGIN = {"left": 52, "right": 18, "top": 18, "bottom": 44}


# ── small formatting / color helpers ──────────────────────────────────────────
def _esc(text) -> str:
    return _html.escape(str(text))


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _interpolate(start: str, end: str, t: float) -> str:
    a, b = _hex_to_rgb(start), _hex_to_rgb(end)
    return _rgb_to_hex(tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3)))


def _luminance(rgb: tuple[int, int, int]) -> float:
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255.0


def _cell_text_color(background: str) -> str:
    return _TEXT_LIGHT if _luminance(_hex_to_rgb(background)) < 0.55 else _TEXT_DARK


def _strength_background(value: float) -> str:
    t = 0.0 if not np.isfinite(value) else min(1.0, max(0.0, value))
    return _interpolate(_STRENGTH_SCALE[0], _STRENGTH_SCALE[1], t)


def _focus_background(label: str, pct: float) -> str:
    base = FOCUS_COLORS.get(label, "#7a7f88")
    t = 0.0 if not np.isfinite(pct) else min(1.0, max(0.0, pct / 100.0))
    intensity = 0.22 + 0.78 * t
    return _interpolate("#ffffff", base, intensity)


def _page_filename(seed: int, player_id: int) -> str:
    return f"seed-{seed}-player-{player_id}.html"


def _detail_link(seed: int, player_id: int, strategist: str, condition: str) -> str:
    query = urlencode({"strategist": strategist, "condition": condition})
    return f"{_page_filename(seed, player_id)}?{query}"


def _fmt_probability(value) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.4f}"


def _fmt_signed(value) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:+.4f}"


def _fmt_pct(value) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.1f}"


def _fmt_pct_short(value) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.0f}%"


# ── document views ────────────────────────────────────────────────────────────
def _summary_lookup(doc: ControlledSeedDocument) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for row in doc.summary_table.to_dict("records"):
        out[(int(row["seed"]), int(row["player_id"]), str(row["strategist"]), str(row["condition"]))] = row
    return out


def _grid(doc: ControlledSeedDocument) -> tuple[list[str], list[tuple[str, str]], list[int], list[int]]:
    """The global row and column grid: observed combos plus player columns.

    Rows list the Vanilla pair first (rendered as its own isolated row), then
    strategist-condition pairs in strategist order and condition order. Columns
    are every observed final player position, numerically sorted.
    """
    vanilla = doc.vanilla_label
    combos = [
        (str(r["strategist"]), str(r["condition"]))
        for r in doc.summary_table.to_dict("records")
        if str(r["strategist"]) != vanilla or str(r["condition"]) != vanilla
    ]
    strategist_rank = {name: i for i, name in enumerate(doc.strategist_order)}
    condition_rank = {name: i for i, name in enumerate(doc.condition_order)}
    combos = sorted(
        set(combos),
        key=lambda pair: (
            strategist_rank.get(pair[0], len(strategist_rank)),
            pair[0],
            condition_rank.get(pair[1], len(condition_rank)),
            pair[1],
        ),
    )
    seeds = sorted({int(r["seed"]) for r in doc.index_table.to_dict("records")})
    players = sorted({int(r["player_id"]) for r in doc.index_table.to_dict("records")})
    return [vanilla], combos, seeds, players


def _has_vanilla_rows(doc: ControlledSeedDocument) -> bool:
    vanilla = doc.vanilla_label
    mask = (doc.summary_table["strategist"] == vanilla) & (
        doc.summary_table["condition"] == vanilla
    )
    return bool(mask.any())


# ── heatmaps ──────────────────────────────────────────────────────────────────
def _heat_cell(row: dict, seed: int, player_id: int, kind: str) -> str:
    """One populated heatmap cell: colored, tooltip-equipped, and clickable."""
    strategist, condition = str(row["strategist"]), str(row["condition"])
    if kind == "strength":
        value = float(row["mean_adjusted_strength"])
        background = _strength_background(value)
        if np.isfinite(value):
            text = f"{value:.2f}"
            value_line = f"Adjusted strength: {value:.4f}"
        else:
            text = ""
            value_line = "Adjusted strength: unavailable"
        tip = (
            f"{value_line}\n"
            f"Civilization: {row['civilization']}\n"
            f"Runs: {int(row['run_count'])}"
        )
    else:
        label = str(row["dominant_focus"])
        pct = float(row["dominant_focus_pct"])
        background = _focus_background(label, pct)
        text = f"{label} {_fmt_pct_short(pct)}" if np.isfinite(pct) else ""
        if np.isfinite(pct):
            value_line = f"Dominant focus: {label} ({pct:.2f}%)"
        else:
            value_line = "Dominant focus: unavailable"
        tip = (
            f"{value_line}\n"
            f"Civilization: {row['civilization']}\n"
            f"Runs: {int(row['run_count'])}"
        )
    color = _cell_text_color(background)
    href = _detail_link(seed, player_id, strategist, condition)
    return (
        f'<td class="heat-cell" style="background-color:{background};'
        f'color:{color}" data-tip="{_esc(tip)}">'
        f'<a href="{_esc(href)}" style="color:inherit">{_esc(text)}</a></td>'
    )


def _empty_heat_cell() -> str:
    return '<td class="heat-cell heat-cell-empty" aria-label="no data"></td>'


def _heatmap(
    doc: ControlledSeedDocument,
    summary: dict[tuple, dict],
    seed: int,
    players: list[int],
    combos: list[tuple[str, str]],
    kind: str,
) -> str:
    vanilla = doc.vanilla_label
    parts: list[str] = []
    parts.append('<div class="table-scroll heat-scroll" role="region" tabindex="0">')
    label = "Mean adjusted strength" if kind == "strength" else "Dominant victory focus"
    parts.append(
        f'<table class="heatmap {"strength" if kind == "strength" else "focus"}-heatmap">'
        f'<caption class="sr-only">Seed {seed}: {label.lower()} by strategist and '
        "condition row and final player position</caption>"
    )
    parts.append("<thead><tr><th scope=\"col\" class=\"row-label\">Strategist | Condition</th>")
    for player_id in players:
        parts.append(f'<th scope="col">Player {player_id}</th>')
    parts.append("</tr></thead>")

    def cells_for(strategist: str, condition: str) -> list[str]:
        out = []
        for player_id in players:
            row = summary.get((seed, player_id, strategist, condition))
            if row is None:
                out.append(_empty_heat_cell())
            else:
                out.append(_heat_cell(row, seed, player_id, kind))
        return out

    if _has_vanilla_rows(doc):
        parts.append('<tbody class="vanilla-body">')
        parts.append(
            f'<tr class="vanilla-row"><th scope="row" class="row-label">'
            f"{_esc(vanilla)}</th>"
        )
        parts.extend(cells_for(vanilla, vanilla))
        parts.append("</tr></tbody>")

    parts.append("<tbody>")
    for strategist, condition in combos:
        row_label = f"{strategist} | {condition}"
        parts.append(
            f'<tr><th scope="row" class="row-label">{_esc(row_label)}</th>'
        )
        parts.extend(cells_for(strategist, condition))
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _focus_legend() -> str:
    items = "".join(
        f'<li><span class="swatch" style="background-color:{color}"></span>'
        f"{_esc(label)}</li>"
        for label, color in FOCUS_COLORS.items()
    )
    return f'<ul class="focus-legend">{items}</ul>'


# ── overview page ─────────────────────────────────────────────────────────────
def _render_overview(doc: ControlledSeedDocument) -> str:
    summary = _summary_lookup(doc)
    _, combos, seeds, players = _grid(doc)
    parts: list[str] = []
    parts.append('<!DOCTYPE html>')
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>{_esc(doc.title)}</title>")
    parts.append('<link rel="stylesheet" href="assets/report.css">')
    parts.append("</head><body>")
    parts.append('<a class="skip-link" href="#main-content">Skip to content</a>')
    parts.append('<main class="content controlled-content" id="main-content">')
    parts.append(f"<h1>{_esc(doc.title)}</h1>")
    if doc.description:
        parts.append(f'<p class="caption">{_esc(doc.description)}</p>')
    parts.append(
        f'<p class="meta">Run {_esc(doc.run_name)} · seed {doc.seed} · '
        f"config <code>{_esc(doc.config_path)}</code></p>"
    )
    parts.append(f"<p>{_esc(doc.summary)}</p>")
    meta_bits = []
    if doc.metadata.get("baseline_experiment"):
        meta_bits.append(
            f"dedicated baseline: {_esc(doc.metadata['baseline_experiment'])}"
        )
    if doc.metadata.get("estimator"):
        meta_bits.append(f"estimator: {_esc(doc.metadata['estimator'])}")
    if doc.metadata.get("strength_table"):
        meta_bits.append(f"strength table: {_esc(doc.metadata['strength_table'])}")
    if meta_bits:
        parts.append(f'<p class="meta">{"; ".join(meta_bits)}</p>')
    parts.append(
        '<p class="caption">Each cell averages every unique run for its seed, final '
        "player position, strategist, and condition; seating rotations and repeated "
        "runs contribute equally. Click a cell to open the matching seed-player "
        "detail page.</p>"
    )

    for seed in seeds:
        parts.append(f'<section aria-labelledby="seed-{seed}-heading">')
        parts.append(f'<h2 id="seed-{seed}">Seed {seed}</h2>')
        parts.append('<figure class="heat-figure">')
        parts.append("<figcaption>Mean adjusted strength (fixed 0 to 1 scale)</figcaption>")
        parts.append(_heatmap(doc, summary, seed, players, combos, "strength"))
        parts.append("</figure>")
        parts.append('<figure class="heat-figure">')
        parts.append("<figcaption>Dominant victory focus (largest mean share)</figcaption>")
        parts.append(_heatmap(doc, summary, seed, players, combos, "focus"))
        parts.append("</figure>")
        parts.append(_focus_legend())
        parts.append("</section>")

    if doc.downloads:
        parts.append('<section aria-labelledby="source-tables-heading">')
        parts.append('<h2 id="source-tables-heading">Source tables</h2>')
        parts.append("<ul>")
        for download in doc.downloads:
            parts.append(
                f'<li><a href="{_esc(download.rel_path)}">{_esc(download.label)}</a></li>'
            )
        parts.append("</ul></section>")

    parts.append("</main>")
    parts.append('<script src="assets/controlled-seed-report.js" defer></script>')
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


# ── detail pages ──────────────────────────────────────────────────────────────
def _page_series(
    doc: ControlledSeedDocument, seed: int, player_id: int
) -> list[dict]:
    """The chart series for one seed-player page, in deterministic display order."""
    vanilla = doc.vanilla_label
    frame = doc.probability_table
    if frame.empty:
        return []
    page = frame[
        (frame["seed"] == seed) & (frame["player_id"] == player_id)
    ]
    if page.empty:
        return []
    strategist_rank = {name: i for i, name in enumerate(doc.strategist_order)}
    condition_rank = {name: i for i, name in enumerate(doc.condition_order)}
    colors = doc.strategist_colors
    series: dict[tuple, dict] = {}
    for row in page.to_dict("records"):
        key = (str(row["strategist"]), str(row["condition"]))
        if key not in series:
            is_vanilla = key == (vanilla, vanilla)
            if is_vanilla:
                color = colors.get(vanilla, "#555555")
            else:
                color = colors.get(key[0], colors.get(vanilla, "#555555"))
            series[key] = {
                "strategist": key[0],
                "condition": key[1],
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


def _comparison_rows(doc: ControlledSeedDocument, seed: int, player_id: int) -> list[dict]:
    vanilla = doc.vanilla_label
    rows = [
        row
        for row in doc.summary_table.to_dict("records")
        if int(row["seed"]) == seed and int(row["player_id"]) == player_id
    ]
    strategist_rank = {name: i for i, name in enumerate(doc.strategist_order)}
    condition_rank = {name: i for i, name in enumerate(doc.condition_order)}
    return sorted(
        rows,
        key=lambda row: (
            str(row["strategist"]) != vanilla,
            strategist_rank.get(str(row["strategist"]), len(strategist_rank)),
            str(row["strategist"]),
            condition_rank.get(str(row["condition"]), len(condition_rank)),
            str(row["condition"]),
        ),
    )


def _comparison_table(doc: ControlledSeedDocument, seed: int, player_id: int) -> str:
    vanilla = doc.vanilla_label
    parts: list[str] = []
    parts.append('<div class="table-scroll" role="region" tabindex="0">')
    parts.append('<table class="comparison">')
    parts.append(
        f'<caption class="sr-only">Seed {seed}, player {player_id}: per-condition means '
        "over every unique run occupying this final position</caption>"
    )
    headers = [
        "Strategist", "Condition", "Run count",
        "Mean weighted victory probability", "Mean adjusted strength",
        "Difference from matched Vanilla adjusted strength",
        "Dominant victory focus", "Domination focus %", "Culture focus %",
        "Diplomatic focus %", "Science focus %",
    ]
    parts.append("<thead><tr>")
    for header in headers:
        parts.append(f'<th scope="col">{_esc(header)}</th>')
    parts.append("</tr></thead><tbody>")
    for row in _comparison_rows(doc, seed, player_id):
        is_vanilla = str(row["strategist"]) == vanilla
        row_open = '<tr class="vanilla-row">' if is_vanilla else "<tr>"
        parts.append(row_open)
        parts.append(f"<td>{_esc(row['strategist'])}</td>")
        parts.append(f"<td>{_esc(row['condition'])}</td>")
        parts.append(f"<td>{int(row['run_count'])}</td>")
        parts.append(f"<td>{_esc(_fmt_probability(row['mean_weighted_victory_probability']))}</td>")
        strength_cell = (
            f'<td class="vanilla-value">{_esc(_fmt_probability(row["mean_adjusted_strength"]))}</td>'
            if is_vanilla
            else f"<td>{_esc(_fmt_probability(row['mean_adjusted_strength']))}</td>"
        )
        parts.append(strength_cell)
        parts.append(f"<td>{_esc(_fmt_signed(row['adjusted_strength_difference']))}</td>")
        dominant = (
            f"{row['dominant_focus']} {_fmt_pct_short(row['dominant_focus_pct'])}"
            if np.isfinite(row["dominant_focus_pct"])
            else ""
        )
        parts.append(f"<td>{_esc(dominant)}</td>")
        for column in (
            "domination_focus_pct",
            "culture_focus_pct",
            "diplomatic_focus_pct",
            "science_focus_pct",
        ):
            parts.append(f"<td>{_esc(_fmt_pct(row[column]))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _chart_data(doc: ControlledSeedDocument, seed: int, player_id: int) -> str:
    series = _page_series(doc, seed, player_id)
    strategists = []
    for entry in series:
        if not entry["vanilla"] and entry["strategist"] not in strategists:
            strategists.append(entry["strategist"])
    data = {
        "strategists": strategists,
        "showAllLabel": "Show all",
        "xLabel": "Turn progress",
        "yLabel": "Mean predicted win probability",
        "series": series,
        "chartWidth": _CHART_WIDTH,
        "chartHeight": _CHART_HEIGHT,
        "margin": _CHART_MARGIN,
    }
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    # Keep the JSON blob from closing its own script element.
    return text.replace("</", "<\\/")


def _render_detail(
    doc: ControlledSeedDocument,
    index_row: dict,
    prev_row: dict | None,
    next_row: dict | None,
) -> str:
    seed = int(index_row["seed"])
    player_id = int(index_row["player_id"])
    vanilla = doc.vanilla_label
    parts: list[str] = []
    parts.append('<!DOCTYPE html>')
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>Seed {seed} · Player {player_id} | {_esc(doc.title)}</title>")
    parts.append('<link rel="stylesheet" href="assets/report.css">')
    parts.append("</head><body>")
    parts.append('<a class="skip-link" href="#main-content">Skip to content</a>')
    parts.append('<main class="content controlled-content" id="main-content">')
    parts.append('<nav class="page-nav" aria-label="Seed player pages">')
    if prev_row is not None:
        prev_seed, prev_player = int(prev_row["seed"]), int(prev_row["player_id"])
        parts.append(f'<a href="{_page_filename(prev_seed, prev_player)}">← Player {prev_player}</a>')
    parts.append(f'<a href="report.html#seed-{seed}" class="return-link">Seed {seed} overview</a>')
    if next_row is not None:
        next_seed, next_player = int(next_row["seed"]), int(next_row["player_id"])
        parts.append(f'<a href="{_page_filename(next_seed, next_player)}">Player {next_player} →</a>')
    parts.append("</nav>")
    parts.append(f'<p class="eyebrow">{_esc(doc.title)}</p>')
    parts.append(f"<h1>Seed {seed} · Player {player_id}</h1>")
    parts.append(
        f'<p class="meta">Civilization: {_esc(index_row["civilization"])} · '
        f"{int(index_row['run_count'])} source run(s)</p>"
    )
    if int(index_row["n_civilizations"]) > 1:
        parts.append(
            '<p class="warning">Multiple civilizations occupy this seed-player pair '
            f"({_esc(index_row['civilization'])}); values are not directly comparable "
            "across runs.</p>"
        )
    if not bool(index_row["has_matched_vanilla"]):
        parts.append(
            '<p class="warning">The dedicated Vanilla baseline is unavailable for this '
            "seed-player pair; the reference curve and the difference column are "
            "omitted.</p>"
        )
    if not bool(index_row["has_probability"]):
        parts.append(
            '<p class="warning">No usable estimator prediction rows cover this '
            "seed-player pair; the victory-probability curve is unavailable.</p>"
        )

    parts.append('<section aria-labelledby="curves-heading">')
    parts.append('<h2 id="curves-heading">Victory-probability curves</h2>')
    parts.append('<div class="chart-controls">')
    parts.append('<label for="strategist-select">Strategist</label>')
    parts.append('<select id="strategist-select"></select>')
    parts.append("</div>")
    parts.append(
        '<div id="curve-chart" class="curve-chart" role="img" '
        'aria-label="Mean victory probability over normalized game progress"></div>'
    )
    parts.append('<ul id="curve-legend" class="curve-legend"></ul>')
    parts.append(
        '<p class="caption">Each curve is the mean of every run\'s interpolated '
        "victory probability on the fixed 0 to 1 progress grid; the "
        f"{_esc(vanilla)} reference curve is drawn thicker when present.</p>"
    )
    parts.append("</section>")

    parts.append('<section aria-labelledby="comparison-heading">')
    parts.append('<h2 id="comparison-heading">Comparison table</h2>')
    parts.append(_comparison_table(doc, seed, player_id))
    parts.append("</section>")

    parts.append("</main>")
    parts.append(
        f'<script type="application/json" id="curve-data">{_chart_data(doc, seed, player_id)}</script>'
    )
    parts.append('<script src="assets/controlled-seed-report.js" defer></script>')
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


# ── site assembly ─────────────────────────────────────────────────────────────
def render_controlled_seed_site(doc: ControlledSeedDocument) -> dict[str, str]:
    """Render the controlled overview and every seed-player page (+ the JS asset)."""
    pages: dict[str, str] = {"report.html": _render_overview(doc)}
    rows = doc.index_table.to_dict("records")
    by_seed: dict[int, list[dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    for seed in sorted(by_seed):
        page_rows = sorted(by_seed[seed], key=lambda r: int(r["player_id"]))
        for position, row in enumerate(page_rows):
            prev_row = page_rows[position - 1] if len(page_rows) > 1 else None
            next_row = page_rows[(position + 1) % len(page_rows)] if len(page_rows) > 1 else None
            filename = _page_filename(int(row["seed"]), int(row["player_id"]))
            pages[filename] = _render_detail(doc, row, prev_row, next_row)
    pages["assets/controlled-seed-report.js"] = CONTROLLED_SEED_JS
    return pages


# ── the static browser script ─────────────────────────────────────────────────
CONTROLLED_SEED_JS = """/* civ-bench controlled-seed report interactions.
   Deterministic vanilla JavaScript: no packages, no network. Tooltips on
   heatmap cells; strategist selection and the SVG probability-curve chart on
   seed-player detail pages. */
(function () {
  "use strict";

  // ── tooltips (overview heatmaps) ─────────────────────────────────────────
  var tooltip = null;

  function ensureTooltip() {
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.id = "heat-tooltip";
      tooltip.className = "heat-tooltip";
      tooltip.setAttribute("role", "tooltip");
      document.body.appendChild(tooltip);
    }
    return tooltip;
  }

  function showTooltip(target) {
    var tip = ensureTooltip();
    tip.textContent = target.getAttribute("data-tip") || "";
    tip.style.display = "block";
    var rect = target.getBoundingClientRect();
    var top = rect.top - tip.offsetHeight - 6;
    if (top < 4) {
      top = rect.bottom + 6;
    }
    tip.style.top = (window.scrollY + top) + "px";
    tip.style.left = (window.scrollX + rect.left) + "px";
  }

  function hideTooltip() {
    if (tooltip) {
      tooltip.style.display = "none";
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.addEventListener("mouseover", function (event) {
      var target = event.target.closest("[data-tip]");
      if (target) {
        showTooltip(target);
      }
    });
    document.addEventListener("mouseout", function (event) {
      if (event.target.closest && event.target.closest("[data-tip]")) {
        hideTooltip();
      }
    });
    document.addEventListener("focusin", function (event) {
      var target = event.target.closest && event.target.closest("[data-tip]");
      if (target) {
        showTooltip(target);
      }
    });
    document.addEventListener("focusout", hideTooltip);
  });

  // ── detail page: strategist selection + SVG curve chart ──────────────────
  function readQuery() {
    var params = {};
    var search = window.location.search.substring(1);
    if (!search) {
      return params;
    }
    search.split("&").forEach(function (pair) {
      var parts = pair.split("=");
      if (parts[0]) {
        params[decodeURIComponent(parts[0].replace(/\\+/g, " "))] =
          decodeURIComponent((parts[1] || "").replace(/\\+/g, " "));
      }
    });
    return params;
  }

  function svgTag(name, attributes) {
    var element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attributes).forEach(function (key) {
      element.setAttribute(key, attributes[key]);
    });
    return element;
  }

  function initChart() {
    var dataNode = document.getElementById("curve-data");
    var select = document.getElementById("strategist-select");
    var chartHost = document.getElementById("curve-chart");
    var legendHost = document.getElementById("curve-legend");
    if (!dataNode || !select || !chartHost) {
      return;
    }
    var data = JSON.parse(dataNode.textContent);
    if (!data.series.length) {
      chartHost.textContent = "No probability curves are available for this page.";
      return;
    }
    var query = readQuery();
    var preselectedCondition = query.condition || null;

    var options = data.strategists.slice();
    if (options.length > 1) {
      options.push(data.showAllLabel);
    }
    options.forEach(function (name) {
      var option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    });
    var requested = query.strategist;
    if (requested && options.indexOf(requested) !== -1) {
      select.value = requested;
    }
    select.disabled = options.length === 0;

    function visibleSeries() {
      var current = select.value;
      if (current === data.showAllLabel) {
        return data.series;
      }
      return data.series.filter(function (entry) {
        return entry.vanilla || entry.strategist === current;
      });
    }

    function render() {
      var margin = data.margin;
      var width = data.chartWidth;
      var height = data.chartHeight;
      var plotWidth = width - margin.left - margin.right;
      var plotHeight = height - margin.top - margin.bottom;

      function x(value) {
        return margin.left + value * plotWidth;
      }
      function y(value) {
        return margin.top + (1 - value) * plotHeight;
      }

      var svg = svgTag("svg", {
        viewBox: "0 0 " + width + " " + height,
        width: "100%",
        preserveAspectRatio: "xMidYMid meet"
      });
      [0, 0.25, 0.5, 0.75, 1].forEach(function (tick) {
        svg.appendChild(svgTag("line", {
          x1: x(tick), y1: y(0), x2: x(tick), y2: y(1),
          stroke: "#e0e5eb", "stroke-width": 1
        }));
        svg.appendChild(svgTag("line", {
          x1: x(0), y1: y(tick), x2: x(1), y2: y(tick),
          stroke: "#e0e5eb", "stroke-width": 1
        }));
        svg.appendChild(svgTag("text", {
          x: x(tick), y: height - margin.bottom + 18,
          "text-anchor": "middle", "font-size": 11, fill: "#687486"
        })).textContent = tick.toFixed(2);
        svg.appendChild(svgTag("text", {
          x: margin.left - 8, y: y(tick) + 4,
          "text-anchor": "end", "font-size": 11, fill: "#687486"
        })).textContent = tick.toFixed(2);
      });
      svg.appendChild(svgTag("text", {
        x: x(0.5), y: height - 6, "text-anchor": "middle",
        "font-size": 12, fill: "#445164"
      })).textContent = data.xLabel;
      var yLabel = svgTag("text", {
        x: 14, y: margin.top + plotHeight / 2, "text-anchor": "middle",
        "font-size": 12, fill: "#445164",
        transform: "rotate(-90 14 " + (margin.top + plotHeight / 2) + ")"
      });
      yLabel.textContent = data.yLabel;
      svg.appendChild(yLabel);

      var series = visibleSeries();
      series.forEach(function (entry) {
        if (!entry.points.length) {
          return;
        }
        var d = entry.points.map(function (point, index) {
          return (index === 0 ? "M" : "L") +
            x(point[0]).toFixed(2) + " " + y(point[1]).toFixed(2);
        }).join(" ");
        svg.appendChild(svgTag("path", {
          d: d, fill: "none", stroke: entry.color,
          "stroke-width": entry.width,
          "stroke-dasharray": entry.dash || "none",
          "stroke-linejoin": "round", "stroke-linecap": "round"
        }));
      });

      chartHost.textContent = "";
      chartHost.appendChild(svg);

      if (legendHost) {
        legendHost.textContent = "";
        series.forEach(function (entry) {
          var item = document.createElement("li");
          var label = entry.strategist + " · " + entry.condition;
          if (entry.vanilla) {
            item.className = "vanilla";
          }
          if (preselectedCondition &&
              entry.condition === preselectedCondition &&
              !entry.vanilla) {
            item.className = (item.className ? item.className + " " : "") +
              "preselected";
          }
          var swatch = document.createElement("span");
          swatch.className = "curve-swatch";
          swatch.style.borderTop = entry.width + "px " +
            (entry.dash ? "dashed" : "solid") + " " + entry.color;
          item.appendChild(swatch);
          item.appendChild(document.createTextNode(" " + label));
          legendHost.appendChild(item);
        });
      }
    }

    select.addEventListener("change", render);
    render();
  }

  document.addEventListener("DOMContentLoaded", function () {
    initChart();
  });
})();
"""
