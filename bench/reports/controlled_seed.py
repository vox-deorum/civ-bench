"""Build and render the report's controlled-seed chapter (stage 5).

:func:`controlled_seed_document` turns the resolved ``performance.controlled_seed_report``
section into a :class:`~bench.reports.model.ControlledSeedDocument` (the default
document carries it as its ``controlled_seed`` field and gives the section its
own chapter, parallel to the analysis families); :func:`render_controlled_seed_site`
renders that chapter: ``controlled-seed/index.html`` (one heatmap overview per
controlled seed) plus one detail page per ``(seed, player_id)`` pair in the same
folder, both framed by the report site's sidebar navigation.

The pages are self-contained and deterministic: one overview page with two heatmap
tables per controlled seed (mean adjusted strength on a fixed RdYlBu scale, red at
0, yellow at 0.5, blue at 1, plus the dominant victory focus with a stable
categorical color per strategy), one detail page per ``(seed, player_id)`` pair,
and one static vanilla-JavaScript asset for tooltips, the strategist checkboxes,
and the probability-curve chart with its per-progress comparison tooltip. Every
value, color, link, and query string is computed server-side, so unchanged inputs
re-render byte-identically; the browser script only reads embedded data.

Rows are strategist-condition combinations and columns are final ``player_id``
values, each column heading pairing the position with its seat-bound
civilization, such as ``1: China`` (the orientation that keeps the dedicated
Vanilla condition a separate, visually isolated row). The renderer completes the
global row and column grid and leaves unobserved combinations blank.
"""

from __future__ import annotations

import html as _html
import json
from typing import Optional
from urllib.parse import urlencode

import numpy as np

from .context import ReportBuildContext
from .errors import ReportError
from .model import ControlledSeedDocument

CONTROLLED_SEED_MODULE = "performance.controlled_seed_report"
CONTROLLED_SEED_TABLES = (
    "seed_player_summary",
    "seed_player_probability",
    "seed_player_index",
)

# The chapter's place in the report site: its own top-level directory beside the
# family pages, one sidebar heading parallel to the families, and one detail
# page per (seed, player_id) pair inside the directory.
CONTROLLED_SEED_DIR = "controlled-seed"
CONTROLLED_SEED_OVERVIEW = f"{CONTROLLED_SEED_DIR}/index.html"
CONTROLLED_SEED_TITLE = "Controlled seed"


# ── document building ─────────────────────────────────────────────────────────
def controlled_seed_document(ctx: ReportBuildContext) -> Optional[ControlledSeedDocument]:
    """Build the controlled-seed document when the report carries its analysis.

    Returns ``None`` when no ``performance.controlled_seed_report`` section is
    among the resolved sections, or when that section produced no artifacts (the
    section then renders as an ordinary empty section and the heatmap pages are
    skipped). Config validation allows at most one enabled instance of the
    analysis, so a resolved report contains at most one such section.
    """
    sections = [s for s in ctx.sections if s.module == CONTROLLED_SEED_MODULE]
    if not sections:
        return None
    if len(sections) > 1:
        raise ReportError(
            "the report carries more than one "
            f"'{CONTROLLED_SEED_MODULE}' section "
            f"({[s.id for s in sections]}); the heatmap pages render from a "
            "single section."
        )
    section = sections[0]
    if section.empty:
        return None
    tables = {
        name: ctx.load_table(section.id, name) for name in CONTROLLED_SEED_TABLES
    }
    return ControlledSeedDocument(
        title=ctx.meta["title"],
        run_name=ctx.meta["run_name"],
        seed=ctx.meta["seed"],
        config_path=ctx.meta["config_path"],
        output_root=ctx.meta["output_root"],
        description=ctx.meta.get("description", "") or "",
        section_id=section.id,
        summary=section.summary,
        metadata=dict(section.metadata or {}),
        summary_table=tables["seed_player_summary"],
        probability_table=tables["seed_player_probability"],
        index_table=tables["seed_player_index"],
        downloads=list(section.downloads),
    )


def chapter_seeds(doc: ControlledSeedDocument) -> list[int]:
    """The sorted controlled seeds the chapter covers (the sidebar sub-entries)."""
    return sorted({int(row["seed"]) for row in doc.index_table.to_dict("records")})

# Stable categorical colors for the four victory-focus strategies.
FOCUS_COLORS = {
    "Domination": "#b3452f",
    "Culture": "#c28e21",
    "Diplomatic": "#4f81bd",
    "Science": "#4e9b4e",
}

# The adjusted-strength heatmap scale is matplotlib's RdYlBu (its 11 ColorBrewer
# anchors), fixed from 0 to 1 across every seed so the panels compare directly:
# 0 is red, 0.5 is yellow, 1 is blue.
_STRENGTH_SCALE = (
    "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090",
    "#ffffbf", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695",
)
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
    n = len(_STRENGTH_SCALE) - 1
    position = t * n
    index = min(int(position), n - 1)
    return _interpolate(
        _STRENGTH_SCALE[index], _STRENGTH_SCALE[index + 1], position - index
    )


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


def _civ_headings(doc: ControlledSeedDocument) -> dict[tuple[int, int], str]:
    """Each column heading pairs the final player position with its civilization."""
    out: dict[tuple[int, int], str] = {}
    for row in doc.index_table.to_dict("records"):
        out[(int(row["seed"]), int(row["player_id"]))] = str(row["civilization"])
    return out


# ── heatmaps ──────────────────────────────────────────────────────────────────
def _cell_tooltip(row: dict) -> str:
    """The shared cell tooltip: civilization and runs, strength, and focus.

    Both overview heatmaps carry the same tooltip so a cell means the same thing
    wherever it is hovered.
    """
    runs = int(row["run_count"])
    strength = float(row["mean_adjusted_strength"])
    pct = float(row["dominant_focus_pct"])
    strength_line = (
        f"Adj strength: {strength:.4f}"
        if np.isfinite(strength)
        else "Adj strength: unavailable"
    )
    focus_line = (
        f"Victory focus: {row['dominant_focus']} ({pct:.1f}%)"
        if np.isfinite(pct)
        else "Victory focus: unavailable"
    )
    return (
        f"{row['civilization']} ({runs} run{'s' if runs != 1 else ''})\n"
        f"{strength_line}\n{focus_line}"
    )


def _heat_cell(row: dict, seed: int, player_id: int, kind: str) -> str:
    """One populated heatmap cell: colored, tooltip-equipped, and clickable."""
    strategist, condition = str(row["strategist"]), str(row["condition"])
    if kind == "strength":
        value = float(row["mean_adjusted_strength"])
        background = _strength_background(value)
        text = f"{value:.2f}" if np.isfinite(value) else ""
    else:
        label = str(row["dominant_focus"])
        pct = float(row["dominant_focus_pct"])
        background = _focus_background(label, pct)
        text = f"{label} {_fmt_pct_short(pct)}" if np.isfinite(pct) else ""
    color = _cell_text_color(background)
    href = _detail_link(seed, player_id, strategist, condition)
    return (
        f'<td class="heat-cell" style="background-color:{background};'
        f'color:{color}" data-tip="{_esc(_cell_tooltip(row))}">'
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
    headings: dict[tuple[int, int], str],
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
        civilization = headings.get((seed, player_id))
        column = f"{player_id}: {civilization}" if civilization else f"Player {player_id}"
        parts.append(f'<th scope="col">{_esc(column)}</th>')
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
def _page_start(
    doc: ControlledSeedDocument,
    page_title: str,
    navigation: Optional[list[str]],
) -> list[str]:
    """The shared chapter-page scaffold: head, skip link, optional sidebar.

    The page lives one level below the report root, so every asset and report
    link carries a ``../`` prefix. Without navigation the page renders
    standalone (centered layout), which keeps the renderer usable in isolation.
    """
    parts: list[str] = []
    parts.append('<!DOCTYPE html>')
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>{_esc(page_title)}</title>")
    parts.append('<link rel="stylesheet" href="../assets/report.css">')
    parts.append("</head><body>")
    parts.append('<a class="skip-link" href="#main-content">Skip to content</a>')
    if navigation:
        parts.extend(navigation)
    main_class = "content" if navigation else "controlled-content"
    parts.append(f'<main class="{main_class}" id="main-content">')
    return parts


def _render_overview(
    doc: ControlledSeedDocument, navigation: Optional[list[str]]
) -> str:
    summary = _summary_lookup(doc)
    _, combos, seeds, players = _grid(doc)
    headings = _civ_headings(doc)
    parts = _page_start(doc, f"{CONTROLLED_SEED_TITLE} | {doc.title}", navigation)
    parts.append(f'<p class="eyebrow">{_esc(doc.title)}</p>')
    parts.append(f"<h1>{_esc(CONTROLLED_SEED_TITLE)}</h1>")
    if doc.description:
        parts.append(f'<p class="caption">{_esc(doc.description)}</p>')
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
        parts.append(
            "<figcaption>Mean adjusted strength (red 0, yellow 0.5, blue 1)</figcaption>"
        )
        parts.append(_heatmap(doc, summary, seed, players, combos, "strength", headings))
        parts.append("</figure>")
        parts.append('<figure class="heat-figure">')
        parts.append("<figcaption>Dominant victory focus (largest mean share)</figcaption>")
        parts.append(_heatmap(doc, summary, seed, players, combos, "focus", headings))
        parts.append("</figure>")
        parts.append(_focus_legend())
        parts.append("</section>")

    if doc.downloads:
        parts.append('<section aria-labelledby="source-tables-heading">')
        parts.append('<h2 id="source-tables-heading">Source tables</h2>')
        parts.append("<ul>")
        for download in doc.downloads:
            parts.append(
                f'<li><a href="../{_esc(download.rel_path)}">{_esc(download.label)}</a></li>'
            )
        parts.append("</ul></section>")

    parts.append("</main>")
    parts.append('<script src="../assets/controlled-seed-report.js" defer></script>')
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
                "label": key[0] if is_vanilla else f"{key[0]} · {key[1]}",
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


def _strength_cell(row: dict, is_vanilla: bool) -> str:
    """The adjusted-strength cell, colored like the overview strength heatmap."""
    value = float(row["mean_adjusted_strength"])
    if not np.isfinite(value):
        return "<td></td>"
    background = _strength_background(value)
    classes = ' class="vanilla-value"' if is_vanilla else ""
    return (
        f'<td{classes} style="background-color:{background};'
        f'color:{_cell_text_color(background)}">{value:.2f}</td>'
    )


def _focus_cell(row: dict) -> str:
    """The dominant-focus cell, colored like the overview focus heatmap."""
    label = str(row["dominant_focus"])
    pct = float(row["dominant_focus_pct"])
    if not np.isfinite(pct):
        return "<td></td>"
    background = _focus_background(label, pct)
    return (
        f'<td style="background-color:{background};'
        f'color:{_cell_text_color(background)}">'
        f"{_esc(label)} {_fmt_pct_short(pct)}</td>"
    )


def _share_cell(row: dict, column: str, label: str) -> str:
    """One focus-share cell, colored with its strategy color at share intensity."""
    pct = float(row[column])
    if not np.isfinite(pct):
        return "<td></td>"
    background = _focus_background(label, pct)
    return (
        f'<td style="background-color:{background};'
        f'color:{_cell_text_color(background)}">{_fmt_pct(pct)}</td>'
    )


# Short header text; the full wording rides along as the title tooltip.
_COMPARISON_HEADERS = [
    ("Strategist", None),
    ("Condition", None),
    ("Runs", "Unique runs averaged"),
    ("Win prob", "Mean weighted victory probability"),
    ("Adj strength", "Mean adjusted strength"),
    ("Focus", "Dominant victory focus"),
    ("Dom %", "Domination focus %"),
    ("Cul %", "Culture focus %"),
    ("Dip %", "Diplomatic focus %"),
    ("Sci %", "Science focus %"),
]


def _comparison_table(doc: ControlledSeedDocument, seed: int, player_id: int) -> str:
    vanilla = doc.vanilla_label
    parts: list[str] = []
    parts.append('<div class="table-scroll" role="region" tabindex="0">')
    parts.append('<table class="comparison">')
    parts.append(
        f'<caption class="sr-only">Seed {seed}, player {player_id}: per-condition means '
        "over every unique run occupying this final position</caption>"
    )
    parts.append("<thead><tr>")
    for header, title in _COMPARISON_HEADERS:
        title_attr = f' title="{_esc(title)}"' if title else ""
        parts.append(f'<th scope="col"{title_attr}>{_esc(header)}</th>')
    parts.append("</tr></thead><tbody>")
    for row in _comparison_rows(doc, seed, player_id):
        is_vanilla = str(row["strategist"]) == vanilla
        row_open = '<tr class="vanilla-row">' if is_vanilla else "<tr>"
        parts.append(row_open)
        parts.append(f"<td>{_esc(row['strategist'])}</td>")
        parts.append(f"<td>{_esc(row['condition'])}</td>")
        parts.append(f"<td>{int(row['run_count'])}</td>")
        parts.append(f"<td>{_esc(_fmt_probability(row['mean_weighted_victory_probability']))}</td>")
        parts.append(_strength_cell(row, is_vanilla))
        parts.append(_focus_cell(row))
        for column, label in (
            ("domination_focus_pct", "Domination"),
            ("culture_focus_pct", "Culture"),
            ("diplomatic_focus_pct", "Diplomatic"),
            ("science_focus_pct", "Science"),
        ):
            parts.append(_share_cell(row, column, label))
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _chart_data(series: list[dict]) -> str:
    data = {
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
    navigation: Optional[list[str]],
) -> str:
    seed = int(index_row["seed"])
    player_id = int(index_row["player_id"])
    vanilla = doc.vanilla_label
    series = _page_series(doc, seed, player_id)
    strategists = []
    for entry in series:
        if not entry["vanilla"] and entry["strategist"] not in strategists:
            strategists.append(entry["strategist"])
    parts = _page_start(
        doc, f"Seed {seed} · Player {player_id} | {doc.title}", navigation
    )
    parts.append('<nav class="page-nav" aria-label="Seed player pages">')
    if prev_row is not None:
        prev_seed, prev_player = int(prev_row["seed"]), int(prev_row["player_id"])
        parts.append(f'<a href="{_page_filename(prev_seed, prev_player)}">← Player {prev_player}</a>')
    parts.append(f'<a href="index.html#seed-{seed}" class="return-link">Seed {seed} overview</a>')
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
            "seed-player pair; the reference curve is omitted.</p>"
        )
    if not bool(index_row["has_probability"]):
        parts.append(
            '<p class="warning">No usable estimator prediction rows cover this '
            "seed-player pair; the victory-probability curve is unavailable.</p>"
        )

    parts.append('<section aria-labelledby="curves-heading">')
    parts.append('<h2 id="curves-heading">Victory-probability curves</h2>')
    if strategists:
        parts.append(
            '<div class="chart-controls" id="strategist-filters" role="group" '
            'aria-label="Strategists">'
        )
        parts.append('<span class="controls-label">Strategists</span>')
        for name in strategists:
            parts.append(
                '<label class="strategist-check">'
                f'<input type="checkbox" value="{_esc(name)}" checked> '
                f"{_esc(name)}</label>"
            )
        parts.append("</div>")
    parts.append(
        '<div id="curve-chart" class="curve-chart" role="img" '
        'aria-label="Mean victory probability over normalized game progress"></div>'
    )
    parts.append('<ul id="curve-legend" class="curve-legend"></ul>')
    parts.append(
        '<p class="caption">Each curve is the mean of every run\'s interpolated '
        "victory probability on the fixed 0 to 1 progress grid; the "
        f"{_esc(vanilla)} reference curve is drawn thicker when present. Hover "
        "the chart to compare every checked condition's probability at one "
        "progress point.</p>"
    )
    parts.append("</section>")

    parts.append('<section aria-labelledby="comparison-heading">')
    parts.append('<h2 id="comparison-heading">Comparison table</h2>')
    parts.append(_comparison_table(doc, seed, player_id))
    parts.append("</section>")

    parts.append("</main>")
    parts.append(
        f'<script type="application/json" id="curve-data">{_chart_data(series)}</script>'
    )
    parts.append('<script src="../assets/controlled-seed-report.js" defer></script>')
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


# ── site assembly ─────────────────────────────────────────────────────────────
def render_controlled_seed_site(
    doc: ControlledSeedDocument, navigation: Optional[list[str]] = None
) -> dict[str, str]:
    """Render the chapter overview and every seed-player page (+ the JS asset).

    ``navigation`` is the report-site sidebar for pages inside the chapter
    directory (links relative to the report root must be ``../``-prefixed; the
    caller renders it accordingly). Without it the pages render standalone.
    """
    pages: dict[str, str] = {
        CONTROLLED_SEED_OVERVIEW: _render_overview(doc, navigation)
    }
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
            pages[f"{CONTROLLED_SEED_DIR}/{filename}"] = _render_detail(
                doc, row, prev_row, next_row, navigation
            )
    pages["assets/controlled-seed-report.js"] = CONTROLLED_SEED_JS
    return pages


# ── the static browser script ─────────────────────────────────────────────────
CONTROLLED_SEED_JS = """/* civ-bench controlled-seed report interactions.
   Deterministic vanilla JavaScript: no packages, no network. Tooltips on
   heatmap cells; strategist checkboxes, the SVG probability-curve chart, and
   the per-progress comparison tooltip on seed-player detail pages. */
(function () {
  "use strict";

  // ── shared tooltip ───────────────────────────────────────────────────────
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

  function showChartTooltip(clientX, clientY, text) {
    var tip = ensureTooltip();
    tip.textContent = text;
    tip.style.display = "block";
    var left = window.scrollX + clientX + 16;
    var top = window.scrollY + clientY - tip.offsetHeight - 12;
    if (top < window.scrollY + 4) {
      top = window.scrollY + clientY + 16;
    }
    var edge = window.scrollX + document.documentElement.clientWidth -
      tip.offsetWidth - 8;
    if (left > edge) {
      left = window.scrollX + clientX - tip.offsetWidth - 16;
    }
    tip.style.left = left + "px";
    tip.style.top = top + "px";
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

  // ── detail page: checkboxes + SVG curve chart + progress tooltip ─────────
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
    var filters = document.getElementById("strategist-filters");
    var chartHost = document.getElementById("curve-chart");
    var legendHost = document.getElementById("curve-legend");
    if (!dataNode || !chartHost) {
      return;
    }
    var data = JSON.parse(dataNode.textContent);
    if (!data.series.length) {
      chartHost.textContent = "No probability curves are available for this page.";
      return;
    }
    var query = readQuery();
    var preselectedCondition = query.condition || null;

    var boxes = [];
    if (filters) {
      boxes = Array.prototype.slice.call(
        filters.querySelectorAll('input[type="checkbox"]')
      );
      // An overview cell link focuses its strategist (the Vanilla row's cells
      // match no checkbox and keep everyone checked); direct entry also keeps
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
      boxes.forEach(function (box) {
        box.addEventListener("change", function () {
          hideTooltip();
          render();
        });
      });
    }

    function visibleSeries() {
      var checked = {};
      boxes.forEach(function (box) {
        checked[box.value] = box.checked;
      });
      return data.series.filter(function (entry) {
        return entry.vanilla || !boxes.length || checked[entry.strategist];
      });
    }

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

    var svg = null;
    var guide = null;        // hover guide: one vertical line + a dot per series
    var guideLine = null;
    var guideDots = [];
    var progressSteps = [];  // sorted union of the visible series' x values

    function collectProgress(series) {
      var seen = {};
      progressSteps = [];
      series.forEach(function (entry) {
        entry.points.forEach(function (point) {
          if (!seen[point[0]]) {
            seen[point[0]] = true;
            progressSteps.push(point[0]);
          }
        });
      });
      progressSteps.sort(function (a, b) { return a - b; });
    }

    function nearestProgress(value) {
      var best = progressSteps[0];
      var bestDistance = Math.abs(value - best);
      for (var i = 1; i < progressSteps.length; i++) {
        var distance = Math.abs(value - progressSteps[i]);
        if (distance < bestDistance) {
          best = progressSteps[i];
          bestDistance = distance;
        }
      }
      return best;
    }

    function buildGuide(series) {
      guide = svgTag("g", { "class": "chart-guide", display: "none" });
      guideLine = svgTag("line", {
        y1: y(0), y2: y(1),
        stroke: "#445164", "stroke-width": 1, "stroke-dasharray": "4 3"
      });
      guide.appendChild(guideLine);
      guideDots = series.map(function (entry) {
        var dot = svgTag("circle", {
          r: 4, fill: entry.color, stroke: "#ffffff",
          "stroke-width": 1.5, display: "none"
        });
        guide.appendChild(dot);
        return { dot: dot, entry: entry };
      });
      svg.appendChild(guide);
    }

    function render() {
      var series = visibleSeries();
      if (!series.length) {
        svg = null;
        guide = null;
        chartHost.textContent = "Check a strategist to show its curves.";
        if (legendHost) {
          legendHost.textContent = "";
        }
        return;
      }

      var chart = svgTag("svg", {
        viewBox: "0 0 " + width + " " + height,
        width: "100%",
        preserveAspectRatio: "xMidYMid meet"
      });
      [0, 0.25, 0.5, 0.75, 1].forEach(function (tick) {
        chart.appendChild(svgTag("line", {
          x1: x(tick), y1: y(0), x2: x(tick), y2: y(1),
          stroke: "#e0e5eb", "stroke-width": 1
        }));
        chart.appendChild(svgTag("line", {
          x1: x(0), y1: y(tick), x2: x(1), y2: y(tick),
          stroke: "#e0e5eb", "stroke-width": 1
        }));
        chart.appendChild(svgTag("text", {
          x: x(tick), y: height - margin.bottom + 18,
          "text-anchor": "middle", "font-size": 11, fill: "#687486"
        })).textContent = tick.toFixed(2);
        chart.appendChild(svgTag("text", {
          x: margin.left - 8, y: y(tick) + 4,
          "text-anchor": "end", "font-size": 11, fill: "#687486"
        })).textContent = tick.toFixed(2);
      });
      chart.appendChild(svgTag("text", {
        x: x(0.5), y: height - 6, "text-anchor": "middle",
        "font-size": 12, fill: "#445164"
      })).textContent = data.xLabel;
      var yLabel = svgTag("text", {
        x: 14, y: margin.top + plotHeight / 2, "text-anchor": "middle",
        "font-size": 12, fill: "#445164",
        transform: "rotate(-90 14 " + (margin.top + plotHeight / 2) + ")"
      });
      yLabel.textContent = data.yLabel;
      chart.appendChild(yLabel);

      series.forEach(function (entry) {
        if (!entry.points.length) {
          return;
        }
        var d = entry.points.map(function (point, index) {
          return (index === 0 ? "M" : "L") +
            x(point[0]).toFixed(2) + " " + y(point[1]).toFixed(2);
        }).join(" ");
        chart.appendChild(svgTag("path", {
          d: d, fill: "none", stroke: entry.color,
          "stroke-width": entry.width,
          "stroke-dasharray": entry.dash || "none",
          "stroke-linejoin": "round", "stroke-linecap": "round"
        }));
      });

      svg = chart;
      chartHost.textContent = "";
      chartHost.appendChild(svg);
      collectProgress(series);
      buildGuide(series);
      svg.addEventListener("mousemove", onMove);
      svg.addEventListener("mouseleave", onLeave);

      if (legendHost) {
        legendHost.textContent = "";
        series.forEach(function (entry) {
          var item = document.createElement("li");
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
          item.appendChild(document.createTextNode(" " + entry.label));
          legendHost.appendChild(item);
        });
      }
    }

    function onMove(event) {
      if (!svg || !guide || !progressSteps.length) {
        return;
      }
      var rect = svg.getBoundingClientRect();
      if (!rect.width) {
        return;
      }
      var scale = width / rect.width;
      var localX = (event.clientX - rect.left) * scale;
      var progress = (localX - margin.left) / plotWidth;
      progress = Math.max(0, Math.min(1, progress));
      var step = nearestProgress(progress);
      var px = x(step);
      guideLine.setAttribute("x1", px.toFixed(2));
      guideLine.setAttribute("x2", px.toFixed(2));
      var lines = ["Turn progress " + step.toFixed(2)];
      var readings = [];
      guideDots.forEach(function (item) {
        var value = null;
        item.entry.points.forEach(function (point) {
          if (point[0] === step) {
            value = point[1];
          }
        });
        if (value === null) {
          item.dot.setAttribute("display", "none");
          return;
        }
        item.dot.setAttribute("cx", px.toFixed(2));
        item.dot.setAttribute("cy", y(value).toFixed(2));
        item.dot.removeAttribute("display");
        readings.push({ label: item.entry.label, value: value });
      });
      guide.removeAttribute("display");
      readings.sort(function (a, b) { return b.value - a.value; });
      readings.forEach(function (reading) {
        lines.push(reading.label + ": " + reading.value.toFixed(3));
      });
      if (readings.length) {
        showChartTooltip(event.clientX, event.clientY, lines.join("\\n"));
      }
    }

    function onLeave() {
      if (guide) {
        guide.setAttribute("display", "none");
      }
      hideTooltip();
    }

    render();
  }

  document.addEventListener("DOMContentLoaded", function () {
    initChart();
  });
})();
"""
