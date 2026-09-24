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
0, yellow at 0.5, blue at 1, with a leading Avg column pooling each condition
row's runs, plus the dominant victory focus with a stable categorical color per
strategy), one detail page per ``(seed, player_id)`` pair, and static
JavaScript assets for the strategist checkboxes and an offline Plotly
probability-curve chart (cell tooltips come from the shared help script). Every value, color, link, and query string is computed
server-side, so unchanged inputs re-render byte-identically; the browser script
only changes trace visibility, color, and emphasis. The chart's Y axis fits the
visible curves, and same-family strategists that share a catalog color are spread
through the shared
:js:func:`civBench.distinguishColors` util in :mod:`bench.reports.assets`.

Rows are strategist-condition combinations and columns are final ``player_id``
values, each column heading pairing the position with its seat-bound
civilization, such as ``1: China`` (the orientation that keeps the dedicated
Vanilla condition a separate, visually isolated row). The renderer completes the
global row and column grid and leaves unobserved combinations blank.
"""

from __future__ import annotations

import html as _html
from typing import Optional
from urllib.parse import urlencode

import numpy as np

from bench.reports.assets import REPORT_COMMON_JS, REPORT_HELP_JS
from bench.reports.game_log import games_url, render_seat_games
from bench.plotting.interactive import figure_html, plotly_javascript
from .context import ReportBuildContext
from bench.reports.content import (
    render_footer_html, render_summary_html, resolve_footer, render_help_html,
    report_summary, metadata_text,
)
from .errors import ReportError
from .heatmap import (
    cell_text_color as _cell_text_color, interpolate as _interpolate, plural, position_background, tip_row,
)
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
CONTROLLED_SEED_TITLE = "Matched Maps"
CONTROLLED_SEED_PURPOSE = (
    "Compare strategists and experimental conditions with a VPAI baseline on "
    "the same maps and starting positions, so differences in performance can "
    "be assessed with the starting setup held constant."
)


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
        footer=resolve_footer(ctx.meta.get("footer")),
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

# The adjusted-strength heatmap scale is the shared RdYlBu scale
# (bench.reports.heatmap), fixed from 0 to 1 across every seed so the panels
# compare directly: 0 is red, 0.5 is yellow, 1 is blue.

# Plotly dash styles cycled by condition position, so one strategist's
# conditions stay distinguishable while sharing its color.
_DASH_PATTERNS = ("solid", "dash", "dot", "dashdot")


# ── small formatting / color helpers ──────────────────────────────────────────
def _esc(text) -> str:
    return _html.escape(str(text))


def _strategist_label(doc: ControlledSeedDocument, strategist: str) -> str:
    return "VPAI" if strategist == doc.vanilla_label else strategist


def _vpai_tooltip(doc: ControlledSeedDocument, strategist: str, condition: str) -> str:
    if strategist != doc.vanilla_label:
        return ""
    if condition == doc.vanilla_label:
        return "VPAI self-play: all players use VPAI."
    return (
        "VPAI in games with LLM players. Performance may differ from self-play "
        "because LLMs can counter VPAI's playstyle."
    )


def _label_html(label: str, tooltip: str) -> str:
    if tooltip:
        return f'<span tabindex="0" data-tip="{_esc(tooltip)}">{_esc(label)}</span>'
    return _esc(label)


def _strength_background(value: float) -> str:
    return position_background(value)


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
def _cell_tooltip(doc: ControlledSeedDocument, row: dict, player_id: int) -> str:
    """The shared cell tooltip: row and seat, then strength, focus, and runs.

    Both overview heatmaps carry the same tooltip so a cell means the same thing
    wherever it is hovered. Lines follow the report's tip format (``tip_row``).
    """
    runs = int(row["run_count"])
    strength = float(row["mean_adjusted_strength"])
    pct = float(row["dominant_focus_pct"])
    lines = [
        _row_title(doc, str(row["strategist"]), str(row["condition"])),
        f"P{player_id} · {row['civilization']}",
        tip_row("Strength", f"{strength:.3f}" if np.isfinite(strength) else "n/a"),
        tip_row("Focus", str(row["dominant_focus"]), f"{pct:.0f}%")
        if np.isfinite(pct) else tip_row("Focus", "n/a"),
        tip_row("Runs", str(runs)),
    ]
    return "\n".join(lines)


def _row_title(doc: ControlledSeedDocument, strategist: str, condition: str) -> str:
    if strategist == doc.vanilla_label and condition == doc.vanilla_label:
        return "VPAI"
    return f"{_strategist_label(doc, strategist)} | {condition}"


def _heat_cell(doc: ControlledSeedDocument, row: dict, seed: int, player_id: int, kind: str) -> str:
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
        f'color:{color}" data-tip="{_esc(_cell_tooltip(doc, row, player_id))}">'
        f'<a href="{_esc(href)}" style="color:inherit">{_esc(text)}</a></td>'
    )


def _empty_heat_cell(extra_class: str = "") -> str:
    classes = "heat-cell heat-cell-empty"
    if extra_class:
        classes += f" {extra_class}"
    return f'<td class="{classes}" aria-label="no data"></td>'


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
    if kind == "strength":
        parts.append(
            '<th scope="col" class="col-avg" title="Pooled mean adjusted strength '
            "across this seed's populated seats\">Avg</th>"
        )
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
                out.append(_heat_cell(doc, row, seed, player_id, kind))
        return out

    def avg_cell(strategist: str, condition: str) -> str:
        """The strength heatmap's leading cell: the pooled mean over the row.

        Each populated seat's cell is itself a mean over its runs, so weighting
        the seat means by their run counts yields the mean over every run of
        this (seed, strategist, condition) row, consistent with the page's
        equal-runs averaging.
        """
        rows = []
        for player_id in players:
            row = summary.get((seed, player_id, strategist, condition))
            if row is not None and np.isfinite(float(row["mean_adjusted_strength"])):
                rows.append(row)
        runs = sum(int(row["run_count"]) for row in rows)
        if not rows or runs <= 0:
            return _empty_heat_cell("col-avg")
        value = (
            sum(
                float(row["mean_adjusted_strength"]) * int(row["run_count"])
                for row in rows
            )
            / runs
        )
        background = _strength_background(value)
        tip = "\n".join([
            _row_title(doc, strategist, condition),
            "Seed average",
            tip_row("Strength", f"{value:.3f}"),
            tip_row("Runs", str(runs), f"over {plural(len(rows), 'seat')}"),
        ])
        return (
            f'<td class="heat-cell heat-cell-avg col-avg" '
            f'style="background-color:{background};'
            f'color:{_cell_text_color(background)}" '
            f'data-tip="{_esc(tip)}">{value:.2f}</td>'
        )

    if _has_vanilla_rows(doc):
        parts.append('<tbody class="vanilla-body">')
        parts.append(
            f'<tr class="vanilla-row"><th scope="row" class="row-label">'
            f'{_label_html("VPAI", _vpai_tooltip(doc, vanilla, vanilla))}</th>'
        )
        if kind == "strength":
            parts.append(avg_cell(vanilla, vanilla))
        parts.extend(cells_for(vanilla, vanilla))
        parts.append("</tr></tbody>")

    parts.append("<tbody>")
    for strategist, condition in combos:
        row_label = f"{_strategist_label(doc, strategist)} | {condition}"
        tooltip = _vpai_tooltip(doc, strategist, condition)
        parts.append(
            f'<tr><th scope="row" class="row-label">{_label_html(row_label, tooltip)}</th>'
        )
        if kind == "strength":
            parts.append(avg_cell(strategist, condition))
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
    parts.append('<script src="../assets/report-help.js" defer></script>')
    parts.append("</head><body>")
    parts.append('<a class="skip-link" href="#main-content">Skip to content</a>')
    if navigation:
        parts.extend(navigation)
    main_class = "content" if navigation else "controlled-content"
    parts.append(f'<main class="{main_class}" id="main-content">')
    return parts


def _chapter_details(doc: ControlledSeedDocument) -> str:
    provenance = {
        key: doc.metadata[key]
        for key in ("baseline_experiment", "estimator", "strength_table")
        if doc.metadata.get(key)
    }
    return "\n\n".join(filter(None, (doc.description, metadata_text(provenance))))


def _render_overview(
    doc: ControlledSeedDocument, navigation: Optional[list[str]]
) -> str:
    summary = _summary_lookup(doc)
    _, combos, seeds, players = _grid(doc)
    headings = _civ_headings(doc)
    parts = _page_start(doc, f"{CONTROLLED_SEED_TITLE} | {doc.title}", navigation)
    parts.append(f'<p class="eyebrow">{_esc(doc.title)}</p>')
    details = _chapter_details(doc) + "\n\n" + (
        'Each cell averages every unique run for its seed, final '
        "player position, strategist, and condition; seating rotations and repeated "
        "runs contribute equally. The strength heatmap's leading Avg column pools "
        "each row's runs."
    )
    help_html = render_help_html(details, "chapter-help", "About this comparison")
    parts.append(f"<h1>{_esc(CONTROLLED_SEED_TITLE)}{help_html}</h1>")
    parts.append(f"<p>{_esc(CONTROLLED_SEED_PURPOSE)}</p>")
    parts.append(f"<p>{render_summary_html(report_summary(doc.summary, doc.metadata))}</p>")
    parts.append('<p>Click a cell to explore that starting position.</p>')

    for seed in seeds:
        parts.append(f'<section aria-labelledby="seed-{seed}">')
        parts.append(f'<h2 id="seed-{seed}">Seed {seed}</h2>')
        if doc.game_log is not None:
            parts.append(
                f'<p><a href="{_esc(games_url("../", seed=seed))}">'
                f'Game Log for seed {seed} →</a></p>'
            )
        parts.append('<figure class="heat-figure">')
        tip = render_help_html("Mean adjusted strength: red 0, yellow 0.5, blue 1. The Avg column pools every run in the row.", f"strength-{seed}-help")
        parts.append(f"<figcaption>Mean adjusted strength{tip}</figcaption>")
        parts.append(_heatmap(doc, summary, seed, players, combos, "strength", headings))
        parts.append("</figure>")
        parts.append('<figure class="heat-figure">')
        tip = render_help_html("The victory focus with the largest mean share across runs.", f"focus-{seed}-help")
        parts.append(f"<figcaption>Dominant victory focus{tip}</figcaption>")
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

    parts.append(render_footer_html(doc.footer))
    parts.append("</main>")
    parts.append('<script src="../assets/report-common.js" defer></script>')
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
                "label": "VPAI" if is_vanilla else f"{_strategist_label(doc, key[0])} · {key[1]}",
                "tooltip": _vpai_tooltip(doc, *key),
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
        strategist, condition = str(row["strategist"]), str(row["condition"])
        is_vanilla = (strategist, condition) == (vanilla, vanilla)
        tooltip = _vpai_tooltip(doc, strategist, condition)
        row_open = '<tr class="vanilla-row">' if is_vanilla else "<tr>"
        parts.append(row_open)
        parts.append(f"<td>{_label_html(_strategist_label(doc, strategist), tooltip)}</td>")
        parts.append(f"<td>{'VPAI' if is_vanilla else _esc(row['condition'])}</td>")
        runs = int(row["run_count"])
        if doc.game_log is not None:
            filters = {"seed": seed, "player": player_id}
            filters["strategist"] = doc.vanilla_label if is_vanilla else strategist
            if not is_vanilla:
                filters["condition"] = condition
            runs_html = f'<a href="{_esc(games_url("../", **filters))}">{runs}</a>'
        else:
            runs_html = str(runs)
        parts.append(f"<td>{runs_html}</td>")
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


def _chart_figure(series: list[dict]):
    """Build the detail-page probability chart with stable trace metadata."""
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


def _render_detail(
    doc: ControlledSeedDocument,
    index_row: dict,
    prev_row: dict | None,
    next_row: dict | None,
    navigation: Optional[list[str]],
) -> str:
    seed = int(index_row["seed"])
    player_id = int(index_row["player_id"])
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
    help_html = render_help_html(_chapter_details(doc), "page-help", "About this benchmark")
    parts.append(f"<h1>Seed {seed} · Player {player_id}{help_html}</h1>")
    parts.append(
        f'<p>Civilization: {_esc(index_row["civilization"])} · '
        f"<strong>{int(index_row['run_count'])}</strong> runs</p>"
    )
    if doc.game_log is not None:
        parts.append(
            f'<p><a href="{_esc(games_url("../", seed=seed, player=player_id))}">'
            'Browse these games in the Game Log →</a></p>'
        )
    if int(index_row["n_civilizations"]) > 1:
        parts.append(
            '<p class="warning">Multiple civilizations occupy this seed-player pair '
            f"({_esc(index_row['civilization'])}); values are not directly comparable "
            "across runs.</p>"
        )
    if not bool(index_row["has_matched_vanilla"]):
        parts.append(
            '<p class="warning">The dedicated VPAI baseline is unavailable for this '
            "seed-player pair; the reference curve is omitted.</p>"
        )
    if not bool(index_row["has_probability"]):
        parts.append(
            '<p class="warning">No usable estimator prediction rows cover this '
            "seed-player pair; the victory-probability curve is unavailable.</p>"
        )

    parts.append('<section aria-labelledby="curves-heading">')
    tip = render_help_html(
        "Each curve is the mean of every run's interpolated victory probability "
        "on the fixed 0 to 1 progress grid. The VPAI reference curve is drawn "
        "thicker when present. The vertical axis fits the visible curves. "
        "Hover the chart to compare every checked condition's probability "
        "at one progress point.", "curves-help",
    )
    parts.append(f'<h2 id="curves-heading">Victory-probability curves{tip}</h2>')
    if strategists:
        parts.append(
            '<div class="chart-controls" id="strategist-filters" role="group" '
            'aria-label="Strategists">'
        )
        parts.append('<span class="controls-label">Strategists</span>')
        for name in strategists:
            tooltip = _vpai_tooltip(doc, name, "")
            tip_attr = f' data-tip="{_esc(tooltip)}"' if tooltip else ""
            parts.append(
                f'<label class="strategist-check"{tip_attr}>'
                f'<input type="checkbox" value="{_esc(name)}" checked> '
                f"{_esc(_strategist_label(doc, name))}</label>"
            )
        parts.append("</div>")
    parts.append(figure_html(
        _chart_figure(series),
        "matched-map",
        full_html=False,
        include_plotlyjs="../assets/plotly.min.js",
    ))
    parts.append("</section>")

    parts.append('<section aria-labelledby="comparison-heading">')
    tip = render_help_html("Per-condition means over every unique run occupying this final position. Seating rotations and repeated runs contribute equally.", "comparison-help")
    parts.append(f'<h2 id="comparison-heading">Comparison table{tip}</h2>')
    parts.append(_comparison_table(doc, seed, player_id))
    parts.append("</section>")

    if doc.game_log is not None:
        parts.append(render_seat_games(doc.game_log, seed, player_id, prefix="../"))

    parts.append(render_footer_html(doc.footer))
    parts.append("</main>")
    parts.append('<script src="../assets/report-common.js" defer></script>')
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
        "assets/report-help.js": REPORT_HELP_JS,
        "assets/plotly.min.js": plotly_javascript(),
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
    pages["assets/report-common.js"] = REPORT_COMMON_JS
    pages["assets/controlled-seed-report.js"] = CONTROLLED_SEED_JS
    return pages


# ── the static browser script ─────────────────────────────────────────────────
CONTROLLED_SEED_JS = """/* civ-bench controlled-seed report interactions.
   Plotly trace controls for seed-player detail pages. Heatmap cell tooltips
   come from the shared assets/report-help.js, and same-family strategist
   colors are spread through the shared civBench.distinguishColors util
   (assets/report-common.js). */
(function () {
  "use strict";

  // ── detail page: Plotly chart controls ──────────────────────────────────
  function readQuery() {
    var params = {};
    var search = new URLSearchParams(window.location.search);
    search.forEach(function (value, key) { params[key] = value; });
    return params;
  }

  function initChart() {
    var filters = document.getElementById("strategist-filters");
    var chart = document.getElementById("plotly-matched-map");
    if (!chart || !window.Plotly || !chart.data) {
      return;
    }
    var query = readQuery();
    var highlightedCondition = query.condition || null;

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
          updateChart();
          updateGames();
        });
      });
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
    function updateChart() {
      var checked = {};
      boxes.forEach(function (box) { checked[box.value] = box.checked; });
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

    function updateGames() {
      var gameRows = document.querySelectorAll(".seat-games .game-row");
      if (!gameRows.length) { return; }
      var checked = {};
      boxes.forEach(function (box) { checked[box.value] = box.checked; });
      gameRows.forEach(function (row) {
        row.hidden = boxes.length > 0 && checked[row.dataset.strategist] !== true;
      });
    }

    updateChart();
    updateGames();
  }

  document.addEventListener("DOMContentLoaded", function () {
    initChart();
  });
})();
"""
