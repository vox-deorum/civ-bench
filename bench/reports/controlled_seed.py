"""Build and render the report's controlled-seed chapter (stage 5).

:func:`controlled_seed_document` turns the resolved ``performance.controlled_seed_report``
section into a :class:`~bench.reports.model.ControlledSeedDocument` (the default
document carries it as its ``controlled_seed`` field and gives the section its
own chapter, parallel to the analysis families); :func:`render_controlled_seed_site`
renders that chapter: ``controlled-seed/index.html`` (one tabbed table group per
controlled seed) plus one detail page per ``(seed, player_id)`` pair in the same
folder, both framed by the report site's sidebar navigation.

The pages are self-contained and deterministic. Each controlled seed on the
overview gets one tab group: Strength (mean adjusted strength on a fixed RdYlBu
scale, red at 0, yellow at 0.5, blue at 1, with a leading Avg column pooling each
condition row's runs), Focus (the dominant victory focus with a stable
categorical color per strategy), then one tab per analysis the stage lists in
``uses.analyses`` (:class:`~bench.reports.model.MatchedMapTab`, such as the
Strategic tab from ``behavior.flavors``). Each ``(seed, player_id)`` detail page
repeats the same tabs for its seat under the probability-curve chart. Every
table renders through the shared :func:`~bench.reports.heatmap.render_heatmap_html`,
so tooltips, the pinned VPAI row, and click-to-sort match the rest of the report.
Static JavaScript assets drive the strategist checkboxes and an offline Plotly
chart (tooltips, tabs, and sorting come from the shared help script). Every
value, color, link, and query string is computed server-side, so unchanged
inputs re-render byte-identically; the browser script only changes trace
visibility, color, and emphasis. The chart's Y axis fits the visible curves, and
same-family strategists that share a catalog color are spread through the shared
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
import pandas as pd

from bench.reports.assets import REPORT_COMMON_JS, REPORT_HELP_JS
from bench.reports.game_log import games_url, render_seat_games
from bench.plotting.interactive import figure_html, plotly_javascript
from .context import ReportBuildContext
from bench.reports.content import (
    render_footer_html, render_summary_html, resolve_footer, render_help_html,
    render_view_group, report_summary, metadata_text,
)
from .errors import ReportError
from .heatmap import (
    HEATMAP_METADATA_KEY, interpolate as _interpolate, plural, render_heatmap_html, tip_row,
)
from .model import ControlledSeedDocument, MatchedMapTab

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
        tabs=_matched_map_tabs(ctx, section.metadata.get("tabs") or []),
    )


def _matched_map_tabs(ctx: ReportBuildContext, stage_ids: list) -> list[MatchedMapTab]:
    """The extra tabs the analysis lists (its ``uses.analyses``), in order.

    Each listed section declares its tab through ``metadata["matched_maps"]``. A
    section that is not in the report, is empty, or declares no tab (for
    example an uncontrolled run) is skipped with a warning.
    """
    by_id = {s.id: s for s in ctx.sections}
    tabs = []
    for stage_id in map(str, stage_ids):
        section = by_id.get(stage_id)
        declared = (section.metadata or {}).get("matched_maps") if section is not None else None
        if section is None or section.empty or not isinstance(declared, dict):
            reason = "is not among the report sections" if section is None else "offers no Matched Maps tab"
            ctx.warnings.append(f"Matched Maps: analysis '{stage_id}' {reason}; its tab is skipped.")
            continue
        specs = (section.metadata or {}).get(HEATMAP_METADATA_KEY) or {}
        seed_name, seat_name = str(declared["seed_table"]), str(declared["seat_table"])
        tabs.append(MatchedMapTab(
            name=stage_id,
            label=str(declared.get("label") or section.title),
            tip=str(declared.get("tip") or section.title),
            seed_table=ctx.load_table(stage_id, seed_name),
            seat_table=ctx.load_table(stage_id, seat_name),
            seed_spec=dict(specs.get(seed_name) or {}),
            seat_spec=dict(specs.get(seat_name) or {}),
        ))
    return tabs


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

# The adjusted-strength scale is the shared RdYlBu scale (bench.reports.heatmap),
# fixed from 0 to 1 across every seed so the panels compare directly: 0 is red,
# 0.5 is yellow, 1 is blue.

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


def _focus_background(label: str, pct: float) -> str:
    base = FOCUS_COLORS.get(label, "#7a7f88")
    t = 0.0 if not np.isfinite(pct) else min(1.0, max(0.0, pct / 100.0))
    intensity = 0.22 + 0.78 * t
    return _interpolate("#ffffff", base, intensity)


def _page_filename(seed: int, player_id: int) -> str:
    return f"seed-{seed}-player-{player_id}.html"


def _detail_link(seed: int, player_id: int, strategist: str, condition: str,
                 view: str = "") -> str:
    """The seat page with this row preselected (and a non-default tab chosen)."""
    params = {"strategist": strategist, "condition": condition}
    if view:
        params["view"] = view
    query = urlencode(params)
    return f"{_page_filename(seed, player_id)}?{query}"


def _fmt_probability(value) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.4f}"


def _fmt_pct(value) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.1f}"


def _fmt_pct_short(value) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.0f}%"


# ── document views ────────────────────────────────────────────────────────────
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


# ── heatmap tables ────────────────────────────────────────────────────────────
# Every table in the chapter renders through the shared heatmap renderer
# (bench.reports.heatmap) from a long frame with one record per cell. The
# frames carry their own text, color, link, and tooltip columns, so the cells
# keep the chapter's formats while sharing the markup, pinned VPAI body, and
# sorting of every other report heatmap.
STRENGTH_VIEW = "strength"
FOCUS_VIEW = "focus"
_ROW_HEADING = "Strategist | Condition"
_STRENGTH_LEGEND = [[0.0, "0"], [0.25, "0.25"], [0.5, "0.5"], [0.75, "0.75"], [1.0, "1"]]
_CELL_COLUMNS = ["row_label", "column", "value", "color_position", "text", "background", "link", "tip"]
_CELL_SPEC = {
    "row": "row_label", "column": "column", "value": "value", "row_heading": _ROW_HEADING,
    "text_column": "text", "background_column": "background",
    "link_column": "link", "tip_column": "tip",
}


def _cell_tooltip(doc: ControlledSeedDocument, row: dict, player_id: int) -> str:
    """The overview cell tooltip: row and seat, then strength, focus, and runs.

    Both overview tables carry the same tooltip so a cell means the same thing
    on either tab. Lines follow the report's tip format (``tip_row``).
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


def _rows_layout(doc: ControlledSeedDocument, rows: list[tuple[str, str]]) -> dict:
    """Row order, the pinned VPAI row, and row-label tooltips for ``rows``."""
    vanilla = (doc.vanilla_label, doc.vanilla_label)
    order, tips = [], {}
    for strategist, condition in rows:
        label = _row_title(doc, strategist, condition)
        order.append(label)
        tip = _vpai_tooltip(doc, strategist, condition)
        if tip:
            tips[label] = tip
    return {
        "row_order": order,
        "reference_rows": ["VPAI"] if vanilla in rows else [],
        "row_tips": tips,
    }


def _cell(label: str, column: str, value, **extra) -> dict:
    record = {"row_label": label, "column": column, "value": value,
              "color_position": float("nan"), "text": "", "background": "", "link": "", "tip": ""}
    record.update(extra)
    return record


def _frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=_CELL_COLUMNS)


def _seat_columns(doc: ControlledSeedDocument, seed: int, players: list[int]) -> tuple[list[str], dict]:
    headings = _civ_headings(doc)
    labels = {}
    for player_id in players:
        civilization = headings.get((seed, player_id))
        labels[str(player_id)] = f"{player_id}: {civilization}" if civilization else f"Player {player_id}"
    return [str(p) for p in players], labels


def _overview_rows(doc: ControlledSeedDocument) -> list[tuple[str, str]]:
    """The global row grid: the VPAI self-play row first, then every combination."""
    _, combos, _, _ = _grid(doc)
    vanilla = doc.vanilla_label
    return ([(vanilla, vanilla)] if _has_vanilla_rows(doc) else []) + combos


def _seed_records(doc: ControlledSeedDocument, seed: int) -> list[dict]:
    return [row for row in doc.summary_table.to_dict("records") if int(row["seed"]) == seed]


def _strength_table(doc: ControlledSeedDocument, seed: int, players: list[int]) -> tuple[pd.DataFrame, dict]:
    """Mean adjusted strength by row and seat, led by the pooled Avg column."""
    records = []
    pooled: dict[tuple[str, str], list[dict]] = {}
    for row in _seed_records(doc, seed):
        strategist, condition = str(row["strategist"]), str(row["condition"])
        player_id = int(row["player_id"])
        value = float(row["mean_adjusted_strength"])
        records.append(_cell(
            _row_title(doc, strategist, condition), str(player_id), value,
            color_position=value, tip=_cell_tooltip(doc, row, player_id),
            link=_detail_link(seed, player_id, strategist, condition),
        ))
        if np.isfinite(value):
            pooled.setdefault((strategist, condition), []).append(row)
    for (strategist, condition), rows in pooled.items():
        # Each seat's value is a mean over its runs, so weighting the seat means
        # by their run counts yields the mean over every run of the row,
        # consistent with the page's equal-runs averaging.
        runs = sum(int(row["run_count"]) for row in rows)
        if runs <= 0:
            continue
        value = sum(float(row["mean_adjusted_strength"]) * int(row["run_count"]) for row in rows) / runs
        title = _row_title(doc, strategist, condition)
        tip = "\n".join([
            title, "Seed average", tip_row("Strength", f"{value:.3f}"),
            tip_row("Runs", str(runs), f"over {plural(len(rows), 'seat')}"),
        ])
        records.append(_cell(title, "avg", value, color_position=value, tip=tip))
    columns, labels = _seat_columns(doc, seed, players)
    spec = {
        **_CELL_SPEC,
        **_rows_layout(doc, _overview_rows(doc)),
        "title": "Adjusted strength",
        "help": "Mean adjusted strength: red 0, yellow 0.5, blue 1. The Avg column pools every run in the row.",
        "complete_grid": True,
        "column_order": ["avg", *columns],
        "column_labels": {"avg": "Avg", **labels},
        "column_tips": {"avg": "Pooled mean adjusted strength across this seed's populated seats"},
        "divider_columns": ["avg"],
        "decimals": 2,
        "legend": _STRENGTH_LEGEND,
    }
    return _frame(records), spec


def _focus_text(label: str, pct: float) -> str:
    return f"{label} {_fmt_pct_short(pct)}" if np.isfinite(pct) else ""


def _focus_table(doc: ControlledSeedDocument, seed: int, players: list[int]) -> tuple[pd.DataFrame, dict]:
    """The dominant victory focus by row and seat, in its strategy color."""
    records = []
    for row in _seed_records(doc, seed):
        strategist, condition = str(row["strategist"]), str(row["condition"])
        player_id = int(row["player_id"])
        label, pct = str(row["dominant_focus"]), float(row["dominant_focus_pct"])
        records.append(_cell(
            _row_title(doc, strategist, condition), str(player_id), pct,
            text=_focus_text(label, pct), background=_focus_background(label, pct),
            tip=_cell_tooltip(doc, row, player_id),
            link=_detail_link(seed, player_id, strategist, condition, FOCUS_VIEW),
        ))
    columns, labels = _seat_columns(doc, seed, players)
    spec = {
        **_CELL_SPEC,
        **_rows_layout(doc, _overview_rows(doc)),
        "title": "Dominant victory focus",
        "help": "The victory focus with the largest mean share across runs. Sorting uses that share.",
        "complete_grid": True,
        "column_order": columns,
        "column_labels": labels,
        "legend": _focus_legend_entries(),
    }
    return _frame(records), spec


def _focus_legend_entries() -> list[list[str]]:
    return [[color, label] for label, color in FOCUS_COLORS.items()]


def _render_table(frame: pd.DataFrame, spec: dict, help_id: str) -> str:
    help_html = render_help_html(str(spec.get("help") or ""), help_id)
    return render_heatmap_html(frame, spec, help_html)


def _tab_slice(frame: pd.DataFrame, **keys) -> pd.DataFrame:
    """The rows of an extra tab's table for one seed (and seat)."""
    if frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    for column, value in keys.items():
        if column not in frame.columns:
            return frame.iloc[0:0]
        mask &= pd.to_numeric(frame[column], errors="coerce") == value
    return frame[mask]


def _extra_views(doc: ControlledSeedDocument, anchor: str, seat: bool, **keys) -> list[tuple]:
    """One view per extra tab (``doc.tabs``), sliced to this seed or seat."""
    views = []
    where = "seat" if seat else "seed"
    for tab in doc.tabs:
        frame = _tab_slice(tab.seat_table if seat else tab.seed_table, **keys)
        spec = tab.seat_spec if seat else tab.seed_spec
        if frame.empty or not spec:
            body = f'<p class="empty">No {_esc(tab.label.lower())} data for this {where}.</p>'
        else:
            body = _render_table(frame, spec, f"{anchor}-{tab.name}-help")
        views.append((tab.name, tab.label, tab.tip, body))
    return views


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
    _, _, seeds, players = _grid(doc)
    parts = _page_start(doc, f"{CONTROLLED_SEED_TITLE} | {doc.title}", navigation)
    parts.append(f'<p class="eyebrow">{_esc(doc.title)}</p>')
    details = _chapter_details(doc) + "\n\n" + (
        'Each cell averages every unique run for its seed, final '
        "player position, strategist, and condition; seating rotations and repeated "
        "runs contribute equally. The strength table's leading Avg column pools "
        "each row's runs. Click a numeric column heading to sort."
    )
    help_html = render_help_html(details, "chapter-help", "About this comparison")
    parts.append(f"<h1>{_esc(CONTROLLED_SEED_TITLE)}{help_html}</h1>")
    parts.append(f"<p>{_esc(CONTROLLED_SEED_PURPOSE)}</p>")
    parts.append(f"<p>{render_summary_html(report_summary(doc.summary, doc.metadata))}</p>")
    parts.append('<p>Click a cell to explore that starting position.</p>')

    for seed in seeds:
        anchor = f"seed-{seed}"
        parts.append(f'<section aria-labelledby="{anchor}">')
        parts.append(f'<h2 id="{anchor}">Seed {seed}</h2>')
        if doc.game_log is not None:
            parts.append(
                f'<p><a href="{_esc(games_url("../", seed=seed))}">'
                f'Game Log for seed {seed} →</a></p>'
            )
        views = [
            (STRENGTH_VIEW, "Strength", "Mean adjusted strength",
             _render_table(*_strength_table(doc, seed, players), f"strength-{seed}-help")),
            (FOCUS_VIEW, "Focus", "Dominant victory focus",
             _render_table(*_focus_table(doc, seed, players), f"focus-{seed}-help")),
            *_extra_views(doc, anchor, seat=False, seed=seed),
        ]
        parts.append(render_view_group(anchor, views))
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


def _seat_tip(doc: ControlledSeedDocument, row: dict, player_id: int, label: str, value: str) -> str:
    lines = [
        _row_title(doc, str(row["strategist"]), str(row["condition"])),
        f"P{player_id} · {row['civilization']}",
        tip_row(label, value),
    ]
    if label != "Runs":
        lines.append(tip_row("Runs", str(int(row["run_count"]))))
    return "\n".join(lines)


def _runs_link(doc: ControlledSeedDocument, row: dict, seed: int, player_id: int) -> str:
    if doc.game_log is None:
        return ""
    strategist, condition = str(row["strategist"]), str(row["condition"])
    is_vanilla = (strategist, condition) == (doc.vanilla_label, doc.vanilla_label)
    filters = {"seed": seed, "player": player_id,
               "strategist": doc.vanilla_label if is_vanilla else strategist}
    if not is_vanilla:
        filters["condition"] = condition
    return games_url("../", **filters)


# Short header text; the full wording rides along as the header tooltip.
_SEAT_STRENGTH_COLUMNS = {
    "runs": ("Runs", "Unique runs averaged"),
    "win_prob": ("Win prob", "Mean weighted victory probability"),
    "strength": ("Adj strength", "Mean adjusted strength"),
}
_SEAT_FOCUS_COLUMNS = {
    "focus": ("Focus", "Dominant victory focus"),
    **{label.lower(): (f"{label[:3]} %", f"{label} focus %") for label in FOCUS_COLORS},
}


def _seat_spec(doc: ControlledSeedDocument, rows: list[dict], columns: dict, title: str, help_text: str) -> dict:
    return {
        **_CELL_SPEC,
        **_rows_layout(doc, [(str(r["strategist"]), str(r["condition"])) for r in rows]),
        "title": title,
        "help": help_text,
        "column_order": list(columns),
        "column_labels": {key: short for key, (short, _full) in columns.items()},
        "column_names": {key: full for key, (_short, full) in columns.items()},
        "column_tips": {key: full for key, (_short, full) in columns.items()},
    }


def _seat_strength_table(doc: ControlledSeedDocument, seed: int, player_id: int) -> tuple[pd.DataFrame, dict]:
    """Runs, win probability, and adjusted strength per row on one seat."""
    rows = _comparison_rows(doc, seed, player_id)
    records = []
    for row in rows:
        label = _row_title(doc, str(row["strategist"]), str(row["condition"]))
        runs = int(row["run_count"])
        probability = float(row["mean_weighted_victory_probability"])
        strength = float(row["mean_adjusted_strength"])
        strength_text = f"{strength:.2f}" if np.isfinite(strength) else ""
        records.append(_cell(label, "runs", runs, text=str(runs),
                             link=_runs_link(doc, row, seed, player_id),
                             tip=_seat_tip(doc, row, player_id, "Runs", str(runs))))
        records.append(_cell(label, "win_prob", probability, text=_fmt_probability(probability),
                             tip=_seat_tip(doc, row, player_id, "Win prob", _fmt_probability(probability))))
        records.append(_cell(label, "strength", strength, color_position=strength, text=strength_text,
                             tip=_seat_tip(doc, row, player_id, "Strength",
                                           f"{strength:.3f}" if np.isfinite(strength) else "n/a")))
    spec = _seat_spec(
        doc, rows, _SEAT_STRENGTH_COLUMNS, "Adjusted strength",
        "Adjusted strength runs red 0, yellow 0.5, blue 1.",
    )
    spec["legend"] = _STRENGTH_LEGEND
    return _frame(records), spec


def _seat_focus_table(doc: ControlledSeedDocument, seed: int, player_id: int) -> tuple[pd.DataFrame, dict]:
    """The dominant focus and each focus share per row on one seat."""
    rows = _comparison_rows(doc, seed, player_id)
    records = []
    for row in rows:
        label = _row_title(doc, str(row["strategist"]), str(row["condition"]))
        focus, pct = str(row["dominant_focus"]), float(row["dominant_focus_pct"])
        records.append(_cell(label, "focus", pct, text=_focus_text(focus, pct),
                             background=_focus_background(focus, pct),
                             tip=_seat_tip(doc, row, player_id, "Focus", _focus_text(focus, pct))))
        for name in FOCUS_COLORS:
            share = float(row[f"{name.lower()}_focus_pct"])
            records.append(_cell(label, name.lower(), share, text=_fmt_pct(share),
                                 background=_focus_background(name, share) if np.isfinite(share) else "",
                                 tip=_seat_tip(doc, row, player_id, f"{name} %", _fmt_pct(share))))
    spec = _seat_spec(
        doc, rows, _SEAT_FOCUS_COLUMNS, "Victory focus",
        "Mean share of each victory focus across runs, and the largest one. Each share is "
        "colored with its strategy color at share intensity.",
    )
    spec["legend"] = _focus_legend_entries()
    return _frame(records), spec


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
    tip = render_help_html(
        "Per-condition means over every unique run occupying this final position. Seating "
        "rotations and repeated runs contribute equally. Click a numeric column heading to sort.",
        "comparison-help",
    )
    parts.append(f'<h2 id="comparison-heading">Comparison{tip}</h2>')
    views = [
        (STRENGTH_VIEW, "Strength", "Runs, win probability, and adjusted strength",
         _render_table(*_seat_strength_table(doc, seed, player_id), "seat-strength-help")),
        (FOCUS_VIEW, "Focus", "Victory focus shares",
         _render_table(*_seat_focus_table(doc, seed, player_id), "seat-focus-help")),
        *_extra_views(doc, "seat", seat=True, seed=seed, player_id=player_id),
    ]
    parts.append(render_view_group("seat", views))
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
