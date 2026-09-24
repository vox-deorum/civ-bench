"""Colored HTML heatmap tables shared across the report (stage 5).

The Matched Maps chapter and the behavior pages draw heatmaps with one set of
helpers: the fixed RdYlBu color scale, the cell text color, and
:func:`render_heatmap_html`, which turns a long table into the Matched Maps
markup (``table.heatmap``, ``td.heat-cell`` with a ``data-tip`` tooltip, and a
pinned reference-row body).

An analysis asks for a heatmap through ``metadata["heatmaps"]``, a mapping of
table name to a layout spec. The table itself carries the numbers: one row per
``(row, column)`` cell with the value column, a ``color_position`` from 0 to 1,
and optionally ``ci_lower`` / ``ci_upper`` / ``median`` / ``n_players`` /
``n_games`` for the tooltip. The spec only describes layout and wording:

``row``, ``column``
    Required. The columns holding the row and column keys.
``value``
    The value column (default ``mean``).
``title``, ``help``, ``row_heading``
    Caption text, its help tooltip, and the row-label column heading.
``row_order``, ``column_order``
    Display order; keys not listed follow in first-appearance order.
``reference_rows``, ``row_tips``
    Rows pinned in their own body at the top, and row-label tooltips.
``column_group``
    Optional column naming each column's group, shown as a spanning header.
``column_names``, ``column_labels``, ``column_tips``
    The full column name (tooltips and the Markdown key), the short header
    text, and the header tooltip.
``decimals``, ``signed``, ``value_label``, ``ci_level``
    Number format and the tooltip's value line.
``range_columns``, ``range_label``
    Optional ``[low, high]`` columns shown as one more tooltip line.
``legend``
    ``[[position, label], ...]`` swatches under the table.

Every value, color, and tooltip is computed here, so unchanged inputs render
byte-identically.
"""

from __future__ import annotations

import html as _html
from typing import Optional

import numpy as np
import pandas as pd

# matplotlib's RdYlBu (its 11 ColorBrewer anchors): position 0 is red, 0.5 is
# pale yellow, 1 is blue.
RDYLBU_SCALE = (
    "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090",
    "#ffffbf", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695",
)
TEXT_DARK = "#18202a"
TEXT_LIGHT = "#ffffff"

HEATMAP_METADATA_KEY = "heatmaps"
COLOR_COLUMN = "color_position"


# ── colors ────────────────────────────────────────────────────────────────────
def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def interpolate(start: str, end: str, t: float) -> str:
    a, b = hex_to_rgb(start), hex_to_rgb(end)
    return rgb_to_hex(tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3)))


def luminance(rgb: tuple[int, int, int]) -> float:
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255.0


def cell_text_color(background: str) -> str:
    return TEXT_LIGHT if luminance(hex_to_rgb(background)) < 0.55 else TEXT_DARK


def position_background(value: float) -> str:
    """The RdYlBu color at ``value`` in [0, 1] (clipped; NaN reads as 0)."""
    t = 0.0 if not np.isfinite(value) else min(1.0, max(0.0, value))
    n = len(RDYLBU_SCALE) - 1
    position = t * n
    index = min(int(position), n - 1)
    return interpolate(RDYLBU_SCALE[index], RDYLBU_SCALE[index + 1], position - index)


# ── spec lookup ───────────────────────────────────────────────────────────────
def heatmap_spec(metadata: dict, name: str) -> Optional[dict]:
    """The layout spec an analysis declared for table ``name``, if any."""
    specs = (metadata or {}).get(HEATMAP_METADATA_KEY)
    if not isinstance(specs, dict):
        return None
    spec = specs.get(name)
    return spec if isinstance(spec, dict) else None


def spec_fits(spec: Optional[dict], frame: pd.DataFrame) -> bool:
    """Whether ``frame`` has every column the spec needs; otherwise render plainly."""
    if not isinstance(spec, dict):
        return False
    row, column = spec.get("row"), spec.get("column")
    if not isinstance(row, str) or not isinstance(column, str):
        return False
    needed = {row, column, str(spec.get("value", "mean")), COLOR_COLUMN}
    group = spec.get("column_group")
    if group is not None:
        if not isinstance(group, str):
            return False
        needed.add(group)
    return needed <= set(frame.columns)


# ── layout ────────────────────────────────────────────────────────────────────
def _esc(text) -> str:
    return _html.escape(str(text))


def _ordered(observed: list[str], preferred) -> list[str]:
    preferred = [str(v) for v in (preferred or []) if str(v) in set(observed)]
    return list(dict.fromkeys(preferred + observed))


def _layout(frame: pd.DataFrame, spec: dict):
    row, column = spec["row"], spec["column"]
    rows = _ordered(list(dict.fromkeys(frame[row].astype(str))), spec.get("row_order"))
    columns = _ordered(list(dict.fromkeys(frame[column].astype(str))), spec.get("column_order"))
    reference = [r for r in rows if r in {str(v) for v in spec.get("reference_rows") or []}]
    body = [r for r in rows if r not in set(reference)]
    cells = {
        (str(rec[row]), str(rec[column])): rec
        for rec in frame.to_dict("records")
    }
    return reference, body, columns, cells


def _number(value, decimals: int, signed: bool) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    text = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"
    # "-0" and "+0" read as noise; a zero is a zero.
    if float(text) == 0:
        text = f"{0:.{decimals}f}"
    return text


def _cell_value(rec: dict, spec: dict) -> str:
    return _number(rec.get(spec.get("value", "mean")), int(spec.get("decimals", 1)),
                   bool(spec.get("signed", False)))


def _column_name(spec: dict, column: str) -> str:
    return str((spec.get("column_names") or {}).get(column, column))


def _column_label(spec: dict, column: str) -> str:
    return str((spec.get("column_labels") or {}).get(column, _column_name(spec, column)))


def _cell_tooltip(rec: dict, spec: dict, row: str, column: str) -> str:
    decimals = int(spec.get("decimals", 1)) + 1
    signed = bool(spec.get("signed", False))
    value = _number(rec.get(spec.get("value", "mean")), decimals, signed)
    lines = [f"{row} · {_column_name(spec, column)}"]
    line = value
    if spec.get("value_label"):
        line = f"{value} {spec['value_label']}"
    lo = _number(rec.get("ci_lower"), decimals, signed)
    hi = _number(rec.get("ci_upper"), decimals, signed)
    if lo and hi:
        level = spec.get("ci_level")
        prefix = f"{float(level) * 100:g}% CI " if isinstance(level, (int, float)) else "CI "
        line += f" ({prefix}{lo} to {hi})"
    lines.append(line)
    median = _number(rec.get("median"), decimals, signed)
    if median:
        lines.append(f"Median {median}")
    bounds = spec.get("range_columns")
    if isinstance(bounds, list) and len(bounds) == 2:
        low = _number(rec.get(bounds[0]), 0, False)
        high = _number(rec.get(bounds[1]), 0, False)
        if low and high:
            lines.append(f"{spec.get('range_label', 'Range')}: {low} to {high}")
    n_players, n_games = rec.get("n_players"), rec.get("n_games")
    if n_players is not None and n_games is not None and np.isfinite(float(n_players)):
        players, games = int(n_players), int(float(n_games))
        lines.append(
            f"{players} player{'s' if players != 1 else ''} in "
            f"{games} game{'s' if games != 1 else ''}"
        )
    return "\n".join(lines)


def _label_html(label: str, tip: str) -> str:
    if tip:
        return f'<span tabindex="0" data-tip="{_esc(tip)}">{_esc(label)}</span>'
    return _esc(label)


def _cell_html(rec: Optional[dict], spec: dict, row: str, column: str) -> str:
    if rec is None:
        return '<td class="heat-cell heat-cell-empty" aria-label="no data"></td>'
    text = _cell_value(rec, spec)
    position = rec.get(COLOR_COLUMN)
    try:
        position = float(position)
    except (TypeError, ValueError):
        position = float("nan")
    tip = _esc(_cell_tooltip(rec, spec, row, column))
    if not text or not np.isfinite(position):
        return f'<td class="heat-cell heat-cell-value" data-tip="{tip}">{_esc(text)}</td>'
    background = position_background(position)
    return (
        f'<td class="heat-cell heat-cell-value" style="background-color:{background};'
        f'color:{cell_text_color(background)}" data-tip="{tip}">{_esc(text)}</td>'
    )


def _header_html(frame: pd.DataFrame, spec: dict, columns: list[str]) -> list[str]:
    heading = _esc(spec.get("row_heading", ""))
    tips = spec.get("column_tips") or {}
    group_col = spec.get("column_group")
    parts = ["<thead>"]
    headers = "".join(
        f'<th scope="col">{_label_html(_column_label(spec, c), str(tips.get(c, "")))}</th>'
        for c in columns
    )
    if group_col:
        group_of = dict(zip(frame[spec["column"]].astype(str), frame[group_col].astype(str)))
        runs: list[list] = []
        for column in columns:
            group = group_of.get(column, "")
            if runs and runs[-1][0] == group:
                runs[-1][1] += 1
            else:
                runs.append([group, 1])
        parts.append(f'<tr><th scope="col" rowspan="2" class="row-label">{heading}</th>')
        parts.extend(
            f'<th scope="colgroup" colspan="{span}" class="heat-group">{_esc(group)}</th>'
            for group, span in runs
        )
        parts.append(f"</tr><tr>{headers}</tr>")
    else:
        parts.append(f'<tr><th scope="col" class="row-label">{heading}</th>{headers}</tr>')
    parts.append("</thead>")
    return parts


def _rows_html(rows: list[str], columns: list[str], cells: dict, spec: dict, reference: bool) -> list[str]:
    tips = spec.get("row_tips") or {}
    parts = ['<tbody class="vanilla-body">' if reference else "<tbody>"]
    row_open = '<tr class="vanilla-row">' if reference else "<tr>"
    for row in rows:
        parts.append(
            f'{row_open}<th scope="row" class="row-label">'
            f'{_label_html(row, str(tips.get(row, "")))}</th>'
        )
        parts.extend(_cell_html(cells.get((row, c)), spec, row, c) for c in columns)
        parts.append("</tr>")
    parts.append("</tbody>")
    return parts


def _legend_html(spec: dict) -> str:
    items = []
    for entry in spec.get("legend") or []:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            position = float(entry[0])
        except (TypeError, ValueError):
            continue
        items.append(
            f'<li><span class="swatch" style="background-color:{position_background(position)}"></span>'
            f"{_esc(entry[1])}</li>"
        )
    return f'<ul class="focus-legend">{"".join(items)}</ul>' if items else ""


def render_heatmap_html(frame: pd.DataFrame, spec: dict, help_html: str = "", csv_link: str = "") -> str:
    """One heatmap figure in the Matched Maps markup."""
    reference, body, columns, cells = _layout(frame, spec)
    title = str(spec.get("title", ""))
    parts = ['<figure class="heat-figure">']
    if title or help_html:
        parts.append(f"<figcaption>{_esc(title)}{help_html}</figcaption>")
    parts.append('<div class="table-scroll heat-scroll" role="region" tabindex="0">')
    parts.append('<table class="heatmap">')
    if title:
        parts.append(f'<caption class="sr-only">{_esc(title)}</caption>')
    parts.extend(_header_html(frame, spec, columns))
    if reference:
        parts.extend(_rows_html(reference, columns, cells, spec, reference=True))
    parts.extend(_rows_html(body, columns, cells, spec, reference=False))
    parts.append("</table></div>")
    parts.append(_legend_html(spec))
    if csv_link:
        parts.append(f'<p class="caption">{csv_link}</p>')
    parts.append("</figure>")
    return "".join(parts)


def render_heatmap_md(frame: pd.DataFrame, spec: dict) -> list[str]:
    """The same heatmap as a pipe table of the cell values, plus a key for short headers."""
    reference, body, columns, cells = _layout(frame, spec)
    heading = str(spec.get("row_heading") or spec["row"])
    header = [heading, *(_column_label(spec, c) for c in columns)]

    def clean(text: str) -> str:
        return str(text).replace("|", "\\|")

    lines = [
        "| " + " | ".join(clean(h) for h in header) + " |",
        "|" + "|".join([":---"] + ["---:"] * len(columns)) + "|",
    ]
    for row in reference + body:
        values = [_cell_value(cells[(row, c)], spec) if (row, c) in cells else "" for c in columns]
        lines.append("| " + " | ".join(clean(v) for v in [row, *values]) + " |")
    key = [
        f"{_column_label(spec, c)} = {_column_name(spec, c)}"
        for c in columns if _column_label(spec, c) != _column_name(spec, c)
    ]
    out = [f"**{spec['title']}**", ""] if spec.get("title") else []
    out.extend(lines)
    out.append("")
    if key:
        out.extend([f"_Columns: {'; '.join(key)}._", ""])
    return out
