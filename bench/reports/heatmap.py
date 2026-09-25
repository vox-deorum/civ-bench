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
``baseline_rows``
    Rows holding the baseline's absolute values: shown unsigned, with an
    ``sd`` line in the tooltip of a signed (relative) table.
``column_group``
    Optional column naming each column's group, shown as a spanning header.
``column_names``, ``column_labels``, ``column_tips``
    The full column name (tooltips and the Markdown key), the short header
    text, and the header tooltip.
``decimals``, ``signed``, ``value_label``, ``ci_level``
    Number format and the tooltip's value line.
``column_decimals``, ``column_value_labels``, ``column_units``, ``unit``
    Optional per-column overrides of ``decimals`` and ``value_label``, and a
    unit appended to the tooltip's numbers (such as ``%``) per column or for
    the whole table, for tables whose columns measure different things.
    ``baseline_units`` gives the units of the baseline rows, which hold
    absolute values even in a signed table.
``range_columns``, ``range_label``, ``range_note``
    Optional ``[low, high]`` columns shown as one more tooltip line, in the
    table's number format (signed in a signed table) and followed by
    ``range_note`` (default ``mean min to max``).
``legend``
    ``[[position, label], ...]`` swatches under the table; a ``#rrggbb`` string
    in place of the position gives the swatch color directly.
``complete_grid``
    When true, ``row_order`` and ``column_order`` are the full grid: listed
    rows and columns render even without data, as blank cells.
``tip_rows``
    Optional ``[{"column", "label", "signed", "decimals", "unit", "units", "text"}, ...]``:
    more tooltip lines from other columns, such as a difference from the
    baseline. ``units`` maps a table column to the line's unit there, else
    ``unit`` applies; a row with ``"text": true`` shows the column's string as is.
``category_column``, ``category_colors``
    Optional categorical coloring, the Matched Maps Focus style: a cell whose
    category column names a key of ``category_colors`` (``#rrggbb``) is shaded
    from white toward that color by its ``color_position`` (see
    :func:`category_background`). Other cells keep the RdYlBu color.

Some tables need more than one number per cell. These optional keys each name
a column of the long table that overrides one part of a cell:

``text_column``
    The cell text (for example ``Science 45%``) instead of the formatted value.
``background_column``
    A ``#rrggbb`` background instead of the ``color_position`` color.
``link_column``
    A link target; the cell text becomes a link.
``tip_column``
    The whole tooltip text instead of the default one.

``divider_columns`` lists columns followed by a thick border (such as a
leading average column).

Every value cell carries its raw value as ``data-value`` and every column
header its position as ``data-col``, which the shared report script uses to
sort a table by a numeric column.

A cell tooltip follows the report script's tip format: the first line is the
title, other plain lines are subtitles, and ``label<TAB>value[<TAB>note]``
lines form a grid with the numbers in an accent color (see :func:`tip_row`).

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
OVERRIDE_KEYS = ("text_column", "background_column", "link_column", "tip_column")


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


def category_background(color: str, value: float) -> str:
    """``color`` at intensity ``value`` in [0, 1]: a faint tint at 0, full at 1."""
    t = 0.0 if not np.isfinite(value) else min(1.0, max(0.0, value))
    return interpolate("#ffffff", color, 0.22 + 0.78 * t)


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


def text_columns(spec: Optional[dict]) -> list[str]:
    """The columns a spec shows as text, which a CSV read must keep as strings.

    Read back as numbers, a cell text such as ``+0.50`` would lose its sign and
    trailing zero.
    """
    if not isinstance(spec, dict):
        return []
    names = [spec.get(key) for key in OVERRIDE_KEYS]
    names += [row.get("column") for row in spec.get("tip_rows") or []
              if isinstance(row, dict) and row.get("text")]
    return list(dict.fromkeys(n for n in names if isinstance(n, str) and n))


def spec_fits(spec: Optional[dict], frame: pd.DataFrame) -> bool:
    """Whether ``frame`` has every column the spec needs; otherwise render plainly."""
    if not isinstance(spec, dict):
        return False
    row, column = spec.get("row"), spec.get("column")
    if not isinstance(row, str) or not isinstance(column, str):
        return False
    needed = {row, column, str(spec.get("value", "mean")), COLOR_COLUMN}
    for key in ("column_group", "category_column", *OVERRIDE_KEYS):
        name = spec.get(key)
        if name is None:
            continue
        if not isinstance(name, str):
            return False
        needed.add(name)
    return needed <= set(frame.columns)


# ── layout ────────────────────────────────────────────────────────────────────
def _esc(text) -> str:
    return _html.escape(str(text))


def _ordered(observed: list[str], preferred) -> list[str]:
    preferred = [str(v) for v in (preferred or []) if str(v) in set(observed)]
    return list(dict.fromkeys(preferred + observed))


def _layout(frame: pd.DataFrame, spec: dict):
    row, column = spec["row"], spec["column"]
    observed_rows = list(dict.fromkeys(frame[row].astype(str)))
    observed_columns = list(dict.fromkeys(frame[column].astype(str)))
    if spec.get("complete_grid"):
        # The declared order is the full grid; unobserved cells render blank.
        observed_rows = [str(v) for v in spec.get("row_order") or []] + observed_rows
        observed_columns = [str(v) for v in spec.get("column_order") or []] + observed_columns
    rows = _ordered(observed_rows, spec.get("row_order"))
    columns = _ordered(observed_columns, spec.get("column_order"))
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


def _signed(spec: dict, row: str) -> bool:
    return bool(spec.get("signed", False)) and row not in set(spec.get("baseline_rows") or [])


def _text(rec: dict, column) -> str:
    """A string cell of an override column; blank for missing values."""
    if not column:
        return ""
    value = rec.get(column)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return str(value)


def _decimals(spec: dict, column: str) -> int:
    """The column's number of decimals, else the table's."""
    per_column = spec.get("column_decimals") or {}
    return int(per_column.get(column, spec.get("decimals", 1)))


def _cell_value(rec: dict, spec: dict, row: str, column: str) -> str:
    text = _text(rec, spec.get("text_column"))
    if text:
        return text
    return _number(rec.get(spec.get("value", "mean")), _decimals(spec, column),
                   _signed(spec, row))


def _raw_value(rec: dict, spec: dict) -> str:
    """The unrounded value a sort compares, or blank."""
    try:
        value = float(rec.get(spec.get("value", "mean")))
    except (TypeError, ValueError):
        return ""
    return f"{value:.6g}" if np.isfinite(value) else ""


def tip_row(label: str, value: str = "", note: str = "") -> str:
    """One grid line of a tooltip: muted label, accent-colored value, muted note."""
    return f"{label}\t{value}\t{note}" if note else f"{label}\t{value}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def _column_name(spec: dict, column: str) -> str:
    return str((spec.get("column_names") or {}).get(column, column))


def _column_label(spec: dict, column: str) -> str:
    return str((spec.get("column_labels") or {}).get(column, _column_name(spec, column)))


def _cell_tooltip(rec: dict, spec: dict, row: str, column: str) -> str:
    """Row and column name, then value, CI, spread, range, and sample size as grid lines."""
    override = _text(rec, spec.get("tip_column"))
    if override:
        return override
    places = _decimals(spec, column)
    decimals = places + 1
    baseline = row in set(spec.get("baseline_rows") or [])
    units = spec["baseline_units"] if baseline and "baseline_units" in spec else spec.get("column_units")
    unit = str((units or {}).get(column, spec.get("unit", "")))
    signed = _signed(spec, row)
    value = _with_unit(_number(rec.get(spec.get("value", "mean")), decimals, signed), unit)
    value_labels = spec.get("column_value_labels") or {}
    label = "Average" if baseline else str(
        value_labels.get(column) or spec.get("value_label") or "Value"
    )
    lines = [row, _column_name(spec, column), tip_row(label, value)]
    lo = _with_unit(_number(rec.get("ci_lower"), decimals, signed), unit)
    hi = _with_unit(_number(rec.get("ci_upper"), decimals, signed), unit)
    if lo and hi:
        level = spec.get("ci_level")
        prefix = f"{float(level) * 100:g}% CI" if isinstance(level, (int, float)) else "CI"
        lines.append(tip_row(prefix, "", f"{lo} to {hi}"))
    sd = _number(rec.get("sd"), decimals, False)
    if baseline and sd and spec.get("signed"):
        lines.append(tip_row("SD", sd, "one legend step"))
    bounds = spec.get("range_columns")
    if isinstance(bounds, list) and len(bounds) == 2:
        low = _number(rec.get(bounds[0]), places, signed)
        high = _number(rec.get(bounds[1]), places, signed)
        if low and high:
            note = str(spec.get("range_note") or "mean min to max")
            lines.append(tip_row(str(spec.get("range_label", "Range")), "",
                                 f"{low} to {high} ({note})"))
    for extra in spec.get("tip_rows") or []:
        if not isinstance(extra, dict) or not extra.get("column"):
            continue
        if extra.get("text"):
            text = _text(rec, extra["column"])
        else:
            text = _with_unit(
                _number(rec.get(extra["column"]), int(extra.get("decimals", decimals)),
                        bool(extra.get("signed", False))),
                str((extra.get("units") or {}).get(column, extra.get("unit", ""))),
            )
        if text:
            lines.append(tip_row(str(extra.get("label", extra["column"])), text))
    n_players, n_games = rec.get("n_players"), rec.get("n_games")
    if n_players is not None and n_games is not None and np.isfinite(float(n_players)):
        players, games = int(n_players), int(float(n_games))
        lines.append(tip_row("Players", str(players), f"in {plural(games, 'game')}"))
    return "\n".join(lines)


def _with_unit(text: str, unit: str) -> str:
    """``text`` with its unit: ``%`` joins the number, a word follows a space."""
    if not text or not unit:
        return text
    return f"{text}{unit}" if unit == "%" else f"{text} {unit}"


def _label_html(label: str, tip: str) -> str:
    if tip:
        return f'<span tabindex="0" data-tip="{_esc(tip)}">{_esc(label)}</span>'
    return _esc(label)


def _background(rec: dict, spec: dict) -> str:
    override = _text(rec, spec.get("background_column"))
    if override.startswith("#") and len(override) == 7:
        return override
    try:
        position = float(rec.get(COLOR_COLUMN))
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(position):
        return ""
    category = _text(rec, spec.get("category_column"))
    color = (spec.get("category_colors") or {}).get(category) if category else None
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        return category_background(color, position)
    return position_background(position)


def _cell_html(rec: Optional[dict], spec: dict, row: str, column: str) -> str:
    divider = " col-divider" if column in set(spec.get("divider_columns") or []) else ""
    if rec is None:
        return f'<td class="heat-cell heat-cell-empty{divider}" aria-label="no data"></td>'
    text = _cell_value(rec, spec, row, column)
    tip = _esc(_cell_tooltip(rec, spec, row, column))
    value = _raw_value(rec, spec)
    value_attr = f' data-value="{value}"' if value else ""
    background = _background(rec, spec) if text else ""
    style = (
        f' style="background-color:{background};color:{cell_text_color(background)}"'
        if background else ""
    )
    href = _text(rec, spec.get("link_column"))
    if href and text:
        return (
            f'<td class="heat-cell heat-cell-link{divider}"{style}{value_attr} data-tip="{tip}">'
            f'<a href="{_esc(href)}">{_esc(text)}</a></td>'
        )
    return (
        f'<td class="heat-cell heat-cell-value{divider}"{style}{value_attr} '
        f'data-tip="{tip}">{_esc(text)}</td>'
    )


def _header_html(frame: pd.DataFrame, spec: dict, columns: list[str]) -> list[str]:
    heading = _esc(spec.get("row_heading", ""))
    tips = spec.get("column_tips") or {}
    group_col = spec.get("column_group")
    parts = ["<thead>"]
    dividers = set(spec.get("divider_columns") or [])

    def header(index: int, column: str) -> str:
        divider = ' class="col-divider"' if column in dividers else ""
        label = _label_html(_column_label(spec, column), str(tips.get(column, "")))
        return f'<th scope="col"{divider} data-col="{index + 1}">{label}</th>'

    headers = "".join(header(i, c) for i, c in enumerate(columns))
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
        if isinstance(entry[0], str) and entry[0].startswith("#"):
            color = entry[0]
        else:
            try:
                color = position_background(float(entry[0]))
            except (TypeError, ValueError):
                continue
        items.append(
            f'<li><span class="swatch" style="background-color:{_esc(color)}"></span>'
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
        values = [_cell_value(cells[(row, c)], spec, row, c) if (row, c) in cells else "" for c in columns]
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
