"""Inline SVG charts for the plain-language front page (stage 5).

Pure functions over a prepared :class:`~bench.reports.model.FrontChart`: they
only place marks and text. Every coordinate is rounded to one decimal so a
re-render is byte-stable. Hover text rides on ``data-tip`` attributes that the
shared report script shows as a tooltip; colors come from the stylesheet,
except the per-series point colors, which arrive on each row.
"""

from __future__ import annotations

import html
import math

from .model import FrontChart

WIDTH = 1000
_CHAR = 7.0  # rough width of one 12 px label character, for label placement


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _f(value: float) -> str:
    return f"{value:.1f}"


def _tip(text: str, focus: bool = True) -> str:
    """The hover tip; ``focus=False`` keeps it off the tab order, as inside a row link."""
    if not text:
        return ""
    return (' tabindex="0"' if focus else "") + f' data-tip="{_esc(text)}"'


def _link(row: dict) -> tuple[str, str]:
    """The element and ``href`` attribute for a row that opens its games, if it has a page."""
    return ("a", f' href="{_esc(row["href"])}"') if row.get("href") else ("g", "")


# ── leaderboard ─────────────────────────────────────────────────────────────
_LB_LABEL, _LB_RIGHT, _ROW = 300, 70, 28


def _span(rows: list[dict], reference: float, pad: float) -> tuple[float, float, float]:
    """A rounded value range covering every row and the reference, plus a tick step."""
    lo = min([row["low"] for row in rows] + [reference])
    hi = max([row["high"] for row in rows] + [reference])
    step = 50 if hi - lo <= 500 else 100
    return math.floor((lo - pad) / step) * step, math.ceil((hi + pad) / step) * step, step


def leaderboard_svgs(chart: FrontChart) -> tuple[str, str]:
    """The top rows, and the remaining rows (empty when all fit), on one scale."""
    rows = chart.rows
    reference = float(chart.extra.get("reference", 1500.0))
    lo, hi, step = _span(rows, reference, 10)
    top = int(chart.extra.get("top", 10))

    def x(value: float) -> float:
        return _LB_LABEL + (value - lo) / (hi - lo) * (WIDTH - _LB_LABEL - _LB_RIGHT)

    def draw(part: list[dict], axis: bool) -> str:
        head = 30 if axis else 8
        height = head + _ROW * len(part) + 8
        out = [f'<svg viewBox="0 0 {WIDTH} {height}" role="group" aria-label="{_esc(chart.title)}">']
        tick = lo + step
        while tick < hi:
            out.append(f'<line class="grid" x1="{_f(x(tick))}" x2="{_f(x(tick))}" y1="{head - 6}" y2="{height - 4}"/>')
            if axis:
                out.append(f'<text class="tick" x="{_f(x(tick))}" y="{head - 12}" text-anchor="middle">{tick:.0f}</text>')
            tick += step
        x0 = x(reference)
        for i, row in enumerate(part):
            y = head + i * _ROW
            cy = y + _ROW / 2
            tag, href = _link(row)
            out.append(
                f'<{tag} class="lb-row {row["kind"]}"{href}{_tip(row["tip"], tag == "g")} '
                f'aria-label="{_esc(row["name"])}, rating {row["elo"]:.0f}">'
            )
            out.append(f'<rect class="hit" x="0" y="{y}" width="{WIDTH}" height="{_ROW}"/>')
            out.append(f'<text class="rank" x="22" y="{_f(cy + 4.5)}" text-anchor="end">{row["rank"]}</text>')
            out.append(f'<text class="name" x="34" y="{_f(cy + 4.5)}">{_esc(row["name"])}</text>')
            if row["mode"]:
                width = round(len(row["mode"]) * 6.2 + 20)
                left = _LB_LABEL - 14 - width
                out.append(
                    f'<g class="mode m{row.get("mode_index", 0)}"{_tip(row["mode_tip"], tag == "g")}>'
                    f'<rect x="{left}" y="{_f(cy - 9)}" width="{width}" height="18" rx="9"/>'
                    f'<text x="{_f(left + width / 2)}" y="{_f(cy + 4)}" text-anchor="middle">{_esc(row["mode"])}</text></g>'
                )
            xe = x(row["elo"])
            if row["kind"] == "ref":
                out.append(f'<circle class="dot" cx="{_f(xe)}" cy="{_f(cy)}" r="5"/>')
            else:
                out.append(
                    f'<rect class="bar" x="{_f(min(x0, xe))}" y="{_f(cy - 7)}" '
                    f'width="{_f(max(abs(xe - x0), 1))}" height="14" rx="2"/>'
                )
            w1, w2 = x(row["low"]), x(row["high"])
            if w2 > w1:
                out.append(f'<line class="whisk" x1="{_f(w1)}" x2="{_f(w2)}" y1="{_f(cy)}" y2="{_f(cy)}"/>')
            up = row["elo"] >= reference
            vx = max(xe, w2) + 6 if up else min(xe, w1) - 6
            anchor = "start" if up else "end"
            out.append(f'<text class="val" x="{_f(vx)}" y="{_f(cy + 4)}" text-anchor="{anchor}">{row["elo"]:.0f}</text>')
            out.append(f"</{tag}>")
        out.append(f'<line class="base" x1="{_f(x0)}" x2="{_f(x0)}" y1="{head - 6}" y2="{height - 4}"/>')
        out.append("</svg>")
        return "\n".join(out)

    rest = rows[top:]
    return draw(rows[:top], True), (draw(rest, False) if rest else "")


# ── cost against skill ──────────────────────────────────────────────────────
_SC_H, _ML, _MR, _MT, _MB = 440, 64, 24, 20, 52
_COST_TICKS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def _shape(shape: str, x: float, y: float) -> str:
    if shape == "circle":
        return f'<circle cx="{_f(x)}" cy="{_f(y)}" r="6.5"/>'
    if shape == "square":
        return f'<rect x="{_f(x - 6)}" y="{_f(y - 6)}" width="12" height="12" rx="1.5"/>'
    if shape == "diamond":
        return f'<path d="M{_f(x)} {_f(y - 7.5)}L{_f(x + 7.5)} {_f(y)}L{_f(x)} {_f(y + 7.5)}L{_f(x - 7.5)} {_f(y)}Z"/>'
    return f'<path d="M{_f(x)} {_f(y - 7.5)}L{_f(x + 7)} {_f(y + 6)}L{_f(x - 7)} {_f(y + 6)}Z"/>'


def shape_icon(shape: str) -> str:
    """A small legend icon for a point shape."""
    return f'<svg viewBox="0 0 16 16" aria-hidden="true">{_shape(shape, 8, 8)}</svg>'


def _place_labels(points: list[tuple[str, float, float, str]], bounds) -> dict[str, tuple[float, float, str]]:
    """Greedy label spots that avoid each other and the points of the same group.

    ``points`` holds ``(key, x, y, text)``. Each label tries right, left,
    above, and below its point; the first spot inside ``bounds`` that hits no
    earlier label or point wins, else the first spot.
    """
    left, top, right, bottom = bounds
    boxes = [(x - 7, y - 7, x + 7, y + 7) for _, x, y, _ in points]
    placed: dict[str, tuple[float, float, str]] = {}
    for key, x, y, text in sorted(points, key=lambda p: (p[2], p[1], p[0])):
        width = len(text) * _CHAR
        options = [
            (x + 10, y + 4, "start"), (x - 10, y + 4, "end"),
            (x + 6, y - 10, "start"), (x - 6, y - 10, "end"),
            (x + 6, y + 19, "start"), (x - 6, y + 19, "end"),
        ]
        chosen = options[0]
        for lx, ly, anchor in options:
            x1 = lx if anchor == "start" else lx - width
            box = (x1, ly - 11, x1 + width, ly + 3)
            inside = box[0] >= left and box[2] <= right and box[1] >= top and box[3] <= bottom
            own = (x - 7, y - 7, x + 7, y + 7)
            clear = all(
                b == own or box[2] <= b[0] or box[0] >= b[2] or box[3] <= b[1] or box[1] >= b[3]
                for b in boxes
            )
            if inside and clear:
                chosen = (lx, ly, anchor)
                break
        lx, ly, anchor = chosen
        x1 = lx if anchor == "start" else lx - width
        boxes.append((x1, ly - 11, x1 + width, ly + 3))
        placed[key] = chosen
    return placed


def cost_svg(chart: FrontChart) -> str:
    rows = chart.rows
    costs = [row["cost"] for row in rows]
    xmin, xmax = math.log10(min(costs)) - 0.15, math.log10(max(costs)) + 0.15
    elos = [row["elo"] for row in rows]
    reference = chart.extra.get("reference")
    span = elos + ([reference] if reference is not None else [])
    ymin = math.floor((min(span) - 20) / 50) * 50
    ymax = math.ceil((max(span) + 20) / 50) * 50

    def sx(cost: float) -> float:
        return _ML + (math.log10(cost) - xmin) / (xmax - xmin) * (WIDTH - _ML - _MR)

    def sy(elo: float) -> float:
        return _MT + (ymax - elo) / (ymax - ymin) * (_SC_H - _MT - _MB)

    out = [f'<svg viewBox="0 0 {WIDTH} {_SC_H}" role="group" aria-label="{_esc(chart.title)}">']
    for cost in _COST_TICKS:
        if not xmin <= math.log10(cost) <= xmax:
            continue
        x = sx(cost)
        label = f"${cost:g}" if cost >= 1 else f"{cost * 100:.0f}¢"
        out.append(f'<line class="grid" x1="{_f(x)}" x2="{_f(x)}" y1="{_MT}" y2="{_SC_H - _MB}"/>')
        out.append(f'<text class="tick" x="{_f(x)}" y="{_SC_H - _MB + 18}" text-anchor="middle">{label}</text>')
    step = 50 if ymax - ymin <= 500 else 100
    for elo in range(int(ymin) + step, int(ymax), step):
        y = sy(elo)
        out.append(f'<line class="grid" x1="{_ML}" x2="{WIDTH - _MR}" y1="{_f(y)}" y2="{_f(y)}"/>')
        out.append(f'<text class="tick" x="{_ML - 8}" y="{_f(y + 4)}" text-anchor="end">{elo}</text>')
    if reference is not None:
        y = sy(reference)
        out.append(f'<line class="base" x1="{_ML}" x2="{WIDTH - _MR}" y1="{_f(y)}" y2="{_f(y)}"/>')
        name = chart.extra.get("reference_name") or ""
        if name:
            out.append(f'<text class="base-label" x="{WIDTH - _MR - 4}" y="{_f(y - 6)}" text-anchor="end">{_esc(name)}</text>')
    fit = chart.extra.get("fit")
    if fit:
        a, b = float(fit["intercept"]), float(fit["slope"])
        c0, c1 = 10 ** (xmin + 0.05), 10 ** (xmax - 0.05)
        out.append(
            f'<line class="fit" x1="{_f(sx(c0))}" y1="{_f(sy(a + b * math.log10(c0)))}" '
            f'x2="{_f(sx(c1))}" y2="{_f(sy(a + b * math.log10(c1)))}"/>'
        )
        out.append(
            f'<text class="fit-label" x="{_f(sx(c1))}" y="{_f(sy(a + b * math.log10(c1)) - 8)}" '
            'text-anchor="end">Typical rating for the price</text>'
        )
    out.append(
        f'<text class="axis" x="{(_ML + WIDTH - _MR) / 2:.0f}" y="{_SC_H - 10}" text-anchor="middle">'
        "Cost per player per game, US dollars (each step multiplies the cost)</text>"
    )
    out.append(
        f'<text class="axis" transform="translate(16 {(_MT + _SC_H - _MB) / 2:.0f}) rotate(-90)" '
        'text-anchor="middle">Rating</text>'
    )
    xy = {row["key"]: (sx(row["cost"]), sy(row["elo"])) for row in rows}
    bounds = (_ML + 2, _MT, WIDTH - _MR, _SC_H - _MB)
    for row in sorted(rows, key=lambda r: (r["elo"], r["key"])):
        x, y = xy[row["key"]]
        out.append(
            f'<g class="pt" style="--c:{_esc(row["color"])}" data-series="{_esc(row["series"])}"'
            f'{_tip(row["tip"])} aria-label="{_esc(row["name"])}">{_shape(row["shape"], x, y)}</g>'
        )
    # Labels: one set shown at rest, and one per series shown while it is picked.
    groups = [("rest", [row for row in rows if row.get("label")])]
    for series in sorted({row["series"] for row in rows}):
        groups.append((series, [row for row in rows if row["series"] == series]))
    for group, members in groups:
        spots = _place_labels([(row["key"], *xy[row["key"]], row["name"]) for row in members], bounds)
        for row in sorted(members, key=lambda r: r["key"]):
            lx, ly, anchor = spots[row["key"]]
            cls = "pt-label rest" if group == "rest" else "pt-label pick"
            out.append(
                f'<text class="{cls}" data-series="{_esc(row["series"])}" x="{_f(lx)}" y="{_f(ly)}" '
                f'text-anchor="{anchor}">{_esc(row["name"])}</text>'
            )
    out.append("</svg>")
    return "\n".join(out)


# ── play styles ─────────────────────────────────────────────────────────────
_ST_LABEL, _ST_ROW = 300, 30


def styles_svg(chart: FrontChart) -> str:
    rows = chart.rows
    gap = 12 if any(row["reference"] for row in rows) else 0
    height = 8 + _ST_ROW * len(rows) + gap + 4
    span = WIDTH - _ST_LABEL - 12
    out = [f'<svg viewBox="0 0 {WIDTH} {height}" role="group" aria-label="{_esc(chart.title)}">']
    for i, row in enumerate(rows):
        y = 8 + i * _ST_ROW + (gap if row["reference"] else 0)
        if row["reference"] and gap:
            out.append(f'<line class="grid" x1="0" x2="{WIDTH}" y1="{_f(y - gap / 2 - 1)}" y2="{_f(y - gap / 2 - 1)}"/>')
        name_cls = "name refname" if row["reference"] else "name"
        mode = f'<tspan class="sub"> · {_esc(row["mode"])}</tspan>' if row["mode"] else ""
        _, href = _link(row)
        if href:
            out.append(f'<a class="st-row"{href} aria-label="{_esc(row["name"])}: see its games">')
        out.append(f'<text class="{name_cls}" x="0" y="{y + 16}">{_esc(row["name"])}{mode}</text>')
        x = float(_ST_LABEL)
        for segment in row["segments"]:
            width = segment["share"] * span
            out.append(
                f'<g class="seg {segment["cls"]}"{_tip(segment["tip"], not href)}>'
                f'<rect x="{_f(x)}" y="{y + 3}" width="{_f(width)}" height="22"/>'
            )
            if segment["share"] >= 0.1:
                out.append(
                    f'<text x="{_f(x + width / 2)}" y="{y + 18}" text-anchor="middle">{segment["share"]:.0%}</text>'
                )
            out.append("</g>")
            x += width
        if href:
            out.append("</a>")
    out.append("</svg>")
    return "\n".join(out)
