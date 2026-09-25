"""Shared condition-pairing resolution, ordering, and row plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class PairingSpec:
    suffixes: tuple[str, ...]
    sort_condition: str = "base"
    base_label: str = "Base"


def resolve_pairing(global_block, stage_override, catalog) -> Optional[PairingSpec]:
    """Resolve a stage override over the global presentation setting."""
    # Imported here: loading bench.analyses at module import would cycle back
    # through the analyses that import this module.
    from ..analyses.errors import AnalysisError

    global_cfg = dict(global_block or {})
    if isinstance(stage_override, bool):
        stage_cfg = {"enabled": stage_override}
        override_present = True
    elif isinstance(stage_override, dict):
        stage_cfg = dict(stage_override)
        override_present = True
    else:
        stage_cfg = {}
        override_present = False

    merged = {
        "enabled": global_cfg.get("enabled", False),
        "suffixes": global_cfg.get("suffixes"),
        "sort_condition": global_cfg.get("sort_condition", "base"),
        "base_label": global_cfg.get("base_label", "Base"),
    }
    merged.update(stage_cfg)
    if override_present and isinstance(stage_override, dict) and "enabled" not in stage_cfg:
        merged["enabled"] = global_cfg.get("enabled", True)
    if not bool(merged["enabled"]):
        return None

    configured = merged.get("suffixes")
    suffixes = catalog.condition_suffixes() if configured is None else list(configured)
    suffixes = sorted(set(suffixes), key=lambda value: (-len(value), value))
    if not suffixes:
        raise AnalysisError(
            "condition pairing is enabled, but no suffix-style player_type_labels "
            "were configured and presentation.condition_pairing.suffixes is null."
        )
    sort_condition = merged.get("sort_condition", "base")
    if sort_condition not in ("base", "best") and sort_condition not in suffixes:
        raise AnalysisError(
            f"condition pairing sort_condition {sort_condition!r} is not in the "
            f"effective suffixes {suffixes}."
        )
    return PairingSpec(
        suffixes=tuple(suffixes),
        sort_condition=sort_condition,
        base_label=str(merged.get("base_label", "Base")),
    )


def condition_display(spec, condition_key: str, vanilla_label: str) -> str:
    if condition_key == "vanilla":
        return vanilla_label
    if condition_key == "base":
        return spec.base_label
    return condition_key.lstrip("-")


def order_conditions(spec, observed_keys: set[str], vanilla_label: str) -> list[str]:
    """Display order: base label, configured suffix order, then unconfigured observed."""
    order = [condition_display(spec, "base", vanilla_label)]
    order += [condition_display(spec, key, vanilla_label) for key in spec.suffixes]
    known = {"vanilla"} | {"base"} | set(spec.suffixes)
    extra = sorted({k for k in observed_keys if k not in known})
    order += [condition_display(spec, key, vanilla_label) for key in extra]
    seen: set[str] = set()
    return [label for label in order if not (label in seen or seen.add(label))]


def order_strategists(catalog, strategists: list[str]) -> list[str]:
    """Catalog order first (by resolved model id), then unlisted identities lexically."""
    catalog_ids = [m["id"] for m in catalog.strategist_models()]
    rank = {model_id: idx for idx, model_id in enumerate(catalog_ids)}

    def sort_key(name: str):
        model_id = catalog.split_player_type(name)["model_id"]
        position = rank.get(str(model_id))
        if position is None:
            return (1, str(model_id), name)
        return (0, position, name)

    return sorted(set(strategists), key=sort_key)


def attach_pair_columns(
    df: pd.DataFrame,
    catalog,
    spec: PairingSpec,
    identity_col: str,
) -> pd.DataFrame:
    """Copy ``df`` and add normalized pairing columns."""
    out = df.copy()
    bases: list[str] = []
    conditions: list[str] = []
    baselines: list[bool] = []
    baseline_names = {catalog.vanilla_label, catalog.null_label}
    for raw in out[identity_col].astype(str):
        base, suffix = catalog.split_condition_suffix(raw, list(spec.suffixes))
        is_baseline = raw in baseline_names
        bases.append(str(base))
        conditions.append("base" if is_baseline or not suffix else suffix)
        baselines.append(is_baseline)
    out["base_identity"] = bases
    out["condition"] = conditions
    out["is_baseline"] = baselines
    return out


def paired_sort_order(
    df: pd.DataFrame,
    spec: PairingSpec,
    value_col: str,
    ascending: bool,
) -> list[str]:
    """Order base identities by the requested condition, with value fallback.

    The input must already carry ``base_identity`` and ``condition``.
    ``spec.sort_condition`` is ``"base"``, ``"best"``, or a suffix. With a
    concrete condition, an identity missing that condition falls back to the
    first available condition. With ``"best"``, each identity is keyed by its
    best value across all its conditions (the maximum when ``ascending`` is
    false, the minimum when true). Identities with no numeric value sort last.
    """
    if df.empty:
        return []

    def identity_value(group: pd.DataFrame) -> float:
        values = pd.to_numeric(group[value_col], errors="coerce").dropna()
        if values.empty:
            return float("nan")
        if spec.sort_condition == "best":
            ordered = values.sort_values(ascending=ascending)
            return float(ordered.iloc[0])
        for condition in (spec.sort_condition, "base", *spec.suffixes):
            candidates = values[group["condition"].astype(str) == condition]
            if not candidates.empty:
                return float(candidates.iloc[0])
        return float("nan")

    keyed: list[tuple[str, float]] = [
        (str(identity), identity_value(group))
        for identity, group in df.groupby("base_identity", sort=False, dropna=False)
    ]

    def key(item):
        identity, value = item
        if pd.isna(value):
            return (1, 0.0, identity)
        return (0, value if ascending else -value, identity)

    return [identity for identity, _ in sorted(keyed, key=key)]


def condition_marker(spec: PairingSpec, condition: str) -> str:
    if condition == "base":
        return "o"
    markers = ("D", "s", "^")
    try:
        return markers[list(spec.suffixes).index(condition) % len(markers)]
    except ValueError:
        return "o"


def _condition_label(spec: PairingSpec, condition: str) -> str:
    return spec.base_label if condition == "base" else condition.lstrip("-")


def _condition_offset(spec: PairingSpec, condition: str) -> float:
    if condition == "base":
        return 0.16
    if len(spec.suffixes) == 1:
        return -0.16
    idx = list(spec.suffixes).index(condition)
    return -0.08 - idx * 0.14


def plot_paired_rows(
    df: pd.DataFrame,
    *,
    catalog,
    spec: PairingSpec,
    value_col: str,
    identity_col: str,
    lo_col: Optional[str] = None,
    hi_col: Optional[str] = None,
    err_col: Optional[str] = None,
    ref_line: Optional[float] = None,
    ref_label: Optional[str] = None,
    annotate_col: Optional[str] = None,
    preliminary_col: Optional[str] = None,
    row_order: Optional[list[str]] = None,
    ascending: bool = False,
    log_x: bool = False,
    provenance_note: Optional[str] = None,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
):
    """Plot one y-row per base identity with condition-specific markers."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .styles import get_player_color

    if df.empty:
        return None
    work = attach_pair_columns(df, catalog, spec, identity_col)
    if row_order is None:
        row_order = paired_sort_order(work, spec, value_col, ascending)
    else:
        row_order = list(dict.fromkeys(str(value) for value in row_order))
    if not row_order:
        return None

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.46 * len(row_order) + 1.8)))
    y_by_identity = {identity: idx for idx, identity in enumerate(row_order)}
    plotted_conditions: set[str] = set()
    for _, row in work.iterrows():
        identity = str(row["base_identity"])
        if identity not in y_by_identity:
            continue
        value = pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0]
        if pd.isna(value) or (log_x and value <= 0):
            continue
        condition = str(row["condition"])
        is_baseline = bool(row["is_baseline"])
        y = y_by_identity[identity] + (0.0 if is_baseline else _condition_offset(spec, condition))
        color = get_player_color(catalog, identity)
        marker = condition_marker(spec, condition)
        marker_face = color if condition == "base" or is_baseline else "none"
        xerr = None
        if lo_col and hi_col and pd.notna(row.get(lo_col)) and pd.notna(row.get(hi_col)):
            lo = float(row[lo_col])
            hi = float(row[hi_col])
            xerr = [[max(0.0, float(value) - lo)], [max(0.0, hi - float(value))]]
        elif err_col and pd.notna(row.get(err_col)):
            xerr = abs(float(row[err_col]))
        ax.errorbar(
            float(value), y, xerr=xerr, fmt=marker, color=color, ecolor=color,
            markerfacecolor=marker_face, markeredgecolor=color, markersize=7,
            elinewidth=1.4, capsize=3, zorder=3,
        )
        plotted_conditions.add("base" if is_baseline else condition)
        annotations: list[str] = []
        if annotate_col and pd.notna(row.get(annotate_col)):
            annotation = str(row[annotate_col])
            if annotation and annotation not in ("False", "0", "0.0"):
                annotations.append(annotation)
        if preliminary_col and bool(row.get(preliminary_col, False)):
            annotations.append("*")
        if annotations:
            ax.annotate(
                " " + " ".join(annotations), (float(value), y), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8, color="#555555",
            )

    if ref_line is not None:
        ax.axvline(ref_line, color="gray", linestyle="--", linewidth=1)
    if log_x:
        ax.set_xscale("log")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.invert_yaxis()

    expected = {"base", *spec.suffixes}
    for idx, identity in enumerate(row_order):
        rows = work[work["base_identity"].astype(str) == identity]
        if rows.empty or bool(rows["is_baseline"].any()):
            continue
        present = set(rows["condition"].astype(str))
        if present != expected:
            ax.get_yticklabels()[idx].set_color("#888888")
            if len(present) == 1:
                only_condition = next(iter(present))
                only = "base" if only_condition == "base" else only_condition.lstrip("-")
                note = f"({only} only)"
            else:
                missing = ", ".join(
                    _condition_label(spec, condition)
                    for condition in ("base", *spec.suffixes)
                    if condition not in present
                )
                note = f"(missing {missing})"
            ax.annotate(
                note, xy=(1.01, idx), xycoords=("axes fraction", "data"),
                va="center", fontsize=8, color="#888888",
            )

    handles = []
    for condition in ("base", *spec.suffixes):
        if condition not in plotted_conditions:
            continue
        handles.append(Line2D(
            [0], [0], marker=condition_marker(spec, condition), linestyle="none",
            markerfacecolor="#555555" if condition == "base" else "none",
            markeredgecolor="#555555", label=_condition_label(spec, condition),
        ))
    if ref_line is not None:
        handles.append(Line2D(
            [0], [0], color="gray", linestyle="--",
            label=ref_label or f"Reference ({ref_line:g})",
        ))
    if handles:
        ax.legend(handles=handles, fontsize=9, loc="best")
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    if provenance_note:
        fig.text(0.01, 0.01, provenance_note, ha="left", va="bottom", fontsize=8, color="#666666")
        fig.tight_layout(rect=(0, 0.04, 0.95, 1))
    else:
        fig.tight_layout(rect=(0, 0, 0.95, 1))
    return fig


_PLOTLY_SYMBOLS = {"o": "circle", "D": "diamond", "s": "square", "^": "triangle-up"}


def _missing_note(spec: PairingSpec, present: set[str]) -> str:
    """``(Base only)`` / ``(missing Brief)`` for an identity lacking some conditions."""
    if len(present) == 1:
        only = next(iter(present))
        return f"({'base' if only == 'base' else only.lstrip('-')} only)"
    missing = ", ".join(
        _condition_label(spec, condition)
        for condition in ("base", *spec.suffixes)
        if condition not in present
    )
    return f"(missing {missing})"


def plot_paired_rows_interactive(
    df: pd.DataFrame,
    *,
    catalog,
    spec: PairingSpec,
    value_col: str,
    identity_col: str,
    lo_col: Optional[str] = None,
    hi_col: Optional[str] = None,
    err_col: Optional[str] = None,
    ref_line: Optional[float] = None,
    ref_label: Optional[str] = None,
    tooltip: Optional[list[tuple[str, str]]] = None,
    tooltip_note: Optional[str] = None,
    row_order: Optional[list[str]] = None,
    ascending: bool = False,
    log_x: bool = False,
    percent_x: bool = False,
    provenance_note: Optional[str] = None,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
):
    """The Plotly version of :func:`plot_paired_rows`, with a table tooltip.

    Rows, order, offsets, and colors follow :func:`plot_paired_rows`. Each point
    is its own trace so it carries its own color and error bar; hidden
    legend-only traces, one per condition, toggle that condition's points (a
    spec without suffixes, pairing off, has no condition legend or subtitle).
    ``tooltip`` lists ``(label, column)`` pairs of pre-formatted strings shown
    under the identity and condition on hover (the first row in bold); without
    it the tooltip shows the value. ``percent_x`` formats the axis as percents
    of 1. Returns ``None`` when nothing can be plotted.
    """
    from html import escape

    import plotly.graph_objects as go

    from .styles import get_player_color

    if df.empty:
        return None
    work = attach_pair_columns(df, catalog, spec, identity_col)
    if row_order is None:
        row_order = paired_sort_order(work, spec, value_col, ascending)
    else:
        row_order = list(dict.fromkeys(str(value) for value in row_order))
    if not row_order:
        return None

    rows = list(tooltip or [(xlabel or value_col, "")])
    y_by_identity = {identity: idx for idx, identity in enumerate(row_order)}
    fig = go.Figure()
    plotted_conditions: set[str] = set()
    for _, row in work.iterrows():
        identity = str(row["base_identity"])
        if identity not in y_by_identity:
            continue
        value = pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0]
        if pd.isna(value) or (log_x and value <= 0):
            continue
        value = float(value)
        condition = str(row["condition"])
        is_baseline = bool(row["is_baseline"])
        group = "base" if is_baseline else condition
        offset = 0.0 if is_baseline or not spec.suffixes else _condition_offset(spec, condition)
        y = y_by_identity[identity] + offset
        color = get_player_color(catalog, identity)
        symbol = _PLOTLY_SYMBOLS.get(condition_marker(spec, condition), "circle")
        if not (condition == "base" or is_baseline):
            symbol += "-open"
        error_x = None
        if lo_col and hi_col and pd.notna(row.get(lo_col)) and pd.notna(row.get(hi_col)):
            error_x = {
                "type": "data", "symmetric": False,
                "array": [max(0.0, float(row[hi_col]) - value)],
                "arrayminus": [max(0.0, value - float(row[lo_col]))],
            }
        elif err_col and pd.notna(row.get(err_col)):
            error_x = {"type": "data", "array": [abs(float(row[err_col]))]}
        if error_x is not None:
            error_x.update({"color": color, "thickness": 1.4, "width": 3})
        details = [
            str(row[identity_col]),
            "" if is_baseline or not spec.suffixes else _condition_label(spec, condition),
            *(
                (str(row[column]) if column and pd.notna(row.get(column)) else "")
                if column else f"{value:.4g}"
                for _label, column in rows
            ),
        ]
        fig.add_trace(go.Scatter(
            x=[value], y=[y], mode="markers", error_x=error_x,
            name=escape(str(row[identity_col])), legendgroup=group, showlegend=False,
            marker={"color": color, "symbol": symbol, "size": 10, "line": {"width": 1.5, "color": color}},
            customdata=[details], meta={"table_tooltip": True}, hoverinfo="none",
        ))
        plotted_conditions.add(group)

    # Without pairing there is one condition, so no condition legend to filter by.
    for condition in ("base", *spec.suffixes) if spec.suffixes else ():
        if condition not in plotted_conditions:
            continue
        symbol = _PLOTLY_SYMBOLS.get(condition_marker(spec, condition), "circle")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=escape(_condition_label(spec, condition)),
            legendgroup=condition, showlegend=True, hoverinfo="skip",
            marker={"color": "#555555", "size": 10,
                    "symbol": symbol if condition == "base" else symbol + "-open"},
        ))
    if ref_line is not None:
        fig.add_trace(go.Scatter(
            x=[ref_line, ref_line], y=[-0.5, len(row_order) - 0.5], mode="lines",
            name=escape(ref_label or f"Reference ({ref_line:g})"), hoverinfo="skip",
            line={"color": "gray", "dash": "dash", "width": 1},
        ))

    expected = {"base", *spec.suffixes}
    ticktext = []
    for identity in row_order:
        label = escape(identity)
        members = work[work["base_identity"].astype(str) == identity]
        if not members.empty and not bool(members["is_baseline"].any()):
            present = set(members["condition"].astype(str))
            if present != expected:
                label = (f'<span style="color:#888888">{label} '
                         f"{escape(_missing_note(spec, present))}</span>")
        ticktext.append(label)

    annotations = []
    if provenance_note:
        annotations.append({
            "x": 0, "y": 0, "xref": "paper", "yref": "paper", "xanchor": "left",
            "yanchor": "top", "yshift": -48, "showarrow": False, "align": "left",
            "text": escape(provenance_note), "font": {"size": 11, "color": "#666666"},
        })
    xaxis = {"title": escape(xlabel) if xlabel else None, "type": "log" if log_x else "linear",
             "showgrid": True, "zeroline": False}
    if percent_x:
        xaxis["tickformat"] = ".0%"
    fig.update_layout(
        template="plotly_white",
        title={"text": f"<b>{escape(title)}</b>" if title else None, "x": 0.02},
        xaxis=xaxis,
        yaxis={"tickmode": "array", "tickvals": list(range(len(row_order))), "ticktext": ticktext,
               "range": [len(row_order) - 0.5, -0.5], "showgrid": False, "zeroline": False,
               "automargin": True},
        height=max(360, 34 * len(row_order) + 190),
        margin={"t": 60, "b": 90 if provenance_note else 60},
        hovermode="closest", annotations=annotations,
        legend={"title": {"text": "Click to filter"}},
        meta={"table_tooltip": {
            "title_index": 0, "subtitle_index": 1, "note": tooltip_note,
            "rows": [
                {"label": label, "index": index + 2, **({"emphasis": True} if index == 0 else {})}
                for index, (label, _column) in enumerate(rows)
            ],
        }},
    )
    return fig
