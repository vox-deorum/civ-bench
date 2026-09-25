"""HTML heatmap tables for the rating pages.

A matchup matrix (row player type against column opponent) and the
per-strategy Elo grid each become a long table that the report renders
through :mod:`bench.reports.heatmap`. The table has one row per cell with the
``value``, a ``color_position`` from 0 to 1, the cell's display text, and
preformatted tooltip columns. The matching layout spec goes in
``metadata["heatmaps"]``.

Rows and columns use the labels and order of :mod:`bench.analyses.heatmap_layout`,
so a matrix reads like the behavior pages: "Strategist | Condition" rows with
the Null and Vanilla baselines pinned at the top. With pairing on, opponent
columns are grouped under their strategist and headed by their condition.

The saved matrix CSVs keep their square shape; these tables sit next to them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..base import AnalysisContext
from ..heatmap_layout import identity_parts

BASELINE_GROUP = "Baselines"


# ── formatting ───────────────────────────────────────────────────────────────
def p_text(p) -> str:
    """A p-value for a tooltip: ``<0.001``, three decimals, or blank."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def sig_stars(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def count_text(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return ""
    return f"{int(n)}" if np.isfinite(n) else ""


# ── colors ───────────────────────────────────────────────────────────────────
def spread(values: pd.Series, center: float, floor: float) -> float:
    """The largest distance of a finite value from ``center``, at least ``floor``."""
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return floor
    return max(float((finite - center).abs().max()), floor)


def diverging_positions(values: pd.Series, center: float, half_range: float) -> pd.Series:
    """0.5 at ``center``, 0 and 1 at ``center -/+ half_range`` (clipped)."""
    values = pd.to_numeric(values, errors="coerce")
    return (0.5 + (values - center) / (2 * half_range)).clip(0.0, 1.0)


def signed_legend(half_range: float, decimals: int, low: str, high: str, center: str) -> list:
    return [
        [0.0, f"{-half_range:+.{decimals}f} ({low})"],
        [0.25, ""],
        [0.5, center],
        [0.75, ""],
        [1.0, f"{half_range:+.{decimals}f} ({high})"],
    ]


# ── matchup matrices ─────────────────────────────────────────────────────────
def matrix_cells(value: pd.DataFrame, **extras: pd.DataFrame) -> pd.DataFrame:
    """Off-diagonal cells with a finite value, as ``player_type``, ``opponent``, ``value``.

    Each extra matrix adds a column of its entries for the same cells.
    """
    rows = []
    for identity in value.index:
        for opponent in value.columns:
            if str(identity) == str(opponent):
                continue
            cell = value.loc[identity, opponent]
            if pd.isna(cell) or not np.isfinite(float(cell)):
                continue
            row = {"player_type": str(identity), "opponent": str(opponent), "value": float(cell)}
            for name, matrix in extras.items():
                row[name] = matrix.loc[identity, opponent]
            rows.append(row)
    columns = ["player_type", "opponent", "value", *extras]
    return pd.DataFrame(rows, columns=columns)


def matrix_heatmap(
    ctx: AnalysisContext,
    cells: pd.DataFrame,
    identities,
    *,
    title: str,
    help_text: str,
    value_label: str,
    decimals: int,
    legend: list,
    signed: bool = False,
    unit: str = "",
    tip_rows: Optional[list] = None,
) -> tuple[pd.DataFrame, dict]:
    """The long table and layout spec of one matchup matrix.

    ``cells`` comes from :func:`matrix_cells` plus the caller's
    ``color_position`` and ``cell_text`` columns and any tooltip columns that
    ``tip_rows`` names. ``identities`` lists every row and column identity,
    so the grid is complete and the diagonal renders as blank cells.
    """
    parts, ordered = identity_parts(ctx, identities)
    paired = ctx.condition_pairing() is not None
    baselines = {ctx.catalog.null_label, ctx.catalog.vanilla_label}
    labels = {identity: parts[identity][2] for identity in ordered}
    order = [labels[identity] for identity in ordered]

    table = cells.copy()
    table["row_label"] = table["player_type"].map(labels)
    table["column_key"] = table["opponent"].map(labels)
    lead = ["player_type", "opponent", "row_label", "column_key"]
    spec = {
        "title": title,
        "help": help_text,
        "row": "row_label",
        "column": "column_key",
        "value": "value",
        "row_heading": "Strategist | Condition" if paired else "Player type",
        "row_order": order,
        "column_order": order,
        "complete_grid": True,
        "reference_rows": [labels[i] for i in ordered if i in baselines],
        "column_names": {label: f"vs {label}" for label in order},
        "decimals": int(decimals),
        "signed": bool(signed),
        "value_label": value_label,
        "text_column": "cell_text",
        "legend": [list(entry) for entry in legend],
    }
    if unit:
        spec["unit"] = unit
    if paired:
        # Opponents group under their strategist and are headed by their condition.
        def group(identity: str) -> str:
            return BASELINE_GROUP if identity in baselines else parts[identity][0]

        table["opponent_group"] = table["opponent"].map(group)
        lead.append("opponent_group")
        spec["column_group"] = "opponent_group"
        spec["column_labels"] = {
            labels[i]: (i if i in baselines else parts[i][1]) for i in ordered
        }
    if tip_rows:
        spec["tip_rows"] = [dict(row) for row in tip_rows]
    rest = [c for c in table.columns if c not in lead]
    table = table[lead + rest]
    rank = {label: i for i, label in enumerate(order)}
    table = table.sort_values(
        ["row_label", "column_key"], key=lambda s: s.map(rank), kind="stable"
    ).reset_index(drop=True)
    return table, spec


# ── per-strategy Elo grid ────────────────────────────────────────────────────
def strategy_heatmap(
    ctx: AnalysisContext,
    ratings: pd.DataFrame,
    general: pd.DataFrame,
    strategies: list[str],
    reference: str,
    title: str,
) -> tuple[pd.DataFrame, dict]:
    """Elo per player type (rows) for ``General`` and each strategy (columns).

    Colors diverge around 1500, the reference, reaching full color at the
    largest gap. Stars mark a z-test against the reference row in the same
    column; the tooltip adds the SE, the p-value, appearances, and the share
    of the row's strategy appearances.
    """
    from scipy.stats import norm

    records = []
    for frame, column in ((general, None), (ratings, "strategy")):
        for row in frame.itertuples(index=False):
            key = "General" if column is None else str(getattr(row, column))
            records.append({
                "player_type": str(getattr(row, "player_type")),
                "column_key": key,
                "elo": float(getattr(row, "elo")),
                "se_elo": float(getattr(row, "se_elo", np.nan)),
                "appearances": float(getattr(row, "appearances", np.nan)),
            })
    table = pd.DataFrame(records, columns=["player_type", "column_key", "elo", "se_elo", "appearances"])
    columns = ["General", *strategies]
    table = table[table["column_key"].isin(columns)].reset_index(drop=True)

    in_strategies = table["column_key"] != "General"
    totals = table[in_strategies].groupby("player_type")["appearances"].sum()
    totals = totals.replace(0, np.nan)
    table["share"] = np.where(
        in_strategies, table["appearances"] / table["player_type"].map(totals) * 100, np.nan
    )

    identities = list(dict.fromkeys(table["player_type"]))
    ref = reference if reference in identities else ctx.catalog.vanilla_label
    ref_rows = table[table["player_type"] == ref].set_index("column_key")
    p_values = []
    for row in table.itertuples(index=False):
        if row.player_type == ref or row.column_key not in ref_rows.index:
            p_values.append(np.nan)
            continue
        ref_row = ref_rows.loc[row.column_key]
        combined = np.sqrt(np.nansum([row.se_elo ** 2, float(ref_row["se_elo"]) ** 2]))
        p_values.append(
            2 * norm.sf(abs((row.elo - float(ref_row["elo"])) / combined)) if combined > 0 else np.nan
        )
    table["p_value_vs_ref"] = p_values
    table["p_text"] = table["p_value_vs_ref"].map(p_text)
    table["cell_text"] = [
        f"{elo:.0f}{sig_stars(p)}" for elo, p in zip(table["elo"], table["p_value_vs_ref"])
    ]
    half = spread(table["elo"], 1500.0, 1.0)
    table["color_position"] = diverging_positions(table["elo"], 1500.0, half)

    parts, ordered = identity_parts(ctx, identities)
    paired = ctx.condition_pairing() is not None
    labels = {identity: parts[identity][2] for identity in ordered}
    order = [labels[identity] for identity in ordered]
    baselines = {ctx.catalog.null_label, ctx.catalog.vanilla_label}
    table.insert(1, "row_label", table["player_type"].map(labels))
    rank = {label: i for i, label in enumerate(order)}
    column_rank = {c: i for i, c in enumerate(columns)}
    table = table.sort_values(
        ["row_label", "column_key"],
        key=lambda s: s.map(rank if s.name == "row_label" else column_rank),
        kind="stable",
    ).reset_index(drop=True)

    spec = {
        "title": title,
        "help": (
            "Elo for each player type overall (General) and within each strategy, with "
            f"{ref} fixed at 1500. Colors run from red (below {ref}) through yellow "
            f"(level with {ref}) to blue (above). Stars mark a z-test against {ref} in "
            "the same column: * p<0.05, ** p<0.01, *** p<0.001. Hover a cell for the "
            "standard error, appearances, and the strategy's share of the row's games."
        ),
        "row": "row_label",
        "column": "column_key",
        "value": "elo",
        "row_heading": "Strategist | Condition" if paired else "Player type",
        "row_order": order,
        "column_order": columns,
        "reference_rows": [labels[i] for i in ordered if i in baselines],
        "column_names": {"General": "General (all strategies)"},
        "divider_columns": ["General"],
        "decimals": 0,
        "value_label": "Elo",
        "text_column": "cell_text",
        "tip_rows": [
            {"column": "se_elo", "label": "SE", "decimals": 0},
            {"column": "p_text", "label": f"p vs {ref}", "text": True},
            {"column": "appearances", "label": "Appearances", "decimals": 0},
            {"column": "share", "label": "Share of games", "decimals": 0, "unit": "%"},
        ],
        "legend": [
            [0.0, f"{1500 - half:.0f}"],
            [0.25, ""],
            [0.5, f"1500 ({ref})"],
            [0.75, ""],
            [1.0, f"{1500 + half:.0f}"],
        ],
    }
    return table, spec
