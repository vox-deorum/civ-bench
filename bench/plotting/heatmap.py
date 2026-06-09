"""Diverging-heatmap helper (ported per-analysis, stage-0 convention).

Used by ``calibration.cell_baseline`` (the implicit-vs-explicit per-cell baseline
heatmap) and available to any analysis rendering a logit-scale, zero-centered
matrix. Provides robust symmetric colour limits (percentile-clipped so a sentinel
value does not saturate the scale) and supports a hatched mask for missing cells.

Imports matplotlib/seaborn, so — like the rest of :mod:`bench.plotting` — it is
off the import-light config/dry-run path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def robust_symmetric_limit(values, pct: float = 98.0, floor: float = 0.1) -> float:
    """Symmetric colour limit = the ``pct`` percentile of |finite values|.

    Percentile-clipping keeps an extreme sentinel (e.g. ``logit(1-eps) ≈ 11.51``)
    from flattening the colour scale; never returns below ``floor``.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return floor
    return max(float(np.percentile(np.abs(arr), pct)), floor)


def plot_diverging_heatmap(
    matrix: pd.DataFrame,
    *,
    annot: Optional[pd.DataFrame] = None,
    vlim: Optional[float] = None,
    mask: Optional[pd.DataFrame] = None,
    sentinel_mask: Optional[pd.DataFrame] = None,
    title: str = "",
    cbar_label: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rule_after_row: Optional[int] = None,
    cmap: str = "RdBu_r",
    figsize: Optional[tuple[float, float]] = None,
):
    """Render a zero-centered diverging heatmap.

    ``mask`` cells are hatched/blanked (missing data); ``sentinel_mask`` cells get
    a distinct marker (clipped/enforced-winner sentinels). ``rule_after_row`` draws
    a horizontal rule below that row index (used to separate the pinned explicit
    reference row from the implicit conditions).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (max(6.0, 0.6 * n_cols + 3), max(3.0, 0.5 * n_rows + 2))
    if vlim is None:
        vlim = robust_symmetric_limit(matrix.to_numpy())

    fig, ax = plt.subplots(figsize=figsize)
    annot_arr = annot.to_numpy() if annot is not None else False
    sns.heatmap(
        matrix,
        annot=annot_arr,
        fmt="" if annot is not None else ".2f",
        cmap=cmap,
        center=0.0,
        vmin=-vlim,
        vmax=vlim,
        mask=mask.to_numpy() if mask is not None else None,
        linewidths=0.4,
        linecolor="lightgray",
        cbar_kws={"label": cbar_label},
        annot_kws={"fontsize": 7},
        ax=ax,
    )

    # Hatch the masked (missing) cells so they read as "no data", not "zero".
    if mask is not None:
        for i in range(n_rows):
            for j in range(n_cols):
                if bool(mask.iloc[i, j]):
                    ax.add_patch(
                        plt.Rectangle((j, i), 1, 1, fill=True, facecolor="white",
                                      hatch="///", edgecolor="lightgray", linewidth=0.4)
                    )
    # Mark sentinel/clipped cells with a corner dot.
    if sentinel_mask is not None:
        for i in range(n_rows):
            for j in range(n_cols):
                if bool(sentinel_mask.iloc[i, j]):
                    ax.plot(j + 0.85, i + 0.15, marker="o", markersize=3, color="black")

    if rule_after_row is not None:
        ax.axhline(rule_after_row + 1, color="black", linewidth=1.6)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    return fig, ax
