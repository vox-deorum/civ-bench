"""Apply a config ``groupings`` entry to a panel (stage 4).

A ``group_by`` dimension beyond the base identity (e.g. ``strategy``) names a
top-level ``groupings`` entry. Today only the ``argmax`` kind is implemented
(benchmark.md §3.2): the label is the positional ``labels[argmax(columns)]`` per
row, exactly the legacy ``strategy_ratings`` "dominant strategy" computation,
but config-driven (the columns/labels come from the run-spec, not hardcoded).
"""

from __future__ import annotations

import pandas as pd

from .errors import AnalysisError


def grouping_label(df: pd.DataFrame, grouping: dict, name: str) -> pd.Series:
    """Return the per-row group label Series for one ``groupings`` entry.

    ``df`` must carry the grouping's source columns (join them from ``panel`` if
    the derived table lacks them). Raises :class:`AnalysisError` when a required
    column is missing so the failure is loud, not a silent NaN column.
    """
    kind = grouping.get("kind")
    if kind != "argmax":
        raise AnalysisError(
            f"grouping '{name}': kind '{kind}' is not implemented (only 'argmax')."
        )
    cols = list(grouping.get("columns") or [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AnalysisError(
            f"grouping '{name}': missing source column(s) {missing} on the panel "
            f"(needed for the '{kind}' grouping)."
        )
    labels = list(grouping.get("labels") or cols)
    idx = df[cols].astype(float).to_numpy().argmax(axis=1)
    return pd.Series([labels[i] for i in idx], index=df.index, name=name)
