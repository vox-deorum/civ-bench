"""Row labels and row order shared by every HTML heatmap an analysis emits.

With condition pairing on and ``by == "player_type"``, a player type reads
"Strategist | Condition" and rows sort like Matched Maps: the Null and Vanilla
baselines first, then catalog strategist order, then condition order.
Otherwise the label is the group value itself, in
:func:`bench.plotting.styles.sort_player_types` order for player types and
alphabetical order for anything else.

:func:`heatmap_rows` labels the rows of a long table; :func:`identity_labels`
labels a bare list of identities, such as the opponent columns of a matchup
matrix, so rows and columns of one heatmap read and sort the same way.
:func:`identity_parts` also returns each identity's strategist and condition.
"""

from __future__ import annotations

import pandas as pd

from ..plotting.pairing import condition_display, order_conditions, order_strategists
from ..plotting.styles import sort_player_types
from .base import AnalysisContext


def _identities(ctx: AnalysisContext, groups: list[str], by: str):
    """``({group: (strategist, condition, label)}, ordered groups)``."""
    spec = ctx.condition_pairing() if by == "player_type" else None
    if spec is None:
        identity = {group: (group, "", group) for group in groups}
        order = sort_player_types(groups) if by == "player_type" else sorted(groups)
        return identity, order

    baselines = [ctx.catalog.null_label, ctx.catalog.vanilla_label]
    identity: dict[str, tuple[str, str, str]] = {}
    keys: set[str] = set()
    for group in groups:
        if group in baselines:
            identity[group] = (group, "", group)
            continue
        strategist, suffix = ctx.catalog.split_condition_suffix(group, list(spec.suffixes))
        key = "base" if not suffix else suffix
        keys.add(key)
        condition = condition_display(spec, key, ctx.catalog.vanilla_label)
        identity[group] = (str(strategist), condition, f"{strategist} | {condition}")
    strategist_rank = {
        name: i for i, name in enumerate(order_strategists(
            ctx.catalog, [v[0] for g, v in identity.items() if g not in baselines]
        ))
    }
    condition_rank = {
        name: i for i, name in enumerate(order_conditions(spec, keys, ctx.catalog.vanilla_label))
    }

    def rank(group: str):
        strategist, condition, _label = identity[group]
        if group in baselines:
            return (0, baselines.index(group), 0, "")
        return (1, strategist_rank.get(strategist, len(strategist_rank)),
                condition_rank.get(condition, len(condition_rank)), group)

    return identity, sorted(groups, key=rank)


def heatmap_rows(ctx: AnalysisContext, summary: pd.DataFrame, by: str) -> tuple[pd.DataFrame, list[str]]:
    """Add ``strategist``, ``condition``, and ``row_label``, and return the row order."""
    out = summary.copy()
    groups = [str(g) for g in dict.fromkeys(out[by].astype(str))]
    identity, ordered = _identities(ctx, groups, by)
    out["strategist"] = out[by].astype(str).map(lambda g: identity[g][0])
    out["condition"] = out[by].astype(str).map(lambda g: identity[g][1])
    out["row_label"] = out[by].astype(str).map(lambda g: identity[g][2])
    return out, [identity[g][2] for g in ordered]


def identity_parts(
    ctx: AnalysisContext, identities, by: str = "player_type"
) -> tuple[dict[str, tuple[str, str, str]], list[str]]:
    """``({identity: (strategist, condition, label)}, identities in display order)``.

    ``condition`` is blank for the baselines and whenever pairing is off.
    """
    groups = [str(g) for g in dict.fromkeys(str(v) for v in identities)]
    return _identities(ctx, groups, by)


def identity_labels(
    ctx: AnalysisContext, identities, by: str = "player_type"
) -> tuple[dict[str, str], list[str]]:
    """``({identity: label}, labels in display order)`` for bare identities."""
    identity, ordered = identity_parts(ctx, identities, by)
    return {g: parts[2] for g, parts in identity.items()}, [identity[g][2] for g in ordered]
