"""Shared token usage aggregation and usage figures.

The helpers in this module operate on data frames supplied by the caller.  They
do not load or save report artifacts, which lets the usage and cost analyses
share the same accounting rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bench.analyses.errors import AnalysisError


CACHE_COLUMNS = ("cached_input_tokens", "repeated_input_tokens")


def cost_note(cached_input: bool, reported_only: list[str] = ()) -> str:
    """Describe how costs treat cached input.

    ``reported_only`` lists models whose ``cache_pricing`` family sets
    ``"estimate": false``.
    """
    if not cached_input:
        return (
            "Costs do not account for cached tokens or cache discounts. "
            "Output token averages include reasoning tokens."
        )
    reported = (
        f"{', '.join(reported_only)} {'uses' if len(reported_only) == 1 else 'use'} "
        "only provider-reported cache counts. "
        if reported_only else ""
    )
    return (
        "Cached input may be estimated. "
        f"{reported}Output token averages include reasoning tokens."
    )


def cache_accounting(models, catalog, enabled: bool) -> tuple[bool, list[str]]:
    """Return whether any of ``models`` is cache-priced, and the reported-only ones."""
    if not enabled:
        return False, []
    policies = {model: catalog.cache_policy(model) for model in models}
    active = any(policies.values())
    reported_only = sorted(model for model, policy in policies.items() if policy and not policy[1])
    return active, reported_only


def _complete_sum(s: pd.Series) -> float:
    """Return a sum only when every observation has telemetry."""
    return float(s.sum()) if s.notna().all() else float("nan")


def reasoning_estimate_note(models: list[str]) -> str:
    """Describe which models have estimated reasoning tokens, or return ''."""
    if not models:
        return ""
    names = ", ".join(models)
    return (
        f"Reasoning tokens for {names} are estimated from the model's calls "
        "that recorded reasoning, scaled by output tokens."
    )


def _estimate_missing_reasoning(df: pd.DataFrame, catalog) -> pd.Series:
    """Fill in reasoning for flagged models; return a per-row estimated flag.

    Some providers report reasoning on only part of a model's calls. For a model
    with ``estimate_reasoning`` set in the catalog, the reasoning-per-output ratio
    is pooled over every call whose reasoning looks fully reported (see
    ``RECORDED_REASONING_MIN_RATIO`` in the token extractor). The other calls'
    reasoning is replaced by that ratio times their output.
    """
    estimated = pd.Series(False, index=df.index)
    flagged = catalog.reasoning_estimate_models()
    present = sorted(set(df["model"]) & flagged)
    if not present:
        return estimated
    columns = ["reasoning_recorded_tokens", "reasoning_recorded_output_tokens"]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise AnalysisError(
            f"Reasoning estimates for {', '.join(present)} need the {missing} "
            "columns in the tokens table. Re-run `civ-bench extract` to rebuild it."
        )
    reasoning = pd.to_numeric(df["reasoning_tokens"], errors="coerce").astype(float)
    output = pd.to_numeric(df["output_tokens"], errors="coerce")
    recorded = pd.to_numeric(df[columns[0]], errors="coerce").fillna(0)
    recorded_output = pd.to_numeric(df[columns[1]], errors="coerce").fillna(0)
    missing_output = (output - recorded_output).clip(lower=0).fillna(0)
    df["reasoning_tokens"] = reasoning
    for model in present:
        rows = df["model"] == model
        denominator = recorded_output[rows].sum()
        if denominator <= 0:
            continue
        ratio = recorded[rows].sum() / denominator
        df.loc[rows, "reasoning_tokens"] = recorded[rows] + ratio * missing_output[rows]
        estimated |= rows & (missing_output > 0)
    return estimated


def _cached_input(df: pd.DataFrame, catalog) -> tuple[pd.Series, pd.Series]:
    """Return per-row cached input tokens and cache-read price ratios.

    Cached input is the larger of the provider-reported cache reads and the
    prompt overlap between consecutive steps, capped at the row's input. Models
    whose ``cache_pricing`` family sets ``"estimate": false`` use only the
    reported count. Without ``cache_pricing`` in the catalog, nothing is cached.
    """
    policies = {model: catalog.cache_policy(model) for model in df["model"].unique()}
    if not any(policies.values()):
        return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
    missing = [column for column in CACHE_COLUMNS if column not in df.columns]
    if missing:
        raise AnalysisError(
            f"Cached-input estimates need the {missing} columns in the tokens table. "
            "Re-run `civ-bench extract` to rebuild it, or set "
            '"token_estimates": {"cached_input": false} in the run spec.'
        )
    ratio = df["model"].map(lambda model: policies[model][0] if policies[model] else 0.0)
    estimate = df["model"].map(lambda model: bool(policies[model] and policies[model][1]))
    enabled = df["model"].map(lambda model: policies[model] is not None)
    reported = pd.to_numeric(df["cached_input_tokens"], errors="coerce").fillna(0)
    repeated = pd.to_numeric(df["repeated_input_tokens"], errors="coerce").fillna(0)
    cached = reported.where(~estimate, np.maximum(reported, repeated))
    cached = np.minimum(cached, df["input_tokens"]).where(enabled, 0.0)
    return cached.astype(float), ratio.astype(float)


def compute_game_costs(
    tokens_df: pd.DataFrame,
    catalog,
    *,
    estimate_reasoning: bool = True,
    cached_input: bool = True,
) -> pd.DataFrame:
    """Price token rows and aggregate them to model/player/game records.

    ``estimate_reasoning`` fills in missing reasoning for flagged models, and
    ``cached_input`` prices cached input at the cache-read rate. Both mirror the
    run spec's root ``token_estimates`` switches.
    """
    pricing = catalog.pricing_per_million()
    df = tokens_df.copy()
    names = df["model_name"].where(df["model_name"].notna(), "Unattributed")
    df["model"] = names.astype(str).apply(catalog.canonicalize_model_name)
    if estimate_reasoning:
        df["reasoning_estimated"] = _estimate_missing_reasoning(df, catalog)
    else:
        df["reasoning_estimated"] = False

    def numeric_column(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    df["combined_output_tokens"] = (
        numeric_column("reasoning_tokens").fillna(0)
        + numeric_column("output_tokens").fillna(0)
    )
    df["input_tokens"] = numeric_column("input_tokens")
    if cached_input:
        df["cached_input_tokens"], read_ratio = _cached_input(df, catalog)
    else:
        df["cached_input_tokens"], read_ratio = 0.0, 0.0
    # Keep missing input telemetry missing so the aggregate stays incomplete.
    df["cached_input_tokens"] = df["cached_input_tokens"].where(df["input_tokens"].notna())
    df["input_per_million"] = df["model"].map(
        lambda model: pricing.get(model, {}).get("input_per_million")
    )
    df["output_per_million"] = df["model"].map(
        lambda model: pricing.get(model, {}).get("output_per_million")
    )
    uncached = df["input_tokens"] - df["cached_input_tokens"]
    df["row_cost"] = (
        (uncached + df["cached_input_tokens"] * read_ratio) / 1_000_000 * df["input_per_million"]
        + df["combined_output_tokens"] / 1_000_000 * df["output_per_million"]
    )
    identity_cols = ["game_id"]
    for column in ("player_id", "player_type"):
        if column in df.columns:
            identity_cols.append(column)
    identity_cols.append("model")
    per_game = df.groupby(identity_cols, as_index=False).agg(
        input_tokens=("input_tokens", _complete_sum),
        cached_input_tokens=("cached_input_tokens", _complete_sum),
        combined_output_tokens=("combined_output_tokens", _complete_sum),
        total_cost=("row_cost", _complete_sum),
        input_per_million=("input_per_million", "first"),
        output_per_million=("output_per_million", "first"),
        reasoning_estimated=("reasoning_estimated", "any"),
    )
    per_game["token_available"] = per_game[
        ["input_tokens", "combined_output_tokens"]
    ].notna().all(axis=1)
    per_game["cost_available"] = per_game[
        ["input_tokens", "combined_output_tokens", "total_cost"]
    ].notna().all(axis=1)
    # Keep the old name for callers that used it as a cost-completeness flag.
    per_game["available"] = per_game["cost_available"]
    return per_game


def summarize_game_costs(
    per_identity_game: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Summarize token and cost completeness independently.

    Token means use complete token telemetry even when a model has no price.
    Cost means and cost completeness use only rows with known pricing.  The
    ``complete_player_games`` and ``na_player_games`` aliases retain their
    historical cost-completeness meaning; explicit token and cost columns are
    included for consumers that need to distinguish the two.
    """
    work = per_identity_game.copy()
    identity_cols = ["game_id"]
    for column in ("player_id", "player_type"):
        if column in work.columns and column not in group_cols:
            identity_cols.append(column)
    keys = [*identity_cols, *group_cols]
    if "cached_input_tokens" not in work.columns:
        work["cached_input_tokens"] = 0.0
    per_game = work.groupby(keys, as_index=False).agg(
        input_tokens=("input_tokens", _complete_sum),
        cached_input_tokens=("cached_input_tokens", _complete_sum),
        combined_output_tokens=("combined_output_tokens", _complete_sum),
        total_cost=("total_cost", _complete_sum),
    )
    per_game["token_available"] = per_game[["input_tokens", "combined_output_tokens"]].notna().all(axis=1)
    per_game["cost_available"] = per_game[["input_tokens", "combined_output_tokens", "total_cost"]].notna().all(axis=1)

    token_valid = per_game[per_game["token_available"]]
    cost_valid = per_game[per_game["cost_available"]]
    game_status = per_game.groupby(["game_id", *group_cols], as_index=False).agg(
        token_available=("token_available", "all"),
        cost_available=("cost_available", "all"),
    )
    games = game_status.groupby(group_cols, as_index=False).agg(
        games=("game_id", "nunique"),
        na_games=("cost_available", lambda values: int((~values).sum())),
        token_na_games=("token_available", lambda values: int((~values).sum())),
    )
    token_av = token_valid.groupby(group_cols, as_index=False).agg(
        avg_input=("input_tokens", "mean"),
        avg_cached_input=("cached_input_tokens", "mean"),
        avg_output=("combined_output_tokens", "mean"),
    )
    cost_av = cost_valid.groupby(group_cols, as_index=False).agg(
        total_cost=("total_cost", "sum"),
        avg_cost_per_player_game=("total_cost", "mean"),
    )
    summary = games.merge(token_av, on=group_cols, how="left").merge(cost_av, on=group_cols, how="left")
    counts = per_game.groupby(group_cols, as_index=False).agg(
        player_games=("game_id", "size"),
        token_complete_player_games=("token_available", "sum"),
        cost_complete_player_games=("cost_available", "sum"),
    )
    summary = summary.merge(counts, on=group_cols, how="left")
    summary["token_complete_player_games"] = summary["token_complete_player_games"].astype(int)
    summary["cost_complete_player_games"] = summary["cost_complete_player_games"].astype(int)
    summary["token_na_player_games"] = summary["player_games"] - summary["token_complete_player_games"]
    summary["cost_na_player_games"] = summary["player_games"] - summary["cost_complete_player_games"]
    summary["complete_player_games"] = summary["cost_complete_player_games"]
    summary["na_player_games"] = summary["cost_na_player_games"]
    summary["complete_games"] = summary["games"] - summary["na_games"]
    summary["avg_cost_per_game"] = summary["avg_cost_per_player_game"]
    if "model" in group_cols:
        prices = work.drop_duplicates("model").set_index("model")[["input_per_million", "output_per_million"]]
        summary = summary.merge(prices, left_on="model", right_index=True, how="left")
    return summary.sort_values("total_cost", ascending=False, na_position="last", kind="stable").reset_index(drop=True)


def _row_order(table: pd.DataFrame, ratings: pd.DataFrame | None, value_col: str) -> list[str]:
    ids = table["player_type"].astype(str).tolist()
    if ratings is not None and {"player_type", "elo"} <= set(ratings.columns):
        rated = ratings.copy()
        rated["player_type"] = rated["player_type"].astype(str)
        rated = rated[rated["player_type"].isin(ids)].sort_values("elo", ascending=False, kind="stable")
        order = rated["player_type"].tolist()
    else:
        order = []
    fallback = table[~table["player_type"].astype(str).isin(order)].sort_values(
        value_col, ascending=False, na_position="last", kind="stable"
    )["player_type"].astype(str).tolist()
    return list(dict.fromkeys(order + fallback))


def plot_usage_figures(
    table: pd.DataFrame,
    ctx,
    ratings: pd.DataFrame | None = None,
    *,
    currency: str = "usd",
) -> dict[str, object]:
    """Create cost, input-token, and output-token figures from a summary table."""
    import matplotlib.pyplot as plt
    from bench.plotting.styles import get_player_color

    baseline = {ctx.catalog.vanilla_label, ctx.catalog.null_label}
    table = table[~table["player_type"].astype(str).isin(baseline)].copy()
    if table.empty:
        return {}
    fields = {
        "cost": ("avg_cost_per_player_game", f"Average cost per player per game ({str(currency).upper()})", "Estimated cost by player type"),
        "input_tokens": ("avg_input", "Average input tokens per player-game", "Average input tokens by player type"),
        "output_tokens": ("avg_output", "Average output tokens per player-game (including reasoning)", "Average output tokens by player type"),
    }
    pairing = ctx.condition_pairing()
    figures = {}
    order = _row_order(table, ratings, fields["cost"][0])
    if pairing is not None:
        from bench.plotting.pairing import (
            attach_pair_columns,
            paired_sort_order,
            plot_paired_rows,
        )

        paired_table = attach_pair_columns(table, ctx.catalog, pairing, "player_type")
        usage_order = paired_sort_order(
            paired_table, pairing, fields["cost"][0], ascending=False
        )
        if ratings is not None and {"player_type", "elo"} <= set(ratings.columns):
            paired_ratings = attach_pair_columns(
                ratings, ctx.catalog, pairing, "player_type"
            )
            rating_order = paired_sort_order(
                paired_ratings, pairing, "elo", ascending=False
            )
            available = set(paired_table["base_identity"].astype(str))
            order = [identity for identity in rating_order if identity in available]
            order.extend(identity for identity in usage_order if identity not in order)
        else:
            order = usage_order
        for name, (column, xlabel, title) in fields.items():
            figures[name] = plot_paired_rows(
                table, catalog=ctx.catalog, spec=pairing, value_col=column,
                identity_col="player_type", row_order=order, ascending=False,
                xlabel=xlabel, title=title,
            )
        return {key: fig for key, fig in figures.items() if fig is not None}
    for name, (column, xlabel, title) in fields.items():
        work = table.set_index("player_type").reindex(order)
        work = work[pd.to_numeric(work[column], errors="coerce").notna()]
        if work.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(work) + 1.5)))
        ax.barh(work.index[::-1], work[column].astype(float).to_numpy()[::-1],
                color=[get_player_color(ctx.catalog, str(v)) for v in work.index[::-1]])
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        figures[name] = fig
    return figures
