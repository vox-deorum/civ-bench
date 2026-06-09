"""Shared bootstrap CIs for the MLE ratings (folded from ``ratings/bootstrap_bt.py``).

A nonparametric game bootstrap, stratified by ``experiment`` by default: each
replicate resamples games with replacement, **re-runs the parts of the strength
adjustment that are themselves re-estimated from the resampled games** (so the
adjustment's uncertainty propagates into the CIs), refits the rating model, and
collects per-identity Elo. Percentile CIs + bootstrap SE are reported around the
full-sample point estimate.

Concretely (benchmark.md / stage3 §5.1): rows the panel marks ``adjust_method ==
"civ"`` get the civ-OLS refit per replicate; ``"cell"`` rows get their per-cell
Vanilla baseline recomputed from the resampled panel (falling back to civ/relative
if the resampled cell has no Vanilla evidence); ``"relative"`` rows pass through.
A quantity therefore moves the CIs exactly when it is recomputed from the sample.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ...catalog import Catalog
from ...stats.transforms import inv_logit


def compute_margin(strength_df: pd.DataFrame, rating_col: str, tie_threshold: float = 0.01) -> float:
    """Median absolute inter-type pairwise rating diff (frozen BT margin, ties excluded)."""
    diffs = []
    for _, game in strength_df.groupby("game_id", sort=False):
        s = game[rating_col].to_numpy()
        t = game["player_type"].to_numpy()
        n = len(s)
        for a in range(n - 1):
            for b in range(a + 1, n):
                if t[a] == t[b]:
                    continue
                d = abs(s[a] - s[b])
                if d >= tie_threshold:
                    diffs.append(d)
    return float(np.median(diffs)) if diffs else 0.0


def resample_games(panel: pd.DataFrame, rng: np.random.Generator, stratified: bool = True) -> pd.DataFrame:
    """Resample games with replacement (stratified by experiment by default).

    Duplicated draws are renamed to unique synthetic game_ids and the result is
    sorted by game_id (the positional slot_id convention the R input expects).
    """
    by_game = {g: sub for g, sub in panel.groupby("game_id", sort=False)}
    if stratified and "experiment" in panel.columns:
        strata = [sorted(sub["game_id"].unique()) for _, sub in panel.groupby("experiment", sort=False)]
    else:
        strata = [sorted(by_game)]

    chunks = []
    counts: dict = {}
    for gids in strata:
        sampled = rng.choice(gids, size=len(gids), replace=True)
        for orig in sampled:
            k = counts.get(orig, 0)
            counts[orig] = k + 1
            sub = by_game[orig].copy()
            sub["game_id"] = f"{orig}__rep{k}"
            chunks.append(sub)
    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values("game_id", kind="stable").reset_index(drop=True)


def _vanilla_mask(df: pd.DataFrame, catalog: Catalog) -> pd.Series:
    mask = df["player_type"] == catalog.vanilla_label
    if "model" in df.columns:
        mask = mask & df["model"].apply(catalog.is_vanilla_model)
    return mask


def readjust(panel: pd.DataFrame, params: dict, catalog: Catalog) -> pd.DataFrame:
    """Recompute ``adjusted_strength`` on a resampled panel, refitting the
    re-estimated quantities (civ OLS / cell baseline) from the resample.

    Requires the panel's persisted ``logit_strength`` + ``adjust_method`` columns.
    Rows whose original method cannot be reproduced on the resample fall back to the
    next available path (cell → civ → relative).
    """
    df = panel.copy()
    if "adjust_method" not in df.columns or "logit_strength" not in df.columns:
        return df  # nothing to refit (block:"none" panels still carry adjusted_strength)
    civ_adjust = params.get("civ_adjust", "ols_logit")
    baseline_experiment = params.get("baseline_experiment")

    # Refit civ effects on the whole resampled panel (matches the stage: fit on all rows).
    civ_effects: dict[str, float] = {}
    if civ_adjust == "ols_logit":
        civ_effects = _fit_civ_effects(df, catalog)

    # Per-cell Vanilla baseline from the resample (logit-scale mean of Vanilla rows).
    vmask = _vanilla_mask(df, catalog) & df.get("controlled", pd.Series(False, index=df.index))
    vdf = df[vmask]
    if baseline_experiment is not None:
        cell_base = vdf[vdf["experiment"] == baseline_experiment].groupby(
            ["seed", "player_id"]
        )["logit_strength"].mean().to_dict()
        cell_key = lambda r: (r["seed"], r["player_id"])  # noqa: E731
    else:
        cell_base = vdf.groupby(["experiment", "seed", "player_id"])["logit_strength"].mean().to_dict()
        cell_key = lambda r: (r["experiment"], r["seed"], r["player_id"])  # noqa: E731

    adjusted = np.empty(len(df))
    logit = df["logit_strength"].to_numpy()
    for pos, (_, row) in enumerate(df.iterrows()):
        method = row["adjust_method"]
        if method == "cell":
            base = cell_base.get(cell_key(row), np.nan)
            if np.isfinite(base):
                adjusted[pos] = inv_logit(logit[pos] - base)
                continue
            method = "civ" if civ_adjust == "ols_logit" else "relative"
        if method == "civ" and civ_adjust == "ols_logit":
            adjusted[pos] = inv_logit(logit[pos] - civ_effects.get(row["civilization"], 0.0))
        else:
            adjusted[pos] = row.get("relative_strength", inv_logit(logit[pos]))
    df["adjusted_strength"] = adjusted
    return df


def _fit_civ_effects(df: pd.DataFrame, catalog: Catalog) -> dict[str, float]:
    from statsmodels.formula.api import ols

    vanilla = catalog.vanilla_label
    formula = (
        "logit_strength ~ C(civilization, Sum) "
        f'+ C(player_type, Treatment(reference="{vanilla}"))'
    )
    try:
        fit = ols(formula, data=df).fit()
    except Exception:
        return {}
    effects: dict[str, float] = {}
    for var in fit.params.index:
        if "C(civilization, Sum)[S." in var:
            effects[var.split("[S.", 1)[1].rstrip("]")] = float(fit.params[var])
    omitted = set(df["civilization"].dropna().unique()) - set(effects)
    for civ in omitted:
        effects[civ] = -sum(effects.values())
    return effects


def run_bootstrap(
    panel: pd.DataFrame,
    point_df: pd.DataFrame,
    calculator: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    group_col: str,
    n: int,
    seed: int,
    ci_level: float = 0.95,
    stratified: bool = True,
    adjust_params: Optional[dict] = None,
    catalog: Optional[Catalog] = None,
    refit_strength: bool = True,
) -> pd.DataFrame:
    """Run ``n`` replicates and return the point estimate joined with percentile CIs.

    ``calculator(resampled_panel) -> rating_df`` performs the rating fit (margin
    frozen by the caller for BT). ``refit_strength`` re-runs the strength
    adjustment per replicate (recommended); set False to bootstrap the frozen
    ``adjusted_strength`` directly.
    """
    rows = []
    for rep in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
        resampled = resample_games(panel, rng, stratified=stratified)
        if refit_strength and adjust_params is not None and catalog is not None:
            resampled = readjust(resampled, adjust_params, catalog)
        try:
            rating = calculator(resampled)
        except Exception:
            continue
        for _, r in rating.iterrows():
            rows.append({"rep": rep, group_col: r[group_col], "elo": r["elo"]})

    reps = pd.DataFrame(rows)
    alpha = (1.0 - ci_level) / 2.0
    point = point_df.rename(columns={"elo": "point_elo"})
    if reps.empty:
        for col in ("ci_lower", "ci_upper", "boot_se_elo", "n_valid"):
            point[col] = np.nan
        return point
    grouped = reps.groupby(group_col)["elo"]
    agg = pd.DataFrame({
        "ci_lower": grouped.quantile(alpha),
        "ci_upper": grouped.quantile(1.0 - alpha),
        "boot_se_elo": grouped.std(ddof=1),
        "n_valid": grouped.size(),
    }).reset_index()
    return point.merge(agg, on=group_col, how="left")
