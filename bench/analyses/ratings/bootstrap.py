"""Shared bootstrap CIs for the MLE ratings (folded from ``ratings/bootstrap_bt.py``).

A nonparametric game bootstrap, stratified by ``experiment`` by default: each
replicate resamples games with replacement, **re-runs the parts of the strength
adjustment that are themselves re-estimated from the resampled games** (so the
adjustment's uncertainty propagates into the CIs), refits the rating model, and
collects per-identity Elo. Percentile CIs + bootstrap SE are reported around the
full-sample point estimate.

Concretely (benchmark.md / stage3 §5.1): rows the panel marks ``adjust_method ==
"civ"`` get the civ-OLS refit per replicate (it is re-estimated from the sample).
``"cell"`` rows — both implicit and explicit ``baseline_experiment`` — use the
**fixed** per-cell baseline persisted in ``cell_baseline.csv`` by the adjust stage
(computed once from the full, unfiltered panel). The cell baseline is held
constant across replicates rather than recomputed from each resample: recomputing
it from the rating-filtered resample loses the Vanilla evidence the point estimate
used (``only_llm``/``min_games`` drop reference rows) and silently diverges from
the point estimate. ``"relative"`` rows pass through. Holding it fixed keeps every
replicate scaled exactly like the point estimate, at the cost of not propagating
the (often small-``n``) baseline's own noise into the CIs.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ...adjust.strength import fit_civ_effects
from ...catalog import Catalog
from ...stats.transforms import inv_logit
from ..errors import AnalysisError


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


def readjust(
    panel: pd.DataFrame,
    params: dict,
    catalog: Catalog,
    fixed_cell_baseline: Optional[dict] = None,
) -> pd.DataFrame:
    """Recompute ``adjusted_strength`` on a resampled panel.

    The civ OLS is re-fit from the resample (it is re-estimated from the sample);
    the per-cell baseline is **fixed** — taken from ``fixed_cell_baseline`` (the
    adjust stage's persisted full-panel trail), not recomputed from the resample —
    so replicates stay scaled exactly like the point estimate (option C). Requires
    the panel's persisted ``logit_strength`` + ``adjust_method`` columns. A
    ``"cell"`` row whose fixed baseline is absent falls back to civ/relative.
    Raises if a requested civ-OLS refit fails (the caller skips the replicate).
    """
    df = panel.copy()
    if "adjust_method" not in df.columns or "logit_strength" not in df.columns:
        return df  # nothing to refit (block:"none" panels still carry adjusted_strength)
    civ_adjust = params.get("civ_adjust", "ols_logit")
    baseline_experiment = params.get("baseline_experiment")

    # Refit civ effects on the whole resampled (pre-narrowing) panel — this is the
    # SAME population the adjust stage fits on, so the refit matches the point
    # estimate (the shared fit_civ_effects, not a diverged local copy).
    civ_effects: dict[str, float] = {}
    if civ_adjust == "ols_logit":
        civ_effects, _ = fit_civ_effects(df, catalog)

    # Fixed per-cell baseline (option C): explicit keys on (seed, player_id),
    # implicit on (experiment, seed, player_id). Held constant across replicates.
    cell_base = fixed_cell_baseline or {}
    if baseline_experiment is not None:
        cell_key = lambda r: (r["seed"], r["player_id"])  # noqa: E731
    else:
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

    # Re-apply the strength stage's optional final re-normalization so replicates
    # are scaled identically to the point estimate (was previously skipped).
    if params.get("post_cell_normalize") == "relative_to_leader" and "controlled" in df.columns:
        mask = df["controlled"].astype(bool)
        if mask.any():
            gmax = df.loc[mask].groupby("game_id")["adjusted_strength"].transform("max")
            df.loc[mask, "adjusted_strength"] = df.loc[mask, "adjusted_strength"] / gmax
    return df


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
    fixed_cell_baseline: Optional[dict] = None,
    narrow: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    max_failure_rate: float = 0.5,
) -> pd.DataFrame:
    """Run ``n`` replicates and return the point estimate joined with percentile CIs.

    Each replicate resamples the **full** (problem-excluded, pre-narrowing) panel,
    re-runs the strength adjustment on it (``refit_strength`` — the civ OLS refit
    needs the full population, incl. the Vanilla reference the rating filters drop),
    optionally applies ``narrow`` (the rating-population filters — only_llm /
    min_games) *after* readjustment, then fits the rating via ``calculator`` (margin
    frozen by the caller for BT).

    Failures are counted, not silently swallowed: a warning naming the last error is
    printed to stderr, and if more than ``max_failure_rate`` of the replicates fail
    an :class:`AnalysisError` is raised rather than returning all-NaN CIs.
    """
    rows = []
    failures = 0
    last_error: Optional[Exception] = None
    for rep in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
        resampled = resample_games(panel, rng, stratified=stratified)
        try:
            if refit_strength and adjust_params is not None and catalog is not None:
                resampled = readjust(
                    resampled, adjust_params, catalog,
                    fixed_cell_baseline=fixed_cell_baseline,
                )
            if narrow is not None:
                resampled = narrow(resampled)
            rating = calculator(resampled)
        except Exception as exc:  # a degenerate/singular replicate — count, don't hide
            failures += 1
            last_error = exc
            continue
        for _, r in rating.iterrows():
            rows.append({"rep": rep, group_col: r[group_col], "elo": r["elo"]})

    if failures:
        print(
            f"WARNING: ratings bootstrap: {failures}/{n} replicate(s) failed "
            f"(last error: {last_error}).",
            file=sys.stderr,
        )
    if failures > max_failure_rate * n:
        raise AnalysisError(
            f"ratings bootstrap: {failures}/{n} replicates failed "
            f"(> {max_failure_rate:.0%}); the CIs would be unreliable. "
            f"Last error: {last_error}."
        )

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
