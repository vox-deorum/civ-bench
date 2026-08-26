"""Strength-panel derivation (the one place ``prepare_strength_data`` lives now).

Ported and consolidated from ``../vox-deorum-analysis/performance/turn_predicted.ipynb``
(also copied into ``ratings/iterative_bt.py`` in the old repo — both are replaced
by this module). Turns an estimator's per-turn ``predicted_win_probability`` into
a per-player-game ``adjusted_strength`` panel:

    late-game weighted average (turn_progress_min, weight)
      → optional relative-to-leader (relative_to="game_leader"; unset/"none" ⇒ raw P(win))
      → winner enforcement (enforce_winner)
      → finite logit_strength = logit(clip(relative_strength, eps, 1-eps))
      → adjustment: civ OLS (uncontrolled) OR matched start-cell baseline (controlled)

Parity notes (so ``block:"none"`` reproduces the legacy per-row values):
  * the estimator's ``predictions.csv`` carries an UNROUNDED ``turn_progress``; the
    legacy pipeline groups/weights on ``round(turn/max_turn, 2)`` — we re-round here.
  * the ``turn_progress_min`` filter is a strict ``>``.
  * ``eps = 1e-5`` (``bench.stats.transforms.LOGIT_EPS``) matches the legacy clip.
  * the legacy Step-6 non-LLM game filter is intentionally NOT applied — the panel
    keeps every experiment so ``ref="Vanilla"`` / the baseline pathways resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..catalog import Catalog
from ..stats.transforms import inv_logit, logit
from .errors import AdjustError


# Coded defaults for unlisted params (benchmark.md §5.1).
DEFAULT_PARAMS = {
    "turn_progress_min": 0.2,
    "weight": "turn_progress",
    # None ("unset") ⇒ no leader normalization (strength is the raw P(win)); set
    # "game_leader" to normalize each seat to its game's strongest seat.
    "relative_to": None,
    "enforce_winner": True,
    "civ_adjust": "ols_logit",
    "block": "auto",
    "baseline_experiment": None,
    "post_cell_normalize": "none",
    # None ⇒ keep every controlled condition; a number in (0, 1] drops conditions
    # whose occupied (seed, rotation) slot fraction is below the threshold (1.0 ⇒
    # drop every condition missing any slot).
    "min_condition_completeness": None,
}

# The persisted panel column contract (Done criterion). The first columns are the
# ratings.* contract; the rest are pass-throughs for the controlled-design report.
PANEL_COLUMNS = [
    "experiment", "game_id", "player_id", "player_type", "civilization",
    "seed", "seating_rotation", "config_slot", "controlled", "is_winner",
    "weighted_strength", "relative_strength", "logit_strength", "adjusted_strength",
    "adjust_method", "cell_logit_advantage",
]
CIV_EFFECTS_COLUMNS = ["civilization", "civ_effect", "n_rows"]
CELL_BASELINE_COLUMNS = [
    "experiment", "pathway", "seed", "player_id", "civilization",
    "cell_baseline", "n_vanilla", "win_rate", "n_games", "n_models",
    "has_vanilla_baseline", "vanilla_connected",
]
CELL_COVERAGE_COLUMNS = [
    "experiment", "seed", "player_id", "civilization",
    "in_entirety", "n_rows", "n_vanilla", "has_baseline", "missing",
]


@dataclass
class StrengthArtifacts:
    """Everything the runner persists: the panel + always-written audit trails."""

    panel: pd.DataFrame
    civ_effects: pd.DataFrame
    cell_baseline: pd.DataFrame
    cell_coverage: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    estimator_id: str = ""


# ── param resolution ─────────────────────────────────────────────────────────
def _resolve_params(params: dict | None) -> dict:
    out = dict(DEFAULT_PARAMS)
    out.update(params or {})
    return out


# ── step 1-4: weighted → relative → enforce → logit (legacy parity) ──────────
def _weighted_strength(pred: pd.DataFrame, turn_progress_min: float, weight_mode: str) -> pd.DataFrame:
    df = pred.copy()
    # Parity: re-round turn_progress to 2dp (legacy load_turn_data overwrote the
    # column with round(turn/max_turn, 2)); the predictions CSV carries it unrounded.
    df["turn_progress"] = (df["turn"] / df["max_turn"]).round(2)
    df = df[df["turn_progress"] > turn_progress_min].copy()

    grouped = (
        df.groupby(["game_id", "player_id", "turn_progress"])
        .agg(
            predicted_win_probability=("predicted_win_probability", "mean"),
            experiment=("experiment", "first"),
            is_winner=("is_winner", "last"),
            civilization=("civilization", "first"),
        )
        .reset_index()
    )
    grouped["weight"] = 1.0 if weight_mode == "uniform" else grouped["turn_progress"]

    records = []
    for (game_id, player_id), grp in grouped.groupby(["game_id", "player_id"]):
        w = grp["weight"]
        p = grp["predicted_win_probability"]
        weighted_avg = (w * p).sum() / w.sum()
        records.append({
            "game_id": game_id,
            "player_id": player_id,
            "experiment": grp["experiment"].iloc[0],
            "civilization": grp["civilization"].iloc[0],
            "weighted_strength": weighted_avg,
            "is_winner": grp["is_winner"].iloc[-1],
        })
    return pd.DataFrame.from_records(records)


def _relative_and_logit(
    strength_df: pd.DataFrame, enforce_winner: bool, relative_to: str | None
) -> pd.DataFrame:
    # game_max is needed in both modes: the game_leader path divides by it, and the
    # absolute path uses it for the raw-scale winner bump.
    game_max = (
        strength_df.groupby("game_id")["weighted_strength"].max().rename("max_weighted_strength")
    )
    strength_df = strength_df.merge(game_max, on="game_id")

    if relative_to == "game_leader":
        # Leader-relative: each seat's strength as a fraction of its game's strongest seat.
        strength_df["relative_strength"] = (
            strength_df["weighted_strength"] / strength_df["max_weighted_strength"]
        )
        winner_mask = (strength_df["is_winner"] == 1) & (strength_df["relative_strength"] < 1.0)
        if enforce_winner:
            strength_df.loc[winner_mask, "weighted_strength"] = (
                strength_df.loc[winner_mask, "max_weighted_strength"] + 0.001
            )
            strength_df.loc[winner_mask, "relative_strength"] = 1.0
    else:
        # Absolute (relative_to unset/"none"): strength is the raw late-game P(win); the
        # leader is NOT relied upon. enforce_winner still guarantees the actual winner holds
        # the top raw strength. relative_strength mirrors weighted_strength so the panel
        # contract and the relative_strength fallbacks keep using the raw (not relative) value.
        winner_mask = (strength_df["is_winner"] == 1) & (
            strength_df["weighted_strength"] < strength_df["max_weighted_strength"]
        )
        if enforce_winner:
            strength_df.loc[winner_mask, "weighted_strength"] = (
                strength_df.loc[winner_mask, "max_weighted_strength"] + 0.001
            )
        strength_df["relative_strength"] = strength_df["weighted_strength"]

    # Audit: relative_strength stays 1.0 for enforced winners (game_leader), but the
    # logit-scale fit/adjustment always uses the clipped value (eps=1e-5) so it is finite.
    strength_df["logit_strength"] = logit(strength_df["relative_strength"].to_numpy())
    return strength_df.drop(columns=["max_weighted_strength"])


# ── identity join (predictions carry no composed player_type) ────────────────
def _join_identity(
    strength_df: pd.DataFrame,
    panel: pd.DataFrame,
    games: pd.DataFrame,
    catalog: Catalog,
) -> pd.DataFrame:
    panel_cols = ["game_id", "player_id", "player_type", "config_slot", "model", "strategist", "civilization"]
    have = [c for c in panel_cols if c in panel.columns]
    keys = panel[have].rename(columns={"civilization": "_civ_panel"})
    merged = strength_df.merge(keys, on=["game_id", "player_id"], how="left")

    # civilization from predictions, panel as fallback (§5.1).
    if "_civ_panel" in merged.columns:
        merged["civilization"] = merged["civilization"].fillna(merged["_civ_panel"])
        merged = merged.drop(columns=["_civ_panel"])
    if "config_slot" in merged.columns:
        merged["config_slot"] = merged["config_slot"].fillna(-1).astype(int)
    else:
        merged["config_slot"] = -1
    if "player_type" not in merged.columns:
        merged["player_type"] = "Player " + merged["player_id"].astype(str)
    merged["player_type"] = merged["player_type"].fillna("Player " + merged["player_id"].astype(str))

    game_cols = [c for c in ("game_id", "seed", "seating_rotation") if c in games.columns]
    gkeys = games[game_cols].drop_duplicates("game_id")
    merged = merged.merge(gkeys, on="game_id", how="left")
    for col in ("seed", "seating_rotation"):
        if col not in merged.columns:
            merged[col] = -1
        merged[col] = merged[col].fillna(-1).astype(int)

    merged["controlled"] = (merged["seed"] != -1) & (merged["seating_rotation"] != -1)
    return merged


# ── uncontrolled path: OLS civilization effects (logit scale) ────────────────
def fit_civ_effects(df: pd.DataFrame, catalog: Catalog) -> tuple[dict, pd.DataFrame]:
    """Fit `logit_strength ~ C(civilization, Sum) + C(player_type, Treatment(ref))` on ALL rows.

    Returns (civ_effects map, civ_effects_df). The omitted Sum-coded level is
    discovered from the data (all civilizations − parsed terms) and set to
    −sum(observed effects), computed ONCE before assignment (so ≥2 omitted civs
    all get the same, correct value); no hardcoded omitted civ.

    Shared with the ratings bootstrap (``analyses.ratings.bootstrap.readjust``) so a
    replicate's civ refit is byte-identical to the adjust stage's — the bootstrap's
    previous private copy diverged (it recomputed −sum inside the omitted loop while
    mutating the dict, mis-scaling every omitted civ after the first).
    """
    from statsmodels.formula.api import ols

    vanilla = catalog.vanilla_label
    formula = (
        "logit_strength ~ C(civilization, Sum) "
        f'+ C(player_type, Treatment(reference="{vanilla}"))'
    )
    try:
        fit = ols(formula, data=df).fit()
    except Exception as exc:  # patsy/statsmodels — surface loudly with context
        raise AdjustError(
            f"strength: OLS civilization fit failed ({exc}). The fit needs the "
            f"vanilla reference player_type '{vanilla}' present in the panel and a "
            f"civilization column with variation."
        ) from exc

    params = fit.params
    civ_effects: dict[str, float] = {}
    for var in params.index:
        if "C(civilization, Sum)[S." in var:
            name = var.split("[S.", 1)[1].rstrip("]")
            civ_effects[name] = float(params[var])

    all_civs = set(df["civilization"].dropna().unique())
    omitted = all_civs - set(civ_effects)
    omitted_effect = -sum(civ_effects.values())
    for civ in omitted:
        civ_effects[civ] = omitted_effect

    counts = df["civilization"].value_counts()
    rows = [
        {"civilization": civ, "civ_effect": eff, "n_rows": int(counts.get(civ, 0))}
        for civ, eff in sorted(civ_effects.items())
    ]
    return civ_effects, pd.DataFrame(rows, columns=CIV_EFFECTS_COLUMNS)


# ── controlled path: matched final-seat-cell Vanilla VPAI baseline ───────────
def _vanilla_baseline_mask(df: pd.DataFrame, catalog: Catalog) -> pd.Series:
    """Rows that count as Vanilla VPAI baseline evidence (§5.1).

    Narrow by design: player_type == vanilla_label AND (when a raw ``model``
    column is present) the model is the VPAI engine. ``Null`` is never evidence.
    """
    mask = df["player_type"] == catalog.vanilla_label
    if "model" in df.columns:
        mask = mask & df["model"].apply(catalog.is_vanilla_model)
    return mask


def _types_connected_to_vanilla(scope_df: pd.DataFrame, vanilla: str) -> set[str]:
    """Union-find: player_types reachable from `vanilla` via game co-occurrence."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for _, grp in scope_df.groupby("game_id"):
        types = list(grp["player_type"].dropna().unique())
        for t in types[1:]:
            ra, rb = find(types[0]), find(t)
            if ra != rb:
                parent[ra] = rb
    if vanilla not in parent:
        return set()
    root = find(vanilla)
    return {t for t in parent if find(t) == root}


def _seat_civ(rows: pd.DataFrame) -> str:
    civs = rows["civilization"].dropna()
    return str(civs.iloc[0]) if len(civs) else ""


def _cell_baseline_row(
    experiment: str,
    pathway: str,
    seed: int,
    player_id: int,
    vanilla_grp: pd.DataFrame,
    cell_rows: pd.DataFrame,
    connected: set[str],
    vanilla: str,
) -> dict:
    cell_types = set(cell_rows["player_type"].dropna().unique())
    nonvanilla = cell_types - {vanilla}
    vanilla_connected = (not nonvanilla) or any(t in connected for t in nonvanilla)
    return {
        "experiment": experiment,
        "pathway": pathway,
        "seed": int(seed),
        "player_id": int(player_id),
        "civilization": _seat_civ(cell_rows if len(cell_rows) else vanilla_grp),
        "cell_baseline": float(vanilla_grp["logit_strength"].mean()),
        "n_vanilla": int(len(vanilla_grp)),
        "win_rate": float(vanilla_grp["is_winner"].mean()),
        "n_games": int(cell_rows["game_id"].nunique()) if len(cell_rows) else int(vanilla_grp["game_id"].nunique()),
        "n_models": int(cell_rows["config_slot"].nunique()) if len(cell_rows) else int(vanilla_grp["config_slot"].nunique()),
        "has_vanilla_baseline": bool(len(vanilla_grp) >= 1),
        "vanilla_connected": bool(vanilla_connected),
    }


def _build_cell_baseline(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    baseline_experiment: str | None,
    catalog: Catalog,
) -> pd.DataFrame:
    """All cell-baseline rows for every pathway that actually ran (implicit always;
    explicit only when a baseline_experiment is set)."""
    vanilla = catalog.vanilla_label
    rows: list[dict] = []

    # Implicit: per controlled experiment × (seed, player_id) wherever Vanilla rows exist.
    connected_by_exp: dict[str, set[str]] = {}
    for exp, vgrp in baseline_df.groupby("experiment"):
        scope = df[df["experiment"] == exp]
        connected = connected_by_exp.setdefault(exp, _types_connected_to_vanilla(scope, vanilla))
        for (seed, pid), cell_vanilla in vgrp.groupby(["seed", "player_id"]):
            cell_rows = scope[(scope["seed"] == seed) & (scope["player_id"] == pid)]
            rows.append(_cell_baseline_row(
                exp, "implicit", seed, pid, cell_vanilla, cell_rows, connected, vanilla
            ))

    # Explicit: the designated baseline experiment's cells (one per seed, player_id).
    if baseline_experiment is not None:
        edf = baseline_df[baseline_df["experiment"] == baseline_experiment]
        scope = df[df["experiment"] == baseline_experiment]
        connected = _types_connected_to_vanilla(scope, vanilla)
        for (seed, pid), cell_vanilla in edf.groupby(["seed", "player_id"]):
            cell_rows = scope[(scope["seed"] == seed) & (scope["player_id"] == pid)]
            rows.append(_cell_baseline_row(
                baseline_experiment, "explicit", seed, pid, cell_vanilla, cell_rows, connected, vanilla
            ))

    return pd.DataFrame(rows, columns=CELL_BASELINE_COLUMNS)


def _selected_baseline_maps(
    baseline_df: pd.DataFrame, baseline_experiment: str | None
) -> tuple[str, dict]:
    """(selected pathway name, cell→baseline map) that feeds adjusted_strength."""
    if baseline_experiment is not None:
        edf = baseline_df[baseline_df["experiment"] == baseline_experiment]
        sel = edf.groupby(["seed", "player_id"])["logit_strength"].mean().to_dict()
        return "explicit", sel
    sel = baseline_df.groupby(["experiment", "seed", "player_id"])["logit_strength"].mean().to_dict()
    return "implicit", sel


def _cell_baseline_per_row(
    df: pd.DataFrame, selected: str, sel_map: dict
) -> pd.Series:
    if selected == "explicit":
        keys = list(zip(df["seed"], df["player_id"]))
    else:
        keys = list(zip(df["experiment"], df["seed"], df["player_id"]))
    values = [sel_map.get(k, np.nan) for k in keys]
    return pd.Series(values, index=df.index)


def _build_cell_coverage(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    baseline_experiment: str | None,
    selected: str,
    sel_map: dict,
    catalog: Catalog,
) -> pd.DataFrame:
    """Per controlled experiment, which (seed, player_id) cells of the entirety grid
    it does not cover. Report-only completeness diagnostic (never fatal)."""
    vanilla = catalog.vanilla_label
    controlled = df[df["controlled"]]
    if controlled.empty:
        return pd.DataFrame(columns=CELL_COVERAGE_COLUMNS)

    # The "entirety" reference grid: the baseline experiment's cells (explicit) or
    # the union of (seed, player_id) cells across the controlled subset (implicit).
    if baseline_experiment is not None:
        grid_src = controlled[controlled["experiment"] == baseline_experiment]
    else:
        grid_src = controlled
    entirety = list(grid_src.groupby(["seed", "player_id"]).groups.keys())

    seat_civ = {
        (seed, pid): _seat_civ(grp)
        for (seed, pid), grp in controlled.groupby(["seed", "player_id"])
    }
    vanilla_mask = _vanilla_baseline_mask(controlled, catalog)

    rows: list[dict] = []
    for exp in sorted(controlled["experiment"].unique()):
        edf = controlled[controlled["experiment"] == exp]
        evan = controlled[vanilla_mask & (controlled["experiment"] == exp)]
        for (seed, pid) in entirety:
            sub = edf[(edf["seed"] == seed) & (edf["player_id"] == pid)]
            n_vanilla = int(len(evan[(evan["seed"] == seed) & (evan["player_id"] == pid)]))
            if selected == "explicit":
                has_baseline = (seed, pid) in sel_map
            else:
                has_baseline = (exp, seed, pid) in sel_map
            rows.append({
                "experiment": exp,
                "seed": int(seed),
                "player_id": int(pid),
                "civilization": seat_civ.get((seed, pid), ""),
                "in_entirety": True,
                "n_rows": int(len(sub)),
                "n_vanilla": n_vanilla,
                "has_baseline": bool(has_baseline),
                "missing": bool(len(sub) == 0),
            })
    return pd.DataFrame(rows, columns=CELL_COVERAGE_COLUMNS)


# ── parameterized condition completeness ─────────────────────────────────────
def _reference_slots(
    df: pd.DataFrame, baseline_experiment: str | None
) -> list[tuple[int, int]]:
    """The controlled reference grid: the ``(seed, seating_rotation)`` slots a
    condition must occupy to be complete. Mirrors the report-side semantics
    (``performance.experiment_completeness``): the explicit baseline experiment's
    grid when set, else the union across the controlled subset."""
    controlled = df[df["controlled"]]
    if baseline_experiment is not None:
        explicit = controlled[controlled["experiment"] == str(baseline_experiment)]
        if not explicit.empty:
            controlled = explicit
    seeds = sorted(controlled["seed"].dropna().astype(int).unique())
    rotations = sorted(controlled["seating_rotation"].dropna().astype(int).unique())
    return [(int(seed), int(rot)) for seed in seeds for rot in rotations]


def _condition_completeness(
    df: pd.DataFrame, baseline_experiment: str | None
) -> dict[str, float]:
    """Per controlled experiment, the fraction of reference ``(seed, rotation)``
    slots it occupies. An experiment missing any reference slot scores below 1.0;
    an absent reference grid yields ``{}``."""
    reference = set(_reference_slots(df, baseline_experiment))
    if not reference:
        return {}
    out: dict[str, float] = {}
    for exp, grp in df[df["controlled"]].groupby("experiment"):
        present = {
            (int(seed), int(rot))
            for seed, rot in grp[["seed", "seating_rotation"]].itertuples(
                index=False, name=None
            )
        }
        out[str(exp)] = len(reference & present) / len(reference)
    return out


def _drop_incomplete_conditions(
    df: pd.DataFrame,
    baseline_experiment: str | None,
    min_condition_completeness: float | None,
    warnings_out: list[str],
) -> pd.DataFrame:
    """Drop every controlled experiment whose slot completeness is below the
    threshold (``None`` ⇒ keep everything). A whole condition is dropped, not just
    its missing cells, so ratings and baselines never mix complete and partial
    evidence from the same experiment. The selected explicit baseline is called out
    loudly because dropping it would void the whole cell adjustment."""
    if min_condition_completeness is None or not df["controlled"].any():
        return df
    incomplete = sorted(
        exp
        for exp, frac in _condition_completeness(df, baseline_experiment).items()
        if frac < min_condition_completeness
    )
    if not incomplete:
        return df
    if baseline_experiment is not None and str(baseline_experiment) in incomplete:
        warnings_out.append(
            f"condition completeness: the selected baseline_experiment "
            f"'{baseline_experiment}' is itself below min_condition_completeness "
            f"{min_condition_completeness} and would be dropped; reconsider the "
            f"threshold or the baseline source."
        )
    dropped = int(df["experiment"].isin(incomplete).sum())
    df = df[~df["experiment"].isin(incomplete)]
    warnings_out.append(
        f"condition completeness: dropped {len(incomplete)} experiment(s) below "
        f"min_condition_completeness {min_condition_completeness}: "
        f"{', '.join(incomplete)} ({dropped} rows)."
    )
    return df


# ── orchestration ────────────────────────────────────────────────────────────
def _effective_block(block: str, controlled_any: bool) -> str:
    if block == "auto":
        return "start_cell" if controlled_any else "none"
    return block


def build_strength_panel(
    predictions_path: str,
    panel_path: str,
    games_path: str,
    params: dict | None,
    catalog: Catalog,
    estimator_id: str = "",
    problem_game_ids: set[str] | None = None,
) -> StrengthArtifacts:
    """Derive the strength panel + audit trails from an estimator's predictions.

    ``problem_game_ids`` (from the malformed-DB ``import_issues.csv``) are dropped
    from all three inputs right after reading, so the civ-OLS effects and the
    Vanilla cell baselines are fit on clean rows only.
    """
    p = _resolve_params(params)
    warnings_out: list[str] = []

    from ..data.loading import drop_problem_games

    pred = drop_problem_games(pd.read_csv(predictions_path), problem_game_ids)
    panel = drop_problem_games(pd.read_csv(panel_path), problem_game_ids)
    games = drop_problem_games(pd.read_csv(games_path), problem_game_ids)

    df = _weighted_strength(pred, p["turn_progress_min"], p["weight"])
    if df.empty:
        raise AdjustError(
            f"strength: no rows survived turn_progress > {p['turn_progress_min']} "
            f"from '{predictions_path}'."
        )
    df = _relative_and_logit(df, p["enforce_winner"], p["relative_to"])
    df = _join_identity(df, panel, games, catalog)

    df = _drop_incomplete_conditions(
        df, p["baseline_experiment"], p["min_condition_completeness"], warnings_out
    )

    controlled_any = bool(df["controlled"].any())
    effective = _effective_block(p["block"], controlled_any)
    cell_active = effective == "start_cell" and controlled_any

    # Uncontrolled path: OLS civ effects, always fit on ALL rows when enabled.
    civ_effects: dict[str, float] = {}
    civ_effects_df = pd.DataFrame(columns=CIV_EFFECTS_COLUMNS)
    if p["civ_adjust"] == "ols_logit":
        civ_effects, civ_effects_df = fit_civ_effects(df, catalog)

    # Controlled path: matched start-cell Vanilla VPAI baseline.
    cell_baseline_df = pd.DataFrame(columns=CELL_BASELINE_COLUMNS)
    cell_coverage_df = pd.DataFrame(columns=CELL_COVERAGE_COLUMNS)
    cell_baseline_series = pd.Series(np.nan, index=df.index)
    selected = None
    if cell_active:
        baseline_df = df[df["controlled"] & _vanilla_baseline_mask(df, catalog)]
        be = p["baseline_experiment"]
        cell_baseline_df = _build_cell_baseline(df, baseline_df, be, catalog)
        selected, sel_map = _selected_baseline_maps(baseline_df, be)
        cell_baseline_series = _cell_baseline_per_row(df, selected, sel_map)

        # A controlled row whose selected baseline cell is missing. EXPLICIT is
        # fatal — the designated baseline source is meant to span the whole grid,
        # so a hole there is a config/data error. IMPLICIT (self-coverage) degrades
        # gracefully: incomplete self-coverage is expected (rotation sparsity), so
        # we still adjust every cell that HAS a baseline and fall the rest back to
        # the uncontrolled adjustment (WARN, never abort).
        missing_mask = df["controlled"] & cell_baseline_series.isna()
        if missing_mask.any():
            offenders = (
                df.loc[missing_mask, ["experiment", "seed", "player_id"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            cells = ", ".join(f"({e}, seed={s}, player_id={pid})" for e, s, pid in offenders)
            if selected == "explicit":
                raise AdjustError(
                    f"strength: selected explicit baseline (experiment '{be}') is "
                    f"missing for controlled cell(s): {cells}. The designated "
                    f"baseline_experiment must cover the full (seed, player_id) grid "
                    f"(and its Vanilla VPAI rows must survive data.filter / predict_subset)."
                )
            n_cells = int(missing_mask.sum())
            fallback = (
                "civ OLS adjustment" if p["civ_adjust"] == "ols_logit"
                else "relative_strength (unadjusted)"
            )
            warnings_out.append(
                f"implicit self-coverage incomplete: {n_cells} controlled row(s) in "
                f"cell(s) {cells} have no own Vanilla VPAI baseline — those rows fall "
                f"back to {fallback}; the complete cells still use the start-cell "
                f"baseline (report-only)."
            )

        cell_coverage_df = _build_cell_coverage(df, baseline_df, be, selected, sel_map, catalog)
        warnings_out.extend(_coverage_warnings(cell_coverage_df, cell_baseline_df, selected, be))

    # ── compute adjusted_strength per row ────────────────────────────────────
    # cell_mask: controlled rows that actually have a start-cell baseline. Controlled
    # rows missing one (implicit fallback) drop into `rest` with the uncontrolled path.
    adjusted = pd.Series(np.nan, index=df.index)
    # The exact start-cell advantage on the logit scale (logit_strength - cell_baseline),
    # captured here *before* the optional post_cell_normalize so it always means
    # "your strength - matched Vanilla VPAI baseline in this cell". NaN for non-cell rows.
    cell_logit_advantage = pd.Series(np.nan, index=df.index)
    if cell_active:
        cell_mask = df["controlled"] & cell_baseline_series.notna()
    else:
        cell_mask = pd.Series(False, index=df.index)
    if cell_mask.any():
        cell_diff = (
            df.loc[cell_mask, "logit_strength"].to_numpy()
            - cell_baseline_series[cell_mask].to_numpy()
        )
        adjusted[cell_mask] = inv_logit(cell_diff)
        cell_logit_advantage[cell_mask] = cell_diff
    rest = ~cell_mask
    if p["civ_adjust"] == "ols_logit":
        civ_adj = df["civilization"].map(civ_effects).fillna(0.0)
        adjusted[rest] = inv_logit(
            df.loc[rest, "logit_strength"].to_numpy() - civ_adj[rest].to_numpy()
        )
    else:
        adjusted[rest] = df.loc[rest, "relative_strength"].to_numpy()
    df["adjusted_strength"] = adjusted
    df["cell_logit_advantage"] = cell_logit_advantage

    # Audit: how each row was adjusted (cell | civ | relative). Lets the report flag
    # implicit-fallback rows as preliminary without re-deriving the coverage.
    rest_method = "civ" if p["civ_adjust"] == "ols_logit" else "relative"
    df["adjust_method"] = np.where(cell_mask.to_numpy(), "cell", rest_method)

    # Optional final re-normalization of the cell-adjusted (controlled) rows.
    if cell_active and p["post_cell_normalize"] == "relative_to_leader":
        mask = df["controlled"]
        gmax = df.loc[mask].groupby("game_id")["adjusted_strength"].transform("max")
        df.loc[mask, "adjusted_strength"] = df.loc[mask, "adjusted_strength"] / gmax

    panel_out = df[[c for c in PANEL_COLUMNS if c in df.columns]].reset_index(drop=True)
    return StrengthArtifacts(
        panel=panel_out,
        civ_effects=civ_effects_df,
        cell_baseline=cell_baseline_df,
        cell_coverage=cell_coverage_df,
        warnings=warnings_out,
        estimator_id=estimator_id,
    )


def _coverage_warnings(
    coverage: pd.DataFrame, cell_baseline: pd.DataFrame, selected: str, baseline_experiment: str | None
) -> list[str]:
    out: list[str] = []
    if not coverage.empty:
        missing = coverage[coverage["missing"]]
        if not missing.empty:
            n = len(missing)
            exps = ", ".join(sorted(missing["experiment"].unique()))
            out.append(
                f"cell coverage: {n} (seed, player_id) cell(s) of the entirety grid "
                f"are uncovered across experiment(s) [{exps}] (report-only)."
            )
    if not cell_baseline.empty:
        disc = cell_baseline[~cell_baseline["vanilla_connected"]]
        if not disc.empty:
            out.append(
                f"connectedness: {len(disc)} cell(s) have model(s) not connected to "
                f"'{selected}' Vanilla baseline (extrapolated; report-only)."
            )
    # Comparison-pathway gaps: when explicit is selected, the implicit comparison
    # may have experiment cells with rows but no own Vanilla evidence. Such cells
    # do not appear in cell_baseline at all, so detect them from the coverage grid.
    if baseline_experiment is not None and not coverage.empty:
        gaps = coverage[(coverage["n_rows"] > 0) & (coverage["n_vanilla"] == 0)]
        if not gaps.empty:
            out.append(
                f"comparison pathway: {len(gaps)} implicit cell(s) lack a Vanilla "
                f"baseline for the implicit-vs-explicit comparison (report-only)."
            )
    return out
