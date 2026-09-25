"""Shared machinery for the descriptive ``behavior.*`` analyses (benchmark.md §6.2).

Every behavior module turns its inputs into one row per player per game with a
set of metric columns, then hands the rows to :func:`build_views`, which returns
two views of the same metrics:

* **relative** (the default): controlled rows only, each metric minus the mean
  of the baseline pool at the same ``(seed, player_id)`` cell. Metrics the pool
  never populates are dropped.
* **absolute**: every filtered row, as measured.

The pool comes from ``params.baseline`` (:func:`resolve_baseline`): one
experiment such as the in-game AI, a list of experiments, or ``"completed"``,
the strategist players of every experiment that fills the controlled grid. It
is read from the unfiltered table (minus problem games and decision-failure
games), so player filters such as ``only_llm`` cannot remove it.

:func:`heatmap_rows` (from :mod:`bench.analyses.heatmap_layout`),
:func:`color_positions`, and :func:`heatmap_spec` turn a
view's summary into the long table and layout spec the report draws as an HTML
heatmap, so every page labels, orders, and colors its cells the same way.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...config import schema as S
from ...plotting.styles import sort_player_types
from ..base import AnalysisContext
from ..errors import AnalysisError
from ..heatmap_layout import heatmap_rows  # noqa: F401  (re-exported for the modules)

RELATIVE = "relative"
ABSOLUTE = "absolute"
# Short tab labels; the long wording rides along as the tab's tooltip.
VIEW_LABELS = {RELATIVE: "Relative", ABSOLUTE: "Absolute"}
VIEW_TIPS = {RELATIVE: "Relative to matched in-game AI", ABSOLUTE: "Absolute values"}
COMPLETED = S.BEHAVIOR_BASELINE_COMPLETED
# A relative cell reaches full color at this many pool SDs from the baseline.
RELATIVE_COLOR_SD = 2.0
# An absolute cell without a fixed range reaches full color at this z-score.
ABSOLUTE_COLOR_Z = 2.0

DEFAULT_RATE = "per_100_turns"
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_CI_LEVEL = 0.95

KEY_COLUMNS = ["experiment", "game_id", "player_id", "player_type"]
SUMMARY_COLUMNS = [
    "metric", "mean", "median", "ci_lower", "ci_upper", "n_players", "n_games",
]


@dataclass(frozen=True)
class Baseline:
    """Where a relative view's reference values come from.

    ``kind`` is ``"completed"`` (every strategist in a completed experiment, each
    part of its own baseline) or ``"experiments"`` (the named experiments, whose
    own rows leave the relative view). ``in_game_ai`` marks the first strength
    stage's baseline experiment alone.
    """

    kind: str
    experiments: tuple[str, ...]
    label: str
    in_game_ai: bool = False

    @property
    def self_included(self) -> bool:
        return self.kind == COMPLETED

    @property
    def view_label(self) -> str:
        return f"Relative to {self.label}"


@dataclass
class BehaviorViews:
    """Per-player metric rows for both views plus the provenance of the relative one."""

    absolute: pd.DataFrame
    relative: Optional[pd.DataFrame]
    metrics: list[str]
    relative_metrics: list[str] = field(default_factory=list)
    baseline: Optional[Baseline] = None
    baseline_sd: dict = field(default_factory=dict)
    # metric → {mean, sd, n_players, n_games} of the pool on controlled games,
    # in absolute terms (the pinned baseline row of the HTML heatmaps).
    baseline_summary: dict = field(default_factory=dict)
    dropped_metrics: list[str] = field(default_factory=list)
    # Companion columns the relative rows hold as differences from the baseline.
    shifted: set = field(default_factory=set)
    # The pool's rows on controlled games, as measured (for pinned-row tooltips).
    pool: Optional[pd.DataFrame] = None
    n_unmatched: int = 0
    n_baseline_rows: int = 0
    n_baseline_experiments: int = 0
    relative_note: str = ""

    def metadata(self) -> dict:
        out = {"n_absolute_players": int(len(self.absolute))}
        if self.baseline is not None:
            out["baseline"] = self.baseline.label
            if not self.baseline.self_included:
                out["baseline_experiments"] = list(self.baseline.experiments)
        if self.relative is not None:
            out["n_relative_players"] = int(len(self.relative))
            out["n_unmatched_controlled_players"] = int(self.n_unmatched)
            out["n_baseline_players"] = int(self.n_baseline_rows)
            if self.baseline is not None and self.baseline.self_included:
                out["n_baseline_experiments"] = int(self.n_baseline_experiments)
            if self.dropped_metrics:
                out["relative_metrics_without_baseline"] = list(self.dropped_metrics)
        elif self.relative_note:
            out["relative_view"] = self.relative_note
        return out


# ── params ────────────────────────────────────────────────────────────────────
def strength_baseline_experiment(ctx: AnalysisContext) -> Optional[str]:
    """The first enabled strength stage's ``baseline_experiment`` (the in-game AI)."""
    for stage in ctx.config.adjust:
        if stage.enabled and stage.raw.get("module") == "strength":
            value = (stage.raw.get("params") or {}).get("baseline_experiment")
            if value:
                return str(value)
    return None


def completed_experiments(ctx: AnalysisContext) -> tuple[str, ...]:
    """Experiments that fill every controlled ``(seed, seating_rotation)`` slot.

    Problem games and decision-failure games do not count, as in
    ``performance.experiment_completeness``.
    """
    from ...data.loading import condition_completeness

    path = (ctx.config.data.get("tables") or {}).get("games")
    if not path or not Path(path).exists():
        return ()
    games = ctx.load_table("games")
    shares = condition_completeness(games, "experiment", ctx.decision_failure_ids())
    return tuple(sorted(name for name, share in shares.items() if share >= 1.0))


def resolve_baseline(ctx: AnalysisContext, params: dict, default: Optional[str] = None) -> Optional[Baseline]:
    """The stage's ``baseline``, else the module ``default``, else the in-game AI.

    ``default`` is ``"completed"`` or ``None``; ``None`` falls back to the first
    strength stage's ``baseline_experiment``.
    """
    value = params.get("baseline", default)
    in_game_ai = strength_baseline_experiment(ctx)
    if value == COMPLETED:
        return Baseline(COMPLETED, completed_experiments(ctx), "completed-experiment average")
    if value is None:
        value = in_game_ai
        if value is None:
            return None
    names = (str(value),) if isinstance(value, str) else tuple(str(v) for v in value)
    if names == (in_game_ai,):
        label = "matched in-game AI"
    elif len(names) == 1:
        label = f"matched '{names[0]}'"
    else:
        label = f"matched average of {len(names)} experiments"
    return Baseline("experiments", names, label, in_game_ai=names == (in_game_ai,))


def stage_rng(ctx: AnalysisContext, *parts) -> np.random.Generator:
    """A generator seeded from the run seed and stable names, independent of call order."""
    keys = [int(ctx.config.seed)] + [zlib.crc32(str(p).encode("utf-8")) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(keys))


# ── inputs ────────────────────────────────────────────────────────────────────
def controlled_cells(ctx: AnalysisContext) -> Optional[pd.DataFrame]:
    """``game_id → seed`` for controlled games, or ``None`` without a games table."""
    path = (ctx.config.data.get("tables") or {}).get("games")
    if not path or not Path(path).exists():
        return None
    games = ctx.load_table("games")
    missing = {"game_id", "seed", "seating_rotation"} - set(games.columns)
    if missing:
        raise AnalysisError(
            f"analysis '{ctx.stage_id}': games table lacks column(s) {sorted(missing)}."
        )
    games = games.copy()
    games["game_id"] = games["game_id"].astype(str)
    for column in ("seed", "seating_rotation"):
        games[column] = pd.to_numeric(games[column], errors="coerce").fillna(-1).astype(int)
    controlled = games[(games["seed"] != -1) & (games["seating_rotation"] != -1)]
    return controlled[["game_id", "seed"]].drop_duplicates("game_id")


def prepare_keys(df: pd.DataFrame, stage_id: str, source: str, *, per_player: bool = True) -> pd.DataFrame:
    """Normalize key dtypes; ``per_player`` keeps one row per ``(game_id, player_id)``."""
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise AnalysisError(f"analysis '{stage_id}': {source} lacks column(s) {missing}.")
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str)
    df["experiment"] = df["experiment"].astype(str)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="raise").astype(int)
    return df.drop_duplicates(["game_id", "player_id"], keep="first") if per_player else df


def numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Coerce metric columns to floats; blanks and ``N/A`` become NaN."""
    df = df.copy()
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def per_100_turns(df: pd.DataFrame, columns, turns_col: str = "survival_turn") -> pd.DataFrame:
    """Divide count columns by turns alive / 100 (turn 0 counts, so turns = last + 1)."""
    df = df.copy()
    turns = pd.to_numeric(df[turns_col], errors="coerce") + 1
    turns = turns.where(turns > 0)
    for column in columns:
        df[column] = df[column] / turns * 100.0
    return df


def baseline_pool(ctx: AnalysisContext, full: pd.DataFrame, baseline: Optional[Baseline]) -> pd.DataFrame:
    """Baseline rows from an unfiltered table, minus decision-failure games.

    A ``"completed"`` pool keeps strategist players only: the in-game AI
    opponents inside those games and the Null strategist are left out.
    """
    if baseline is None or not baseline.experiments:
        return full.iloc[0:0]
    failed = {str(g) for g in ctx.decision_failure_ids()}
    pool = full[full["experiment"].isin(baseline.experiments)]
    pool = pool[~pool["game_id"].isin(failed)]
    if baseline.self_included:
        pool = pool[~pool["player_type"].isin({ctx.catalog.vanilla_label, ctx.catalog.null_label})]
    return pool


# ── views ─────────────────────────────────────────────────────────────────────
def build_views(
    rows: pd.DataFrame,
    pool: pd.DataFrame,
    metrics: list[str],
    cells: Optional[pd.DataFrame],
    baseline: Optional[Baseline],
    companions: Optional[dict] = None,
) -> BehaviorViews:
    """Build the absolute rows and, when the design allows, the relative rows.

    ``rows`` are the filtered players, ``pool`` the unfiltered baseline rows,
    both already carrying numeric metric columns. Players with no value for any
    metric (for example in-game AI opponents on the flavor page) are left out
    of both views, so player counts cover only players who were measured.
    ``companions`` maps a metric to more columns of the same unit (such as its
    per-player min and max); relative rows shift them by the metric's baseline
    cell mean too, so a range reads as a difference from the baseline.
    """
    rows = rows[rows[metrics].notna().any(axis=1)] if metrics else rows
    views = BehaviorViews(absolute=rows.reset_index(drop=True), relative=None,
                          metrics=list(metrics), baseline=baseline)
    if baseline is None:
        views.relative_note = "no baseline is configured"
        return views
    if not baseline.experiments:
        views.relative_note = "no experiment is complete"
        return views
    if cells is None or cells.empty:
        views.relative_note = "no controlled games"
        return views

    pool = pool.merge(cells, on="game_id", how="inner")
    if pool.empty:
        views.relative_note = f"the {baseline.label} has no controlled games"
        return views
    usable = [m for m in metrics if pool[m].notna().any()]
    views.dropped_metrics = [m for m in metrics if m not in usable]
    if not usable:
        views.relative_note = f"the {baseline.label} populates none of the metrics"
        return views
    means = pool.groupby(["seed", "player_id"])[usable].mean()
    views.baseline_sd = {m: float(pool[m].std(ddof=1)) for m in usable}
    for metric in usable:
        values = pool[["game_id", metric]].dropna()
        views.baseline_summary[metric] = {
            "mean": float(values[metric].mean()),
            "sd": views.baseline_sd[metric],
            "n_players": int(len(values)),
            "n_games": int(values["game_id"].nunique()),
        }
    views.n_baseline_rows = int(len(pool))
    views.n_baseline_experiments = int(pool["experiment"].nunique())
    views.pool = pool.reset_index(drop=True)

    in_baseline = rows["experiment"].isin(baseline.experiments)
    candidates = rows if baseline.self_included else rows[~in_baseline]
    candidates = candidates.merge(cells, on="game_id", how="inner")
    matched = candidates.join(means, on=["seed", "player_id"], rsuffix="__baseline", how="left")
    has_cell = matched[[f"{m}__baseline" for m in usable]].notna().any(axis=1)
    views.n_unmatched = int((~has_cell).sum())
    matched = matched[has_cell].copy()
    for metric in usable:
        base = matched.pop(f"{metric}__baseline")
        matched[metric] = matched[metric] - base
        for column in (companions or {}).get(metric, ()):
            if column in matched.columns:
                matched[column] = matched[column] - base
                views.shifted.add(column)
    for metric in views.dropped_metrics:
        matched = matched.drop(columns=metric)
    if matched.empty:
        views.relative_note = f"no controlled player has a {baseline.label} cell"
        return views
    views.relative = matched.reset_index(drop=True)
    views.relative_metrics = usable
    return views


def summarize(
    ctx: AnalysisContext,
    df: pd.DataFrame,
    metrics: list[str],
    by: str,
    view: str,
    bootstrap_n: int,
    ci_level: float,
) -> pd.DataFrame:
    """Long table: one row per ``(by, metric)`` with mean, median, and a game-level bootstrap CI."""
    records = []
    for group, grp in df.groupby(by, sort=False):
        for metric in metrics:
            values = grp[["game_id", metric]].dropna()
            if values.empty:
                continue
            lo, hi = _game_bootstrap(
                values, metric, bootstrap_n, ci_level,
                stage_rng(ctx, ctx.stage_id, view, metric, group),
            )
            records.append({
                by: group,
                "metric": metric,
                "mean": float(values[metric].mean()),
                "median": float(values[metric].median()),
                "ci_lower": lo,
                "ci_upper": hi,
                "n_players": int(len(values)),
                "n_games": int(values["game_id"].nunique()),
            })
    out = pd.DataFrame(records, columns=[by, *SUMMARY_COLUMNS])
    return order_groups(out, by)


def _game_bootstrap(values: pd.DataFrame, metric: str, n: int, ci_level: float, rng) -> tuple:
    """Percentile CI of the mean, resampling whole games (players in a game move together)."""
    per_game = values.groupby("game_id", sort=True)[metric].agg(["sum", "count"])
    if len(per_game) < 2 or n < 1:
        return float("nan"), float("nan")
    sums = per_game["sum"].to_numpy()
    counts = per_game["count"].to_numpy()
    idx = rng.integers(0, len(per_game), size=(n, len(per_game)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def order_groups(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Stable group order: pinned baselines first, then alphabetical; metric order kept."""
    if df.empty:
        return df.reset_index(drop=True)
    groups = [str(g) for g in df[by].unique()]
    order = sort_player_types(groups) if by == "player_type" else sorted(groups)
    rank = {g: i for i, g in enumerate(order)}
    metric_rank = {m: i for i, m in enumerate(dict.fromkeys(df["metric"]))} if "metric" in df else {}
    keyed = df.assign(
        _g=df[by].astype(str).map(rank),
        _m=df["metric"].map(metric_rank) if "metric" in df else 0,
    )
    return keyed.sort_values(["_g", "_m"], kind="stable").drop(columns=["_g", "_m"]).reset_index(drop=True)


def check_by(df: pd.DataFrame, by: str, stage_id: str) -> None:
    if by not in df.columns:
        raise AnalysisError(f"analysis '{stage_id}': params.by column '{by}' is not in the input.")


# ── view labels ──────────────────────────────────────────────────────────────
def views_metadata(declared: dict, baseline: Optional[Baseline] = None,
                   extra: Optional[dict] = None) -> dict:
    """``metadata["views"]`` in display order, skipping empty views.

    Relative comes first, then absolute, then any other declared view in
    declaration order, labeled by ``extra[name] = (label, tip)``.
    """
    labels = dict(VIEW_LABELS)
    tips = dict(VIEW_TIPS)
    if baseline is not None:
        tips[RELATIVE] = baseline.view_label
    for name, (label, tip) in (extra or {}).items():
        labels[name], tips[name] = label, tip
    order = [RELATIVE, ABSOLUTE] + [n for n in declared if n not in (RELATIVE, ABSOLUTE)]
    out = {}
    for name in order:
        spec = declared.get(name)
        if spec and (spec.get("tables") or spec.get("figures")):
            out[name] = {"label": labels.get(name, name), "tip": tips.get(name, ""), **spec}
    return out


# ── HTML heatmap tables ───────────────────────────────────────────────────────
def color_positions(
    summary: pd.DataFrame,
    view: str,
    views: BehaviorViews,
    fixed: Optional[dict] = None,
) -> pd.Series:
    """Each cell's place on the report's 0 to 1 color scale (0.5 is neutral).

    * relative: the mean difference in pool SDs, full color at
      ±``RELATIVE_COLOR_SD``;
    * absolute: the metric's ``fixed[metric] = (low, high)`` range when given,
      else a z-score across every absolute player, full color at
      ±``ABSOLUTE_COLOR_Z``.
    """
    fixed = fixed or {}
    positions = []
    for rec in summary.itertuples(index=False):
        mean = float(rec.mean)
        if view == RELATIVE:
            sd = views.baseline_sd.get(rec.metric, float("nan"))
            score = mean / sd / RELATIVE_COLOR_SD if sd and np.isfinite(sd) else 0.0
            position = 0.5 + 0.5 * float(np.clip(score, -1.0, 1.0))
        elif rec.metric in fixed:
            low, high = fixed[rec.metric]
            position = float(np.clip((mean - low) / (high - low), 0.0, 1.0))
        else:
            values = views.absolute[rec.metric].dropna()
            sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            z = (mean - float(values.mean())) / sd if sd and np.isfinite(sd) else 0.0
            position = 0.5 + 0.5 * float(np.clip(z / ABSOLUTE_COLOR_Z, -1.0, 1.0))
        positions.append(round(position, 6))
    return pd.Series(positions, index=summary.index, dtype=float)


def heatmap_spec(
    *,
    view: str,
    title: str,
    help_text: str,
    row_order: list[str],
    reference_rows: list[str],
    row_tips: dict,
    column_order: list[str],
    column_names: dict,
    column_labels: dict,
    column_tips: dict,
    grouped: bool,
    decimals: int,
    value_label: str,
    ci_level: float,
    legend: list,
    range_label: str = "",
    range_note: str = "",
    baseline_rows: Optional[list] = None,
    column_decimals: Optional[dict] = None,
    column_units: Optional[dict] = None,
    tip_rows: Optional[list] = None,
    baseline_units: Optional[dict] = None,
) -> dict:
    """The ``metadata["heatmaps"]`` entry for one view's table (layout only).

    ``column_decimals`` and ``column_units`` override the number format and add
    a tooltip unit per column (``baseline_units`` for the pinned baseline rows,
    which stay absolute); ``tip_rows`` lists extra tooltip lines
    (:mod:`bench.reports.heatmap`).
    """
    spec = {
        "view": view,
        "title": title,
        "help": help_text,
        "row": "row_label",
        "column": "metric",
        "value": "mean",
        "row_heading": "Strategist | Condition",
        "row_order": list(row_order),
        "reference_rows": list(reference_rows),
        "row_tips": dict(row_tips),
        "column_order": list(column_order),
        "column_names": dict(column_names),
        "column_labels": dict(column_labels),
        "column_tips": dict(column_tips),
        "decimals": int(decimals),
        "signed": view == RELATIVE,
        "value_label": value_label,
        "ci_level": float(ci_level),
        "legend": [list(entry) for entry in legend],
    }
    if grouped:
        spec["column_group"] = "metric_group"
    if range_label:
        spec["range_columns"] = ["mean_min", "mean_max"]
        spec["range_label"] = range_label
        if range_note:
            spec["range_note"] = range_note
    if baseline_rows:
        spec["baseline_rows"] = list(baseline_rows)
    if column_decimals:
        spec["column_decimals"] = {m: int(v) for m, v in column_decimals.items() if m in column_order}
    if column_units:
        spec["column_units"] = {m: str(v) for m, v in column_units.items() if m in column_order}
    if baseline_units is not None and baseline_rows:
        spec["baseline_units"] = {m: str(v) for m, v in baseline_units.items() if m in column_order}
    if tip_rows:
        spec["tip_rows"] = [dict(row) for row in tip_rows]
    return spec

