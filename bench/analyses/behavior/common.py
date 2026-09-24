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

:func:`heatmap_rows`, :func:`color_positions`, and :func:`heatmap_spec` turn a
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
from ...plotting.heatmap import plot_diverging_heatmap
from ...plotting.pairing import condition_display, order_conditions, order_strategists
from ...plotting.styles import sort_player_types
from ..base import AnalysisContext
from ..errors import AnalysisError

RELATIVE = "relative"
ABSOLUTE = "absolute"
VIEW_LABELS = {RELATIVE: "Relative to matched in-game AI", ABSOLUTE: "Absolute"}
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
    own rows leave the relative view).
    """

    kind: str
    experiments: tuple[str, ...]
    label: str

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
    dropped_metrics: list[str] = field(default_factory=list)
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
    return Baseline("experiments", names, label)


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
) -> BehaviorViews:
    """Build the absolute rows and, when the design allows, the relative rows.

    ``rows`` are the filtered players, ``pool`` the unfiltered baseline rows,
    both already carrying numeric metric columns. Players with no value for any
    metric (for example in-game AI opponents on the flavor page) are left out
    of both views, so player counts cover only players who were measured.
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
    views.n_baseline_rows = int(len(pool))
    views.n_baseline_experiments = int(pool["experiment"].nunique())

    in_baseline = rows["experiment"].isin(baseline.experiments)
    candidates = rows if baseline.self_included else rows[~in_baseline]
    candidates = candidates.merge(cells, on="game_id", how="inner")
    matched = candidates.join(means, on=["seed", "player_id"], rsuffix="__baseline", how="left")
    has_cell = matched[[f"{m}__baseline" for m in usable]].notna().any(axis=1)
    views.n_unmatched = int((~has_cell).sum())
    matched = matched[has_cell].copy()
    for metric in usable:
        matched[metric] = matched[metric] - matched.pop(f"{metric}__baseline")
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


# ── labels and figures ───────────────────────────────────────────────────────
def metric_label(metric: str, rate_scaled: set) -> str:
    label = metric.replace("_", " ")
    return f"{label} /100 turns" if metric in rate_scaled else label


def summary_heatmap(
    summary: pd.DataFrame,
    by: str,
    metrics: list[str],
    color: dict,
    *,
    labels: dict,
    title: str,
    cbar_label: str,
    signed: bool,
):
    """Heatmap of ``by`` × metric: colored by ``color[(group, metric)]``, annotated with the mean."""
    groups = list(dict.fromkeys(summary[by].astype(str)))
    columns = [m for m in metrics if m in set(summary["metric"])]
    matrix = pd.DataFrame(0.0, index=groups, columns=[labels[m] for m in columns])
    annot = pd.DataFrame("", index=groups, columns=matrix.columns)
    mask = pd.DataFrame(True, index=groups, columns=matrix.columns)
    for rec in summary.itertuples(index=False):
        group, metric = str(getattr(rec, by)), rec.metric
        if metric not in columns:
            continue
        col = labels[metric]
        value = color.get((group, metric), 0.0)
        matrix.loc[group, col] = 0.0 if not np.isfinite(value) else value
        annot.loc[group, col] = _fmt(rec.mean, signed)
        mask.loc[group, col] = False
    fig, _ax = plot_diverging_heatmap(
        matrix, annot=annot, mask=mask, vlim=2.0, title=title, cbar_label=cbar_label,
        ylabel=by.replace("_", " "),
    )
    return fig


def _fmt(value: float, signed: bool) -> str:
    if not np.isfinite(value):
        return ""
    magnitude = abs(value)
    text = f"{value:+.2f}" if signed else f"{value:.2f}"
    if magnitude >= 100:
        text = f"{value:+.0f}" if signed else f"{value:.0f}"
    elif magnitude >= 10:
        text = f"{value:+.1f}" if signed else f"{value:.1f}"
    return text


def absolute_colors(summary: pd.DataFrame, rows: pd.DataFrame, by: str) -> dict:
    """Group mean as a z-score against all players in the absolute view."""
    colors = {}
    for rec in summary.itertuples(index=False):
        values = rows[rec.metric].dropna()
        sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        z = (rec.mean - float(values.mean())) / sd if sd and np.isfinite(sd) else 0.0
        colors[(str(getattr(rec, by)), rec.metric)] = z
    return colors


def relative_colors(summary: pd.DataFrame, baseline_sd: dict, by: str) -> dict:
    """Mean difference in units of the baseline's spread for that metric."""
    colors = {}
    for rec in summary.itertuples(index=False):
        sd = baseline_sd.get(rec.metric, float("nan"))
        colors[(str(getattr(rec, by)), rec.metric)] = (
            rec.mean / sd if sd and np.isfinite(sd) else 0.0
        )
    return colors


def views_metadata(declared: dict, baseline: Optional[Baseline] = None) -> dict:
    """``metadata["views"]`` in display order, relative first, skipping empty views."""
    labels = dict(VIEW_LABELS)
    if baseline is not None:
        labels[RELATIVE] = baseline.view_label
    out = {}
    for name in (RELATIVE, ABSOLUTE):
        spec = declared.get(name)
        if spec and (spec.get("tables") or spec.get("figures")):
            out[name] = {"label": labels[name], **spec}
    return out


def largest_departure(summary: pd.DataFrame, colors: dict, by: str, min_players: int = 1):
    """The ``(group, metric, mean)`` with the largest |color| among well-supported cells."""
    best = None
    for rec in summary.itertuples(index=False):
        if rec.n_players < min_players:
            continue
        score = abs(colors.get((str(getattr(rec, by)), rec.metric), 0.0))
        if best is None or score > best[0]:
            best = (score, str(getattr(rec, by)), rec.metric, float(rec.mean))
    return None if best is None else best[1:]


def stacked_shares_figure(shares: pd.DataFrame, *, title: str, xlabel: str = "share of players"):
    """Horizontal 100% bars: rows are groups, columns are categories summing to 1."""
    import matplotlib.pyplot as plt

    n_rows = max(1, len(shares))
    fig, ax = plt.subplots(figsize=(10, 0.4 * n_rows + 1.4))
    cmap = plt.get_cmap("tab10")
    left = np.zeros(len(shares))
    labels = [str(i) for i in shares.index]
    for idx, column in enumerate(shares.columns):
        values = shares[column].fillna(0.0).to_numpy(dtype=float)
        ax.barh(labels, values, left=left, color=cmap(idx % 10), label=str(column),
                edgecolor="white", linewidth=0.5)
        left = left + values
    ax.set_xlim(0, 1)
    ax.margins(y=0.01)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def share_table(df: pd.DataFrame, by: str, category: str, categories: list[str]) -> pd.DataFrame:
    """Per-group share of rows in each category, plus ``n_players``; categories in order."""
    counts = pd.crosstab(df[by].astype(str), df[category].astype(str))
    counts = counts.reindex(columns=categories, fill_value=0)
    shares = counts.div(counts.sum(axis=1).where(lambda s: s > 0), axis=0)
    shares.insert(0, "n_players", counts.sum(axis=1))
    shares.index.name = by
    out = shares.reset_index()
    order = sort_player_types(out[by]) if by == "player_type" else sorted(out[by])
    return out.set_index(by).loc[order].reset_index()


# ── HTML heatmap tables ───────────────────────────────────────────────────────
def heatmap_rows(ctx: AnalysisContext, summary: pd.DataFrame, by: str) -> tuple[pd.DataFrame, list[str]]:
    """Add ``strategist``, ``condition``, and ``row_label``, and return the row order.

    With condition pairing on and ``by == "player_type"``, a row reads
    "Strategist | Condition" and rows sort like Matched Maps: the Null and
    Vanilla baselines first, then catalog strategist order, then condition
    order. Otherwise the row label is the group value, in :func:`order_groups`.
    """
    out = summary.copy()
    groups = [str(g) for g in dict.fromkeys(out[by].astype(str))]
    spec = ctx.condition_pairing() if by == "player_type" else None
    baselines = [ctx.catalog.null_label, ctx.catalog.vanilla_label]
    if spec is None:
        out["strategist"], out["condition"] = out[by].astype(str), ""
        out["row_label"] = out[by].astype(str)
        order = sort_player_types(groups) if by == "player_type" else sorted(groups)
        return out, order

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

    ordered = sorted(groups, key=rank)
    out["strategist"] = out[by].astype(str).map(lambda g: identity[g][0])
    out["condition"] = out[by].astype(str).map(lambda g: identity[g][1])
    out["row_label"] = out[by].astype(str).map(lambda g: identity[g][2])
    return out, [identity[g][2] for g in ordered]


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
) -> dict:
    """The ``metadata["heatmaps"]`` entry for one view's table (layout only)."""
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
    return spec

