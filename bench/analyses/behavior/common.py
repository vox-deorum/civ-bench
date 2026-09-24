"""Shared machinery for the descriptive ``behavior.*`` analyses (benchmark.md §6.2).

Every behavior module turns its inputs into one row per player per game with a
set of metric columns, then hands the rows to :func:`build_views`, which returns
two views of the same metrics:

* **relative** (the default): controlled rows only, each metric minus the mean
  of the baseline experiment's in-game AI at the same ``(seed, player_id)``
  cell. Rows without a matched cell and the baseline rows themselves are left
  out; metrics the baseline never populates are dropped.
* **absolute**: every filtered row, as measured.

The baseline pool is read from the unfiltered table (minus problem games and
decision-failure games), so player filters such as ``only_llm`` cannot remove
the in-game AI rows the comparison needs.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...plotting.heatmap import plot_diverging_heatmap
from ...plotting.styles import sort_player_types
from ..base import AnalysisContext
from ..errors import AnalysisError

RELATIVE = "relative"
ABSOLUTE = "absolute"
VIEW_LABELS = {RELATIVE: "Relative to matched in-game AI", ABSOLUTE: "Absolute"}

DEFAULT_RATE = "per_100_turns"
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_CI_LEVEL = 0.95

KEY_COLUMNS = ["experiment", "game_id", "player_id", "player_type"]
SUMMARY_COLUMNS = [
    "metric", "mean", "median", "ci_lower", "ci_upper", "n_players", "n_games",
]


@dataclass
class BehaviorViews:
    """Per-player metric rows for both views plus the provenance of the relative one."""

    absolute: pd.DataFrame
    relative: Optional[pd.DataFrame]
    metrics: list[str]
    relative_metrics: list[str] = field(default_factory=list)
    baseline_experiment: Optional[str] = None
    baseline_sd: dict = field(default_factory=dict)
    dropped_metrics: list[str] = field(default_factory=list)
    n_unmatched: int = 0
    n_baseline_rows: int = 0
    relative_note: str = ""

    def metadata(self) -> dict:
        out = {"n_absolute_players": int(len(self.absolute))}
        if self.baseline_experiment:
            out["baseline_experiment"] = self.baseline_experiment
        if self.relative is not None:
            out["n_relative_players"] = int(len(self.relative))
            out["n_unmatched_controlled_players"] = int(self.n_unmatched)
            out["n_baseline_players"] = int(self.n_baseline_rows)
            if self.dropped_metrics:
                out["relative_metrics_without_baseline"] = list(self.dropped_metrics)
        elif self.relative_note:
            out["relative_view"] = self.relative_note
        return out


# ── params ────────────────────────────────────────────────────────────────────
def resolve_baseline_experiment(ctx: AnalysisContext, params: dict) -> Optional[str]:
    """The stage's ``baseline_experiment``, else the first strength stage's one."""
    if params.get("baseline_experiment"):
        return str(params["baseline_experiment"])
    for stage in ctx.config.adjust:
        if stage.enabled and stage.raw.get("module") == "strength":
            value = (stage.raw.get("params") or {}).get("baseline_experiment")
            if value:
                return str(value)
    return None


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


def baseline_pool(ctx: AnalysisContext, full: pd.DataFrame, baseline: Optional[str]) -> pd.DataFrame:
    """Baseline-experiment rows from an unfiltered table, minus decision-failure games."""
    if not baseline:
        return full.iloc[0:0]
    failed = {str(g) for g in ctx.decision_failure_ids()}
    pool = full[full["experiment"] == baseline]
    return pool[~pool["game_id"].isin(failed)]


# ── views ─────────────────────────────────────────────────────────────────────
def build_views(
    rows: pd.DataFrame,
    pool: pd.DataFrame,
    metrics: list[str],
    cells: Optional[pd.DataFrame],
    baseline: Optional[str],
) -> BehaviorViews:
    """Build the absolute rows and, when the design allows, the relative rows.

    ``rows`` are the filtered players, ``pool`` the unfiltered baseline rows,
    both already carrying numeric metric columns.
    """
    views = BehaviorViews(absolute=rows.reset_index(drop=True), relative=None,
                          metrics=list(metrics), baseline_experiment=baseline)
    if not baseline:
        views.relative_note = "no baseline_experiment is configured"
        return views
    if cells is None or cells.empty:
        views.relative_note = "no controlled games"
        return views

    pool = pool.merge(cells, on="game_id", how="inner")
    if pool.empty:
        views.relative_note = f"baseline experiment '{baseline}' has no controlled games"
        return views
    usable = [m for m in metrics if pool[m].notna().any()]
    views.dropped_metrics = [m for m in metrics if m not in usable]
    if not usable:
        views.relative_note = "the baseline populates none of the metrics"
        return views
    means = pool.groupby(["seed", "player_id"])[usable].mean()
    views.baseline_sd = {m: float(pool[m].std(ddof=1)) for m in usable}

    candidates = rows[rows["experiment"] != baseline].merge(cells, on="game_id", how="inner")
    views.n_baseline_rows = int((rows["experiment"] == baseline).sum())
    matched = candidates.join(means, on=["seed", "player_id"], rsuffix="__baseline", how="left")
    has_cell = matched[[f"{m}__baseline" for m in usable]].notna().any(axis=1)
    views.n_unmatched = int((~has_cell).sum())
    matched = matched[has_cell].copy()
    for metric in usable:
        matched[metric] = matched[metric] - matched.pop(f"{metric}__baseline")
    for metric in views.dropped_metrics:
        matched = matched.drop(columns=metric)
    if matched.empty:
        views.relative_note = "no controlled player has a matched baseline cell"
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


def views_metadata(declared: dict) -> dict:
    """``metadata["views"]`` in display order, relative first, skipping empty views."""
    out = {}
    for name in (RELATIVE, ABSOLUTE):
        spec = declared.get(name)
        if spec and (spec.get("tables") or spec.get("figures")):
            out[name] = {"label": VIEW_LABELS[name], **spec}
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
