"""``performance.controlled_seed_report``: report-ready tables for the controlled-seed report.

The data side of the dedicated controlled-seed HTML report (see
``plans/controlled-seed-report.md``). The module consumes the canonical ``games``
and ``panel`` tables, one configured strength adjust table, and exactly one
estimator, then emits three deterministic, report-ready tables:

* ``seed_player_summary``: one row per observed ``(seed, player_id, strategist,
  condition)`` plus one dedicated Vanilla row per ``(seed, player_id)`` covered by
  the strength stage's ``baseline_experiment``.
* ``seed_player_probability``: mean victory-probability curves on the fixed
  101-point normalized-progress grid for the same keys.
* ``seed_player_index``: one row per available ``(seed, player_id)`` page.

The renderer completes the global row/column grid and leaves unobserved
combinations blank; this module emits only observed combinations. Ordering and
color metadata ride along in the result manifest so the report stage never reads
the catalogs or canonical tables to rebuild them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError

# The fixed normalized-progress grid every individual game curve is interpolated
# onto before averaging (0 to 1 inclusive).
GRID_POINTS = 101

# Final victory-focus labels in tie-break order; the first maximum wins.
FOCUS_ORDER = ["Domination", "Culture", "Diplomatic", "Science"]
FOCUS_RATIO_COLUMNS = {
    "Domination": "domination_ratio",
    "Culture": "culture_ratio",
    "Diplomatic": "diplomatic_ratio",
    "Science": "science_ratio",
}

SUMMARY_COLUMNS = [
    "seed", "player_id", "strategist", "condition", "player_type", "experiment",
    "civilization", "run_count", "mean_weighted_victory_probability",
    "mean_adjusted_strength", "matched_vanilla_adjusted_strength",
    "adjusted_strength_difference", "has_matched_vanilla", "dominant_focus",
    "dominant_focus_pct", "domination_focus_pct", "culture_focus_pct",
    "diplomatic_focus_pct", "science_focus_pct",
]
PROBABILITY_COLUMNS = [
    "seed", "player_id", "strategist", "condition", "turn_progress",
    "mean_predicted_win_probability", "n_runs",
]
INDEX_COLUMNS = [
    "seed", "player_id", "civilization", "n_civilizations", "run_count",
    "has_matched_vanilla", "has_probability",
]


def _require_columns(df: pd.DataFrame, columns: list[str], source: str, stage_id: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise AnalysisError(
            f"performance.controlled_seed_report '{stage_id}': {source} is missing "
            f"required column(s) {missing}."
        )


def _unique_per_game_player(df: pd.DataFrame, source: str, stage_id: str) -> pd.DataFrame:
    """Enforce one record per ``(game_id, player_id)``; duplicates are an error."""
    dup = df.duplicated(["game_id", "player_id"], keep=False)
    if dup.any():
        bad = df[dup][["game_id", "player_id"]].drop_duplicates().head(5)
        keys = ", ".join(f"({r.game_id}, {r.player_id})" for r in bad.itertuples(index=False))
        raise AnalysisError(
            f"performance.controlled_seed_report '{stage_id}': {source} has duplicate "
            f"records per (game_id, player_id), e.g. {keys}. Duplicate source rows are "
            "an analysis error, not extra observations."
        )
    return df


def _condition_display(spec, condition_key: str, vanilla_label: str) -> str:
    if condition_key == "vanilla":
        return vanilla_label
    if condition_key == "base":
        return spec.base_label
    return condition_key.lstrip("-")


def _condition_order(spec, observed_keys: set[str], vanilla_label: str) -> list[str]:
    """Display order: base label, configured suffix order, then unconfigured observed."""
    order = [_condition_display(spec, "base", vanilla_label)]
    order += [_condition_display(spec, key, vanilla_label) for key in spec.suffixes]
    known = {"vanilla"} | {"base"} | set(spec.suffixes)
    extra = sorted({k for k in observed_keys if k not in known})
    order += [_condition_display(spec, key, vanilla_label) for key in extra]
    seen: set[str] = set()
    return [label for label in order if not (label in seen or seen.add(label))]


def _strategist_order(catalog, strategists: list[str]) -> list[str]:
    """Catalog order first (by resolved model id), then unlisted identities lexically."""
    catalog_ids = [m["id"] for m in catalog.strategist_models()]
    rank = {model_id: idx for idx, model_id in enumerate(catalog_ids)}

    def sort_key(name: str):
        model_id = catalog.split_player_type(name)["model_id"]
        position = rank.get(str(model_id))
        if position is None:
            return (1, str(model_id), name)
        return (0, position, name)

    return sorted(set(strategists), key=sort_key)


def _dominant_focus(row: dict) -> tuple[str, float]:
    """Largest focus mean; exact ties resolve in FOCUS_ORDER (deterministic)."""
    best_label = FOCUS_ORDER[0]
    best_value = float(row[f"{FOCUS_ORDER[0].lower()}_focus_pct"])
    for label in FOCUS_ORDER[1:]:
        value = float(row[f"{label.lower()}_focus_pct"])
        if value > best_value:
            best_label, best_value = label, value
    return best_label, best_value


def _aggregate_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """One summary row per ``(seed, player_id, strategist, condition)`` key."""
    out: list[dict] = []
    for key, grp in rows.groupby(
        ["seed", "player_id", "strategist", "condition"], sort=True
    ):
        seed, player_id, strategist, condition = key
        record = {
            "seed": int(seed),
            "player_id": int(player_id),
            "strategist": strategist,
            "condition": condition,
            "player_type": "|".join(sorted(grp["player_type"].astype(str).unique())),
            "experiment": "|".join(sorted(grp["experiment"].astype(str).unique())),
            "civilization": ", ".join(sorted(grp["civilization"].astype(str).unique())),
            "run_count": int(grp["game_id"].nunique()),
            "mean_weighted_victory_probability": round(
                float(grp["weighted_strength"].mean()), 6
            ),
            "mean_adjusted_strength": round(float(grp["adjusted_strength"].mean()), 6),
        }
        for label, column in FOCUS_RATIO_COLUMNS.items():
            record[f"{label.lower()}_focus_pct"] = round(
                float(grp[column].mean()) * 100.0, 4
            )
        dominant, dominant_pct = _dominant_focus(record)
        record["dominant_focus"] = dominant
        record["dominant_focus_pct"] = dominant_pct
        out.append(record)
    return pd.DataFrame(out, columns=SUMMARY_COLUMNS)


def _interpolate_curves(
    runs: pd.DataFrame, grid: np.ndarray
) -> dict[tuple, list[tuple[float, float, int]]]:
    """Mean interpolated curve per ``(seed, player_id, strategist, condition)`` key.

    Each individual game curve is linearly interpolated onto the fixed grid inside
    its own observed progress range only; no extrapolation and no endpoint
    holding. Each grid point averages exactly the runs that cover it.
    """
    curves: dict[tuple, list[tuple[float, float, int]]] = {}
    run_groups = runs.groupby(
        ["seed", "player_id", "strategist", "condition", "game_id"],
        sort=True,
    )
    sums: dict[tuple, np.ndarray] = {}
    counts: dict[tuple, np.ndarray] = {}
    for key, grp in run_groups:
        seed, player_id, strategist, condition = key[0], key[1], key[2], key[3]
        if grp["turn_progress"].isna().all():
            continue
        points = grp.dropna(subset=["turn_progress", "predicted_win_probability"])
        if points.empty:
            continue
        xs = points["turn_progress"].to_numpy(dtype=float)
        ys = points["predicted_win_probability"].to_numpy(dtype=float)
        order = np.argsort(xs, kind="stable")
        xs, ys = xs[order], ys[order]
        if xs.size < 2:
            # A single observation has no range to interpolate inside; it cannot
            # contribute to the grid without extrapolating.
            continue
        covered = (grid >= xs[0]) & (grid <= xs[-1])
        if not covered.any():
            continue
        cell = (int(seed), int(player_id), strategist, condition)
        if cell not in sums:
            sums[cell] = np.zeros(grid.size)
            counts[cell] = np.zeros(grid.size, dtype=int)
        sums[cell][covered] += np.interp(grid[covered], xs, ys)
        counts[cell][covered] += 1
    for cell, count in counts.items():
        mask = count > 0
        values = sums[cell][mask] / count[mask]
        curves[cell] = [
            (float(grid[i]), round(float(v), 6), int(count[i]))
            for i, v in zip(np.nonzero(mask)[0], values)
        ]
    return curves


class PerformanceControlledSeedReport(Analysis):
    module = "performance.controlled_seed_report"
    friendly_name = "Controlled seed comparison"
    description = (
        "Aggregates controlled-seed games by seed and final seat into the tables "
        "behind the dedicated controlled-seed HTML report."
    )
    report_defaults = {"tables": [], "figures": []}

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        stage = self.stage_id
        estimators = ctx.uses_estimators()
        if len(estimators) != 1:
            raise AnalysisError(
                f"performance.controlled_seed_report '{stage}': requires exactly one "
                f"estimator in uses.estimators (got {len(estimators)})."
            )
        estimator_id = estimators[0]
        table_id = ctx.strength_table_id()

        spec = ctx.condition_pairing()
        if spec is None:
            raise AnalysisError(
                f"performance.controlled_seed_report '{stage}': condition pairing must "
                "be enabled for this report (presentation.condition_pairing or "
                "params.condition_pairing)."
            )

        adjust_stage = next((s for s in ctx.config.adjust if s.id == table_id), None)
        baseline_experiment = (
            (adjust_stage.raw.get("params") or {}).get("baseline_experiment")
            if adjust_stage is not None
            else None
        )
        if not baseline_experiment:
            raise AnalysisError(
                f"performance.controlled_seed_report '{stage}': the configured strength "
                f"stage '{table_id}' must set params.baseline_experiment; it is the sole "
                "dedicated Vanilla source for this report."
            )
        baseline_experiment = str(baseline_experiment)
        vanilla_label = ctx.catalog.vanilla_label

        rows = self._load_rows(ctx, table_id, baseline_experiment)
        rows["strategist"], rows["condition_key"] = self._identities(
            ctx, spec, rows, vanilla_label, baseline_experiment
        )
        rows["condition"] = [
            _condition_display(spec, key, vanilla_label)
            for key in rows["condition_key"]
        ]

        baseline_mask = rows["experiment"].astype(str) == baseline_experiment
        baseline_rows = rows[baseline_mask]
        has_baseline = not baseline_rows.empty

        summary = _aggregate_summary(rows)
        summary = self._attach_baseline(summary, baseline_rows, vanilla_label)

        curves = self._build_curves(ctx, estimator_id, rows)
        probability = self._probability_table(curves)
        index = self._index_table(rows, probability, vanilla_label)

        strategist_order = _strategist_order(
            ctx.catalog, [s for s in summary["strategist"].unique() if s != vanilla_label]
        )
        condition_keys = set(rows["condition_key"].astype(str))
        condition_order = _condition_order(spec, condition_keys, vanilla_label)
        colors = self._strategist_colors(ctx, strategist_order, vanilla_label)

        n_seeds = int(rows["seed"].nunique())
        n_players = int(rows["player_id"].nunique())
        n_combos = int(
            summary.loc[summary["strategist"] != vanilla_label].shape[0]
        )
        summary_text = (
            f"Controlled seed comparison covers {n_seeds} seed(s) and {n_players} final "
            f"seat(s) with {n_combos} strategist-condition combination(s) plus the "
            f"dedicated Vanilla baseline '{baseline_experiment}' from "
            f"{int(rows['game_id'].nunique())} controlled game(s)."
        )
        no_baseline = int((~index["has_matched_vanilla"].astype(bool)).sum())
        no_probability = int((~index["has_probability"].astype(bool)).sum())
        notes = []
        if not has_baseline:
            notes.append(
                f"the dedicated baseline experiment '{baseline_experiment}' has no "
                "controlled rows"
            )
        elif no_baseline:
            notes.append(f"{no_baseline} seed-player pair(s) lack a matched Vanilla baseline")
        if no_probability:
            notes.append(
                f"{no_probability} seed-player pair(s) have no usable prediction rows"
            )
        if notes:
            summary_text += " Note: " + "; ".join(notes) + "."

        metadata = {
            "strategist_order": strategist_order,
            "condition_order": condition_order,
            "strategist_colors": colors,
            "base_label": spec.base_label,
            "vanilla_label": vanilla_label,
            "focus_order": list(FOCUS_ORDER),
            "grid_points": GRID_POINTS,
            "estimator": estimator_id,
            "strength_table": table_id,
            "baseline_experiment": baseline_experiment,
            "has_baseline": has_baseline,
            "seeds": sorted(int(s) for s in rows["seed"].unique()),
            "player_ids": sorted(int(p) for p in rows["player_id"].unique()),
        }
        return AnalysisResult(
            tables={
                "seed_player_summary": summary,
                "seed_player_probability": probability,
                "seed_player_index": index,
            },
            summary=summary_text,
            metadata=metadata,
        )

    # ── input preparation ──────────────────────────────────────────────────────
    def _load_rows(
        self, ctx: AnalysisContext, table_id: str, baseline_experiment: str
    ) -> pd.DataFrame:
        stage = self.stage_id
        games = ctx.load_table("games")
        _require_columns(
            games, ["game_id", "experiment", "seed", "seating_rotation"], "games", stage
        )
        games = games.copy()
        games["game_id"] = games["game_id"].astype(str)
        games["seed"] = (
            pd.to_numeric(games["seed"], errors="coerce").fillna(-1).astype(int)
        )
        games["seating_rotation"] = (
            pd.to_numeric(games["seating_rotation"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )
        controlled = games[
            (games["seed"] != -1) & (games["seating_rotation"] != -1)
        ]
        if controlled.empty:
            raise AnalysisError(
                f"performance.controlled_seed_report '{stage}': no controlled rows "
                "(every game has seed=-1 or seating_rotation=-1); this report needs a "
                "controlled design."
            )
        controlled = controlled[
            ["game_id", "experiment", "seed", "seating_rotation"]
        ]

        panel = ctx.load_table("panel")
        panel_columns = [
            "game_id", "player_id", "player_type", "civilization",
            *FOCUS_RATIO_COLUMNS.values(),
        ]
        _require_columns(panel, panel_columns, "panel", stage)
        panel = panel[panel_columns].copy()
        panel["game_id"] = panel["game_id"].astype(str)
        panel["player_id"] = pd.to_numeric(panel["player_id"], errors="raise").astype(int)
        panel = panel[panel["game_id"].isin(set(controlled["game_id"]))]
        panel = _unique_per_game_player(panel, "panel", stage)

        strength = ctx.load_table(table_id)
        strength_columns = ["game_id", "player_id", "weighted_strength", "adjusted_strength"]
        _require_columns(strength, strength_columns, f"strength table '{table_id}'", stage)
        strength = strength[strength_columns].copy()
        strength["game_id"] = strength["game_id"].astype(str)
        strength["player_id"] = pd.to_numeric(
            strength["player_id"], errors="raise"
        ).astype(int)
        strength = strength[strength["game_id"].isin(set(controlled["game_id"]))]
        strength = _unique_per_game_player(strength, "strength table", stage)

        rows = (
            controlled.merge(panel, on="game_id", how="inner")
            .merge(strength, on=["game_id", "player_id"], how="inner")
        )
        if rows.empty:
            raise AnalysisError(
                f"performance.controlled_seed_report '{stage}': no controlled rows "
                "survived the panel/strength join; run extract and the adjust stage first."
            )
        for column in FOCUS_RATIO_COLUMNS.values():
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
        for column in ("weighted_strength", "adjusted_strength"):
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
        return rows

    def _identities(
        self,
        ctx: AnalysisContext,
        spec,
        rows: pd.DataFrame,
        vanilla_label: str,
        baseline_experiment: str,
    ) -> tuple[list[str], list[str]]:
        """Split player_type into strategist and condition keys.

        Dedicated baseline rows bypass condition splitting: the catalog pools
        Vanilla across condition suffixes, so every row of the baseline experiment
        becomes ``strategist = condition = Vanilla``.
        """
        strategists: list[str] = []
        condition_keys: list[str] = []
        baseline_mask = rows["experiment"].astype(str) == baseline_experiment
        for (_, row), baseline in zip(rows.iterrows(), baseline_mask):
            if baseline:
                strategists.append(vanilla_label)
                condition_keys.append("vanilla")
                continue
            base, suffix = ctx.catalog.split_condition_suffix(
                row["player_type"], list(spec.suffixes)
            )
            strategists.append(str(base))
            condition_keys.append("base" if not suffix else suffix)
        return strategists, condition_keys

    def _attach_baseline(
        self, summary: pd.DataFrame, baseline_rows: pd.DataFrame, vanilla_label: str
    ) -> pd.DataFrame:
        """Join the matched Vanilla adjusted strength and its difference.

        The matched value is the baseline mean for the ``(seed, player_id)`` pair,
        averaged over all baseline rotations and repeated games. Treatment rows
        keep their own run counts; the difference is blank for Vanilla itself and
        when the matched baseline is unavailable.
        """
        out = summary.copy()
        if baseline_rows.empty:
            out["matched_vanilla_adjusted_strength"] = float("nan")
            out["adjusted_strength_difference"] = float("nan")
            out["has_matched_vanilla"] = False
            return out[SUMMARY_COLUMNS]
        baseline_means = {
            (int(seed), int(player_id)): float(mean)
            for (seed, player_id), mean in baseline_rows.groupby(
                ["seed", "player_id"], sort=True
            )["adjusted_strength"].mean().items()
        }
        matched = [
            baseline_means.get((int(seed), int(player_id)))
            for seed, player_id in zip(out["seed"], out["player_id"])
        ]
        out["matched_vanilla_adjusted_strength"] = [
            round(value, 6) if value is not None and value == value else float("nan")
            for value in matched
        ]
        # A mean over no finite adjusted_strength values is not a usable baseline.
        out["has_matched_vanilla"] = [
            value is not None and value == value for value in matched
        ]
        is_vanilla = out["strategist"] == vanilla_label
        matched_series = pd.to_numeric(out["matched_vanilla_adjusted_strength"])
        difference = (out["mean_adjusted_strength"] - matched_series).round(6)
        out["adjusted_strength_difference"] = difference.where(
            matched_series.notna() & ~is_vanilla
        )
        out.loc[is_vanilla, "matched_vanilla_adjusted_strength"] = float("nan")
        return out[SUMMARY_COLUMNS]

    # ── probability curves ─────────────────────────────────────────────────────
    def _build_curves(
        self, ctx: AnalysisContext, estimator_id: str, rows: pd.DataFrame
    ) -> dict[tuple, list[tuple[float, float, int]]]:
        stage = self.stage_id
        pred = ctx.load_predictions(estimator_id)
        _require_columns(
            pred,
            ["game_id", "player_id", "turn_progress", "predicted_win_probability"],
            f"estimator '{estimator_id}' predictions",
            stage,
        )
        pred = pred[["game_id", "player_id", "turn_progress", "predicted_win_probability"]]
        pred = pred.copy()
        pred["game_id"] = pred["game_id"].astype(str)
        pred["player_id"] = pd.to_numeric(pred["player_id"], errors="raise").astype(int)
        pred["turn_progress"] = pd.to_numeric(pred["turn_progress"], errors="coerce")
        pred["predicted_win_probability"] = pd.to_numeric(
            pred["predicted_win_probability"], errors="coerce"
        )
        pred = pred.dropna(
            subset=["turn_progress", "predicted_win_probability"]
        ).drop_duplicates()
        pred = pred[pred["game_id"].isin(set(rows["game_id"]))]
        key = ["game_id", "player_id", "turn_progress"]
        conflicts = pred[pred.duplicated(key, keep=False)]
        if not conflicts.empty:
            bad = conflicts[key].drop_duplicates().head(5)
            keys = ", ".join(
                f"({r.game_id}, {r.player_id}, {r.turn_progress})"
                for r in bad.itertuples(index=False)
            )
            raise AnalysisError(
                f"performance.controlled_seed_report '{stage}': conflicting duplicate "
                f"prediction points at (game_id, player_id, turn_progress), e.g. {keys}."
            )
        runs = rows.merge(pred, on=["game_id", "player_id"], how="inner")
        grid = np.round(np.arange(GRID_POINTS) / (GRID_POINTS - 1), 10)
        return _interpolate_curves(runs, grid)

    def _probability_table(
        self, curves: dict[tuple, list[tuple[float, float, int]]]
    ) -> pd.DataFrame:
        records = []
        for (seed, player_id, strategist, condition), points in curves.items():
            for turn_progress, mean_predicted, n_runs in points:
                records.append({
                    "seed": seed,
                    "player_id": player_id,
                    "strategist": strategist,
                    "condition": condition,
                    "turn_progress": round(turn_progress, 2),
                    "mean_predicted_win_probability": mean_predicted,
                    "n_runs": n_runs,
                })
        return pd.DataFrame(records, columns=PROBABILITY_COLUMNS).sort_values(
            ["seed", "player_id", "strategist", "condition", "turn_progress"],
            kind="mergesort",
        ).reset_index(drop=True)

    def _index_table(
        self,
        rows: pd.DataFrame,
        probability: pd.DataFrame,
        vanilla_label: str,
    ) -> pd.DataFrame:
        has_probability_pairs = set(
            zip(
                probability["seed"].astype(int),
                probability["player_id"].astype(int),
            )
        ) if not probability.empty else set()
        records = []
        for (seed, player_id), grp in rows.groupby(["seed", "player_id"], sort=True):
            civilizations = sorted(grp["civilization"].astype(str).unique())
            has_vanilla = bool(
                (grp["strategist"] == vanilla_label).any()
            )
            records.append({
                "seed": int(seed),
                "player_id": int(player_id),
                "civilization": ", ".join(civilizations),
                "n_civilizations": len(civilizations),
                "run_count": int(grp["game_id"].nunique()),
                "has_matched_vanilla": has_vanilla,
                "has_probability": (int(seed), int(player_id)) in has_probability_pairs,
            })
        return pd.DataFrame(records, columns=INDEX_COLUMNS).sort_values(
            ["seed", "player_id"], kind="mergesort"
        ).reset_index(drop=True)

    def _strategist_colors(
        self, ctx: AnalysisContext, strategist_order: list[str], vanilla_label: str
    ) -> dict[str, str]:
        from ...plotting.styles import get_player_color

        colors = {vanilla_label: get_player_color(ctx.catalog, vanilla_label)}
        for name in strategist_order:
            colors[name] = get_player_color(ctx.catalog, name)
        return colors
