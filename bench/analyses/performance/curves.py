"""Shared victory-probability curve helpers for the performance analyses.

``performance.controlled_seed_report`` (one curve per seed, seat, strategist,
and condition) and ``performance.turn_predicted`` (one curve per strategist and
condition over every game) build their curves the same way: load one
estimator's per-turn predictions, split each ``player_type`` into a strategist
and a condition, interpolate every run onto a fixed normalized-progress grid,
and average the runs that cover each grid point. This module holds those steps
so the two report charts always mean the same thing. :func:`adjusted_curves`
builds the controlled counterpart through the strength stage's own
adjustment against the matched VPAI baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import AnalysisError

# The fixed normalized-progress grid every individual game curve is interpolated
# onto before averaging (0 to 1 inclusive).
GRID_POINTS = 101

PREDICTION_COLUMNS = ["game_id", "player_id", "turn_progress", "predicted_win_probability"]


def progress_grid() -> np.ndarray:
    """The fixed 0 to 1 grid, rounded so grid values compare exactly."""
    return np.round(np.arange(GRID_POINTS) / (GRID_POINTS - 1), 10)


def split_identities(
    catalog,
    spec,
    player_types: pd.Series,
    vanilla_label: str,
    reference_mask: pd.Series,
) -> tuple[list[str], list[str]]:
    """Split each ``player_type`` into a strategist and a condition key.

    Rows in ``reference_mask`` are the dedicated VPAI self-play reference and
    become ``strategist = Vanilla`` with the ``"vanilla"`` condition key. With a
    pairing ``spec`` the other rows split through the catalog's suffix rules
    (``"base"`` when no suffix matches); without one every row keeps its whole
    identity with an empty condition key.
    """
    strategists: list[str] = []
    condition_keys: list[str] = []
    for player_type, reference in zip(player_types, reference_mask):
        if reference:
            strategists.append(vanilla_label)
            condition_keys.append("vanilla")
            continue
        if spec is None:
            strategists.append(str(player_type))
            condition_keys.append("")
            continue
        base, suffix = catalog.split_condition_suffix(player_type, list(spec.suffixes))
        strategists.append(str(base))
        condition_keys.append("base" if not suffix else suffix)
    return strategists, condition_keys


def load_curve_predictions(ctx, estimator_id: str, game_ids: set, where: str) -> pd.DataFrame:
    """One estimator's cleaned prediction points for the given games."""
    pred = clean_curve_predictions(ctx.load_predictions(estimator_id), estimator_id, where)
    return pred[pred["game_id"].isin(game_ids)]


def clean_curve_predictions(
    pred: pd.DataFrame, estimator_id: str, where: str, keep: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Prediction points ready for :func:`interpolate_curves`.

    Numeric columns are coerced, incomplete points and exact duplicates are
    dropped, and two different probabilities at the same
    ``(game_id, player_id, turn_progress)`` are an error. ``keep`` names extra
    columns carried through unchanged; ``where`` names the calling stage in
    error messages.
    """
    missing = [c for c in PREDICTION_COLUMNS if c not in pred.columns]
    if missing:
        raise AnalysisError(
            f"{where}: estimator '{estimator_id}' predictions is missing "
            f"required column(s) {missing}."
        )
    pred = pred[[*PREDICTION_COLUMNS, *keep]].copy()
    pred["game_id"] = pred["game_id"].astype(str)
    pred["player_id"] = pd.to_numeric(pred["player_id"], errors="raise").astype(int)
    pred["turn_progress"] = pd.to_numeric(pred["turn_progress"], errors="coerce")
    pred["predicted_win_probability"] = pd.to_numeric(
        pred["predicted_win_probability"], errors="coerce"
    )
    pred = pred.dropna(
        subset=["turn_progress", "predicted_win_probability"]
    ).drop_duplicates(subset=PREDICTION_COLUMNS)
    key = ["game_id", "player_id", "turn_progress"]
    conflicts = pred[pred.duplicated(key, keep=False)]
    if not conflicts.empty:
        bad = conflicts[key].drop_duplicates().head(5)
        keys = ", ".join(
            f"({r.game_id}, {r.player_id}, {r.turn_progress})"
            for r in bad.itertuples(index=False)
        )
        raise AnalysisError(
            f"{where}: conflicting duplicate prediction points at "
            f"(game_id, player_id, turn_progress), e.g. {keys}."
        )
    return pred


def _run_curves(runs: pd.DataFrame, grid: np.ndarray, keys: list[str]):
    """Yield ``(cell, (game_id, player_id), covered, values)`` for every run.

    A run is one player's curve in one game (``game_id`` and ``player_id``),
    even when a cell holds several seats of the same game, such as the VPAI
    seats of one match. Each run is linearly interpolated onto the fixed grid
    inside its own observed progress range only; no extrapolation and no
    endpoint holding. ``covered`` masks the grid points the run spans and
    ``values`` holds the run's interpolated values there. Integer-like key
    values are returned as ``int`` so cells compare and sort deterministically.
    """
    # Grouping by game alone would merge seats into one run with repeated
    # progress values, and np.interp would then pick one seat at exact grid hits.
    run_keys = [*keys, *(c for c in ("game_id", "player_id") if c not in keys)]
    game_at, player_at = run_keys.index("game_id"), run_keys.index("player_id")
    for key, grp in runs.groupby(run_keys, sort=True):
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
        cell = tuple(
            int(value) if isinstance(value, (int, np.integer)) else value
            for value in key[: len(keys)]
        )
        run = (str(key[game_at]), int(key[player_at]))
        yield cell, run, covered, np.interp(grid[covered], xs, ys)


class _CurveSums:
    """Per-cell running sums and run counts on the grid."""

    def __init__(self, size: int):
        self.size = size
        self.sums: dict[tuple, np.ndarray] = {}
        self.counts: dict[tuple, np.ndarray] = {}

    def add(self, cell: tuple, covered: np.ndarray, values: np.ndarray) -> None:
        if cell not in self.sums:
            self.sums[cell] = np.zeros(self.size)
            self.counts[cell] = np.zeros(self.size, dtype=int)
        self.sums[cell][covered] += values
        self.counts[cell][covered] += 1

    def add_point(self, cell: tuple, i: int, value: float) -> None:
        self.add(cell, slice(i, i + 1), value)

    def points(self, grid: np.ndarray) -> dict[tuple, list[tuple[float, float, int]]]:
        curves = {}
        for cell, count in self.counts.items():
            mask = count > 0
            values = self.sums[cell][mask] / count[mask]
            curves[cell] = [
                (float(grid[i]), round(float(v), 6), int(count[i]))
                for i, v in zip(np.nonzero(mask)[0], values)
            ]
        return curves


def interpolate_curves(
    runs: pd.DataFrame, grid: np.ndarray, keys: list[str]
) -> dict[tuple, list[tuple[float, float, int]]]:
    """Mean interpolated curve per ``keys`` cell.

    Every run (:func:`_run_curves`) adds its interpolated values, and each grid
    point averages exactly the runs that cover it.
    """
    sums = _CurveSums(grid.size)
    for cell, _run, covered, values in _run_curves(runs, grid, keys):
        sums.add(cell, covered, values)
    return sums.points(grid)


# The run attributes the strength adjustment reads besides weighted_strength.
ADJUST_COLUMNS = ["experiment", "player_type", "seed", "seating_rotation"]


def adjusted_curves(
    runs: pd.DataFrame,
    pool: pd.DataFrame,
    grid: np.ndarray,
    keys: list[str],
    catalog,
    strength_params: dict | None,
) -> dict[tuple, list[tuple[float, float, int]]] | None:
    """Mean adjusted-strength curve per ``keys`` cell (the Relative view).

    At every grid point, each controlled run's interpolated probability takes
    the place of ``weighted_strength`` and goes through the strength stage's
    own controlled adjustment (:func:`bench.adjust.strength.cell_adjust_rows`)
    with that stage's ``params``: leader normalization when ``relative_to`` is
    set, the clipped logit, the matched start-cell VPAI baseline, the inverse
    logit of the difference, and ``post_cell_normalize``. 0.5 means level with
    the matched baseline. Winner enforcement is off, since the final outcome
    would otherwise lift the winner's whole curve. ``pool`` holds every seat
    of the candidate games (so the baseline and game leaders are complete) with
    :data:`ADJUST_COLUMNS`; ``runs`` names the displayed runs and their
    ``keys``. Runs without a matched baseline are left out. Returns ``None``
    when the strength params turn the cell adjustment off.
    """
    from ...adjust.strength import cell_adjust_rows

    seed = pool["seed"].fillna(-1)
    rotation = pool["seating_rotation"].fillna(-1)
    pool = pool[(seed != -1) & (rotation != -1)]
    if pool.empty:
        return None
    attributes = [c for c in (*ADJUST_COLUMNS, "model") if c in pool.columns]
    first = pool.drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"])
    run_ids, covers, values = [], [], []
    for _cell, run, covered, curve in _run_curves(pool, grid, []):
        full = np.full(grid.size, np.nan)
        full[covered] = curve
        run_ids.append(run)
        covers.append(covered)
        values.append(full)
    if not run_ids:
        return None
    index = pd.MultiIndex.from_tuples(run_ids, names=["game_id", "player_id"])
    frame = first.loc[index, attributes].reset_index()
    frame["controlled"] = True
    frame["is_winner"] = 0
    covers, values = np.array(covers), np.array(values)

    shown = runs.drop_duplicates(["game_id", "player_id"])
    cell_of = {
        (str(row[0]), int(row[1])): tuple(
            int(v) if isinstance(v, (int, np.integer)) else v for v in row[2:]
        )
        for row in shown[["game_id", "player_id", *keys]].itertuples(index=False, name=None)
    }
    cells = [cell_of.get(run) for run in run_ids]

    sums = _CurveSums(grid.size)
    for i in range(grid.size):
        at = covers[:, i]
        if not at.any():
            continue
        rows = frame[at].copy()
        rows["weighted_strength"] = values[at, i]
        adjusted = cell_adjust_rows(rows, catalog, strength_params, enforce_winner=False)
        if adjusted is None:
            return None
        for cell, value in zip(
            (c for c, hit in zip(cells, at) if hit), adjusted["adjusted_strength"].to_numpy()
        ):
            if cell is not None and np.isfinite(value):
                sums.add_point(cell, i, float(value))
    return sums.points(grid)


def curve_records(
    curves: dict[tuple, list[tuple[float, float, int]]],
    keys: list[str],
    value: str = "mean_predicted_win_probability",
) -> list[dict]:
    """Flatten curve output into one record per grid point, the mean in ``value``."""
    records = []
    for cell, points in curves.items():
        for turn_progress, mean_value, n_runs in points:
            record = dict(zip(keys, cell))
            record["turn_progress"] = round(turn_progress, 2)
            record[value] = mean_value
            record["n_runs"] = n_runs
            records.append(record)
    return records
