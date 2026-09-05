"""Python ↔ R bridge for the MLE rating models (ported from ``ratings/*.py``).

Consolidates ``bradley_terry.py`` + ``plackett_luce.py``: assign per-game slot
ids, write the input CSV, call the bundled ``.R`` script via the cross-platform
:func:`bench.rscript.find_rscript` (PATH / ``CIV_BENCH_RSCRIPT`` only; the old
hardcoded ``C:``/``D:\\Program Files\\R`` scan is dropped), read the result, and
add the Elo conversion (``1500 + 400*log10(worth)``).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...rscript import find_rscript
from ..errors import AnalysisError

_R_DIR = Path(__file__).resolve().parent
RATING_COL = "adjusted_strength"


def _assign_slot_ids(strength_df: pd.DataFrame) -> pd.DataFrame:
    """Per game, suffix duplicate player_types with ``_0``, ``_1``, … (legacy convention)."""
    r_input = strength_df[["game_id", "player_type", RATING_COL]].copy()
    slot_ids = []
    for _, game in r_input.groupby("game_id", sort=False):
        counters: dict[str, int] = {}
        for idx in game.index:
            pt = game.at[idx, "player_type"]
            count = counters.get(pt, 0)
            slot_ids.append(f"{pt}_{count}")
            counters[pt] = count + 1
    r_input["slot_id"] = slot_ids
    return r_input


def _add_elo(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["elo"] = 1500 + 400 * np.log10(results["worth"])
    results["se_elo"] = 400 / np.log(10) * results["se_log_worth"]
    results["mu"] = results["log_worth"]
    results["sigma"] = results["se_log_worth"]
    return results.sort_values("elo", ascending=False).reset_index(drop=True)


def _run_r(
    strength_df: pd.DataFrame,
    script: str,
    extra_args: list[str],
    timeout: int = 180,
) -> pd.DataFrame:
    rscript_exe = find_rscript(required=False)
    if rscript_exe is None:
        raise AnalysisError(
            "Rscript not found. Install R (https://cran.r-project.org/) and ensure "
            "Rscript is on PATH, or set CIV_BENCH_RSCRIPT to its full path. The "
            "R-backed ratings analyses (BradleyTerry2 / PlackettLuce) require it."
        )
    r_script = str(_R_DIR / script)
    r_input = _assign_slot_ids(strength_df)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as fh:
        input_path = fh.name
        r_input.to_csv(fh, index=False)
    output_path = input_path.replace(".csv", "_output.csv")
    try:
        cmd = [rscript_exe, r_script, input_path, output_path, *extra_args]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise AnalysisError(
                f"R rating script '{script}' failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
            )
        results = pd.read_csv(output_path)
        return _add_elo(results)
    finally:
        for path in (input_path, output_path,
                     output_path.replace(".csv", "_diagnostics.csv"),
                     output_path.replace(".csv", "_slots.csv")):
            if os.path.exists(path):
                os.unlink(path)


def calculate_ratings_bt(
    strength_df: pd.DataFrame,
    margin: Optional[float] = None,
    reference: Optional[str] = None,
) -> pd.DataFrame:
    """Bradley-Terry MLE ratings via R ``BradleyTerry2`` (pairwise, score-weighted)."""
    ref = reference or "Vanilla"
    margin_arg = "NA" if margin is None else str(margin)
    return _run_r(strength_df, "bradley_terry.R", [margin_arg, ref])


def calculate_ratings_pl(
    strength_df: pd.DataFrame,
    reference: Optional[str] = None,
) -> pd.DataFrame:
    """Plackett-Luce MLE ratings via R ``PlackettLuce`` (full-ranking MLE)."""
    ref = reference or "Vanilla"
    return _run_r(strength_df, "plackett_luce.R", [ref])
