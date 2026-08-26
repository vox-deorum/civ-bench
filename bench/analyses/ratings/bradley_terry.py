"""``ratings.bradley_terry`` — Bradley-Terry MLE Elo (R ``BradleyTerry2``).

The ``weighted`` param toggles the score-margin weighting: ``true`` (default) uses
the auto-detected median pairwise margin; ``false`` flattens the per-pair weights
to ~uniform. For the bootstrap the margin is frozen on the point sample so margin
auto-detection is not a hidden source of CI variability (legacy parity).
"""

from __future__ import annotations

import pandas as pd

from .base import RatingsAnalysis, _STRENGTH_COL
from .bootstrap import compute_margin
from .r_interop import calculate_ratings_bt

_FLAT_MARGIN = 1e9  # weights = 1 + log1p(diff/margin) ≈ 1 → ~unweighted BT


class RatingsBradleyTerry(RatingsAnalysis):
    module = "ratings.bradley_terry"
    friendly_name = "Bradley-Terry ratings"
    description = "Fitted Elo-style strength ratings from a paired game-comparison (Bradley-Terry) model."
    report_defaults = {"tables": [], "figures": ["ratings"]}

    def _margin_for(self, strength_df: pd.DataFrame):
        if not bool(self.params.get("weighted", True)):
            return _FLAT_MARGIN
        return None  # auto-detect (median pairwise margin)

    def _calculate(self, strength_df: pd.DataFrame, reference: str) -> pd.DataFrame:
        return calculate_ratings_bt(strength_df, margin=self._margin_for(strength_df), reference=reference)

    def _frozen_calculator(self, strength_df: pd.DataFrame, reference: str):
        if bool(self.params.get("weighted", True)):
            frozen = compute_margin(strength_df, _STRENGTH_COL)
        else:
            frozen = _FLAT_MARGIN
        return lambda df: calculate_ratings_bt(df, margin=frozen, reference=reference)
