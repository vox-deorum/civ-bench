"""``ratings.plackett_luce`` — Plackett-Luce MLE Elo (R ``PlackettLuce``).

Deterministic full-ranking MLE; no margin/weighting. The bundled R script accepts
a reference player_type (generalized from the legacy hardcoded ``Vanilla``) so the
per-strategy composite path can re-center on its own Vanilla composite.
"""

from __future__ import annotations

import pandas as pd

from .base import RatingsAnalysis
from .r_interop import calculate_ratings_pl


class RatingsPlackettLuce(RatingsAnalysis):
    module = "ratings.plackett_luce"
    report_defaults = {"tables": [], "figures": ["ratings"]}

    def _calculate(self, strength_df: pd.DataFrame, reference: str) -> pd.DataFrame:
        return calculate_ratings_pl(strength_df, reference=reference)
