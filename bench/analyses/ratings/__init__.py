"""Ratings analyses: Bradley-Terry, Plackett-Luce, matchups.

strategy-Elo and bootstrap are NOT separate modules — strategy-Elo is
``group_by:["player_type","strategy"]`` and bootstrap is the shared ``bootstrap``
param (see :mod:`bench.analyses.ratings.base` / :mod:`bench.analyses.ratings.bootstrap`).
Pulls ``Rscript`` (BT/PL) lazily at run time, never on the config/dry-run path.
"""

from __future__ import annotations

from .bradley_terry import RatingsBradleyTerry
from .matchups import RatingsMatchups
from .plackett_luce import RatingsPlackettLuce

__all__ = ["RatingsBradleyTerry", "RatingsMatchups", "RatingsPlackettLuce"]
