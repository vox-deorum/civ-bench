"""Analysis-module registry — name → :class:`Analysis` subclass (stage 4).

The JSON ``analyses[].module`` string selects an implemented module here. The
schema registry (``bench.config.schema.ANALYSIS_MODULES``) is broader: it also
lists the **reserved optional** modules (``enabled:false`` in
``benchmark.full.template.json``) so the config validates, but those have no class
yet — selecting an unimplemented module at run time fails loud here. Importing
this module pulls matplotlib/statsmodels (and, for ratings, calls Rscript), so it
is loaded only from the CLI run path, never the import-light config layer.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Analysis
from .calibration import (
    CalibrationCellBaseline,
    CalibrationCivEffects,
    CalibrationLossByProgress,
    CalibrationReliability,
)
from .errors import AnalysisError
from .exploratory import ExploratoryCostVsRating, ExploratoryModelTokenCosts
from .performance import (
    PerformanceControlledSeedReport,
    PerformanceExperimentCompleteness,
    PerformanceScoreRatio,
    PerformanceStrengthPanel,
    PerformanceTurnPredicted,
)
from .prediction import PredictionCompare, PredictionEvaluate
from .ratings import (
    RatingsBradleyTerry,
    RatingsMatchups,
    RatingsOutcomeMatchups,
    RatingsPlackettLuce,
)

ANALYSIS_REGISTRY: Dict[str, Type[Analysis]] = {
    # ratings.*
    "ratings.bradley_terry": RatingsBradleyTerry,
    "ratings.plackett_luce": RatingsPlackettLuce,
    "ratings.matchups": RatingsMatchups,
    "ratings.outcome_matchups": RatingsOutcomeMatchups,
    # prediction.*
    "prediction.evaluate": PredictionEvaluate,
    "prediction.compare": PredictionCompare,
    # calibration.*
    "calibration.reliability": CalibrationReliability,
    "calibration.loss_by_progress": CalibrationLossByProgress,
    "calibration.civ_effects": CalibrationCivEffects,
    "calibration.cell_baseline": CalibrationCellBaseline,
    # performance.*
    "performance.experiment_completeness": PerformanceExperimentCompleteness,
    "performance.score_ratio": PerformanceScoreRatio,
    "performance.strength_panel": PerformanceStrengthPanel,
    "performance.turn_predicted": PerformanceTurnPredicted,
    "performance.controlled_seed_report": PerformanceControlledSeedReport,
    # exploratory.*
    "exploratory.model_token_costs": ExploratoryModelTokenCosts,
    "exploratory.cost_vs_rating": ExploratoryCostVsRating,
}


def get_analysis(name: str) -> Type[Analysis]:
    """Return the analysis class registered under ``name`` (fail loud otherwise)."""
    if name not in ANALYSIS_REGISTRY:
        from ..config import schema as S  # import-light; lists every reserved module

        if name in S.ANALYSIS_MODULES:
            raise AnalysisError(
                f"analysis module '{name}' is registry-reserved (optional) but not "
                f"implemented yet; it ships only as enabled:false. Enable it once a "
                f"stage lands its implementation."
            )
        available = ", ".join(sorted(ANALYSIS_REGISTRY))
        raise AnalysisError(f"unknown analysis module '{name}'. Available: {available}")
    return ANALYSIS_REGISTRY[name]


def list_analyses() -> list[str]:
    return sorted(ANALYSIS_REGISTRY)
