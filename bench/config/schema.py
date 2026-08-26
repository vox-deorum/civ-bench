"""Static schema constants for the benchmark run-spec.

This is the single place that enumerates the allowed keys, enum domains, and the
module/registry names a stage-0 validator can check without importing the (not
yet built) stage implementations. As later stages land their modules, the
registries here are the authoritative list `benchmark.md` must stay in parity
with (see AGENTS.md: update the schema in the same change as the module).
"""

from __future__ import annotations

# ── top level ──────────────────────────────────────────────────────────────
TOP_LEVEL_REQUIRED = ("name", "seed", "data", "analyses", "report")
TOP_LEVEL_OPTIONAL = (
    "friendly_name",
    "description",
    "output",
    "presentation",
    "catalogs",
    "filters",
    "groupings",
    "estimators",
    "adjust",
)
TOP_LEVEL_KEYS = set(TOP_LEVEL_REQUIRED) | set(TOP_LEVEL_OPTIONAL)

# ── output (§2.1) ──────────────────────────────────────────────────────────
OUTPUT_KEYS = {"root", "suffix"}
DEFAULT_OUTPUT_ROOT = "reports"
DEFAULT_OUTPUT_SUFFIX = ""

# ── presentation (§2.2) ─────────────────────────────────────────────────
PRESENTATION_KEYS = {"condition_pairing", "matchup_display"}
CONDITION_PAIRING_KEYS = {"enabled", "suffixes", "base_label", "sort_condition"}
CONDITION_PAIRING_OVERRIDE_KEYS = {"enabled", "suffixes", "sort_condition"}
MATCHUP_DISPLAY = {"matrix", "vs_reference"}

# ── catalogs ───────────────────────────────────────────────────────────────
CATALOG_KEYS = {"paths", "models", "experiments"}

# ── data (§3) ──────────────────────────────────────────────────────────────
DATA_KEYS = {"extract", "tables", "filter"}
EXTRACT_KEYS = {
    "enabled",
    "runs_dir",
    "outputs",
    "max_dbs",
    "prune_missing",
    "force_rebuild",
    "auto_fix",
    "issues_path",
}
TABLE_NAMES = ("turns", "panel", "games", "tokens")

# ── filters (§3.1) ─────────────────────────────────────────────────────────
FILTER_KEYS = {
    "experiments",
    "exclude_experiments",
    "players",
    "only_llm",
    "min_games",
    "turn_range",
}

# ── groupings (§3.2) ───────────────────────────────────────────────────────
GROUPING_KEYS = {"kind", "columns", "labels", "column", "edges"}
GROUPING_KINDS_IMPLEMENTED = {"argmax"}
GROUPING_KINDS_RESERVED = {"bucket"}

# ── estimators (§4) ────────────────────────────────────────────────────────
ESTIMATOR_KEYS = {
    "id",
    "model",
    "fit",
    "predict",
    "enabled",
    "params",
    "features",
    "predict_subset",
    "save_predictions",
    "needs",
    "tune",
    "train",
    "pretrained",
}
FIT_VALUES = {"train", "pretrained"}
PREDICT_VALUES = {"in_sample", "cross_val"}
FEATURES_KEYS = {"include", "exclude"}
TUNE_KEYS = {
    "enabled",
    "engine",
    "search",
    "n_trials",
    "objective",
    "n_splits",
    "resample",
    "n_jobs",
    "storage",
    "save_params",
    "load_params",
}
TRAIN_KEYS = {
    "train_subset",
    "resample",
    "save_model",
    "n_splits",
    "save_importance",
}
PRETRAINED_KEYS = {"model_dir"}

# ── adjust (§5) ────────────────────────────────────────────────────────────
ADJUST_KEYS = {"id", "module", "enabled", "uses", "save", "needs", "params"}
ADJUST_MODULES = {"strength"}
USES_KEYS = {"estimators", "tables", "analyses"}
# strength params (§5.1) — enum domains validated in stage 0
STRENGTH_PARAM_KEYS = {
    "turn_progress_min",
    "weight",
    "relative_to",
    "enforce_winner",
    "civ_adjust",
    "block",
    "baseline_experiment",
    "post_cell_normalize",
    "min_condition_completeness",
}
STRENGTH_WEIGHT = {"turn_progress", "uniform"}
STRENGTH_RELATIVE_TO = {"game_leader", "none"}
STRENGTH_CIV_ADJUST = {"none", "ols_logit"}
STRENGTH_BLOCK = {"none", "start_cell", "auto"}
STRENGTH_POST_CELL_NORMALIZE = {"none", "relative_to_leader"}
# `baseline_experiment` is a free-form experiment id (not an enum). It is type-checked in
# stage 0, but not validated against the legacy experiments catalog because ids can be
# inferred from extracted data.

# ── analyses (§6) ──────────────────────────────────────────────────────────
ANALYSIS_KEYS = {
    "id", "module", "name", "description", "enabled", "needs", "uses", "filter", "params",
}

# The analysis registry: every module the harness will ship (core + optional).
# Per-module param schemas are validated by each module as it lands (stages 3-5);
# stage 0 validates the common envelope + cross-cutting rules (group_by/bootstrap).
ANALYSIS_MODULES = {
    # ratings.*
    "ratings.bradley_terry",
    "ratings.plackett_luce",
    "ratings.matchups",
    "ratings.outcome_matchups",
    "ratings.ablation_bt",
    "ratings.vanilla_slot_effect",
    # prediction.*
    "prediction.evaluate",
    "prediction.compare",
    "prediction.winner_trajectories",
    "prediction.elo_comparison",
    "prediction.context_slicing",
    # calibration.*
    "calibration.reliability",
    "calibration.loss_by_progress",
    "calibration.civ_effects",
    "calibration.cell_baseline",
    # performance.*
    "performance.experiment_completeness",
    "performance.score_ratio",
    "performance.strength_panel",
    "performance.turn_predicted",
    "performance.permutation_importance",
    # exploratory.*
    "exploratory.model_token_costs",
    "exploratory.cost_vs_rating",
    "exploratory.panel",
    "exploratory.turn",
    "exploratory.strategy_profiles",
}
RATINGS_PREFIX = "ratings."
STRENGTH_RATING_MODULES = {
    "ratings.bradley_terry",
    "ratings.plackett_luce",
    "ratings.matchups",
}

# ── per-module analysis param schemas (§6, validated in loader._validate_analysis) ──
# Allowed param keys per core module. Cross-cutting `group_by`/`bootstrap` (ratings)
# are validated separately. A module absent here accepts no params (empty schema).
ANALYSIS_PARAM_KEYS = {
    "ratings.bradley_terry": {"group_by", "bootstrap", "weighted", "ref", "min_games", "only_llm", "condition_pairing"},
    "ratings.plackett_luce": {"group_by", "bootstrap", "ref", "min_games", "only_llm", "condition_pairing"},
    "ratings.matchups": {"mode", "validate_ols", "display"},
    "ratings.outcome_matchups": {"include_score_ratio", "display"},
    "prediction.evaluate": {"metrics"},
    "prediction.compare": set(),
    "calibration.reliability": {"n_bins"},
    "calibration.loss_by_progress": {"n_bins", "metrics"},
    "calibration.civ_effects": set(),
    "calibration.cell_baseline": set(),
    "performance.experiment_completeness": {"emit_seating"},
    "performance.score_ratio": {"target", "predictors", "condition_pairing"},
    "performance.strength_panel": {
        "metric", "by", "min_games_preliminary", "bootstrap_n", "ci_level",
        "condition_pairing",
    },
    "performance.turn_predicted": {"aggregate", "by"},
    "exploratory.model_token_costs": {
        "currency", "by_player_type", "by_strategist", "condition_pairing",
    },
    "exploratory.cost_vs_rating": {"currency", "log_x", "annotate", "condition_pairing"},
}
# Enum domains for select analysis params.
PREDICTION_METRICS = {"roc_auc", "brier_score", "log_loss", "balanced_accuracy", "accuracy"}
MATCHUPS_MODE = {"mean", "winrate", "both"}
TURN_PREDICTED_AGGREGATE = {"mean", "median"}

# ── report (§7) ────────────────────────────────────────────────────────────
REPORT_KEYS = {
    "template",
    "out_dir",
    "formats",
    "sections",
    "overview_sections",
    "section_overrides",
    "title",
    "include_disabled",
}
REPORT_SECTION_OVERRIDE_KEYS = {"tables", "figures"}
REPORT_FORMATS = {"md", "html", "pdf"}
