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
    "behavior",
}
TABLE_NAMES = ("turns", "panel", "games", "tokens", "behavior")

# ── data.extract.behavior (§3.0) ───────────────────────────────────────────
# Game-DB column names each behavior family may summarize. The extract layer
# builds its SQL from these, so they are the only names that reach a query.
FLAVOR_NAMES = (
    "Offense", "Defense", "Mobilization", "CityDefense", "MilitaryTraining",
    "Recon", "Ranged", "Mobile", "Nuke", "UseNuke", "Naval", "NavalRecon",
    "NavalGrowth", "NavalTileImprovement", "Air", "AirCarrier", "Antiair",
    "Airlift", "Expansion", "Growth", "TileImprovement", "Infrastructure",
    "Production", "WaterConnection", "Gold", "Science", "Culture", "Happiness",
    "GreatPeople", "Wonder", "Religion", "Diplomacy", "Spaceship", "Espionage",
)
# Flavor → (short header, report group, the set-flavors tool's description).
# The descriptions mirror the MCP server's docs/strategies/flavors.json.
FLAVOR_INFO = {
    'Offense': ('Off', 'Military', 'Pivots the military towards offensive stances. Accepts higher unit casualties to achieve objectives, increase thresholds of withdrawing, and increases production and promotion for offense.'),
    'Defense': ('Def', 'Military', 'Increases military production and unit promotion for defensive purposes.'),
    'CityDefense': ('CDef', 'Military', 'Prioritizes unit promotions for static defense (friendly territory bonuses, capital defense, garrisoned attacks).'),
    'Mobilization': ('Mob', 'Military', 'Increases the proportion of military production (compared with civilian production).'),
    'MilitaryTraining': ('MTrn', 'Military', 'Prioritizes military quality through training buildings and upgrading units with gold.'),
    'Recon': ('Rec', 'Military', 'Increases explorer unit production and tactical exploration of the map.'),
    'Ranged': ('Rng', 'Military', 'Increases firepower by raising target composition of ranged units and range-extending promotions.'),
    'Mobile': ('Mobl', 'Military', 'Increases mobility and maneuverability by raising target composition of mobile units and prioritizes related promotions.'),
    'Nuke': ('Nuke', 'Nuclear', 'Increases nuclear weapon arsenals for strategic deterrence.'),
    'UseNuke': ('UNuk', 'Nuclear', 'Sets (flavor%) per-turn probability of launching nuclear strikes during war when strategic conditions are met.'),
    'Naval': ('Nav', 'Naval and air', 'Increases the production of naval production and prioritizes combat-effective naval promotions. Naval size is also based on coastal city percentage and geography.'),
    'NavalRecon': ('NRec', 'Naval and air', 'Increases the production of more naval melee units for sea control and exploration.'),
    'Air': ('Air', 'Naval and air', 'Increases air control by raising target composition of air units and prioritizes related promotions.'),
    'Antiair': ('AA', 'Naval and air', 'Increases anti-air unit composition ratio to counter enemy aircraft.'),
    'AirCarrier': ('Carr', 'Naval and air', 'Increases carrier production to support naval-based air force operations.'),
    'Airlift': ('Lift', 'Naval and air', 'Prioritizes airlift infrastructure for rapid troop deployment, favoring centralized reserves over forward garrisons.'),
    'Expansion': ('Exp', 'Economy', 'Prioritizes settler production and lowers location criteria to settle more cities. Essential in early game.'),
    'Growth': ('Gro', 'Economy', 'Prioritizes population growth through food-focused tile improvements and buildings.'),
    'TileImprovement': ('Tile', 'Economy', 'Prioritizes worker production and peaceful tile development.'),
    'Infrastructure': ('Infr', 'Economy', 'Prioritizes road construction and city connections for income and military readiness.'),
    'Production': ('Prod', 'Economy', 'Prioritizes production-focused tiles and buildings.'),
    'Gold': ('Gold', 'Economy', 'Prioritizes gold-focused tiles and buildings.'),
    'Science': ('Sci', 'Economy', 'Prioritizes science-focused tiles and buildings.'),
    'Culture': ('Cul', 'Economy', 'Prioritizes culture-focused tiles and buildings.'),
    'Happiness': ('Hap', 'Economy', 'Prioritizes happiness-generating buildings and luxury resources.'),
    'NavalGrowth': ('NGro', 'Economy', 'Prioritizes naval economic infrastructure in coastal cities.'),
    'NavalTileImprovement': ('NTile', 'Economy', 'Develops water resources for immediate economic benefit.'),
    'WaterConnection': ('WCon', 'Economy', 'Prioritizes lighthouse construction for coastal city connectivity.'),
    'GreatPeople': ('GP', 'Other', 'Prioritizes specialist buildings and great person generation infrastructure.'),
    'Wonder': ('Wond', 'Other', 'Prioritizes wonder construction.'),
    'Religion': ('Rel', 'Other', 'Prioritizes religious infrastructure, missionary production, and faith generation.'),
    'Diplomacy': ('Dip', 'Other', 'Prioritizes diplomatic unit production and city-state investment.'),
    'Espionage': ('Esp', 'Other', 'Prioritizes counterintelligence protection for high-value science cities.'),
    'Spaceship': ('Spc', 'Other', 'Prioritizes late-game science victory components and spaceship part production.'),
}
# Vox Populi eras in game order; the `Era` column holds the localized name
# ("Modern Era"), matched on its first word.
ERA_ORDER = ("Ancient", "Classical", "Medieval", "Renaissance", "Industrial", "Modern", "Atomic", "Information")
# Flavor → the bar a player must clear before its setting counts: the first turn
# it researches one of `techs` (PlayerSummaries.CurrentResearch), or the first
# turn in `era` or later. A gated flavor is summarized from that turn on and is
# blank for a player who never clears it. `data.extract.behavior.flavor_gates`
# replaces this map; `{}` turns gating off.
_NUCLEAR_TECHS = ["Nuclear Fission", "Satellites", "Advanced Ballistics"]
DEFAULT_FLAVOR_GATES = {
    "Nuke": {"techs": list(_NUCLEAR_TECHS)},
    "UseNuke": {"techs": list(_NUCLEAR_TECHS)},
    "Air": {"era": "Modern"},
    "Antiair": {"era": "Modern"},
    "AirCarrier": {"era": "Modern"},
    "Airlift": {"era": "Modern"},
}
FLAVOR_GATE_KINDS = ("techs", "era")
PERSONA_NAMES = (
    "VictoryCompetitiveness", "WonderCompetitiveness", "MinorCivCompetitiveness",
    "Boldness", "WarBias", "HostileBias", "WarmongerHate", "NeutralBias",
    "FriendlyBias", "GuardedBias", "AfraidBias", "DiplomaticBalance",
    "Friendliness", "WorkWithWillingness", "WorkAgainstWillingness", "Loyalty",
    "MinorCivFriendlyBias", "MinorCivNeutralBias", "MinorCivHostileBias",
    "MinorCivWarBias", "DenounceWillingness", "Forgiveness", "Meanness",
    "Neediness", "Chattiness", "DeceptiveBias",
)
# Persona trait → (short name, group, description of the set-persona tool). Every
# trait runs 1 to 10 (the tool clamps to that range).
PERSONA_INFO = {
    'VictoryCompetitiveness': ('Vic', 'Competitiveness', 'How aggressively the AI reacts to others pursuing victories.'),
    'WonderCompetitiveness': ('Won', 'Competitiveness', 'How aggressively the AI reacts to others competing for wonders.'),
    'MinorCivCompetitiveness': ('CS', 'Competitiveness', 'How aggressively the AI reacts to others competing for city-state influence.'),
    'Boldness': ('Bold', 'Competitiveness', 'Military risk-taking, territorial claim, and conquest desire.'),
    'WarBias': ('War', 'War and peace', 'Likelihood to plan for or declare offensive war.'),
    'HostileBias': ('Host', 'War and peace', 'Tendency toward hostile relationships without direct wars.'),
    'WarmongerHate': ('WmH', 'War and peace', 'How negatively the AI reacts to warlike behaviors.'),
    'NeutralBias': ('Neu', 'War and peace', 'Tendency toward neutral relationships.'),
    'FriendlyBias': ('Fbi', 'War and peace', 'Tendency toward friendly relationships.'),
    'GuardedBias': ('Grd', 'War and peace', 'Tendency to be guarded or cautiously defensive in diplomacy.'),
    'AfraidBias': ('Afr', 'War and peace', 'Tendency to be afraid of stronger civs.'),
    'DiplomaticBalance': ('Bal', 'Diplomacy', 'Increases relationship with non-competitive civilizations and peaceful resolution of wars.'),
    'Friendliness': ('Frd', 'Diplomacy', 'Desire for friendship declarations and increases maximum DoFs.'),
    'WorkWithWillingness': ('With', 'Diplomacy', 'Tendency to support or collaborate with allies. Increases opinions of shared friends.'),
    'WorkAgainstWillingness': ('Agst', 'Diplomacy', 'Tendency to bond over shared enemies and jointly act against them.'),
    'Loyalty': ('Loy', 'Diplomacy', 'Loyalty to allies. Lower values allow for backstabbing.'),
    'MinorCivFriendlyBias': ('CS F', 'City-states', 'Tendency to be friendly with city-states.'),
    'MinorCivNeutralBias': ('CS N', 'City-states', 'Tendency to be neutral with city-states.'),
    'MinorCivHostileBias': ('CS H', 'City-states', 'Tendency to be hostile with city-states.'),
    'MinorCivWarBias': ('CS W', 'City-states', 'Likelihood to attack city-states.'),
    'DenounceWillingness': ('Den', 'Personality', 'Readiness to denounce other civs.'),
    'Forgiveness': ('Fgv', 'Personality', 'How quickly to forgive past transgressions.'),
    'Meanness': ('Mean', 'Personality', 'Aggressiveness in general. Demanding and bullying more while less likely to accept peace.'),
    'Neediness': ('Need', 'Personality', 'Desire for support from friends.'),
    'Chattiness': ('Chat', 'Personality', 'How often the AI initiates diplomatic contact.'),
    'DeceptiveBias': ('Dec', 'Personality', 'Tendency to be deceptively friendly.'),
}
PERSONA_RANGE = (1.0, 10.0)
BEHAVIOR_EVENTS = ("wars_declared", "wars_received", "cities_nuked", "cities_razed")
BEHAVIOR_POLICIES = (
    "policy_changes",
    "tradition", "authority", "progress",
    "fealty", "statecraft", "artistry",
    "industry", "imperialism", "rationalism",
    "freedom", "autocracy", "order",
)
# Relationship column → kind. Counts are 0 when the player set nothing; the
# averages and shares are blank without any pair-turn to weigh. The distinct-
# target count is a `level`, because a per-turn rate of it means nothing.
BEHAVIOR_RELATIONSHIP_KINDS = {
    "relationship_changes": "count",
    "relationship_targets": "level",
    "stance_public_avg": "level",
    "stance_private_avg": "level",
    "stance_net_avg": "level",
    "stance_masked_hostility_share": "share",
    "stance_masked_goodwill_share": "share",
}
BEHAVIOR_RELATIONSHIPS = tuple(BEHAVIOR_RELATIONSHIP_KINDS)
# Policy branches that are ideologies; the rest compete for the opening branch.
BEHAVIOR_IDEOLOGIES = ("freedom", "autocracy", "order")
# Policy branch → (name, short header, tier, unlock rule). Tiers and unlocks
# follow Vox Populi's PolicyTreeChanges.sql; `behavior.policies` groups its
# columns by tier in this order.
POLICY_INFO = {
    "tradition": ("Tradition", "Trad", "Ancient", "Open from the start."),
    "authority": ("Authority", "Auth", "Ancient", "Open from the start."),
    "progress": ("Progress", "Prog", "Ancient", "Open from the start."),
    "fealty": ("Fealty", "Feal", "Medieval", "Opens in the Medieval era after 6 adopted policies."),
    "statecraft": ("Statecraft", "Stat", "Medieval", "Opens in the Medieval era after 6 adopted policies."),
    "artistry": ("Artistry", "Art", "Medieval", "Opens in the Medieval era after 6 adopted policies."),
    "industry": ("Industry", "Ind", "Industrial", "Opens in the Industrial era after 12 adopted policies."),
    "imperialism": ("Imperialism", "Imp", "Industrial", "Opens in the Industrial era after 12 adopted policies."),
    "rationalism": ("Rationalism", "Rat", "Industrial", "Opens in the Industrial era after 12 adopted policies."),
    "freedom": ("Freedom", "Free", "Ideology", "An ideology; a player usually adopts one."),
    "autocracy": ("Autocracy", "Auto", "Ideology", "An ideology; a player usually adopts one."),
    "order": ("Order", "Ord", "Ideology", "An ideology; a player usually adopts one."),
}
# The in-game AI's grand strategies → (display name, the victory type whose
# color it shares in reports). Unknown strategies still show, uncolored.
GRAND_STRATEGY_INFO = {
    "Conquest": ("Conquest", "Domination"),
    "Culture": ("Culture", "Culture"),
    "UnitedNations": ("United Nations", "Diplomatic"),
    "Spaceship": ("Spaceship", "Science"),
}
# Persona traits `behavior.diplomacy` reads (as `persona_<trait>_avg`) by default.
BEHAVIOR_DIPLOMACY_TRAITS = (
    "DiplomaticBalance", "Friendliness", "WorkWithWillingness",
    "WorkAgainstWillingness", "Loyalty", "DenounceWillingness", "Forgiveness",
    "Meanness", "Neediness", "Chattiness", "DeceptiveBias",
)
BEHAVIOR_STATS = ("min", "avg", "max")
# Column kinds tell analyses how to treat a behavior column: `count` is
# normalized by turns alive, `turn` is a first-adoption turn with "N/A" for never.
BEHAVIOR_COLUMN_KINDS = ("count", "level", "share", "turn")
# Family key → allowed selections. `stats` is the only non-family key.
BEHAVIOR_FAMILIES = {
    "flavor": FLAVOR_NAMES,
    "persona": PERSONA_NAMES,
    "events": BEHAVIOR_EVENTS,
    "policies": BEHAVIOR_POLICIES,
    "relationships": BEHAVIOR_RELATIONSHIPS,
}
BEHAVIOR_DEFAULTS = {
    "stats": ["min", "avg", "max"],
    "flavor": list(FLAVOR_NAMES),
    "persona": ["Boldness", "WarBias", "HostileBias", "WarmongerHate", "Meanness",
                "DeceptiveBias", "Forgiveness", "DenounceWillingness",
                "MinorCivWarBias", "VictoryCompetitiveness",
                "DiplomaticBalance", "Friendliness", "WorkWithWillingness",
                "WorkAgainstWillingness", "Loyalty", "Neediness", "Chattiness"],
    "events": list(BEHAVIOR_EVENTS),
    "policies": list(BEHAVIOR_POLICIES),
    "relationships": list(BEHAVIOR_RELATIONSHIPS),
}

# ── filters (§3.1) ─────────────────────────────────────────────────────────
FILTER_KEYS = {
    "experiments",
    "exclude_experiments",
    "players",
    "only_llm",
    "min_games",
    "turn_range",
    "min_condition_completeness",
    "max_decision_failure_pct",
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
# strength params (§5.1); enum domains validated in stage 0
STRENGTH_PARAM_KEYS = {
    "turn_progress_min",
    "weight",
    "relative_to",
    "enforce_winner",
    "civ_adjust",
    "block",
    "baseline_experiment",
    "post_cell_normalize",
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
    "performance.controlled_seed_report",
    "performance.game_log",
    "performance.usage_efficiency",
    # behavior.*
    "behavior.flavors",
    "behavior.diplomacy",
    "behavior.commitment",
    "behavior.policies",
    # exploratory.*
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

# Analyses that consume exactly one strength table through uses.tables. The
# turn_predicted module also takes its estimator from that strength stage.
SINGLE_STRENGTH_TABLE_MODULES = {
    "performance.controlled_seed_report",
    "performance.turn_predicted",
}

# Modules the controlled-seed report may list in uses.analyses: each one adds a
# tab to the Matched Maps overview and seat pages (§7.1).
MATCHED_MAPS_TAB_MODULES = {
    "behavior.flavors", "behavior.diplomacy", "behavior.commitment", "behavior.policies",
}

# Params every behavior.* module accepts (§6.2). `baseline` is "completed", one
# experiment id, or a list of experiment ids.
BEHAVIOR_COMMON_PARAMS = {"baseline", "rate", "by", "bootstrap_n", "ci_level", "condition_pairing"}
BEHAVIOR_BASELINE_COMPLETED = "completed"
BEHAVIOR_RATES = ("per_100_turns", "per_game")

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
    "performance.turn_predicted": {"aggregate", "by", "condition_pairing"},
    "performance.controlled_seed_report": {"condition_pairing"},
    "performance.game_log": {"condition_pairing"},
    "performance.usage_efficiency": {
        "currency", "log_x", "annotate", "condition_pairing",
    },
    "behavior.flavors": {"flavors", *(BEHAVIOR_COMMON_PARAMS - {"rate"})},
    "behavior.diplomacy": {"traits", *BEHAVIOR_COMMON_PARAMS},
    "behavior.commitment": set(BEHAVIOR_COMMON_PARAMS),
    "behavior.policies": {"branches", *(BEHAVIOR_COMMON_PARAMS - {"rate"})},
}
# Enum domains for select analysis params.
PREDICTION_METRICS = {"roc_auc", "brier_score", "log_loss", "balanced_accuracy", "accuracy"}
MATCHUPS_MODE = {"mean", "winrate", "both"}
TURN_PREDICTED_AGGREGATE = {"mean", "median"}

# ── report (§7) ────────────────────────────────────────────────────────────
REPORT_KEYS = {
    "out_dir",
    "formats",
    "sections",
    "overview_sections",
    "section_overrides",
    "title",
    "footer",
    "benchmark_citation",
    "include_disabled",
    "replay",
    "publish",
}
REPORT_PUBLISH_KEYS = {"enabled"}
REPORT_REPLAY_KEYS = {"enabled", "viewer_url", "saves", "base_url", "latest_game"}
REPORT_REPLAY_SAVES = {"all", "controlled"}
REPORT_DEFAULT_VIEWER_URL = "https://vox-deorum.github.io/vox-deorum-replay/"
REPORT_BENCHMARK_CITATION_KEYS = {"title", "url"}
REPORT_SECTION_OVERRIDE_KEYS = {"tables", "figures"}
REPORT_FORMATS = {"md", "html", "pdf"}
# Formats the report renders; also the default when `report.formats` is omitted.
REPORT_DEFAULT_FORMATS = ["md", "html"]
REPORT_DEFAULT_FOOTER = (
    "Generated by [CivBench](https://github.com/vox-deorum/civ-bench) "
    "([Chen, 2026](https://arxiv.org/abs/2604.07733))"
)
