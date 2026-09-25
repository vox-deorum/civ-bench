# `benchmark.json`: the run-spec convention

A **benchmark run-spec** (conventionally `benchmark.json`) is the **top-level config**: it declares the input data, which estimators to train or load, which analyses to run, and how to render the report. It is the only file you edit to define a new benchmark run. This document is its authoritative schema.

> **Templates vs. local configs.** The repo tracks *example* run-specs as `configs/*.template.json` (`benchmark.template.json`, `benchmark.full.template.json`, `benchmark.pretrained.template.json`). To actually run, copy one to a **local, gitignored** `configs/benchmark*.json` (e.g. `configs/benchmark.dev.json`) and point it at your data roots. This schema applies to all of them; "benchmark.json" below names the run-spec *format*, not a tracked file. See [AGENTS.md](../AGENTS.md) §"Templates vs. local configs".

`benchmark.json` is validated on load: **unknown keys and missing required fields are hard errors** (invariant 1: config over code). When you add or rename a module, update this file in the same change.

> JSON has no comments. This reference uses `jsonc` fences with `//` notes for explanation; a real `configs/benchmark*.json` must be plain JSON. See [benchmark.template.json](benchmark.template.json) for a comment-free worked example.

---

## 1. The pipeline as a DAG

A run is a **directed acyclic graph of stages** in five kinds, executed in dependency order:

```
extract ──▶ estimators ──▶ adjust ──▶ analyses ──▶ report
```

`adjust` is the bridge between raw predictions and the strength-based rating models: it turns an estimator's per-turn win-probabilities into a per-player-game **strength panel** (`adjusted_strength`) and registers it as a named table that fitted ratings, strength matchups, and some `performance.*` analyses consume (§5). It is optional; a run with no estimators or strength-based ratings simply omits it.

Each stage has a stable **`id`**. Edges come from three places, all resolved into one topological sort before anything runs:

1. **Kind ordering (implicit).** `extract` → `estimators` → `adjust` → `analyses` → `report`, always.
2. **`needs` (explicit).** A stage may list other stage `id`s it must run after. Use this to force ordering the harness can't infer (e.g. one analysis consuming another's CSV).
3. **`uses` (referential).** When a stage references an estimator `id`, a named table, or another analysis stage in its `uses` block, an edge is created automatically, so you don't also have to write `needs`. A `uses.tables` name may be a canonical table from `data.tables` **or** a table produced by an `adjust` stage (§5); `uses.analyses` names analysis ids whose persisted artifacts the consumer reads.

A cycle, an unknown `id`, or a reference to a disabled stage is a validation error. Disabled stages (`"enabled": false`) are dropped from the graph, and anything that *needed* one is a validation error. There is **no graceful degradation for missing dependencies**: if a stage needs `torch` / `xgboost` / `optuna` / R and it isn't installed, the run **fails loud**: install everything first (see [AGENTS.md](../AGENTS.md#dependencies) and `scripts/install`).

The dependency graph is resolved once at config-load time and then reused by the dry-run printer and runner. Validation and execution therefore share the same interpretation of `needs`, `uses.estimators`, `uses.tables`, and `uses.analyses`. For analysis modules that opt in to the all-estimator default, omitted or empty `uses.estimators` means "all enabled estimators"; listing ids is an explicit subset override.

**Determinism.** The top-level `seed` is threaded into every stage that uses randomness (CV splits, torch init, resampling, bootstrap). Same `benchmark.json` + same `runs/` ⇒ byte-stable outputs.

---

## 2. Top-level shape

```jsonc
{
  "name": "staff-standard-2026",        // required: run id; names the report + reports/ subdir
  "friendly_name": "Staff benchmark 2026",  // optional: human title shown on the report page
  "description": "Staff line-up, standard 8-seat map",  // optional, free text; shown on the report page
  "seed": 42,                            // required: global RNG seed (determinism)

  "output":     { /* §2.1 */ },          // optional: run output root + variant suffix (→ reports/, reports-cross/)
  "presentation": { /* §2.2 */ },        // optional: paired-condition and matchup figure display

  "catalogs": {                          // optional: override lazily loaded sibling config files
    "paths":       "configs/paths.json",
    "models":      "configs/models.json",
    "experiments": "configs/experiments.json"
  },

  "filters":    { /* §3.1 */ },          // optional: named, reusable filter presets
  "groupings":  { /* §3.2 */ },          // optional: named rating-identity grouping dimensions

  "data":       { /* §3  */ },           // required: extraction + canonical tables + global filter
  "estimators": [ /* §4  */ ],           // optional: prediction-model producers
  "adjust":     [ /* §5  */ ],           // optional: derived-table stages (e.g. the strength panel)
  "analyses":   [ /* §6  */ ],           // required: the modules to run
  "report":     { /* §7  */ }            // required: rendering
}
```

Top-level keys: `name`, `seed`, `data`, `analyses`, `report` are **required**; `friendly_name`, `description`, `output`, `presentation`, `catalogs`, `filters`, `groupings`, `estimators`, and `adjust` are optional. Omit `estimators` (and `adjust`) for a run with no strength-based ratings; conversely, `ratings.bradley_terry`, `ratings.plackett_luce`, and `ratings.matchups` need an `adjust` stage to supply the `strength` table they rate. `friendly_name` is an optional human title: when set it becomes the report page title unless `report.title` overrides it, and `description` renders under the title on the report page (§7).

**`catalogs` is lazy.** If omitted, each catalog-using stage resolves `paths.json`, `models.json`, and `experiments.json` from the **same directory as this run-spec file** only when it needs that catalog. Set a key only to point at a file elsewhere; unset keys still fall back to the sibling. A missing catalog is a load error only when an enabled stage needs it (for example, an estimator needs `models.json`, and orthodox `player_type` composition needs the model/experiment catalogs). Runs that do not touch a catalog do not require that sibling file to exist.

### 2.1 `output`: the run output root (and the `-cross` variant)

Every stage that writes (`estimators` `save_predictions`/`save_model`, `adjust` `save`, `report` `out_dir`) writes **under a single run output root**, resolved once and threaded into all stages. By default that root is `reports/`. `output` lets a run redirect everything to a sibling root by appending a configurable **suffix**: the mechanism behind the **`-cross` (llm/non-llm) variant**, where an estimator trained on non-LLM seats predicts everyone and its whole report lands in `reports-cross/` instead of clobbering the normal run.

```jsonc
"output": {
  "root":   "reports",   // base output directory for ALL stages (default "reports")
  "suffix": ""           // appended to root → "" ⇒ reports/ ;  "-cross" ⇒ reports-cross/
}
```

- A stage save-path written as `reports/estimators/<id>/…` is interpreted **relative to the resolved root** (`<root><suffix>/estimators/<id>/…`). Authoring paths under `reports/` keeps the default run unchanged; setting `suffix: "-cross"` moves the *same* config's outputs to `reports-cross/`.
- **Only save-paths are re-rooted; input paths are read as-authored.** `estimators.pretrained.model_dir` (§4.6), `data.tables.*`, and `data.extract.runs_dir` are inputs the run *reads*; the suffix never rewrites them. This is what lets the tracked `pretrained/<model_id>/` store feed every variant: a `-dev` run loads the same `pretrained/score/` but writes its predictions under `reports-dev/`.
- The **cross variant is otherwise an ordinary `fit: train` run** with `train.train_subset: "non_llm"` (§4.4); there is no separate "cross" estimator kind and no special prediction-loading path. Pair the non-LLM training subset with `output.suffix: "-cross"` (typically as its own config, e.g. `benchmark.cross.json`, or a CLI suffix override) so the two variants coexist on disk.
- `output` is optional; omit it for the default `reports/` root and no suffix.

### 2.2 `presentation`: condition pairing and matchup display

`presentation` changes figures only; analysis fits and existing result tables are
unchanged. Omitting the block guarantees the legacy figures: independent
player-type rows and N×N matchup matrices.

```jsonc
"presentation": {
  "condition_pairing": {
    "enabled": true,             // default false
    "suffixes": null,            // derive suffix labels from experiments.json
    "base_label": "Every-turn", // legend label; default "Base"
    "sort_condition": "-Per-5"  // "base", "best", or an effective suffix
  },
  "matchup_display": "vs_reference" // "matrix" (default) | "vs_reference"
}
```

With pairing enabled, suffix-style values from `experiments.json`
`player_type_labels` (§3.3), including per-slot dictionary values, define the
conditions. For example, `"*-per-5": "-Per-5"` pairs
`GLM-5.2-Simple` and `GLM-5.2-Simple-Per-5` on one row. `suffixes:null`
derives all such values; an explicit non-empty list restricts them. Vanilla and
Null remain unsplit. Rows sort by `sort_condition`: `"base"`, `"best"` (each
identity keys off its best value across all conditions, honoring the plot's
top-first direction), or a specific suffix. A concrete condition falls back to
whichever condition has a value for incomplete pairs; all-NaN identities sort
last.

The fitted-rating forests, player-type strength panels, score-ratio effects, and
token-cost view accept `params.condition_pairing` as either a boolean or an
object containing `enabled`, `suffixes`, and `sort_condition`. The stage value
overrides the global block. Matchup modules accept `params.display`; in
`vs_reference` mode they retain their matrix CSVs but render row-vs-Vanilla
points (paired when pairing is enabled). If Vanilla is absent, they warn and
render the matrix instead.

---

## 3. `data`: extract + canonical tables + global filter

`data` owns everything before the analysis modules: turning raw DBs into canonical CSVs, naming those CSVs, and the **global** row filter that every downstream stage inherits (stages may narrow further, never widen).

```jsonc
"data": {
  "extract": {
    "enabled": true,                     // false → reuse existing CSVs, never touch runs/ DBs
    "runs_dir": "runs/",                 // root searched for *.db game DBs
    "outputs": ["turns", "panel", "games", "tokens", "behavior"],  // which canonical CSVs to (re)build
    "max_dbs": null,                     // int → only first N discovered DBs (smoke tests)
    "prune_missing": false,              // true → only drop rows for missing DBs, no new extract
    "force_rebuild": false,              // true → rebuild even if outputs exist & are newer
    "auto_fix": true,                    // true → repair malformed DBs & re-import (--no-fix disables)
    "issues_path": "runs/import_issues.csv",  // where malformed/locked-DB import issues are recorded
    "behavior": {                        // optional; omitted keys use these defaults
      "stats":   ["min", "avg", "max"],
      "flavor":  ["Offense", "Defense", "Mobilization", "CityDefense", "MilitaryTraining", "Recon", "Ranged", "Mobile", "Nuke", "UseNuke", "Naval", "NavalRecon", "NavalGrowth", "NavalTileImprovement", "Air", "AirCarrier", "Antiair", "Airlift", "Expansion", "Growth", "TileImprovement", "Infrastructure", "Production", "WaterConnection", "Gold", "Science", "Culture", "Happiness", "GreatPeople", "Wonder", "Religion", "Diplomacy", "Spaceship", "Espionage"],
      "flavor_gates": {                  // replaces the defaults below; {} disables gating (§3.0)
        "Nuke":    {"techs": ["Nuclear Fission", "Satellites", "Advanced Ballistics"]},
        "UseNuke": {"techs": ["Nuclear Fission", "Satellites", "Advanced Ballistics"]},
        "Air": {"era": "Modern"}, "Antiair": {"era": "Modern"},
        "AirCarrier": {"era": "Modern"}, "Airlift": {"era": "Modern"}
      },
      "persona": ["Boldness", "WarBias", "HostileBias", "WarmongerHate", "Meanness", "DeceptiveBias", "Forgiveness", "DenounceWillingness", "MinorCivWarBias", "VictoryCompetitiveness", "DiplomaticBalance", "Friendliness", "WorkWithWillingness", "WorkAgainstWillingness", "Loyalty", "Neediness", "Chattiness"],
      "events":  ["wars_declared", "wars_received", "cities_nuked", "cities_razed"],
      "policies": ["policy_changes", "tradition", "authority", "progress", "fealty", "statecraft", "artistry", "industry", "imperialism", "rationalism", "freedom", "autocracy", "order"],
      "relationships": ["relationship_changes", "relationship_targets", "stance_public_avg", "stance_private_avg", "stance_net_avg", "stance_masked_hostility_share", "stance_masked_goodwill_share"]
    }
  },

  "tables": {                            // canonical CSV locations (extract writes / loaders read)
    "turns":      "runs/turn_data.csv",          // per-player per-turn panel (prediction features); carries player_type, NOT seed
    "panel":      "runs/panel_data.csv",         // per-player per-game outcomes/strategy summaries (+ player_type/model/strategist/config_slot)
    "games":      "runs/game_data.csv",          // per-GAME row: game_id, timestamp, experiment, seed, seating_rotation (-1 ⇒ uncontrolled)
    "tokens":     "runs/model_token_usage.csv",  // token use plus failed strategist turns per player trace
    "behavior":   "runs/behavior_data.csv"     // per-player per-game behavior and policy summaries
  },

  "filter": "llm_only"                   // GLOBAL selector: inline object OR a preset name (§3.1)
}
```

- **`filter` is optional**: omit it for "all rows". It accepts either an inline filter object or the **name of a preset** from top-level `filters` (§3.1). Every stage inherits this global filter and may narrow it (§6.1), never widen it.
- **`extract.enabled: false`** is the "I already have CSVs" switch: the `extract` stage is dropped from the DAG and loaders read `tables.*` directly. Combine with `--skip extract` on the CLI for the same effect ad hoc.
- The `extract` stage is **skipped automatically** when every `outputs` CSV already exists and is newer than the DBs, unless `force_rebuild: true`.
- When a fresh extract records malformed DBs, they are **auto-repaired and re-imported** in place (extract → `fix` → re-import) before the rest of the DAG runs. Disable with `auto_fix: false` or the CLI `--no-fix` flag; games that recovery cannot save stay flagged and are excluded downstream, exactly as before.
- When the selected stages need experiment ids or player-type names, they resolve through `catalogs.experiments` + `catalogs.models`. **`player_type` is composed at extract from the per-player game metadata** (`model-{id}` + `strategist-{id}`) via the catalog's template + aliases + unified label map (§3.3); the old seat→model mapping is only an optional fallback; never spell out seat→model mappings here.

### 3.0 `data.extract.behavior`: behavior summary table

The `behavior` table has one row per major player per game. Its key columns are `experiment`, `game_id`, `player_id`, `player_type`, `civilization`, and `survival_turn` (the player's last recorded turn). The remaining columns follow `data.extract.behavior`:

| Key | Allowed values | Columns produced |
| --- | --- | --- |
| `stats` | `min`, `avg`, `max` | applied to every `flavor` and `persona` entry |
| `flavor` | any `FlavorChanges` column (for example `Offense`, `UseNuke`) | `flavor_<snake_name>_<stat>`, for example `flavor_use_nuke_max` |
| `flavor_gates` | per-flavor access gate: exactly one of `{"techs": [names]}` or `{"era": name}` | none directly; decides which turns a gated `flavor` column summarizes |
| `persona` | any `PersonaChanges` personality column (for example `Boldness`, `WarBias`) | `persona_<snake_name>_<stat>`, for example `persona_war_bias_avg` |
| `events` | `wars_declared`, `wars_received`, `cities_nuked`, `cities_razed` | one count column per name |
| `policies` | `policy_changes`, `tradition`, `authority`, `progress`, `fealty`, `statecraft`, `artistry`, `industry`, `imperialism`, `rationalism`, `freedom`, `autocracy`, `order` | real policy-change count plus first-adoption turn for each selected branch |
| `relationships` | `relationship_changes`, `relationship_targets`, `stance_public_avg`, `stance_private_avg`, `stance_net_avg`, `stance_masked_hostility_share`, `stance_masked_goodwill_share` | one column per name |

- Each key is optional and falls back to the default shown above. An empty list turns that family off. Unknown names, stats, and duplicates are config errors. Policy fields have no `stats`; only flavor and persona use the `stats` selection.
- Flavor and persona values are the state in effect on each turn: the last row of a turn wins, and it carries forward until the next row. `min` and `max` range over the states in effect from the player's first row to `survival_turn`. `avg` is turn-weighted. Rows written by the in-game AI (`Tweaked by In-Game AI`) count, because the game used them. A player with no rows (for example an in-game AI player with no `FlavorChanges`) gets blank cells. If an older DB lacks a selected column, only that column's cells are blank.
- **`flavor_gates`** keeps a flavor's `min`/`avg`/`max` from summarizing settings the player could not yet act on. Each gate is exactly one of:
  - `{"techs": [names]}`: the access turn is the player's first `PlayerSummaries` turn whose `CurrentResearch` starts with one of the names (a prefix match, so research progress suffixes are fine).
  - `{"era": name}`: the access turn is the player's first turn whose `PlayerSummaries.Era` first word is that era or a later one, in game order: Ancient, Classical, Medieval, Renaissance, Industrial, Modern, Atomic, Information.
  A gated flavor is summarized only from its access turn on; the state in effect at that turn counts from the turn on. A player who never clears the gate gets blanks, and so does every player when the DB lacks the gate's column. The default gates the nuclear flavors (`Nuke`, `UseNuke`) behind `Nuclear Fission`, `Satellites`, or `Advanced Ballistics` and the air flavors (`Air`, `Antiair`, `AirCarrier`, `Airlift`) behind the Modern era. Set `"flavor_gates": {}` to disable gating; a given map replaces the default entirely (it is not merged). Unknown flavors, both gate kinds at once, unknown eras, and empty tech lists are config errors.
- Event counts come from `GameEvents`, with exact duplicate events (same type, turn, and payload) counted once:
  - `wars_declared`: `DeclareWar` events the player originated as aggressor. This includes the automatic declarations on the target's defensive-pact partners and excludes wars joined as a vassal.
  - `wars_received`: every `DeclareWar` event whose target team is the player's team, from any originator.
  - `cities_nuked`: `NuclearDetonation` events by the player whose plot held a city.
  - `cities_razed`: `CityRazed` events by the player.
- `policy_changes` counts `PolicyChanges` rows containing a real field mutation, excluding rationale-only and empty rows. Each branch column records its first adoption turn from `PlayerAdoptPolicyBranch` or `IdeologyAdopted`; branches the player never adopted contain `N/A`.
- Relationship columns come from the `RelationshipChanges` table. State is tracked per `(player, target)` pair: only major-civ targets, never self; the last row of a turn wins and carries forward.
  - `relationship_changes` counts rows that change the pair's public or private value (the starting state is 0 and 0; re-setting the same values does not count). `relationship_targets` counts distinct major targets ever set.
  - The stance averages weight each pair-turn once, from the first row on that target through the earlier of the two players' survival turns (inclusive). Resets to 0 still count, and targets never set are excluded. `stance_net_avg` is public plus private.
  - `stance_masked_hostility_share` is the share of pair-turns with public > 0 > private; `stance_masked_goodwill_share` is the share with public < 0 < private.
  - When the table exists, the two counts are 0 for players who set nothing and the averages and shares are blank; when the table is missing, every cell is blank. The documented range of the values is -100 to 100 but it is not enforced; values are kept raw.
- Every behavior column has a kind the analyses use: `count` (events, `policy_changes`, `relationship_changes`; scaled by turns alive), `level` (flavor and persona stats, `relationship_targets`, stance averages), `share` (stance shares), and `turn` (policy branch first-adoption turns).
- Changing the selection changes the CSV header, so the next extract rebuilds the whole table. `flavor_gates` is the exception: the columns stay the same, so a changed gate map does not trigger a rebuild by itself. After editing it, refresh the table with `civ-bench extract --force-rebuild`.

### 3.1 `filters`: named, reusable filter presets

A filter is the **same shape everywhere** it appears (`data.filter` and every stage's `filter`), so define the common ones once and reference them by name instead of repeating the object. Every field is optional; an omitted field means "no constraint", except `max_decision_failure_pct`, whose effective default is `0.2`.

```jsonc
"filters": {
  "llm_only":     { "only_llm": true, "min_games": 5 },
  "staff_recent": { "experiments": ["2026-staff-standard"], "min_games": 5, "max_decision_failure_pct": 0.2 },
  "late_game":    { "turn_range": [200, null] }
}
```

The full filter shape (every field optional; omitted ⇒ no constraint, except `max_decision_failure_pct`, whose effective default is `0.2`):

```jsonc
{
  "experiments":          null,  // null = all conditions; or ["2026-staff-standard", ...]
  "exclude_experiments":  [],    // subtracted from the above (or from "all")
  "players":              null,  // null = all player types; or ["Sonnet-4.5-Briefed", ...]
  "only_llm":            false,  // true → keep only LLM seats (drop Vanilla/Null, per experiments.json)
  "min_games":           1,      // drop player types with fewer games than this
  "turn_range":          null,   // null = all turns; or [min_turn, max_turn] (absolute turn numbers;
                                 //   either bound may be null, e.g. [200, null] = turn 200 onward)
  "min_condition_completeness": null, // null = keep every controlled condition; a number in (0, 1]
                                      //   drops, as a whole, controlled conditions whose occupied
                                      //   (seed, seating_rotation) slot fraction is below it (1.0 ⇒
                                      //   drop any condition missing a slot)
  "max_decision_failure_pct": 0.2     // effective default 0.2; null disables; number in (0, 1]
}
```

**Wherever a filter is accepted, the value may be:**

- an **inline object** (the shape above),
- a **string** naming a preset in `filters`, or
- a **list** mixing preset names and inline objects, which are merged left-to-right (later entries win per field). E.g. `"filter": ["llm_only", { "turn_range": [200, null] }]`.

A stage's `filter` is then **intersected** with the resolved global `data.filter`: a stage can only narrow, never widen (§6.1). Referencing an undefined preset name is a validation error.

The same resolved filter object is what the shared data loaders accept, so analysis modules should pass config-shaped filters through the loader helpers instead of translating `experiments`, `players`, `only_llm`, `min_games`, `turn_range`, `min_condition_completeness`, and `max_decision_failure_pct` by hand.

- **`max_decision_failure_pct` is a global game filter.** Its effective default is `0.2`; set it to `null` to disable it. A game is accepted only when every player trace has `failed_turn_count / valid_turn_count < threshold`. The comparison is exclusive, so a trace at or above the threshold excludes the whole game. The cutoff uses raw exact telemetry counters, collapses repeated model rows once per player, and is evaluated from whole-game telemetry independently of player or turn filters. The resolved cutoff is inherited by estimator, adjust, and analysis stages. Games with unknown failure rates, including missing, legacy, or zero-denominator telemetry, are retained. Completeness warns when an experiment has no usable telemetry. A stage-level threshold may only narrow the resolved global cutoff.

- **`min_condition_completeness` is a global condition filter.** A controlled *condition* is an experiment's `seed × seating_rotation` grid. The filter evaluates every controlled experiment against the union of controlled slots it can see (the whole controlled design) and drops, as a whole, every experiment whose occupied-slot fraction is below the threshold; `1.0` means "skip every condition missing any slot". Occupied slots are measured after the decision-failure quality filter, while the raw canonical `panel` and `games` design remains available for scheduling and diagnostics. A condition with zero accepted games remains visible as a zero-accepted experiment. Diagnostics bypass this filter so the schedule remains visible; a seating slot with only rejected games stays open until a valid replacement completes it. Because it is a global filter it applies consistently to every stage that loads a table: the estimator predictions, the strength panel, the ratings, and the descriptive analyses such as `performance.usage_efficiency`. The grid is resolved once from the canonical `games` table, so tables that carry no `seed`/`seating_rotation` columns (tokens, turns, panel, predictions) still drop the same incomplete conditions by `experiment`. The `games` table therefore must be present (and extracted) when the filter is set.

### 3.2 `groupings`: named rating-identity dimensions

A **grouping** derives one categorical dimension from the strength panel that `ratings.*` analyses can fold into the rated identity (§6.2, `group_by`). Like `filters`, groupings are defined once by name and referenced by name: config over code, so adding a way to slice the field is a config edit, not a new module. `groupings` is **optional**; omit it for plain per-`player_type` ratings.

```jsonc
"groupings": {
  "strategy": {                          // composite identity {player_type}-{dominant strategy}
    "kind": "argmax",                    // dimension value = column with the largest value per row
    "columns": ["domination_ratio","culture_ratio","diplomatic_ratio","science_ratio"],
    "labels":  ["Domination","Culture","Diplomatic","Science"]   // optional; positional with columns
  }
}
```

- **`kind: "argmax"`** (the only kind implemented) assigns each player-game the label of whichever `columns` value is largest, exactly the dominant-strategy rule the old `strategy_ratings.py` uses. `labels` is optional; when given it must be positional with `columns` (else the raw column name is used).
- A grouping referenced by `group_by` but **absent** from this catalog is a validation error (§8).

> **Reserved / deferred: do not rely on yet.** Other `kind`s are planned but **not implemented**. In particular `kind: "bucket"` (e.g. `{ "kind": "bucket", "column": "turn_progress", "edges": [0, 0.33, 0.66, 1.0], "labels": ["early","mid","late"] }`) would enable per-game-stage ratings, but that additionally requires the `adjust`/`strength` stage to emit *per-stage* strength, which is its own follow-up. Treat anything beyond `argmax` as designed-but-unbuilt.

### 3.3 Player identity: the orthodox `player_type`

`player_type` (the identity every rating is fit over) is **composed at extract time from the per-player game metadata**, not from a hand-maintained seat map. Each seat records `model-{player_id}` (e.g. `Sonnet-4.5`, or `VPAI` for vanilla) and `strategist-{player_id}` (e.g. `simple-strategist-briefed`) in the game's `GameMetadata`; because the identity travels with the player, it stays correct even when controlled seating rotates a model through different seats. The catalog (`catalogs.models` / `catalogs.experiments`) supplies three knobs:

- **`player_type_template`**: the format string, e.g. `"{model}-{variant}{suffix}"`. `model` is alias-normalized and `strategist`→`variant` is mapped via the catalog; `model == "VPAI"` resolves to the vanilla baseline label by strategist (`null-strategist`→`Null`, default→`Vanilla`).
- **Unknown model names**: registered aliases take precedence. An unregistered name drops any `@variant` and provider path prefix (everything through the final `/`), then title-cases the remaining hyphen-separated segments. Uppercase segments and acronyms found in catalog model IDs, such as `GPT`, `GLM`, and `OSS`, retain their capitalization. Empty model metadata resolves to `N/A`.
- **`player_type_labels`**: a single unified map keyed by condition and, optionally, `(condition, slot)` (slot wins; `slot` is the player's `config_slot`, the pre-rotation configured seat). A condition key may also be a **glob pattern** containing `*` (e.g. `"*-per-5"` to label every per-5 rotation condition at once); an **exact** condition key always beats a pattern, and among matching patterns the **most specific** wins (most non-wildcard characters, ties broken lexicographically), so a broad `"*"` never shadows a narrower `"oss-*-per-5"`. The looked-up value is polymorphic: **if it starts with `"-"` it is a SUFFIX** appended to the composed type (e.g. distinguishing two tweaks of the same model+strategist, `…-Briefed-A` vs `…-Briefed-B`, or tagging a rotation variant `…-Simple-Per-5`); **otherwise it is a full player_type OVERRIDE** that replaces the composed value. This one map therefore serves both the tweak-suffix and the legacy explicit-label needs. Suffixes are skipped for VPAI/Vanilla so the baseline pools across conditions; full overrides still apply.
- **optional fallback**: when a game predates the metadata keys, the old static `(condition, slot)` → player_type map is consulted; otherwise `"Player {id}"`.

The same composition feeds both `panel_data` and `model_token_usage.csv` (single source of truth), replacing the old load-time `(condition, player_id)` merge.

---

## 4. `estimators`: prediction-model producers

Estimators are the pipeline's reason for having stages at all: a `performance.turn_predicted` or `calibration.reliability` analysis needs `predicted_win_probability`, which only exists once a predictor has run. Declare each predictor once in `estimators`; analyses reference it by `id`.

An estimator answers two **independent** questions, and that separation is the whole design:

1. **Where does the model come from?** The answer is `fit`: `"train"` (fit on this run's data, optionally tuned first) or `"pretrained"` (load a saved model dir).
2. **How are the predictions it hands downstream generated?** The answer is `predict`: `"in_sample"` (one model predicts the rows) or `"cross_val"` (k-fold; honest *out-of-fold* predictions).

It always emits the same artifact, regardless of how it was obtained:

> **estimator artifact** = a `predictions.csv` (the input rows + `predicted_win_probability`) and, when applicable, a saved model dir (`metadata.json` + state), a tuning result, and a feature- importance table.

**Computing metrics is not the estimator's job.** Scoring a `predictions.csv` (ROC-AUC, Brier, …) is the separate `prediction.evaluate` / `prediction.compare` analysis step (§6.2). See §4.7 for why that split matters.

### 4.1 Entry shape

```jsonc
{
  "id": "attention",                     // required: unique; what analyses reference in `uses`
  "model": "attention_mlp",              // required: a prediction_models id from models.json
  "fit": "train",                        // required: "train" | "pretrained"
  "predict": "in_sample",                // optional: "in_sample" (default) | "cross_val"
  "enabled": true,

  "params": { /* model __init__ kwargs */ },   // optional explicit hyperparameter override
  "features": { /* §4.5 */ },                   // optional; omit → the estimator's own defaults
  "predict_subset": "all",                      // optional: inference subset (in_sample/pretrained)
  "save_predictions": "reports/estimators/attention/predictions.csv",  // optional, sensible default
  "needs": [],                                  // optional explicit deps (usually inferred)

  "tune":       { /* §4.3, fit == "train" only */ },
  "train":      { /* §4.4, required when fit == "train"      */ },
  "pretrained": { /* §4.6, required when fit == "pretrained" */ }
}
```

### 4.2 The two axes

**`fit`, where the weights come from:**

| `fit`        | What runs                                          | Use when |
|--------------|----------------------------------------------------|----------|
| `train`      | (optional tune →) fit on the current data          | You want a fresh model fit to this run's data. |
| `pretrained` | load a saved model dir, no training                | You have a model trained on a reference dataset and want to apply it here. |

**`predict`, how downstream predictions are generated (only meaningful with `fit: train`):**

| `predict`    | What it emits                                                | Use when |
|--------------|--------------------------------------------------------------|----------|
| `in_sample`  | one model fit on the train subset, predicting `predict_subset` | You want a single deployed model + its predictions. |
| `cross_val`  | k-fold out-of-fold predictions (K models, held-out per fold) | You want **honest** predictions to evaluate/calibrate on. |

`fit: pretrained` always predicts in-sample (you can't cross-validate weights you didn't train here), so `predict` is ignored for it.

This pair is the **train-before-others vs. use-a-pre-trained-estimator** switch the pipeline is built around. A `pretrained` estimator still runs *inference* on the current `tables.turns`, so it depends on `extract` (or on the CSVs already existing) but not on training. Either way, downstream analyses reference the estimator `id` and get its `predictions.csv`.

### 4.3 `tune`: optional Optuna pre-step (gates `train`)

Tuning is its own sub-stage that runs **before** the fit it feeds. You can run it fresh, or skip it entirely by pointing at a previously saved best-params file (a *pre-trained hyperparameter set*).

```jsonc
"tune": {
  "enabled": true,
  "engine": "optuna",
  "search": "hyperparameters",           // "hyperparameters" | "features" | "both"
  "n_trials": 200,
  "objective": "brier_score",            // SINGLE scalar to optimize: brier_score | log_loss
                                         //   | roc_auc | balanced_accuracy
  "n_splits": 5,
  "resample": "none",                    // none | oversample | undersample | combined
  "n_jobs": 1,
  "storage": "reports/estimators/attention/tuning.db",  // null = in-memory, non-resumable
  "save_params": "reports/estimators/attention/best_params.json",
  "load_params": null                    // path → load saved best params, DO NOT run a search
}
```

- Tuning optimizes a **single scalar** `objective` (an objective must be scalar). Reporting *many* metrics is a different concern: that's `prediction.evaluate` (§6.2), which takes a `metrics` list.
- `load_params` set ⇒ tuning is skipped; the saved `best_params.json` is loaded as the hyperparameters. This is the cheap path for "reuse the tuning we already did": a *pre-trained hyperparameter set*.
- **Hyperparameter precedence** (highest wins): explicit `params` → `load_params` → fresh `save_params` → the model class's coded defaults.

### 4.4 `train` (required when `fit == "train"`)

```jsonc
"train": {
  "train_subset": "all",                 // "all" | "non_llm" | "llm" | {"experiments": [...]}
  "resample": "none",
  "save_model": "reports/estimators/attention/",  // in_sample: dir for metadata.json + state
  "n_splits": 5,                         // cross_val only: number of folds
  "save_importance": true                // cross_val only: → feature_importance.csv
}
```

- `train_subset` controls which rows the model is fit on (e.g. `"non_llm"` trains only on Vanilla/Null seats, then predicts on everyone: the held-out generalization setup). For `predict: in_sample`, the inference subset is the entry's top-level `predict_subset`; for `cross_val`, predictions are out-of-fold over the `train_subset` and `predict_subset` is ignored.
- `save_model` applies to `predict: in_sample` (a single fitted model). Saved models written here are exactly what a later run's `pretrained.model_dir` can point at. For `cross_val` there is no single model to deploy, so `save_model` is ignored and the artifact is the OOF `predictions.csv` (+ feature importance).

### 4.5 `features`: selection (omit to use the estimator's own defaults)

`features` is **optional**. When omitted, each estimator uses its own coded `DEFAULT_FEATURES` (or the feature set baked into a tuned `best_params.json`). Provide it only to override per-estimator:

```jsonc
"features": {
  "include": null,                       // null = the estimator's default set; or ["science_*", ...]
  "exclude": ["civ_*"]                   // wildcards allowed; applied after include
}
```

Ignored for `fit: pretrained`: the saved `metadata.json` already carries the selected features.

### 4.6 `pretrained` (required when `fit == "pretrained"`)

```jsonc
"pretrained": {
  "model_dir": "pretrained/attention_mlp/"  // dir with metadata.json (dispatches class)
}
```

No `params` / `features` / `tune` are consulted: the saved `metadata.json` carries the architecture and selected features; the class is resolved from the registry by `metadata.model_class`. Inference runs on the entry's `predict_subset` and writes `save_predictions`.

**`model_dir` is an INPUT, read as-authored; it is NOT re-rooted by `output.suffix`** (only save-paths are; §2.1). The repo ships tracked reference snapshots under `pretrained/<model_id>/` (one per `prediction_models` id); pointing `model_dir` there means the same store serves the default run *and* the `-dev`/`-cross` variants (a `suffix: "-dev"` run loads `pretrained/score/` yet writes to `reports-dev/estimators/score/`).

### 4.7 Why evaluation is a *step*, not a *source*

Earlier drafts had an `evaluate` source alongside `train`/`pretrained`. It was removed because it conflated two separable things:

- **Generating predictions**: in-sample vs. out-of-fold vs. loaded-and-inferred. This genuinely is a property of the estimator-*producer*: it changes which `predictions.csv` comes out. It now lives on the `predict` axis (`in_sample` | `cross_val`).
- **Scoring predictions**: computing ROC-AUC / Brier / log-loss / balanced-accuracy from a `predictions.csv`. That reads an artifact and reports numbers; it owns no model. It is, and always was, an **analysis**: `prediction.evaluate` / `prediction.compare` (§6.2).

So **"evaluate an estimator" = point a `prediction.evaluate` analysis at it.** To evaluate honestly, give the estimator `predict: cross_val` so the analysis scores held-out predictions; to inspect a deployed or pre-trained model's behavior, leave it `in_sample`/`pretrained`. The metrics step is shared, multi-metric, and works identically across all estimators, which is exactly what you'd lose by burying scoring inside each producer.

---

## 5. `adjust`: derived tables (the strength panel)

`adjust` is an **optional list of derived-table stages** that run after `estimators` and before `analyses`. Each entry takes an estimator's win-probabilities and emits a per-player-game table that downstream analyses reference by name via `uses.tables`. Today there is one module, `strength`, but the kind is a list so a run can derive several tables (or the same one from different estimators).

The reason it exists: a `ratings.bradley_terry` fit is not run over raw `panel_data`; it is run over **`adjusted_strength`**, a skill estimate distilled from an estimator's per-turn `predicted_win_probability`. That distillation (late-game weighted average → relative-to-leader → winner enforcement → civilization *or* matched start-cell adjustment) is real work with its own knobs, shared by every rating, so it is its own stage rather than buried inside each `ratings.*` module.

### 5.1 Entry shape

```jsonc
{
  "id": "strength",                      // required: unique; ALSO the produced table name
                                         //   (downstream stages do uses.tables: ["strength"])
  "module": "strength",                  // required: adjust-registry name (currently only "strength")
  "enabled": true,
  "uses": { "estimators": ["attention"] },  // required: the estimator whose P(win) defines strength
                                            //   (creates the estimator → adjust edge)
  "save": "reports/adjust/player_strength_panel.csv",  // optional, sensible default
  "needs": [],                           // optional explicit deps (usually inferred from `uses`)

  "params": {                            // module-specific; unlisted keys → coded defaults
    "turn_progress_min": 0.2,            // ignore the opening; average over late-game turns only
    "weight": "turn_progress",           // weight each turn's P(win) by progress when averaging
    "relative_to": "game_leader",        // "game_leader" normalizes each seat to its game's strongest seat;
                                         //   "none"/null (default) ⇒ strength is the raw late-game P(win)
    "enforce_winner": true,              // force the actual winner to the top raw strength
                                         //   (= relative_strength = 1.0 only under "game_leader")
    "civ_adjust": "ols_logit",           // UNCONTROLLED games: subtract civ effects (OLS on the logit);
                                         //   "none" leaves adjusted = relative_strength
    "block": "auto",                     // CONTROLLED games: matched final-seat-cell adjustment (replaces civ_adjust):
                                         //   "none" | "start_cell" | "auto" (= start_cell when controlled)
    "baseline_experiment": null,         // explicit per-cell baseline source (e.g. a pure VP self-play experiment id);
                                          //   null ⇒ implicit per-experiment VPAI extraction. Need not be catalog-listed.
    "post_cell_normalize": "none",       // optional final re-normalization after the cell effect: "none" | "relative_to_leader"
  }
}
```

- The stage `id` doubles as the **table name** it registers, exactly as an estimator `id` names its `predictions.csv`. A consumer writes `uses.tables: ["strength"]` (not the stage's literal output path), and the harness adds the edge.
- `uses.estimators` is **required and single-source** in practice: strength is defined relative to one predictor's win-probabilities. Point it at a `cross_val` estimator for out-of-fold-honest strength, or an `in_sample`/`pretrained` one to mirror a deployed model.
- The emitted table is per-player-game with at least `game_id, player_id, player_type, civilization, relative_strength, logit_strength, adjusted_strength`: the exact columns `ratings.*` require (§6.2), plus audit columns, plus `seed`/`seating_rotation` joined from `games` and `config_slot` carried from `panel_data` for the controlled-design diagnostics. Because the estimator's `predictions.csv` does **not** carry the composed identity, `player_type` (and `config_slot`) are joined from `panel_data` by `(game_id, player_id)`; `civilization` comes from the predictions (panel as fallback). `enforce_winner:true` forces the actual winner to the top strength (under `relative_to:"game_leader"` that is `relative_strength = 1.0`; under `relative_to` unset/`"none"` the winner is bumped to the top raw strength and `relative_strength` mirrors the raw value), but `logit_strength` is always finite via clipping before any logit-scale adjustment.
- **The panel retains every experiment**: it is *not* filtered to LLM seats. Vanilla/Null seats stay in so `ratings.* ref="Vanilla"` and the baseline pathways (§5.1) resolve; LLM-only narrowing is each analysis's `only_llm` filter, not baked into the panel.
- **Condition completeness lives in the global filter (§3.1), not here.** `data.filter.min_condition_completeness` drops, as a whole, every controlled condition whose `seed × seating_rotation` slot coverage falls below the threshold. Because it filters the estimator predictions and every analysis table, incomplete conditions never reach the strength panel, the ratings, the baselines, or the audit trails in the first place, so no per-stage option is needed.
- `civ_adjust: "none"` skips the OLS step (then `adjusted_strength == relative_strength`); any other value selects an adjustment scheme (currently `ols_logit`). The `ols_logit` fit is `logit_strength ~ C(civilization, Sum) + C(player_type, Treatment(reference=<vanilla_label>))` (reference from the catalog's vanilla label), fit over **all** panel rows; only uncontrolled rows have the civ effect subtracted.
- **Controlled-design adjustment (`block`).** When a game carries controlled seeds **and** seating (`seed != -1` and `seating_rotation != -1` in the `games` table), `block` replaces `civ_adjust` on those rows with a **matched final-seat cell** correction: it subtracts the per-`(seed, player_id)` **Vanilla VPAI baseline**. `seed` identifies the generated map/start layout, and `player_id` identifies the final in-game seat/start position after rotation. `config_slot` is retained for diagnostics about which configured model rotated into that cell, but it is not the baseline cell. "Vanilla VPAI" means rows whose raw model is VPAI when available and whose composed `player_type` is the catalog's vanilla label (`Vanilla` by default); `Null` is not baseline evidence. Because the final-seat cell fixes the seat-bound civilization, this subsumes the civ adjustment. Uncontrolled rows fall back to `civ_adjust`; `block: "none"` ⇒ pure legacy behavior except for harmless audit columns/files. The baseline is built via one or both pathways:
  - **Explicit** (`baseline_experiment` set, e.g. a pure VP self-play condition): for each `(seed, player_id)` cell, average the designated experiment's Vanilla/VPAI `logit_strength` rows. In seated mode, rotations for a seed are repeated observations of the same generated map/start cells, so each final-seat cell's explicit baseline is the logit-scale mean of those repeated Vanilla observations.
  - **Implicit** (`baseline_experiment: null`, default): for each controlled experiment and `(seed, player_id)` cell, average that experiment's own Vanilla/VPAI `logit_strength` rows. Incomplete self-coverage is expected (rotation sparsity) and is **not fatal**: a controlled seat whose own `(seed, player_id)` cell has no VPAI counterpart **falls back to the uncontrolled adjustment** (`civ_adjust` when `ols_logit`, else `relative_strength`) and is named in a WARN; every cell that does have a Vanilla observation still uses its start-cell baseline. A per-row `adjust_method` column (`cell|civ|relative`) records which path each row took. Cell-adjusted rows additionally persist `cell_logit_advantage` = `logit_strength − cell_baseline`: the exact start-cell advantage on the logit scale, captured **before** any `post_cell_normalize` so it always means "your strength − matched Vanilla VPAI baseline in this cell"; it is `NaN` for non-cell (`civ`/`relative`) rows.

  The pathway selected by `baseline_experiment` feeds `adjusted_strength`: `null` selects implicit; a string selects explicit. If explicit is selected, implicit is still computed per controlled experiment wherever Vanilla rows exist, so the report can show implicit-vs-explicit deltas. **A missing selected *explicit* baseline cell is fatal** (the designated `baseline_experiment` is meant to span the whole grid); a missing selected *implicit* cell warns and falls back. Per-model coverage gaps, cells with no VPAI baseline in a comparison pathway, and player types disconnected from `Vanilla` are **warned, never fatal** (keep all games, proceed).
- **Estimator/filter precondition.** The strength stage consumes the referenced estimator's saved predictions; it does not re-infer missing rows. Controlled `block != "none"` therefore expects the estimator artifact to include the Vanilla rows needed by the selected baseline pathway. A global `data.filter` or estimator `predict_subset` that drops Vanilla references (for example `only_llm`) removes the baseline evidence: under **explicit** that is fatal at the missing cell, and under **implicit** the affected cells fall back to `civ_adjust` (which itself needs the vanilla reference level present, or its OLS fit fails). Keep estimator prediction broad and narrow later in each analysis.
- **Intermediate adjustment diagnostics: always written (no config).** Like every other audit trail, the stage *always* emits the per-group values it subtracts, next to the strength panel (in the directory of `save`, default `reports/adjust/`), so the correction can be inspected without opting in:
  - `civ_effects.csv`: per-`civilization` OLS-logit effect table (`civilization, civ_effect, n_rows`), the **civilization-level effect** from the uncontrolled (`civ_adjust`) path.
  - `cell_baseline.csv`, the **VPAI seating×seed effect** from the controlled (`block`) path: `experiment, pathway, seed, player_id, civilization, cell_baseline, n_vanilla, win_rate, n_games, n_models, has_vanilla_baseline, vanilla_connected`. It carries every pathway that actually ran (`pathway ∈ {explicit, implicit}`); a default `baseline_experiment:null` run normally contains only implicit rows. When both explicit and implicit exist for a condition/cell, the report compares `implicit - explicit` on the logit scale.
  - `cell_coverage.csv`, the **controlled-design cell coverage report**: for each controlled experiment, which `(seed, player_id)` cells of the **entirety** reference grid it is missing. Columns: `experiment, seed, player_id, civilization, in_entirety, n_rows, n_vanilla, has_baseline, missing` (`missing = true` ⇒ the experiment has no rows for that cell). The "entirety" is the full baseline cell set: the `baseline_experiment`'s cells when set (a pure VP self-play spans every seat across every seed/rotation), else the union of `(seed, player_id)` cells observed across the controlled subset. This is a **report-only** diagnostic (WARN, never fatal; distinct from the hard-error "a row needs adjustment but its selected baseline cell is missing"); written only when controlled rows exist.

  `performance.experiment_completeness` derives compact game-level completeness tables from canonical panel, games, and token telemetry, using the strength stage's baseline configuration as its own report section. `experiment_completeness.csv` contains `experiment, required_games, present_games, missing_games, excluded_games, completeness_pct, repeated_slots, failed_turn_count, avg_failure_count, failure_pct, warning`. The average is the mean failed turn count per player trace. The percentage is total failed turns divided by total selected turn roots. `excluded_games` counts games rejected by the decision-failure cutoff. `decision_turn_failures.csv` identifies every affected `experiment, game_id, player_id`, lists its failed turns, and includes `excluded_game`. Telemetry diagnostics retain all player traces for the selected games, including rejected traces and players outside the player filter. `repeated_games.csv` lists exact duplicate game ids by seed and rotation, and gap or issue detail tables are written when needed. `required_games` is the full controlled `seed × seating_rotation` grid, using the explicit `baseline_experiment` grid when configured and present, else the controlled union. `present_games` counts distinct accepted `game_id`s, and `warning` is `ok` or a readable issue summary rather than a boolean. An experiment without usable telemetry receives a completeness warning.

  `condition_progress.csv` contains `experiment, player_type, completed_games, required_games, completed_at` for each controlled condition's non-baseline player types. `completed_games` counts occupied accepted slots, excluding duplicates. `completed_at` is an ISO UTC timestamp when every required slot has a dated accepted game; it is blank otherwise. This saved table supplies the front-page announcements (§7).

  A mixed dataset writes these files; each is simply empty/absent when its path didn't run (`cell_baseline.csv`/`cell_coverage.csv` are empty on a fully uncontrolled run). `performance.strength_panel` always surfaces whichever exist (§6.2); no flag.

---

## 6. `analyses`: the pluggable modules

A list of analysis stages. Every entry shares a common envelope; the `params` block is module-specific (catalog in §6.2). Order in the list is irrelevant; the DAG decides execution.

### 6.1 Common envelope

```jsonc
{
  "id": "bt_main",                       // required: unique stage id
  "module": "ratings.bradley_terry",     // required: analysis registry name
  "name": "Main ratings",                // optional: friendly heading, overrides the module default on the report
  "description": "The headline Elo table.",  // optional: one-line module description (overrides the module default)
  "enabled": true,
  "needs": [],                           // optional explicit deps
  "uses": {                              // optional artifact references (create auto-edges)
    "estimators": ["attention", "score"],   // estimator ids → their predictions.csv
    "tables": ["panel", "strength"],         // canonical (data.tables) OR an adjust stage's table (§5)
    "analyses": ["bt_main"]                  // analysis ids → persisted table/figure artifacts
  },
  "filter": "late_game",                 // optional: preset name, inline object, or list (§3.1);
                                         //   NARROWS the global filter for this stage
  "params": { /* module-specific, see §6.2 */ }
}
```

- `uses.estimators` is how estimator-consuming modules get win-probabilities. For implemented modules whose analysis class opts in to the all-estimator default (`prediction.*`, `calibration.reliability`, and `calibration.loss_by_progress`), omit it (or set an empty list) to consume every enabled estimator; provide ids only to narrow to a subset. The DAG adds edges to the resolved estimators either way. `performance.turn_predicted` is the exception: it reads the strength stage's estimator unless `uses.estimators` names one override.
- `name` and `description` are optional per-stage identity overrides. Every analysis module ships a coded friendly name and one-line description (§6.3). Rating module instances resolve a distinct coded identity from `params.group_by`, so `bt_main` and `bt_strategy` receive different headings without config name fields. A supplied stage `name` or `description` still replaces the resolved identity for that section. When omitted, the resolved module identity is used.
- `uses.tables` names a canonical table (`data.tables`) or one an `adjust` stage emits (§5). Strength-based ratings consume the derived `strength` table this way; referencing it adds the edge to the `adjust` stage (and transitively to its estimator). Observed matchup analyses can consume canonical tables such as `panel`.
- `uses.analyses` names analysis stages whose persisted artifacts are inputs. It creates a strict dependency edge; unknown, disabled, and self references are errors even on disabled consumers. `performance.usage_efficiency`, for example, reads `ratings.csv` from the first declared ratings stage.
- `filter` accepts the same preset-name / inline / list forms as `data.filter` (§3.1). It is intersected with the resolved global filter; a stage can only narrow, never widen.

### 6.2 Module params catalog

Every registry name `civ-bench` will ship, with its key params. Unlisted params fall back to coded defaults; unknown params are validation errors.

#### `ratings.*`: fitted ratings, strength matchups, and observed matchups

The fitted ratings (`bradley_terry`, `plackett_luce`) and strength matchup view rate `adjusted_strength`, so they reference an `adjust` stage's table via `uses.tables: ["strength"]` and depend transitively on the estimator that fed the `adjust` stage. The observed outcome matchup view reads the canonical `panel` table directly because it reports actual `is_winner` rates and score-ratio margins.

Two cross-cutting params apply to both fitted ratings (`bradley_terry`, `plackett_luce`):

- **`group_by`** (default `["player_type"]`): the identity the rating is fit over. Extra dimensions past the base must name a grouping in top-level `groupings` (§3.2); the rated identity is the composite formed by joining the dimension values with `-`. So **per-strategy Elo is not a separate module**; it is `group_by: ["player_type", "strategy"]` on the ordinary BT/PL fit. `min_games` then filters *composite* identities post-fit, and ref/vanilla re-centering is preserved.
- **`bootstrap`** (default omitted → point estimate only): when set, a shared resample-and-refit helper draws games with replacement (`stratified` by experiment by default), re-runs the same fit `n` times, and emits percentile CIs + rank stability alongside the point estimate. Bootstrap CIs are **not a separate module**; the resampling is seeded from the top-level `seed`. When the `strength` table uses a controlled-design `block` adjustment, the resample **re-runs the `adjust/strength` fit inside each replicate** (not a fixed panel) so the start-cell baseline's uncertainty is reflected in the CIs.

When `group_by` includes the `strategy` grouping, fitted ratings render a heatmap
matching the legacy Vox Deorum analysis: rows are player types, columns are
`General` plus the configured strategy labels (normally Domination, Culture,
Diplomatic, Science), and annotations include Elo, SE, and game support.

```jsonc
// ratings.bradley_terry: BT MLE with pairwise score weights (R: BradleyTerry2)
{ "module": "ratings.bradley_terry",
  "uses": { "tables": ["strength"] },
  "params": { "group_by": ["player_type"],          // ["player_type","strategy"] → per-strategy Elo
              "weighted": true, "ref": "Vanilla", "min_games": 5, "only_llm": false,
              "bootstrap": null } }                  // or { "n": 1000, "stratified": true } for CIs

// ratings.plackett_luce: PL MLE over per-game rankings (R: PlackettLuce)
{ "module": "ratings.plackett_luce",
  "uses": { "tables": ["strength"] },
  "params": { "group_by": ["player_type"],          // ["player_type","strategy"] → same heatmap view
              "ref": "Vanilla", "min_games": 5, "bootstrap": null } }

// ratings.matchups: adjusted-strength head-to-head matrices + OLS validation
{ "module": "ratings.matchups",
  "uses": { "tables": ["strength"] },
  "params": { "mode": "both", "validate_ols": true,
              "display": "matrix" } }  // display: matrix | vs_reference

// ratings.outcome_matchups: observed win rates + score_ratio margins
{ "module": "ratings.outcome_matchups",
  "uses": { "tables": ["panel"] },
  "params": { "include_score_ratio": true, "display": "matrix" } }
```

`ratings.outcome_matchups` reports wins per player appearance. A player type
occupying two seats in one game contributes two appearances. Each matrix cell
uses games shared by the row and column player types; repeated opponents of the
same type do not multiply the row player's appearances. Summaries and victory
charts show raw wins / player appearances (for example, `10/48`) alongside rates;
the `vs_reference` table includes the raw `wins` and appearance count `n`.
The expected victory rate assumes equal chances: `1 / player count`, or
**12.5% in eight-player games**.
Player counts come from the full panel before analysis filters. For mixed game
sizes, expected rates are averaged over the same player appearances as observed
rates. The `expected_win_rate` table stores these baselines, and `vs_reference`
includes an `expected_win_rate` column. Its `p_value_win_rate` tests against this
baseline when game sizes are fixed and the player type appears once per game;
otherwise the p-value is blank.

**Optional `ratings.*` (off by default, registry-reserved, shipped only in `benchmark.full.template.json`):**

```jsonc
// ratings.ablation_bt: incrementally add each player type's games (chronologically) and track
//   Elo convergence; isolates each game's marginal contribution. Writes the ablation_bt_* artifacts.
{ "module": "ratings.ablation_bt", "enabled": false,
  "uses": { "tables": ["strength"] }, "params": { "weighted": true } }

// ratings.vanilla_slot_effect: seat/start-position confound check. With the strength stage's start-cell
//   adjustment ON it doubles as the VALIDATION that the cell effect nulled the slot effect: significant on the
//   raw panel, ~null on the adjusted panel.
{ "module": "ratings.vanilla_slot_effect", "enabled": false,
  "uses": { "tables": ["strength"] }, "params": {} }
```

#### `prediction.*`: score one or more estimators

`prediction.*` is the *scoring* family: it answers "how good is the win predictor". The implemented prediction modules opt in to scoring every enabled estimator by default; add `uses.estimators` only when a stage should compare a subset. Calibration views live in their own family (`calibration.*`, below).

```jsonc
// prediction.evaluate: metrics table across estimators (ROC-AUC/Brier/log-loss/bal-acc)
{ "module": "prediction.evaluate",
  "params": { "metrics": ["roc_auc","brier_score","log_loss","balanced_accuracy"] } }

// prediction.compare: side-by-side comparison table + ranking
{ "module": "prediction.compare",
  "params": {} }

// Explicit subset override, when needed:
{ "module": "prediction.evaluate",
  "uses": { "estimators": ["score","attention"] },
  "params": { "metrics": ["brier_score","log_loss"] } }
```

**Optional `prediction.*` (off by default, registry-reserved, shipped only in `benchmark.full.template.json`):**

```jsonc
// prediction.winner_trajectories: P(win) trajectories for eventual winners
{ "module": "prediction.winner_trajectories", "enabled": false,
  "uses": { "estimators": ["attention"] }, "params": { "sample_games": 12 } }

// prediction.elo_comparison: predicted-strength vs rating-based Elo cross-check
{ "module": "prediction.elo_comparison", "enabled": false,
  "uses": { "estimators": ["attention"] }, "needs": ["bt_main"], "params": {} }

// prediction.context_slicing: metrics sliced by experiment / player type / turn range
{ "module": "prediction.context_slicing", "enabled": false,
  "uses": { "estimators": ["attention"] }, "params": { "by": ["experiment","player_type"] } }
```

#### `calibration.*`: calibration of estimators

Two single-purpose views of how well estimator probabilities are calibrated: one across the **probability** axis, one across the **game-progress** axis. Both implemented calibration views opt in to consuming every enabled estimator by default; add `uses.estimators` only to narrow the comparison.

```jsonc
// calibration.reliability, reliability diagram: observed win-rate vs predicted P(win) per bin
{ "module": "calibration.reliability",
  "params": { "n_bins": 10 } }

// calibration.loss_by_progress: Brier/log-loss across turn_progress (game-stage) bins
{ "module": "calibration.loss_by_progress",
  "params": { "n_bins": 20, "metrics": ["brier_score","log_loss"] } }
```

#### `performance.*`: strength + score, some need the `strength` table

```jsonc
// performance.score_ratio: per-player score-ratio regressions (consumes panel)
{ "module": "performance.score_ratio",
  "params": { "target": "score_ratio", "predictors": ["player_type","civilization"] } }

// performance.strength_panel: summarizes the adjust stage's strength table by player type.
//   It CONSUMES the `strength` table (does not derive it; that's the adjust stage, §5).
//   It always reports per-identity n_games + bootstrap CI AND flags identities below
//   `min_games_preliminary` as preliminary (the controlled design exists to estimate strength from
//   very few games, so the small-sample basis is surfaced, not hidden).
//   It also surfaces the adjust stage's always-written adjustment diagnostics: the civilization-level
//   effect table and the VPAI seating×seed effect (cell baseline, explicit/implicit pathways,
//   §5.1), whenever they exist. No flag on the diagnostics.
//   CONTROLLED-ONLY alternative view: when the panel has cell-adjusted rows (a non-null
//   `cell_logit_advantage`, §5.1), it adds a `by_identity_logit_advantage` table + `logit_advantage`
//   figure: the per-identity mean start-cell advantage on the logit scale (0 = the matched Vanilla
//   cell baseline). Vanilla is summarized like every identity (NOT pinned to 0), so the report shows
//   where it actually lands. Absent on a fully uncontrolled run.
{ "module": "performance.strength_panel",
  "uses": { "tables": ["strength"] },
  "params": { "metric": "adjusted_strength", "by": "player_type",
              "min_games_preliminary": 5 } }   // < this ⇒ flagged preliminary (defaults to ratings min_games)

// performance.experiment_completeness: compact controlled-design game-grid completeness.
//   Emits a standalone report section with per-experiment required/present/missing games,
//   failed-turn rates, readable warning text, exact repeated game ids, and failure details.
{ "module": "performance.experiment_completeness",
  "uses": { "tables": ["strength", "tokens"] },
  "params": {} }

// performance.turn_predicted: predicted win probability over the game,
//   from the strength stage's estimator
{ "module": "performance.turn_predicted",
  "uses": { "tables": ["strength"] },
  "params": { "aggregate": "mean", "by": "player_type" } }

// performance.controlled_seed_report: report-ready tables for the controlled-seed chapter
//   the report renders automatically (§7.1). Emits seed_player_summary,
//   seed_player_probability, and seed_player_index; the renderer completes the grid.
//   uses.analyses optionally lists stages whose Matched Maps tabs render beside
//   Strength and Focus (§7.1).
{ "module": "performance.controlled_seed_report",
  "uses": { "estimators": ["attention"], "tables": ["strength"], "analyses": ["beh_flavors"] },
  "params": {} }
```

#### `performance.game_log`: per-game report data

```jsonc
{ "id": "game_log", "module": "performance.game_log", "enabled": true,
  "uses": {}, "params": {} }
```

The Game Log reads the canonical `games` and `panel` tables and emits one
`games` table plus one `game_players` table. It records the game date, map
identity, outcome, controlled seed and rotation, strategist condition label,
and the player seats. Rows are ordered newest first. With condition pairing
enabled, labels use the same `<strategist> | <condition>` wording as the rest
of the report; VPAI seats are marked so report pages can hide them. The stage
also records ordering metadata and the latest game for the report renderer.

The stage is required when `report.replay.enabled` is true. It has no figures
or default inline tables. Set `params.condition_pairing` to override the
shared presentation pairing for this stage.

**`performance.turn_predicted` in detail.** The module plots predicted win
probability over the game from exactly one estimator: the one listed in the
strength adjust stage's `uses.estimators` (for example `attention`). An
optional `uses.estimators` on the analysis may name at most one estimator to
override that choice; it does not default to all enabled estimators. The
entry must reference exactly one strength table in `uses.tables`, or config
validation rejects it. Key behavior:

- **Splitting.** When `by` is `player_type` and condition pairing is enabled
  (§2.2), each player_type splits into strategist and condition, like the
  Matched Maps pages; the stage accepts `params.condition_pairing` as an
  override. The strength stage's `params.baseline_experiment` games form the
  VPAI reference curve (without one, every Vanilla seat does).
- **Curves.** Each run is interpolated onto a fixed 101-point turn-progress
  grid (0 to 1), and each grid point averages the runs that cover it.
- **Tables.** `over_progress` (strategist, condition, turn_progress,
  mean_predicted_win_probability, n_runs) and `by_identity` (`<by>`,
  strategist, condition, mean_predicted, n_rows, n_games).
- **Rendering.** The report renders the same interactive chart as the Matched
  Maps seat pages (strategist checkboxes, thick VPAI line) inline on the
  Performance page. There is no figure file.

**`performance.controlled_seed_report` in detail.** The module aggregates the
controlled design by `(seed, player_id)` cell so the report can expose
game-to-game variation per seed. It consumes the canonical `games` table
(`seed`, `seating_rotation`, experiment membership), the canonical `panel`
table (player identity, civilization, the four victory-focus ratios), exactly
one strength adjust table (`weighted_strength`, `adjusted_strength`), and
exactly one estimator (per-turn `predicted_win_probability`). Key rules:

- **Condition pairing must be enabled** (§2.2); the stage also accepts
  `params.condition_pairing` as an override. `player_type` splits into its base
  strategist identity and display condition (base label, then configured
  suffixes); each cell averages every unique run for its
  `(seed, player_id, strategist, condition)` key; seating rotations and
  genuine repeated games contribute equally, and no confidence interval is
  computed.
- **The dedicated VPAI baseline.** The configured strength stage's
  `baseline_experiment` (§5.1) is the sole Vanilla source, and it must be set.
  Baseline rows bypass condition splitting and canonicalize to
  `strategist = "Vanilla"`, `condition = "Vanilla"` (the catalog label); the
  report places that row before strategist rows. Baseline strength and
  probability evidence matches by `(seed, player_id)`, averaging all baseline
  rotations and repeated games; the Vanilla row carries its own unique-game run
  count, and treatment rows keep their own run counts even when the matched
  baseline mean came from a different number of runs. Vanilla opponents from
  treatment conditions are never substituted when the baseline is missing.
- **Curves.** Each individual game's predicted-win-probability curve is
  linearly interpolated onto a fixed 101-point normalized-progress grid from 0
  to 1, only inside that run's observed progress range (no extrapolation, no
  endpoint holding); each grid point averages exactly the runs covering it.
  One prediction per `(game_id, player_id, turn_progress)` is required;
  conflicting duplicate progress points are an analysis error, as are duplicate
  panel or strength records per `(game_id, player_id)`. Runs are unique
  `game_id`s.
- **Scope.** Only rows with both `seed != -1` and `seating_rotation != -1`
  participate; uncontrolled rows are excluded from mixed inputs, and an input
  with no controlled rows is a clear analysis error. The module reads its
  inputs as a census of the controlled design: it does not apply the global
  `data.filter` (an `only_llm` or `min_games` filter would punch holes in the
  seed grid and remove the dedicated VPAI baseline). Narrow the inputs by
  controlling what is extracted.
- **Focus.** The four final victory-focus ratios average per cell before the
  dominant focus is chosen; exact ties resolve in the order Domination,
  Culture, Diplomatic, Science.
- **Baseline gaps are not fatal.** A `(seed, player_id)` pair without matched
  baseline rows keeps its treatment rows with blank differences and a visible
  page note; a pair without usable prediction rows keeps its scalar summary and
  marks the curve unavailable.
- **Extra Matched Maps tabs.** An optional `uses.analyses` list adds one tab to
  the Matched Maps chapter per entry (§7.1), rendered in list order after
  Strength and Focus. Only modules in `MATCHED_MAPS_TAB_MODULES` (currently
  `behavior.flavors`, `behavior.diplomacy`, `behavior.commitment`, and
  `behavior.policies`) may be listed; any other module is a config validation
  error. A module may offer more than one tab (`behavior.commitment` offers
  Commitment and Grand strategy). The stage records the list as `metadata.tabs` so the report renders
  the tabs without reading the config.

The emitted tables carry ordering and color metadata in the result manifest
(catalog strategist order, configured condition order, `configs/models.json`
colors), so the report stage rebuilds the presentation without reading the
catalogs or canonical tables.

**Optional `performance.*` (off by default, registry-reserved, shipped only in `benchmark.full.template.json`):**

```jsonc
// performance.permutation_importance: grouped permutation importance over feature families
{ "module": "performance.permutation_importance", "enabled": false,
  "uses": { "estimators": ["attention"] },
  "params": { "n_repeats": 20, "groups": "feature_families" } }
```

#### `performance.usage_efficiency`: model usage and efficiency

```jsonc
// Usage and cost summaries plus rating efficiency, grouped by player identity.
{ "module": "performance.usage_efficiency",
  "uses": { "tables": ["tokens"], "analyses": ["bt_main"] },
  "params": { "currency": "usd", "log_x": true, "annotate": false } }
```

The module writes `usage` and `usage_vs_rating` tables. It emits three static
figures, `cost`, `input_tokens`, and `output_tokens`, as downloadable attachments,
each showing the average metric per player per game. Models used by the same
player in a game are summed
before averaging across complete player-game records. Output tokens include
reasoning. Token averages remain available when pricing is unknown; cost averages
require complete telemetry and known prices. Costs exclude cache discounts.
The `usage` table includes unrated identities and baselines. Charts omit the
Vanilla and Null baselines, and `usage_vs_rating` contains only rated identities.
All three static charts follow the same Elo ordering, with unrated identities
appended by cost. `condition_pairing` follows the shared presentation settings.

For each metric, the interactive chart fits Elo as `intercept + slope *
log10(average usage)` across player identities and conditions, with one equally
weighted observation per identity. The efficiency score is observed Elo minus
fitted Elo; higher is better. At least three finite rated identities with
positive usage and two distinct usage values are required for a fit. Zero values
have no efficiency score and are excluded from logarithmic views. `log_x:false`
uses a linear display axis while keeping the logarithmic regression.

Only `usage_vs_rating` appears inline by default. The resource selector switches
cost, input tokens, and output tokens together with their fitted dotted curve,
equation, R², and residual scores. R² is `1 - sum(residual²) / sum((Elo - mean(Elo))²)`
over the observations used in each fit, and is saved as `r_squared` in that fit's
metadata. Constant Elo has no defined R² and displays `N/A`. A neutral table
tooltip highlights Elo, baseline Elo, and Elo above the selected fit. The baseline
line uses Vanilla's rating, labeled VPAI, when available, otherwise the 1500 Elo
reference. When the ratings table contains a finite rating for the configured
Null identity, a second line labeled `Null baseline` marks the lower-bound
reference in all three resource views. Its rating is saved as `null_baseline_elo`
in metadata, or null when unavailable. The line does not clip lower-rated points
or contribute an observation to the fitted curve. `annotate:true` labels
the identities with the highest and lowest residuals for the selected metric.
The summary names the cost-efficiency extremes. Fits are descriptive comparisons
within the displayed cohort, not predictions of gains from spending more.

`usage_skill_fits` metadata and output columns use the metric keys `cost`, `input`,
and `output`, such as `expected_elo_cost` and `efficiency_elo_cost`. Tables include
`avg_cost_per_player_game`, `avg_input`, `avg_output`, and `player_games`.
`token_complete_player_games` and `token_na_player_games` count token coverage;
`cost_complete_player_games` and `cost_na_player_games` count cost coverage.
`complete_player_games` and `na_player_games` are aliases for the cost counts.
`avg_cost_per_game` aliases `avg_cost_per_player_game`; `games`, `complete_games`,
and `na_games` count distinct games with complete or incomplete cost records.

**Optional `exploratory.*` (off by default, registry-reserved, shipped only in `benchmark.full.template.json`):**

```jsonc
{ "module": "exploratory.panel", "enabled": false, "params": {} }
{ "module": "exploratory.turn",  "enabled": false, "params": {} }
// strategy_profiles joins strategy mix against adjusted_strength → needs the strength table
{ "module": "exploratory.strategy_profiles", "enabled": false,
  "uses": { "tables": ["strength"] }, "params": { "by": "player_type" } }
```

#### `behavior.*`: descriptive behavior analyses

The `behavior.*` family reads the canonical `behavior` table (§3.0) and profiles how each strategist drives the in-game AI: its flavor settings, its diplomatic persona and stances, its strategic commitment, and its policy paths. All params are optional and unknown keys are config errors.

```jsonc
{ "id": "beh_flavors",    "module": "behavior.flavors",    "enabled": true, "params": {} }
{ "id": "beh_diplomacy",  "module": "behavior.diplomacy",  "enabled": true, "params": { "rate": "per_100_turns" } }
{ "id": "beh_commitment", "module": "behavior.commitment", "enabled": true, "params": {} }
{ "id": "beh_policies",   "module": "behavior.policies",   "enabled": true, "params": { "baseline": "vanilla-standard-fixed" } }
```

Params shared by all four modules:

- **`baseline`**: `"completed"`, one experiment id, or a non-empty list of experiment ids. `"completed"` means every experiment that fills every controlled (seed, seating_rotation) slot, after dropping problem games and decision-failure games; its pool keeps strategist players only (Null and Vanilla rows are left out), and completed experiments are part of their own baseline. With a named experiment or list, those experiments' rows are the pool and leave the relative view. When `baseline` is omitted, `behavior.flavors` and `behavior.commitment` use `"completed"`, and `behavior.diplomacy` and `behavior.policies` use the first enabled strength stage's `baseline_experiment` (the in-game AI, §5.1).
- **`by`**: the grouping column; default `"player_type"`.
- **`bootstrap_n`** (integer >= 1, default 1000) and **`ci_level`** (default 0.95): the bootstrap confidence intervals.
- **`condition_pairing`**: the same override shape as other modules (§2.2).
- **`rate`**: `"per_100_turns"` (default) or `"per_game"`; accepted by `behavior.diplomacy` and `behavior.commitment` only. It scales the count columns by turns alive (`survival_turn` + 1).

Per-module params:

- `behavior.flavors`: optional **`flavors`**, a non-empty list of flavor names (each must be extracted with `avg`); the default is every extracted flavor. Older behavior tables with fewer flavors still work: the module shows the extracted ones and lists the rest in metadata as `flavors_not_extracted`.
- `behavior.diplomacy`: optional **`traits`**, a list of persona names; the default is DiplomaticBalance, Friendliness, WorkWithWillingness, WorkAgainstWillingness, Loyalty, DenounceWillingness, Forgiveness, Meanness, Neediness, Chattiness, and DeceptiveBias.
- `behavior.policies`: optional **`branches`**, a list of extracted policy branches.

**Two views.** Every module produces two views of the same metrics. The **relative** view (the default, shown first) covers controlled players only: each player's value minus the baseline pool's mean at the same (seed, player_id) cell. The **absolute** view covers every filtered player. Summaries per group and metric give mean, median, a bootstrap CI resampling whole games (seeded from the run seed), `n_players`, and `n_games`. If there is no baseline, no controlled games, or no complete experiment, only the absolute view is written and the reason appears in the section tooltip. The baseline pool is read from the unfiltered table, so player filters never remove it.

**The flavors page.** The in-game AI records no flavor values, and 50 is its own balanced value, so the relative baseline is the completed-experiment average. Its two tables, `flavors_relative` and `flavors_absolute`, titled `Relative flavor setting` and `Average flavor setting`, render as HTML heatmaps (§7): one column per flavor under a short name (for example Off for Offense), grouped into Military, Nuclear, Naval and air, Economy, and Other, with the full name and the set-flavors tool description in the header tooltip. The Null strategist, which never moves its flavors off 50, is left out of both tables entirely. In controlled runs both views pin one extra first row with the baseline pool itself (rows with `row_kind == "baseline"`, others `"group"`): it carries the capitalized baseline label (for example `Completed-experiment average` or `Matched 'vanilla-standard'`), holds the pool's absolute mean, SD, and player and game counts colored on the fixed 0 to 100 scale in both views, and has no CI or median. The heatmap spec lists the label under `baseline_rows`, and the report renders those rows unsigned and labels their value line `Average`. Flavors gated by `data.extract.behavior.flavor_gates` (§3.0), by default the nuclear and air ones, read blank until a player gains access, and their header tooltip adds one sentence saying when the flavor starts to count. Rows read `Strategist | Condition` when condition pairing is on, and Vanilla is pinned at the top. Absolute colors use a fixed 0 to 100 scale with 50 at the midpoint; relative colors are the difference in baseline-pool SDs, full color at 2 SDs. Cell tooltips lead with the row label and the column name, then grid lines: the value (labeled `Difference` or `Average`), the CI, an `SD` line for a baseline row in the signed table, the in-game range (`Range`, the mean of each player's min and max, when both are extracted; in the relative view both are minus the baseline at the player's own seed and seat, signed, and the line reads `mean min to max, vs. baseline`), and the player and game counts. In controlled runs with a baseline the module also writes `flavors_by_seed` (keyed by `seed`) and `flavors_by_seat` (keyed by `seed` and `player_id`): absolute tables on the fixed 0 to 100 scale, one row per (cell, row, flavor) with `mean`, `difference` (the mean relative value, the player minus the baseline at its own seed and seat), `n_players`, `n_games`, `metric_group`, and `color_position`, each seed or seat pinning the baseline pool's average for that cell as its first row (`row_kind == "baseline"`) and carrying no bootstrap CI. They are downloadable supporting files, not inline on the behavior page, and feed the Matched Maps Strategy tab through `metadata.matched_maps` (`label`, `tip`, `seed_table`, `seat_table`). They leave out Vanilla players, since the pinned row is the reference. Uncontrolled runs write neither table.

**The diplomacy page.** Persona traits run from 1 to 10, and the relative baseline defaults to the matched in-game AI. The page has three views. **Relative** and **Absolute** hold the persona tables `diplomacy_relative` and `diplomacy_absolute`, titled `Relative persona trait` and `Average persona trait`. They render as HTML heatmaps laid out like the flavors page: one column per trait under a short name (for example Frd for Friendliness), grouped into Competitiveness, War and peace, Diplomacy, City-states, and Personality (`PERSONA_INFO`), with the full name and the set-persona tool description in the header tooltip. They also share its pinned baseline row (here `Matched in-game AI`, on the 1 to 10 scale), its tooltips and signed relative range, one decimal, a fixed 1 to 10 absolute scale with 5.5 at the midpoint, and relative colors in baseline-pool SDs. When the baseline is the first strength stage's `baseline_experiment`, Vanilla players are left out of both tables, because the pinned row stands for them. The Null strategist, which records the default 5 on every trait, is left out of the persona tables and the Diplomacy tab. **Stance** is absolute only, because the in-game AI sets no relationship modifiers, and it leaves out Null and Vanilla players. It holds two heatmaps. `stance_absolute` (`Stance toward rivals`) shows Public, Private, and Net stance on a fixed diverging scale centered at 0: -50 to +50 for public and private, -100 to +100 for net, red hostile and blue warm. `stance_signals_absolute` (`Relationship changes and mixed signals`) shows relationship changes (scaled by `rate`) under Activity, and masked hostility and masked goodwill as percents under Mixed signals, colored as z-scores across players. `relationship_targets` is not shown. In controlled runs with a baseline the module also writes `diplomacy_by_seed` and `diplomacy_by_seat`, built like the flavor ones on the fixed 1 to 10 scale, for the Matched Maps Diplomacy tab. Traits the extract lacks are listed in metadata as `traits_not_extracted`.

**The commitment page.** It reads the per-turn table (`is_decision`, `is_changed`, the `flavor_*` settings, and `grand_strategy`) and the panel's `persona_changes`, and the relative baseline defaults to the completed-experiment average. The page has three views. **Relative** and **Absolute** hold `commitment_relative` and `commitment_absolute` (`Relative commitment`, `Commitment`), HTML heatmaps without column groups. The columns are Acts % (share of turns with a decision), Revises % (share of decisions that changed at least one flavor), Touched (flavors changed per revision), Step (the average size of one flavor change, in points on the 0 to 100 scale), Net % (the distance from each flavor's first to its last setting as a share of all flavor movement, so 100% means every change pushed the same way), and Persona (real persona changes, scaled by `rate`). A revision is a turn whose flavor settings differ from the previous turn's. The Touched and Step tooltips add the total change per revision (`flavor_shift`). Each column has its own decimals and tooltip unit (`%` in the absolute view, `pts` for percentage-point differences in the relative view). Absolute colors are z-scores across strategists; relative colors are baseline-pool SDs. The in-game AI and the Null strategist have no strategist, so they are left out of both tables; the pinned baseline row is the pool's absolute mean. **Grand strategy** holds `grand_strategy`, laid out like the Matched Maps Focus tab: Main (the grand strategy with the largest average share of turns, written like `Spaceship 62%`), one share column per grand strategy, Switches (per 100 recorded turns), and Pivot (the first turn off the opening strategy, among players who change it). Main and the share columns take the victory color of their strategy (`GRAND_STRATEGY_INFO`: Conquest as Domination, Culture, United Nations as Diplomatic, Spaceship as Science) at share intensity; Switches and Pivot are uncolored. The Main tooltip adds the strategy, the time each player spent in its own most-used strategy, switches, and the first pivot. Vanilla rows come only from the strength stage's `baseline_experiment` (the in-game AI in its own games) when one is configured, and Null stays. In controlled runs with a baseline the module offers two Matched Maps tabs: Commitment (`commitment_by_seed`, `commitment_by_seat`, built like the flavor ones with z-score colors) and Grand strategy (`grand_strategy_by_seed`, one Main cell per seat under the Focus tab's seat headings, and `grand_strategy_by_seat`, the view's columns for one seat).

**The policies page.** Columns are the extracted policy branches grouped by the tier in which they open (`POLICY_INFO`, from Vox Populi's `PolicyTreeChanges.sql`): Ancient (Tradition, Authority, Progress), Medieval (Fealty, Statecraft, Artistry; the Medieval era after 6 policies), Industrial (Industry, Imperialism, Rationalism; the Industrial era after 12 policies), and Ideology (Freedom, Autocracy, Order), with short headers such as Trad. The relative baseline defaults to the matched in-game AI, pinned as the top row, and Vanilla players leave the tables when it is. Each view has two heatmaps. `adoption_relative` / `adoption_absolute` (`Branch adoption`) show the percent of players who adopted the branch, on a fixed 0 to 100% scale in the absolute view and in percentage points in the relative view; the absolute tooltip adds the average first-adoption turn and the share of players whose first pick in that tier was this branch (for Ancient, the opening branch; for Ideology, the first ideology). `adoption_turn_relative` / `adoption_turn_absolute` (`First-adoption turn`) show the first-adoption turn among adopters, as z-scores within each branch in the absolute view (red earlier, blue later) and as turns earlier or later than the baseline in the relative view; the absolute tooltip adds the share of players who adopted it. In controlled runs with a baseline the module writes `policies_by_seed` and `policies_by_seat` for the Matched Maps Policies tab: adoption percents on the fixed scale, with the difference from the baseline, the first-adoption turn, and the first-pick share in the tooltip. `policy_changes` is not shown.

### 6.3 Module friendly names and descriptions

Every implemented analysis module ships a coded friendly name and one-line
description; the report renders the friendly name as the section heading and the
description under it (§7). The module instance resolves its identity from its
parameters before execution. In particular, the fitted rating modules use their
grouped identity when `group_by` includes an extra dimension such as `strategy`.
The resolved name and description are persisted in the result manifest, so the
report can render them without importing the analysis code. Both are overridable
per stage via the optional `name`/`description` envelope keys (§6.1).

| registry module | friendly name | description |
| --- | --- | --- |
| `ratings.bradley_terry` | Pairwise skill ratings | Estimates each player type's relative skill from pairwise comparisons of model-adjusted strength within each game (Bradley-Terry Elo ratings). With `group_by: ["player_type", "strategy"]`, the identity is **Pairwise strategy ratings**: estimates relative skill for each player type and strategy combination from the same pairwise comparisons. |
| `ratings.plackett_luce` | Rank-based skill ratings | Estimates each player type's relative skill from the full within-game ranking of model-adjusted strength (Plackett-Luce ratings). With `group_by: ["player_type", "strategy"]`, the identity is **Rank-based strategy ratings**: estimates relative skill for each player type and strategy combination from the same full rankings. |
| `ratings.matchups` | Adjusted-strength matchups | Compares every pair of player types using model-adjusted strength, including mean differences and win rates. |
| `ratings.outcome_matchups` | Victory matchups | Compares every pair of player types using actual wins and final-score margins from completed games. |
| `prediction.evaluate` | Prediction quality | Measures how well each estimator identifies likely winners and matches observed outcomes (discrimination and calibration). |
| `prediction.compare` | Estimator agreement | Shows how closely estimators agree on win probabilities and on the within-turn ranking of players. |
| `calibration.reliability` | Prediction reliability | Checks whether predicted win probabilities match observed win rates (reliability curves and expected calibration error). |
| `calibration.loss_by_progress` | Prediction error over time | Tracks win-probability error from the opening turns through the end of the game (Brier score and log loss by game progress). |
| `calibration.civ_effects` | Civilization strength effects | Estimates how much civilization choice shifts player strength in uncontrolled games (ordinary least squares). |
| `calibration.cell_baseline` | Starting-position baselines | Shows baseline AI strength for each map seed and starting position in the controlled experiment. |
| `performance.experiment_completeness` | Experiment coverage | Reports completed, missing, and repeated games across the planned map, seat, and condition combinations. |
| `performance.score_ratio` | Final-score effects | Estimates how player identity, civilization, and other factors affect final score share (regression analysis). |
| `performance.strength_panel` | Gameplay strength | Summarizes model-adjusted strength, uncertainty, and experiment coverage for each player identity (bootstrap confidence intervals). |
| `performance.turn_predicted` | Win-probability trends | Shows how each player identity's predicted chance of winning changes from the opening turns through the end of the game. |
| `performance.controlled_seed_report` | Matched Maps | Aggregates games on shared maps by seed and final seat into the tables behind the report's Matched Maps chapter (§7.1). |
| `performance.game_log` | Game Log | Lists completed games and player seats for filtering, provenance, and optional replay links. |
| `performance.usage_efficiency` | Usage, cost, and skill | Compares cost and token use per player per game with skill, and measures Elo above or below the fitted usage-skill curve. |
| `behavior.flavors` | Strategic settings | Shows how each strategist sets the in-game AI's flavors (0 to 100, 50 is balanced), against the average of completed experiments on the same map and seat and in absolute terms. |
| `behavior.diplomacy` | Diplomatic behavior | Describes diplomatic persona traits and the public and private stances strategists set toward rivals, including how often the two conflict. |
| `behavior.commitment` | Strategic commitment | Shows how often strategists act and revise their settings, how large and how lasting their changes are, and which grand strategy they hold. |
| `behavior.policies` | Policy paths | Shows which policy branches and ideologies each player type adopts, which it picks first in each tier, and how early, against the in-game AI on the same map and seat. |

Registry-reserved modules (`enabled:false` placeholders, §6.2) have no coded
identity; they cannot run, so they never appear on a report.

---

## 7. `report`: rendering

```jsonc
"report": {
  "out_dir": "reports/",                 // authored under the base output root (§2.1); run writes <root><suffix>/<name>/
  "formats": ["md", "html"],             // md + html implemented; pdf is schema-reserved (errors at render time);
                                         //   omitted ⇒ ["md", "html"]
  "sections": null,                      // null = every enabled analysis (canonical family order, members in config order);
                                         //   or priority stage ids, followed by all remaining enabled analyses
  "overview_sections": ["bt_main", "matchup_winrates", "pred_metrics", "cal_reliability", "perf_strength", "perf_experiment_completeness", "perf_usage_efficiency"],
                                         // null = a summary card for every resolved section; a list selects compact overview cards
  "section_overrides": {                 // optional inline-artifact selection by analysis stage id
    "bt_main": {"tables": ["ratings"], "figures": ["ratings"]}
  },
  "title": null,                         // null = derive from `friendly_name`, else `name`
  "footer": null,                        // optional markdown footer; null uses the default, empty hides it
  "benchmark_citation": null,             // optional object with required `title` and `url` strings
  "include_disabled": false,             // never render skipped/disabled stages
  "replay": null                          // optional replay saves and viewer links
}
```

`replay` is omitted or `null` by default. When enabled, it requires an
enabled `performance.game_log` stage and copies matching `.Civ5Save` files
into the report when HTML is rendered. `saves` is `"all"` by default; set it
to `"controlled"` to include only games with a controlled seed. `viewer_url`
must be an absolute HTTP(S) URL for the hosted Vox Deorum replayer. Set
`base_url` to the absolute published report directory when links must work
without JavaScript. `latest_game` controls the latest-game card on the
overview page.

```jsonc
"replay": {
  "enabled": true,
  "viewer_url": "https://vox-deorum.github.io/vox-deorum-replay/",
  "saves": "all",       // "all" or "controlled"
  "base_url": null,      // optional absolute URL of the published report dir
  "latest_game": true
}
```

`footer` is an optional Markdown string appended to the bottom of `report.md` and every generated HTML page. When omitted or `null`, it defaults to `Generated by [CivBench](https://github.com/vox-deorum/civ-bench) ([Chen, 2026](https://arxiv.org/abs/2604.07733))`. An empty or whitespace-only string hides the footer. HTML pages render CommonMark links, emphasis, lists, and paragraphs, with raw HTML escaped. The Markdown report preserves the configured Markdown.

`benchmark_citation` is optional. When omitted or `null`, the report includes only the fixed CivBench paper citation. When configured, it must contain exactly two non-blank string fields, `title` and `url`. The report adds an `@misc` BibTeX citation for the benchmark results, using `title` and `url`, alongside the paper citation in `report.md` and `index.html`, before the configurable footer. Citations appear only in the main overview and Markdown report.

The report walks each produced `AnalysisResult` (tables + figures + summary) and renders one section per analysis. With `sections: null` or `sections: []`, every enabled analysis appears in canonical family order (ratings / prediction / calibration / performance / behavior / exploratory), with members in config order. An ordered `id` list puts those sections first, then automatically appends every remaining enabled analysis in the default order. Duplicate ids appear only once; unknown ids stop rendering. Automatic fill excludes disabled analyses, while explicitly listed disabled ids follow `include_disabled`. Chapters follow their first section's position in the resolved list, including Matched Maps. For example, `"sections": ["controlled_seed"]` puts Matched Maps first and includes all other enabled analyses. Disable an analysis to exclude it from the default report. Each family page opens with a sentence explaining what its analyses help readers assess; result summaries appear with the individual analyses.

Each section is headed by the module instance's **resolved friendly name** (§6.3), overridden by that stage's optional `name` when given. A question-mark tooltip after the heading contains its description, registry module, and metadata. Tooltips open on hover, keyboard focus, or click/tap, and Escape dismisses them. The stage `id` supplies its stable anchor for report links and curation (TOC, overview, sidebar). The page title is `report.title`, else the config's `friendly_name`, else its `name`. The config's `description` renders under the title on the main overview and `report.md`; other HTML pages carry it in the page heading's tooltip. Markdown sections collect technical details in a collapsible disclosure. Result summaries keep key numbers bold and use VPAI for the baseline, with its configured experiment identifier in the technical details. Headings fall back to stage ids when no friendly identity is configured.

`overview_sections` controls the compact cards in `index.html`. Set it to `null` to include a card for every resolved report section, or provide an ordered list of stage ids. The tracked templates use the compact seven-section list shown above. Every card shows the analysis's one-sentence result summary, and the same sentence appears in its detailed section. A legacy or custom analysis without a summary receives an explicit fallback sentence in both views. `section_overrides` selects the inline `tables` and `figures` for a stage. Each dimension is optional and inherits the analysis default when omitted. The selected names replace that dimension's inline list. Unknown stage ids stop rendering; requested artifact names that were not emitted produce a warning and are skipped. Hidden artifacts remain downloadable supporting files.

**Section views.** An analysis can offer alternative views of its results through `metadata.views`, an ordered mapping of view name to its `label`, optional `tip`, `tables`, and `figures`. The first view is the default. Labels are short (`Relative` and `Absolute` for the behavior views) and the optional `tip` carries the long wording (for example `Relative to completed-experiment average`), shown as the tab button's tooltip. The HTML page shows one view at a time behind a toggle, and one click switches every section on the page that offers a view of the same name, so all behavior sections flip between relative and absolute together. A `?view=<name>` query in a page URL picks the initial view wherever a group offers it. Without JavaScript every view stays visible under its label; the Markdown report renders each view under a bold label.

**HTML heatmap tables.** An analysis asks for a table to render as a colored heatmap through `metadata.heatmaps`, a mapping from table name to a layout spec: the row and column keys, their order, pinned reference rows, an optional `baseline_rows` list (rows holding the baseline's absolute values, shown unsigned and with an SD line in a signed table's tooltip), short column labels with header tooltips, an optional group column for spanning headers, the number format, and a legend. The spec may also set `complete_grid` (`row_order` and `column_order` are the full grid: listed rows and columns render even without data, as blank cells), `divider_columns` (columns followed by a thick border), `tip_rows` (extra tooltip lines from other columns, each with `column`, `label`, `signed`, `decimals`, and a `unit`, per-column `units`, or `text: true` for a string column), per-column `column_decimals`, `column_value_labels`, and `column_units` (a unit after the tooltip's numbers, such as `%`; `unit` sets one for the whole table and `baseline_units` the units of baseline rows, which stay absolute in a signed table), `category_column` with `category_colors` (a cell whose category names a `#rrggbb` color is shaded from white toward it by its `color_position`, the Focus style), and `range_columns` / `range_label` / `range_note` (a `[low, high]` pair of columns shown as one tooltip line in the table's decimals, signed in a signed table, followed by the note in parentheses, by default `mean min to max`); a legend entry may give a `#rrggbb` color instead of a position. A table that needs more than one number per cell overrides parts of a cell through keys naming its columns: `text_column` replaces the cell text, `background_column` replaces the background color with a `#rrggbb`, `link_column` turns the cell text into a link to that target, and `tip_column` replaces the whole tooltip. Every value cell carries its raw value as `data-value` and each column header its position as `data-col`. The table itself carries each cell's value and a `color_position` from 0 to 1 on the RdYlBu scale shared with Matched Maps (0 red, 0.5 pale yellow, 1 blue). Cells show hover and keyboard tooltips in the report's tip format: the first line is the tip's bold title (usually the row label), later plain lines are muted subtitles, and `label<TAB>value<TAB>note` lines form a grid with the value and any numbers in the note in an accent color. A behavior heatmap cell reads row label, column name, then the value (its label comes from the spec's `value_label`), the confidence interval, an SD line for a baseline row in a signed table, the optional range, and the player and game counts. The Markdown report shows the same grid as a pipe table with a key for the short column names. A heatmap table is kept whole instead of capped at the inline row limit. A spec that does not fit its table falls back to a plain table. None of `views`, `heatmaps`, `matched_maps`, or `tabs` appears in section tooltips.

**Sorting.** Every HTML report table under `.table-scroll` can be sorted by a numeric column. Clicking a numeric column heading sorts descending, then ascending, then back to the original order. A column counts as numeric when at least two cells hold numbers and no non-empty cell is text; heatmap cells sort by `data-value`, other cells by their text (percent signs, commas, a leading plus, and the unicode minus are handled). Blank cells sort last, pinned reference rows (VPAI, the baseline row) never move, and each table body sorts on its own. The Game Log keeps its own sorting (`no-auto-sort`). Sorting happens in the browser, so the generated HTML is unchanged.

**Output layout.** The run writes `<root><suffix>/<name>/` containing `report.md`, the HTML overview `index.html`, one HTML page per represented family (`ratings.html`, `prediction.html`, `calibration.html`, `performance.html`, `behavior.html`, `exploratory.html`), `assets/report.css`, and a self-contained `assets/<id>/` tree (figures + the full table CSVs the inline tables link to). Only families represented by the resolved report sections get a page. When the resolved sections carry an enabled, non-empty `performance.controlled_seed_report` analysis, its section leaves the performance family and becomes the controlled-seed chapter (§7.1): a `controlled-seed/` directory beside the family pages. When replay is enabled, the report also contains `games.html`, `assets/game-log.js`, and copied saves under `saves/<experiment>/<game_id>.Civ5Save`. Inline tables are capped (the full data is the linked CSV). Rendering is **deterministic**: dates come from saved artifacts, so `civ-bench report --config …` re-renders the same document **byte-identically** from existing artifacts.

The overview shows a compact announcement for the most recently completed rated condition, using the first overall Bradley-Terry or Plackett-Luce section in report order. A second announcement lists incomplete conditions as `Model (20/24)` when the global `min_condition_completeness` filter is configured. It disappears when all conditions are complete. Both use the coverage analysis's saved `condition_progress` table; rerun that analysis to add announcements to older artifacts. Completion dates use game timestamps in UTC, taking the earliest accepted game in each required slot and then the latest slot date. Missing dates or ratings omit the score announcement; missing coverage artifacts omit both announcements.

The score announcement highlights each model name and shows all its available condition scores from the selected ratings table, including conditions completed earlier. Each score includes its rank among models in the same condition, ordered by unrounded Elo, with tied scores sharing competition ranks (1, 1, 3). Pooled Vanilla and Null references are excluded from these ranks. Condition names use the configured pairing suffixes and base label, including stage overrides. For example: **Qwen-27B** scored **1430 Elo** (Every turn, #3) or **1490 Elo** (Per-5, #2).

### 7.1 The Matched Maps chapter

Matched Maps is a chapter of its own, parallel to the analysis families. It renders automatically whenever the resolved report sections carry an enabled, non-empty `performance.controlled_seed_report` analysis (§6.2); no separate template or config. The site sidebar lists it next to the families with one sub-entry per seed, the overview card (when the section is among `overview_sections`) links to it, and the chapter's pages carry the same sidebar as every other page. The section itself renders in `report.md` as a `Matched Maps` chapter; when `html` is among `report.formats`, its downloads list also carries a link to the chapter pages (a `report.formats` list without `html` skips the pages with a warning and omits the link). The chapter loads the section's three persisted tables, plus the Matched Maps tables of every analysis listed in the stage's `uses.analyses`, through the report build context (a containment-checked loader for the full named CSV artifacts of each selected manifest; the report stage never reads canonical tables or estimator predictions directly), so `civ-bench report` re-renders it from artifacts alone. At most one enabled `performance.controlled_seed_report` analysis is allowed per run. Chapter and player-page header tooltips contain the benchmark description and baseline provenance; chart tooltips explain aggregation, scales, and curve interpolation. The dedicated baseline appears as VPAI in tables and chart legends.

The chapter lives in its own directory beside the family pages:

```text
<report-dir>/
  controlled-seed/
    index.html                       # the chapter page: tabbed tables per controlled seed
    seed-<seed>-player-<player_id>.html # one detail page per available pair
  assets/report-common.js            # shared vanilla JS util (color spreading), no packages/network
  assets/report-help.js              # shared heading tooltips on every HTML page
  assets/controlled-seed-report.js   # the chapter's vanilla JS, no packages/network
  assets/<analysis-id>/*.csv         # the three source tables
```

- **Seed overview.** Each seed gets one tab group on the same axes: rows are
  strategist `|` condition combinations (catalog strategist order, configured
  condition order), columns are final `player_id` values, each heading pairing
  the position with its seat-bound civilization, such as `0: Rome`. The tabs
  are Strength (mean `adjusted_strength` on a fixed RdYlBu scale from 0 to 1,
  red at 0, yellow at 0.5, blue at 1, the rounded value in the cell, led by
  an `Avg` column that pools each condition row's runs as the run-weighted
  mean over the seed's populated seats, not a link), Focus (the dominant
  victory focus, name and percentage, a stable categorical color per strategy
  with intensity by share), then one tab per `uses.analyses` entry (§6.2),
  Strategy for `behavior.flavors`, Diplomacy for `behavior.diplomacy`, Commitment
  and Grand strategy for `behavior.commitment`, and Policies for
  `behavior.policies`. Tab labels are short and the long wording
  rides on the tab button's tooltip; choosing a tab switches every seed on
  the page. All chapter tables render through the shared heatmap renderer
  (§7), so they carry the same markup, the pinned VPAI body, tooltips, and
  sorting as every other report heatmap. The Strength and Focus captions are
  `Adjusted strength` and `Dominant victory focus`, and both carry legends:
  0 to 1 swatches for strength, the four focus colors for focus. The
  dedicated `Vanilla | Vanilla` condition renders as a separate, visually
  isolated row before the strategist rows, and the renderer completes the
  global row and column grid, leaving unobserved combinations blank.
  Strength and Focus cells carry the same tooltip: the row title
  (`Strategist | Condition`, or `VPAI` for the self-play row), the seat line
  `P<player_id> · <civilization>`, then grid lines for the mean adjusted
  strength (three decimals), the dominant victory focus with its share, and
  the run count. The `Avg` cell's tooltip shows the row title, the words
  `Seed average`, then the pooled strength and run lines. Clicking a Strength
  cell opens the matching `(seed, player_id)` detail page with that strategist
  and condition preselected (encoded in the query string); a Focus cell's
  link adds `view=focus` so the detail page opens on its Focus tab. The
  Strategy tab shows that seed's `flavors_by_seed` (§6.2): rows are
  `Strategist | Condition`, flavor columns are grouped as on the behavior
  page, the seed's baseline average is pinned on top, values are colored on
  the fixed 0 to 100 scale, and each tooltip adds a `Vs. baseline` difference
  line. The Diplomacy tab shows `diplomacy_by_seed` the same way, with
  persona trait columns on the fixed 1 to 10 scale and the matched in-game
  AI pinned on top. The Commitment and Policies tabs follow the same pattern
  (§6.2). The Grand strategy tab copies the Focus layout: one column per
  seat, each cell the main grand strategy and its share in its victory
  color. An analysis declares its tabs in `metadata.matched_maps`, one entry
  (`label`, `tip`, `seed_table`, `seat_table`) or a list of them; an entry's
  optional `key` names the tab `<stage id>-<key>`. A seed table's spec with
  `seat_columns: true` gets its columns, headings, and full grid from the
  chapter's seats, as the Focus tab does. If a listed analysis is missing from the report, is empty, or declares
  no tab, the chapter skips that tab with a report warning.
- **Seed-player detail page.** The header shows the seed, player ID, matched
  civilization, and total source runs, with previous/next player links and a
  return link to the seed overview. Multiple civilizations on one pair produce
  a visible comparability warning. The victory-probability chart draws each
  run's interpolated curve averaged on the fixed 101-point grid (§6.2): one
  checkbox per strategist, all checked by default, toggles that strategist's
  conditions; the matched Vanilla curve stays visible as a thicker reference
  line; the vertical axis fits the visible curves; and hovering the chart
  snaps to the nearest grid progress and lists every checked condition's
  probability at that point. The chart uses the shared Plotly renderer and a
  local JavaScript bundle, with no network access required. Strategists that share a catalog
  color (typically one model family) are spread through the shared
  `civBench.distinguishColors` util in `assets/report-common.js` so their
  curves stay distinguishable. Below the chart, the **Comparison** section
  offers the same tabs as the overview, sliced to this seat. The Strength
  tab keeps one row per strategist-condition combination plus the pinned
  VPAI row and shows `Runs` (linked to the Game Log when that analysis is
  present), `Win prob` (mean weighted victory probability), and
  `Adj strength` (colored on the overview's 0 to 1 RdYlBu scale); the Focus
  tab shows `Focus`, `Dom %`, `Cul %`, `Dip %`, and `Sci %`, each share in
  its strategy color at share intensity; then one tab per `uses.analyses`
  entry shows that entry's seat-keyed table (`flavors_by_seat` for
  `behavior.flavors`, `diplomacy_by_seat` for `behavior.diplomacy`). A tab with no rows for a seed or seat says so. Missing
  baselines and missing prediction rows are visible page notes, never fatal.

### 7.2 Game Log and replay links

When `report.replay.enabled` is true, the report includes a `games.html` Game
Log page. It lists every game from `performance.game_log`, with filters for
strategist, condition, seed, seat, victory, and controlled games. Matched Maps
links to relevant Game Log rows, and the overview can show a latest-game card
when `latest_game` is true. The Game Log is linked from those pages and is not
added to the sidebar chapters.

Each game with a matching save gets one replay link. The link resolves the
relative save path under `saves/<experiment>/<game_id>.Civ5Save` against the
report location, then opens the configured viewer with `file`, `player<i>`,
and `winner` query parameters. Only strategist-controlled seats receive
`player<i>` labels; VPAI seats retain the viewer's civilization names. A
missing save is shown as `no replay` and does not fail report generation.

The viewer loads saves with an HTTP request. Serve the report from an HTTP(S)
host with CORS enabled for the viewer origin, such as GitHub Pages. A
`file://` report cannot launch the viewer, but its save link remains available
for manual drag and drop. `base_url` emits absolute viewer links for static
publishing without relying on JavaScript.

**The manifest.** Each analysis persists a `result.json` beside its artifacts (`<root>/analyses/<id>/result.json`: id, module, `module_name`/`module_description`, summary, metadata, ordered table/figure filenames, `empty` flag). `module_name`/`module_description` carry the module instance's resolved friendly identity, including the grouped BT or PL identity selected from `group_by`, so the report renders it without importing the analysis registry; the report combines them with any current per-stage `name`/`description` override. This is what the report reads, so a plain `civ-bench report` reproduces the document from disk without re-running any analysis (a manifest from before friendly names simply falls back to the stage id). An analysis that legitimately produced nothing renders as an explicit empty section (it is not mistaken for a never-run stage).

---

## 8. Validation rules (enforced on load)

1. **Required keys present**: `name`, `seed`, `data`, `analyses`, `report`. `catalogs`, `estimators`, and `adjust` are optional. `catalogs` defaults to sibling paths but is loaded lazily; a catalog must resolve to a readable file only if an enabled stage needs it.
2. **No unknown keys** at any level: typos fail loud. Fields documented as arrays of ids/names (`needs`, `uses.estimators`, `uses.tables`, `uses.analyses`, `group_by`, extract `outputs`, report `formats`, report `overview_sections`, and report override artifact lists) must be JSON arrays of strings; a bare string is an error. Text fields (`friendly_name`, `analyses[].name`, `analyses[].description`) are null or strings. `report.section_overrides` is a mapping of stage ids to objects whose only optional keys are `tables` and `figures`.
3. **Unique ids** across `estimators` + `adjust` + `analyses`; explicit `needs`/`uses` must reference existing, enabled ids. `uses.analyses` additionally rejects self references. These checks apply to disabled stages too, so optional template stages cannot contain stale references. Omitted or empty `uses.estimators` on analyses that opt in to the all-estimator default resolves to all enabled estimators. The config loader resolves and caches this graph once, and the pipeline consumes that resolved graph directly.
4. **Acyclic** after edge resolution; a cycle is an error naming the cycle.
5. **Estimator consistency**: `fit` matches exactly the one sub-block present (`train`/`pretrained`); `predict: cross_val` and `tune` are valid only with `fit: train`.
6. **Registry membership**: every analysis `module` resolves in the analysis registry; every `adjust` `module` resolves in the adjust registry (currently `strength`); every estimator `model` resolves in `catalogs.models` `prediction_models`.
7. **Adjust wiring**: each `adjust` stage must declare exactly one estimator in `uses.estimators`. A `uses.tables` name must resolve to either a `data.tables` key or an enabled `adjust` stage `id`; strength-based ratings (`ratings.bradley_terry`, `ratings.plackett_luce`, `ratings.matchups`) must reference a `strength` table (no `adjust` stage ⇒ a validation error, since there is nothing to rate).
8. **Filter resolution**: every preset name in any `filter` exists in top-level `filters`; list filters merge left-to-right; a stage `filter` may not select experiments/players/turns excluded by the resolved global `data.filter` or lower constraints like `only_llm`/`min_games`/`min_condition_completeness`; a `turn_range` must be `[min, max]` with `min <= max` (either bound nullable); `min_condition_completeness` and `max_decision_failure_pct` are each null or a number in `(0, 1]`; a stage `max_decision_failure_pct` may only be stricter than the global value.
9. **Grouping resolution**: in a `ratings.*` `group_by`, every dimension past the base (`group_by[0]`, typically `player_type`) must name a grouping defined in top-level `groupings` (§3.2); referencing an undefined grouping is an error. Each grouping's `kind` must be implemented (currently only `argmax`); an `argmax` grouping's `labels`, when present, must be positional with `columns`.
10. **Bootstrap**: a `ratings.*` `bootstrap`, when not null, requires an integer `n >= 1`; its resampling is seeded from the top-level `seed` (determinism: same config ⇒ same CIs).
11. **No missing dependencies**: there is no graceful degradation. A stage requiring an uninstalled package (torch/xgboost/optuna/R) **aborts the run** with an install hint. Run `scripts/install` first so every dependency is present.
12. **Output root** (§2.1): `output`, when present, accepts only `root` (string) and `suffix` (string); both optional, defaulting to `"reports"` and `""`. Every stage save-path resolves under `<root><suffix>/`; two runs that differ only in `suffix` must not write to the same directory.
13. **Strength controlled-design params** (§5.1): `turn_progress_min` is numeric in `[0, 1]`; `weight ∈ {turn_progress, uniform}`; `relative_to ∈ {game_leader}`; `enforce_winner` is boolean; `civ_adjust ∈ {none, ols_logit}`; `block ∈ {none, start_cell, auto}`; `post_cell_normalize ∈ {none, relative_to_leader}`; `baseline_experiment` is null or a string experiment id. The id need not be listed in `experiments.json`; the adjust stage resolves it from extracted data. A `block` other than `none` affects only rows from the controlled subset (`seed != -1 and seating_rotation != -1`); uncontrolled rows always use `civ_adjust`. Missing cells in the selected implicit pathway and missing source rows for the selected explicit pathway are fatal; non-selected-pathway gaps, per-model gaps, disconnected models, and incomplete cycles warn. Condition completeness is filtered globally (§3.1: `data.filter.min_condition_completeness`), so by the time the cell adjustment resolves, incomplete conditions are already absent from the panel.
14. **Extract invariants** (§3, §3.3): Vox Deorum can record distinct sync/map seeds, but civ-bench's controlled-design benchmark requires matched starts. Therefore, for a controlled game, a configured `configuredSyncRandSeed` must equal `configuredMapRandSeed`: a mismatch **aborts extraction** with a policy error. The `games` table stores one `seed` (the controlled value, else `-1`) and `seating_rotation` (else `-1`); `-1` is the uncontrolled sentinel (controlled seeds are `≥ 1`; `0` is Civ's "pick random" and rejected for controlled runs; and rotations are `≥ 0`). Per-player `config_slot` lives in `panel_data` and is joined by `(game_id, player_id)` where needed. A `player_type_labels` value is read as a **suffix** when it begins with `-`, else as a full **override**.
15. **Presentation** (§2.2): only `condition_pairing` and `matchup_display` are accepted. Pairing `suffixes` is null or a non-empty list of `-`-prefixed strings; `sort_condition` is `"base"`, `"best"`, or a `-`-prefixed suffix and, with an explicit list, a suffix must be a member (`"base"` and `"best"` need no membership). `matchup_display` and per-matchup `display` are `matrix|vs_reference`. An enabled derived suffix set and its sort membership are checked lazily against the experiment catalog at analysis runtime.
16. **Report identity** (§2, §6.1, §7): top-level `friendly_name` is null or a string; optional `analyses[].name`/`description` are null or strings. Neither affects the DAG, a stage's fit, or `result.json` artifacts beyond the friendly-name manifest fields; they are pure presentation overrides resolved at render time.
17. **Controlled-seed report wiring** (§6.2, §7.1): a `performance.controlled_seed_report` stage must declare exactly one estimator in `uses.estimators` and exactly one strength-table reference (an enabled strength-module adjust stage id) in `uses.tables`; at most one such stage may be enabled per run. Each `uses.analyses` entry must name an analysis whose module offers a Matched Maps tab (`MATCHED_MAPS_TAB_MODULES`, currently `behavior.flavors`, `behavior.diplomacy`, `behavior.commitment`, and `behavior.policies`); any other module is an error. At run time the module additionally requires enabled condition pairing (§2.2) and a configured strength-stage `baseline_experiment` (§5.1), and it rejects uncontrolled-only inputs, duplicate per-`(game_id, player_id)` panel/strength records, and conflicting duplicate prediction points. The controlled-seed chapter renders from its section automatically when `html` is among `report.formats` (§7.1).
