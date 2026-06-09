# `benchmark.json` — the run-spec convention

A **benchmark run-spec** (conventionally `benchmark.json`) is the **top-level config**: it declares the input data, which estimators to train or load, which analyses to run, and how to render the report. It is the only file you edit to define a new benchmark run. This document is its authoritative schema.

> **Templates vs. local configs.** The repo tracks *example* run-specs as `configs/*.template.json` (`benchmark.template.json`, `benchmark.full.template.json`, `benchmark.pretrained.template.json`). To actually run, copy one to a **local, gitignored** `configs/benchmark*.json` (e.g. `configs/benchmark.dev.json`) and point it at your data roots. This schema applies to all of them; "benchmark.json" below names the run-spec *format*, not a tracked file. See [AGENTS.md](../AGENTS.md) §"Templates vs. local configs".

`benchmark.json` is validated on load — **unknown keys and missing required fields are hard errors** (invariant 1: config over code). When you add or rename a module, update this file in the same change.

> JSON has no comments. This reference uses `jsonc` fences with `//` notes for explanation; a real `configs/benchmark*.json` must be plain JSON. See [benchmark.template.json](benchmark.template.json) for a comment-free worked example.

---

## 1. The pipeline as a DAG

A run is a **directed acyclic graph of stages** in five kinds, executed in dependency order:

```
extract ──▶ estimators ──▶ adjust ──▶ analyses ──▶ report
```

`adjust` is the bridge between raw predictions and the rating models: it turns an estimator's per-turn win-probabilities into a per-player-game **strength panel** (`adjusted_strength`) and registers it as a named table that `ratings.*` (and some `performance.*`) analyses consume (§5). It is optional — a run with no estimators or ratings simply omits it.

Each stage has a stable **`id`**. Edges come from three places, all resolved into one topological sort before anything runs:

1. **Kind ordering (implicit).** `extract` → `estimators` → `adjust` → `analyses` → `report`, always.
2. **`needs` (explicit).** A stage may list other stage `id`s it must run after. Use this to force ordering the harness can't infer (e.g. one analysis consuming another's CSV).
3. **`uses` (referential).** When a stage references an estimator `id` or a named table in its `uses` block, an edge is created automatically — you don't also have to write `needs`. A `uses.tables` name may be a canonical table from `data.tables` **or** a table produced by an `adjust` stage (§5); referencing it creates an edge to that stage.

A cycle, an unknown `id`, or a reference to a disabled stage is a validation error. Disabled stages (`"enabled": false`) are dropped from the graph, and anything that *needed* one is a validation error. There is **no graceful degradation for missing dependencies**: if a stage needs `torch` / `xgboost` / `optuna` / R and it isn't installed, the run **fails loud** — install everything first (see [AGENTS.md](../AGENTS.md#dependencies) and `scripts/install`).

The dependency graph is resolved once at config-load time and then reused by the dry-run printer and runner. Validation and execution therefore share the same interpretation of `needs`, `uses.estimators`, and `uses.tables`. For analysis modules that opt in to the all-estimator default, omitted or empty `uses.estimators` means "all enabled estimators"; listing ids is an explicit subset override.

**Determinism.** The top-level `seed` is threaded into every stage that uses randomness (CV splits, torch init, resampling, bootstrap). Same `benchmark.json` + same `runs/` ⇒ byte-stable outputs.

---

## 2. Top-level shape

```jsonc
{
  "name": "staff-standard-2026",        // required: run id; names the report + reports/ subdir
  "description": "Staff line-up, standard 8-seat map",  // optional, free text
  "seed": 42,                            // required: global RNG seed (determinism)

  "output":     { /* §2.1 */ },          // optional: run output root + variant suffix (→ reports/, reports-cross/)

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

Top-level keys: `name`, `seed`, `data`, `analyses`, `report` are **required**; `description`, `output`, `catalogs`, `filters`, `groupings`, `estimators`, and `adjust` are optional. Omit `estimators` (and `adjust`) for a ratings-free run; conversely, any `ratings.*` analysis needs an `adjust` stage to supply the `strength` table it rates.

**`catalogs` is lazy.** If omitted, each catalog-using stage resolves `paths.json`, `models.json`, and `experiments.json` from the **same directory as this run-spec file** only when it needs that catalog. Set a key only to point at a file elsewhere; unset keys still fall back to the sibling. A missing catalog is a load error only when an enabled stage needs it (for example, an estimator needs `models.json`, and orthodox `player_type` composition needs the model/experiment catalogs). Runs that do not touch a catalog do not require that sibling file to exist.

### 2.1 `output` — the run output root (and the `-cross` variant)

Every stage that writes (`estimators` `save_predictions`/`save_model`, `adjust` `save`, `report` `out_dir`) writes **under a single run output root**, resolved once and threaded into all stages. By default that root is `reports/`. `output` lets a run redirect everything to a sibling root by appending a configurable **suffix** — the mechanism behind the **`-cross` (llm/non-llm) variant**, where an estimator trained on non-LLM seats predicts everyone and its whole report lands in `reports-cross/` instead of clobbering the normal run.

```jsonc
"output": {
  "root":   "reports",   // base output directory for ALL stages (default "reports")
  "suffix": ""           // appended to root → "" ⇒ reports/ ;  "-cross" ⇒ reports-cross/
}
```

- A stage save-path written as `reports/estimators/<id>/…` is interpreted **relative to the resolved root** (`<root><suffix>/estimators/<id>/…`). Authoring paths under `reports/` keeps the default run unchanged; setting `suffix: "-cross"` moves the *same* config's outputs to `reports-cross/`.
- **Only save-paths are re-rooted — input paths are read as-authored.** `estimators.pretrained.model_dir` (§4.6), `data.tables.*`, and `data.extract.runs_dir` are inputs the run *reads*; the suffix never rewrites them. This is what lets the tracked `pretrained/<model_id>/` store feed every variant: a `-dev` run loads the same `pretrained/score/` but writes its predictions under `reports-dev/`.
- The **cross variant is otherwise an ordinary `fit: train` run** with `train.train_subset: "non_llm"` (§4.4) — there is no separate "cross" estimator kind and no special prediction-loading path. Pair the non-LLM training subset with `output.suffix: "-cross"` (typically as its own config, e.g. `benchmark.cross.json`, or a CLI suffix override) so the two variants coexist on disk.
- `output` is optional; omit it for the default `reports/` root and no suffix.

---

## 3. `data` — extract + canonical tables + global filter

`data` owns everything before the analysis modules: turning raw DBs into canonical CSVs, naming those CSVs, and the **global** row filter that every downstream stage inherits (stages may narrow further, never widen).

```jsonc
"data": {
  "extract": {
    "enabled": true,                     // false → reuse existing CSVs, never touch runs/ DBs
    "runs_dir": "runs/",                 // root searched for *.sqlite game DBs
    "outputs": ["turns", "panel", "games", "tokens"],  // which canonical CSVs to (re)build
    "max_dbs": null,                     // int → only first N discovered DBs (smoke tests)
    "prune_missing": false,              // true → only drop rows for missing DBs, no new extract
    "force_rebuild": false               // true → rebuild even if outputs exist & are newer
  },

  "tables": {                            // canonical CSV locations (extract writes / loaders read)
    "turns":      "runs/turn_data.csv",          // per-player per-turn panel (prediction features) — carries player_type, NOT seed
    "panel":      "runs/panel_data.csv",         // per-player per-game outcomes/strategies/strength (+ player_type/model/strategist/config_slot)
    "games":      "runs/game_data.csv",          // per-GAME row: game_id, timestamp, experiment, seed, seating_rotation (-1 ⇒ uncontrolled)
    "tokens":     "runs/model_token_usage.csv"
  },

  "filter": "llm_only"                   // GLOBAL selector: inline object OR a preset name (§3.1)
}
```

- **`filter` is optional** — omit it for "all rows". It accepts either an inline filter object or the **name of a preset** from top-level `filters` (§3.1). Every stage inherits this global filter and may narrow it (§6.1), never widen it.
- **`extract.enabled: false`** is the "I already have CSVs" switch — the `extract` stage is dropped from the DAG and loaders read `tables.*` directly. Combine with `--skip extract` on the CLI for the same effect ad hoc.
- The `extract` stage is **skipped automatically** when every `outputs` CSV already exists and is newer than the DBs, unless `force_rebuild: true`.
- When the selected stages need experiment ids or player-type names, they resolve through `catalogs.experiments` + `catalogs.models`. **`player_type` is composed at extract from the per-player game metadata** (`model-{id}` + `strategist-{id}`) via the catalog's template + aliases + unified label map (§3.3); the old seat→model mapping is only an optional fallback — never spell out seat→model mappings here.

### 3.1 `filters` — named, reusable filter presets

A filter is the **same shape everywhere** it appears (`data.filter` and every stage's `filter`), so define the common ones once and reference them by name instead of repeating the object. Every field is optional; an omitted field means "no constraint".

```jsonc
"filters": {
  "llm_only":     { "only_llm": true, "min_games": 5 },
  "staff_recent": { "experiments": ["2026-staff-standard"], "min_games": 5 },
  "late_game":    { "turn_range": [200, null] }
}
```

The full filter shape (every field optional; omitted ⇒ no constraint):

```jsonc
{
  "experiments":          null,  // null = all conditions; or ["2026-staff-standard", ...]
  "exclude_experiments":  [],    // subtracted from the above (or from "all")
  "players":              null,  // null = all player types; or ["Sonnet-4.5-Briefed", ...]
  "only_llm":            false,  // true → keep only LLM seats (drop Vanilla/Null, per experiments.json)
  "min_games":           1,      // drop player types with fewer games than this
  "turn_range":          null    // null = all turns; or [min_turn, max_turn] (absolute turn numbers;
                                 //   either bound may be null, e.g. [200, null] = turn 200 onward)
}
```

**Wherever a filter is accepted, the value may be:**

- an **inline object** (the shape above),
- a **string** naming a preset in `filters`, or
- a **list** mixing preset names and inline objects, which are merged left-to-right (later entries win per field). E.g. `"filter": ["llm_only", { "turn_range": [200, null] }]`.

A stage's `filter` is then **intersected** with the resolved global `data.filter` — a stage can only narrow, never widen (§6.1). Referencing an undefined preset name is a validation error.

The same resolved filter object is what the shared data loaders accept, so analysis modules should pass config-shaped filters through the loader helpers instead of translating `experiments`, `players`, `only_llm`, `min_games`, and `turn_range` by hand.

### 3.2 `groupings` — named rating-identity dimensions

A **grouping** derives one categorical dimension from the strength panel that `ratings.*` analyses can fold into the rated identity (§6.2, `group_by`). Like `filters`, groupings are defined once by name and referenced by name — config over code, so adding a way to slice the field is a config edit, not a new module. `groupings` is **optional**; omit it for plain per-`player_type` ratings.

```jsonc
"groupings": {
  "strategy": {                          // composite identity {player_type}-{dominant strategy}
    "kind": "argmax",                    // dimension value = column with the largest value per row
    "columns": ["domination_ratio","culture_ratio","diplomatic_ratio","science_ratio"],
    "labels":  ["Domination","Culture","Diplomatic","Science"]   // optional; positional with columns
  }
}
```

- **`kind: "argmax"`** (the only kind implemented) assigns each player-game the label of whichever `columns` value is largest — exactly the dominant-strategy rule the old `strategy_ratings.py` uses. `labels` is optional; when given it must be positional with `columns` (else the raw column name is used).
- A grouping referenced by `group_by` but **absent** from this catalog is a validation error (§8).

> **Reserved / deferred — do not rely on yet.** Other `kind`s are planned but **not implemented**. In particular `kind: "bucket"` (e.g. `{ "kind": "bucket", "column": "turn_progress", "edges": [0, 0.33, 0.66, 1.0], "labels": ["early","mid","late"] }`) would enable per-game-stage ratings, but that additionally requires the `adjust`/`strength` stage to emit *per-stage* strength, which is its own follow-up. Treat anything beyond `argmax` as designed-but-unbuilt.

### 3.3 Player identity — the orthodox `player_type`

`player_type` (the identity every rating is fit over) is **composed at extract time from the per-player game metadata**, not from a hand-maintained seat map. Each seat records `model-{player_id}` (e.g. `Sonnet-4.5`, or `VPAI` for vanilla) and `strategist-{player_id}` (e.g. `simple-strategist-briefed`) in the game's `GameMetadata`; because the identity travels with the player, it stays correct even when controlled seating rotates a model through different seats. The catalog (`catalogs.models` / `catalogs.experiments`) supplies three knobs:

- **`player_type_template`** — the format string, e.g. `"{model}-{variant}{suffix}"`. `model` is alias-normalized and `strategist`→`variant` is mapped via the catalog; `model == "VPAI"` resolves to the vanilla baseline label by strategist (`null-strategist`→`Null`, default→`Vanilla`).
- **`player_type_labels`** — a single unified map keyed by condition and, optionally, `(condition, slot)` (slot wins; `slot` is the player's `config_slot`, the pre-rotation configured seat). The looked-up value is polymorphic: **if it starts with `"-"` it is a SUFFIX** appended to the composed type (e.g. distinguishing two tweaks of the same model+strategist, `…-Briefed-A` vs `…-Briefed-B`); **otherwise it is a full player_type OVERRIDE** that replaces the composed value. This one map therefore serves both the tweak-suffix and the legacy explicit-label needs. Suffixes are skipped for VPAI/Vanilla so the baseline pools across conditions; full overrides still apply.
- **optional fallback** — when a game predates the metadata keys, the old static `(condition, slot)` → player_type map is consulted; otherwise `"Player {id}"`.

The same composition feeds both `panel_data` and `model_token_usage.csv` (single source of truth), replacing the old load-time `(condition, player_id)` merge.

---

## 4. `estimators` — prediction-model producers

Estimators are the pipeline's reason for having stages at all: a `performance.turn_predicted` or `calibration.reliability` analysis needs `predicted_win_probability`, which only exists once a predictor has run. Declare each predictor once in `estimators`; analyses reference it by `id`.

An estimator answers two **independent** questions, and that separation is the whole design:

1. **Where does the model come from?** — `fit`: `"train"` (fit on this run's data, optionally tuned first) or `"pretrained"` (load a saved model dir).
2. **How are the predictions it hands downstream generated?** — `predict`: `"in_sample"` (one model predicts the rows) or `"cross_val"` (k-fold; honest *out-of-fold* predictions).

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

**`fit` — where the weights come from:**

| `fit`        | What runs                                          | Use when |
|--------------|----------------------------------------------------|----------|
| `train`      | (optional tune →) fit on the current data          | You want a fresh model fit to this run's data. |
| `pretrained` | load a saved model dir, no training                | You have a model trained on a reference dataset and want to apply it here. |

**`predict` — how downstream predictions are generated (only meaningful with `fit: train`):**

| `predict`    | What it emits                                                | Use when |
|--------------|--------------------------------------------------------------|----------|
| `in_sample`  | one model fit on the train subset, predicting `predict_subset` | You want a single deployed model + its predictions. |
| `cross_val`  | k-fold out-of-fold predictions (K models, held-out per fold) | You want **honest** predictions to evaluate/calibrate on. |

`fit: pretrained` always predicts in-sample (you can't cross-validate weights you didn't train here), so `predict` is ignored for it.

This pair is the **train-before-others vs. use-a-pre-trained-estimator** switch the pipeline is built around. A `pretrained` estimator still runs *inference* on the current `tables.turns`, so it depends on `extract` (or on the CSVs already existing) but not on training. Either way, downstream analyses reference the estimator `id` and get its `predictions.csv`.

### 4.3 `tune` — optional Optuna pre-step (gates `train`)

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

- Tuning optimizes a **single scalar** `objective` (an objective must be scalar). Reporting *many* metrics is a different concern — that's `prediction.evaluate` (§6.2), which takes a `metrics` list.
- `load_params` set ⇒ tuning is skipped; the saved `best_params.json` is loaded as the hyperparameters. This is the cheap path for "reuse the tuning we already did" — a *pre-trained hyperparameter set*.
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

- `train_subset` controls which rows the model is fit on (e.g. `"non_llm"` trains only on Vanilla/Null seats, then predicts on everyone — the held-out generalization setup). For `predict: in_sample`, the inference subset is the entry's top-level `predict_subset`; for `cross_val`, predictions are out-of-fold over the `train_subset` and `predict_subset` is ignored.
- `save_model` applies to `predict: in_sample` (a single fitted model). Saved models written here are exactly what a later run's `pretrained.model_dir` can point at. For `cross_val` there is no single model to deploy, so `save_model` is ignored and the artifact is the OOF `predictions.csv` (+ feature importance).

### 4.5 `features` — selection (omit to use the estimator's own defaults)

`features` is **optional**. When omitted, each estimator uses its own coded `DEFAULT_FEATURES` (or the feature set baked into a tuned `best_params.json`). Provide it only to override per-estimator:

```jsonc
"features": {
  "include": null,                       // null = the estimator's default set; or ["science_*", ...]
  "exclude": ["civ_*"]                   // wildcards allowed; applied after include
}
```

Ignored for `fit: pretrained` — the saved `metadata.json` already carries the selected features.

### 4.6 `pretrained` (required when `fit == "pretrained"`)

```jsonc
"pretrained": {
  "model_dir": "pretrained/attention_mlp/"  // dir with metadata.json (dispatches class)
}
```

No `params` / `features` / `tune` are consulted — the saved `metadata.json` carries the architecture and selected features; the class is resolved from the registry by `metadata.model_class`. Inference runs on the entry's `predict_subset` and writes `save_predictions`.

**`model_dir` is an INPUT, read as-authored — it is NOT re-rooted by `output.suffix`** (only save-paths are; §2.1). The repo ships tracked reference snapshots under `pretrained/<model_id>/` (one per `prediction_models` id); pointing `model_dir` there means the same store serves the default run *and* the `-dev`/`-cross` variants (a `suffix: "-dev"` run loads `pretrained/score/` yet writes to `reports-dev/estimators/score/`).

### 4.7 Why evaluation is a *step*, not a *source*

Earlier drafts had an `evaluate` source alongside `train`/`pretrained`. It was removed because it conflated two separable things:

- **Generating predictions** — in-sample vs. out-of-fold vs. loaded-and-inferred. This genuinely is a property of the estimator-*producer*: it changes which `predictions.csv` comes out. It now lives on the `predict` axis (`in_sample` | `cross_val`).
- **Scoring predictions** — computing ROC-AUC / Brier / log-loss / balanced-accuracy from a `predictions.csv`. That reads an artifact and reports numbers; it owns no model. It is, and always was, an **analysis**: `prediction.evaluate` / `prediction.compare` (§6.2).

So **"evaluate an estimator" = point a `prediction.evaluate` analysis at it.** To evaluate honestly, give the estimator `predict: cross_val` so the analysis scores held-out predictions; to inspect a deployed or pre-trained model's behavior, leave it `in_sample`/`pretrained`. The metrics step is shared, multi-metric, and works identically across all estimators — which is exactly what you'd lose by burying scoring inside each producer.

---

## 5. `adjust` — derived tables (the strength panel)

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
    "post_cell_normalize": "none"        // optional final re-normalization after the cell effect: "none" | "relative_to_leader"
  }
}
```

- The stage `id` doubles as the **table name** it registers — exactly as an estimator `id` names its `predictions.csv`. A consumer writes `uses.tables: ["strength"]` (not the stage's literal output path), and the harness adds the edge.
- `uses.estimators` is **required and single-source** in practice: strength is defined relative to one predictor's win-probabilities. Point it at a `cross_val` estimator for out-of-fold-honest strength, or an `in_sample`/`pretrained` one to mirror a deployed model.
- The emitted table is per-player-game with at least `game_id, player_id, player_type, civilization, relative_strength, logit_strength, adjusted_strength` — the exact columns `ratings.*` require (§6.2), plus audit columns — plus `seed`/`seating_rotation` joined from `games` and `config_slot` carried from `panel_data` for the controlled-design diagnostics. Because the estimator's `predictions.csv` does **not** carry the composed identity, `player_type` (and `config_slot`) are joined from `panel_data` by `(game_id, player_id)`; `civilization` comes from the predictions (panel as fallback). `enforce_winner:true` forces the actual winner to the top strength (under `relative_to:"game_leader"` that is `relative_strength = 1.0`; under `relative_to` unset/`"none"` the winner is bumped to the top raw strength and `relative_strength` mirrors the raw value), but `logit_strength` is always finite via clipping before any logit-scale adjustment.
- **The panel retains every experiment** — it is *not* filtered to LLM seats. Vanilla/Null seats stay in so `ratings.* ref="Vanilla"` and the baseline pathways (§5.1) resolve; LLM-only narrowing is each analysis's `only_llm` filter, not baked into the panel.
- `civ_adjust: "none"` skips the OLS step (then `adjusted_strength == relative_strength`); any other value selects an adjustment scheme (currently `ols_logit`). The `ols_logit` fit is `logit_strength ~ C(civilization, Sum) + C(player_type, Treatment(reference=<vanilla_label>))` (reference from the catalog's vanilla label), fit over **all** panel rows; only uncontrolled rows have the civ effect subtracted.
- **Controlled-design adjustment (`block`).** When a game carries controlled seeds **and** seating (`seed != -1` and `seating_rotation != -1` in the `games` table), `block` replaces `civ_adjust` on those rows with a **matched final-seat cell** correction: it subtracts the per-`(seed, player_id)` **Vanilla VPAI baseline**. `seed` identifies the generated map/start layout, and `player_id` identifies the final in-game seat/start position after rotation. `config_slot` is retained for diagnostics about which configured model rotated into that cell, but it is not the baseline cell. "Vanilla VPAI" means rows whose raw model is VPAI when available and whose composed `player_type` is the catalog's vanilla label (`Vanilla` by default); `Null` is not baseline evidence. Because the final-seat cell fixes the seat-bound civilization, this subsumes the civ adjustment. Uncontrolled rows fall back to `civ_adjust`; `block: "none"` ⇒ pure legacy behavior except for harmless audit columns/files. The baseline is built via one or both pathways:
  - **Explicit** (`baseline_experiment` set, e.g. a pure VP self-play condition): for each `(seed, player_id)` cell, average the designated experiment's Vanilla/VPAI `logit_strength` rows. In seated mode, rotations for a seed are repeated observations of the same generated map/start cells, so each final-seat cell's explicit baseline is the logit-scale mean of those repeated Vanilla observations.
  - **Implicit** (`baseline_experiment: null`, default): for each controlled experiment and `(seed, player_id)` cell, average that experiment's own Vanilla/VPAI `logit_strength` rows. Incomplete self-coverage is expected (rotation sparsity) and is **not fatal**: a controlled seat whose own `(seed, player_id)` cell has no VPAI counterpart **falls back to the uncontrolled adjustment** (`civ_adjust` when `ols_logit`, else `relative_strength`) and is named in a WARN; every cell that does have a Vanilla observation still uses its start-cell baseline. A per-row `adjust_method` column (`cell|civ|relative`) records which path each row took.

  The pathway selected by `baseline_experiment` feeds `adjusted_strength`: `null` selects implicit; a string selects explicit. If explicit is selected, implicit is still computed per controlled experiment wherever Vanilla rows exist, so the report can show implicit-vs-explicit deltas. **A missing selected *explicit* baseline cell is fatal** (the designated `baseline_experiment` is meant to span the whole grid); a missing selected *implicit* cell warns and falls back. Per-model coverage gaps, cells with no Vanilla baseline in a comparison pathway, and player types disconnected from `Vanilla` are **warned, never fatal** (keep all games, proceed).
- **Estimator/filter precondition.** The strength stage consumes the referenced estimator's saved predictions; it does not re-infer missing rows. Controlled `block != "none"` therefore expects the estimator artifact to include the Vanilla rows needed by the selected baseline pathway. A global `data.filter` or estimator `predict_subset` that drops Vanilla references (for example `only_llm`) removes the baseline evidence: under **explicit** that is fatal at the missing cell, and under **implicit** the affected cells fall back to `civ_adjust` (which itself needs the vanilla reference level present, or its OLS fit fails). Keep estimator prediction broad and narrow later in each analysis.
- **Intermediate adjustment diagnostics — always written (no config).** Like every other audit trail, the stage *always* emits the per-group values it subtracts, next to the strength panel (in the directory of `save`, default `reports/adjust/`), so the correction can be inspected without opting in:
  - `civ_effects.csv` — per-`civilization` OLS-logit effect table (`civilization, civ_effect, n_rows`) — the **civilization-level effect** from the uncontrolled (`civ_adjust`) path.
  - `cell_baseline.csv` — the **VPAI seating×seed effect** from the controlled (`block`) path: `experiment, pathway, seed, player_id, civilization, cell_baseline, n_vanilla, n_games, n_models, has_vanilla_baseline, vanilla_connected`. It carries every pathway that actually ran (`pathway ∈ {explicit, implicit}`); a default `baseline_experiment:null` run normally contains only implicit rows. When both explicit and implicit exist for a condition/cell, the report compares `implicit - explicit` on the logit scale.
  - `cell_coverage.csv` — the **controlled-design completeness report**: for each controlled experiment, which `(seed, player_id)` cells of the **entirety** reference grid it is missing. Columns: `experiment, seed, player_id, civilization, in_entirety, n_rows, n_vanilla, has_baseline, missing` (`missing = true` ⇒ the experiment has no rows for that cell). The "entirety" is the full baseline cell set — the `baseline_experiment`'s cells when set (a pure VP self-play spans every seat across every seed/rotation), else the union of `(seed, player_id)` cells observed across the controlled subset. This is a **report-only** diagnostic (WARN, never fatal — distinct from the hard-error "a row needs adjustment but its selected baseline cell is missing"); written only when controlled rows exist.

  A mixed dataset writes these files; each is simply empty/absent when its path didn't run (`cell_baseline.csv`/`cell_coverage.csv` are empty on a fully uncontrolled run). `performance.strength_panel` always surfaces whichever exist (§6.2) — no flag.

---

## 6. `analyses` — the pluggable modules

A list of analysis stages. Every entry shares a common envelope; the `params` block is module-specific (catalog in §6.2). Order in the list is irrelevant — the DAG decides execution.

### 6.1 Common envelope

```jsonc
{
  "id": "bt_main",                       // required: unique stage id
  "module": "ratings.bradley_terry",     // required: analysis registry name
  "enabled": true,
  "needs": [],                           // optional explicit deps
  "uses": {                              // optional artifact references (create auto-edges)
    "estimators": ["attention", "score"],   // estimator ids → their predictions.csv
    "tables": ["panel", "strength"]          // canonical (data.tables) OR an adjust stage's table (§5)
  },
  "filter": "late_game",                 // optional: preset name, inline object, or list (§3.1);
                                         //   NARROWS the global filter for this stage
  "params": { /* module-specific, see §6.2 */ }
}
```

- `uses.estimators` is how estimator-consuming modules get win-probabilities. For implemented modules whose analysis class opts in to the all-estimator default (`prediction.*`, `calibration.reliability`, `calibration.loss_by_progress`, and `performance.turn_predicted`), omit it (or set an empty list) to consume every enabled estimator; provide ids only to narrow to a subset. The DAG adds edges to the resolved estimators either way.
- `uses.tables` names a canonical table (`data.tables`) or one an `adjust` stage emits (§5). The `ratings.*` family consumes the derived `strength` table this way; referencing it adds the edge to the `adjust` stage (and transitively to its estimator).
- `filter` accepts the same preset-name / inline / list forms as `data.filter` (§3.1). It is intersected with the resolved global filter; a stage can only narrow, never widen.

### 6.2 Module params catalog

Every registry name `civ-bench` will ship, with its key params. Unlisted params fall back to coded defaults; unknown params are validation errors.

#### `ratings.*` — consume the `strength` table from an `adjust` stage (§5)

Every `ratings.*` analysis rates `adjusted_strength`, so each one references an `adjust` stage's table via `uses.tables: ["strength"]` (shown once below; the same `uses` applies to all). They do **not** read `panel_data` directly, and they depend transitively on the estimator that fed the `adjust` stage.

Two cross-cutting params apply to both fitted ratings (`bradley_terry`, `plackett_luce`):

- **`group_by`** (default `["player_type"]`) — the identity the rating is fit over. Extra dimensions past the base must name a grouping in top-level `groupings` (§3.2); the rated identity is the composite formed by joining the dimension values with `-`. So **per-strategy Elo is not a separate module** — it is `group_by: ["player_type", "strategy"]` on the ordinary BT/PL fit. `min_games` then filters *composite* identities post-fit, and ref/vanilla re-centering is preserved.
- **`bootstrap`** (default omitted → point estimate only) — when set, a shared resample-and-refit helper draws games with replacement (`stratified` by experiment by default), re-runs the same fit `n` times, and emits percentile CIs + rank stability alongside the point estimate. Bootstrap CIs are **not a separate module**; the resampling is seeded from the top-level `seed`. When the `strength` table uses a controlled-design `block` adjustment, the resample **re-runs the `adjust/strength` fit inside each replicate** (not a fixed panel) so the start-cell baseline's uncertainty is reflected in the CIs.

```jsonc
// ratings.bradley_terry — BT MLE with pairwise score weights (R: BradleyTerry2)
{ "module": "ratings.bradley_terry",
  "uses": { "tables": ["strength"] },
  "params": { "group_by": ["player_type"],          // ["player_type","strategy"] → per-strategy Elo
              "weighted": true, "ref": "Vanilla", "min_games": 5, "only_llm": false,
              "bootstrap": null } }                  // or { "n": 1000, "stratified": true } for CIs

// ratings.plackett_luce — PL MLE over per-game rankings (R: PlackettLuce)
{ "module": "ratings.plackett_luce",
  "uses": { "tables": ["strength"] },
  "params": { "group_by": ["player_type"], "ref": "Vanilla", "min_games": 5, "bootstrap": null } }

// ratings.matchups — empirical head-to-head matrices + OLS validation
{ "module": "ratings.matchups",
  "uses": { "tables": ["strength"] },
  "params": { "mode": "mean", "validate_ols": true } }
```

**Optional `ratings.*` (off by default — registry-reserved, shipped only in `benchmark.full.template.json`):**

```jsonc
// ratings.ablation_bt — incrementally add each player type's games (chronologically) and track
//   Elo convergence; isolates each game's marginal contribution. Writes the ablation_bt_* artifacts.
{ "module": "ratings.ablation_bt", "enabled": false,
  "uses": { "tables": ["strength"] }, "params": { "weighted": true } }

// ratings.vanilla_slot_effect — seat/start-position confound check. With the strength stage's start-cell
//   adjustment ON it doubles as the VALIDATION that the cell effect nulled the slot effect: significant on the
//   raw panel, ~null on the adjusted panel.
{ "module": "ratings.vanilla_slot_effect", "enabled": false,
  "uses": { "tables": ["strength"] }, "params": {} }
```

#### `prediction.*` — score one or more estimators

`prediction.*` is the *scoring* family: it answers "how good is the win predictor". The implemented prediction modules opt in to scoring every enabled estimator by default; add `uses.estimators` only when a stage should compare a subset. Calibration views live in their own family (`calibration.*`, below).

```jsonc
// prediction.evaluate — metrics table across estimators (ROC-AUC/Brier/log-loss/bal-acc)
{ "module": "prediction.evaluate",
  "params": { "metrics": ["roc_auc","brier_score","log_loss","balanced_accuracy"] } }

// prediction.compare — side-by-side comparison table + ranking
{ "module": "prediction.compare",
  "params": {} }

// Explicit subset override, when needed:
{ "module": "prediction.evaluate",
  "uses": { "estimators": ["score","attention"] },
  "params": { "metrics": ["brier_score","log_loss"] } }
```

**Optional `prediction.*` (off by default — registry-reserved, shipped only in `benchmark.full.template.json`):**

```jsonc
// prediction.winner_trajectories — P(win) trajectories for eventual winners
{ "module": "prediction.winner_trajectories", "enabled": false,
  "uses": { "estimators": ["attention"] }, "params": { "sample_games": 12 } }

// prediction.elo_comparison — predicted-strength vs rating-based Elo cross-check
{ "module": "prediction.elo_comparison", "enabled": false,
  "uses": { "estimators": ["attention"] }, "needs": ["bt_main"], "params": {} }

// prediction.context_slicing — metrics sliced by experiment / player type / turn range
{ "module": "prediction.context_slicing", "enabled": false,
  "uses": { "estimators": ["attention"] }, "params": { "by": ["experiment","player_type"] } }
```

#### `calibration.*` — calibration of estimators

Two single-purpose views of how well estimator probabilities are calibrated — one across the **probability** axis, one across the **game-progress** axis. Both implemented calibration views opt in to consuming every enabled estimator by default; add `uses.estimators` only to narrow the comparison.

```jsonc
// calibration.reliability — reliability diagram: observed win-rate vs predicted P(win) per bin
{ "module": "calibration.reliability",
  "params": { "n_bins": 10 } }

// calibration.loss_by_progress — Brier/log-loss across turn_progress (game-stage) bins
{ "module": "calibration.loss_by_progress",
  "params": { "n_bins": 20, "metrics": ["brier_score","log_loss"] } }
```

#### `performance.*` — strength + score, some need the `strength` table

```jsonc
// performance.score_ratio — per-player score-ratio regressions (consumes panel)
{ "module": "performance.score_ratio",
  "params": { "target": "score_ratio", "predictors": ["player_type","civilization"] } }

// performance.strength_panel — summarizes the adjust stage's strength table by player type.
//   It CONSUMES the `strength` table (does not derive it — that's the adjust stage, §5).
//   It always reports per-identity n_games + bootstrap CI AND flags identities below
//   `min_games_preliminary` as preliminary (the controlled design exists to estimate strength from
//   very few games, so the small-sample basis is surfaced, not hidden).
//   It also surfaces the adjust stage's always-written adjustment diagnostics — the civilization-level
//   effect table and the VPAI seating×seed effect (cell baseline, explicit/implicit pathways,
//   §5.1) — whenever they exist. No flag on the diagnostics.
{ "module": "performance.strength_panel",
  "uses": { "tables": ["strength"] },
  "params": { "metric": "adjusted_strength", "by": "player_type",
              "min_games_preliminary": 5 } }   // < this ⇒ flagged preliminary (defaults to ratings min_games)

// performance.turn_predicted — P(win) / strength trajectory over the game, per estimator
{ "module": "performance.turn_predicted",
  "params": { "aggregate": "mean", "by": "player_type" } }
```

**Optional `performance.*` (off by default — registry-reserved, shipped only in `benchmark.full.template.json`):**

```jsonc
// performance.permutation_importance — grouped permutation importance over feature families
{ "module": "performance.permutation_importance", "enabled": false,
  "uses": { "estimators": ["attention"] },
  "params": { "n_repeats": 20, "groups": "feature_families" } }
```

#### `exploratory.*` — dataset descriptives

```jsonc
// model_token_costs uses tokens table + pricing from models.json (cost-efficiency is a benchmark axis)
{ "module": "exploratory.model_token_costs","uses": { "tables": ["tokens"] },
  "params": { "currency": "usd" } }
```

**Optional `exploratory.*` (off by default — registry-reserved, shipped only in `benchmark.full.template.json`):**

```jsonc
{ "module": "exploratory.panel", "enabled": false, "params": {} }
{ "module": "exploratory.turn",  "enabled": false, "params": {} }
// strategy_profiles joins strategy mix against adjusted_strength → needs the strength table
{ "module": "exploratory.strategy_profiles", "enabled": false,
  "uses": { "tables": ["strength"] }, "params": { "by": "player_type" } }
```

> **`behavior.*` is deferred.** The whole behavioral family (flavor-change clusters/decomposition, pivot/nuke rationale, victory commitment) scores no strategist, so it is **not in this schema**. We will revisit how to bring behavioral profiling back as an opt-in extension later.

---

## 7. `report` — rendering

```jsonc
"report": {
  "template": "default",                 // template name under bench/reports/
  "out_dir": "reports/",                 // resolved under the run output root (§2.1); run writes <root><suffix>/<name>/
  "formats": ["md", "html"],             // any of: md, html, pdf
  "sections": null,                      // null = every produced AnalysisResult, in DAG order;
                                         //   or an explicit ordered list of stage ids to include
  "title": null,                         // null = derive from `name`
  "include_disabled": false              // never render skipped/disabled stages
}
```

The report walks each produced `AnalysisResult` (tables + figures + summary). With `sections: null`, every enabled analysis appears in dependency order; pass an ordered `id` list to curate and reorder. No analysis hardcodes its place in the document (invariant 3).

---

## 8. Validation rules (enforced on load)

1. **Required keys present**: `name`, `seed`, `data`, `analyses`, `report`. `catalogs`, `estimators`, and `adjust` are optional. `catalogs` defaults to sibling paths but is loaded lazily; a catalog must resolve to a readable file only if an enabled stage needs it.
2. **No unknown keys** at any level — typos fail loud. Fields documented as arrays of ids/names (`needs`, `uses.estimators`, `uses.tables`, `group_by`, extract `outputs`, report `formats`, etc.) must be JSON arrays of strings; a bare string is an error.
3. **Unique ids** across `estimators` + `adjust` + `analyses`; explicit `needs`/`uses` must reference existing, enabled ids. Omitted or empty `uses.estimators` on analyses that opt in to the all-estimator default resolves to all enabled estimators. The config loader resolves and caches this graph once, and the pipeline consumes that resolved graph directly.
4. **Acyclic** after edge resolution; a cycle is an error naming the cycle.
5. **Estimator consistency**: `fit` matches exactly the one sub-block present (`train`/`pretrained`); `predict: cross_val` and `tune` are valid only with `fit: train`.
6. **Registry membership**: every analysis `module` resolves in the analysis registry; every `adjust` `module` resolves in the adjust registry (currently `strength`); every estimator `model` resolves in `catalogs.models` `prediction_models`.
7. **Adjust wiring**: each `adjust` stage must declare exactly one estimator in `uses.estimators`. A `uses.tables` name must resolve to either a `data.tables` key or an enabled `adjust` stage `id`; any `ratings.*` analysis must reference a `strength` table (no `adjust` stage ⇒ a `ratings.*` analysis is a validation error, since there is nothing to rate).
8. **Filter resolution**: every preset name in any `filter` exists in top-level `filters`; list filters merge left-to-right; a stage `filter` may not select experiments/players/turns excluded by the resolved global `data.filter` or lower constraints like `only_llm`/`min_games`; a `turn_range` must be `[min, max]` with `min <= max` (either bound nullable).
9. **Grouping resolution**: in a `ratings.*` `group_by`, every dimension past the base (`group_by[0]`, typically `player_type`) must name a grouping defined in top-level `groupings` (§3.2); referencing an undefined grouping is an error. Each grouping's `kind` must be implemented (currently only `argmax`); an `argmax` grouping's `labels`, when present, must be positional with `columns`.
10. **Bootstrap**: a `ratings.*` `bootstrap`, when not null, requires an integer `n >= 1`; its resampling is seeded from the top-level `seed` (determinism — same config ⇒ same CIs).
11. **No missing dependencies**: there is no graceful degradation. A stage requiring an uninstalled package (torch/xgboost/optuna/R) **aborts the run** with an install hint. Run `scripts/install` first so every dependency is present.
12. **Output root** (§2.1): `output`, when present, accepts only `root` (string) and `suffix` (string); both optional, defaulting to `"reports"` and `""`. Every stage save-path resolves under `<root><suffix>/`; two runs that differ only in `suffix` must not write to the same directory.
13. **Strength controlled-design params** (§5.1): `turn_progress_min` is numeric in `[0, 1]`; `weight ∈ {turn_progress, uniform}`; `relative_to ∈ {game_leader}`; `enforce_winner` is boolean; `civ_adjust ∈ {none, ols_logit}`; `block ∈ {none, start_cell, auto}`; `post_cell_normalize ∈ {none, relative_to_leader}`; `baseline_experiment` is null or a string experiment id. The id need not be listed in `experiments.json`; the adjust stage resolves it from extracted data. A `block` other than `none` affects only rows from the controlled subset (`seed != -1 and seating_rotation != -1`); uncontrolled rows always use `civ_adjust`. Missing cells in the selected implicit pathway and missing source rows for the selected explicit pathway are fatal; non-selected-pathway gaps, per-model gaps, disconnected models, and incomplete cycles warn.
14. **Extract invariants** (§3, §3.3): Vox Deorum can record distinct sync/map seeds, but civ-bench's controlled-design benchmark requires matched starts. Therefore, for a controlled game, a configured `configuredSyncRandSeed` must equal `configuredMapRandSeed` — a mismatch **aborts extraction** with a policy error. The `games` table stores one `seed` (the controlled value, else `-1`) and `seating_rotation` (else `-1`); `-1` is the uncontrolled sentinel (controlled seeds are `≥ 1` — `0` is Civ's "pick random" and rejected for controlled runs — and rotations are `≥ 0`). Per-player `config_slot` lives in `panel_data` and is joined by `(game_id, player_id)` where needed. A `player_type_labels` value is read as a **suffix** when it begins with `-`, else as a full **override**.
