# `benchmark.json` — the run-spec convention

`configs/benchmark.json` is the **top-level run spec**: it declares the input data, which
estimators to train or load, which analyses to run, and how to render the report. It is the only
file you edit to define a new benchmark run. This document is its authoritative schema.

`benchmark.json` is validated on load — **unknown keys and missing required fields are hard
errors** (invariant 1: config over code). When you add or rename a module, update this file in the
same change.

> JSON has no comments. This reference uses `jsonc` fences with `//` notes for explanation; the
> real `configs/benchmark.json` must be plain JSON. See [benchmark.json](benchmark.json) for a
> comment-free worked example.

---

## 1. The pipeline as a DAG

A run is a **directed acyclic graph of stages** in four kinds, executed in dependency order:

```
extract ──▶ estimators ──▶ analyses ──▶ report
```

Each stage has a stable **`id`**. Edges come from three places, all resolved into one topological
sort before anything runs:

1. **Kind ordering (implicit).** `extract` → `estimators` → `analyses` → `report`, always.
2. **`needs` (explicit).** A stage may list other stage `id`s it must run after. Use this to force
   ordering the harness can't infer (e.g. one analysis consuming another's CSV).
3. **`uses` (referential).** When a stage references an estimator `id` (or a named table) in its
   `uses` block, an edge is created automatically — you don't also have to write `needs`.

A cycle, an unknown `id`, or a reference to a disabled stage is a validation error. Disabled stages
(`"enabled": false`) are dropped from the graph, and anything that *needed* one is a validation
error. There is **no graceful degradation for missing dependencies**: if a stage needs `torch` /
`xgboost` / `optuna` / R and it isn't installed, the run **fails loud** — install everything first
(see [AGENTS.md](../AGENTS.md#dependencies) and `scripts/install`).

**Determinism.** The top-level `seed` is threaded into every stage that uses randomness (CV splits,
torch init, resampling, bootstrap). Same `benchmark.json` + same `runs/` ⇒ byte-stable outputs.

---

## 2. Top-level shape

```jsonc
{
  "name": "staff-standard-2026",        // required: run id; names the report + reports/ subdir
  "description": "Staff line-up, standard 8-seat map",  // optional, free text
  "seed": 42,                            // required: global RNG seed (determinism)

  "catalogs": {                          // optional: override the sibling config files
    "paths":       "configs/paths.json",
    "models":      "configs/models.json",
    "experiments": "configs/experiments.json"
  },

  "filters":    { /* §3.1 */ },          // optional: named, reusable filter presets

  "data":       { /* §3  */ },           // required: extraction + canonical tables + global filter
  "estimators": [ /* §4  */ ],           // optional: prediction-model producers
  "analyses":   [ /* §5  */ ],           // required: the modules to run
  "report":     { /* §6  */ }            // required: rendering
}
```

Top-level keys: `name`, `seed`, `data`, `analyses`, `report` are **required**; `description`,
`catalogs`, `filters`, and `estimators` are optional (omit `estimators` for a ratings/behavior-only
run).

**`catalogs` defaults to sibling files.** If omitted, the harness loads `paths.json`, `models.json`,
and `experiments.json` from the **same directory as this `benchmark.json`**. Set a key only to point
at a file elsewhere; unset keys still fall back to the sibling. (A missing required catalog — e.g.
no `models.json` next to the config and no override — is a load error.)

---

## 3. `data` — extract + canonical tables + global filter

`data` owns everything before the analysis modules: turning raw DBs into canonical CSVs, naming
those CSVs, and the **global** row filter that every downstream stage inherits (stages may narrow
further, never widen).

```jsonc
"data": {
  "extract": {
    "enabled": true,                     // false → reuse existing CSVs, never touch runs/ DBs
    "runs_dir": "runs/",                 // root searched for *.sqlite game DBs
    "outputs": ["turns", "panel", "timestamps", "tokens"],  // which canonical CSVs to (re)build
    "max_dbs": null,                     // int → only first N discovered DBs (smoke tests)
    "prune_missing": false,              // true → only drop rows for missing DBs, no new extract
    "force_rebuild": false               // true → rebuild even if outputs exist & are newer
  },

  "tables": {                            // canonical CSV locations (extract writes / loaders read)
    "turns":      "runs/turn_data.csv",          // per-player per-turn panel (prediction features)
    "panel":      "runs/panel_data.csv",         // per-player per-game outcomes/strategies/strength
    "timestamps": "runs/game_timestamps.csv",
    "tokens":     "runs/model_token_usage.csv"
  },

  "filter": "llm_only"                   // GLOBAL selector: inline object OR a preset name (§3.1)
}
```

- **`filter` is optional** — omit it for "all rows". It accepts either an inline filter object or the
  **name of a preset** from top-level `filters` (§3.1). Every stage inherits this global filter and
  may narrow it (§5.1), never widen it.
- **`extract.enabled: false`** is the "I already have CSVs" switch — the `extract` stage is dropped
  from the DAG and loaders read `tables.*` directly. Combine with `--skip extract` on the CLI for
  the same effect ad hoc.
- The `extract` stage is **skipped automatically** when every `outputs` CSV already exists and is
  newer than the DBs, unless `force_rebuild: true`.
- Experiment ids, player-type names, and the vanilla/null groupings all resolve through
  `catalogs.experiments` + `catalogs.models`; never spell out seat→model mappings here.

### 3.1 `filters` — named, reusable filter presets

A filter is the **same shape everywhere** it appears (`data.filter` and every stage's `filter`), so
define the common ones once and reference them by name instead of repeating the object. Every field
is optional; an omitted field means "no constraint".

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
- a **list** mixing preset names and inline objects, which are merged left-to-right (later entries
  win per field). E.g. `"filter": ["llm_only", { "turn_range": [200, null] }]`.

A stage's `filter` is then **intersected** with the resolved global `data.filter` — a stage can only
narrow, never widen (§5.1). Referencing an undefined preset name is a validation error.

---

## 4. `estimators` — prediction-model producers

Estimators are the pipeline's reason for having stages at all: a `performance.turn_predicted` or
`prediction.calibration` analysis needs `predicted_win_probability`, which only exists once a
predictor has run. Declare each predictor once in `estimators`; analyses reference it by `id`.

An estimator answers two **independent** questions, and that separation is the whole design:

1. **Where does the model come from?** — `fit`: `"train"` (fit on this run's data, optionally tuned
   first) or `"pretrained"` (load a saved model dir).
2. **How are the predictions it hands downstream generated?** — `predict`: `"in_sample"` (one model
   predicts the rows) or `"cross_val"` (k-fold; honest *out-of-fold* predictions).

It always emits the same artifact, regardless of how it was obtained:

> **estimator artifact** = a `predictions.csv` (the input rows + `predicted_win_probability`) and,
> when applicable, a saved model dir (`metadata.json` + state), a tuning result, and a feature-
> importance table.

**Computing metrics is not the estimator's job.** Scoring a `predictions.csv` (ROC-AUC, Brier, …) is
the separate `prediction.evaluate` / `prediction.compare` analysis step (§5.2). See §4.7 for why
that split matters.

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

`fit: pretrained` always predicts in-sample (you can't cross-validate weights you didn't train here),
so `predict` is ignored for it.

This pair is the **train-before-others vs. use-a-pre-trained-estimator** switch the pipeline is built
around. A `pretrained` estimator still runs *inference* on the current `tables.turns`, so it depends
on `extract` (or on the CSVs already existing) but not on training. Either way, downstream analyses
reference the estimator `id` and get its `predictions.csv`.

### 4.3 `tune` — optional Optuna pre-step (gates `train`)

Tuning is its own sub-stage that runs **before** the fit it feeds. You can run it fresh, or skip it
entirely by pointing at a previously saved best-params file (a *pre-trained hyperparameter set*).

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

- Tuning optimizes a **single scalar** `objective` (an objective must be scalar). Reporting *many*
  metrics is a different concern — that's `prediction.evaluate` (§5.2), which takes a `metrics` list.
- `load_params` set ⇒ tuning is skipped; the saved `best_params.json` is loaded as the
  hyperparameters. This is the cheap path for "reuse the tuning we already did" — a *pre-trained
  hyperparameter set*.
- **Hyperparameter precedence** (highest wins): explicit `params` → `load_params` → fresh
  `save_params` → the model class's coded defaults.

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

- `train_subset` controls which rows the model is fit on (e.g. `"non_llm"` trains only on
  Vanilla/Null seats, then predicts on everyone — the held-out generalization setup). For
  `predict: in_sample`, the inference subset is the entry's top-level `predict_subset`; for
  `cross_val`, predictions are out-of-fold over the `train_subset` and `predict_subset` is ignored.
- `save_model` applies to `predict: in_sample` (a single fitted model). Saved models written here
  are exactly what a later run's `pretrained.model_dir` can point at. For `cross_val` there is no
  single model to deploy, so `save_model` is ignored and the artifact is the OOF `predictions.csv`
  (+ feature importance).

### 4.5 `features` — selection (omit to use the estimator's own defaults)

`features` is **optional**. When omitted, each estimator uses its own coded `DEFAULT_FEATURES` (or
the feature set baked into a tuned `best_params.json`). Provide it only to override per-estimator:

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
  "model_dir": "reports/reference/attention_2026q1/"  // dir with metadata.json (dispatches class)
}
```

No `params` / `features` / `tune` are consulted — the saved `metadata.json` carries the architecture
and selected features; the class is resolved from the registry by `metadata.model_class`. Inference
runs on the entry's `predict_subset` and writes `save_predictions`.

### 4.7 Why evaluation is a *step*, not a *source*

Earlier drafts had an `evaluate` source alongside `train`/`pretrained`. It was removed because it
conflated two separable things:

- **Generating predictions** — in-sample vs. out-of-fold vs. loaded-and-inferred. This genuinely is
  a property of the estimator-*producer*: it changes which `predictions.csv` comes out. It now lives
  on the `predict` axis (`in_sample` | `cross_val`).
- **Scoring predictions** — computing ROC-AUC / Brier / log-loss / balanced-accuracy from a
  `predictions.csv`. That reads an artifact and reports numbers; it owns no model. It is, and always
  was, an **analysis**: `prediction.evaluate` / `prediction.compare` (§5.2).

So **"evaluate an estimator" = point a `prediction.evaluate` analysis at it.** To evaluate honestly,
give the estimator `predict: cross_val` so the analysis scores held-out predictions; to inspect a
deployed or pre-trained model's behavior, leave it `in_sample`/`pretrained`. The metrics step is
shared, multi-metric, and works identically across all estimators — which is exactly what you'd lose
by burying scoring inside each producer.

---

## 5. `analyses` — the pluggable modules

A list of analysis stages. Every entry shares a common envelope; the `params` block is
module-specific (catalog in §5.2). Order in the list is irrelevant — the DAG decides execution.

### 5.1 Common envelope

```jsonc
{
  "id": "bt_main",                       // required: unique stage id
  "module": "ratings.bradley_terry",     // required: analysis registry name
  "enabled": true,
  "needs": [],                           // optional explicit deps
  "uses": {                              // optional artifact references (create auto-edges)
    "estimators": ["attention", "score"],   // estimator ids → their predictions.csv
    "tables": ["panel"]                      // canonical table names from data.tables
  },
  "filter": "late_game",                 // optional: preset name, inline object, or list (§3.1);
                                         //   NARROWS the global filter for this stage
  "params": { /* module-specific, see §5.2 */ }
}
```

- `uses.estimators` is how `performance.*` / `prediction.*` modules get win-probabilities, and it is
  what makes them depend on (run after) those estimators.
- `filter` accepts the same preset-name / inline / list forms as `data.filter` (§3.1). It is
  intersected with the resolved global filter; a stage can only narrow, never widen.

### 5.2 Module params catalog

Every registry name `civ-bench` will ship, with its key params. Unlisted params fall back to coded
defaults; unknown params are validation errors.

#### `ratings.*` — consume `panel` (no estimator)

```jsonc
// ratings.bradley_terry — BT MLE with pairwise score weights (R: BradleyTerry2)
{ "module": "ratings.bradley_terry",
  "params": { "weighted": true, "ref": "Vanilla", "min_games": 5, "only_llm": false } }

// ratings.plackett_luce — PL MLE over per-game rankings (R: PlackettLuce)
{ "module": "ratings.plackett_luce",
  "params": { "ref": "Vanilla", "min_games": 5 } }

// ratings.elo — sequential/online Elo over game outcomes
{ "module": "ratings.elo",
  "params": { "k": 32, "initial": 1500, "order_by": "timestamp" } }

// ratings.strategy_elo — composite {PlayerType}-{Strategy} identities, single BT fit
{ "module": "ratings.strategy_elo",
  "params": { "strategy_cols": ["domination_ratio","culture_ratio","diplomatic_ratio","science_ratio"],
              "min_games": 5 } }

// ratings.matchups — empirical head-to-head matrices + OLS validation
{ "module": "ratings.matchups",
  "params": { "mode": "mean", "validate_ols": true } }

// ratings.bootstrap_bt — bootstrap CIs around BT ratings
{ "module": "ratings.bootstrap_bt",
  "params": { "n_bootstrap": 1000, "weighted": true } }

// ratings.iterative_bt — iterative/leave-one-experiment-out BT for stability
{ "module": "ratings.iterative_bt",
  "params": { "weighted": true } }

// ratings.vanilla_slot_effect — tests whether seat position confounds Vanilla rating
{ "module": "ratings.vanilla_slot_effect", "params": {} }

// ratings.ablation_bt — feature/condition ablations of the BT fit
{ "module": "ratings.ablation_bt",
  "params": { "ablate": ["weighted", "score_margin"] } }
```

#### `prediction.*` — consume one or more estimators (`uses.estimators` required)

```jsonc
// prediction.evaluate — metrics table across estimators (ROC-AUC/Brier/log-loss/bal-acc)
{ "module": "prediction.evaluate",
  "uses": { "estimators": ["naive","score","attention"] },
  "params": { "metrics": ["roc_auc","brier_score","log_loss","balanced_accuracy"] } }

// prediction.compare — side-by-side comparison table + ranking
{ "module": "prediction.compare",
  "uses": { "estimators": ["score","grouped_mlp","attention"] }, "params": {} }

// prediction.calibration — reliability diagrams per estimator
{ "module": "prediction.calibration",
  "uses": { "estimators": ["attention"] }, "params": { "n_bins": 10 } }

// prediction.loss_by_progress — loss binned by turn_progress
{ "module": "prediction.loss_by_progress",
  "uses": { "estimators": ["score","attention"] }, "params": { "n_bins": 20 } }

// prediction.winner_trajectories — P(win) trajectories for eventual winners
{ "module": "prediction.winner_trajectories",
  "uses": { "estimators": ["attention"] }, "params": { "sample_games": 12 } }

// prediction.elo_comparison — predicted-strength vs rating-based Elo cross-check
{ "module": "prediction.elo_comparison",
  "uses": { "estimators": ["attention"] }, "needs": ["bt_main"], "params": {} }

// prediction.context_slicing — metrics sliced by experiment / player type / turn range
{ "module": "prediction.context_slicing",
  "uses": { "estimators": ["attention"] }, "params": { "by": ["experiment","player_type"] } }
```

#### `performance.*` — strength + score, some need an estimator

```jsonc
// performance.score_ratio — per-player score-ratio regressions (consumes panel)
{ "module": "performance.score_ratio",
  "params": { "target": "score_ratio", "predictors": ["player_type","civilization"] } }

// performance.strength_panel — adjusted-strength panels by player type (consumes panel)
{ "module": "performance.strength_panel",
  "params": { "metric": "adjusted_strength", "by": "player_type" } }

// performance.turn_predicted — strength derived from an estimator's win-probabilities
{ "module": "performance.turn_predicted",
  "uses": { "estimators": ["attention"] },
  "params": { "aggregate": "mean", "by": "player_type" } }

// performance.permutation_importance — grouped permutation importance over feature families
{ "module": "performance.permutation_importance",
  "uses": { "estimators": ["attention"] },
  "params": { "n_repeats": 20, "groups": "feature_families" } }
```

#### `behavior.*` — descriptive over turn/token/flavor data

```jsonc
{ "module": "behavior.flavor_change_clusters",     "params": { "n_clusters": 8 } }
{ "module": "behavior.flavor_change_decomposition","params": { "components": 5 } }
{ "module": "behavior.pivot_rationale",            "params": { "min_pivots": 1 } }
{ "module": "behavior.victory_commitment",         "params": { "window": 10 } }
{ "module": "behavior.nuke_flavor_rationale",      "params": {} }
```

#### `exploratory.*` — dataset descriptives

```jsonc
{ "module": "exploratory.panel",            "params": {} }
{ "module": "exploratory.turn",             "params": {} }
{ "module": "exploratory.strategy_profiles","params": { "by": "player_type" } }
// model_token_costs uses tokens table + pricing from models.json
{ "module": "exploratory.model_token_costs","uses": { "tables": ["tokens"] },
  "params": { "currency": "usd" } }
```

---

## 6. `report` — rendering

```jsonc
"report": {
  "template": "default",                 // template name under civ_bench/reports/
  "out_dir": "reports/",                 // output root; run writes reports/<name>/
  "formats": ["md", "html"],             // any of: md, html, pdf
  "sections": null,                      // null = every produced AnalysisResult, in DAG order;
                                         //   or an explicit ordered list of stage ids to include
  "title": null,                         // null = derive from `name`
  "include_disabled": false              // never render skipped/disabled stages
}
```

The report walks each produced `AnalysisResult` (tables + figures + summary). With
`sections: null`, every enabled analysis appears in dependency order; pass an ordered `id` list to
curate and reorder. No analysis hardcodes its place in the document (invariant 3).

---

## 7. Validation rules (enforced on load)

1. **Required keys present**: `name`, `seed`, `data`, `analyses`, `report`. `catalogs` is optional
   (defaults to sibling files; each catalog must still resolve to a readable file).
2. **No unknown keys** at any level — typos fail loud.
3. **Unique ids** across `estimators` + `analyses`; `needs`/`uses` must reference existing,
   enabled ids.
4. **Acyclic** after edge resolution; a cycle is an error naming the cycle.
5. **Estimator consistency**: `fit` matches exactly the one sub-block present (`train`/`pretrained`);
   `predict: cross_val` and `tune` are valid only with `fit: train`.
6. **Registry membership**: every `module` resolves in the analysis registry; every estimator
   `model` resolves in `catalogs.models` `prediction_models`.
7. **Filter resolution**: every preset name in any `filter` exists in top-level `filters`; a stage
   `filter` may not select experiments/players/turns excluded by the resolved global `data.filter`;
   a `turn_range` must be `[min, max]` with `min <= max` (either bound nullable).
8. **No missing dependencies**: there is no graceful degradation. A stage requiring an uninstalled
   package (torch/xgboost/optuna/R) **aborts the run** with an install hint. Run `scripts/install`
   first so every dependency is present.
