# Configuration guide

Everything `civ-bench` does is driven by one JSON file, the **benchmark run-spec**. This guide is a readable tour of that file: what each block is for, how the blocks fit together, and the edits you make most often. It is the companion to [configs/benchmark.md](../configs/benchmark.md), which is the **authoritative, field-by-field schema** (when the two disagree, the schema wins). For a first run, start with the [Getting Started guide](getting-started.md) instead.

A guiding principle: **config over code.** Anything that changes between datasets, experiments, model line-ups, or report selections lives here, never in Python. The run-spec is validated on load, so unknown keys and missing required fields are hard errors. Typos fail loud rather than silently doing nothing.

> The repo tracks example run-specs as `configs/*.template.json`. Copy one to a local, gitignored `configs/benchmark*.json` and edit that. Never edit a template in place.

---

## The shape of a run-spec

```jsonc
{
  "name": "staff-standard-2026",   // required: names the report and its output subdirectory
  "description": "free text",       // optional
  "seed": 42,                       // required: global RNG seed, threaded everywhere for determinism

  "output":     { ... },            // optional: output root + variant suffix
  "catalogs":   { ... },            // optional: override the sibling config files
  "filters":    { ... },            // optional: named, reusable row-filter presets
  "groupings":  { ... },            // optional: named rating-identity dimensions

  "data":       { ... },            // required: extraction + canonical table paths + global filter
  "estimators": [ ... ],            // optional: victory-probability predictors
  "adjust":     [ ... ],            // optional: derived tables (the strength panel)
  "analyses":   [ ... ],            // required: the modules to run
  "report":     { ... }             // required: rendering
}
```

`name`, `seed`, `data`, `analyses`, and `report` are required; the rest are optional. Omit `estimators` and `adjust` for a run with no prediction-derived strength ratings. Conversely, the strength-based ratings (`ratings.bradley_terry`, `ratings.plackett_luce`, `ratings.matchups`) need an `adjust` stage to supply the `strength` table they rate.

**Determinism.** The top-level `seed` is threaded into every stage that uses randomness (cross-validation splits, torch init, bootstrap resampling). The same run-spec over the same `runs/` data produces byte-stable outputs.

---

## How stages connect: the DAG

A run is a **directed acyclic graph of stages** in five kinds, executed in dependency order:

```
extract  ->  estimators  ->  adjust  ->  analyses  ->  report
```

You do not write the edges by hand. They come from three places, all resolved into one topological sort before anything runs:

1. **Kind ordering (implicit).** The five kinds always run in the order above.
2. **`needs` (explicit).** A stage may list other stage `id`s it must run after. Use this for ordering the harness cannot infer (for example, one analysis that reads another's CSV).
3. **`uses` (referential).** When a stage references an estimator `id` or a named table in its `uses` block, the harness creates the edge automatically. You do not also write `needs`.

A cycle, an unknown `id`, or a reference to a disabled stage is a validation error. Run `civ-bench run --config <file> --dry-run` to print the resolved DAG without executing anything.

---

## `data`: input and the global filter

`data` owns everything before the analysis modules: turning raw databases into canonical CSVs, naming those CSVs, and the global row filter every downstream stage inherits.

```jsonc
"data": {
  "extract": {
    "enabled": true,                 // false: reuse existing CSVs, never touch runs/ DBs
    "runs_dir": "runs/",             // root searched for *.sqlite game DBs
    "outputs": ["turns", "panel", "games", "tokens"],
    "force_rebuild": false,          // true: rebuild even if outputs are newer than the DBs
    "auto_fix": true                 // true: repair malformed DBs & re-import (CLI --no-fix disables)
  },
  "tables": {
    "turns":  "runs/turn_data.csv",          // per-player per-turn features
    "panel":  "runs/panel_data.csv",         // per-player per-game outcomes, strategy, civ
    "games":  "runs/game_data.csv",          // per-game: timestamp, experiment, seed, seating
    "tokens": "runs/model_token_usage.csv"
  },
  "filter": "llm_only"               // global selector: a preset name, inline object, or list
}
```

- Set `extract.enabled: false` (or pass `--skip extract`) when you already have the four CSVs and just want to analyze them.
- Extraction is **skipped automatically** when every output CSV exists and is newer than the game DBs, unless `force_rebuild` is set.
- When a fresh extract records malformed DBs, they are **auto-repaired and re-imported** (extract → `fix` → re-import) before the rest of the run. Turn this off with `auto_fix: false` or the CLI `--no-fix` flag.
- The global `data.filter` is inherited by every stage. A stage may *narrow* it but never *widen* it.

### Filters: named, reusable, composable

A filter is the same shape everywhere it appears, so define the common ones once by name and reference them. Every field is optional; an omitted field means "no constraint."

```jsonc
"filters": {
  "llm_only":  { "only_llm": true, "min_games": 5 },
  "late_game": { "turn_range": [200, null] }
}
```

The full filter shape:

```jsonc
{
  "experiments":         null,   // null = all; or ["2026-staff-standard", ...]
  "exclude_experiments": [],     // subtracted from the above
  "players":             null,   // null = all player types; or ["Sonnet-4.5-Briefed", ...]
  "only_llm":            false,  // true: drop Vanilla/Null seats
  "min_games":           1,      // drop player types with fewer games than this
  "turn_range":          null    // null = all turns; or [min, max], either bound nullable
}
```

Anywhere a filter is accepted, the value may be an **inline object**, a **string** naming a preset, or a **list** mixing the two (merged left to right, later entries win per field). For example: `"filter": ["llm_only", { "turn_range": [200, null] }]`.

### Groupings: how ratings slice the field

A grouping derives one categorical dimension from the strength panel that `ratings.*` analyses can fold into the rated identity. Like filters, you define them once by name. This is why per-strategy Elo is a config edit, not a new module.

```jsonc
"groupings": {
  "strategy": {
    "kind": "argmax",
    "columns": ["domination_ratio", "culture_ratio", "diplomatic_ratio", "science_ratio"],
    "labels":  ["Domination", "Culture", "Diplomatic", "Science"]
  }
}
```

`kind: "argmax"` (the only kind implemented) labels each player-game by whichever column is largest, the dominant-strategy rule. Reference it from a rating's `group_by` (below).

### Player identity: the orthodox `player_type`

The identity every rating is fit over, `player_type` (for example `Sonnet-4.5-Briefed`), is **composed at extract time from the per-player game metadata**, not from a hand-maintained seat map. Each seat records its model and its strategist scaffold, and the catalog supplies a template, alias normalization, and a unified label map. Because the identity travels with the player, it stays correct even when controlled seating rotates a model through different seats. You never spell out seat-to-model mappings in the run-spec. See [configs/benchmark.md](../configs/benchmark.md) section 3.3 for the composition rules.

---

## `estimators`: the victory-probability predictors

An estimator emits one artifact, a `predictions.csv` with a `predicted_win_probability` column, and it answers two **independent** questions. That separation is the heart of the design.

```jsonc
{
  "id": "attention",            // unique; what analyses reference in `uses`
  "model": "attention_mlp",     // a prediction_models id from models.json
  "fit": "train",               // "train" or "pretrained"
  "predict": "in_sample",       // "in_sample" (default) or "cross_val"
  "enabled": true,
  "predict_subset": "all",
  "save_predictions": "reports/estimators/attention/predictions.csv",

  "tune":       { ... },        // optional Optuna pre-step, fit == train only
  "train":      { ... },        // required when fit == train
  "pretrained": { ... }         // required when fit == pretrained
}
```

**Axis 1, `fit`: where do the weights come from?**

| `fit`        | What runs                                  | Use when |
|--------------|--------------------------------------------|----------|
| `train`      | (optional tune, then) fit on this run's data | You want a fresh model fit to this data. |
| `pretrained` | load a saved model directory, no training  | You have a model trained elsewhere and want to apply it here. |

**Axis 2, `predict`: how are downstream predictions generated?** (only meaningful with `fit: train`)

| `predict`    | What it emits                                   | Use when |
|--------------|-------------------------------------------------|----------|
| `in_sample`  | one model predicting `predict_subset`           | You want a single deployed model and its predictions. |
| `cross_val`  | k-fold out-of-fold predictions (honest)         | You want honest held-out predictions to evaluate and calibrate on. |

The paper uses 5-fold cross-validation grouped by game for its honest predictions. A `pretrained` estimator always predicts in-sample (you cannot cross-validate weights you did not train here).

The available models, increasing in complexity, are `naive`, `score`, `baseline`, `xgboost`, `mlp`, `grouped_mlp`, `interaction_mlp`, and `attention_mlp` (the paper's primary estimator).

> **Scoring is not the estimator's job.** Computing ROC-AUC, Brier, and so on from a `predictions.csv` is a separate analysis step, `prediction.evaluate` / `prediction.compare`. That keeps scoring shared, multi-metric, and identical across every estimator.

---

## `adjust`: the strength panel

`adjust` is an optional list of derived-table stages that run after `estimators` and before `analyses`. Today there is one module, `strength`, which turns an estimator's per-turn win probabilities into the per-player-game skill estimate that ratings are fit over.

```jsonc
{
  "id": "strength",                       // also the table name downstream stages reference
  "module": "strength",
  "uses": { "estimators": ["attention"] }, // the predictor whose P(win) defines strength
  "save": "reports/adjust/player_strength_panel.csv",
  "params": {
    "turn_progress_min": 0.2,             // ignore the opening, average over late-game turns
    "weight": "turn_progress",            // weight each turn's P(win) by progress
    "relative_to": "game_leader",         // normalize each seat to its game's strongest seat
    "enforce_winner": true,               // force the actual winner to the top
    "civ_adjust": "ols_logit",            // uncontrolled games: subtract civilization effects
    "block": "auto"                       // controlled games: matched start-cell correction
  }
}
```

The derivation follows the paper: progress-weighted average to relative standing against the strongest player, a winner-preserving correction, then an OLS fit on the logit scale to remove civilization effects (the paper's *revised standing*; the code's `adjusted_strength`). In **controlled** games with fixed seeds and seating, `block` swaps the civilization adjustment for a matched start-cell correction that subtracts the Vanilla baseline of the same `(seed, seat)` cell, removing the start-position confound. The full controlled-design behavior, the baseline pathways, and the diagnostic files it always writes are documented in [configs/benchmark.md](../configs/benchmark.md) section 5.

Why a separate stage rather than logic inside each rating: every rating consumes the same strength estimate, so the derivation lives in one place instead of being copy-pasted.

---

## `analyses`: the pluggable modules

A list of analysis stages. Every entry shares one envelope; the `params` block is module-specific.

```jsonc
{
  "id": "bt_main",
  "module": "ratings.bradley_terry",
  "enabled": true,
  "uses": { "tables": ["strength"] },     // canonical table or an adjust stage's table
  "filter": "late_game",                  // optional, narrows the global filter for this stage
  "params": { "group_by": ["player_type"], "ref": "Vanilla", "min_games": 5 }
}
```

The modules, grouped into five families:

- **ratings** rate skill: `bradley_terry`, `plackett_luce`, `matchups`, `outcome_matchups`. Per-strategy Elo is `group_by: ["player_type", "strategy"]`, and confidence intervals are a `bootstrap` param, both on the ordinary fit rather than separate modules.
- **prediction** scores the predictor: `evaluate`, `compare`. These opt in to scoring every enabled estimator by default; add `uses.estimators` only to narrow.
- **calibration** checks honesty: `reliability`, `loss_by_progress`, `civ_effects`, `cell_baseline`.
- **performance**: `score_ratio`, `strength_panel`, `experiment_completeness`, `turn_predicted`.
- **exploratory**: `model_token_costs` (uses the token table and pricing from `models.json`).

Optional modules are listed in `benchmark.full.template.json` with `"enabled": false`. Some are registry-reserved placeholders until their implementation lands; if you enable one too early, the run fails with a clear "reserved but not implemented" error. The full per-module parameter catalog is in [configs/benchmark.md](../configs/benchmark.md) section 6.2.

### Two cross-cutting rating params

```jsonc
// per-strategy Elo: the rated identity becomes player_type-strategy
"params": { "group_by": ["player_type", "strategy"] }

// bootstrap confidence intervals (seeded from the top-level seed)
"params": { "bootstrap": { "n": 1000, "stratified": true } }
```

When the strength table uses a controlled-design `block` adjustment, the bootstrap re-runs the strength fit inside each replicate, so the start-cell baseline's uncertainty is reflected in the intervals.

---

## `report`: rendering

```jsonc
"report": {
  "template": "default",          // only "default" today
  "out_dir": "reports/",          // under the resolved output root
  "formats": ["md", "html"],      // md and html implemented
  "sections": null,               // null = every enabled analysis in canonical family order;
                                  //   or an explicit ordered list of stage ids to curate
  "title": null,                  // null = derive from name
  "include_disabled": false
}
```

The report walks each analysis result and renders one section per analysis. With `sections: null`, every enabled analysis appears, bucketed into the five families in canonical order. Pass an ordered list of ids to curate and reorder. Each analysis persists a `result.json` beside its artifacts, so `civ-bench report` re-renders the document from disk, deterministically and byte-identically, without re-running any analysis.

---

## `output`: variants that coexist on disk

Every stage that writes does so under a single run output root, `reports/` by default. `output` lets a run redirect everything to a sibling root by appending a suffix. This is the mechanism behind the `-cross` variant (train on non-LLM seats, write to `reports-cross/`).

```jsonc
"output": { "root": "reports", "suffix": "-cross" }   // writes under reports-cross/
```

Only **save** paths are re-rooted. **Inputs** are read as authored, so the tracked `pretrained/<model_id>/` snapshots feed every variant without being moved. A `-dev` run can load `pretrained/score/` yet write its predictions under `reports-dev/`.

---

## Validation, in one breath

The loader enforces, on load: required keys present; no unknown keys anywhere; unique ids across estimators, adjust, and analyses; `needs` and `uses` reference existing enabled ids; the graph is acyclic; `fit` matches exactly the one sub-block present; every `module` resolves in its registry; every preset name and grouping name resolves; bootstrap `n` is a positive integer; and strength-stage params are in range. There is no graceful degradation for missing packages: a stage that needs torch, xgboost, optuna, or R aborts the run with an install hint. The complete, numbered rule list is [configs/benchmark.md](../configs/benchmark.md) section 8.

---

## Where to go deeper

- **[configs/benchmark.md](../configs/benchmark.md)** is the authoritative schema, every field and every validation rule.
- **[Getting Started](getting-started.md)** is the hands-on first-run walkthrough and tutorial.
- **[Developer guide](development.md)** covers extending the harness with new modules.
