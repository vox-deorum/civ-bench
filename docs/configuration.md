# Configuration guide

One JSON file, the **benchmark run-spec**, controls a `civ-bench` run.

| Task | Reference |
| --- | --- |
| Set up a first run | [Getting Started](getting-started.md) |
| Find a report section's ID or display name | [Report section reference](#report-section-reference) |
| Move sections or choose overview cards | [Report settings](#report-rendering) |
| Look up every field and validation rule | [Complete schema](../configs/benchmark.md) |

- Put dataset, experiment, model, and report choices in JSON.
- Copy `configs/*.template.json` to a local, gitignored `configs/benchmark*.json` before editing machine-specific settings.
- Validation rejects unknown keys and missing required fields.

---

## The shape of a run-spec

```jsonc
{
  "name": "staff-standard-2026",   // required: names the report and its output subdirectory
  "friendly_name": "Staff 2026",    // optional: human title shown on the report page
  "description": "free text",       // optional; shown on the report page
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

| Setting | Rule |
| --- | --- |
| Required fields | `name`, `seed`, `data`, `analyses`, `report` |
| `estimators`, `adjust` | Optional when analyses do not need predicted strength |
| Strength-based ratings | `ratings.bradley_terry`, `ratings.plackett_luce`, and `ratings.matchups` need a strength-producing adjust stage |
| `friendly_name` | Default report title; `report.title` overrides it |
| `description` | Description on the report overview |
| `seed` | Shared seed for cross-validation, model initialization, and bootstrap resampling |

The same run-spec and input data produce byte-stable outputs.

---

## How stages connect: the DAG

A run is a **directed acyclic graph of stages** in five kinds, executed in dependency order:

```
extract  ->  estimators  ->  adjust  ->  analyses  ->  report
```

| Dependency source | Meaning |
| --- | --- |
| Stage kind | The five kinds run in the order above |
| `needs` | Explicit stage IDs that must run first |
| `uses` | Estimator, table, or analysis references create dependencies automatically; no duplicate `needs` entry is required |

- Cycles, unknown IDs, and references to disabled stages fail validation.
- Preview the resolved graph with `civ-bench run --config <file> --dry-run`.

---

## `data`: input and the global filter

`data` owns everything before the analysis modules: turning raw databases into canonical CSVs, naming those CSVs, and the global row filter every downstream stage inherits.

```jsonc
"data": {
  "extract": {
    "enabled": true,                 // false: reuse existing CSVs, never touch runs/ DBs
    "runs_dir": "runs/",             // root searched for *.db game DBs
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

| Filter form | Example |
| --- | --- |
| Inline object | `"filter": {"only_llm": true}` |
| Named preset | `"filter": "llm_only"` |
| Combined list | `"filter": ["llm_only", {"turn_range": [200, null]}]` |

Lists merge left to right; later entries win per field. Stage filters still cannot widen the global filter.

### Groupings: how ratings slice the field

A named grouping derives a category from the strength panel for use in a rating's `group_by`.

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

### Player identity: `player_type`

| Property | Behavior |
| --- | --- |
| Example | `Sonnet-4.5-Briefed` |
| Source | Per-player model and strategist metadata, composed during extraction |
| Catalog role | Supplies the naming template, alias normalization, and label map |
| Seat rotations | Identity follows the player across seats |
| Run-spec setup | No seat-to-model mapping needed |

See [configs/benchmark.md](../configs/benchmark.md), section 3.3, for the composition rules.

---

## `estimators`: the victory-probability predictors

An estimator emits `predictions.csv` with a `predicted_win_probability` column. Configure weight fitting and prediction generation separately.

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

| Choice | Options or behavior |
| --- | --- |
| Available models | `naive`, `score`, `baseline`, `xgboost`, `mlp`, `grouped_mlp`, `interaction_mlp`, `attention_mlp` |
| Paper setup | `attention_mlp`, five-fold cross-validation grouped by game |
| Pretrained weights | Always use in-sample prediction |
| Prediction scoring | Separate analyses: `prediction.evaluate` and `prediction.compare` |

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
    "block": "auto",                      // controlled games: matched start-cell correction
  }
}
```

| Step | Behavior |
| --- | --- |
| Aggregate predictions | Average win probabilities with progress weights |
| Normalize | Compare each player with the game's strongest player |
| Preserve the winner | Keep the actual winner at the top |
| Uncontrolled games | Remove civilization effects with an OLS fit on the logit scale |
| Controlled games | Use `block` to correct against VPAI in the same `(seed, seat)` cell |
| Shared result | Ratings consume the same `adjusted_strength` estimate |

To exclude incomplete controlled conditions:

- Set `data.filter.min_condition_completeness` in the global filter.
- Use `1.0` to exclude any condition missing a `seed × seating_rotation` slot.
- The filter removes those conditions from every table before strength fitting and ratings.

See [configs/benchmark.md](../configs/benchmark.md), sections 3.1 and 5, for completeness filtering, baseline choices, and diagnostics.

---

## `analyses`: the pluggable modules

A list of analysis stages. Every entry shares one envelope; the `params` block is module-specific.

```jsonc
{
  "id": "bt_main",
  "module": "ratings.bradley_terry",
  "name": "Main ratings",              // optional: friendly heading for this section on the report
  "description": "The headline Elo table.",  // optional: one-line description under the heading
  "enabled": true,
  "uses": { "tables": ["strength"] },     // canonical table or an adjust stage's table
  "filter": "late_game",                  // optional, narrows the global filter for this stage
  "params": { "group_by": ["player_type"], "ref": "Vanilla", "min_games": 5 }
}
```

| Family | Purpose | Implemented modules |
| --- | --- | --- |
| `ratings` | Compare skill and outcomes | `bradley_terry`, `plackett_luce`, `matchups`, `outcome_matchups` |
| `prediction` | Evaluate predictors | `evaluate`, `compare` |
| `calibration` | Check prediction reliability and adjustment effects | `reliability`, `loss_by_progress`, `civ_effects`, `cell_baseline` |
| `performance` | Compare strength, coverage, progress, cost, and games | `score_ratio`, `strength_panel`, `experiment_completeness`, `turn_predicted`, `controlled_seed_report`, `game_log`, `usage_efficiency` |

- `prediction` analyses use all enabled estimators by default; `uses.estimators` narrows the selection.
- `name` and `description` override a module's display text for that stage.
- Strategy-grouped ratings select their own display names automatically.
- The [report section reference](#report-section-reference) maps stage IDs to display names.
- The [full template](../configs/benchmark.full.template.json) includes disabled optional stages. Reserved modules fail with a "reserved but not implemented" error if enabled.
- See [schema sections 6.2 and 6.3](../configs/benchmark.md) for parameters and display descriptions.

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

Use analysis **stage IDs** in `sections`, `overview_sections`, and `section_overrides`.

| Name type | Example | Where to use it |
| --- | --- | --- |
| Stage ID | `bt_main` | Report lists and cross-stage references |
| Module | `ratings.bradley_terry` | An analysis's `module` field |
| Display name | Pairwise skill ratings | Report heading; override with the analysis's `name` |
| Chapter name | Ratings | Generated navigation label, determined by the module family |

### Report section reference

These IDs come from the [full template](../configs/benchmark.full.template.json). Use the IDs in your own `analyses` list if you have renamed stages. A stage must exist in your config before you can reference it.

| Chapter | Stage ID | Default display name |
| --- | --- | --- |
| Matched Maps | `controlled_seed` | Matched Maps |
| Ratings | `bt_main` | Pairwise skill ratings |
| Ratings | `bt_strategy` | Pairwise strategy ratings |
| Ratings | `pl_main` | Rank-based skill ratings |
| Ratings | `pl_strategy` | Rank-based strategy ratings |
| Ratings | `matchup_strength` | Adjusted-strength matchups |
| Ratings | `matchup_winrates` | Victory matchups |
| Prediction | `pred_metrics` | Prediction quality |
| Prediction | `pred_compare` | Estimator agreement |
| Calibration | `cal_reliability` | Prediction reliability |
| Calibration | `cal_loss_progress` | Prediction error over time |
| Calibration | `cal_civ_effects` | Civilization strength effects |
| Calibration | `cal_cell_baseline` | Starting-position baselines |
| Performance | `perf_score_ratio` | Final-score effects |
| Performance | `perf_strength` | Gameplay strength |
| Performance | `perf_experiment_completeness` | Experiment coverage |
| Performance | `perf_turn_predicted` | Win-probability trends |
| Performance | `game_log` | Game Log |
| Performance | `perf_usage_efficiency` | Usage, cost, and skill |

- `controlled_seed`, `pl_main`, and `pl_strategy` are disabled in the full template; enable them before including them in a report.
- Strategy rating names depend on `group_by: ["player_type", "strategy"]`.
- Exploratory modules in the full template are reserved placeholders, so they have no implemented report sections.
- Friendly names and chapter names are display text. For example, use `"bt_main"`, not `"Ratings"`, in `report.sections`.

### Report settings

Example: put Matched Maps first and show two selected overview cards. This assumes `controlled_seed` is enabled.

```jsonc
"report": {
  "out_dir": "reports/",
  "formats": ["md", "html"],
  "sections": ["controlled_seed"],
  "overview_sections": ["bt_main", "perf_usage_efficiency"],
  "section_overrides": {},
  "title": null,
  "include_disabled": false,
  "replay": null
}
```

| Field | Behavior |
| --- | --- |
| `out_dir` | Report directory under the resolved output root; the run's `name` supplies its subdirectory |
| `formats` | Defaults to `["md", "html"]`; `pdf` is reserved and cannot render yet |
| `sections` | Priority stage IDs, followed by every remaining enabled analysis; `null` or `[]` uses the default order |
| `overview_sections` | Exact ordered selection of overview cards; `null` includes every resolved section, `[]` shows none |
| `section_overrides` | Inline table and figure selections by stage ID |
| `title` | Report title; `null` falls back to `friendly_name`, then `name` |
| `include_disabled` | Defaults to `false`; `true` allows explicitly listed disabled stages with saved results. Automatic fill still includes only enabled analyses |
| `footer` | Markdown footer; `null` uses the default CivBench citation, `""` hides it |
| `benchmark_citation` | Optional benchmark citation with `title` and `url`; see [schema section 7](../configs/benchmark.md#7-report-rendering) |
| `replay` | Optional replay saves and viewer links; requires an enabled `performance.game_log` stage |

Set `report.replay` to enable the Game Log and copied `.Civ5Save` files:

```jsonc
"replay": {
  "enabled": true,
  "viewer_url": "https://vox-deorum.github.io/vox-deorum-replay/",
  "saves": "all",       // "all" or "controlled"
  "base_url": null,      // optional absolute published report URL
  "latest_game": true
}
```

The Game Log filters games and links to the hosted viewer. `saves: "controlled"`
limits copied saves to games with a controlled seed. Reports served from
`file://` cannot launch the viewer; serve the report over HTTP(S) with CORS
enabled for the viewer origin.

### Ordering and overview cards

| Goal | Setting inside `report` |
| --- | --- |
| Put Matched Maps first, then include the rest | `"sections": ["controlled_seed"]` |
| Put Ratings before Matched Maps | `"sections": ["bt_main", "controlled_seed"]` |
| Put cost and usage before the other Performance sections | `"sections": ["perf_usage_efficiency"]` |
| Use the default chapter and section order | `"sections": null` |
| Show only skill and cost cards on the overview | `"overview_sections": ["bt_main", "perf_usage_efficiency"]` |
| Show a card for every resolved section | `"overview_sections": null` |

- Default family order: Ratings, Prediction, Calibration, Performance, Exploratory. Within a family, config order applies.
- Chapters follow their first section's position in the resolved list. Sections in the same family stay together.
- Matched Maps forms its own chapter when its analysis is non-empty. Its default position follows its `performance` module's place in the resolved list.
- Partial `sections` lists append the rest automatically. Disable an analysis to exclude it from the default report.
- Partial `overview_sections` lists select only those cards. The tracked templates select seven cards.
- Unknown stage IDs are errors; duplicate IDs appear only once.

### Inline tables, figures, and display text

| Setting | Effect |
| --- | --- |
| `"section_overrides": {"bt_main": {"tables": ["ratings"]}}` | Show the named ratings table; keep the default figures |
| `"section_overrides": {"bt_main": {"figures": []}}` | Hide inline figures; keep their downloads |
| Omitted `tables` or `figures` | Use the analysis's default list for that dimension |
| Artifact name not emitted by the analysis | Warn and skip that artifact |
| Analysis `name` | Override the section's display heading |
| Analysis `description` | Override the description in its heading tooltip |
| Top-level `description` | Display under the title in `index.html` and `report.md` |

Artifact names come from the analysis's saved `result.json`. Hidden inline artifacts remain downloadable.

### Generated pages

| Output | Contents |
| --- | --- |
| `index.html` | Overview cards and navigation |
| `report.md` | Combined Markdown report |
| `ratings.html`, `prediction.html`, `calibration.html`, `performance.html`, `exploratory.html` | One page per represented family |
| `controlled-seed/index.html` | Matched Maps tables by seed, in Strength, Focus, and Strategic tabs |
| `controlled-seed/seed-<seed>-player-<position>.html` | Probability curves and the same tabs for one seed and player position |
| `games.html` | Filterable per-game log and replay links, when `report.replay.enabled` |
| `saves/<experiment>/<game_id>.Civ5Save` | Copied replay saves, when available and in scope |
| `assets/` | Styles, scripts, figures, and downloadable tables |

- Only requested formats and represented chapters are written.
- Matched Maps requires an enabled, non-empty `performance.controlled_seed_report` analysis. Its HTML pages require `html` in `formats`.
- Matched Maps shows adjusted strength and dominant victory focus, with a separate VPAI baseline row.
- `civ-bench report --config <file>` rebuilds reports from saved analysis results without rerunning analyses. Rendering is deterministic and byte-stable.
- See [schema section 7.1](../configs/benchmark.md#71-the-matched-maps-chapter) for Matched Maps details.
- See [schema section 7.2](../configs/benchmark.md#72-game-log-and-replay-links) for Game Log and replay details.

---

## `output`: variants that coexist on disk

Use `output.suffix` to keep run variants under separate output roots.

```jsonc
"output": { "root": "reports", "suffix": "-cross" }   // writes under reports-cross/
```

| Path type | Effect of the suffix |
| --- | --- |
| Save paths under `output.root` | Re-rooted, for example from `reports/` to `reports-cross/` |
| Input paths | Read as authored |
| Pretrained snapshots | Stay at paths such as `pretrained/score/` and can serve multiple variants |

---

## Validation

| Check | Requirement |
| --- | --- |
| Keys and types | Required fields present; no unknown keys; correct value types |
| Stage IDs | Unique across estimators, adjust stages, and analyses |
| Dependencies | `needs` and `uses` resolve to existing enabled stages; no cycles |
| Estimator fitting | `fit` matches exactly one `train` or `pretrained` block |
| Modules and presets | Registry modules, filter presets, and groupings must resolve |
| Numeric parameters | Positive bootstrap `n`; strength parameters within allowed ranges |
| Installed packages | Missing required Python or R dependencies stop execution with an install hint |

See [configs/benchmark.md](../configs/benchmark.md), section 8, for the complete rules.

---

## Where to go deeper

- **[configs/benchmark.md](../configs/benchmark.md)** is the authoritative schema, every field and every validation rule.
- **[Getting Started](getting-started.md)** is the hands-on first-run walkthrough and tutorial.
- **[Developer guide](development.md)** covers extending the harness with new modules.
