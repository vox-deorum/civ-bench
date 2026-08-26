# civ-bench

**The benchmark harness behind CivBench: a progress-based evaluation of LLM strategists playing *Civilization V: Vox Populi*.**

`civ-bench` takes the raw data from games that AI models played (via the [Vox Deorum](https://arxiv.org/abs/2512.18564) platform) and turns it into a finished report: tables, figures, and skill ratings that tell you which strategist looks strongest. You point it at your game data, hand it a JSON config, and it does the rest. It pulls the data out of the game databases, fits turn-level victory-probability models, distills those into skill ratings, checks how trustworthy the predictions are, and renders everything into Markdown and HTML.

The whole thing is driven by JSON. Adding a new strategist to the line-up, swapping in a different dataset, or turning an analysis on or off is a config edit, never a code change.

> **New here? Read the [Getting Started guide](docs/getting-started.md).** It walks you through installation, making your first config, running the pipeline, and reading the report, with a hands-on tutorial.

This repository implements the analysis from the CivBench paper (Chen, Cheng, Gurkan, and Lin, [arXiv:2604.07733](https://arxiv.org/abs/2604.07733)). [About CivBench](docs/about.md) explains the research problem and the project's design principles.

---

## How the pipeline works

A run is not a fixed script. It is a **directed acyclic graph (DAG) of stages** that the harness wires up from your config and runs in dependency order. There are five kinds of stage:

```text
raw game DBs ──> extract ──> estimators ──> adjust ──> analyses ──> report
                   |            |             |           |            |
               canonical    train OR      strength    ratings /    templated
                 CSVs       pre-trained     panel      prediction / markdown
                            predictors   (adjusted-    calibration / + html
                                          strength)    performance
```

- **extract** walks your game SQLite databases and writes four canonical CSVs: per-turn data, per-player-game panel data, per-game metadata (timestamps, controlled seeds, seating), and model token usage. It figures out each player's *identity* (the `player_type`, for example `Sonnet-4.5-Briefed`) from the game metadata, so the identity follows a model even when controlled games rotate it through different seats.
- **estimators** are the victory-probability predictors. Each one is either **trained** on the current data (optionally tuned first with Optuna) or **loaded pre-trained** from a saved model directory. Either way it emits the same artifact: a `predictions.csv` with a `predicted_win_probability` column.
- **adjust** turns raw predictions into the `adjusted_strength` panel used by ratings. It averages progress-weighted win probabilities, measures each player against the game leader, preserves the winner's position, and corrects for civilization effects. Controlled games use a matched Vanilla baseline to remove the start-position confound. The shared calculation lives in `bench/adjust/strength.py`.
- **analyses** are the pluggable modules, grouped into five families: **ratings** (who is stronger), **prediction** (how good is the predictor), **calibration** (are its probabilities honest), **performance** (strength panels, score regressions, trajectories), and **exploratory** (descriptives like token cost). They consume data and return structured results; they do not write files or hardcode where they appear in the report.
- **report** walks the analysis results and renders one section per analysis into deterministic Markdown and HTML.

Stage ordering comes from three places, all resolved into a single topological sort before anything runs: the implicit kind order above, explicit `needs` edges you can add, and automatic edges inferred when one stage `uses` another's output (an estimator's predictions, or a named table like `strength`).

> **No graceful degradation.** Every dependency is installed up front. If a stage needs `torch`, `xgboost`, `optuna`, or R and it is not there, the run fails loudly with an install hint. It is never silently skipped.

---

## Configuring a run

One file, the **benchmark run-spec**, defines an entire run: the input data, which estimators to train or load, which analyses to run, and how to render the report. It is validated on load, so **unknown keys and missing required fields are hard errors**. Typos fail loud instead of silently doing nothing.

```jsonc
{
  "name": "staff-standard-2026",   // names the report and its output subdirectory
  "seed": 42,                       // global RNG seed, for deterministic runs
  "filters":    { ... },            // named, reusable row-filter presets
  "groupings":  { ... },            // named rating-identity dimensions (e.g. "strategy")
  "data":       { ... },            // extraction + canonical table paths + global filter
  "estimators": [ ... ],            // the victory-probability predictors
  "adjust":     [ ... ],            // derived tables (the strength panel)
  "analyses":   [ ... ],            // the modules to run
  "report":     { ... }             // rendering
}
```

Two documents cover the run-spec in depth:

- **[docs/configuration.md](docs/configuration.md)** is a readable, narrative guide: what each block is for, how filters and groupings compose, the two estimator axes, and the edits you make most often.
- **[configs/benchmark.md](configs/benchmark.md)** is the authoritative, field-by-field schema and the numbered validation rules.

---

## What's in the box (module inventory)

Modules are selected by registry name from your config. The implemented **core** set ships enabled in the default template. The full template also lists registry-reserved optional modules with `"enabled": false`; those entries document the intended surface area, and they fail loudly if selected before their implementation lands.

- **estimators:** `naive`, `score`, `baseline`, `xgboost`, `mlp`, `grouped_mlp`, `interaction_mlp`, `attention_mlp` (the eight estimators of the paper, increasing in complexity from a score transform to multi-head self-attention).
- **adjust:** `strength` (the revised-standing / adjusted-strength panel).
- **ratings:** `bradley_terry`, `plackett_luce`, `matchups`, `outcome_matchups` *(implemented core)*; `ablation_bt`, `vanilla_slot_effect` *(reserved optional)*.
  - Per-strategy Elo and bootstrap confidence intervals are **not separate modules**. They are a `group_by` param and a `bootstrap` param on the ordinary Bradley-Terry / Plackett-Luce fit. The convergence ablation from the paper is `ablation_bt`.
- **prediction:** `evaluate`, `compare` *(implemented core)*; `winner_trajectories`, `elo_comparison`, `context_slicing` *(reserved optional)*.
- **calibration:** `reliability`, `loss_by_progress`, `civ_effects`, `cell_baseline`.
- **performance:** `score_ratio`, `strength_panel`, `experiment_completeness`, `turn_predicted` *(implemented core)*; `permutation_importance` *(reserved optional, the construct-validity feature analysis)*.
- **exploratory:** `model_token_costs` *(implemented core)*; `panel`, `turn`, `strategy_profiles` *(reserved optional)*.

> The behavioral analysis family (flavor-change clusters, strategy-pivot and nuke rationale, victory commitment) is **deferred**. It scores no strategist, so it is not built yet. It may return later as an opt-in extension.

---

## Repository layout

```text
civ-bench/
├── AGENTS.md            # agent and repository conventions
├── README.md            # you are here (overview)
├── LICENSE              # MIT
├── docs/                # the human documentation
│   ├── about.md                      # motivation, the paper, design principles
│   ├── getting-started.md            # install, configure, run, and a tutorial
│   ├── configuration.md              # the readable run-spec guide
│   └── development.md                # extending the harness (developer guide)
├── pyproject.toml       # installable package; exposes the `civ-bench` CLI
├── configs/             # the JSON control surface
│   ├── benchmark.md                  # authoritative run-spec schema
│   ├── benchmark.*.template.json     # tracked example run-specs (copy to run)
│   ├── models.json                   # strategist + prediction-model registry
│   ├── experiments.json              # experiment registry + player-type labels
│   └── paths.json                    # data + output roots
├── bench/               # the Python package
│   ├── cli.py                # `civ-bench extract|run|report|fix`
│   ├── pipeline/             # DAG builder + scheduler
│   ├── config/               # load + validate JSON into typed config objects
│   ├── extract/              # game SQLite DBs to canonical CSVs
│   ├── fix/                  # recover malformed game DBs listed in import_issues.csv
│   ├── data/                 # canonical CSV loading + filtering
│   ├── catalog/              # model + experiment catalogs backed by configs/
│   ├── estimators/           # predictor producers: tune / train / load / infer
│   ├── adjust/               # derived tables (the strength panel)
│   ├── analyses/             # the pluggable analysis modules (5 families)
│   ├── stats/                # OLS/logistic regression + heatmap helpers
│   ├── reports/              # assemble analysis artifacts into markdown/html
│   └── plotting/             # shared styles, colors, figure helpers
├── tests/               # pytest suite, one file per area
├── pretrained/          # tracked reference model snapshots (one per model id)
├── plans/               # the roadmap + the staged build plan
├── runs/                # raw game data + canonical CSVs (gitignored)
└── reports/             # generated reports, figures, trained models (gitignored)
```

---

## Running the tests

The test suite lives in [tests/](tests/) at the repo root, one file per area (`test_config.py`, `test_pipeline.py`, `test_extract.py`, and so on). It exercises config validation, DAG resolution, catalog logic, and the analysis modules against small synthetic fixtures. It never touches real game data.

```bash
pip install -e ".[test]"
pytest
```

When you add a module or a validation rule, add or extend its test in the same change.

---

## Where to read next

- **[docs/about.md](docs/about.md)** is the motivation: the problem, the progress-based evaluation idea from the paper, and the design principles.
- **[docs/getting-started.md](docs/getting-started.md)** is the hands-on guide: install, make a config, run the pipeline, read the report, and a tutorial with common recipes.
- **[docs/configuration.md](docs/configuration.md)** is the readable run-spec guide: every block, how filters and groupings compose, and the common edits.
- **[docs/development.md](docs/development.md)** is the developer guide: the plugin contract and how to add an analysis, an estimator, or an adjust module.
- **[AGENTS.md](AGENTS.md)** defines the conventions that apply to every change.
- **[configs/benchmark.md](configs/benchmark.md)** is the complete run-spec schema, field by field, with examples for every module.
- **[plans/](plans/)** contains the roadmap and implementation notes.
- **The CivBench paper** ([arXiv:2604.07733](https://arxiv.org/abs/2604.07733)) is the research this harness implements, including the validity framework, the estimator design, and the full results.

---

## License

Released under the [MIT License](LICENSE). Copyright (c) 2025 John Chen, University of Arizona.
