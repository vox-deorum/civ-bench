# civ-bench — build plan

A modular, JSON-configurable Python **benchmark harness** for LLM strategists in
*Civilization V: Vox Populi* (via the Vox Deorum platform). Point it at raw game-run data, hand it
a config, and it produces a report — extraction, analysis, and rendering all driven by JSON.

> **This is a greenfield rewrite of `../vox-deorum-analysis`.** That repo is a *paper repo*: a flat
> pile of one-off scripts, notebooks, and checked-in result artifacts wired to a single fixed
> dataset. `civ-bench` keeps its proven analysis logic but restructures it into an installable
> package where **everything experiment-specific is config, not code**, and reports are generated
> automatically rather than hand-assembled in notebooks.

This document is the *roadmap*. For the rules every change must follow, read [AGENTS.md](AGENTS.md).
For the run-spec schema, read [configs/benchmark.md](configs/benchmark.md).

## End-to-end flow

The package is `civ_bench/`. The pipeline is a configurable DAG of five stage kinds, all driven by
`configs/benchmark.json`:

```
raw game DBs ──▶ extract ──▶ estimators ──▶ adjust ──▶ analyses ──▶ report
                  │             │             │           │            │
              canonical     train OR      strength    registry +   templated
                CSVs       pre-trained      panel      configs/    markdown/html
                           predictors    (adjusted-   benchmark    → reports/
                                          strength)
```

- **extract** — game SQLite DBs → `turn_data` / `panel_data` / `game_timestamps` /
  `model_token_usage` CSVs.
- **estimators** — victory-probability predictors, either trained (optionally tuned) on the current
  data or loaded **pre-trained**, then run to emit `predicted_win_probability`.
- **adjust** — turns an estimator's per-turn win-probabilities into the per-player-game **strength
  panel** (`adjusted_strength`), registered as a named `strength` table. This is the bridge the
  rating models sit on; it makes `ratings.*` depend (transitively) on an estimator.
- **analyses** — pluggable modules: `ratings.*` (rate the `strength` table), `prediction.*`,
  `performance.*`, `behavior.*`, `exploratory.*`.
- **report** — assembles the produced artifacts into markdown/html.

See [AGENTS.md](AGENTS.md#the-pipeline-model) for the stage model and [configs/benchmark.md](configs/benchmark.md)
for how dependencies and pre-trained estimators are expressed.

## Migration map: what comes from `vox-deorum-analysis`

Port the **logic**, drop the **flatness and the data**. Rough correspondence:

| Old (`vox-deorum-analysis`)              | New (`civ_bench/…`)                          | Notes |
|---|---|---|
| `extract/` (`python -m extract`)         | `civ_bench/extract/`                         | Keep the DB-discovery + turn/panel/token extractors; drive roots from `configs/paths.json`. |
| `shared/data_loading.py`                 | `civ_bench/data/`                            | `load_turn_data` / `load_panel_data` + filters. |
| `shared/model_catalog.py`, `experiments.py` | `civ_bench/catalog/`                      | Config-backed catalogs; preserve alias normalization + seat expansion. |
| `shared/config/*.json`                   | `configs/*.json`                             | Same schema; now the primary control surface, not a side file. |
| `shared/plot_styles.py`, `plot_utilities.py`, `regression_utilities.py` | `civ_bench/plotting/` | Keep style/color logic; trim notebook-only helpers. |
| `models/` (compare/evaluate/tune + registry) | `civ_bench/estimators/`                  | Predictor registry pattern is already good — fold tune/train/load/infer behind the estimator config block. |
| `performance/turn_predicted.ipynb` → `prepare_strength_data` (also copied in `ratings/iterative_bt.py`) | `civ_bench/adjust/strength.py` | The strength-panel derivation: estimator `predicted_win_probability` → `adjusted_strength`. It is its own **`adjust` stage**, not an analysis, because every `ratings.*` consumes its `strength` table. |
| `predict/` (loader + calibration/comparison notebooks) | `civ_bench/analyses/prediction/` | Convert the visualize-* notebooks into generated report sections. |
| `ratings/` (BT, PL, strategy-Elo, matchups; R+py) | `civ_bench/analyses/ratings/`           | Keep R interop where it exists; wrap each as an Analysis. Each rates the `adjust` stage's `strength` table (`uses.tables: ["strength"]`), **not** raw `panel_data`. |
| `performance/`, `exploratory/`, `behaviors/` notebooks | `civ_bench/analyses/{performance,behavior}/` + report templates | Convert notebook narratives into generated report sections. |
| checked-in CSVs, `*/output/`, `ratings/output/`, `behaviors/nuke/` | **not ported** | Raw data goes in `runs/` (gitignored); results regenerate into `reports/`. |
| `template.md` + `sync-template.sh` + template branch | **obsolete** | The whole repo is now reusable by design; no template branch needed. |

When porting a module, do **not** paste it verbatim. Strip the hardcoded paths, the
`print(...)`-as-output style, and any single-dataset assumptions; route those through config and
return values instead.

## Module inventory

Every module the harness will include, grouped by stage kind. The registry name in the left column
is what `configs/benchmark.json` selects. (See [configs/benchmark.md](configs/benchmark.md) for the
per-module params.)

- **extract** — `extract`
- **estimators** (`civ_bench/estimators/`, selected by `model` id): `naive`, `score`, `baseline`,
  `xgboost`, `mlp`, `grouped_mlp`, `interaction_mlp`, `attention_mlp`
- **adjust** (`civ_bench/adjust/`, derived tables): `strength`
- **ratings** — `ratings.bradley_terry`, `ratings.plackett_luce`,
  `ratings.strategy_elo`, `ratings.matchups`, `ratings.bootstrap_bt`, `ratings.iterative_bt`,
  `ratings.vanilla_slot_effect`
- **prediction** (estimator-facing analyses) — `prediction.evaluate`, `prediction.compare`,
  `prediction.calibration`, `prediction.loss_by_progress`, `prediction.winner_trajectories`,
  `prediction.elo_comparison`, `prediction.context_slicing`
- **performance** — `performance.score_ratio`, `performance.strength_panel`,
  `performance.turn_predicted`, `performance.permutation_importance`
- **behavior** — `behavior.flavor_change_clusters`, `behavior.flavor_change_decomposition`,
  `behavior.pivot_rationale`, `behavior.victory_commitment`, `behavior.nuke_flavor_rationale`
- **exploratory** — `exploratory.panel`, `exploratory.turn`, `exploratory.strategy_profiles`,
  `exploratory.model_token_costs`
- **report** — `report`

## Status

Greenfield. The repo is an empty checkout; this document plus [AGENTS.md](AGENTS.md) and
[configs/benchmark.md](configs/benchmark.md) are the build plan.

**Where to start:**

1. Stand up `configs/` (port `models.json`, `experiments.json`, `paths.json`; write the first
   `benchmark.json`).
2. Build `civ_bench/config/` (load + validate) and `civ_bench/pipeline/` (resolve `needs`,
   topo-sort, run stages).
3. Port `civ_bench/extract/` and `civ_bench/data/`.
4. Port one estimator end-to-end (`score` is the cleanest) through `civ_bench/estimators/`.
5. Port the `adjust` stage (`strength`, from `turn_predicted`'s `prepare_strength_data`) so the
   estimator's predictions become the `adjusted_strength` panel.
6. Port one analysis end-to-end (`ratings.bradley_terry`, consuming the `strength` table) through to
   a generated report — this proves the full extract → estimator → adjust → analysis → report
   pipeline before migrating the rest.
