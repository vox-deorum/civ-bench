# civ-bench — build plan

A modular, JSON-configurable Python **benchmark harness** for LLM strategists in
*Civilization V: Vox Populi* (via the Vox Deorum platform). Point it at raw game-run data, hand it
a config, and it produces a report — extraction, analysis, and rendering all driven by JSON.

> **This is a greenfield rewrite of `../vox-deorum-analysis`.** That repo is a *paper repo*: a flat
> pile of one-off scripts, notebooks, and checked-in result artifacts wired to a single fixed
> dataset. `civ-bench` keeps its proven analysis logic but restructures it into an installable
> package where **everything experiment-specific is config, not code**, and reports are generated
> automatically rather than hand-assembled in notebooks.

This document is the *roadmap*. For the rules every change must follow, read [AGENTS.md](../AGENTS.md).
For the run-spec schema, read [configs/benchmark.md](../configs/benchmark.md). For the **ordered,
stage-by-stage build plan** (which this roadmap is refined into), read [stages.md](stages.md)
and `stage0.md … stage6.md`. The `plans/` files are **live documentation** — keep them in
parity with the code as you implement (update the relevant stage file in the same change).

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
  `calibration.*`, `performance.*`, `exploratory.*`. (`behavior.*` is deferred.)
- **report** — assembles the produced artifacts into markdown/html.

See [AGENTS.md](../AGENTS.md#the-pipeline-model) for the stage model and [configs/benchmark.md](../configs/benchmark.md)
for how dependencies and pre-trained estimators are expressed.

## Migration map: what comes from `vox-deorum-analysis`

Port the **logic**, drop the **flatness and the data**. Rough correspondence:

| Old (`vox-deorum-analysis`)              | New (`civ_bench/…`)                          | Notes |
|---|---|---|
| `extract/` (`python -m extract`)         | `civ_bench/extract/`                         | Keep the DB-discovery + turn/panel/token extractors; drive roots from `configs/paths.json`. |
| `shared/data_loading.py`                 | `civ_bench/data/`                            | `load_turn_data` / `load_panel_data` + filters. |
| `shared/model_catalog.py`, `experiments.py` | `civ_bench/catalog/`                      | Config-backed catalogs; preserve alias normalization + seat expansion. |
| `shared/config/*.json`                   | `configs/*.json`                             | Same schema; now the primary control surface, not a side file. |
| `shared/plot_styles.py`, `plot_utilities.py` | `civ_bench/plotting/` | Keep style/color logic; trim notebook-only helpers. |
| `shared/regression_utilities.py` | `civ_bench/stats/` | The **statistics** layer (OLS/logistic regression wrappers, clustered/weighted fits, coefficient/odds-ratio heatmaps). Imported by `performance.score_ratio`, `ratings.matchups` (`validate_ols`), and `adjust/strength.py`'s civ-adjustment — it is real analysis code, **not** a plotting helper, so it does not belong under `plotting/`. |
| `models/` (compare/evaluate/tune + registry) | `civ_bench/estimators/`                  | Predictor registry pattern is already good — fold tune/train/load/infer behind the estimator config block. |
| `performance/turn_predicted.ipynb` → `prepare_strength_data` (also copied in `ratings/iterative_bt.py`) | `civ_bench/adjust/strength.py` | The strength-panel derivation: estimator `predicted_win_probability` → `adjusted_strength`. It is its own **`adjust` stage**, not an analysis, because every `ratings.*` consumes its `strength` table. |
| `predict/` (loader + calibration/comparison notebooks) | `civ_bench/analyses/prediction/` + `…/calibration/` | Scoring views (`evaluate`/`compare`) become `prediction.*`; the reliability + loss-by-progress views become the `calibration.*` family (§ module inventory). |
| `ratings/` (BT, PL, strategy-Elo, matchups, bootstrap; R+py) | `civ_bench/analyses/ratings/`       | Keep R interop where it exists; wrap each as an Analysis rating the `adjust` stage's `strength` table (`uses.tables: ["strength"]`), **not** raw `panel_data`. **strategy-Elo and bootstrap are NOT separate modules**: strategy-Elo is `group_by: ["player_type","strategy"]` (it already just swaps BT's identity column and reuses `calculate_ratings()`); bootstrap is a shared resample-and-refit `bootstrap` param. **R discovery must be cross-platform**: when porting `bradley_terry.py`/`plackett_luce.py`, find `Rscript` via `PATH` and an override env var (`CIV_BENCH_RSCRIPT`) — drop the old `_find_rscript()` hardcoded `C:`/`D:\Program Files\R` scan, which silently fails on Linux/macOS. |
| `performance/`, `exploratory/` notebooks | `civ_bench/analyses/{performance,exploratory}/` + report templates | Convert notebook narratives into generated report sections. **Orphans resolved:** `exploratory/group_permutation_importance.ipynb` is a duplicate of `performance/group_permutation_importance.ipynb` → keep only `performance.permutation_importance`; if the exploratory narrative genuinely differs it is a descriptive re-cut and is dropped, not re-added as a module. `exploratory/panel_exploratory_vanilla.ipynb` (vanilla-only cut, `condition_include=["observe-vanilla-standard"]`) has **no new module**: it is `exploratory.panel` run under a vanilla `filter` — config, not code. |
| `behaviors/` notebooks | **deferred** | The behavioral family scores no strategist; not ported now. Revisit as an opt-in extension later. The derived CSVs `flavor_change_decomposition.ipynb` writes (`atom_dictionary.csv`, `multi_flavor_events.csv`) are **read by nothing else** (grep-confirmed), so deferring the whole family drops no input any retained stage depends on. |
| checked-in CSVs, `*/output/`, `ratings/output/`, `behaviors/nuke/` | **not ported** | Raw data goes in `runs/` (gitignored); results regenerate into `reports/`. |
| `template.md` + `sync-template.sh` + template branch | **obsolete** | The whole repo is now reusable by design; no template branch needed. |
| `models/tune_colab.ipynb` (Colab tuning) | **obsolete** | Tuning is the estimator `tune` block ([configs/benchmark.md](../configs/benchmark.md) §4.3); no separate Colab notebook. |

When porting a module, do **not** paste it verbatim. Strip the hardcoded paths, the
`print(...)`-as-output style, and any single-dataset assumptions; route those through config and
return values instead.

## Module inventory

Every module the harness will include, grouped by stage kind. The registry name in the left column
is what `configs/benchmark.json` selects. (See [configs/benchmark.md](../configs/benchmark.md) for the
per-module params.) Modules are split into a **core** set (implemented, in the default
`benchmark.json`) and an **optional** set (registry-reserved, `"enabled": false`, shipped only in
`benchmark.full.json`). See *Core vs. optional* below.

- **extract** — `extract`
- **estimators** (`civ_bench/estimators/`, selected by `model` id): `naive`, `score`, `baseline`,
  `xgboost`, `mlp`, `grouped_mlp`, `interaction_mlp`, `attention_mlp`
- **adjust** (`civ_bench/adjust/`, derived tables): `strength`
- **ratings** — *core:* `ratings.bradley_terry`, `ratings.plackett_luce`, `ratings.matchups`.
  *optional:* `ratings.ablation_bt`, `ratings.vanilla_slot_effect`.
  BT/PL take a `group_by` param (default `["player_type"]`; `["player_type","strategy"]` =
  per-strategy Elo) and a `bootstrap` param (CIs) — so the former `strategy_elo`/`bootstrap_bt`
  modules are folded in, not separate.
- **prediction** (scoring) — *core:* `prediction.evaluate`, `prediction.compare`.
  *optional:* `prediction.winner_trajectories`, `prediction.elo_comparison`,
  `prediction.context_slicing`.
- **calibration** (calibration of the estimator) — *core:* `calibration.reliability`,
  `calibration.loss_by_progress`.
- **performance** — *core:* `performance.score_ratio`, `performance.strength_panel`,
  `performance.turn_predicted`. *optional:* `performance.permutation_importance`.
- **exploratory** — *core:* `exploratory.model_token_costs`. *optional:* `exploratory.panel`,
  `exploratory.turn`, `exploratory.strategy_profiles`.
- **report** — `report`

> **`behavior.*` is deferred** — the whole behavioral family (flavor-change, pivot/nuke rationale,
> victory commitment) scores no strategist and is not in this inventory. Revisit as an opt-in
> extension later.

### Core vs. optional

The shipped default `configs/benchmark.json` is the **lean core** (~10 modules): it answers the two
benchmark questions — *which strategist is stronger* (ratings) and *how good is the win predictor*
(prediction + calibration) — plus the strength panel that feeds ratings and a cost-efficiency axis.
The kitchen-sink `configs/benchmark.full.json` adds every optional module with `"enabled": false`,
so opting in is a one-flag config edit, never a code change (invariant 1). New rating *slices*
(per-strategy, and later per-stage) arrive as `groupings` catalog entries + a `group_by` value, also
config-only. **Deferred (not built now):** the optional modules above, the `behavior.*` family, and
`groupings` dimensions beyond `strategy` (notably a per-game-stage `kind: "bucket"`, which also needs
the `adjust`/`strength` stage to emit per-stage strength).

## Status

Greenfield. The repo is an empty checkout; this document plus [AGENTS.md](../AGENTS.md),
[configs/benchmark.md](../configs/benchmark.md), and the staged [plans/](stages.md) are the build plan.

**Build order — load-only first.** We bring up the pipeline using **pre-trained estimators loaded
from copied model dirs** (zero training), then implement training/tuning last. The authoritative
ordering lives in [stages.md](stages.md); in brief:

0. **Scaffold** — `configs/` (port `models.json`/`experiments.json`/`paths.json`), `civ_bench/config/`
   (load + validate, resolve the output root §2.1), `civ_bench/pipeline/` (DAG), and the shared
   `data`/`catalog`/`stats`/`plotting` infra. ([stage0.md](stage0.md))
1. **extract** → canonical CSVs. ([stage1.md](stage1.md))
2. **estimators (load only)** — `fit:"pretrained"` loads a copied `model_dir` and infers; no training
   yet (`configs/benchmark.pretrained.json`). ([stage2.md](stage2.md))
3. **adjust** — the `strength` panel. ([stage3.md](stage3.md))
4. **analyses** — the core modules. ([stage4.md](stage4.md))
5. **report** — full pipeline end-to-end with no training. ([stage5.md](stage5.md))
6. **train / tune** — implement `fit:"train"`+`tune`; bring up the train-based `benchmark.json` and
   the **cross variant** (estimator trained on non-LLM seats → `reports-cross/`, configurable suffix
   §2.1). ([stage6.md](stage6.md))
