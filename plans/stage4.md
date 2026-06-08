<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 4 — analyses

**Goal.** Implement the pluggable analysis layer and the **core** modules. Each is a self-contained unit behind the `Analysis` interface + registry; JSON selects it by name; it returns an `AnalysisResult` (tables + figures + summary), never side effects.

## Files to create / port

- `bench/analyses/base.py` — `Analysis` interface + registry (name → class) and `AnalysisResult`.
- **ratings** (`analyses/ratings/`, consume `strength` via `uses.tables:["strength"]`): `ratings.bradley_terry`, `ratings.plackett_luce`, `ratings.matchups`. Port from `../vox-deorum-analysis/ratings/` (keep R interop; `Rscript` via `PATH`/`CIV_BENCH_RSCRIPT`). **strategy-Elo and bootstrap are params, not modules:** `group_by` (default `["player_type"]`; `["player_type","strategy"]` = per-strategy) and `bootstrap` (shared resample-and-refit). **When the `strength` table uses a controlled-design `block` adjustment (stage3), the `bootstrap` resample must re-run the `adjust/strength` fit inside each replicate** (resample games upstream of the strength fit, not a fixed panel) so the start-cell baseline's uncertainty is reflected in the CIs.
- **prediction** (scoring, `analyses/prediction/`): `prediction.evaluate` (multi-metric table), `prediction.compare`. Port from `predict/` loader + comparison logic.
- **calibration** (`analyses/calibration/`): `calibration.reliability`, `calibration.loss_by_progress`.
- **performance** (`analyses/performance/`): `performance.score_ratio` (uses `stats/`), `performance.strength_panel` (consumes `strength`), `performance.turn_predicted` (uses an estimator). **`performance.strength_panel` always reports per-identity `n_games` + bootstrap CI *and* flags identities below `min_games_preliminary` (default = ratings `min_games`) as preliminary; it also renders the adjust stage's civilization-level effect (`civ_effects.csv`), VPAI seating×seed effect (`cell_baseline.csv`, both explicit+implicit pathways, stage3 §5.1), and the controlled-design cell-coverage report (`cell_coverage.csv` — which `(seed, player_id)` cells of the entirety each controlled experiment is missing).**
- **exploratory** (`analyses/exploratory/`): `exploratory.model_token_costs` (tokens table + pricing).
- Optional modules (ablation_bt, vanilla_slot_effect, winner_trajectories, elo_comparison, context_slicing, permutation_importance, panel, turn, strategy_profiles) are **registry-reserved, `enabled:false`**, shipped only in `benchmark.full.template.json`. `behavior.*` is deferred (not built). **`ratings.vanilla_slot_effect` doubles as the controlled-design validation**: with stage3's start-cell adjustment on, the seat/start-position effect should be significant on the raw panel and ~null on the adjusted panel.

## Config wiring (`analyses[]`)

Common envelope: `id`, `module`, `uses.{estimators,tables}`, `filter` (narrows the global filter), `params`. `ratings.*` must reference a `strength` table; metrics live in `prediction.evaluate`, not in the estimator.

## Done

Every core analysis runs and returns a populated `AnalysisResult`; unknown `params` are validation errors; `group_by:["player_type","strategy"]` produces per-strategy ratings via the `strategy` grouping.

## Verification

- Each core module's tables match the old notebook/script output on the same inputs (within tolerance).
- R-backed ratings run on a machine where `Rscript` is only on `PATH` (no hardcoded path).
