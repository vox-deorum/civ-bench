<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 6 — train / tune (and the cross variant)

**Goal.** Fill in the estimator producer's `fit:"train"` path (with the optional `tune` Optuna
pre-step) behind the same config block, then bring up the train-based `benchmark.json` AND the **cross
variant** (estimator trained on non-LLM seats → `reports-cross/`).

## Files to create / port

- `civ_bench/estimators/` — implement the stubs from stage 2: port the train/eval logic from
  `../vox-deorum-analysis/models/evaluate_model.py` (full-train + cross-val) and the Optuna search from
  `models/tune_model.py`. Fold them behind the estimator config — **no separate CLIs/notebooks**
  (`tune_colab.ipynb` is obsolete).

## Config wiring

- `predict: in_sample | cross_val` (cross_val emits honest out-of-fold predictions + feature importance).
- `train.train_subset`: `all` | `llm` | `non_llm` | `{experiments:[...]}`. **`non_llm` is the cross
  split** (train on Vanilla/Null, predict on all).
- `tune`: `search`, `n_trials`, `objective` (single scalar), `n_splits`, `storage`, `save_params`,
  `load_params`. Hyperparameter precedence: `params → load_params → save_params → coded defaults`.
- `save_model`/`save_importance` resolve under the output root.
- **Cross variant:** a config (e.g. `configs/benchmark.cross.json` or a suffix overlay on
  `benchmark.json`) sets the output suffix → `reports-cross/` and estimator `train_subset:"non_llm"`.
- Determinism: thread top-level `seed` into CV splits, torch init, resampling, bootstrap.

## Done

- `civ-bench run --config configs/benchmark.json` trains (+ tunes) and reports → `reports/`.
- The cross run trains on non-LLM, predicts all, and reports → `reports-cross/` (configurable suffix).

## Verification

- A fresh in-sample `train` of `score`/`xgboost` reproduces the copied normal `predictions.csv` within
  tolerance (sanity that the port matches the reference run).
- Re-running an identical config is byte-stable (determinism via `seed`).
- `load_params` skips the search and reuses a saved `best_params.json`.
