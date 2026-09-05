<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 6: train / tune (and the cross variant)

**Goal.** Fill in the estimator producer's `fit:"train"` path (with the optional `tune` Optuna pre-step) behind the same config block, then bring up the train-based `benchmark.template.json` AND the **cross variant** (estimator trained on non-LLM seats → `reports-cross/`).

## Files created / ported: *done, as built*

- `bench/estimators/training.py`: the train/eval pipeline ported from `../vox-deorum-analysis/models/utils/model_evaluator.py` (`run_full_prediction` → `run_full_train`; `run_kfold_evaluation` → `run_cross_val`) plus the resampling / GroupKFold helpers from `models/utils/data_utils.py`. The CLI/notebook `print(...)`-as-output style is stripped: everything returns a `TrainResult` (`predictions`, `model`, `feature_importance`, `selected_features`). Per-model `FILTER_ZERO_SCORE` / `DISABLE_RESAMPLING` / `REQUIRES_ID_COLUMNS` are honoured exactly as the source.
- `bench/estimators/tuning.py`: the Optuna search ported from `models/tune_model.py`: the per-model `SEARCH_SPACES`, the feature-variant families (`suggest_feature_variants` / `reconstruct_include_features` / `_reverse_map_features`), `convert_best_params`, and the CV objective with the source's per-fold overfitting penalty. `run_tune` runs a fresh study; `load_best_params` reuses a saved `best_params.json`. Single-scalar `objective`; single-process (`n_jobs:1`): the multi-process Colab worker is dropped (`tune_colab.ipynb` is obsolete).
- `bench/estimators/runner.py`: `run_estimator` now dispatches `fit:"train"` to `_run_train` (the stage-2 load path is `_run_pretrained`, unchanged). No separate CLIs/notebooks.

## Config wiring: *as built*

- `predict: in_sample | cross_val`. `in_sample` fits one model on `train.train_subset` and predicts the entry's `predict_subset` (the deployed model is saved to `train.save_model`); `cross_val` emits honest out-of-fold predictions over the whole frame (each row scored by the fold that held its game out) plus an aggregated `feature_importance.csv` next to the predictions when `train.save_importance` is set. `predict_subset` is ignored for `cross_val` (benchmark.md §4.4).
- `train.train_subset`: `all` | `llm` | `non_llm` | `{experiments:[...]}`, resolved to a concrete experiment list against the loaded frame (`_subset_experiment_list`). **`non_llm` is the cross split**: train on the Vanilla/Null experiments, predict everyone. For `in_sample` it narrows the training frame; for `cross_val` it narrows each fold's *training* indices while OOF coverage stays the whole frame (the source's `train_non_llm_only` generalization setup).
- `tune`: `search` (`hyperparameters` → params / `features` → variants / `both`), `n_trials`, `objective` (single scalar: `brier_score`/`log_loss` minimized, `roc_auc`/`balanced_accuracy` maximized), `n_splits`, `resample`, `storage`, `save_params`, `load_params`. **Hyperparameter precedence** (highest wins): explicit `params` → `load_params` → a fresh `save_params` search → the model class's coded defaults. `load_params` and `tune.enabled:false` both skip the search.
- `save_model` / `save_importance` / `tune.save_params` / `tune.storage` resolve under the output root (`cfg.output.resolve`). `tune.load_params` is an **input** read as-authored (a pre-trained hyperparameter set), like `pretrained.model_dir`.
- **Cross variant:** `configs/benchmark.cross.template.json`, a tracked copy of `benchmark.template.json` with `output.suffix:"-cross"` (→ `reports-cross/`) and every estimator's `train.train_subset:"non_llm"`. The variant is config, not code (no separate "cross" estimator kind).
- Determinism: the top-level `seed` threads into the GroupKFold split order, the resamplers, the Optuna `TPESampler` seed, and each model's torch/xgboost init.
- **Validator fix:** `features.include`/`features.exclude` may be `null` (= "unset", fall back to the model's coded `DEFAULT_FEATURES`); only a present, non-null value must be a string list. The train-based `benchmark.template.json` relies on this (`attention.features.include: null`).

## Done

- `civ-bench run --config <local copy of configs/benchmark.template.json>` trains (+ tunes) and reports → `reports/`.
- The cross run (`configs/benchmark.cross.template.json` copy) trains on non-LLM, predicts all, and reports → `reports-cross/` (configurable suffix).

## Verification: *done*

- **Port fidelity.** A fresh in-sample `train` of `score` (default exponent 4.236) on the real `runs/turn_data.csv` (631 528 rows / 192 games) reproduces a `fit:"pretrained"` load of `pretrained/score/` **exactly** (max abs diff `0.0`); the train pipeline matches the reference inference. The `xgboost` `cross_val` path emits 631 528 OOF rows + a 3-column aggregated `feature_importance.csv`.
- **Determinism.** Re-running an identical `score` train config is byte-stable (verified on real data and in `tests/test_estimators.py::test_train_is_byte_stable`).
- **`load_params`** skips the search and reuses a saved `best_params.json`; **explicit `params` beat `load_params`** (precedence test).

## Tests: *done*

`tests/test_estimators.py` adds: in-sample train writes + saves a reloadable model dir; train reproduces the pretrained inference (miniature); byte-stability; `train_subset` narrows training while `predict_subset:"all"` predicts everyone; empty `train_subset` raises; `cross_val` OOF covers every row + writes feature importance; a real (tiny) Optuna search runs and writes `best_params.json`; `load_params` reuse; explicit-params-over-`load_params` precedence. `tests/test_config.py` adds: the default train template and the cross template both load (incl. `features.include: null`) and the cross template redirects to `reports-cross/` with `train_subset:"non_llm"`.
