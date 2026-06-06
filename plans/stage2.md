<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 2 — estimators (LOAD ONLY)

**Goal.** Stand up `bench/estimators/` with the **load path only**: `fit:"pretrained"` loads a saved `model_dir`, re-runs inference on the current `turns` table, and emits `predictions.csv`. No training. This unblocks adjust → analyses → report using the copied pre-trained models.

## Files to create / port  — *done, as built*

- `bench/estimators/registry.py` — port of `../vox-deorum-analysis/models/utils/model_registry.py` (`MODEL_REGISTRY` name → predictor class; `get_model`/`list_models`/`register_model`; `load_model(dir)` dispatches on `metadata.model_class`). The source's `HAS_XGBOOST` try/except soft-fail is dropped — xgboost is imported directly (AGENTS.md: all deps mandatory).
- `bench/estimators/models/` — ported predictor classes: `base_predictor.py`, `base_torch_predictor.py`, and `score`/`naive`/`baseline`/`xgboost`/`mlp`/`grouped_mlp`/`interaction_mlp`/`attention_mlp`. `fit()` is ported too (it rides along with each class) but the load-only path only exercises `_load_model_state()` + `predict_proba()`. Only edits vs. source: relative imports, `baseline`'s feature-default import now from `bench.estimators.features`, and xgboost's direct import.
- `bench/estimators/features.py` — port of the **feature-engineering half** of `models/utils/data_utils.py` (`apply_city_adjustments` → `add_relative_features` → `add_competitive_features`/`add_raw_share_features` → `drop_transformed_columns` → `prepare_features`, plus `FEATURE_GROUPS`/`SELECTED_FEATURES`/`needs_variant_columns`). Adds `build_feature_frame()` (the inference-side `load_and_prepare_base_data`). Training-only pieces (CV splits, resampling/SMOTE) are deferred to stage 6. The `turn_progress` *feature* is the unrounded `turn/max_turn` from `add_relative_features` (the loaders' rounded column is overwritten), matching the source byte-for-byte.
- `bench/estimators/runner.py` — `run_estimator(cfg, stage_raw, catalog=None)` resolves a `pretrained` block (via `cfg.output.resolve(model_dir)`), loads, builds the feature frame, narrows to `predict_subset` (`all`/`non_llm`/`llm`/`{experiments:[…]}`), strips ID columns per `REQUIRES_ID_COLUMNS`, and writes `<root>/estimators/<id>/predictions.csv` (the `PREDICTION_OUTPUT_COLUMNS`). `EstimatorResult` carries id/model/path/n_rows/model_dir.
- `bench/cli.py` — `civ-bench run` now executes the implemented prefix of the resolved DAG (`extract` + `estimators`) honouring `--only`/`--skip` (`_resolve_subset` keeps a node + its transitive deps); on reaching an unimplemented kind (adjust/analyses/report) it stops loudly with a stage-N pointer. The estimators package is imported lazily (it pulls torch/xgboost) so dry-runs stay light.
- `fit:"train"` and a `tune` block **raise NotImplementedError** (stage 6 fills them).

## Tracked pretrained store — `pretrained/<model_id>/` (committed)

The pre-trained snapshots live in a **tracked** top-level `pretrained/` dir — one per
`prediction_models` id — so the repo ships runnable load-only models (no manual copy).
They were copied verbatim (`metadata.json` + state files) from
`D:\Cache\…\colm-2026\analysis\models\output\<class>_model\`:

| `model_id` (`pretrained/<id>/`) | source dir | state file(s) |
|---|---|---|
| `naive` | `naive_model/` | `model.json` |
| `score` | `score_model/` | `model.json` |
| `baseline` | `baseline_model/` | `model.pkl` |
| `xgboost` | `xgboost_model/` | `model.pkl` |
| `mlp` | `mlp_model/` | `model_state.pt` + `torch_state.json` |
| `grouped_mlp` | `grouped_mlp_model/` | `model_state.pt` + `torch_state.json` |
| `interaction_mlp` | `interaction_mlp_model/` | `model_state.pt` + `torch_state.json` |
| `attention_mlp` | `attention_mlp_model/` | `model_state.pt` + `torch_state.json` |

Keyed by **model id** (so `attention`'s estimator points at `pretrained/attention_mlp/`). ~1.6 MB total.
The `reports-cross/estimators/<id>/` group is produced by stage 6, not stored here.

## Config wiring

The tracked example is `configs/benchmark.pretrained.template.json`; run a local copy (e.g.
`configs/benchmark.dev.json`): estimators with `fit:"pretrained"`, `predict:"in_sample"`,
`pretrained.model_dir` → `pretrained/<model_id>/`; `extract.enabled:false` (the template) or
`true` with a machine `runs_dir` (the dev config).

**`model_dir` is an INPUT read as-authored — it is *not* re-rooted by `output.suffix`** (only
save-paths are; benchmark.md §2.1). So the same tracked `pretrained/` store serves the default
run *and* the `-dev`/`-cross` variants: `benchmark.dev.json` (suffix `-dev`) loads
`pretrained/score/` while still writing predictions to `reports-dev/estimators/score/`.

`fit:"pretrained"` is **`model_dir`-only** — we load saved weights and re-infer, never read a raw
prediction CSV as an estimator artifact (so column-name reconciliation of old prediction files is a
non-issue). Honest cross-val/OOF predictions and the non-LLM **cross** split are deferred to stage
6's `fit:"train"` pipeline (`train_subset:"non_llm"` → `reports-cross/`); we don't keep a saved cross model.

## Done

`civ-bench run --config configs/benchmark.dev.json --only <estimator>` emits a validated
`predictions.csv` (input rows + `predicted_win_probability`) for each core estimator — extract
skips when fresh, the model loads from the tracked `pretrained/` store, and output lands under
`reports-dev/`. No training run.

## Verification — *done*

- The load→infer path is **byte-faithful to the source's `evaluate_model.py --inference-only`** (same load_and_prepare → prepare_features → strip-ids → predict_proba). Re-inferring the copied `score`/`attention`/`xgboost` models on the original `colm-2026/turn_data.csv` and comparing to the source's own inference-only output: `score` and `xgboost` match to `≤1.1e-16`, `attention` to `≤3e-8` (float32 GPU/CPU nondeterminism — well within tolerance).
- **Caveat re: the checked-in `output/<model>_predictions.csv`.** For the deterministic `score` model, re-inference reproduces the copied CSV exactly (`1.1e-16`). For the *trained* models (`attention`/`xgboost`) the copied CSVs are **stale relative to the saved model snapshots** (they were generated by a different fit), so they do **not** match a load of the current `*_model/` dir — the saved-model re-inference is correct, the copied CSV is just from another run. The right invariant is "a saved model reproduces *its own* inference," which holds; we do not treat the legacy prediction files as ground truth (consistent with stage 2's "`model_dir`-only, never read a raw prediction CSV as an estimator artifact").
- A `pretrained` block with a missing/inconsistent `model_dir` fails loudly (`FileNotFoundError` on a missing `metadata.json`; `ValueError` on a `model_class` not in the registry) — surfaced by `civ-bench run` as a non-zero exit, and covered by `tests/test_estimators.py`.

## Tests

`tests/test_estimators.py` (synthetic turns fixture, no machine-specific roots): registry `load_model` dispatch + error cases, the feature pipeline, `run_estimator` end-to-end (validated `predictions.csv` columns + per-(game,turn) softmax sum), saved-model-reproduces-direct-inference, `predict_subset` narrowing, and `fit:train`/`tune` → `NotImplementedError`.

## One-time copy — *done*

Copied `score_model/` → `reports/estimators/score/model/`, `attention_mlp_model/` → `reports/estimators/attention/model/`, `xgboost_model/` → `reports/estimators/xgboost/model/` (gitignored `reports/`). A local `configs/benchmark.local.json` (copy of the pretrained template) drives `civ-bench run --config configs/benchmark.local.json --only <estimator>`.
