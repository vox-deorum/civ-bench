<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 2 — estimators (LOAD ONLY)

**Goal.** Stand up `civ_bench/estimators/` with the **load path only**: `fit:"pretrained"` loads a
saved `model_dir`, re-runs inference on the current `turns` table, and emits `predictions.csv`. No
training. This unblocks adjust → analyses → report using the copied pre-trained models.

## Files to create / port

- `civ_bench/estimators/registry.py` — port `../vox-deorum-analysis/models/utils/model_registry.py`
  (name → predictor class; `load_model(dir)` dispatches on `metadata.model_class`).
- `civ_bench/estimators/models/` — port the predictor classes (`score`, `baseline`, `xgboost`, `mlp`,
  `grouped_mlp`, `interaction_mlp`, `attention_mlp`, `naive`) from `models/models/`; for load-only we
  need each class's `_load_model_state()` + `predict_proba()` (training methods can land in stage 6).
- Feature pipeline — port `models/utils/data_utils.py` (feature engineering) so re-inference rebuilds
  the exact feature matrix the saved `metadata.selected_features` expects.
- `civ_bench/estimators/runner.py` (or similar) — resolve a `pretrained` block, load, infer on
  `predict_subset`, write `<root>/estimators/<id>/predictions.csv`.
- `fit:"train"` and `tune` are present but **raise NotImplementedError** (stage 6 fills them).

## One-time copy / reorg (normal group → `reports/`)

From `D:\Cache\…\colm-2026\analysis\models\output\`:

| `id` | source dir | → target |
|---|---|---|
| `score` | `score_model/` | `reports/estimators/score/model/` |
| `attention` | `attention_mlp_model/` | `reports/estimators/attention/model/` |
| `xgboost` | `xgboost_model/` | `reports/estimators/xgboost/model/` |

(Copy `metadata.json` + state files verbatim. naive/baseline/mlp/grouped/interaction copy the same
way when wired. The `reports-cross/estimators/<id>/` group is produced by stage 6, not copied.)

## Config wiring

Author `configs/benchmark.pretrained.json`: estimators `score`/`attention`/`xgboost` with
`fit:"pretrained"`, `predict:"in_sample"`, `pretrained.model_dir` pointing at the copied dirs;
`extract.enabled:false`.

`fit:"pretrained"` is **`model_dir`-only** — we load saved weights and re-infer, never read a raw
prediction CSV as an estimator artifact (so column-name reconciliation of old prediction files is a
non-issue). Honest cross-val/OOF predictions and the non-LLM **cross** split are deferred to stage 6's
`fit:"train"` pipeline (`train_subset:"non_llm"` → `reports-cross/`); we don't keep a saved cross model.

## Done

`civ-bench run --config configs/benchmark.pretrained.json --only <estimator>` emits a validated
`predictions.csv` (input rows + `predicted_win_probability`) for each core estimator — no training run.

## Verification

- Loaded-model predictions match the copied `output/<model>_predictions.csv` within float tolerance.
- A `pretrained` block with a missing/inconsistent `model_dir` fails loudly.
