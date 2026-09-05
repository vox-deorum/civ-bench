<!-- PARITY: This folder is the build plan for civ-bench. It is LIVE documentation, not a
     historical record. Any agent that lands code or config MUST update the relevant stage file (and this plan) in the SAME change so the plans stay in parity with the repo. A stage file that disagrees with the code is a bug. -->

# civ-bench — staged build plan

This folder is the **executable roadmap** for building `civ-bench`. It refines the *what* in [plan.md](plan.md) into an ordered *how*, and it must obey the *rules* in [../AGENTS.md](../AGENTS.md) and the *schema* in [../configs/benchmark.md](../configs/benchmark.md).

- **Main plan** — this file: goals, invariants, build order, the load-only bootstrap, the two-group ("cross") concept, and cross-cutting concerns.
- **Stage plans** — [stage0.md](stage0.md) … [stage6.md](stage6.md): one file per build stage, each with goal / files to port / config wiring / done-criteria / verification.

## Goals & invariants

Build a modular, JSON-configurable Python benchmark harness for LLM strategists in *Civ V: Vox Populi*, by porting the proven logic of `../vox-deorum-analysis` into an installable package where **everything experiment-specific is config, not code**. Hold the three invariants (see AGENTS.md): **(1) config over code**, **(2) modular + pluggable**, **(3) reports are generated, never authored**.

## Build order

We build **in the order the pipeline runs, but with estimators in LOAD-ONLY mode first** and training/tuning last — so the early stages never depend on `torch`/`xgboost`/`optuna` training:

```
stage0 scaffold ─▶ stage1 extract ─▶ stage2 estimators (LOAD) ─▶ stage3 adjust
        ─▶ stage4 analyses ─▶ stage5 report ─▶ stage6 train/tune
```

Stages 1–5 run against a **local config copied from `configs/benchmark.pretrained.template.json`** (estimators `fit:"pretrained"`, loaded from copied model dirs — zero training); `configs/benchmark.dev.json` is the worked, gitignored dev instance. Stage 6 implements `fit:"train"`/`tune` and brings up the train-based **`configs/benchmark.template.json`** plus the cross variant. (The `*.template.json` files are tracked examples; the config you actually run is a local, gitignored copy — see [../AGENTS.md](../AGENTS.md) §"Templates vs. local configs".)

## The load-only bootstrap

The pre-trained estimators were copied out of `D:\Cache\…\Vox Deorum\colm-2026\analysis\models\output\<model>_model\` (each has `metadata.json`
+ model state) into `reports/estimators/<id>/model/`. `fit:"pretrained"` loads a `model_dir`, re-runs inference on the current `turns` table, and emits `predictions.csv`. This proves extract→adjust→analyses→report end-to-end before any training code exists.

## Two output groups — the "cross" split

"cross" = the **llm/non-llm generalization split**: an estimator trained on **non-LLM** seats (Vanilla/Null) that predicts on everyone (old setup: `../vox-deorum-analysis/models/compare_models.py:133-144`, `--train-non-llm-only`). The **whole pipeline** can run in two variants:

| variant | estimator training | output root |
|---|---|---|
| normal | all (or LLM) seats | `reports/` |
| cross | non-LLM seats, predict all | `reports-cross/` (**`-cross` suffix is configurable**) |

So `estimators/<id>/`, `adjust/`, and the rendered report each exist under **both roots**. The cross group is produced by the **normal train pipeline** (estimator `train_subset:"non_llm"`); there is **no saved cross model and we don't need one now**, so the load-only bootstrap populates only `reports/`. The configurable output root is a schema addition (see [../configs/benchmark.md](../configs/benchmark.md), "output root") and is resolved once in stage0 and threaded into every stage's save paths.

We do **not** load the old OOF / `output_cross/` prediction CSVs as estimators: `fit:"pretrained"` stays `model_dir`-only (re-infer from saved weights), so column-name reconciliation of raw prediction CSVs is moot. The cross split is therefore an ordinary `fit:"train"` + `train_subset:"non_llm"` run (stage6), not a separate estimator kind — there is no saved cross model and we don't need one now.

## Cross-cutting

- **Controlled-design (seeds + seating).** One feature threaded across stages: **stage1** imports the controlled `seed`/`seating_rotation` (into `game_data`; `-1` ⇒ uncontrolled, sync==map asserted) and composes the **orthodox `player_type`** from per-player `model-{id}`/`strategist-{id}` metadata (not a seat map — [../configs/benchmark.md](../configs/benchmark.md) §3.3); **stage3**'s `strength` swaps the civ adjustment for a matched **start-cell** Vanilla-baseline correction on controlled games; **stage4**'s `ratings` bootstrap re-runs strength per replicate for the parts re-estimated from the sample (the civ OLS under `civ_adjust:"ols_logit"`; the per-cell baseline — implicit **and** explicit — is held **constant** across replicates from the persisted full-panel trail, option C) and `vanilla_slot_effect` validates it; **stage4+5** ship the controlled-seed chapter: the `performance.controlled_seed_report` analysis ([controlled-seed-report.md](controlled-seed-report.md)) plus its automatic rendering as a chapter of the single report (per-seed heatmap overview and one detail page per seed-player pair under `controlled-seed/`, rendered when html is among report.formats). All gated by the strength `block` param — `none` ⇒ legacy behavior.
- **Determinism.** Thread top-level `seed` into every stage with randomness; same config + same `runs/` ⇒ byte-stable outputs. Never call un-seeded RNGs.
- **No graceful degradation.** Every dependency is installed up front (`scripts/install`); a missing package aborts loudly. No `try/except ImportError` skips.
- **Config over code.** New model/experiment/slice = a config edit. When you add or rename a module, update `../configs/benchmark.md` in the same change.
- **Tooling.** Windows host, PowerShell default; relative paths; built-in tools over shell.

## Stage dependency graph

`stage0` is a prerequisite for all. `stage1 → stage2 → stage3 → stage4 → stage5` is the linear data path; `stage6` retro-fills training behind the same estimator config and adds the cross variant. Each stage's exit criterion is its **Done** line in the stage file.
