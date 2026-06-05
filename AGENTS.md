# AGENTS.md — conventions for working in `civ-bench`

This file is the **rulebook**. It tells any agent (human or model) *how* to build and change `civ-bench`. For *what* we're building and the current roadmap, see [plans/plan.md](plans/plan.md). For the full schema of the run spec, see [configs/benchmark.md](configs/benchmark.md).

`civ-bench` is a modular, JSON-configurable Python **benchmark harness** for LLM strategists in *Civilization V: Vox Populi* (via the Vox Deorum platform). Point it at raw game-run data, hand it a config, and it produces a report — extraction, analysis, and rendering all driven by JSON.

## The three invariants

Hold the line on these. They are the whole point of the rewrite; a change that violates one is wrong even if it "works":

1. **Config over code.** Anything that changes between datasets, experiments, model line-ups, or report selections lives in a JSON file under `configs/`, never hardcoded in a module. Adding a new model or experiment must require *zero* Python edits.
2. **Modular + pluggable.** Each analysis (ratings, prediction, calibration, performance, exploratory) is a self-contained unit behind a common interface and a registry. Adding an analysis is adding one module + one registry entry + one config block — touching nothing else. Robustness variants of an existing analysis (bootstrap CIs, identity groupings like per-strategy Elo) are **params on the parent module**, not new modules — trimming or adding them is a config edit.
3. **Reports are generated, never authored.** The harness consumes raw game-run data and emits a complete report (tables + figures + narrative scaffolding). No notebook is the source of truth for a result. If a result can't be regenerated from `civ-bench run`, it doesn't belong in the repo.

## Repo layout (canonical)

This is the authoritative tree. Build it out incrementally; not all of it exists yet.

```text
civ-bench/
├── AGENTS.md                     # this file — conventions
├── plans/                        # roadmap + staged build plan
├── README.md
├── pyproject.toml                # installable: `pip install -e .`, exposes `civ-bench` CLI
├── configs/                      # ── the JSON control surface ──
│   ├── benchmark.template.json            # TEMPLATE: DEFAULT run spec = the lean CORE (~10 modules; see configs/benchmark.md)
│   ├── benchmark.full.template.json       # TEMPLATE: kitchen-sink — core + every optional module with "enabled": false
│   ├── benchmark.pretrained.template.json # TEMPLATE: load-only driver (fit:pretrained, no training)
│   ├── benchmark.md              # the benchmark.json convention / schema reference (applies to every benchmark*.json)
│   ├── models.json               # strategist + prediction model registry (id/aliases/color/pricing; player_type_template)
│   ├── experiments.json          # experiment registry; unified player_type_labels (leading-"-"=suffix, else override); vanilla/null groups; legacy seat map = optional fallback
│   └── paths.json                # repo-relative data + output roots
│   #  ── tracked: the *.template.json examples + the catalogs above + benchmark.md (schema) ──
│   #  ── gitignored: the ACTUAL run-specs (any other configs/benchmark*.json, e.g. benchmark.dev.json) — they
│   #     point at machine-specific data roots; copy a template to make one. See "Templates vs. local configs" below.
│   #  groupings (rating-identity dimensions, e.g. "strategy") + filters live INLINE in each benchmark*.json
├── bench/
│   ├── __init__.py
│   ├── cli.py                    # `civ-bench extract|run|report ...`
│   ├── pipeline/                 # DAG builder + scheduler: resolve `needs`, topo-sort, run stages
│   ├── config/                   # load + validate JSON, expose typed config objects
│   ├── extract/                  # game SQLite DBs → turn/panel/token CSVs
│   ├── data/                     # canonical CSV loading, filtering, player-type mapping
│   ├── catalog/                  # model + experiment catalogs backed by configs/*.json
│   ├── estimators/               # prediction-model producers: tune / train / load / infer
│   │   ├── registry.py           # name → predictor class (port of models/utils/model_registry.py)
│   │   └── models/               # score / baseline / xgboost / mlp / grouped / interaction / attention
│   ├── adjust/                   # derived-table stages: estimator P(win) → strength panel
│   │   ├── registry.py           # name → adjust class (currently: strength)
│   │   ├── strength.py           # adjusted_strength derivation (port of turn_predicted's prepare_strength_data); controlled games use the matched start-cell Vanilla-baseline correction
│   │   └── strength_lmm.R        # lme4 shrinkage fit for the per-(seed,seat) Vanilla baseline (controlled-design)
│   ├── analyses/                 # ── pluggable analysis modules ──
│   │   ├── base.py               # Analysis interface + registry (name → class)
│   │   ├── ratings/              # Bradley-Terry / Plackett-Luce / matchups (group_by + bootstrap params)
│   │   ├── prediction/           # estimator scoring: evaluate + compare
│   │   ├── calibration/          # calibration of the estimator: reliability + loss-by-progress
│   │   ├── performance/          # strength panels, score-ratio regressions, turn-predicted
│   │   └── exploratory/          # dataset descriptives (token costs, panel/turn summaries)
│   │   #   behavior/ is DEFERRED — not built now (revisit as an opt-in extension)
│   ├── stats/                    # statistics layer: OLS/logistic wrappers, clustered/weighted fits,
│   │                             #   coeff/odds-ratio heatmaps (port of shared/regression_utilities.py).
│   │                             #   Imported by performance.score_ratio, ratings.matchups, adjust/strength.py
│   ├── reports/                  # assemble analysis artifacts → markdown/html report
│   └── plotting/                 # shared styles, colors, figure helpers
├── tests/                        # pytest suite (mirrors bench/ by area: config, pipeline, catalog, …)
├── runs/                         # raw game-run input data + canonical CSVs (gitignored)
└── reports/                      # generated reports, figures, trained estimators (gitignored)
```

**Tests** live in `tests/` at the repo root (not inside `bench/`), one file per area
(`test_config.py`, `test_pipeline.py`, `test_catalog.py`, …). Install the test tool with
`pip install -e ".[test]"` and run `pytest` from the repo root. Tests must not touch
machine-specific data roots or execute stages against real `runs/` data — they exercise
config/validation/DAG/catalog logic and use small synthetic fixtures. When you add a module
or a validation rule, add or extend its test in the same change.

### Templates vs. local configs

The repo tracks **example** run-specs as `configs/*.template.json` (the lean core, the kitchen-sink, and the load-only driver). They are documentation, not runnable configs — they reference experiments/paths that are illustrative. The **actual** config you run is a *local, gitignored* copy:

- Copy a template to `configs/benchmark<whatever>.json` (e.g. `configs/benchmark.dev.json`) and edit `data.extract.runs_dir` / `data.tables.*` / `pretrained.model_dir` for your machine.
- `.gitignore` tracks `configs/benchmark*.template.json` (and the catalogs + `benchmark.md`) but ignores every other `configs/benchmark*.json`, so real run-specs with machine-specific paths never land in git.
- `configs/benchmark.dev.json` is the worked dev instance: extract raw DBs from `J:/`, load pre-trained estimators, run the full core analysis + report into `reports-dev/`.
- `benchmark.md` is the schema for **all** of them; "benchmark.json" throughout the docs names the run-spec *format*, not a tracked file.

## The pipeline model

`civ-bench` is a **directed acyclic graph of stages**, not a fixed script. There are five stage kinds, and they run in dependency order:

```
extract ──▶ estimators ──▶ adjust ──▶ analyses ──▶ report
   │            │             │           │            │
canonical   trained or     strength    ratings /    templated
  CSVs       pre-trained     panel      prediction /   markdown
            predictors    (adjusted_   calibration /  + html
                           strength)   performance
```

- **`extract`** turns raw game DBs in `runs/` into canonical CSVs (`turn_data`, `panel_data`, `game_data`, `model_token_usage`). It also imports controlled seeds/seating into `game_data` (`seed`/`seating_rotation`, `-1` ⇒ uncontrolled) and composes the **orthodox `player_type`** from per-player metadata (benchmark.md §3.3). It is skipped when its outputs already exist unless forced.
- **`estimators`** are prediction-model producers. Each is either **trained** on the current data (optionally tuned first) or loaded **pre-trained** from a saved model directory, then run to emit `predicted_win_probability`. They must run **before** any analysis that consumes their output.
- **`adjust`** (optional) turns an estimator's per-turn win-probabilities into a per-player-game **strength panel** (`adjusted_strength`): late-game weighted average → relative-to-leader → winner enforcement → civilization adjustment (uncontrolled games) **or** a matched **start-cell** Vanilla-baseline correction (controlled games — fixed seeds + seating). It registers that panel as a named `strength` table, and it is what makes the rating models depend (transitively) on an estimator.
- **`analyses`** are the pluggable modules. **`ratings.*` consume the `strength` table** (they rate `adjusted_strength`, not raw `panel_data`, so they depend on an `adjust` stage; their `group_by` / `bootstrap` params fold in what were once separate strategy-Elo and bootstrap modules); `prediction.*` (scoring) and `calibration.*` (calibration of the estimator) and the predicted-strength `performance.*` **depend on an estimator**; `exploratory.*` are mostly descriptive over the canonical CSVs, though a few (e.g. `strategy_profiles`) also read the `strength` table.
- **`report`** walks the produced artifacts and renders them.

There is **no graceful degradation**: every dependency is installed up front via `scripts/install`, and a stage that needs a missing package aborts the run loudly rather than being silently skipped.

Ordering is expressed in the run-spec (`configs/benchmark*.json`) via implicit kind-ordering plus explicit `needs` edges and `uses` references. The full grammar — including how to train an estimator up front versus point at a pre-trained one — is specified in [configs/benchmark.md](configs/benchmark.md). **When you add or change a module, update that schema in the same change.**

## The analysis plugin contract

Every analysis implements one small interface and registers under a string name so JSON can select it. Keep the contract minimal — config in, structured artifacts out:

```python
# bench/analyses/base.py  (sketch)
class Analysis:
    name: str                      # registry key used in the run-spec (configs/benchmark*.json)
    def run(self, ctx, params: dict) -> AnalysisResult: ...
        # ctx: loaded data + catalogs + paths + resolved estimator artifacts
        # params: this analysis's JSON config block
        # returns tables (DataFrames), figures (paths), and a short text summary
```

The report stage walks the produced `AnalysisResult`s and renders them — no analysis hardcodes its place in the document.

## Conventions

- **Python package, not loose scripts.** Use absolute imports from `bench.*`. No `cd models && python compare_models.py`; commands run from repo root via the CLI.
- **JSON is the source of truth.** Config files under `configs/` are validated on load (fail loud on unknown keys / missing required fields). A new dataset = new config, never a code branch.
- **Analyses return data, not side effects.** They produce `AnalysisResult` objects; writing files and printing summaries is the report/CLI layer's job. This makes them testable and composable.
- **Estimators are producers, not analyses.** Training, tuning, saving, and loading predictors live under `bench/estimators/`. An estimator either trains on the current run or loads pre-trained weights; either way it exposes the same predictions artifact to downstream analyses.
- **`adjust` stages are derived-table producers, not analyses.** They live under `bench/adjust/`, consume an estimator's predictions, and emit a named table (e.g. `strength`) that downstream stages reference via `uses.tables`. The derivation logic (the strength panel) belongs here once, not duplicated inside each `ratings.*` module — which is exactly the bug being fixed from the old repo, where `prepare_strength_data` was copied between `turn_predicted.ipynb` and `iterative_bt.py`.
- **Determinism.** Same config + same `runs/` data ⇒ byte-stable tables. Thread `seed` from config; never call un-seeded RNGs.
- **All dependencies are mandatory — no soft-fail.** `torch` / `xgboost` / `optuna` / R packages are installed by `scripts/install`, not gated behind try-imports. Import them directly; if one is missing, the run fails loudly with an install hint. Do **not** add `try/except ImportError` skip-with-warning fallbacks.
- **Nothing experiment-specific in git** beyond `configs/` examples. `runs/` and `reports/` are gitignored.

## Commands (target CLI)

```bash
pip install -e .                                       # install package + civ-bench entrypoint
# Run against a LOCAL config (copy a *.template.json first); benchmark.dev.json is the worked dev instance.
civ-bench extract --config configs/benchmark.dev.json  # raw game DBs → canonical CSVs
civ-bench run     --config configs/benchmark.dev.json  # run the full DAG (extract→estimators→analyses→report)
civ-bench report  --config configs/benchmark.dev.json  # re-render report from existing analysis artifacts
civ-bench run --config configs/benchmark.dev.json --only ratings.bradley_terry   # one stage + its deps
civ-bench run --config configs/benchmark.dev.json --skip extract                 # reuse existing CSVs
```

Until the CLI exists, mirror the old entrypoints (`python -m bench.extract`, etc.) but always parameterize by config path.

## Dependencies

**Everything is required and installed up front — there are no optional extras.** Run the install script before anything else:

```bash
scripts/install.ps1        # Windows (PowerShell) — primary host
scripts/install.sh         # Linux/macOS
```

It installs the package (`pip install -e .`) and every dependency, then verifies each import and the R packages, failing loudly if any is absent. The full set:

- **Python core:** `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `tabulate`.
- **Python heavy (still required):** `torch`, `xgboost`, `optuna`, `imbalanced-learn`.
- **R (for ratings + the controlled-design strength baseline):** `BradleyTerry2`, `PlackettLuce`, and `lme4` (+`Matrix`) for the `adjust/strength` start-cell shrinkage fit (needs `Rscript` discoverable on `PATH`, or via the `CIV_BENCH_RSCRIPT` env override — **not** a hardcoded Windows path like the old repo's `C:`/`D:\Program Files\R` scan, which silently fails on Linux/macOS).

`pyproject.toml` declares the Python set as plain install requirements (one environment, no `civ-bench[torch]`-style extras). Code imports these directly; a missing dependency is a hard error, never a skipped stage (see the dependency convention above).

## Tool-calling rules (reduce permission prompts)

- **Use built-in tools, not bash equivalents** — Read not `cat`/`head`/`tail`, Edit not `sed`/`awk`, Write not `echo >`, Grep not `grep`/`rg`, Glob not `find`/`ls`.
- **Never `cd` into the working directory** — pass repo-relative paths to commands instead.
- **Always use relative paths** in shell commands — never absolute paths like `f:\vox-deorum\...` or `/f/vox-deorum/...`.
- **Avoid noisy shell idioms** — skip `2>/dev/null || echo "not found"`; the dedicated tools handle missing paths gracefully.
- This is a Windows host with PowerShell as the default shell — prefer PowerShell syntax (`$null`, `$env:VAR`) when a shell command is unavoidable.
