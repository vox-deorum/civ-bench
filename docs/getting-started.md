# Getting Started with civ-bench

This guide takes you from a fresh checkout to a finished report and covers common config edits. See the [README](../README.md) for an overview, the [Configuration guide](configuration.md) for a run-spec tour, and [configs/benchmark.md](../configs/benchmark.md) for the complete schema.

---

## 1. Install

Run the install script for your platform. It installs and verifies every required dependency, including the R rating packages:

```powershell
scripts\install.ps1     # Windows (PowerShell), the primary host
```
```bash
scripts/install.sh      # Linux / macOS
```

Under the hood this is just `pip install -e .` plus the dependency set. You will also need **R** on your `PATH` (or pointed to via the `CIV_BENCH_RSCRIPT` environment variable) with the `BradleyTerry2` and `PlackettLuce` packages. The install script sets these up and checks them for you.

The dependency set, for reference:

- **Python core:** pandas, numpy, scipy, statsmodels, matplotlib, seaborn, plotly, scikit-learn, tabulate
- **Python heavy:** torch, xgboost, optuna, imbalanced-learn
- **R:** BradleyTerry2, PlackettLuce (for the rating analyses)

## 2. Make a config

The repo tracks **example** run-specs as `configs/*.template.json`. They are documentation, not runnable as-is, because they reference illustrative experiments and machine-specific data paths. To actually run, copy one to a **local, gitignored** config and edit the data paths for your machine:

- [configs/benchmark.template.json](../configs/benchmark.template.json) is the lean **core** run: which strategist is stronger, how good the predictor is, and token cost.
- [configs/benchmark.full.template.json](../configs/benchmark.full.template.json) is the kitchen sink: core plus every optional or reserved module, shipped with `"enabled": false`.
- [configs/benchmark.pretrained.template.json](../configs/benchmark.pretrained.template.json) is a load-only driver that uses pre-trained estimators (zero training).
- [configs/benchmark.cross.template.json](../configs/benchmark.cross.template.json) is the "cross" variant: train on non-LLM seats, predict everyone, write to a separate `reports-cross/` root.

```powershell
# Copy a template, then edit data.extract.runs_dir / data.tables.* / pretrained.model_dir
copy configs\benchmark.template.json configs\benchmark.dev.json
```

Any `configs/benchmark*.json` that is not a `*.template.json` is gitignored, so your real run-specs with local paths never land in git.

## 3. Run it

```bash
# Run the whole DAG: extract, estimators, adjust, analyses, report
civ-bench run --config configs/benchmark.dev.json

# Just extract raw game DBs into canonical CSVs
civ-bench extract --config configs/benchmark.dev.json

# Re-render the report from existing analysis artifacts (no re-computation)
civ-bench report --config configs/benchmark.dev.json
```

Handy flags:

- `--dry-run` loads and validates the config, prints the resolved DAG and output root, and runs nothing. (Great for sanity-checking a config edit.)
- `--only <stage-id>` runs just that stage and its dependencies. Repeatable.
- `--skip <stage-id>` drops a stage. `--skip all` is equivalent to a dry run. Repeatable.
- `--force-rebuild` (or `-f`) re-extracts even when the CSVs are newer than the game DBs.

Your finished report lands in `reports/<run-name>/`. It contains `report.html`, a compact overview; `ratings.html`, `prediction.html`, `calibration.html`, `performance.html`, and `exploratory.html` for represented analysis families; `report.md`; and a self-contained `assets/` tree with `report.css`, figures, and full-table CSVs. Rendering is **deterministic** (no timestamps), so re-running `civ-bench report` reproduces every document byte-for-byte.

---

## Tutorial: from raw games to a finished report

This is a hands-on walkthrough. It assumes you have run the install script once and have a terminal open at the repo root. We will start with the fastest path to a real result, then look at how to read the output, then cover the edits you will make day to day.

### Step 0: sanity-check your setup with a dry run

Before touching any data, prove your config loads and the pipeline wires up the way you expect. A **dry run** validates the config and prints the resolved DAG without executing a single stage:

```bash
civ-bench run --config configs/benchmark.pretrained.template.json --dry-run
```

You will see the stages in dependency order (extract, estimators, adjust, analyses, report) and the output root they will write to. If you have a typo in your config, an unknown key, a missing required field, or a reference to a stage that does not exist, this is where it fails loudly, before any expensive work happens. Get into the habit of dry-running every config edit.

> The template references illustrative experiments and paths, so it will not *run* as-is, but it will *validate*, which is all a dry run does. Copy it to a local config (Step 2 above) before a real run.

### Step 1: understand what goes in

`civ-bench` reads two things:

- **Raw game databases**, the `*.db` files Vox Deorum writes, one per game, sitting somewhere under a `runs_dir` you point at. The **extract** stage turns these into four canonical CSVs in `runs/`:
  - `turn_data.csv`, one row per player per turn (the features the victory-probability models learn from)
  - `panel_data.csv`, one row per player per game (final outcomes, strategy mix, civilization)
  - `game_data.csv`, one row per game (timestamp, experiment, and the controlled `seed` and `seating_rotation`)
  - `model_token_usage.csv`, token counts per model plus failed strategist-turn counts
- **A run-spec**, the JSON config that names the data, the estimators, the analyses, and the report.

If you already have the four canonical CSVs (for example, someone handed them to you, or you generated them on another machine), you can skip extraction entirely by setting `data.extract.enabled: false` and dropping the CSVs into `runs/`. That is exactly what the **pretrained** template does.

### Step 2: make your own config from a template

Never edit a `*.template.json` directly. Copy it to a local, gitignored name and edit that:

```powershell
copy configs\benchmark.pretrained.template.json configs\benchmark.dev.json
```

Open `configs/benchmark.dev.json` and adjust three things for your machine:

- **`data.extract.runs_dir`**, where your `*.db` game databases live (only matters if extraction is enabled).
- **`data.tables.*`**, where the canonical CSVs should be written and read (the defaults under `runs/` are usually fine).
- **`data.filter`**, which experiments to include. The templates filter to `"staff_recent"` (the `2026-staff-standard` experiment). Change the experiment id in the `filters` block to match *your* data, or set `data.filter` to `"llm_only"` to keep all LLM seats across every experiment.

The pretrained template loads its victory-probability models from the tracked snapshots under [pretrained/](../pretrained/), so there is **no training**, no torch/xgboost/optuna training paths, and you get a report in minutes. That makes it the ideal first run.

### Step 3: run the whole pipeline

```bash
civ-bench run --config configs/benchmark.dev.json
```

Watch the stages execute in order. Each one prints what it produced: the estimators announce their `predictions.csv`, the adjust stage announces the strength panel and how many rows it wrote, each analysis reports how many tables and figures it emitted, and the report stage prints the files it wrote. When it finishes, open:

```
reports/staff-standard-2026/report.html
```

(`staff-standard-2026` is the `name` field in the config. Rename it and the output folder follows.)

### Step 4: read the report

`report.html` is the compact overview. Use its links to open the full family pages. The templates select eight overview cards by default: headline ratings, win rates, prediction metrics, reliability, strength, experiment completeness, token cost, and cost versus rating. Set `report.overview_sections` to `null` for a card for every selected report section, or provide your own ordered list. The report is organized into the five analysis families, in this order. Here is what to look at and why, with the connection to the CivBench paper's framework noted:

- **Ratings** are the headline. The Bradley-Terry and Plackett-Luce tables rank every `player_type` by skill, centered on the `Vanilla` baseline (the stock Vox Populi AI sits at Elo 1500 in the paper, so a rating above that means "stronger than stock Civ AI"). This is the answer to "who is the better player." The `bt_strategy` and `pl_strategy` sections break the same ratings down by dominant victory path (Domination, Culture, Diplomatic, Science). The matchup matrices show head-to-head strength and actual win rates.
- **Prediction** tells you how much to trust the rest. The metrics table (ROC-AUC, Brier, log-loss, balanced accuracy) is the paper's **predictive validity**. The ratings are only as trustworthy as the strength estimates feeding them, which come from this predictor (the paper's primary estimator, AttentionMLP, reaches AUC 0.865).
- **Calibration** asks whether the probabilities are honest. The reliability diagram should track the diagonal (when the model says 70 percent, it should win about 70 percent of the time). Loss-by-progress shows whether it is well-behaved early versus late in a game. The civ-effects and cell-baseline tables expose the confounds the adjust stage corrected for.
- **Performance** covers strength panels per player type (with confidence intervals and small-sample flags), score-ratio regressions, victory-probability trajectories over the course of a game, and a controlled-design completeness check.
- **Exploratory** reports token cost per player type and per model, so you can weigh strength against price.

Every inline table is capped for readability; the full data sits next to it as a linked CSV under `assets/<analysis-id>/`. To keep a family page focused, use `report.section_overrides` to select a stage's inline table or figure names. Unselected artifacts remain downloadable.

### Step 5: re-render without recomputing

The analysis results are cached to disk (each analysis writes a `result.json` beside its artifacts). If you only want to tweak the *report*, to reorder sections, change the title, or switch formats, you do not need to re-run the whole pipeline:

```bash
civ-bench report --config configs/benchmark.dev.json
```

This reads the existing artifacts and re-renders, deterministically and in seconds.

### Common recipes

Once you have a run working, these are the edits you will reach for most. All of them are config-only, no Python. For the full set of knobs, see the [Configuration guide](configuration.md).

**Add a strategist to the line-up.** New player types arrive *in the data*, not in analysis code. Register the model's pricing, color, and aliases in [configs/models.json](../configs/models.json) so its `player_type` composes and its token cost is priced, make sure the games are in your `runs_dir`, and re-run. No analysis changes; the new player just shows up in every ranking.

**Narrow to a subset of games.** Edit the `filters` block or `data.filter`. For example, to look only at the back half of games:

```jsonc
"filters": { "late_game": { "turn_range": [200, null] } },
"data":    { "filter": ["llm_only", "late_game"] }   // merged left-to-right
```

A stage can also carry its own `filter` that *narrows* the global one, handy for a single analysis that should only see one experiment.

**Add confidence intervals to the ratings.** Bootstrap CIs are not a separate module; they are a param. On any `ratings.bradley_terry` or `ratings.plackett_luce` entry:

```jsonc
"params": { "group_by": ["player_type"], "ref": "Vanilla",
            "bootstrap": { "n": 1000, "stratified": true } }
```

**Rate by strategy instead of just by model.** Also a param. Set `group_by` to two dimensions and the rated identity becomes the composite:

```jsonc
"params": { "group_by": ["player_type", "strategy"] }   // per-strategy Elo
```

(The `strategy` grouping is defined once in the top-level `groupings` block.)

**Run just one stage while iterating.** Use `--only` to run a stage and its dependencies, or `--skip` to drop one:

```bash
civ-bench run --config configs/benchmark.dev.json --only bt_main      # one rating + its deps
civ-bench run --config configs/benchmark.dev.json --skip extract       # reuse existing CSVs
```

**Train models locally.** Use [configs/benchmark.template.json](../configs/benchmark.template.json), where each estimator has `fit: "train"`. The attention model also has a `tune` block for Optuna hyperparameter search. Local training takes much longer than loading snapshots.

**Turn on an optional analysis.** The full template lists optional modules with `"enabled": false`. When an implementation exists, flip the flag to `true` (or copy the entry into your config) and it joins the run. Registry-reserved placeholders fail loudly if enabled before their implementation lands, which is intentional.

---

## Troubleshooting

- **A dependency is missing.** Re-run `scripts/install.ps1` or `scripts/install.sh`. Required packages stop the run when absent.
- **`Rscript not found`.** Install R from [CRAN](https://cran.r-project.org/) and make sure `Rscript` is on your `PATH`, or set `CIV_BENCH_RSCRIPT` to its full path. The `ratings.*` analyses need it.
- **A config error on load.** The loader fails loud on unknown keys and missing required fields. Read the message; it names the offending key. Cross-check against [configs/benchmark.md](../configs/benchmark.md).
- **Extraction seems to do nothing.** It is skipped automatically when every output CSV already exists and is newer than the game DBs. Pass `--force-rebuild` to rebuild anyway.
