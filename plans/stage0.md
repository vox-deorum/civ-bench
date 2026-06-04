<!-- PARITY: keep this file in sync with the code it describes. If the package skeleton, config
     schema, or output-root resolution changes, update this file in the same change. -->

# Stage 0 — scaffold & infrastructure

**Goal.** Stand up the installable package, the JSON config loader/validator, the DAG pipeline, and the shared infra (catalogs, data, stats, plotting) that every later stage imports. After this stage a config can be loaded, validated, and its DAG printed — without running any analysis.

## Files to create / port

- `pyproject.toml` — `pip install -e .`, exposes the `civ-bench` CLI entrypoint; declares the full dependency set (see AGENTS.md §Dependencies) as plain requirements (no extras).
- `civ_bench/__init__.py`, `civ_bench/cli.py` — `civ-bench extract|run|report` + `--only`/`--skip`/`--config`.
- `civ_bench/config/` — load + validate `benchmark.json` per [../configs/benchmark.md](../configs/benchmark.md) §8 (unknown keys / missing required = hard error); expose typed config objects; **resolve the output root** (default `reports`, configurable suffix → `reports-cross`) and thread it to all stages.
- `civ_bench/pipeline/` — build the stage DAG from kind-ordering + `needs` + `uses`, topo-sort, run.
- `civ_bench/catalog/` — port `shared/model_catalog.py` + `shared/experiments.py` (alias normalization, vanilla/null groups). **Add the orthodox `player_type` composition** (benchmark.md §3.3): `compose_player_type(model, strategist, condition, config_slot)` builds the identity from the per-player game metadata (`model-{id}` + `strategist-{id}`) via a `player_type_template` + alias maps, then applies the unified `player_type_labels` map (leading-`-` ⇒ suffix, else full override; `(condition, slot)` beats `condition`). The old static `(condition, seat)` mapping is demoted to an **optional fallback** (legacy games with no metadata).
- `civ_bench/data/` — port `shared/data_loading.py` (`load_turn_data`/`load_panel_data` + filters).
- `civ_bench/stats/` — port `shared/regression_utilities.py` (OLS/logistic, clustered/weighted fits, coeff/odds-ratio heatmaps). Used by `performance.score_ratio`, `ratings.matchups`, `adjust/strength`.
- `civ_bench/plotting/` — port `shared/plot_styles.py` + `plot_utilities.py` (trim notebook-only helpers).
- `configs/models.json`, `configs/experiments.json`, `configs/paths.json` — port from `../vox-deorum-analysis/shared/config/` and extend the schema where the harness needs richer metadata; these are now the primary control surface. **Add the catalog fields backing benchmark.md §3.3:** `player_type_template`, model/strategist alias maps, the vanilla/null labels, and the unified `player_type_labels` map (per-`condition` and per-`(condition, slot)`). The legacy per-seat `CONDITION_PLAYER_MAPPING` is kept only as the optional fallback.

## Config wiring

- `catalogs` defaults to sibling `configs/*.json` paths, but catalogs are loaded lazily. The loader fails only when a selected stage actually needs a catalog and neither an override nor the sibling file exists.
- **R discovery is cross-platform:** find `Rscript` via `PATH` and the `CIV_BENCH_RSCRIPT` env override — **drop** the old `_find_rscript()` hardcoded `C:`/`D:\Program Files\R` scan (D5).
- Add the **output-root** field (see benchmark.md): all stage save-paths (`estimators` `save_*`, `adjust` `save`, `report.out_dir`) resolve relative to the resolved `<root><suffix>/`.

## Done

`civ-bench run --config configs/benchmark.pretrained.json --skip <all>` (or a dry-run flag) loads and validates the config and prints the resolved DAG + the resolved output root, with no stage executed.

## Verification

- A deliberately malformed config (unknown key, missing required field, cycle, dangling `needs`/`uses`) fails loudly with a precise message.
- `import civ_bench` works after `pip install -e .`; `scripts/install.ps1` verifies every dependency.
