<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 1 — extract

**Goal.** Turn raw game SQLite DBs in `runs/` into the canonical CSVs every later stage reads:
`turn_data`, `panel_data`, `game_timestamps`, `model_token_usage`.

## Files to create / port

- `civ_bench/extract/` — port `../vox-deorum-analysis/extract/` (`__main__.py`, `extract_turns.py`,
  `extract_panel.py`, `extract_model_tokens.py`, `utilities.py`). Keep the DB-discovery + extractor
  logic; **strip hardcoded paths** — drive roots from `configs/paths.json`.

## Config wiring (`data.extract`, `data.tables`)

Honor `enabled`, `runs_dir`, `outputs` (`["turns","panel","timestamps","tokens"]`), `max_dbs`,
`prune_missing`, `force_rebuild`. Skip the stage automatically when every `outputs` CSV exists and is
newer than the DBs (unless `force_rebuild`). `extract.enabled:false` drops the stage and loaders read
`tables.*` directly.

## Bootstrap shortcut (so stages 2–5 are testable before raw DBs are wired)

Set `extract.enabled:false` in `benchmark.pretrained.json` and drop the OneDrive analysis-root CSVs
(`turn_data.csv`, `panel_data.csv`, `game_timestamps.csv`, `model_token_usage.csv`) into `runs/`.

## Done

`civ-bench extract --config …` produces the four canonical CSVs at `data.tables.*`, and
`civ_bench/data/` loads them with experiment/player-type mapping applied.

## Verification

- Row/column sanity vs. the old `turn_data.csv`/`panel_data.csv` (same schema after porting).
- Re-running with unchanged DBs is a no-op (skip-if-newer); `force_rebuild:true` rebuilds.
