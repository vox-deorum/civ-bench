<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 1 — extract

**Goal.** Turn raw game SQLite DBs in `runs/` into the canonical CSVs every later stage reads: `turn_data`, `panel_data`, **`game_data`** (per-game row; was `game_timestamps`), `model_token_usage`.

## Files to create / port

- `bench/extract/` — port `../vox-deorum-analysis/extract/` (`extract_turns.py`, `extract_panel.py`, `extract_model_tokens.py`, `utilities.py`). Keep the DB-discovery + extractor logic; **strip hardcoded paths** — drive roots from `data.extract.runs_dir` / `data.tables.*`, consulting the paths catalog only when an enabled configuration actually provides or requires it. The source `__init__.py`/`__main__.py` orchestration (which read a global `shared.paths.ROOT_DIR`) is replaced by a config-driven [`runner.py`](../bench/extract/runner.py) (`run_extract(cfg, catalog)`) invoked from `civ-bench extract`. New helpers: [`identity.py`](../bench/extract/identity.py) (orthodox `player_type` composition shared by panel/turn/token), [`extract_games.py`](../bench/extract/extract_games.py) (the renamed `game_data`), and `utilities.extract_seeding_fields` (controlled-seed/seating reduction). The old `shared.model_catalog` imports in the token extractor route through `bench.catalog` instead.

### New: import controlled seeds/seating + orthodox player_type (benchmark.md §3.3)

The game runner now records, in each DB's `GameMetadata` (Key→Value), controlled-experiment parameters and per-player identity. Extraction pulls them and writes them **lean** — per-game facts in `game_data`, per-player identity in `panel_data`, nothing new on the huge `turn_data` except the composed `player_type`:

- **`extract_seeding_fields(metadata)`** (in `utilities.py`): read `configuredMapRandSeed` / `configuredSyncRandSeed`. Vox Deorum supports distinct sync/map seeds, but civ-bench's controlled-design benchmark intentionally requires matched starts, so configured controlled rows must have equal sync/map values; a mismatch **ABORTS extraction** with a clear policy error. Set a single `seed` = the controlled value, else **`-1`** (uncontrolled). Read `seatingRotation`, else **`-1`**. Read `seatingMap` and invert it to a per-player `config_slot` (`player_id` when uncontrolled). `seating_seed_index` is **not extracted** — the actual `seed` value subsumes the array index. The two `-1` sentinels replace what would otherwise be `seeds_controlled` / `seating_controlled` boolean columns; `controlled` is derived downstream as `seed != -1 and seating_rotation != -1`.
- **Orthodox `player_type`** — compose at extract via `bench/catalog/compose_player_type` from the per-player `model-{id}` + `strategist-{id}` metadata (benchmark.md §3.3). This **replaces** the old load-time `(condition, player_id)` static merge as the primary path (the static map is fallback only), and is correct under seat rotation because the identity travels with the player. The same composition feeds `model_token_usage.csv` (single source of truth).

### Canonical CSV schema changes

- **`game_data.csv`** (rename of `game_timestamps.csv`, config key `games`) — one row per game: `game_id, timestamp, experiment, seed, seating_rotation`. `seed`/`seating_rotation` use the `-1` uncontrolled sentinel. It does **not** carry per-player fields.
- **`panel_data.csv`** — gains player-level extras: `player_type` (orthodox), `model`, `strategist`, `config_slot`. `civilization` stays (seat-bound).
- **`turn_data.csv`** — **keeps `player_type`** (composed once per (game, player) and broadcast) but stores **no `seed`** and no seating columns; the strength stage joins `seed` from `game_data` by `game_id` only where it needs the start-cell. No `seat` column is materialized anywhere — it is `player_id`; `start_cell = (seed, player_id)` is derived at strength-build time.

## Config wiring (`data.extract`, `data.tables`)

Honor `enabled`, `runs_dir`, `outputs` (`["turns","panel","games","tokens"]`), `max_dbs`, `prune_missing`, `force_rebuild`. Skip the stage automatically when every `outputs` CSV exists and is newer than the DBs (unless `force_rebuild`). `extract.enabled:false` drops the stage and loaders read `tables.*` directly.

## Bootstrap shortcut (so stages 2–5 are testable before raw DBs are wired)

Set `extract.enabled:false` in your local load-only config (copied from `benchmark.pretrained.template.json`) and drop the OneDrive analysis-root CSVs (`turn_data.csv`, `panel_data.csv`, `game_data.csv`, `model_token_usage.csv`) into `runs/`. (The worked dev config `benchmark.dev.json` instead sets `extract.enabled:true` with `runs_dir: "J:/"` to exercise extract from raw DBs.)

## Done

`civ-bench extract --config …` produces the four canonical CSVs at `data.tables.*` (incl. `game_data.csv` with `seed`/`seating_rotation` and the `-1` sentinels), and `bench/data/` loads them with the orthodox `player_type` applied (joining `player_type`/`seed` by `(game_id, player_id)` / `game_id`, not the static seat map). `player_type` is composed once per (game, player) at extract from `GameMetadata` (`model-{id}` + `strategist-{id}`) and written to `panel_data` (with `model`/`strategist`/`config_slot`), broadcast across `turn_data`, and reused as the single source of truth in `model_token_usage`. The extract stage honors `enabled`/`runs_dir`/`outputs`/`max_dbs`/`prune_missing`/`force_rebuild` and auto-skips when every output CSV is newer than every source DB.

## Verification (all passing — `tests/test_extract.py`, 17 tests)

- A fabricated controlled game with `configuredSyncRandSeed != configuredMapRandSeed` **aborts** with a clear civ-bench `ExtractError` policy error; uncontrolled games and absent configured seeds still use `-1` sentinels, and a configured `0` ("pick random") reads as uncontrolled; a `seatingRotation` of `0` stays a valid (≥ 0) rotation.
- `seatingMap` (`{config_slot: player_id}`) inverts to per-player `config_slot`; `player_type` follows `model-{id}`/`strategist-{id}` through a seat rotation (identity travels with the player, not the seat); a `VPAI` seat resolves to the `Vanilla` baseline; a legacy game with no metadata falls back to the static `(condition, slot)` seat map.
- End-to-end `game_data` extraction writes `seed`/`seating_rotation` (incl. the `-1` sentinels) and the filename `timestamp`; end-to-end `panel_data` extraction lands `player_type`/`model`/`config_slot` for a two-player synthetic DB.
- `run_extract` skips when outputs are newer than the DBs and runs anyway under `force_rebuild`; a disabled `extract` is reported as skipped (loaders read `tables.*` directly).
- Config validation: `data.extract` scalar types (`enabled`/`runs_dir`/`max_dbs ≥ 1`/`prune_missing`/`force_rebuild`) and `data.tables` string paths fail loud on the wrong shape.
