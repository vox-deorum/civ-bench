<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 1 — extract

**Goal.** Turn raw game SQLite DBs in `runs/` into the canonical CSVs every later stage reads:
`turn_data`, `panel_data`, **`game_data`** (per-game row; was `game_timestamps`), `model_token_usage`.

## Files to create / port

- `civ_bench/extract/` — port `../vox-deorum-analysis/extract/` (`__main__.py`, `extract_turns.py`,
  `extract_panel.py`, `extract_model_tokens.py`, `utilities.py`). Keep the DB-discovery + extractor
  logic; **strip hardcoded paths** — drive roots from `configs/paths.json`.

### New: import controlled seeds/seating + orthodox player_type (benchmark.md §3.3)

The game runner now records, in each DB's `GameMetadata` (Key→Value), controlled-experiment parameters and
per-player identity. Extraction pulls them and writes them **lean** — per-game facts in `game_data`, per-player
identity in `panel_data`, nothing new on the huge `turn_data` except the composed `player_type`:

- **`extract_seeding_fields(metadata)`** (in `utilities.py`): read `configuredMapRandSeed` /
  `configuredSyncRandSeed` and **assert they are equal — a mismatch ABORTS extraction** (game_seed ≡ map_seed
  is an invariant). Set a single `seed` = the controlled value, else **`-1`** (uncontrolled). Read
  `seatingRotation`, else **`-1`**. Read `seatingMap` and invert it to a per-player `config_slot`
  (`player_id` when uncontrolled). `seating_seed_index` is **not extracted** — the actual `seed` value
  subsumes the array index. The two `-1` sentinels replace what would otherwise be `seeds_controlled` /
  `seating_controlled` boolean columns; `controlled` is derived downstream as
  `seed != -1 and seating_rotation != -1`.
- **Orthodox `player_type`** — compose at extract via `civ_bench/catalog/compose_player_type` from the
  per-player `model-{id}` + `strategist-{id}` metadata (benchmark.md §3.3). This **replaces** the old
  load-time `(condition, player_id)` static merge as the primary path (the static map is fallback only), and
  is correct under seat rotation because the identity travels with the player. The same composition feeds
  `model_token_usage.csv` (single source of truth).

### Canonical CSV schema changes

- **`game_data.csv`** (rename of `game_timestamps.csv`, config key `games`) — one row per game:
  `game_id, timestamp, experiment, seed, seating_rotation`. `seed`/`seating_rotation` use the `-1`
  uncontrolled sentinel.
- **`panel_data.csv`** — gains player-level extras: `player_type` (orthodox), `model`, `strategist`,
  `config_slot`. `civilization` stays (seat-bound).
- **`turn_data.csv`** — **keeps `player_type`** (composed once per (game, player) and broadcast) but stores
  **no `seed`** and no seating columns; the strength stage joins `seed` from `game_data` by `game_id` only
  where it needs the start-cell. No `seat` column is materialized anywhere — it is `player_id`;
  `start_cell = (seed, player_id)` is derived at strength-build time.

## Config wiring (`data.extract`, `data.tables`)

Honor `enabled`, `runs_dir`, `outputs` (`["turns","panel","games","tokens"]`), `max_dbs`,
`prune_missing`, `force_rebuild`. Skip the stage automatically when every `outputs` CSV exists and is
newer than the DBs (unless `force_rebuild`). `extract.enabled:false` drops the stage and loaders read
`tables.*` directly.

## Bootstrap shortcut (so stages 2–5 are testable before raw DBs are wired)

Set `extract.enabled:false` in `benchmark.pretrained.json` and drop the OneDrive analysis-root CSVs
(`turn_data.csv`, `panel_data.csv`, `game_data.csv`, `model_token_usage.csv`) into `runs/`.

## Done

`civ-bench extract --config …` produces the four canonical CSVs at `data.tables.*` (incl. `game_data.csv`
with `seed`/`seating_rotation` and the `-1` sentinels), and `civ_bench/data/` loads them with the orthodox
`player_type` applied (joining `player_type`/`seed` by `(game_id, player_id)` / `game_id`, not the static
seat map).

## Verification

- Row/column sanity vs. the old `turn_data.csv`/`panel_data.csv` — turn_data unchanged except for the
  retained `player_type`; `seed`/seating live only in `game_data`; no `seat` column.
- A fabricated game with `configuredSyncRandSeed != configuredMapRandSeed` **aborts** with a clear error.
- `player_type` follows `model-{id}`/`strategist-{id}` through a seat rotation; a leading-`-` label appends,
  a non-`-` label overrides, `(condition, slot)` beats `condition`; a legacy game with no metadata falls back
  to the static map.
- Re-running with unchanged DBs is a no-op (skip-if-newer); `force_rebuild:true` rebuilds.
