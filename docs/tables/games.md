# The `games` table

**One row per game.** It is the game list: which condition each game belongs to, when it was played, and where it sits in the controlled design. Other tables point to it by `game_id`.

- **File:** `runs/game_data.csv` (set by `data.tables.games`)
- **Built from:** the game database's file name and `GameMetadata`
- **Read by:** the extract, trim, and prune steps (to know which games exist), filters that need completeness, the `strength` adjust stage, and several `performance.*` and `behavior.*` analyses

## The ideas behind it

| Column | What it means |
| --- | --- |
| `game_id` | The game's identifier, from the database file name. |
| `timestamp` | When the game was saved, from the file name. |
| `experiment` | The condition the game was run under (its folder). |
| `seed` | The map seed in a controlled design. The same seed means the same map and starting positions. `-1` means the game was not controlled. |
| `seating_rotation` | Which rotation of the seating the game used, so each model plays each start position. `-1` when uncontrolled. |

## Key conventions

- **A controlled game has both a seed and a rotation.** Analyses that compare models on the same maps use only those.
- **Each player's seat in the design** (`config_slot`) is in [`panel`](panel.md), not here.
- **One database per game.** If a game was saved more than once, the newest file wins.

## Reference: exact rules

- `seed` comes from `configuredSyncRandSeed` and `configuredMapRandSeed`, which must agree; a mismatch stops extraction. A seed of 0 (Civ's "pick random") is stored as `-1`.
- `seating_rotation` is kept when it is 0 or more, else `-1`.
- A database that cannot be opened, or whose metadata cannot be read, gets no row and is recorded as an import issue rather than written as an uncontrolled game.
