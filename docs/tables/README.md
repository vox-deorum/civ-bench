# Canonical tables

The `extract` stage turns raw game files into seven CSV tables. Everything after it (estimators, the strength panel, every analysis, and the report) reads these tables instead of the raw files. This page explains what each table is for and how they fit together. Each linked page leads with the concepts and keeps the exact column rules at the end.

## Where the data comes from

Each game leaves two kinds of files under `runs/`:

- **A game database** (`<game>_<timestamp>.db`), written by the game. It holds what happened: turn summaries, cities, events, and the strategist's setting changes.
- **A telemetry database per player** (`<game>-player-<n>.db`), written by the strategist host. It holds how the strategist worked: model calls, tokens, tool calls, and failed turns.

## The seven tables

| Table | One row per | Built from | What it is for |
| --- | --- | --- | --- |
| [`games`](games.md) | game | game DB | The game list: condition, date, and place in the controlled design |
| [`panel`](panel.md) | player per game | game DB | Each player's final result, identity, and victory focus |
| [`turns`](turns.md) | player per turn | game DB | The game state turn by turn; what the prediction models learn from |
| [`behavior`](behavior.md) | player per game | game DB | How each player played: settings, events, policies, diplomacy |
| [`world`](world.md) | game | game DB | What kind of world each game became: wars, conquest, nukes |
| [`tokens`](tokens.md) | model per player per game | telemetry | What each strategist cost, and how often its turns failed |
| [`agentic`](agentic.md) | tool call | telemetry | How strategists with a workspace used it |

## How they connect

```text
games ──game_id──> panel, turns, behavior, world, tokens, agentic
panel ──(game_id, player_id)──> turns, behavior, tokens, agentic
```

- `game_id` links every table to its game. `player_id` is the seat, so (`game_id`, `player_id`) is one player in one game.
- `player_type` (model plus strategist, for example `GPT-6.1-Sol-Simple-Agentic`) is composed once at extract and written into every player-level table, so analyses group by it directly.
- `experiment` is the condition a game was run under.

## Shared behavior

- **Extraction is incremental.** A game already in a table is skipped, so a new run adds only new games. Changing a table's columns rebuilds that table.
- **Removed games are pruned after confirmation.** When a game's files are gone, extract lists the game and asks before dropping its rows.
- **Broken files are recorded, not guessed.** A database that cannot be read is listed in `runs/import_issues.csv` and gets no rows; downstream stages leave that game out.
- **Where the tables live** is set in `data.tables`, and which ones extract rebuilds in `data.extract.outputs` ([configs/benchmark.md §3](../../configs/benchmark.md#3-data-extract--canonical-tables--global-filter)).
