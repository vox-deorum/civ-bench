# The `panel` table

**One row per major player per game.** It is each player's result: final score and rank, whether it won, how long it survived, how much its strategist intervened, and which victory it pursued. Where [`turns`](turns.md) follows the game as it unfolds, `panel` is the scorecard at the end.

- **File:** `runs/panel_data.csv` (set by `data.tables.panel`)
- **Built from:** the game database of each game
- **Read by:** the `strength` adjust stage (as the source of identity and seating), and most `ratings.*`, `prediction.*`, `performance.*`, `behavior.*`, and `world.*` analyses

## The ideas behind it

| Group | What it captures | Example question it answers |
| --- | --- | --- |
| **Game settings** | Map, size, difficulty, speed, final turn, how the game was won and by whom | Which games ended in a science victory? |
| **Who played** | The seat, the composed `player_type`, the raw model and strategist names, the civilization, and the controlled seating slot | Which model played Rome in this game? |
| **Result** | Best score, its rank, its ratio to the game's best, last turn alive, and whether the player won | How close did this player come? |
| **Intervention** | How many times the strategist acted, and how many times it changed flavors, persona, or research | How hands-on was this model? |
| **Victory focus** | The share of its life the player spent on each grand strategy: conquest, culture, diplomacy, science | What did this model try to win by? |

## Key conventions

- **Only major players.** City-states have no rows.
- **Game settings repeat on every player's row**, so the table can be read without joining.
- **`N/A` means unknown.** Missing metadata reads `N/A`. Controlled seating uses `-1` for seats outside the design.
- **Early cultural wins are relabeled.** A Cultural victory by turn 300 is recorded as Survival.

## Reference: exact rules

Rows are unique by (`game_id`, `player_id`). If the database cannot be read, the game has no rows. If one player cannot be read, that player's own fields read `N/A`.

| Columns | Meaning and source |
| --- | --- |
| `experiment`, `game_id` | The condition folder and the game. |
| `turn`, `map_type`, `map_size`, `difficulty`, `game_speed`, `victory_type`, `victory_player_id` | From `GameMetadata`. `turn` is the final turn. |
| `player_id`, `player_type`, `model`, `strategist`, `civilization` | The seat, the composed identity, the raw `model-<id>` and `strategist-<id>` metadata it came from (unmarked seats read `VPAI`), and the civilization. |
| `config_slot` | The seat's slot in the controlled seating design, from `GameMetadata.seatingMap`; `-1` outside it. |
| `score`, `score_rank`, `score_ratio` | The player's highest `PlayerSummaries.Score`, its rank among major players (1 is best), and its ratio to the game's highest score. |
| `survival_turn`, `is_winner` | The player's last turn (`PlayerSummaries.IsLatest`), and 1 for the winner. |
| `decisions`, `strategy_changes` | `FlavorChanges` rows where the strategist acted, and the subset that changed a flavor value. Older games fall back to `StrategyChanges`. |
| `persona_changes`, `research_changes` | Distinct persona states the strategist set (rows written by the in-game AI are skipped), and research changes that changed something. |
| `domination_ratio`, `culture_ratio`, `diplomatic_ratio`, `science_ratio` | The share of the player's life under each `StrategyChanges.GrandStrategy` (Conquest, Culture, UnitedNations, Spaceship). Each strategy lasts until the next change. |
