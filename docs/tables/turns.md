# The `turns` table

**One row per major player per turn.** It is the game's state as each player saw it, turn by turn: score, cities, economy, military, diplomacy, and the strategist's current settings. It is the largest table and the one the prediction models learn from.

- **File:** `runs/turn_data.csv` (set by `data.tables.turns`)
- **Built from:** the game database of each game
- **Read by:** every estimator (to train and predict win probabilities), `world.*` (progress and conflict), and `behavior.commitment`

## The ideas behind it

| Group | What it captures | Example question it answers |
| --- | --- | --- |
| **Standing** | Score, rank, and the best score that turn | Who was leading on turn 150? |
| **Empire** | Cities, population, territory, technologies, policies | How fast did this civ grow? |
| **Economy** | Gold, production, food, science, culture, faith, tourism per turn | Was it a science-heavy game? |
| **Military** | Strength, units, supply | Was this player building an army? |
| **Diplomacy** | Active wars, war weariness, truces, friendships, defensive pacts, city-state allies, World Congress votes | How isolated was this player? |
| **Religion** | Share of the world's cities following the player's religion | Did its religion spread? |
| **Strategist settings** | The flavors and grand strategy in effect, and whether the strategist acted or changed anything this turn | How often does the model actually change course? |
| **Outcome** | Whether the player eventually won | (The label the prediction models learn) |

## Key conventions

- **Every turn has a row for every major player**, including players already eliminated. Their rows are mostly zeros, which is how the models see elimination.
- **Settings carry forward.** A flavor stays at its last value until the strategist changes it. Before the first setting it is blank.
- **Acting is not the same as changing.** A strategist that reviews the game and keeps everything as is has made a decision (`is_decision`) without a change (`is_changed`).
- **Replayed turns count once.** If a turn was written twice, the latest record wins.
- **No seed here.** Map seed and seating live in [`games`](games.md) and are joined by `game_id` where needed.

## Reference: exact rules

**Row set.** Every distinct `PlayerSummaries.Turn` in the game, crossed with every major player (`PlayerInformations.IsMajor = 1`). A game with no major players or no summaries gives no rows. Rows are unique by (`game_id`, `player_id`, `turn`).

| Columns | Meaning and source |
| --- | --- |
| `experiment`, `game_id`, `player_id`, `player_type`, `civilization` | The condition folder, the game, the seat, the composed identity ([configs/benchmark.md §3.3](../../configs/benchmark.md#33-player-identity-the-orthodox-player_type)), and the civilization. |
| `turn`, `max_turn` | This turn and the game's last turn. |
| `score`, `cities`, `population`, `territory`, `technologies`, `military_strength`, `military_units`, `military_supply`, `gold`, `gold_per_turn`, `happiness_percentage`, `culture_per_turn`, `science_per_turn`, `tourism_per_turn`, `faith_per_turn`, `votes` | Same-named `PlayerSummaries` columns from the latest row (highest `ID`) of each player and turn. Missing values are 0. |
| `rank`, `max_score` | Rank by score among living players (ties share a rank; eliminated players all get the worst rank), and the best living score that turn. |
| `production_per_turn`, `food_per_turn` | Summed over the player's cities (`CityInformations`), using the latest row (highest `ID`) for each city and turn before the sum. |
| `policies` | Number of policies adopted, from `PlayerSummaries.PolicyBranches`. |
| `religion_percentage` | Percent of all cities whose majority religion is the one this player founded; 0 when it founded none. |
| `minor_allies` | City-states allied to this player. |
| `highest_war_weariness`, `active_wars`, `truces`, `friendships`, `defensive_pacts` | Parsed from `PlayerSummaries.Relationships`. |
| `is_winner` | 1 for the player named in `GameMetadata.victoryPlayerID`. |
| `is_decision`, `is_changed` | `is_decision` is 1 when the strategist acted this turn, including a keep-as-is with a rationale. `is_changed` is 1 when it changed at least one flavor value. |
| `flavor_*` (34 columns) | The flavor in effect from `FlavorChanges`, carried forward; blank before the first setting. |
| `grand_strategy` | From `FlavorChanges`, carried forward, falling back to `StrategyChanges` in older games. |
| `rationale` | The strategist's stated reason, only on the turn it was written. |

When two `FlavorChanges` rows share a turn, a real change beats a no-change row, otherwise the later one wins, and the displaced row moves to the turn before.
