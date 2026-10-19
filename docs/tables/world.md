# The `world` table

**One row per game.** It describes what kind of world a game turned into: how many wars, how much conquest, whether nukes flew, how many civilizations died. The [`behavior`](behavior.md) table credits each event to the player who caused it. This table counts each event once for the whole game.

- **File:** `runs/world_data.csv` (set by `data.tables.world`)
- **Built from:** the game database of each game
- **Read by:** the `world.*` analyses (world report pages)

## The ideas behind it

| Group | What it captures | Example question it answers |
| --- | --- | --- |
| **Game size** | How long the game ran and how many major civs played | Did this game end early? |
| **Conflict** | Wars between civs, wars on city-states, and the most wars open at the same time | Do games with this model turn into world wars? |
| **Conquest and losses** | Cities and capitals captured, cities razed, units killed, civs eliminated | How destructive was the game? |
| **Nuclear use** | Nukes built and detonated | Did anyone go nuclear? |

## Key conventions

- **A war counts once, not once per side.**
- **"At the same time" is replayed from the game.** The peak war count follows declarations and peace deals in turn order. A war ends at peace, or once every civ on one side is gone.
- **A civ is eliminated** when it has no positive score on the final turn.

## Reference: exact rules

**Columns.** `experiment`, `game_id`, `final_turn` (the last `PlayerSummaries` turn), `n_civs` (major players), then the totals `wars_declared`, `wars_on_city_states`, `peak_wars`, `units_lost`, `units_killed`, `cities_captured`, `capitals_captured`, `city_state_cities_captured`, `civs_eliminated`, `cities_razed`, `nukes_built`, and `nukes_detonated`.

- The counts follow the [behavior event rules](behavior.md#reference-exact-rules), counted once per game.
- Two counters differ from their behavior namesakes. `wars_declared` counts only aggressor declarations on a major civ. `wars_on_city_states` counts aggressor declarations on city-states.
- `peak_wars` and `civs_eliminated` read only the latest `PlayerSummaries` row (highest `ID`) of each player and turn, as the `turns` table does, so a replayed turn's score of 0 overrides an earlier positive one. A civ is gone after its last turn with a positive score.
