# The `behavior` table

**One row per major player per game.** It describes how a player played: how it tuned the game AI it steers, what it did on the map, which policies it took, and how it treated other civilizations. Outcome tables say *how well* a player did. This one says *what it did*.

- **File:** `runs/behavior_data.csv` (set by `data.tables.behavior`)
- **Built from:** the game database of each game
- **Read by:** the `behavior.*` analyses (behavior report pages)
- **Configured by:** `data.extract.behavior` in the run spec ([configs/benchmark.md §3.0](../../configs/benchmark.md#30-canonical-tables-and-dataextractbehavior)), which picks which columns exist

## The ideas behind it

An LLM strategist does not move units. It steers the in-game AI through a few levers and makes some choices directly. The table summarizes each of them over the whole game:

| Group | What it captures | Example question it answers |
| --- | --- | --- |
| **Flavors** | The weights the strategist gives the AI's priorities, such as Offense, Science, or Expansion, on a 0 to 100 scale | Does this model lean military? |
| **Persona** | The AI's personality settings, such as Boldness or WarBias | Does this model make the AI more aggressive in diplomacy? |
| **Events** | Counts of things that happened: wars declared, cities captured, units lost, nukes built | How warlike was the game this player made? |
| **Policies** | How often the player changed policies, and when it first adopted each policy branch | When does a model commit to an ideology? |
| **Relationships** | How the player set its public and private stance toward each rival | Does the model smile in public while planning hostility? |

## Key conventions

- **Settings are summarized over time, not sampled once.** A flavor or persona value is the one in effect on each turn. The table keeps its lowest, average, and highest value over the player's life, so a model that spikes Offense for five turns differs from one that keeps it high all game.
- **Some flavors count only once they matter.** A nuclear flavor set before the player can build nukes says little, so it is summarized from the turn the player gains access. Players who never reach that point get blanks.
- **Events belong to whoever caused them.** A war counts for the declarer, a capture for the new owner.
- **Counts and levels are compared differently.** Analyses scale counts by how long the player lived, so a player eliminated early is not compared on raw totals.
- **Blank means "does not apply" or "no data".** For example, the in-game AI never sets flavors, so its flavor cells are blank.

## Reference: exact rules

These rules make the numbers reproducible. They matter when you check a surprising value, not when reading a report.

**Columns.** The key columns are `experiment`, `game_id`, `player_id`, `player_type`, `civilization`, and `survival_turn` (the player's last recorded turn). Flavor and persona columns are `flavor_<name>_<stat>` and `persona_<name>_<stat>`, for example `flavor_use_nuke_max`. Each selected event, policy, and relationship name gets its own column.

**Flavors and persona.**
- Values come from `FlavorChanges` and `PersonaChanges`. The last row of a turn wins and carries forward until the next row.
- `min` and `max` range over the states in effect from the player's first row to `survival_turn`. `avg` is turn-weighted.
- Rows written by the in-game AI (`Tweaked by In-Game AI`) count, because the game used them.
- A player with no rows gets blank cells. If an older database lacks a column, only that column is blank.

**Access gates** (`flavor_gates`). A gated flavor is summarized only from its access turn on, starting with the state in effect at that turn.
- A tech gate opens on the player's first `PlayerSummaries` turn whose `CurrentResearch` starts with one of the listed techs.
- An era gate opens on the first turn whose `PlayerSummaries.Era` is that era or later (Ancient, Classical, Medieval, Renaissance, Industrial, Modern, Atomic, Information).
- By default `Nuke` and `UseNuke` wait for `Nuclear Fission`, `Satellites`, or `Advanced Ballistics`, and `Air`, `Antiair`, `AirCarrier`, and `Airlift` wait for the Modern era.

**Events.** Counts come from `GameEvents`. An exact duplicate (same type, turn, and payload) counts once.

| Column | Counts |
| --- | --- |
| `wars_declared` | `DeclareWar` events the player started as aggressor, including the automatic declarations on the target's defensive-pact partners, but not wars joined as a vassal |
| `wars_received` | `DeclareWar` events against the player's team, from anyone |
| `wars_on_city_states` | Aggressor declarations on a city-state (also in `wars_declared`) |
| `cities_nuked` | `NuclearDetonation` events by the player on a plot with a city |
| `nukes_detonated` | Every `NuclearDetonation` by the player |
| `nukes_built` | Atomic Bombs and Nuclear Missiles the player trained or bought; the free Manhattan Project bomb does not count |
| `cities_razed` | `CityRazed` events by the player |
| `units_lost`, `units_killed` | `UnitKilledInCombat` events where the player's unit died, or made the kill against anyone |
| `cities_captured` | Conquests where the player took a major civ's city |
| `capitals_captured` | The subset where that city was the old owner's capital at the time |
| `city_state_cities_captured` | Conquests of city-state cities |

**Policies.** `policy_changes` counts `PolicyChanges` rows that really change something, not rationale-only or empty rows. Each branch column holds the first adoption turn from `PlayerAdoptPolicyBranch` or `IdeologyAdopted`, or `N/A` if never adopted.

**Relationships.** These come from `RelationshipChanges`, tracked per (player, rival) pair. Only major-civ rivals count. The last row of a turn wins and carries forward.
- `relationship_changes` counts rows that change the public or private value, starting from 0 and 0. `relationship_targets` counts the distinct rivals ever set.
- The stance averages weight each pair-turn once, from the first row on that rival until either player's last turn. `stance_net_avg` is public plus private.
- `stance_masked_hostility_share` is the share of pair-turns with public above 0 and private below 0. `stance_masked_goodwill_share` is the reverse.
- When the table exists but a player set nothing, the counts are 0 and the averages are blank. When the table is missing, every cell is blank. Values are kept raw; the documented range is -100 to 100.

**Column kinds.** Analyses treat each column by kind: `count` (events, `policy_changes`, `relationship_changes`; scaled by turns alive), `level` (flavor and persona stats, `relationship_targets`, stance averages), `share` (stance shares), and `turn` (first-adoption turns).
