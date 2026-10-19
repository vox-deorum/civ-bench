# The `tokens` table

**One row per model per player per game.** It records what a strategist cost to run, in tokens and tool calls, and how often its turns failed. It is built from the per-player telemetry, not the game database.

- **File:** `runs/model_token_usage.csv` (set by `data.tables.tokens`)
- **Built from:** the per-player telemetry databases (`<game>-player-<n>.db`)
- **Read by:** `performance.usage_efficiency` (cost and efficiency), `performance.experiment_completeness`, and the decision-failure filter that every stage applies

## The ideas behind it

| Group | What it captures | Example question it answers |
| --- | --- | --- |
| **Who** | The player, its `player_type`, and the model as reported by telemetry | Which model ran this seat? |
| **Tokens** | Input, cached input, reasoning, and output tokens, plus how much of the input repeated from one step to the next | How expensive is this model per game? |
| **Activity** | Tool calls the strategist made | How busy is the strategist? |
| **Failures** | How many turns the strategist ran, how many failed, and which ones | Is this model reliable enough to keep the game? |

## Key conventions

- **A retried turn counts once**, using its latest attempt. The [`agentic`](agentic.md) table follows the same rule.
- **Failures gate games.** A game whose worst player failed too many turns (`max_decision_failure_pct`, default 20%) is dropped from every estimator, adjust, and analysis stage.
- **Some providers under-report.** Reasoning and cached tokens are kept raw here; the usage analyses estimate what is missing (see [configs/benchmark.md](../../configs/benchmark.md#performanceusage_efficiency-model-usage-and-efficiency)).
- **Players without telemetry tokens still get a row** when needed: an in-game AI seat reads `Vanilla`, and a player with failed turns but no recorded model call reads `Unattributed`, both with zero tokens.

## Reference: exact rules

Rows are unique by (`game_id`, `player_id`, `model_base`). A player that used several models gets one row per model, and its turn counts repeat on each.

| Columns | Meaning and source |
| --- | --- |
| `experiment`, `game_id`, `player_id`, `player_type` | The condition folder, the game, the seat (from the telemetry file name), and the identity from the game database (`N/A` if that cannot be read). |
| `model_name`, `model_base`, `model_variants`, `agent_names` | The catalog name of the model, the normalized raw model string, the raw strings seen, and the agents that ran. |
| `input_tokens`, `cached_input_tokens`, `reasoning_tokens`, `output_tokens`, `total_tokens` | Sums of the token attributes over each turn's agent runs. `total_tokens` is input plus reasoning plus output. |
| `repeated_input_tokens` | For each run, the input that overlaps between consecutive steps: the sum of the smaller of each pair of step inputs. |
| `reasoning_recorded_tokens`, `reasoning_recorded_output_tokens` | Reasoning and output tokens from calls whose reasoning looks fully reported (more reasoning than output). |
| `tool_count`, `focus_briefer_count` | Tool calls inside agent runs, and how many of them were the `focus-briefer` helper. |
| `valid_turn_count`, `failed_turn_count`, `failure_pct`, `failed_turns` | Turns with a `strategist.turn.*` trace, those that ended in error, their ratio, and the failed turn numbers. |
