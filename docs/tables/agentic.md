# The `agentic` table

**One row per tool call a strategist made while deciding a turn**, for strategists that were given a workspace. It records how a model used that workspace: what it looked up, whether it kept notes, how often its commands failed, and whether it read about a choice before making it.

- **File:** `runs/agentic_data.csv` (set by `data.tables.agentic`)
- **Built from:** the per-player telemetry databases (`<game>-player-<n>.db`)
- **Read by:** nothing yet; the agentic report family is planned

## The workspace

Some conditions (today the `-files-per-5` ones, labeled Agentic) give the strategist a shell next to its usual game actions. The shell sees four folders:

| Folder | The model can | Holds |
| --- | --- | --- |
| Records | read | Reports of past and current turns: events, options, cities, rivals |
| Reference | read | The game rules: technologies, policies, buildings, units |
| Notes | read and write, kept all game | Whatever the model writes down, plus an editable guide (`AGENTS.md`) |
| Scratch (`/tmp`) | read and write | Temporary files |

Anything written elsewhere vanishes after the command.

## The ideas behind it

A strategist decides each turn in a **run**: a short loop of **steps**. In each step the model replies once and may make several **calls**. A call is either a **workspace command** (a shell command) or a **game action** (such as choosing research). The table has one row per call, so it can answer questions at every level:

| Group | What it captures | Example question it answers |
| --- | --- | --- |
| **Engagement** | Which calls are workspace commands and which are actions, step by step | Does the model look things up, or act straight away? |
| **Foraging** | Which folder and which kind of record or rule each command read, how much it printed, and how far back in the game it looked | Does the model read the rules, or only this turn's events? |
| **Memory** | What each command wrote and where it landed: kept, scratch, lost, or refused | Does the model keep notes, and do they survive? |
| **Reliability** | Whether a command worked, and why not if it failed | How often do its commands break? |
| **Grounding** | For research and policy choices, whether the model had looked up that option before choosing it | Are the model's choices informed by the rules? |

## Key conventions

- **Conditions without a workspace add no rows.** A player is included only if its telemetry shows a workspace, so a new condition with one is picked up automatically and the others stay out.
- **Every run is represented.** A run that ended before any call, for example on a timeout, still gets one row, so per-run rates have the right denominator.
- **Paths are read, not run.** The table reads each shell command to work out the files it touched. It follows simple variables, loops, and brace lists, but a path built while the command runs is missed. The numbers are close, not exact.
- **"No match" is not an error.** A search that finds nothing is its own outcome, separate from failures.
- **Grounding is a lower bound.** A choice counts as grounded only if the option's own entry (a heading or a name) appeared in an output from an earlier step of the same run. Reading it in an earlier turn, or only seeing it mentioned in passing, does not count. Grounding shows the information was in front of the model, not that the choice was good.

## Reference: exact rules

**Which calls count.** Only the latest `strategist.turn.*` trace of each turn counts, as in the [`tokens`](tokens.md) table. Host calls made outside a run, such as the prefetched `get-*` reports, are not rows. A player is included when its first agent spans show a workspace: a step span with `host.capability`, or an agent span whose `context.files` describes workspace folders (runs without one record `false`). Only the first spans are checked, so the many players without a workspace cost little to skip.

**Columns.**

| Columns | Meaning |
| --- | --- |
| `experiment`, `game_id`, `player_id`, `player_type`, `turn` | The player and the decision turn. |
| `agent`, `run`, `step`, `call` | The agent (for example `simple-strategist`), the run's position among the player's runs (1-based, by start time), the step number, and the call's position in the run (by step, then start time). `step` and `call` are blank on a `none` row. |
| `tool`, `kind`, `capability` | The tool name. `kind` is `workspace` (a `simple-tool.*` call), `action` (an `mcp-tool.*` call), `helper` (a host helper such as `focus-briefer`), or `none` (a run without calls). `capability` is the step's `host.capability`, such as `write`. |
| `option`, `action_ok`, `grounded` | Actions only. `option` is the technology or policy chosen by `set-research` or `set-policy`. `action_ok` is 1 when the action reported success. `grounded` is 1 when a workspace output from an earlier step showed `option` as a Markdown heading or a JSON `"Name"`. Outputs of the action's own step do not count, since they came from the same reply. A heading behind a `grep -n`, multi-file `grep`, or `cat -n` prefix (`modern.md:12:## Plastics`) counts. |
| `exit_code`, `outcome`, `out_chars`, `truncated_chars`, `duration_ms` | Workspace calls only. `outcome` is `ok`, `no_match` (exit 1 with nothing on stderr), `read_only`, `missing_file`, `syntax`, `not_found` (exit 127 or "command not found"), or `other`. `out_chars` is the length of stdout and `truncated_chars` the characters the host cut from it. |
| `read_style` | `full` (a whole file reached the output), `targeted` (part of a file, or a count, through `grep`, `sed`, `awk`, `head`, `wc`, a piped filter, and similar), `listing` (names or existence only), or `none`. |
| `src_records`, `src_reference`, `src_notes`, `src_guide`, `src_tmp`, `src_other` | 1 when the call read that area. `src_other` is any other path under `/workspace`. |
| `rec_*` | 1 when the call read that record section: `catalog`, `situation`, `civilization`, `events_all`, `events_economic`, `events_military`, `events_diplomatic`, `events_others`, `options`, `strategies`, `victory`, `players`, `cities`, `military`, or `other`. A turn folder alone names no section; a glob names every section it matches. |
| `ref_technologies`, `ref_policies`, `ref_buildings`, `ref_units`, `ref_other` | 1 when the call read that reference category. |
| `lookback` | The decision turn minus the oldest `turn-<N>` folder read, counting only folders at or before the decision turn; blank when none was read. |
| `writes_persisted`, `writes_scratch`, `writes_lost`, `writes_rejected` | Paths written, by where they landed: kept in `/workspace/game`, scratch in `/tmp`, lost (any other path, including relative ones), or rejected (the read-only folders, or a failed write to `/workspace/game`). |
| `guide_edit`, `note_missing` | `guide_edit` is 1 when the call changed `AGENTS.md`. `note_missing` is 1 when it read a note that did not exist. |

Workspace columns are blank on action rows, and action columns are blank on workspace rows. Variables, `for` lists, and brace lists expand to at most 64 values each, and only the file arguments of known commands count as paths.
