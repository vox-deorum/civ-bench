# AGENTS.md

NEVER STAGE YOUR CHANGES UNLESS EXPLICITLY ASKED! However, if a change gets externally staged, it is from the human reviewer. 

When asking questions, come with a clear, plain description with an example. Do not assume the owner knows every detail in your context. DO NOT ASK asynchronous questions.

## Use Subagents When Appropriate

Delegate less critical/lower-level BATCH work to subagents with less capabilities for exploring/batch editing. Always designate a model for subagents and report which model (or tool) you used in response text. Such work may involve exploring repo structure, finding references, summarizing information, or conducting less sophisticated edits in batches.

Use OpenCode delegation if such a skill exists, with clear, bounded instructions. If OpenCode does not work, switch back to native subagents.

DO NOT use weak models for complex diagnosis. For independent review, use OpenCode. For exploration and simple implementation task:
- Claude Code: always delegate to OpenCode. Never use Sonnet or Haiku.
- Codex: always delegate to OpenCode or GPT-6-Luna. Never use Sol.

## Writing Style

Use plain, natural language in documentation, comments, commit messages, release notes, and responses. Use lists, tables, or diagrams to improve clarity. Do not use em-dashes. These rules also apply to delegates. Rewrite documentation and plans for a coherent final result. Include revision history only when requested, and comparisons only when they help the reader make a decision.

## Core rules

1. **Config over code.** Put dataset, experiment, model, filter, grouping, and report choices in JSON under `configs/`. Adding a model or experiment must not require Python changes.
2. **Keep stages modular.** Each analysis implements the shared interface and has one registry entry and one config block. Put variants such as bootstrap confidence intervals and per-strategy ratings in the parent module's parameters.
3. **Generate reports.** Results must be reproducible with `civ-bench run`. Analyses return structured data, and the report layer renders it.

## Architecture

The pipeline is a directed acyclic graph with five stage types:

```text
extract -> estimators -> adjust -> analyses -> report
```

- `extract` converts game databases into canonical CSV files and builds `player_type` from per-player metadata.
- `estimators` train or load predictors and emit win probabilities.
- `adjust` creates derived tables. The `strength` module owns the shared `adjusted_strength` calculation.
- `analyses` consume canonical or derived tables and return `AnalysisResult` objects.
- `report` renders saved analysis results as Markdown and HTML.

Dependencies come from stage order, explicit `needs`, and references in `uses`. Update [configs/benchmark.md](configs/benchmark.md) whenever a selectable module or config field changes.

The main directories are:

| Path | Purpose |
| --- | --- |
| `bench/` | Python package and pipeline implementation |
| `configs/` | Run-spec templates, catalogs, and schema documentation |
| `docs/` | User and developer guides |
| `tests/` | Tests built on small synthetic fixtures |
| `pretrained/` | Tracked reference model snapshots |
| `runs/` | Local inputs and extracted tables, ignored by Git |
| `reports/` | Generated outputs, ignored by Git |

For a fuller package map, see [docs/development.md](docs/development.md).

## Development conventions

- Use absolute imports from `bench.*` and run commands from the repository root.
- Keep the config layer import-light. Load heavy analysis and estimator dependencies only on execution paths.
- Validate unknown keys, required fields, types, registry entries, and stage references when loading config.
- Keep estimators under `bench/estimators/`, derived-table producers under `bench/adjust/`, and analyses under `bench/analyses/`.
- Keep file writing and console output in the runner, CLI, or report layer. Analyses return data.
- Thread the configured seed through every random operation. The same config and input data must produce byte-stable tables.
- Import required dependencies directly. Missing Python or R packages must stop the run with an install hint.
- Keep machine-specific paths and experiment data out of Git.

When adding a module or validation rule, update its schema documentation and tests in the same change. Tests must not read machine-specific data or execute against real data under `runs/`.

## Config files

Tracked `configs/benchmark*.template.json` files are examples. Copy one to a local `configs/benchmark*.json` file and set the machine-specific paths there. Local run specs are ignored by Git. [configs/benchmark.md](configs/benchmark.md) defines the complete format.

## Common commands

```powershell
scripts/install.ps1
civ-bench extract --config configs/benchmark.dev.json
civ-bench run --config configs/benchmark.dev.json
civ-bench report --config configs/benchmark.dev.json
civ-bench fix --config configs/benchmark.dev.json --dry-run
pytest
```

Use `--only <stage-id>` to run one stage with its dependencies. Use `--skip <stage-id>` to omit a stage. After `civ-bench fix`, run extraction with `--force-rebuild` to refresh the canonical tables. Use `--no-publish` with `run` or `report` to skip the offer to commit and push the rendered report.

All dependencies are installed up front. Use `scripts/install.sh` on Linux or macOS. `Rscript` must be on `PATH` or set through `CIV_BENCH_RSCRIPT`.