# Agents Guide

This file is for AI coding agents working on the CivBench repository. It captures the writing style we want and gives a quick orientation to the project. NEVER STAGE YOUR CHANGES UNLESS EXPLICITLY ASKED! However, if a change gets externally staged, it is from the human reviewer.

`civ-bench` is a modular, JSON-configurable Python **benchmark harness** for LLM strategists in *Civilization V: Vox Populi* (via the Vox Deorum platform). Point it at raw game-run data, hand it a config, and it produces a report — extraction, analysis, and rendering all driven by JSON.

## Use Subagents Whenever Appropriate

ALWAYS delegate less critical/lower-level BATCH work to subagents with less capabilities, e.g., from Claude Fable to Sonnet/Haiku, or from GPT Sol to Terra (reviewing/implementing)/Luna (exploring/batch editing). Report which model you used to spawn that agent in response text. Such work may involve exploring repo structure, finding references, summarizing information, or conducting less sophisticated edits in batches.

## Writing Style

Write everything in natural language: docs, code comments, commit messages, release notes, console output, and the AGENTS.md files themselves. Keep the prose plain and easy to follow. Bullets, subbullets, and tables are encouraged wherever they make the content easier to scan. Do not use em-dashes anywhere. Reach for a colon, a comma, parentheses, or two separate sentences instead. Every agent working in this repo must follow this rule.

Do not produce layered writings (e.g., instead of X we chose to do Y) that document revision histories, unless explicitly instructed to do so. A reader needs the rule, not the case for it. State what is true and stop there, maximize the language efficiency. **Readability is the top priority.**

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

Use `--only <stage-id>` to run one stage with its dependencies. Use `--skip <stage-id>` to omit a stage. After `civ-bench fix`, run extraction with `--force-rebuild` to refresh the canonical tables.

All dependencies are installed up front. Use `scripts/install.sh` on Linux or macOS. `Rscript` must be on `PATH` or set through `CIV_BENCH_RSCRIPT`.

## Tool use

- Use dedicated read, edit, search, and file-listing tools when available.
- Pass the working directory to shell tools. Do not change directories inside a command.
- Use repository-relative paths in shell commands.
- Prefer PowerShell syntax on Windows.
- Do not stage changes unless the user explicitly asks.
