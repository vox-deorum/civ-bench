# About civ-bench: motivation and design

This page covers *why* `civ-bench` exists and the principles behind its shape. For what it does and how to run it, see the [README](../README.md) and the [Getting Started guide](getting-started.md).

This repository is the reusable, config-driven reimplementation of the analysis used in the CivBench paper:

> Chen, Cheng, Gurkan, and Lin. *CivBench: Progress-Based Evaluation for LLMs' Strategic Decision-Making in Civilization V.* arXiv:2604.07733 ([abstract](https://arxiv.org/abs/2604.07733)).

---

## What problem does this actually solve?

When you have several LLMs, or several variants of the same LLM with different prompts and strategist scaffolds, play many games of Civ V, you end up with a pile of game databases and a hard question: **who is actually better, and how confident can you be about it?**

The paper puts the core difficulty plainly: *"terminal win/loss is too sparse a signal in games spanning hundreds of turns and multiple opponents."* A single win or loss at the end of a 300-turn, multi-opponent game tells you very little, and map luck, starting position, and civilization choice all swing the outcome. So you cannot just count wins.

CivBench's answer is **progress-based evaluation**: instead of waiting for the final result, it *"trains models on turn-level game state to estimate victory probabilities throughout play."* `civ-bench` is the harness that runs that whole pipeline. In broad strokes it:

- **Extracts** every game into clean, canonical tables (one row per turn, one row per player-game, and so on).
- **Estimates victory probability** at each turn using a small zoo of models, from a simple score transform up to a multi-head self-attention network (the paper's primary estimator, AttentionMLP, reaches ROC-AUC 0.865).
- **Distills skill** from those per-turn probabilities into a single per-player "strength" number, correcting for confounds like which civilization a player was handed or where they started on the map.
- **Rates the players** against each other with Bradley-Terry and Plackett-Luce models (the statistics behind Elo and ranked-choice ratings), centered on the stock-AI baseline, complete with confidence intervals.
- **Validates the predictor itself** along the paper's three axes (predictive, construct, and convergent validity): is it accurate, are its learned signals strategically plausible, and do independent estimators agree?
- **Profiles strategy and cost**, surfacing the distinct strategic profiles (domination, science, culture, diplomacy) the paper highlights, plus token cost so you can weigh strength against price.
- **Writes it all up** into a report you can hand to someone.

In the study that introduced it, CivBench was run over **307 games with 7 LLMs** across multiple agent conditions, and it held up as an *"unsaturated benchmark"* that reveals *"distinct strategic profiles not visible through outcome-only evaluation."*

---

## The three ideas that shape everything

Three rules shape the project. The full repository conventions are in [AGENTS.md](../AGENTS.md).

1. **Config over code.** Anything that changes between datasets, experiments, or strategist line-ups lives in a JSON file under [configs/](../configs/), never hardcoded. Adding a new player type or experiment requires *zero* Python edits.
2. **Modular and pluggable.** Every analysis (ratings, prediction, calibration, performance, exploratory) is a self-contained unit behind a common interface and a registry. Adding one means writing a module, registering it, and referencing it from config, touching nothing else.
3. **Reports are generated, never authored.** No notebook is the source of truth for a result. If a number cannot be reproduced by running `civ-bench`, it does not belong in the repo.

The [developer guide](development.md) explains how these rules apply to modules and config validation.
