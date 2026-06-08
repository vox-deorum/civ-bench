<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 3 — adjust (strength panel)

**Goal.** Turn an estimator's per-turn `predicted_win_probability` into the per-player-game **`adjusted_strength`** panel and register it as the named `strength` table that `ratings.*` (and some `performance.*`) consume. This is the bridge that makes ratings depend transitively on an estimator.

## Files to create / port

- `bench/adjust/registry.py` — name → adjust class (currently only `strength`).
- `bench/adjust/strength.py` — port `prepare_strength_data` from `../vox-deorum-analysis/performance/turn_predicted.ipynb` (also copied in `ratings/iterative_bt.py` — consolidate to one place here). Core pipeline (unchanged): late-game weighted average (`turn_progress_min`, `weight`) → relative-to-leader (`relative_to`) → winner enforcement (`enforce_winner`) → `logit_strength`.
- `bench/adjust/strength_lmm.R` — the lme4 shrinkage fit for the controlled-design cell baseline (below); invoked through the shared cross-platform `Rscript` locator (stage0/`bench/stats/`), the same one the R-backed ratings use.

## The adjustment step — civ (uncontrolled) vs matched start-cell (controlled)

After `logit_strength`, the panel is split by `controlled = (seed != -1 and seating_rotation != -1)`. Join game-level `seed`/`seating_rotation` from the `games` table by `game_id`; keep per-player `config_slot` from `panel_data` by `(game_id, player_id)`:

- **Uncontrolled rows** keep today's behavior: OLS-logit **civ** adjustment (`civ_adjust`, uses `bench/stats/`). `civ_adjust:"none"` ⇒ `adjusted_strength == relative_strength`. The per-civ effects it subtracts are always written to `<adjust dir>/civ_effects.csv` (below).
- **Controlled rows** take the **matched start-cell** path (benchmark.md §5.1), selected by `block` (`none|seed|start_cell|auto`; `auto` ⇒ `start_cell` when controlled rows exist). Derive `start_cell = (seed, player_id)`, then subtract a per-cell **Vanilla/VPAI baseline** via one of two pathways:
  - **Explicit baseline** (`baseline_experiment` set, e.g. a pure VP self-play condition where every seat is VPAI): estimate the per-cell baseline from the *designated* experiment's VPAI rows. **When that experiment has only one VPAI row per `(player, seat)` cell, use the observed value directly as the fixed per-cell baseline** (no shrinkage — the dedicated baseline run is ground truth for that cell); fall back to the shrinkage fit below when a cell has multiple rows.
  - **Implicit baseline** (`baseline_experiment` unset, default): estimate the per-cell baseline from **each experiment's own VPAI rows** with **shrinkage** — a variance-components model `logit_strength_vanilla ~ (1|seed) + (1|player_id) + (1|seed:player_id)` (drop seat/cell terms for `block:"seed"`). One game per `(seed, seat)`-cell-per-condition means the interaction is single-shot; shrinkage (not fixed dummies, which would be saturated and erase signal) is what makes the estimate sound. **Throw** if any controlled seat feeding the panel has no VPAI counterpart for its `(seed, seat)` cell. `baseline_source` pools VPAI rotations within the same condition first (cleanest match) or across conditions (`pooled`). `engine` = `r_lmer` (lme4, default, via `strength_lmm.R`) or `statsmodels` (MixedLM variance-components).
  - Subtract: `adjusted_strength = inv_logit(logit_strength − cell_baseline)`. The start-cell fixes the seat-bound civ, so this **replaces** civ adjustment for controlled rows. Optional final `post_cell_normalize` (`none|relative_to_leader`).
  - **Both pathways are computed every run.** The pathway selected by `baseline_experiment` feeds `adjusted_strength` (and enforces the throw above); the other is computed best-effort for the report only (coverage gaps shown, never fatal). Even fully controlled cells carry residual randomness — the shrinkage fit's variance components (σ²_seed/σ²_seat/σ²_cell/σ²_resid) quantify it and are reported.
- **Coverage — throw vs. warn.** *Hard error (selected implicit pathway only):* any controlled seat with no VPAI counterpart for its `(seed, seat)` cell aborts the stage, naming the offending `(experiment, seed, seat)` cells — a missing reference makes the comparison undefined (a deliberate, scoped departure from the otherwise "never abort" stance). *WARN, never abort (unchanged):* per-model cell coverage (using `config_slot` from `panel_data` and `seating_rotation` from `game_data`), connectedness-to-`Vanilla` (union-find; extrapolated models warned), and any coverage gap in the *non-selected* pathway computed only for the report. Outside the selected-implicit abort, keep all games and proceed.
- **Always-written adjustment trails (no config — like every other audit artifact).** The stage writes the intermediate per-group values it subtracts, next to the panel (the directory of `save`, default `reports/adjust/`): `civ_effects.csv` (per-`civilization`: `civilization, civ_effect, n_rows`, the civilization-level effect from the uncontrolled path) and `cell_baseline.csv` (the VPAI seating×seed effect from the controlled path: `experiment, pathway, seed, player_id, civilization, cell_baseline, n_vanilla, n_games, n_models, has_vanilla_baseline, vanilla_connected` + the fitted variance components — null for fixed-baseline explicit cells). `cell_baseline.csv` carries **both** pathways' rows (`pathway ∈ {explicit, implicit}`), each whenever it ran, so the report can compare them regardless of which fed downstream. `civ_effects.csv` is generated whenever the uncontrolled path ran. `performance.strength_panel` surfaces them in the report.

`bench/stats/` gains a `MixedLMResult` wrapper + a `fit_cell_baseline(...)` dispatcher alongside the OLS `RegressionResult`.

## Config wiring (`adjust[]`)

The stage `id` doubles as the table name. `uses.estimators` is required and single-source (point at `attention` per `benchmark.pretrained.template.json` / `benchmark.dev.json`). Saves to `<root>/adjust/player_strength_panel.csv`. New params: `block`, `baseline_source`, `post_cell_normalize`, `engine` (defaults `auto`/`same_condition_first`/`none`/ `r_lmer`) and **`baseline_experiment`** (string experiment id, default `null` ⇒ implicit per-experiment path; a value ⇒ explicit-baseline path). `baseline_experiment`, when set, must name a known experiment id (validated at config-load → `ConfigError`). With `block:"none"` (or no controlled rows) the stage is byte-identical to the civ-only pipeline.

## Done

`<root>/adjust/player_strength_panel.csv` exists with at least `game_id, player_id, player_type, civilization, adjusted_strength` (plus `experiment`, `controlled` = `seed != -1 and seating_rotation != -1`, and `seed`/`seating_rotation`/`config_slot` pass-through so the report can judge preliminary-ness), and is registered so a downstream `uses.tables:["strength"]` resolves and adds the edge. The intermediate trails `<root>/adjust/civ_effects.csv` and `<root>/adjust/cell_baseline.csv` are written automatically (both pathways for the latter, whichever path ran).

## Verification

- Column contract present (incl. `experiment`, `controlled` pass-through); winner rows have `relative_strength == 1.0` when `enforce_winner:true`.
- `block:"none"` reproduces the old `turn_predicted` output byte-for-byte on the same estimator.
- On controlled synthetic data with a known per-cell Vanilla baseline + a known uplift: the estimated baseline recovers the baseline, `adjusted_strength` recovers the uplift, and `ratings.vanilla_slot_effect` goes from significant (raw) to ~null (adjusted).
- **Explicit pathway:** a designated `baseline_experiment` (pure VP self-play) with one VPAI row per `(player, seat)` cell uses that value as the fixed baseline and recovers the injected uplift.
- **Implicit pathway:** **throws** when a controlled experiment has a seat with no VPAI counterpart; succeeds and recovers the baseline when every seat has one.
- **Both pathways** appear in `cell_baseline.csv` (`pathway ∈ {explicit, implicit}`) even though only the selected one feeds the panel; variance components are emitted and non-degenerate where a shrinkage fit ran.
- Non-selected-pathway / per-model / disconnected coverage gaps warn (never abort); `r_lmer` vs `statsmodels` baselines agree (~1e-2).
- The `civ_effects.csv` / `cell_baseline.csv` trails reconcile with the panel: subtracting the trail value (civ effect or cell baseline) from `logit_strength` reproduces `adjusted_strength`.
