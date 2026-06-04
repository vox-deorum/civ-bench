<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 3 — adjust (strength panel)

**Goal.** Turn an estimator's per-turn `predicted_win_probability` into the per-player-game
**`adjusted_strength`** panel and register it as the named `strength` table that `ratings.*` (and some
`performance.*`) consume. This is the bridge that makes ratings depend transitively on an estimator.

## Files to create / port

- `civ_bench/adjust/registry.py` — name → adjust class (currently only `strength`).
- `civ_bench/adjust/strength.py` — port `prepare_strength_data` from
  `../vox-deorum-analysis/performance/turn_predicted.ipynb` (also copied in `ratings/iterative_bt.py` —
  consolidate to one place here). Core pipeline (unchanged): late-game weighted average (`turn_progress_min`,
  `weight`) → relative-to-leader (`relative_to`) → winner enforcement (`enforce_winner`) → `logit_strength`.
- `civ_bench/adjust/strength_lmm.R` — the lme4 shrinkage fit for the controlled-design cell baseline (below);
  invoked through the shared cross-platform `Rscript` locator (stage0/`civ_bench/stats/`), the same one the
  R-backed ratings use.

## The adjustment step — civ (uncontrolled) vs matched start-cell (controlled)

After `logit_strength`, the panel is split by `controlled = (seed != -1 and seating_rotation != -1)` (joining
`seed` from the `games` table by `game_id`):

- **Uncontrolled rows** keep today's behavior: OLS-logit **civ** adjustment (`civ_adjust`, uses
  `civ_bench/stats/`). `civ_adjust:"none"` ⇒ `adjusted_strength == relative_strength`.
- **Controlled rows** take the **matched start-cell** path (benchmark.md §5.1), selected by `block`
  (`none|seed|start_cell|auto`; `auto` ⇒ `start_cell` when controlled rows exist):
  - Derive `start_cell = (seed, player_id)`.
  - Estimate the per-cell **Vanilla/VPAI baseline** from Vanilla rows only, with **shrinkage** — a
    variance-components model `logit_strength_vanilla ~ (1|seed) + (1|player_id) + (1|seed:player_id)`
    (drop seat/cell terms for `block:"seed"`). One game per `(seed, seat)`-cell-per-condition means the
    interaction is single-shot; shrinkage (not fixed dummies, which would be saturated and erase signal) is
    what makes the estimate sound. `baseline_source` pools VPAI rotations within the same condition first
    (cleanest match) or across conditions (`pooled`). `engine` = `r_lmer` (lme4, default, via
    `strength_lmm.R`) or `statsmodels` (MixedLM variance-components).
  - Subtract: `adjusted_strength = inv_logit(logit_strength − cell_baseline)`. The start-cell fixes the
    seat-bound civ, so this **replaces** civ adjustment for controlled rows.
  - Optional final `post_cell_normalize` (`none|relative_to_leader`).
- **Coverage diagnostic (WARN, never abort):** cells with no Vanilla baseline (still predicted via the
  seed/seat margin — flagged), per-model cell coverage (from `config_slot`/`seating_rotation`), and
  connectedness-to-`Vanilla` (union-find; extrapolated models warned). Keep all games, proceed.

`civ_bench/stats/` gains a `MixedLMResult` wrapper + a `fit_cell_baseline(...)` dispatcher alongside the OLS
`RegressionResult`.

## Config wiring (`adjust[]`)

The stage `id` doubles as the table name. `uses.estimators` is required and single-source (point at
`attention` per `benchmark.pretrained.json`). Saves to `<root>/adjust/player_strength_panel.csv`. New params:
`block`, `baseline_source`, `post_cell_normalize`, `engine` (defaults `auto`/`same_condition_first`/`none`/
`r_lmer`). With `block:"none"` (or no controlled rows) the stage is byte-identical to the civ-only pipeline.

## Done

`<root>/adjust/player_strength_panel.csv` exists with at least
`game_id, player_id, player_type, civilization, adjusted_strength` (plus `seed`/`config_slot` pass-through),
and is registered so a downstream `uses.tables:["strength"]` resolves and adds the edge.

## Verification

- Column contract present; winner rows have `relative_strength == 1.0` when `enforce_winner:true`.
- `block:"none"` reproduces the old `turn_predicted` output byte-for-byte on the same estimator.
- On controlled synthetic data with a known per-cell Vanilla baseline + a known uplift: the estimated baseline
  recovers the baseline, `adjusted_strength` recovers the uplift, and `ratings.vanilla_slot_effect` goes from
  significant (raw) to ~null (adjusted).
- Sparse/no-baseline/disconnected cells warn (never abort); `r_lmer` vs `statsmodels` baselines agree (~1e-2).
