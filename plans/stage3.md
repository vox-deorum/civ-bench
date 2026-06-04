<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 3 — adjust (strength panel)

**Goal.** Turn an estimator's per-turn `predicted_win_probability` into the per-player-game
**`adjusted_strength`** panel and register it as the named `strength` table that `ratings.*` (and some
`performance.*`) consume. This is the bridge that makes ratings depend transitively on an estimator.

## Files to create / port

- `civ_bench/adjust/registry.py` — name → adjust class (currently only `strength`).
- `civ_bench/adjust/strength.py` — port `prepare_strength_data` from
  `../vox-deorum-analysis/performance/turn_predicted.ipynb` (also copied in `ratings/iterative_bt.py` —
  consolidate to one place here). Pipeline: late-game weighted average (`turn_progress_min`, `weight`)
  → relative-to-leader (`relative_to`) → winner enforcement (`enforce_winner`) → OLS-logit civ
  adjustment (`civ_adjust`, uses `civ_bench/stats/`).

## Config wiring (`adjust[]`)

The stage `id` doubles as the table name. `uses.estimators` is required and single-source (point at
`attention` per `benchmark.pretrained.json`). Saves to `<root>/adjust/player_strength_panel.csv`.
`civ_adjust:"none"` ⇒ `adjusted_strength == relative_strength`.

## Done

`<root>/adjust/player_strength_panel.csv` exists with at least
`game_id, player_id, player_type, civilization, adjusted_strength`, and is registered so a downstream
`uses.tables:["strength"]` resolves and adds the edge.

## Verification

- Column contract present; winner rows have `relative_strength == 1.0` when `enforce_winner:true`.
- Spot-check `adjusted_strength` against the old `turn_predicted` output on the same estimator.
