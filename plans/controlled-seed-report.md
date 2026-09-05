# Controlled seed comparison report

<!-- PARITY: this feature is BUILT. The sections below are the as-built design; see
     configs/benchmark.md §6.2 (the analysis) and §7.1 (the heatmap pages) for the
     user-facing schema docs, and tests/test_controlled_seed_report.py for the
     verified behavior. -->

**Status: built & validated.** As-built notes on top of the original design:

- The analysis module is `bench/analyses/performance/controlled_seed_report.py`; `bench/reports/controlled_seed.py` owns the annex end-to-end (the `controlled_seed_document` builder and the `render_controlled_seed_site` renderer); the build context is `bench/reports/context.py`; the document model addition is `ControlledSeedDocument` in `bench/reports/model.py`, carried on `ReportDocument.controlled_seed` with `render_html_site` emitting its pages.
- The module reads its inputs as a census of the controlled design: it does not apply the global `data.filter`, because an `only_llm` or `min_games` filter would punch holes in the seed grid and remove the dedicated Vanilla baseline.
- A missing `baseline_experiment` on the strength stage is a configuration error (the report has no dedicated Vanilla source at all). A configured baseline that lacks rows for a specific `(seed, player_id)` pair is not fatal: the pair keeps blank differences and a visible page note.
- The exactly-one-estimator / one-strength-table references and the at-most-one-enabled-instance rule are validated at config load; the HTML-only formats rule is enforced at render time (`report.formats` without `html` skips the pages with a warning and omits the section's link).
- Previous and next player links on a detail page wrap cyclically within the seed's available player list, so both links always exist.

## Goal

Add the controlled-seed heatmap pages as an annex of the report. The pages expose game-to-game variation without repeating the aggregate rankings already covered by the family pages.

The annex has two levels:

1. An overview with two heatmaps for every controlled seed.
2. One detail page for every `(seed, player_id)` pair.

The current dataset has three seeds and eight final player positions. It therefore produces six overview heatmaps and 24 detail pages.

## Seed overview

Each seed gets two heatmaps with the same axes:

- Rows are strategist and condition combinations, such as `GPT-5.6-Sol | Every-turn` and `GPT-5.6-Sol | Per-5`.
- Columns are final `player_id` values.
- The dedicated Vanilla condition is a separate, visually isolated row. It is not a strategist column.
- Each cell averages all unique runs for its `(seed, player_id, strategist, condition)` key. Seating rotations and genuine repeated runs contribute equally. No confidence interval is shown.
- A cell tooltip shows the exact value, civilization, and run count.
- Clicking a cell opens the matching `(seed, player_id)` detail page with that strategist and condition selected.

This row and column orientation follows the requirement that Vanilla is a condition row. It supersedes the earlier sketch that placed strategist-condition combinations on columns.

### Adjusted strength heatmap

The first heatmap shows mean `adjusted_strength`. Use a fixed probability scale from 0 to 1 across every seed so the panels can be compared directly. Display the rounded value in each populated cell and leave missing combinations blank.

### Dominant victory focus heatmap

The second heatmap summarizes the four final victory-focus percentages:

- Domination
- Culture
- Diplomatic
- Science

Average each percentage over the runs in a cell, then select the largest mean. Annotate the cell with the strategy name and percentage, such as `Science 72%`. Use a stable categorical color for the winning strategy and vary its intensity by the percentage. Resolve exact ties in the listed order so output remains deterministic.

## Seed and player detail page

Write one page for each controlled `(seed, player_id)` pair, for example `seed-2-player-1.html`. The page combines all rotations and repeated runs in which a strategist-condition combination occupied that final player position.

The header shows the seed, player ID, matched civilization, and total source runs. It includes previous and next player links plus a return link to the seed overview. A controlled seed-player pair should have one seat-bound civilization. If the source contains more than one, show the sorted civilization list and a visible comparability warning.

### Victory-probability curves

- Use the estimator referenced by the report analysis.
- Interpolate each individual game curve onto a fixed 101-point normalized-progress grid from 0 to 1, then take the arithmetic mean at each grid point. Interpolate only inside each run's observed progress range and do not extrapolate or hold endpoints. Each point averages the runs that cover that point. Do not draw a confidence interval.
- Provide a strategist selector. Selecting a strategist shows all of its conditions together so Every-turn and Per-5 can be compared directly.
- Keep the matched dedicated Vanilla curve visible as a thicker reference line when it exists.
- Opening the page from an overview cell preselects the clicked strategist and condition. Direct navigation selects the first strategist in deterministic display order.
- Allow an explicit `Show all` choice for inspecting every curve, but do not use it as the default because the chart can contain many conditions.

### Comparison table

Show one row per strategist-condition combination, plus the separate Vanilla condition row. Keep these columns:

- Strategist
- Condition
- Run count
- Mean weighted victory probability
- Mean adjusted strength
- Difference from matched Vanilla adjusted strength
- Dominant victory focus and percentage
- Domination focus %
- Culture focus %
- Diplomatic focus %
- Science focus %

Do not include a rotations-represented column. Highlight the Vanilla row and its adjusted-strength value. Leave the difference blank for Vanilla itself or when the matched baseline is unavailable.

## Data preparation

Add an analysis module named `performance.controlled_seed_report`. It consumes:

- The canonical `games` table for `seed`, `seating_rotation`, and experiment membership.
- The canonical `panel` table for player identity, civilization, winner, and the four focus ratios.
- One configured strength adjust table for `weighted_strength` and `adjusted_strength`.
- Exactly one configured estimator for per-turn `predicted_win_probability`.

The module emits report-ready, deterministic tables rather than HTML:

- `seed_player_summary.csv` has one row per observed `(seed, player_id, strategist, condition)` plus dedicated Vanilla rows. It contains civilization, run counts, means, matched-baseline values, differences, dominant focus, and all four focus means. The renderer completes the global row and column grid and leaves unobserved combinations blank.
- `seed_player_probability.csv` has the normalized probability curve points for the same keys.
- `seed_player_index.csv` records every available seed-player page, its civilization, and source-run count.

Use the existing condition-pairing rules to split `player_type` into its base strategist identity and display condition. Preserve full `player_type` and experiment as audit columns. Require condition pairing to be enabled for this report. Use its configured base label and suffix order, then append any observed unconfigured suffixes in lexical order. Resolve model ordering and colors through `configs/models.json`.

Vanilla bypasses ordinary condition splitting because the catalog intentionally pools it across condition suffixes. Give dedicated baseline rows the canonical values `strategist = "Vanilla"` and `condition = "Vanilla"`, and place that row before strategist rows.

The configured strength stage's `baseline_experiment` is the sole dedicated Vanilla source. Match baseline strength and probability evidence by `(seed, player_id)`, averaging all baseline rotations and repeated games for that pair. The Vanilla row carries its own unique-game run count. Treatment rows keep their own run counts, even when the matched baseline mean was calculated from a different number of runs. Do not substitute Vanilla opponents from treatment conditions when the dedicated baseline is missing.

Count unique `game_id` values as runs. Before aggregation, enforce one panel and strength record per `(game_id, player_id)`. Duplicate source rows that violate that invariant are an analysis error rather than extra observations.

Probability input may contain many turns for one run. Require one prediction per `(game_id, player_id, turn_progress)` after sorting. Conflicting duplicate progress points are an analysis error. The scalar `Mean weighted victory probability` comes from the strength table's `weighted_strength`; it is distinct from the pointwise mean curve built from estimator predictions.

Only rows with both `seed != -1` and `seating_rotation != -1` participate. Exclude uncontrolled rows from mixed inputs. An input with no controlled rows fails with a clear `AnalysisError` instead of producing an empty report.

## Report template and output

The heatmap pages render automatically as an annex of the single report: when the resolved sections include an enabled, non-empty `performance.controlled_seed_report` analysis, the report also emits them. At most one enabled instance is allowed per run. The pages are HTML-only: a `report.formats` list without `html` skips them with a warning and omits the analysis section's link to `controlled-seed.html`; an omitted `formats` list defaults to `["md", "html"]`. The family pages are unchanged.

The report stage continues to render persisted analysis artifacts. It must not read canonical tables or estimator predictions directly. Introduce a report build context that contains run metadata, resolved sections, and a containment-checked loader for full named CSV artifacts from each selected analysis manifest. Adapt the default document builder to this context without changing its output.

Add a `ControlledSeedDocument` report model and a controlled-seed HTML renderer, both in `bench/reports/controlled_seed.py`: `controlled_seed_document` loads the three required analysis tables through the build context and constructs the document, and `render_controlled_seed_site` renders it. The document rides on `ReportDocument.controlled_seed`, and `render_html_site` emits its pages beside the family pages.

The annex adds these files to the report directory:

```text
<report-dir>/
  controlled-seed.html
  seed-<seed>-player-<player_id>.html
  assets/controlled-seed-report.js
  assets/<analysis-id>/*.csv
```

Use accessible HTML tables for the heatmaps, with deterministic vanilla JavaScript for filtering, tooltips, and page selection. Do not add a browser-side package or network dependency. Encode the selected strategist and condition in the detail-page query string so overview links remain stable.

`configs/benchmark.full.template.json` wires the analysis with `"enabled": false`. Update `configs/benchmark.md` with the analysis inputs, aggregation rules, baseline behavior, the automatic annex rendering, and the output layout.

## Validation and failure behavior

- Config validation requires one estimator reference and one strength-table reference for `performance.controlled_seed_report`, and allows at most one enabled instance per run.
- The pages render only when `html` is among `report.formats`; otherwise the run warns, skips the pages, and omits the analysis section's link to `controlled-seed.html`.
- A missing dedicated baseline leaves baseline curves and differences unavailable, with a visible page note. It is not fatal.
- A condition missing a seed-player combination produces a blank cell in the completed heatmap grid and no detail-table row for that unobserved combination.
- A seed-player pair with no usable prediction rows keeps its scalar summary and marks the probability curve unavailable.
- All ordering is stable: seeds numerically, Vanilla first, strategists by catalog order, conditions by configured pairing order, and player IDs numerically.
- Generated HTML, CSV, and JavaScript are byte-stable for unchanged inputs and configuration.

## Test plan

### Analysis tests

- Accept a fully controlled synthetic dataset and reject an uncontrolled-only dataset.
- Exclude uncontrolled rows from mixed data.
- Average rotations and genuine repeated games by seed, player ID, strategist, and condition without a confidence interval.
- Count unique game IDs and reject duplicate per-player source rows.
- Compute the four focus means before choosing the dominant focus, including deterministic tie handling.
- Match Vanilla by seed and player ID, compute adjusted-strength differences, and preserve rows when the baseline is missing.
- Interpolate unequal-length games onto the fixed normalized-progress grid and average them correctly.
- Reject conflicting duplicate probability points and avoid extrapolation outside each run's observed range.
- Give treatment and matched Vanilla rows independent run counts.
- Canonicalize the dedicated baseline as `Vanilla | Vanilla` without applying suffix pairing.
- Keep strategist and condition labels, catalog order, and colors stable.

### Report tests

- Render exactly two heatmaps per seed and one page per available seed-player pair.
- Render Vanilla as a separate condition row in both heatmaps and the detail table.
- Link every populated heatmap cell to the correct page and preselection query.
- Show run count but no rotations-represented column on detail pages.
- Emphasize the Vanilla curve and strength value when present, and show an unavailable note when absent.
- Escape labels and query parameters safely.
- Link the analysis's section to `controlled-seed.html` from its downloads list when html is rendered.
- Warn, skip the pages, and omit the link when `report.formats` excludes html.
- Render a comparability warning when a seed-player pair has multiple civilizations.
- Preserve the existing default report output and deterministic re-render tests.

## Review points

- Confirm the heatmap orientation: strategist-condition rows and player-ID columns. This orientation is used because Vanilla must be a separate condition row.
- Confirm that `weighted_strength` is the intended scalar labeled as mean victory probability in the detail table.
- Confirm that a 101-point normalized-progress grid is sufficient for averaged probability curves.
