<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 5 — report

**Goal.** Assemble the produced `AnalysisResult`s into a complete templated report — proving the full **extract → load → adjust → analyses → report** pipeline with **zero training**.

## Files to create / port

- `civ_bench/reports/` — default template + renderer: walk each produced `AnalysisResult` in DAG order, render its tables/figures/summary to `md` and `html`. No analysis hardcodes its place (invariant 3). Convert the old notebook narratives (`performance/`, `exploratory/`, `predict/`) into generated sections, not authored ones.

## Config wiring (`report`)

`template`, `out_dir` (resolves under the output root → `reports/<name>/` or `reports-cross/<name>/`), `formats` (`md`/`html`/`pdf`), `sections` (null = every enabled analysis in DAG order; or an ordered id list), `title`, `include_disabled:false`.

## Done

`civ-bench run --config configs/benchmark.dev.json` (a local load-only config copied from `benchmark.pretrained.template.json`) produces the report under the resolved output root — e.g. `reports-dev/staff-standard-2026-dev/` for `benchmark.dev.json` (or `reports/staff-standard-2026/` for the no-suffix template) — (md + html) end-to-end from the copied estimators — **no training invoked**. This is the milestone: the whole pipeline works on pre-trained estimators.

## Verification

- The report contains sections for every enabled core analysis, in dependency order.
- `report` re-render (`civ-bench report --config …`) reproduces the document from existing artifacts.
