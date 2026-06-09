<!-- PARITY: keep this file in sync with the code it describes. -->

# Stage 5 — report

**Goal.** Assemble the produced `AnalysisResult`s into a complete templated report — proving the full **extract → load → adjust → analyses → report** pipeline with **zero training**. **Status: built & validated.**

**Layout as built.** `bench/reports/` holds `errors.py` (`ReportError`), `model.py` (the renderer-agnostic document IR: `ReportDocument` → `FamilyGroup` → `Section` → `Figure`/`Table`), `templates.py` (template registry; the shipped `default` template groups sections into the five analysis families — ratings / prediction / calibration / performance / exploratory), `render.py` (`render_markdown` + `render_html` walk the *same* IR so the two formats stay faithful; tables via `DataFrame.to_markdown`/`to_html`, a tiny inline-markdown subset converted for HTML), and `runner.py` (`run_report` → resolve sections, read each analysis's `result.json` manifest, copy artifacts into a self-contained `assets/<id>/` tree, build the document, render). The CLI dispatches `node.kind == "report"` to `run_report` at the tail of `run`, and the standalone `civ-bench report` re-renders from existing artifacts without re-running any analysis. The package is import-light (pandas + stdlib only — figures are already PNGs on disk), so it pulls neither matplotlib nor R.

**The manifest.** The report re-renders from disk, so the **analysis runner now writes a `result.json` manifest** beside each analysis's artifacts (`<root>/analyses/<id>/result.json`: id, module, summary, metadata, ordered table/figure filenames, `empty` flag). This is the contract that lets `civ-bench report` reproduce the document from artifacts alone (invariant 3) — and it is written even for an empty result, so an empty section renders as *produced-but-empty* rather than being mistaken for a never-run stage.

## Files created

- `bench/reports/` — `errors.py`, `model.py`, `templates.py`, `render.py`, `runner.py`, `__init__.py`. The old notebook narratives (`performance/`, `exploratory/`, `predict/`) are now *generated* sections — each analysis's `AnalysisResult` is walked in DAG order and rendered; no analysis hardcodes its place (invariant 3).
- `bench/analyses/runner.py` — extended to persist the `result.json` manifest (see above).
- `bench/cli.py` — `report` added to `_IMPLEMENTED_KINDS`; `run` dispatches the report node; the standalone `report` command renders from existing artifacts.
- `tests/test_report.py` — section resolution, manifest→document→md/html render, asset copying, empty-section handling, byte-stable re-render, and the loud errors (missing manifest, unknown section id, unsupported format, unknown template).

## Config wiring (`report`)

`template` (registry name under `bench/reports/`; default `default`), `out_dir` (authored under the base output root and re-rooted by `output.suffix` §2.1; the run writes `<root><suffix>/<name>/`), `formats` (`md`/`html` implemented; `pdf` is schema-reserved and errors loudly at render time), `sections` (null = every enabled analysis in canonical-family order, members in dependency order; or an ordered id list, whose order — including across families — is preserved), `title` (null = derive from `name`), `include_disabled:false`.

## Done

`civ-bench run --config configs/benchmark.dev.json` (a local load-only config copied from `benchmark.pretrained.template.json`) produces the report under the resolved output root — `reports-dev/civbench-dev/` (`report.md` + `report.html` + `assets/`) end-to-end from the copied pre-trained estimators — **no training invoked**. This is the milestone: the whole pipeline works on pre-trained estimators. Verified on real data (15 sections across the five families).

## Verification

- ✅ The report contains a section for every enabled core analysis, grouped into families (members in dependency order).
- ✅ `report` re-render (`civ-bench report --config …`) reproduces the document **byte-identically** from existing artifacts (deterministic: no timestamps).
- ✅ Section ordering: `sections:null` ⇒ canonical family order; an explicit `sections` list ⇒ authored order. Missing manifest / unknown section id / `pdf` format / unknown template all fail loud.
