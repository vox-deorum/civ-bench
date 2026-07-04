"""``civ-bench`` command-line entrypoint.

    civ-bench extract|run|report --config <path> [--only ID] [--skip ID] [--dry-run]

Stage 0 implements config loading + validation + DAG resolution + dry-run
printing; stage 1 adds the ``extract`` stage (raw game DBs → canonical CSVs);
stage 2 adds the **load-only** ``estimators`` stage (``fit:"pretrained"`` →
``predictions.csv``); stages 3-4 add ``adjust`` (the strength panel) and the
``analyses`` modules; stage 5 adds ``report`` rendering. ``run`` executes the full
resolved DAG (extract → estimators → adjust → analyses → report); the standalone
``report`` command re-renders the document from existing analysis artifacts.

``extract`` (standalone and inside ``run``) auto-repairs by default: when a run
records malformed DBs in ``import_issues.csv`` it repairs them (``civ-bench fix``)
and re-imports so the ledger reflects the fixed state, before the rest of the DAG
runs. Disable via ``data.extract.auto_fix: false`` or ``--no-fix``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .catalog import Catalog
from .config import ConfigError, load_config
from .extract import ExtractError, run_extract
from .pipeline import Dag, build_dag, render_dag


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True,
        help="path to a benchmark run-spec (configs/benchmark*.json)",
    )
    parser.add_argument(
        "--only", action="append", default=[], metavar="ID",
        help="run only this stage id (and its deps); repeatable",
    )
    parser.add_argument(
        "--skip", action="append", default=[], metavar="ID",
        help="skip this stage id (or 'all'); repeatable",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="load + validate the config and print the resolved DAG + output "
             "root, without executing any stage",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="civ-bench",
        description="JSON-configurable benchmark harness for LLM strategists in "
                    "Civ V: Vox Populi.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd, help_text in (
        ("extract", "raw game DBs → canonical CSVs"),
        ("run", "run the full DAG (extract→estimators→adjust→analyses→report)"),
        ("report", "re-render the report from existing analysis artifacts"),
    ):
        p = sub.add_parser(cmd, help=help_text)
        _add_common(p)
        if cmd in ("extract", "run"):
            p.add_argument(
                "--force-rebuild", "--force_rebuild", "-f",
                dest="force_rebuild", action="store_true",
                help="re-extract even when outputs are newer than the DBs "
                     "(overrides data.extract.force_rebuild)",
            )
            p.add_argument(
                "--no-fix", dest="no_fix", action="store_true",
                help="do not auto-repair malformed DBs and re-import "
                     "(overrides data.extract.auto_fix)",
            )

    # `fix` repairs the malformed DBs recorded in import_issues.csv. It takes no
    # stage selection (--only/--skip are meaningless), so it gets a lean subparser
    # with its own --dry-run (preview) and --force (overwrite an existing .bak).
    fix = sub.add_parser("fix", help="repair malformed game DBs from import_issues.csv")
    fix.add_argument(
        "--config", required=True,
        help="path to a benchmark run-spec (configs/benchmark*.json)",
    )
    fix.add_argument(
        "--dry-run", action="store_true",
        help="list the problem DBs that would be repaired, without writing anything",
    )
    fix.add_argument(
        "--force", action="store_true",
        help="re-repair even when a <name>.db.bak backup already exists (overwrites it)",
    )
    return parser


def _report_extract_issues(result) -> None:
    """Print the one-line malformed-DB issue count to stderr (no-op when clean)."""
    if not result.skipped and result.issues:
        print(
            f"civ-bench: extract — {len(result.issues)} problem database(s) "
            f"recorded in {result.issues_path}",
            file=sys.stderr,
        )


def _extract_with_autofix(cfg, catalog, *, force_rebuild, auto_fix):
    """Run extract; when it records malformed DBs and ``auto_fix`` is on, repair the
    flagged DBs (``civ-bench fix``) and re-import so the ledger reflects the repaired
    state. Returns the final :class:`ExtractResult` (the re-import's, when one ran)."""
    result = run_extract(cfg, catalog=catalog, force_rebuild=force_rebuild)
    if result.skipped:
        print(f"civ-bench: extract skipped — {result.reason}")
    _report_extract_issues(result)

    # Only fix when this run FRESHLY recorded issues. A skipped (fresh-outputs) run and
    # a prune-only run evaluate nothing, so they never have fresh issues; carried-forward
    # (already-unrecoverable) games from a prior run must not be re-attempted every
    # invocation — pass --force-rebuild to re-examine (and re-fix) a stale ledger.
    if not auto_fix or result.skipped or not result.issues.has_fresh_issues:
        return result

    from .fix import FixError, run_fix

    try:
        # Repair only the games that failed this run, not the whole carried-forward ledger.
        fix_result = run_fix(cfg, only_game_ids=result.issues.fresh_game_ids)  # prints its own summary
    except FixError as exc:  # a fix failure must never abort the pipeline
        print(f"civ-bench: auto-fix skipped — {exc}", file=sys.stderr)
        return result
    if not fix_result.repaired:  # nothing repairable → the re-import would be a no-op
        return result

    print("civ-bench: re-importing repaired databases …")
    catalog = catalog or Catalog.from_run_config(cfg)
    # force_rebuild + prune_missing=False so the repaired DBs are actually re-inspected
    # (a prune-only re-import would evaluate nothing and never clear the ledger).
    result = run_extract(cfg, catalog=catalog, force_rebuild=True, prune_missing=False)
    _report_extract_issues(result)
    return result


def _is_dry(args: argparse.Namespace) -> bool:
    # `--skip all` is the documented equivalent of a dry run (Done criterion).
    return bool(args.dry_run) or any(s.lower() == "all" for s in args.skip)


# Every stage kind now has an executable implementation (report landed in stage 5).
_IMPLEMENTED_KINDS = {"extract", "estimators", "adjust", "analyses", "report"}


def _resolve_subset(dag: Dag, only: list[str], skip: list[str]) -> list[str]:
    """Return the topo-ordered node ids to execute, honouring ``--only``/``--skip``.

    ``--only ID`` keeps that node plus its transitive deps; ``--skip ID`` drops a
    node. Unknown ids are a ``ValueError`` (fail loud, never silently no-op).
    """
    skip_ids = {s for s in skip if s.lower() != "all"}
    for sid in skip_ids:
        if sid not in dag.nodes:
            raise ValueError(f"--skip references unknown stage id '{sid}'.")

    if only:
        for oid in only:
            if oid not in dag.nodes:
                raise ValueError(f"--only references unknown stage id '{oid}'.")
        keep: set[str] = set()
        stack = list(only)
        while stack:
            nid = stack.pop()
            if nid in keep:
                continue
            keep.add(nid)
            stack.extend(dag.nodes[nid].deps)
    else:
        keep = set(dag.order)

    keep -= skip_ids
    return [nid for nid in dag.order if nid in keep]


def _run_pipeline(cfg, dag: Dag, subset: list[str], force_rebuild: bool, auto_fix: bool) -> int:
    """Execute every stage in ``subset`` (topo order).

    Every stage kind is now implemented (extract → estimators → adjust → analyses
    → report). The ``_IMPLEMENTED_KINDS`` guard is retained as a defensive net for
    any future, not-yet-wired kind: such a stage is skipped (not aborted) and the
    run exits non-zero with a pointer rather than failing an executed stage.
    """
    catalog: Optional[Catalog] = None
    skipped: list[tuple[str, str]] = []
    for nid in subset:
        node = dag.nodes[nid]
        if node.kind not in _IMPLEMENTED_KINDS:
            skipped.append((nid, node.kind))
            continue
        if catalog is None:
            catalog = Catalog.from_run_config(cfg)
        if node.kind == "extract":
            _extract_with_autofix(
                cfg, catalog, force_rebuild=force_rebuild, auto_fix=auto_fix
            )
        elif node.kind == "estimators":
            from .estimators import run_estimator  # lazy: pulls torch/xgboost

            result = run_estimator(cfg, node.raw, catalog=catalog)
            print(
                f"civ-bench: estimator '{result.id}' ({result.model}) → "
                f"{result.predictions_path} ({result.n_rows} rows)"
            )
        elif node.kind == "adjust":
            from .adjust import run_adjust  # lazy: pulls statsmodels

            result = run_adjust(cfg, node.raw, catalog=catalog)
            print(
                f"civ-bench: adjust '{result.id}' ({result.module}, est={result.estimator_id}) → "
                f"{result.table_path} ({result.n_rows} rows)"
            )
            for warning in result.warnings:
                print(f"civ-bench: adjust '{result.id}' WARN — {warning}", file=sys.stderr)
        elif node.kind == "analyses":
            from .analyses import run_analysis  # lazy: pulls matplotlib/statsmodels/Rscript

            result = run_analysis(cfg, node.raw, catalog=catalog)
            n_t, n_f = len(result.table_paths), len(result.figure_paths)
            status = "empty (no artifacts)" if result.empty else f"{n_t} table(s), {n_f} figure(s)"
            print(
                f"civ-bench: analysis '{result.id}' ({result.module}) → "
                f"{cfg.output.resolve(f'{cfg.output.root}/analyses/{result.id}')} ({status})"
            )
            if result.summary:
                print(f"           {result.summary}")
        elif node.kind == "report":
            from .reports import run_report  # lazy: pulls pandas only

            result = run_report(cfg)
            print(
                f"civ-bench: report → {result.report_dir} "
                f"({result.n_sections} section(s); {', '.join(result.formats)})"
            )
            for path in result.written:
                print(f"           wrote {path}")
            for warning in result.warnings:
                print(f"civ-bench: report WARN — {warning}", file=sys.stderr)

    if skipped:
        kinds = ", ".join(sorted({k for _, k in skipped}))
        print(
            f"civ-bench: ran the implemented stages; skipped {len(skipped)} "
            f"unimplemented stage(s) [{kinds}]. "
            f"Use --dry-run to inspect the full DAG, or --only on a stage to run just that.",
            file=sys.stderr,
        )
        return 3
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"civ-bench: config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "fix":
        # `fix` repairs raw DB files; it needs no DAG and owns its own --dry-run
        # (preview), so it is handled before the DAG build and the generic _is_dry path.
        from .fix import FixError, run_fix

        try:
            result = run_fix(
                cfg, dry_run=bool(args.dry_run), force=bool(getattr(args, "force", False))
            )
        except (ConfigError, FixError) as exc:
            print(f"civ-bench: fix error: {exc}", file=sys.stderr)
            return 2
        # Best-effort tool: non-zero only when the run resolved nothing yet left
        # unresolved DBs (every attempt failed, or none could be located).
        return 2 if (result.unresolved and not result.repaired) else 0

    try:
        dag = build_dag(cfg)
    except ConfigError as exc:
        print(f"civ-bench: config error: {exc}", file=sys.stderr)
        return 2

    if _is_dry(args):
        print(render_dag(dag, cfg))
        print()
        print("Dry run: config loaded and validated; no stage executed.")
        return 0

    force_rebuild = getattr(args, "force_rebuild", False)
    # Auto-fix default lives in config (data.extract.auto_fix, default on); --no-fix is
    # a per-invocation override. Ignored by report/fix (no --no-fix, no extract stage).
    extract_cfg = cfg.data.get("extract", {}) or {}
    auto_fix = bool(extract_cfg.get("auto_fix", True)) and not getattr(args, "no_fix", False)

    if args.command == "extract":
        try:
            _extract_with_autofix(cfg, None, force_rebuild=force_rebuild, auto_fix=auto_fix)
        except (ConfigError, ExtractError) as exc:
            print(f"civ-bench: extract error: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.command == "report":
        # Re-render the document from existing analysis artifacts (no stage re-run).
        from .reports import ReportError, run_report

        try:
            result = run_report(cfg)
        except (ConfigError, ReportError) as exc:
            print(f"civ-bench: report error: {exc}", file=sys.stderr)
            return 2
        print(
            f"civ-bench: report → {result.report_dir} "
            f"({result.n_sections} section(s); {', '.join(result.formats)})"
        )
        for path in result.written:
            print(f"           wrote {path}")
        for warning in result.warnings:
            print(f"civ-bench: report WARN — {warning}", file=sys.stderr)
        return 0

    # `run`: execute the implemented prefix of the resolved DAG.
    try:
        subset = _resolve_subset(dag, args.only, args.skip)
    except ValueError as exc:
        print(f"civ-bench: {exc}", file=sys.stderr)
        return 2

    try:
        return _run_pipeline(cfg, dag, subset, force_rebuild, auto_fix)
    except (ConfigError, ExtractError) as exc:
        print(f"civ-bench: run error: {exc}", file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(f"civ-bench: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # estimator / adjust / analysis / report / load failures — fail loud
        from .adjust import AdjustError
        from .analyses import AnalysisError
        from .estimators import EstimatorError
        from .reports import ReportError

        if isinstance(exc, (EstimatorError, AdjustError, AnalysisError, ReportError,
                            FileNotFoundError, ValueError)):
            print(f"civ-bench: run error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
