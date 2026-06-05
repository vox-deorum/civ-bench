"""``civ-bench`` command-line entrypoint.

    civ-bench extract|run|report --config <path> [--only ID] [--skip ID] [--dry-run]

Stage 0 implements config loading + validation + DAG resolution + dry-run
printing. Actual stage execution (extract/estimators/adjust/analyses/report) is
filled in by later stages; running without ``--dry-run`` reports what is not yet
implemented rather than silently doing nothing.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .config import ConfigError, load_config
from .pipeline import build_dag, render_dag


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
    return parser


def _is_dry(args: argparse.Namespace) -> bool:
    # `--skip all` is the documented equivalent of a dry run (Done criterion).
    return bool(args.dry_run) or any(s.lower() == "all" for s in args.skip)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config)
        dag = build_dag(cfg)
    except ConfigError as exc:
        print(f"civ-bench: config error: {exc}", file=sys.stderr)
        return 2

    if _is_dry(args):
        print(render_dag(dag, cfg))
        print()
        print("Dry run: config loaded and validated; no stage executed.")
        return 0

    # Stage execution is built out in later stages (stage1+).
    print(render_dag(dag, cfg))
    print()
    print(
        f"civ-bench: '{args.command}' execution is not implemented yet "
        f"(stage 0 is scaffold only). Re-run with --dry-run to validate + print "
        f"the DAG.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
