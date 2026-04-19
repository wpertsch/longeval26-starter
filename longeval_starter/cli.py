"""Command-line entry point.

Invoked as ``python -m longeval_starter <command> [--snapshot ...]`` or
via the ``Makefile``. All heavy lifting lives in the other modules —
this file is just argument parsing plus a main function.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from longeval_starter.config import DEFAULT_CONFIG_PATH, load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longeval_starter",
        description="LongEval 2026 Task-1 BM25 starter.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config.yaml (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # Every subcommand accepts --snapshot; "all" iterates over the
    # snapshots list from config.yaml.
    def add_snapshot(p: argparse.ArgumentParser, *, required: bool = True):
        p.add_argument(
            "--snapshot",
            required=required,
            help="Snapshot name, e.g. snapshot-1.",
        )

    p_index = sub.add_parser("index", help="Build a Terrier index for a snapshot.")
    add_snapshot(p_index)

    p_retrieve = sub.add_parser(
        "retrieve",
        help="Run the retrieval pipeline and write a TREC run file.",
    )
    add_snapshot(p_retrieve)

    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate the pipeline on a snapshot's (training) qrels.",
    )
    add_snapshot(p_eval)

    sub.add_parser(
        "all",
        help="Index + retrieve + evaluate for every snapshot in config.yaml.",
    )

    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    cfg = load_config(args.config)

    # Imported lazily so `--help` doesn't pay the JVM / ir_datasets startup cost.
    from longeval_starter.evaluate import evaluate as do_evaluate
    from longeval_starter.index import build_or_load_index
    from longeval_starter.retrieve import retrieve as do_retrieve

    log = logging.getLogger("longeval_starter")

    if args.command == "index":
        build_or_load_index(cfg, args.snapshot)

    elif args.command == "retrieve":
        do_retrieve(cfg, args.snapshot)

    elif args.command == "evaluate":
        do_evaluate(cfg, args.snapshot)

    elif args.command == "all":
        for snapshot in cfg.snapshots:
            log.info("=" * 72)
            log.info("Snapshot: %s", snapshot)
            log.info("=" * 72)
            build_or_load_index(cfg, snapshot)
            try:
                do_evaluate(cfg, snapshot)
            except RuntimeError as exc:
                # Snapshots without qrels just skip evaluation.
                log.warning("Skipping evaluation for %s: %s", snapshot, exc)
            do_retrieve(cfg, snapshot)

    else:  # pragma: no cover -- argparse enforces the choices
        raise SystemExit(f"Unknown command: {args.command!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
