"""Animus CLI — argparse entry point for subcommands.

Wires ``animus ingest`` and future structured commands outside the
interactive REPL.
"""

from __future__ import annotations

import argparse
import sys

from animus.workflows.ingest import ingest


def _cmd_ingest(args: argparse.Namespace) -> int:
    result = ingest(
        args.url,
        synthesize=args.synthesize,
        tag=args.tag,
    )
    if result.item:
        print(f"item: {result.item.title}")
    if result.synthesis:
        print(f"synthesis: {result.synthesis.title}")
    if result.memory_tags:
        print(f"memory_tags: {len(result.memory_tags)}")
    for err in result.errors:
        print(f"warning: {err.stage} failed — {err.message}", file=sys.stderr)
    if not result.success:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="animus")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a URL")
    ingest_parser.add_argument("url")
    ingest_parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Run Ogma synthesis after fetching",
    )
    ingest_parser.add_argument(
        "--tag",
        action="store_true",
        help="Push structured concepts to semantic memory",
    )
    ingest_parser.set_defaults(func=_cmd_ingest)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
