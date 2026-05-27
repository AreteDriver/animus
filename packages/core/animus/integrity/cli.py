"""CLI for the integrity checker.

Run::

    python -m animus.integrity.cli verify        # check; non-zero exit on drift
    python -m animus.integrity.cli regenerate    # re-baseline (after intentional changes)
    python -m animus.integrity.cli show          # print current manifest
"""

from __future__ import annotations

import argparse
import json
import sys

from animus.integrity.checker import (
    IntegrityMismatchError,
    IntegrityNotInitializedError,
    compute_current,
    default_baseline_dir,
    load_baseline,
    regenerate_baseline,
    verify_or_raise,
)


def cmd_verify() -> int:
    try:
        verify_or_raise(default_baseline_dir())
    except (IntegrityMismatchError, IntegrityNotInitializedError) as e:
        print(str(e), file=sys.stderr)
        return 1
    print("Integrity check passed.")
    return 0


def cmd_regenerate(note: str = "") -> int:
    manifest = regenerate_baseline(default_baseline_dir(), note=note)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_show() -> int:
    try:
        manifest = load_baseline(default_baseline_dir())
    except IntegrityNotInitializedError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    print("\n--- Current hashes (for comparison) ---", file=sys.stderr)
    print(json.dumps(compute_current(), indent=2), file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Animus integrity baseline tooling.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    regen = sub.add_parser("regenerate")
    regen.add_argument("--note", default="", help="Human-readable reason for the regen.")
    sub.add_parser("show")
    args = parser.parse_args()

    if args.cmd == "verify":
        return cmd_verify()
    if args.cmd == "regenerate":
        return cmd_regenerate(note=args.note)
    if args.cmd == "show":
        return cmd_show()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
