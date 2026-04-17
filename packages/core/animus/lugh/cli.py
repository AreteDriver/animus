"""Lugh transcripts CLI.

Examples:
    python -m animus.lugh.cli                                    # summary
    python -m animus.lugh.cli --rollup                           # per-cwd table
    python -m animus.lugh.cli --project monolith                 # filter
    python -m animus.lugh.cli --since 2026-04-01 --rollup        # date-bounded
    python -m animus.lugh.cli --session <uuid>                   # drift payload
    python -m animus.lugh.cli --no-sidechains                    # exclude subagents
    python -m animus.lugh.cli --no-cache                         # force re-parse
    python -m animus.lugh.cli --vacuum                           # drop stale cache rows
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from animus.lugh.cache import TranscriptCache
from animus.lugh.transcripts import (
    drift_signals,
    harvest_transcripts,
    rollup_by_cwd,
)


def _human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="animus.lugh.cli", description="Claude Code transcript harvester"
    )
    ap.add_argument("--project", help="Substring filter on session path (e.g., 'monolith')")
    ap.add_argument("--since", help="YYYY-MM-DD lower bound on session date")
    ap.add_argument("--session", help="Emit drift_signals JSON for one session id")
    ap.add_argument("--no-sidechains", action="store_true", help="Exclude subagent transcripts")
    ap.add_argument("--rollup", action="store_true", help="Per-cwd cost table")
    ap.add_argument("--no-cache", action="store_true", help="Bypass the SQLite cache")
    ap.add_argument(
        "--vacuum",
        action="store_true",
        help="Drop cache rows for files no longer on disk, then exit",
    )
    ap.add_argument("--json", action="store_true", help="Emit summary as JSON")
    args = ap.parse_args(argv)

    if args.vacuum:
        cache = TranscriptCache()
        base = Path.home() / ".claude" / "projects"
        valid = {
            str(f.relative_to(base)) for f in base.glob("**/*.jsonl") if f.is_relative_to(base)
        }
        n = cache.vacuum(valid)
        print(f"Vacuumed {n} stale rows; {cache.count()} remaining")
        return 0

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    sessions = list(
        harvest_transcripts(
            project=args.project,
            since=since,
            include_sidechains=not args.no_sidechains,
            use_cache=not args.no_cache,
        )
    )

    if args.session:
        match = [s for s in sessions if s.session_id == args.session]
        if not match:
            print(f"no session matching {args.session}")
            return 1
        print(json.dumps(drift_signals(match[0]), indent=2, default=str))
        return 0

    if args.rollup:
        roll = rollup_by_cwd(sessions)
        if args.json:
            print(json.dumps(roll, indent=2, default=str))
            return 0
        print(f"{'cwd':50}  {'cost':>10}  {'sessions':>8}  {'turns':>7}  {'tools':>7}")
        for cwd, d in sorted(roll.items(), key=lambda kv: -kv[1]["cost"])[:20]:
            print(
                f"{cwd[:50]:50}  ${d['cost']:9.2f}  "
                f"{d['session_count']:>8}  {d['turns']:>7}  {d['tools']:>7}"
            )
        return 0

    # Default summary
    main_sessions = [s for s in sessions if not s.is_sidechain]
    side_sessions = [s for s in sessions if s.is_sidechain]
    tot_cost = sum(s.total_cost_usd for s in sessions)
    tot_turns = sum(s.turn_count for s in sessions)
    tot_tools = sum(s.tool_call_count for s in sessions)
    tot_mcp = sum(s.mcp_call_count for s in sessions)

    act_sum: dict[str, float] = defaultdict(float)
    for s in sessions:
        for k, v in s.activity_breakdown.items():
            act_sum[k] += v * s.turn_count
    act_avg = {k: round(v / tot_turns, 1) for k, v in act_sum.items()} if tot_turns else {}

    # Collect novel schema signals so the user knows if CC changed format.
    schema_drift: set[str] = set()
    unpriced: set[str] = set()
    for s in sessions:
        schema_drift.update(s.unknown_line_types)
        unpriced.update(s.unknown_models)

    out = {
        "sessions": len(sessions),
        "main": len(main_sessions),
        "subagent": len(side_sessions),
        "total_cost_usd": round(tot_cost, 2),
        "main_cost_usd": round(sum(s.total_cost_usd for s in main_sessions), 2),
        "subagent_cost_usd": round(sum(s.total_cost_usd for s in side_sessions), 2),
        "turns": tot_turns,
        "tool_calls": tot_tools,
        "mcp_calls": tot_mcp,
        "activity_pct": act_avg,
        "schema_drift_line_types": sorted(schema_drift),
        "unpriced_models": sorted(unpriced),
    }

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"Sessions:   {out['sessions']:>6}  (main: {out['main']}, subagent: {out['subagent']})")
    print(
        f"Total cost: ${out['total_cost_usd']:>9.2f}  "
        f"(main: ${out['main_cost_usd']:.2f}, subagent: ${out['subagent_cost_usd']:.2f})"
    )
    print(f"Turns:      {out['turns']:>6}")
    print(f"Tools:      {out['tool_calls']:>6}  MCP: {out['mcp_calls']}")
    print()
    print("Activity mix (turn-weighted avg):")
    for k, v in sorted(act_avg.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(v / 2)
        print(f"  {k:14}  {v:5.1f}%  {bar}")
    if schema_drift:
        print(f"\n  schema drift — unknown line types: {sorted(schema_drift)}")
    if unpriced:
        print(f"  unpriced models (using default pricing): {sorted(unpriced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
