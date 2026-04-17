#!/usr/bin/env python3
"""
Lugh Watchlist Cron — weekly competition scanner.

Runs run_watchlist_scan(), prints a summary, and posts a Discord webhook
notification with results.

Usage:
    python tools/lugh_cron.py              # Run scan + notify
    python tools/lugh_cron.py --dry-run    # Preview without Discord post
    python tools/lugh_cron.py --quiet      # Minimal stdout output

Cron (Sunday 6am):
    0 6 * * 0 ~/projects/animus/packages/core/.venv/bin/python \
        ~/projects/animus/tools/lugh_cron.py \
        >> ~/.animus/lugh_cron.log 2>&1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

# Ensure animus core is importable when run standalone (e.g., from cron)
_CORE_DIR = os.path.join(os.path.dirname(__file__), "..", "packages", "core")
if os.path.isdir(_CORE_DIR) and _CORE_DIR not in sys.path:
    sys.path.insert(0, os.path.realpath(_CORE_DIR))

from animus.lugh.transcripts import drift_signals, harvest_transcripts, rollup_by_cwd  # noqa: E402
from animus.lugh.watchlist import run_watchlist_scan  # noqa: E402

DISCORD_WEBHOOK_ENV = "LUGH_DISCORD_WEBHOOK"
_LEGACY_DISCORD_WEBHOOK_ENV = "HARVEST_DISCORD_WEBHOOK"

# Colors for Discord embeds
COLOR_GREEN = 0x2ECC71  # No changes / all good
COLOR_YELLOW = 0xF1C40F  # Changes detected
COLOR_RED = 0xE74C3C  # Errors occurred


def build_discord_payload(report: dict) -> dict:
    """Build a Discord webhook payload from the scan report."""
    scanned = report.get("scanned", 0)
    changes = report.get("changes", [])
    no_changes = report.get("no_changes", [])
    errors = report.get("errors", [])

    # Pick embed color based on results
    if errors:
        color = COLOR_RED
    elif changes:
        color = COLOR_YELLOW
    else:
        color = COLOR_GREEN

    fields = []

    # Summary field
    summary_lines = [
        f"Scanned: **{scanned}** repos",
        f"Changes: **{len(changes)}**",
        f"No changes: **{len(no_changes)}**",
        f"Errors: **{len(errors)}**",
    ]
    fields.append(
        {
            "name": "Summary",
            "value": "\n".join(summary_lines),
            "inline": False,
        }
    )

    # Detail fields for repos with changes
    for change in changes:
        repo = change.get("repo", "unknown")
        parts = []

        if change.get("score_change"):
            parts.append(f"Score: {change['score_change']}")
        if change.get("new_patterns"):
            parts.append(f"New patterns: {', '.join(change['new_patterns'][:5])}")
        if change.get("removed_patterns"):
            parts.append(f"Removed: {', '.join(change['removed_patterns'][:5])}")
        if change.get("new_dependencies"):
            parts.append(f"New deps: {', '.join(change['new_dependencies'][:5])}")
        if change.get("alert"):
            parts.append(f"Alert: {change['alert']}")

        fields.append(
            {
                "name": repo,
                "value": "\n".join(parts) if parts else "Changes detected",
                "inline": False,
            }
        )

    # No-changes repos (compact)
    if no_changes:
        fields.append(
            {
                "name": "No Changes",
                "value": ", ".join(no_changes),
                "inline": False,
            }
        )

    # Error details
    for err in errors:
        fields.append(
            {
                "name": f"Error: {err.get('repo', '?')}",
                "value": str(err.get("error", "Unknown error"))[:1024],
                "inline": False,
            }
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "embeds": [
            {
                "title": "Weekly Lugh Report",
                "color": color,
                "fields": fields,
                "timestamp": timestamp,
                "footer": {"text": "Animus Lugh — repo watchlist + Claude Code transcripts"},
            }
        ],
    }


def send_discord_webhook(webhook_url: str, payload: dict) -> bool:
    """Send a payload to a Discord webhook. Returns True on success."""
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 204)
    except urllib.error.HTTPError as e:
        print(f"Discord webhook HTTP error: {e.code} {e.reason}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Discord webhook error: {e}", file=sys.stderr)
        return False


def print_summary(report: dict, quiet: bool = False) -> None:
    """Print a human-readable summary of the scan report."""
    scanned = report.get("scanned", 0)
    changes = report.get("changes", [])
    no_changes = report.get("no_changes", [])
    errors = report.get("errors", [])

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n[Harvest Cron] {now}")
    print(f"  Scanned: {scanned} repos")
    print(f"  Changes: {len(changes)}")
    print(f"  No changes: {len(no_changes)}")
    print(f"  Errors: {len(errors)}")

    if not quiet:
        for change in changes:
            repo = change.get("repo", "unknown")
            score = change.get("score_change", "?")
            alert = change.get("alert", "")
            print(f"\n  [{repo}] Score: {score}")
            if alert:
                print(f"    Alert: {alert}")
            if change.get("new_patterns"):
                print(f"    New patterns: {', '.join(change['new_patterns'][:5])}")
            if change.get("new_dependencies"):
                print(f"    New deps: {', '.join(change['new_dependencies'][:5])}")

        if no_changes:
            print(f"\n  Unchanged: {', '.join(no_changes)}")

        for err in errors:
            print(f"\n  [ERROR] {err.get('repo', '?')}: {err.get('error', '?')}")


async def run() -> dict:
    """Execute the watchlist scan."""
    return await run_watchlist_scan()


def build_transcript_summary(days: int = 7, drift_threshold: float = 70.0) -> dict:
    """Roll up the last `days` of Claude Code transcripts.

    Returns a dict with total spend, per-cwd top 5, activity mix, and
    any sessions above the efficiency-drift threshold.
    """
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    since = _dt.now(_tz.utc) - _td(days=days)
    sessions = list(harvest_transcripts(since=since))
    if not sessions:
        return {"window_days": days, "session_count": 0, "total_cost_usd": 0.0}

    total_cost = sum(s.total_cost_usd for s in sessions)
    total_turns = sum(s.turn_count for s in sessions)
    main_cost = sum(s.total_cost_usd for s in sessions if not s.is_sidechain)
    side_cost = sum(s.total_cost_usd for s in sessions if s.is_sidechain)

    roll = rollup_by_cwd(sessions)
    top5 = sorted(roll.items(), key=lambda kv: -kv[1]["cost"])[:5]

    act_sum: dict[str, float] = {}
    for s in sessions:
        for k, v in s.activity_breakdown.items():
            act_sum[k] = act_sum.get(k, 0.0) + v * s.turn_count
    act_avg = {k: round(v / total_turns, 1) for k, v in act_sum.items()} if total_turns else {}

    flagged = []
    for s in sessions:
        p = drift_signals(s)
        if p["signals"]["efficiency_drift"] >= drift_threshold:
            flagged.append(
                {
                    "session_id": s.session_id,
                    "primary_cwd": s.primary_cwd,
                    "efficiency_drift": p["signals"]["efficiency_drift"],
                    "total_cost_usd": s.total_cost_usd,
                }
            )
    flagged.sort(key=lambda x: -x["efficiency_drift"])

    schema_drift = sorted({t for s in sessions for t in s.unknown_line_types})
    unpriced = sorted({m for s in sessions for m in s.unknown_models})

    return {
        "window_days": days,
        "session_count": len(sessions),
        "main_count": sum(1 for s in sessions if not s.is_sidechain),
        "subagent_count": sum(1 for s in sessions if s.is_sidechain),
        "total_cost_usd": round(total_cost, 2),
        "main_cost_usd": round(main_cost, 2),
        "subagent_cost_usd": round(side_cost, 2),
        "top_cwds": [
            {"cwd": c, "cost": round(d["cost"], 2), "sessions": d["session_count"]} for c, d in top5
        ],
        "activity_pct": act_avg,
        "drift_flagged": flagged[:5],
        "schema_drift_line_types": schema_drift,
        "unpriced_models": unpriced,
    }


def append_transcript_fields(payload: dict, ts_summary: dict) -> None:
    """Attach a weekly transcript summary to an existing Discord embed payload."""
    if not payload.get("embeds") or ts_summary.get("session_count", 0) == 0:
        return
    fields = payload["embeds"][0].setdefault("fields", [])

    lines = [
        f"Sessions: **{ts_summary['session_count']}** "
        f"({ts_summary.get('main_count', 0)} main, "
        f"{ts_summary.get('subagent_count', 0)} subagent)",
        f"Spend: **${ts_summary['total_cost_usd']:.2f}** "
        f"(main ${ts_summary.get('main_cost_usd', 0):.2f}, "
        f"subagent ${ts_summary.get('subagent_cost_usd', 0):.2f})",
    ]
    if ts_summary.get("top_cwds"):
        top = "\n".join(
            f"• `{c['cwd']}`: ${c['cost']:.2f} ({c['sessions']}s)" for c in ts_summary["top_cwds"]
        )
        lines.append(f"Top cwds:\n{top}")
    if ts_summary.get("activity_pct"):
        mix = ", ".join(
            f"{k}={v}%"
            for k, v in sorted(ts_summary["activity_pct"].items(), key=lambda kv: -kv[1])[:5]
        )
        lines.append(f"Activity: {mix}")
    if ts_summary.get("drift_flagged"):
        flags = "\n".join(
            f"• `{f['session_id'][:8]}` @ {f['primary_cwd']}: "
            f"{f['efficiency_drift']}% conv, ${f['total_cost_usd']:.2f}"
            for f in ts_summary["drift_flagged"]
        )
        lines.append(f"Drift flagged (efficiency ≥ 70%):\n{flags}")
    if ts_summary.get("schema_drift_line_types"):
        lines.append(f"⚠️ Unknown JSONL types: {', '.join(ts_summary['schema_drift_line_types'])}")
    if ts_summary.get("unpriced_models"):
        lines.append(f"⚠️ Unpriced models: {', '.join(ts_summary['unpriced_models'])}")

    fields.append(
        {
            "name": f"Claude Code — Last {ts_summary['window_days']} Days",
            "value": "\n".join(lines)[:1024],
            "inline": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly Lugh watchlist + transcript cron")
    parser.add_argument("--dry-run", action="store_true", help="Skip Discord notification")
    parser.add_argument("--quiet", action="store_true", help="Minimal stdout output")
    parser.add_argument(
        "--no-transcripts", action="store_true", help="Skip the Claude Code transcript summary"
    )
    parser.add_argument(
        "--transcript-days", type=int, default=7, help="Window (days) for the transcript summary"
    )
    args = parser.parse_args()

    try:
        report = asyncio.run(run())
    except Exception as e:
        print(f"[Lugh Cron] FATAL: {e}", file=sys.stderr)
        return 1

    print_summary(report, quiet=args.quiet)

    ts_summary: dict = {}
    if not args.no_transcripts:
        try:
            ts_summary = build_transcript_summary(days=args.transcript_days)
            if not args.quiet and ts_summary.get("session_count", 0) > 0:
                print(
                    f"\n[Transcripts — last {ts_summary['window_days']}d] "
                    f"{ts_summary['session_count']} sessions, "
                    f"${ts_summary['total_cost_usd']:.2f}"
                )
                for c in ts_summary.get("top_cwds", [])[:3]:
                    print(f"  {c['cwd']:45}  ${c['cost']:8.2f}  {c['sessions']}s")
                if ts_summary.get("schema_drift_line_types"):
                    print(f"  ⚠️ Unknown types: {ts_summary['schema_drift_line_types']}")
                if ts_summary.get("unpriced_models"):
                    print(f"  ⚠️ Unpriced models: {ts_summary['unpriced_models']}")
        except Exception as e:
            # Never let transcript errors break the watchlist cron
            print(f"[Lugh Cron] transcript summary failed: {e}", file=sys.stderr)
            ts_summary = {}

    # Discord notification
    webhook_url = os.environ.get(DISCORD_WEBHOOK_ENV, "") or os.environ.get(
        _LEGACY_DISCORD_WEBHOOK_ENV, ""
    )
    if not webhook_url:
        if not args.quiet:
            print(f"\n  [{DISCORD_WEBHOOK_ENV} not set — skipping Discord notification]")
    elif args.dry_run:
        if not args.quiet:
            print("\n  [Dry run — skipping Discord notification]")
    else:
        payload = build_discord_payload(report)
        if ts_summary:
            append_transcript_fields(payload, ts_summary)
        if send_discord_webhook(webhook_url, payload):
            if not args.quiet:
                print("\n  Discord notification sent.")
        else:
            print("\n  [WARNING] Discord notification failed.", file=sys.stderr)

    # Exit 1 if any errors occurred
    if report.get("errors"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
