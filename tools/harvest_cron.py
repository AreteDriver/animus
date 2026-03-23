#!/usr/bin/env python3
"""
Harvest Watchlist Cron — weekly competition scanner.

Runs run_watchlist_scan(), prints a summary, and posts a Discord webhook
notification with results.

Usage:
    python tools/harvest_cron.py              # Run scan + notify
    python tools/harvest_cron.py --dry-run    # Preview without Discord post
    python tools/harvest_cron.py --quiet      # Minimal stdout output

Cron (Sunday 6am):
    0 6 * * 0 /home/arete/projects/animus/packages/core/.venv/bin/python \
        /home/arete/projects/animus/tools/harvest_cron.py \
        >> /home/arete/.animus/harvest_cron.log 2>&1
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

from animus.harvest_watchlist import run_watchlist_scan

DISCORD_WEBHOOK_ENV = "HARVEST_DISCORD_WEBHOOK"

# Colors for Discord embeds
COLOR_GREEN = 0x2ECC71   # No changes / all good
COLOR_YELLOW = 0xF1C40F  # Changes detected
COLOR_RED = 0xE74C3C     # Errors occurred


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
    fields.append({
        "name": "Summary",
        "value": "\n".join(summary_lines),
        "inline": False,
    })

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

        fields.append({
            "name": repo,
            "value": "\n".join(parts) if parts else "Changes detected",
            "inline": False,
        })

    # No-changes repos (compact)
    if no_changes:
        fields.append({
            "name": "No Changes",
            "value": ", ".join(no_changes),
            "inline": False,
        })

    # Error details
    for err in errors:
        fields.append({
            "name": f"Error: {err.get('repo', '?')}",
            "value": str(err.get("error", "Unknown error"))[:1024],
            "inline": False,
        })

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "embeds": [{
            "title": "Weekly Harvest Report",
            "color": color,
            "fields": fields,
            "timestamp": timestamp,
            "footer": {"text": "Animus Harvest Scanner"},
        }],
    }


def send_discord_webhook(webhook_url: str, payload: dict) -> bool:
    """Send a payload to a Discord webhook. Returns True on success."""
    import urllib.request
    import urllib.error

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly harvest watchlist cron job")
    parser.add_argument("--dry-run", action="store_true", help="Skip Discord notification")
    parser.add_argument("--quiet", action="store_true", help="Minimal stdout output")
    args = parser.parse_args()

    try:
        report = asyncio.run(run())
    except Exception as e:
        print(f"[Harvest Cron] FATAL: {e}", file=sys.stderr)
        return 1

    print_summary(report, quiet=args.quiet)

    # Discord notification
    webhook_url = os.environ.get(DISCORD_WEBHOOK_ENV, "")
    if not webhook_url:
        if not args.quiet:
            print(f"\n  [{DISCORD_WEBHOOK_ENV} not set — skipping Discord notification]")
    elif args.dry_run:
        if not args.quiet:
            print("\n  [Dry run — skipping Discord notification]")
    else:
        payload = build_discord_payload(report)
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
