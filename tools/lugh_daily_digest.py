#!/usr/bin/env python3
"""
Lugh Daily Digest — harvest → categorize → render MD + Discord.

Typical daily run (cron-ready):

    python tools/lugh_daily_digest.py --fetch --post

    # What it does:
    # 1. Fetch fresh items from all configured Lugh sources
    # 2. Build a 24h digest against the SourceCache
    # 3. Write ~/projects/notes/ogma/digests/YYYY-MM-DD.md (action plan)
    # 4. Post Discord embed to LUGH_DISCORD_WEBHOOK (or
    #    HARVEST_DISCORD_WEBHOOK legacy)

Flags:

    --fetch              Run `cli fetch` before building the digest
    --post               Actually hit Discord webhook (default: dry-run print)
    --hours N            Window size, default 24
    --min-score X        Relevance floor, default 0.05
    --md-dir PATH        Where to write the MD plan (default:
                         ~/projects/notes/ogma/digests)
    --no-md              Skip writing the MD file
    --json               Also print the raw Discord payload as JSON

Cron example (daily 8am local):

    0 8 * * * cd ~/projects/animus && /usr/bin/env python \
        tools/lugh_daily_digest.py --fetch --post \
        >> ~/.animus/lugh_daily_digest.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_CORE_DIR = os.path.join(os.path.dirname(__file__), "..", "packages", "core")
if os.path.isdir(_CORE_DIR) and _CORE_DIR not in sys.path:
    sys.path.insert(0, os.path.realpath(_CORE_DIR))

from animus.lugh.digest import build_digest, to_discord_payload, to_markdown  # noqa: E402
from animus.lugh.sources.base import SourceCache  # noqa: E402
from animus.lugh.sources.cli import main as sources_cli_main  # noqa: E402

DISCORD_WEBHOOK_ENV = "LUGH_DISCORD_WEBHOOK"
_LEGACY_DISCORD_WEBHOOK_ENV = "HARVEST_DISCORD_WEBHOOK"

DEFAULT_MD_DIR = Path("~/projects/notes/ogma/digests").expanduser()


def _post_discord(webhook_url: str, payload: dict, timeout: float = 15.0) -> bool:
    """POST payload to Discord webhook. Returns True on 2xx; fail-open (logs on error)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 204)
    except urllib.error.HTTPError as e:
        print(f"Discord HTTP {e.code}: {e.reason}", file=sys.stderr)
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"Discord transport error: {e}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lugh daily digest → MD + Discord")
    ap.add_argument(
        "--fetch",
        action="store_true",
        help="Run `animus.lugh.sources.cli fetch` before digesting",
    )
    ap.add_argument(
        "--post",
        action="store_true",
        help="Actually POST to Discord webhook (default: dry-run print)",
    )
    ap.add_argument("--hours", type=int, default=24, help="Window size in hours")
    ap.add_argument("--min-score", type=float, default=0.05, help="Relevance floor")
    ap.add_argument(
        "--md-dir",
        type=Path,
        default=DEFAULT_MD_DIR,
        help="Directory for daily MD plan (default: ~/projects/notes/ogma/digests)",
    )
    ap.add_argument("--no-md", action="store_true", help="Skip writing the MD file")
    ap.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Print raw Discord payload JSON in addition to the preview",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Override cache path (default: ~/.animus/lugh_sources.sqlite)",
    )
    args = ap.parse_args(argv)

    # Step 1 — fresh fetch
    if args.fetch:
        print("[digest] fetching fresh items…")
        rc = sources_cli_main(["fetch", "--quiet"])
        if rc != 0:
            print(f"[digest] fetch returned {rc} — continuing with existing cache", file=sys.stderr)

    # Step 2 — digest
    cache = SourceCache(path=args.cache) if args.cache else SourceCache()
    digest = build_digest(cache, window_hours=args.hours, min_score=args.min_score)

    print(
        f"[digest] {digest.date} · {digest.total_items} items · "
        f"top_for_ogma={len(digest.top_for_ogma)}"
    )
    for cat, items in digest.by_category.items():
        print(f"  {cat.value:9s}  {len(items)}")

    # Step 3 — write MD plan
    if not args.no_md:
        args.md_dir.mkdir(parents=True, exist_ok=True)
        md_path = args.md_dir / f"{digest.date}.md"
        md_path.write_text(to_markdown(digest), encoding="utf-8")
        print(f"[digest] wrote {md_path}")

    # Step 4 — Discord
    payload = to_discord_payload(digest)
    if args.emit_json:
        print(json.dumps(payload, indent=2, default=str))

    webhook = os.environ.get(DISCORD_WEBHOOK_ENV) or os.environ.get(_LEGACY_DISCORD_WEBHOOK_ENV)

    if args.post:
        if not webhook:
            print(
                f"[digest] --post set but neither {DISCORD_WEBHOOK_ENV} nor "
                f"{_LEGACY_DISCORD_WEBHOOK_ENV} is set",
                file=sys.stderr,
            )
            return 2
        ok = _post_discord(webhook, payload)
        print("[digest] Discord posted." if ok else "[digest] Discord post FAILED.")
        return 0 if ok else 1

    # Dry-run: show the embed text preview
    print("\n--- Discord embed preview (dry-run; use --post to send) ---")
    for embed in payload.get("embeds", []):
        print(f"\n  {embed.get('title', '?')}")
        print(f"  {embed.get('description', '')}")
        for f in embed.get("fields", []):
            print(f"\n  [{f['name']}]")
            for line in f["value"].splitlines():
                print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
