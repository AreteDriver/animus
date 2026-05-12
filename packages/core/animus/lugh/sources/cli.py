"""Lugh external-sources CLI.

Examples:
    python -m animus.lugh.sources.cli list
    python -m animus.lugh.sources.cli fetch --limit 5
    python -m animus.lugh.sources.cli fetch --source arxiv:cs.LG --limit 3
    python -m animus.lugh.sources.cli probe https://feeds.example.com/rss
    python -m animus.lugh.sources.cli recent --source hn:front_page --limit 10
    python -m animus.lugh.sources.cli add-podcast all-in https://... --name "All-In"
    python -m animus.lugh.sources.cli add-youtube @AIDailyBrief --name "The AI Daily Brief"
    python -m animus.lugh.sources.cli probe-youtube @LatentSpacePod
    python -m animus.lugh.sources.cli remove podcast:all-in
    python -m animus.lugh.sources.cli explain <source_id> <item_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from animus.lugh.sources.base import SourceCache
from animus.lugh.sources.podcasts import probe_feed
from animus.lugh.sources.registry import (
    add_podcast,
    add_youtube,
    instantiate,
    load_registry,
    remove_source,
)
from animus.lugh.sources.relevance import default_scorer
from animus.lugh.sources.youtube import probe_channel


def cmd_list(args: argparse.Namespace) -> int:
    entries = load_registry(path=args.registry)
    sources = instantiate(entries)
    if args.json:
        print(
            json.dumps(
                [
                    {"kind": e.get("kind"), "entry": e, "source_id": s.source_id}
                    for e, s in zip(entries, sources, strict=False)
                ],
                indent=2,
            )
        )
        return 0
    if not sources:
        print("No sources configured. Run `cli add-podcast` or edit registry.")
        return 0
    for s in sources:
        print(f"  {s.source_id}")
    print(f"\n  total: {len(sources)}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    entries = load_registry(path=args.registry)
    sources = instantiate(entries)
    if args.source:
        sources = [s for s in sources if s.source_id == args.source]
        if not sources:
            print(f"No source matching {args.source}", file=sys.stderr)
            return 1

    cache = SourceCache(path=args.cache)
    scorer = default_scorer()
    totals: dict[str, dict[str, int]] = {}

    for s in sources:
        items = list(s.fetch(limit=args.limit))
        result = cache.put_many(items, scorer=scorer)
        totals[s.source_id] = result
        if not args.quiet:
            print(
                f"{s.source_id:28}  fetched={len(items):3}  "
                f"new={result['inserted']:3}  dedup={result['skipped']:3}"
            )

    if args.json:
        print(json.dumps(totals, indent=2))
    elif not args.quiet:
        grand_new = sum(r["inserted"] for r in totals.values())
        grand_dup = sum(r["skipped"] for r in totals.values())
        print(f"\n  total new: {grand_new}, dedup: {grand_dup}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    result = probe_feed(args.url)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def cmd_recent(args: argparse.Namespace) -> int:
    cache = SourceCache(path=args.cache)
    items = cache.recent(
        source_id=args.source,
        limit=args.limit,
        min_score=args.min_score,
    )
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "source_id": i.source_id,
                        "item_id": i.item_id,
                        "title": i.title,
                        "url": i.url,
                        "published": i.published.isoformat() if i.published else None,
                        "summary": i.summary,
                    }
                    for i in items
                ],
                indent=2,
            )
        )
        return 0
    for i in items:
        pub = i.published.strftime("%Y-%m-%d") if i.published else "????-??-??"
        print(f"  {pub}  {i.source_id:22}  {i.title[:80]}")
    print(f"\n  {len(items)} items")
    return 0


def cmd_add_podcast(args: argparse.Namespace) -> int:
    ok = add_podcast(
        show_id=args.show_id,
        feed_url=args.feed_url,
        show_name=args.name or "",
        fetch_transcripts=args.transcripts,
        path=args.registry,
    )
    if not ok:
        print(f"podcast:{args.show_id} already exists", file=sys.stderr)
        return 1
    print(f"added podcast:{args.show_id} -> {args.feed_url}")
    return 0


def cmd_add_youtube(args: argparse.Namespace) -> int:
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    ok = add_youtube(
        channel=args.channel,
        show_name=args.name or "",
        fetch_captions=not args.no_captions,
        tags=tags,
        path=args.registry,
    )
    if not ok:
        print(f"youtube:{args.channel} already exists", file=sys.stderr)
        return 1
    mode = "list-only" if args.no_captions else "with captions"
    print(f"added youtube:{args.channel} ({mode})")
    return 0


def cmd_probe_youtube(args: argparse.Namespace) -> int:
    result = probe_channel(args.channel)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def cmd_remove(args: argparse.Namespace) -> int:
    if remove_source(args.source_id, path=args.registry):
        print(f"removed {args.source_id}")
        return 0
    print(f"no source matching {args.source_id}", file=sys.stderr)
    return 1


def cmd_explain(args: argparse.Namespace) -> int:
    entries = load_registry(path=args.registry)
    sources = {s.source_id: s for s in instantiate(entries)}
    source = sources.get(args.source_id)
    if source is None:
        print(f"no source {args.source_id}", file=sys.stderr)
        return 1
    items = [i for i in source.fetch() if i.item_id == args.item_id]
    if not items:
        print(f"no item {args.item_id} in {args.source_id}", file=sys.stderr)
        return 1
    scorer = default_scorer()
    explanation = scorer.explain(items[0])
    explanation["item"] = {
        "title": items[0].title,
        "url": items[0].url,
    }
    print(json.dumps(explanation, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="animus.lugh.sources.cli")
    ap.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry path (defaults to ~/.animus/lugh_sources.json)",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache path (defaults to ~/.animus/lugh_sources.sqlite)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub_list = sub.add_parser("list", help="List configured sources")
    sub_list.add_argument("--json", action="store_true")
    sub_list.set_defaults(func=cmd_list)

    sub_fetch = sub.add_parser("fetch", help="Fetch items from all (or one) source")
    sub_fetch.add_argument("--source", help="Restrict to a single source_id")
    sub_fetch.add_argument("--limit", type=int, default=None)
    sub_fetch.add_argument("--json", action="store_true")
    sub_fetch.add_argument("--quiet", action="store_true")
    sub_fetch.set_defaults(func=cmd_fetch)

    sub_probe = sub.add_parser("probe", help="Validate a feed URL before adding")
    sub_probe.add_argument("url")
    sub_probe.set_defaults(func=cmd_probe)

    sub_recent = sub.add_parser("recent", help="Show recently-harvested items")
    sub_recent.add_argument("--source", default=None)
    sub_recent.add_argument("--limit", type=int, default=20)
    sub_recent.add_argument(
        "--min-score", type=float, default=None, dest="min_score", help="Filter by relevance"
    )
    sub_recent.add_argument("--json", action="store_true")
    sub_recent.set_defaults(func=cmd_recent)

    sub_add = sub.add_parser("add-podcast", help="Register a podcast feed")
    sub_add.add_argument("show_id")
    sub_add.add_argument("feed_url")
    sub_add.add_argument("--name", default=None)
    sub_add.add_argument("--transcripts", action="store_true", help="Fetch iTunes transcripts")
    sub_add.set_defaults(func=cmd_add_podcast)

    sub_add_yt = sub.add_parser("add-youtube", help="Register a YouTube channel (via yt-dlp)")
    sub_add_yt.add_argument("channel", help="Channel handle, e.g. @AIDailyBrief")
    sub_add_yt.add_argument("--name", default=None)
    sub_add_yt.add_argument("--tags", default=None, help="Comma-separated tags")
    sub_add_yt.add_argument(
        "--no-captions",
        action="store_true",
        dest="no_captions",
        help="List videos only; don't download auto-captions",
    )
    sub_add_yt.set_defaults(func=cmd_add_youtube)

    sub_probe_yt = sub.add_parser("probe-youtube", help="Validate a YouTube channel handle")
    sub_probe_yt.add_argument("channel")
    sub_probe_yt.set_defaults(func=cmd_probe_youtube)

    sub_remove = sub.add_parser("remove", help="Remove a source by source_id")
    sub_remove.add_argument("source_id")
    sub_remove.set_defaults(func=cmd_remove)

    sub_explain = sub.add_parser("explain", help="Explain relevance score for an item")
    sub_explain.add_argument("source_id")
    sub_explain.add_argument("item_id")
    sub_explain.set_defaults(func=cmd_explain)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
