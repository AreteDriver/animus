"""Ogma CLI verbs — wires the ``read`` flow to ``/ogma read <id>``.

Mounted from ``animus/__main__.py``'s REPL command tree. ``brief``/``gap``/
``audit``/``sweep`` print "deferred" stubs in v0 — they fill in in v0.1+
per ``docs/OGMA.md`` §Roadmap.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from animus.lugh.sources.base import SourceCache
from animus.ogma.read import OgmaSynthesisError, synthesize

console = Console()

USAGE = (
    "ogma read <source_id>:<item_id>     # synthesize one cached item\n"
    "ogma brief                          # deferred to v0.1\n"
    'ogma gap "<concept>"                # deferred to v0.1\n'
    "ogma audit <target>                 # deferred to v0.x\n"
    "ogma sweep                          # deferred to v0.x"
)


def handle_ogma(argv: list[str], *, cache: SourceCache | None = None) -> int:
    """Dispatch one ``ogma <verb> [...args]`` invocation.

    Args:
        argv: tokens after ``ogma`` (e.g. ``["read", "youtube:@AIDailyBrief:JWS8bYZrtnQ"]``).
        cache: an open ``SourceCache``; defaults to the on-disk one.

    Returns:
        ``0`` on success, non-zero on any error path (unknown verb, cache miss, etc).
    """
    if not argv:
        console.print(USAGE)
        return 2
    verb = argv[0]
    rest = argv[1:]
    if verb == "read":
        return _handle_read(rest, cache=cache)
    if verb in ("brief", "gap", "audit", "sweep"):
        console.print(f"[yellow]ogma {verb}: deferred — not implemented in v0[/yellow]")
        return 0
    console.print(f"[red]unknown verb: {verb}[/red]\n{USAGE}")
    return 2


def _handle_read(argv: list[str], *, cache: SourceCache | None) -> int:
    if not argv:
        console.print("[red]usage: ogma read <source_id>:<item_id>[/red]")
        return 2
    target = argv[0]
    if ":" not in target:
        console.print(
            "[red]ogma read: expected <source_id>:<item_id> (e.g. youtube:@AIDailyBrief:JWS8bYZrtnQ)[/red]"
        )
        return 2
    source_id, _, item_id = target.rpartition(":")
    if not source_id or not item_id:
        console.print(f"[red]ogma read: malformed identifier {target!r}[/red]")
        return 2
    cache = cache or SourceCache()
    item = _lookup(cache, source_id, item_id)
    if item is None:
        console.print(f"[red]not in lugh cache: {source_id}:{item_id}[/red]")
        return 1
    try:
        output = synthesize(item)
    except OgmaSynthesisError as e:
        console.print(f"[red]ogma read: synthesis failed: {e}[/red]")
        return 1
    if output is None:
        console.print("[yellow]ogma read: item gated out by relevance threshold[/yellow]")
        return 0
    written = (
        Path("~/projects/notes/ogma").expanduser() / f"{output.date.isoformat()}-{output.slug}.md"
    )
    console.print(f"[green]ogma read: wrote {written}[/green]")
    return 0


def _lookup(cache: SourceCache, source_id: str, item_id: str):
    """Find a cached item by ``(source_id, item_id)``.

    SourceCache exposes ``recent(source_id=..., limit=N)`` — we scan the
    most recent 500 items for that source and match on ``item_id``. Cheap
    enough at current cache scale; gets a proper indexed lookup if v0.1
    needs it.
    """
    for cached in cache.recent(source_id=source_id, limit=500):
        if cached.item_id == item_id:
            return cached
    return None
