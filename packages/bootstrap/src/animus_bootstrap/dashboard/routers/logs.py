"""Log viewer router with SSE streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

_LOG_FILE = Path("~/.local/share/animus/animus.log").expanduser()


async def _tail_log() -> AsyncGenerator[dict[str, str], None]:
    """Yield new log lines as SSE events.

    Opens the log file, seeks to end, then watches for new content.
    If the file does not exist, sends a single event and stops.
    """
    if not _LOG_FILE.is_file():
        yield {"event": "log", "data": "No logs yet"}
        return

    with open(_LOG_FILE) as fh:
        # Seek to end — only stream new lines
        fh.seek(0, 2)

        while True:
            line = fh.readline()
            if line:
                yield {"event": "log", "data": line.rstrip("\n")}
            else:
                await asyncio.sleep(0.5)


@router.get("/logs")
async def logs_page(request: Request) -> object:
    """Render the log viewer page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "logs.html",
        {"request": request},
    )


@router.get("/logs/stream")
async def logs_stream() -> EventSourceResponse:
    """SSE endpoint that tails the Animus log file."""
    return EventSourceResponse(_tail_log())
