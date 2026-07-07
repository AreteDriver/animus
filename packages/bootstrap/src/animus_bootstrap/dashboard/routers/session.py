"""Session continuity router — load/save Head checkpoints for the PWA.

Bridges the kernel's SQLite-backed checkpoint store to the Bootstrap dashboard
API so the PWA can resume conversation context across browser sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Checkpoint shape returned to / loaded from the PWA
@dataclass
class CheckpointPayload:
    messages: list[dict[str, str]]
    summary: str
    turns: int
    session_id: str | None = None


def _checkpoint_store() -> Any | None:
    """Import the kernel checkpoint store lazily to avoid bootstrap → kernel
tight coupling at import time."""
    try:
        from animus_kernel.head.checkpoint import HeadCheckpointStore

        db_path = Path.home() / ".animus" / "sessions" / "head.db"
        return HeadCheckpointStore(db_path=db_path)
    except Exception as exc:
        logger.warning("Could not initialise HeadCheckpointStore: %s", exc)
        return None


@router.get("/api/session/checkpoint")
async def get_checkpoint(request: Request) -> JSONResponse:
    """Return the most recent HeadCheckpoint for this PWA session.

    Falls back to an empty checkpoint when the store is unreadable.
    """
    store = _checkpoint_store()
    if store is None:
        return JSONResponse(content={"messages": [], "summary": "", "turns": 0})

    recent = store.list_recent(limit=1)
    if not recent:
        return JSONResponse(content={"messages": [], "summary": "", "turns": 0})

    cp = recent[0]
    return JSONResponse(
        content={
            "session_id": cp.session_id,
            "messages": cp.messages,
            "summary": cp.summary,
            "turns": cp.turns,
        }
    )


@router.post("/api/session/checkpoint")
async def post_checkpoint(request: Request) -> JSONResponse:
    """Persist a checkpoint from the PWA into the kernel store.

    Creates or overwrites the checkpoint for the given session_id.
    """
    body = await request.json()
    session_id = body.get("session_id", "pwa-session")
    messages = body.get("messages", [])
    summary = body.get("summary", "")
    turns = body.get("turns", 0)

    store = _checkpoint_store()
    if store is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Checkpoint store unavailable."},
        )

    try:
        from animus_kernel.head.checkpoint import HeadCheckpoint

        cp = HeadCheckpoint(
            session_id=session_id,
            started_at=datetime.now(UTC),
            last_active_at=datetime.now(UTC),
            messages=messages,
            summary=summary,
            turns=turns,
        )
        store.save(cp)
    except Exception as exc:
        logger.exception("Failed to save checkpoint")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Save failed: {exc}"},
        )

    return JSONResponse(content={"saved": True, "session_id": session_id})
