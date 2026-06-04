"""Quick-capture router — store a thought/note into memory from the PWA."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from animus_bootstrap.intelligence.tools.builtin.memory_tools import _store_memory

logger = logging.getLogger(__name__)

router = APIRouter()


class CaptureRequest(BaseModel):
    """A captured note to persist into memory."""

    text: str
    memory_type: str = "episodic"


@router.post("/api/capture")
async def capture(payload: CaptureRequest) -> JSONResponse:
    """Persist a quick-capture note via the memory store.

    Reuses the memory tool's store path, which delegates to the live
    MemoryManager when wired and falls back to an in-memory list otherwise.
    """
    text = payload.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"detail": "Empty capture."})

    result = await _store_memory(text, payload.memory_type)
    logger.info("Quick-capture stored (%d chars)", len(text))
    return JSONResponse(content={"ok": True, "message": result})
