"""Memory browser router — surfaces hit-rate scoring + recall log stats."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from animus_bootstrap.config import ConfigManager

router = APIRouter()


@dataclass
class ScoredMemory:
    """A memory plus its hit-rate score, prepared for the template."""

    memory_id: str
    n_recalls: int
    raw_mean: float
    effective_score: float
    last_updated: str
    content: str
    memory_type: str


def _get_runtime(request: Request) -> object | None:
    return getattr(request.app.state, "runtime", None)


async def _resolve_scored(runtime, score_records, now, limit: int) -> list[ScoredMemory]:
    """Resolve a list of MemoryScore records to ScoredMemory dataclasses with content."""
    scored: list[ScoredMemory] = []
    score_store = runtime._memory_score_store
    memory_manager = runtime.memory_manager
    for record in score_records[:limit]:
        content = ""
        memory_type = ""
        if memory_manager is not None:
            try:
                resolved = await memory_manager.get_by_id(record.memory_id)
            except Exception:
                resolved = None
            if resolved:
                content = resolved.get("content", "")
                memory_type = resolved.get("type", "")
        scored.append(
            ScoredMemory(
                memory_id=record.memory_id,
                n_recalls=record.n_recalls,
                raw_mean=round(record.raw_mean, 4),
                effective_score=round(score_store.effective_score(record.memory_id, now=now), 4),
                last_updated=record.last_updated.isoformat(),
                content=content[:240],
                memory_type=memory_type,
            )
        )
    return scored


@router.get("/memory")
async def memory_page(request: Request) -> object:
    """Render the memory browser with hit-rate scoring panel."""
    templates = request.app.state.templates

    runtime = _get_runtime(request)
    score_stats = {"scored_memories": 0, "total_recalls_attributed": 0, "mean_score": 0.0}
    recall_count = 0
    top_memories: list[ScoredMemory] = []
    bottom_memories: list[ScoredMemory] = []

    score_store = getattr(runtime, "_memory_score_store", None) if runtime is not None else None
    recall_log = getattr(runtime, "_recall_log", None) if runtime is not None else None

    if score_store is not None:
        score_stats = score_store.stats()
        now = datetime.now(UTC)
        top_records = score_store.top_memories(limit=10)
        bottom_records = score_store.bottom_memories(limit=10)
        top_memories = await _resolve_scored(runtime, top_records, now, limit=10)
        bottom_memories = await _resolve_scored(runtime, bottom_records, now, limit=10)

    if recall_log is not None:
        try:
            recall_count = recall_log.recall_count()
        except Exception:
            recall_count = 0

    return templates.TemplateResponse(
        request,
        "memory.html",
        {
            "score_stats": score_stats,
            "recall_count": recall_count,
            "top_memories": top_memories,
            "bottom_memories": bottom_memories,
        },
    )


@router.get("/memory/scores")
async def memory_scores_json(request: Request) -> JSONResponse:
    """JSON endpoint for top + bottom memory scores."""
    runtime = _get_runtime(request)
    score_store = getattr(runtime, "_memory_score_store", None) if runtime is not None else None
    if score_store is None:
        raise HTTPException(status_code=503, detail="memory score store not initialized")
    now = datetime.now(UTC)
    top = await _resolve_scored(runtime, score_store.top_memories(limit=20), now, limit=20)
    bottom = await _resolve_scored(runtime, score_store.bottom_memories(limit=20), now, limit=20)
    return JSONResponse(
        content={
            "stats": score_store.stats(),
            "top": [s.__dict__ for s in top],
            "bottom": [s.__dict__ for s in bottom],
        }
    )


# Field-name hints for secret values. ``public`` is exempt so the VAPID
# *public* key (shared with the browser) is not redacted; redaction is
# string-only so int budgets like ``max_response_tokens`` are untouched.
_SECRET_HINTS: tuple[str, ...] = ("key", "token", "secret", "password", "private", "credential")


def _looks_secret(name: str) -> bool:
    n = name.lower()
    if "public" in n:
        return False
    return any(h in n for h in _SECRET_HINTS)


def _redact_secrets(obj: object) -> object:
    """Recursively redact secret-looking string fields from a dumped config."""
    if isinstance(obj, dict):
        return {
            k: (
                "***redacted***"
                if isinstance(v, str) and v and _looks_secret(k)
                else _redact_secrets(v)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_secrets(x) for x in obj]
    return obj


@router.get("/memory/export")
async def memory_export() -> JSONResponse:
    """Return a SECRET-REDACTED config snapshot (placeholder for memory export).

    Security (C92): the previous version returned ``cfg.model_dump()`` verbatim,
    leaking every API key, the auth token, the TLS key, and the VAPID private
    key. This endpoint is reachable by an authenticated remote client, so it
    must never serialize secrets — they are redacted by field name before
    leaving the box. (It is also now bearer-gated for remote clients by
    AuthMiddleware; this is the defense-in-depth second layer.)
    """
    config_manager = ConfigManager()
    cfg = config_manager.load()
    return JSONResponse(content=_redact_secrets(cfg.model_dump()))
