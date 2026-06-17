"""FastAPI chat endpoint for the Animus kernel.

Routes POST /chat requests through TerminalAgent and streams
SSE tokens produced by the local OllamaProvider.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from animus_kernel.budget.manager import BudgetConfig, BudgetManager
from animus_kernel.providers.base import CompletionRequest
from animus_kernel.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

# In-memory budget for the mobile UI (600 ET ceiling, TASK-007)
_budget_config = BudgetConfig(total_budget=600)
_budget_manager = BudgetManager(config=_budget_config)

# Placeholder build queue until builder integration lands
_build_queue: list[dict] = []

app = FastAPI(title="Animus Kernel Chat")

_cors_origins_raw = os.environ.get("ANIMUS_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
CORS_ORIGINS = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Incoming chat payload."""

    message: str


def _error_response(error: str, detail: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {"error": error, "detail": detail},
        status_code=status_code,
    )


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream an LLM response for the supplied instruction.

    The message is forwarded to Ollama via OllamaProvider; tokens are
    emitted as Server-Sent Events and the stream terminates with a
    ``done`` event.
    """
    provider = OllamaProvider()

    if not provider.is_configured():
        return _error_response(
            "Service Unavailable",
            "Ollama is not reachable. Make sure Ollama is running.",
            503,
        )

    try:
        provider.initialize()
    except Exception as exc:
        logger.warning("Failed to initialise OllamaProvider: %s", exc)
        return _error_response(
            "Service Unavailable",
            "Ollama is not reachable. Make sure Ollama is running.",
            503,
        )

    completion_req = CompletionRequest(prompt=request.message)

    async def event_generator():
        streamed_chars = 0
        try:
            async for chunk in provider.complete_stream_async(completion_req):
                if chunk.content:
                    streamed_chars += len(chunk.content)
                    payload = json.dumps({"token": chunk.content})
                    yield f"data: {payload}\n\n"
            yield 'event: done\ndata: {"status":"complete"}\n\n'
        except Exception as exc:
            logger.exception("Streaming error")
            payload = json.dumps(
                {"error": type(exc).__name__, "detail": str(exc)}
            )
            yield f"event: error\ndata: {payload}\n\n"
        finally:
            # Rough token estimate (1 token ~ 4 characters) for budget sync
            if streamed_chars:
                tokens_est = max(1, streamed_chars // 4)
                _budget_manager.record_usage(
                    agent_id="chat_user",
                    tokens=tokens_est,
                    operation="chat_completion",
                    model="ollama",
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.get("/api/budget")
async def get_budget():
    """Return current budget state for the mobile UI."""
    return {
        "total": _budget_manager.total_budget,
        "used": _budget_manager.used,
        "remaining": _budget_manager.remaining,
        "status": _budget_manager.status.value,
        "percent": round(_budget_manager.usage_percent, 1),
    }


@app.get("/api/queue")
async def get_queue():
    """Return the active build queue."""
    return _build_queue


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    return _error_response(
        error=exc.detail if isinstance(exc.detail, str) else "HTTP Error",
        detail=str(exc.detail),
        status_code=exc.status_code,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _error_response(
        error="Validation Error",
        detail=str(exc),
        status_code=400,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception("Unhandled exception in request")
    return _error_response(
        error=type(exc).__name__,
        detail=str(exc),
        status_code=500,
    )


# Serve the mobile UI static files (TASK-007)
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
