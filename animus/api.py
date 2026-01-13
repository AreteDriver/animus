"""
Animus HTTP API

FastAPI-based REST API for external access to Animus functionality.

Requires: pip install 'animus[api]'
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from animus.cognitive import CognitiveLayer, ReasoningMode, detect_mode
from animus.config import AnimusConfig
from animus.decision import DecisionFramework
from animus.logging import get_logger
from animus.memory import Conversation, MemoryLayer, MemoryType
from animus.tasks import TaskStatus, TaskTracker
from animus.tools import ToolRegistry

if TYPE_CHECKING:
    pass

logger = get_logger("api")

# Check for FastAPI dependency
try:
    import uvicorn
    from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
    from fastapi.security import APIKeyHeader

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# =============================================================================
# Request/Response Models (Pydantic, always available)
# =============================================================================


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., description="User message")
    mode: str = Field(default="auto", description="Reasoning mode: auto, quick, deep, research")
    conversation_id: str | None = Field(default=None, description="Continue existing conversation")


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    conversation_id: str
    mode_used: str


class MemoryCreate(BaseModel):
    """Create memory request."""

    content: str
    memory_type: str = "semantic"
    tags: list[str] = []
    source: str = "stated"
    confidence: float = 1.0


class MemoryResponse(BaseModel):
    """Memory response model."""

    id: str
    content: str
    memory_type: str
    tags: list[str]
    source: str
    confidence: float
    created_at: str
    updated_at: str


class MemorySearchResponse(BaseModel):
    """Memory search response."""

    memories: list[MemoryResponse]
    count: int


class TaskCreate(BaseModel):
    """Create task request."""

    description: str
    tags: list[str] = []
    priority: int = 0


class TaskResponse(BaseModel):
    """Task response model."""

    id: str
    description: str
    status: str
    tags: list[str]
    priority: int
    created_at: str


class TaskUpdate(BaseModel):
    """Update task request."""

    status: str | None = None
    description: str | None = None


class ToolExecute(BaseModel):
    """Tool execution request."""

    params: dict = {}


class ToolResponse(BaseModel):
    """Tool response model."""

    name: str
    success: bool
    output: str | None
    error: str | None


class DecisionRequest(BaseModel):
    """Decision analysis request."""

    question: str
    options: list[str] | None = None
    criteria: list[str] | None = None


class StatusResponse(BaseModel):
    """System status response."""

    status: str
    version: str
    memory_count: int
    task_count: int
    model_provider: str
    model_name: str


class BriefResponse(BaseModel):
    """Briefing response."""

    briefing: str
    topic: str | None


# =============================================================================
# Application State
# =============================================================================


@dataclass
class AppState:
    """Shared application state."""

    config: AnimusConfig
    memory: MemoryLayer
    cognitive: CognitiveLayer
    tools: ToolRegistry
    tasks: TaskTracker
    decisions: DecisionFramework
    conversations: dict[str, Conversation]


_state: AppState | None = None


# =============================================================================
# API Server Class
# =============================================================================


class APIServer:
    """
    Manages the API server lifecycle.

    Provides a simple interface to start/stop the FastAPI server
    from the CLI without blocking the main thread.
    """

    def __init__(
        self,
        memory: MemoryLayer,
        cognitive: CognitiveLayer,
        tools: ToolRegistry,
        tasks: TaskTracker,
        decisions: DecisionFramework,
        host: str = "127.0.0.1",
        port: int = 8420,
        api_key: str | None = None,
    ):
        """
        Initialize API server.

        Args:
            memory: MemoryLayer instance
            cognitive: CognitiveLayer instance
            tools: ToolRegistry instance
            tasks: TaskTracker instance
            decisions: DecisionFramework instance
            host: Host to bind to
            port: Port to bind to
            api_key: Optional API key for authentication
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Install with: pip install 'animus[api]'")

        self.memory = memory
        self.cognitive = cognitive
        self.tools = tools
        self.tasks = tasks
        self.decisions = decisions
        self.host = host
        self.port = port
        self.api_key = api_key

        self._server_thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._is_running = False

    def start(self) -> bool:
        """Start the API server in a background thread."""
        global _state

        if self._is_running:
            logger.warning("Server already running")
            return False

        # Initialize global state
        _state = AppState(
            config=AnimusConfig.load(),
            memory=self.memory,
            cognitive=self.cognitive,
            tools=self.tools,
            tasks=self.tasks,
            decisions=self.decisions,
            conversations={},
        )

        # Update config with API key if provided
        if self.api_key:
            _state.config.api.api_key = self.api_key

        def run_server():
            import asyncio

            # Create app inside thread to avoid issues
            app = create_app()

            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                log_level="warning",
            )
            self._server = uvicorn.Server(config)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._server.serve())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self._is_running = True

        logger.info(f"API server started on http://{self.host}:{self.port}")
        return True

    def stop(self) -> bool:
        """Stop the API server."""
        if not self._is_running:
            return False

        if self._server:
            self._server.should_exit = True

        self._is_running = False
        logger.info("API server stopped")
        return True

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._is_running

    @property
    def url(self) -> str:
        """Get server URL."""
        return f"http://{self.host}:{self.port}"


# =============================================================================
# FastAPI App Factory
# =============================================================================


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory function creates the app with all routes registered.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Install with: pip install 'animus[api]'")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        """Application lifespan management."""
        logger.info("Animus API starting up")
        yield
        logger.info("Animus API shutting down")

    app = FastAPI(
        title="Animus API",
        description="Personal AI exocortex API",
        version="0.3.0",
        lifespan=lifespan,
    )

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def get_state() -> AppState:
        """Get application state."""
        if _state is None:
            raise HTTPException(status_code=503, detail="Server not initialized")
        return _state

    async def verify_api_key(api_key: str | None = Depends(api_key_header)) -> bool:
        """Verify API key if configured."""
        state = get_state()
        configured_key = state.config.api.api_key

        if configured_key is None:
            return True  # No auth required

        if api_key is None or api_key != configured_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return True

    # =========================================================================
    # Endpoints
    # =========================================================================

    @app.get("/status", response_model=StatusResponse)
    async def get_status():
        """Get system status."""
        state = get_state()
        memories = state.memory.store.list_all()
        tasks = state.tasks.list(include_completed=True)

        return StatusResponse(
            status="running",
            version="0.3.0",
            memory_count=len(memories),
            task_count=len(tasks),
            model_provider=state.config.model.provider,
            model_name=state.config.model.name,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, _auth: bool = Depends(verify_api_key)):
        """Send a chat message and get a response."""
        state = get_state()

        # Get or create conversation
        if request.conversation_id and request.conversation_id in state.conversations:
            conversation = state.conversations[request.conversation_id]
        else:
            conversation = Conversation.new()
            state.conversations[conversation.id] = conversation

        # Determine mode
        if request.mode == "auto":
            mode = detect_mode(request.message)
        else:
            mode = ReasoningMode(request.mode)

        # Get context from memory
        context_memories = state.memory.recall(request.message, limit=3)
        context = "\n".join(m.content for m in context_memories) if context_memories else None

        # Generate response
        conversation.add_message("user", request.message)
        response = state.cognitive.think(request.message, context=context, mode=mode)
        conversation.add_message("assistant", response)

        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            mode_used=mode.value,
        )

    # Memory endpoints

    @app.post("/memory", response_model=MemoryResponse)
    async def create_memory(request: MemoryCreate, _auth: bool = Depends(verify_api_key)):
        """Create a new memory."""
        state = get_state()

        try:
            mem_type = MemoryType(request.memory_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid memory type: {request.memory_type}"
            )

        memory = state.memory.remember(
            content=request.content,
            memory_type=mem_type,
            tags=request.tags,
            source=request.source,
            confidence=request.confidence,
        )

        return MemoryResponse(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type.value,
            tags=memory.tags,
            source=memory.source,
            confidence=memory.confidence,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        )

    @app.get("/memory/search", response_model=MemorySearchResponse)
    async def search_memories(
        query: str = Query(..., description="Search query"),
        limit: int = Query(default=10, le=100),
        tags: str | None = Query(default=None, description="Comma-separated tags"),
        _auth: bool = Depends(verify_api_key),
    ):
        """Search memories."""
        state = get_state()

        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        memories = state.memory.recall(query, tags=tag_list, limit=limit)

        return MemorySearchResponse(
            memories=[
                MemoryResponse(
                    id=m.id,
                    content=m.content,
                    memory_type=m.memory_type.value,
                    tags=m.tags,
                    source=m.source,
                    confidence=m.confidence,
                    created_at=m.created_at.isoformat(),
                    updated_at=m.updated_at.isoformat(),
                )
                for m in memories
            ],
            count=len(memories),
        )

    @app.get("/memory/{memory_id}", response_model=MemoryResponse)
    async def get_memory(memory_id: str, _auth: bool = Depends(verify_api_key)):
        """Get a specific memory by ID."""
        state = get_state()
        memory = state.memory.get_memory(memory_id)

        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        return MemoryResponse(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type.value,
            tags=memory.tags,
            source=memory.source,
            confidence=memory.confidence,
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
        )

    @app.delete("/memory/{memory_id}")
    async def delete_memory(memory_id: str, _auth: bool = Depends(verify_api_key)):
        """Delete a memory."""
        state = get_state()

        if state.memory.forget(memory_id):
            return {"status": "deleted", "id": memory_id}
        else:
            raise HTTPException(status_code=404, detail="Memory not found")

    # Tool endpoints

    @app.get("/tools")
    async def list_tools(_auth: bool = Depends(verify_api_key)):
        """List available tools."""
        state = get_state()
        tools = state.tools.list_tools()

        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "requires_approval": t.requires_approval,
                }
                for t in tools
            ]
        }

    @app.post("/tools/{tool_name}", response_model=ToolResponse)
    async def execute_tool(
        tool_name: str, request: ToolExecute, _auth: bool = Depends(verify_api_key)
    ):
        """Execute a tool."""
        state = get_state()

        tool = state.tools.get(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

        if tool.requires_approval:
            raise HTTPException(
                status_code=403,
                detail=f"Tool '{tool_name}' requires approval (not available via API)",
            )

        result = state.tools.execute(tool_name, request.params)

        return ToolResponse(
            name=result.tool_name,
            success=result.success,
            output=str(result.output) if result.output else None,
            error=result.error,
        )

    # Task endpoints

    @app.get("/tasks")
    async def list_tasks(
        status: str | None = Query(default=None),
        include_completed: bool = Query(default=False),
        _auth: bool = Depends(verify_api_key),
    ):
        """List tasks."""
        state = get_state()

        status_filter = TaskStatus(status) if status else None
        tasks = state.tasks.list(status=status_filter, include_completed=include_completed)

        return {
            "tasks": [
                TaskResponse(
                    id=t.id,
                    description=t.description,
                    status=t.status.value,
                    tags=t.tags,
                    priority=t.priority,
                    created_at=t.created_at.isoformat(),
                )
                for t in tasks
            ]
        }

    @app.post("/tasks", response_model=TaskResponse)
    async def create_task(request: TaskCreate, _auth: bool = Depends(verify_api_key)):
        """Create a new task."""
        state = get_state()

        task = state.tasks.add(
            description=request.description,
            tags=request.tags,
            priority=request.priority,
        )

        return TaskResponse(
            id=task.id,
            description=task.description,
            status=task.status.value,
            tags=task.tags,
            priority=task.priority,
            created_at=task.created_at.isoformat(),
        )

    @app.patch("/tasks/{task_id}", response_model=TaskResponse)
    async def update_task(task_id: str, request: TaskUpdate, _auth: bool = Depends(verify_api_key)):
        """Update a task."""
        state = get_state()

        task = state.tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if request.status:
            try:
                state.tasks.update_status(task_id, TaskStatus(request.status))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {request.status}")

        task = state.tasks.get(task_id)

        return TaskResponse(
            id=task.id,
            description=task.description,
            status=task.status.value,
            tags=task.tags,
            priority=task.priority,
            created_at=task.created_at.isoformat(),
        )

    @app.delete("/tasks/{task_id}")
    async def delete_task(task_id: str, _auth: bool = Depends(verify_api_key)):
        """Delete a task."""
        state = get_state()

        if state.tasks.delete(task_id):
            return {"status": "deleted", "id": task_id}
        else:
            raise HTTPException(status_code=404, detail="Task not found")

    # Decision endpoint

    @app.post("/decide")
    async def analyze_decision(request: DecisionRequest, _auth: bool = Depends(verify_api_key)):
        """Perform decision analysis."""
        state = get_state()

        decision = state.decisions.analyze(
            question=request.question,
            options=request.options,
            criteria=request.criteria,
        )

        return {
            "question": decision.question,
            "options": decision.options,
            "criteria": decision.criteria,
            "analysis": decision.analysis,
            "recommendation": decision.recommendation,
            "reasoning": decision.reasoning,
        }

    # Briefing endpoint

    @app.get("/brief", response_model=BriefResponse)
    async def get_briefing(
        topic: str | None = Query(default=None), _auth: bool = Depends(verify_api_key)
    ):
        """Generate a situation briefing."""
        state = get_state()

        briefing = state.cognitive.brief(state.memory, topic=topic)

        return BriefResponse(briefing=briefing, topic=topic)

    # WebSocket endpoint

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """WebSocket endpoint for streaming chat."""
        await websocket.accept()
        state = get_state()

        conversation = Conversation.new()
        state.conversations[conversation.id] = conversation

        try:
            while True:
                data = await websocket.receive_json()
                message = data.get("message", "")

                if not message:
                    await websocket.send_json({"error": "No message provided"})
                    continue

                # Get mode
                mode_str = data.get("mode", "auto")
                if mode_str == "auto":
                    mode = detect_mode(message)
                else:
                    mode = ReasoningMode(mode_str)

                # Get context
                context_memories = state.memory.recall(message, limit=3)
                context = (
                    "\n".join(m.content for m in context_memories) if context_memories else None
                )

                # Generate response
                conversation.add_message("user", message)
                response = state.cognitive.think(message, context=context, mode=mode)
                conversation.add_message("assistant", response)

                await websocket.send_json(
                    {
                        "response": response,
                        "conversation_id": conversation.id,
                        "mode_used": mode.value,
                    }
                )

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {conversation.id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close(code=1011)

    return app
