"""Central runtime — boots and holds all live components."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus_bootstrap.config import ConfigManager
from animus_bootstrap.config.schema import AnimusConfig

if TYPE_CHECKING:
    from animus_bootstrap.gateway.session import SessionManager
    from animus_bootstrap.identity import IdentityFileManager
    from animus_bootstrap.intelligence.feedback import FeedbackStore
    from animus_bootstrap.intelligence.memory import MemoryManager
    from animus_bootstrap.intelligence.proactive.engine import ProactiveEngine
    from animus_bootstrap.intelligence.router import IntelligentRouter
    from animus_bootstrap.intelligence.tools.executor import ToolExecutor
    from animus_bootstrap.intelligence.tools.mcp_bridge import MCPBridge
    from animus_bootstrap.personas.context import ContextAdapter
    from animus_bootstrap.personas.engine import PersonaEngine

logger = logging.getLogger(__name__)


class AnimusRuntime:
    """Central runtime — boots and holds all live components.

    The runtime is the single place that wires config to components.
    Dashboard, CLI, and daemon all use it to get live references.
    """

    def __init__(self, config: AnimusConfig | None = None) -> None:
        self._config = config or ConfigManager().load()
        self._started = False

        # Component references (populated by start())
        self.identity_manager: IdentityFileManager | None = None
        self.feedback_store: FeedbackStore | None = None
        self.session_manager: SessionManager | None = None
        self.memory_manager: MemoryManager | None = None
        self.tool_executor: ToolExecutor | None = None
        self.proactive_engine: ProactiveEngine | None = None
        self.automation_engine: Any = None
        self.router: IntelligentRouter | None = None
        self.cognitive_backend: Any = None
        self.persona_engine: PersonaEngine | None = None
        self.context_adapter: ContextAdapter | None = None
        self._mcp_bridge: MCPBridge | None = None

    @property
    def config(self) -> AnimusConfig:
        return self._config

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        """Initialize and start all components based on config."""
        if self._started:
            logger.warning("Runtime already started")
            return

        logger.info("Animus runtime starting...")
        data_dir = self._config.get_data_path()
        data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Identity file manager
        from animus_bootstrap.identity.manager import IdentityFileManager

        identity_dir = Path(self._config.identity.identity_dir).expanduser()
        self.identity_manager = IdentityFileManager(identity_dir)
        logger.info("Identity manager initialized: %s", identity_dir)

        # 2. Session manager
        from animus_bootstrap.gateway.session import SessionManager

        session_db = data_dir / "sessions.db"
        self.session_manager = SessionManager(session_db)
        logger.info("Session manager initialized: %s", session_db)

        # 2. Cognitive backend
        self.cognitive_backend = self._create_cognitive_backend()
        logger.info("Cognitive backend: %s", self._config.gateway.default_backend)

        # 3. Memory manager (if intelligence enabled)
        if self._config.intelligence.enabled:
            self.memory_manager = self._create_memory_manager()
            logger.info(
                "Memory manager initialized: %s",
                self._config.intelligence.memory_backend,
            )

        # 4. Tool executor (if intelligence enabled)
        if self._config.intelligence.enabled:
            self.tool_executor = self._create_tool_executor()

            # Persistent tool history store
            from animus_bootstrap.intelligence.tools.history_store import (
                ToolHistoryStore,
            )

            history_db = data_dir / "tool_history.db"
            self._tool_history_store = ToolHistoryStore(history_db)
            self.tool_executor.set_history_store(self._tool_history_store)
            logger.info("Tool history store initialized: %s", history_db)

            logger.info(
                "Tool executor initialized: %d tools registered",
                len(self.tool_executor.list_tools()),
            )
            # MCP tool auto-discovery
            if self._config.intelligence.mcp.auto_discover:
                await self._discover_mcp_tools()

            # Wire self-improvement dependencies
            from animus_bootstrap.intelligence.tools.builtin.self_improve import (
                set_improvement_store,
                set_self_improve_deps,
            )

            set_self_improve_deps(self.tool_executor, self.cognitive_backend)

            # Persistent improvement store
            from animus_bootstrap.intelligence.tools.builtin.improvement_store import (
                ImprovementStore,
            )

            improvements_db = data_dir / "improvements.db"
            self._improvement_store = ImprovementStore(improvements_db)
            set_improvement_store(self._improvement_store)
            logger.info("Improvement store initialized: %s", improvements_db)

            # Wire identity tools to live identity manager
            if self.identity_manager is not None:
                from animus_bootstrap.intelligence.tools.builtin.identity_tools import (
                    set_identity_improvement_store,
                    set_identity_manager,
                )

                set_identity_manager(self.identity_manager)
                set_identity_improvement_store(self._improvement_store)
                logger.info("Identity tools wired to live identity manager")

            # Wire memory tools to live memory manager
            if self.memory_manager is not None:
                from animus_bootstrap.intelligence.tools.builtin.memory_tools import (
                    set_memory_manager,
                )

                set_memory_manager(self.memory_manager)

        # 5. Automation engine (if intelligence enabled)
        if self._config.intelligence.enabled:
            from animus_bootstrap.intelligence.automations.engine import AutomationEngine

            automations_db = data_dir / "automations.db"
            self.automation_engine = AutomationEngine(automations_db)
            logger.info("Automation engine initialized")

        # 6. Persona engine
        if self._config.personas.enabled:
            self.persona_engine = self._create_persona_engine()
            from animus_bootstrap.personas.context import ContextAdapter

            self.context_adapter = ContextAdapter()
            logger.info(
                "Persona engine initialized: %d personas",
                self.persona_engine.persona_count,
            )

        # 7. Router (intelligent if components available, basic otherwise)
        self.router = self._create_router()
        logger.info("Message router initialized")

        # Wire gateway tools to live router
        if self.tool_executor is not None:
            from animus_bootstrap.intelligence.tools.builtin.gateway_tools import (
                set_gateway_router,
            )

            set_gateway_router(self.router)

        # 8. Proactive engine
        if self._config.proactive.enabled and self._config.intelligence.enabled:
            self.proactive_engine = await self._create_proactive_engine()
            logger.info("Proactive engine started")

        # 8b. Wire reflection dependencies
        if self._config.self_improvement.reflection_enabled:
            from animus_bootstrap.intelligence.proactive.checks.reflection import (
                set_reflection_deps,
            )

            set_reflection_deps(
                identity_manager=self.identity_manager,
                feedback_store=getattr(self, "feedback_store", None),
                memory_manager=self.memory_manager,
                cognitive_backend=self.cognitive_backend,
                config=self._config,
            )
            logger.info("Reflection loop dependencies wired")

        # 9. Persistent timer store + restore saved timers
        if self._config.intelligence.enabled:
            from animus_bootstrap.intelligence.tools.builtin.timer_ctl import (
                restore_timers,
                set_timer_store,
            )
            from animus_bootstrap.intelligence.tools.builtin.timer_store import (
                TimerStore,
            )

            timers_db = data_dir / "timers.db"
            self._timer_store = TimerStore(timers_db)
            set_timer_store(self._timer_store)
            restored = restore_timers()
            if restored:
                logger.info("Restored %d timers from persistent store", restored)

        # 10. Feedback store
        from animus_bootstrap.intelligence.feedback import FeedbackStore

        feedback_db = data_dir / "feedback.db"
        self.feedback_store = FeedbackStore(feedback_db)
        logger.info("Feedback store initialized: %s", feedback_db)

        self._started = True
        logger.info("Animus runtime started successfully")

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        if not self._started:
            return

        logger.info("Animus runtime stopping...")

        if self._mcp_bridge is not None:
            await self._mcp_bridge.close()
            logger.info("MCP bridge closed")

        if self.proactive_engine is not None:
            await self.proactive_engine.stop()
            self.proactive_engine.close()
            logger.info("Proactive engine stopped")

        if self.automation_engine is not None:
            self.automation_engine.close()
            logger.info("Automation engine closed")

        if hasattr(self, "_timer_store") and self._timer_store is not None:
            self._timer_store.close()
            logger.info("Timer store closed")

        if hasattr(self, "_improvement_store") and self._improvement_store is not None:
            self._improvement_store.close()
            logger.info("Improvement store closed")

        if hasattr(self, "_tool_history_store") and self._tool_history_store is not None:
            self._tool_history_store.close()
            logger.info("Tool history store closed")

        if self.feedback_store is not None:
            self.feedback_store.close()
            logger.info("Feedback store closed")

        if self.memory_manager is not None:
            self.memory_manager.close()
            logger.info("Memory manager closed")

        if self.session_manager is not None:
            self.session_manager.close()
            logger.info("Session manager closed")

        self._started = False
        logger.info("Animus runtime stopped")

    def _create_cognitive_backend(self) -> Any:
        """Create cognitive backend based on config."""
        backend_type = self._config.gateway.default_backend

        anthropic_key = self._config.api.anthropic_key
        if backend_type == "anthropic" and anthropic_key and len(anthropic_key) > 40:
            from animus_bootstrap.gateway.cognitive import AnthropicBackend

            return AnthropicBackend(api_key=anthropic_key)

        if backend_type == "ollama":
            from animus_bootstrap.gateway.cognitive import OllamaBackend

            ollama_cfg = self._config.ollama
            return OllamaBackend(
                model=ollama_cfg.model,
                host=f"http://{ollama_cfg.host}:{ollama_cfg.port}",
            )

        if backend_type == "forge" and self._config.forge.enabled:
            from animus_bootstrap.gateway.cognitive import ForgeBackend

            return ForgeBackend(
                host=self._config.forge.host,
                port=self._config.forge.port,
                api_key=self._config.forge.api_key,
            )

        # Fallback to Ollama (always available locally)
        from animus_bootstrap.gateway.cognitive import OllamaBackend

        logger.warning(
            "Backend '%s' not configured, falling back to Ollama",
            backend_type,
        )
        ollama_cfg = self._config.ollama
        return OllamaBackend(
            model=ollama_cfg.model,
            host=f"http://{ollama_cfg.host}:{ollama_cfg.port}",
        )

    def _create_memory_manager(self) -> Any:
        """Create memory manager based on config."""
        from animus_bootstrap.intelligence.memory import MemoryManager

        backend_type = self._config.intelligence.memory_backend
        db_path = Path(self._config.intelligence.memory_db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if backend_type == "sqlite":
            from animus_bootstrap.intelligence.memory_backends.sqlite_backend import (
                SQLiteMemoryBackend,
            )

            backend = SQLiteMemoryBackend(db_path)
            return MemoryManager(backend)

        if backend_type == "chromadb":
            try:
                from animus_bootstrap.intelligence.memory_backends.chromadb_backend import (
                    ChromaDBMemoryBackend,
                )

                persist_dir = str(self._config.get_data_path() / "chromadb")
                backend = ChromaDBMemoryBackend(persist_directory=persist_dir)
                return MemoryManager(backend)
            except (RuntimeError, ImportError):
                logger.warning("ChromaDB not available, falling back to SQLite")
                from animus_bootstrap.intelligence.memory_backends.sqlite_backend import (
                    SQLiteMemoryBackend,
                )

                backend = SQLiteMemoryBackend(db_path)
                return MemoryManager(backend)

        if backend_type == "animus":
            from animus_bootstrap.intelligence.memory_backends.animus_backend import (
                AnimusMemoryBackend,
            )

            backend = AnimusMemoryBackend()
            return MemoryManager(backend)

        # Default to SQLite
        from animus_bootstrap.intelligence.memory_backends.sqlite_backend import (
            SQLiteMemoryBackend,
        )

        backend = SQLiteMemoryBackend(db_path)
        return MemoryManager(backend)

    def _create_tool_executor(self) -> Any:
        """Create tool executor with built-in tools registered."""
        from animus_bootstrap.intelligence.tools.builtin import get_all_builtin_tools
        from animus_bootstrap.intelligence.tools.executor import ToolExecutor
        from animus_bootstrap.intelligence.tools.permissions import (
            PermissionLevel,
            ToolPermissionManager,
        )

        perm_mgr = ToolPermissionManager(
            default=PermissionLevel(self._config.intelligence.tool_approval_default)
        )
        executor = ToolExecutor(
            max_calls_per_turn=self._config.intelligence.max_tool_calls_per_turn,
            timeout_seconds=float(self._config.intelligence.tool_timeout_seconds),
            permission_manager=perm_mgr,
        )

        for tool in get_all_builtin_tools():
            executor.register(tool)

        return executor

    async def _discover_mcp_tools(self) -> None:
        """Discover and import tools from configured MCP servers."""
        from animus_bootstrap.intelligence.tools.mcp_bridge import MCPToolBridge

        config_path = Path(self._config.intelligence.mcp.config_path).expanduser()
        bridge = MCPToolBridge(config_path)
        server_names = await bridge.discover_servers()

        if not server_names:
            return

        self._mcp_bridge = bridge
        total = 0
        for name in server_names:
            tools = await bridge.import_tools(name)
            for tool in tools:
                try:
                    self.tool_executor.register(tool)
                    total += 1
                except ValueError:
                    logger.warning("MCP tool '%s' conflicts with existing tool", tool.name)

        if total:
            logger.info("Imported %d tools from %d MCP servers", total, len(server_names))

    def _create_router(self) -> Any:
        """Create message router -- intelligent if components are available."""
        if (
            self.memory_manager
            or self.tool_executor
            or self.automation_engine
            or self.persona_engine
        ):
            from animus_bootstrap.intelligence.router import IntelligentRouter

            return IntelligentRouter(
                cognitive=self.cognitive_backend,
                session_manager=self.session_manager,
                memory=self.memory_manager,
                tools=self.tool_executor,
                automations=self.automation_engine,
                system_prompt=self._config.gateway.system_prompt,
                persona_engine=self.persona_engine,
                context_adapter=self.context_adapter,
                identity_manager=self.identity_manager,
            )

        from animus_bootstrap.gateway.router import MessageRouter

        return MessageRouter(
            cognitive=self.cognitive_backend,
            session_manager=self.session_manager,
        )

    def _create_persona_engine(self) -> Any:
        """Create persona engine from config."""
        from animus_bootstrap.personas.engine import PersonaEngine, PersonaProfile
        from animus_bootstrap.personas.voice import VoiceConfig

        engine = PersonaEngine()

        # Register default persona from config
        cfg = self._config.personas
        default_persona = PersonaProfile(
            name=cfg.default_name,
            system_prompt=cfg.default_system_prompt,
            voice=VoiceConfig(
                tone=cfg.default_tone,
                max_response_length=cfg.default_max_response_length,
                emoji_policy=cfg.default_emoji_policy,
            ),
            is_default=True,
        )
        engine.register_persona(default_persona)

        # Register named profiles from config
        for profile_name, profile_cfg in cfg.profiles.items():
            persona = PersonaProfile(
                name=profile_cfg.name or profile_name,
                description=profile_cfg.description,
                system_prompt=profile_cfg.system_prompt,
                voice=VoiceConfig(tone=profile_cfg.tone),
                knowledge_domains=profile_cfg.knowledge_domains,
                excluded_topics=profile_cfg.excluded_topics,
                channel_bindings=profile_cfg.channel_bindings,
            )
            engine.register_persona(persona)

        return engine

    async def _create_proactive_engine(self) -> Any:
        """Create and start proactive engine."""
        from animus_bootstrap.intelligence.proactive.checks import get_builtin_checks
        from animus_bootstrap.intelligence.proactive.engine import ProactiveEngine

        data_dir = self._config.get_data_path()
        proactive_db = data_dir / "proactive.db"

        # Build send callback that routes through the gateway
        async def send_nudge(text: str, channels: list[str]) -> None:
            if self.router:
                await self.router.broadcast(text, channels)

        engine = ProactiveEngine(
            db_path=proactive_db,
            quiet_hours=(
                self._config.proactive.quiet_hours_start,
                self._config.proactive.quiet_hours_end,
            ),
            send_callback=send_nudge,
        )

        # Register built-in checks
        for check in get_builtin_checks():
            # Override from config if present
            check_config = self._config.proactive.checks.get(check.name)
            if check_config:
                check.enabled = check_config.enabled
                if check_config.schedule:
                    check.schedule = check_config.schedule
                if check_config.channels:
                    check.channels = check_config.channels
            engine.register_check(check)

        await engine.start()

        # Wire timer tools to the live proactive engine
        from animus_bootstrap.intelligence.tools.builtin.timer_ctl import (
            set_proactive_engine,
        )

        set_proactive_engine(engine)

        return engine


# Module-level singleton
_runtime: AnimusRuntime | None = None


def get_runtime() -> AnimusRuntime:
    """Get or create the global runtime singleton."""
    global _runtime  # noqa: PLW0603
    if _runtime is None:
        _runtime = AnimusRuntime()
    return _runtime


def set_runtime(runtime: AnimusRuntime) -> None:
    """Set the global runtime singleton (for testing)."""
    global _runtime  # noqa: PLW0603
    _runtime = runtime


def reset_runtime() -> None:
    """Reset the global runtime singleton (for testing)."""
    global _runtime  # noqa: PLW0603
    _runtime = None
