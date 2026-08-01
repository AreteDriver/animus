"""Animus Kernel — Autonomous builder engine.

Core primitives for autonomous code improvement:
- agents: Multi-agent supervision and delegation
- budget: Token economy, atomic reservations, cost audit
- builder: Terminal agent and command runner
- channels: Discord bot integration
- config: Settings and offline defaults
- contracts: Validation and enforcement
- coordination: Workflow evolution and auto-promotion
- evaluation: Outcome comparison
- executions: Execution manager and models
- executor: Workflow orchestration with checkpoint/resume
- head: Session context, checkpointing, and daemon
- integrations: External system bridges
- intelligence: Cost learning and cross-workflow memory
- memory: HOT/WARM/COLD tiered memory with multiple backends
- metrics: Cost tracking, debt monitoring, audit checks
- monitoring: MCP tool usage and parallel execution tracking
- network: Egress handling
- protocols: Memory protocol definitions
- providers: Multi-provider LLM abstraction (Anthropic, OpenAI, Ollama)
- ratelimit: Token-bucket rate limiting
- resilience: Bulkheads, fallbacks, concurrency limits
- safety: PII gates and tool safety
- sandbox: Isolated build-test-lint-rollback cycles
- security: PI wrapping
- server: FastAPI application
- skills: Skill library, loader, and consensus
- state: Agent context, memory backends, checkpointing
- tools: Filesystem, registry, and proposals
- utils: Retry, circuit breaker, validation
"""

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("animus-kernel")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+dev"
