"""MCP server for Animus — exposes memory, tasks, and tools to Claude Code.

Run: python -m animus.mcp_server
Or add to Claude Code MCP config.

Auth: Set ANIMUS_MCP_API_KEY env var to require authentication.
"""

from __future__ import annotations

import asyncio
import json
import os

from animus.config import AnimusConfig
from animus.logging import get_logger
from animus.memory import MemoryLayer, MemoryType
from animus.tasks import TaskTracker

logger = get_logger("mcp_server")

# Optional API key for MCP server authentication
_MCP_API_KEY = os.environ.get("ANIMUS_MCP_API_KEY")


def _check_auth(api_key: str = "") -> str | None:
    """Validate API key if one is configured. Returns error message or None."""
    if not _MCP_API_KEY:
        return None  # No auth configured
    if api_key == _MCP_API_KEY:
        return None
    return "Authentication required. Pass api_key parameter matching ANIMUS_MCP_API_KEY."


def create_mcp_server():
    """Create and configure the Animus MCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "MCP server requires the mcp SDK. Install with: pip install 'mcp>=1.0.0'"
        ) from exc

    config = AnimusConfig.load()
    config.ensure_dirs()
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)
    tasks = TaskTracker(config.data_dir)

    mcp = FastMCP("animus", instructions="Animus exocortex — persistent memory, tasks, and tools.")

    # -----------------------------------------------------------------------
    # Memory tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def animus_remember(
        content: str, tags: str = "", memory_type: str = "semantic", api_key: str = ""
    ) -> str:
        """Store a memory in Animus.

        Args:
            content: Text to remember (fact, decision, observation, pattern).
            tags: Comma-separated tags for categorization.
            memory_type: One of: semantic (facts), episodic (events), procedural (how-tos).
            api_key: API key (required if ANIMUS_MCP_API_KEY is set).
        """
        auth_err = _check_auth(api_key)
        if auth_err:
            return auth_err
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        try:
            mt = MemoryType(memory_type)
        except ValueError:
            mt = MemoryType.SEMANTIC

        mem = memory.remember(content=content, memory_type=mt, tags=tag_list, source="mcp")
        return f"Stored memory {mem.id[:8]} ({mt.value}, {len(tag_list)} tags)"

    @mcp.tool()
    def animus_recall(query: str, limit: int = 5) -> str:
        """Search Animus memory by semantic similarity.

        Args:
            query: What to search for.
            limit: Maximum results to return (default 5).
        """
        results = memory.recall(query=query, limit=limit)
        if not results:
            return "No matching memories found."

        lines = []
        for m in results:
            tags = f" [{', '.join(m.tags)}]" if m.tags else ""
            lines.append(f"- [{m.id[:8]}] {m.content[:200]}{tags}")
        return "\n".join(lines)

    @mcp.tool()
    def animus_search_tags(tags: str, limit: int = 10) -> str:
        """Find memories by tags.

        Args:
            tags: Comma-separated tags to filter by (all must match).
            limit: Maximum results.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        if not tag_list:
            return "No tags provided."

        results = memory.recall_by_tags(tags=tag_list, limit=limit)
        if not results:
            return f"No memories found with tags: {', '.join(tag_list)}"

        lines = []
        for m in results:
            lines.append(f"- [{m.id[:8]}] {m.content[:200]}")
        return "\n".join(lines)

    @mcp.tool()
    def animus_memory_stats() -> str:
        """Get Animus memory statistics."""
        stats = memory.get_statistics()
        return json.dumps(stats, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Task tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def animus_list_tasks(status: str = "pending") -> str:
        """List tasks in Animus task tracker.

        Args:
            status: Filter by status: pending, in_progress, completed, all.
        """
        all_tasks = tasks.list()
        if status != "all":
            all_tasks = [t for t in all_tasks if t.status.value == status]

        if not all_tasks:
            return f"No {status} tasks."

        lines = []
        for t in all_tasks:
            lines.append(f"- [{t.id[:8]}] [{t.status}] {t.description}")
        return "\n".join(lines)

    @mcp.tool()
    def animus_create_task(description: str, priority: int = 5, api_key: str = "") -> str:
        """Create a new task.

        Args:
            description: What needs to be done.
            priority: Priority 1-10 (1=highest, 10=lowest). Default 5.
            api_key: API key (required if ANIMUS_MCP_API_KEY is set).
        """
        auth_err = _check_auth(api_key)
        if auth_err:
            return auth_err
        task = tasks.add(description=description, priority=priority)
        return f"Created task {task.id[:8]}: {description}"

    @mcp.tool()
    def animus_complete_task(task_id: str, api_key: str = "") -> str:
        """Mark a task as completed.

        Args:
            task_id: Task ID or partial ID prefix.
            api_key: API key (required if ANIMUS_MCP_API_KEY is set).
        """
        auth_err = _check_auth(api_key)
        if auth_err:
            return auth_err
        success = tasks.complete(task_id)
        if success:
            return f"Task {task_id} marked complete."
        return f"Task {task_id} not found."

    # -----------------------------------------------------------------------
    # Brief / context tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def animus_brief(topic: str = "") -> str:
        """Generate a situation briefing from Animus memory.

        Args:
            topic: Optional topic to focus the briefing on.
        """
        query = topic or "recent important context"
        recent = memory.recall(query=query, limit=10)

        if not recent:
            return "No relevant context in memory."

        lines = ["## Animus Briefing", ""]
        for m in recent:
            prefix = f"[{m.memory_type.value}]" if hasattr(m, "memory_type") else ""
            lines.append(f"- {prefix} {m.content[:300]}")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Workflow tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def animus_run_workflow(
        workflow_path: str, task_description: str = "", api_key: str = ""
    ) -> str:
        """Run a Forge workflow pipeline.

        Args:
            workflow_path: Path to workflow YAML file (e.g., configs/examples/build_task.yaml).
            task_description: Optional task description to inject into the first agent's prompt.
            api_key: API key (required if ANIMUS_MCP_API_KEY is set).
        """
        auth_err = _check_auth(api_key)
        if auth_err:
            return auth_err
        from pathlib import Path

        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.forge import ForgeEngine
        from animus.forge.loader import load_workflow
        from animus.forge.models import ForgeError
        from animus.tools import create_default_registry

        wf_path = Path(workflow_path)
        if not wf_path.exists():
            return f"Workflow not found: {workflow_path}"

        try:
            wf_config = load_workflow(wf_path)
        except ForgeError as e:
            return f"Failed to load workflow: {e}"

        if task_description and wf_config.agents:
            existing = wf_config.agents[0].system_prompt or ""
            wf_config.agents[0].system_prompt = f"{existing}\n\n## Task\n{task_description}"

        # Use default model config
        model_config = ModelConfig.ollama()
        cognitive = CognitiveLayer(model_config)
        tools = create_default_registry()

        cp_dir = config.data_dir / "checkpoints"
        cp_dir.mkdir(exist_ok=True)
        engine = ForgeEngine(cognitive=cognitive, checkpoint_dir=cp_dir, tools=tools)

        try:
            state = engine.run(wf_config)
            lines = [f"Workflow '{wf_config.name}' {state.status}"]
            for result in state.results:
                status = "OK" if result.success else "FAIL"
                lines.append(f"  [{status}] {result.agent_name} ({result.tokens_used} tokens)")
                if result.error:
                    lines.append(f"        {result.error}")
            lines.append(f"Total: {state.total_tokens} tokens, ${state.total_cost:.4f}")
            return "\n".join(lines)
        except Exception as e:
            return f"Workflow failed: {e}"

    # -----------------------------------------------------------------------
    # Harvest tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def animus_harvest(
        target: str,
        compare: bool = True,
        depth: str = "quick",
        api_key: str = "",
    ) -> str:
        """Scan an external GitHub repo and extract learnable patterns.

        Clones the repo, runs anchormd analysis, extracts architecture,
        dependencies, testing patterns, and CI setup. Optionally compares
        against our projects and stores findings in memory.

        Args:
            target: GitHub repo URL or username/repo (e.g., 'fastapi/fastapi').
            compare: Compare against our projects (default True).
            depth: Scan depth: 'quick' (shallow clone) or 'deep' (full clone).
            api_key: API key (required if ANIMUS_MCP_API_KEY is set).
        """
        auth_err = _check_auth(api_key)
        if auth_err:
            return auth_err

        from animus.harvest import harvest_repo

        try:
            result = harvest_repo(
                target=target,
                compare=compare,
                depth=depth,
                memory_layer=memory,
            )
            return json.dumps(result.to_dict(), indent=2)
        except (ValueError, RuntimeError) as e:
            return f"Harvest failed: {e}"
        except Exception as e:
            return f"Harvest error: {e}"

    # -----------------------------------------------------------------------
    # Harvest watchlist tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def animus_watchlist_add(
        target: str,
        tags: str = "",
        notes: str = "",
        api_key: str = "",
    ) -> str:
        """Add a GitHub repo to the competition watchlist for periodic scanning.

        Args:
            target: GitHub repo URL or username/repo (e.g., 'fastapi/fastapi').
            tags: Comma-separated tags (e.g., 'competitor,eve-frontier').
            notes: Notes about why this repo matters.
            api_key: API key (required if ANIMUS_MCP_API_KEY is set).
        """
        auth_err = _check_auth(api_key)
        if auth_err:
            return auth_err

        from animus.harvest_watchlist import add_to_watchlist

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

        try:
            entry = add_to_watchlist(target=target, tags=tag_list, notes=notes or None)
            return json.dumps(entry, indent=2)
        except ValueError as e:
            return f"Watchlist add failed: {e}"
        except Exception as e:
            return f"Watchlist error: {e}"

    @mcp.tool()
    def animus_watchlist_remove(target: str, api_key: str = "") -> str:
        """Remove a GitHub repo from the competition watchlist.

        Args:
            target: GitHub repo URL or username/repo to remove.
            api_key: API key (required if ANIMUS_MCP_API_KEY is set).
        """
        auth_err = _check_auth(api_key)
        if auth_err:
            return auth_err

        from animus.harvest_watchlist import remove_from_watchlist

        removed = remove_from_watchlist(target)
        if removed:
            return f"Removed '{target}' from watchlist."
        return f"'{target}' not found on watchlist."

    @mcp.tool()
    def animus_watchlist_list() -> str:
        """List all repos on the competition watchlist with their last scan data."""
        from animus.harvest_watchlist import get_watchlist

        repos = get_watchlist()
        if not repos:
            return "Watchlist is empty."
        return json.dumps(repos, indent=2)

    @mcp.tool()
    def animus_watchlist_scan(
        interval_hours: int = 0,
        api_key: str = "",
    ) -> str:
        """Run harvest scans on all due repos and return a changes report.

        Args:
            interval_hours: Override scan interval in hours (0 = use default 168h/7 days).
            api_key: API key (required if ANIMUS_MCP_API_KEY is set).
        """
        auth_err = _check_auth(api_key)
        if auth_err:
            return auth_err

        from animus.harvest_watchlist import run_watchlist_scan

        interval = interval_hours if interval_hours > 0 else None
        try:
            report = asyncio.run(run_watchlist_scan(memory=memory, interval_hours=interval))
            return json.dumps(report, indent=2)
        except Exception as e:
            return f"Watchlist scan failed: {e}"

    # -----------------------------------------------------------------------
    # Self-improvement tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def animus_self_improve(
        codebase_path: str,
        provider: str = "ollama",
        focus: str = "",
        auto_approve: bool = True,
        api_key: str = "",
    ) -> str:
        """Run the Forge self-improvement pipeline on a codebase.

        Analyzes code, generates an improvement plan, tests changes in a sandbox,
        and creates a PR if everything passes.

        Args:
            codebase_path: Path to the codebase to improve.
            provider: AI provider — 'ollama', 'anthropic', or 'openai'.
            focus: Optional focus category (e.g., 'testing', 'security', 'performance').
            auto_approve: Auto-approve all stages (default True for MCP use).
            api_key: API key (required if ANIMUS_MCP_API_KEY is set).
        """
        auth_err = _check_auth(api_key)
        if auth_err:
            return auth_err

        from pathlib import Path

        cpath = Path(codebase_path)
        if not cpath.exists():
            return f"Path not found: {codebase_path}"

        try:
            from animus_forge.agents.provider_wrapper import create_agent_provider
            from animus_forge.self_improve.orchestrator import SelfImproveOrchestrator
        except ImportError:
            return (
                "Forge not installed. Install with: pip install animus-forge\n"
                "Or run from the monorepo: pip install -e packages/forge/"
            )

        try:
            agent_provider = create_agent_provider(provider)
        except Exception as e:
            return f"Failed to create {provider} provider: {e}"

        orchestrator = SelfImproveOrchestrator(
            codebase_path=cpath,
            provider=agent_provider,
        )

        try:
            result = asyncio.run(
                orchestrator.run(
                    focus_category=focus or None,
                    auto_approve=auto_approve,
                )
            )
        except Exception as e:
            return f"Self-improve failed: {e}"

        lines = [f"Stage: {result.stage_reached.value}"]
        lines.append(f"Success: {result.success}")
        if result.plan:
            lines.append(f"Plan: {result.plan.title}")
            lines.append(f"Suggestions: {len(result.plan.suggestions)}")
            for s in result.plan.suggestions[:5]:
                lines.append(f"  - {s.description[:80]}")
        if result.error:
            lines.append(f"Error: {result.error}")
        if result.sandbox_result:
            passed = "passed" if result.sandbox_result.tests_passed else "failed"
            lines.append(f"Tests: {passed}")
        if result.pull_request:
            lines.append(f"PR: {result.pull_request.url or result.pull_request.branch}")
        return "\n".join(lines)

    return mcp


def main():
    """Run the MCP server via stdio."""
    mcp = create_mcp_server()
    mcp.run()


if __name__ == "__main__":
    main()
