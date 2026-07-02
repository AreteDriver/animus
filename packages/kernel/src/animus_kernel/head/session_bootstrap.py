"""Session bootstrap for Animus Head.

Automatically loads context on REPL startup:
- Active tasks from TODO.md
- Recent semantic memories relevant to the project
- Recent decisions from decisions/
- Previous session checkpoint (if within 24h)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from animus_kernel.head.checkpoint import HeadCheckpointStore
from animus_kernel.memory.stores.local import LocalMemoryStore

logger = logging.getLogger(__name__)


class SessionBootstrap:
    """Bootstraps a Head session with relevant context."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        memory_dir: str | Path | None = None,
        checkpoint_store: HeadCheckpointStore | None = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.checkpoint_store = checkpoint_store or HeadCheckpointStore()

        if memory_dir is None:
            memory_dir = Path.home() / ".animus" / "memory"
        self._memory = LocalMemoryStore(data_dir=Path(memory_dir))

    def bootstrap(self) -> dict[str, Any]:
        """Gather all bootstrap context.

        Returns:
            Dict with keys: active_tasks, recent_memories, recent_decisions,
            previous_session, project_name
        """
        context: dict[str, Any] = {
            "project_name": self.project_root.name,
            "project_root": str(self.project_root),
        }

        context["active_tasks"] = self._load_tasks()
        context["recent_memories"] = self._load_memories()
        context["recent_decisions"] = self._load_decisions()
        context["previous_session"] = self._load_previous_session()

        return context

    def build_system_prompt(self, context: dict[str, Any]) -> str:
        """Build a system prompt from bootstrap context."""
        lines = [
            "You are Animus Head — a local-first agentic assistant.",
            "You run entirely on local hardware using Ollama models.",
            "You have access to filesystem tools, shell execution, and memory.",
            "Be concise. Prefer tool calls over long explanations.",
            "",
            f"Current project: {context.get('project_name', 'unknown')}",
            f"Project root: {context.get('project_root', '.')}",
        ]

        # Active tasks
        tasks = context.get("active_tasks", [])
        if tasks:
            lines.append("")
            lines.append("ACTIVE TASKS:")
            for task in tasks[:10]:
                lines.append(f"  - {task}")

        # Recent decisions
        decisions = context.get("recent_decisions", [])
        if decisions:
            lines.append("")
            lines.append("RECENT DECISIONS:")
            for decision in decisions[:5]:
                lines.append(f"  - {decision}")

        # Recent memories
        memories = context.get("recent_memories", [])
        if memories:
            lines.append("")
            lines.append("RELEVANT PRIOR KNOWLEDGE:")
            for mem in memories[:5]:
                lines.append(f"  - {mem}")

        # Previous session
        prev = context.get("previous_session")
        if prev and isinstance(prev, dict):
            lines.append("")
            lines.append("PREVIOUS SESSION:")
            if prev.get("summary"):
                lines.append(f"  Summary: {prev['summary']}")
            lines.append(f"  Turns: {prev.get('turns', 0)}")

        lines.extend(
            [
                "",
                "INSTRUCTIONS:",
                "1. Use tools to read files, search code, run tests, or execute commands.",
                "2. Use 'remember' to save important discoveries.",
                "3. Use 'recall' to retrieve prior knowledge.",
                "4. Use 'create_task' to track new work items.",
                "5. When writing files, prefer small, focused edits.",
            ]
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context loaders
    # ------------------------------------------------------------------

    def _load_tasks(self) -> list[str]:
        """Load active tasks from TODO.md."""
        todo_path = self.project_root / "TODO.md"
        if not todo_path.exists():
            todo_path = self.project_root.parent / "TODO.md"
        if not todo_path.exists():
            return []

        try:
            content = todo_path.read_text()
            lines = content.splitlines()
            pending = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("- [ ]"):
                    pending.append(stripped.replace("- [ ]", "").strip())
            return pending
        except Exception:
            logger.debug("Failed to read TODO.md", exc_info=True)
            return []

    def _load_memories(self) -> list[str]:
        """Load recent memories relevant to project name."""
        try:
            query = self.project_root.name
            memories = self._memory.search(query, limit=5)
            return [m.content[:200] for m in memories]
        except Exception:
            logger.debug("Failed to load memories", exc_info=True)
            return []

    def _load_decisions(self) -> list[str]:
        """Load recent decisions from decisions/YYYY-MM.md."""
        decisions_dir = self.project_root / "decisions"
        if not decisions_dir.exists():
            # Try parent (for monorepo setups where project is a subdir)
            decisions_dir = self.project_root.parent / "decisions"
        if not decisions_dir.exists():
            return []

        try:
            # Find the most recent decision file
            files = sorted(decisions_dir.glob("*.md"), reverse=True)
            if not files:
                return []

            latest = files[0]
            content = latest.read_text()
            lines = content.splitlines()

            # Extract ADL entries (lines starting with "## ADL-" or similar patterns)
            decisions = []
            for line in lines:
                if line.strip().startswith("## ADL-") or line.strip().startswith("- "):
                    decisions.append(line.strip())
            return decisions[:10]
        except Exception:
            logger.debug("Failed to load decisions", exc_info=True)
            return []

    def _load_previous_session(self) -> dict[str, Any] | None:
        """Load previous session checkpoint if within 24 hours."""
        try:
            recent = self.checkpoint_store.list_recent(limit=1)
            if not recent:
                return None

            checkpoint = recent[0]
            cutoff = datetime.now(UTC) - timedelta(hours=24)
            if checkpoint.last_active_at.replace(tzinfo=UTC) < cutoff:
                return None

            return {
                "session_id": checkpoint.session_id,
                "summary": checkpoint.summary,
                "turns": checkpoint.turns,
                "total_tokens": checkpoint.total_tokens,
                "messages": checkpoint.messages,
            }
        except Exception:
            logger.debug("Failed to load previous session", exc_info=True)
            return None
