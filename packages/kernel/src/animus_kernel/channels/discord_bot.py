"""Discord slash-command bot for Animus kernel build visibility.

Lightweight, stateless bot that exposes ``/build`` commands.
Requires ``discord.py`` (soft dependency — guarded at runtime).

Usage::

    bot = DiscordBot(
        token=os.getenv("DISCORD_TOKEN"),
        message_bus=message_bus,
        budget_manager=budget_manager,
        approval_gate=approval_gate,
    )
    bot.run()
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus_kernel.agents.message_bus import AgentMessage, AgentMessageBus
    from animus_kernel.budget.manager import BudgetManager
    from animus_kernel.sandbox.approval import ApprovalGate

logger = logging.getLogger(__name__)

# Soft dependency — module is importable without discord.py installed.
try:
    import discord
    from discord import app_commands
    from discord.ext import commands

    _HAS_DISCORD = True
except ImportError:  # pragma: no cover
    discord = None  # type: ignore[assignment]
    app_commands = None  # type: ignore[assignment]
    commands = None  # type: ignore[assignment]
    _HAS_DISCORD = False


class _BuildEventCache:
    """Thread-safe cache fed by :class:`AgentMessageBus` subscriptions."""

    def __init__(self, max_size: int = 500) -> None:
        self._lock = threading.Lock()
        self._messages: deque[Any] = deque(maxlen=max_size)
        self._active_builds: dict[str, dict[str, Any]] = {}

    def on_message(self, msg: AgentMessage) -> None:
        with self._lock:
            self._messages.append(msg)
            payload = getattr(msg, "payload", {}) or {}
            topic = getattr(msg, "topic", "")

            if topic == "build.start":
                build_id = payload.get("build_id", getattr(msg, "id", "unknown"))
                self._active_builds[build_id] = {
                    "id": build_id,
                    "project": payload.get("project", "unknown"),
                    "stage": payload.get("stage", "starting"),
                    "started_at": getattr(msg, "timestamp", 0.0),
                }
            elif topic in ("build.complete", "build.failed"):
                build_id = payload.get("build_id")
                if build_id and build_id in self._active_builds:
                    self._active_builds[build_id]["status"] = (
                        "completed" if topic == "build.complete" else "failed"
                    )
                    self._active_builds[build_id]["ended_at"] = getattr(
                        msg, "timestamp", 0.0
                    )

    @property
    def active_builds(self) -> list[dict[str, Any]]:
        with self._lock:
            now = datetime.datetime.now(datetime.UTC).timestamp()
            active: list[dict[str, Any]] = []
            for b in self._active_builds.values():
                ended = b.get("ended_at")
                if ended is not None and now - ended > 300:
                    continue
                active.append(b.copy())
            return active

    @property
    def recent_messages(self) -> list[Any]:
        with self._lock:
            return list(self._messages)


class _BuildGroup(app_commands.Group):
    """``/build`` slash-command group."""

    def __init__(self, bot_ref: DiscordBot) -> None:
        super().__init__(name="build", description="Build management commands")
        self._bot_ref = bot_ref

    @app_commands.command(name="status", description="Active builds, budget remaining, ET today")
    async def status(self, interaction: discord.Interaction) -> None:
        await self._bot_ref._cmd_status(interaction)  # noqa: SLF001

    @app_commands.command(name="approve", description="Approve a build job")
    @app_commands.describe(job_id="Job ID to approve")
    async def approve(self, interaction: discord.Interaction, job_id: str) -> None:
        await self._bot_ref._cmd_approve(interaction, job_id)  # noqa: SLF001

    @app_commands.command(name="queue", description="List pending tasks for a project")
    @app_commands.describe(project="Project name")
    async def queue(self, interaction: discord.Interaction, project: str) -> None:
        await self._bot_ref._cmd_queue(interaction, project)  # noqa: SLF001


class DiscordBot:
    """Stateless Discord bot wired to the Animus kernel.

    Args:
        token: Discord bot token. Falls back to ``DISCORD_TOKEN`` env var.
        message_bus: Shared :class:`AgentMessageBus` for build events.
        budget_manager: Shared :class:`BudgetManager` for budget queries.
        approval_gate: Shared :class:`ApprovalGate` for job approvals.
    """

    def __init__(
        self,
        token: str | None = None,
        message_bus: AgentMessageBus | None = None,
        budget_manager: BudgetManager | None = None,
        approval_gate: ApprovalGate | None = None,
    ) -> None:
        if not _HAS_DISCORD:
            raise RuntimeError(
                "discord.py is required for DiscordBot. Install: pip install discord.py"
            )

        self.token = token or os.environ.get("DISCORD_TOKEN")
        if not self.token:
            raise RuntimeError("DISCORD_TOKEN env var or token argument is required")

        self._message_bus = message_bus
        self._budget = budget_manager
        self._approval = approval_gate
        self._event_cache = _BuildEventCache()

        if self._message_bus:
            self._message_bus.subscribe("build.*", self._event_cache.on_message)
            self._message_bus.subscribe("task.*", self._event_cache.on_message)

        intents = discord.Intents.default()
        intents.message_content = False

        # Auto-shard when added to >1 guild (DiscordBot specification requirement).
        self._client: commands.AutoShardedBot = commands.AutoShardedBot(
            command_prefix="!",
            help_command=None,
            intents=intents,
        )
        self._client.tree.on_error = self._on_tree_error
        self._client.tree.add_command(_BuildGroup(self))

    # ------------------------------------------------------------------ error handling

    async def _on_tree_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        """Suppress stack traces in Discord — return user-friendly text only."""
        friendly = "Something went wrong. Please try again later."
        if isinstance(error, app_commands.MissingPermissions):
            friendly = "You don't have permission to use this command."
        elif isinstance(error, app_commands.CommandInvokeError):
            cause = getattr(error, "original", None) or error.__cause__
            logger.exception("Command invoke error: %s", cause)
        else:
            logger.exception("Discord command error: %s", error)

        if interaction.response.is_done():
            await interaction.followup.send(friendly, ephemeral=True)
        else:
            await interaction.response.send_message(friendly, ephemeral=True)

    # ------------------------------------------------------------------ commands

    async def _cmd_status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)

        embed = discord.Embed(title="Build Status", color=discord.Color.blue())

        # Active builds
        active = self._event_cache.active_builds
        if active:
            lines = []
            for b in active[:10]:
                lines.append(
                    f"• `{b['id']}` — {b['project']} ({b['stage']})"
                )
            embed.add_field(
                name="Active Builds",
                value="\n".join(lines),
                inline=False,
            )
        else:
            embed.add_field(
                name="Active Builds",
                value="No active builds.",
                inline=False,
            )

        # Budget & ET
        if self._budget:
            remaining = self._budget.remaining
            total = self._budget.total_budget
            pct = self._budget.usage_percent
            status = self._budget.status.value
            embed.add_field(
                name="Budget",
                value=f"{remaining:,} / {total:,} tokens ({pct:.1f}% used) — `{status}`",
                inline=False,
            )

            # ET today — sum effective tokens for usage records created today.
            today = datetime.datetime.now(datetime.UTC).date()
            today_et = 0.0
            for rec in self._budget.get_usage_history(limit=500):
                ts = rec.timestamp
                if isinstance(ts, datetime.datetime) and ts.date() == today:
                    from animus_kernel.budget.manager import effective_tokens

                    today_et += effective_tokens(
                        rec, self._budget.config.model_multipliers
                    )
            embed.add_field(
                name="ET Today",
                value=f"{today_et:,.1f}",
                inline=False,
            )
        else:
            embed.add_field(
                name="Budget",
                value="Budget manager not available.",
                inline=False,
            )
            embed.add_field(
                name="ET Today",
                value="N/A",
                inline=False,
            )

        await interaction.followup.send(embed=embed)

    async def _cmd_approve(
        self, interaction: discord.Interaction, job_id: str
    ) -> None:
        await interaction.response.defer(thinking=True)

        if not self._approval:
            embed = discord.Embed(
                title="Approval",
                description="Approval gate is not configured.",
                color=discord.Color.red(),
            )
            await interaction.followup.send(embed=embed)
            return

        result = self._approval.approve(
            request_id=job_id,
            approved_by=str(interaction.user),
        )
        if result:
            embed = discord.Embed(
                title="Approved",
                description=f"Job `{job_id}` approved by {interaction.user.mention}.",
                color=discord.Color.green(),
            )
        else:
            embed = discord.Embed(
                title="Not Found",
                description=f"No pending job with ID `{job_id}`.",
                color=discord.Color.orange(),
            )
        await interaction.followup.send(embed=embed)

    async def _cmd_queue(
        self, interaction: discord.Interaction, project: str
    ) -> None:
        await interaction.response.defer(thinking=True)

        pending_tasks: list[dict[str, Any]] = []

        if self._message_bus:
            for topic in self._message_bus.get_topics():
                if topic.startswith("task.") or topic.startswith("build.queue"):
                    for msg in self._message_bus.get_messages(topic, limit=50):
                        payload = getattr(msg, "payload", {}) or {}
                        msg_project = payload.get("project", "")
                        if (
                            not project
                            or msg_project == project
                            or project.lower() in msg_project.lower()
                        ):
                            pending_tasks.append(
                                {
                                    "id": payload.get(
                                        "task_id", getattr(msg, "id", "unknown")
                                    ),
                                    "topic": topic,
                                    "priority": str(
                                        getattr(msg, "priority", "normal")
                                    ).upper(),
                                    "project": msg_project,
                                    "et": payload.get(
                                        "estimated_time",
                                        payload.get("et", "—"),
                                    ),
                                }
                            )

        # Deduplicate
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for t in pending_tasks:
            tid = t["id"]
            if tid not in seen:
                seen.add(tid)
                unique.append(t)

        embed = discord.Embed(
            title=f"Queue: {project}",
            color=discord.Color.purple(),
        )
        if unique:
            lines = [
                f"• `{t['id']}` — P:{t['priority']} — ET {t['et']}"
                for t in unique[:10]
            ]
            embed.description = "\n".join(lines)
        else:
            embed.description = "No pending tasks found."

        await interaction.followup.send(embed=embed)

    # ------------------------------------------------------------------ lifecycle

    def run(self) -> None:
        """Start the bot synchronously (blocks)."""
        self._client.run(self.token)

    async def start(self) -> None:
        """Start the bot asynchronously."""
        await self._client.start(self.token)

    async def close(self) -> None:
        """Shut down and unsubscribe from message bus."""
        if self._message_bus:
            self._message_bus.unsubscribe("build.*", self._event_cache.on_message)
            self._message_bus.unsubscribe("task.*", self._event_cache.on_message)
        await self._client.close()
