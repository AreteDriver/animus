#!/usr/bin/env python3
"""
Animus Discord Bot — operational intel bot with slash commands.

Pushes harvest intel, searches Animus memory, manages watchlist,
and provides daily briefs via Discord slash commands.

Standalone bot — does not modify the Forge Discord bot.

Usage:
    python tools/animus_discord_bot.py

Environment:
    ANIMUS_DISCORD_TOKEN  — Discord bot token (required)
    ANIMUS_DISCORD_CHANNEL — Channel ID for auto-push intel (required for auto-push)
    DISCORD_BOT_TOKEN     — Fallback token if ANIMUS_DISCORD_TOKEN not set
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure animus core is importable when run standalone
_CORE_DIR = os.path.join(os.path.dirname(__file__), "..", "packages", "core")
if os.path.isdir(_CORE_DIR) and _CORE_DIR not in sys.path:
    sys.path.insert(0, os.path.realpath(_CORE_DIR))

import discord
from discord import app_commands
from discord.ext import tasks

from animus.config import AnimusConfig
from animus.memory import MemoryLayer, MemoryType

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("animus_discord")

# ---------------------------------------------------------------------------
# Embed colors
# ---------------------------------------------------------------------------

COLOR_BLUE = 0x3498DB
COLOR_GREEN = 0x2ECC71
COLOR_YELLOW = 0xF1C40F
COLOR_RED = 0xE74C3C
COLOR_PURPLE = 0x9B59B6

# ---------------------------------------------------------------------------
# Discord field value max length
# ---------------------------------------------------------------------------

FIELD_MAX = 1024
EMBED_DESC_MAX = 4096


def _trunc(text: str, limit: int = FIELD_MAX) -> str:
    """Truncate text to fit Discord limits."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Memory / config helpers (initialized lazily)
# ---------------------------------------------------------------------------

_memory: MemoryLayer | None = None
_config: AnimusConfig | None = None


def _get_config() -> AnimusConfig:
    global _config
    if _config is None:
        _config = AnimusConfig.load()
        _config.ensure_dirs()
    return _config


def _get_memory() -> MemoryLayer:
    global _memory
    if _memory is None:
        cfg = _get_config()
        _memory = MemoryLayer(cfg.data_dir, backend=cfg.memory.backend)
    return _memory


# ---------------------------------------------------------------------------
# Harvest state file — used for auto-push change detection
# ---------------------------------------------------------------------------

HARVEST_STATE_FILE = Path("~/.animus/last_harvest_report.json").expanduser()


def _load_last_harvest() -> dict[str, Any] | None:
    """Load the last harvest report from disk."""
    if not HARVEST_STATE_FILE.exists():
        return None
    try:
        return json.loads(HARVEST_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_harvest_state(report: dict[str, Any]) -> None:
    """Save the latest harvest report to disk for change detection."""
    HARVEST_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    report["_checked_at"] = datetime.now(timezone.utc).isoformat()
    HARVEST_STATE_FILE.write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Bot class
# ---------------------------------------------------------------------------


class AnimusBot(discord.Client):
    """Animus operational Discord bot with slash commands."""

    def __init__(self, intel_channel_id: int | None = None) -> None:
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.intel_channel_id = intel_channel_id
        self._register_commands()

    async def setup_hook(self) -> None:
        """Sync slash commands on startup."""
        await self.tree.sync()
        logger.info("Slash commands synced")

        # Start the background harvest check loop
        if self.intel_channel_id:
            self.harvest_check_loop.start()

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (id=%s)", self.user, self.user.id)
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="the fleet",
            )
        )

    # -------------------------------------------------------------------
    # Background task: harvest change detection
    # -------------------------------------------------------------------

    @tasks.loop(minutes=30)
    async def harvest_check_loop(self) -> None:
        """Periodically check if harvest_cron produced new results."""
        if not self.intel_channel_id:
            return

        try:
            report = _load_last_harvest()
            if report is None:
                return

            # Only push if report is fresh (checked_at not set = never pushed)
            checked = report.get("_checked_at")
            if checked:
                return  # Already pushed this report

            channel = self.get_channel(self.intel_channel_id)
            if channel is None:
                channel = await self.fetch_channel(self.intel_channel_id)

            embed = _build_harvest_report_embed(report)
            await channel.send(embed=embed)

            # Mark as pushed
            _save_harvest_state(report)
            logger.info("Auto-pushed harvest report to channel %s", self.intel_channel_id)

        except Exception:
            logger.exception("Error in harvest check loop")

    @harvest_check_loop.before_loop
    async def _before_harvest_check(self) -> None:
        await self.wait_until_ready()

    # -------------------------------------------------------------------
    # Slash command registration
    # -------------------------------------------------------------------

    def _register_commands(self) -> None:
        tree = self.tree

        # /harvest <repo>
        @tree.command(name="harvest", description="Scan a GitHub repo and extract patterns")
        @app_commands.describe(repo="GitHub repo (user/repo or full URL)")
        async def harvest_cmd(interaction: discord.Interaction, repo: str) -> None:
            await interaction.response.defer(thinking=True)
            try:
                from animus.harvest import harvest_repo

                result = harvest_repo(
                    target=repo,
                    compare=True,
                    depth="quick",
                    memory_layer=_get_memory(),
                )
                embed = _build_harvest_result_embed(result)
                await interaction.followup.send(embed=embed)
            except Exception as e:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="Harvest Failed",
                        description=_trunc(str(e), EMBED_DESC_MAX),
                        color=COLOR_RED,
                    )
                )

        # /watchlist
        @tree.command(name="watchlist", description="Show the current harvest watchlist")
        async def watchlist_cmd(interaction: discord.Interaction) -> None:
            from animus.harvest_watchlist import get_watchlist

            repos = get_watchlist()
            if not repos:
                await interaction.response.send_message("Watchlist is empty.")
                return

            embed = discord.Embed(
                title="Harvest Watchlist",
                description=f"{len(repos)} repos tracked",
                color=COLOR_BLUE,
            )
            for entry in repos[:25]:  # Discord max 25 fields
                target = entry.get("target", "?")
                score = entry.get("last_score")
                last = entry.get("last_scanned") or "never"
                tags = ", ".join(entry.get("tags", [])) or "none"
                value = f"Score: {score or '?'} | Last: {last}\nTags: {tags}"
                if entry.get("notes"):
                    value += f"\n{entry['notes'][:100]}"
                embed.add_field(name=target, value=_trunc(value), inline=False)

            embed.set_footer(text="Animus Harvest")
            await interaction.response.send_message(embed=embed)

        # /watchlist-add <repo> [tags] [notes]
        @tree.command(name="watchlist-add", description="Add a repo to the harvest watchlist")
        @app_commands.describe(
            repo="GitHub repo (user/repo or full URL)",
            tags="Comma-separated tags",
            notes="Why this repo matters",
        )
        async def watchlist_add_cmd(
            interaction: discord.Interaction,
            repo: str,
            tags: str = "",
            notes: str = "",
        ) -> None:
            from animus.harvest_watchlist import add_to_watchlist

            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
            try:
                entry = add_to_watchlist(target=repo, tags=tag_list, notes=notes or None)
                embed = discord.Embed(
                    title="Added to Watchlist",
                    description=f"**{entry['target']}** is now tracked.",
                    color=COLOR_GREEN,
                )
                if tag_list:
                    embed.add_field(name="Tags", value=", ".join(tag_list), inline=True)
                if notes:
                    embed.add_field(name="Notes", value=_trunc(notes), inline=False)
                embed.set_footer(text="Animus Harvest")
                await interaction.response.send_message(embed=embed)
            except ValueError as e:
                await interaction.response.send_message(f"Failed: {e}")

        # /watchlist-scan
        @tree.command(
            name="watchlist-scan",
            description="Run harvest scan on all due watchlist repos",
        )
        async def watchlist_scan_cmd(interaction: discord.Interaction) -> None:
            await interaction.response.defer(thinking=True)
            try:
                from animus.harvest_watchlist import run_watchlist_scan

                report = await run_watchlist_scan(memory=_get_memory())
                embed = _build_harvest_report_embed(report)
                await interaction.followup.send(embed=embed)

                # Save state for auto-push tracking
                _save_harvest_state(report)
            except Exception as e:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="Watchlist Scan Failed",
                        description=_trunc(str(e), EMBED_DESC_MAX),
                        color=COLOR_RED,
                    )
                )

        # /recall <query>
        @tree.command(name="recall", description="Search Animus memory")
        @app_commands.describe(query="What to search for", limit="Max results (default 5)")
        async def recall_cmd(
            interaction: discord.Interaction, query: str, limit: int = 5
        ) -> None:
            memory = _get_memory()
            results = memory.recall(query=query, limit=limit)

            if not results:
                await interaction.response.send_message("No matching memories found.")
                return

            embed = discord.Embed(
                title=f"Recall: {query}",
                description=f"{len(results)} memories found",
                color=COLOR_PURPLE,
            )
            for m in results:
                tags = f" [{', '.join(m.tags)}]" if m.tags else ""
                mem_type = m.memory_type.value if hasattr(m, "memory_type") else "?"
                name = f"[{m.id[:8]}] ({mem_type}){tags}"
                embed.add_field(name=name, value=_trunc(m.content[:500]), inline=False)

            embed.set_footer(text="Animus Memory")
            await interaction.response.send_message(embed=embed)

        # /remember <content>
        @tree.command(name="remember", description="Store something in Animus memory")
        @app_commands.describe(
            content="Text to remember",
            tags="Comma-separated tags",
            memory_type="semantic, episodic, or procedural",
        )
        async def remember_cmd(
            interaction: discord.Interaction,
            content: str,
            tags: str = "",
            memory_type: str = "semantic",
        ) -> None:
            memory = _get_memory()
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            try:
                mt = MemoryType(memory_type)
            except ValueError:
                mt = MemoryType.SEMANTIC

            mem = memory.remember(
                content=content,
                memory_type=mt,
                tags=tag_list,
                source="discord",
            )
            embed = discord.Embed(
                title="Stored",
                description=f"Memory `{mem.id[:8]}` ({mt.value}, {len(tag_list)} tags)",
                color=COLOR_GREEN,
            )
            embed.set_footer(text="Animus Memory")
            await interaction.response.send_message(embed=embed)

        # /ask <question>
        @tree.command(
            name="ask",
            description="Ask Animus a question (searches memory for relevant context)",
        )
        @app_commands.describe(question="Your question")
        async def ask_cmd(interaction: discord.Interaction, question: str) -> None:
            memory = _get_memory()
            results = memory.recall(query=question, limit=8)

            if not results:
                await interaction.response.send_message(
                    "No relevant context found in memory for that question."
                )
                return

            # Format a useful response from memory hits
            embed = discord.Embed(
                title=f"Animus: {_trunc(question, 200)}",
                color=COLOR_BLUE,
            )

            context_lines = []
            for i, m in enumerate(results, 1):
                tags = f" [{', '.join(m.tags[:3])}]" if m.tags else ""
                mem_type = m.memory_type.value if hasattr(m, "memory_type") else "?"
                context_lines.append(f"**{i}.** ({mem_type}){tags}\n{m.content[:300]}")

            body = "\n\n".join(context_lines)
            embed.description = _trunc(body, EMBED_DESC_MAX)
            embed.set_footer(text=f"Based on {len(results)} memories | Animus")
            await interaction.response.send_message(embed=embed)

        # /brief
        @tree.command(name="brief", description="Get a daily brief from Animus")
        async def brief_cmd(interaction: discord.Interaction) -> None:
            await interaction.response.defer(thinking=True)
            embed = await _build_brief_embed()
            await interaction.followup.send(embed=embed)


# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------


def _build_harvest_result_embed(result: Any) -> discord.Embed:
    """Build a Discord embed from a HarvestResult."""
    embed = discord.Embed(
        title=f"Harvest: {result.repo}",
        description=f"Score: **{result.score}/100** | Arch: {result.architecture}",
        color=COLOR_BLUE,
    )

    if result.notable_patterns:
        embed.add_field(
            name="Notable Patterns",
            value=_trunc("\n".join(f"- {p}" for p in result.notable_patterns[:8])),
            inline=False,
        )

    if result.tools_worth_adopting:
        embed.add_field(
            name="Tools Worth Adopting",
            value=_trunc(", ".join(result.tools_worth_adopting[:10])),
            inline=False,
        )

    if result.testing_approach:
        embed.add_field(name="Testing", value=_trunc(result.testing_approach), inline=True)

    if result.comparison:
        for key, items in result.comparison.items():
            if items:
                label = key.replace("_", " ").title()
                embed.add_field(
                    name=label,
                    value=_trunc("\n".join(f"- {i}" for i in items[:5])),
                    inline=False,
                )

    embed.set_footer(text="Animus Harvest")
    return embed


def _build_harvest_report_embed(report: dict[str, Any]) -> discord.Embed:
    """Build a Discord embed from a watchlist scan report."""
    scanned = report.get("scanned", 0)
    changes = report.get("changes", [])
    no_changes = report.get("no_changes", [])
    errors = report.get("errors", [])

    if errors:
        color = COLOR_RED
    elif changes:
        color = COLOR_YELLOW
    else:
        color = COLOR_GREEN

    embed = discord.Embed(
        title="Harvest Watchlist Report",
        description=(
            f"Scanned: **{scanned}** | "
            f"Changes: **{len(changes)}** | "
            f"Unchanged: **{len(no_changes)}** | "
            f"Errors: **{len(errors)}**"
        ),
        color=color,
    )

    for change in changes[:10]:
        repo = change.get("repo", "?")
        parts = []
        if change.get("score_change"):
            parts.append(f"Score: {change['score_change']}")
        if change.get("new_patterns"):
            parts.append(f"New: {', '.join(change['new_patterns'][:3])}")
        if change.get("alert"):
            parts.append(f"Alert: {change['alert']}")
        embed.add_field(
            name=repo,
            value=_trunc("\n".join(parts) if parts else "Changes detected"),
            inline=False,
        )

    if no_changes:
        embed.add_field(
            name="Unchanged",
            value=_trunc(", ".join(no_changes)),
            inline=False,
        )

    for err in errors[:5]:
        embed.add_field(
            name=f"Error: {err.get('repo', '?')}",
            value=_trunc(str(err.get("error", "?"))),
            inline=False,
        )

    embed.set_footer(text="Animus Harvest")
    embed.timestamp = datetime.now(timezone.utc)
    return embed


async def _build_brief_embed() -> discord.Embed:
    """Build a daily brief embed from memory stats, harvest, and tasks."""
    embed = discord.Embed(
        title="Animus Daily Brief",
        color=COLOR_BLUE,
        timestamp=datetime.now(timezone.utc),
    )

    # Memory stats
    try:
        memory = _get_memory()
        stats = memory.get_statistics()
        total = stats.get("total_memories", stats.get("total", "?"))
        by_type = stats.get("by_type", {})
        type_line = ", ".join(f"{k}: {v}" for k, v in by_type.items()) if by_type else "N/A"
        embed.add_field(
            name="Memory",
            value=f"Total: **{total}**\n{type_line}",
            inline=False,
        )
    except Exception as e:
        embed.add_field(name="Memory", value=f"Error: {e}", inline=False)

    # Recent harvest state
    try:
        from animus.harvest_watchlist import get_watchlist, get_due_repos

        watchlist = get_watchlist()
        due = get_due_repos()
        embed.add_field(
            name="Harvest Watchlist",
            value=f"Tracked: **{len(watchlist)}** | Due for scan: **{len(due)}**",
            inline=False,
        )

        # Show top-scoring tracked repos
        scored = sorted(
            (r for r in watchlist if r.get("last_score") is not None),
            key=lambda r: r["last_score"],
            reverse=True,
        )
        if scored:
            top = "\n".join(
                f"- {r['target']}: {r['last_score']}/100" for r in scored[:5]
            )
            embed.add_field(name="Top Tracked Repos", value=_trunc(top), inline=False)
    except Exception as e:
        embed.add_field(name="Harvest", value=f"Error: {e}", inline=False)

    # Tasks
    try:
        cfg = _get_config()
        from animus.tasks import TaskTracker

        tracker = TaskTracker(cfg.data_dir)
        all_tasks = tracker.list()
        pending = [t for t in all_tasks if t.status.value == "pending"]
        in_progress = [t for t in all_tasks if t.status.value == "in_progress"]

        task_lines = []
        if pending:
            task_lines.append(f"Pending: **{len(pending)}**")
        if in_progress:
            task_lines.append(f"In progress: **{len(in_progress)}**")
        for t in (in_progress + pending)[:5]:
            task_lines.append(f"- [{t.id[:8]}] {t.description[:80]}")

        if task_lines:
            embed.add_field(name="Tasks", value=_trunc("\n".join(task_lines)), inline=False)
        else:
            embed.add_field(name="Tasks", value="No active tasks", inline=False)
    except Exception as e:
        embed.add_field(name="Tasks", value=f"Error: {e}", inline=False)

    # Recent memories (last 24h context)
    try:
        memory = _get_memory()
        recent = memory.recall(query="recent important context", limit=3)
        if recent:
            lines = []
            for m in recent:
                lines.append(f"- {m.content[:150]}")
            embed.add_field(
                name="Recent Context",
                value=_trunc("\n".join(lines)),
                inline=False,
            )
    except Exception:
        pass  # Not critical

    embed.set_footer(text="Animus Daily Brief")
    return embed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    token = os.environ.get("ANIMUS_DISCORD_TOKEN") or os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print(
            "Error: Set ANIMUS_DISCORD_TOKEN or DISCORD_BOT_TOKEN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    channel_id_str = os.environ.get("ANIMUS_DISCORD_CHANNEL", "")
    intel_channel: int | None = None
    if channel_id_str:
        try:
            intel_channel = int(channel_id_str)
        except ValueError:
            print(
                f"Warning: ANIMUS_DISCORD_CHANNEL '{channel_id_str}' is not a valid integer.",
                file=sys.stderr,
            )

    bot = AnimusBot(intel_channel_id=intel_channel)

    logger.info("Starting Animus Discord bot...")
    if intel_channel:
        logger.info("Intel auto-push channel: %s", intel_channel)
    else:
        logger.info("No ANIMUS_DISCORD_CHANNEL set — auto-push disabled")

    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
