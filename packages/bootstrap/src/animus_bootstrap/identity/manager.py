"""Identity file manager — reads, writes, and assembles identity files."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_MAX_LEARNED_LINES = 500


class IdentityFileManager:
    """Manage identity files with immutability guardrails.

    CORE_VALUES.md is locked — only ``write_locked()`` (wizard/dashboard)
    can modify it.  All other identity files can be written normally.

    Args:
        identity_dir: Directory where identity files are stored.
    """

    LOCKED_FILES: tuple[str, ...] = ("CORE_VALUES.md",)
    EDITABLE_FILES: tuple[str, ...] = (
        "IDENTITY.md",
        "CONTEXT.md",
        "GOALS.md",
        "PREFERENCES.md",
        "LEARNED.md",
    )
    ALL_FILES: tuple[str, ...] = LOCKED_FILES + EDITABLE_FILES

    def __init__(self, identity_dir: Path | str) -> None:
        self._dir = Path(identity_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
            autoescape=select_autoescape(),
        )

    @property
    def identity_dir(self) -> Path:
        """Return the identity directory path."""
        return self._dir

    def read(self, filename: str) -> str:
        """Read an identity file, returning empty string if missing."""
        self._validate_filename(filename)
        path = self._dir / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write(self, filename: str, content: str) -> None:
        """Write to an editable identity file.

        Raises:
            PermissionError: If ``filename`` is in ``LOCKED_FILES``.
        """
        self._validate_filename(filename)
        if filename in self.LOCKED_FILES:
            msg = f"{filename} is immutable. Use write_locked() for wizard/dashboard edits."
            raise PermissionError(msg)
        path = self._dir / filename
        path.write_text(content, encoding="utf-8")
        logger.info("Identity file written: %s", filename)

    def write_locked(self, filename: str, content: str) -> None:
        """Write to a locked identity file (wizard generation / dashboard only)."""
        self._validate_filename(filename)
        path = self._dir / filename
        path.write_text(content, encoding="utf-8")
        logger.info("Locked identity file written: %s", filename)

    def read_all(self) -> dict[str, str]:
        """Read all identity files, returning a mapping of filename to content."""
        return {f: self.read(f) for f in self.ALL_FILES}

    def exists(self, filename: str) -> bool:
        """Check whether an identity file exists on disk."""
        self._validate_filename(filename)
        return (self._dir / filename).exists()

    def append_to_learned(self, section: str, entry: str) -> None:
        """Append a timestamped entry to LEARNED.md under a section header.

        Trims oldest entries per section when the file exceeds
        ``_MAX_LEARNED_LINES``.
        """
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        path = self._dir / "LEARNED.md"

        current = path.read_text(encoding="utf-8") if path.exists() else ""
        section_header = f"## {section}"
        new_line = f"- [{timestamp}] {entry}"

        if section_header in current:
            # Insert after section header
            idx = current.index(section_header)
            end_of_header = current.index("\n", idx) + 1
            current = current[:end_of_header] + new_line + "\n" + current[end_of_header:]
        else:
            # Add new section at end
            if current and not current.endswith("\n"):
                current += "\n"
            current += f"\n{section_header}\n{new_line}\n"

        # Trim if over line cap
        lines = current.splitlines(keepends=True)
        if len(lines) > _MAX_LEARNED_LINES:
            current = "".join(lines[:_MAX_LEARNED_LINES])
            logger.info("LEARNED.md trimmed to %d lines", _MAX_LEARNED_LINES)

        path.write_text(current, encoding="utf-8")

    def get_identity_prompt(self, memory_context: str = "") -> str:
        """Assemble all identity files into a single system prompt block.

        Assembly order per spec:
            CORE_VALUES.md → IDENTITY.md → CONTEXT.md → GOALS.md →
            PREFERENCES.md → LEARNED.md → memory_context
        """
        parts: list[str] = []

        core = self.read("CORE_VALUES.md")
        if core:
            parts.append(core)

        identity = self.read("IDENTITY.md")
        if identity:
            parts.append(f"\n## Who I'm Talking To\n{identity}")

        context = self.read("CONTEXT.md")
        if context:
            parts.append(f"\n## Current Context\n{context}")

        goals = self.read("GOALS.md")
        if goals:
            parts.append(f"\n## Goals\n{goals}")

        prefs = self.read("PREFERENCES.md")
        if prefs:
            parts.append(f"\n## Communication Preferences\n{prefs}")

        learned = self.read("LEARNED.md")
        if learned:
            parts.append(f"\n## What I've Learned About You\n{learned}")

        if memory_context:
            parts.append(f"\n## Relevant Memory\n{memory_context}")

        return "\n".join(parts) if parts else ""

    def get_condensed_prompt(self) -> str:
        """Build a short, directive system prompt for local LLMs.

        Full identity files are too long for small models to follow.
        This extracts key constraints into ~200 words that a 14B model
        can reliably obey.
        """
        # Extract key facts from identity files
        identity = self.read("IDENTITY.md") or ""
        context = self.read("CONTEXT.md") or ""
        prefs = self.read("PREFERENCES.md") or ""

        # Pull name from identity
        name_line = ""
        for line in identity.splitlines():
            if line.startswith("**Name:**"):
                name_line = line.split("**Name:**")[1].strip()
                break

        # Pull anti-patterns from preferences
        anti_patterns: list[str] = []
        in_anti = False
        for line in prefs.splitlines():
            if "Anti-Pattern" in line or "Never Do" in line:
                in_anti = True
                continue
            if in_anti and line.startswith("- "):
                anti_patterns.append(line[2:].strip())
            elif in_anti and line.startswith("#"):
                break

        # Pull "What I Am" section from context
        what_i_am: list[str] = []
        in_what = False
        for line in context.splitlines():
            if "What I Am" in line:
                in_what = True
                continue
            if in_what and line.startswith("##"):
                break
            if in_what and line.strip():
                what_i_am.append(line.strip())

        parts = [
            f"You are Animus, sovereign AI exocortex for {name_line or 'Arete'}.",
            "You are ALREADY BUILT — 13,188 tests, 4 packages, production code.",
            "You run locally on Ollama. No data leaves this machine.",
        ]

        if what_i_am:
            parts.append("\n".join(what_i_am[:8]))

        # Pull key interaction rules from identity
        for line in identity.splitlines():
            if line.startswith("**Operating principle"):
                parts.append(line)
                break

        parts.append(
            "\nRULES (OBEY STRICTLY):\n"
            "- Be direct. No filler, no preamble, no restating the question.\n"
            "- NEVER propose building things that already exist.\n"
            "- NEVER generate generic project plans or numbered wishlists.\n"
            "- NEVER hallucinate commands, repos, or frameworks.\n"
            "- When unsure about the codebase, say 'I'd need to check' — do NOT guess.\n"
            "- Treat Arete as expert peer (17+ years ops). No hand-holding."
        )

        if anti_patterns:
            parts.append("\nNEVER DO:\n" + "\n".join(f"- {ap}" for ap in anti_patterns[:5]))

        return "\n".join(parts)

    def generate_from_templates(self, context: dict) -> None:
        """Generate all identity files from Jinja2 templates.

        Uses ``write_locked()`` for CORE_VALUES.md and ``write()`` for
        editable files.  Skips files that already exist on disk.

        Args:
            context: Template variables (name, timezone, etc.).
        """
        for filename in self.ALL_FILES:
            if self.exists(filename):
                logger.info("Identity file already exists, skipping: %s", filename)
                continue

            template_name = f"{filename}.j2"
            try:
                template = self._jinja.get_template(template_name)
            except Exception:
                logger.warning("No template found for %s", filename)
                continue

            content = template.render(**context)

            if filename in self.LOCKED_FILES:
                self.write_locked(filename, content)
            else:
                self.write(filename, content)

    def _validate_filename(self, filename: str) -> None:
        """Ensure filename is one of the known identity files."""
        if filename not in self.ALL_FILES:
            msg = f"Unknown identity file: {filename}. Must be one of {self.ALL_FILES}"
            raise ValueError(msg)
