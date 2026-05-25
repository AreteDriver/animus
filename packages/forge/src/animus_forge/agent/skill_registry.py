"""Skill registry — loads Claude Code skills from `~/.claude/skills/`.

Spec §4.7 + R6: discovers `SKILL.md` (or `skill.md`) under each
subdirectory and parses optional YAML frontmatter for `name`,
`description`, `invokable` (default True). Per-skill load failures
log + skip; aggregate count surfaces in `skill_load_warnings`.

This registry is INDEPENDENT of Animus's existing YAML skill resolver
per RD3 — v0 reads Claude Code format only.

Spec deviation (2026-05-25): 27% of installed skills (34 / 127) ship
without frontmatter. To meet the C4 floor of 127, the loader falls back
to `name = directory_name` and `description = first heading or
directory_name` when frontmatter is absent. Will be documented in RD9b
in a follow-on spec amendment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from .types import Skill

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_DIR = Path.home() / ".claude" / "skills"
_SKILL_FILENAMES = ("SKILL.md", "skill.md")
_FRONTMATTER_RX = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_HEADING_RX = re.compile(r"^#+\s+(.+?)\s*$", re.MULTILINE)


def _first_heading(body: str) -> str:
    """Return the first markdown heading text, or empty string."""
    match = _HEADING_RX.search(body)
    return match.group(1).strip() if match else ""


def _parse_skill(skill_md: Path) -> Skill:
    """Parse one SKILL.md file into a Skill.

    Tries YAML frontmatter first; falls back to directory-name + first
    heading when frontmatter is absent or malformed.
    """
    content = skill_md.read_text(encoding="utf-8")
    dir_name = skill_md.parent.name

    match = _FRONTMATTER_RX.match(content)
    if match:
        yaml_block, body = match.group(1), match.group(2)
        try:
            meta = yaml.safe_load(yaml_block)
            if not isinstance(meta, dict):
                meta = {}
        except yaml.YAMLError:
            logger.warning("invalid YAML frontmatter in %s; falling back", skill_md)
            meta, body = {}, content
    else:
        meta, body = {}, content

    name = str(meta.get("name") or dir_name)
    description = str(meta.get("description") or _first_heading(body) or dir_name)
    invokable = bool(meta.get("invokable", True))
    return Skill(
        name=name,
        description=description,
        body=body,
        invokable=invokable,
        source_path=skill_md,
    )


def load_skills(
    skills_dir: Path | None = None,
) -> tuple[dict[str, Skill], list[str]]:
    """Discover and load all skills under `skills_dir`.

    Args:
        skills_dir: Root to scan. Defaults to `~/.claude/skills/`.

    Returns:
        Tuple of:
        - dict mapping skill name → Skill (later wins on duplicate name,
          but a warning is appended)
        - list of human-readable warnings for per-skill load failures
    """
    root = skills_dir if skills_dir is not None else DEFAULT_SKILLS_DIR
    registry: dict[str, Skill] = {}
    warnings: list[str] = []

    if not root.exists() or not root.is_dir():
        warnings.append(f"skills dir not found: {root}")
        return registry, warnings

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = None
        for candidate in _SKILL_FILENAMES:
            if (entry / candidate).is_file():
                skill_md = entry / candidate
                break
        if skill_md is None:
            warnings.append(f"{entry.name}: no SKILL.md found")
            continue
        try:
            skill = _parse_skill(skill_md)
        except Exception as e:  # noqa: BLE001 — per-skill isolation
            logger.warning("failed to parse skill at %s: %s", skill_md, e)
            warnings.append(f"{entry.name}: {e}")
            continue
        if skill.name in registry:
            warnings.append(
                f"{entry.name}: duplicate name '{skill.name}' "
                f"(was {registry[skill.name].source_path}); later wins"
            )
        registry[skill.name] = skill
    return registry, warnings


class SkillRegistry:
    """Convenience wrapper exposing `list()` + `get()` over a loaded registry.

    Held by the agent for its lifetime; not thread-safe (one agent run is
    single-threaded). Caching the loaded skill set in-memory satisfies
    spec MAY-16.
    """

    def __init__(
        self,
        skills_dir: Path | None = None,
    ):
        self._skills, self._warnings = load_skills(skills_dir)

    def list(self) -> list[str]:
        """Sorted list of skill names."""
        return sorted(self._skills.keys())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._skills
