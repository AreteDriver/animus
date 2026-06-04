"""SkillResolver — maps a skill name to its on-disk ResolvedSkill.

Resolution order (R2):
  1. ai-skills `registry.yaml` (canonical)
  2. Filesystem scan of `~/projects/ai-skills/{personas,agents,workflows}/`
     with WARN log

Out-of-repo paths (e.g. symlinks pointing outside the ai-skills tree)
are rejected with SkillResolutionError per R3.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml

from .errors import SkillResolutionError
from .types import AgentSchema, RegistryEntry, ResolvedSkill, Tier

logger = logging.getLogger(__name__)

DEFAULT_AI_SKILLS_REPO = Path.home() / "projects" / "ai-skills"
TIERS: tuple[Tier, ...] = ("persona", "agent", "workflow")
TIER_DIR_NAMES: dict[Tier, str] = {
    "persona": "personas",
    "agent": "agents",
    "workflow": "workflows",
}
CACHE_MAX_ENTRIES = 500


class SkillResolver:
    """Resolve skill names against the ai-skills repo.

    The registry is parsed lazily on first resolve() call and cached
    for the lifetime of the resolver. Per-skill metadata (schema.yaml,
    output.schema.yaml) is cached via lru_cache up to CACHE_MAX_ENTRIES.
    """

    def __init__(self, repo_path: Path | None = None) -> None:
        env_path = os.environ.get("AI_SKILLS_REPO")
        if repo_path is None and env_path:
            repo_path = Path(env_path)
        self.repo_path = (repo_path or DEFAULT_AI_SKILLS_REPO).resolve()
        self._registry: dict[tuple[str, Tier], RegistryEntry] | None = None

    def resolve(self, name: str, tier: Tier) -> ResolvedSkill:
        """Look up `name` in the requested `tier`.

        Returns a ResolvedSkill. Raises SkillResolutionError on:
          - unknown skill
          - tier mismatch (registry says it's in a different tier)
          - skill_md_path resolves outside the repo (symlink attack)
          - missing SKILL.md
        """
        if tier not in TIERS:
            raise SkillResolutionError(f"unknown tier: {tier}")

        registry = self._ensure_registry_loaded()
        entry = registry.get((name, tier))

        if entry is not None:
            skill_dir = self.repo_path / entry.path
        else:
            # Filesystem fallback (R2)
            skill_dir = self._scan_filesystem(name, tier)
            if skill_dir is None:
                raise SkillResolutionError(f"skill not found: {name!r} (tier={tier})")
            logger.warning(
                "skill %r (tier=%s) not in registry — using filesystem fallback at %s",
                name,
                tier,
                skill_dir,
            )

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise SkillResolutionError(f"SKILL.md missing at {skill_md}")

        # R3 — refuse anything that resolves outside the repo
        resolved_md = skill_md.resolve()
        if not _is_within(resolved_md, self.repo_path):
            raise SkillResolutionError(f"refusing out-of-repo skill path: {resolved_md}")

        schema = self._load_schema(skill_dir) if tier == "agent" else None
        risk_level = self._effective_risk_level(entry, schema)

        return ResolvedSkill(
            name=name,
            tier=tier,
            skill_md_path=skill_md,
            risk_level=risk_level,
            schema=schema,
            registry_entry=entry,
        )

    def _ensure_registry_loaded(self) -> dict[tuple[str, Tier], RegistryEntry]:
        if self._registry is not None:
            return self._registry

        registry_path = self.repo_path / "registry.yaml"
        registry: dict[tuple[str, Tier], RegistryEntry] = {}

        if not registry_path.exists():
            logger.warning(
                "registry.yaml missing at %s — all resolves use filesystem fallback", registry_path
            )
            self._registry = registry
            return registry

        try:
            data = yaml.safe_load(registry_path.read_text()) or {}
        except yaml.YAMLError as exc:
            logger.warning("registry.yaml malformed (%s) — falling back to filesystem", exc)
            self._registry = registry
            return registry

        registry.update(self._parse_personas(data.get("personas", {})))
        registry.update(self._parse_agents(data.get("agents", {})))
        registry.update(self._parse_workflows(data.get("workflows", [])))
        self._registry = registry
        return registry

    @staticmethod
    def _parse_personas(personas: dict) -> dict[tuple[str, Tier], RegistryEntry]:
        out: dict[tuple[str, Tier], RegistryEntry] = {}
        if not isinstance(personas, dict):
            return out
        for _category, entries in personas.items():
            if not isinstance(entries, list):
                continue
            for raw in entries:
                if not isinstance(raw, dict) or "name" not in raw:
                    continue
                entry = RegistryEntry(
                    name=raw["name"],
                    path=raw.get("path", ""),
                    tier="persona",
                    description=raw.get("description", ""),
                    has_references=bool(raw.get("has_references", False)),
                    risk_level=raw.get("risk_level"),
                )
                out[(entry.name, "persona")] = entry
        return out

    @staticmethod
    def _parse_agents(agents: dict) -> dict[tuple[str, Tier], RegistryEntry]:
        out: dict[tuple[str, Tier], RegistryEntry] = {}
        if not isinstance(agents, dict):
            return out
        for _category, entries in agents.items():
            if not isinstance(entries, list):
                continue
            for raw in entries:
                if not isinstance(raw, dict) or "name" not in raw:
                    continue
                entry = RegistryEntry(
                    name=raw["name"],
                    path=raw.get("path", ""),
                    tier="agent",
                    description=raw.get("description", ""),
                    has_references=bool(raw.get("has_references", False)),
                    risk_level=raw.get("risk_level"),
                )
                out[(entry.name, "agent")] = entry
        return out

    @staticmethod
    def _parse_workflows(workflows: list) -> dict[tuple[str, Tier], RegistryEntry]:
        out: dict[tuple[str, Tier], RegistryEntry] = {}
        if not isinstance(workflows, list):
            return out
        for raw in workflows:
            if not isinstance(raw, dict) or "name" not in raw:
                continue
            entry = RegistryEntry(
                name=raw["name"],
                path=raw.get("path", ""),
                tier="workflow",
                description=raw.get("description", ""),
                has_references=bool(raw.get("has_references", False)),
                risk_level=raw.get("risk_level"),
            )
            out[(entry.name, "workflow")] = entry
        return out

    def _scan_filesystem(self, name: str, tier: Tier) -> Path | None:
        """Walk the tier directory looking for a matching skill folder."""
        tier_root = self.repo_path / TIER_DIR_NAMES[tier]
        if not tier_root.exists():
            return None
        # Personas + agents are nested under category subdirs;
        # workflows is flat.
        if tier == "workflow":
            candidate = tier_root / name
            return candidate if candidate.is_dir() else None
        for category_dir in tier_root.iterdir():
            if not category_dir.is_dir():
                continue
            candidate = category_dir / name
            if candidate.is_dir():
                return candidate
        return None

    @lru_cache(maxsize=CACHE_MAX_ENTRIES)
    def _load_schema(self, skill_dir: Path) -> AgentSchema | None:
        """Parse schema.yaml + check for output.schema.yaml."""
        schema_path = skill_dir / "schema.yaml"
        if not schema_path.exists():
            return None
        try:
            data = yaml.safe_load(schema_path.read_text()) or {}
        except yaml.YAMLError as exc:
            logger.warning("schema.yaml malformed at %s (%s)", schema_path, exc)
            return None
        return AgentSchema(
            inputs=data.get("inputs", {}) or {},
            outputs=data.get("outputs", {}) or {},
            risk_level=data.get("risk_level", "safe"),
            has_output_schema=(skill_dir / "output.schema.yaml").exists(),
        )

    @staticmethod
    def _effective_risk_level(
        entry: RegistryEntry | None,
        schema: AgentSchema | None,
    ):
        """Apply OQ3 resolution order: registry → schema → safe."""
        if entry and entry.risk_level:
            return entry.risk_level
        if schema and schema.risk_level:
            return schema.risk_level
        return "safe"


def _is_within(path: Path, root: Path) -> bool:
    """True if `path` is `root` or a descendant of it (resolved)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
