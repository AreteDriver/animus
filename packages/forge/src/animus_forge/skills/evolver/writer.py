"""Write skill definitions to disk as YAML + update registry."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from animus_forge.skills.models import SkillDefinition

logger = logging.getLogger(__name__)


class SkillWriter:
    """Writes skill definitions to the filesystem.

    Args:
        skills_dir: Root skills directory (contains registry.yaml and
            category subdirectories).
    """

    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir

    def skill_to_yaml(self, skill: SkillDefinition) -> str:
        """Serialize a SkillDefinition to deterministic YAML.

        Args:
            skill: The skill definition to serialize.

        Returns:
            YAML string with sorted keys.
        """
        data: dict = {
            "name": skill.name,
            "version": skill.version,
            "type": skill.type,
            "agent": skill.agent,
            "category": skill.category,
            "description": skill.description,
            "status": skill.status,
            "risk_level": skill.risk_level,
            "consensus_level": skill.consensus_level,
            "trust": skill.trust,
            "parallel_safe": skill.parallel_safe,
        }

        if skill.tools:
            data["tools"] = skill.tools

        if skill.dependencies:
            data["dependencies"] = skill.dependencies

        if skill.routing:
            routing: dict = {}
            if skill.routing.use_when:
                routing["use_when"] = skill.routing.use_when
            if skill.routing.do_not_use_when:
                routing["do_not_use_when"] = [
                    exc.model_dump(exclude_defaults=True) for exc in skill.routing.do_not_use_when
                ]
            data["routing"] = routing

        if skill.capabilities:
            caps = []
            for cap in skill.capabilities:
                cap_dict = cap.model_dump(exclude_defaults=True)
                caps.append(cap_dict)
            data["capabilities"] = caps

        if skill.verification:
            data["verification"] = skill.verification.model_dump(exclude_defaults=True)

        if skill.error_handling:
            data["error_handling"] = skill.error_handling.model_dump(exclude_defaults=True)

        if skill.contracts:
            data["contracts"] = skill.contracts.model_dump(exclude_defaults=True)

        if skill.skill_inputs:
            data["inputs"] = skill.skill_inputs
        if skill.skill_outputs:
            data["outputs"] = skill.skill_outputs

        return yaml.dump(data, default_flow_style=False, sort_keys=True, allow_unicode=True)

    def write_skill(self, skill: SkillDefinition, category: str) -> Path:
        """Write a skill definition to disk.

        Creates ``skills/{category}/{skill.name}/schema.yaml``.

        Args:
            skill: The skill definition to write.
            category: Category subdirectory name.

        Returns:
            Path to the written schema.yaml file.
        """
        skill_dir = self._skills_dir / category / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)

        schema_path = skill_dir / "schema.yaml"
        yaml_content = self.skill_to_yaml(skill)
        schema_path.write_text(yaml_content)

        logger.info("Wrote skill %s v%s to %s", skill.name, skill.version, schema_path)
        return schema_path

    def update_registry(self, skill: SkillDefinition, category: str) -> None:
        """Add or update a skill entry in registry.yaml.

        Args:
            skill: The skill definition.
            category: Category subdirectory name.
        """
        registry_path = self._skills_dir / "registry.yaml"
        if registry_path.exists():
            with open(registry_path) as f:
                registry_data = yaml.safe_load(f) or {}
        else:
            registry_data = {"version": "1.0.0", "categories": {}, "skills": []}

        skills_list: list[dict] = registry_data.get("skills", [])

        # Update existing or append new
        found = False
        for entry in skills_list:
            if entry.get("name") == skill.name:
                entry["version"] = skill.version
                entry["status"] = skill.status
                entry["path"] = f"{category}/{skill.name}"
                entry["agent"] = skill.agent
                entry["description"] = skill.description
                found = True
                break

        if not found:
            skills_list.append(
                {
                    "name": skill.name,
                    "path": f"{category}/{skill.name}",
                    "agent": skill.agent,
                    "version": skill.version,
                    "status": skill.status,
                    "description": skill.description,
                    "capabilities": [c.name for c in skill.capabilities],
                }
            )

        registry_data["skills"] = skills_list

        with open(registry_path, "w") as f:
            yaml.dump(
                registry_data, f, default_flow_style=False, sort_keys=True, allow_unicode=True
            )

        logger.info("Updated registry for skill %s", skill.name)

    def read_raw_yaml(self, skill_name: str) -> str:
        """Read the raw YAML content of a skill's schema.yaml.

        Searches all category subdirectories for the skill.

        Args:
            skill_name: Name of the skill.

        Returns:
            Raw YAML string, or empty string if not found.
        """
        for category_dir in self._skills_dir.iterdir():
            if not category_dir.is_dir():
                continue
            schema_path = category_dir / skill_name / "schema.yaml"
            if schema_path.exists():
                return schema_path.read_text()
        return ""
