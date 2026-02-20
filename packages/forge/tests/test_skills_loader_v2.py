"""Tests for v2 schema loading in skills/loader.py."""

import pytest
import yaml

from animus_forge.skills.loader import load_skill


@pytest.fixture
def v2_schema_dir(tmp_path):
    """Create a temp dir with a v2 schema.yaml."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()

    schema = {
        "name": "test-skill",
        "version": "2.0.0",
        "description": "A test v2 skill",
        "agent": "system",
        "type": "agent",
        "category": "analysis",
        "risk_level": "medium",
        "consensus_level": "majority",
        "trust": "supervised",
        "parallel_safe": True,
        "tools": ["Read", "Write", "Grep"],
        "routing": {
            "use_when": [
                "Filesystem operations needing safety",
                "Batch operations requiring rollback",
            ],
            "do_not_use_when": [
                {
                    "condition": "Single file read with known path",
                    "instead": "Read tool directly",
                    "reason": "Skill overhead unnecessary",
                },
                {
                    "condition": "Content search by pattern",
                    "instead": "Grep tool",
                    "reason": "Grep is faster",
                },
            ],
        },
        "inputs": {
            "operation": {"type": "string", "required": True},
            "path": {"type": "string", "required": True},
        },
        "outputs": {
            "success": {"type": "boolean"},
            "error": {"type": "string"},
        },
        "capabilities": [
            {
                "name": "read_file",
                "description": "Read file contents",
                "risk": "low",
                "consensus": "any",
                "parallel_safe": True,
                "intent_required": True,
                "inputs": {
                    "path": {"type": "string", "required": True},
                },
                "outputs": {
                    "content": {"type": "string"},
                },
                "post_execution": [
                    "Assess if content is sufficient",
                    "Re-read if truncated",
                ],
            },
            {
                "name": "delete_file",
                "description": "Permanently remove a file",
                "risk": "high",
                "consensus": "majority",
                "parallel_safe": False,
                "intent_required": True,
                "inputs": {
                    "path": {"type": "string", "required": True},
                    "force": {"type": "boolean", "required": False},
                },
                "outputs": {
                    "success": {"type": "boolean"},
                },
                "post_execution": ["Verify file no longer exists"],
            },
        ],
        "verification": {
            "pre_conditions": ["Target path exists"],
            "post_conditions": ["No protected paths modified"],
            "checkpoints": [
                {
                    "trigger": "Before any destructive operation",
                    "action": "Verify backup exists",
                }
            ],
            "completion_checklist": [
                "All operations succeeded",
                "Backups created for destructive ops",
            ],
        },
        "error_handling": {
            "escalation": [
                {
                    "error_class": "recoverable",
                    "description": "Permission denied, file locked",
                    "action": "retry",
                    "max_retries": 3,
                    "fallback": "Report error with context",
                },
                {
                    "error_class": "environment",
                    "description": "Disk full",
                    "action": "report",
                    "max_retries": 0,
                    "fallback": "Report immediately",
                },
            ],
            "self_correction": [
                "Destructive op without backup: back up remaining state",
            ],
        },
        "contracts": {
            "provides": [
                {
                    "name": "file_content",
                    "type": "string",
                    "consumers": ["context-mapper"],
                    "description": "Raw file content",
                },
            ],
            "requires": [
                {
                    "name": "target_paths",
                    "type": "array",
                    "provider": "context-mapper",
                    "description": "File paths from context mapping",
                },
            ],
        },
        "protected_paths": ["/etc/passwd", "/boot"],
    }

    with open(skill_dir / "schema.yaml", "w") as f:
        yaml.dump(schema, f)

    # Also add a SKILL.md
    (skill_dir / "SKILL.md").write_text("# Test Skill\nThis is a test.")

    return skill_dir


@pytest.fixture
def v1_schema_dir(tmp_path):
    """Create a temp dir with a v1 (dict-style capabilities) schema.yaml."""
    skill_dir = tmp_path / "old-skill"
    skill_dir.mkdir()

    schema = {
        "skill_name": "old_skill",
        "version": "1.0.0",
        "agent": "browser",
        "description": "A v1 skill",
        "capabilities": {
            "search": {
                "description": "Search the web",
                "inputs": {"query": {"type": "string"}},
                "outputs": {"results": "list"},
                "risk_level": "low",
                "consensus_required": "any",
            },
            "scrape": {
                "description": "Scrape a page",
                "risk_level": "medium",
                "consensus_required": "majority",
            },
        },
        "protected_paths": [],
        "blocked_patterns": [],
    }

    with open(skill_dir / "schema.yaml", "w") as f:
        yaml.dump(schema, f)

    return skill_dir


class TestLoadSkillV2:
    def test_basic_fields(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        assert skill.name == "test-skill"
        assert skill.version == "2.0.0"
        assert skill.agent == "system"
        assert skill.type == "agent"
        assert skill.category == "analysis"
        assert skill.risk_level == "medium"
        assert skill.consensus_level == "majority"
        assert skill.trust == "supervised"
        assert skill.parallel_safe is True
        assert skill.tools == ["Read", "Write", "Grep"]

    def test_capabilities_parsed_as_list(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        assert len(skill.capabilities) == 2
        assert skill.capability_names == ["read_file", "delete_file"]

    def test_capability_v2_field_mapping(self, v2_schema_dir):
        """v2 uses 'risk' and 'consensus' shorthand — mapped to risk_level/consensus_required."""
        skill = load_skill(v2_schema_dir)
        read_cap = skill.get_capability("read_file")
        assert read_cap is not None
        assert read_cap.risk_level == "low"
        assert read_cap.consensus_required == "any"
        assert read_cap.parallel_safe is True
        assert read_cap.intent_required is True

        delete_cap = skill.get_capability("delete_file")
        assert delete_cap is not None
        assert delete_cap.risk_level == "high"
        assert delete_cap.consensus_required == "majority"
        assert delete_cap.parallel_safe is False

    def test_post_execution_parsed(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        read_cap = skill.get_capability("read_file")
        assert len(read_cap.post_execution) == 2
        assert "truncated" in read_cap.post_execution[1]

    def test_routing_parsed(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        assert skill.routing is not None
        assert len(skill.routing.use_when) == 2
        assert len(skill.routing.do_not_use_when) == 2
        exc = skill.routing.do_not_use_when[0]
        assert exc.condition == "Single file read with known path"
        assert exc.instead == "Read tool directly"
        assert exc.reason == "Skill overhead unnecessary"

    def test_verification_parsed(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        assert skill.verification is not None
        assert len(skill.verification.pre_conditions) == 1
        assert len(skill.verification.post_conditions) == 1
        assert len(skill.verification.checkpoints) == 1
        assert skill.verification.checkpoints[0].trigger == "Before any destructive operation"
        assert len(skill.verification.completion_checklist) == 2

    def test_error_handling_parsed(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        assert skill.error_handling is not None
        assert len(skill.error_handling.escalation) == 2
        assert skill.error_handling.escalation[0].error_class == "recoverable"
        assert skill.error_handling.escalation[0].max_retries == 3
        assert skill.error_handling.escalation[1].action == "report"
        assert len(skill.error_handling.self_correction) == 1

    def test_contracts_parsed(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        assert skill.contracts is not None
        assert len(skill.contracts.provides) == 1
        assert skill.contracts.provides[0].name == "file_content"
        assert skill.contracts.provides[0].type == "string"
        assert skill.contracts.provides[0].consumers == ["context-mapper"]
        assert len(skill.contracts.requires) == 1
        assert skill.contracts.requires[0].provider == "context-mapper"

    def test_skill_inputs_outputs(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        assert "operation" in skill.skill_inputs
        assert "success" in skill.skill_outputs

    def test_skill_doc_loaded(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        assert "Test Skill" in skill.skill_doc

    def test_protected_paths_preserved(self, v2_schema_dir):
        skill = load_skill(v2_schema_dir)
        assert "/etc/passwd" in skill.protected_paths
        assert "/boot" in skill.protected_paths


class TestLoadSkillV1Backward:
    def test_v1_dict_capabilities_still_work(self, v1_schema_dir):
        skill = load_skill(v1_schema_dir)
        assert skill.name == "old_skill"
        assert skill.agent == "browser"
        assert len(skill.capabilities) == 2
        names = {c.name for c in skill.capabilities}
        assert "search" in names
        assert "scrape" in names

    def test_v1_risk_and_consensus_fields(self, v1_schema_dir):
        skill = load_skill(v1_schema_dir)
        search = skill.get_capability("search")
        assert search.risk_level == "low"
        assert search.consensus_required == "any"

        scrape = skill.get_capability("scrape")
        assert scrape.risk_level == "medium"
        assert scrape.consensus_required == "majority"

    def test_v1_no_v2_sections(self, v1_schema_dir):
        skill = load_skill(v1_schema_dir)
        assert skill.routing is None
        assert skill.verification is None
        assert skill.error_handling is None
        assert skill.contracts is None
        assert skill.skill_inputs == {}
        assert skill.skill_outputs == {}


class TestLoadSkillEdgeCases:
    def test_missing_schema_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_skill(tmp_path / "nonexistent")

    def test_empty_capabilities(self, tmp_path):
        skill_dir = tmp_path / "empty-caps"
        skill_dir.mkdir()
        schema = {"name": "empty", "agent": "test", "capabilities": []}
        with open(skill_dir / "schema.yaml", "w") as f:
            yaml.dump(schema, f)
        skill = load_skill(skill_dir)
        assert skill.capabilities == []

    def test_capabilities_with_non_dict_items_skipped(self, tmp_path):
        """Non-dict items in v2 capabilities list are skipped."""
        skill_dir = tmp_path / "bad-caps"
        skill_dir.mkdir()
        schema = {
            "name": "bad",
            "agent": "test",
            "capabilities": ["not_a_dict", {"name": "valid", "description": "OK"}],
        }
        with open(skill_dir / "schema.yaml", "w") as f:
            yaml.dump(schema, f)
        skill = load_skill(skill_dir)
        assert len(skill.capabilities) == 1
        assert skill.capabilities[0].name == "valid"

    def test_routing_with_non_dict_exclusions_skipped(self, tmp_path):
        """Non-dict items in do_not_use_when are skipped."""
        skill_dir = tmp_path / "bad-routing"
        skill_dir.mkdir()
        schema = {
            "name": "bad-routing",
            "agent": "test",
            "routing": {
                "use_when": ["test"],
                "do_not_use_when": [
                    "not_a_dict",
                    {"condition": "valid exclusion"},
                ],
            },
        }
        with open(skill_dir / "schema.yaml", "w") as f:
            yaml.dump(schema, f)
        skill = load_skill(skill_dir)
        assert skill.routing is not None
        assert len(skill.routing.do_not_use_when) == 1

    def test_v2_name_field_preferred_over_skill_name(self, tmp_path):
        """v2 uses 'name', v1 uses 'skill_name'. 'name' takes priority."""
        skill_dir = tmp_path / "name-test"
        skill_dir.mkdir()
        schema = {"name": "v2-name", "skill_name": "v1-name", "agent": "test"}
        with open(skill_dir / "schema.yaml", "w") as f:
            yaml.dump(schema, f)
        skill = load_skill(skill_dir)
        assert skill.name == "v2-name"

    def test_fallback_to_skill_name(self, tmp_path):
        """When 'name' is missing, falls back to 'skill_name'."""
        skill_dir = tmp_path / "fallback-test"
        skill_dir.mkdir()
        schema = {"skill_name": "v1-name", "agent": "test"}
        with open(skill_dir / "schema.yaml", "w") as f:
            yaml.dump(schema, f)
        skill = load_skill(skill_dir)
        assert skill.name == "v1-name"

    def test_fallback_to_dir_name(self, tmp_path):
        """When both 'name' and 'skill_name' are missing, use dir name."""
        skill_dir = tmp_path / "dir-fallback"
        skill_dir.mkdir()
        schema = {"agent": "test"}
        with open(skill_dir / "schema.yaml", "w") as f:
            yaml.dump(schema, f)
        skill = load_skill(skill_dir)
        assert skill.name == "dir-fallback"
