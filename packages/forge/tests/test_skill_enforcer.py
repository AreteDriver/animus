"""Tests for SkillEnforcer runtime constraint validation."""

from unittest.mock import MagicMock

import pytest

from animus_forge.skills.enforcer import (
    EnforcementAction,
    EnforcementResult,
    SkillEnforcer,
    Violation,
    ViolationType,
)
from animus_forge.skills.models import SkillDefinition

ROLE_MAP = {"builder": ["system"]}


def _make_skill(
    name: str = "file_operations",
    agent: str = "system",
    protected_paths: list[str] | None = None,
    blocked_patterns: list[str] | None = None,
) -> SkillDefinition:
    return SkillDefinition(
        name=name,
        agent=agent,
        protected_paths=protected_paths or [],
        blocked_patterns=blocked_patterns or [],
    )


def _make_enforcer(library, skills, role_map=None):
    library.get_skills_for_agent.return_value = skills
    return SkillEnforcer(library, role_skill_agents=role_map or ROLE_MAP)


@pytest.fixture
def library():
    lib = MagicMock()
    lib.get_skills_for_agent.return_value = []
    return lib


class TestEnforcementResult:
    def test_allow_result_passed(self):
        r = EnforcementResult(action=EnforcementAction.ALLOW)
        assert r.passed is True
        assert r.has_violations is False

    def test_warn_result_not_passed(self):
        r = EnforcementResult(
            action=EnforcementAction.WARN,
            violations=[
                Violation(
                    type=ViolationType.PROTECTED_PATH,
                    severity="high",
                    message="test",
                    matched_text="/etc/passwd",
                )
            ],
        )
        assert r.passed is False
        assert r.has_violations is True

    def test_to_dict(self):
        r = EnforcementResult(action=EnforcementAction.ALLOW)
        d = r.to_dict()
        assert d["action"] == "allow"
        assert d["passed"] is True
        assert d["violations"] == []


class TestProtectedPaths:
    def test_exact_match(self, library):
        skill = _make_skill(protected_paths=["/etc/passwd"])
        enforcer = _make_enforcer(library, [skill])
        result = enforcer.check_output("builder", "I modified /etc/passwd")

        assert result.action == EnforcementAction.WARN
        assert len(result.violations) == 1
        assert result.violations[0].type == ViolationType.PROTECTED_PATH
        assert result.violations[0].matched_text == "/etc/passwd"

    def test_glob_match(self, library):
        skill = _make_skill(protected_paths=["~/.ssh/id_*"])
        enforcer = _make_enforcer(library, [skill])
        result = enforcer.check_output("builder", "Reading ~/.ssh/id_rsa for keys")

        assert result.action == EnforcementAction.WARN
        assert result.violations[0].matched_text == "~/.ssh/id_rsa"

    def test_safe_path_no_violation(self, library):
        skill = _make_skill(protected_paths=["/etc/passwd"])
        enforcer = _make_enforcer(library, [skill])
        result = enforcer.check_output("builder", "I edited /home/user/app.py")

        assert result.passed is True


class TestBlockedPatterns:
    def test_rm_rf_root(self, library):
        skill = _make_skill(blocked_patterns=[r"rm\s+-rf\s+/"])
        enforcer = _make_enforcer(library, [skill])
        result = enforcer.check_output("builder", "Run: rm -rf /tmp/data")

        assert result.action == EnforcementAction.BLOCK
        assert result.violations[0].type == ViolationType.BLOCKED_PATTERN
        assert result.violations[0].severity == "critical"

    def test_fork_bomb(self, library):
        skill = _make_skill(blocked_patterns=[r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"])
        enforcer = _make_enforcer(library, [skill])
        result = enforcer.check_output("builder", ":(){ :|:& };:")

        assert result.action == EnforcementAction.BLOCK

    def test_safe_output_passes(self, library):
        skill = _make_skill(blocked_patterns=[r"rm\s+-rf\s+/"])
        enforcer = _make_enforcer(library, [skill])
        result = enforcer.check_output("builder", "Created file /home/user/test.py")

        assert result.passed is True


class TestMultipleViolations:
    def test_collects_all_violations(self, library):
        skill = _make_skill(
            protected_paths=["/etc/passwd", "/etc/shadow"],
            blocked_patterns=[r"rm\s+-rf\s+/"],
        )
        enforcer = _make_enforcer(library, [skill])
        result = enforcer.check_output(
            "builder",
            "Read /etc/passwd and /etc/shadow then rm -rf /var",
        )

        assert result.action == EnforcementAction.BLOCK
        assert len(result.violations) == 3  # 2 paths + 1 pattern


class TestEdgeCases:
    def test_empty_output_allows(self, library):
        enforcer = _make_enforcer(library, [])
        result = enforcer.check_output("builder", "")
        assert result.passed is True

    def test_no_skills_for_role_allows(self, library):
        library.get_skills_for_agent.return_value = []
        enforcer = _make_enforcer(library, [])
        result = enforcer.check_output("builder", "anything")
        assert result.passed is True

    def test_unmapped_role_allows(self, library):
        enforcer = _make_enforcer(library, [_make_skill(blocked_patterns=[r"rm\s+-rf\s+/"])])
        result = enforcer.check_output("unknown_role", "rm -rf /")
        assert result.passed is True

    def test_invalid_regex_skipped(self, library):
        skill = _make_skill(blocked_patterns=["[invalid"])
        enforcer = _make_enforcer(library, [skill])
        result = enforcer.check_output("builder", "some output")
        assert result.passed is True
