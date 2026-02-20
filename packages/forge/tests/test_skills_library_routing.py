"""Tests for v2 routing queries and enhanced context building in SkillLibrary."""

import pytest

from animus_forge.skills.library import SkillLibrary, _tokenize
from animus_forge.skills.models import (
    EscalationRule,
    RoutingExclusion,
    SkillCapability,
    SkillDefinition,
    SkillErrorHandling,
    SkillRegistry,
    SkillRouting,
    SkillVerification,
)

# --- Tokenizer tests ---


class TestTokenize:
    def test_basic(self):
        assert _tokenize("hello world") == {"hello", "world"}

    def test_short_words_excluded(self):
        assert _tokenize("a is to the for") == {"the", "for"}

    def test_case_insensitive(self):
        assert _tokenize("Hello WORLD") == {"hello", "world"}

    def test_empty_string(self):
        assert _tokenize("") == set()


# --- Fixtures ---


def _make_v2_skill(
    name="test-skill",
    agent="system",
    use_when=None,
    do_not_use_when=None,
    capabilities=None,
    consensus_level="any",
    verification=None,
    error_handling=None,
    contracts=None,
    status="active",
):
    routing = None
    if use_when or do_not_use_when:
        exclusions = [
            RoutingExclusion(**exc) if isinstance(exc, dict) else exc
            for exc in (do_not_use_when or [])
        ]
        routing = SkillRouting(
            use_when=use_when or [],
            do_not_use_when=exclusions,
        )
    return SkillDefinition(
        name=name,
        agent=agent,
        version="2.0.0",
        description=f"Test skill {name}",
        capabilities=capabilities or [],
        capability_names=[c.name for c in (capabilities or [])],
        routing=routing,
        consensus_level=consensus_level,
        verification=verification,
        error_handling=error_handling,
        contracts=contracts,
        status=status,
    )


@pytest.fixture
def library_with_v2_skills(tmp_path):
    """Build a SkillLibrary with in-memory v2 skills (bypassing disk loading)."""

    def _make(skills):
        lib = SkillLibrary.__new__(SkillLibrary)
        lib._skills_dir = tmp_path
        lib._registry = SkillRegistry(
            version="2.0.0",
            skills=skills,
        )
        return lib

    return _make


# --- find_skills_for_task tests ---


class TestFindSkillsForTask:
    def test_matches_use_when(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="file-ops",
            use_when=["Filesystem operations needing safety controls"],
        )
        lib = library_with_v2_skills([skill])
        results = lib.find_skills_for_task("Need filesystem operations for this project")
        assert len(results) == 1
        assert results[0].name == "file-ops"

    def test_no_match(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="file-ops",
            use_when=["Filesystem operations needing safety controls"],
        )
        lib = library_with_v2_skills([skill])
        results = lib.find_skills_for_task("Deploy kubernetes cluster")
        assert results == []

    def test_empty_task(self, library_with_v2_skills):
        skill = _make_v2_skill(name="file-ops", use_when=["anything"])
        lib = library_with_v2_skills([skill])
        assert lib.find_skills_for_task("") == []

    def test_short_words_not_matched(self, library_with_v2_skills):
        """Words shorter than 3 chars are excluded from matching."""
        skill = _make_v2_skill(
            name="file-ops",
            use_when=["An agent needs to read a file"],
        )
        lib = library_with_v2_skills([skill])
        # "a" and "to" are < 3 chars, should not trigger match
        results = lib.find_skills_for_task("a to")
        assert results == []

    def test_multiple_skills_matched(self, library_with_v2_skills):
        s1 = _make_v2_skill(name="s1", use_when=["operations on files"])
        s2 = _make_v2_skill(name="s2", use_when=["batch file processing"])
        lib = library_with_v2_skills([s1, s2])
        results = lib.find_skills_for_task("Process batch files")
        assert len(results) == 2

    def test_inactive_skill_skipped(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="inactive",
            use_when=["anything matching"],
            status="inactive",
        )
        lib = library_with_v2_skills([skill])
        results = lib.find_skills_for_task("anything matching")
        assert results == []

    def test_skill_without_routing_skipped(self, library_with_v2_skills):
        skill = SkillDefinition(name="no-routing", agent="system")
        lib = library_with_v2_skills([skill])
        results = lib.find_skills_for_task("filesystem operations")
        assert results == []


# --- get_routing_exclusions tests ---


class TestGetRoutingExclusions:
    def test_matches_exclusion(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="file-ops",
            do_not_use_when=[
                RoutingExclusion(
                    condition="Single file read with known path",
                    instead="Read tool directly",
                    reason="Overhead unnecessary",
                )
            ],
        )
        lib = library_with_v2_skills([skill])
        results = lib.get_routing_exclusions("Read single file from path")
        assert len(results) == 1
        assert results[0]["skill"].name == "file-ops"
        assert results[0]["exclusion"].instead == "Read tool directly"

    def test_no_match(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="file-ops",
            do_not_use_when=[RoutingExclusion(condition="Content search by pattern")],
        )
        lib = library_with_v2_skills([skill])
        results = lib.get_routing_exclusions("Delete all temporary files")
        assert results == []

    def test_empty_task(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="x",
            do_not_use_when=[RoutingExclusion(condition="anything")],
        )
        lib = library_with_v2_skills([skill])
        assert lib.get_routing_exclusions("") == []


# --- get_skill_consensus tests ---


class TestGetSkillConsensus:
    def test_returns_capability_consensus(self, library_with_v2_skills):
        cap = SkillCapability(name="delete_file", consensus_required="unanimous")
        skill = _make_v2_skill(
            name="file-ops",
            capabilities=[cap],
            consensus_level="majority",
        )
        lib = library_with_v2_skills([skill])
        assert lib.get_skill_consensus("file-ops", "delete_file") == "unanimous"

    def test_falls_back_to_skill_level(self, library_with_v2_skills):
        cap = SkillCapability(name="read_file", consensus_required="any")
        skill = _make_v2_skill(
            name="file-ops",
            capabilities=[cap],
            consensus_level="majority",
        )
        lib = library_with_v2_skills([skill])
        # Capability not found — fall back to skill level
        assert lib.get_skill_consensus("file-ops", "nonexistent") == "majority"

    def test_skill_level_when_no_capability(self, library_with_v2_skills):
        skill = _make_v2_skill(name="file-ops", consensus_level="unanimous")
        lib = library_with_v2_skills([skill])
        assert lib.get_skill_consensus("file-ops") == "unanimous"

    def test_unknown_skill_returns_any(self, library_with_v2_skills):
        lib = library_with_v2_skills([])
        assert lib.get_skill_consensus("nonexistent") == "any"


# --- build_skill_context v2 tests ---


class TestBuildSkillContextV2:
    def test_includes_routing(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="file-ops",
            use_when=["Filesystem operations"],
            do_not_use_when=[
                RoutingExclusion(
                    condition="Single file read",
                    instead="Read tool",
                )
            ],
        )
        lib = library_with_v2_skills([skill])
        ctx = lib.build_skill_context("system")
        assert "When to use" in ctx
        assert "Filesystem operations" in ctx
        assert "When NOT to use" in ctx
        assert "Single file read" in ctx
        assert "Read tool" in ctx

    def test_includes_post_execution(self, library_with_v2_skills):
        cap = SkillCapability(
            name="delete",
            description="Delete a file",
            post_execution=["Verify file removed", "Check backup"],
        )
        skill = _make_v2_skill(name="file-ops", capabilities=[cap])
        lib = library_with_v2_skills([skill])
        ctx = lib.build_skill_context("system")
        assert "post-execution:" in ctx
        assert "Verify file removed" in ctx

    def test_includes_capability_inputs_outputs(self, library_with_v2_skills):
        cap = SkillCapability(
            name="read_file",
            description="Read a file",
            inputs={"path": {"type": "string", "required": True}},
            outputs={"content": {"type": "string"}},
        )
        skill = _make_v2_skill(name="file-ops", capabilities=[cap])
        lib = library_with_v2_skills([skill])
        ctx = lib.build_skill_context("system")
        assert "input: `path`: string (required)" in ctx
        assert "output: `content`: string" in ctx

    def test_includes_completion_checklist(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="file-ops",
            verification=SkillVerification(
                completion_checklist=["All ops succeeded", "Backups created"],
            ),
        )
        lib = library_with_v2_skills([skill])
        ctx = lib.build_skill_context("system")
        assert "Completion checklist" in ctx
        assert "All ops succeeded" in ctx

    def test_includes_error_escalation(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="file-ops",
            error_handling=SkillErrorHandling(
                escalation=[
                    EscalationRule(
                        error_class="recoverable",
                        action="retry",
                        max_retries=3,
                    ),
                    EscalationRule(
                        error_class="environment",
                        action="report",
                    ),
                ],
            ),
        )
        lib = library_with_v2_skills([skill])
        ctx = lib.build_skill_context("system")
        assert "Error escalation" in ctx
        assert "**recoverable**: retry (max 3 retries)" in ctx
        assert "**environment**: report" in ctx

    def test_empty_agent_returns_empty(self, library_with_v2_skills):
        lib = library_with_v2_skills([])
        assert lib.build_skill_context("nonexistent") == ""

    def test_v1_style_skill_still_renders(self, library_with_v2_skills):
        """v1 skills without v2 sections still render basic context."""
        cap = SkillCapability(name="search", description="Search the web")
        skill = SkillDefinition(
            name="web-search",
            agent="browser",
            capabilities=[cap],
            capability_names=["search"],
            protected_paths=["/etc"],
        )
        lib = library_with_v2_skills([skill])
        ctx = lib.build_skill_context("browser")
        assert "web-search" in ctx
        assert "search" in ctx
        assert "Protected paths" in ctx
        # No v2 sections
        assert "When to use" not in ctx
        assert "Completion checklist" not in ctx


# --- build_routing_summary tests ---


class TestBuildRoutingSummary:
    def test_includes_all_active_skills(self, library_with_v2_skills):
        s1 = _make_v2_skill(name="skill-a", use_when=["analysis"])
        s2 = _make_v2_skill(name="skill-b", use_when=["building"])
        lib = library_with_v2_skills([s1, s2])
        summary = lib.build_routing_summary()
        assert "skill-a" in summary
        assert "skill-b" in summary

    def test_includes_routing_info(self, library_with_v2_skills):
        skill = _make_v2_skill(
            name="file-ops",
            use_when=["Filesystem operations"],
            do_not_use_when=[RoutingExclusion(condition="Single file read")],
        )
        lib = library_with_v2_skills([skill])
        summary = lib.build_routing_summary()
        assert "Use when:" in summary
        assert "Filesystem operations" in summary
        assert "Do NOT use when:" in summary
        assert "Single file read" in summary

    def test_includes_capability_names(self, library_with_v2_skills):
        cap = SkillCapability(name="read_file", description="Read a file")
        skill = _make_v2_skill(name="file-ops", capabilities=[cap])
        lib = library_with_v2_skills([skill])
        summary = lib.build_routing_summary()
        assert "read_file" in summary

    def test_inactive_skills_excluded(self, library_with_v2_skills):
        skill = _make_v2_skill(name="inactive", status="inactive")
        lib = library_with_v2_skills([skill])
        summary = lib.build_routing_summary()
        assert "inactive" not in summary
