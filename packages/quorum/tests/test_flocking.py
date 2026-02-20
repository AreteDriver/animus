"""Tests for convergent.flocking swarm coordination."""

from __future__ import annotations

from convergent.flocking import FlockingCoordinator, _extract_keywords
from convergent.stigmergy import StigmergyField


def _field() -> StigmergyField:
    """Create an in-memory StigmergyField for testing."""
    return StigmergyField(":memory:")


class TestCheckAlignment:
    def test_returns_pattern_markers(self) -> None:
        field = _field()
        field.leave_marker(
            "agent-1", "pattern_found", "src/api.py", "Use repository pattern for DB"
        )
        coord = FlockingCoordinator(field)
        patterns = coord.check_alignment("agent-2", ["src/api.py"])
        assert len(patterns) == 1
        assert "repository pattern" in patterns[0]

    def test_excludes_own_markers(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "pattern_found", "src/api.py", "Use snake_case")
        coord = FlockingCoordinator(field)
        patterns = coord.check_alignment("agent-1", ["src/api.py"])
        assert len(patterns) == 0

    def test_excludes_non_pattern_markers(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "file_modified", "src/api.py", "Changed endpoint")
        field.leave_marker("agent-1", "known_issue", "src/api.py", "Race condition")
        coord = FlockingCoordinator(field)
        patterns = coord.check_alignment("agent-2", ["src/api.py"])
        assert len(patterns) == 0

    def test_deduplicates_same_content(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "pattern_found", "a.py", "Use type hints")
        field.leave_marker("agent-2", "pattern_found", "a.py", "Use type hints")
        coord = FlockingCoordinator(field)
        patterns = coord.check_alignment("agent-3", ["a.py"])
        assert len(patterns) == 1

    def test_multiple_files(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "pattern_found", "a.py", "Pattern A")
        field.leave_marker("agent-1", "pattern_found", "b.py", "Pattern B")
        coord = FlockingCoordinator(field)
        patterns = coord.check_alignment("agent-2", ["a.py", "b.py"])
        assert len(patterns) == 2

    def test_empty_files_returns_empty(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        assert coord.check_alignment("agent-1", []) == []

    def test_no_markers_returns_empty(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        assert coord.check_alignment("agent-1", ["src/api.py"]) == []


class TestCheckCohesion:
    def test_identical_text_zero_drift(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        drift = coord.check_cohesion(
            "Implement user authentication",
            "Implement user authentication",
        )
        assert drift == 0.0

    def test_completely_different_high_drift(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        drift = coord.check_cohesion(
            "Implement user authentication login endpoint",
            "Refactor database migration schema tooling",
        )
        assert drift > 0.7

    def test_partial_overlap_moderate_drift(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        drift = coord.check_cohesion(
            "Add authentication to the API endpoint",
            "Fix authentication bug in the API response",
        )
        assert 0.0 < drift < 0.8

    def test_empty_task_zero_drift(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        assert coord.check_cohesion("", "working on stuff") == 0.0

    def test_empty_work_zero_drift(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        assert coord.check_cohesion("implement auth", "") == 0.0

    def test_stop_words_ignored(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        # These differ only in stop words — keywords are the same
        drift = coord.check_cohesion(
            "the authentication system",
            "an authentication system",
        )
        assert drift == 0.0


class TestCheckSeparation:
    def test_detects_file_conflict(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "file_modified", "src/auth.py", "Changed login", strength=1.0)
        coord = FlockingCoordinator(field)
        conflicts = coord.check_separation("agent-2", ["src/auth.py"])
        assert conflicts == ["src/auth.py"]

    def test_ignores_own_markers(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "file_modified", "src/auth.py", "Changed login", strength=1.0)
        coord = FlockingCoordinator(field)
        conflicts = coord.check_separation("agent-1", ["src/auth.py"])
        assert conflicts == []

    def test_ignores_weak_markers(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "file_modified", "src/auth.py", "Old change", strength=0.1)
        coord = FlockingCoordinator(field, separation_threshold=0.3)
        conflicts = coord.check_separation("agent-2", ["src/auth.py"])
        assert conflicts == []

    def test_ignores_non_file_modified_markers(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "known_issue", "src/auth.py", "Bug here", strength=1.0)
        coord = FlockingCoordinator(field)
        conflicts = coord.check_separation("agent-2", ["src/auth.py"])
        assert conflicts == []

    def test_one_conflict_per_file(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "file_modified", "a.py", "c1", strength=1.0)
        field.leave_marker("agent-2", "file_modified", "a.py", "c2", strength=1.0)
        coord = FlockingCoordinator(field)
        conflicts = coord.check_separation("agent-3", ["a.py"])
        assert conflicts == ["a.py"]  # Only listed once

    def test_multiple_files_some_conflicting(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "file_modified", "a.py", "c1", strength=1.0)
        coord = FlockingCoordinator(field)
        conflicts = coord.check_separation("agent-2", ["a.py", "b.py", "c.py"])
        assert conflicts == ["a.py"]

    def test_no_conflicts_returns_empty(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        assert coord.check_separation("agent-1", ["a.py"]) == []


class TestGenerateConstraints:
    def test_combines_all_three_rules(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "pattern_found", "a.py", "Use dataclasses")
        field.leave_marker("agent-1", "file_modified", "b.py", "Editing B", strength=1.0)
        coord = FlockingCoordinator(field)
        result = coord.generate_constraints(
            agent_id="agent-2",
            task_description="implement auth login",
            current_work="refactoring database migration tooling",
            file_paths=["a.py", "b.py"],
        )
        assert "Alignment" in result
        assert "Use dataclasses" in result
        assert "Cohesion Warning" in result
        assert "Separation" in result
        assert "`b.py`" in result

    def test_empty_when_no_constraints(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        result = coord.generate_constraints(
            agent_id="agent-1",
            task_description="implement auth",
            current_work="implement auth",
            file_paths=["a.py"],
        )
        assert result == ""

    def test_alignment_only(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "pattern_found", "a.py", "Use snake_case")
        coord = FlockingCoordinator(field)
        result = coord.generate_constraints(
            agent_id="agent-2",
            task_description="implement auth",
            current_work="implement auth",
            file_paths=["a.py"],
        )
        assert "Alignment" in result
        assert "Cohesion" not in result
        assert "Separation" not in result

    def test_separation_only(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "file_modified", "a.py", "Editing", strength=1.0)
        coord = FlockingCoordinator(field)
        result = coord.generate_constraints(
            agent_id="agent-2",
            task_description="implement auth",
            current_work="implement auth",
            file_paths=["a.py"],
        )
        assert "Separation" in result
        assert "Alignment" not in result

    def test_cohesion_only_when_drifting(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        result = coord.generate_constraints(
            agent_id="agent-1",
            task_description="implement authentication login endpoint",
            current_work="refactoring database migration schema tooling",
            file_paths=[],
        )
        assert "Cohesion Warning" in result
        assert "drift=" in result

    def test_no_cohesion_when_on_task(self) -> None:
        field = _field()
        coord = FlockingCoordinator(field)
        result = coord.generate_constraints(
            agent_id="agent-1",
            task_description="implement auth",
            current_work="implement auth",
            file_paths=[],
        )
        assert result == ""

    def test_header_present(self) -> None:
        field = _field()
        field.leave_marker("agent-1", "pattern_found", "a.py", "Pattern")
        coord = FlockingCoordinator(field)
        result = coord.generate_constraints(
            agent_id="agent-2",
            task_description="task",
            current_work="task",
            file_paths=["a.py"],
        )
        assert "## Flocking Constraints" in result


class TestExtractKeywords:
    def test_basic_extraction(self) -> None:
        keywords = _extract_keywords("Implement user authentication")
        assert "implement" in keywords
        assert "user" in keywords
        assert "authentication" in keywords

    def test_stop_words_removed(self) -> None:
        keywords = _extract_keywords("the quick brown fox is very fast")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "very" not in keywords
        assert "quick" in keywords

    def test_short_words_removed(self) -> None:
        keywords = _extract_keywords("go to db")
        # "go", "to", "db" are all <= 2 chars
        assert keywords == []

    def test_empty_string(self) -> None:
        assert _extract_keywords("") == []

    def test_case_insensitive(self) -> None:
        keywords = _extract_keywords("Auth LOGIN System")
        assert "auth" in keywords
        assert "login" in keywords
        assert "system" in keywords

    def test_underscored_identifiers(self) -> None:
        keywords = _extract_keywords("use snake_case for function_names")
        assert "snake_case" in keywords
        assert "function_names" in keywords


class TestPublicAPI:
    def test_import_from_convergent(self) -> None:
        import convergent

        assert hasattr(convergent, "FlockingCoordinator")

    def test_all_exports_listed(self) -> None:
        import convergent

        assert "FlockingCoordinator" in convergent.__all__
