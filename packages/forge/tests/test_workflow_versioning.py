"""Tests for workflow versioning functionality."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, "src")

from animus_forge.state.backends import SQLiteBackend
from animus_forge.state.migrations import run_migrations
from animus_forge.workflow.version_manager import WorkflowVersionManager
from animus_forge.workflow.versioning import (
    SemanticVersion,
    WorkflowVersion,
    compare_versions,
    compute_content_hash,
)

SAMPLE_WORKFLOW_V1 = """name: test-workflow
version: 1.0.0
description: Test workflow version 1

steps:
  - id: step1
    type: shell
    params:
      command: echo "Hello"
"""

SAMPLE_WORKFLOW_V2 = """name: test-workflow
version: 2.0.0
description: Test workflow version 2 with changes

steps:
  - id: step1
    type: shell
    params:
      command: echo "Hello World"
  - id: step2
    type: shell
    params:
      command: echo "New step"
"""


class TestSemanticVersion:
    """Tests for SemanticVersion class."""

    def test_parse_valid_version(self):
        """Can parse valid version string."""
        v = SemanticVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_zero_version(self):
        """Can parse version with zeros."""
        v = SemanticVersion.parse("0.0.1")
        assert v.major == 0
        assert v.minor == 0
        assert v.patch == 1

    def test_parse_large_numbers(self):
        """Can parse large version numbers."""
        v = SemanticVersion.parse("100.200.300")
        assert v.major == 100
        assert v.minor == 200
        assert v.patch == 300

    def test_parse_invalid_format_raises(self):
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError) as exc:
            SemanticVersion.parse("1.2")
        assert "Invalid version format" in str(exc.value)

    def test_parse_non_numeric_raises(self):
        """Non-numeric version raises ValueError."""
        with pytest.raises(ValueError):
            SemanticVersion.parse("1.2.x")

    def test_parse_with_whitespace(self):
        """Strips whitespace from version string."""
        v = SemanticVersion.parse("  1.2.3  ")
        assert str(v) == "1.2.3"

    def test_str_conversion(self):
        """Can convert to string."""
        v = SemanticVersion(1, 2, 3)
        assert str(v) == "1.2.3"

    def test_equality(self):
        """Version equality comparison."""
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 3)
        v3 = SemanticVersion(1, 2, 4)
        assert v1 == v2
        assert v1 != v3

    def test_less_than(self):
        """Version less-than comparison."""
        assert SemanticVersion(1, 0, 0) < SemanticVersion(2, 0, 0)
        assert SemanticVersion(1, 1, 0) < SemanticVersion(1, 2, 0)
        assert SemanticVersion(1, 1, 1) < SemanticVersion(1, 1, 2)

    def test_greater_than(self):
        """Version greater-than comparison."""
        assert SemanticVersion(2, 0, 0) > SemanticVersion(1, 0, 0)
        assert SemanticVersion(1, 2, 0) > SemanticVersion(1, 1, 0)
        assert SemanticVersion(1, 1, 2) > SemanticVersion(1, 1, 1)

    def test_less_than_or_equal(self):
        """Version less-than-or-equal comparison."""
        assert SemanticVersion(1, 0, 0) <= SemanticVersion(1, 0, 0)
        assert SemanticVersion(1, 0, 0) <= SemanticVersion(2, 0, 0)

    def test_greater_than_or_equal(self):
        """Version greater-than-or-equal comparison."""
        assert SemanticVersion(1, 0, 0) >= SemanticVersion(1, 0, 0)
        assert SemanticVersion(2, 0, 0) >= SemanticVersion(1, 0, 0)

    def test_hash(self):
        """Versions can be used in sets/dicts."""
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 3)
        assert hash(v1) == hash(v2)
        assert {v1} == {v2}

    def test_bump_patch(self):
        """Bump patch version."""
        v = SemanticVersion(1, 2, 3)
        bumped = v.bump_patch()
        assert str(bumped) == "1.2.4"

    def test_bump_minor(self):
        """Bump minor version resets patch."""
        v = SemanticVersion(1, 2, 3)
        bumped = v.bump_minor()
        assert str(bumped) == "1.3.0"

    def test_bump_major(self):
        """Bump major version resets minor and patch."""
        v = SemanticVersion(1, 2, 3)
        bumped = v.bump_major()
        assert str(bumped) == "2.0.0"

    def test_bump_by_type(self):
        """Bump by type string."""
        v = SemanticVersion(1, 2, 3)
        assert str(v.bump("patch")) == "1.2.4"
        assert str(v.bump("minor")) == "1.3.0"
        assert str(v.bump("major")) == "2.0.0"

    def test_bump_invalid_type_raises(self):
        """Invalid bump type raises ValueError."""
        v = SemanticVersion(1, 2, 3)
        with pytest.raises(ValueError) as exc:
            v.bump("invalid")
        assert "Invalid bump type" in str(exc.value)


class TestWorkflowVersion:
    """Tests for WorkflowVersion model."""

    def test_create_version(self):
        """Can create a WorkflowVersion."""
        wv = WorkflowVersion.create(
            workflow_name="test",
            version="1.0.0",
            content="name: test\nsteps: []",
            description="Test version",
            author="test-user",
        )
        assert wv.workflow_name == "test"
        assert wv.version == "1.0.0"
        assert wv.version_major == 1
        assert wv.version_minor == 0
        assert wv.version_patch == 0
        assert wv.description == "Test version"
        assert wv.author == "test-user"
        assert wv.content_hash is not None

    def test_create_with_semantic_version(self):
        """Can create with SemanticVersion object."""
        sem_ver = SemanticVersion(2, 1, 0)
        wv = WorkflowVersion.create(
            workflow_name="test",
            version=sem_ver,
            content="name: test",
        )
        assert wv.version == "2.1.0"
        assert wv.version_major == 2
        assert wv.version_minor == 1

    def test_get_semantic_version(self):
        """Can get SemanticVersion from WorkflowVersion."""
        wv = WorkflowVersion.create(
            workflow_name="test",
            version="3.2.1",
            content="name: test",
        )
        sem_ver = wv.get_semantic_version()
        assert sem_ver == SemanticVersion(3, 2, 1)


class TestContentHash:
    """Tests for content hash computation."""

    def test_same_content_same_hash(self):
        """Same content produces same hash."""
        content = "name: test\nversion: 1.0.0"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content produces different hash."""
        hash1 = compute_content_hash("content1")
        hash2 = compute_content_hash("content2")
        assert hash1 != hash2

    def test_whitespace_normalized(self):
        """Leading/trailing whitespace is normalized."""
        hash1 = compute_content_hash("content")
        hash2 = compute_content_hash("  content  ")
        assert hash1 == hash2


class TestCompareVersions:
    """Tests for version diff comparison."""

    def test_no_changes(self):
        """No changes detected for identical content."""
        diff = compare_versions(SAMPLE_WORKFLOW_V1, SAMPLE_WORKFLOW_V1)
        assert not diff.has_changes
        assert diff.added_lines == 0
        assert diff.removed_lines == 0

    def test_changes_detected(self):
        """Changes detected for different content."""
        diff = compare_versions(SAMPLE_WORKFLOW_V1, SAMPLE_WORKFLOW_V2)
        assert diff.has_changes
        assert diff.added_lines > 0

    def test_unified_diff_generated(self):
        """Unified diff is generated."""
        diff = compare_versions(SAMPLE_WORKFLOW_V1, SAMPLE_WORKFLOW_V2)
        assert "+" in diff.unified_diff or "-" in diff.unified_diff

    def test_changed_sections_detected(self):
        """Changed YAML sections are identified."""
        diff = compare_versions(SAMPLE_WORKFLOW_V1, SAMPLE_WORKFLOW_V2)
        assert len(diff.changed_sections) > 0


class TestWorkflowVersionManager:
    """Tests for WorkflowVersionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a manager with temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)
            run_migrations(backend)
            manager = WorkflowVersionManager(backend=backend)
            yield manager
            backend.close()

    def test_save_version(self, manager):
        """Can save a workflow version."""
        wv = manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
            description="Initial version",
        )
        assert wv.workflow_name == "test"
        assert wv.version == "1.0.0"
        assert wv.is_active

    def test_save_version_auto_bump(self, manager):
        """Version auto-bumps from latest."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        # Add different content without explicit version
        wv2 = manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
        )
        assert wv2.version == "1.0.1"  # Auto-bumped patch

    def test_save_version_auto_bump_minor(self, manager):
        """Version auto-bumps minor when specified."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        wv2 = manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            auto_bump="minor",
        )
        assert wv2.version == "1.1.0"

    def test_save_version_duplicate_content_skipped(self, manager):
        """Duplicate content returns existing version."""
        wv1 = manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        # Same content
        wv2 = manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
        )

        assert wv2.version == wv1.version

    def test_save_version_existing_version_raises(self, manager):
        """Saving existing version raises ValueError."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        with pytest.raises(ValueError) as exc:
            manager.save_version(
                workflow_name="test",
                content=SAMPLE_WORKFLOW_V2,
                version="1.0.0",
            )
        assert "already exists" in str(exc.value)

    def test_save_version_invalid_yaml_raises(self, manager):
        """Invalid YAML content raises ValueError."""
        with pytest.raises(ValueError) as exc:
            manager.save_version(
                workflow_name="test",
                content="invalid: yaml: content:",
            )
        assert "Invalid YAML" in str(exc.value)

    def test_get_version(self, manager):
        """Can retrieve a specific version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        wv = manager.get_version("test", "1.0.0")
        assert wv is not None
        assert wv.version == "1.0.0"

    def test_get_version_not_found(self, manager):
        """Returns None for non-existent version."""
        wv = manager.get_version("nonexistent", "1.0.0")
        assert wv is None

    def test_get_active_version(self, manager):
        """Can get currently active version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        active = manager.get_active_version("test")
        assert active is not None
        assert active.version == "2.0.0"  # Latest saved becomes active

    def test_get_latest_version(self, manager):
        """Can get latest (highest) version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="1.1.0",
            activate=False,
        )

        latest = manager.get_latest_version("test")
        assert latest is not None
        assert latest.version == "1.1.0"

    def test_list_versions(self, manager):
        """Can list all versions of a workflow."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        versions = manager.list_versions("test")
        assert len(versions) == 2
        # Should be sorted by version descending
        assert versions[0].version == "2.0.0"
        assert versions[1].version == "1.0.0"

    def test_list_versions_with_pagination(self, manager):
        """List versions supports pagination."""
        for i in range(5):
            manager.save_version(
                workflow_name="test",
                content=f"name: test\nversion: 1.0.{i}",
                version=f"1.0.{i}",
            )

        page1 = manager.list_versions("test", limit=2, offset=0)
        page2 = manager.list_versions("test", limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        # Page 1 should have newest, page 2 older
        assert page1[0].version_patch > page2[0].version_patch

    def test_set_active(self, manager):
        """Can activate a specific version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        manager.set_active("test", "1.0.0")

        active = manager.get_active_version("test")
        assert active.version == "1.0.0"

    def test_set_active_nonexistent_raises(self, manager):
        """Activating non-existent version raises ValueError."""
        with pytest.raises(ValueError) as exc:
            manager.set_active("test", "99.0.0")
        assert "doesn't exist" in str(exc.value)

    def test_rollback(self, manager):
        """Can rollback to previous version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        rolled_back = manager.rollback("test")

        assert rolled_back is not None
        assert rolled_back.version == "1.0.0"
        assert manager.get_active_version("test").version == "1.0.0"

    def test_rollback_no_previous_version(self, manager):
        """Rollback returns None when no previous version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        rolled_back = manager.rollback("test")
        # No previous version, stays at 1.0.0
        assert rolled_back is None or rolled_back.version == "1.0.0"

    def test_rollback_to_specific_version(self, manager):
        """Can rollback to a specific version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        rolled_back = manager.rollback_to("test", "1.0.0")

        assert rolled_back.version == "1.0.0"

    def test_compare_versions(self, manager):
        """Can compare two versions."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        diff = manager.compare_versions("test", "1.0.0", "2.0.0")

        assert diff.from_version == "1.0.0"
        assert diff.to_version == "2.0.0"
        assert diff.has_changes

    def test_compare_versions_nonexistent_raises(self, manager):
        """Comparing non-existent version raises ValueError."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        with pytest.raises(ValueError):
            manager.compare_versions("test", "1.0.0", "99.0.0")

    def test_get_unified_diff(self, manager):
        """Can get unified diff between versions."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        diff = manager.get_unified_diff("test", "1.0.0", "2.0.0")
        assert isinstance(diff, str)
        assert "+" in diff or "-" in diff

    def test_delete_version(self, manager):
        """Can delete a non-active version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        result = manager.delete_version("test", "1.0.0")
        assert result is True

        wv = manager.get_version("test", "1.0.0")
        assert wv is None

    def test_delete_active_version_raises(self, manager):
        """Cannot delete the active version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        with pytest.raises(ValueError) as exc:
            manager.delete_version("test", "1.0.0")
        assert "Cannot delete active version" in str(exc.value)

    def test_list_workflows(self, manager):
        """Can list all workflows with version info."""
        manager.save_version(
            workflow_name="workflow1",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="workflow2",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        workflows = manager.list_workflows()

        assert len(workflows) == 2
        names = [w["workflow_name"] for w in workflows]
        assert "workflow1" in names
        assert "workflow2" in names


class TestWorkflowVersionManagerFileOps:
    """Tests for file import/export operations."""

    @pytest.fixture
    def manager(self):
        """Create a manager with temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)
            run_migrations(backend)
            manager = WorkflowVersionManager(backend=backend)
            yield manager
            backend.close()

    def test_import_from_file(self, manager):
        """Can import workflow from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(SAMPLE_WORKFLOW_V1)
            f.flush()

            try:
                wv = manager.import_from_file(f.name)

                assert wv.workflow_name == "test-workflow"
                assert wv.version == "1.0.0"
            finally:
                os.unlink(f.name)

    def test_import_nonexistent_file_raises(self, manager):
        """Importing non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            manager.import_from_file("/nonexistent/file.yaml")

    def test_import_invalid_yaml_raises(self, manager):
        """Importing invalid YAML raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            f.flush()

            try:
                with pytest.raises(ValueError) as exc:
                    manager.import_from_file(f.name)
                assert "Invalid YAML" in str(exc.value)
            finally:
                os.unlink(f.name)

    def test_export_to_file(self, manager):
        """Can export workflow to file."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "exported.yaml")
            result_path = manager.export_to_file("test", file_path=export_path)

            assert os.path.exists(result_path)
            with open(result_path) as f:
                content = f.read()
            assert "test-workflow" in content

    def test_export_specific_version(self, manager):
        """Can export a specific version."""
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )
        manager.save_version(
            workflow_name="test",
            content=SAMPLE_WORKFLOW_V2,
            version="2.0.0",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "exported.yaml")
            manager.export_to_file("test", version="1.0.0", file_path=export_path)

            with open(export_path) as f:
                content = f.read()
            assert "version: 1.0.0" in content

    def test_migrate_existing_workflows(self, manager):
        """Can migrate existing workflow files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a workflow file
            wf_path = os.path.join(tmpdir, "test.yaml")
            with open(wf_path, "w") as f:
                f.write(SAMPLE_WORKFLOW_V1)

            migrated = manager.migrate_existing_workflows(tmpdir)

            assert len(migrated) == 1
            assert migrated[0].workflow_name == "test-workflow"

    def test_migrate_skips_already_versioned(self, manager):
        """Migration skips workflows that already have versions."""
        # Pre-save a version
        manager.save_version(
            workflow_name="test-workflow",
            content=SAMPLE_WORKFLOW_V1,
            version="1.0.0",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            wf_path = os.path.join(tmpdir, "test.yaml")
            with open(wf_path, "w") as f:
                f.write(SAMPLE_WORKFLOW_V1)

            migrated = manager.migrate_existing_workflows(tmpdir)

            # Should skip since already has versions
            assert len(migrated) == 0
