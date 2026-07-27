"""Tests for the WorkspaceManager (Git worktree isolation)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from animus_forge.workspace import WorkspaceManager


@pytest.fixture()
def temp_workspace_dir():
    """Create a temporary directory for workspaces."""
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture()
def fake_repo():
    """Create a fake Git repository for worktree tests."""
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        # Initialise Git repo
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        # Create initial commit
        (repo / "README.md").write_text("# test")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        yield repo


class TestWorkspaceManager:
    def test_create_workspace(self, temp_workspace_dir, fake_repo):
        wm = WorkspaceManager(base_dir=temp_workspace_dir)
        mid = uuid4()
        manifest = wm.create(
            mission_id=mid,
            repo_path=str(fake_repo),
            allowed_paths=["src/**", "tests/**"],
            protected_paths=[".github/**"],
        )
        assert manifest.mission_id == str(mid)
        assert manifest.branch.startswith("animus/mission-")
        assert Path(manifest.worktree_path).exists()
        assert (Path(manifest.worktree_path) / "README.md").exists()

    def test_manifest_roundtrip(self, temp_workspace_dir, fake_repo):
        wm = WorkspaceManager(base_dir=temp_workspace_dir)
        mid = uuid4()
        wm.create(
            mission_id=mid,
            repo_path=str(fake_repo),
            allowed_paths=["src/**"],
            protected_paths=[".github/**"],
        )
        fetched = wm.get_manifest(mid)
        assert fetched is not None
        assert fetched.allowed_paths == ["src/**"]
        assert fetched.protected_paths == [".github/**"]

    def test_path_allowed(self, temp_workspace_dir, fake_repo):
        wm = WorkspaceManager(base_dir=temp_workspace_dir)
        mid = uuid4()
        wm.create(
            mission_id=mid,
            repo_path=str(fake_repo),
            allowed_paths=["src/**"],
            protected_paths=[".github/**"],
        )
        assert wm.is_path_allowed(mid, "src/main.py") is True
        assert wm.is_path_allowed(mid, ".github/workflows/ci.yml") is False
        assert wm.is_path_allowed(mid, "docs/readme.md") is False

    def test_empty_allowlist_allows_non_protected(self, temp_workspace_dir, fake_repo):
        wm = WorkspaceManager(base_dir=temp_workspace_dir)
        mid = uuid4()
        wm.create(
            mission_id=mid,
            repo_path=str(fake_repo),
            allowed_paths=[],
            protected_paths=["migrations/**"],
        )
        assert wm.is_path_allowed(mid, "src/main.py") is True
        assert wm.is_path_allowed(mid, "migrations/001.sql") is False

    def test_destroy_workspace(self, temp_workspace_dir, fake_repo):
        wm = WorkspaceManager(base_dir=temp_workspace_dir)
        mid = uuid4()
        manifest = wm.create(
            mission_id=mid,
            repo_path=str(fake_repo),
        )
        worktree = Path(manifest.worktree_path)
        assert worktree.exists()
        wm.destroy(mid)
        assert not worktree.exists()

    def test_destroy_missing_is_noop(self, temp_workspace_dir):
        wm = WorkspaceManager(base_dir=temp_workspace_dir)
        mid = uuid4()
        # Should not raise
        wm.destroy(mid)

    def test_manifest_missing(self, temp_workspace_dir):
        wm = WorkspaceManager(base_dir=temp_workspace_dir)
        assert wm.get_manifest(uuid4()) is None

    def test_match_pattern(self):
        wm = WorkspaceManager
        assert wm._match("src/main.py", "src/**") is True
        assert wm._match("src/main.py", "src/*.py") is True
        assert wm._match("src/main.py", "tests/**") is False
        assert wm._match(".github/workflows/ci.yml", ".github/**") is True
