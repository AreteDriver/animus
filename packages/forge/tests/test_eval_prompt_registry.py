"""Tests for evaluation/prompt_registry.py — versioned prompt artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from animus_forge.evaluation.prompt_registry import (
    PromptRegistry,
    PromptVersion,
)


def test_register_auto_version_starts_at_v1(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    pv = registry.register("supervisor", "You are a supervisor.")
    assert pv.version == "v1"
    assert pv.name == "supervisor"
    assert pv.content_hash
    assert Path(pv.path).exists()


def test_register_auto_increments_version(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    registry.register("supervisor", "v1 content")
    pv2 = registry.register("supervisor", "v2 content")
    assert pv2.version == "v2"
    assert pv2.parent == "v1"


def test_register_explicit_version(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    pv = registry.register("supervisor", "content", version="v5")
    assert pv.version == "v5"


def test_register_collision_raises(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    registry.register("supervisor", "content", version="v1")
    with pytest.raises(FileExistsError):
        registry.register("supervisor", "other", version="v1")


def test_versions_sorted_numerically(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    for i in range(1, 12):
        registry.register("p", f"content v{i}", version=f"v{i}")
    # v10 should come after v9, not between v1 and v2
    versions = registry.versions("p")
    assert versions == [f"v{i}" for i in range(1, 12)]


def test_versions_missing_prompt_returns_empty(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    assert registry.versions("nope") == []


def test_load_latest(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    registry.register("p", "first")
    registry.register("p", "second")
    pv = registry.load("p", version="latest")
    assert pv.version == "v2"
    assert "second" in pv.content


def test_load_specific_version(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    registry.register("p", "first")
    registry.register("p", "second")
    pv = registry.load("p", version="v1")
    assert "first" in pv.content


def test_load_unknown_version_raises(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    registry.register("p", "content")
    with pytest.raises(FileNotFoundError, match="Version 'v99'"):
        registry.load("p", version="v99")


def test_load_unknown_prompt_raises(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="No versioned prompts"):
        registry.load("nope")


def test_load_preserves_frontmatter_metadata(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    registry.register(
        "supervisor",
        "body text",
        description="test prompt",
        tags=["role:supervisor", "phase:v1"],
    )
    pv = registry.load("supervisor")
    assert pv.description == "test prompt"
    assert "role:supervisor" in pv.tags
    assert pv.created_at is not None


def test_content_hash_stable_roundtrip(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    pv1 = registry.register("p", "the exact content")
    pv2 = registry.load("p")
    assert pv1.content_hash == pv2.content_hash


def test_content_hash_changes_with_content(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    pv1 = registry.register("p", "content a")
    pv2 = registry.register("p", "content b")
    assert pv1.content_hash != pv2.content_hash


def test_diff_between_versions(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    registry.register("p", "line one\nline two\n")
    registry.register("p", "line one\nline three\n")
    diff = registry.diff("p", "v1", "v2")
    assert "-line two" in diff
    assert "+line three" in diff


def test_list_prompts_summarizes(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    registry.register("supervisor", "a")
    registry.register("supervisor", "b")
    registry.register("editor", "c")
    summaries = {s["name"]: s for s in registry.list_prompts()}
    assert summaries["supervisor"]["count"] == 2
    assert summaries["supervisor"]["latest"] == "v2"
    assert summaries["editor"]["count"] == 1


def test_list_prompts_skips_reserved_dirs(tmp_path: Path):
    (tmp_path / "modes").mkdir()
    (tmp_path / "modes" / "v1.md").write_text("---\nname: ignore\nversion: v1\n---\nbody")
    registry = PromptRegistry(prompts_dir=tmp_path)
    summaries = registry.list_prompts()
    assert all(s["name"] != "modes" for s in summaries)


def test_list_prompts_skips_dirs_without_versions(tmp_path: Path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "README.md").write_text("not a version file")
    registry = PromptRegistry(prompts_dir=tmp_path)
    summaries = registry.list_prompts()
    assert not summaries


def test_list_prompts_empty_dir(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path)
    assert registry.list_prompts() == []


def test_list_prompts_missing_dir(tmp_path: Path):
    registry = PromptRegistry(prompts_dir=tmp_path / "nonexistent")
    assert registry.list_prompts() == []


def test_prompt_version_identifier():
    pv = PromptVersion(
        name="supervisor",
        version="v3",
        content="body",
        content_hash="abc",
    )
    assert pv.identifier == "supervisor@v3"
