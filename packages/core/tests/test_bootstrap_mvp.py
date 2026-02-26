"""Tests for the Bootstrap MVP — the self-improvement loop.

Tests the three-layer integration:
  Core (identity + memory + cognitive)
    → Forge (workflow execution)
      → Quorum (two-agent consensus)

All tests use MockModel — no external LLM or services needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from animus.cognitive import CognitiveLayer, ModelConfig
from animus.identity import AnimusIdentity
from animus.memory import MemoryLayer, MemoryType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def codebase_root(tmp_path: Path) -> Path:
    """Create a fake codebase structure for identity to inspect."""
    root = tmp_path / "animus"
    root.mkdir()

    # Create CLAUDE.md so identity detection works
    (root / "CLAUDE.md").write_text("# CLAUDE.md\n")

    # Create package structure
    core_dir = root / "packages" / "core" / "animus"
    core_dir.mkdir(parents=True)
    (core_dir / "__init__.py").write_text('__version__ = "2.0.0"\n')
    (core_dir / "identity.py").write_text(
        '"""Identity module."""\n\nclass AnimusIdentity:\n    name = "Animus"\n'
    )
    (core_dir / "memory.py").write_text('"""Memory module."""\n\nclass MemoryLayer:\n    pass\n')
    (core_dir / "cognitive.py").write_text(
        '"""Cognitive module."""\n\nclass CognitiveLayer:\n    pass\n'
    )

    # Create other package dirs
    for pkg in [
        "forge/src/animus_forge",
        "quorum/python/convergent",
        "bootstrap/src/animus_bootstrap",
    ]:
        d = root / "packages" / pkg
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("")

    return root


@pytest.fixture
def identity(codebase_root: Path) -> AnimusIdentity:
    """AnimusIdentity pointed at the fake codebase."""
    return AnimusIdentity(codebase_root=str(codebase_root))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Temp directory for memory/checkpoint storage."""
    d = tmp_path / "animus_data"
    d.mkdir()
    return d


@pytest.fixture
def mock_cognitive() -> CognitiveLayer:
    """CognitiveLayer with mock responses shaped for the self-review workflow.

    Uses ## headings so ForgeAgent._parse_outputs() can extract structured outputs.
    """
    response = (
        "## analysis\n"
        "The code implements a clean identity system with file I/O.\n\n"
        "## suggestions\n"
        "1. Add input validation for file paths\n"
        "2. Add type hints to all methods\n\n"
        "## approved_improvements\n"
        "1. Add input validation\n"
        "2. Add type hints\n\n"
        "## score\n"
        "0.85\n"
    )
    return CognitiveLayer(
        ModelConfig.mock(
            default_response=response,
            response_map={},
        )
    )


# ===========================================================================
# Identity Tests
# ===========================================================================


class TestAnimusIdentity:
    """Tests for the identity system — Animus knowing itself."""

    def test_identity_creation_defaults(self):
        """Identity has sensible defaults."""
        ident = AnimusIdentity()
        assert ident.name == "Animus"
        assert ident.version == "2.0.0"
        assert "memory_persistence" in ident.capabilities
        assert "self_reflection" in ident.capabilities
        assert ident.reflection_count == 0
        assert ident.created_at  # Should be auto-populated

    def test_identity_custom_codebase(self, codebase_root: Path):
        """Identity accepts a custom codebase root."""
        ident = AnimusIdentity(codebase_root=str(codebase_root))
        assert ident.root == codebase_root
        assert ident.codebase_root == str(codebase_root)

    def test_identity_package_paths(self, identity: AnimusIdentity, codebase_root: Path):
        """Package paths resolve correctly."""
        core_path = identity.package_path("core")
        assert core_path == codebase_root / "packages" / "core" / "animus"
        assert core_path.exists()

    def test_identity_unknown_package(self, identity: AnimusIdentity):
        """Unknown package raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            identity.package_path("nonexistent")

    def test_read_own_file(self, identity: AnimusIdentity):
        """Animus can read its own source files."""
        content = identity.read_own_file("packages/core/animus/__init__.py")
        assert '__version__ = "2.0.0"' in content

    def test_read_own_file_not_found(self, identity: AnimusIdentity):
        """Reading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            identity.read_own_file("packages/core/animus/nonexistent.py")

    def test_list_own_files(self, identity: AnimusIdentity):
        """Can list Python files in a package."""
        files = identity.list_own_files("core")
        assert len(files) >= 3
        rel_paths = [f.replace("\\", "/") for f in files]  # Normalize for Windows
        assert any("identity.py" in f for f in rel_paths)
        assert any("memory.py" in f for f in rel_paths)
        assert any("cognitive.py" in f for f in rel_paths)

    def test_list_own_files_empty_package(self, identity: AnimusIdentity):
        """Listing files for a package with no .py files returns near-empty."""
        files = identity.list_own_files("forge")
        # forge has just __init__.py
        assert len(files) >= 1

    def test_record_reflection(self, identity: AnimusIdentity):
        """Recording a reflection updates identity state."""
        assert identity.reflection_count == 0
        identity.record_reflection(
            summary="First self-review of identity module",
            improvements=["Add docstrings", "Better error messages"],
        )
        assert identity.reflection_count == 1
        assert identity.last_reflection
        assert len(identity.improvement_log) == 1
        assert identity.improvement_log[0]["cycle"] == 1
        assert "Add docstrings" in identity.improvement_log[0]["improvements"]

    def test_multiple_reflections(self, identity: AnimusIdentity):
        """Multiple reflections increment correctly."""
        for i in range(5):
            identity.record_reflection(f"Reflection #{i + 1}")
        assert identity.reflection_count == 5
        assert len(identity.improvement_log) == 5

    def test_save_and_load(self, identity: AnimusIdentity, tmp_path: Path):
        """Identity can persist and reload."""
        identity.record_reflection("Test reflection")
        save_path = tmp_path / "identity.json"
        identity.save(save_path)

        assert save_path.exists()

        loaded = AnimusIdentity.load(save_path)
        assert loaded.name == identity.name
        assert loaded.version == identity.version
        assert loaded.reflection_count == 1
        assert len(loaded.improvement_log) == 1

    def test_save_default_location(self, identity: AnimusIdentity):
        """Save to default location creates .animus/identity.json."""
        path = identity.save()
        assert path.exists()
        assert path.name == "identity.json"
        assert path.parent.name == ".animus"

    def test_load_nonexistent(self, tmp_path: Path):
        """Loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AnimusIdentity.load(tmp_path / "nope.json")

    def test_to_dict_roundtrip(self, identity: AnimusIdentity):
        """to_dict produces valid serializable data."""
        identity.record_reflection("Test")
        d = identity.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "Animus"
        assert d["reflection_count"] == 1
        # Verify JSON-serializable
        json_str = json.dumps(d)
        assert json.loads(json_str) == d

    def test_repr(self, identity: AnimusIdentity):
        """repr is informative."""
        r = repr(identity)
        assert "Animus" in r
        assert "2.0.0" in r


# ===========================================================================
# Self-Review Workflow Tests (Forge Integration)
# ===========================================================================


class TestSelfReviewWorkflow:
    """Tests for the Forge-based self-review workflow."""

    def test_build_workflow_config(self, codebase_root: Path):
        """Workflow config is valid."""
        from animus.bootstrap_loop import build_self_review_workflow

        wf = build_self_review_workflow(
            files_content="def hello(): pass",
            provider="mock",
            model="mock",
        )
        assert wf.name == "self-review"
        assert len(wf.agents) == 2
        assert wf.agents[0].name == "analyst"
        assert wf.agents[1].name == "reviewer"
        assert "analyst.analysis" in wf.agents[1].inputs
        assert wf.max_cost_usd == 0.10

    def test_workflow_has_gate(self):
        """Workflow includes a quality gate after reviewer."""
        from animus.bootstrap_loop import build_self_review_workflow

        wf = build_self_review_workflow("code here")
        assert len(wf.gates) == 1
        assert wf.gates[0].name == "quality-check"
        assert wf.gates[0].after == "reviewer"
        assert "reviewer.score" in wf.gates[0].pass_condition

    def test_workflow_execution_with_mock(self, mock_cognitive: CognitiveLayer, data_dir: Path):
        """Forge engine can execute the self-review workflow end-to-end."""
        from animus.bootstrap_loop import build_self_review_workflow
        from animus.forge.engine import ForgeEngine

        engine = ForgeEngine(
            cognitive=mock_cognitive,
            checkpoint_dir=data_dir / "checkpoints",
        )

        wf = build_self_review_workflow(
            files_content="class AnimusIdentity:\n    name = 'Animus'\n",
            provider="mock",
            model="mock",
        )

        state = engine.run(wf)
        assert state.status == "completed"
        assert len(state.results) == 2
        assert state.results[0].agent_name == "analyst"
        assert state.results[1].agent_name == "reviewer"
        assert state.results[0].success
        assert state.results[1].success


# ===========================================================================
# Consensus Tests (Quorum Integration)
# ===========================================================================


class TestConsensus:
    """Tests for the two-agent consensus check."""

    def test_both_approve(self):
        """Both agents approving yields approved consensus."""
        from animus.bootstrap_loop import run_consensus

        result = run_consensus(
            question="Accept improvements?",
            context="Analysis looks good.",
            agent_a_vote="approve",
            agent_a_confidence=0.9,
            agent_b_vote="approve",
            agent_b_confidence=0.8,
        )
        assert result.approved is True
        assert result.approve_weight > 0
        assert result.reject_weight == 0.0

    def test_both_reject(self):
        """Both agents rejecting yields rejected consensus."""
        from animus.bootstrap_loop import run_consensus

        result = run_consensus(
            question="Accept bad improvements?",
            context="Analysis has flaws.",
            agent_a_vote="reject",
            agent_a_confidence=0.9,
            agent_b_vote="reject",
            agent_b_confidence=0.8,
        )
        assert result.approved is False
        assert result.reject_weight > 0

    def test_split_vote_higher_confidence_wins(self):
        """Split vote resolves by weighted score."""
        from animus.bootstrap_loop import run_consensus

        result = run_consensus(
            question="Accept contested improvements?",
            context="Mixed quality.",
            agent_a_vote="approve",
            agent_a_confidence=0.9,
            agent_b_vote="reject",
            agent_b_confidence=0.3,
        )
        # Agent A has higher weighted score (0.5 * 0.9 = 0.45 vs 0.5 * 0.3 = 0.15)
        assert result.approved is True

    def test_consensus_has_reasoning(self):
        """Consensus result includes agent reasoning."""
        from animus.bootstrap_loop import run_consensus

        result = run_consensus(
            question="Test reasoning?",
            context="Context.",
            agent_a_reasoning="Code quality is excellent.",
            agent_b_reasoning="Tests all pass.",
        )
        assert len(result.reasoning) >= 2
        assert any("excellent" in r for r in result.reasoning)
        assert any("pass" in r for r in result.reasoning)

    def test_consensus_summary(self):
        """Consensus result has a readable summary."""
        from animus.bootstrap_loop import run_consensus

        result = run_consensus(
            question="Summary test?",
            context="Context.",
        )
        summary = result.summary
        assert "APPROVED" in summary or "REJECTED" in summary
        assert "approve=" in summary

    def test_local_fallback(self):
        """Local fallback works when convergent isn't available."""
        from animus.bootstrap_loop import _local_consensus_fallback

        result = _local_consensus_fallback(
            vote_a="approve",
            confidence_a=0.9,
            vote_b="approve",
            confidence_b=0.8,
        )
        assert result.approved is True
        assert result.approve_weight > 0

    def test_local_fallback_rejection(self):
        """Local fallback correctly rejects."""
        from animus.bootstrap_loop import _local_consensus_fallback

        result = _local_consensus_fallback(
            vote_a="reject",
            confidence_a=0.9,
            vote_b="reject",
            confidence_b=0.8,
        )
        assert result.approved is False


# ===========================================================================
# Bootstrap Loop Integration Tests
# ===========================================================================


class TestBootstrapLoop:
    """End-to-end tests for the full bootstrap loop."""

    def test_full_cycle(
        self,
        identity: AnimusIdentity,
        mock_cognitive: CognitiveLayer,
        data_dir: Path,
    ):
        """Complete bootstrap cycle: read → analyze → consensus → write."""
        from animus.bootstrap_loop import BootstrapLoop

        memory = MemoryLayer(data_dir=data_dir, backend="json")
        loop = BootstrapLoop(
            identity=identity,
            cognitive=mock_cognitive,
            memory=memory,
            data_dir=data_dir,
        )

        result = loop.run_cycle(
            files=["packages/core/animus/__init__.py"],
        )

        # Workflow completed
        assert result.cycle == 1
        assert len(result.files_reviewed) == 1
        assert result.analysis  # Non-empty analysis
        assert result.suggestions  # Non-empty suggestions

        # Consensus reached
        assert result.consensus is not None
        assert result.consensus.approved is True

        # Identity updated
        assert identity.reflection_count == 1
        assert result.improvements_written is True

    def test_cycle_increments(
        self,
        identity: AnimusIdentity,
        mock_cognitive: CognitiveLayer,
        data_dir: Path,
    ):
        """Multiple cycles increment the counter."""
        from animus.bootstrap_loop import BootstrapLoop

        memory = MemoryLayer(data_dir=data_dir, backend="json")
        loop = BootstrapLoop(
            identity=identity,
            cognitive=mock_cognitive,
            memory=memory,
            data_dir=data_dir,
        )

        r1 = loop.run_cycle(files=["packages/core/animus/__init__.py"])
        r2 = loop.run_cycle(files=["packages/core/animus/identity.py"])

        assert r1.cycle == 1
        assert r2.cycle == 2
        assert loop.cycle_count == 2
        assert identity.reflection_count == 2

    def test_memory_persistence(
        self,
        identity: AnimusIdentity,
        mock_cognitive: CognitiveLayer,
        data_dir: Path,
    ):
        """Bootstrap results are saved to memory."""
        from animus.bootstrap_loop import BootstrapLoop

        memory = MemoryLayer(data_dir=data_dir, backend="json")
        loop = BootstrapLoop(
            identity=identity,
            cognitive=mock_cognitive,
            memory=memory,
            data_dir=data_dir,
        )

        loop.run_cycle(files=["packages/core/animus/__init__.py"])

        # Check memory contains the reflection
        memories = memory.recall("bootstrap", memory_type=MemoryType.PROCEDURAL)
        assert len(memories) >= 1
        assert "bootstrap" in memories[0].tags
        assert "self-review" in memories[0].tags

    def test_no_files_returns_early(
        self,
        identity: AnimusIdentity,
        mock_cognitive: CognitiveLayer,
        data_dir: Path,
    ):
        """Cycle with no reviewable files returns gracefully."""
        from animus.bootstrap_loop import BootstrapLoop

        memory = MemoryLayer(data_dir=data_dir, backend="json")
        loop = BootstrapLoop(
            identity=identity,
            cognitive=mock_cognitive,
            memory=memory,
            data_dir=data_dir,
        )

        result = loop.run_cycle(files=[])

        assert result.files_reviewed == []
        assert result.analysis == "No files found to review."
        assert result.consensus is None
        assert result.improvements_written is False

    def test_auto_file_selection(
        self,
        identity: AnimusIdentity,
        mock_cognitive: CognitiveLayer,
        data_dir: Path,
    ):
        """Without explicit files, auto-selects from package."""
        from animus.bootstrap_loop import BootstrapLoop

        memory = MemoryLayer(data_dir=data_dir, backend="json")
        loop = BootstrapLoop(
            identity=identity,
            cognitive=mock_cognitive,
            memory=memory,
            data_dir=data_dir,
        )

        result = loop.run_cycle(package="core", max_files=2)

        assert len(result.files_reviewed) <= 2
        assert len(result.files_reviewed) >= 1

    def test_get_history(
        self,
        identity: AnimusIdentity,
        mock_cognitive: CognitiveLayer,
        data_dir: Path,
    ):
        """History returns serializable records."""
        from animus.bootstrap_loop import BootstrapLoop

        memory = MemoryLayer(data_dir=data_dir, backend="json")
        loop = BootstrapLoop(
            identity=identity,
            cognitive=mock_cognitive,
            memory=memory,
            data_dir=data_dir,
        )

        loop.run_cycle(files=["packages/core/animus/__init__.py"])
        history = loop.get_history()

        assert len(history) == 1
        assert history[0]["cycle"] == 1
        assert "files_reviewed" in history[0]
        # Verify JSON-serializable
        json_str = json.dumps(history)
        assert json.loads(json_str) == history

    def test_result_to_dict(
        self,
        identity: AnimusIdentity,
        mock_cognitive: CognitiveLayer,
        data_dir: Path,
    ):
        """BootstrapResult.to_dict is serializable."""
        from animus.bootstrap_loop import BootstrapLoop

        memory = MemoryLayer(data_dir=data_dir, backend="json")
        loop = BootstrapLoop(
            identity=identity,
            cognitive=mock_cognitive,
            memory=memory,
            data_dir=data_dir,
        )

        result = loop.run_cycle(files=["packages/core/animus/__init__.py"])
        d = result.to_dict()

        assert d["cycle"] == 1
        assert d["consensus_approved"] is True
        assert d["improvements_written"] is True
        assert isinstance(d["timestamp"], str)

    def test_skipped_file_warning(
        self,
        identity: AnimusIdentity,
        mock_cognitive: CognitiveLayer,
        data_dir: Path,
    ):
        """Nonexistent files are skipped, not fatal."""
        from animus.bootstrap_loop import BootstrapLoop

        memory = MemoryLayer(data_dir=data_dir, backend="json")
        loop = BootstrapLoop(
            identity=identity,
            cognitive=mock_cognitive,
            memory=memory,
            data_dir=data_dir,
        )

        # Mix real and nonexistent files
        result = loop.run_cycle(
            files=[
                "packages/core/animus/__init__.py",
                "packages/core/animus/nonexistent.py",
            ]
        )

        # Should still succeed with the one real file
        assert len(result.files_reviewed) == 2  # Both listed
        assert result.consensus is not None


# ===========================================================================
# ConsensusResult Tests
# ===========================================================================


class TestConsensusResult:
    """Tests for the ConsensusResult dataclass."""

    def test_approved_summary(self):
        from animus.bootstrap_loop import ConsensusResult

        r = ConsensusResult(approved=True, approve_weight=0.85, reject_weight=0.0)
        assert "APPROVED" in r.summary
        assert "0.85" in r.summary

    def test_rejected_summary(self):
        from animus.bootstrap_loop import ConsensusResult

        r = ConsensusResult(approved=False, approve_weight=0.2, reject_weight=0.7)
        assert "REJECTED" in r.summary


# ===========================================================================
# BootstrapResult Tests
# ===========================================================================


class TestBootstrapResult:
    """Tests for the BootstrapResult dataclass."""

    def test_to_dict_truncates_long_content(self):
        from animus.bootstrap_loop import BootstrapResult, ConsensusResult

        result = BootstrapResult(
            cycle=1,
            files_reviewed=["file.py"],
            analysis="x" * 1000,
            suggestions="y" * 1000,
            consensus=ConsensusResult(approved=True),
            improvements_written=True,
        )
        d = result.to_dict()
        assert len(d["analysis"]) <= 500
        assert len(d["suggestions"]) <= 500

    def test_to_dict_none_consensus(self):
        from animus.bootstrap_loop import BootstrapResult

        result = BootstrapResult(
            cycle=1,
            files_reviewed=[],
            analysis="",
            suggestions="",
            consensus=None,
            improvements_written=False,
        )
        d = result.to_dict()
        assert d["consensus_approved"] is None
