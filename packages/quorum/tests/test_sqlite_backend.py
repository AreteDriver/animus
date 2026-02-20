"""Tests for SQLiteBackend — persistent intent graph."""

from __future__ import annotations

import pytest
from convergent.intent import (
    Constraint,
    ConstraintSeverity,
    Evidence,
    EvidenceKind,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.resolver import IntentResolver, PythonGraphBackend
from convergent.sqlite_backend import SQLiteBackend
from convergent.versioning import VersionedGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(name: str, tags: list[str] | None = None) -> InterfaceSpec:
    return InterfaceSpec(
        name=name,
        kind=InterfaceKind.FUNCTION,
        signature="(x: int) -> int",
        module_path="mod",
        tags=tags or [],
    )


def _make_intent(
    agent_id: str,
    intent: str,
    provides: list[InterfaceSpec] | None = None,
    requires: list[InterfaceSpec] | None = None,
    constraints: list[Constraint] | None = None,
    evidence: list[Evidence] | None = None,
) -> Intent:
    return Intent(
        agent_id=agent_id,
        intent=intent,
        provides=provides or [],
        requires=requires or [],
        constraints=constraints or [],
        evidence=evidence or [],
    )


@pytest.fixture
def backend():
    """In-memory SQLite backend."""
    b = SQLiteBackend(":memory:")
    yield b
    b.close()


@pytest.fixture
def file_db(tmp_path):
    """File-based SQLite backend path."""
    return str(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestPublish:
    def test_publish_returns_stability(self, backend):
        intent = _make_intent("a1", "build widget")
        stability = backend.publish(intent)
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0

    def test_publish_increments_count(self, backend):
        assert backend.count() == 0
        backend.publish(_make_intent("a1", "task1"))
        assert backend.count() == 1
        backend.publish(_make_intent("a2", "task2"))
        assert backend.count() == 2

    def test_publish_same_id_replaces(self, backend):
        intent = _make_intent("a1", "task1")
        backend.publish(intent)
        backend.publish(intent)  # same ID
        assert backend.count() == 1

    def test_publish_with_evidence_affects_stability(self, backend):
        intent = _make_intent(
            "a1",
            "tested task",
            evidence=[
                Evidence.test_pass("test 1"),
                Evidence.test_pass("test 2"),
                Evidence.code_committed("sha123"),
            ],
        )
        stability = backend.publish(intent)
        assert stability > 0.3  # base is 0.3


class TestQueryAll:
    def test_query_all_empty(self, backend):
        assert backend.query_all() == []

    def test_query_all_returns_all(self, backend):
        backend.publish(_make_intent("a1", "t1"))
        backend.publish(_make_intent("a2", "t2"))
        results = backend.query_all()
        assert len(results) == 2

    def test_query_all_min_stability_filter(self, backend):
        low = _make_intent("a1", "low stab")
        high = _make_intent(
            "a2",
            "high stab",
            evidence=[
                Evidence.test_pass("t1"),
                Evidence.test_pass("t2"),
                Evidence.code_committed("sha"),
                Evidence(kind=EvidenceKind.MANUAL_APPROVAL, description="approved"),
            ],
        )
        backend.publish(low)
        backend.publish(high)
        results = backend.query_all(min_stability=0.8)
        assert len(results) == 1
        assert results[0].agent_id == "a2"

    def test_query_all_none_stability_returns_all(self, backend):
        backend.publish(_make_intent("a1", "t1"))
        assert len(backend.query_all(min_stability=None)) == 1


class TestQueryByAgent:
    def test_query_by_agent(self, backend):
        backend.publish(_make_intent("alice", "t1"))
        backend.publish(_make_intent("bob", "t2"))
        backend.publish(_make_intent("alice", "t3"))
        results = backend.query_by_agent("alice")
        assert len(results) == 2
        assert all(i.agent_id == "alice" for i in results)

    def test_query_by_agent_empty(self, backend):
        assert backend.query_by_agent("ghost") == []


class TestFindOverlapping:
    def test_find_overlapping_by_name(self, backend):
        backend.publish(_make_intent("a1", "t1", provides=[_make_spec("create_user")]))
        specs = [_make_spec("create_user")]
        results = backend.find_overlapping(specs, "a2", 0.0)
        assert len(results) == 1

    def test_find_overlapping_excludes_own_agent(self, backend):
        backend.publish(_make_intent("a1", "t1", provides=[_make_spec("create_user")]))
        specs = [_make_spec("create_user")]
        results = backend.find_overlapping(specs, "a1", 0.0)
        assert len(results) == 0

    def test_find_overlapping_by_tags(self, backend):
        backend.publish(
            _make_intent(
                "a1",
                "t1",
                provides=[_make_spec("build_widget", tags=["auth", "users", "api"])],
            )
        )
        specs = [_make_spec("different_name", tags=["auth", "users"])]
        results = backend.find_overlapping(specs, "a2", 0.0)
        assert len(results) == 1

    def test_find_overlapping_respects_min_stability(self, backend):
        backend.publish(_make_intent("a1", "low", provides=[_make_spec("func")]))
        specs = [_make_spec("func")]
        results = backend.find_overlapping(specs, "a2", 0.9)
        assert len(results) == 0

    def test_find_overlapping_empty_specs(self, backend):
        backend.publish(_make_intent("a1", "t1", provides=[_make_spec("func")]))
        results = backend.find_overlapping([], "a2", 0.0)
        assert len(results) == 0

    def test_find_overlapping_no_match(self, backend):
        backend.publish(_make_intent("a1", "t1", provides=[_make_spec("func_a")]))
        specs = [_make_spec("func_b")]
        results = backend.find_overlapping(specs, "a2", 0.0)
        assert len(results) == 0


class TestCount:
    def test_count_empty(self, backend):
        assert backend.count() == 0

    def test_count_after_publishes(self, backend):
        for i in range(5):
            backend.publish(_make_intent(f"a{i}", f"task{i}"))
        assert backend.count() == 5


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_interface_spec_round_trip(self, backend):
        spec = InterfaceSpec(
            name="process_data",
            kind=InterfaceKind.ENDPOINT,
            signature="(data: bytes) -> dict",
            module_path="services.data",
            tags=["data", "processing", "api"],
        )
        intent = _make_intent("a1", "process", provides=[spec])
        backend.publish(intent)
        result = backend.query_all()[0]
        assert len(result.provides) == 1
        rt = result.provides[0]
        assert rt.name == "process_data"
        assert rt.kind == InterfaceKind.ENDPOINT
        assert rt.signature == "(data: bytes) -> dict"
        assert rt.module_path == "services.data"
        assert rt.tags == ["data", "processing", "api"]

    def test_constraint_round_trip(self, backend):
        constraint = Constraint(
            target="database",
            requirement="use_postgres",
            severity=ConstraintSeverity.CRITICAL,
            affects_tags=["db", "storage"],
        )
        intent = _make_intent("a1", "constrained", constraints=[constraint])
        backend.publish(intent)
        result = backend.query_all()[0]
        assert len(result.constraints) == 1
        rc = result.constraints[0]
        assert rc.target == "database"
        assert rc.requirement == "use_postgres"
        assert rc.severity == ConstraintSeverity.CRITICAL
        assert rc.affects_tags == ["db", "storage"]

    def test_evidence_round_trip(self, backend):
        evidence = [
            Evidence.test_pass("all tests green"),
            Evidence.code_committed("abc123"),
            Evidence.conflict("version mismatch"),
        ]
        intent = _make_intent("a1", "evidenced", evidence=evidence)
        backend.publish(intent)
        result = backend.query_all()[0]
        assert len(result.evidence) == 3
        assert result.evidence[0].kind == EvidenceKind.TEST_PASS
        assert result.evidence[1].kind == EvidenceKind.CODE_COMMITTED
        assert result.evidence[2].kind == EvidenceKind.CONFLICT

    def test_full_intent_round_trip(self, backend):
        intent = Intent(
            agent_id="agent_x",
            intent="full round trip test",
            provides=[_make_spec("func_a", tags=["tag1", "tag2"])],
            requires=[_make_spec("func_b")],
            constraints=[
                Constraint(
                    target="version",
                    requirement=">=3.10",
                    affects_tags=["python"],
                )
            ],
            evidence=[Evidence.test_pass("ok")],
            parent_id="parent-123",
        )
        backend.publish(intent)
        result = backend.query_all()[0]
        assert result.agent_id == "agent_x"
        assert result.intent == "full round trip test"
        assert result.id == intent.id
        assert len(result.provides) == 1
        assert len(result.requires) == 1
        assert len(result.constraints) == 1
        assert len(result.evidence) == 1
        assert result.parent_id == "parent-123"


# ---------------------------------------------------------------------------
# Persistence (file-based DB)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_file_db_survives_close_reopen(self, file_db):
        b1 = SQLiteBackend(file_db)
        b1.publish(_make_intent("a1", "persistent task"))
        b1.close()

        b2 = SQLiteBackend(file_db)
        assert b2.count() == 1
        result = b2.query_all()[0]
        assert result.intent == "persistent task"
        b2.close()

    def test_file_db_multiple_intents(self, file_db):
        b = SQLiteBackend(file_db)
        for i in range(10):
            b.publish(_make_intent(f"a{i}", f"task_{i}"))
        b.close()

        b2 = SQLiteBackend(file_db)
        assert b2.count() == 10
        b2.close()


# ---------------------------------------------------------------------------
# Drop-in replacement (same behavior as PythonGraphBackend)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["sqlite", "python"])
def graph_backend(request):
    if request.param == "sqlite":
        b = SQLiteBackend(":memory:")
        yield b
        b.close()
    else:
        yield PythonGraphBackend()


class TestDropIn:
    """Parametrized tests that run against both backends."""

    def test_publish_and_query(self, graph_backend):
        intent = _make_intent("a1", "task", provides=[_make_spec("func")])
        graph_backend.publish(intent)
        assert graph_backend.count() == 1
        results = graph_backend.query_all()
        assert len(results) == 1
        assert results[0].intent == "task"

    def test_find_overlapping_consistency(self, graph_backend):
        graph_backend.publish(_make_intent("a1", "t1", provides=[_make_spec("shared_func")]))
        graph_backend.publish(_make_intent("a2", "t2", provides=[_make_spec("other_func")]))
        specs = [_make_spec("shared_func")]
        results = graph_backend.find_overlapping(specs, "a2", 0.0)
        assert len(results) == 1
        assert results[0].agent_id == "a1"

    def test_query_by_agent_consistency(self, graph_backend):
        graph_backend.publish(_make_intent("alice", "a_task"))
        graph_backend.publish(_make_intent("bob", "b_task"))
        results = graph_backend.query_by_agent("alice")
        assert len(results) == 1
        assert results[0].agent_id == "alice"


# ---------------------------------------------------------------------------
# VersionedGraph integration
# ---------------------------------------------------------------------------


class TestVersionedGraphIntegration:
    def _vg_intent(self, agent_id, name):
        """Create an intent valid for VersionedGraph (needs provides)."""
        return _make_intent(agent_id, name, provides=[_make_spec(f"{name}_func")])

    def test_sqlite_main_graph(self):
        vg = VersionedGraph(
            "main",
            resolver=IntentResolver(
                backend=SQLiteBackend(":memory:"),
                min_stability=0.0,
            ),
        )
        vg.publish(self._vg_intent("a1", "versioned_task"))
        snap = vg.snapshot()
        assert snap.intent_count == 1

    def test_branch_from_sqlite_main(self):
        """Branch from SQLite main — branch uses PythonGraphBackend (ephemeral)."""
        vg = VersionedGraph(
            "main",
            resolver=IntentResolver(
                backend=SQLiteBackend(":memory:"),
                min_stability=0.0,
            ),
        )
        vg.publish(self._vg_intent("a1", "base_task"))
        branch = vg.branch("feature")
        branch.publish(self._vg_intent("a2", "branch_task"))
        assert branch.resolver.backend.count() == 2
        assert vg.resolver.backend.count() == 1  # main unaffected

    def test_merge_branch_back_to_sqlite(self):
        vg = VersionedGraph(
            "main",
            resolver=IntentResolver(
                backend=SQLiteBackend(":memory:"),
                min_stability=0.0,
            ),
        )
        vg.publish(self._vg_intent("a1", "base"))
        branch = vg.branch("feat")
        branch.publish(self._vg_intent("a2", "new"))
        result = vg.merge(branch)
        assert result.success
        assert vg.resolver.backend.count() == 2

    def test_backend_factory(self):
        """VersionedGraph with backend_factory creates SQLite backend."""
        vg = VersionedGraph(
            "main",
            backend_factory=SQLiteBackend,
        )
        vg.publish(self._vg_intent("a1", "factory_task"))
        assert vg.resolver.backend.count() == 1
        assert isinstance(vg.resolver.backend, SQLiteBackend)


class TestClose:
    def test_close_is_idempotent(self, backend):
        backend.close()
        backend.close()  # should not raise
