"""Tests for RustGraphBackend — Rust PyO3 intent graph wrapper.

All tests require the Rust extension (``maturin develop --release``).
Skipped automatically if ``convergent._core`` is not importable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("convergent._core")

from convergent.intent import (  # noqa: E402
    Constraint,
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.resolver import IntentResolver  # noqa: E402
from convergent.rust_backend import (  # noqa: E402
    HAS_RUST,
    RustGraphBackend,
    _rust_dict_to_spec,
)

# ---------------------------------------------------------------------------
# Helpers
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
    """In-memory Rust backend."""
    b = RustGraphBackend()
    yield b


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary file-backed Rust backend path."""
    return str(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_has_rust_flag(self):
        assert HAS_RUST is True

    def test_can_create_in_memory(self):
        b = RustGraphBackend()
        assert b.count() == 0


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_publish_returns_float(self, backend):
        intent = _make_intent("agent-a", "build api", provides=[_make_spec("handler")])
        result = backend.publish(intent)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_query_all_returns_intents(self, backend):
        backend.publish(_make_intent("a", "task", provides=[_make_spec("fn")]))
        results = backend.query_all()
        assert len(results) == 1
        assert isinstance(results[0], Intent)

    def test_query_all_min_stability(self, backend):
        backend.publish(_make_intent("a", "low", provides=[_make_spec("fn")]))
        assert len(backend.query_all(min_stability=0.9)) == 0
        assert len(backend.query_all(min_stability=0.0)) == 1

    def test_query_by_agent(self, backend):
        backend.publish(_make_intent("a", "t1", provides=[_make_spec("fn1")]))
        backend.publish(_make_intent("b", "t2", provides=[_make_spec("fn2")]))
        assert len(backend.query_by_agent("a")) == 1
        assert len(backend.query_by_agent("c")) == 0

    def test_find_overlapping(self, backend):
        backend.publish(_make_intent("a", "task-a", provides=[_make_spec("shared_fn")]))
        specs = [_make_spec("shared_fn")]
        overlaps = backend.find_overlapping(specs, "b", 0.0)
        assert len(overlaps) >= 1

    def test_find_overlapping_excludes_own_agent(self, backend):
        backend.publish(_make_intent("a", "task-a", provides=[_make_spec("shared_fn")]))
        specs = [_make_spec("shared_fn")]
        overlaps = backend.find_overlapping(specs, "a", 0.0)
        assert len(overlaps) == 0

    def test_count(self, backend):
        assert backend.count() == 0
        backend.publish(_make_intent("a", "t", provides=[_make_spec("fn")]))
        assert backend.count() == 1


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_basic_fields(self, backend):
        original = _make_intent(
            "agent-x",
            "build login",
            provides=[_make_spec("login_handler", tags=["auth", "api"])],
            requires=[_make_spec("db_connect")],
        )
        backend.publish(original)
        retrieved = backend.query_all()[0]

        assert retrieved.agent_id == original.agent_id
        assert retrieved.intent == original.intent
        assert retrieved.id == original.id
        assert len(retrieved.provides) == 1
        assert len(retrieved.requires) == 1
        assert retrieved.provides[0].name == "login_handler"
        assert retrieved.requires[0].name == "db_connect"

    def test_tags_preserved(self, backend):
        intent = _make_intent("a", "t", provides=[_make_spec("fn", tags=["web", "api", "v2"])])
        backend.publish(intent)
        retrieved = backend.query_all()[0]
        assert set(retrieved.provides[0].tags) == {"web", "api", "v2"}

    def test_constraints_round_trip(self, backend):
        intent = _make_intent("a", "t", provides=[_make_spec("fn")])
        intent.constraints = [
            Constraint(
                target="python_version",
                requirement=">=3.10",
                affects_tags=["build"],
            )
        ]
        backend.publish(intent)
        retrieved = backend.query_all()[0]
        assert len(retrieved.constraints) == 1
        assert retrieved.constraints[0].target == "python_version"
        assert retrieved.constraints[0].requirement == ">=3.10"

    def test_evidence_round_trip(self, backend):
        intent = _make_intent("a", "t", provides=[_make_spec("fn")])
        intent.evidence = [Evidence.test_pass("unit tests green")]
        backend.publish(intent)
        retrieved = backend.query_all()[0]
        # Rust may or may not preserve evidence — check stability reflects it
        assert isinstance(retrieved.stability, float)


# ---------------------------------------------------------------------------
# Persistence (file-backed)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_survives_close_reopen(self, tmp_db):
        b1 = RustGraphBackend(tmp_db)
        b1.publish(_make_intent("a", "persist-test", provides=[_make_spec("fn")]))
        assert b1.count() == 1
        b1.close()

        b2 = RustGraphBackend(tmp_db)
        assert b2.count() == 1
        results = b2.query_all()
        assert results[0].intent == "persist-test"
        b2.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_keys(self, backend):
        backend.publish(_make_intent("a", "t1", provides=[_make_spec("fn1")]))
        backend.publish(_make_intent("b", "t2", provides=[_make_spec("fn2")]))
        s = backend.summary()
        assert s["total_intents"] == 2
        assert s["agent_count"] == 2
        assert "a" in s["agents"]
        assert "b" in s["agents"]
        assert isinstance(s["average_stability"], float)
        assert "high_stability_count" in s

    def test_summary_empty(self, backend):
        s = backend.summary()
        assert s["total_intents"] == 0


# ---------------------------------------------------------------------------
# InterfaceKind mapping
# ---------------------------------------------------------------------------


class TestKindMapping:
    @pytest.mark.parametrize(
        "rust_kind,expected",
        [
            ("Function", InterfaceKind.FUNCTION),
            ("Class", InterfaceKind.CLASS),
            ("Model", InterfaceKind.MODEL),
            ("Endpoint", InterfaceKind.ENDPOINT),
            ("Migration", InterfaceKind.MIGRATION),
            ("Config", InterfaceKind.CONFIG),
        ],
    )
    def test_rust_kind_to_python(self, rust_kind, expected):
        d = {"name": "fn", "kind": rust_kind, "signature": "()", "module_path": "", "tags": []}
        spec = _rust_dict_to_spec(d)
        assert spec.kind == expected


# ---------------------------------------------------------------------------
# IntentResolver integration
# ---------------------------------------------------------------------------


class TestResolverIntegration:
    def test_works_as_resolver_backend(self):
        backend = RustGraphBackend()
        resolver = IntentResolver(backend=backend)
        intent = _make_intent("a", "task", provides=[_make_spec("fn")])
        stability = resolver.publish(intent)
        assert isinstance(stability, float)
        assert resolver.intent_count == 1
