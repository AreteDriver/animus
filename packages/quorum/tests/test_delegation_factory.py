"""Tests for create_delegation_checker factory."""

from __future__ import annotations

from convergent import (
    IntentResolver,
    PythonGraphBackend,
    SQLiteBackend,
    create_delegation_checker,
)


class TestCreateDelegationChecker:
    def test_returns_intent_resolver(self):
        resolver = create_delegation_checker()
        assert isinstance(resolver, IntentResolver)

    def test_default_backend_is_python(self):
        resolver = create_delegation_checker()
        assert isinstance(resolver.backend, PythonGraphBackend)

    def test_custom_min_stability(self):
        resolver = create_delegation_checker(min_stability=0.5)
        assert resolver.min_stability == 0.5

    def test_custom_backend(self):
        backend = SQLiteBackend(":memory:")
        resolver = create_delegation_checker(backend=backend)
        assert isinstance(resolver.backend, SQLiteBackend)
        backend.close()
