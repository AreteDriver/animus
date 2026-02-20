"""Tests for IntentResolver event callback (hook) system."""

from __future__ import annotations

import pytest
from convergent.intent import (
    ConflictReport,
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
    ResolutionResult,
)
from convergent.resolver import IntentResolver


def _make_spec(name: str, tags: list[str] | None = None) -> InterfaceSpec:
    return InterfaceSpec(
        name=name,
        kind=InterfaceKind.FUNCTION,
        signature="(x: int) -> int",
        tags=tags or [],
    )


def _make_intent(
    agent_id: str,
    intent: str,
    provides: list[InterfaceSpec] | None = None,
    evidence: list[Evidence] | None = None,
) -> Intent:
    return Intent(
        agent_id=agent_id,
        intent=intent,
        provides=provides or [],
        evidence=evidence or [],
    )


# ---------------------------------------------------------------------------
# Registration and removal
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_add_hook_valid_event(self):
        resolver = IntentResolver()
        calls = []
        resolver.add_hook("publish", lambda *a: calls.append(a))
        assert len(resolver._hooks["publish"]) == 1

    def test_add_hook_invalid_event_raises(self):
        resolver = IntentResolver()
        with pytest.raises(ValueError, match="Unknown event 'bogus'"):
            resolver.add_hook("bogus", lambda: None)

    def test_remove_hook(self):
        resolver = IntentResolver()
        cb = lambda *a: None  # noqa: E731
        resolver.add_hook("publish", cb)
        resolver.remove_hook("publish", cb)
        assert len(resolver._hooks["publish"]) == 0

    def test_remove_hook_invalid_event_raises(self):
        resolver = IntentResolver()
        with pytest.raises(ValueError, match="Unknown event"):
            resolver.remove_hook("nope", lambda: None)

    def test_remove_hook_identity_comparison(self):
        """remove_hook uses `is`, not equality."""
        resolver = IntentResolver()
        cb1 = lambda *a: None  # noqa: E731
        cb2 = lambda *a: None  # noqa: E731
        resolver.add_hook("publish", cb1)
        resolver.add_hook("publish", cb2)
        resolver.remove_hook("publish", cb1)
        assert len(resolver._hooks["publish"]) == 1
        # The remaining one is cb2
        assert resolver._hooks["publish"][0] is cb2

    def test_multiple_hooks_per_event(self):
        resolver = IntentResolver()
        resolver.add_hook("publish", lambda *a: None)
        resolver.add_hook("publish", lambda *a: None)
        resolver.add_hook("publish", lambda *a: None)
        assert len(resolver._hooks["publish"]) == 3


# ---------------------------------------------------------------------------
# Publish hooks
# ---------------------------------------------------------------------------


class TestPublishHooks:
    def test_publish_fires_hook(self):
        resolver = IntentResolver()
        calls = []
        resolver.add_hook("publish", lambda intent, stab: calls.append((intent, stab)))

        intent = _make_intent("a1", "task")
        stability = resolver.publish(intent)

        assert len(calls) == 1
        assert calls[0][0] is intent
        assert calls[0][1] == stability

    def test_publish_fires_multiple_hooks_in_order(self):
        resolver = IntentResolver()
        order = []
        resolver.add_hook("publish", lambda *a: order.append("first"))
        resolver.add_hook("publish", lambda *a: order.append("second"))

        resolver.publish(_make_intent("a1", "task"))
        assert order == ["first", "second"]

    def test_publish_correct_stability_value(self):
        resolver = IntentResolver()
        stabs = []
        resolver.add_hook("publish", lambda i, s: stabs.append(s))

        intent = _make_intent(
            "a1",
            "task",
            evidence=[Evidence.test_pass("t1"), Evidence.code_committed("sha")],
        )
        resolver.publish(intent)
        assert stabs[0] == intent.compute_stability()


# ---------------------------------------------------------------------------
# Resolve hooks
# ---------------------------------------------------------------------------


class TestResolveHooks:
    def test_resolve_fires_hook(self):
        resolver = IntentResolver()
        calls = []
        resolver.add_hook("resolve", lambda intent, result: calls.append((intent, result)))

        intent = _make_intent("a1", "task")
        resolver.publish(intent)
        resolve_intent = _make_intent("a2", "task2")
        resolver.resolve(resolve_intent)

        assert len(calls) == 1
        assert calls[0][0] is resolve_intent
        assert isinstance(calls[0][1], ResolutionResult)

    def test_resolve_hook_receives_correct_result(self):
        resolver = IntentResolver()
        results = []
        resolver.add_hook("resolve", lambda i, r: results.append(r))

        resolver.publish(
            _make_intent("a1", "t1", provides=[_make_spec("shared", tags=["x", "y", "z"])])
        )
        intent = _make_intent("a2", "t2", provides=[_make_spec("shared", tags=["x", "y", "z"])])
        resolver.resolve(intent)

        assert len(results) == 1
        # Should have detected overlap → conflict or adjustment
        result = results[0]
        assert result.original_intent_id == intent.id


# ---------------------------------------------------------------------------
# Conflict hooks
# ---------------------------------------------------------------------------


class TestConflictHooks:
    def test_conflict_fires_on_overlap(self):
        resolver = IntentResolver(min_stability=0.0)
        conflicts = []
        resolver.add_hook("conflict", lambda intent, conflict: conflicts.append(conflict))

        # Two agents providing same function with equal stability → conflict
        resolver.publish(_make_intent("a1", "t1", provides=[_make_spec("create_user")]))
        intent = _make_intent("a2", "t2", provides=[_make_spec("create_user")])
        resolver.resolve(intent)

        assert len(conflicts) >= 1
        assert isinstance(conflicts[0], ConflictReport)

    def test_no_conflict_hook_when_clean(self):
        resolver = IntentResolver(min_stability=0.0)
        conflicts = []
        resolver.add_hook("conflict", lambda i, c: conflicts.append(c))

        resolver.publish(_make_intent("a1", "t1", provides=[_make_spec("func_a")]))
        resolver.resolve(_make_intent("a2", "t2", provides=[_make_spec("func_b")]))

        assert len(conflicts) == 0


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------


class TestExceptionSafety:
    def test_hook_exception_does_not_break_publish(self):
        resolver = IntentResolver()
        resolver.add_hook("publish", lambda *a: 1 / 0)  # ZeroDivisionError

        # Should not raise
        stability = resolver.publish(_make_intent("a1", "task"))
        assert isinstance(stability, float)

    def test_hook_exception_does_not_break_resolve(self):
        resolver = IntentResolver()
        resolver.add_hook("resolve", lambda *a: 1 / 0)

        resolver.publish(_make_intent("a1", "task"))
        result = resolver.resolve(_make_intent("a2", "task2"))
        assert isinstance(result, ResolutionResult)

    def test_hook_exception_allows_subsequent_hooks(self):
        resolver = IntentResolver()
        calls = []
        resolver.add_hook("publish", lambda *a: 1 / 0)
        resolver.add_hook("publish", lambda *a: calls.append("ok"))

        resolver.publish(_make_intent("a1", "task"))
        assert calls == ["ok"]


# ---------------------------------------------------------------------------
# No-op when empty
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_empty_hooks_no_overhead(self):
        resolver = IntentResolver()
        # No hooks registered — should work fine
        stability = resolver.publish(_make_intent("a1", "task"))
        assert isinstance(stability, float)
        result = resolver.resolve(_make_intent("a2", "task2"))
        assert isinstance(result, ResolutionResult)
