"""Tests for graph visualization outputs."""

from __future__ import annotations

import pytest
from convergent.intent import (
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)
from convergent.resolver import IntentResolver
from convergent.visualization import dot_graph, html_report, overlap_matrix, text_table


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
    requires: list[InterfaceSpec] | None = None,
    evidence: list[Evidence] | None = None,
) -> Intent:
    return Intent(
        agent_id=agent_id,
        intent=intent,
        provides=provides or [],
        requires=requires or [],
        evidence=evidence or [],
    )


@pytest.fixture
def empty_resolver():
    return IntentResolver(min_stability=0.0)


@pytest.fixture
def populated_resolver():
    resolver = IntentResolver(min_stability=0.0)
    resolver.publish(
        _make_intent(
            "alice",
            "build auth",
            provides=[_make_spec("login", tags=["auth", "users"])],
            requires=[_make_spec("db_connection")],
            evidence=[Evidence.test_pass("auth tests pass")],
        )
    )
    resolver.publish(
        _make_intent(
            "bob",
            "build api",
            provides=[_make_spec("api_endpoint")],
            requires=[_make_spec("login", tags=["auth", "users"])],
        )
    )
    resolver.publish(
        _make_intent(
            "alice",
            "add logging",
            provides=[_make_spec("logger")],
        )
    )
    return resolver


# ---------------------------------------------------------------------------
# text_table
# ---------------------------------------------------------------------------


class TestTextTable:
    def test_empty_graph(self, empty_resolver):
        result = text_table(empty_resolver)
        assert result == "(empty graph)"

    def test_single_intent(self):
        resolver = IntentResolver(min_stability=0.0)
        resolver.publish(_make_intent("a1", "task", provides=[_make_spec("func")]))
        result = text_table(resolver)
        assert "a1" in result
        assert "task" in result
        assert "func" in result

    def test_multiple_agents(self, populated_resolver):
        result = text_table(populated_resolver)
        assert "alice" in result
        assert "bob" in result
        assert "build auth" in result
        assert "build api" in result

    def test_header_present(self, populated_resolver):
        result = text_table(populated_resolver)
        assert "Agent" in result
        assert "Intent" in result
        assert "Stab" in result

    def test_evidence_display(self, populated_resolver):
        result = text_table(populated_resolver, show_evidence=True)
        assert "test_pass" in result
        assert "auth tests pass" in result

    def test_evidence_hidden_by_default(self, populated_resolver):
        result = text_table(populated_resolver, show_evidence=False)
        assert "test_pass" not in result

    def test_no_provides_shows_dash(self):
        resolver = IntentResolver(min_stability=0.0)
        resolver.publish(_make_intent("a1", "bare", requires=[_make_spec("x")]))
        result = text_table(resolver)
        assert "-" in result


# ---------------------------------------------------------------------------
# dot_graph
# ---------------------------------------------------------------------------


class TestDotGraph:
    def test_empty_graph(self, empty_resolver):
        result = dot_graph(empty_resolver)
        assert "digraph convergent" in result
        assert "}" in result

    def test_contains_subgraph_per_agent(self, populated_resolver):
        result = dot_graph(populated_resolver)
        assert "subgraph cluster_" in result
        assert '"alice"' in result or "alice" in result
        assert '"bob"' in result or "bob" in result

    def test_contains_edges_for_overlaps(self):
        """Intents from different agents with overlapping specs get edges."""
        resolver = IntentResolver(min_stability=0.0)
        resolver.publish(_make_intent("a1", "t1", provides=[_make_spec("shared_func")]))
        resolver.publish(_make_intent("a2", "t2", provides=[_make_spec("shared_func")]))
        result = dot_graph(resolver)
        assert "->" in result
        assert "dashed" in result

    def test_no_edges_when_no_overlap(self, empty_resolver):
        empty_resolver.publish(_make_intent("a1", "t1", provides=[_make_spec("func_a")]))
        empty_resolver.publish(_make_intent("a2", "t2", provides=[_make_spec("func_b")]))
        result = dot_graph(empty_resolver)
        assert "->" not in result

    def test_stability_coloring(self, populated_resolver):
        result = dot_graph(populated_resolver)
        assert "fillcolor" in result
        assert "style=filled" in result

    def test_min_stability_filter(self):
        resolver = IntentResolver(min_stability=0.0)
        resolver.publish(_make_intent("a1", "low"))
        result = dot_graph(resolver, min_stability=0.99)
        # Low stability intent should be filtered out
        assert "low" not in result or "subgraph" not in result.split("low")[0]

    def test_valid_dot_syntax(self, populated_resolver):
        result = dot_graph(populated_resolver)
        assert result.startswith("digraph")
        assert result.strip().endswith("}")


# ---------------------------------------------------------------------------
# html_report
# ---------------------------------------------------------------------------


class TestHtmlReport:
    def test_contains_html_tags(self, populated_resolver):
        result = html_report(populated_resolver)
        assert "<html>" in result
        assert "</html>" in result

    def test_self_contained(self, populated_resolver):
        result = html_report(populated_resolver)
        assert "<style>" in result
        assert "</style>" in result

    def test_all_agents_present(self, populated_resolver):
        result = html_report(populated_resolver)
        assert "alice" in result
        assert "bob" in result

    def test_summary_stats(self, populated_resolver):
        result = html_report(populated_resolver)
        assert "Total intents: 3" in result
        assert "Agents: 2" in result

    def test_empty_graph(self, empty_resolver):
        result = html_report(empty_resolver)
        assert "<html>" in result
        assert "Total intents: 0" in result

    def test_html_escaping(self):
        resolver = IntentResolver(min_stability=0.0)
        resolver.publish(_make_intent("<script>", "alert('xss')", provides=[_make_spec("x<y")]))
        result = html_report(resolver)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ---------------------------------------------------------------------------
# overlap_matrix
# ---------------------------------------------------------------------------


class TestOverlapMatrix:
    def test_empty_graph(self, empty_resolver):
        result = overlap_matrix(empty_resolver)
        assert result == "(empty graph)"

    def test_no_overlaps(self):
        resolver = IntentResolver(min_stability=0.0)
        resolver.publish(_make_intent("a1", "t1", provides=[_make_spec("func_a")]))
        resolver.publish(_make_intent("a2", "t2", provides=[_make_spec("func_b")]))
        result = overlap_matrix(resolver)
        assert "X" not in result
        assert "." in result  # self-intersection markers

    def test_known_overlaps_marked(self):
        resolver = IntentResolver(min_stability=0.0)
        resolver.publish(_make_intent("a1", "t1", provides=[_make_spec("shared")]))
        resolver.publish(_make_intent("a2", "t2", provides=[_make_spec("shared")]))
        result = overlap_matrix(resolver)
        assert "X" in result

    def test_self_intersection_dot(self):
        resolver = IntentResolver(min_stability=0.0)
        resolver.publish(_make_intent("a1", "t1", provides=[_make_spec("f")]))
        result = overlap_matrix(resolver)
        assert "." in result

    def test_legend_present(self, populated_resolver):
        result = overlap_matrix(populated_resolver)
        # Legend shows index to label mapping
        assert "0:" in result
