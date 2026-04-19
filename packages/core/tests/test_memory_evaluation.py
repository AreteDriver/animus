"""Tests for animus.memory.evaluation — retrieval metrics + fixture runner."""

from __future__ import annotations

from pathlib import Path

from animus.memory import LocalMemoryStore, MemoryType
from animus.memory.evaluation import (
    EvaluationResult,
    Fixture,
    LabeledQuery,
    evaluate,
    load_fixture,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "memory_eval_corpus.json"


class TestPrecisionAtK:
    def test_all_retrieved_relevant(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial_hit(self):
        # 1 of 3 top-3 retrieved is relevant
        assert precision_at_k(["a", "x", "y"], ["a", "b"], k=3) == 1 / 3

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], k=5) == 0.0

    def test_k_exceeds_retrieved(self):
        # Only 2 items retrieved; k=5 caps at whatever came back
        assert precision_at_k(["a", "b"], ["a"], k=5) == 0.5


class TestRecallAtK:
    def test_all_expected_retrieved(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_one_of_two_expected(self):
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5

    def test_empty_expected_returns_zero(self):
        assert recall_at_k(["a", "b"], [], k=5) == 0.0

    def test_expected_outside_k_misses(self):
        # expected is at rank 6 but k=5
        assert recall_at_k(["x"] * 5 + ["a"], ["a"], k=5) == 0.0


class TestReciprocalRank:
    def test_first_result_is_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_result_is_relevant(self):
        assert reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5

    def test_third_result_is_relevant(self):
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == 1 / 3

    def test_no_relevant_returns_zero(self):
        assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

    def test_multiple_expected_uses_first_hit(self):
        # 'b' appears at rank 2, 'c' at rank 3; first hit wins
        assert reciprocal_rank(["x", "b", "c"], ["b", "c"]) == 0.5


class TestLoadFixture:
    def test_fixture_file_exists(self):
        assert FIXTURE_PATH.exists()

    def test_fixture_loads(self):
        fixture = load_fixture(FIXTURE_PATH)
        assert isinstance(fixture, Fixture)

    def test_fixture_has_corpus(self):
        fixture = load_fixture(FIXTURE_PATH)
        assert len(fixture.memories) == 40
        assert all(m.id.startswith("mem-") for m in fixture.memories)

    def test_fixture_has_queries(self):
        fixture = load_fixture(FIXTURE_PATH)
        assert len(fixture.queries) == 20
        assert all(isinstance(q, LabeledQuery) for q in fixture.queries)

    def test_queries_reference_valid_memory_ids(self):
        fixture = load_fixture(FIXTURE_PATH)
        mem_ids = {m.id for m in fixture.memories}
        for q in fixture.queries:
            for expected in q.expected_ids:
                assert expected in mem_ids, f"{q.id} expects {expected} which is not in corpus"

    def test_memory_type_filter_parsed(self):
        fixture = load_fixture(FIXTURE_PATH)
        # q-08 has memory_type "procedural"
        q08 = next(q for q in fixture.queries if q.id == "q-08")
        assert q08.memory_type == MemoryType.PROCEDURAL


class TestEvaluate:
    def test_evaluate_returns_result_shape(self, tmp_path):
        fixture = load_fixture(FIXTURE_PATH)
        store = LocalMemoryStore(tmp_path)
        result = evaluate(store, fixture)
        assert isinstance(result, EvaluationResult)
        assert result.corpus_size == 40
        assert result.query_count == 20
        assert 0.0 <= result.precision_at_k <= 1.0
        assert 0.0 <= result.recall_at_k <= 1.0
        assert 0.0 <= result.mrr <= 1.0
        assert len(result.per_query) == 20

    def test_evaluate_on_local_store_produces_real_metric(self, tmp_path):
        """LocalMemoryStore uses substring match — single-keyword queries should hit."""
        fixture = load_fixture(FIXTURE_PATH)
        store = LocalMemoryStore(tmp_path)
        result = evaluate(store, fixture)
        # "Animus" (q-01) → mem-001 has 'Animus' in content; substring should find it
        q01 = next(r for r in result.per_query if r.query_id == "q-01")
        assert "mem-001" in q01.retrieved_ids
        assert q01.reciprocal_rank > 0

    def test_evaluate_store_name_override(self, tmp_path):
        fixture = load_fixture(FIXTURE_PATH)
        store = LocalMemoryStore(tmp_path)
        result = evaluate(store, fixture, store_name="test-store")
        assert result.store_name == "test-store"

    def test_summary_line_format(self, tmp_path):
        fixture = load_fixture(FIXTURE_PATH)
        store = LocalMemoryStore(tmp_path)
        line = evaluate(store, fixture).summary_line()
        assert "corpus=" in line
        assert "P@" in line
        assert "R@" in line
        assert "MRR=" in line

    def test_result_serializes_to_dict(self, tmp_path):
        fixture = load_fixture(FIXTURE_PATH)
        store = LocalMemoryStore(tmp_path)
        d = evaluate(store, fixture).to_dict()
        assert d["corpus_size"] == 40
        assert isinstance(d["per_query"], list)

    def test_small_fixture_inline(self, tmp_path):
        """Sanity check — hand-rolled fixture isolates metric math."""
        from animus.memory.types import Memory

        memories = [
            Memory.create(content="apple banana"),
            Memory.create(content="cherry date"),
            Memory.create(content="elderberry fig"),
        ]
        # Tag first by content for expected_ids mapping
        expected_apple = memories[0].id
        fixture = Fixture(
            memories=memories,
            queries=[
                LabeledQuery(id="t1", query="apple", expected_ids=[expected_apple], k=3),
                LabeledQuery(
                    id="t2",
                    query="pomegranate",
                    expected_ids=[memories[0].id],
                    k=3,
                ),  # won't hit
            ],
        )
        store = LocalMemoryStore(tmp_path)
        result = evaluate(store, fixture, store_name="inline-test")
        t1 = next(r for r in result.per_query if r.query_id == "t1")
        t2 = next(r for r in result.per_query if r.query_id == "t2")
        # t1 should hit
        assert t1.reciprocal_rank > 0
        # t2's query doesn't match any content — should miss
        assert t2.reciprocal_rank == 0.0
