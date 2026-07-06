"""Tests for eval_evidence integration module."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from animus.citizens.eval_evidence import (
    build_eval_evidence_item,
    query_eval_runs,
    read_eval_results_from_dir,
    read_eval_results_from_memory,
)


class TestQueryEvalRuns:
    def test_returns_empty_when_forge_not_installed(self):
        with patch.dict("sys.modules", {"animus_forge": None}):
            results = query_eval_runs()
        assert results == []

    def test_queries_forge_store(self):
        # Reset the module-level cache so this test isn't affected by prior runs.
        import animus.citizens.eval_evidence as ev_mod
        ev_mod._eval_db_available = None

        mock_store_cls = MagicMock()
        mock_store = MagicMock()
        mock_store.query_runs.return_value = [
            {"suite_name": "personal-quality", "score": 0.85, "timestamp": "2026-07-01T00:00:00"},
        ]
        mock_store_cls.return_value = mock_store

        with (
            patch("animus.citizens.eval_evidence._try_import_eval_store", return_value=mock_store_cls),
            patch("animus.citizens.eval_evidence._try_create_backend", return_value=MagicMock()),
        ):
            results = query_eval_runs(suite_name="personal-quality")

        assert len(results) == 1
        assert results[0]["suite_name"] == "personal-quality"
        assert results[0]["score"] == 0.85


class TestReadEvalResultsFromMemory:
    def test_returns_empty_without_memory(self):
        assert read_eval_results_from_memory(None) == []

    def test_reads_eval_from_memory(self):
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {
                "content": "Eval run",
                "metadata": {
                    "suite": "personal-quality",
                    "score": 0.9,
                    "timestamp": "2026-07-01T00:00:00",
                },
            },
            {
                "content": "Something else",
                "metadata": {" unrelated": True},
            },
        ]

        results = read_eval_results_from_memory(mock_memory)
        assert len(results) == 1
        assert results[0]["suite_name"] == "personal-quality"
        assert results[0]["score"] == 0.9


class TestBuildEvalEvidenceItem:
    def test_builds_evidence_dict(self):
        run = {
            "suite_name": "personal-quality",
            "score": 0.85,
            "status": "PASSED",
            "failure_mode": "",
            "rubric_band": "A",
            "pass_rate": 0.95,
            "timestamp": "2026-07-01T00:00:00",
        }
        item = build_eval_evidence_item(run)
        assert item["source"] == "eval_system"
        assert "personal-quality" in item["description"]
        assert item["data"]["score"] == 0.85
        assert item["data"]["rubric_band"] == "A"

    def test_handles_missing_fields(self):
        run = {"suite_name": "test", "score": 0.5}
        item = build_eval_evidence_item(run)
        assert item["data"]["rubric_band"] == ""
        assert item["data"]["failure_mode"] == ""


class TestReadEvalResultsFromDir:
    def test_reads_json_files(self, tmp_path):
        eval_dir = tmp_path / "evals"
        eval_dir.mkdir()
        (eval_dir / "run1.json").write_text(
            '{"suite_name": "s1", "score": 0.8, "timestamp": "2026-07-01T00:00:00"}'
        )

        results = read_eval_results_from_dir(eval_dir)
        assert len(results) == 1
        assert results[0]["suite_name"] == "s1"

    def test_returns_empty_for_missing_dir(self):
        assert read_eval_results_from_dir("/nonexistent/path") == []
