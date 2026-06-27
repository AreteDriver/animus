"""Extra tests to push coverage above 97%."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from animus_prospector.config import ProspectorConfig
from animus_prospector.evaluator import Evaluator
from animus_prospector.feed import OpportunityFeed
from animus_prospector.models import (
    Action,
    EvaluationResult,
)
from animus_prospector.orchestrator import Prospector
from animus_prospector.store import EvaluationStore, OpportunityStore


class TestStoreEdgeCases:
    def test_load_all_with_bad_json(self, tmp_data_dir):
        store = OpportunityStore(tmp_data_dir)
        # Write invalid JSON line
        with open(store._file, "w") as f:
            f.write("not json\n")
            f.write(
                json.dumps(
                    {
                        "title": "valid",
                        "url": "http://valid",
                        "opportunity_type": "faucet",
                        "source_type": "manual",
                    }
                )
                + "\n"
            )
        loaded = store.load_all()
        assert len(loaded) == 1

    def test_load_seen_ids_with_bad_json(self, tmp_data_dir):
        store = OpportunityStore(tmp_data_dir)
        with open(store._file, "w") as f:
            f.write("bad line\n")
        store._load_seen_ids()  # should not crash

    def test_eval_store_load_bad_json(self, tmp_data_dir):
        store = EvaluationStore(tmp_data_dir)
        with open(store._file, "w") as f:
            f.write("bad\n")
            f.write(
                json.dumps(
                    {
                        "opportunity_id": "x",
                        "recommended_action": "skip",
                    }
                )
                + "\n"
            )
        loaded = store.load_all()
        assert len(loaded) == 1

    def test_eval_store_get_for_not_found(self, tmp_data_dir):
        store = EvaluationStore(tmp_data_dir)
        store.record(
            EvaluationResult(
                opportunity_id="abc",
                roi_score=1.0,
                scam_probability=0.0,
                recommended_action=Action.SKIP,
                reasoning="test",
            )
        )
        assert store.get_for_opportunity("xyz") is None
        assert store.get_for_opportunity("abc") is not None


class TestEvaluatorLLMParsing:
    def test_parse_response_with_no_json_match(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        result = evaluator._parse_llm_response(sample_opportunity, "no json here at all!!!")
        assert result.recommended_action == Action.FLAG_FOR_REVIEW

    def test_parse_response_invalid_json_in_braces(self, sample_config, sample_opportunity):
        evaluator = Evaluator(sample_config)
        result = evaluator._parse_llm_response(sample_opportunity, "text {not: valid: json:} text")
        assert result.recommended_action == Action.FLAG_FOR_REVIEW


class TestFeedEdgeCases:
    def test_get_actionable_no_eval(self, tmp_data_dir, sample_opportunity):
        feed = OpportunityFeed(tmp_data_dir)
        feed.record_opportunity(sample_opportunity)
        # No evaluation recorded
        actionable = feed.get_actionable()
        assert len(actionable) == 0

    def test_get_flagged_no_eval(self, tmp_data_dir, sample_opportunity):
        feed = OpportunityFeed(tmp_data_dir)
        feed.record_opportunity(sample_opportunity)
        flagged = feed.get_flagged()
        assert len(flagged) == 0

    def test_get_new_faucets_no_eval(self, tmp_data_dir, sample_opportunity):
        feed = OpportunityFeed(tmp_data_dir)
        feed.record_opportunity(sample_opportunity)
        faucets = feed.get_new_faucets()
        assert len(faucets) == 0

    def test_summary_with_skip_actions(self, tmp_data_dir, sample_opportunity):
        feed = OpportunityFeed(tmp_data_dir)
        feed.record_opportunity(sample_opportunity)
        feed.record_evaluation(
            EvaluationResult(
                opportunity_id=sample_opportunity.opportunity_id,
                roi_score=0.0,
                scam_probability=0.9,
                recommended_action=Action.SKIP,
                reasoning="scam",
            )
        )
        summary = feed.get_summary()
        assert summary["by_action"]["skip"] == 1


class TestToolFunctions:
    def _setup(self, tmp_data_dir):
        import animus_prospector.tools.prospector_tools as pt

        pt._prospector = None
        pt._config = None

        config = ProspectorConfig(data_dir=tmp_data_dir)
        config.ensure_dirs()
        pt._config = config
        pt._prospector = Prospector(config)
        return pt

    def test_prospector_feed_actionable(self, tmp_data_dir):
        self._setup(tmp_data_dir)
        from animus_prospector.tools.prospector_tools import prospector_feed

        result = prospector_feed({"view": "actionable"})
        assert result["success"] is True
        assert "opportunities" in result

    def test_prospector_feed_flagged(self, tmp_data_dir):
        self._setup(tmp_data_dir)
        from animus_prospector.tools.prospector_tools import prospector_feed

        result = prospector_feed({"view": "flagged"})
        assert result["success"] is True

    def test_prospector_feed_faucets(self, tmp_data_dir):
        self._setup(tmp_data_dir)
        from animus_prospector.tools.prospector_tools import prospector_feed

        result = prospector_feed({"view": "faucets"})
        assert result["success"] is True

    def test_prospector_feed_summary(self, tmp_data_dir):
        self._setup(tmp_data_dir)
        from animus_prospector.tools.prospector_tools import prospector_feed

        result = prospector_feed({"view": "summary"})
        assert result["success"] is True
        assert "summary" in result

    def test_prospector_feed_invalid_view(self, tmp_data_dir):
        self._setup(tmp_data_dir)
        from animus_prospector.tools.prospector_tools import prospector_feed

        result = prospector_feed({"view": "invalid"})
        assert result["success"] is False

    def test_prospector_feed_default_view(self, tmp_data_dir):
        self._setup(tmp_data_dir)
        from animus_prospector.tools.prospector_tools import prospector_feed

        result = prospector_feed({})
        assert result["success"] is True

    def test_prospector_health(self, tmp_data_dir):
        pt = self._setup(tmp_data_dir)
        from animus_prospector.tools.prospector_tools import prospector_health

        for hunter in pt._prospector._hunters:
            hunter.health_check = AsyncMock(return_value=True)

        with patch("animus_prospector.tools.prospector_tools.asyncio") as mock_asyncio:
            mock_asyncio.get_event_loop.return_value.run_until_complete.return_value = {
                "aggregator": True
            }
            result = prospector_health({})

        assert result["success"] is True

    def test_prospector_scan(self, tmp_data_dir):
        self._setup(tmp_data_dir)
        from animus_prospector.orchestrator import ScanReport
        from animus_prospector.tools.prospector_tools import prospector_scan

        mock_report = ScanReport(total_scanned=5, new_discovered=2)

        with patch("animus_prospector.tools.prospector_tools.asyncio") as mock_asyncio:
            mock_asyncio.get_event_loop.return_value.run_until_complete.return_value = mock_report
            result = prospector_scan({})

        assert result["success"] is True

    def test_ensure_initialized_default(self, tmp_path):
        import animus_prospector.tools.prospector_tools as pt

        pt._prospector = None
        pt._config = None

        with patch("animus_prospector.tools.prospector_tools.ProspectorConfig") as mock_cls:
            mock_config = MagicMock()
            mock_config.data_dir = tmp_path
            mock_config.config_file = tmp_path / "config.yaml"
            mock_cls.return_value = mock_config
            from animus_prospector.tools.prospector_tools import _ensure_initialized

            p, c = _ensure_initialized()
            assert p is not None

    def test_ensure_initialized_with_yaml(self, tmp_data_dir):
        import animus_prospector.tools.prospector_tools as pt

        pt._prospector = None
        pt._config = None

        config_file = tmp_data_dir / "config.yaml"
        config_file.write_text("llm:\n  model: test")

        with patch("animus_prospector.tools.prospector_tools.ProspectorConfig") as mock_cls:
            mock_config = MagicMock()
            mock_config.data_dir = tmp_data_dir
            mock_config.config_file = config_file
            mock_cls.return_value = mock_config

            mock_yaml_config = MagicMock()
            mock_yaml_config.data_dir = tmp_data_dir
            mock_cls.from_yaml.return_value = mock_yaml_config

            from animus_prospector.tools.prospector_tools import _ensure_initialized

            _ensure_initialized()
            mock_cls.from_yaml.assert_called_once()
