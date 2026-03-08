"""Tests for chainlog_bridge optional integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.agents.chainlog_bridge import (
    _make_noop_coroutine,
    get_chainlog,
    reset_chainlog,
    trace_agent_action,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Reset the chainlog singleton before and after each test."""
    reset_chainlog()
    yield
    reset_chainlog()


class TestGetChainlog:
    def test_returns_none_without_chainlog_installed(self) -> None:
        with patch("animus_forge.agents.chainlog_bridge.HAS_CHAINLOG", False):
            assert get_chainlog() is None

    def test_returns_none_without_env_var(self) -> None:
        with (
            patch("animus_forge.agents.chainlog_bridge.HAS_CHAINLOG", True),
            patch.dict("os.environ", {}, clear=True),
        ):
            assert get_chainlog() is None

    def test_creates_instance_with_env_vars(self) -> None:
        mock_chainlog = MagicMock()
        mock_config = MagicMock()

        with (
            patch("animus_forge.agents.chainlog_bridge.HAS_CHAINLOG", True),
            patch("animus_forge.agents.chainlog_bridge.ChainLog", mock_chainlog),
            patch("animus_forge.agents.chainlog_bridge.ChainLogConfig", mock_config),
            patch.dict(
                "os.environ",
                {
                    "CHAINLOG_CONTRACT_ADDRESS": "0x1234567890abcdef",
                    "CHAINLOG_PRIVATE_KEY": "0xkey",
                },
            ),
        ):
            result = get_chainlog()
            assert result is not None
            mock_config.assert_called_once()
            mock_chainlog.assert_called_once()

    def test_returns_cached_instance(self) -> None:
        mock_chainlog = MagicMock()
        mock_config = MagicMock()

        with (
            patch("animus_forge.agents.chainlog_bridge.HAS_CHAINLOG", True),
            patch("animus_forge.agents.chainlog_bridge.ChainLog", mock_chainlog),
            patch("animus_forge.agents.chainlog_bridge.ChainLogConfig", mock_config),
            patch.dict(
                "os.environ",
                {"CHAINLOG_CONTRACT_ADDRESS": "0x1234567890abcdef"},
            ),
        ):
            first = get_chainlog()
            second = get_chainlog()
            assert first is second
            assert mock_chainlog.call_count == 1

    def test_returns_none_on_init_failure(self) -> None:
        mock_chainlog = MagicMock(side_effect=RuntimeError("init failed"))
        mock_config = MagicMock()

        with (
            patch("animus_forge.agents.chainlog_bridge.HAS_CHAINLOG", True),
            patch("animus_forge.agents.chainlog_bridge.ChainLog", mock_chainlog),
            patch("animus_forge.agents.chainlog_bridge.ChainLogConfig", mock_config),
            patch.dict(
                "os.environ",
                {"CHAINLOG_CONTRACT_ADDRESS": "0x1234567890abcdef"},
            ),
        ):
            assert get_chainlog() is None


class TestTraceAgentAction:
    async def test_returns_none_when_disabled(self) -> None:
        with patch("animus_forge.agents.chainlog_bridge.get_chainlog", return_value=None):
            result = await trace_agent_action(
                agent_id="builder",
                action_type="delegation",
                input_data={"task": "test"},
                output="done",
                duration_ms=100,
            )
            assert result is None

    async def test_traces_when_enabled(self) -> None:
        mock_result = MagicMock()
        mock_result.action_hash = "0x" + "ab" * 32
        mock_result.stored = "local"

        mock_cl = MagicMock()
        mock_cl.trace_action = AsyncMock(return_value=mock_result)

        with patch("animus_forge.agents.chainlog_bridge.get_chainlog", return_value=mock_cl):
            result = await trace_agent_action(
                agent_id="builder",
                action_type="delegation",
                input_data={"task": "build feature"},
                output="feature built",
                duration_ms=250,
                model_id="claude-sonnet-4-6",
            )
            assert result is mock_result
            mock_cl.trace_action.assert_called_once()
            call_kwargs = mock_cl.trace_action.call_args.kwargs
            assert call_kwargs["agent_id"] == "forge-builder"
            assert call_kwargs["action_type"] == "delegation"

    async def test_returns_none_on_trace_failure(self) -> None:
        mock_cl = MagicMock()
        mock_cl.trace_action = AsyncMock(side_effect=RuntimeError("chain down"))

        with patch("animus_forge.agents.chainlog_bridge.get_chainlog", return_value=mock_cl):
            result = await trace_agent_action(
                agent_id="tester",
                action_type="delegation",
                input_data={},
                output="test output",
                duration_ms=50,
            )
            assert result is None


class TestMakeNoopCoroutine:
    async def test_returns_value(self) -> None:
        coro_fn = _make_noop_coroutine("hello")
        result = await coro_fn()
        assert result == "hello"

    async def test_returns_dict(self) -> None:
        data = {"key": "value"}
        coro_fn = _make_noop_coroutine(data)
        result = await coro_fn()
        assert result == data


class TestResetChainlog:
    def test_reset_clears_singleton(self) -> None:
        mock_cl = MagicMock()
        mock_config = MagicMock()

        with (
            patch("animus_forge.agents.chainlog_bridge.HAS_CHAINLOG", True),
            patch("animus_forge.agents.chainlog_bridge.ChainLog", mock_cl),
            patch("animus_forge.agents.chainlog_bridge.ChainLogConfig", mock_config),
            patch.dict(
                "os.environ",
                {"CHAINLOG_CONTRACT_ADDRESS": "0xabc"},
            ),
        ):
            first = get_chainlog()
            assert first is not None
            reset_chainlog()

            # After reset, should create a new instance
            get_chainlog()
            assert mock_cl.call_count == 2
