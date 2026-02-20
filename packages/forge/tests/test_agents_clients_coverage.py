"""Coverage tests for agent/client modules.

Targets:
  - agents/provider_wrapper.py (~52 uncovered lines, 16% coverage)
  - api_clients/resilience.py (~39 uncovered lines, 64% coverage)
  - api_clients/calendar_client.py (~173 uncovered lines, 28% coverage)
  - chat/router.py (~159 uncovered lines)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_provider():
    """Create a mock Provider for AgentProvider tests."""
    provider = MagicMock()
    provider._initialized = True
    provider._async_client = None
    provider.default_model = "claude-test"

    response = MagicMock()
    response.content = "mock response"
    provider.complete_async = AsyncMock(return_value=response)
    return provider


@pytest.fixture()
def mock_provider_uninitialized():
    """Provider that hasn't been initialized yet."""
    provider = MagicMock()
    provider._initialized = False
    provider._async_client = None
    provider.default_model = "claude-test"

    response = MagicMock()
    response.content = "initialized response"
    provider.complete_async = AsyncMock(return_value=response)
    return provider


@pytest.fixture()
def sample_messages():
    """Standard message list for tests."""
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]


@pytest.fixture()
def sample_api_event():
    """Sample Google Calendar API event response."""
    return {
        "id": "evt-123",
        "summary": "Team Meeting",
        "description": "Weekly sync",
        "location": "Room 42",
        "start": {"dateTime": "2026-02-15T10:00:00Z"},
        "end": {"dateTime": "2026-02-15T11:00:00Z"},
        "attendees": [
            {"email": "alice@example.com"},
            {"email": "bob@example.com"},
        ],
        "reminders": {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": 10}],
        },
        "recurrence": ["RRULE:FREQ=WEEKLY"],
        "status": "confirmed",
        "htmlLink": "https://calendar.google.com/event/evt-123",
    }


@pytest.fixture()
def sample_allday_event():
    """All-day event API response."""
    return {
        "id": "evt-allday",
        "summary": "Holiday",
        "start": {"date": "2026-03-01"},
        "end": {"date": "2026-03-02"},
        "status": "confirmed",
    }


@pytest.fixture()
def mock_session():
    """Mock ChatSession for router tests."""
    session = MagicMock()
    session.id = "sess-test"
    session.title = "Test Session"
    session.project_path = "/tmp/project"
    session.mode = "assistant"
    session.status = "active"
    session.created_at = datetime(2026, 2, 15, 10, 0, 0)
    session.updated_at = datetime(2026, 2, 15, 10, 0, 0)
    session.messages = []
    session.allowed_paths = []
    return session


@pytest.fixture()
def mock_session_manager(mock_session):
    """Mock ChatSessionManager."""
    manager = MagicMock()
    manager.create_session.return_value = mock_session
    manager.get_session.return_value = mock_session
    manager.get_session_with_messages.return_value = mock_session
    manager.list_sessions.return_value = [mock_session]
    manager.update_session.return_value = mock_session
    manager.delete_session.return_value = True
    manager.get_message_count.return_value = 5
    manager.get_messages.return_value = []
    manager.generate_title.return_value = "Generated Title"
    manager.get_session_jobs.return_value = ["job-1", "job-2"]
    return manager


# ===========================================================================
# 1. agents/provider_wrapper.py
# ===========================================================================


class TestAgentProviderInit:
    """Test AgentProvider initialization."""

    def test_init_with_initialized_provider(self, mock_provider):
        """Provider already initialized -- no initialize() call."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        agent = AgentProvider(mock_provider)
        assert agent.provider is mock_provider
        mock_provider.initialize.assert_not_called()

    def test_init_with_uninitialized_provider(self, mock_provider_uninitialized):
        """Provider not yet initialized -- initialize() is called."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        agent = AgentProvider(mock_provider_uninitialized)
        mock_provider_uninitialized.initialize.assert_called_once()
        assert agent.provider is mock_provider_uninitialized


class TestAgentProviderComplete:
    """Test AgentProvider.complete()."""

    def test_complete_basic(self, mock_provider, sample_messages):
        """Basic completion with system + user messages."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        agent = AgentProvider(mock_provider)
        result = asyncio.run(agent.complete(sample_messages))

        assert result == "mock response"
        mock_provider.complete_async.assert_awaited_once()
        req = mock_provider.complete_async.call_args[0][0]
        assert req.prompt == "Hello"
        assert req.system_prompt == "You are helpful."
        assert req.temperature == 0.7
        assert req.max_tokens == 4096

    def test_complete_multiple_system_prompts(self, mock_provider):
        """Multiple system messages are concatenated."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        messages = [
            {"role": "system", "content": "Part 1"},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "Go"},
        ]
        agent = AgentProvider(mock_provider)
        asyncio.run(agent.complete(messages))

        req = mock_provider.complete_async.call_args[0][0]
        assert req.system_prompt == "Part 1\n\nPart 2"

    def test_complete_no_system_prompt(self, mock_provider):
        """No system message uses default."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        messages = [{"role": "user", "content": "Hello"}]
        agent = AgentProvider(mock_provider)
        asyncio.run(agent.complete(messages))

        req = mock_provider.complete_async.call_args[0][0]
        assert req.system_prompt == "You are a helpful assistant."

    def test_complete_empty_messages(self, mock_provider):
        """Empty message list still works (prompt='')."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        agent = AgentProvider(mock_provider)
        asyncio.run(agent.complete([]))

        req = mock_provider.complete_async.call_args[0][0]
        assert req.prompt == ""

    def test_complete_only_system_messages(self, mock_provider):
        """Only system messages -- no user content, prompt is empty."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        messages = [{"role": "system", "content": "System only"}]
        agent = AgentProvider(mock_provider)
        asyncio.run(agent.complete(messages))

        req = mock_provider.complete_async.call_args[0][0]
        assert req.prompt == ""
        assert req.system_prompt == "System only"


class TestAgentProviderStreamCompletion:
    """Test AgentProvider.stream_completion()."""

    def test_stream_fallback_no_async_client(self, mock_provider, sample_messages):
        """Falls back to non-streaming when no _async_client."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_provider._async_client = None
        agent = AgentProvider(mock_provider)

        async def _collect():
            chunks = []
            async for chunk in agent.stream_completion(sample_messages):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())
        assert chunks == ["mock response"]

    def test_stream_fallback_async_client_false(self, mock_provider, sample_messages):
        """Falls back when _async_client is falsy."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_provider._async_client = False
        agent = AgentProvider(mock_provider)

        async def _collect():
            chunks = []
            async for chunk in agent.stream_completion(sample_messages):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())
        assert chunks == ["mock response"]

    def test_stream_with_async_client_fallback_on_error(self, mock_provider, sample_messages):
        """When streaming raises, falls back to non-streaming."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_provider._async_client = MagicMock()

        agent = AgentProvider(mock_provider)
        # Make _stream_anthropic raise
        with patch.object(
            agent,
            "_stream_anthropic",
            side_effect=RuntimeError("stream failed"),
        ):

            async def _collect():
                chunks = []
                async for chunk in agent.stream_completion(sample_messages):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_collect())
            assert chunks == ["mock response"]


class TestAgentProviderStreamAnthropic:
    """Test AgentProvider._stream_anthropic()."""

    def test_stream_anthropic_success(self, mock_provider, sample_messages):
        """Anthropic streaming yields text chunks."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_client = MagicMock()
        mock_provider._async_client = mock_client

        async def _fake_text_stream():
            for chunk in ["Hello", " World"]:
                yield chunk

        mock_stream = AsyncMock()
        mock_stream.text_stream = _fake_text_stream()

        # AsyncContextManager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream.return_value = mock_cm

        agent = AgentProvider(mock_provider)

        async def _collect():
            chunks = []
            async for chunk in agent._stream_anthropic(sample_messages):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())
        assert chunks == ["Hello", " World"]

    def test_stream_anthropic_error_raises(self, mock_provider, sample_messages):
        """Anthropic streaming error is re-raised."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_client = MagicMock()
        mock_provider._async_client = mock_client

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("API error"))
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream.return_value = mock_cm

        agent = AgentProvider(mock_provider)

        async def _collect():
            chunks = []
            async for chunk in agent._stream_anthropic(sample_messages):
                chunks.append(chunk)
            return chunks

        with pytest.raises(RuntimeError, match="API error"):
            asyncio.run(_collect())

    def test_stream_anthropic_no_system_messages(self, mock_provider):
        """Streaming without system messages uses default."""
        from animus_forge.agents.provider_wrapper import AgentProvider

        mock_client = MagicMock()
        mock_provider._async_client = mock_client

        async def _fake_text_stream():
            yield "response"

        mock_stream = AsyncMock()
        mock_stream.text_stream = _fake_text_stream()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_client.messages.stream.return_value = mock_cm

        agent = AgentProvider(mock_provider)
        messages = [{"role": "user", "content": "Hey"}]

        async def _collect():
            chunks = []
            async for chunk in agent._stream_anthropic(messages):
                chunks.append(chunk)
            return chunks

        asyncio.run(_collect())
        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."


class TestCreateAgentProvider:
    """Test create_agent_provider() factory."""

    @patch("animus_forge.config.get_settings")
    @patch("animus_forge.providers.anthropic_provider.AnthropicProvider")
    def test_create_anthropic(self, mock_cls, mock_settings):
        """Creates an AnthropicProvider-based agent."""
        from animus_forge.agents.provider_wrapper import create_agent_provider

        mock_settings.return_value.anthropic_api_key = "sk-test"
        mock_instance = MagicMock()
        mock_instance._initialized = True
        mock_cls.return_value = mock_instance

        agent = create_agent_provider("anthropic")
        assert agent.provider is mock_instance
        mock_cls.assert_called_once_with(api_key="sk-test")

    @patch("animus_forge.config.get_settings")
    @patch("animus_forge.providers.openai_provider.OpenAIProvider")
    def test_create_openai(self, mock_cls, mock_settings):
        """Creates an OpenAIProvider-based agent."""
        from animus_forge.agents.provider_wrapper import create_agent_provider

        mock_settings.return_value.openai_api_key = "sk-openai"
        mock_instance = MagicMock()
        mock_instance._initialized = True
        mock_cls.return_value = mock_instance

        agent = create_agent_provider("openai")
        assert agent.provider is mock_instance
        mock_cls.assert_called_once_with(api_key="sk-openai")

    def test_create_unknown_raises(self):
        """Unknown provider type raises ValueError."""
        from animus_forge.agents.provider_wrapper import create_agent_provider

        with pytest.raises(ValueError, match="Unknown provider type: llama"):
            create_agent_provider("llama")


# ===========================================================================
# 2. api_clients/resilience.py — uncovered paths
# ===========================================================================


class TestResilientCallRateLimitDenied:
    """Test rate-limit denial path in resilient_call."""

    def test_sync_rate_limit_denied(self):
        """Rate limit not acquired raises RuntimeError."""
        from animus_forge.api_clients.resilience import resilient_call

        mock_limiter = MagicMock()
        mock_limiter.acquire.return_value = False

        with patch(
            "animus_forge.api_clients.resilience.get_provider_limiter",
            return_value=mock_limiter,
        ):

            @resilient_call("openai")
            def my_func():
                return "result"

            with pytest.raises(RuntimeError, match="Rate limit exceeded"):
                my_func()

    def test_sync_bulkhead_denied(self):
        """Bulkhead not acquired raises RuntimeError."""
        from animus_forge.api_clients.resilience import resilient_call

        mock_limiter = MagicMock()
        mock_limiter.acquire.return_value = True

        mock_bh = MagicMock()
        mock_bh.acquire.return_value = False

        with (
            patch(
                "animus_forge.api_clients.resilience.get_provider_limiter",
                return_value=mock_limiter,
            ),
            patch(
                "animus_forge.api_clients.resilience.get_provider_bulkhead",
                return_value=mock_bh,
            ),
        ):

            @resilient_call("openai")
            def my_func():
                return "result"

            with pytest.raises(RuntimeError, match="Bulkhead full"):
                my_func()


class TestResilientCallAsyncDenied:
    """Test async denial paths."""

    def test_async_rate_limit_denied(self):
        """Async rate limit denial."""
        from animus_forge.api_clients.resilience import resilient_call_async

        mock_limiter = MagicMock()
        mock_limiter.acquire_async = AsyncMock(return_value=False)

        with patch(
            "animus_forge.api_clients.resilience.get_provider_limiter",
            return_value=mock_limiter,
        ):

            @resilient_call_async("openai")
            async def my_func():
                return "result"

            with pytest.raises(RuntimeError, match="Rate limit exceeded"):
                asyncio.run(my_func())

    def test_async_bulkhead_denied(self):
        """Async bulkhead denial."""
        from animus_forge.api_clients.resilience import resilient_call_async

        mock_limiter = MagicMock()
        mock_limiter.acquire_async = AsyncMock(return_value=True)

        mock_bh = MagicMock()
        mock_bh.acquire_async = AsyncMock(return_value=False)

        with (
            patch(
                "animus_forge.api_clients.resilience.get_provider_limiter",
                return_value=mock_limiter,
            ),
            patch(
                "animus_forge.api_clients.resilience.get_provider_bulkhead",
                return_value=mock_bh,
            ),
        ):

            @resilient_call_async("openai")
            async def my_func():
                return "result"

            with pytest.raises(RuntimeError, match="Bulkhead full"):
                asyncio.run(my_func())

    def test_async_no_rate_limit(self):
        """Async with rate_limit=False skips limiter."""
        from animus_forge.api_clients.resilience import resilient_call_async

        @resilient_call_async("openai", rate_limit=False)
        async def my_func():
            return "ok"

        result = asyncio.run(my_func())
        assert result == "ok"

    def test_async_no_bulkhead(self):
        """Async with bulkhead=False skips bulkhead."""
        from animus_forge.api_clients.resilience import resilient_call_async

        @resilient_call_async("openai", bulkhead=False)
        async def my_func():
            return "no-bh"

        result = asyncio.run(my_func())
        assert result == "no-bh"


class TestResilientContext:
    """Test _ResilientContext sync context manager."""

    def test_sync_context_enter_exit(self):
        """Sync context acquires and releases resources."""
        from animus_forge.api_clients.resilience import _ResilientContext

        mock_limiter = MagicMock()
        mock_bh = MagicMock()

        with (
            patch(
                "animus_forge.api_clients.resilience.get_provider_limiter",
                return_value=mock_limiter,
            ),
            patch(
                "animus_forge.api_clients.resilience.get_provider_bulkhead",
                return_value=mock_bh,
            ),
        ):
            ctx = _ResilientContext("openai", rate_limit=True, bulkhead=True, is_async=False)
            with ctx:
                mock_limiter.acquire.assert_called_once_with(wait=True)
                mock_bh.acquire.assert_called_once()
            mock_bh.release.assert_called_once()

    def test_sync_context_no_rate_limit(self):
        """Sync context with rate_limit=False."""
        from animus_forge.api_clients.resilience import _ResilientContext

        mock_limiter = MagicMock()

        with patch(
            "animus_forge.api_clients.resilience.get_provider_limiter",
            return_value=mock_limiter,
        ):
            ctx = _ResilientContext("openai", rate_limit=False, bulkhead=False, is_async=False)
            with ctx:
                pass
            mock_limiter.acquire.assert_not_called()

    def test_sync_context_no_bulkhead(self):
        """Sync context with bulkhead=False -- no release on exit."""
        from animus_forge.api_clients.resilience import _ResilientContext

        ctx = _ResilientContext("openai", rate_limit=False, bulkhead=False, is_async=False)
        with ctx:
            pass
        # _bh should be None, no release
        assert ctx._bh is None

    def test_sync_context_returns_false_on_exit(self):
        """__exit__ returns False (doesn't suppress exceptions)."""
        from animus_forge.api_clients.resilience import _ResilientContext

        ctx = _ResilientContext("openai", rate_limit=False, bulkhead=False, is_async=False)
        result = ctx.__exit__(None, None, None)
        assert result is False


class TestResilientContextAsync:
    """Test _ResilientContextAsync."""

    def test_async_context_enter_exit(self):
        """Async context acquires and releases resources."""
        from animus_forge.api_clients.resilience import _ResilientContextAsync

        mock_limiter = MagicMock()
        mock_limiter.acquire_async = AsyncMock(return_value=True)
        mock_bh = MagicMock()
        mock_bh.acquire_async = AsyncMock(return_value=True)
        mock_bh.release_async = AsyncMock()

        with (
            patch(
                "animus_forge.api_clients.resilience.get_provider_limiter",
                return_value=mock_limiter,
            ),
            patch(
                "animus_forge.api_clients.resilience.get_provider_bulkhead",
                return_value=mock_bh,
            ),
        ):

            async def _run():
                ctx = _ResilientContextAsync("openai", rate_limit=True, bulkhead=True)
                async with ctx:
                    pass
                return ctx

            asyncio.run(_run())
            mock_limiter.acquire_async.assert_awaited_once_with(wait=True)
            mock_bh.acquire_async.assert_awaited_once()
            mock_bh.release_async.assert_awaited_once()

    def test_async_context_no_rate_limit(self):
        """Async context with rate_limit=False skips limiter."""
        from animus_forge.api_clients.resilience import _ResilientContextAsync

        mock_limiter = MagicMock()
        mock_limiter.acquire_async = AsyncMock()

        with patch(
            "animus_forge.api_clients.resilience.get_provider_limiter",
            return_value=mock_limiter,
        ):

            async def _run():
                ctx = _ResilientContextAsync("openai", rate_limit=False, bulkhead=False)
                async with ctx:
                    pass

            asyncio.run(_run())
            mock_limiter.acquire_async.assert_not_awaited()

    def test_async_context_no_bulkhead_no_release(self):
        """Async context with bulkhead=False -- no release on exit."""
        from animus_forge.api_clients.resilience import _ResilientContextAsync

        async def _run():
            ctx = _ResilientContextAsync("openai", rate_limit=False, bulkhead=False)
            async with ctx:
                pass
            return ctx

        ctx = asyncio.run(_run())
        assert ctx._bh is None

    def test_async_context_aexit_returns_false(self):
        """__aexit__ returns False."""
        from animus_forge.api_clients.resilience import _ResilientContextAsync

        async def _run():
            ctx = _ResilientContextAsync("openai", rate_limit=False, bulkhead=False)
            result = await ctx.__aexit__(None, None, None)
            return result

        result = asyncio.run(_run())
        assert result is False


class TestResilientClientMixinContexts:
    """Test ResilientClientMixin.resilient_context / resilient_context_async."""

    def test_resilient_context_sync(self):
        """Mixin produces sync context manager."""
        from animus_forge.api_clients.resilience import (
            ResilientClientMixin,
            _ResilientContext,
        )

        class MyClient(ResilientClientMixin):
            PROVIDER = "github"

        client = MyClient()
        ctx = client.resilient_context()
        assert isinstance(ctx, _ResilientContext)
        assert ctx.provider == "github"

    def test_resilient_context_async(self):
        """Mixin produces async context manager."""
        from animus_forge.api_clients.resilience import (
            ResilientClientMixin,
            _ResilientContextAsync,
        )

        class MyClient(ResilientClientMixin):
            PROVIDER = "notion"

        client = MyClient()
        ctx = client.resilient_context_async()
        assert isinstance(ctx, _ResilientContextAsync)
        assert ctx.provider == "notion"

    def test_mixin_custom_flags(self):
        """Mixin passes rate_limit/bulkhead flags through."""
        from animus_forge.api_clients.resilience import ResilientClientMixin

        class MyClient(ResilientClientMixin):
            PROVIDER = "gmail"

        client = MyClient()
        ctx = client.resilient_context(rate_limit=False, bulkhead=False)
        assert ctx.rate_limit is False
        assert ctx.bulkhead is False


class TestGetProviderBulkhead:
    """Test get_provider_bulkhead with known/unknown providers."""

    def test_known_provider(self):
        """Known provider uses configured limits."""
        from animus_forge.api_clients.resilience import get_provider_bulkhead

        bh = get_provider_bulkhead("openai")
        assert bh is not None

    def test_unknown_provider_uses_defaults(self):
        """Unknown provider falls back to defaults."""
        from animus_forge.api_clients.resilience import get_provider_bulkhead

        bh = get_provider_bulkhead("unknown_provider")
        assert bh is not None


class TestGetAllProviderStatsError:
    """Test get_all_provider_stats error handling."""

    def test_stats_with_error(self):
        """Provider that raises gets an error entry."""
        from animus_forge.api_clients.resilience import get_all_provider_stats

        with patch(
            "animus_forge.ratelimit.provider.get_provider_limiter",
            side_effect=RuntimeError("broken"),
        ):
            stats = get_all_provider_stats()
            for provider_name in stats:
                assert "error" in stats[provider_name]


# ===========================================================================
# 3. api_clients/calendar_client.py — CalendarEvent + CalendarClient
# ===========================================================================


class TestCalendarEventToApiFormat:
    """Test CalendarEvent.to_api_format()."""

    def test_timed_event(self):
        """Timed event with all fields."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        start = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
        end = datetime(2026, 3, 1, 11, 0, 0, tzinfo=UTC)
        event = CalendarEvent(
            summary="Meeting",
            description="Sync up",
            location="Office",
            start=start,
            end=end,
            attendees=["alice@example.com", "bob@example.com"],
            reminders=[{"method": "popup", "minutes": 10}],
            recurrence=["RRULE:FREQ=WEEKLY"],
        )
        api = event.to_api_format()

        assert api["summary"] == "Meeting"
        assert api["description"] == "Sync up"
        assert api["location"] == "Office"
        assert "dateTime" in api["start"]
        assert api["start"]["timeZone"] == "UTC"
        assert "dateTime" in api["end"]
        assert len(api["attendees"]) == 2
        assert api["attendees"][0]["email"] == "alice@example.com"
        assert api["reminders"]["useDefault"] is False
        assert api["recurrence"] == ["RRULE:FREQ=WEEKLY"]

    def test_allday_event(self):
        """All-day event uses date format."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        start = datetime(2026, 3, 1, tzinfo=UTC)
        end = datetime(2026, 3, 2, tzinfo=UTC)
        event = CalendarEvent(
            summary="Holiday",
            all_day=True,
            start=start,
            end=end,
        )
        api = event.to_api_format()

        assert "date" in api["start"]
        assert api["start"]["date"] == "2026-03-01"
        assert "date" in api["end"]
        assert "dateTime" not in api.get("start", {})

    def test_no_start_end(self):
        """Event with no start/end timestamps."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        event = CalendarEvent(summary="No times")
        api = event.to_api_format()

        assert "start" not in api
        assert "end" not in api

    def test_no_attendees(self):
        """No attendees omits attendees key."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        event = CalendarEvent(summary="Solo")
        api = event.to_api_format()

        assert "attendees" not in api

    def test_no_reminders_uses_default(self):
        """Empty reminders enables useDefault."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        event = CalendarEvent(summary="Default reminders")
        api = event.to_api_format()

        assert api["reminders"]["useDefault"] is True

    def test_no_recurrence(self):
        """Empty recurrence omits recurrence key."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        event = CalendarEvent(summary="One-time")
        api = event.to_api_format()

        assert "recurrence" not in api

    def test_allday_no_start(self):
        """All-day event with start=None."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        event = CalendarEvent(summary="No start", all_day=True, end=datetime(2026, 3, 2))
        api = event.to_api_format()
        assert "start" not in api

    def test_allday_no_end(self):
        """All-day event with end=None."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        event = CalendarEvent(summary="No end", all_day=True, start=datetime(2026, 3, 1))
        api = event.to_api_format()
        assert "end" not in api


class TestCalendarEventFromApiResponse:
    """Test CalendarEvent.from_api_response()."""

    def test_timed_event(self, sample_api_event):
        """Parse timed event from API response."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        event = CalendarEvent.from_api_response(sample_api_event)

        assert event.id == "evt-123"
        assert event.summary == "Team Meeting"
        assert event.description == "Weekly sync"
        assert event.location == "Room 42"
        assert event.all_day is False
        assert event.start is not None
        assert event.end is not None
        assert event.attendees == ["alice@example.com", "bob@example.com"]
        assert len(event.reminders) == 1
        assert event.recurrence == ["RRULE:FREQ=WEEKLY"]
        assert event.status == "confirmed"
        assert event.html_link == "https://calendar.google.com/event/evt-123"

    def test_allday_event(self, sample_allday_event):
        """Parse all-day event."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        event = CalendarEvent.from_api_response(sample_allday_event)

        assert event.all_day is True
        assert event.start is not None
        assert event.start.tzinfo == UTC
        assert event.end is not None

    def test_minimal_event(self):
        """Parse event with minimal fields."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        data = {"id": "evt-min", "start": {}, "end": {}}
        event = CalendarEvent.from_api_response(data)

        assert event.id == "evt-min"
        assert event.summary == ""
        assert event.start is None
        assert event.end is None
        assert event.all_day is False
        assert event.attendees == []
        assert event.reminders == []

    def test_attendees_filter_empty_emails(self):
        """Attendees with missing email are filtered out."""
        from animus_forge.api_clients.calendar_client import CalendarEvent

        data = {
            "start": {},
            "end": {},
            "attendees": [
                {"email": "valid@example.com"},
                {"displayName": "No Email"},
                {},
            ],
        }
        event = CalendarEvent.from_api_response(data)
        assert event.attendees == ["valid@example.com"]


class TestCalendarClientInit:
    """Test CalendarClient initialization."""

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_init_with_credentials(self, mock_settings):
        """Init uses explicit credentials_path."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/default/creds.json"
        client = CalendarClient(credentials_path="/custom/creds.json")
        assert client.credentials_path == "/custom/creds.json"
        assert client.service is None
        assert client._authenticated is False

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_init_uses_settings_default(self, mock_settings):
        """Init falls back to settings credentials path."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/settings/creds.json"
        client = CalendarClient()
        assert client.credentials_path == "/settings/creds.json"

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_is_configured_true(self, mock_settings):
        """is_configured returns True when credentials_path set."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/some/path"
        client = CalendarClient()
        assert client.is_configured() is True

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_is_configured_false(self, mock_settings):
        """is_configured returns False when credentials_path is None."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = None
        client = CalendarClient()
        assert client.is_configured() is False


class TestCalendarClientUnauthenticated:
    """Test CalendarClient methods when not authenticated (service=None)."""

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_list_events_unauthenticated(self, mock_settings):
        """list_events returns empty list when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        assert client.list_events() == []

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_get_event_unauthenticated(self, mock_settings):
        """get_event returns None when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        assert client.get_event("evt-123") is None

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_create_event_unauthenticated(self, mock_settings):
        """create_event returns None when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient, CalendarEvent

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        event = CalendarEvent(summary="Test")
        assert client.create_event(event) is None

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_update_event_unauthenticated(self, mock_settings):
        """update_event returns None when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient, CalendarEvent

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        event = CalendarEvent(id="evt-123", summary="Test")
        assert client.update_event(event) is None

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_update_event_no_id(self, mock_settings):
        """update_event returns None when event has no ID."""
        from animus_forge.api_clients.calendar_client import CalendarClient, CalendarEvent

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        client.service = MagicMock()  # Pretend authenticated
        event = CalendarEvent(summary="No ID")
        assert client.update_event(event) is None

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_delete_event_unauthenticated(self, mock_settings):
        """delete_event returns False when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        assert client.delete_event("evt-123") is False

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_check_availability_unauthenticated(self, mock_settings):
        """check_availability returns empty list when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        now = datetime.now(UTC)
        assert client.check_availability(now, now + timedelta(hours=1)) == []

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_list_calendars_unauthenticated(self, mock_settings):
        """list_calendars returns empty list when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        assert client.list_calendars() == []

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_quick_add_unauthenticated(self, mock_settings):
        """quick_add returns None when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        assert client.quick_add("Lunch tomorrow") is None


class TestCalendarClientAuthenticate:
    """Test CalendarClient.authenticate()."""

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_authenticate_not_configured(self, mock_settings):
        """authenticate returns False when not configured."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = None
        client = CalendarClient()
        assert client.authenticate() is False

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_authenticate_exception(self, mock_settings):
        """authenticate returns False on exception."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/fake/path"
        client = CalendarClient()
        # Will fail because google libraries are mocked/not available
        result = client.authenticate()
        assert result is False


class TestCalendarClientGetUpcomingToday:
    """Test CalendarClient.get_upcoming_today()."""

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_get_upcoming_today_unauthenticated(self, mock_settings):
        """get_upcoming_today returns empty when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        assert client.get_upcoming_today() == []

    @patch("animus_forge.api_clients.calendar_client.get_settings")
    def test_get_tomorrow_unauthenticated(self, mock_settings):
        """get_tomorrow returns empty when not authenticated."""
        from animus_forge.api_clients.calendar_client import CalendarClient

        mock_settings.return_value.gmail_credentials_path = "/path"
        client = CalendarClient()
        assert client.get_tomorrow() == []
