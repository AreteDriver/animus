"""Additional coverage tests for api_clients resilience module."""

import sys

import pytest

sys.path.insert(0, "src")


class TestGetProviderConfigs:
    def test_with_settings(self):
        from animus_forge.api_clients.resilience import _get_provider_configs

        configs = _get_provider_configs()
        assert "openai" in configs
        assert "anthropic" in configs
        assert "github" in configs
        assert "notion" in configs
        assert "gmail" in configs
        assert "max_concurrent" in configs["openai"]

    def test_fallback_defaults(self):
        from animus_forge.api_clients.resilience import _get_provider_configs

        # Test that configs have expected structure
        configs = _get_provider_configs()
        for provider in ["openai", "anthropic", "github", "notion", "gmail"]:
            assert "max_concurrent" in configs[provider]
            assert "timeout" in configs[provider]


class TestResilientCall:
    def test_basic_decorator(self):
        from animus_forge.api_clients.resilience import resilient_call

        @resilient_call("openai")
        def my_func():
            return "result"

        assert my_func() == "result"

    def test_decorator_with_args(self):
        from animus_forge.api_clients.resilience import resilient_call

        @resilient_call("openai")
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_decorator_failure(self):
        from animus_forge.api_clients.resilience import resilient_call

        @resilient_call("openai")
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            fail()

    def test_no_rate_limit(self):
        from animus_forge.api_clients.resilience import resilient_call

        @resilient_call("openai", rate_limit=False)
        def my_func():
            return "ok"

        assert my_func() == "ok"

    def test_no_bulkhead(self):
        from animus_forge.api_clients.resilience import resilient_call

        @resilient_call("openai", bulkhead=False)
        def my_func():
            return "ok"

        assert my_func() == "ok"


class TestAsyncResilientCall:
    def test_basic(self):
        import asyncio

        from animus_forge.api_clients.resilience import resilient_call_async

        @resilient_call_async("openai")
        async def my_func():
            return "async_result"

        result = asyncio.run(my_func())
        assert result == "async_result"

    def test_failure(self):
        import asyncio

        from animus_forge.api_clients.resilience import resilient_call_async

        @resilient_call_async("openai")
        async def fail():
            raise ValueError("async boom")

        with pytest.raises(ValueError):
            asyncio.run(fail())


class TestGetAllProviderStats:
    def test_stats(self):
        from animus_forge.api_clients.resilience import get_all_provider_stats

        stats = get_all_provider_stats()
        assert isinstance(stats, dict)


class TestResilientClientMixin:
    def test_mixin(self):
        from animus_forge.api_clients.resilience import ResilientClientMixin

        class MyClient(ResilientClientMixin):
            _provider_name = "openai"

        client = MyClient()
        assert client._provider_name == "openai"
