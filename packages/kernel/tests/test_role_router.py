"""Tests for role-based provider routing."""

from __future__ import annotations

from animus_kernel.agents.supervisor import AgentRole
from animus_kernel.providers.manager import ProviderManager
from animus_kernel.providers.mock_provider import MockProvider
from animus_kernel.providers.role_router import RoleRouter, RoleRoutingConfig


class TestRoleRouterOffline:
    def test_builder_routes_to_ollama_offline(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        pm = ProviderManager()
        pm.register("ollama", provider=MockProvider(), set_default=True)
        router = RoleRouter(pm)
        decision = router.route(AgentRole.BUILDER, "implement oauth")
        assert decision.provider_name == "ollama"
        assert decision.model and "hermes" in decision.model
        assert "role=builder" in decision.reason

    def test_planner_routes_to_qwen_offline(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        pm = ProviderManager()
        pm.register("ollama", provider=MockProvider(), set_default=True)
        router = RoleRouter(pm)
        decision = router.route(AgentRole.PLANNER, "design api")
        assert decision.provider_name == "ollama"
        assert decision.model and "qwen2.5" in decision.model

    def test_unmapped_role_fallback_to_tier_router(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        pm = ProviderManager()
        pm.register("ollama", provider=MockProvider(), set_default=True)
        router = RoleRouter(pm)
        decision = router.route(AgentRole.TESTER, "write tests")
        assert decision.provider_name == "ollama"


class TestRoleRouterCloud:
    def test_builder_routes_to_anthropic_cloud(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        pm = ProviderManager()
        anthropic = MockProvider()
        pm.register("anthropic", provider=anthropic, set_default=True)
        router = RoleRouter(pm)
        decision = router.route(AgentRole.BUILDER, "implement oauth")
        assert decision.provider_name == "anthropic"
        assert decision.model is None

    def test_planner_routes_to_anthropic_cloud(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        pm = ProviderManager()
        anthropic = MockProvider()
        pm.register("anthropic", provider=anthropic, set_default=True)
        router = RoleRouter(pm)
        decision = router.route(AgentRole.PLANNER, "design api")
        assert decision.provider_name == "anthropic"

    def test_tester_routes_to_haiku_cloud(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        pm = ProviderManager()
        anthropic = MockProvider()
        pm.register("anthropic", provider=anthropic, set_default=True)
        router = RoleRouter(pm)
        decision = router.route(AgentRole.TESTER, "write tests")
        assert decision.provider_name == "anthropic"
        assert decision.model == "claude-3-5-haiku-20241022"


class TestRoleRouterConfig:
    def test_custom_role_config(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        pm = ProviderManager()
        pm.register("mock", provider=MockProvider(), set_default=True)
        config = RoleRoutingConfig(
            offline_defaults={
                AgentRole.BUILDER: ("mock", "custom-model"),
            }
        )
        router = RoleRouter(pm, role_config=config)
        decision = router.route(AgentRole.BUILDER, "do work")
        assert decision.provider_name == "mock"
        assert decision.model == "custom-model"

    def test_cloud_defaults_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        pm = ProviderManager()
        pm.register("openai", provider=MockProvider(), set_default=True)
        config = RoleRoutingConfig(
            cloud_defaults={
                AgentRole.BUILDER: ("openai", "gpt-4o"),
            }
        )
        router = RoleRouter(pm, role_config=config)
        decision = router.route(AgentRole.BUILDER, "do work")
        assert decision.provider_name == "openai"
        assert decision.model == "gpt-4o"
