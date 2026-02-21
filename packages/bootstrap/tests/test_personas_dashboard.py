"""Tests for the persona and routing dashboard pages."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from animus_bootstrap.runtime import reset_runtime


class TestPersonasDashboardPage:
    """Tests for the /personas page."""

    def setup_method(self) -> None:
        reset_runtime()

    def teardown_method(self) -> None:
        reset_runtime()

    def test_personas_page_returns_200(self) -> None:
        """GET /personas returns 200."""
        from animus_bootstrap.dashboard.app import app

        # Ensure no runtime is set (empty state)
        if hasattr(app.state, "runtime"):
            delattr(app.state, "runtime")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/personas")
        assert resp.status_code == 200

    def test_personas_page_has_table(self) -> None:
        """GET /personas contains the personas heading."""
        from animus_bootstrap.dashboard.app import app

        if hasattr(app.state, "runtime"):
            delattr(app.state, "runtime")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/personas")
        assert "Personas" in resp.text
        assert "No personas configured yet" in resp.text

    def test_personas_page_with_runtime_personas(self) -> None:
        """GET /personas shows personas from runtime.persona_engine."""
        from animus_bootstrap.dashboard.app import app
        from animus_bootstrap.personas.engine import PersonaProfile
        from animus_bootstrap.personas.voice import VoiceConfig

        rt = MagicMock()
        persona = PersonaProfile(
            name="TestBot",
            description="A test persona",
            voice=VoiceConfig(tone="formal"),
            is_default=True,
            knowledge_domains=["python"],
            channel_bindings={"discord": True},
        )
        rt.persona_engine.list_personas.return_value = [persona]
        app.state.runtime = rt

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/personas")
        assert resp.status_code == 200
        assert "TestBot" in resp.text
        assert "formal" in resp.text


class TestRoutingDashboardPage:
    """Tests for the /routing page."""

    def setup_method(self) -> None:
        reset_runtime()

    def teardown_method(self) -> None:
        reset_runtime()

    def test_routing_page_returns_200(self) -> None:
        """GET /routing returns 200."""
        from animus_bootstrap.dashboard.app import app

        if hasattr(app.state, "runtime"):
            delattr(app.state, "runtime")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/routing")
        assert resp.status_code == 200

    def test_routing_page_has_channels(self) -> None:
        """GET /routing shows all channel names."""
        from animus_bootstrap.dashboard.app import app

        if hasattr(app.state, "runtime"):
            delattr(app.state, "runtime")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/routing")
        assert "Channel Routing" in resp.text
        assert "webchat" in resp.text
        assert "telegram" in resp.text
        assert "discord" in resp.text
        assert "Default" in resp.text

    def test_routing_page_with_runtime_personas(self) -> None:
        """GET /routing shows persona assignments from runtime."""
        from animus_bootstrap.dashboard.app import app
        from animus_bootstrap.personas.engine import PersonaProfile

        rt = MagicMock()
        persona = PersonaProfile(
            name="SlackBot",
            channel_bindings={"slack": True},
        )
        rt.persona_engine.list_personas.return_value = [persona]
        app.state.runtime = rt

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/routing")
        assert resp.status_code == 200
        assert "SlackBot" in resp.text
