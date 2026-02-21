"""Tests for identity dashboard router."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from animus_bootstrap.dashboard.routers.identity_page import router
from animus_bootstrap.identity.manager import IdentityFileManager


@pytest.fixture()
def app_with_identity(tmp_path):
    """Create a test app with identity manager wired."""
    app = FastAPI()
    app.include_router(router)

    # Set up templates
    from pathlib import Path

    from fastapi.templating import Jinja2Templates

    template_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "animus_bootstrap"
        / "dashboard"
        / "templates"
    )
    templates = Jinja2Templates(directory=str(template_dir))
    app.state.templates = templates

    # Set up identity manager
    mgr = IdentityFileManager(tmp_path / "identity")
    mgr.generate_from_templates({"name": "TestUser", "timezone": "UTC"})

    runtime = MagicMock()
    runtime.identity_manager = mgr
    app.state.runtime = runtime

    return app, mgr


@pytest.fixture()
def client(app_with_identity):
    app, _ = app_with_identity
    return TestClient(app)


class TestIdentityPage:
    def test_get_identity_page(self, client):
        resp = client.get("/identity")
        assert resp.status_code == 200
        assert "CORE_VALUES.md" in resp.text
        assert "IDENTITY.md" in resp.text

    def test_identity_page_no_runtime(self):
        app = FastAPI()
        app.include_router(router)
        from pathlib import Path

        from fastapi.templating import Jinja2Templates

        template_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "animus_bootstrap"
            / "dashboard"
            / "templates"
        )
        templates = Jinja2Templates(directory=str(template_dir))
        app.state.templates = templates
        app.state.runtime = None

        c = TestClient(app)
        resp = c.get("/identity")
        assert resp.status_code == 200


class TestIdentityEdit:
    def test_get_edit_form(self, client):
        resp = client.get("/identity/edit/IDENTITY.md")
        assert resp.status_code == 200
        assert "textarea" in resp.text

    def test_edit_locked_file_shows_readonly(self, client):
        resp = client.get("/identity/edit/CORE_VALUES.md")
        assert resp.status_code == 200
        assert "readonly" in resp.text

    def test_edit_unknown_file(self, client):
        resp = client.get("/identity/edit/BAD.md")
        assert resp.status_code == 200
        assert "Unknown file" in resp.text


class TestIdentitySave:
    def test_save_editable_file(self, app_with_identity):
        app, mgr = app_with_identity
        c = TestClient(app)
        resp = c.put("/identity/IDENTITY.md", data={"content": "Updated content"})
        assert resp.status_code == 200
        assert mgr.read("IDENTITY.md") == "Updated content"

    def test_save_locked_file_via_dashboard(self, app_with_identity):
        app, mgr = app_with_identity
        c = TestClient(app)
        resp = c.put("/identity/CORE_VALUES.md", data={"content": "New values"})
        assert resp.status_code == 200
        # Dashboard CAN edit locked files (human edit)
        assert mgr.read("CORE_VALUES.md") == "New values"


class TestIdentityView:
    def test_view_file(self, client):
        resp = client.get("/identity/view/IDENTITY.md")
        assert resp.status_code == 200
        assert "IDENTITY.md" in resp.text
