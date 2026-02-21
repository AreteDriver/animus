"""Tests for proposals dashboard router."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from animus_bootstrap.dashboard.routers.proposals import router
from animus_bootstrap.identity.manager import IdentityFileManager
from animus_bootstrap.intelligence.tools.builtin.improvement_store import ImprovementStore


@pytest.fixture()
def app_with_proposals(tmp_path):
    """Create a test app with improvement store and identity manager."""
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

    mgr = IdentityFileManager(tmp_path / "identity")
    mgr.generate_from_templates({"name": "TestUser", "timezone": "UTC"})

    store = ImprovementStore(tmp_path / "improvements.db")

    runtime = MagicMock()
    runtime._improvement_store = store
    runtime.identity_manager = mgr
    app.state.runtime = runtime

    return app, store, mgr


@pytest.fixture()
def client(app_with_proposals):
    app, _, _ = app_with_proposals
    return TestClient(app)


class TestProposalsPage:
    def test_empty_proposals_page(self, client):
        resp = client.get("/proposals")
        assert resp.status_code == 200
        assert "No pending proposals" in resp.text

    def test_proposals_with_pending(self, app_with_proposals):
        app, store, _ = app_with_proposals
        store.save(
            {
                "area": "identity:CONTEXT.md",
                "description": "Update context",
                "status": "proposed",
                "timestamp": "2026-02-21T00:00:00",
                "patch": "New context content",
            }
        )
        c = TestClient(app)
        resp = c.get("/proposals")
        assert resp.status_code == 200
        assert "Update context" in resp.text
        assert "#1" in resp.text


class TestApproveProposal:
    def test_approve_writes_file(self, app_with_proposals):
        app, store, mgr = app_with_proposals
        store.save(
            {
                "area": "identity:CONTEXT.md",
                "description": "Update context",
                "status": "proposed",
                "timestamp": "2026-02-21T00:00:00",
                "patch": "Approved content here",
            }
        )

        c = TestClient(app)
        resp = c.post("/proposals/1/approve")
        assert resp.status_code == 200
        assert "Approved" in resp.text

        # Verify file was updated
        assert mgr.read("CONTEXT.md") == "Approved content here"

        # Verify status updated
        p = store.get(1)
        assert p["status"] == "approved"

    def test_approve_locked_file_blocked(self, app_with_proposals):
        app, store, _ = app_with_proposals
        store.save(
            {
                "area": "identity:CORE_VALUES.md",
                "description": "Try to change values",
                "status": "proposed",
                "timestamp": "2026-02-21T00:00:00",
                "patch": "hacked",
            }
        )

        c = TestClient(app)
        resp = c.post("/proposals/1/approve")
        assert resp.status_code == 200
        assert "locked" in resp.text.lower()

    def test_approve_nonexistent_proposal(self, client):
        resp = client.post("/proposals/999/approve")
        assert resp.status_code == 200
        assert "not found" in resp.text.lower()


class TestRejectProposal:
    def test_reject_updates_status(self, app_with_proposals):
        app, store, _ = app_with_proposals
        store.save(
            {
                "area": "identity:GOALS.md",
                "description": "Change goals",
                "status": "proposed",
                "timestamp": "2026-02-21T00:00:00",
            }
        )

        c = TestClient(app)
        resp = c.post("/proposals/1/reject")
        assert resp.status_code == 200
        assert "Rejected" in resp.text

        p = store.get(1)
        assert p["status"] == "rejected"

    def test_reject_nonexistent_proposal(self, client):
        resp = client.post("/proposals/999/reject")
        assert resp.status_code == 200
        assert "not found" in resp.text.lower()
