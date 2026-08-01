"""Tests for the Cognitive Event Ledger dashboard router."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.dashboard.app import app
from animus_bootstrap.ledger import IntegrityChain, LedgerEvent, LedgerStore


@pytest.fixture()
def client() -> TestClient:
    """TestClient wired with a temporary LedgerStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "ledger.db"
        store = LedgerStore(db_path=db)
        app.state.ledger_store = store
        tc = TestClient(app)
        # Prime CSRF cookie via a GET request
        tc.get("/ledger")
        yield tc
        del app.state.ledger_store


def _csrf_headers(client: TestClient) -> dict[str, str]:
    """Build headers with CSRF token from cookie."""
    cookie = client.cookies.get("animus_csrf", "")
    return {"X-CSRF-Token": cookie}


def _valid_event_dict(event_id: str = "evt-router-001", **overrides: Any) -> dict:
    payload = {"msg": "hello"}
    base = {
        "event_id": event_id,
        "event_type": "created",
        "object_id": "obj-router",
        "object_version": 1,
        "principal": "user-test",
        "workspace_id": "ws-test",
        "payload": payload,
        "integrity_hash": IntegrityChain.hash_payload(payload),
    }
    base.update(overrides)
    if "payload" in overrides:
        base["integrity_hash"] = IntegrityChain.hash_payload(overrides["payload"])
    return base


class TestLedgerPage:
    """HTML page rendering."""

    def test_ledger_page_returns_200(self, client: TestClient) -> None:
        response = client.get("/ledger")
        assert response.status_code == 200
        assert "Cognitive Event Ledger" in response.text


class TestListEvents:
    """GET /api/ledger/events"""

    def test_empty_list(self, client: TestClient) -> None:
        response = client.get("/api/ledger/events")
        assert response.status_code == 200
        data = response.json()
        assert data["events"] == []
        assert data["total"] == 0

    def test_lists_events(self, client: TestClient) -> None:
        store = app.state.ledger_store
        store.append(LedgerEvent.model_validate(_valid_event_dict("evt-list-001")))
        response = client.get("/api/ledger/events")
        assert response.status_code == 200
        data = response.json()
        assert len(data["events"]) == 1
        assert data["total"] == 1

    def test_filter_by_object_id(self, client: TestClient) -> None:
        store = app.state.ledger_store
        store.append(LedgerEvent.model_validate(_valid_event_dict("evt-f-001", object_id="obj-a")))
        store.append(LedgerEvent.model_validate(_valid_event_dict("evt-f-002", object_id="obj-b")))
        response = client.get("/api/ledger/events?object_id=obj-a")
        data = response.json()
        assert len(data["events"]) == 1
        assert data["events"][0]["object_id"] == "obj-a"

    def test_filter_by_event_type(self, client: TestClient) -> None:
        store = app.state.ledger_store
        store.append(
            LedgerEvent.model_validate(_valid_event_dict("evt-ft-001", event_type="created"))
        )
        store.append(
            LedgerEvent.model_validate(_valid_event_dict("evt-ft-002", event_type="updated"))
        )
        response = client.get("/api/ledger/events?event_type=created")
        data = response.json()
        assert len(data["events"]) == 1
        assert data["events"][0]["event_type"] == "created"

    def test_pagination_limit_offset(self, client: TestClient) -> None:
        store = app.state.ledger_store
        for i in range(5):
            store.append(LedgerEvent.model_validate(_valid_event_dict(f"evt-p-{i}")))
        response = client.get("/api/ledger/events?limit=2&offset=0")
        data = response.json()
        assert len(data["events"]) == 2


class TestGetEvent:
    """GET /api/ledger/events/{event_id}"""

    def test_get_existing(self, client: TestClient) -> None:
        store = app.state.ledger_store
        store.append(LedgerEvent.model_validate(_valid_event_dict("evt-get-001")))
        response = client.get("/api/ledger/events/evt-get-001")
        assert response.status_code == 200
        data = response.json()
        assert data["event_id"] == "evt-get-001"

    def test_get_missing_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/ledger/events/evt-missing")
        assert response.status_code == 404


class TestAppendEvent:
    """POST /api/ledger/events"""

    def test_append_valid_event(self, client: TestClient) -> None:
        payload = _valid_event_dict("evt-post-001")
        response = client.post("/api/ledger/events", json=payload, headers=_csrf_headers(client))
        assert response.status_code == 201
        data = response.json()
        assert data["event_id"] == "evt-post-001"
        assert data["chain_hash"] is not None

    def test_append_invalid_event_returns_422(self, client: TestClient) -> None:
        payload = {"event_id": "bad"}  # missing required fields
        response = client.post("/api/ledger/events", json=payload, headers=_csrf_headers(client))
        assert response.status_code == 422

    def test_append_duplicate_returns_409(self, client: TestClient) -> None:
        payload = _valid_event_dict("evt-dup-post")
        response1 = client.post("/api/ledger/events", json=payload, headers=_csrf_headers(client))
        assert response1.status_code == 201
        response2 = client.post("/api/ledger/events", json=payload, headers=_csrf_headers(client))
        # Duplicate event_id triggers IntegrityError → 409 Conflict
        assert response2.status_code == 409


class TestObjectChain:
    """GET /api/ledger/objects/{object_id}/chain"""

    def test_chain_for_object(self, client: TestClient) -> None:
        store = app.state.ledger_store
        for i in range(3):
            store.append(
                LedgerEvent.model_validate(
                    _valid_event_dict(f"evt-ch-{i}", object_id="obj-chain-test")
                )
            )
        response = client.get("/api/ledger/objects/obj-chain-test/chain")
        assert response.status_code == 200
        data = response.json()
        assert data["object_id"] == "obj-chain-test"
        assert data["count"] == 3
        assert data["integrity_valid"] is True

    def test_chain_for_missing_object(self, client: TestClient) -> None:
        response = client.get("/api/ledger/objects/obj-nope/chain")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["integrity_valid"] is True


class TestVerifyLedger:
    """GET /api/ledger/verify"""

    def test_verify_empty(self, client: TestClient) -> None:
        response = client.get("/api/ledger/verify")
        assert response.status_code == 200
        data = response.json()
        assert data["integrity_valid"] is True

    def test_verify_after_appends(self, client: TestClient) -> None:
        store = app.state.ledger_store
        store.append(LedgerEvent.model_validate(_valid_event_dict("evt-v-001")))
        response = client.get("/api/ledger/verify")
        data = response.json()
        assert data["integrity_valid"] is True
