"""Tests for Web Push: subscription store, sender, key gen, and router."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.config.schema import AnimusConfig
from animus_bootstrap.dashboard.app import app
from animus_bootstrap.intelligence import push_sender
from animus_bootstrap.intelligence.push_store import PushSubscriptionStore


def _crypto_available() -> bool:
    try:
        push_sender.generate_vapid_keys()
        return True
    except BaseException:  # noqa: BLE001 — native binding may panic
        return False


CRYPTO_OK = _crypto_available()


# ------------------------------------------------------------------
# PushSubscriptionStore
# ------------------------------------------------------------------


class TestPushStore:
    def test_add_list_remove(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = PushSubscriptionStore(tmp_path / "push.db")
        try:
            sub = {"endpoint": "https://push.example/abc", "keys": {"p256dh": "x", "auth": "y"}}
            store.add(sub)
            assert store.count() == 1
            assert store.all()[0]["endpoint"] == "https://push.example/abc"

            # Re-adding the same endpoint is idempotent.
            store.add(sub)
            assert store.count() == 1

            store.remove("https://push.example/abc")
            assert store.count() == 0
        finally:
            store.close()

    def test_add_requires_endpoint(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = PushSubscriptionStore(tmp_path / "push.db")
        try:
            with pytest.raises(ValueError):
                store.add({"keys": {}})
        finally:
            store.close()


# ------------------------------------------------------------------
# PushSender
# ------------------------------------------------------------------


class TestPushSender:
    def _store_with(self, tmp_path, *endpoints: str) -> PushSubscriptionStore:  # type: ignore[no-untyped-def]
        store = PushSubscriptionStore(tmp_path / "push.db")
        for ep in endpoints:
            store.add({"endpoint": ep, "keys": {"p256dh": "x", "auth": "y"}})
        return store

    def test_send_no_pywebpush_is_noop(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Ensure pywebpush import fails.
        monkeypatch.setitem(sys.modules, "pywebpush", None)
        store = self._store_with(tmp_path, "https://push.example/a")
        try:
            sender = push_sender.PushSender(store, "PRIVKEY", "mailto:a@b.c")
            assert sender.send("Title", "Body") == 0
        finally:
            store.close()

    def test_send_delivers_and_prunes(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Build a fake pywebpush module.
        fake = types.ModuleType("pywebpush")

        class WebPushException(Exception):
            def __init__(self, message: str, status: int) -> None:
                super().__init__(message)
                self.response = types.SimpleNamespace(status_code=status)

        def webpush(*, subscription_info, data, vapid_private_key, vapid_claims):  # type: ignore[no-untyped-def]
            endpoint = subscription_info["endpoint"]
            if endpoint.endswith("gone"):
                raise WebPushException("gone", 410)
            if endpoint.endswith("err"):
                raise WebPushException("server error", 500)
            return MagicMock()

        fake.WebPushException = WebPushException  # type: ignore[attr-defined]
        fake.webpush = webpush  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pywebpush", fake)

        store = self._store_with(
            tmp_path,
            "https://push.example/ok",
            "https://push.example/gone",
            "https://push.example/err",
        )
        try:
            sender = push_sender.PushSender(store, "PRIVKEY", "mailto:a@b.c")
            sent = sender.send("Title", "Body", url="/x")
            assert sent == 1
            # Only the 410 (expired) subscription is pruned; the 500 stays.
            endpoints = {s["endpoint"] for s in store.all()}
            assert endpoints == {"https://push.example/ok", "https://push.example/err"}
        finally:
            store.close()


# ------------------------------------------------------------------
# VAPID key helpers
# ------------------------------------------------------------------


class TestVapidKeys:
    def test_ensure_keys_idempotent_when_present(self) -> None:
        cfg = AnimusConfig()
        cfg.services.vapid_private_key = "PRIV"
        cfg.services.vapid_public_key = "PUB"
        manager = MagicMock()
        priv, pub = push_sender.ensure_vapid_keys(cfg, manager)
        assert (priv, pub) == ("PRIV", "PUB")
        manager.save.assert_not_called()

    @pytest.mark.skipif(not CRYPTO_OK, reason="cryptography native binding unavailable")
    def test_generate_and_persist(self) -> None:
        cfg = AnimusConfig()
        manager = MagicMock()
        priv, pub = push_sender.ensure_vapid_keys(cfg, manager)
        assert "BEGIN PRIVATE KEY" in priv
        assert pub  # base64url public key
        assert cfg.services.vapid_public_key == pub
        manager.save.assert_called_once_with(cfg)


# ------------------------------------------------------------------
# Push router
# ------------------------------------------------------------------


@pytest.fixture()
def restore_push_store() -> Iterator[None]:
    had = hasattr(app.state, "push_store")
    original = getattr(app.state, "push_store", None)
    try:
        yield
    finally:
        if had:
            app.state.push_store = original
        elif hasattr(app.state, "push_store"):
            delattr(app.state, "push_store")


class TestPushRouter:
    def test_subscribe_and_unsubscribe(self, tmp_path, restore_push_store: None) -> None:  # type: ignore[no-untyped-def]
        store = PushSubscriptionStore(tmp_path / "push.db")
        app.state.push_store = store
        client = TestClient(app)
        try:
            sub = {"endpoint": "https://push.example/z", "keys": {"p256dh": "x", "auth": "y"}}
            resp = client.post("/api/push/subscribe", json={"subscription": sub})
            assert resp.status_code == 200
            assert store.count() == 1

            resp = client.post("/api/push/unsubscribe", json={"endpoint": "https://push.example/z"})
            assert resp.status_code == 200
            assert store.count() == 0
        finally:
            store.close()

    def test_subscribe_unavailable_without_store(self, restore_push_store: None) -> None:
        app.state.push_store = None
        client = TestClient(app)
        resp = client.post("/api/push/subscribe", json={"subscription": {"endpoint": "e"}})
        assert resp.status_code == 503

    def test_unsubscribe_unavailable_without_store(self, restore_push_store: None) -> None:
        app.state.push_store = None
        client = TestClient(app)
        resp = client.post("/api/push/unsubscribe", json={"endpoint": "e"})
        assert resp.status_code == 503

    def test_subscribe_rejects_missing_endpoint(self, tmp_path, restore_push_store: None) -> None:  # type: ignore[no-untyped-def]
        store = PushSubscriptionStore(tmp_path / "push.db")
        app.state.push_store = store
        client = TestClient(app)
        try:
            resp = client.post("/api/push/subscribe", json={"subscription": {"keys": {}}})
            assert resp.status_code == 400
        finally:
            store.close()

    def test_vapid_public_key_returns_configured_key(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        original = app.state.config
        cfg = AnimusConfig()
        cfg.services.vapid_private_key = "PRIV"
        cfg.services.vapid_public_key = "PUBKEY123"
        app.state.config = cfg
        client = TestClient(app)
        try:
            resp = client.get("/api/push/vapid-public-key")
            assert resp.status_code == 200
            assert resp.json()["publicKey"] == "PUBKEY123"
        finally:
            app.state.config = original
