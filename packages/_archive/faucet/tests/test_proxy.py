"""Tests for proxy rotator."""

from animus_faucet.config import ProxyConfig
from animus_faucet.proxy.rotator import ProxyRotator, _redact


class TestProxyRotator:
    def test_disabled(self):
        config = ProxyConfig(enabled=False)
        rotator = ProxyRotator(config)
        assert rotator.get_proxy() is None

    def test_provider_proxy(self):
        config = ProxyConfig(
            enabled=True,
            provider="brightdata",
            endpoint="proxy.example.com:8080",
            username="user",
            password="pass",
        )
        rotator = ProxyRotator(config)
        proxy = rotator.get_proxy()
        assert proxy is not None
        assert "proxy.example.com" in proxy

    def test_custom_proxy_list(self):
        config = ProxyConfig(enabled=True)
        rotator = ProxyRotator(config)
        rotator.set_proxy_list(
            [
                "http://proxy1:8080",
                "http://proxy2:8080",
                "http://proxy3:8080",
            ]
        )
        proxy = rotator.get_proxy()
        assert proxy in ["http://proxy1:8080", "http://proxy2:8080", "http://proxy3:8080"]

    def test_rotation(self):
        config = ProxyConfig(enabled=True, rotation_interval_seconds=0)
        rotator = ProxyRotator(config)
        rotator.set_proxy_list(["http://p1:80", "http://p2:80"])

        proxies = set()
        for _ in range(10):
            rotator._state.last_rotated_at = 0  # force rotation
            proxies.add(rotator.get_proxy())

        assert len(proxies) == 2

    def test_report_ban(self):
        config = ProxyConfig(enabled=True)
        rotator = ProxyRotator(config)
        rotator.set_proxy_list(["http://p1:80", "http://p2:80"])
        rotator.report_ban("http://p1:80")
        assert "http://p1:80" in rotator._state.banned_proxies
        assert rotator._state.ban_count == 1

    def test_banned_proxy_excluded_from_list(self):
        config = ProxyConfig(enabled=True)
        rotator = ProxyRotator(config)
        rotator.report_ban("http://bad:80")
        rotator.set_proxy_list(["http://bad:80", "http://good:80"])
        assert len(rotator._proxy_list) == 1

    def test_report_success(self):
        config = ProxyConfig(enabled=True)
        rotator = ProxyRotator(config)
        rotator.report_success()
        assert rotator._state.request_count == 1

    def test_stats(self):
        config = ProxyConfig(enabled=True)
        rotator = ProxyRotator(config)
        stats = rotator.stats
        assert stats["enabled"] is True
        assert stats["request_count"] == 0
        assert stats["ban_count"] == 0

    def test_empty_proxy_list(self):
        config = ProxyConfig(enabled=True)
        rotator = ProxyRotator(config)
        rotator.set_proxy_list([])
        assert rotator._get_from_list(rotate=True) is None

    def test_no_rotation_within_interval(self):
        config = ProxyConfig(enabled=True, rotation_interval_seconds=3600)
        rotator = ProxyRotator(config)
        rotator.set_proxy_list(["http://p1:80", "http://p2:80"])

        first = rotator.get_proxy()
        second = rotator.get_proxy()
        assert first == second  # shouldn't rotate

    def test_provider_non_brightdata(self):
        config = ProxyConfig(
            enabled=True,
            provider="oxylabs",
            endpoint="proxy.ox:8080",
            username="user",
            password="pass",
        )
        rotator = ProxyRotator(config)
        rotator._state.last_rotated_at = 0
        proxy = rotator.get_proxy()
        assert proxy is not None
        assert "session" not in proxy  # only brightdata adds session


class TestRedact:
    def test_redact_with_auth(self):
        assert _redact("http://user:pass@proxy:8080") == "http://***@proxy:8080"

    def test_redact_without_auth(self):
        assert _redact("http://proxy:8080") == "http://proxy:8080"

    def test_redact_none(self):
        assert _redact(None) == "<none>"

    def test_redact_no_scheme(self):
        assert _redact("user:pass@host") == "user:pass@host"
