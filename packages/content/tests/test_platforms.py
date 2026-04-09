"""Tests for platform adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_content.models import Article, ContentPlatform
from animus_content.platforms.medium import MediumPlatform
from animus_content.platforms.mirror import MirrorPlatform
from animus_content.platforms.paragraph import ParagraphPlatform


class TestBasePlatform:
    def test_is_configured_with_key(self) -> None:
        p = MirrorPlatform(api_key="test-key")
        assert p.is_configured() is True

    def test_is_configured_without_key(self) -> None:
        p = MirrorPlatform()
        assert p.is_configured() is False

    def test_extra_kwargs(self) -> None:
        p = MirrorPlatform(api_key="k", custom="value")
        assert p.extra["custom"] == "value"


class TestMirrorPlatform:
    @pytest.fixture
    def mirror(self) -> MirrorPlatform:
        return MirrorPlatform(api_key="test-key")

    @pytest.fixture
    def article(self) -> Article:
        return Article(id="art1", title="Test", body="Body", tags=["crypto"])

    @pytest.mark.asyncio
    async def test_publish_not_configured(self, article: Article) -> None:
        p = MirrorPlatform()
        result = await p.publish(article)
        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_publish_success(self, mirror: MirrorPlatform, article: Article) -> None:
        with patch("animus_content.platforms.mirror.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"url": "https://mirror.xyz/test/art1"}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            result = await mirror.publish(article)
            assert result.success is True
            assert "mirror.xyz" in result.published_url

    @pytest.mark.asyncio
    async def test_publish_http_error(self, mirror: MirrorPlatform, article: Article) -> None:
        with patch("animus_content.platforms.mirror.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("fail"))

            result = await mirror.publish(article)
            assert result.success is False
            assert "fail" in result.error

    @pytest.mark.asyncio
    async def test_fetch_earnings_not_configured(self) -> None:
        p = MirrorPlatform()
        records = await p.fetch_earnings("art1")
        assert records == []

    @pytest.mark.asyncio
    async def test_fetch_earnings_success(self, mirror: MirrorPlatform) -> None:
        with patch("animus_content.platforms.mirror.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "earnings": [{"amount": 0.1, "token": "ETH", "amount_usd": 300.0}]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            records = await mirror.fetch_earnings("art1")
            assert len(records) == 1
            assert records[0].amount == 0.1

    @pytest.mark.asyncio
    async def test_fetch_earnings_http_error(self, mirror: MirrorPlatform) -> None:
        with patch("animus_content.platforms.mirror.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("fail"))

            records = await mirror.fetch_earnings("art1")
            assert records == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, mirror: MirrorPlatform) -> None:
        with patch("animus_content.platforms.mirror.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_resp)

            assert await mirror.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mirror: MirrorPlatform) -> None:
        with patch("animus_content.platforms.mirror.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("down"))

            assert await mirror.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_non_200(self, mirror: MirrorPlatform) -> None:
        with patch("animus_content.platforms.mirror.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_client.get = AsyncMock(return_value=mock_resp)

            assert await mirror.health_check() is False

    def test_default_base_url(self) -> None:
        p = MirrorPlatform()
        assert p.base_url == "https://mirror.xyz/api"

    def test_custom_base_url(self) -> None:
        p = MirrorPlatform(base_url="https://custom.api")
        assert p.base_url == "https://custom.api"


class TestParagraphPlatform:
    @pytest.fixture
    def paragraph(self) -> ParagraphPlatform:
        return ParagraphPlatform(api_key="test-key")

    @pytest.fixture
    def article(self) -> Article:
        return Article(id="art1", title="Test", body="Body", tags=["web3"])

    @pytest.mark.asyncio
    async def test_publish_not_configured(self, article: Article) -> None:
        p = ParagraphPlatform()
        result = await p.publish(article)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_publish_success(self, paragraph: ParagraphPlatform, article: Article) -> None:
        with patch("animus_content.platforms.paragraph.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"url": "https://paragraph.xyz/test/art1"}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            result = await paragraph.publish(article)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_publish_http_error(self, paragraph: ParagraphPlatform, article: Article) -> None:
        with patch("animus_content.platforms.paragraph.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("fail"))

            result = await paragraph.publish(article)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_fetch_earnings_not_configured(self) -> None:
        p = ParagraphPlatform()
        assert await p.fetch_earnings("art1") == []

    @pytest.mark.asyncio
    async def test_fetch_earnings_success(self, paragraph: ParagraphPlatform) -> None:
        with patch("animus_content.platforms.paragraph.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "earnings": [{"amount": 5.0, "token": "MATIC", "amount_usd": 2.5}]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            records = await paragraph.fetch_earnings("art1")
            assert len(records) == 1

    @pytest.mark.asyncio
    async def test_fetch_earnings_http_error(self, paragraph: ParagraphPlatform) -> None:
        with patch("animus_content.platforms.paragraph.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("fail"))

            assert await paragraph.fetch_earnings("art1") == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, paragraph: ParagraphPlatform) -> None:
        with patch("animus_content.platforms.paragraph.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_resp)

            assert await paragraph.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, paragraph: ParagraphPlatform) -> None:
        with patch("animus_content.platforms.paragraph.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("down"))

            assert await paragraph.health_check() is False

    def test_default_base_url(self) -> None:
        p = ParagraphPlatform()
        assert "paragraph.xyz" in p.base_url

    def test_platform_type(self) -> None:
        p = ParagraphPlatform()
        assert p.platform == ContentPlatform.PARAGRAPH


class TestMediumPlatform:
    @pytest.fixture
    def medium(self) -> MediumPlatform:
        return MediumPlatform(api_key="test-key", user_id="user123")

    @pytest.fixture
    def article(self) -> Article:
        return Article(id="art1", title="Test", body="Body", tags=["crypto"])

    @pytest.mark.asyncio
    async def test_publish_not_configured(self, article: Article) -> None:
        p = MediumPlatform()
        result = await p.publish(article)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_publish_success(self, medium: MediumPlatform, article: Article) -> None:
        with patch("animus_content.platforms.medium.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": {"url": "https://medium.com/test/art1"}}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            result = await medium.publish(article)
            assert result.success is True
            assert "medium.com" in result.published_url

    @pytest.mark.asyncio
    async def test_publish_resolves_user_id(self, article: Article) -> None:
        medium = MediumPlatform(api_key="test-key")
        with patch("animus_content.platforms.medium.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            # First call resolves user ID, second publishes
            me_resp = MagicMock()
            me_resp.json.return_value = {"data": {"id": "resolved_id"}}
            me_resp.raise_for_status = MagicMock()

            post_resp = MagicMock()
            post_resp.json.return_value = {"data": {"url": "https://medium.com/art1"}}
            post_resp.raise_for_status = MagicMock()

            mock_client.get = AsyncMock(return_value=me_resp)
            mock_client.post = AsyncMock(return_value=post_resp)

            result = await medium.publish(article)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_publish_user_id_resolve_fails(self, article: Article) -> None:
        medium = MediumPlatform(api_key="test-key")
        with patch("animus_content.platforms.medium.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            me_resp = MagicMock()
            me_resp.json.return_value = {"data": {}}
            me_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=me_resp)

            result = await medium.publish(article)
            assert result.success is False
            assert "user ID" in result.error

    @pytest.mark.asyncio
    async def test_publish_http_error(self, medium: MediumPlatform, article: Article) -> None:
        with patch("animus_content.platforms.medium.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("fail"))

            result = await medium.publish(article)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_fetch_earnings_returns_empty(self, medium: MediumPlatform) -> None:
        records = await medium.fetch_earnings("art1")
        assert records == []

    @pytest.mark.asyncio
    async def test_health_check_200(self, medium: MediumPlatform) -> None:
        with patch("animus_content.platforms.medium.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_resp)

            assert await medium.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_401_still_reachable(self, medium: MediumPlatform) -> None:
        with patch("animus_content.platforms.medium.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_client.get = AsyncMock(return_value=mock_resp)

            assert await medium.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, medium: MediumPlatform) -> None:
        with patch("animus_content.platforms.medium.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("down"))

            assert await medium.health_check() is False

    def test_default_base_url(self) -> None:
        p = MediumPlatform()
        assert "medium.com" in p.base_url

    def test_platform_type(self) -> None:
        p = MediumPlatform()
        assert p.platform == ContentPlatform.MEDIUM
