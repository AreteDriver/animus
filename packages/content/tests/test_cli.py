"""Tests for the CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from animus_content.cli import app
from animus_content.config import ContentConfig
from animus_content.models import (
    Article,
    ContentPlatform,
    ContentStatus,
    ContentTopic,
    PublishResult,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_cli_config(tmp_path: Path) -> ContentConfig:
    """Override config loading for CLI tests."""
    config = ContentConfig(data_dir=tmp_path / "content")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    with patch("animus_content.cli._load_config", return_value=config):
        yield config


class TestGenerateCommand:
    def test_generate(self, mock_cli_config: ContentConfig) -> None:
        mock_article = Article(
            id="gen1",
            title="CLI Article",
            word_count=500,
            platform=ContentPlatform.MIRROR,
            status=ContentStatus.GENERATED,
        )
        with patch("animus_content.cli.ContentGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_article)
            mock_gen_cls.return_value = mock_gen

            result = runner.invoke(app, ["generate", "test topic"])
            assert result.exit_code == 0
            assert "gen1" in result.output

    def test_generate_with_options(self, mock_cli_config: ContentConfig) -> None:
        mock_article = Article(
            id="gen2",
            title="Custom",
            word_count=1500,
            platform=ContentPlatform.MEDIUM,
            status=ContentStatus.GENERATED,
        )
        with patch("animus_content.cli.ContentGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_article)
            mock_gen_cls.return_value = mock_gen

            result = runner.invoke(
                app, ["generate", "test", "--platform", "medium", "--word-count", "1500"]
            )
            assert result.exit_code == 0


class TestPublishCommand:
    def test_publish_article_not_found(self, mock_cli_config: ContentConfig) -> None:
        result = runner.invoke(app, ["publish", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_publish_success(self, mock_cli_config: ContentConfig) -> None:
        from animus_content.store import ContentStore

        store = ContentStore(mock_cli_config.data_dir)
        store.save_article(Article(id="pub1", title="Test", status=ContentStatus.GENERATED))

        with patch("animus_content.cli.Publisher") as mock_pub_cls:
            mock_pub = MagicMock()
            mock_pub.publish = AsyncMock(
                return_value=PublishResult(
                    article_id="pub1",
                    platform=ContentPlatform.MIRROR,
                    success=True,
                    published_url="https://mirror.xyz/test",
                )
            )
            mock_pub_cls.return_value = mock_pub

            result = runner.invoke(app, ["publish", "pub1"])
            assert result.exit_code == 0
            assert "Published" in result.output

    def test_publish_failure(self, mock_cli_config: ContentConfig) -> None:
        from animus_content.store import ContentStore

        store = ContentStore(mock_cli_config.data_dir)
        store.save_article(Article(id="pub2", title="Test", status=ContentStatus.GENERATED))

        with patch("animus_content.cli.Publisher") as mock_pub_cls:
            mock_pub = MagicMock()
            mock_pub.publish = AsyncMock(
                return_value=PublishResult(
                    article_id="pub2",
                    platform=ContentPlatform.MIRROR,
                    success=False,
                    error="API error",
                )
            )
            mock_pub_cls.return_value = mock_pub

            result = runner.invoke(app, ["publish", "pub2"])
            assert result.exit_code == 0
            assert "Failed" in result.output


class TestListCommand:
    def test_list_empty(self, mock_cli_config: ContentConfig) -> None:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0

    def test_list_with_articles(self, mock_cli_config: ContentConfig) -> None:
        from animus_content.store import ContentStore

        store = ContentStore(mock_cli_config.data_dir)
        store.save_article(Article(id="a1", title="First Article", word_count=500))
        store.save_article(Article(id="a2", title="Second Article", word_count=800))

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "First Article" in result.output

    def test_list_filter_platform(self, mock_cli_config: ContentConfig) -> None:
        from animus_content.store import ContentStore

        store = ContentStore(mock_cli_config.data_dir)
        store.save_article(Article(id="a1", platform=ContentPlatform.MIRROR))

        result = runner.invoke(app, ["list", "--platform", "mirror"])
        assert result.exit_code == 0

    def test_list_filter_status(self, mock_cli_config: ContentConfig) -> None:
        result = runner.invoke(app, ["list", "--status", "draft"])
        assert result.exit_code == 0


class TestEarningsCommand:
    def test_earnings_empty(self, mock_cli_config: ContentConfig) -> None:
        result = runner.invoke(app, ["earnings"])
        assert result.exit_code == 0
        assert "$0.00" in result.output

    def test_earnings_with_data(self, mock_cli_config: ContentConfig) -> None:
        from animus_content.models import EarningsRecord
        from animus_content.store import ContentStore

        store = ContentStore(mock_cli_config.data_dir)
        store.save_earnings(
            EarningsRecord(article_id="a1", platform=ContentPlatform.MIRROR, amount_usd=123.45)
        )

        result = runner.invoke(app, ["earnings"])
        assert result.exit_code == 0
        assert "$123.45" in result.output


class TestTopicsCommand:
    def test_topics_empty(self, mock_cli_config: ContentConfig) -> None:
        with patch("animus_content.cli.ContentGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.suggest_topics = AsyncMock(return_value=[])
            mock_gen_cls.return_value = mock_gen

            result = runner.invoke(app, ["topics"])
            assert result.exit_code == 0
            assert "No topics" in result.output

    def test_topics_with_results(self, mock_cli_config: ContentConfig) -> None:
        mock_topics = [
            ContentTopic(topic="Bitcoin", keywords=["btc"], estimated_engagement=0.8),
        ]
        with patch("animus_content.cli.ContentGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.suggest_topics = AsyncMock(return_value=mock_topics)
            mock_gen_cls.return_value = mock_gen

            result = runner.invoke(app, ["topics"])
            assert result.exit_code == 0
            assert "Bitcoin" in result.output

    def test_topics_with_options(self, mock_cli_config: ContentConfig) -> None:
        with patch("animus_content.cli.ContentGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.suggest_topics = AsyncMock(return_value=[])
            mock_gen_cls.return_value = mock_gen

            result = runner.invoke(app, ["topics", "--platform", "medium", "--count", "3"])
            assert result.exit_code == 0


class TestInitCommand:
    def test_init_creates_config(self, mock_cli_config: ContentConfig) -> None:
        config_path = mock_cli_config.data_dir / "config.yaml"
        if config_path.exists():
            config_path.unlink()
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Config created" in result.output

    def test_init_already_exists(self, mock_cli_config: ContentConfig) -> None:
        config_path = mock_cli_config.data_dir / "config.yaml"
        config_path.write_text("existing: true")

        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "already exists" in result.output
