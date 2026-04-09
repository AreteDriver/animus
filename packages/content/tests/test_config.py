"""Tests for configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from animus_content.config import ContentConfig, LLMConfig, PlatformCredentials


class TestPlatformCredentials:
    def test_defaults(self) -> None:
        c = PlatformCredentials()
        assert c.api_key == ""
        assert c.api_secret == ""
        assert c.base_url == ""
        assert c.extra == {}

    def test_with_values(self) -> None:
        c = PlatformCredentials(api_key="key123", base_url="https://api.test.com")
        assert c.api_key == "key123"


class TestLLMConfig:
    def test_defaults(self) -> None:
        c = LLMConfig()
        assert c.provider == "ollama"
        assert c.model == "qwen2.5:14b"
        assert c.base_url == "http://localhost:11434"
        assert c.api_key == ""
        assert c.max_tokens == 4096
        assert c.temperature == 0.7

    def test_anthropic_config(self) -> None:
        c = LLMConfig(provider="anthropic", model="claude-3-haiku", api_key="sk-test")
        assert c.provider == "anthropic"


class TestContentConfig:
    def test_defaults(self) -> None:
        c = ContentConfig()
        assert c.data_dir == Path.home() / ".animus" / "content"
        assert c.max_articles_per_day == 5
        assert c.min_word_count == 500
        assert c.max_word_count == 2500
        assert c.duplicate_topic_cooldown_days == 7
        assert "crypto" in c.default_tags

    def test_custom_values(self) -> None:
        c = ContentConfig(min_word_count=1000, max_articles_per_day=10)
        assert c.min_word_count == 1000
        assert c.max_articles_per_day == 10

    def test_load_missing_file(self, tmp_path: Path) -> None:
        config = ContentConfig.load(tmp_path / "nonexistent.yaml")
        assert config.min_word_count == 500  # defaults

    def test_load_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("")
        config = ContentConfig.load(path)
        assert config.min_word_count == 500

    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        original = ContentConfig(
            data_dir=tmp_path,
            min_word_count=1000,
            max_articles_per_day=3,
        )
        original.save(path)
        assert path.exists()

        loaded = ContentConfig.load(path)
        assert loaded.min_word_count == 1000
        assert loaded.max_articles_per_day == 3

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "config.yaml"
        config = ContentConfig(data_dir=tmp_path)
        config.save(path)
        assert path.exists()

    def test_load_with_platforms(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        data = {
            "platforms": {
                "mirror": {"api_key": "key123", "base_url": "https://custom.mirror.xyz"},
            },
            "min_word_count": 800,
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

        config = ContentConfig.load(path)
        assert "mirror" in config.platforms
        assert config.platforms["mirror"].api_key == "key123"
        assert config.min_word_count == 800

    def test_load_with_llm_config(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        data = {
            "llm": {
                "provider": "anthropic",
                "model": "claude-3-haiku",
                "api_key": "sk-test",
            }
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

        config = ContentConfig.load(path)
        assert config.llm.provider == "anthropic"
        assert config.llm.model == "claude-3-haiku"
