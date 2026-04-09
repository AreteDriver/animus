"""Tests for referral configuration."""

from __future__ import annotations

import yaml

from animus_referral.config import DEFAULT_DATA_DIR, ReferralConfig
from animus_referral.models import ProgramType, ReferralProgram


class TestReferralConfig:
    def test_default_config(self):
        config = ReferralConfig()
        assert config.high_value_threshold_usd == 50.0
        assert config.auto_discover is True
        assert config.programs == []

    def test_data_dir_default(self):
        config = ReferralConfig()
        assert config.data_dir == DEFAULT_DATA_DIR

    def test_data_dir_custom(self, tmp_data_dir):
        config = ReferralConfig(data_dir=tmp_data_dir)
        assert config.data_dir == tmp_data_dir

    def test_data_dir_from_env(self, tmp_data_dir, monkeypatch):
        monkeypatch.setenv("REFERRAL_DATA_DIR", str(tmp_data_dir))
        config = ReferralConfig()
        assert config.data_dir == tmp_data_dir

    def test_threshold_from_env(self, monkeypatch):
        monkeypatch.setenv("REFERRAL_HIGH_VALUE_THRESHOLD", "100.0")
        config = ReferralConfig()
        assert config.high_value_threshold_usd == 100.0

    def test_ensure_dirs(self, tmp_data_dir):
        config = ReferralConfig(data_dir=tmp_data_dir / "new_dir")
        config.ensure_dirs()
        assert config.data_dir.exists()

    def test_config_file_path(self, tmp_data_dir):
        config = ReferralConfig(data_dir=tmp_data_dir)
        assert config.config_file == tmp_data_dir / "referral.yaml"

    def test_add_program(self, config, sample_program):
        assert config.add_program(sample_program) is True
        assert len(config.programs) == 1

    def test_add_program_duplicate(self, config, sample_program):
        config.add_program(sample_program)
        assert config.add_program(sample_program) is False
        assert len(config.programs) == 1

    def test_get_program(self, config, sample_program):
        config.add_program(sample_program)
        found = config.get_program("test_exchange")
        assert found is not None
        assert found.name == "test_exchange"

    def test_get_program_not_found(self, config):
        assert config.get_program("nonexistent") is None

    def test_get_active_programs(self, config, sample_program):
        config.add_program(sample_program)
        inactive = ReferralProgram(
            name="inactive",
            platform_url="https://x.com",
            program_type=ProgramType.OTHER,
            active=False,
        )
        config.add_program(inactive)
        active = config.get_active_programs()
        assert len(active) == 1
        assert active[0].name == "test_exchange"

    def test_get_high_value_programs(self, config, sample_program):
        config.high_value_threshold_usd = 5.0
        config.add_program(sample_program)
        low = ReferralProgram(
            name="low_value",
            platform_url="https://x.com",
            program_type=ProgramType.OTHER,
            bonus_referrer_usd=1.0,
        )
        config.add_program(low)
        high = config.get_high_value_programs()
        assert len(high) == 1
        assert high[0].name == "test_exchange"

    def test_to_dict(self, config, sample_program):
        config.add_program(sample_program)
        d = config.to_dict()
        assert "programs" in d
        assert len(d["programs"]) == 1
        assert d["high_value_threshold_usd"] == 50.0

    def test_from_yaml(self, tmp_data_dir, sample_program):
        yaml_path = tmp_data_dir / "test.yaml"
        data = {
            "programs": [sample_program.to_dict()],
            "high_value_threshold_usd": 25.0,
            "auto_discover": False,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        config = ReferralConfig.from_yaml(yaml_path)
        assert len(config.programs) == 1
        assert config.programs[0].name == "test_exchange"
        assert config.high_value_threshold_usd == 25.0
        assert config.auto_discover is False

    def test_from_yaml_empty(self, tmp_data_dir):
        yaml_path = tmp_data_dir / "empty.yaml"
        with open(yaml_path, "w") as f:
            f.write("")

        config = ReferralConfig.from_yaml(yaml_path)
        assert config.programs == []

    def test_from_yaml_with_data_dir(self, tmp_data_dir):
        yaml_path = tmp_data_dir / "test.yaml"
        custom_dir = tmp_data_dir / "custom"
        data = {"data_dir": str(custom_dir)}
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        config = ReferralConfig.from_yaml(yaml_path)
        assert config.data_dir == custom_dir

    def test_save_yaml(self, tmp_data_dir, sample_program):
        config = ReferralConfig(data_dir=tmp_data_dir)
        config.add_program(sample_program)
        yaml_path = tmp_data_dir / "saved.yaml"
        config.save_yaml(yaml_path)

        loaded = ReferralConfig.from_yaml(yaml_path)
        assert len(loaded.programs) == 1
        assert loaded.programs[0].name == "test_exchange"

    def test_save_yaml_creates_parent(self, tmp_data_dir):
        config = ReferralConfig(data_dir=tmp_data_dir)
        yaml_path = tmp_data_dir / "subdir" / "saved.yaml"
        config.save_yaml(yaml_path)
        assert yaml_path.exists()
