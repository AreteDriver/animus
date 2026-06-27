"""Tests for default faucet configurations."""

import yaml

from animus_faucet.defaults import DEFAULT_FAUCETS_YAML
from animus_faucet.models import FaucetDefinition


class TestDefaultFaucetsYaml:
    def test_parses_valid_yaml(self):
        data = yaml.safe_load(DEFAULT_FAUCETS_YAML)
        assert isinstance(data, dict)

    def test_has_faucets(self):
        data = yaml.safe_load(DEFAULT_FAUCETS_YAML)
        assert "faucets" in data
        assert len(data["faucets"]) > 0

    def test_all_faucets_valid(self):
        data = yaml.safe_load(DEFAULT_FAUCETS_YAML)
        for faucet_data in data["faucets"]:
            f = FaucetDefinition.from_dict(faucet_data)
            assert f.name
            assert f.chain

    def test_has_sui_faucet(self):
        data = yaml.safe_load(DEFAULT_FAUCETS_YAML)
        names = [f["name"] for f in data["faucets"]]
        assert "sui_cli_faucet" in names

    def test_has_proxy_config(self):
        data = yaml.safe_load(DEFAULT_FAUCETS_YAML)
        assert "proxy" in data
        assert data["proxy"]["enabled"] is False

    def test_has_vision_config(self):
        data = yaml.safe_load(DEFAULT_FAUCETS_YAML)
        assert "vision" in data

    def test_has_consolidation_config(self):
        data = yaml.safe_load(DEFAULT_FAUCETS_YAML)
        assert "consolidation" in data

    def test_has_alerts_config(self):
        data = yaml.safe_load(DEFAULT_FAUCETS_YAML)
        assert "alerts" in data

    def test_jitter_default(self):
        data = yaml.safe_load(DEFAULT_FAUCETS_YAML)
        assert data["jitter_seconds"] == 300
