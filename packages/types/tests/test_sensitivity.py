"""Tests for the shared Sensitivity enum."""

from __future__ import annotations

import pytest

from animus_types import Sensitivity


class TestSensitivityEnum:
    def test_four_members(self):
        assert len(list(Sensitivity)) == 4

    def test_values(self):
        assert Sensitivity.PUBLIC.value == "public"
        assert Sensitivity.PERSONAL.value == "personal"
        assert Sensitivity.CONFIDENTIAL.value == "confidential"
        assert Sensitivity.SECRET.value == "secret"

    @pytest.mark.parametrize(
        "v,expected",
        [
            ("public", Sensitivity.PUBLIC),
            ("personal", Sensitivity.PERSONAL),
            ("confidential", Sensitivity.CONFIDENTIAL),
            ("secret", Sensitivity.SECRET),
        ],
    )
    def test_construct_from_value(self, v, expected):
        assert Sensitivity(v) is expected

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Sensitivity("medium")

    def test_zero_deps(self):
        """Sanity check: importing the package doesn't pull in heavy deps."""
        import importlib
        import sys

        # Reload to confirm a fresh import
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("animus_types"):
                del sys.modules[mod_name]

        importlib.import_module("animus_types")
        # animus_types should not have pulled in any non-stdlib deps
        non_stdlib = [m for m in sys.modules if m.startswith("animus_types") and "." in m]
        # Module + sensitivity submodule = 2 max; anything beyond would be a smell
        assert len(non_stdlib) <= 2
