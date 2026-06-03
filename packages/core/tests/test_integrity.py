"""Tests for the integrity checker (10/10 polish — tampering detection)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from animus.integrity.checker import (
    _TRACKED_RELATIVE_PATHS,
    IntegrityMismatchError,
    IntegrityNotInitializedError,
    baseline_path,
    compute_current,
    regenerate_baseline,
    tracked_files,
    tracked_module_hashes,
    verify_or_raise,
)


@pytest.fixture
def fake_tree(tmp_path: Path):
    """Build a synthetic mini-tree mirroring the layout of the tracked files."""
    # Layout: tmp_path/animus_pkg/{memory,network,audit}/*.py + mcp_server.py
    pkg_root = tmp_path / "animus_pkg"
    for rel in _TRACKED_RELATIVE_PATHS:
        p = pkg_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# original content of {rel}\n")
    data_dir = tmp_path / "data"
    return pkg_root, data_dir


class TestComputeCurrent:
    def test_hashes_every_tracked_file(self, fake_tree):
        pkg_root, _ = fake_tree
        hashes = compute_current(pkg_root)
        expected = set(_TRACKED_RELATIVE_PATHS) | set(tracked_module_hashes())
        assert set(hashes.keys()) == expected
        for h in hashes.values():
            assert len(h) == 64  # sha256 hex
            assert all(c in "0123456789abcdef" for c in h)

    def test_raises_if_tracked_file_missing(self, fake_tree):
        pkg_root, _ = fake_tree
        (pkg_root / _TRACKED_RELATIVE_PATHS[0]).unlink()
        with pytest.raises(FileNotFoundError):
            compute_current(pkg_root)


class TestRegenerateBaseline:
    def test_writes_manifest(self, fake_tree):
        pkg_root, data_dir = fake_tree
        manifest = regenerate_baseline(data_dir, root=pkg_root, note="initial")
        path = baseline_path(data_dir)
        assert path.exists()
        on_disk = json.loads(path.read_text())
        # tracked_files is a tuple in-memory; JSON round-trip yields a list.
        # Compare the meaningful contents independently.
        assert on_disk["hashes"] == manifest["hashes"]
        assert on_disk["note"] == manifest["note"] == "initial"
        assert on_disk["schema_version"] == 1
        assert list(on_disk["tracked_files"]) == list(manifest["tracked_files"])
        assert "generated_at" in manifest
        expected = set(_TRACKED_RELATIVE_PATHS) | set(tracked_module_hashes())
        assert set(manifest["hashes"].keys()) == expected

    def test_round_trip_idempotent_on_unchanged_tree(self, fake_tree):
        pkg_root, data_dir = fake_tree
        m1 = regenerate_baseline(data_dir, root=pkg_root)
        m2 = regenerate_baseline(data_dir, root=pkg_root)
        # Hashes identical; generated_at + note may differ
        assert m1["hashes"] == m2["hashes"]


class TestVerifyOrRaise:
    def test_passes_on_clean_baseline(self, fake_tree):
        pkg_root, data_dir = fake_tree
        regenerate_baseline(data_dir, root=pkg_root)
        verify_or_raise(data_dir, root=pkg_root)  # no raise

    def test_raises_on_drift(self, fake_tree):
        pkg_root, data_dir = fake_tree
        regenerate_baseline(data_dir, root=pkg_root)
        # Tamper with one file
        target = pkg_root / _TRACKED_RELATIVE_PATHS[0]
        target.write_text(target.read_text() + "\n# malicious patch\n")
        with pytest.raises(IntegrityMismatchError) as exc:
            verify_or_raise(data_dir, root=pkg_root)
        assert _TRACKED_RELATIVE_PATHS[0] in str(exc.value)
        assert "drifted from baseline" in str(exc.value)

    def test_raises_on_missing_tracked_file(self, fake_tree):
        pkg_root, data_dir = fake_tree
        regenerate_baseline(data_dir, root=pkg_root)
        (pkg_root / _TRACKED_RELATIVE_PATHS[1]).unlink()
        # Should fail at compute_current with FileNotFoundError, not
        # IntegrityMismatchError — the failure mode for a tracked file
        # vanishing is "the gate is gone entirely", a louder error.
        with pytest.raises(FileNotFoundError):
            verify_or_raise(data_dir, root=pkg_root)

    def test_raises_when_baseline_missing(self, fake_tree):
        pkg_root, data_dir = fake_tree
        # No baseline written
        with pytest.raises(IntegrityNotInitializedError):
            verify_or_raise(data_dir, root=pkg_root)

    def test_override_env_bypasses_check(self, fake_tree, monkeypatch, caplog):
        pkg_root, data_dir = fake_tree
        # Don't even regenerate — override should skip the missing-baseline
        # raise entirely.
        monkeypatch.setenv("ANIMUS_INTEGRITY_OVERRIDE", "1")
        with caplog.at_level("WARNING"):
            verify_or_raise(data_dir, root=pkg_root)  # no raise
        assert any("ANIMUS_INTEGRITY_OVERRIDE=1" in record.message for record in caplog.records)

    def test_extra_files_in_actual_flagged_as_drift(self, fake_tree, monkeypatch):
        """If the tracked-files set is extended without regenerating the
        baseline, the new file shows as drift."""
        pkg_root, data_dir = fake_tree
        regenerate_baseline(data_dir, root=pkg_root)

        # Simulate a new tracked file added in code but not in baseline
        import animus.integrity.checker as checker

        orig_tracked = checker._TRACKED_RELATIVE_PATHS
        extended = orig_tracked + ("new/file.py",)
        monkeypatch.setattr(checker, "_TRACKED_RELATIVE_PATHS", extended)
        (pkg_root / "new").mkdir()
        (pkg_root / "new" / "file.py").write_text("# new tracked file\n")

        with pytest.raises(IntegrityMismatchError) as exc:
            verify_or_raise(data_dir, root=pkg_root)
        assert "new/file.py" in str(exc.value)


class TestTrackedFiles:
    def test_returns_paths_under_root(self, fake_tree):
        pkg_root, _ = fake_tree
        files = tracked_files(pkg_root)
        # Relative tracked files resolve under the given root; cross-package
        # module files resolve to their real installed locations.
        n_modules = len(tracked_module_hashes())
        assert len(files) == len(_TRACKED_RELATIVE_PATHS) + n_modules
        under_root = [p for p in files if pkg_root in p.parents]
        assert len(under_root) == len(_TRACKED_RELATIVE_PATHS)


class TestRealAnimusPackage:
    """Smoke check against the actually-installed animus package — every
    tracked relative path must point to a real file in the live package."""

    def test_all_tracked_files_exist_in_live_package(self):
        # No root override — uses _animus_pkg_root() default
        hashes = compute_current()  # raises if any file is missing
        expected = set(_TRACKED_RELATIVE_PATHS) | set(tracked_module_hashes())
        assert set(hashes.keys()) == expected


class TestCrossPackageTracking:
    """A6: the baseline must hash the checker itself and the REAL (cross-
    package) egress/DLP logic, not just core's re-export shim."""

    def test_self_hash_and_real_egress_modules_tracked(self):
        keys = set(compute_current().keys())
        assert "integrity/checker.py" in keys  # self-hash
        assert "learning/guardrails.py" in keys
        assert "module:animus_types.egress" in keys  # real egress logic
        assert "module:animus_types.secrets" in keys  # credential patterns

    def test_forge_enforcement_modules_tracked_when_installed(self):
        """C5 — when forge is importable, the modules that actually gate every
        cloud call (egress engine, content-aware DLP helper, sensitivity router)
        must be in the baseline. Skip cleanly if forge isn't installed."""
        import importlib.util

        if importlib.util.find_spec("animus_forge") is None:
            pytest.skip("animus_forge not installed in this interpreter")

        keys = set(tracked_module_hashes().keys())
        assert "module:animus_forge.network.egress" in keys
        assert "module:animus_forge.providers.base" in keys
        assert "module:animus_forge.providers.router" in keys

    def test_module_drift_trips_verify(self, fake_tree, monkeypatch):
        pkg_root, data_dir = fake_tree
        regenerate_baseline(data_dir, root=pkg_root)

        import animus.integrity.checker as checker

        real = checker.tracked_module_hashes

        def tampered():
            h = dict(real())
            first = next(iter(h))
            h[first] = "0" * 64  # simulate a tampered cross-package module
            return h

        monkeypatch.setattr(checker, "tracked_module_hashes", tampered)
        with pytest.raises(IntegrityMismatchError):
            verify_or_raise(data_dir, root=pkg_root)
