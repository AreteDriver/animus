"""Tests for durability export + cold-rebuild (roadmap A8)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from animus.durability.export import (
    ARCHIVE_SCHEMA_VERSION,
    DurabilityError,
    ManifestMismatchError,
    export_all,
    rebuild,
    verify_archive,
)


def _populate(data_dir: Path) -> None:
    """Build a representative data dir (nested dirs, binary + text)."""
    (data_dir / "memory").mkdir(parents=True, exist_ok=True)
    (data_dir / "entities").mkdir(parents=True, exist_ok=True)
    (data_dir / "memory" / "store.db").write_bytes(b"\x00\x01sqlite-ish\xff")
    (data_dir / "memory" / "index.jsonl").write_text('{"id": 1, "content": "hi"}\n')
    (data_dir / "entities" / "people.json").write_text('{"arete": {"role": "owner"}}')
    (data_dir / "tasks.json").write_text('[{"id": "t1", "done": false}]')


class TestExportAll:
    def test_writes_archive_with_manifest_and_data(self, tmp_path):
        data_dir = tmp_path / "data"
        _populate(data_dir)
        archive = export_all(data_dir, tmp_path / "exp")
        assert archive.name == "exp.zip"
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            assert "manifest.json" in names
            assert "data/memory/store.db" in names
            assert "data/tasks.json" in names
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["schema_version"] == ARCHIVE_SCHEMA_VERSION
        assert manifest["file_count"] == 4
        assert "data/memory/store.db" not in manifest["files"]  # keys are rel-to-data
        assert "memory/store.db" in manifest["files"]
        assert all("sha256" in m and "bytes" in m for m in manifest["files"].values())

    def test_config_snapshot_is_redacted(self, tmp_path):
        data_dir = tmp_path / "data"
        _populate(data_dir)
        cfg = {
            "api": {"anthropic_key": "AAAsecretAAA", "openai_key": "BBBsecretBBB"},
            "services": {"auth_token": "CCCsecretCCC", "vapid_public_key": "PUB", "port": 7700},
        }
        archive = export_all(data_dir, tmp_path / "exp", config_snapshot=cfg)
        manifest = verify_archive(archive)
        snap = manifest["config_snapshot"]
        assert snap["api"]["anthropic_key"] == "***redacted***"
        assert snap["api"]["openai_key"] == "***redacted***"
        assert snap["services"]["auth_token"] == "***redacted***"
        assert snap["services"]["vapid_public_key"] == "PUB"  # public — kept
        assert snap["services"]["port"] == 7700  # int — kept
        blob = json.dumps(snap)
        for secret in ("AAAsecretAAA", "BBBsecretBBB", "CCCsecretCCC"):
            assert secret not in blob

    def test_export_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(DurabilityError, match="does not exist"):
            export_all(tmp_path / "nope", tmp_path / "out")


class TestVerifyArchive:
    def test_clean_archive_verifies(self, tmp_path):
        data_dir = tmp_path / "data"
        _populate(data_dir)
        archive = export_all(data_dir, tmp_path / "exp")
        manifest = verify_archive(archive)
        assert manifest["file_count"] == 4

    def test_tampered_member_fails_checksum(self, tmp_path):
        data_dir = tmp_path / "data"
        _populate(data_dir)
        archive = export_all(data_dir, tmp_path / "exp")
        # Rewrite a data member without updating the manifest.
        good = zipfile.ZipFile(archive)
        contents = {n: good.read(n) for n in good.namelist()}
        good.close()
        contents["data/tasks.json"] = b"[]  # tampered"
        with zipfile.ZipFile(archive, "w") as zf:
            for n, b in contents.items():
                zf.writestr(n, b)
        with pytest.raises(ManifestMismatchError, match="checksum mismatch"):
            verify_archive(archive)

    def test_non_animus_zip_rejected(self, tmp_path):
        bogus = tmp_path / "bogus.zip"
        with zipfile.ZipFile(bogus, "w") as zf:
            zf.writestr("hello.txt", "not an export")
        with pytest.raises(DurabilityError, match="no manifest"):
            verify_archive(bogus)


class TestColdRebuild:
    def test_round_trip_restores_state(self, tmp_path):
        """The headline A8 guarantee: export → wipe → rebuild → state identical."""
        src = tmp_path / "data"
        _populate(src)
        # Capture original bytes for every file.
        original = {
            p.relative_to(src).as_posix(): p.read_bytes() for p in src.rglob("*") if p.is_file()
        }

        archive = export_all(src, tmp_path / "exp")

        # Simulate loss-of-machine: restore into a brand-new empty dir.
        target = tmp_path / "restored"
        manifest = rebuild(archive, target)

        restored = {
            p.relative_to(target).as_posix(): p.read_bytes()
            for p in target.rglob("*")
            if p.is_file()
        }
        assert restored == original  # byte-for-byte, including the sqlite-ish binary
        assert manifest["file_count"] == len(original)

    def test_rebuild_refuses_nonempty_target_without_overwrite(self, tmp_path):
        src = tmp_path / "data"
        _populate(src)
        archive = export_all(src, tmp_path / "exp")
        target = tmp_path / "restored"
        target.mkdir()
        (target / "preexisting.txt").write_text("keep me")
        with pytest.raises(DurabilityError, match="not empty"):
            rebuild(archive, target)

    def test_rebuild_overwrite_allows_nonempty(self, tmp_path):
        src = tmp_path / "data"
        _populate(src)
        archive = export_all(src, tmp_path / "exp")
        target = tmp_path / "restored"
        target.mkdir()
        (target / "stale.txt").write_text("old")
        manifest = rebuild(archive, target, overwrite=True)
        assert (target / "tasks.json").read_text() == '[{"id": "t1", "done": false}]'
        assert manifest["file_count"] == 4

    def test_rebuild_verifies_before_touching_target(self, tmp_path):
        # A corrupt archive must not leave a half-written target.
        src = tmp_path / "data"
        _populate(src)
        archive = export_all(src, tmp_path / "exp")
        contents = {}
        with zipfile.ZipFile(archive) as zf:
            for n in zf.namelist():
                contents[n] = zf.read(n)
        contents["data/memory/index.jsonl"] = b"corrupted"
        with zipfile.ZipFile(archive, "w") as zf:
            for n, b in contents.items():
                zf.writestr(n, b)
        target = tmp_path / "restored"
        with pytest.raises(ManifestMismatchError):
            rebuild(archive, target)
        assert not target.exists() or not any(target.iterdir())
