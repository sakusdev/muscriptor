import io
import tarfile
from pathlib import Path

import pytest

from muscriptor import flash_dataset_fetch


def _make_musicnet_tree(root: Path) -> Path:
    nested = root / "musicnet"
    for name in ("train_data", "train_labels", "test_data", "test_labels"):
        (nested / name).mkdir(parents=True, exist_ok=True)
    return nested


def test_find_musicnet_root_supports_nested_archive_layout(tmp_path: Path):
    expected = _make_musicnet_tree(tmp_path / "extract")

    resolved = flash_dataset_fetch.find_musicnet_root(tmp_path / "extract")

    assert resolved == expected.resolve()


def test_file_md5_matches_known_digest(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"MuScripter Flash")

    assert flash_dataset_fetch.file_md5(path) == "89eebef67eec6c755cdf269babea52e7"


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"nope"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe archive path"):
        flash_dataset_fetch._safe_extract_tar(archive, tmp_path / "out")

    assert not (tmp_path / "escape.txt").exists()


def test_ensure_musicnet_reuses_existing_dataset(monkeypatch, tmp_path: Path):
    expected = _make_musicnet_tree(tmp_path / "dataset")

    def fail_download(*args, **kwargs):
        raise AssertionError("download should not run")

    monkeypatch.setattr(flash_dataset_fetch, "_download_with_resume", fail_download)

    resolved = flash_dataset_fetch.ensure_musicnet(
        tmp_path / "dataset",
        minimum_free_gib=0.0,
    )

    assert resolved == expected.resolve()
