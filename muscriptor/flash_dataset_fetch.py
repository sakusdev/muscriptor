"""Dataset download helpers for MuScripter Flash notebooks and local training."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

MUSICNET_URL = "https://zenodo.org/records/5120004/files/musicnet.tar.gz?download=1"
MUSICNET_MD5 = "844764911fa0d5b97c97da944a057590"
_MIB = 1024 * 1024


def find_musicnet_root(root: str | Path) -> Path | None:
    """Return the directory containing all four official MusicNet split folders."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return None

    required = {"train_data", "train_labels", "test_data", "test_labels"}
    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    for candidate in candidates:
        names = {child.name for child in candidate.iterdir() if child.is_dir()}
        if required.issubset(names):
            return candidate
    return None


def file_md5(path: str | Path, *, chunk_size: int = 8 * _MIB) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_resume(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0

    headers = {"User-Agent": "MuScripter-Flash/0.3"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(
            f"Resuming MusicNet download at {existing / (1024**3):.2f} GiB",
            flush=True,
        )
    else:
        print("Starting MusicNet download", flush=True)

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = int(getattr(response, "status", 200) or 200)
        append = bool(existing and status == 206)
        if existing and not append:
            print("Server did not honor Range; restarting download", flush=True)
            existing = 0

        content_length = int(response.headers.get("Content-Length", "0") or 0)
        total = existing + content_length if content_length else 0
        mode = "ab" if append else "wb"
        downloaded = existing
        next_report = downloaded + 512 * _MIB

        with partial.open(mode) as handle:
            while True:
                chunk = response.read(8 * _MIB)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total:
                        print(
                            f"Downloaded {downloaded / (1024**3):.2f} / "
                            f"{total / (1024**3):.2f} GiB",
                            flush=True,
                        )
                    else:
                        print(
                            f"Downloaded {downloaded / (1024**3):.2f} GiB",
                            flush=True,
                        )
                    next_report = downloaded + 512 * _MIB

    os.replace(partial, destination)
    return destination


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"refusing archive link entry: {member.name}")
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive path: {member.name}")
        archive.extractall(destination, members=members)


def ensure_musicnet(
    dataset_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    keep_archive: bool = False,
    verify_md5: bool = True,
    minimum_free_gib: float = 30.0,
    url: str = MUSICNET_URL,
) -> Path:
    """Ensure the official MusicNet dataset exists and return its usable root.

    Existing extracted data is reused. Otherwise the official Zenodo archive is
    downloaded with HTTP Range resume support, verified against the published
    MD5 checksum, then safely extracted. The 11.1 GB archive can optionally live
    in a persistent cache directory such as Google Drive.
    """
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    existing = find_musicnet_root(dataset_dir)
    if existing is not None:
        print(f"MusicNet already available: {existing}", flush=True)
        return existing

    dataset_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(dataset_dir).free / (1024**3)
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"MusicNet extraction target has only {free_gib:.1f} GiB free; "
            f"at least {minimum_free_gib:.1f} GiB is required by this helper"
        )

    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir
        else dataset_dir.parent / ".cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root / "musicnet.tar.gz"

    if archive_path.is_file() and verify_md5:
        checksum = file_md5(archive_path)
        if checksum != MUSICNET_MD5:
            print(
                "Cached MusicNet archive checksum mismatch; downloading again",
                flush=True,
            )
            archive_path.unlink()

    if not archive_path.is_file():
        _download_with_resume(url, archive_path)

    if verify_md5:
        print("Verifying MusicNet MD5 checksum", flush=True)
        checksum = file_md5(archive_path)
        if checksum != MUSICNET_MD5:
            raise RuntimeError(
                f"MusicNet checksum mismatch: expected {MUSICNET_MD5}, got {checksum}"
            )

    print(f"Extracting MusicNet to {dataset_dir}", flush=True)
    _safe_extract_tar(archive_path, dataset_dir)
    root = find_musicnet_root(dataset_dir)
    if root is None:
        raise RuntimeError(
            "MusicNet archive extracted but expected split folders were not found"
        )

    if not keep_archive:
        archive_path.unlink(missing_ok=True)
    print(f"MusicNet ready: {root}", flush=True)
    return root
