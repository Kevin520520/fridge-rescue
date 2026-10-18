"""Download and verify the fixed-revision Epicurious recipe CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from fridge_rescue.epicurious_pipeline import (
    SOURCE_DATASET,
    SOURCE_DATASET_URL,
    SOURCE_EXPECTED_ROWS,
    SOURCE_FILE_NAME,
    SOURCE_FILE_SHA256,
    SOURCE_FILE_SIZE_BYTES,
    SOURCE_LICENSE,
    SOURCE_REPOSITORY_URL,
    SOURCE_REVISION,
)
from fridge_rescue.settings import PROJECT_ROOT

CHUNK_SIZE = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_file(path: Path) -> tuple[int, str]:
    """Require the reviewed byte length and digest before accepting a source file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    digest = sha256(path)
    if size != SOURCE_FILE_SIZE_BYTES:
        raise ValueError(f"Source size mismatch: expected {SOURCE_FILE_SIZE_BYTES}, got {size}")
    if digest.casefold() != SOURCE_FILE_SHA256:
        raise ValueError(f"Source SHA-256 mismatch: expected {SOURCE_FILE_SHA256}, got {digest}")
    return size, digest


def download_atomic(destination: Path, *, force: bool = False) -> bool:
    """Download to a .part file, verify it, then atomically publish it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        validate_source_file(destination)
        return False

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        SOURCE_DATASET_URL,
        headers={"User-Agent": "Fridge-Rescue-Course-Project/3.0"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
        validate_source_file(temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return True


def write_metadata(path: Path, source: Path) -> None:
    size, digest = validate_source_file(source)
    metadata = {
        "dataset_name": SOURCE_DATASET,
        "file_name": source.name,
        "file_size_bytes": size,
        "expected_size_bytes": SOURCE_FILE_SIZE_BYTES,
        "sha256": digest,
        "expected_sha256": SOURCE_FILE_SHA256,
        "expected_row_count": SOURCE_EXPECTED_ROWS,
        "downloaded_or_verified_at_utc": datetime.now(UTC).isoformat(),
        "license": SOURCE_LICENSE,
        "source_url": SOURCE_DATASET_URL,
        "source_repository_url": SOURCE_REPOSITORY_URL,
        "source_revision": SOURCE_REVISION,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / SOURCE_FILE_NAME,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    downloaded = download_atomic(args.destination, force=args.force)
    metadata_path = args.destination.parent / "source_metadata.json"
    write_metadata(metadata_path, args.destination)

    action = "Downloaded" if downloaded else "Already present and verified"
    print(f"{action}: {args.destination}")
    print(f"Size: {args.destination.stat().st_size:,} bytes")
    print(f"SHA-256: {SOURCE_FILE_SHA256}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
