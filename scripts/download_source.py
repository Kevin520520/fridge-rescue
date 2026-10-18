"""Download the single fixed-version Food.com recipe file from Kaggle."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import kagglehub

from fridge_rescue.settings import PROJECT_ROOT, get_settings

EXPECTED_SIZE_BYTES = 178_723_234
EXPECTED_SHA256 = "9f591abe9f8d1c691bbc630b0431ec6613f09fabe93f1789af87a06484aae9fb"
LICENSE = "CC0-1.0"
SOURCE_URL = "https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_parquet(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as source:
        header = source.read(4)
        source.seek(-4, 2)
        footer = source.read(4)
    return header == b"PAR1" and footer == b"PAR1"


def unpack_kaggle_wrapper(path: Path) -> bool:
    """Replace Kaggle's single-file ZIP wrapper with the actual Parquet file."""

    if not zipfile.is_zipfile(path):
        return False

    temporary = path.with_suffix(path.suffix + ".part")
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(path.name)
        if Path(member.filename).name != path.name or member.file_size != EXPECTED_SIZE_BYTES:
            raise ValueError("Unexpected file inside Kaggle ZIP wrapper")
        with archive.open(member) as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)

    if not is_parquet(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError("Extracted file does not have valid Parquet magic bytes")
    temporary.replace(path)
    return True


def main() -> None:
    settings = get_settings()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    versioned_handle = (
        f"{settings.kaggle_dataset_handle}/versions/{settings.kaggle_dataset_version}"
    )

    recipe_path = raw_dir / settings.kaggle_dataset_file
    if not recipe_path.exists():
        downloaded = Path(
            kagglehub.dataset_download(
                versioned_handle,
                path=settings.kaggle_dataset_file,
                output_dir=str(raw_dir),
            )
        )
        recipe_path = downloaded if downloaded.is_file() else recipe_path
    if not recipe_path.exists():
        raise FileNotFoundError("KaggleHub completed, but recipes file was not found")

    unpacked_zip_wrapper = unpack_kaggle_wrapper(recipe_path)
    if not is_parquet(recipe_path):
        raise ValueError("Downloaded file is neither valid Parquet nor the expected ZIP wrapper")

    file_hash = sha256(recipe_path)
    if recipe_path.stat().st_size != EXPECTED_SIZE_BYTES or file_hash != EXPECTED_SHA256:
        raise ValueError("Downloaded Food.com v2 file failed size or SHA-256 verification")

    metadata = {
        "dataset_handle": settings.kaggle_dataset_handle,
        "dataset_version": settings.kaggle_dataset_version,
        "file_name": recipe_path.name,
        "file_size_bytes": recipe_path.stat().st_size,
        "expected_size_bytes": EXPECTED_SIZE_BYTES,
        "sha256": file_hash,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "kaggle_zip_wrapper_unpacked": unpacked_zip_wrapper,
        "license": LICENSE,
        "source_url": SOURCE_URL,
    }
    metadata_path = raw_dir / "source_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Downloaded: {recipe_path}")
    print(f"Size: {metadata['file_size_bytes']:,} bytes")
    if metadata["file_size_bytes"] != EXPECTED_SIZE_BYTES:
        print(
            "Warning: file size differs from the reviewed v2 metadata; inspect before continuing."
        )
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
