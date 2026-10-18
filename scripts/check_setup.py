"""Check project prerequisites without displaying any secret values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fridge_rescue.epicurious_pipeline import (
    CURATION_VERSION,
    REQUIRED_COLUMNS,
    SOURCE_EXPECTED_ROWS,
    SOURCE_FILE_SHA256,
    SOURCE_FILE_SIZE_BYTES,
    SOURCE_LICENSE,
)
from fridge_rescue.settings import PROJECT_ROOT, get_settings

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "streamlit": "streamlit",
    "langgraph": "langgraph",
    "openai": "openai",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "dotenv": "python-dotenv",
    "yaml": "PyYAML",
}
REBUILD_PACKAGES: dict[str, str] = {}
CURATED_CORPUS_NAME = "recipes_2000.parquet"
EXPECTED_CURATED_ROWS = 2_000
EXPECTED_CURATION_VERSION = CURATION_VERSION
EXPECTED_CURATED_OUTPUTS = frozenset(
    {CURATED_CORPUS_NAME, "manifest.csv", "qa_sample.csv", "DATA_CARD.md"}
)


def status(ok: bool, label: str, detail: str, *, required: bool = True) -> bool:
    """Print one redaction-safe check result and return its Boolean status."""

    marker = "READY" if ok else ("NEEDED" if required else "OPTION")
    print(f"[{marker:6}] {label}: {detail}")
    return ok


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_metadata(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read optional source metadata and return a validation error when malformed."""

    if not path.exists():
        return None, None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"cannot read {path.name}: {error}"
    if not isinstance(metadata, dict):
        return None, f"{path.name} must contain a JSON object"
    return metadata, None


def validate_curated_corpus(corpus: Path, summary_path: Path) -> tuple[bool, str]:
    """Verify the packaged 2,000-recipe corpus against its curation record."""

    if not corpus.is_file():
        return False, f"packaged corpus not found: {corpus}"

    try:
        parquet = pq.ParquetFile(corpus)
        row_count = parquet.metadata.num_rows
    except (OSError, ValueError) as error:
        return False, f"packaged corpus is not readable Parquet: {error}"

    if row_count != EXPECTED_CURATED_ROWS:
        return False, (
            f"row-count mismatch: expected {EXPECTED_CURATED_ROWS:,}, got {row_count:,}"
        )

    summary, summary_error = load_source_metadata(summary_path)
    if summary_error:
        return False, summary_error
    if summary is None:
        return False, f"integrity metadata not found: {summary_path}"

    recorded_rows = summary.get("selected_rows")
    if recorded_rows != EXPECTED_CURATED_ROWS:
        return False, (
            f"{summary_path.name} selected_rows must be {EXPECTED_CURATED_ROWS:,}; "
            f"got {recorded_rows!r}"
        )

    if summary.get("curation_version") != EXPECTED_CURATION_VERSION:
        return False, (
            f"{summary_path.name} curation_version must be "
            f"{EXPECTED_CURATION_VERSION}; got {summary.get('curation_version')!r}"
        )

    output_hashes = summary.get("output_sha256")
    if not isinstance(output_hashes, dict):
        return False, f"{summary_path.name} has no output_sha256 object"
    missing_hashes = sorted(EXPECTED_CURATED_OUTPUTS - set(output_hashes))
    if missing_hashes:
        return False, (
            f"{summary_path.name} has no SHA-256 for: {', '.join(missing_hashes)}"
        )

    output_dir = summary_path.parent
    for name in sorted(EXPECTED_CURATED_OUTPUTS):
        expected_sha = output_hashes.get(name)
        if not isinstance(expected_sha, str) or not expected_sha.strip():
            return False, f"{summary_path.name} has an invalid SHA-256 for {name}"
        artifact = output_dir / name
        if not artifact.is_file():
            return False, f"curated package artifact not found: {artifact}"
        if sha256(artifact).casefold() != expected_sha.strip().casefold():
            return False, f"SHA-256 mismatch for {name}; rebuild or restore the package"

    required_columns = [
        "recipe_id",
        "source_record_id",
        "record_hash",
        "dedup_hash",
        "source_row_hash",
        "category_group",
        "complexity_bucket",
        "ingredient_lines_original",
        "ingredients_normalized",
        "instructions_original",
        "measurement_coverage",
        "source_file_sha256",
        "source_license",
    ]
    try:
        corpus_rows = pq.read_table(corpus, columns=required_columns).to_pylist()
    except (OSError, ValueError, KeyError) as error:
        return False, f"curated corpus is missing required integrity fields: {error}"

    corpus_ids = [str(row["recipe_id"]) for row in corpus_rows]
    if len(set(corpus_ids)) != EXPECTED_CURATED_ROWS:
        return False, "curated corpus recipe IDs are not unique"
    if len({row["record_hash"] for row in corpus_rows}) != EXPECTED_CURATED_ROWS:
        return False, "curated corpus record hashes are not unique"
    if len({row["dedup_hash"] for row in corpus_rows}) != EXPECTED_CURATED_ROWS:
        return False, "curated corpus dedup hashes are not unique"
    if len({row["source_row_hash"] for row in corpus_rows}) != EXPECTED_CURATED_ROWS:
        return False, "curated corpus source-row hashes are not unique"
    if len({str(row["source_record_id"]) for row in corpus_rows}) != EXPECTED_CURATED_ROWS:
        return False, "curated corpus source record IDs are not unique"
    if any(
        len(row["ingredient_lines_original"] or []) < 3
        or any(not str(line).strip() for line in row["ingredient_lines_original"] or [])
        for row in corpus_rows
    ):
        return False, "curated corpus contains incomplete source ingredient lines"
    if any(
        len(row["instructions_original"] or []) < 2
        or any(not str(step).strip() for step in row["instructions_original"] or [])
        for row in corpus_rows
    ):
        return False, "curated corpus contains incomplete source instructions"
    if any(
        not 0.0 <= float(row["measurement_coverage"]) <= 1.0
        for row in corpus_rows
    ):
        return False, "curated corpus has measurement coverage outside 0..1"
    if any(
        str(row["source_file_sha256"]).casefold() != SOURCE_FILE_SHA256.casefold()
        for row in corpus_rows
    ):
        return False, "curated corpus does not point to the pinned source SHA-256"
    if any(row["source_license"] != SOURCE_LICENSE for row in corpus_rows):
        return False, "curated corpus does not preserve the pinned source license"

    manifest_path = output_dir / "manifest.csv"
    qa_path = output_dir / "qa_sample.csv"
    try:
        with manifest_path.open(encoding="utf-8-sig", newline="") as source:
            manifest_rows = list(csv.DictReader(source))
        with qa_path.open(encoding="utf-8-sig", newline="") as source:
            qa_rows = list(csv.DictReader(source))
    except (OSError, csv.Error) as error:
        return False, f"curated CSV artifact is unreadable: {error}"

    if len(manifest_rows) != EXPECTED_CURATED_ROWS:
        return False, f"manifest must contain {EXPECTED_CURATED_ROWS:,} rows"
    manifest_by_id = {row.get("recipe_id", ""): row for row in manifest_rows}
    if len(manifest_by_id) != EXPECTED_CURATED_ROWS or set(manifest_by_id) != set(corpus_ids):
        return False, "manifest recipe IDs do not match the curated corpus"
    for row in corpus_rows:
        manifest_row = manifest_by_id[str(row["recipe_id"])]
        if manifest_row.get("record_hash") != row["record_hash"]:
            return False, f"manifest record_hash mismatch for recipe {row['recipe_id']}"
        if manifest_row.get("dedup_hash") != row["dedup_hash"]:
            return False, f"manifest dedup_hash mismatch for recipe {row['recipe_id']}"

    qa_ids = [row.get("recipe_id", "") for row in qa_rows]
    if len(qa_ids) != 30 or len(set(qa_ids)) != 30:
        return False, "QA sample must contain 30 unique recipe IDs"
    if not set(qa_ids).issubset(set(corpus_ids)):
        return False, "QA sample contains recipe IDs outside the curated corpus"

    expected_distribution = summary.get("strata_selected")
    actual_distribution = Counter(
        f"{row['category_group']}|{row['complexity_bucket']}" for row in corpus_rows
    )
    if expected_distribution != dict(sorted(actual_distribution.items())):
        return False, "curated corpus strata do not match curation_summary.json"

    return True, (
        f"{corpus.name}; {row_count:,} rows; package hashes, manifest, QA sample, "
        "source provenance and strata verified"
    )


def validate_raw_source(source: Path, metadata_path: Path) -> tuple[bool, str]:
    """Validate the pinned Epicurious CSV by bytes, digest, schema and row count."""

    if not source.is_file():
        return False, f"configured file not found: {source}"

    actual_size = source.stat().st_size
    if actual_size != SOURCE_FILE_SIZE_BYTES:
        return False, (
            f"size mismatch: expected {SOURCE_FILE_SIZE_BYTES:,}, "
            f"got {actual_size:,} bytes"
        )
    actual_sha = sha256(source)
    if actual_sha.casefold() != SOURCE_FILE_SHA256.casefold():
        return False, "SHA-256 mismatch; re-download the fixed dataset version"

    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            row_count = sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error) as error:
        return False, f"configured file is not readable CSV: {error}"

    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    if missing_columns:
        return False, f"missing required columns: {', '.join(missing_columns)}"
    if row_count != SOURCE_EXPECTED_ROWS:
        return False, (
            f"row-count mismatch: expected {SOURCE_EXPECTED_ROWS:,}, got {row_count:,}"
        )

    metadata, metadata_error = load_source_metadata(metadata_path)
    if metadata_error:
        return False, metadata_error

    if metadata is not None:
        metadata_expectations = {
            "expected_size_bytes": SOURCE_FILE_SIZE_BYTES,
            "expected_row_count": SOURCE_EXPECTED_ROWS,
            "expected_sha256": SOURCE_FILE_SHA256,
            "license": SOURCE_LICENSE,
        }
        for key, expected in metadata_expectations.items():
            if metadata.get(key) != expected:
                return False, f"{metadata_path.name} has an unexpected {key} value"

    return True, (
        f"{source.name}; {row_count:,} rows; pinned size and SHA-256 verified; "
        "required columns present"
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for runtime and rebuild checks."""

    parser = argparse.ArgumentParser(
        description="Check Fridge Rescue runtime prerequisites without exposing secrets."
    )
    parser.add_argument(
        "--check-rebuild",
        "--rebuild",
        dest="check_rebuild",
        action="store_true",
        help="also validate data/raw/epicurious_recipes.csv for rebuilding the corpus",
    )
    return parser


def run_checks(*, check_rebuild: bool = False, project_root: Path | None = None) -> bool:
    """Run required application checks and optionally raw-data rebuild checks."""

    checks: list[bool] = []
    root = project_root or PROJECT_ROOT
    settings = get_settings()

    print("Application runtime (required)")
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 14)
    checks.append(
        status(
            supported_python,
            "Python",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    missing_packages = [
        display_name
        for import_name, display_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]
    checks.append(
        status(
            not missing_packages,
            "Dependencies",
            "all installed" if not missing_packages else ", ".join(missing_packages),
        )
    )

    curated_ok, curated_detail = validate_curated_corpus(
        root / "data" / "curated" / CURATED_CORPUS_NAME,
        root / "data" / "curated" / "curation_summary.json",
    )
    checks.append(status(curated_ok, "Curated recipe corpus", curated_detail))

    print("\nOnline AI mode (optional)")
    status(
        settings.has_valid_llm_key,
        "API config",
        (
            f"{settings.active_provider} / {settings.active_model}; key present"
            if settings.has_valid_llm_key
            else (
                f"{settings.active_provider} / {settings.active_model}; "
                f"optional for offline mode; add {settings.active_key_env_name} to .env "
                "for live AI modes"
            )
        ),
        required=False,
    )

    if check_rebuild:
        print("\nCorpus rebuild (explicit check)")
        missing_rebuild_packages = [
            display_name
            for import_name, display_name in REBUILD_PACKAGES.items()
            if importlib.util.find_spec(import_name) is None
        ]
        checks.append(
            status(
                not missing_rebuild_packages,
                "Rebuild dependencies",
                (
                    "all installed"
                    if not missing_rebuild_packages
                    else ", ".join(missing_rebuild_packages)
                ),
            )
        )
        source_ok, source_detail = validate_raw_source(
            settings.raw_recipe_path,
            root / "data" / "raw" / "source_metadata.json",
        )
        checks.append(status(source_ok, "Raw recipe data", source_detail))
    else:
        print(
            "\nCorpus rebuild: not checked "
            "(run scripts/check_setup.py --check-rebuild when raw data is available)."
        )

    print("\nSecrets were checked only for presence and were not displayed.")
    return all(checks)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return 0 if run_checks(check_rebuild=args.check_rebuild) else 1


if __name__ == "__main__":
    raise SystemExit(main())
