"""Profile the reviewed Epicurious CSV before corpus curation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fridge_rescue.data_pipeline import parse_iso8601_minutes
from fridge_rescue.epicurious_pipeline import (
    REQUIRED_COLUMNS,
    SOURCE_FILE_NAME,
    measurement_coverage,
    parse_ingredient_lines,
    source_row_hash,
    split_instruction_steps,
)
from fridge_rescue.settings import PROJECT_ROOT

BATCH_SIZE = 8192
LEGACY_REQUIRED_COLUMNS = frozenset(
    {
        "RecipeId",
        "Name",
        "CookTime",
        "PrepTime",
        "TotalTime",
        "Description",
        "RecipeCategory",
        "Keywords",
        "RecipeIngredientQuantities",
        "RecipeIngredientParts",
        "AggregatedRating",
        "ReviewCount",
        "RecipeServings",
        "RecipeYield",
        "RecipeInstructions",
    }
)
LEGACY_PROFILE_COLUMNS = (
    "RecipeId",
    "Name",
    "Description",
    "RecipeCategory",
    "TotalTime",
    "RecipeIngredientQuantities",
    "RecipeIngredientParts",
    "RecipeInstructions",
)
LEGACY_TEXT_COLUMNS = frozenset({"Name", "Description", "RecipeCategory", "TotalTime"})
LEGACY_LIST_COLUMNS = frozenset(
    {"RecipeIngredientQuantities", "RecipeIngredientParts", "RecipeInstructions"}
)


def count_empty(values: Iterable[Any], *, list_column: bool) -> int:
    if list_column:
        return sum(value is not None and len(value) == 0 for value in values)
    return sum(isinstance(value, str) and not value.strip() for value in values)


def _profile_legacy_parquet(source: Path, *, batch_size: int) -> dict[str, Any]:
    """Keep the old profiler available for tests and archived source audits."""

    parquet = pq.ParquetFile(source)
    schema_names = set(parquet.schema_arrow.names)
    missing_columns = sorted(LEGACY_REQUIRED_COLUMNS - schema_names)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    key_columns = LEGACY_TEXT_COLUMNS | LEGACY_LIST_COLUMNS
    null_counts = {name: 0 for name in key_columns}
    empty_counts = {name: 0 for name in key_columns}
    duration_counts = {
        "missing_or_blank": 0,
        "invalid_format": 0,
        "non_positive": 0,
        "over_1440_minutes": 0,
        "valid_1_to_1440_minutes": 0,
    }
    recipe_id_counts = {
        "null_or_non_finite": 0,
        "non_integral": 0,
        "non_positive": 0,
        "valid_integral_rows": 0,
        "duplicate_rows": 0,
    }
    seen_recipe_ids: set[int] = set()
    scanned_rows = 0

    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(LEGACY_PROFILE_COLUMNS)):
        scanned_rows += batch.num_rows
        batch_columns = {
            name: batch.column(batch.schema.get_field_index(name))
            for name in LEGACY_PROFILE_COLUMNS
        }
        for name in key_columns:
            array = batch_columns[name]
            null_counts[name] += array.null_count
            empty_counts[name] += count_empty(
                array.to_pylist(), list_column=name in LEGACY_LIST_COLUMNS
            )
        for value in batch_columns["TotalTime"].to_pylist():
            if value is None or (isinstance(value, str) and not value.strip()):
                duration_counts["missing_or_blank"] += 1
                continue
            minutes = parse_iso8601_minutes(value)
            if minutes is None:
                duration_counts["invalid_format"] += 1
            elif minutes <= 0:
                duration_counts["non_positive"] += 1
            elif minutes > 1440:
                duration_counts["over_1440_minutes"] += 1
            else:
                duration_counts["valid_1_to_1440_minutes"] += 1
        for value in batch_columns["RecipeId"].to_pylist():
            if value is None:
                recipe_id_counts["null_or_non_finite"] += 1
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                recipe_id_counts["null_or_non_finite"] += 1
                continue
            if not math.isfinite(numeric):
                recipe_id_counts["null_or_non_finite"] += 1
            elif not numeric.is_integer():
                recipe_id_counts["non_integral"] += 1
            elif numeric <= 0:
                recipe_id_counts["non_positive"] += 1
            else:
                recipe_id = int(numeric)
                recipe_id_counts["valid_integral_rows"] += 1
                if recipe_id in seen_recipe_ids:
                    recipe_id_counts["duplicate_rows"] += 1
                else:
                    seen_recipe_ids.add(recipe_id)

    metadata_rows = parquet.metadata.num_rows
    if scanned_rows != metadata_rows:
        raise ValueError("Parquet batch scan differs from metadata row count")
    recipe_id_counts["unique_valid_ids"] = len(seen_recipe_ids)
    recipe_id_counts["all_rows_unique_positive_integers"] = (
        recipe_id_counts["valid_integral_rows"] == scanned_rows
        and recipe_id_counts["duplicate_rows"] == 0
    )
    return {
        "source": str(source),
        "row_groups": parquet.num_row_groups,
        "total_rows": metadata_rows,
        "scan_batch_size": batch_size,
        "required_columns": {"all_present": True, "missing": []},
        "schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in parquet.schema_arrow
        ],
        "quality": {
            "key_columns": {
                name: {
                    "null_count": null_counts[name],
                    "empty_string_or_list_count": empty_counts[name],
                }
                for name in sorted(key_columns)
            },
            "total_time": duration_counts,
            "recipe_id": recipe_id_counts,
        },
    }


def _count_bucket(value: int, *, kind: str) -> str:
    if kind == "ingredients":
        if value < 3:
            return "00-02"
        if value <= 6:
            return "03-06"
        if value <= 12:
            return "07-12"
        return "13+"
    if value < 2:
        return "00-01"
    if value <= 4:
        return "02-04"
    if value <= 8:
        return "05-08"
    return "09+"


def _profile_epicurious_csv(source: Path) -> dict[str, Any]:
    blank_counts: Counter[str] = Counter()
    ingredient_parse_errors: Counter[str] = Counter()
    ingredient_count_buckets: Counter[str] = Counter()
    instruction_count_buckets: Counter[str] = Counter()
    measurement_buckets: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    seen_rows: set[str] = set()
    duplicate_source_rows = 0
    coverage_sum = 0.0
    coverage_min = 1.0
    coverage_max = 0.0
    total_rows = 0

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        for total_rows, row in enumerate(reader, start=1):
            for name in REQUIRED_COLUMNS:
                if not str(row.get(name) or "").strip():
                    blank_counts[name or "source_record_id"] += 1

            source_id = str(row.get("") or "").strip()
            if source_id:
                if source_id in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(source_id)

            row_hash = source_row_hash(row)
            if row_hash in seen_rows:
                duplicate_source_rows += 1
            else:
                seen_rows.add(row_hash)

            lines, error = parse_ingredient_lines(row.get("Ingredients"))
            if error:
                ingredient_parse_errors[error] += 1
            else:
                assert lines is not None
                ingredient_count_buckets[_count_bucket(len(lines), kind="ingredients")] += 1
                coverage = measurement_coverage(lines)
                coverage_sum += coverage
                coverage_min = min(coverage_min, coverage)
                coverage_max = max(coverage_max, coverage)
                if coverage < 0.5:
                    measurement_buckets["0.00-0.49"] += 1
                elif coverage < 0.75:
                    measurement_buckets["0.50-0.74"] += 1
                elif coverage < 1.0:
                    measurement_buckets["0.75-0.99"] += 1
                else:
                    measurement_buckets["1.00"] += 1

            steps = split_instruction_steps(row.get("Instructions"))
            instruction_count_buckets[_count_bucket(len(steps), kind="instructions")] += 1

    parsed_rows = total_rows - sum(ingredient_parse_errors.values())
    return {
        "source": str(source),
        "format": "csv",
        "total_rows": total_rows,
        "required_columns": {"all_present": True, "missing": []},
        "schema": [{"name": name, "type": "string"} for name in fieldnames],
        "quality": {
            "blank_fields": dict(sorted(blank_counts.items())),
            "source_record_id": {
                "unique_nonblank": len(seen_ids),
                "duplicate_rows": duplicate_ids,
            },
            "ingredient_parsing": {
                "valid_list_rows": parsed_rows,
                "errors": dict(sorted(ingredient_parse_errors.items())),
                "count_buckets": dict(sorted(ingredient_count_buckets.items())),
            },
            "instructions": {"count_buckets": dict(sorted(instruction_count_buckets.items()))},
            "source_measurement_coverage": {
                "mean": round(coverage_sum / parsed_rows, 6) if parsed_rows else 0.0,
                "minimum": round(coverage_min, 6) if parsed_rows else 0.0,
                "maximum": round(coverage_max, 6) if parsed_rows else 0.0,
                "buckets": dict(sorted(measurement_buckets.items())),
            },
            "exact_duplicate_source_rows": duplicate_source_rows,
        },
    }


def profile_source(source: Path, *, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    """Profile either the current CSV or an archived Food.com Parquet source."""

    with source.open("rb") as handle:
        is_parquet = handle.read(4) == b"PAR1"
    if is_parquet:
        return _profile_legacy_parquet(source, batch_size=batch_size)
    return _profile_epicurious_csv(source)


def print_report(report: dict[str, Any], output: Path) -> None:
    print(f"Source: {report['source']}")
    print(f"Total rows: {report['total_rows']:,}")
    if report.get("format") == "csv":
        parsing = report["quality"]["ingredient_parsing"]
        coverage = report["quality"]["source_measurement_coverage"]
        print(f"Valid ingredient-list rows: {parsing['valid_list_rows']:,}")
        print(f"Ingredient parse errors: {sum(parsing['errors'].values()):,}")
        print(f"Mean source measurement coverage: {coverage['mean']:.1%}")
        print(f"Duplicate source IDs: {report['quality']['source_record_id']['duplicate_rows']:,}")
    else:
        print(f"Columns found: {len(report['schema'])}")
    print(f"Schema and quality report: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / SOURCE_FILE_NAME,
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if not args.source.exists():
        raise FileNotFoundError(
            f"{args.source} is missing. Run scripts/download_source.py before inspection."
        )

    report = profile_source(args.source, batch_size=args.batch_size)
    output = PROJECT_ROOT / "data" / "raw" / "source_schema.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print_report(report, output)


if __name__ == "__main__":
    main()
