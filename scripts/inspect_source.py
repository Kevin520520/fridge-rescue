"""Stream a data-quality profile of the configured Food.com Parquet source."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fridge_rescue.data_pipeline import parse_iso8601_minutes
from fridge_rescue.settings import PROJECT_ROOT, get_settings

BATCH_SIZE = 8192
REQUIRED_COLUMNS = frozenset(
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
PROFILE_COLUMNS = (
    "RecipeId",
    "Name",
    "Description",
    "RecipeCategory",
    "TotalTime",
    "RecipeIngredientQuantities",
    "RecipeIngredientParts",
    "RecipeInstructions",
)
TEXT_COLUMNS = frozenset({"Name", "Description", "RecipeCategory", "TotalTime"})
LIST_COLUMNS = frozenset(
    {"RecipeIngredientQuantities", "RecipeIngredientParts", "RecipeInstructions"}
)


def count_empty(values: Iterable[Any], *, list_column: bool) -> int:
    """Count empty non-null strings or lists in a single batch."""

    if list_column:
        return sum(value is not None and len(value) == 0 for value in values)
    return sum(isinstance(value, str) and not value.strip() for value in values)


def profile_source(source: Path, *, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    """Profile source quality batch-by-batch and return a JSON-serializable report."""

    parquet = pq.ParquetFile(source)
    schema_names = set(parquet.schema_arrow.names)
    missing_columns = sorted(REQUIRED_COLUMNS - schema_names)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    key_columns = TEXT_COLUMNS | LIST_COLUMNS
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

    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(PROFILE_COLUMNS)):
        scanned_rows += batch.num_rows
        batch_columns = {
            name: batch.column(batch.schema.get_field_index(name)) for name in PROFILE_COLUMNS
        }

        for name in key_columns:
            array = batch_columns[name]
            null_counts[name] += array.null_count
            empty_counts[name] += count_empty(array.to_pylist(), list_column=name in LIST_COLUMNS)

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
        raise ValueError(
            f"Batch scan row count ({scanned_rows:,}) differs from Parquet metadata "
            f"({metadata_rows:,})"
        )

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
        "required_columns": {
            "all_present": True,
            "missing": [],
        },
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


def print_report(report: dict[str, Any], output: Path) -> None:
    """Print the concise, decision-relevant portion of the profile."""

    print(f"Source: {report['source']}")
    print(f"Total rows: {report['total_rows']:,}")
    print(f"Columns found: {len(report['schema'])}")
    print("Key-column missing values (null / empty):")
    for name, counts in report["quality"]["key_columns"].items():
        print(f"- {name}: {counts['null_count']:,} / {counts['empty_string_or_list_count']:,}")

    duration = report["quality"]["total_time"]
    print("TotalTime quality:")
    print(f"- missing or blank: {duration['missing_or_blank']:,}")
    print(f"- invalid format: {duration['invalid_format']:,}")
    print(f"- non-positive: {duration['non_positive']:,}")
    print(f"- over 1,440 minutes: {duration['over_1440_minutes']:,}")

    recipe_ids = report["quality"]["recipe_id"]
    print("RecipeId quality:")
    print(f"- valid integral rows: {recipe_ids['valid_integral_rows']:,}")
    print(f"- unique valid IDs: {recipe_ids['unique_valid_ids']:,}")
    print(f"- duplicate rows: {recipe_ids['duplicate_rows']:,}")
    print(f"- non-integral: {recipe_ids['non_integral']:,}")
    print(f"- non-positive: {recipe_ids['non_positive']:,}")
    print(f"- null/non-finite: {recipe_ids['null_or_non_finite']:,}")
    print(f"Schema and quality report: {output}")


def main() -> None:
    settings = get_settings()
    source = settings.raw_recipe_path
    if not source.exists():
        raise FileNotFoundError(
            f"{source} is missing. Run scripts/download_source.py before inspection."
        )

    report = profile_source(source)
    output = PROJECT_ROOT / "data" / "raw" / "source_schema.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report, output)


if __name__ == "__main__":
    main()
