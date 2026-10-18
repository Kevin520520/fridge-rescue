"""Profile the reviewed Epicurious CSV before corpus curation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

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
    """Profile the pinned Epicurious CSV used by the current corpus."""

    del batch_size
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
