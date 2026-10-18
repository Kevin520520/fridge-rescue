"""Build the reproducible 2,000-recipe Fridge Rescue Curation v3 corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from fridge_rescue.epicurious_pipeline import (
    CURATION_VERSION,
    REQUIRED_COLUMNS,
    SOURCE_DATASET,
    SOURCE_DATASET_URL,
    SOURCE_FILE_NAME,
    SOURCE_FILE_SHA256,
    SOURCE_FILE_SIZE_BYTES,
    SOURCE_LICENSE,
    SOURCE_REPOSITORY_URL,
    SOURCE_REVISION,
    build_epicurious_record,
    normalize_recipe_reference,
)
from fridge_rescue.settings import PROJECT_ROOT

CURATED_SCHEMA = pa.schema(
    [
        pa.field("recipe_id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("category_original", pa.string()),
        pa.field("category_group", pa.string(), nullable=False),
        pa.field("description_original", pa.string()),
        pa.field("keywords_original", pa.list_(pa.string()), nullable=False),
        pa.field("ingredient_lines_original", pa.list_(pa.string()), nullable=False),
        pa.field("ingredients_normalized", pa.list_(pa.string()), nullable=False),
        pa.field("source_ingredient_line_count", pa.int64(), nullable=False),
        pa.field("removed_non_ingredient_line_count", pa.int64(), nullable=False),
        pa.field("quantity_format_correction_count", pa.int64(), nullable=False),
        pa.field("instructions_original", pa.list_(pa.string()), nullable=False),
        pa.field("prep_time_minutes", pa.int64()),
        pa.field("cook_time_minutes", pa.int64()),
        pa.field("total_time_minutes_derived", pa.int64()),
        pa.field("rating", pa.float64()),
        pa.field("review_count", pa.int64()),
        pa.field("servings", pa.int64()),
        pa.field("yield_original", pa.string()),
        pa.field("equipment_tags_inferred", pa.list_(pa.string()), nullable=False),
        pa.field("vegetarian_status_derived", pa.string(), nullable=False),
        pa.field("vegan_status_derived", pa.string(), nullable=False),
        pa.field("complexity_bucket", pa.string(), nullable=False),
        pa.field("measurement_coverage", pa.float64(), nullable=False),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_dataset_url", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("source_file_sha256", pa.string(), nullable=False),
        pa.field("source_license", pa.string(), nullable=False),
        pa.field("source_row_hash", pa.string(), nullable=False),
        pa.field("dataset_version", pa.string(), nullable=False),
        pa.field("quality_flags", pa.list_(pa.string()), nullable=False),
        pa.field("content_for_embedding", pa.string(), nullable=False),
        pa.field("dedup_hash", pa.string(), nullable=False),
        pa.field("record_hash", pa.string(), nullable=False),
    ]
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def allocate_sqrt_quotas(counts: Counter[str], target: int) -> dict[str, int]:
    """Allocate an exact target with sqrt weighting and largest remainders."""

    if target < 0:
        raise ValueError("target cannot be negative")
    if target == 0:
        return {key: 0 for key in counts}
    if sum(counts.values()) < target:
        raise ValueError(f"Only {sum(counts.values())} eligible artifacts for target {target}")
    weights = {key: math.sqrt(count) for key, count in counts.items() if count > 0}
    total_weight = sum(weights.values())
    raw = {key: target * weight / total_weight for key, weight in weights.items()}
    quotas = {key: min(counts[key], math.floor(value)) for key, value in raw.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda key: (raw[key] - math.floor(raw[key]), weights[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] >= counts[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("Quota allocation stopped before reaching the target")
    return quotas


def deterministic_priority(recipe_id: str, seed: int) -> int:
    payload = f"{seed}:{recipe_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def iter_rows(source: Path):
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        yield from reader


def _curation_thresholds(rules: dict[str, Any]) -> tuple[int, int, float]:
    curation = rules.get("curation", {})
    minimum_ingredients = int(curation.get("minimum_ingredients", 3))
    minimum_instructions = int(curation.get("minimum_instructions", 2))
    minimum_measurement_coverage = float(curation.get("minimum_measurement_coverage", 0.0))
    if minimum_ingredients < 3:
        raise ValueError("curation.minimum_ingredients must be at least 3")
    if minimum_instructions < 2:
        raise ValueError("curation.minimum_instructions must be at least 2")
    if not 0.0 <= minimum_measurement_coverage <= 1.0:
        raise ValueError("curation.minimum_measurement_coverage must be between 0 and 1")
    return minimum_ingredients, minimum_instructions, minimum_measurement_coverage


def build_record(
    row: dict[str, Any],
    rules: dict[str, Any],
    source_sha256: str,
    known_recipe_titles: set[str] | frozenset[str] | None = None,
):
    minimum_ingredients, minimum_instructions, minimum_coverage = _curation_thresholds(rules)
    return build_epicurious_record(
        row,
        source_file_sha256=source_sha256,
        ingredient_aliases=rules.get("ingredient_aliases", {}),
        equipment_rules=rules.get("equipment_keywords", {}),
        equipment_exclusion_phrases=rules.get("equipment_exclusion_phrases", {}),
        diet_conflict_markers=rules.get("diet_conflict_markers", {}),
        scope_exclusion_markers=rules.get("scope_exclusion_markers", {}),
        reviewed_source_exclusions=rules.get("reviewed_source_exclusions", {}),
        minimum_ingredients=minimum_ingredients,
        minimum_instructions=minimum_instructions,
        minimum_measurement_coverage=minimum_coverage,
        known_recipe_titles=known_recipe_titles,
    )


def _stratum(record: dict[str, Any]) -> str:
    return f"{record['category_group']}|{record['complexity_bucket']}"


def first_pass(
    source: Path,
    rules: dict[str, Any],
    source_sha256: str,
) -> tuple[Counter[str], Counter[str], int, int]:
    strata: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    valid_before_dedup = 0
    source_rows = 0

    for source_rows, row in enumerate(iter_rows(source), start=1):
        record, reason = build_record(row, rules, source_sha256)
        if record is None:
            rejections[reason or "unknown_rejection"] += 1
            continue
        valid_before_dedup += 1
        recipe_id = record["recipe_id"]
        if recipe_id in seen_ids:
            rejections["duplicate_source_record_id"] += 1
            continue
        seen_ids.add(recipe_id)
        dedup_hash = record["dedup_hash"]
        if dedup_hash in seen_content:
            rejections["duplicate_content"] += 1
            continue
        seen_content.add(dedup_hash)
        strata[_stratum(record)] += 1

    return strata, rejections, valid_before_dedup, source_rows


def push_bounded(
    heap: list[tuple[int, str, dict[str, Any]]],
    record: dict[str, Any],
    priority: int,
    limit: int,
) -> None:
    if limit <= 0:
        return
    entry = (-priority, str(record["recipe_id"]), record)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def second_pass(
    source: Path,
    rules: dict[str, Any],
    source_sha256: str,
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_content: set[str] = set()

    for row in iter_rows(source):
        record, _ = build_record(row, rules, source_sha256)
        if record is None:
            continue
        recipe_id = record["recipe_id"]
        if recipe_id in seen_ids:
            continue
        seen_ids.add(recipe_id)
        dedup_hash = record["dedup_hash"]
        if dedup_hash in seen_content:
            continue
        seen_content.add(dedup_hash)
        stratum = _stratum(record)
        priority = deterministic_priority(recipe_id, seed)
        push_bounded(heaps[stratum], record, priority, quotas.get(stratum, 0))

    selected: list[dict[str, Any]] = []
    for stratum in sorted(quotas):
        entries = sorted(heaps[stratum], key=lambda item: (-item[0], item[1]))
        if len(entries) != quotas[stratum]:
            raise RuntimeError(
                f"Stratum {stratum} selected {len(entries)} of quota {quotas[stratum]}"
            )
        selected.extend(record for _, _, record in entries)
    selected.sort(key=lambda record: record["source_record_id"])
    return selected


def collect_eligible(
    source: Path,
    rules: dict[str, Any],
    source_sha256: str,
) -> tuple[list[dict[str, Any]], Counter[str], int, int]:
    """Validate and deduplicate the modest 13k-row source in one pass."""

    eligible: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    valid_before_dedup = 0
    source_records = list(iter_rows(source))
    known_recipe_titles = {
        normalize_recipe_reference(str(row.get("Title") or ""))
        for row in source_records
        if normalize_recipe_reference(str(row.get("Title") or ""))
    }
    source_rows = len(source_records)
    for row in source_records:
        record, reason = build_record(
            row, rules, source_sha256, known_recipe_titles
        )
        if record is None:
            rejections[reason or "unknown_rejection"] += 1
            continue
        valid_before_dedup += 1
        recipe_id = record["recipe_id"]
        if recipe_id in seen_ids:
            rejections["duplicate_source_record_id"] += 1
            continue
        seen_ids.add(recipe_id)
        dedup_hash = record["dedup_hash"]
        if dedup_hash in seen_content:
            rejections["duplicate_content"] += 1
            continue
        seen_content.add(dedup_hash)
        eligible.append(record)
    return eligible, rejections, valid_before_dedup, source_rows


def sample_eligible(
    records: list[dict[str, Any]], quotas: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        stratum = _stratum(record)
        priority = deterministic_priority(record["recipe_id"], seed)
        push_bounded(heaps[stratum], record, priority, quotas.get(stratum, 0))

    selected: list[dict[str, Any]] = []
    for stratum in sorted(quotas):
        entries = sorted(heaps[stratum], key=lambda item: (-item[0], item[1]))
        if len(entries) != quotas[stratum]:
            raise RuntimeError(
                f"Stratum {stratum} selected {len(entries)} of quota {quotas[stratum]}"
            )
        selected.extend(record for _, _, record in entries)
    selected.sort(key=lambda record: record["source_record_id"])
    return selected


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_parquet_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        table = pa.Table.from_pylist(records, schema=CURATED_SCHEMA)
        pq.write_table(table, temporary, compression="zstd")
        parquet = pq.ParquetFile(temporary)
        try:
            if parquet.metadata.num_rows != len(records):
                raise AssertionError("Temporary curated Parquet failed row-count verification")
            if parquet.schema_arrow != CURATED_SCHEMA:
                raise AssertionError("Temporary curated Parquet failed schema verification")
        finally:
            parquet.close()
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def stable_qa_sample(records: list[dict[str, Any]], size: int = 30) -> list[dict[str, Any]]:
    """Select a deterministic sample spread across available curation strata."""

    if not records or size <= 0:
        return []
    sample_size = min(size, len(records))
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        stratum = (
            f"{row.get('category_group', 'other')}|"
            f"{row.get('complexity_bucket', 'unknown')}"
        )
        by_stratum[stratum].append(row)
    quotas = allocate_sqrt_quotas(
        Counter({key: len(rows) for key, rows in by_stratum.items()}), sample_size
    )
    selected: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        ranked = sorted(
            by_stratum[stratum],
            key=lambda row: hashlib.sha256(f"qa-v3:{row['recipe_id']}".encode()).hexdigest(),
        )
        selected.extend(ranked[: quotas[stratum]])
    selected.sort(key=lambda row: str(row["recipe_id"]))

    qa_rows: list[dict[str, Any]] = []
    for row in selected:
        ingredient_lines = row.get("ingredient_lines_original") or row.get(
            "ingredients_normalized", []
        )
        instructions = row.get("instructions_original", [])
        qa_rows.append(
            {
                "recipe_id": row["recipe_id"],
                "source_record_id": row.get("source_record_id", ""),
                "title": row["title"],
                "category_group": row.get("category_group", "other"),
                "complexity_bucket": row.get("complexity_bucket", ""),
                "ingredient_count": len(ingredient_lines),
                "ingredient_lines_original": " | ".join(ingredient_lines),
                "instruction_count": len(instructions),
                "instructions_original": " | ".join(instructions),
                "measurement_coverage": row.get("measurement_coverage", ""),
                "removed_non_ingredient_line_count": row.get(
                    "removed_non_ingredient_line_count", 0
                ),
                "quantity_format_correction_count": row.get(
                    "quantity_format_correction_count", 0
                ),
                "equipment_tags_inferred": " | ".join(row.get("equipment_tags_inferred", [])),
                "vegetarian_status": row.get("vegetarian_status_derived", "unknown"),
                "vegan_status": row.get("vegan_status_derived", "unknown"),
                "qa_status": "",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
            }
        )
    return qa_rows


def write_data_card(
    path: Path,
    *,
    source_sha: str,
    target_size: int,
    seed: int,
    rules_sha: str,
    distribution: Counter[str],
    thresholds: tuple[int, int, float],
) -> None:
    minimum_ingredients, minimum_instructions, minimum_coverage = thresholds
    distribution_lines = "\n".join(
        f"- `{stratum}`: {count}" for stratum, count in sorted(distribution.items())
    )
    content = f"""# Fridge Rescue corpus data card

## Dataset summary

- Source dataset: {SOURCE_DATASET}
- Source repository: {SOURCE_REPOSITORY_URL}
- Pinned raw file: {SOURCE_DATASET_URL}
- Pinned revision: `{SOURCE_REVISION}`
- Source file: `{SOURCE_FILE_NAME}`
- Source SHA-256: `{source_sha}`
- Source and derived-corpus license: {SOURCE_LICENSE}
- Curated artifacts: {target_size}
- Fixed seed: {seed}
- Curation version: {CURATION_VERSION}
- Curation rules SHA-256: `{rules_sha}`

## Selection and cleaning

The CSV is parsed with Python's CSV reader so embedded newlines remain attached to their records.
`Ingredients` must be a valid Python list containing only non-blank strings. Each selected recipe
contains at least {minimum_ingredients} ingredient lines and {minimum_instructions} newline-delimited
instruction steps. The minimum source-measurement coverage is {minimum_coverage:.0%}. High-precision
phrases remove non-food and pet-food records, reviewed source exclusions are applied by source row
ID, source rows containing markup, missing quantities or unresolved sub-recipe references are
rejected, standalone equipment/metadata/narrative lines are removed, and exact content duplicates
are removed.

Every retained value in `ingredient_lines_original` comes from the source list. Quantities and units
are never guessed. The only text correction restores spacing in unambiguous joined mixed fractions
(for example, `11/2 cups` becomes `1 1/2 cups`) and its count is recorded. The source-line count,
removed non-ingredient-line count and format-correction count remain in every curated row.
`ingredients_normalized` is a lowercase/whitespace-normalized copy of the same complete lines for
matching. Each row also contains the fixed source-file hash and a hash of its original CSV fields.

The final sample is allocated with square-root weighting across derived broad-category and recipe-
complexity strata. Within each stratum, selection uses a fixed-seed SHA-256 priority.

## Important limitations

- Source text is community-published recipe content; deterministic checks improve structural and
  scope quality but do not prove that every recipe has been kitchen-tested.
- Some seasonings are legitimately listed without a numeric quantity (for example, salt to taste).
  Their source wording is preserved and the per-record measurement coverage is reported.
- This source does not provide dependable time, rating, review-count, serving or yield fields; the
  corresponding curated fields are null rather than inferred.
- Equipment and dietary statuses are conservative text-derived labels. `unknown` is not a safety
  guarantee, and allergy-related use still requires human checking.
- The corpus is primarily English and does not represent every cuisine or household.

## Final category and complexity distribution

{distribution_lines}
"""
    write_text_atomic(path, content)


def _verify_fixed_source(source: Path) -> str:
    size = source.stat().st_size
    digest = file_sha256(source)
    if size != SOURCE_FILE_SIZE_BYTES:
        raise ValueError(f"Source size mismatch: expected {SOURCE_FILE_SIZE_BYTES}, got {size}")
    if digest != SOURCE_FILE_SHA256:
        raise ValueError(f"Source SHA-256 mismatch: expected {SOURCE_FILE_SHA256}, got {digest}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / SOURCE_FILE_NAME,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "curated",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_yaml(PROJECT_ROOT / "config.yaml")
    rules_path = PROJECT_ROOT / "rules.yaml"
    rules = load_yaml(rules_path)
    rules_sha = file_sha256(rules_path)
    target_size = args.target_size or int(config["project"]["corpus_size"])
    seed = int(config["project"]["random_seed"])
    if not args.source.exists():
        raise FileNotFoundError("Run scripts/download_source.py first")
    source_sha = _verify_fixed_source(args.source)
    thresholds = _curation_thresholds(rules)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Validating and deduplicating the fixed 13k source")
    eligible, rejections, valid_before_dedup, source_rows = collect_eligible(
        args.source, rules, source_sha
    )
    strata = Counter(_stratum(record) for record in eligible)
    quotas = allocate_sqrt_quotas(strata, target_size)

    print("Applying deterministic category/complexity sampling")
    selected = sample_eligible(eligible, quotas, seed)
    if len(selected) != target_size:
        raise AssertionError(f"Expected {target_size} selected records, got {len(selected)}")
    if len({row["recipe_id"] for row in selected}) != target_size:
        raise AssertionError("Selected recipe IDs are not unique")
    if len({row["record_hash"] for row in selected}) != target_size:
        raise AssertionError("Selected record hashes are not unique")
    if len({row["dedup_hash"] for row in selected}) != target_size:
        raise AssertionError("Selected content hashes are not unique")
    if any(not row["ingredient_lines_original"] for row in selected):
        raise AssertionError("A selected record has no authoritative ingredient lines")

    corpus_path = args.output_dir / f"recipes_{target_size}.parquet"
    write_parquet_atomic(corpus_path, selected)

    selected_distribution = Counter(_stratum(row) for row in selected)
    manifest_rows = [
        {
            "recipe_id": row["recipe_id"],
            "source_record_id": row["source_record_id"],
            "source_row_hash": row["source_row_hash"],
            "record_hash": row["record_hash"],
            "dedup_hash": row["dedup_hash"],
            "category_group": row["category_group"],
            "complexity_bucket": row["complexity_bucket"],
            "measurement_coverage": f"{row['measurement_coverage']:.6f}",
            "removed_non_ingredient_line_count": row[
                "removed_non_ingredient_line_count"
            ],
            "quantity_format_correction_count": row[
                "quantity_format_correction_count"
            ],
            "selection_priority": deterministic_priority(row["recipe_id"], seed),
            "quality_flags": " | ".join(row["quality_flags"]),
            "source_dataset": SOURCE_DATASET,
            "source_url": SOURCE_DATASET_URL,
            "source_revision": SOURCE_REVISION,
            "source_sha256": source_sha,
            "curation_rules_sha256": rules_sha,
            "license": SOURCE_LICENSE,
            "curation_version": CURATION_VERSION,
        }
        for row in selected
    ]
    manifest_path = args.output_dir / "manifest.csv"
    write_csv_atomic(manifest_path, manifest_rows, list(manifest_rows[0]))

    qa_rows = stable_qa_sample(selected)
    qa_path = args.output_dir / "qa_sample.csv"
    write_csv_atomic(qa_path, qa_rows, list(qa_rows[0]))

    data_card_path = args.output_dir / "DATA_CARD.md"
    write_data_card(
        data_card_path,
        source_sha=source_sha,
        target_size=target_size,
        seed=seed,
        rules_sha=rules_sha,
        distribution=selected_distribution,
        thresholds=thresholds,
    )

    coverage_values = [row["measurement_coverage"] for row in selected]
    summary = {
        "curation_version": CURATION_VERSION,
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_DATASET_URL,
        "source_repository_url": SOURCE_REPOSITORY_URL,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "source_rows": source_rows,
        "valid_before_dedup": valid_before_dedup,
        "eligible_after_dedup": sum(strata.values()),
        "selected_rows": len(selected),
        "random_seed": seed,
        "minimum_ingredients": thresholds[0],
        "minimum_instructions": thresholds[1],
        "minimum_measurement_coverage": thresholds[2],
        "selected_measurement_coverage": {
            "minimum": min(coverage_values),
            "mean": sum(coverage_values) / len(coverage_values),
            "maximum": max(coverage_values),
        },
        "selected_records_with_removed_non_ingredient_lines": sum(
            bool(row["removed_non_ingredient_line_count"]) for row in selected
        ),
        "selected_removed_non_ingredient_lines": sum(
            row["removed_non_ingredient_line_count"] for row in selected
        ),
        "selected_records_with_quantity_format_corrections": sum(
            bool(row["quantity_format_correction_count"]) for row in selected
        ),
        "selected_quantity_format_corrections": sum(
            row["quantity_format_correction_count"] for row in selected
        ),
        "rejection_counts": dict(sorted(rejections.items())),
        "strata_candidates": dict(sorted(strata.items())),
        "strata_quotas": dict(sorted(quotas.items())),
        "strata_selected": dict(sorted(selected_distribution.items())),
        "source_sha256": source_sha,
        "curation_rules_sha256": rules_sha,
        "output_sha256": {
            corpus_path.name: file_sha256(corpus_path),
            manifest_path.name: file_sha256(manifest_path),
            qa_path.name: file_sha256(qa_path),
            data_card_path.name: file_sha256(data_card_path),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = args.output_dir / "curation_summary.json"
    write_text_atomic(summary_path, json.dumps(summary, indent=2))

    print(f"Curated corpus: {corpus_path}")
    print(f"Selected artifacts: {len(selected):,}")
    print(f"Eligible artifacts: {sum(strata.values()):,}")
    print(
        f"Mean source measurement coverage: {summary['selected_measurement_coverage']['mean']:.1%}"
    )
    print(f"Elapsed seconds: {summary['elapsed_seconds']}")


if __name__ == "__main__":
    main()
