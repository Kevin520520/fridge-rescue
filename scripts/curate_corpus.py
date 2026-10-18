"""Build the reproducible 2,000-recipe Fridge Rescue corpus."""

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

from fridge_rescue.data_pipeline import (
    build_curated_record,
    deterministic_priority,
)
from fridge_rescue.settings import PROJECT_ROOT, get_settings

CURATION_VERSION = "2.0"
SOURCE_COLUMNS = [
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
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def allocate_sqrt_quotas(counts: Counter[str], target: int) -> dict[str, int]:
    """Allocate an exact target with sqrt weighting and largest remainders."""

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


def iter_rows(source: Path, batch_size: int):
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=SOURCE_COLUMNS):
        yield from batch.to_pylist()


def build_record(row: dict[str, Any], rules: dict[str, Any], dataset_version: int):
    return build_curated_record(
        row,
        aliases=rules.get("ingredient_aliases", {}),
        equipment_rules=rules.get("equipment_keywords", {}),
        diet_conflict_markers=rules.get("diet_conflict_markers", {}),
        dataset_version=dataset_version,
        minimum_review_count=int(rules.get("curation", {}).get("minimum_review_count", 0)),
        require_aligned_quantities=True,
        scope_exclusion_markers=rules.get("scope_exclusion_markers", {}),
        equipment_exclusion_phrases=rules.get("equipment_exclusion_phrases", {}),
        reviewed_source_exclusions=rules.get("reviewed_source_exclusions", {}),
    )


def first_pass(
    source: Path,
    rules: dict[str, Any],
    dataset_version: int,
    batch_size: int,
) -> tuple[Counter[str], Counter[str], int, int]:
    strata: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    seen_content: set[str] = set()
    valid_before_dedup = 0
    source_rows = 0

    for source_rows, row in enumerate(iter_rows(source, batch_size), start=1):
        record, reason = build_record(row, rules, dataset_version)
        if record is None:
            rejections[reason or "unknown_rejection"] += 1
            continue
        valid_before_dedup += 1
        dedup_hash = record["dedup_hash"]
        if dedup_hash in seen_content:
            rejections["duplicate_content"] += 1
            continue
        seen_content.add(dedup_hash)
        stratum = f"{record['category_group']}|{record['time_bucket']}"
        strata[stratum] += 1
        if source_rows % 100_000 == 0:
            print(f"Pass 1: scanned {source_rows:,} rows")

    return strata, rejections, valid_before_dedup, source_rows


def push_bounded(
    heap: list[tuple[int, int, dict[str, Any]]],
    record: dict[str, Any],
    priority: int,
    limit: int,
) -> None:
    if limit <= 0:
        return
    entry = (-priority, int(record["recipe_id"]), record)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def second_pass(
    source: Path,
    rules: dict[str, Any],
    dataset_version: int,
    batch_size: int,
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    seen_content: set[str] = set()

    for row_number, row in enumerate(iter_rows(source, batch_size), start=1):
        record, _ = build_record(row, rules, dataset_version)
        if record is None:
            continue
        dedup_hash = record["dedup_hash"]
        if dedup_hash in seen_content:
            continue
        seen_content.add(dedup_hash)
        stratum = f"{record['category_group']}|{record['time_bucket']}"
        priority = deterministic_priority(record["recipe_id"], seed)
        push_bounded(heaps[stratum], record, priority, quotas.get(stratum, 0))
        if row_number % 100_000 == 0:
            print(f"Pass 2: scanned {row_number:,} rows")

    selected: list[dict[str, Any]] = []
    for stratum in sorted(quotas):
        entries = sorted(heaps[stratum], key=lambda item: (-item[0], item[1]))
        if len(entries) != quotas[stratum]:
            raise RuntimeError(
                f"Stratum {stratum} selected {len(entries)} of quota {quotas[stratum]}"
            )
        selected.extend(record for _, _, record in entries)
    selected.sort(key=lambda record: int(record["recipe_id"]))
    return selected


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stable_qa_sample(records: list[dict[str, Any]], size: int = 30) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda row: hashlib.sha256(f"qa-v2:{row['recipe_id']}".encode()).hexdigest(),
    )[:size]
    return [
        {
            "recipe_id": row["recipe_id"],
            "title": row["title"],
            "category_group": row["category_group"],
            "total_time_minutes": row["total_time_minutes_derived"],
            "ingredients_normalized": " | ".join(row["ingredients_normalized"]),
            "instruction_count": len(row["instructions_original"]),
            "quantity_alignment_status": row["quantity_alignment_status"],
            "vegetarian_status": row["vegetarian_status_derived"],
            "vegan_status": row["vegan_status_derived"],
            "qa_status": "",
            "reviewer": "",
            "reviewed_at": "",
            "notes": "",
        }
        for row in ranked
    ]


def write_data_card(
    path: Path,
    *,
    source_sha: str,
    target_size: int,
    seed: int,
    rules_sha: str,
    distribution: Counter[str],
) -> None:
    distribution_lines = "\n".join(
        f"- `{stratum}`: {count}" for stratum, count in sorted(distribution.items())
    )
    content = f"""# Fridge Rescue corpus data card

## Dataset summary

- Source: Food.com Recipes and Reviews on Kaggle
- Source handle: `irkaal/foodcom-recipes-and-reviews/versions/2`
- Source file: `recipes.parquet`
- Source SHA-256: `{source_sha}`
- License: CC0 1.0 Public Domain
- Curated artifacts: {target_size}
- Fixed seed: {seed}
- Curation version: {CURATION_VERSION}
- Curation rules SHA-256: `{rules_sha}`

## Selection and cleaning

The source is scanned in PyArrow batches. Records must have a positive integral recipe ID,
non-empty title, description and category, at least three normalized ingredients, at least two
instructions, and a parseable total time from 1 to 1,440 minutes. HTML entities are decoded before
normalization and hashing. Ingredient and quantity lists must have the same length, with no blank
quantity, and each selected record must have at least two independent source reviews. High-precision
rules remove clearly non-human-food and non-food records; a conservative check rejects a required
wait that is longer than the declared total time. Confirmed source defects from frozen QA audits are
recorded in `rules.yaml` rather than silently repaired. Exact content duplicates are removed using
normalized title, ingredients and instructions. The final sample is allocated across coarse category
and time strata with square-root weighting, then selected by a deterministic SHA-256 priority.

## Important limitations

- Every selected record has aligned, non-blank source quantities, but this structural check cannot
  guarantee that the source recipe is semantically complete.
- Requiring two reviews favors older or more popular Food.com records and does not represent all
  cuisines or newer contributions equally.
- Equipment labels are inferred from small reviewed keyword groups.
- Vegetarian and vegan labels are conservative text screens and may be `unknown`.
- Allergy-related checks are risk flags only, not medical safety certification.
- The corpus is primarily English and is not representative of every cuisine or household.

## Final distribution

{distribution_lines}
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    started = time.perf_counter()
    settings = get_settings()
    config = load_yaml(PROJECT_ROOT / "config.yaml")
    rules_path = PROJECT_ROOT / "rules.yaml"
    rules = load_yaml(rules_path)
    rules_sha = file_sha256(rules_path)
    target_size = args.target_size or int(config["project"]["corpus_size"])
    seed = int(config["project"]["random_seed"])
    source = settings.raw_recipe_path
    if not source.exists():
        raise FileNotFoundError("Run scripts/download_source.py first")

    source_metadata = json.loads(
        (PROJECT_ROOT / "data" / "raw" / "source_metadata.json").read_text(encoding="utf-8")
    )
    output_dir = PROJECT_ROOT / "data" / "curated"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Pass 1: validating, deduplicating and counting strata")
    strata, rejections, valid_before_dedup, source_rows = first_pass(
        source, rules, settings.kaggle_dataset_version, args.batch_size
    )
    quotas = allocate_sqrt_quotas(strata, target_size)

    print("Pass 2: deterministic bounded sampling")
    selected = second_pass(
        source,
        rules,
        settings.kaggle_dataset_version,
        args.batch_size,
        quotas,
        seed,
    )
    if len(selected) != target_size:
        raise AssertionError(f"Expected {target_size} selected records, got {len(selected)}")
    if len({row["recipe_id"] for row in selected}) != target_size:
        raise AssertionError("Selected recipe IDs are not unique")
    if len({row["record_hash"] for row in selected}) != target_size:
        raise AssertionError("Selected record hashes are not unique")
    if any(row["quantity_alignment_status"] != "aligned" for row in selected):
        raise AssertionError("Curation v2 selected a record without aligned quantities")

    corpus_path = output_dir / f"recipes_{target_size}.parquet"
    corpus_temp = corpus_path.with_suffix(corpus_path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(selected), corpus_temp, compression="zstd")
    verified = pq.ParquetFile(corpus_temp)
    try:
        if verified.metadata.num_rows != target_size:
            raise AssertionError("Temporary curated Parquet failed row-count verification")
    finally:
        verified.close()
    corpus_temp.replace(corpus_path)

    selected_distribution = Counter(
        f"{row['category_group']}|{row['time_bucket']}" for row in selected
    )
    manifest_rows = [
        {
            "recipe_id": row["recipe_id"],
            "record_hash": row["record_hash"],
            "dedup_hash": row["dedup_hash"],
            "category_group": row["category_group"],
            "time_bucket": row["time_bucket"],
            "selection_priority": deterministic_priority(row["recipe_id"], seed),
            "quality_flags": " | ".join(row["quality_flags"]),
            "source_handle": settings.kaggle_dataset_handle,
            "source_version": settings.kaggle_dataset_version,
            "source_sha256": source_metadata["sha256"],
            "curation_rules_sha256": rules_sha,
            "license": "CC0-1.0",
            "curation_version": CURATION_VERSION,
        }
        for row in selected
    ]
    manifest_path = output_dir / "manifest.csv"
    write_csv_atomic(manifest_path, manifest_rows, list(manifest_rows[0]))

    qa_rows = stable_qa_sample(selected)
    qa_path = output_dir / "qa_sample.csv"
    write_csv_atomic(qa_path, qa_rows, list(qa_rows[0]))

    data_card_path = output_dir / "DATA_CARD.md"
    write_data_card(
        data_card_path,
        source_sha=source_metadata["sha256"],
        target_size=target_size,
        seed=seed,
        rules_sha=rules_sha,
        distribution=selected_distribution,
    )

    summary = {
        "curation_version": CURATION_VERSION,
        "source_rows": source_rows,
        "valid_before_content_dedup": valid_before_dedup,
        "eligible_after_content_dedup": sum(strata.values()),
        "selected_rows": len(selected),
        "random_seed": seed,
        "minimum_review_count": int(
            rules.get("curation", {}).get("minimum_review_count", 0)
        ),
        "batch_size": args.batch_size,
        "rejection_counts": dict(sorted(rejections.items())),
        "strata_candidates": dict(sorted(strata.items())),
        "strata_quotas": dict(sorted(quotas.items())),
        "strata_selected": dict(sorted(selected_distribution.items())),
        "source_sha256": source_metadata["sha256"],
        "curation_rules_sha256": rules_sha,
        "output_sha256": {
            corpus_path.name: file_sha256(corpus_path),
            manifest_path.name: file_sha256(manifest_path),
            qa_path.name: file_sha256(qa_path),
            data_card_path.name: file_sha256(data_card_path),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = output_dir / "curation_summary.json"
    summary_temp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_temp.replace(summary_path)

    print(f"Curated corpus: {corpus_path}")
    print(f"Selected artifacts: {len(selected):,}")
    print(f"Eligible artifacts: {sum(strata.values()):,}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']}")


if __name__ == "__main__":
    main()
