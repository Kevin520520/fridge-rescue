"""Reproducibly audit the Curation v3.1 recipe corpus.

The audit separates hard data-contract failures from review flags. Hard failures
make the command exit with status 1. Review flags remain visible in
``flagged_records.csv`` and never silently remove a record. Any ingredient line
that violates the v3.1 curation contract is a hard failure in the final corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from fridge_rescue.curation_common import scope_exclusion_reason
from fridge_rescue.epicurious_pipeline import (
    content_dedup_hash,
    ingredient_line_rejection_reason,
    is_unexpanded_subrecipe_reference,
    measurement_coverage,
    merge_scope_markers,
    normalize_ingredient_lines,
    normalize_joined_mixed_fraction,
    normalize_recipe_reference,
    stable_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_VERSION = "1.1"
DEFAULT_EXPECTED_ROWS = 2_000
MINIMUM_MEASUREMENT_COVERAGE = 0.60

REQUIRED_COLUMNS = frozenset(
    {
        "recipe_id",
        "title",
        "ingredient_lines_original",
        "ingredients_normalized",
        "source_ingredient_line_count",
        "removed_non_ingredient_line_count",
        "instructions_original",
        "measurement_coverage",
        "source_dataset",
        "source_dataset_url",
        "source_record_id",
        "source_file_sha256",
        "source_license",
        "dataset_version",
        "dedup_hash",
        "record_hash",
    }
)

# These columns indicate reconstructed quantities rather than source-preserved units.
# Their presence would make the displayed quantities and units ambiguous.
DISALLOWED_SPLIT_QUANTITY_COLUMNS = frozenset(
    {
        "ingredients_original",
        "ingredient_quantities_original",
        "quantity_alignment_status",
        "ingredient_units_original",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEADING_NUMBER = re.compile(
    r"^\s*(?:about\s+|approximately\s+|approx\.?\s+)?"
    r"(?P<amount>\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_RANGE_TAIL = re.compile(
    r"^\s*(?:-|–|—|to)\s*\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)
_UNIT_AT_START = re.compile(
    r"^\s*(?:[-–—]\s*)?(?:"
    r"%|kg\b|kilograms?\b|g\b|grams?\b|mg\b|milligrams?\b|"
    r"lb\b|lbs\b|pounds?\b|oz\b|ounces?\b|"
    r"ml\b|millilit(?:er|re)s?\b|l\b|lit(?:er|re)s?\b|"
    r"cups?\b|c\.\b|tablespoons?\b|tbsp\.?\b|"
    r"teaspoons?\b|tsp\.?\b|pints?\b|quarts?\b|gallons?\b|"
    r"fl\.?\s*oz\b|fluid\s+ounces?\b|"
    r"pieces?\b|"
    r"inches?\b|cm\b|centimeters?\b|mm\b|millimeters?\b"
    r")",
    re.IGNORECASE,
)

# These patterns flag probable equipment or non-food supplies for inspection.
# They do not delete records automatically.
_EQUIPMENT_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("equipment_label", r"^\s*(?:special\s+)?equipment\s*:"),
        ("cheesecloth", r"\bcheesecloth\b"),
        ("twine_or_string", r"\b(?:kitchen|butcher[’']?s)\s+(?:twine|string)\b"),
        ("skewer", r"\b(?:wooden|bamboo|metal)\s+skewers?\b"),
        ("pick", r"\b(?:toothpicks?|cocktail\s+(?:picks?|sticks?))\b"),
        ("paper_or_wrap", r"\b(?:parchment|wax(?:ed)?)\s+paper\b|\bplastic\s+wrap\b"),
        ("foil", r"\b(?:heavy-duty\s+)?alumin(?:um|ium)\s+foil\b|\bheavy-duty\s+foil\b"),
        ("thermometer", r"\b(?:instant-read|deep-fry|candy)\s+thermometer\b"),
        ("ice_cream_maker", r"\bice[- ]cream\s+maker\b"),
        ("cookie_cutter", r"\b(?:cookie|biscuit)\s+cutters?\b"),
        ("springform_pan", r"\bspringform\s+pan\b"),
        ("muffin_tin", r"\bmuffin\s+tins?\b"),
    )
)
_FOOD_APPLIED_TO_EQUIPMENT = re.compile(
    r"\b(?:oil|butter|flour|cooking\s+spray)\b.*\bfor\s+(?:the|a|an)\b",
    re.IGNORECASE,
)
_MARKUP_RESIDUE = re.compile(r"</?[a-z][^>]*>", re.IGNORECASE)
_SUBRECIPE_REFERENCE_MARKER = re.compile(
    r"(?:\breserved\s+from\b|\bsee\s+(?:the\s+)?recipe\b|"
    r"^\s*(?:one|\d+(?:\s+\d+/\d+)?|\d+/\d+)\s+recipe(?:\s+of)?\b)",
    re.IGNORECASE,
)

RECORD_AUDIT_FIELDS = [
    "row_index",
    "recipe_id",
    "title",
    "hard_pass",
    "hard_failure_codes",
    "warning_codes",
    "ingredient_count",
    "instruction_count",
    "measurement_coverage",
    "recomputed_measurement_coverage",
    "scope_exclusion_reason",
    "large_unitless_line_count",
    "equipment_line_count",
    "retained_rejected_line_count",
    "joined_fraction_line_count",
    "subrecipe_reference_line_count",
]

FLAGGED_FIELDS = [
    "scope",
    "row_index",
    "recipe_id",
    "title",
    "severity",
    "flag_code",
    "field",
    "line_index",
    "text",
    "details",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def large_unitless_amount(line: str, *, threshold: float = 100.0) -> float | None:
    """Return a suspicious leading amount when no source unit follows it."""

    match = _LEADING_NUMBER.search(line)
    if match is None:
        return None
    amount = float(match.group("amount").replace(",", ""))
    if amount < threshold:
        return None
    remainder = line[match.end() :]
    range_match = _RANGE_TAIL.search(remainder)
    if range_match is not None:
        remainder = remainder[range_match.end() :]
    if _UNIT_AT_START.search(remainder):
        return None
    return amount


def equipment_line_markers(line: str) -> list[str]:
    """Return high-precision reasons that a line may be equipment-only."""

    if (
        _FOOD_APPLIED_TO_EQUIPMENT.search(line)
        or ingredient_line_rejection_reason(line) != "equipment_only"
    ):
        return []
    markers = [name for name, pattern in _EQUIPMENT_MARKERS if pattern.search(line)]
    return markers or ["pipeline_equipment_prefix"]


def _is_nonblank_sequence(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _source_contract(summary: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source_dataset": str(summary.get("source_dataset") or "").strip(),
        "source_dataset_url": str(summary.get("source_url") or "").strip(),
        "source_file_sha256": str(summary.get("source_sha256") or "").strip(),
        "source_license": str(summary.get("source_license") or "").strip(),
        "dataset_version": str(summary.get("curation_version") or "").strip(),
        "source_revision": str(summary.get("source_revision") or "").strip(),
    }


def _append_record_flag(
    flagged: list[dict[str, Any]],
    hard_codes: set[str],
    warning_codes: set[str],
    row_index: int,
    recipe_id: str,
    title: str,
    code: str,
    *,
    severity: str,
    field: str = "",
    line_index: int | str = "",
    text: str = "",
    details: str = "",
) -> None:
    if severity == "hard_failure":
        hard_codes.add(code)
    else:
        warning_codes.add(code)
    flagged.append(
        {
            "scope": "record",
            "row_index": row_index,
            "recipe_id": recipe_id,
            "title": title,
            "severity": severity,
            "flag_code": code,
            "field": field,
            "line_index": line_index,
            "text": text,
            "details": details,
        }
    )


def audit_corpus(
    *,
    corpus_path: Path,
    summary_path: Path,
    rules_path: Path,
    output_dir: Path,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
) -> dict[str, Any]:
    """Audit one curated corpus and write all three evidence artifacts."""

    if expected_rows < 1:
        raise ValueError("expected_rows must be positive")
    with summary_path.open(encoding="utf-8") as source:
        summary = json.load(source)
    rules = load_yaml(rules_path)
    table = pq.read_table(corpus_path)
    rows: list[dict[str, Any]] = table.to_pylist()
    columns = set(table.column_names)
    contract = _source_contract(summary)
    scope_markers = merge_scope_markers(rules.get("scope_exclusion_markers", {}))

    global_failures: list[dict[str, str]] = []
    flagged: list[dict[str, Any]] = []

    def add_global(code: str, details: str) -> None:
        global_failures.append({"code": code, "details": details})
        flagged.append(
            {
                "scope": "corpus",
                "row_index": "",
                "recipe_id": "",
                "title": "",
                "severity": "hard_failure",
                "flag_code": code,
                "field": "",
                "line_index": "",
                "text": "",
                "details": details,
            }
        )

    if len(rows) != expected_rows:
        add_global("unexpected_row_count", f"expected {expected_rows}, found {len(rows)}")
    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    if missing_columns:
        add_global("missing_required_columns", ", ".join(missing_columns))
    disallowed_columns = sorted(DISALLOWED_SPLIT_QUANTITY_COLUMNS & columns)
    if disallowed_columns:
        add_global("split_quantity_columns_present", ", ".join(disallowed_columns))

    if summary.get("selected_rows") != expected_rows:
        add_global(
            "summary_selected_rows_mismatch",
            f"expected {expected_rows}, summary has {summary.get('selected_rows')!r}",
        )
    configured_minimum = summary.get("minimum_measurement_coverage")
    try:
        configured_minimum_value = float(configured_minimum)
    except (TypeError, ValueError):
        configured_minimum_value = -1.0
    if configured_minimum_value < MINIMUM_MEASUREMENT_COVERAGE:
        add_global(
            "summary_measurement_threshold_too_low",
            f"required at least {MINIMUM_MEASUREMENT_COVERAGE:.2f}, "
            f"summary has {configured_minimum!r}",
        )

    for field, value in contract.items():
        if not value:
            add_global("blank_source_contract_value", field)
    if contract["source_file_sha256"] and not _SHA256.fullmatch(
        contract["source_file_sha256"]
    ):
        add_global("invalid_summary_source_sha256", contract["source_file_sha256"])
    revision = contract["source_revision"]
    if revision and revision not in contract["source_dataset_url"]:
        add_global(
            "unpinned_summary_source_url",
            "source_revision is absent from the summary source_url",
        )

    unique_fields = ("recipe_id", "source_record_id", "record_hash", "dedup_hash")
    value_counts = {
        field: Counter(str(row.get(field) or "").strip() for row in rows)
        for field in unique_fields
    }
    normalized_recipe_titles = {
        normalized
        for row in rows
        if (normalized := normalize_recipe_reference(str(row.get("title") or "")))
    }

    record_audit: list[dict[str, Any]] = []
    hard_counts: Counter[str] = Counter(item["code"] for item in global_failures)
    warning_counts: Counter[str] = Counter()

    for row_index, row in enumerate(rows, start=1):
        recipe_id = str(row.get("recipe_id") or "").strip()
        title = str(row.get("title") or "").strip()
        hard_codes: set[str] = set()
        warning_codes: set[str] = set()
        scope_reason = ""
        large_count = 0
        equipment_count = 0
        retained_rejected_count = 0
        joined_fraction_count = 0
        subrecipe_reference_count = 0
        add_record_flag = partial(
            _append_record_flag,
            flagged,
            hard_codes,
            warning_codes,
            row_index,
            recipe_id,
            title,
        )

        if not recipe_id:
            add_record_flag("blank_recipe_id", severity="hard_failure", field="recipe_id")
        if not title:
            add_record_flag("blank_title", severity="hard_failure", field="title")

        for field in unique_fields:
            value = str(row.get(field) or "").strip()
            if not value:
                add_record_flag(
                    f"blank_{field}", severity="hard_failure", field=field
                )
            elif value_counts[field][value] > 1:
                add_record_flag(
                    f"duplicate_{field}",
                    severity="hard_failure",
                    field=field,
                    text=value,
                )

        ingredients = row.get("ingredient_lines_original")
        normalized = row.get("ingredients_normalized")
        instructions = row.get("instructions_original")
        ingredient_count = len(ingredients) if isinstance(ingredients, list) else 0
        instruction_count = len(instructions) if isinstance(instructions, list) else 0

        if not _is_nonblank_sequence(ingredients):
            add_record_flag(
                "blank_or_invalid_ingredient_lines",
                severity="hard_failure",
                field="ingredient_lines_original",
            )
        if ingredient_count < 3:
            add_record_flag(
                "fewer_than_three_ingredients",
                severity="hard_failure",
                field="ingredient_lines_original",
                details=f"found {ingredient_count}",
            )
        source_ingredient_line_count = row.get("source_ingredient_line_count")
        if source_ingredient_line_count != ingredient_count:
            add_record_flag(
                "source_ingredient_line_count_mismatch",
                severity="hard_failure",
                field="source_ingredient_line_count",
                text=str(source_ingredient_line_count),
                details=f"source={source_ingredient_line_count}, retained={ingredient_count}",
            )
        removed_non_ingredient_line_count = row.get(
            "removed_non_ingredient_line_count"
        )
        if removed_non_ingredient_line_count != 0:
            add_record_flag(
                "hidden_source_ingredient_lines",
                severity="hard_failure",
                field="removed_non_ingredient_line_count",
                text=str(removed_non_ingredient_line_count),
                details="released recipes must retain every accepted source ingredient line",
            )
        if not _is_nonblank_sequence(normalized):
            add_record_flag(
                "blank_or_invalid_normalized_ingredients",
                severity="hard_failure",
                field="ingredients_normalized",
            )
        if isinstance(ingredients, list) and isinstance(normalized, list):
            if len(ingredients) != len(normalized):
                add_record_flag(
                    "ingredient_array_length_mismatch",
                    severity="hard_failure",
                    field="ingredients_normalized",
                    details=f"original={len(ingredients)}, normalized={len(normalized)}",
                )
            elif (
                _is_nonblank_sequence(ingredients)
                and _is_nonblank_sequence(normalized)
                and normalize_ingredient_lines(ingredients) != normalized
            ):
                add_record_flag(
                    "normalized_ingredients_mismatch",
                    severity="hard_failure",
                    field="ingredients_normalized",
                )

        if not _is_nonblank_sequence(instructions):
            add_record_flag(
                "blank_or_invalid_instruction_steps",
                severity="hard_failure",
                field="instructions_original",
            )
        if instruction_count < 2:
            add_record_flag(
                "fewer_than_two_instructions",
                severity="hard_failure",
                field="instructions_original",
                details=f"found {instruction_count}",
            )

        stored_coverage = row.get("measurement_coverage")
        try:
            stored_coverage_value = float(stored_coverage)
        except (TypeError, ValueError):
            stored_coverage_value = -1.0
            add_record_flag(
                "invalid_measurement_coverage",
                severity="hard_failure",
                field="measurement_coverage",
                text=str(stored_coverage),
            )
        recomputed_coverage = (
            measurement_coverage(ingredients)
            if _is_nonblank_sequence(ingredients)
            else 0.0
        )
        if stored_coverage_value < MINIMUM_MEASUREMENT_COVERAGE:
            add_record_flag(
                "measurement_coverage_below_0_60",
                severity="hard_failure",
                field="measurement_coverage",
                text=str(stored_coverage),
            )
        if not 0.0 <= stored_coverage_value <= 1.0:
            add_record_flag(
                "measurement_coverage_out_of_range",
                severity="hard_failure",
                field="measurement_coverage",
                text=str(stored_coverage),
            )
        if abs(stored_coverage_value - recomputed_coverage) > 1e-9:
            add_record_flag(
                "measurement_coverage_mismatch",
                severity="hard_failure",
                field="measurement_coverage",
                details=f"stored={stored_coverage_value}, recomputed={recomputed_coverage}",
            )

        for field, expected in contract.items():
            if field == "source_revision":
                continue
            actual = str(row.get(field) or "").strip()
            if actual != expected:
                add_record_flag(
                    f"inconsistent_{field}",
                    severity="hard_failure",
                    field=field,
                    text=actual,
                    details=f"expected {expected!r}",
                )
        if revision and revision not in str(row.get("source_dataset_url") or ""):
            add_record_flag(
                "unpinned_record_source_url",
                severity="hard_failure",
                field="source_dataset_url",
            )
        row_source_sha = str(row.get("source_file_sha256") or "").strip()
        if row_source_sha and not _SHA256.fullmatch(row_source_sha):
            add_record_flag(
                "invalid_record_source_sha256",
                severity="hard_failure",
                field="source_file_sha256",
                text=row_source_sha,
            )

        if isinstance(ingredients, list) and isinstance(instructions, list):
            text_parts = [
                title,
                *(item for item in ingredients if isinstance(item, str)),
                *(item for item in instructions if isinstance(item, str)),
            ]
            scope_reason = scope_exclusion_reason(text_parts, scope_markers) or ""
            if scope_reason:
                add_record_flag(
                    "known_non_food_or_pet_marker",
                    severity="hard_failure",
                    field="title_ingredients_instructions",
                    details=scope_reason,
                )

        if isinstance(ingredients, list):
            for line_index, line in enumerate(ingredients, start=1):
                if not isinstance(line, str) or not line.strip():
                    continue
                rejection_reason = ingredient_line_rejection_reason(line)
                if rejection_reason is not None:
                    retained_rejected_count += 1
                    if rejection_reason == "equipment_only":
                        equipment_count += 1
                    add_record_flag(
                        f"retained_{rejection_reason}",
                        severity="hard_failure",
                        field="ingredient_lines_original",
                        line_index=line_index,
                        text=line,
                        details=(
                            "a line rejected by the v3.1 curation contract remains "
                            "in the curated corpus"
                        ),
                    )

                _, correction_count = normalize_joined_mixed_fraction(line)
                if correction_count:
                    joined_fraction_count += 1
                    add_record_flag(
                        "joined_mixed_fraction_residue",
                        severity="hard_failure",
                        field="ingredient_lines_original",
                        line_index=line_index,
                        text=line,
                        details=f"found {correction_count} uncorrected mixed fraction(s)",
                    )

                if (
                    _SUBRECIPE_REFERENCE_MARKER.search(line)
                    or is_unexpanded_subrecipe_reference(
                        line, normalized_recipe_titles
                    )
                ):
                    subrecipe_reference_count += 1
                    add_record_flag(
                        "unexpanded_subrecipe_reference",
                        severity="hard_failure",
                        field="ingredient_lines_original",
                        line_index=line_index,
                        text=line,
                        details="ingredient line refers to another recipe",
                    )

                suspicious_amount = large_unitless_amount(line)
                if suspicious_amount is not None:
                    large_count += 1
                    add_record_flag(
                        "large_unitless_numeric_ingredient",
                        severity="warning",
                        field="ingredient_lines_original",
                        line_index=line_index,
                        text=line,
                        details=f"leading amount {suspicious_amount:g} has no adjacent source unit",
                    )
                if _MARKUP_RESIDUE.search(line) and rejection_reason is None:
                    add_record_flag(
                        "retained_source_markup_residue",
                        severity="hard_failure",
                        field="ingredient_lines_original",
                        line_index=line_index,
                        text=line,
                    )

        supplied_hash = str(row.get("record_hash") or "").strip()
        if supplied_hash:
            hash_payload = dict(row)
            hash_payload.pop("record_hash", None)
            if stable_hash(hash_payload) != supplied_hash:
                add_record_flag(
                    "record_hash_mismatch",
                    severity="hard_failure",
                    field="record_hash",
                )
        supplied_dedup_hash = str(row.get("dedup_hash") or "").strip()
        if (
            supplied_dedup_hash
            and title
            and _is_nonblank_sequence(ingredients)
            and _is_nonblank_sequence(instructions)
            and content_dedup_hash(title, ingredients, instructions) != supplied_dedup_hash
        ):
            add_record_flag(
                "dedup_hash_mismatch",
                severity="hard_failure",
                field="dedup_hash",
            )

        hard_counts.update(hard_codes)
        warning_counts.update(warning_codes)
        record_audit.append(
            {
                "row_index": row_index,
                "recipe_id": recipe_id,
                "title": title,
                "hard_pass": not hard_codes,
                "hard_failure_codes": " | ".join(sorted(hard_codes)),
                "warning_codes": " | ".join(sorted(warning_codes)),
                "ingredient_count": ingredient_count,
                "instruction_count": instruction_count,
                "measurement_coverage": stored_coverage,
                "recomputed_measurement_coverage": f"{recomputed_coverage:.12f}",
                "scope_exclusion_reason": scope_reason,
                "large_unitless_line_count": large_count,
                "equipment_line_count": equipment_count,
                "retained_rejected_line_count": retained_rejected_count,
                "joined_fraction_line_count": joined_fraction_count,
                "subrecipe_reference_line_count": subrecipe_reference_count,
            }
        )

    record_audit.sort(key=lambda item: (str(item["recipe_id"]), int(item["row_index"])))
    flagged.sort(
        key=lambda item: (
            0 if item["scope"] == "corpus" else 1,
            str(item["recipe_id"]),
            str(item["flag_code"]),
            str(item["line_index"]),
        )
    )
    hard_record_count = sum(not bool(item["hard_pass"]) for item in record_audit)
    warning_record_count = sum(bool(item["warning_codes"]) for item in record_audit)
    warning_occurrences = Counter(
        str(item["flag_code"])
        for item in flagged
        if item["severity"] == "warning"
    )
    passed = not global_failures and hard_record_count == 0
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "passed": passed,
        "status": (
            "failed"
            if not passed
            else "pass_with_review_warnings"
            if warning_record_count
            else "pass"
        ),
        "review_required": warning_record_count > 0,
        "corpus_file": corpus_path.name,
        "corpus_sha256": file_sha256(corpus_path),
        "expected_rows": expected_rows,
        "actual_rows": len(rows),
        "required_minimum_measurement_coverage": MINIMUM_MEASUREMENT_COVERAGE,
        "required_columns": sorted(REQUIRED_COLUMNS),
        "disallowed_split_quantity_columns": sorted(DISALLOWED_SPLIT_QUANTITY_COLUMNS),
        "disallowed_columns_present": disallowed_columns,
        "source_contract": contract,
        "global_hard_failures": global_failures,
        "records_with_hard_failures": hard_record_count,
        "records_with_review_warnings": warning_record_count,
        "hard_failure_counts": dict(sorted(hard_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "warning_occurrence_counts": dict(sorted(warning_occurrences.items())),
        "flagged_rows": len(flagged),
        "output_files": ["corpus_audit.json", "record_audit.csv", "flagged_records.csv"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(output_dir / "record_audit.csv", record_audit, RECORD_AUDIT_FIELDS)
    _write_csv_atomic(output_dir / "flagged_records.csv", flagged, FLAGGED_FIELDS)
    _write_json_atomic(output_dir / "corpus_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "curated" / "recipes_2000.parquet",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "data" / "curated" / "curation_summary.json",
    )
    parser.add_argument("--rules", type=Path, default=PROJECT_ROOT / "rules.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "curated" / "audits" / "v31_20261018",
    )
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    args = parser.parse_args()

    report = audit_corpus(
        corpus_path=args.corpus,
        summary_path=args.summary,
        rules_path=args.rules,
        output_dir=args.output_dir,
        expected_rows=args.expected_rows,
    )
    print(f"Corpus audit passed: {report['passed']}")
    print(f"Rows: {report['actual_rows']}/{report['expected_rows']}")
    print(f"Records with hard failures: {report['records_with_hard_failures']}")
    print(f"Records with review warnings: {report['records_with_review_warnings']}")
    print(f"Evidence directory: {args.output_dir}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
