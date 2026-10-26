"""Read authoritative recipe facts from the curated local Parquet corpus."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fridge_rescue.cooking_metadata import enrich_cooking_metadata
from fridge_rescue.schemas import RecipeRecord
from fridge_rescue.settings import PROJECT_ROOT

DEFAULT_CURATED_CORPUS_PATH = PROJECT_ROOT / "data" / "curated" / "recipes_2000.parquet"

_COLUMN_MAP = {
    "recipe_id": "recipe_id",
    "title": "title",
    "ingredient_lines_original": "ingredient_lines_original",
    "ingredients_normalized": "ingredients_normalized",
    "instructions_original": "instructions_original",
    "total_time_minutes_derived": "total_time_minutes",
    "rating": "rating",
    "review_count": "review_count",
    "equipment_tags_inferred": "equipment_tags_inferred",
    "vegetarian_status_derived": "vegetarian_status",
    "vegan_status_derived": "vegan_status",
    "source_record_id": "source_record_id",
}


def _none_if_not_finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _record_from_row(row: dict[str, Any]) -> RecipeRecord:
    """Map the curation schema to the smaller workflow-facing schema."""

    values = {
        model_field: _none_if_not_finite(row.get(parquet_field))
        for parquet_field, model_field in _COLUMN_MAP.items()
    }
    return enrich_cooking_metadata(RecipeRecord.model_validate(values))


class ParquetRecipeRepository:
    """In-memory view of the small curated corpus.

    The curated corpus contains 2,000 rows. Loading the workflow-facing columns
    once avoids repeated Parquet reads. ``get_many`` preserves requested order
    and silently ignores unknown IDs, matching the workflow repository
    protocol.
    """

    def __init__(self, path: str | Path = DEFAULT_CURATED_CORPUS_PATH) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Curated recipe corpus was not found: {self.path}")

        schema_names = set(pq.read_schema(self.path).names)
        missing_columns = sorted(set(_COLUMN_MAP) - schema_names)
        if missing_columns:
            raise ValueError(
                "Curated recipe corpus is missing required columns: " + ", ".join(missing_columns)
            )

        table = pq.read_table(self.path, columns=list(_COLUMN_MAP))
        records: dict[str, RecipeRecord] = {}
        for row_number, row in enumerate(table.to_pylist(), start=1):
            try:
                record = _record_from_row(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid curated recipe at row {row_number} in {self.path}: {exc}"
                ) from exc
            if record.recipe_id in records:
                raise ValueError(f"Duplicate recipe_id in curated corpus: {record.recipe_id}")
            records[record.recipe_id] = record

        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def get_many(self, recipe_ids: Sequence[str]) -> list[RecipeRecord]:
        """Return known records in exactly the requested order."""

        return [self._records[recipe_id] for recipe_id in recipe_ids if recipe_id in self._records]

    def list_all(self) -> list[RecipeRecord]:
        """Return a stable snapshot for deterministic local retrieval."""

        return list(self._records.values())
