"""Typed boundaries shared by the UI and workflow."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_EQUIPMENT = {
    "oven",
    "microwave",
    "blender_or_food_processor",
    "slow_cooker",
    "grill",
    "waffle_iron",
}


class BaselineGeneratedRecipe(BaseModel):
    """One model-generated recipe used only by the direct Baseline A."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    ingredients: list[str] = Field(min_length=1, max_length=60)
    instructions: list[str] = Field(min_length=1, max_length=40)
    total_time_minutes: int = Field(ge=1, le=1440)
    equipment: list[str] = Field(default_factory=list, max_length=20)
    reason: str = Field(min_length=1, max_length=600)

    @field_validator("title", "reason")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("text must not be blank")
        return cleaned

    @field_validator("ingredients", "instructions", "equipment")
    @classmethod
    def string_lists_must_not_contain_blanks(cls, value: list[str]) -> list[str]:
        cleaned = [" ".join(item.strip().split()) for item in value]
        if any(not item for item in cleaned):
            raise ValueError("list values must not be blank")
        return list(dict.fromkeys(cleaned))


class UserRequest(BaseModel):
    """Structured input accepted by the MVP."""

    model_config = ConfigDict(extra="forbid")

    available_ingredients: list[str] = Field(min_length=1)
    max_time_minutes: int | None = Field(default=None, ge=1, le=1440)
    excluded_ingredients: list[str] = Field(default_factory=list)
    unavailable_equipment: set[str] = Field(default_factory=set)
    diet_preference: Literal["none", "vegetarian", "vegan"] = "none"
    meal_type: str | None = None
    additional_notes: str | None = None

    @staticmethod
    def _clean_ingredients(value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            normalized = " ".join(item.lower().strip().split())
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    @field_validator("available_ingredients")
    @classmethod
    def clean_available_ingredients(cls, value: list[str]) -> list[str]:
        cleaned = cls._clean_ingredients(value)
        if not cleaned:
            raise ValueError("At least one non-blank ingredient is required")
        return cleaned

    @field_validator("excluded_ingredients")
    @classmethod
    def clean_excluded_ingredients(cls, value: list[str]) -> list[str]:
        return cls._clean_ingredients(value)

    @field_validator("unavailable_equipment")
    @classmethod
    def equipment_must_be_supported(cls, value: set[str]) -> set[str]:
        unknown = value - SUPPORTED_EQUIPMENT
        if unknown:
            raise ValueError(f"Unsupported equipment values: {sorted(unknown)}")
        return value


class RecipeRecord(BaseModel):
    """Normalized recipe facts hydrated from the curated corpus."""

    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    title: str
    ingredients_original: list[str]
    ingredient_quantities_original: list[str | None] = Field(default_factory=list)
    quantity_alignment_status: Literal["aligned", "unreliable"] = "unreliable"
    ingredients_normalized: list[str]
    instructions_original: list[str]
    total_time_minutes: int | None = Field(default=None, ge=0)
    rating: float | None = Field(default=None, ge=0, le=5)
    review_count: int | None = Field(default=None, ge=0)
    equipment_tags_inferred: set[str] = Field(default_factory=set)
    vegetarian_status: Literal["match", "conflict", "unknown"] = "unknown"
    vegan_status: Literal["match", "conflict", "unknown"] = "unknown"
    source_record_id: str


class SelectedRecipe(BaseModel):
    """The only recipe-specific information an LLM is allowed to choose."""

    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    reason: str = Field(min_length=1, max_length=600)
    substitution_notes: list[str] = Field(default_factory=list)


class LLMRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selections: list[SelectedRecipe] = Field(default_factory=list, max_length=3)
    no_result_reason: str | None = None

    @model_validator(mode="after")
    def result_must_be_consistent(self) -> LLMRecommendation:
        has_selections = bool(self.selections)
        has_no_result = bool(self.no_result_reason and self.no_result_reason.strip())
        if has_selections == has_no_result:
            raise ValueError("Provide selections or a no-result reason, but not both")
        recipe_ids = [selection.recipe_id for selection in self.selections]
        if len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError("Selected recipe IDs must be unique")
        return self


class WorkflowState(TypedDict, total=False):
    run_id: str
    request: UserRequest
    retrieved_candidate_ids: list[str]
    candidate_records: list[RecipeRecord]
    draft: LLMRecommendation
    validation_errors: list[str]
    validation_passed: bool
    retry_count: int
    trace: list[str]
