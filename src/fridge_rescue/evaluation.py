"""Reproducible, offline evaluation for the baseline and agentic pipelines.

The evaluation layer is decoupled from OpenAI and LangGraph. Baseline and
agentic adapters implement ``run(case)`` (or a callable equivalent), allowing
both systems to share the same scoring code during offline evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fridge_rescue.retrieval import ingredient_match_details
from fridge_rescue.rule_checker import (
    canonicalize_ingredient,
    contains_phrase,
    find_excluded_matches,
    ingredient_terms_are_non_equivalent,
    ingredient_terms_overlap,
    normalize_token,
)
from fridge_rescue.schemas import BaselineGeneratedRecipe, RecipeRecord, UserRequest

DEFAULT_DIET_CONFLICT_MARKERS: dict[str, list[str]] = {
    "vegetarian": [
        "beef",
        "chicken",
        "pork",
        "bacon",
        "ham",
        "turkey",
        "lamb",
        "fish",
        "salmon",
        "tuna",
        "shrimp",
        "prawn",
        "anchovy",
        "gelatin",
    ],
    "vegan": [
        "beef",
        "chicken",
        "pork",
        "bacon",
        "ham",
        "turkey",
        "lamb",
        "fish",
        "salmon",
        "tuna",
        "shrimp",
        "prawn",
        "anchovy",
        "gelatin",
        "egg",
        "milk",
        "butter",
        "cheese",
        "cream",
        "yogurt",
        "honey",
    ],
}

DEFAULT_DIET_MARKER_EXEMPTIONS: dict[str, tuple[str, ...]] = {
    "milk": (
        "almond milk",
        "cashew milk",
        "coconut milk",
        "dairy-free milk",
        "non dairy milk",
        "non-dairy milk",
        "nondairy milk",
        "oat milk",
        "milk alternative",
        "milk substitute",
        "plant based milk",
        "plant-based milk",
        "plant milk",
        "rice milk",
        "soy milk",
        "vegan milk",
    ),
    "butter": (
        "almond butter",
        "apple butter",
        "cashew butter",
        "cocoa butter",
        "coconut butter",
        "dairy-free butter",
        "non dairy butter",
        "non-dairy butter",
        "peanut butter",
        "butter alternative",
        "butter substitute",
        "plant based butter",
        "plant-based butter",
        "pumpkin butter",
        "seed butter",
        "vegan butter",
    ),
    "cream": (
        "cashew cream",
        "coconut cream",
        "cream of tartar",
        "dairy-free cream",
        "non dairy cream",
        "non-dairy cream",
        "nondairy cream",
        "plant based cream",
        "plant-based cream",
        "vegan cream",
    ),
    "cheese": (
        "dairy-free cheese",
        "non dairy cheese",
        "non-dairy cheese",
        "nondairy cheese",
        "plant based cheese",
        "plant-based cheese",
        "vegan cheese",
    ),
    "yogurt": (
        "almond yogurt",
        "coconut yogurt",
        "non dairy yogurt",
        "non-dairy yogurt",
        "plant based yogurt",
        "plant-based yogurt",
        "soy yogurt",
        "vegan yogurt",
    ),
    "egg": ("chia egg", "egg substitute", "flax egg", "vegan egg"),
    "honey": ("honey alternative", "vegan honey"),
}

DEFAULT_EQUIPMENT_ALIASES: dict[str, list[str]] = {
    "oven": ["oven"],
    "microwave": ["microwave"],
    "blender_or_food_processor": ["blender", "food processor"],
    "slow_cooker": ["slow cooker", "crock pot", "crockpot"],
    "grill": ["grill"],
    "waffle_iron": ["waffle iron"],
}


def estimate_token_cost_usd(
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    input_cost_per_million_usd: float | None,
    output_cost_per_million_usd: float | None,
) -> float | None:
    """Estimate one run's token cost from explicitly supplied price rates.

    Prices are intentionally not hard-coded because provider rates can change.
    A cost is returned only when both token counts and both rates are known.
    """

    values = (
        input_tokens,
        output_tokens,
        input_cost_per_million_usd,
        output_cost_per_million_usd,
    )
    if any(value is None for value in values):
        return None
    if input_tokens < 0 or output_tokens < 0:  # type: ignore[operator]
        raise ValueError("Token counts must be non-negative")
    if input_cost_per_million_usd < 0 or output_cost_per_million_usd < 0:
        raise ValueError("Token prices must be non-negative")
    return (
        input_tokens * input_cost_per_million_usd
        + output_tokens * output_cost_per_million_usd
    ) / 1_000_000


class EvaluationCase(BaseModel):
    """One frozen request and optional human-reviewed no-result labels.

    ``expected_no_result`` means that no practical answer should be produced
    by *any* arm.  ``expected_corpus_no_result`` is separate because an LLM may
    generate an answer even when the local corpus has no suitable record.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=80)
    split: Literal["dev", "final"]
    request: UserRequest
    expected_no_result: bool | None = None
    expected_corpus_no_result: bool | None = None
    label_status: Literal["placeholder", "reviewed"] = "placeholder"
    notes: str | None = None

    @model_validator(mode="after")
    def reviewed_case_needs_a_label(self) -> EvaluationCase:
        if self.label_status == "reviewed" and self.expected_no_result is None:
            raise ValueError("A reviewed case must define expected_no_result")
        return self


class RunnerPrediction(BaseModel):
    """Common output contract for corpus-backed and directly generated arms.

    ``recipe_ids`` is for Baseline B and the agentic system.  The required
    direct single-prompt Baseline A instead supplies ``generated_recipes``.
    The two fields are mutually exclusive to distinguish generated content
    from corpus-backed results.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "no_result", "error"]
    recipe_ids: list[str] = Field(default_factory=list, max_length=3)
    generated_recipes: list[BaselineGeneratedRecipe] = Field(default_factory=list, max_length=3)
    no_result_reason: str | None = None
    error_message: str | None = None
    provenance_kind: Literal["corpus", "generated"] | None = None
    model_call_count: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    application_output: dict[str, object] | None = None
    workflow_trace: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    retry_count: int | None = Field(default=None, ge=0)
    retrieved_candidate_ids: list[str] | None = None
    failure_stage: Literal[
        "input_validation",
        "retrieval",
        "provider_or_output",
        "output_validation",
    ] | None = None

    @model_validator(mode="after")
    def output_shape_matches_status(self) -> RunnerPrediction:
        self.recipe_ids = [str(value).strip() for value in self.recipe_ids]
        if any(not value for value in self.recipe_ids):
            raise ValueError("recipe_ids cannot contain blank values")
        if len(self.recipe_ids) != len(set(self.recipe_ids)):
            raise ValueError("recipe_ids must be unique")

        reason = self.no_result_reason.strip() if self.no_result_reason else None
        error = self.error_message.strip() if self.error_message else None
        self.no_result_reason = reason
        self.error_message = error

        has_ids = bool(self.recipe_ids)
        has_generated = bool(self.generated_recipes)
        if has_ids and has_generated:
            raise ValueError("Use recipe_ids or generated_recipes, never both")
        if has_ids and self.provenance_kind == "generated":
            raise ValueError("recipe_ids cannot claim generated provenance")
        if has_generated and self.provenance_kind == "corpus":
            raise ValueError("generated_recipes cannot claim corpus provenance")

        if self.status == "success":
            if has_ids == has_generated:
                raise ValueError(
                    "A successful prediction must contain recipe IDs or generated recipes"
                )
            if reason or error:
                raise ValueError("A successful prediction cannot contain no-result or error text")
        elif self.status == "no_result":
            if has_ids or has_generated or not reason or error:
                raise ValueError("A no-result prediction needs a reason and no recipe output")
        elif has_ids or has_generated or reason or not error:
            raise ValueError("An error prediction needs only an error_message")
        return self


RunnerOutput = RunnerPrediction | Mapping[str, object]


class EvaluationRunner(Protocol):
    """Object-style injection boundary for either experiment arm."""

    def run(self, case: EvaluationCase) -> RunnerOutput: ...


RunnerCallable = Callable[[EvaluationCase], RunnerOutput]
RunnerLike = EvaluationRunner | RunnerCallable


class DirectBaselineAnswer(Protocol):
    recipes: list[BaselineGeneratedRecipe]
    no_result_reason: str | None


class DirectBaselineLike(Protocol):
    """Minimal boundary implemented by :class:`ResponsesSinglePromptBaseline`."""

    def generate(self, request: UserRequest) -> DirectBaselineAnswer: ...


class AgenticRecommendationLike(Protocol):
    recipe: RecipeRecord


class AgenticOutcomeLike(Protocol):
    status: str
    recommendations: Sequence[AgenticRecommendationLike]
    no_result_reason: str | None
    validation_errors: Sequence[str]
    trace: Sequence[str]
    input_tokens: int | None
    output_tokens: int | None


class AgenticWorkflowLike(Protocol):
    """Minimal boundary implemented by :class:`FridgeRescueWorkflow`."""

    def run(self, request: UserRequest) -> AgenticOutcomeLike: ...


class DirectBaselineEvaluationRunner:
    """Adapt the one-call Baseline A output to :class:`RunnerPrediction`."""

    def __init__(
        self,
        baseline: DirectBaselineLike,
        *,
        input_cost_per_million_usd: float | None = None,
        output_cost_per_million_usd: float | None = None,
    ) -> None:
        self.baseline = baseline
        self.input_cost_per_million_usd = input_cost_per_million_usd
        self.output_cost_per_million_usd = output_cost_per_million_usd

    def _usage(self) -> tuple[int | None, int | None]:
        usage = getattr(self.baseline, "last_usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if (
            not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or input_tokens < 0
        ):
            input_tokens = None
        if (
            not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 0
        ):
            output_tokens = None
        return input_tokens, output_tokens

    def _estimated_cost(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        return estimate_token_cost_usd(
            input_tokens,
            output_tokens,
            input_cost_per_million_usd=self.input_cost_per_million_usd,
            output_cost_per_million_usd=self.output_cost_per_million_usd,
        )

    def run(self, case: EvaluationCase) -> RunnerPrediction:
        try:
            answer = self.baseline.generate(case.request)
            recipes = [BaselineGeneratedRecipe.model_validate(item) for item in answer.recipes]
            no_result_reason = answer.no_result_reason
        except Exception as exc:  # noqa: BLE001 - preserve provider failures as rows
            message = str(exc).strip() or type(exc).__name__
            input_tokens, output_tokens = self._usage()
            return RunnerPrediction(
                status="error",
                error_message=message,
                provenance_kind="generated",
                model_call_count=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=self._estimated_cost(input_tokens, output_tokens),
                workflow_trace=["single_prompt"],
                failure_stage="provider_or_output",
            )
        input_tokens, output_tokens = self._usage()
        application_output = (
            answer.model_dump(mode="json")
            if isinstance(answer, BaseModel)
            else {
                "recipes": [recipe.model_dump(mode="json") for recipe in recipes],
                "no_result_reason": no_result_reason,
            }
        )
        if recipes:
            return RunnerPrediction(
                status="success",
                generated_recipes=recipes,
                provenance_kind="generated",
                model_call_count=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=self._estimated_cost(input_tokens, output_tokens),
                application_output=application_output,
                workflow_trace=["single_prompt"],
            )
        return RunnerPrediction(
            status="no_result",
            no_result_reason=no_result_reason,
            provenance_kind="generated",
            model_call_count=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=self._estimated_cost(input_tokens, output_tokens),
            application_output=application_output,
            workflow_trace=["single_prompt"],
        )


class AgenticEvaluationRunner:
    """Adapt the LangGraph workflow result without making any API call itself."""

    def __init__(
        self,
        workflow: AgenticWorkflowLike,
        *,
        retrieval_diagnostics: object | None = None,
        input_cost_per_million_usd: float | None = None,
        output_cost_per_million_usd: float | None = None,
    ) -> None:
        self.workflow = workflow
        self.retrieval_diagnostics = retrieval_diagnostics
        self.input_cost_per_million_usd = input_cost_per_million_usd
        self.output_cost_per_million_usd = output_cost_per_million_usd

    def _retrieved_ids(self) -> list[str] | None:
        if self.retrieval_diagnostics is None:
            return None
        value = getattr(self.retrieval_diagnostics, "last_retrieved_ids", None)
        if value is None:
            return None
        return [str(item) for item in value]

    def _estimated_cost(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        return estimate_token_cost_usd(
            input_tokens,
            output_tokens,
            input_cost_per_million_usd=self.input_cost_per_million_usd,
            output_cost_per_million_usd=self.output_cost_per_million_usd,
        )

    def run(self, case: EvaluationCase) -> RunnerPrediction:
        reset_diagnostics = getattr(self.retrieval_diagnostics, "reset", None)
        if callable(reset_diagnostics):
            reset_diagnostics()
        try:
            outcome = self.workflow.run(case.request)
        except Exception as exc:  # noqa: BLE001 - preserve workflow failures as rows
            message = str(exc).strip() or type(exc).__name__
            return RunnerPrediction(
                status="error",
                error_message=message,
                provenance_kind="corpus",
                retrieved_candidate_ids=self._retrieved_ids(),
                failure_stage="provider_or_output",
            )

        model_calls = sum(node in {"generate_answer", "repair_once"} for node in outcome.trace)
        input_tokens = getattr(outcome, "input_tokens", None)
        output_tokens = getattr(outcome, "output_tokens", None)
        application_output = (
            outcome.model_dump(mode="json")
            if isinstance(outcome, BaseModel)
            else {
                "status": outcome.status,
                "recommendations": [
                    item.recipe.model_dump(mode="json")
                    if isinstance(item.recipe, BaseModel)
                    else {"recipe_id": item.recipe.recipe_id}
                    for item in outcome.recommendations
                ],
                "no_result_reason": outcome.no_result_reason,
                "validation_errors": list(outcome.validation_errors),
                "trace": list(outcome.trace),
            }
        )
        common_diagnostics = {
            "estimated_cost_usd": self._estimated_cost(input_tokens, output_tokens),
            "application_output": application_output,
            "workflow_trace": list(outcome.trace),
            "validation_errors": list(outcome.validation_errors),
            "retry_count": getattr(outcome, "retry_count", None),
            "retrieved_candidate_ids": self._retrieved_ids(),
        }
        if outcome.status == "success":
            return RunnerPrediction(
                status="success",
                recipe_ids=[item.recipe.recipe_id for item in outcome.recommendations],
                provenance_kind="corpus",
                model_call_count=model_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                **common_diagnostics,
            )
        if outcome.status in {"no_result", "no_candidates"}:
            return RunnerPrediction(
                status="no_result",
                no_result_reason=outcome.no_result_reason
                or "No feasible corpus recipe was returned.",
                provenance_kind="corpus",
                model_call_count=model_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                **common_diagnostics,
            )
        detail = "; ".join(outcome.validation_errors) or outcome.status
        failure_stage = {
            "invalid_request": "input_validation",
            "failed_validation": "output_validation",
            "service_error": "provider_or_output",
        }.get(outcome.status, "provider_or_output")
        return RunnerPrediction(
            status="error",
            error_message=detail,
            provenance_kind="corpus",
            model_call_count=model_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            failure_stage=failure_stage,
            **common_diagnostics,
        )


class RecipeCatalog(Protocol):
    """Minimal authoritative recipe lookup needed by the scorers."""

    def get_many(self, recipe_ids: Sequence[str]) -> list[RecipeRecord]: ...


CatalogLike = RecipeCatalog | Mapping[str, RecipeRecord]


class CaseEvaluation(BaseModel):
    """Machine-readable measurements for one runner on one case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    split: Literal["dev", "final"]
    request: UserRequest
    expected_no_result: bool | None = None
    expected_corpus_no_result: bool | None = None
    label_status: Literal["placeholder", "reviewed"]
    runner_name: str
    prediction_status: Literal["success", "no_result", "error"]
    output_kind: Literal["corpus", "generated", "none"]
    output_item_count: int = Field(ge=0)
    generated_recipes: list[BaselineGeneratedRecipe] = Field(default_factory=list)
    selected_recipe_ids: list[str] = Field(default_factory=list)
    valid_recipe_ids: list[str] = Field(default_factory=list)
    invalid_recipe_ids: list[str] = Field(default_factory=list)
    hard_constraint_passed_ids: list[str] = Field(default_factory=list)
    hard_constraint_failed_ids: list[str] = Field(default_factory=list)
    hard_constraint_violations: dict[str, list[str]] = Field(default_factory=dict)
    valid_recipe_id_rate: float | None = Field(default=None, ge=0, le=1)
    hard_constraint_pass_rate: float | None = Field(default=None, ge=0, le=1)
    ingredient_coverage_rate: float | None = Field(default=None, ge=0, le=1)
    common_hard_constraint_violations: dict[str, list[str]] = Field(default_factory=dict)
    common_hard_constraint_pass_rate: float | None = Field(default=None, ge=0, le=1)
    common_ingredient_coverage_rate: float | None = Field(default=None, ge=0, le=1)
    common_input_utilization_rate: float | None = Field(default=None, ge=0, le=1)
    best_common_input_utilization_rate: float | None = Field(default=None, ge=0, le=1)
    all_requested_ingredients_used: bool = False
    all_requested_ingredients_metric_applicable: bool = True
    output_completeness_rate: float | None = Field(default=None, ge=0, le=1)
    common_success: bool | None = None
    provenance_applicable: bool | None = None
    latency_ms: float = Field(ge=0)
    model_call_count: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    application_output: dict[str, object] | None = None
    workflow_trace: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    retry_count: int | None = Field(default=None, ge=0)
    retrieved_candidate_ids: list[str] | None = None
    retrieved_candidate_metrics: dict[str, dict[str, object]] = Field(default_factory=dict)
    valid_retrieved_candidate_ids: list[str] = Field(default_factory=list)
    invalid_retrieved_candidate_ids: list[str] = Field(default_factory=list)
    retrieved_candidate_count: int | None = Field(default=None, ge=0)
    best_retrieved_input_utilization_rate: float | None = Field(default=None, ge=0, le=1)
    selection_input_utilization_regret: float | None = Field(default=None, ge=0, le=1)
    failure_stage: Literal[
        "input_validation",
        "retrieval",
        "provider_or_output",
        "output_validation",
    ] | None = None
    no_result_reasonable: bool | None = None
    corpus_no_result_reasonable: bool | None = None
    no_result_reason: str | None = None
    error_message: str | None = None


class RunnerSummary(BaseModel):
    """Aggregate metrics for one experiment arm."""

    model_config = ConfigDict(extra="forbid")

    runner_name: str
    case_count: int = Field(ge=0)
    error_case_count: int = Field(ge=0)
    total_output_items: int = Field(ge=0)
    total_selected_recipe_ids: int = Field(ge=0)
    valid_recipe_id_rate: float | None = Field(default=None, ge=0, le=1)
    hard_constraint_pass_rate: float | None = Field(default=None, ge=0, le=1)
    mean_ingredient_coverage_rate: float | None = Field(default=None, ge=0, le=1)
    common_hard_constraint_pass_rate: float | None = Field(default=None, ge=0, le=1)
    mean_common_ingredient_coverage_rate: float | None = Field(default=None, ge=0, le=1)
    mean_common_input_utilization_rate: float | None = Field(default=None, ge=0, le=1)
    mean_best_input_utilization_rate: float | None = Field(default=None, ge=0, le=1)
    all_requested_ingredients_case_rate: float | None = Field(default=None, ge=0, le=1)
    applicable_all_requested_ingredients_case_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    all_requested_ingredients_applicable_case_count: int = Field(ge=0)
    output_completeness_rate: float | None = Field(default=None, ge=0, le=1)
    common_success_rate: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float | None = Field(default=None, ge=0)
    median_latency_ms: float | None = Field(default=None, ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    no_result_reasonableness_rate: float | None = Field(default=None, ge=0, le=1)
    no_result_labeled_case_count: int = Field(ge=0)
    corpus_no_result_accuracy: float | None = Field(default=None, ge=0, le=1)
    corpus_no_result_labeled_case_count: int = Field(ge=0)
    mean_model_calls: float | None = Field(default=None, ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    total_estimated_cost_usd: float | None = Field(default=None, ge=0)
    retrieval_diagnostic_case_count: int = Field(ge=0)
    mean_retrieved_candidate_count: float | None = Field(default=None, ge=0)
    mean_best_retrieved_input_utilization_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    mean_selection_input_utilization_regret: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class ExperimentReport(BaseModel):
    """Complete comparison artifact before serialization."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1"] = "1.1"
    case_set_sha256: str
    case_count: int = Field(ge=1)
    results: list[CaseEvaluation]
    summaries: list[RunnerSummary]
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


def _invoke_runner(runner: RunnerLike, case: EvaluationCase) -> RunnerOutput:
    run_method = getattr(runner, "run", None)
    if callable(run_method):
        return run_method(case)
    if callable(runner):
        return runner(case)
    raise TypeError("runner must be callable or provide run(case)")


def _lookup_records(catalog: CatalogLike, recipe_ids: Sequence[str]) -> dict[str, RecipeRecord]:
    unique_ids = list(dict.fromkeys(recipe_ids))
    if isinstance(catalog, Mapping):
        values = [catalog[recipe_id] for recipe_id in unique_ids if recipe_id in catalog]
    else:
        values = catalog.get_many(unique_ids)

    records: dict[str, RecipeRecord] = {}
    for value in values:
        record = RecipeRecord.model_validate(value)
        if record.recipe_id in unique_ids:
            records[record.recipe_id] = record
    return records


def hard_constraint_violations(
    request: UserRequest,
    recipe: RecipeRecord,
    *,
    allergen_aliases: Mapping[str, list[str]] | None = None,
    ingredient_aliases: Mapping[str, str] | None = None,
) -> list[str]:
    """Apply the same observable hard constraints to either experiment arm."""

    violations: list[str] = []
    if request.max_time_minutes is not None:
        if recipe.total_time_minutes is None:
            violations.append("total_time_unknown")
        elif recipe.total_time_minutes > request.max_time_minutes:
            violations.append("exceeds_time_limit")

    unavailable = recipe.equipment_tags_inferred & request.unavailable_equipment
    if unavailable:
        violations.append("requires_unavailable_equipment:" + "|".join(sorted(unavailable)))

    exclusions = find_excluded_matches(
        recipe.ingredients_normalized,
        request.excluded_ingredients,
        dict(allergen_aliases or {}),
        ingredient_aliases,
    )
    if exclusions:
        violations.append("contains_excluded_ingredient:" + "|".join(sorted(exclusions)))

    if request.diet_preference == "vegetarian" and recipe.vegetarian_status != "match":
        violations.append("vegetarian_not_confirmed")
    if request.diet_preference == "vegan" and recipe.vegan_status != "match":
        violations.append("vegan_not_confirmed")
    return violations


def ingredient_coverage(
    request: UserRequest,
    recipe: RecipeRecord,
    *,
    pantry_staples: Sequence[str] = (),
    ingredient_aliases: Mapping[str, str] | None = None,
) -> float | None:
    """Return lexical coverage of unique required ingredients by the user's inventory."""

    required = list(
        dict.fromkeys(
            canonicalize_ingredient(value, ingredient_aliases)
            for value in recipe.ingredients_normalized
            if value.strip()
        )
    )
    if not required:
        return None
    available = list(
        dict.fromkeys(
            canonicalize_ingredient(value, ingredient_aliases)
            for value in [*request.available_ingredients, *pantry_staples]
            if value.strip()
        )
    )
    covered = sum(
        any(
            ingredient_terms_overlap(required_item, available_item)
            for available_item in available
        )
        for required_item in required
    )
    return covered / len(required)


def _simple_singular(value: str) -> str:
    """Conservatively singularize one ingredient token for lexical screening."""

    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("oes") and len(value) > 4:
        return value[:-2]
    if value.endswith("s") and not value.endswith(("ss", "us")) and len(value) > 3:
        return value[:-1]
    return value


def _ingredient_phrase_present(text: str, phrase: str) -> bool:
    if ingredient_terms_are_non_equivalent(text, phrase):
        return False
    if ingredient_terms_overlap(text, phrase):
        return True
    text_tokens = {_simple_singular(token) for token in re.findall(r"[a-z]+", text.lower())}
    phrase_tokens = [_simple_singular(token) for token in re.findall(r"[a-z]+", phrase.lower())]
    return bool(phrase_tokens) and all(token in text_tokens for token in phrase_tokens)


def generated_ingredient_coverage(
    request: UserRequest,
    recipe: BaselineGeneratedRecipe,
    *,
    pantry_staples: Sequence[str] = (),
    ingredient_aliases: Mapping[str, str] | None = None,
) -> float | None:
    """Coverage for free-text Baseline A ingredient lines.

    This is only a lexical estimate.  Quantity text is retained, while simple
    singular/plural variants are accepted to avoid treating ``tomatoes`` and
    ``tomato`` as different foods.
    """

    required = list(
        dict.fromkeys(
            canonicalize_ingredient(item, ingredient_aliases)
            for item in recipe.ingredients
        )
    )
    if not required:
        return None
    available = list(
        dict.fromkeys(
            canonicalize_ingredient(value, ingredient_aliases)
            for value in [*request.available_ingredients, *pantry_staples]
            if value.strip()
        )
    )
    covered = sum(
        any(
            _ingredient_phrase_present(required_item, available_item)
            for available_item in available
        )
        for required_item in required
    )
    return covered / len(required)


def ingredient_input_utilization(
    request: UserRequest,
    recipe: RecipeRecord,
    *,
    ingredient_aliases: Mapping[str, str] | None = None,
) -> float:
    """Return the share of user-provided ingredients used by a corpus recipe.

    This differs from :func:`ingredient_coverage`, whose denominator includes
    every recipe requirement. For input such as ``egg, tomato``, this metric
    measures whether both supplied ingredients are used rather than favoring
    a short recipe that uses only one.
    """

    requested = list(
        dict.fromkeys(
            canonicalize_ingredient(value, ingredient_aliases)
            for value in request.available_ingredients
        )
    )
    recipe_ingredients = [
        canonicalize_ingredient(value, ingredient_aliases)
        for value in recipe.ingredients_normalized
        if value.strip()
    ]
    matched = sum(
        any(
            _ingredient_phrase_present(recipe_item, requested_item)
            for recipe_item in recipe_ingredients
        )
        for requested_item in requested
    )
    return matched / len(requested)


def generated_input_utilization(
    request: UserRequest,
    recipe: BaselineGeneratedRecipe,
    *,
    ingredient_aliases: Mapping[str, str] | None = None,
) -> float:
    """Return the share of user-provided ingredients used by a generated recipe."""

    requested = list(
        dict.fromkeys(
            canonicalize_ingredient(value, ingredient_aliases)
            for value in request.available_ingredients
        )
    )
    generated_ingredients = [
        canonicalize_ingredient(value, ingredient_aliases)
        for value in recipe.ingredients
    ]
    matched = sum(
        any(
            _ingredient_phrase_present(generated_item, requested_item)
            for generated_item in generated_ingredients
        )
        for requested_item in requested
    )
    return matched / len(requested)


def generated_hard_constraint_violations(
    request: UserRequest,
    recipe: BaselineGeneratedRecipe,
    *,
    allergen_aliases: Mapping[str, list[str]] | None = None,
    ingredient_aliases: Mapping[str, str] | None = None,
    diet_conflict_markers: Mapping[str, list[str]] | None = None,
    equipment_aliases: Mapping[str, list[str]] | None = None,
) -> list[str]:
    """Text-screen declared Baseline A facts for hard-constraint risks.

    These checks constitute a declared-text risk screen and do not convert
    generated content into authoritative corpus data.
    """

    violations: list[str] = []
    if (
        request.max_time_minutes is not None
        and recipe.total_time_minutes > request.max_time_minutes
    ):
        violations.append("exceeds_time_limit")

    normalized_equipment = [normalize_token(item) for item in recipe.equipment]
    aliases_by_equipment = (
        equipment_aliases if equipment_aliases is not None else DEFAULT_EQUIPMENT_ALIASES
    )
    for unavailable in sorted(request.unavailable_equipment):
        aliases = aliases_by_equipment.get(unavailable, [unavailable])
        if any(contains_phrase(item, alias) for item in normalized_equipment for alias in aliases):
            violations.append(f"requires_unavailable_equipment:{unavailable}")

    screened_ingredients = [
        re.sub(
            r"\((?:check|verify|ensure)\b[^)]*(?:\bno\b|free|without)[^)]*\)",
            " ",
            ingredient,
            flags=re.IGNORECASE,
        )
        for ingredient in recipe.ingredients
    ]
    for excluded in request.excluded_ingredients:
        term = re.escape(normalize_token(excluded))
        safe_contexts = (
            rf"(?<!\w){term}\s*-\s*free(?!\w)",
            rf"(?<!\w){term}\s+free(?!\w)",
            rf"(?<!\w)(?:free\s+(?:of|from)|without|no)\s+{term}(?!\w)",
            rf"(?<!\w)(?:avoid|omit)\s+(?:the\s+)?{term}(?!\w)",
            rf"(?<!\w)instead\s+of\s+{term}(?!\w)",
            rf"(?<!\w)check\s+(?:the\s+)?labels?\s+for\s+{term}(?!\w)",
        )
        screened_ingredients = [
            re.sub("|".join(safe_contexts), " ", ingredient, flags=re.IGNORECASE)
            for ingredient in screened_ingredients
        ]

    exclusions = find_excluded_matches(
        screened_ingredients,
        request.excluded_ingredients,
        dict(allergen_aliases or {}),
        ingredient_aliases,
    )
    if exclusions:
        violations.append("contains_excluded_ingredient:" + "|".join(sorted(exclusions)))

    markers_by_diet = (
        diet_conflict_markers
        if diet_conflict_markers is not None
        else DEFAULT_DIET_CONFLICT_MARKERS
    )
    if request.diet_preference in {"vegetarian", "vegan"}:
        conflicts: list[str] = []
        normalized_ingredients = [normalize_token(item) for item in recipe.ingredients]
        for marker in markers_by_diet.get(request.diet_preference, []):
            marker_conflict = False
            for ingredient in normalized_ingredients:
                if not contains_phrase(ingredient, marker):
                    continue
                screened = ingredient
                for exemption in DEFAULT_DIET_MARKER_EXEMPTIONS.get(marker, ()):
                    screened = re.sub(
                        rf"(?<!\w){re.escape(exemption)}(?!\w)",
                        " ",
                        screened,
                    )
                if contains_phrase(screened, marker):
                    marker_conflict = True
                    break
            if marker_conflict:
                conflicts.append(marker)
        conflicts.sort()
        if conflicts:
            violations.append(f"{request.diet_preference}_text_conflict:" + "|".join(conflicts))
    return violations


def _corpus_recipe_complete(recipe: RecipeRecord) -> bool:
    return bool(
        recipe.title.strip()
        and recipe.ingredient_lines_original
        and recipe.instructions_original
    )


def _generated_recipe_complete(recipe: BaselineGeneratedRecipe) -> bool:
    return bool(
        recipe.title.strip()
        and recipe.ingredients
        and recipe.instructions
        and recipe.total_time_minutes is not None
    )


def evaluate_runner(
    cases: Sequence[EvaluationCase],
    *,
    runner_name: str,
    runner: RunnerLike,
    catalog: CatalogLike,
    allergen_aliases: Mapping[str, list[str]] | None = None,
    ingredient_aliases: Mapping[str, str] | None = None,
    diet_conflict_markers: Mapping[str, list[str]] | None = None,
    equipment_aliases: Mapping[str, list[str]] | None = None,
    pantry_staples: Sequence[str] = (),
    clock: Callable[[], float] = perf_counter,
) -> list[CaseEvaluation]:
    """Run and score one experiment arm without aborting on a single bad output."""

    results: list[CaseEvaluation] = []
    for case_value in cases:
        case = EvaluationCase.model_validate(case_value)
        runner_case = case.model_copy(deep=True)
        started = clock()
        try:
            raw_prediction = _invoke_runner(runner, runner_case)
            prediction = RunnerPrediction.model_validate(raw_prediction)
        # Network or provider failures at the runner boundary remain visible
        # as error rows instead of being silently dropped.
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc).strip() or type(exc).__name__
            prediction = RunnerPrediction(status="error", error_message=error_text)
        latency_ms = max(0.0, (clock() - started) * 1000)

        selected_ids = prediction.recipe_ids
        generated_recipes = prediction.generated_recipes
        records = _lookup_records(catalog, selected_ids)
        valid_ids = [recipe_id for recipe_id in selected_ids if recipe_id in records]
        invalid_ids = [recipe_id for recipe_id in selected_ids if recipe_id not in records]

        violations_by_id: dict[str, list[str]] = {
            recipe_id: hard_constraint_violations(
                case.request,
                records[recipe_id],
                allergen_aliases=allergen_aliases,
                ingredient_aliases=ingredient_aliases,
            )
            for recipe_id in valid_ids
        }
        passed_ids = [recipe_id for recipe_id in valid_ids if not violations_by_id[recipe_id]]
        failed_ids = [
            recipe_id
            for recipe_id in selected_ids
            if recipe_id in invalid_ids or violations_by_id.get(recipe_id)
        ]
        for recipe_id in invalid_ids:
            violations_by_id[recipe_id] = ["invalid_recipe_id"]

        selected_count = len(selected_ids)
        valid_rate = len(valid_ids) / selected_count if selected_count else None
        hard_pass_rate = len(passed_ids) / selected_count if selected_count else None
        coverage_values = [
            score
            for recipe_id in valid_ids
            if (
                score := ingredient_coverage(
                    case.request,
                    records[recipe_id],
                    pantry_staples=pantry_staples,
                    ingredient_aliases=ingredient_aliases,
                )
            )
            is not None
        ]
        coverage_rate = fmean(coverage_values) if coverage_values else None
        corpus_input_utilizations = [
            ingredient_input_utilization(
                case.request,
                records[recipe_id],
                ingredient_aliases=ingredient_aliases,
            )
            for recipe_id in valid_ids
        ]

        retrieved_ids = prediction.retrieved_candidate_ids
        retrieved_records = _lookup_records(catalog, retrieved_ids or [])
        valid_retrieved_ids = [
            recipe_id for recipe_id in (retrieved_ids or []) if recipe_id in retrieved_records
        ]
        invalid_retrieved_ids = [
            recipe_id for recipe_id in (retrieved_ids or []) if recipe_id not in retrieved_records
        ]
        retrieved_details = {
            recipe_id: ingredient_match_details(
                retrieved_records[recipe_id],
                case.request.available_ingredients,
                pantry_staples,
                ingredient_aliases,
            )
            for recipe_id in valid_retrieved_ids
        }
        retrieved_candidate_metrics: dict[str, dict[str, object]] = {
            recipe_id: {
                "title": retrieved_records[recipe_id].title,
                "rank": rank,
                "input_utilization": details.input_utilization,
                "recipe_coverage": details.recipe_coverage,
                "missing_count": details.missing_count,
                "matched_inputs": list(details.matched_inputs),
                "unmatched_inputs": list(details.unmatched_inputs),
                "missing_recipe_ingredients": list(details.missing_recipe_ingredients),
            }
            for rank, (recipe_id, details) in enumerate(retrieved_details.items(), start=1)
        }
        retrieved_input_utilizations = [
            details.input_utilization for details in retrieved_details.values()
        ]
        best_retrieved_input_utilization_rate = (
            max(retrieved_input_utilizations) if retrieved_input_utilizations else None
        )

        output_kind: Literal["corpus", "generated", "none"] = "none"
        if selected_ids:
            output_kind = "corpus"
        elif generated_recipes:
            output_kind = "generated"
        output_count = len(selected_ids) + len(generated_recipes)

        common_violations: dict[str, list[str]] = {
            f"corpus:{recipe_id}": violations_by_id[recipe_id] for recipe_id in selected_ids
        }
        generated_coverages: list[float] = []
        generated_input_utilizations: list[float] = []
        complete_count = sum(_corpus_recipe_complete(records[recipe_id]) for recipe_id in valid_ids)
        common_pass_count = len(passed_ids)
        common_successful_item_count = sum(
            not violations_by_id[recipe_id] and _corpus_recipe_complete(records[recipe_id])
            for recipe_id in valid_ids
        )
        for index, generated in enumerate(generated_recipes, start=1):
            key = f"generated:{index}"
            generated_violations = generated_hard_constraint_violations(
                case.request,
                generated,
                allergen_aliases=allergen_aliases,
                ingredient_aliases=ingredient_aliases,
                diet_conflict_markers=diet_conflict_markers,
                equipment_aliases=equipment_aliases,
            )
            common_violations[key] = generated_violations
            generated_complete = _generated_recipe_complete(generated)
            common_pass_count += not generated_violations
            complete_count += generated_complete
            common_successful_item_count += generated_complete and not generated_violations
            generated_coverage = generated_ingredient_coverage(
                case.request,
                generated,
                pantry_staples=pantry_staples,
                ingredient_aliases=ingredient_aliases,
            )
            if generated_coverage is not None:
                generated_coverages.append(generated_coverage)
            generated_input_utilizations.append(
                generated_input_utilization(
                    case.request,
                    generated,
                    ingredient_aliases=ingredient_aliases,
                )
            )

        common_coverages = [*coverage_values, *generated_coverages]
        common_input_utilizations = [
            *corpus_input_utilizations,
            *generated_input_utilizations,
        ]
        common_hard_pass_rate = common_pass_count / output_count if output_count else None
        common_coverage_rate = fmean(common_coverages) if common_coverages else None
        common_input_utilization_rate = (
            fmean(common_input_utilizations) if common_input_utilizations else None
        )
        best_common_input_utilization_rate = (
            max(common_input_utilizations) if common_input_utilizations else None
        )
        selection_input_utilization_regret = None
        if (
            output_kind == "corpus"
            and best_retrieved_input_utilization_rate is not None
            and best_common_input_utilization_rate is not None
        ):
            selection_input_utilization_regret = max(
                0.0,
                best_retrieved_input_utilization_rate
                - best_common_input_utilization_rate,
            )
        all_requested_ingredients_used = bool(
            best_common_input_utilization_rate is not None
            and math.isclose(best_common_input_utilization_rate, 1.0)
        )
        completeness_rate = complete_count / output_count if output_count else None

        provenance_kind = prediction.provenance_kind
        if provenance_kind is None and output_kind != "none":
            provenance_kind = output_kind
        provenance_applicable = provenance_kind == "corpus" if provenance_kind is not None else None
        all_requested_ingredients_metric_applicable = case.expected_no_result is not True and not (
            provenance_applicable is True and case.expected_corpus_no_result is True
        )

        no_result_reasonable: bool | None = None
        if case.label_status == "reviewed" and case.expected_no_result is not None:
            if case.expected_no_result:
                no_result_reasonable = prediction.status == "no_result"
            else:
                no_result_reasonable = prediction.status == "success" and bool(common_pass_count)

        corpus_no_result_reasonable: bool | None = None
        if provenance_applicable and case.expected_corpus_no_result is not None:
            if case.expected_corpus_no_result:
                corpus_no_result_reasonable = prediction.status == "no_result"
            else:
                corpus_no_result_reasonable = prediction.status == "success" and bool(passed_ids)

        common_success: bool | None
        if prediction.status == "success":
            common_success = bool(common_successful_item_count)
        elif prediction.status == "no_result":
            common_success = no_result_reasonable
        else:
            common_success = False

        results.append(
            CaseEvaluation(
                case_id=case.case_id,
                split=case.split,
                request=case.request,
                expected_no_result=case.expected_no_result,
                expected_corpus_no_result=case.expected_corpus_no_result,
                label_status=case.label_status,
                runner_name=runner_name,
                prediction_status=prediction.status,
                output_kind=output_kind,
                output_item_count=output_count,
                generated_recipes=generated_recipes,
                selected_recipe_ids=selected_ids,
                valid_recipe_ids=valid_ids,
                invalid_recipe_ids=invalid_ids,
                hard_constraint_passed_ids=passed_ids,
                hard_constraint_failed_ids=failed_ids,
                hard_constraint_violations=violations_by_id,
                valid_recipe_id_rate=valid_rate,
                hard_constraint_pass_rate=hard_pass_rate,
                ingredient_coverage_rate=coverage_rate,
                common_hard_constraint_violations=common_violations,
                common_hard_constraint_pass_rate=common_hard_pass_rate,
                common_ingredient_coverage_rate=common_coverage_rate,
                common_input_utilization_rate=common_input_utilization_rate,
                best_common_input_utilization_rate=best_common_input_utilization_rate,
                all_requested_ingredients_used=all_requested_ingredients_used,
                all_requested_ingredients_metric_applicable=(
                    all_requested_ingredients_metric_applicable
                ),
                output_completeness_rate=completeness_rate,
                common_success=common_success,
                provenance_applicable=provenance_applicable,
                latency_ms=latency_ms,
                model_call_count=prediction.model_call_count,
                input_tokens=prediction.input_tokens,
                output_tokens=prediction.output_tokens,
                estimated_cost_usd=prediction.estimated_cost_usd,
                application_output=prediction.application_output,
                workflow_trace=prediction.workflow_trace,
                validation_errors=prediction.validation_errors,
                retry_count=prediction.retry_count,
                retrieved_candidate_ids=retrieved_ids,
                retrieved_candidate_metrics=retrieved_candidate_metrics,
                valid_retrieved_candidate_ids=valid_retrieved_ids,
                invalid_retrieved_candidate_ids=invalid_retrieved_ids,
                retrieved_candidate_count=(len(retrieved_ids) if retrieved_ids is not None else None),
                best_retrieved_input_utilization_rate=(
                    best_retrieved_input_utilization_rate
                ),
                selection_input_utilization_regret=selection_input_utilization_regret,
                failure_stage=prediction.failure_stage,
                no_result_reasonable=no_result_reasonable,
                corpus_no_result_reasonable=corpus_no_result_reasonable,
                no_result_reason=prediction.no_result_reason,
                error_message=prediction.error_message,
            )
        )
    return results


def summarize_results(results: Sequence[CaseEvaluation], runner_name: str) -> RunnerSummary:
    """Aggregate selection-level correctness and case-level latency metrics."""

    selected = [result for result in results if result.runner_name == runner_name]
    total_outputs = sum(result.output_item_count for result in selected)
    total_ids = sum(len(result.selected_recipe_ids) for result in selected)
    valid_ids = sum(len(result.valid_recipe_ids) for result in selected)
    hard_passes = sum(len(result.hard_constraint_passed_ids) for result in selected)
    coverages = [
        result.ingredient_coverage_rate
        for result in selected
        if result.ingredient_coverage_rate is not None
    ]
    common_coverages = [
        result.common_ingredient_coverage_rate
        for result in selected
        if result.common_ingredient_coverage_rate is not None
    ]
    common_input_utilizations = [
        result.common_input_utilization_rate
        for result in selected
        if result.common_input_utilization_rate is not None
    ]
    best_input_utilizations = [
        result.best_common_input_utilization_rate
        for result in selected
        if result.best_common_input_utilization_rate is not None
    ]
    applicable_all_input_results = [
        result for result in selected if result.all_requested_ingredients_metric_applicable
    ]
    common_hard_passes = sum(
        result.common_hard_constraint_pass_rate * result.output_item_count
        for result in selected
        if result.common_hard_constraint_pass_rate is not None
    )
    complete_outputs = sum(
        result.output_completeness_rate * result.output_item_count
        for result in selected
        if result.output_completeness_rate is not None
    )
    common_successes = [
        result.common_success for result in selected if result.common_success is not None
    ]
    latencies = sorted(result.latency_ms for result in selected)
    labeled = [
        result.no_result_reasonable
        for result in selected
        if result.no_result_reasonable is not None
    ]
    corpus_labeled = [
        result.corpus_no_result_reasonable
        for result in selected
        if result.corpus_no_result_reasonable is not None
    ]
    model_calls = [result.model_call_count for result in selected]
    input_tokens = [result.input_tokens for result in selected]
    output_tokens = [result.output_tokens for result in selected]
    costs = [result.estimated_cost_usd for result in selected]
    retrieval_diagnostic_results = [
        result for result in selected if result.retrieved_candidate_ids is not None
    ]
    retrieved_counts = [
        result.retrieved_candidate_count
        for result in retrieval_diagnostic_results
        if result.retrieved_candidate_count is not None
    ]
    best_retrieved_utilizations = [
        result.best_retrieved_input_utilization_rate
        for result in retrieval_diagnostic_results
        if result.best_retrieved_input_utilization_rate is not None
    ]
    selection_regrets = [
        result.selection_input_utilization_regret
        for result in retrieval_diagnostic_results
        if result.selection_input_utilization_regret is not None
    ]
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1) if latencies else 0

    def complete_sum(values: Sequence[int | float | None]) -> int | float | None:
        if not values or any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    total_calls = complete_sum(model_calls)
    total_input = complete_sum(input_tokens)
    total_output = complete_sum(output_tokens)
    total_cost = complete_sum(costs)

    return RunnerSummary(
        runner_name=runner_name,
        case_count=len(selected),
        error_case_count=sum(result.prediction_status == "error" for result in selected),
        total_output_items=total_outputs,
        total_selected_recipe_ids=total_ids,
        valid_recipe_id_rate=valid_ids / total_ids if total_ids else None,
        hard_constraint_pass_rate=hard_passes / total_ids if total_ids else None,
        mean_ingredient_coverage_rate=fmean(coverages) if coverages else None,
        common_hard_constraint_pass_rate=(
            common_hard_passes / total_outputs if total_outputs else None
        ),
        mean_common_ingredient_coverage_rate=(
            fmean(common_coverages) if common_coverages else None
        ),
        mean_common_input_utilization_rate=(
            fmean(common_input_utilizations) if common_input_utilizations else None
        ),
        mean_best_input_utilization_rate=(
            fmean(best_input_utilizations) if best_input_utilizations else None
        ),
        all_requested_ingredients_case_rate=(
            sum(result.all_requested_ingredients_used for result in selected) / len(selected)
            if selected
            else None
        ),
        applicable_all_requested_ingredients_case_rate=(
            sum(result.all_requested_ingredients_used for result in applicable_all_input_results)
            / len(applicable_all_input_results)
            if applicable_all_input_results
            else None
        ),
        all_requested_ingredients_applicable_case_count=len(applicable_all_input_results),
        output_completeness_rate=(complete_outputs / total_outputs if total_outputs else None),
        common_success_rate=(
            sum(common_successes) / len(common_successes) if common_successes else None
        ),
        mean_latency_ms=fmean(latencies) if latencies else None,
        median_latency_ms=median(latencies) if latencies else None,
        p95_latency_ms=latencies[p95_index] if latencies else None,
        no_result_reasonableness_rate=(sum(labeled) / len(labeled) if labeled else None),
        no_result_labeled_case_count=len(labeled),
        corpus_no_result_accuracy=(
            sum(corpus_labeled) / len(corpus_labeled) if corpus_labeled else None
        ),
        corpus_no_result_labeled_case_count=len(corpus_labeled),
        mean_model_calls=(
            float(total_calls) / len(selected) if total_calls is not None and selected else None
        ),
        total_input_tokens=int(total_input) if total_input is not None else None,
        total_output_tokens=int(total_output) if total_output is not None else None,
        total_estimated_cost_usd=float(total_cost) if total_cost is not None else None,
        retrieval_diagnostic_case_count=len(retrieval_diagnostic_results),
        mean_retrieved_candidate_count=(
            fmean(retrieved_counts) if retrieved_counts else None
        ),
        mean_best_retrieved_input_utilization_rate=(
            fmean(best_retrieved_utilizations) if best_retrieved_utilizations else None
        ),
        mean_selection_input_utilization_regret=(
            fmean(selection_regrets) if selection_regrets else None
        ),
    )


def _case_set_hash(cases: Sequence[EvaluationCase]) -> str:
    canonical = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_multi_arm_comparison(
    cases: Sequence[EvaluationCase],
    *,
    runners: Mapping[str, RunnerLike],
    catalog: CatalogLike,
    allergen_aliases: Mapping[str, list[str]] | None = None,
    ingredient_aliases: Mapping[str, str] | None = None,
    diet_conflict_markers: Mapping[str, list[str]] | None = None,
    equipment_aliases: Mapping[str, list[str]] | None = None,
    pantry_staples: Sequence[str] = (),
    clock: Callable[[], float] = perf_counter,
    metadata: Mapping[str, str | int | float | bool | None] | None = None,
) -> ExperimentReport:
    """Evaluate any ordered set of arms over exactly the same frozen cases.

    The standard configuration uses ``baseline_a``, optional ``baseline_b``
    and ``agentic`` keys. The generic mapping also supports offline test
    doubles and records the arm order explicitly in the saved report.
    """

    validated_cases = [EvaluationCase.model_validate(case) for case in cases]
    if not validated_cases:
        raise ValueError("At least one evaluation case is required")
    case_ids = [case.case_id for case in validated_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be unique")

    if not runners:
        raise ValueError("At least one evaluation runner is required")
    runner_names = list(runners)
    if any(
        not isinstance(name, str) or not name.strip() or name != name.strip()
        for name in runner_names
    ):
        raise ValueError("Runner names must be non-blank and have no surrounding whitespace")
    if len(runner_names) != len(set(runner_names)):
        raise ValueError("Runner names must be unique")

    results: list[CaseEvaluation] = []
    for name, runner in runners.items():
        results.extend(
            evaluate_runner(
                validated_cases,
                runner_name=name,
                runner=runner,
                catalog=catalog,
                allergen_aliases=allergen_aliases,
                ingredient_aliases=ingredient_aliases,
                diet_conflict_markers=diet_conflict_markers,
                equipment_aliases=equipment_aliases,
                pantry_staples=pantry_staples,
                clock=clock,
            )
        )
    summaries = [summarize_results(results, name) for name in runner_names]
    return ExperimentReport(
        case_set_sha256=_case_set_hash(validated_cases),
        case_count=len(validated_cases),
        results=results,
        summaries=summaries,
        metadata=dict(metadata or {}),
    )


def run_interleaved_comparison(
    cases: Sequence[EvaluationCase],
    *,
    runners: Mapping[str, RunnerLike],
    catalog: CatalogLike,
    allergen_aliases: Mapping[str, list[str]] | None = None,
    ingredient_aliases: Mapping[str, str] | None = None,
    diet_conflict_markers: Mapping[str, list[str]] | None = None,
    equipment_aliases: Mapping[str, list[str]] | None = None,
    pantry_staples: Sequence[str] = (),
    clock: Callable[[], float] = perf_counter,
    metadata: Mapping[str, str | int | float | bool | None] | None = None,
    reverse_first_case_order: bool = False,
    on_result: Callable[[CaseEvaluation], None] | None = None,
) -> ExperimentReport:
    """Evaluate runners case by case while alternating which arm runs first.

    Live API latency can drift during a long experiment.
    :func:`run_multi_arm_comparison` runs one complete arm at a time for
    offline evaluation. Formal live comparison interleaves the arms so one
    system is not always measured earlier in the session. Runner order is
    reversed for every second case while summaries retain the configured
    runner order.

    ``on_result`` is called immediately after each scored system-case result.
    The final-evaluation CLI uses it to append a crash-resistant JSONL log.
    """

    validated_cases = [EvaluationCase.model_validate(case) for case in cases]
    if not validated_cases:
        raise ValueError("At least one evaluation case is required")
    case_ids = [case.case_id for case in validated_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be unique")

    if not runners:
        raise ValueError("At least one evaluation runner is required")
    runner_names = list(runners)
    if any(
        not isinstance(name, str) or not name.strip() or name != name.strip()
        for name in runner_names
    ):
        raise ValueError("Runner names must be non-blank and have no surrounding whitespace")
    if len(runner_names) != len(set(runner_names)):
        raise ValueError("Runner names must be unique")

    results: list[CaseEvaluation] = []
    for case_index, evaluation_case in enumerate(validated_cases):
        reverse_order = bool(case_index % 2) ^ reverse_first_case_order
        ordered_names = list(reversed(runner_names)) if reverse_order else runner_names
        for runner_name in ordered_names:
            result = evaluate_runner(
                [evaluation_case],
                runner_name=runner_name,
                runner=runners[runner_name],
                catalog=catalog,
                allergen_aliases=allergen_aliases,
                ingredient_aliases=ingredient_aliases,
                diet_conflict_markers=diet_conflict_markers,
                equipment_aliases=equipment_aliases,
                pantry_staples=pantry_staples,
                clock=clock,
            )[0]
            results.append(result)
            if on_result is not None:
                on_result(result)

    summaries = [summarize_results(results, name) for name in runner_names]
    return ExperimentReport(
        case_set_sha256=_case_set_hash(validated_cases),
        case_count=len(validated_cases),
        results=results,
        summaries=summaries,
        metadata=dict(metadata or {}),
    )


def run_comparison(
    cases: Sequence[EvaluationCase],
    *,
    baseline_runner: RunnerLike,
    agentic_runner: RunnerLike,
    catalog: CatalogLike,
    allergen_aliases: Mapping[str, list[str]] | None = None,
    ingredient_aliases: Mapping[str, str] | None = None,
    diet_conflict_markers: Mapping[str, list[str]] | None = None,
    equipment_aliases: Mapping[str, list[str]] | None = None,
    pantry_staples: Sequence[str] = (),
    clock: Callable[[], float] = perf_counter,
    metadata: Mapping[str, str | int | float | bool | None] | None = None,
) -> ExperimentReport:
    """Backward-compatible two-arm wrapper for legacy tests and scripts."""

    return run_multi_arm_comparison(
        cases,
        runners={"baseline": baseline_runner, "agentic": agentic_runner},
        catalog=catalog,
        allergen_aliases=allergen_aliases,
        ingredient_aliases=ingredient_aliases,
        diet_conflict_markers=diet_conflict_markers,
        equipment_aliases=equipment_aliases,
        pantry_staples=pantry_staples,
        clock=clock,
        metadata=metadata,
    )


def _parse_json_string_list(raw: str, field_name: str) -> list[str]:
    if not raw.strip():
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must contain a JSON array") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must contain a JSON array of strings")
    return value


def _parse_optional_bool(raw: str) -> bool | None:
    normalized = raw.strip().lower()
    if not normalized:
        return None
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("expected_no_result must be true, false or blank")


def load_cases_csv(path: str | Path) -> list[EvaluationCase]:
    """Load the CSV case format defined by ``evaluation/cases_template.csv``."""

    cases: list[EvaluationCase] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                max_time_raw = (row.get("max_time_minutes") or "").strip()
                request = UserRequest(
                    available_ingredients=_parse_json_string_list(
                        row.get("available_ingredients_json") or "",
                        "available_ingredients_json",
                    ),
                    max_time_minutes=int(max_time_raw) if max_time_raw else None,
                    excluded_ingredients=_parse_json_string_list(
                        row.get("excluded_ingredients_json") or "",
                        "excluded_ingredients_json",
                    ),
                    unavailable_equipment=set(
                        _parse_json_string_list(
                            row.get("unavailable_equipment_json") or "",
                            "unavailable_equipment_json",
                        )
                    ),
                    diet_preference=(row.get("diet_preference") or "none").strip() or "none",
                    meal_type=(row.get("meal_type") or "").strip() or None,
                    additional_notes=(row.get("additional_notes") or "").strip() or None,
                )
                cases.append(
                    EvaluationCase(
                        case_id=(row.get("case_id") or "").strip(),
                        split=(row.get("split") or "").strip(),
                        request=request,
                        expected_no_result=_parse_optional_bool(
                            row.get("expected_no_result") or ""
                        ),
                        expected_corpus_no_result=_parse_optional_bool(
                            row.get("expected_corpus_no_result") or ""
                        ),
                        label_status=(row.get("label_status") or "placeholder").strip(),
                        notes=(row.get("notes") or "").strip() or None,
                    )
                )
            except (ValueError, ValidationError) as exc:
                raise ValueError(f"Invalid evaluation case on CSV row {row_number}: {exc}") from exc
    return cases


def save_evaluation_results(
    report: ExperimentReport,
    output_dir: str | Path,
    *,
    stem: str = "comparison",
) -> tuple[Path, Path]:
    """Atomically save detailed CSV measurements and a complete JSON report."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
        raise ValueError("stem may contain only letters, numbers, dot, underscore and hyphen")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{stem}_cases.csv"
    json_path = output_path / f"{stem}_report.json"
    csv_temp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    json_temp = json_path.with_suffix(json_path.suffix + ".tmp")

    fieldnames = list(CaseEvaluation.model_fields)
    with csv_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in report.results:
            row = result.model_dump(mode="json")
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            writer.writerow(row)

    json_temp.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    csv_temp.replace(csv_path)
    json_temp.replace(json_path)
    return csv_path, json_path
