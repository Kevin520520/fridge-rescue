"""Small, offline-testable LangGraph workflow for Fridge Rescue.

The language-model-facing selector is limited to choosing candidate recipe IDs
and providing short explanatory text. Recipe facts in the final result are
loaded from ``RecipeRepository`` after validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fridge_rescue.retrieval import ingredient_match_details
from fridge_rescue.rule_checker import find_excluded_matches, normalize_token
from fridge_rescue.schemas import LLMRecommendation, RecipeRecord, UserRequest
from fridge_rescue.selector_errors import SelectorOutputError, SelectorServiceError


class RecipeRepository(Protocol):
    """Read normalized recipe facts from the curated corpus."""

    def get_many(self, recipe_ids: Sequence[str]) -> list[RecipeRecord]: ...


class RecipeRetriever(Protocol):
    """Return ranked recipe IDs; it must not invent recipe content."""

    def retrieve(self, request: UserRequest, *, limit: int) -> list[str]: ...


class RecommendationSelector(Protocol):
    """The replaceable LLM boundary used for the first draft and one repair.

    Implementations raise :class:`SelectorOutputError` for malformed model
    drafts that one bounded repair may fix, and :class:`SelectorServiceError`
    for provider failures that must stop immediately.
    """

    def select(
        self,
        request: UserRequest,
        candidates: Sequence[RecipeRecord],
        *,
        validation_errors: Sequence[str] = (),
        previous_draft: LLMRecommendation | None = None,
    ) -> LLMRecommendation: ...


class SequenceSelector:
    """Offline selector that returns predefined responses in order.

    The deterministic sequence requires no API key and implements the same
    ``select`` interface as provider-backed adapters.
    """

    def __init__(self, responses: Sequence[LLMRecommendation | Mapping[str, object]]) -> None:
        if not responses:
            raise ValueError("SequenceSelector needs at least one response")
        self._responses = list(responses)
        self.call_count = 0

    def select(
        self,
        request: UserRequest,
        candidates: Sequence[RecipeRecord],
        *,
        validation_errors: Sequence[str] = (),
        previous_draft: LLMRecommendation | None = None,
    ) -> LLMRecommendation:
        del request, candidates, validation_errors, previous_draft
        if self.call_count >= len(self._responses):
            raise RuntimeError("SequenceSelector has no response left")
        response = self._responses[self.call_count]
        self.call_count += 1
        return LLMRecommendation.model_validate(response)


class HydratedRecommendation(BaseModel):
    """A selected explanation joined to authoritative repository facts."""

    model_config = ConfigDict(extra="forbid")

    recipe: RecipeRecord
    reason: str
    substitution_notes: list[str] = Field(default_factory=list)


OutcomeStatus = Literal[
    "success",
    "no_result",
    "no_candidates",
    "invalid_request",
    "failed_validation",
    "service_error",
]


class WorkflowOutcome(BaseModel):
    """Stable public result returned by :meth:`FridgeRescueWorkflow.run`."""

    model_config = ConfigDict(extra="forbid")

    status: OutcomeStatus
    recommendations: list[HydratedRecommendation] = Field(default_factory=list)
    no_result_reason: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0, le=1)
    trace: list[str] = Field(default_factory=list)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class AgentWorkflowState(TypedDict, total=False):
    request: UserRequest | Mapping[str, object]
    retrieved_candidate_ids: list[str]
    candidate_records: list[RecipeRecord]
    draft: LLMRecommendation
    selector_error: str | None
    validation_errors: list[str]
    validation_passed: bool
    retry_count: int
    allowed_substitution_notes: list[str]
    outcome_status: OutcomeStatus
    outcome: WorkflowOutcome
    trace: list[str]
    input_tokens: int
    output_tokens: int
    usage_complete: bool


def _append_trace(state: AgentWorkflowState, node: str) -> list[str]:
    return [*state.get("trace", []), node]


def _unique_ids(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        recipe_id = str(value).strip()
        if recipe_id and recipe_id not in seen:
            seen.add(recipe_id)
            result.append(recipe_id)
    return result


class FridgeRescueWorkflow:
    """Five-node deterministic-control workflow with at most one repair."""

    def __init__(
        self,
        *,
        repository: RecipeRepository,
        retriever: RecipeRetriever,
        selector: RecommendationSelector,
        top_k: int = 10,
        allowed_substitution_notes: Sequence[str] = (),
        allergen_aliases: Mapping[str, list[str]] | None = None,
        ingredient_aliases: Mapping[str, str] | None = None,
        pantry_staples: Sequence[str] = (),
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.repository = repository
        self.retriever = retriever
        self.selector = selector
        self.top_k = top_k
        self.allowed_substitution_notes = _unique_ids(list(allowed_substitution_notes))
        self.allergen_aliases = dict(allergen_aliases or {})
        self.ingredient_aliases = {
            normalize_token(str(key)): normalize_token(str(value))
            for key, value in (ingredient_aliases or {}).items()
        }
        self.pantry_staples = tuple(
            dict.fromkeys(normalize_token(item) for item in pantry_staples if item.strip())
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentWorkflowState)
        builder.add_node("validate_and_normalize", self._validate_and_normalize)
        builder.add_node("retrieve_and_rank", self._retrieve_and_rank)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("validate_answer", self._validate_answer)
        builder.add_node("repair_once", self._repair_once)

        builder.add_edge(START, "validate_and_normalize")
        builder.add_conditional_edges(
            "validate_and_normalize",
            self._after_request_validation,
            {"continue": "retrieve_and_rank", "end": END},
        )
        builder.add_conditional_edges(
            "retrieve_and_rank",
            self._after_retrieval,
            {"continue": "generate_answer", "end": END},
        )
        builder.add_conditional_edges(
            "generate_answer",
            self._after_selector_call,
            {"validate": "validate_answer", "end": END},
        )
        builder.add_conditional_edges(
            "validate_answer",
            self._after_answer_validation,
            {"repair": "repair_once", "end": END},
        )
        builder.add_conditional_edges(
            "repair_once",
            self._after_selector_call,
            {"validate": "validate_answer", "end": END},
        )
        return builder.compile()

    def run(self, request: UserRequest | Mapping[str, object]) -> WorkflowOutcome:
        """Run the graph and return only the stable, hydrated public outcome."""

        final_state = self.graph.invoke(
            {
                "request": request,
                "retry_count": 0,
                "allowed_substitution_notes": self.allowed_substitution_notes,
                "trace": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "usage_complete": True,
            }
        )
        return WorkflowOutcome.model_validate(final_state["outcome"])

    @staticmethod
    def _outcome_usage(state: AgentWorkflowState) -> dict[str, int | None]:
        if not state.get("usage_complete", True):
            return {"input_tokens": None, "output_tokens": None}
        return {
            "input_tokens": state.get("input_tokens", 0),
            "output_tokens": state.get("output_tokens", 0),
        }

    def _selector_usage_update(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Add usage from the selector call that just finished.

        An absent or malformed provider usage object is recorded as unknown,
        never as zero.  Totals remain available only when every model call in
        the workflow reported both counters.
        """

        usage = getattr(self.selector, "last_usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        valid_input = (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and input_tokens >= 0
        )
        valid_output = (
            isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens >= 0
        )
        if not valid_input or not valid_output:
            return {"usage_complete": False}
        return {
            "input_tokens": state.get("input_tokens", 0) + input_tokens,
            "output_tokens": state.get("output_tokens", 0) + output_tokens,
            "usage_complete": state.get("usage_complete", True),
        }

    def _validate_and_normalize(self, state: AgentWorkflowState) -> AgentWorkflowState:
        trace = _append_trace(state, "validate_and_normalize")
        try:
            request = UserRequest.model_validate(state.get("request"))
        except ValidationError as exc:
            errors = [error["msg"] for error in exc.errors()]
            outcome = WorkflowOutcome(
                status="invalid_request",
                validation_errors=errors,
                retry_count=state.get("retry_count", 0),
                trace=trace,
                **self._outcome_usage(state),
            )
            return {
                "validation_passed": False,
                "validation_errors": errors,
                "outcome_status": "invalid_request",
                "outcome": outcome,
                "trace": trace,
            }
        return {
            "request": request,
            "validation_passed": True,
            "validation_errors": [],
            "trace": trace,
        }

    @staticmethod
    def _after_request_validation(state: AgentWorkflowState) -> str:
        return "end" if state.get("outcome_status") == "invalid_request" else "continue"

    def _retrieve_and_rank(self, state: AgentWorkflowState) -> AgentWorkflowState:
        request = UserRequest.model_validate(state["request"])
        ranked_ids = _unique_ids(self.retriever.retrieve(request, limit=self.top_k))[: self.top_k]
        records_by_id = {
            record.recipe_id: record for record in self.repository.get_many(ranked_ids)
        }
        candidate_ids = [recipe_id for recipe_id in ranked_ids if recipe_id in records_by_id]
        candidates = [records_by_id[recipe_id] for recipe_id in candidate_ids]
        trace = _append_trace(state, "retrieve_and_rank")

        update: AgentWorkflowState = {
            "retrieved_candidate_ids": candidate_ids,
            "candidate_records": candidates,
            "trace": trace,
        }
        if not candidates:
            outcome = WorkflowOutcome(
                status="no_candidates",
                no_result_reason="No matching recipes were found in the curated corpus.",
                retry_count=state.get("retry_count", 0),
                trace=trace,
                **self._outcome_usage(state),
            )
            update.update(outcome_status="no_candidates", outcome=outcome)
        return update

    @staticmethod
    def _after_retrieval(state: AgentWorkflowState) -> str:
        return "end" if state.get("outcome_status") == "no_candidates" else "continue"

    def _call_selector(
        self,
        state: AgentWorkflowState,
        *,
        trace_node: str,
        retry_count: int | None = None,
    ) -> AgentWorkflowState:
        trace = _append_trace(state, trace_node)
        current_retry_count = state.get("retry_count", 0) if retry_count is None else retry_count
        try:
            draft = self.selector.select(
                UserRequest.model_validate(state["request"]),
                state.get("candidate_records", []),
                validation_errors=state.get("validation_errors", []),
                previous_draft=state.get("draft") if trace_node == "repair_once" else None,
            )
            draft = LLMRecommendation.model_validate(draft)
        except SelectorServiceError:
            usage_update = self._selector_usage_update(state)
            usage_state: AgentWorkflowState = {**state, **usage_update}
            outcome = WorkflowOutcome(
                status="service_error",
                retry_count=current_retry_count,
                trace=trace,
                **self._outcome_usage(usage_state),
            )
            return {
                "selector_error": "The recommendation service is unavailable.",
                "validation_passed": False,
                "retry_count": current_retry_count,
                "outcome_status": "service_error",
                "outcome": outcome,
                "trace": trace,
                **usage_update,
            }
        except (SelectorOutputError, ValidationError, ValueError, TypeError) as exc:
            usage_update = self._selector_usage_update(state)
            return {
                "selector_error": f"Selector returned an invalid response: {exc}",
                "validation_passed": False,
                "retry_count": current_retry_count,
                "trace": trace,
                **usage_update,
            }
        except RuntimeError:
            # Unknown selector failures stop as service failures, preventing a
            # second paid call and suppressing provider diagnostics.
            usage_update = self._selector_usage_update(state)
            usage_state = {**state, **usage_update}
            outcome = WorkflowOutcome(
                status="service_error",
                retry_count=current_retry_count,
                trace=trace,
                **self._outcome_usage(usage_state),
            )
            return {
                "selector_error": "The recommendation service is unavailable.",
                "validation_passed": False,
                "retry_count": current_retry_count,
                "outcome_status": "service_error",
                "outcome": outcome,
                "trace": trace,
                **usage_update,
            }
        usage_update = self._selector_usage_update(state)
        return {
            "draft": draft,
            "selector_error": None,
            "retry_count": current_retry_count,
            "trace": trace,
            **usage_update,
        }

    def _generate_answer(self, state: AgentWorkflowState) -> AgentWorkflowState:
        return self._call_selector(state, trace_node="generate_answer")

    def _repair_once(self, state: AgentWorkflowState) -> AgentWorkflowState:
        return self._call_selector(
            state,
            trace_node="repair_once",
            retry_count=state.get("retry_count", 0) + 1,
        )

    @staticmethod
    def _after_selector_call(state: AgentWorkflowState) -> str:
        return "end" if state.get("outcome_status") == "service_error" else "validate"

    def _hard_constraint_errors(self, request: UserRequest, record: RecipeRecord) -> list[str]:
        errors: list[str] = []
        if request.max_time_minutes is not None:
            if record.total_time_minutes is None:
                errors.append(f"{record.recipe_id}: total time is unknown")
            elif record.total_time_minutes > request.max_time_minutes:
                errors.append(f"{record.recipe_id}: exceeds the requested time limit")

        unavailable = record.equipment_tags_inferred & request.unavailable_equipment
        if unavailable:
            errors.append(
                f"{record.recipe_id}: requires unavailable equipment {sorted(unavailable)}"
            )

        excluded = find_excluded_matches(
            record.ingredients_normalized,
            request.excluded_ingredients,
            self.allergen_aliases,
            self.ingredient_aliases,
        )
        if excluded:
            errors.append(f"{record.recipe_id}: contains excluded ingredients {sorted(excluded)}")

        if request.diet_preference == "vegetarian" and record.vegetarian_status != "match":
            errors.append(f"{record.recipe_id}: vegetarian status is not a confirmed match")
        if request.diet_preference == "vegan" and record.vegan_status != "match":
            errors.append(f"{record.recipe_id}: vegan status is not a confirmed match")
        return errors

    def _draft_errors(self, state: AgentWorkflowState) -> list[str]:
        if state.get("selector_error"):
            return [state["selector_error"]]  # type: ignore[list-item]
        draft = state.get("draft")
        if draft is None:
            return ["Selector did not return a recommendation"]

        request = UserRequest.model_validate(state["request"])
        records = {record.recipe_id: record for record in state.get("candidate_records", [])}
        feasible_ids = {
            recipe_id
            for recipe_id, record in records.items()
            if not self._hard_constraint_errors(request, record)
        }
        if draft.no_result_reason is not None:
            if feasible_ids:
                return ["A no-result answer is invalid because feasible candidates exist"]
            return []

        errors: list[str] = []
        allowed_notes = {
            normalize_token(note) for note in state.get("allowed_substitution_notes", [])
        }
        for selection in draft.selections:
            record = records.get(selection.recipe_id)
            if record is None:
                errors.append(f"{selection.recipe_id}: ID is not in the retrieved candidates")
                continue
            errors.extend(self._hard_constraint_errors(request, record))
            for note in selection.substitution_notes:
                if normalize_token(note) not in allowed_notes:
                    errors.append(
                        f"{selection.recipe_id}: substitution note is not in the reviewed allowlist"
                    )

        feasible_records = [records[recipe_id] for recipe_id in feasible_ids]
        selected_records = [
            records[selection.recipe_id]
            for selection in draft.selections
            if selection.recipe_id in feasible_ids
        ]
        if feasible_records and selected_records:
            candidate_utilizations = {
                record.recipe_id: ingredient_match_details(
                    record,
                    request.available_ingredients,
                    self.pantry_staples,
                    self.ingredient_aliases,
                ).input_utilization
                for record in feasible_records
            }
            best_candidate_utilization = max(candidate_utilizations.values())
            best_selected_utilization = max(
                ingredient_match_details(
                    record,
                    request.available_ingredients,
                    self.pantry_staples,
                    self.ingredient_aliases,
                ).input_utilization
                for record in selected_records
            )
            if best_selected_utilization + 1e-9 < best_candidate_utilization:
                best_candidate_ids = sorted(
                    recipe_id
                    for recipe_id, utilization in candidate_utilizations.items()
                    if abs(utilization - best_candidate_utilization) <= 1e-9
                )
                errors.append(
                    "The selected recipes have lower input_utilization than the best "
                    "feasible candidates. Include at least one of these recipe IDs: "
                    f"{', '.join(best_candidate_ids)}."
                )
        return errors

    def _hydrate(self, state: AgentWorkflowState) -> list[HydratedRecommendation]:
        draft = state["draft"]
        selected_ids = [selection.recipe_id for selection in draft.selections]
        authoritative = {
            record.recipe_id: record for record in self.repository.get_many(selected_ids)
        }
        missing = [recipe_id for recipe_id in selected_ids if recipe_id not in authoritative]
        if missing:
            raise ValueError(f"Repository could not hydrate selected IDs: {missing}")
        return [
            HydratedRecommendation(
                recipe=authoritative[selection.recipe_id],
                reason=selection.reason,
                substitution_notes=selection.substitution_notes,
            )
            for selection in draft.selections
        ]

    def _validate_answer(self, state: AgentWorkflowState) -> AgentWorkflowState:
        trace = _append_trace(state, "validate_answer")
        errors = self._draft_errors(state)
        recommendations: list[HydratedRecommendation] = []
        if not errors and state["draft"].selections:
            try:
                recommendations = self._hydrate(state)
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            retry_count = state.get("retry_count", 0)
            update: AgentWorkflowState = {
                "validation_passed": False,
                "validation_errors": errors,
                "trace": trace,
            }
            if retry_count >= 1:
                outcome = WorkflowOutcome(
                    status="failed_validation",
                    validation_errors=errors,
                    retry_count=retry_count,
                    trace=trace,
                    **self._outcome_usage(state),
                )
                update.update(outcome_status="failed_validation", outcome=outcome)
            return update

        draft = state["draft"]
        status: OutcomeStatus = "success" if recommendations else "no_result"
        outcome = WorkflowOutcome(
            status=status,
            recommendations=recommendations,
            no_result_reason=draft.no_result_reason,
            retry_count=state.get("retry_count", 0),
            trace=trace,
            **self._outcome_usage(state),
        )
        return {
            "validation_passed": True,
            "validation_errors": [],
            "outcome_status": status,
            "outcome": outcome,
            "trace": trace,
        }

    @staticmethod
    def _after_answer_validation(state: AgentWorkflowState) -> str:
        if state.get("validation_passed"):
            return "end"
        if state.get("retry_count", 0) < 1:
            return "repair"
        return "end"
