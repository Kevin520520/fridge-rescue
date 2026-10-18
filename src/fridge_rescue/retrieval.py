"""Fast deterministic retrieval over the curated recipe corpus."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from fridge_rescue.repository import ParquetRecipeRepository
from fridge_rescue.rule_checker import (
    find_excluded_matches,
    ingredient_terms_overlap,
    normalize_token,
)
from fridge_rescue.schemas import RecipeRecord, UserRequest
from fridge_rescue.settings import PROJECT_ROOT

DEFAULT_RULES_PATH = PROJECT_ROOT / "rules.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class RetrievalPolicy:
    """Configurable candidate-admission thresholds.

    Ranking is not a weighted score. Candidates that pass these thresholds are
    ordered lexicographically by ``_RankedRecipe.sort_key``.
    """

    minimum_coverage: float = 0.25
    max_missing_ingredients: int = 6

    def __post_init__(self) -> None:
        if isinstance(self.minimum_coverage, bool) or not isinstance(
            self.minimum_coverage, int | float
        ):
            raise TypeError("minimum_coverage must be a number")
        if not 0 < float(self.minimum_coverage) <= 1:
            raise ValueError("minimum_coverage must be greater than 0 and at most 1")
        if isinstance(self.max_missing_ingredients, bool) or not isinstance(
            self.max_missing_ingredients, int
        ):
            raise TypeError("max_missing_ingredients must be an integer")
        if self.max_missing_ingredients < 0:
            raise ValueError("max_missing_ingredients must be non-negative")

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> RetrievalPolicy:
        """Load thresholds from the ``retrieval`` section of project config."""

        config_path = Path(path)
        with config_path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Config file must contain a mapping: {config_path}")

        retrieval = payload.get("retrieval", {})
        if not isinstance(retrieval, dict):
            raise TypeError("retrieval config must be a mapping")
        return cls(
            minimum_coverage=retrieval.get("minimum_coverage", cls.minimum_coverage),
            max_missing_ingredients=retrieval.get(
                "max_missing_ingredients", cls.max_missing_ingredients
            ),
        )


class CorpusReader(Protocol):
    """Minimum corpus access needed by the offline retriever."""

    def list_all(self) -> list[RecipeRecord]: ...


@dataclass(frozen=True)
class RetrievalRules:
    """Reviewed normalization and safety rules used during retrieval."""

    pantry_staples: tuple[str, ...] = ()
    ingredient_aliases: Mapping[str, str] | None = None
    allergen_aliases: Mapping[str, list[str]] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_RULES_PATH) -> RetrievalRules:
        rules_path = Path(path)
        with rules_path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Rules file must contain a mapping: {rules_path}")

        pantry = payload.get("pantry_staples", [])
        ingredient_aliases = payload.get("ingredient_aliases", {})
        allergen_aliases = payload.get("allergen_aliases", {})
        if not isinstance(pantry, list):
            raise TypeError("pantry_staples must be a list")
        if not isinstance(ingredient_aliases, dict):
            raise TypeError("ingredient_aliases must be a mapping")
        if not isinstance(allergen_aliases, dict):
            raise TypeError("allergen_aliases must be a mapping")

        return cls(
            pantry_staples=tuple(str(item) for item in pantry),
            ingredient_aliases={str(key): str(value) for key, value in ingredient_aliases.items()},
            allergen_aliases={
                str(key): [str(alias) for alias in aliases]
                for key, aliases in allergen_aliases.items()
                if isinstance(aliases, list)
            },
        )


@dataclass(frozen=True)
class _RankedRecipe:
    recipe_id: str
    input_utilization: float
    matched_input_count: int
    coverage: float
    missing_count: int
    quality: float
    rating: float
    review_count: int

    @property
    def sort_key(
        self,
    ) -> tuple[float, int, int, float, float, float, int, tuple[int, int | str]]:
        """Rank inventory use first, then shopping burden and corpus quality."""

        numeric_id = int(self.recipe_id) if self.recipe_id.isdigit() else None
        stable_id: tuple[int, int | str] = (
            (0, numeric_id) if numeric_id is not None else (1, self.recipe_id)
        )
        return (
            -self.input_utilization,
            -self.matched_input_count,
            self.missing_count,
            -self.coverage,
            -self.quality,
            -self.rating,
            -self.review_count,
            stable_id,
        )


def _canonical_term(value: str, aliases: Mapping[str, str]) -> str:
    normalized = normalize_token(value)
    exact_match = aliases.get(normalized)
    if exact_match is not None:
        return exact_match

    # Source data often adds preparation words around a reviewed alias, such
    # as "diced tomatoes". Replace only complete words/phrases so aliases can
    # still canonicalize that ingredient without making egg match eggplant.
    for source, target in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, normalized)
    return normalized


def _terms_overlap(first: str, second: str) -> bool:
    """Allow reviewed canonical terms to match a more specific ingredient phrase."""

    return ingredient_terms_overlap(first, second)


@dataclass(frozen=True)
class IngredientMatchDetails:
    """Explain how a recipe uses the inventory supplied in one request."""

    recipe_coverage: float
    missing_count: int
    input_utilization: float
    matched_input_count: int
    matched_inputs: tuple[str, ...]
    unmatched_inputs: tuple[str, ...]
    missing_recipe_ingredients: tuple[str, ...]


def ingredient_match_details(
    recipe: RecipeRecord,
    available_ingredients: Sequence[str],
    pantry_staples: Sequence[str] = (),
    aliases: Mapping[str, str] | None = None,
) -> IngredientMatchDetails:
    """Return both recipe-side coverage and user-input utilization details."""

    normalized_aliases = aliases or {}
    available = list(
        dict.fromkeys(_canonical_term(item, normalized_aliases) for item in available_ingredients)
    )
    pantry = [_canonical_term(item, normalized_aliases) for item in pantry_staples]
    required = [_canonical_term(item, normalized_aliases) for item in recipe.ingredients_normalized]
    non_pantry = [
        ingredient
        for ingredient in required
        if not any(_terms_overlap(ingredient, staple) for staple in pantry)
    ]

    target_inputs = [
        available_item
        for available_item in available
        if not any(_terms_overlap(available_item, staple) for staple in pantry)
    ] or available
    matched_inputs = tuple(
        available_item
        for available_item in target_inputs
        if any(_terms_overlap(available_item, ingredient) for ingredient in non_pantry)
    )
    unmatched_inputs = tuple(
        available_item for available_item in target_inputs if available_item not in matched_inputs
    )
    missing_recipe_ingredients = tuple(
        dict.fromkeys(
            ingredient
            for ingredient in non_pantry
            if not any(_terms_overlap(ingredient, available_item) for available_item in available)
        )
    )

    if not non_pantry:
        recipe_coverage = 1.0
        missing_count = 0
    else:
        matched_recipe_count = len(non_pantry) - sum(
            not any(_terms_overlap(ingredient, available_item) for available_item in available)
            for ingredient in non_pantry
        )
        recipe_coverage = matched_recipe_count / len(non_pantry)
        missing_count = len(non_pantry) - matched_recipe_count
    input_utilization = len(matched_inputs) / len(target_inputs) if target_inputs else 0.0
    return IngredientMatchDetails(
        recipe_coverage=recipe_coverage,
        missing_count=missing_count,
        input_utilization=input_utilization,
        matched_input_count=len(matched_inputs),
        matched_inputs=matched_inputs,
        unmatched_inputs=unmatched_inputs,
        missing_recipe_ingredients=missing_recipe_ingredients,
    )


def _ingredient_metrics(
    recipe: RecipeRecord,
    available_ingredients: Sequence[str],
    pantry_staples: Sequence[str],
    aliases: Mapping[str, str],
) -> tuple[float, int, float, int]:
    details = ingredient_match_details(
        recipe,
        available_ingredients,
        pantry_staples,
        aliases,
    )
    return (
        details.recipe_coverage,
        details.missing_count,
        details.input_utilization,
        details.matched_input_count,
    )


def _quality_metrics(recipe: RecipeRecord) -> tuple[float, float, int]:
    rating = recipe.rating if recipe.rating is not None else 0.0
    review_count = recipe.review_count if recipe.review_count is not None else 0
    rating_score = max(0.0, min(rating / 5.0, 1.0))
    review_score = min(math.log1p(review_count) / math.log1p(100), 1.0)
    return 0.7 * rating_score + 0.3 * review_score, rating, review_count


class DeterministicRecipeRetriever:
    """Filter hard constraints, then rank recipes using corpus facts only.

    Candidates first have to meet the configured minimum ingredient coverage
    and maximum missing-ingredient count. The survivors are ranked by this
    exact lexicographic order: user-input utilization descending, matched user
    ingredient count descending, missing count ascending, recipe-side
    ingredient coverage descending, quality descending, rating descending, review
    count descending, and recipe ID ascending. No embedding or semantic-
    similarity score is used. The final ID tie-breaker makes identical calls
    reproducible across processes and machines.
    """

    def __init__(
        self,
        repository: CorpusReader,
        *,
        rules: RetrievalRules | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self.repository = repository
        self.rules = rules or RetrievalRules()
        self.policy = policy or RetrievalPolicy()
        self._ingredient_aliases = {
            normalize_token(str(key)): normalize_token(str(value))
            for key, value in (self.rules.ingredient_aliases or {}).items()
        }
        self._allergen_aliases = {
            normalize_token(str(key)): [normalize_token(alias) for alias in aliases]
            for key, aliases in (self.rules.allergen_aliases or {}).items()
        }

    @classmethod
    def from_local_corpus(
        cls,
        corpus_path: str | Path | None = None,
        *,
        rules_path: str | Path = DEFAULT_RULES_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> DeterministicRecipeRetriever:
        """Build the offline retriever from the corpus, rules, and project config."""

        repository = (
            ParquetRecipeRepository(corpus_path)
            if corpus_path is not None
            else ParquetRecipeRepository()
        )
        return cls(
            repository,
            rules=RetrievalRules.from_yaml(rules_path),
            policy=RetrievalPolicy.from_yaml(config_path),
        )

    def _meets_hard_constraints(self, request: UserRequest, recipe: RecipeRecord) -> bool:
        if request.max_time_minutes is not None:
            if recipe.total_time_minutes is None:
                return False
            if recipe.total_time_minutes > request.max_time_minutes:
                return False

        if recipe.equipment_tags_inferred & request.unavailable_equipment:
            return False

        if find_excluded_matches(
            recipe.ingredients_normalized,
            request.excluded_ingredients,
            self._allergen_aliases,
        ):
            return False

        if request.diet_preference == "vegetarian" and recipe.vegetarian_status != "match":
            return False
        return not (request.diet_preference == "vegan" and recipe.vegan_status != "match")

    def retrieve(self, request: UserRequest, *, limit: int) -> list[str]:
        """Return ranked recipe IDs compatible with ``workflow.RecipeRetriever``."""

        if limit < 1:
            raise ValueError("limit must be at least 1")

        ranked: list[_RankedRecipe] = []
        for recipe in self.repository.list_all():
            if not self._meets_hard_constraints(request, recipe):
                continue
            coverage, missing_count, input_utilization, matched_input_count = _ingredient_metrics(
                recipe,
                request.available_ingredients,
                self.rules.pantry_staples,
                self._ingredient_aliases,
            )
            if coverage < self.policy.minimum_coverage:
                continue
            if missing_count > self.policy.max_missing_ingredients:
                continue
            quality, rating, review_count = _quality_metrics(recipe)
            ranked.append(
                _RankedRecipe(
                    recipe_id=recipe.recipe_id,
                    input_utilization=input_utilization,
                    matched_input_count=matched_input_count,
                    coverage=coverage,
                    missing_count=missing_count,
                    quality=quality,
                    rating=rating,
                    review_count=review_count,
                )
            )

        ranked.sort(key=lambda item: item.sort_key)
        return [item.recipe_id for item in ranked[:limit]]
