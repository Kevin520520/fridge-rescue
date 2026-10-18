"""Fast deterministic retrieval over the curated recipe corpus."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml

from fridge_rescue.repository import ParquetRecipeRepository
from fridge_rescue.rule_checker import (
    NON_EQUIVALENT_INGREDIENT_PRODUCTS,
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
    rescue_minimum_coverage: float | None = None
    rescue_max_missing_ingredients: int | None = None

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
        rescue_values = (
            self.rescue_minimum_coverage,
            self.rescue_max_missing_ingredients,
        )
        if (rescue_values[0] is None) != (rescue_values[1] is None):
            raise ValueError("Both rescue thresholds must be supplied together")
        if self.rescue_minimum_coverage is not None:
            if isinstance(self.rescue_minimum_coverage, bool) or not isinstance(
                self.rescue_minimum_coverage, int | float
            ):
                raise TypeError("rescue_minimum_coverage must be a number")
            if not 0 < float(self.rescue_minimum_coverage) <= self.minimum_coverage:
                raise ValueError(
                    "rescue_minimum_coverage must be greater than 0 and no higher "
                    "than minimum_coverage"
                )
            if isinstance(self.rescue_max_missing_ingredients, bool) or not isinstance(
                self.rescue_max_missing_ingredients, int
            ):
                raise TypeError("rescue_max_missing_ingredients must be an integer")
            if self.rescue_max_missing_ingredients < self.max_missing_ingredients:
                raise ValueError(
                    "rescue_max_missing_ingredients must be at least "
                    "max_missing_ingredients"
                )

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
            rescue_minimum_coverage=retrieval.get("rescue_minimum_coverage"),
            rescue_max_missing_ingredients=retrieval.get(
                "rescue_max_missing_ingredients"
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


@dataclass(frozen=True)
class _PreparedRecipe:
    """Corpus record with query-independent text work completed once."""

    record: RecipeRecord
    source_ingredients: tuple[str, ...]
    canonical_ingredients: tuple[str, ...]
    non_pantry_ingredients: tuple[str, ...]


class _AliasCanonicalizer:
    """Canonicalize one term with one compiled alias expression and a value cache."""

    def __init__(self, alias_items: tuple[tuple[str, str], ...]) -> None:
        self.aliases = dict(alias_items)
        sources = sorted(self.aliases, key=len, reverse=True)
        self.pattern = (
            re.compile(
                rf"(?<!\w)(?:{'|'.join(re.escape(source) for source in sources)})(?!\w)"
            )
            if sources
            else None
        )
        self.cache: dict[str, str] = {}

    def canonicalize(self, value: str) -> str:
        normalized = normalize_token(value)
        cached = self.cache.get(normalized)
        if cached is not None:
            return cached

        exact_match = self.aliases.get(normalized)
        if exact_match is not None:
            canonical = exact_match
        elif self.pattern is None:
            canonical = normalized
        else:
            canonical = self.pattern.sub(
                lambda match: self.aliases[match.group(0)],
                normalized,
            )
        self.cache[normalized] = canonical
        return canonical


def _normalized_alias_items(aliases: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (normalize_token(str(source)), normalize_token(str(target)))
            for source, target in aliases.items()
        )
    )


@lru_cache(maxsize=32)
def _cached_canonicalizer(
    alias_items: tuple[tuple[str, str], ...],
) -> _AliasCanonicalizer:
    return _AliasCanonicalizer(alias_items)


def _canonical_term(value: str, aliases: Mapping[str, str]) -> str:
    return _cached_canonicalizer(_normalized_alias_items(aliases)).canonicalize(value)


def _is_word_character(value: str) -> bool:
    r"""Mirror the ``\w`` boundary used by the previous regular expressions."""

    return value == "_" or value.isalnum()


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    """Find a normalized phrase on complete-word boundaries without compiling regexes."""

    if not phrase:
        return False

    start = 0
    while True:
        position = text.find(phrase, start)
        if position < 0:
            return False
        end = position + len(phrase)
        left_is_clear = position == 0 or not _is_word_character(text[position - 1])
        right_is_clear = end == len(text) or not _is_word_character(text[end])
        if left_is_clear and right_is_clear:
            return True
        start = position + 1


def _normalized_terms_are_non_equivalent(first: str, second: str) -> bool:
    for general, products in NON_EQUIVALENT_INGREDIENT_PRODUCTS.items():
        if first == general and any(
            _contains_normalized_phrase(second, product) for product in products
        ):
            return True
        if second == general and any(
            _contains_normalized_phrase(first, product) for product in products
        ):
            return True
    return False


def _normalized_terms_overlap(first: str, second: str) -> bool:
    if _normalized_terms_are_non_equivalent(first, second):
        return False
    return _contains_normalized_phrase(first, second) or _contains_normalized_phrase(
        second, first
    )


def _terms_overlap(first: str, second: str) -> bool:
    """Allow reviewed canonical terms to match a more specific ingredient phrase."""

    return _normalized_terms_overlap(normalize_token(first), normalize_token(second))


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


def _match_details_from_canonical_terms(
    required: Sequence[str],
    available: Sequence[str],
    pantry: Sequence[str],
) -> IngredientMatchDetails:
    non_pantry = [
        ingredient
        for ingredient in required
        if not any(_normalized_terms_overlap(ingredient, staple) for staple in pantry)
    ]
    return _match_details_for_non_pantry(
        non_pantry,
        available,
        available,
        input_required=required,
    )


def _match_details_for_non_pantry(
    non_pantry: Sequence[str],
    available: Sequence[str],
    target_inputs: Sequence[str],
    *,
    input_required: Sequence[str] | None = None,
) -> IngredientMatchDetails:
    """Compute request-specific metrics for an already prepared ingredient list."""

    match_matrix = [
        [
            _normalized_terms_overlap(ingredient, available_item)
            for available_item in available
        ]
        for ingredient in non_pantry
    ]
    recipe_matches = [any(matches) for matches in match_matrix]
    input_match_matrix = [
        [
            _normalized_terms_overlap(ingredient, available_item)
            for available_item in available
        ]
        for ingredient in (input_required if input_required is not None else non_pantry)
    ]
    input_matches = [
        any(matches[input_index] for matches in input_match_matrix)
        for input_index in range(len(available))
    ]
    available_match_map = dict(zip(available, input_matches, strict=True))
    matched_inputs = tuple(
        available_item
        for available_item in target_inputs
        if available_match_map[available_item]
    )
    unmatched_inputs = tuple(
        available_item
        for available_item in target_inputs
        if not available_match_map[available_item]
    )
    missing_recipe_ingredients = tuple(
        dict.fromkeys(
            ingredient
            for ingredient, matched in zip(non_pantry, recipe_matches, strict=True)
            if not matched
        )
    )

    if not non_pantry:
        recipe_coverage = 1.0
        missing_count = 0
    else:
        matched_recipe_count = sum(recipe_matches)
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


def ingredient_match_details(
    recipe: RecipeRecord,
    available_ingredients: Sequence[str],
    pantry_staples: Sequence[str] = (),
    aliases: Mapping[str, str] | None = None,
) -> IngredientMatchDetails:
    """Return both recipe-side coverage and user-input utilization details."""

    canonicalizer = _cached_canonicalizer(_normalized_alias_items(aliases or {}))
    available = list(
        dict.fromkeys(canonicalizer.canonicalize(item) for item in available_ingredients)
    )
    pantry = [canonicalizer.canonicalize(item) for item in pantry_staples]
    required = [
        canonicalizer.canonicalize(item) for item in recipe.ingredients_normalized
    ]
    return _match_details_from_canonical_terms(
        required,
        available,
        pantry,
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
        alias_items = _normalized_alias_items(self.rules.ingredient_aliases or {})
        self._ingredient_aliases = dict(alias_items)
        self._canonicalizer = _cached_canonicalizer(alias_items)
        self._pantry_staples = tuple(
            self._canonicalizer.canonicalize(item) for item in self.rules.pantry_staples
        )
        self._allergen_aliases = {
            normalize_token(str(key)): [normalize_token(alias) for alias in aliases]
            for key, aliases in (self.rules.allergen_aliases or {}).items()
        }
        self._prepared_recipes = tuple(
            self._prepare_recipe(recipe) for recipe in self.repository.list_all()
        )

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

    def _prepare_recipe(self, recipe: RecipeRecord) -> _PreparedRecipe:
        source_ingredients = tuple(
            normalize_token(ingredient) for ingredient in recipe.ingredients_normalized
        )
        canonical_ingredients = tuple(
            self._canonicalizer.canonicalize(ingredient) for ingredient in source_ingredients
        )
        non_pantry_ingredients = tuple(
            ingredient
            for ingredient in canonical_ingredients
            if not any(
                _normalized_terms_overlap(ingredient, staple)
                for staple in self._pantry_staples
            )
        )
        return _PreparedRecipe(
            record=recipe,
            source_ingredients=source_ingredients,
            canonical_ingredients=canonical_ingredients,
            non_pantry_ingredients=non_pantry_ingredients,
        )

    def _has_excluded_match(
        self,
        recipe: _PreparedRecipe,
        excluded_ingredients: Sequence[str],
    ) -> bool:
        for exclusion in excluded_ingredients:
            source_key = normalize_token(exclusion)
            key = self._canonicalizer.canonicalize(source_key)
            allergen_terms = [
                *self._allergen_aliases.get(source_key, []),
                *self._allergen_aliases.get(key, []),
            ]
            terms = {
                key,
                *(self._canonicalizer.canonicalize(term) for term in allergen_terms),
            }
            if any(
                _contains_normalized_phrase(ingredient, term)
                for ingredient in recipe.canonical_ingredients
                for term in terms
            ):
                return True
        return False

    def _meets_hard_constraints(
        self,
        request: UserRequest,
        prepared: _PreparedRecipe,
    ) -> bool:
        recipe = prepared.record
        if request.max_time_minutes is not None:
            if recipe.total_time_minutes is None:
                return False
            if recipe.total_time_minutes > request.max_time_minutes:
                return False

        if recipe.equipment_tags_inferred & request.unavailable_equipment:
            return False

        if self._has_excluded_match(
            prepared,
            request.excluded_ingredients,
        ):
            return False

        if request.diet_preference == "vegetarian" and recipe.vegetarian_status != "match":
            return False
        return not (request.diet_preference == "vegan" and recipe.vegan_status != "match")

    def retrieve(self, request: UserRequest, *, limit: int) -> list[str]:
        """Return ranked recipe IDs compatible with ``workflow.RecipeRetriever``."""

        if limit < 1:
            raise ValueError("limit must be at least 1")

        available = list(
            dict.fromkeys(
                self._canonicalizer.canonicalize(item)
                for item in request.available_ingredients
            )
        )
        target_inputs = available
        ranked: list[_RankedRecipe] = []
        rescue_ranked: list[_RankedRecipe] = []
        for prepared in self._prepared_recipes:
            recipe = prepared.record
            if not self._meets_hard_constraints(request, prepared):
                continue
            match = _match_details_for_non_pantry(
                prepared.non_pantry_ingredients,
                available,
                target_inputs,
                input_required=prepared.canonical_ingredients,
            )
            coverage = match.recipe_coverage
            missing_count = match.missing_count
            input_utilization = match.input_utilization
            matched_input_count = match.matched_input_count
            quality, rating, review_count = _quality_metrics(recipe)
            ranked_recipe = _RankedRecipe(
                recipe_id=recipe.recipe_id,
                input_utilization=input_utilization,
                matched_input_count=matched_input_count,
                coverage=coverage,
                missing_count=missing_count,
                quality=quality,
                rating=rating,
                review_count=review_count,
            )
            if (
                coverage >= self.policy.minimum_coverage
                and missing_count <= self.policy.max_missing_ingredients
            ):
                ranked.append(ranked_recipe)
                continue
            if (
                self.policy.rescue_minimum_coverage is not None
                and self.policy.rescue_max_missing_ingredients is not None
                and input_utilization == 1.0
                and coverage >= self.policy.rescue_minimum_coverage
                and missing_count <= self.policy.rescue_max_missing_ingredients
            ):
                rescue_ranked.append(ranked_recipe)

        if not any(item.input_utilization == 1.0 for item in ranked):
            ranked.extend(rescue_ranked)

        ranked.sort(key=lambda item: item.sort_key)
        return [item.recipe_id for item in ranked[:limit]]
