"""A corpus-backed ingredient graph with three deterministic, read-only query tools.

Edges are ALIAS_OF (rules.yaml), USED_IN (original corpus ingredient lines), and
CAN_REPLACE (reviewed, directional and conditional rules). Co-occurrence is a
two-hop ingredient -> recipe <- ingredient traversal, not an invented similarity.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fridge_rescue.retrieval import ingredient_normalizer, ingredient_terms_overlap
from fridge_rescue.rule_checker import normalize_token
from fridge_rescue.schemas import RecipeRecord


class IngredientPath(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_ingredient: str
    corpus_ingredient: str
    source_line: str


class RecipeGraphEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recipe_id: str
    source_record_id: str
    paths: list[IngredientPath] = Field(default_factory=list)


class CompanionMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ingredient: str
    recipe_count: int
    evidence_recipe_id: str
    source_record_id: str


class SubstitutionMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    original: str
    substitute: str
    allowed_contexts: list[str]
    warning: str
    source: str = "rules.yaml / substitutions"


class RecipeIngredientGraph:
    """Build once from 2,000 authoritative records; store no user data in the graph."""

    def __init__(
        self, recipes: Sequence[RecipeRecord], *,
        ingredient_aliases: Mapping[str, str] | None = None,
        substitutions: Mapping[str, Any] | None = None,
        pantry_staples: Sequence[str] = (),
    ) -> None:
        self.aliases = {
            normalize_token(key): normalize_token(value)
            for key, value in (ingredient_aliases or {}).items()
        }
        self._canonicalize = ingredient_normalizer(self.aliases)
        entity_names = sorted(set(self.aliases.values()), key=lambda term: (-len(term), term))
        self._entity_pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(term) for term in entity_names if term) + r")(?!\w)"
        ) if entity_names else None
        self._recipes = {recipe.recipe_id: recipe for recipe in recipes}
        self._ingredient_recipes: dict[str, set[str]] = defaultdict(set)
        self._recipe_ingredients: dict[str, tuple[str, ...]] = {}
        self._source_lines: dict[tuple[str, str], str] = {}
        self._pantry = {self.canonical(term) for term in pantry_staples}
        for recipe in recipes:
            ingredients = []
            for normalized, source in zip(
                recipe.ingredients_normalized, recipe.ingredient_lines_original, strict=True
            ):
                ingredient = self.canonical(normalized)
                ingredients.append(ingredient)
                self._ingredient_recipes[ingredient].add(recipe.recipe_id)
                self._source_lines.setdefault((recipe.recipe_id, ingredient), source)
            self._recipe_ingredients[recipe.recipe_id] = tuple(dict.fromkeys(ingredients))
        self._substitutions: list[SubstitutionMatch] = []
        for original, rule in (substitutions or {}).items():
            if not isinstance(rule, Mapping):
                continue
            substitute = str(rule.get("substitute", "")).strip()
            contexts = rule.get("allowed_contexts", [])
            if not substitute or not isinstance(contexts, list) or not contexts:
                continue
            self._substitutions.append(SubstitutionMatch(
                original=self.canonical(original), substitute=self.canonical(substitute),
                allowed_contexts=[str(context) for context in contexts],
                warning=str(rule.get("warning", "")),
            ))

    def canonical(self, ingredient: str) -> str:
        return self._canonicalize(ingredient)

    def _entities(self, source_term: str) -> set[str]:
        """Group quantities/forms by known vocabulary for meaningful co-occurrence counts."""

        entities = {
            match.group(0) for match in self._entity_pattern.finditer(source_term)
            if ingredient_terms_overlap(match.group(0), source_term)
        } if self._entity_pattern is not None else set()
        return entities or {source_term}

    @property
    def stats(self) -> dict[str, int]:
        return {
            "recipes": len(self._recipes), "ingredients": len(self._ingredient_recipes),
            "aliases": len(self.aliases),
            "ingredient_edges": sum(len(ids) for ids in self._ingredient_recipes.values()),
            "substitutions": len(self._substitutions),
        }

    def _matching_terms(self, ingredient: str) -> tuple[str, ...]:
        canonical = self.canonical(ingredient)
        if not canonical:
            return ()
        return tuple(sorted(
            term for term in self._ingredient_recipes
            if ingredient_terms_overlap(canonical, term)
        ))

    def find_recipe_ids(
        self, ingredients: Sequence[str], *, require_all: bool = False
    ) -> list[str]:
        """Tool 1: traverse aliases/ingredient nodes to recipe IDs, in stable order."""

        matches = []
        for ingredient in dict.fromkeys(ingredients):
            ids: set[str] = set()
            for term in self._matching_terms(ingredient):
                ids.update(self._ingredient_recipes[term])
            matches.append(ids)
        if not matches:
            return []
        found = set.intersection(*matches) if require_all else set.union(*matches)
        return sorted(found, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))

    def explain(
        self, recipe_ids: Sequence[str], ingredients: Sequence[str], *, limit_per_recipe: int = 8
    ) -> list[RecipeGraphEvidence]:
        """Return source lines for bounded ingredient -> recipe evidence paths."""

        evidence = []
        for recipe_id in recipe_ids:
            recipe = self._recipes.get(recipe_id)
            if recipe is None:
                continue
            paths = []
            present = set(self._recipe_ingredients[recipe_id])
            for query in dict.fromkeys(ingredients):
                for term in self._matching_terms(query):
                    if term in present:
                        paths.append(IngredientPath(
                            query_ingredient=query, corpus_ingredient=term,
                            source_line=self._source_lines[(recipe_id, term)],
                        ))
            evidence.append(RecipeGraphEvidence(
                recipe_id=recipe_id, source_record_id=recipe.source_record_id,
                paths=paths[:limit_per_recipe],
            ))
        return evidence

    def find_companions(self, ingredient: str, *, limit: int = 8) -> list[CompanionMatch]:
        """Tool 2: count shared recipes for each two-hop co-occurring ingredient."""

        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        source_terms = set(self._matching_terms(ingredient))
        counts: Counter[str] = Counter()
        first_recipe: dict[str, str] = {}
        for recipe_id in self.find_recipe_ids([ingredient]):
            entities = {
                entity for term in self._recipe_ingredients[recipe_id]
                if term not in source_terms for entity in self._entities(term)
                if entity not in self._pantry
                and not ingredient_terms_overlap(self.canonical(ingredient), entity)
            }
            for entity in entities:
                counts[entity] += 1
                first_recipe.setdefault(entity, recipe_id)
        return [
            CompanionMatch(
                ingredient=term, recipe_count=counts[term],
                evidence_recipe_id=first_recipe[term],
                source_record_id=self._recipes[first_recipe[term]].source_record_id,
            )
            for term in sorted(counts, key=lambda term: (-counts[term], term))[:limit]
        ]

    def find_substitutes(self, ingredient: str) -> list[SubstitutionMatch]:
        """Tool 3: traverse one reviewed directional substitution edge with its conditions."""

        canonical = self.canonical(ingredient)
        return [
            rule.model_copy(deep=True) for rule in self._substitutions
            if canonical and canonical == rule.original
        ]
