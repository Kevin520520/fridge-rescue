"""Small deterministic helpers used before and after LLM calls."""

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache

# A general ingredient name must not match a distinct processed product.
# Ordinary forms such as chicken breast or cooked rice still match their
# general ingredient.
NON_EQUIVALENT_INGREDIENT_PRODUCTS: dict[str, tuple[str, ...]] = {
    "chicken": (
        "chicken broth",
        "chicken stock",
        "chicken bouillon",
        "chicken base",
        "chicken flavor",
        "chicken flavored",
        "chicken flavour",
        "chicken flavoured",
        "chicken gravy",
    ),
    "beef": (
        "beef broth",
        "beef stock",
        "beef bouillon",
        "beef base",
        "beef flavor",
        "beef flavored",
        "beef flavour",
        "beef flavoured",
        "beef gravy",
    ),
    "rice": (
        "rice flour",
        "rice vinegar",
        "rice wine",
        "rice syrup",
        "rice milk",
        "rice mix",
        "rice noodles",
        "rice paper",
        "rice-a-roni",
        "rice bran",
        "rice cereal",
    ),
}


def normalize_token(value: str) -> str:
    return " ".join(value.lower().strip().split())


@lru_cache(maxsize=1024)
def _compiled_phrase_pattern(normalized_phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)")


def contains_phrase(text: str, phrase: str) -> bool:
    """Match a phrase on word boundaries, avoiding egg/eggplant mistakes."""

    normalized_text = normalize_token(text)
    normalized_phrase = normalize_token(phrase)
    return _compiled_phrase_pattern(normalized_phrase).search(normalized_text) is not None


def contains_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    """Match any configured phrase while normalizing the text only once."""

    normalized_text = normalize_token(text)
    return any(
        _compiled_phrase_pattern(normalize_token(phrase)).search(normalized_text) is not None
        for phrase in phrases
    )


@lru_cache(maxsize=32)
def _compiled_ingredient_aliases(
    items: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern[str] | None, dict[str, str]]:
    lookup = dict(items)
    if not lookup:
        return None, lookup
    alternatives = "|".join(
        re.escape(source) for source in sorted(lookup, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)"), lookup


def canonicalize_ingredient(
    value: str,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Apply configured ingredient aliases to a normalized term or full line."""

    normalized = normalize_token(value)
    items = tuple(
        sorted(
            (normalize_token(str(source)), normalize_token(str(target)))
            for source, target in (aliases or {}).items()
            if normalize_token(str(source)) and normalize_token(str(target))
        )
    )
    pattern, lookup = _compiled_ingredient_aliases(items)
    exact = lookup.get(normalized)
    if exact is not None:
        return exact
    if pattern is None:
        return normalized
    return pattern.sub(lambda match: lookup[match.group(0)], normalized)


def ingredient_terms_are_non_equivalent(first: str, second: str) -> bool:
    """Return true for a configured general-item/processed-product mismatch."""

    normalized_first = normalize_token(first)
    normalized_second = normalize_token(second)
    for general, products in NON_EQUIVALENT_INGREDIENT_PRODUCTS.items():
        if normalized_first == general and any(
            contains_phrase(normalized_second, product) for product in products
        ):
            return True
        if normalized_second == general and any(
            contains_phrase(normalized_first, product) for product in products
        ):
            return True
    return False


def ingredient_terms_overlap(first: str, second: str) -> bool:
    """Match ingredient phrases while rejecting configured false equivalences."""

    if ingredient_terms_are_non_equivalent(first, second):
        return False
    return contains_phrase(first, second) or contains_phrase(second, first)


def find_excluded_matches(
    recipe_ingredients: Iterable[str],
    excluded_ingredients: Iterable[str],
    allergen_aliases: dict[str, list[str]] | None = None,
    ingredient_aliases: Mapping[str, str] | None = None,
) -> set[str]:
    """Return exclusions whose exact normalized term or configured alias is present.

    This is a text risk screen, not a medical allergy guarantee.
    """

    haystack = [
        canonicalize_ingredient(item, ingredient_aliases) for item in recipe_ingredients
    ]
    alias_map = allergen_aliases or {}
    matches: set[str] = set()

    for exclusion in excluded_ingredients:
        key = normalize_token(exclusion)
        canonical_key = canonicalize_ingredient(key, ingredient_aliases)
        allergen_terms = [
            *alias_map.get(key, []),
            *alias_map.get(canonical_key, []),
        ]
        terms = {
            canonical_key,
            *(
                canonicalize_ingredient(alias, ingredient_aliases)
                for alias in allergen_terms
            ),
        }
        if any(contains_phrase(ingredient, term) for ingredient in haystack for term in terms):
            matches.add(key)

    return matches


def missing_ingredients(
    recipe_ingredients: Iterable[str],
    available_ingredients: Iterable[str],
    pantry_staples: Iterable[str] = (),
) -> list[str]:
    available = {normalize_token(item) for item in available_ingredients}
    pantry = {normalize_token(item) for item in pantry_staples}
    required = [normalize_token(item) for item in recipe_ingredients]
    return [item for item in required if item not in available and item not in pantry]
