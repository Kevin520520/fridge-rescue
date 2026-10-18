"""Shared deterministic helpers used by the Epicurious curation pipeline."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from fridge_rescue.rule_checker import contains_any_phrase, normalize_token

DietStatus = Literal["match", "conflict", "unknown"]

_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("beverage", ("beverage", "drink", "smoothie", "cocktail")),
    ("dessert", ("dessert", "cake", "cookie", "pie", "candy", "ice cream")),
    ("breakfast", ("breakfast", "brunch", "pancake", "waffle")),
    ("soup", ("soup", "stew", "chowder")),
    ("salad", ("salad",)),
    ("sauce_condiment", ("sauce", "dressing", "spread", "condiment")),
    ("bread_baking", ("bread", "muffin", "biscuit", "yeast")),
    ("grain_pasta", ("rice", "pasta", "noodle", "grain")),
    ("vegetable_legume", ("vegetable", "bean", "lentil", "legume")),
    (
        "main_protein",
        ("main dish", "chicken", "beef", "pork", "seafood", "poultry", "meat"),
    ),
    ("snack_appetizer", ("snack", "appetizer", "side dish")),
)

_PROTEIN_TITLE_MARKERS = (
    "chicken",
    "beef",
    "pork",
    "sausage",
    "shrimp",
    "prawn",
    "lamb",
    "fish",
    "salmon",
    "tuna",
    "turkey",
    "meat",
    "bacon",
)


def clean_text(value: object) -> str:
    """Decode HTML entities and normalize whitespace deterministically."""

    if value is None:
        return ""
    return " ".join(html.unescape(str(value)).strip().split())


def _mask_phrases(text: str, phrases: Sequence[str]) -> str:
    masked = normalize_token(text)
    for phrase in phrases:
        normalized = normalize_token(phrase)
        if not normalized:
            continue
        masked = re.sub(rf"(?<!\w){re.escape(normalized)}(?!\w)", " ", masked)
    return " ".join(masked.split())


def infer_equipment(
    text_parts: Sequence[str],
    rules: Mapping[str, Sequence[str]],
    exclusion_phrases: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Infer reviewed equipment tags from recipe text."""

    searchable = " ".join(text_parts)
    exclusions = exclusion_phrases or {}
    inferred: list[str] = []
    for equipment, phrases in rules.items():
        equipment_text = _mask_phrases(searchable, exclusions.get(equipment, ()))
        if contains_any_phrase(equipment_text, phrases):
            inferred.append(equipment)
    return sorted(inferred)


def scope_exclusion_reason(
    text_parts: Sequence[str], markers: Mapping[str, Sequence[str]]
) -> str | None:
    """Return a reviewed out-of-scope reason when a marker matches."""

    searchable = " ".join(text_parts)
    for scope, phrases in markers.items():
        if contains_any_phrase(searchable, phrases):
            safe_scope = re.sub(r"[^a-z0-9]+", "_", normalize_token(scope)).strip("_")
            return f"out_of_scope_{safe_scope or 'other'}"
    return None


def infer_diet_status(
    ingredients: Sequence[str],
    keywords: Sequence[str],
    preference: Literal["vegetarian", "vegan"],
    conflict_markers: Mapping[str, Sequence[str]],
) -> DietStatus:
    """Infer only reviewed vegetarian or vegan states from source text."""

    markers = conflict_markers.get(preference, ())
    if contains_any_phrase(" ".join(ingredients), markers):
        return "conflict"

    normalized_keywords = {normalize_token(keyword) for keyword in keywords}
    if preference == "vegan" and any("vegan" in keyword for keyword in normalized_keywords):
        return "match"
    if preference == "vegetarian" and any(
        "vegetarian" in keyword or "vegan" in keyword for keyword in normalized_keywords
    ):
        return "match"
    return "unknown"


def broad_category(title: str, category: str, keywords: Sequence[str]) -> str:
    """Map source categories and reviewed text markers to broad strata."""

    normalized_title = normalize_token(title)
    if "rendered" in normalized_title and "fat" in normalized_title:
        return "sauce_condiment"
    if contains_any_phrase(normalized_title, _PROTEIN_TITLE_MARKERS):
        return "main_protein"

    normalized_category = normalize_token(category)
    for group, phrases in _CATEGORY_RULES:
        if any(phrase in normalized_category for phrase in phrases):
            return group

    searchable = " ".join([title, *keywords])
    for group, phrases in _CATEGORY_RULES:
        if contains_any_phrase(searchable, phrases):
            return group
    return "other"
