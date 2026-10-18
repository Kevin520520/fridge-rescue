"""Deterministic Food.com cleaning helpers used by the corpus builder."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from fridge_rescue.rule_checker import contains_any_phrase, normalize_token

DietStatus = Literal["match", "conflict", "unknown"]

_DURATION_PATTERN = re.compile(
    r"^P(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)

_LEADING_INGREDIENT_FRAGMENT = re.compile(
    r"^of (?=(?:fresh|dried|ground|chopped|minced|crushed|grated|sliced)\b)",
    re.IGNORECASE,
)
_REVERSED_INGREDIENT_FORM = re.compile(
    r"^(?P<base>[a-z][a-z\s-]*),\s*(?P<form>zest|juice|rind) of$",
    re.IGNORECASE,
)
_REQUIRED_WAIT_ACTION = re.compile(
    r"\b(?:marinat(?:e|ing)|marinad(?:e|ing)|refrigerat(?:e|ing)|chill(?:ing)?|"
    r"let\s+(?:it\s+)?stand|leave\s+(?:it\s+)?to\s+stand|"
    r"rest(?:ing)?|soak(?:ing)?|ris(?:e|ing)|proof(?:ing)?|ferment(?:ing)?|"
    r"cook(?:ing)?|simmer(?:ing)?|bak(?:e|ing)|roast(?:ing)?|boil(?:ing)?|"
    r"grill(?:ing)?|fry(?:ing)?|steam(?:ing)?)\b",
    re.IGNORECASE,
)
_OPTIONAL_OR_STORAGE_CONTEXT = re.compile(
    r"\b(?:up to|if desired|if you wish|optional(?:ly)?|for storage|"
    r"store|leftovers?|can be|may be)\b",
    re.IGNORECASE,
)
_DURATION_RANGE = re.compile(
    r"\b(?P<low>\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?\s*"
    r"(?P<unit>minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)
_WORD_DURATION_RANGE = re.compile(
    r"\b(?P<low>one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+to\s+"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+"
    r"(?P<unit>minutes?|hours?|days?)\b",
    re.IGNORECASE,
)
_NUMERIC_DURATION = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)

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
    ("main_protein", ("main dish", "chicken", "beef", "pork", "seafood", "poultry", "meat")),
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
_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
}


def parse_iso8601_minutes(value: object) -> int | None:
    """Parse the simple ISO-8601 durations used by Food.com into whole minutes."""

    if not isinstance(value, str) or not value.strip():
        return None
    match = _DURATION_PATTERN.fullmatch(value.strip().upper())
    if not match or not any(match.groupdict().values()):
        return None
    parts = {name: float(raw or 0) for name, raw in match.groupdict().items()}
    total_seconds = (
        parts["days"] * 86_400 + parts["hours"] * 3_600 + parts["minutes"] * 60 + parts["seconds"]
    )
    return round(total_seconds / 60)


def normalize_recipe_id(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        return None
    return str(int(numeric))


def clean_text(value: object) -> str:
    """Decode source HTML entities and normalize whitespace deterministically."""

    if value is None:
        return ""
    return " ".join(html.unescape(str(value)).strip().split())


def clean_text_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    cleaned: list[str] = []
    for item in value:
        if item is None:
            continue
        text = clean_text(item)
        if text:
            cleaned.append(text)
    return cleaned


def clean_instructions(value: object) -> list[str]:
    """Clean instructions and repair a narrow single-letter split artifact."""

    cleaned = clean_text_list(value)
    repaired: list[str] = []
    for instruction in cleaned:
        if (
            repaired
            and re.search(r"\b[A-Za-z]\.$", repaired[-1])
            and re.match(r"^[a-z]", instruction)
        ):
            repaired[-1] = repaired[-1][:-1] + instruction
        else:
            repaired.append(instruction)
    return repaired


def normalize_ingredients(ingredients: Sequence[str], aliases: Mapping[str, str]) -> list[str]:
    normalized: list[str] = []
    alias_lookup = {normalize_token(key): normalize_token(value) for key, value in aliases.items()}
    for ingredient in ingredients:
        value = normalize_token(clean_text(ingredient)).strip(" ,.;")
        value = _LEADING_INGREDIENT_FRAGMENT.sub("", value)
        reversed_form = _REVERSED_INGREDIENT_FORM.fullmatch(value)
        if reversed_form:
            value = f"{reversed_form.group('base')} {reversed_form.group('form')}"
        value = alias_lookup.get(value, value)
        if value and value not in normalized:
            normalized.append(value)
    return normalized


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
    """Return a high-precision out-of-scope reason, if a reviewed phrase matches."""

    searchable = " ".join(text_parts)
    for scope, phrases in markers.items():
        if contains_any_phrase(searchable, phrases):
            safe_scope = re.sub(r"[^a-z0-9]+", "_", normalize_token(scope)).strip("_")
            return f"out_of_scope_{safe_scope or 'other'}"
    return None


def _duration_to_minutes(value: float, unit: str) -> int:
    normalized = unit.casefold()
    if normalized.startswith(("hour", "hr")):
        value *= 60
    elif normalized.startswith("day"):
        value *= 1440
    return round(value)


def required_wait_duration_minutes(instruction: str) -> int | None:
    """Extract a conservative mandatory wait duration from one instruction."""

    text = clean_text(instruction)
    if not _REQUIRED_WAIT_ACTION.search(text) or _OPTIONAL_OR_STORAGE_CONTEXT.search(text):
        return None

    normalized = normalize_token(text)
    if re.search(r"\bovernight\b", normalized):
        return 480
    if re.search(r"\bseveral hours?\b", normalized):
        return 120
    if re.search(r"\bhalf (?:an|and) hour\b", normalized):
        return 30
    if re.search(r"\b(?:an|one) hour\b", normalized):
        return 60

    range_match = _DURATION_RANGE.search(normalized)
    if range_match:
        return _duration_to_minutes(float(range_match.group("low")), range_match.group("unit"))
    word_range_match = _WORD_DURATION_RANGE.search(normalized)
    if word_range_match:
        return _duration_to_minutes(
            _WORD_NUMBERS[word_range_match.group("low").casefold()],
            word_range_match.group("unit"),
        )
    numeric_match = _NUMERIC_DURATION.search(normalized)
    if numeric_match:
        return _duration_to_minutes(
            float(numeric_match.group("value")), numeric_match.group("unit")
        )
    return None


def infer_diet_status(
    ingredients: Sequence[str],
    keywords: Sequence[str],
    preference: Literal["vegetarian", "vegan"],
    conflict_markers: Mapping[str, Sequence[str]],
) -> DietStatus:
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


def time_bucket(total_minutes: int) -> str:
    if total_minutes <= 15:
        return "00-15"
    if total_minutes <= 30:
        return "16-30"
    if total_minutes <= 60:
        return "31-60"
    return "61-1440"


def deterministic_priority(recipe_id: str, seed: int) -> int:
    payload = f"{seed}:{recipe_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def record_hash(record: Mapping[str, Any]) -> str:
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_dedup_hash(title: str, ingredients: Sequence[str], instructions: Sequence[str]) -> str:
    payload = {
        "title": normalize_token(title),
        "ingredients": sorted(normalize_token(item) for item in ingredients),
        "instructions": [normalize_token(item) for item in instructions],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_curated_record(
    row: Mapping[str, Any],
    *,
    aliases: Mapping[str, str],
    equipment_rules: Mapping[str, Sequence[str]],
    diet_conflict_markers: Mapping[str, Sequence[str]],
    dataset_version: int,
    max_total_minutes: int = 1440,
    minimum_review_count: int = 0,
    require_aligned_quantities: bool = False,
    scope_exclusion_markers: Mapping[str, Sequence[str]] | None = None,
    equipment_exclusion_phrases: Mapping[str, Sequence[str]] | None = None,
    reviewed_source_exclusions: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Turn one source row into a curated artifact or a named rejection reason."""

    recipe_id = normalize_recipe_id(row.get("RecipeId"))
    if recipe_id is None:
        return None, "invalid_recipe_id"

    reviewed_reason = (reviewed_source_exclusions or {}).get(recipe_id)
    if reviewed_reason:
        safe_reason = re.sub(
            r"[^a-z0-9]+", "_", normalize_token(str(reviewed_reason))
        ).strip("_")
        return None, f"reviewed_source_exclusion_{safe_reason or 'quality'}"

    review_raw = row.get("ReviewCount")
    review_count = (
        int(review_raw)
        if isinstance(review_raw, int | float)
        and math.isfinite(float(review_raw))
        and review_raw >= 0
        else 0
    )
    if review_count < minimum_review_count:
        return None, "insufficient_independent_reviews"

    title = clean_text(row.get("Name"))
    if not title:
        return None, "missing_title"

    ingredients_original = clean_text_list(row.get("RecipeIngredientParts"))
    ingredients_normalized = normalize_ingredients(ingredients_original, aliases)
    if len(ingredients_normalized) < 3:
        return None, "fewer_than_3_ingredients"

    instructions = clean_instructions(row.get("RecipeInstructions"))
    if len(instructions) < 2:
        return None, "fewer_than_2_instructions"

    prep_minutes = parse_iso8601_minutes(row.get("PrepTime"))
    cook_minutes = parse_iso8601_minutes(row.get("CookTime"))
    total_minutes = parse_iso8601_minutes(row.get("TotalTime"))
    if total_minutes is None or not 1 <= total_minutes <= max_total_minutes:
        return None, "invalid_total_time"

    keywords = clean_text_list(row.get("Keywords"))
    category = clean_text(row.get("RecipeCategory"))
    description = clean_text(row.get("Description"))
    if not category:
        return None, "missing_category"
    if not description:
        return None, "missing_description"
    scope_reason = scope_exclusion_reason(
        [title, category, *keywords], scope_exclusion_markers or {}
    )
    if scope_reason:
        return None, scope_reason

    quantities_raw = row.get("RecipeIngredientQuantities")
    quantities = (
        [clean_text(item) for item in quantities_raw]
        if isinstance(quantities_raw, Sequence) and not isinstance(quantities_raw, str | bytes)
        else []
    )
    quantity_alignment_status = (
        "aligned"
        if len(quantities) == len(ingredients_original) and all(quantities)
        else "unreliable"
    )
    if require_aligned_quantities and quantity_alignment_status != "aligned":
        return None, "quantity_alignment_unreliable"

    longest_required_wait = max(
        (required_wait_duration_minutes(item) or 0 for item in instructions),
        default=0,
    )
    if longest_required_wait > total_minutes:
        return None, "instruction_duration_exceeds_total"

    equipment = infer_equipment(
        instructions,
        equipment_rules,
        equipment_exclusion_phrases,
    )
    vegetarian_status = infer_diet_status(
        ingredients_normalized, keywords, "vegetarian", diet_conflict_markers
    )
    vegan_status = infer_diet_status(
        ingredients_normalized, keywords, "vegan", diet_conflict_markers
    )
    group = broad_category(title, category, keywords)
    bucket = time_bucket(total_minutes)

    quality_flags: list[str] = []
    if quantity_alignment_status == "unreliable":
        quality_flags.append("quantity_alignment_unreliable")
    if vegetarian_status == "unknown":
        quality_flags.append("vegetarian_unknown")
    if vegan_status == "unknown":
        quality_flags.append("vegan_unknown")

    rating_raw = row.get("AggregatedRating")
    rating = (
        float(rating_raw)
        if isinstance(rating_raw, int | float) and math.isfinite(float(rating_raw))
        else None
    )
    servings_raw = row.get("RecipeServings")
    servings = (
        int(servings_raw)
        if isinstance(servings_raw, int | float)
        and math.isfinite(float(servings_raw))
        and servings_raw >= 0
        else None
    )

    record: dict[str, Any] = {
        "recipe_id": recipe_id,
        "title": title,
        "category_original": category,
        "category_group": group,
        "description_original": description,
        "keywords_original": keywords,
        "ingredients_original": ingredients_original,
        "ingredient_quantities_original": quantities,
        "quantity_alignment_status": quantity_alignment_status,
        "ingredients_normalized": ingredients_normalized,
        "instructions_original": instructions,
        "prep_time_minutes": prep_minutes,
        "cook_time_minutes": cook_minutes,
        "total_time_minutes_derived": total_minutes,
        "rating": rating,
        "review_count": review_count,
        "servings": servings,
        "yield_original": clean_text(row.get("RecipeYield")),
        "equipment_tags_inferred": equipment,
        "vegetarian_status_derived": vegetarian_status,
        "vegan_status_derived": vegan_status,
        "time_bucket": bucket,
        "source_dataset": "Food.com Recipes and Reviews",
        "source_record_id": recipe_id,
        "dataset_version": dataset_version,
        "license": "CC0-1.0",
        "quality_flags": quality_flags,
        "content_for_embedding": " | ".join([title, category, *keywords, *ingredients_normalized]),
    }
    record["dedup_hash"] = content_dedup_hash(title, ingredients_normalized, instructions)
    record["record_hash"] = record_hash(record)
    return record, None
