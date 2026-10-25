"""Explainable runtime labels for recipes whose source omits cooking metadata.

The original corpus and its hashes stay reproducible. Duration is a planning estimate,
stored separately from source total time. Diet labels require a recognized ingredient
on every line and reject animal markers/ambiguous prepared products first.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from fridge_rescue.schemas import RecipeRecord

COOKING_METADATA_VERSION = "cooking-tools-v2-bilingual"


def _phrases(values: str) -> re.Pattern[str]:
    terms = sorted((item.strip() for item in values.split("|")), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(item) for item in terms) + r")(?:s|es)?(?!\w)", re.I)


_ANIMAL = _phrases(
    "beef|chicken|pork|bacon|ham|turkey|lamb|mutton|veal|venison|duck|goose|rabbit|"
    "quail|pheasant|meat|poultry|sausage|chorizo|prosciutto|pancetta|salami|pepperoni|"
    "pastrami|mortadella|lard|suet|tallow|gelatin|gelatine|fish|seafood|salmon|tuna|"
    "cod|halibut|haddock|trout|sole|sardine|anchovy|shrimp|prawn|crab|lobster|"
    "clam|mussel|oyster|scallop|squid|octopus|calamari|bonito|caviar|roe|surimi|"
    "fish sauce|oyster sauce|worcestershire|dashi"
)
_DAIRY_EGG = _phrases(
    "egg|milk|buttermilk|butter|ghee|cheese|cream|yogurt|yoghurt|kefir|whey|casein|"
    "honey|mozzarella|parmesan|parmigiano|pecorino|cheddar|feta|ricotta|mascarpone|"
    "gruyere|gruyère|gouda|brie|burrata|halloumi|half-and-half"
)
_PLANT_EXCEPTIONS = _phrases(
    "coconut milk|coconut cream|coconut butter|almond milk|soy milk|soya milk|oat milk|"
    "rice milk|cashew milk|peanut butter|almond butter|cashew butter|apple butter|"
    "cocoa butter|cream of tartar|cream of coconut|butter lettuce|butter bean"
)
_KNOWN_PLANT = _phrases(
    "tomato|potato|onion|garlic|shallot|scallion|leek|carrot|celery|spinach|kale|"
    "lettuce|cabbage|broccoli|cauliflower|asparagus|zucchini|courgette|squash|pumpkin|"
    "cucumber|eggplant|aubergine|mushroom|artichoke|radish|turnip|beet|fennel|"
    "pepper|chile|chili|chilli|jalapeno|jalapeño|avocado|corn|bean|chickpea|lentil|"
    "pea|tofu|tempeh|edamame|soybean|rice|pasta|spaghetti|linguine|fettuccine|penne|"
    "macaroni|noodle|couscous|quinoa|bulgur|barley|oat|wheat|flour|cornmeal|starch|"
    "bread|breadcrumb|panko|tortilla|lemon|lime|orange|grapefruit|apple|pear|peach|"
    "plum|apricot|cherry|cherries|berry|berries|grape|raisin|currant|cranberry|fig|"
    "date|prune|banana|pineapple|mango|papaya|melon|watermelon|coconut|nut|almond|"
    "walnut|pecan|hazelnut|pistachio|cashew|peanut|pine nut|sesame|sunflower|seed|"
    "tahini|flax|chia|oil|olive|vinegar|wine|beer|cider|sugar|syrup|molasses|agave|"
    "salt|water|ice|baking powder|baking soda|yeast|cocoa|dark chocolate|coffee|tea|"
    "vanilla|cinnamon|nutmeg|clove|ginger|turmeric|cumin|coriander|paprika|saffron|"
    "sumac|cardamom|allspice|cayenne|mustard|basil|parsley|cilantro|oregano|thyme|"
    "rosemary|sage|dill|mint|tarragon|chive|marjoram|bay leaf|bay leaves|capers|"
    "soy sauce|tamari|miso|hoisin|ketchup|tomato paste|tomato sauce|salsa|jam|jelly"
)
_AMBIGUOUS_PREPARED = _phrases(
    "stock|broth|bouillon|soup|gravy|stuffing|filling|dressing|pesto|marshmallow|"
    "shortening|margarine|cake mix|pancake mix|prepared meal|ravioli|tortellini"
)
_PLANT_PREPARED = _phrases("vegetable stock|vegetable broth|vegetarian broth|vegan broth|mushroom broth")


def ingredient_diet_status(ingredients: list[str], *, vegan: bool) -> Literal["match", "conflict", "unknown"]:
    known = True
    for line in ingredients:
        # Plant dairy names are exceptions to both the conflict check and recognition.
        plant_variant = bool(_PLANT_EXCEPTIONS.search(line))
        text = _PLANT_EXCEPTIONS.sub(" plant alternative ", line.casefold())
        if _ANIMAL.search(text) or (vegan and _DAIRY_EGG.search(text)):
            return "conflict"
        prepared_variant = bool(_PLANT_PREPARED.search(text))
        text = _PLANT_PREPARED.sub(" vegetable ", text)
        if _AMBIGUOUS_PREPARED.search(text):
            known = False
        if not (plant_variant or prepared_variant or _KNOWN_PLANT.search(text)
                or (not vegan and _DAIRY_EGG.search(text))):
            known = False
    return "match" if known and ingredients else "unknown"


_DURATION = re.compile(
    r"(?<![\w.])(?P<low>\d+(?:\.\d+)?)(?:\s*(?:to|[-–—]|�+C)\s*"
    r"(?P<high>\d+(?:\.\d+)?))?\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
    re.I,
)
_NUMBER_WORDS = {"a": "1", "an": "1", "one": "1", "two": "2", "three": "3", "four": "4",
                 "five": "5", "ten": "10", "fifteen": "15", "twenty": "20", "thirty": "30",
                 "half": "0.5"}
_COOKING = _phrases("cook|bake|roast|fry|simmer|boil|broil|grill|steam|saute|sauté|microwave|chill|freeze|marinate|rest|cool|rise")
_OPTIONAL_STORAGE = re.compile(r"\b(?:can be (?:made|prepared|stored|kept)|keeps? for|make ahead|leftovers?|up to \d+ days? ahead)\b", re.I)


def estimate_cooking_minutes(instructions: list[str]) -> tuple[int | None, list[str]]:
    """Sum stated upper durations plus disclosed preparation/untimed-step allowances."""

    if not instructions:
        return None, []
    minutes = 0.0
    untimed_cooking = 0
    evidence = []
    for step in instructions:
        sentences = re.split(r"(?<=[.!?])\s+", step)
        mandatory = " ".join(sentence for sentence in sentences if not _OPTIONAL_STORAGE.search(sentence))
        text = mandatory.casefold()
        text = re.sub(
            r"(?<![\w/])(?:(\d+)\s+)?(\d+)/(\d+)(?![\w/])",
            lambda match: str(float(match.group(1) or 0) + int(match.group(2)) / max(1, int(match.group(3)))),
            text,
        )
        text = re.sub(
            r"\b(" + "|".join(_NUMBER_WORDS) + r")\s+(?=(?:second|minute|hour|day))",
            lambda match: _NUMBER_WORDS[match.group(1)] + " ", text,
        )
        matches = list(_DURATION.finditer(text))
        step_minutes = 0.0
        for match in matches:
            amount = float(match.group("high") or match.group("low"))
            unit = match.group("unit")
            multiplier = 1 / 60 if unit.startswith("sec") else 60 if unit.startswith(("hour", "hr")) else 1440 if unit.startswith("day") else 1
            amount *= multiplier
            if re.match(r"\s*(?:on |per )?(?:each|a) side", text[match.end():]):
                amount *= 2
            step_minutes += amount
        if "overnight" in text:
            step_minutes = max(step_minutes, 720)
        if matches or "overnight" in text:
            minutes += step_minutes
            evidence.append(
                f"Step duration: about {math.ceil(step_minutes)} min · 步骤时长约 {math.ceil(step_minutes)} 分钟：{mandatory[:220]}"
            )
        elif _COOKING.search(text):
            untimed_cooking += 1
    allowance = 10 + 2 * len(instructions) + 5 * untimed_cooking
    evidence.append(
        f"Preparation allowance: {allowance} min = 10 min prep + 2 min × {len(instructions)} steps "
        f"+ 5 min × {untimed_cooking} untimed cooking steps. / "
        f"操作预留 {allowance} 分钟：基础准备 10 + 每步 2 × {len(instructions)} + 未标时烹饪步骤 5 × {untimed_cooking}。"
    )
    return max(1, math.ceil(minutes + allowance)), evidence


def enrich_cooking_metadata(recipe: RecipeRecord) -> RecipeRecord:
    updates = {}
    if recipe.total_time_minutes is None:
        estimate, evidence = estimate_cooking_minutes(recipe.instructions_original)
        updates.update(estimated_time_minutes=estimate, time_estimate_basis=evidence)
    for field, vegan in (("vegetarian_status", False), ("vegan_status", True)):
        derived = ingredient_diet_status(recipe.ingredients_normalized, vegan=vegan)
        original = getattr(recipe, field)
        if (original == "unknown" and derived != "unknown") or (derived == "conflict" and original != "conflict"):
            updates[field] = derived
            updates["diet_status_source"] = "ingredient_rules"
    return recipe.model_copy(update=updates)
