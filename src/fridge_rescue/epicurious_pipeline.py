"""Deterministic parsing and curation helpers for the pinned Epicurious CSV."""

from __future__ import annotations

import ast
import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from typing import Any

from fridge_rescue.curation_common import (
    broad_category,
    clean_text,
    infer_diet_status,
    infer_equipment,
    scope_exclusion_reason,
)
from fridge_rescue.rule_checker import normalize_token

SOURCE_DATASET = "Epicurious Recipes (13k)"
SOURCE_REVISION = "c5a1e8dab69d46841d914be1f20cca6fbe4b6260"
SOURCE_DATASET_URL = (
    "https://raw.githubusercontent.com/josephrmartinez/recipe-dataset/"
    f"{SOURCE_REVISION}/13k-recipes.csv"
)
SOURCE_REPOSITORY_URL = "https://github.com/josephrmartinez/recipe-dataset"
SOURCE_FILE_NAME = "epicurious_recipes.csv"
SOURCE_FILE_SIZE_BYTES = 26_595_455
SOURCE_FILE_SHA256 = "d0e3977da7c1dc287ed75d6770f4064a7caeaa20ef6f64708b730e02f4d5f17a"
SOURCE_LICENSE = "CC BY-SA 3.0"
SOURCE_EXPECTED_ROWS = 13_501
CURATION_VERSION = "3.1"

REQUIRED_COLUMNS = frozenset(
    {"", "Title", "Ingredients", "Instructions", "Image_Name", "Cleaned_Ingredients"}
)

# These default phrases are intentionally narrow; project rules may add more.
DEFAULT_SCOPE_EXCLUSION_MARKERS: dict[str, tuple[str, ...]] = {
    "non_food": (
        "snowdrop",
        "snowdrops",
    ),
    "pet_food": (
        "dog food",
        "dog treats",
        "dog biscuits",
        "dog cookies",
        "cat food",
        "cat treats",
        "cat biscuits",
        "cat cookies",
        "pet food",
        "pet treats",
        "puppy treats",
        "kitten treats",
    ),
    "personal_care": (
        "face mask",
        "facial mask",
        "facial scrub",
        "body scrub",
        "skin scrub",
        "foot soak",
        "bath salts",
        "lip balm",
        "hand cream",
    ),
    "household": (
        "household cleaner",
        "cleaning solution",
        "insect repellent",
        "play dough",
        "modeling clay",
    ),
}

_MEASURED_PREFIX = re.compile(
    r"^(?:about\s+|approximately\s+|approx\.\s+)?(?:"
    r"\d|[\u00bc-\u00be\u2150-\u215e]|"
    r"a\b|an\b|one\b|two\b|three\b|four\b|five\b|six\b|"
    r"pinch\b|dash\b|handful\b|few\b|several\b)",
    re.IGNORECASE,
)
_QUALITATIVE_MEASUREMENT = re.compile(
    r"\b(?:to taste|as needed|as required|for serving|for garnish)\b",
    re.IGNORECASE,
)
_NON_INGREDIENT_EQUIPMENT_LINE = re.compile(r"^(?:special\s+)?equipment\s*:", re.IGNORECASE)
_SOURCE_MARKUP_RESIDUE = re.compile(r"</?[a-z][^>]*>|\bepi:recipelink\b", re.IGNORECASE)
_ENCODING_RESIDUE = re.compile(
    r"&(?:\d+|(?:deg|def)|[a-z]{4,});|[\u0080-\u009f\ufffd]",
    re.IGNORECASE,
)
_MAX_INGREDIENT_LINE_LENGTH = 300
_UNIT_WITHOUT_AMOUNT = re.compile(
    r"^\s*(?:cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|"
    r"pounds?|lbs?\.?|ounces?|oz\.?|grams?|g|kilograms?|kg|"
    r"millilit(?:er|re)s?|ml|lit(?:er|re)s?|pints?|quarts?|gallons?)\b",
    re.IGNORECASE,
)
_METADATA_OR_INSTRUCTION_PREFIX = re.compile(
    r"^\s*(?:ingredient\s+(?:info|tip)\s*:|\*?note\s*:|\*?available\b|"
    r"test[- ]kitchen\s+(?:tip|note)\s*:|cooks?[’']?\s+note\s*:|"
    r"shopping\s+(?:tip|note)\s*:|what\s+to\s+drink\s*:|you\s+will\s+need\b|"
    r"each\s+version\s+makes\b|preheat\b|heat\s+(?:the\s+)?oven\b|"
    r"pan\s+preparation\s*:|indirect\s+grilling\b|"
    r"line\s+(?:a|an|the)\b|use\s+(?:a|an|the)\b|"
    r"soften\s+about\b|oil\s+from\s+the\s+can\b|salmon\s+will\s+be\b|"
    r"you[’']ll\s+often\s+see\b|when\s+purchasing\b|even\s+easier\s*:|"
    r"find\s+\S+\s+at\b|choose\s+(?:a|an|the)\b|"
    r"buy\s+\S+\b|seek\s+out\b)",
    re.IGNORECASE,
)
_INSTRUCTION_METADATA_PREFIX = re.compile(
    r"^\s*(?:\*+\s*|preparation\s+time\s*:|test[- ]kitchen\s+tip\s*:|"
    r"this\s+recipe\s+is\s+made\s+available\b|"
    r"(?:recipe\s+)?source\s*:|courtesy\s+of\b)",
    re.IGNORECASE,
)
_PURCHASE_NOTE = re.compile(
    r"\b(?:is|are|can\s+be)\s+(?:available|sold|found)\b",
    re.IGNORECASE,
)
_SHOPPING_METADATA = re.compile(
    r"\b(?:can\s+be\s+ordered\s+online|"
    r"available\s+at\s+.{0,80}\b(?:markets?|stores?))\b",
    re.IGNORECASE,
)
_URL_OR_DOMAIN_METADATA = re.compile(
    r"(?:https?://|www\.|\b[a-z0-9][a-z0-9-]*\.(?:com|org|net)\b)",
    re.IGNORECASE,
)
_PRICED_WINE_RECOMMENDATION = re.compile(
    r"^\s*[A-Z].*\b(?:19|20)\d{2}\b.*\(\$\d+(?:\.\d{2})?\)\s*\.?\s*$"
)
_FOOTNOTE_LINE = re.compile(r"^\s*\*+\s*\S", re.IGNORECASE)
_PLACEHOLDER_INGREDIENT = re.compile(
    r"^\s*(?:n\s*/?\s*a|not\s+applicable|none)(?:\b|\s)",
    re.IGNORECASE,
)
_ORPHANED_FOOTNOTE_MARKER = re.compile(r"\*")
_ZERO_QUANTITY = re.compile(
    r"^\s*(?:about\s+|approximately\s+|approx\.?\s+)?0(?=\s|$)",
    re.IGNORECASE,
)
_CORRUPT_UNIT_TOKEN = re.compile(
    r"\b(?:tablespoonss|tablespoona|tablesepoon|tablepoons|teaspooon|"
    r"oounce|bupounds|cupsespagnole|cupspanko)\b",
    re.IGNORECASE,
)
_DUPLICATED_UNIT_TOKEN = re.compile(
    r"\b(cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|pounds?|lbs?\.?|"
    r"ounces?|oz\.?)\s+\1\b",
    re.IGNORECASE,
)
_CORRUPT_TEXT_TOKEN = re.compile(
    r"\b(?:aout|oldfashioned|coursely|andgizzard|chooped|caxn|bitesize|"
    r"mediumrare|mediumlow|thats)\b|\bpairing\s+knife\b|"
    r"\bturmeric\s+power\b|\byou\s+with\s+either\s+need\b|"
    r"\bmore\s+at\s+needed\b|\bgarlic\s+gloves\b|\bdivded\b|"
    r"\bdrained\s+kosher\s+salt\b|\+\s*,|"
    r"(?<=[A-Za-z])\s+,|(?<=[A-Za-z]),(?=[A-Za-z])",
    re.IGNORECASE,
)
_DANGLING_CONNECTOR = re.compile(r"\b(?:plus|or|and)\s*$", re.IGNORECASE)
_REPEATED_CONNECTOR_OR_PHRASE = re.compile(
    r"\b(?P<word>with|and|the|to|of|for|in)\s+(?P=word)\b|"
    r"\b(?P<phrase>can\s+be\s+made)\s+(?P=phrase)\b",
    re.IGNORECASE,
)
_MALFORMED_DECIMAL_UNIT = re.compile(
    r"\b\d+\.\s*(?:lb|lbs|oz|g|kg|ml|l)\b",
    re.IGNORECASE,
)
_SINGULAR_AMOUNT_PLURAL_UNIT = re.compile(
    r"^\s*(?:about\s+|approximately\s+|approx\.?\s+)?1\s+"
    r"(?:cups|tablespoons|teaspoons|pounds|lbs|ounces|grams|"
    r"kilograms|milliliters|liters|pints|quarts|gallons)\b",
    re.IGNORECASE,
)
_FRACTIONAL_AMOUNT_PLURAL_UNIT = re.compile(
    r"(?<!\d\s)(?<![\d/-])(?:1/2|1/3|2/3|1/4|3/4|1/8|3/8|5/8|7/8)\s+"
    r"(?:cups|tablespoons|teaspoons|pounds|lbs|ounces|grams|kilograms|"
    r"milliliters|liters|pints|quarts|gallons)\b",
    re.IGNORECASE,
)
_PLURAL_AMOUNT_SINGULAR_UNIT = re.compile(
    r"(?<![\d/])(?:[2-9]|\d{2,})(?:\s+\d+/\d+)?\s+"
    r"(?:cup|tablespoon|teaspoon|pound|ounce|gram|kilogram|"
    r"milliliter|liter|pint|quart|gallon)\b",
    re.IGNORECASE,
)
_MIXED_AMOUNT_SINGULAR_UNIT = re.compile(
    r"(?<![\d/])1\s+(?:1/2|1/3|2/3|1/4|3/4|1/8|3/8|5/8|7/8)\s+"
    r"(?:cup|tablespoon|teaspoon|pound|ounce|gram|kilogram|"
    r"milliliter|liter|pint|quart|gallon)\b",
    re.IGNORECASE,
)
_MALFORMED_FRACTION_SPACING = re.compile(r"\b\d+\s+/\s*\d+\b")
_HYPHENATED_PLURAL_UNIT = re.compile(
    r"\b\d+(?:\s+\d+/\d+|/\d+|\.\d+)?-(?:cups|tablespoons|teaspoons|"
    r"pounds|lbs|ounces|grams|kilograms|milliliters|liters|pints|quarts|"
    r"gallons)\b",
    re.IGNORECASE,
)
_UNIT_GLUED_TO_TITLE_WORD = re.compile(
    r"\b(?:cup|tablespoon|teaspoon|pound|ounce|gram|kilogram|milliliter|"
    r"liter|pint|quart|gallon)(?=[A-Z])"
)
_JOINED_PLUS_ENTRY = re.compile(r"\b\w+\+\s*\d", re.IGNORECASE)
_DUAL_MASS_CONVERSION = re.compile(
    r"(?P<amount_a>\d+(?:\s+\d+/\d+|/\d+|\.\d+)?)\s*"
    r"(?P<unit_a>ounces?|oz\.?|grams?|g)\s*/\s*"
    r"(?P<amount_b>\d+(?:\s+\d+/\d+|/\d+|\.\d+)?)\s*"
    r"(?P<unit_b>ounces?|oz\.?|grams?|g)\b",
    re.IGNORECASE,
)
_CUP_FLUID_OUNCE_CONVERSION = re.compile(
    r"(?P<cups>\d+(?:\s+\d+/\d+|/\d+|\.\d+)?)\s+cups?\b"
    r"[^()]{0,100}\(\s*(?P<fluid_ounces>\d+(?:\s+\d+/\d+|/\d+|\.\d+)?)\s+"
    r"(?:fl\.?\s*oz\.?|fluid\s+ounces?)\s*\)",
    re.IGNORECASE,
)
_GLUED_QUANTITY_TOKEN = re.compile(
    r"(?:^\s*\d+\s+\d{2,}\s*,|"
    r"^\s*\d+(?=(?:medium|large|small|cloves?|cups?|tablespoons?|teaspoons?|"
    r"ounces?|pounds?|grams?|kilograms?)\b)|"
    r"\d/\d(?=(?:cups?|tablespoons?|teaspoons?|ounces?|pounds?|grams?|"
    r"kilograms?)\b)|\)(?=\d)|\bplus(?=\d))",
    re.IGNORECASE,
)
_UNIT_LESS_ONE_SEASONING = re.compile(
    r"^\s*1\s+(?:sea\s+salt|freshly\s+ground\s+(?:black\s+)?pepper|"
    r"unsalted\s+butter|pure\s+maple\s+syrup)\s*$",
    re.IGNORECASE,
)
_INCOMPLETE_ALTERNATIVE_AMOUNT = re.compile(
    r"\bor\s+\d+(?:\s+\d+/\d+|/\d+|\.\d+)?\s+"
    r"(?:cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|pounds?|lbs?\.?|"
    r"ounces?|oz\.?)\s*$",
    re.IGNORECASE,
)
_CONCATENATED_INGREDIENT_ENTRY = re.compile(
    r"(?:\bdivided\s+[A-Z][A-Za-z'’-]+\s+for\b|"
    r"\(about\s+\d+(?:\s+\d+/\d+)?\s+cups?\)\s+"
    r"\d+(?:\s+\d+/\d+)?\s+(?:tablespoons?|teaspoons?|cups?|ounces?|pounds?)\b|"
    r"\)\s+\d+(?:\s+\d+/\d+|/\d+|\.\d+)?\s+"
    r"(?:tablespoons?|teaspoons?|cups?|ounces?|pounds?)\b|"
    r"\b(?:sliced|chopped|minced|divided|melted|softened|peeled)\s+"
    r"\d+(?:\s+\d+/\d+|/\d+|\.\d+)?\s+"
    r"(?:tablespoons?|teaspoons?|cups?|ounces?|pounds?)\b)",
    re.IGNORECASE,
)
_BROKEN_INGREDIENT_FRAGMENT = re.compile(
    r"^\s*(?:"
    r"\d+(?:\s+\d+/\d+)?\s+(?:flour|sugar|salt|pepper|oil|butter)|"
    r"\d+(?:\s+\d+/\d+)?\s+plus\s+more(?:\s+for\b.*)?|"
    r"plus\s+more\s+for\s+\w+(?:\s+\w+){0,3}|"
    r"halved\s+through\s+root|"
    r"plus\s+additional\s+in\s+case\s+some\s+tear|"
    r"\([a-z][a-z -]{1,30}\),\s*(?:peeled|chopped|sliced|cut)\b.*"
    r")\s*$",
    re.IGNORECASE,
)
_JOINED_MIXED_FRACTION = re.compile(
    r"(?<![\d/])(?P<whole>\d{1,2})(?P<numerator>[1-7])/(?P<denominator>[2348])"
    r"(?=(?:\s|[-‐‑‒–—])(?:"
    r"cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|pounds?|lbs?\.?|"
    r"ounces?|oz\.?|grams?|g\b|kilograms?|kg\b|millilit(?:er|re)s?|ml\b|"
    r"lit(?:er|re)s?|pints?|quarts?|gallons?|inches?|inch\b"
    r"|sticks?"
    r"))",
    re.IGNORECASE,
)
_DIRECT_SUBRECIPE_REFERENCE = re.compile(
    r"(?:\breserved\s+from\b|\bsee\s+(?:the\s+)?recipe\b|"
    r"\brecipe\s+(?:follows|below|above|available\s+at)\b|"
    r"^\s*(?:one|\d+(?:\s+\d+/\d+)?|\d+/\d+)\s+(?:"
    r"recipe(?:\s+of)?|batch(?:\s+of)?|quantity(?:\s+of)?|x\b.*\brecipe\b))",
    re.IGNORECASE,
)
_LOCAL_COMPONENT_POINTER = re.compile(
    r"^\s*the\s+[a-z][^()]{0,80}\(\s*see\s+above\s*\)\s*$",
    re.IGNORECASE,
)
_REFERENCE_AMOUNT_PREFIX = re.compile(
    r"^\s*(?:(?:about|approximately|approx\.)\s+)?(?:"
    r"\d+(?:\s+\d+/\d+|/\d+|\.\d+)?|"
    r"[\u00bc-\u00be\u2150-\u215e]|one|two|three|four|five|six|a|an"
    r")(?:\s*(?:-|to)\s*\d+(?:\s+\d+/\d+|/\d+|\.\d+)?)?\s*(?:"
    r"(?:cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|pounds?|lbs?\.?|"
    r"ounces?|oz\.?|grams?|g|kilograms?|kg|millilit(?:er|re)s?|ml|"
    r"lit(?:er|re)s?|pints?|quarts?|gallons?|bunch(?:es)?|cloves?|"
    r"slices?|cans?|packages?|envelopes?|pieces?|sprigs?|heads?|jars?|"
    r"bottles?|dozen)\b\.?\s*)?",
    re.IGNORECASE,
)
_REFERENCE_ADDITIONAL_AMOUNT_PREFIX = re.compile(
    r"^\s*(?:plus|and)\s+"
    r"(?:\d+(?:\s+\d+/\d+|/\d+|\.\d+)?|[¼-¾⅐-⅞])\s*"
    r"(?:cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|pounds?|lbs?\.?|"
    r"ounces?|oz\.?|grams?|g|kilograms?|kg|millilit(?:er|re)s?|ml|"
    r"lit(?:er|re)s?|pints?|quarts?|gallons?)\b\.?\s*",
    re.IGNORECASE,
)
_RETAIL_COMPONENT_ALTERNATIVE = re.compile(
    r"\b(?:store[- ]?bought|storebought|purchased|commercial(?:ly)?|"
    r"canned|bottled|ready[- ]made)\b",
    re.IGNORECASE,
)
_REFERENCE_TRAILING_DETAIL = re.compile(
    r"\s*(?:,|;|\(|\bor\b|\bat\s+room\s+temperature\b|"
    r"\bfor\s+(?:serving|drizzling|garnish)\b).*$",
    re.IGNORECASE,
)
_FOOD_QUANTITY_PREFIX = re.compile(
    r"^\s*(?:about\s+|approximately\s+)?(?:\d|[\u00bc-\u00be\u2150-\u215e]).{0,12}?\s"
    r"(?:cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|pounds?|lbs?\.?|"
    r"ounces?|oz\.?|grams?|g\b|kilograms?|kg\b|millilit(?:er|re)s?|ml\b|"
    r"lit(?:er|re)s?|pints?|quarts?|gallons?|bunch(?:es)?|cloves?|slices?|"
    r"cans?|packages?|envelopes?)\b",
    re.IGNORECASE,
)
_NON_RECIPE_TITLES = frozenset(
    {
        "dog biscuits",
        "napkin rings and a decorated vase from corrugated cardboard",
        "fabric-covered vase centerpiece",
        "robin's-egg place card",
        "harvest topiary",
        "to dye easter eggs",
    }
)

# Epicurious sometimes stores a final equipment/supply line inside the source
# ingredient list.  Only the first eight normalized words are inspected so a
# real ingredient such as "thyme, tied with kitchen twine" is retained while a
# standalone line such as "Kitchen twine" is removed.
_EQUIPMENT_PREFIX_MARKERS = re.compile(
    r"(?:"
    r"\bcheesecloth\b|"
    r"\b(?:kitchen|butcher s) (?:twine|string)\b|"
    r"\b(?:wooden|bamboo|metal) skewers?\b|"
    r"\b(?:toothpicks?|cocktail (?:picks?|sticks?))\b|"
    r"\b(?:parchment|wax(?:ed)?) paper\b|"
    r"\bplastic wrap\b|"
    r"\b(?:heavy duty )?(?:aluminum |aluminium )?foil\b|"
    r"\b(?:instant read|deep fry|candy) thermometer\b|"
    r"\b(?:deep fat|frying|oven|meat) thermometer\b|"
    r"\bice cream maker\b|"
    r"\b(?:cookie|biscuit) cutters?\b|"
    r"\bspringform pan\b|"
    r"\b(?:mini )?muffin (?:tins?|pans?|cups?)\b|"
    r"\b(?:coffee|spice) grinder\b|"
    r"\badjustable blade slicer\b|"
    r"\bstand mixer\b|"
    r"\belectric (?:hand )?mixer\b|"
    r"\bfood processor\b|"
    r"\bpasta machine\b|"
    r"\bpastry (?:bag|brush|blender|rings?|tip)\b|"
    r"\b(?:cake|bread|loaf|pie|tartlet) pans?\b|"
    r"\b(?:bundt|tube|tart|popover|madeleine) pans?\b|"
    r"\b(?:baking|cookie) sheets?\b|"
    r"\b(?:baking|souffle|gratin) dish\b|"
    r"\b(?:wire|cooling|flat|v) rack\b|"
    r"\brazor blade\b|"
    r"\bcedar plank\b|"
    r"\bpizza (?:stone|peel|oven attachment)\b|"
    r"\b(?:kitchen )?(?:blowtorch|torch)\b|"
    r"\bmortar and pestle\b|"
    r"\bnut grater\b|"
    r"\bwok with (?:a )?lid\b|"
    r"\b(?:large |heavy )?wok\b|"
    r"\bice pop sticks?\b"
    r"|\bround cutters?\b"
    r"|\bjuice extractor\b"
    r"|\b(?:flour|drum|fine mesh) sifters?\b"
    r"|\b(?:ring|ramekin|dariole|savarin|tim[b]?ale|popsicle|ice pop) molds?\b"
    r"|\b(?:julienne )?peeler\b"
    r"|\bbox grater\b"
    r"|\binstant pot\b"
    r"|\bheavy pot\b"
    r"|\broasting pan\b"
    r"|\bletters printed on paper\b"
    r")",
    re.IGNORECASE,
)

_EQUIPMENT_LINE_START = re.compile(
    r"^\s*(?:(?:special\s+)?equipment\s*:\s*|you\s+will\s+(?:also\s+)?need\s+)?"
    r"(?:(?:a|an|one|two|three|four|assorted)\s+|"
    r"\d+(?:\s+\d+/\d+)?(?:[-‐‑‒–—](?:qt|in(?:ch)?|cm|mm))?\s+)?"
    r"(?:\([^)]*\)\s*)?(?:(?:large|small|mini|heavy|metal|wooden|bamboo|"
    r"disposable|electric|nonstick|rimmed|shallow|ceramic|glass|fluted|round|"
    r"square|rectangular|adjustable|fine[- ]mesh|paper|or)\s+){0,6}"
    r"(?:layers?\s+)?(?:cheesecloth|kitchen\s+(?:twine|string)|"
    r"butcher[’']?s\s+(?:twine|string)|pliers|"
    r"skewers?|toothpicks?|cocktail\s+(?:picks?|sticks?)|parchment\s+paper|"
    r"wax(?:ed)?\s+paper|plastic\s+wrap|(?:aluminum|aluminium)\s+foil|"
    r"(?:deep[- ]fat|deep[- ]fry|candy|instant[- ]read|oven|meat)\s+thermometer|"
    r"ice[- ]cream\s+maker|(?:cookie|biscuit)\s+cutters?|springform\s+pan|"
    r"muffin\s+(?:tins?|pans?|cups?)|(?:coffee|spice)\s+(?:mill|grinder)|"
    r"food\s+processor|stand\s+mixer|electric\s+(?:hand\s+)?mixer|"
    r"pasta\s+machine|pastry\s+(?:bag|brush|blender|rings?|tip)|"
    r"(?:bundt|tube|tart|tartlet|cake|bread|loaf|pie|popover|madeleine)\s+pans?|"
    r"(?:baking|cookie)\s+sheets?|(?:baking|souffle|gratin)\s+dish|"
    r"(?:wire|cooling|flat|v)\s+rack|razor\s+blade|cedar\s+plank|"
    r"pizza\s+(?:stone|peel|oven\s+attachment)|(?:kitchen\s+)?(?:blowtorch|torch)|"
    r"mortar\s+and\s+pestle|nut\s+grater|wok\b|ice[- ]pop\s+sticks?|"
    r"round\s+cutters?|juice\s+extractor|(?:flour|drum)\s+sifters?|"
    r"(?:ring|ramekin|dariole|savarin|timbale|popsicle|ice[- ]pop)\s+molds?|"
    r"(?:fine[- ]mesh\s+)?sieves?|paper\s+muffin\s+cup\s+liners?|"
    r"julienne\s+peeler|box\s+grater|instant\s+pot|heavy\s+pot|roasting\s+pan|"
    r"letters\s+printed\s+on\s+paper)\b",
    re.IGNORECASE,
)

# Source-wide equipment screening is anchored to a configured equipment head
# noun.  The words before that head may describe only count, size, shape, or
# material.  This avoids treating food as equipment merely because a later
# preparation note names a tool (for example, thyme tied with kitchen twine).
_SOURCE_WIDE_EQUIPMENT_HEAD = re.compile(
    r"\b(?:"
    r"baking\s+dish(?:es)?|pressure\s+cookers?|instant\s+pots?|"
    r"waffle\s+(?:irons?|makers?)|dutch\s+ovens?|"
    r"casserole\s+dish(?:es)?|casseroles?|braisers?|"
    r"slow\s+cookers?|ice[- ]cube\s+trays?|"
    r"cookie\s+or\s+biscuit\s+cutters?|biscuit\s+or\s+cookie\s+cutters?|"
    r"(?:cookie|biscuit|melon[- ]ball|round|fluted|french[- ]fry)\s+cutters?|"
    r"cutters?|"
    r"pastry\s+(?:tips?|bags?|brush(?:es)?|blenders?|rings?)|"
    r"tube\s+or\s+bundt\s+pans?|"
    r"springform\s+(?:cake\s+)?pans?|"
    r"(?:tube|bundt|tart|tartlet|cake|loaf|bread|baking|pie|"
    r"popover|madeleine)\s+pans?|pie\s+(?:dish(?:es)?|plates?)|"
    r"muffin\s+(?:tins?|pans?)|baking\s+cups?|paper\s+muffin[- ]cup\s+liners?|"
    r"panettone\s+molds?|brioche\s+molds?|ice[- ]pop\s+molds?|ramekins?|"
    r"glasses,\s*ramekins,\s*or\s*cups|"
    r"(?:gratin|souffl[eé])\s+dish(?:es)?|stock\s+pots?|"
    r"heavy\s+(?:nonreactive\s+)?pots?|"
    r"roasting\s+pans?|saucepans?|skillets?|woks?|griddles?|comals?|"
    r"food\s+mills?|(?:herb|spice|coffee)\s+(?:mills?|grinders?)|"
    r"coffee/spice\s+grinders?|mortar\s+and\s+pestle|"
    r"hand[- ]held\s+whisks?|whisks?|stand\s+mixers?|mixers?|blenders?|"
    r"food\s+processors?|mini\s+processors?|processors?|mandolines?|"
    r"adjustable[- ]blade\s+slicers?|japanese\s+benriner|juice\s+extractors?|"
    r"(?:vegetable|julienne)\s+peelers?|(?:microplane|box|nut)\s+graters?|"
    r"graters?|funnels?|sifters?|sieves?|strainers?|spatulas?|fondant\s+rollers?|"
    r"(?:cooling|wire|flat|v)\s+racks?|"
    r"pastry\s+or\s+bench\s+scrapers?|(?:bench|pastry)\s+scrapers?|ricers?|"
    r"(?:deep[- ]fat|deep[- ]fry|candy|instant[- ]read|oven|meat)?\s*"
    r"thermometers?|skewers?|toothpicks?|cocktail\s+(?:picks?|sticks?)|"
    r"lollipop\s+sticks?|candy\s+sticks?|craft\s+sticks?|"
    r"ice[- ]pop\s+sticks?|popsicle\s+sticks?|cheesecloth|"
    r"kitchen\s+(?:twine|string)|butcher[’']?s\s+(?:twine|string)|"
    r"parchment\s+or\s+waxed\s+paper|parchment\s+paper|waxed\s+paper|"
    r"butcher\s+paper|plastic\s+wrap|"
    r"(?:heavy[- ]duty\s+)?(?:aluminum\s+|aluminium\s+)?foil|"
    r"foil[- ]wrapped\s+bricks?|paper\s+straws?|pie\s+weights?|"
    r"bakeware\s+liners?|green[- ]bean\s+frenchers?|"
    r"(?:shot|tall\s+narrow)\s+glass(?:es)?|"
    r"parfait\s+or\s+other\s+tall\s+narrow\s+glass(?:es)?|"
    r"jars?\s+or\s+containers?\s+with\s+lids?|heatproof\s+jars?|canning\s+jars?|"
    r"glass(?:es)?\s+or\s+ramekins?|"
    r"cedar\s+planks?|pizza[- ]oven\s+attachments?|pizza\s+pans?|"
    r"pizza\s+(?:baking\s+)?(?:stones?|peels?)|(?:kitchen\s+)?(?:blowtorches?|torches?)|"
    r"custard\s+cups?|measuring\s+cups?|terrines?|crocks?|cleavers?|"
    r"steamer\s+inserts?|offset\s+spatulas?|paintbrushes?|"
    r"pasta\s+(?:machines?|makers?)|ice[- ]cream\s+makers?|"
    r"kettle\s+grills?|chimney\s+starters?|grilling\s+mitts?|"
    r"(?:baking|cookie)\s+sheets?|cake\s+boards?|"
    r"carpet\s+or\s+upholstery\s+needles?|"
    r"shallow\s+casserole\s+or\s+baking\s+dish"
    r")\b",
    re.IGNORECASE,
)
_SOURCE_WIDE_EQUIPMENT_ALLOWED_PREFIX_WORDS = frozenset(
    """
    a an one two three four five six seven eight nine ten twelve thirty dozen
    about approximately assorted standard very good electric hand held handheld
    heavy duty bottomed large small mini medium size sized fine mesh metal wooden
    bamboo paper colored ceramic glass fluted round square squares rectangular
    shallow deep nonstick rimless rimmed cast iron ovenproof flameproof
    nonreactive disposable clean untreated old fashioned well seasoned japanese
    other adjustable blade v u shaped flexible thin freezer oven safe wire star
    pastry dry french dark plastic with sides diameter inch inches in cm mm qt y
    pint pints quart quarts cup cups ounce ounces oz l ml liter litre by x or and of layers
    layer not candy deep fry instant read soup sheets jumbo texas red crescent
    moon tree offset long unglazed wide flat foil bonbon to fat belgian charcoal
    table holiday diameter larger oz floured
    """.split()  # noqa: SIM905
)
_SOURCE_WIDE_EQUIPMENT_FOOD_WEIGHT_PREFIX = re.compile(
    r"^\s*\d+(?:\s+\d+/\d+)?\s+"
    r"(?:ounces?|oz\.?|pounds?|lbs?\.?|grams?|g|kg|kilograms?|"
    r"millilit(?:er|re)s?|ml|lit(?:er|re)s?)\b",
    re.IGNORECASE,
)
_SOURCE_WIDE_EQUIPMENT_SPECIAL_PREFIX = re.compile(
    r"^\s*special\s+(?:equipment|equpment)\s*:",
    re.IGNORECASE,
)
_SOURCE_WIDE_EQUIPMENT_ODDITIES = re.compile(
    r"^\s*(?:"
    r"(?:[^,;]{0,20}\s+)?letters\s+printed\s+on\s+paper\b|"
    r"\d+\s+tablespoons?\s+uncooked\s+rice\s+or\s+dried\s+beans\b|"
    r"\d+\s+l\s*\([^)]*\)\s*jar,\s*pounder\b|"
    r"(?:(?:a|an)|\d+(?:\s+\d+/\d+)?)\s+"
    r"(?:heatproof\s+)?(?:\d+(?:\s+\d+/\d+)?[- ](?:pint|quart|qt)\.?\s+)?"
    r"(?:glass\s+)?jars?\s*$|"
    r"(?:a\s+)?soda\s+syphon\b.*\b(?:seltzer|cream)\s+charges?\b|"
    r"\d+\s*\([^)]*quart[^)]*\)\s+dish\s+or\b|"
    r"kettle,\s*small\s+wide\s+bowl,\s*large\s+tray\b|"
    r"\d+(?:\s+\d+/\d+)?\s+cups?\s+.{0,80}?\b(?:hardwood|oak|mesquite|"
    r"hickory|pecan)\s+(?:wood\s+)?chips\b"
    r")",
    re.IGNORECASE,
)


def is_standalone_equipment_or_supply(line: str) -> bool:
    """Return whether a source ingredient line is only equipment or supplies."""

    normalized = (
        normalize_token(line)
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", " x ")
    )
    if re.match(
        r"^(?:pan juices|roasting pan with (?:pan )?juices)\b",
        normalized,
    ):
        return False
    if _SOURCE_WIDE_EQUIPMENT_FOOD_WEIGHT_PREFIX.search(normalized):
        return False
    if _SOURCE_WIDE_EQUIPMENT_SPECIAL_PREFIX.search(normalized):
        return True
    if normalized.startswith("the pan must be"):
        return True
    if _SOURCE_WIDE_EQUIPMENT_ODDITIES.search(normalized):
        return True

    normalized = re.sub(
        r"^(?:for assembling the [^:]{1,40}:|you will also need)\s*",
        "",
        normalized,
    )
    match = _SOURCE_WIDE_EQUIPMENT_HEAD.search(normalized)
    if match is None:
        return False

    prefix = normalized[: match.start()].replace("-", " ")
    prefix_tokens = re.findall(r"[a-z]+|\d+(?:/\d+)?", prefix)
    if len(prefix_tokens) > 15:
        return False
    return all(
        token in _SOURCE_WIDE_EQUIPMENT_ALLOWED_PREFIX_WORDS
        or re.fullmatch(r"\d+(?:/\d+)?", token) is not None
        for token in prefix_tokens
    )


def stable_hash(payload: Any) -> str:
    """Return a canonical SHA-256 digest for JSON-compatible content."""

    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_ingredient_lines(value: object) -> tuple[list[str] | None, str | None]:
    """Parse a Python-list string without evaluating arbitrary source code."""

    if not isinstance(value, str) or not value.strip():
        return None, "blank_ingredients"
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None, "invalid_ingredient_list"
    if not isinstance(parsed, list):
        return None, "ingredients_not_list"
    if any(not isinstance(item, str) for item in parsed):
        return None, "non_string_ingredient"
    lines = [html.unescape(item).strip() for item in parsed]
    if any(not item for item in lines):
        return None, "blank_ingredient_line"
    return lines, None


def split_instruction_steps(value: object) -> list[str]:
    """Split the source newline-delimited instructions into non-blank steps."""

    if not isinstance(value, str):
        return []
    return [
        html.unescape(line).strip()
        for line in value.splitlines()
        if line.strip()
    ]


def normalize_ingredient_lines(lines: Sequence[str]) -> list[str]:
    """Normalize full source lines while retaining every quantity and unit token."""

    return [normalize_token(line) for line in lines]


def has_exact_duplicated_text_fragment(line: str) -> bool:
    """Detect a source field made by joining the same text to itself."""

    text = line.strip()
    if len(text) < 24 or len(text) % 2:
        return False
    midpoint = len(text) // 2
    return text[:midpoint] == text[midpoint:]


def has_repeated_semicolon_segment(line: str) -> bool:
    """Return whether a semicolon list repeats one complete non-trivial segment."""

    content = re.sub(r"^\s*accompaniments?\s*:\s*", "", line, flags=re.IGNORECASE)
    segments = [normalize_token(part).strip(" .,:;") for part in content.split(";")]
    meaningful = [segment for segment in segments if len(segment) >= 4]
    return len(meaningful) != len(set(meaningful))


def _parse_amount(value: str) -> float:
    parts = value.strip().split()
    if len(parts) == 2:
        return float(Fraction(parts[0])) + float(Fraction(parts[1]))
    return float(Fraction(parts[0]))


def has_inconsistent_mass_conversion(line: str, *, tolerance: float = 0.20) -> bool:
    """Detect an oz/g pair whose two source amounts cannot be the same mass."""

    for match in _DUAL_MASS_CONVERSION.finditer(line):
        amount_a = _parse_amount(match.group("amount_a"))
        amount_b = _parse_amount(match.group("amount_b"))
        unit_a = match.group("unit_a").lower()
        unit_b = match.group("unit_b").lower()
        a_is_ounce = unit_a.startswith("o")
        b_is_ounce = unit_b.startswith("o")
        if a_is_ounce == b_is_ounce:
            continue
        grams_a = amount_a * 28.349523125 if a_is_ounce else amount_a
        grams_b = amount_b * 28.349523125 if b_is_ounce else amount_b
        if max(grams_a, grams_b) <= 0:
            continue
        relative_error = abs(grams_a - grams_b) / min(grams_a, grams_b)
        if relative_error > tolerance:
            return True
    return False


def has_inconsistent_volume_conversion(line: str, *, tolerance: float = 0.12) -> bool:
    """Detect a US cup/fluid-ounce pair that cannot represent one volume."""

    for match in _CUP_FLUID_OUNCE_CONVERSION.finditer(line):
        cups = _parse_amount(match.group("cups"))
        fluid_ounces = _parse_amount(match.group("fluid_ounces"))
        expected_fluid_ounces = cups * 8.0
        if min(expected_fluid_ounces, fluid_ounces) <= 0:
            continue
        relative_error = abs(expected_fluid_ounces - fluid_ounces) / min(
            expected_fluid_ounces,
            fluid_ounces,
        )
        if relative_error > tolerance:
            return True
    return False


def ingredient_line_rejection_reason(line: str) -> str | None:
    """Return a high-precision reason for rejecting one polluted source line.

    Any confirmed pollution invalidates the whole recipe row. This preserves a
    simple provenance rule for the released corpus: every retained ingredient
    line belongs to the source recipe and no questionable line was hidden.
    """

    if _SOURCE_MARKUP_RESIDUE.search(line):
        return "source_markup_residue"
    if _ENCODING_RESIDUE.search(line):
        return "encoding_residue"
    if line.count("(") != line.count(")"):
        return "unbalanced_parentheses"
    if _PLACEHOLDER_INGREDIENT.search(line):
        return "placeholder_ingredient"
    if _ORPHANED_FOOTNOTE_MARKER.search(line):
        return "orphaned_footnote_marker"
    if _ZERO_QUANTITY.search(line):
        return "zero_quantity_ingredient"
    if _CORRUPT_UNIT_TOKEN.search(line):
        return "corrupt_unit_token"
    if _DUPLICATED_UNIT_TOKEN.search(line):
        return "corrupt_unit_token"
    if has_inconsistent_mass_conversion(line):
        return "corrupt_quantity_token"
    if has_inconsistent_volume_conversion(line):
        return "corrupt_quantity_token"
    if _GLUED_QUANTITY_TOKEN.search(line):
        return "corrupt_quantity_token"
    if _UNIT_LESS_ONE_SEASONING.search(line):
        return "corrupt_quantity_token"
    if _INCOMPLETE_ALTERNATIVE_AMOUNT.search(line):
        return "broken_ingredient_fragment"
    if _DANGLING_CONNECTOR.search(line):
        return "broken_ingredient_fragment"
    if _MALFORMED_DECIMAL_UNIT.search(line):
        return "corrupt_quantity_token"
    if _SINGULAR_AMOUNT_PLURAL_UNIT.search(line):
        return "corrupt_quantity_token"
    if _FRACTIONAL_AMOUNT_PLURAL_UNIT.search(line):
        return "corrupt_quantity_token"
    if _PLURAL_AMOUNT_SINGULAR_UNIT.search(line):
        return "corrupt_quantity_token"
    if _MIXED_AMOUNT_SINGULAR_UNIT.search(line):
        return "corrupt_quantity_token"
    if _MALFORMED_FRACTION_SPACING.search(line):
        return "corrupt_quantity_token"
    if _HYPHENATED_PLURAL_UNIT.search(line):
        return "corrupt_quantity_token"
    if _UNIT_GLUED_TO_TITLE_WORD.search(line):
        return "corrupt_quantity_token"
    if _CORRUPT_TEXT_TOKEN.search(line):
        return "source_text_corruption"
    if _REPEATED_CONNECTOR_OR_PHRASE.search(line):
        return "source_text_corruption"
    if _JOINED_PLUS_ENTRY.search(line):
        return "concatenated_ingredient_entries"
    if _CONCATENATED_INGREDIENT_ENTRY.search(line):
        return "concatenated_ingredient_entries"
    if has_exact_duplicated_text_fragment(line):
        return "duplicated_text_fragment"
    if has_repeated_semicolon_segment(line):
        return "duplicated_text_fragment"
    if _BROKEN_INGREDIENT_FRAGMENT.search(line):
        return "broken_ingredient_fragment"
    if _UNIT_WITHOUT_AMOUNT.search(line):
        return "unit_without_amount"
    if len(line) > _MAX_INGREDIENT_LINE_LENGTH:
        return "metadata_or_narrative"
    if _METADATA_OR_INSTRUCTION_PREFIX.search(line):
        return "metadata_or_narrative"
    if _FOOTNOTE_LINE.search(line):
        return "metadata_or_narrative"
    # A complete measured ingredient may legitimately contain a parenthetical
    # shopping note (for example, "2 bay leaves ... if only dried leaves are
    # available").  A stray digit inside standalone purchase prose is not a
    # measurement, so the decision uses the same measured-prefix detector as
    # the coverage calculation instead of merely looking for any digit.
    if _PURCHASE_NOTE.search(line) and not _MEASURED_PREFIX.search(line):
        return "metadata_or_narrative"
    if _SHOPPING_METADATA.search(line):
        return "metadata_or_narrative"
    if _URL_OR_DOMAIN_METADATA.search(line):
        return "metadata_or_narrative"
    if _NON_INGREDIENT_EQUIPMENT_LINE.search(line):
        return "equipment_only"

    if is_standalone_equipment_or_supply(line):
        return "equipment_only"
    return None


def normalize_joined_mixed_fraction(line: str) -> tuple[str, int]:
    """Restore missing spaces in unambiguous mixed fractions next to a unit."""

    correction_count = 0

    def replacement(match: re.Match[str]) -> str:
        nonlocal correction_count
        numerator = int(match.group("numerator"))
        denominator = int(match.group("denominator"))
        if numerator >= denominator:
            return match.group(0)
        correction_count += 1
        return (
            f"{match.group('whole')} "
            f"{match.group('numerator')}/{match.group('denominator')}"
        )

    return _JOINED_MIXED_FRACTION.sub(replacement, line), correction_count


def normalize_recipe_reference(value: str) -> str:
    """Normalize punctuation variants used when one recipe names another."""

    normalized = normalize_token(value)
    normalized = re.sub(r"[‐‑‒–—−]", "-", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _named_component_candidate(line: str) -> str:
    """Return a conservative title candidate from a measured ingredient line.

    Epicurious often copied the title of a separately published component into
    another recipe's ingredient list (for example, ``4 Jammy Soft-Boiled
    Eggs``).  Quantity/unit prefixes and serving notes are not part of that
    title, so they are removed only for this cross-reference check.  The source
    ingredient line itself is never rewritten.
    """

    candidate = _REFERENCE_AMOUNT_PREFIX.sub("", line, count=1).strip()
    candidate = _REFERENCE_ADDITIONAL_AMOUNT_PREFIX.sub("", candidate, count=1).strip()
    candidate = _REFERENCE_TRAILING_DETAIL.sub("", candidate, count=1).strip()
    return candidate


def _has_distinctive_title_casing(value: str) -> bool:
    """Limit implicit title matching to source text formatted as a named item."""

    minor_words = {"a", "an", "and", "for", "in", "of", "or", "the", "to", "with"}
    words = re.findall(r"[^\W\d_][\w’'\-]*", value, flags=re.UNICODE)
    content_words = [word for word in words if word.casefold() not in minor_words]
    return len(content_words) >= 2 and all(word[0].isupper() for word in content_words)


def is_unexpanded_subrecipe_reference(
    line: str,
    known_recipe_titles: set[str] | frozenset[str],
) -> bool:
    """Detect source lines that depend on another recipe rather than ingredients."""

    normalized = normalize_recipe_reference(line)
    if not normalized:
        return False
    if _DIRECT_SUBRECIPE_REFERENCE.search(line):
        return True
    if _LOCAL_COMPONENT_POINTER.search(line):
        return True
    if not known_recipe_titles:
        return False
    if normalized in known_recipe_titles and len(normalized.split()) >= 2:
        return True

    # A bare ``see below`` is not sufficient evidence: some records expand the
    # component inside the same ingredient list and instructions (source 4249
    # is an example). Implicit links instead require a distinctive,
    # title-cased component name that exactly matches another source recipe.
    # Explicit retail alternatives remain usable inputs and are not rejected
    # merely because a similarly named recipe also exists in the corpus.
    if not _RETAIL_COMPONENT_ALTERNATIVE.search(line):
        candidate = _named_component_candidate(line)
        normalized_candidate = normalize_recipe_reference(candidate)
        if (
            _has_distinctive_title_casing(candidate)
            and len(normalized_candidate.split()) >= 2
            and normalized_candidate in known_recipe_titles
        ):
            return True

    reference_markers = (
        " reserved from ",
        " see recipe",
    )
    if any(marker in f" {normalized} " for marker in reference_markers):
        return any(
            len(title.split()) >= 2 and title in normalized
            for title in known_recipe_titles
        )

    leading_recipe = re.sub(
        r"^(?:one|\d+(?:\s+\d+/\d+)?|\d+/\d+)\s+recipe(?:\s+of)?\s+",
        "",
        normalized,
    )
    return (
        leading_recipe != normalized
        and leading_recipe in known_recipe_titles
        and len(leading_recipe.split()) >= 2
    )


@lru_cache(maxsize=8)
def _compiled_aliases(
    items: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern[str] | None, dict[str, str]]:
    lookup = dict(items)
    if not lookup:
        return None, lookup
    alternatives = "|".join(re.escape(source) for source in sorted(lookup, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)"), lookup


def _alias_lines_for_screening(
    lines: Sequence[str], aliases: Mapping[str, str] | None
) -> list[str]:
    """Apply configured aliases only to an internal inference copy of source lines."""

    normalized_aliases = tuple(
        sorted(
            (
                (normalize_token(source), normalize_token(target))
                for source, target in (aliases or {}).items()
                if normalize_token(source) and normalize_token(target)
            )
        )
    )
    pattern, lookup = _compiled_aliases(normalized_aliases)
    if pattern is None:
        return list(lines)
    return [pattern.sub(lambda match: lookup[match.group(0)], line) for line in lines]


def has_source_measurement(line: str) -> bool:
    """Detect an explicit or qualitative source measurement without inventing one."""

    text = line.strip()
    return bool(_MEASURED_PREFIX.search(text) or _QUALITATIVE_MEASUREMENT.search(text))


def measurement_coverage(lines: Sequence[str]) -> float:
    if not lines:
        return 0.0
    return sum(has_source_measurement(line) for line in lines) / len(lines)


def merge_scope_markers(
    configured: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    merged = {key: tuple(values) for key, values in DEFAULT_SCOPE_EXCLUSION_MARKERS.items()}
    for key, values in (configured or {}).items():
        existing = list(merged.get(key, ()))
        for value in values:
            phrase = clean_text(value)
            if phrase and phrase not in existing:
                existing.append(phrase)
        merged[key] = tuple(existing)
    return merged


def complexity_bucket(ingredient_count: int, instruction_count: int) -> str:
    """Create a coarse complexity stratum from fields actually present in the source."""

    if ingredient_count <= 6 and instruction_count <= 4:
        return "simple"
    if ingredient_count <= 12 and instruction_count <= 8:
        return "moderate"
    return "complex"


def source_row_hash(row: Mapping[str, Any]) -> str:
    """Hash all source CSV fields so a curated row can be traced to source bytes."""

    return stable_hash({name: row.get(name) for name in sorted(REQUIRED_COLUMNS)})


def content_dedup_hash(
    title: str, ingredient_lines: Sequence[str], instructions: Sequence[str]
) -> str:
    return stable_hash(
        {
            "title": normalize_token(title),
            "ingredient_lines": sorted(normalize_ingredient_lines(ingredient_lines)),
            "instructions": [normalize_token(step) for step in instructions],
        }
    )


def build_epicurious_record(
    row: Mapping[str, Any],
    *,
    source_file_sha256: str,
    ingredient_aliases: Mapping[str, str] | None = None,
    equipment_rules: Mapping[str, Sequence[str]] | None = None,
    equipment_exclusion_phrases: Mapping[str, Sequence[str]] | None = None,
    diet_conflict_markers: Mapping[str, Sequence[str]] | None = None,
    scope_exclusion_markers: Mapping[str, Sequence[str]] | None = None,
    reviewed_source_exclusions: Mapping[str, str] | Sequence[str | int] | None = None,
    minimum_ingredients: int = 3,
    minimum_instructions: int = 2,
    minimum_measurement_coverage: float = 0.0,
    known_recipe_titles: set[str] | frozenset[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one source row and build the Curation v3 artifact."""

    source_record_id = clean_text(row.get(""))
    if not source_record_id:
        return None, "missing_source_record_id"

    reviewed_exclusions = reviewed_source_exclusions or {}
    if isinstance(reviewed_exclusions, Mapping):
        normalized_exclusions = {
            str(key): str(value) for key, value in reviewed_exclusions.items()
        }
        reviewed_reason = normalized_exclusions.get(
            source_record_id
        ) or normalized_exclusions.get(f"epicurious-{source_record_id}")
    else:
        normalized_exclusion_ids = {str(value) for value in reviewed_exclusions}
        reviewed_reason = (
            "confirmed_adversarial_pollution"
            if source_record_id in normalized_exclusion_ids
            or f"epicurious-{source_record_id}" in normalized_exclusion_ids
            else None
        )
    if reviewed_reason:
        safe_reason = re.sub(r"[^a-z0-9]+", "_", normalize_token(str(reviewed_reason))).strip("_")
        return None, f"reviewed_source_exclusion_{safe_reason or 'quality'}"

    title = clean_text(row.get("Title"))
    if not title:
        return None, "missing_title"
    normalized_title = normalize_token(title)
    if normalized_title in _NON_RECIPE_TITLES:
        return None, "known_non_recipe_title"
    if _SOURCE_MARKUP_RESIDUE.search(title):
        return None, "source_markup_residue"
    if _ENCODING_RESIDUE.search(title):
        return None, "encoding_residue"
    if _REPEATED_CONNECTOR_OR_PHRASE.search(title):
        return None, "source_text_corruption"

    ingredient_lines, ingredient_error = parse_ingredient_lines(row.get("Ingredients"))
    if ingredient_error:
        return None, ingredient_error
    assert ingredient_lines is not None
    source_ingredient_line_count = len(ingredient_lines)
    retained_ingredient_lines: list[str] = []
    removed_non_ingredient_line_count = 0
    quantity_format_correction_count = 0
    normalized_source_titles = known_recipe_titles or frozenset()
    for line in ingredient_lines:
        rejection_reason = ingredient_line_rejection_reason(line)
        if rejection_reason in {
            "source_markup_residue",
            "encoding_residue",
            "unbalanced_parentheses",
            "placeholder_ingredient",
            "orphaned_footnote_marker",
            "zero_quantity_ingredient",
            "corrupt_unit_token",
            "corrupt_quantity_token",
            "source_text_corruption",
            "concatenated_ingredient_entries",
            "duplicated_text_fragment",
            "broken_ingredient_fragment",
            "unit_without_amount",
        }:
            return None, rejection_reason
        if is_unexpanded_subrecipe_reference(line, normalized_source_titles):
            return None, "unexpanded_subrecipe_reference"
        if rejection_reason in {"equipment_only", "metadata_or_narrative"}:
            return None, rejection_reason
        corrected_line, correction_count = normalize_joined_mixed_fraction(line)
        quantity_format_correction_count += correction_count
        retained_ingredient_lines.append(corrected_line)
    ingredient_lines = retained_ingredient_lines
    if len(ingredient_lines) < minimum_ingredients:
        return None, "fewer_than_minimum_ingredients"

    instructions = split_instruction_steps(row.get("Instructions"))
    if len(instructions) < minimum_instructions:
        return None, "fewer_than_minimum_instructions"
    normalized_steps = [normalize_token(step) for step in instructions]
    # Source 12179 intentionally repeats two procedural templates for two
    # different gelée components. Other exact duplicate steps are ingestion
    # artifacts and invalidate the record.
    if source_record_id != "12179" and len(normalized_steps) != len(set(normalized_steps)):
        return None, "duplicated_instruction_step"
    if any(_SOURCE_MARKUP_RESIDUE.search(step) for step in instructions):
        return None, "source_markup_residue"
    if any(_ENCODING_RESIDUE.search(step) for step in instructions):
        return None, "encoding_residue"
    if any(_INSTRUCTION_METADATA_PREFIX.search(step) for step in instructions):
        return None, "metadata_or_narrative"
    if any(_SHOPPING_METADATA.search(step) for step in instructions):
        return None, "metadata_or_narrative"
    if any(
        _URL_OR_DOMAIN_METADATA.search(step) or _PRICED_WINE_RECOMMENDATION.search(step)
        for step in instructions
    ):
        return None, "metadata_or_narrative"
    if any(step.count("(") != step.count(")") for step in instructions):
        return None, "unbalanced_parentheses"
    if any(_CORRUPT_TEXT_TOKEN.search(step) for step in instructions):
        return None, "source_text_corruption"
    if any(re.fullmatch(r"\s*fin\.?\s*", step, re.IGNORECASE) for step in instructions):
        return None, "source_text_corruption"
    if any(_REPEATED_CONNECTOR_OR_PHRASE.search(step) for step in instructions):
        return None, "source_text_corruption"
    if any(
        _JOINED_MIXED_FRACTION.search(step)
        or _GLUED_QUANTITY_TOKEN.search(step)
        or _MALFORMED_FRACTION_SPACING.search(step)
        or _FRACTIONAL_AMOUNT_PLURAL_UNIT.search(step)
        or _PLURAL_AMOUNT_SINGULAR_UNIT.search(step)
        or _MIXED_AMOUNT_SINGULAR_UNIT.search(step)
        or _HYPHENATED_PLURAL_UNIT.search(step)
        or _UNIT_GLUED_TO_TITLE_WORD.search(step)
        or has_inconsistent_mass_conversion(step)
        or has_inconsistent_volume_conversion(step)
        for step in instructions
    ):
        return None, "corrupt_quantity_token"
    if any(re.search(r"\bstraincooking\b", step, re.IGNORECASE) for step in instructions):
        return None, "source_text_corruption"

    scope_reason = scope_exclusion_reason(
        [title, *ingredient_lines, *instructions],
        merge_scope_markers(scope_exclusion_markers),
    )
    if scope_reason:
        return None, scope_reason

    coverage = measurement_coverage(ingredient_lines)
    if not 0.0 <= minimum_measurement_coverage <= 1.0:
        raise ValueError("minimum_measurement_coverage must be between 0 and 1")
    if coverage < minimum_measurement_coverage:
        return None, "insufficient_source_measurement_coverage"

    ingredients_normalized = normalize_ingredient_lines(ingredient_lines)
    screening_ingredients = _alias_lines_for_screening(ingredients_normalized, ingredient_aliases)
    category_group = broad_category(title, "", ingredients_normalized)
    complexity = complexity_bucket(len(ingredient_lines), len(instructions))
    equipment = infer_equipment(
        instructions,
        equipment_rules or {},
        equipment_exclusion_phrases or {},
    )
    diet_markers = diet_conflict_markers or {}
    vegetarian_status = infer_diet_status(
        screening_ingredients, [title], "vegetarian", diet_markers
    )
    vegan_status = infer_diet_status(screening_ingredients, [title], "vegan", diet_markers)

    source_hash = source_row_hash(row)
    recipe_id = f"epicurious-{source_record_id}"
    quality_flags: list[str] = []
    if coverage < 1.0:
        quality_flags.append("some_source_lines_use_no_explicit_measurement")
    if removed_non_ingredient_line_count:
        quality_flags.append("source_non_ingredient_lines_removed")
    if quantity_format_correction_count:
        quality_flags.append("joined_mixed_fraction_spacing_restored")
    if vegetarian_status == "unknown":
        quality_flags.append("vegetarian_unknown")
    if vegan_status == "unknown":
        quality_flags.append("vegan_unknown")

    record: dict[str, Any] = {
        "recipe_id": recipe_id,
        "title": title,
        "category_original": None,
        "category_group": category_group,
        "description_original": None,
        "keywords_original": [],
        "ingredient_lines_original": ingredient_lines,
        "ingredients_normalized": ingredients_normalized,
        "source_ingredient_line_count": source_ingredient_line_count,
        "removed_non_ingredient_line_count": removed_non_ingredient_line_count,
        "quantity_format_correction_count": quantity_format_correction_count,
        "instructions_original": instructions,
        "prep_time_minutes": None,
        "cook_time_minutes": None,
        "total_time_minutes_derived": None,
        "rating": None,
        "review_count": None,
        "servings": None,
        "yield_original": None,
        "equipment_tags_inferred": equipment,
        "vegetarian_status_derived": vegetarian_status,
        "vegan_status_derived": vegan_status,
        "complexity_bucket": complexity,
        "measurement_coverage": coverage,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_url": SOURCE_DATASET_URL,
        "source_record_id": source_record_id,
        "source_file_sha256": source_file_sha256,
        "source_license": SOURCE_LICENSE,
        "source_row_hash": source_hash,
        "dataset_version": CURATION_VERSION,
        "quality_flags": quality_flags,
        "content_for_embedding": " | ".join([title, category_group, *ingredients_normalized]),
    }
    record["dedup_hash"] = content_dedup_hash(title, ingredient_lines, instructions)
    record["record_hash"] = stable_hash(record)
    return record, None
