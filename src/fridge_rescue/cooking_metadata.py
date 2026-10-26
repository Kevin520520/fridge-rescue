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

COOKING_METADATA_VERSION = "cooking-tools-v3-bilingual"


def _phrases(values: str) -> re.Pattern[str]:
    terms = sorted((item.strip() for item in values.split("|")), key=len, reverse=True)
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(item) for item in terms) + r")(?:s|es)?(?!\w)",
        re.IGNORECASE,
    )


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
    "|vegan butter|vegan cream cheese|vegan cheese|vegan cream|vegan yogurt|vegan yoghurt|"
    "vegan sausage|vegan chorizo|vegan chorizo sausage|vegetarian sausage|"
    "vegetarian chorizo|vegetarian chorizo sausage|veggie sausage|veggie chorizo|"
    "veggie chorizo sausage|meatless sausage|meatless chorizo|plant-based sausage|"
    "plant-based chorizo|oyster mushroom|king oyster mushroom"
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
    r"(?<![\w.])(?P<low>\d+(?:\.\d+)?)(?:\s*(?:to|or|[-–—])\s*"
    r"(?P<high>\d+(?:\.\d+)?))?\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {"a": "1", "an": "1", "one": "1", "two": "2", "three": "3", "four": "4",
                 "five": "5", "ten": "10", "fifteen": "15", "twenty": "20", "thirty": "30",
                 "half": "0.5"}
_COOKING = _phrases("cook|bake|roast|fry|simmer|boil|broil|grill|steam|saute|sauté|microwave|chill|freeze|marinate|rest|cool|rise")
_OPTIONAL_STORAGE_ACTION = re.compile(
    r"\b(?:(?:can|may) be (?:made|prepared|stored|kept|refrigerated|frozen|done)|"
    r"store(?:d)?|"
    r"keeps?(?: well)?(?: for)?|(?:will\s+)?lasts?|will stay|make ahead|leftovers?)\b",
    re.IGNORECASE,
)
_OPTIONAL_BRANCH_MARKER = re.compile(
    r"(?<!\w)(?:optional\s*:|alternatively\b|alternately\b|"
    r"if you (?:want|wish|prefer|choose|would like)\b)",
    re.IGNORECASE,
)
_OPTIONAL_CHOICE_MARKER = re.compile(
    r"\bif you (?:want|wish|prefer|choose|would like)\b",
    re.IGNORECASE,
)
_OPTIONAL_CONTINUATION_PREFIX = re.compile(
    r"^\s*(?:\d+[.)]\s*)?if you are (?:making|preparing|using|choosing)\b",
    re.IGNORECASE,
)
_OPTIONAL_PATH_REJOIN = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?:to (?:serve|finish|assemble)|"
    r"continue with|return to|when ready to serve)\b",
    re.IGNORECASE,
)
_OPTIONAL_WHOLE_STEP_PREFIX = re.compile(
    r"^\s*(?:\d+[.)]\s*)?make a more traditional\b",
    re.IGNORECASE,
)
_DO_AHEAD_MARKER = re.compile(
    r"(?<!\w)do(?:[\s_-]+)ahead(?=\b|_)",
    re.IGNORECASE,
)
_DO_AHEAD_REJOIN = re.compile(
    r"\b(?:just assembled|if serving (?:immediately|right away)|"
    r"if not (?:making|preparing)(?: it)? ahead|without (?:chilling|refrigerating))\b",
    re.IGNORECASE,
)
_OPTIONAL_AHEAD_PREFIX = re.compile(
    r"(?:\bor\s+|\band\s+)?(?:for\s+)?up to\s*$",
    re.IGNORECASE,
)
_OPTIONAL_AHEAD_CONTEXT = re.compile(
    r"\b(?:can|may) be\b|\bif you can\b|\bdo ahead\b|\bmake ahead\b|"
    r"\bbest to (?:make|prepare|cook|bake|chill|refrigerate)\b",
    re.IGNORECASE,
)
_STORAGE_UP_TO_PREFIX = re.compile(
    r"\b(?:store|stored|keep|kept|left|refrigerate|refrigerated|freeze|frozen|chill|chilled)\b"
    r"[^.;!?]*\b(?:for\s+)?up to\s*$",
    re.IGNORECASE,
)
_UP_TO_RANGE_CONNECTOR = re.compile(
    r"\s*[,;]?\s*(?:(?:and|or)\s+)?(?:for\s+)?up to\s*",
    re.IGNORECASE,
)
_PREFERRED_EXTENSION_CONNECTOR = re.compile(
    r"\s*(?:and|or)\s+(?:preferably|ideally)(?:\s+for)?\s*",
    re.IGNORECASE,
)
_ALTERNATIVE_WAIT_CONNECTOR = re.compile(
    r"\s*,?\s*or\s+(?:instead\s+)?(?:chill|refrigerate|freeze|"
    r"rest|rise|ferment|marinate|soak|dry|let(?: it)? (?:rise|rest|stand))"
    r"(?:\s+for)?\s*",
    re.IGNORECASE,
)
_WAIT_ACTION = re.compile(
    r"\b(?:chill|refrigerate|freeze|cure|marinate|rest|rise|stand|"
    r"soak|ferment|dry|steep|sit|cool)\b",
    re.IGNORECASE,
)
_REPEATED_WAIT_REFERENCE = re.compile(
    r"(?:^\s*after(?:\s+(?:about|approximately|roughly))?\s*$|"
    r"\b(?:when|once)\b[^.;!?]*\bafter(?:\s+(?:about|approximately|roughly))?\s*$)",
    re.IGNORECASE,
)
_SCHEDULING_REFERENCE_PREFIX = re.compile(
    r"^\s*(?:(?:at least|about|approximately|roughly|around)\s*)?$",
    re.IGNORECASE,
)
_SCHEDULING_REFERENCE_SUFFIX = re.compile(
    r"^\s*before\s+(?:[a-z]+ing\b|you\b|the\b)",
    re.IGNORECASE,
)
_PARALLEL_STEP_PREFIX = re.compile(r"^\s*(?:\d+[.)]\s*)?meanwhile\b", re.IGNORECASE)
_OVERNIGHT = re.compile(r"\bovernight\b", re.IGNORECASE)
_OVERNIGHT_ALTERNATIVE = re.compile(
    r"(?:,|\bor\b)\s*(?:or\s+)?overnight\b|\bovernight\b\s*,?\s*or\b",
    re.IGNORECASE,
)
_OVERNIGHT_BEFORE_DURATION = re.compile(
    r"\bovernight\b[^.;!?]{0,48}\bor\s+(?:for\s+)?(?:at least\s+|up to\s+)?$",
    re.IGNORECASE,
)
_OVERNIGHT_CHAIN_CONNECTOR = re.compile(
    r"\s*,?\s*overnight\s*,?\s*or\s+(?:for\s+)?(?:at least\s+|up to\s+)?",
    re.IGNORECASE,
)
_OMITTED_FIRST_UNIT_RANGE = re.compile(
    r"(?P<lead>\bat least\s+)(?P<low>\d+(?:\.\d+)?)"
    r"(?P<join>\s*,?\s*(?:and\s+)?up to\s*)"
    r"(?P<high>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)
_UNICODE_FRACTIONS = {
    "¼": 0.25,
    "½": 0.5,
    "¾": 0.75,
    "⅐": 1 / 7,
    "⅑": 1 / 9,
    "⅒": 0.1,
    "⅓": 1 / 3,
    "⅔": 2 / 3,
    "⅕": 0.2,
    "⅖": 0.4,
    "⅗": 0.6,
    "⅘": 0.8,
    "⅙": 1 / 6,
    "⅚": 5 / 6,
    "⅛": 0.125,
    "⅜": 0.375,
    "⅝": 0.625,
    "⅞": 0.875,
}


def _normalize_duration_numbers(text: str) -> str:
    """Convert ASCII and Unicode recipe fractions to decimal numbers."""

    text = re.sub(
        r"(?<![\w/])(?:(\d+)\s+)?(\d+)/(\d+)(?![\w/])",
        lambda match: str(
            float(match.group(1) or 0)
            + int(match.group(2)) / max(1, int(match.group(3)))
        ),
        text,
    )

    fraction_chars = "".join(_UNICODE_FRACTIONS)

    def replace_unicode_fraction(match: re.Match[str]) -> str:
        whole = float(match.group("whole") or 0)
        return str(whole + _UNICODE_FRACTIONS[match.group("fraction")])

    text = re.sub(
        rf"(?<!\w)(?P<whole>\d+)?\s*(?P<fraction>[{fraction_chars}])(?!\w)",
        replace_unicode_fraction,
        text,
    )
    return _OMITTED_FIRST_UNIT_RANGE.sub(
        lambda match: (
            f"{match.group('lead')}{match.group('low')} {match.group('unit')}"
            f"{match.group('join')}{match.group('high')} {match.group('unit')}"
        ),
        text,
    )


def _required_time_fragments(step: str) -> list[str]:
    """Remove explicitly optional branches while retaining required text in the same step."""

    if _OPTIONAL_WHOLE_STEP_PREFIX.search(step):
        return []
    fragments: list[str] = []
    in_do_ahead_branch = False
    for sentence in re.split(r"(?<=[.!?])\s+", step):
        sentence = re.sub(
            r"\(\s*(?:if|when)\b[^()]*\)",
            "",
            sentence,
            flags=re.IGNORECASE,
        )
        if not sentence.strip():
            continue
        if in_do_ahead_branch:
            rejoin = _DO_AHEAD_REJOIN.search(sentence)
            if rejoin is None:
                continue
            sentence = sentence[rejoin.start():]
            in_do_ahead_branch = False

        branch_markers = [
            (match.start(), "optional", match)
            for match in _OPTIONAL_BRANCH_MARKER.finditer(sentence)
        ]
        branch_markers.extend(
            (match.start(), "do_ahead", match)
            for match in _DO_AHEAD_MARKER.finditer(sentence)
        )
        if not branch_markers:
            fragments.append(sentence)
            continue

        _, marker_kind, marker = min(branch_markers, key=lambda item: item[0])
        opening_parenthesis = sentence.rfind("(", 0, marker.start())
        closing_parenthesis = sentence.find(")", marker.end())
        marker_is_parenthetical = (
            opening_parenthesis != -1 and closing_parenthesis != -1
        )
        prefix_end = opening_parenthesis if marker_is_parenthetical else marker.start()
        prefix = sentence[:prefix_end].strip()
        if re.search(r"[A-Za-z]", prefix):
            fragments.append(prefix)

        if marker_is_parenthetical:
            suffix = sentence[closing_parenthesis + 1:].strip()
            if suffix:
                fragments.append(suffix)
        elif marker_kind == "do_ahead":
            rest = sentence[marker.end():]
            rejoin = _DO_AHEAD_REJOIN.search(rest)
            if rejoin is not None:
                fragments.append(rest[rejoin.start():])
            else:
                in_do_ahead_branch = True
    return fragments


def _clause_prefix(text: str, position: int) -> str:
    clause_start = max(text.rfind(mark, 0, position) for mark in ".;!?")
    return text[clause_start + 1:position]


def _is_optional_storage_duration(text: str, match: re.Match[str]) -> bool:
    """Identify a duration that describes storage rather than required preparation."""

    clause_start = max(text.rfind(mark, 0, match.start()) for mark in ".;!?")
    following_stops = [
        position for mark in ".;!?"
        if (position := text.find(mark, match.end())) != -1
    ]
    clause_end = min(following_stops, default=len(text))
    prefix = text[clause_start + 1:match.start()]
    suffix = text[match.end():clause_end]
    if re.match(r"\s*(?:ahead|in advance)\b", suffix, re.IGNORECASE):
        return True
    if _OVERNIGHT_BEFORE_DURATION.search(prefix):
        return False
    if _OPTIONAL_STORAGE_ACTION.search(prefix):
        return True
    if _STORAGE_UP_TO_PREFIX.search(prefix):
        return True
    if re.search(r"\b(?:ahead|in advance)\b", suffix, re.IGNORECASE):
        return bool(
            _OPTIONAL_AHEAD_PREFIX.search(prefix)
            or _OPTIONAL_AHEAD_CONTEXT.search(prefix)
        )
    return False


def _duration_is_overnight_alternative(text: str, match: re.Match[str]) -> bool:
    prefix = _clause_prefix(text, match.start())
    following_stops = [
        position for mark in ".;!?"
        if (position := text.find(mark, match.end())) != -1
    ]
    clause_end = min(following_stops, default=len(text))
    suffix = text[match.end():clause_end]
    return bool(
        _OVERNIGHT_BEFORE_DURATION.search(prefix)
        or re.match(r"\s*(?:(?:,?\s*or)|,)\s*overnight\b", suffix, re.IGNORECASE)
    )


def _has_required_overnight(text: str) -> bool:
    """Check overnight waits using only the clause that contains each occurrence."""

    for match in _OVERNIGHT.finditer(text):
        if not _OPTIONAL_STORAGE_ACTION.search(_clause_prefix(text, match.start())):
            return True
    return False


def _duration_minutes(match: re.Match[str]) -> float:
    amount = float(match.group("high") or match.group("low"))
    unit = match.group("unit")
    if unit.startswith("sec"):
        return amount / 60
    if unit.startswith(("hour", "hr")):
        return amount * 60
    if unit.startswith("day"):
        return amount * 1440
    return amount


def _is_repeated_wait_reference(
    text: str,
    match: re.Match[str],
    seen_wait_minutes: set[float],
) -> bool:
    """Ignore a later narrative reference to a wait already counted earlier."""

    duration = round(_duration_minutes(match), 6)
    return bool(
        duration in seen_wait_minutes
        and _REPEATED_WAIT_REFERENCE.search(_clause_prefix(text, match.start()))
    )


def _duration_describes_wait(text: str, match: re.Match[str]) -> bool:
    return bool(_WAIT_ACTION.search(_clause_prefix(text, match.start())))


def _is_scheduling_reference_duration(text: str, match: re.Match[str]) -> bool:
    """Identify a relative clock marker rather than elapsed recipe work."""

    return bool(
        _SCHEDULING_REFERENCE_PREFIX.fullmatch(_clause_prefix(text, match.start()))
        and _SCHEDULING_REFERENCE_SUFFIX.match(text[match.end():])
    )


def _required_duration_matches(text: str) -> list[re.Match[str]]:
    """Keep durations needed for the earliest supported completion path."""

    matches: list[re.Match[str]] = []
    for match in _DURATION.finditer(text):
        if _is_scheduling_reference_duration(text, match):
            continue
        if _is_optional_storage_duration(text, match):
            continue
        if matches and _OVERNIGHT_CHAIN_CONNECTOR.fullmatch(
            text[matches[-1].end():match.start()]
        ):
            continue
        if matches and _PREFERRED_EXTENSION_CONNECTOR.fullmatch(
            text[matches[-1].end():match.start()]
        ):
            continue
        if matches and _ALTERNATIVE_WAIT_CONNECTOR.fullmatch(
            text[matches[-1].end():match.start()]
        ):
            if _duration_minutes(match) < _duration_minutes(matches[-1]):
                matches[-1] = match
            continue
        if matches and _UP_TO_RANGE_CONNECTOR.fullmatch(text[matches[-1].end():match.start()]):
            # ``at least X and up to Y`` describes a flexible wait. The time
            # filter answers the earliest the recipe can be completed, so X is
            # the relevant bound and the optional extension Y is ignored.
            continue
        matches.append(match)
    return matches


def estimate_cooking_minutes(instructions: list[str]) -> tuple[int | None, list[str]]:
    """Estimate the earliest supported path plus preparation and untimed-step allowances."""

    if not instructions:
        return None, []
    duration_groups: list[float] = []
    seen_wait_minutes: set[float] = set()
    optional_path_active = False
    untimed_cooking = 0
    evidence = []
    for step_index, step in enumerate(instructions):
        if optional_path_active:
            if _OPTIONAL_PATH_REJOIN.search(step):
                optional_path_active = False
            else:
                continue

        next_step = instructions[step_index + 1] if step_index + 1 < len(instructions) else ""
        starts_multi_step_optional_path = bool(
            _OPTIONAL_CHOICE_MARKER.search(step)
            and _OPTIONAL_CONTINUATION_PREFIX.search(next_step)
        )
        step_minutes = 0.0
        step_has_duration = False
        step_has_untimed_cooking = False
        for fragment in _required_time_fragments(step):
            text = _normalize_duration_numbers(fragment.casefold())
            text = re.sub(
                r"\b(" + "|".join(_NUMBER_WORDS) + r")\s+(?=(?:second|minute|hour|day))",
                lambda match: _NUMBER_WORDS[match.group(1)] + " ", text,
            )
            matches = [
                match
                for match in _required_duration_matches(text)
                if not _is_repeated_wait_reference(text, match, seen_wait_minutes)
            ]
            has_scheduling_reference = any(
                _is_scheduling_reference_duration(text, match)
                for match in _DURATION.finditer(text)
            )
            fragment_minutes = 0.0
            overnight_is_alternative = bool(
                matches and _OVERNIGHT_ALTERNATIVE.search(text)
            )
            for match in matches:
                amount = _duration_minutes(match)
                if _duration_is_overnight_alternative(text, match):
                    amount = min(amount, 720)
                if re.match(r"\s*(?:on |per )?(?:each|a) side", text[match.end():]):
                    amount *= 2
                if _duration_describes_wait(text, match):
                    seen_wait_minutes.add(round(amount, 6))
                fragment_minutes += amount
            required_overnight = _has_required_overnight(text)
            if required_overnight and not overnight_is_alternative:
                fragment_minutes = max(fragment_minutes, 720)
            if matches or required_overnight:
                step_minutes += fragment_minutes
                step_has_duration = True
            elif _COOKING.search(text) and not has_scheduling_reference:
                step_has_untimed_cooking = True
        if step_has_duration:
            is_parallel = bool(_PARALLEL_STEP_PREFIX.search(step) and duration_groups)
            if is_parallel:
                duration_groups[-1] = max(duration_groups[-1], step_minutes)
            else:
                duration_groups.append(step_minutes)
            evidence_label = "Parallel step duration" if is_parallel else "Step duration"
            evidence.append(
                f"{evidence_label}: about {math.ceil(step_minutes)} min · "
                f"步骤时长约 {math.ceil(step_minutes)} 分钟：{step[:220]}"
            )
        elif step_has_untimed_cooking:
            untimed_cooking += 1
        if starts_multi_step_optional_path:
            optional_path_active = True
    minutes = sum(duration_groups)
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
    for field, source_field, vegan in (
        ("vegetarian_status", "vegetarian_status_source", False),
        ("vegan_status", "vegan_status_source", True),
    ):
        derived = ingredient_diet_status(recipe.ingredients_normalized, vegan=vegan)
        original = getattr(recipe, field)
        corrects_known_plant_product = (
            original == "conflict"
            and derived == "match"
            and any(_PLANT_EXCEPTIONS.search(line) for line in recipe.ingredients_normalized)
        )
        if (original == "unknown" and derived != "unknown") or corrects_known_plant_product:
            updates[field] = derived
            updates[source_field] = "runtime_ingredient_rules"
    return recipe.model_copy(update=updates)
