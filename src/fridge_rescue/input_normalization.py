"""Deterministic parsing and configured bilingual normalization for UI input."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from fridge_rescue.rule_checker import normalize_token

_SEPARATOR_PATTERN = re.compile(r"[,，、;；\r\n]+")
_CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]")


class UnsupportedChineseIngredientError(ValueError):
    """Raised when Chinese input has no configured deterministic translation."""

    def __init__(self, ingredients: list[str]) -> None:
        self.ingredients = tuple(ingredients)
        joined = "、".join(ingredients)
        super().__init__(
            f"暂不支持以下中文食材：{joined}。请改用英文，或使用已支持的常见中文名称。"
        )


def contains_chinese(value: str) -> bool:
    """Return whether a string contains a CJK Unified Ideograph."""

    return _CJK_PATTERN.search(value) is not None


def _normalized_aliases(aliases: Mapping[str, str]) -> dict[str, str]:
    return {
        normalize_token(unicodedata.normalize("NFKC", str(source))): normalize_token(
            unicodedata.normalize("NFKC", str(target))
        )
        for source, target in aliases.items()
    }


def parse_ingredient_text(text: str, aliases: Mapping[str, str]) -> list[str]:
    """Split UI text and map configured aliases to canonical English terms.

    English comma, Chinese comma, enumeration comma, semicolon, and line
    breaks are accepted as separators. Chinese terms are supported only when
    they have an exact configured mapping; unknown English terms remain valid.
    """

    normalized_text = unicodedata.normalize("NFKC", text)
    alias_lookup = _normalized_aliases(aliases)
    parsed: list[str] = []
    unsupported_chinese: list[str] = []

    for raw_item in _SEPARATOR_PATTERN.split(normalized_text):
        item = normalize_token(raw_item)
        if not item:
            continue
        canonical = alias_lookup.get(item, item)
        if contains_chinese(canonical):
            if item not in unsupported_chinese:
                unsupported_chinese.append(item)
            continue
        if canonical not in parsed:
            parsed.append(canonical)

    if unsupported_chinese:
        raise UnsupportedChineseIngredientError(unsupported_chinese)
    return parsed
