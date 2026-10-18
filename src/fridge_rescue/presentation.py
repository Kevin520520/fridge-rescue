"""Pure display helpers for authoritative recipe facts."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence

from fridge_rescue.schemas import RecipeRecord


def ai_privacy_notice(provider_label: str, *, uses_external_ai: bool) -> str | None:
    """Return the bilingual privacy notice for modes that contact an AI provider."""

    if not uses_external_ai:
        return None
    label = provider_label.strip() or "AI service provider"
    return (
        "🔒 Privacy / 隐私：Ingredient information is sent to "
        f"{label} in this AI mode. Do not enter personal or sensitive information. / "
        f"此 AI 模式会将食材信息发送给 {label}，请勿输入个人或敏感信息。"
    )


def ingredient_display_lines(recipe: RecipeRecord) -> list[str]:
    """Return the authoritative full ingredient lines from the source row."""

    return list(recipe.ingredient_lines_original)


def authoritative_source_line_html(
    text: str,
    *,
    prefix: str,
    highlight_phrases: Sequence[str] = (),
) -> str:
    """Render escaped source text and highlight only matched ingredient phrases."""

    safe_prefix = html.escape(prefix, quote=True)
    spans: list[tuple[int, int]] = []
    for phrase in sorted(
        {value.strip() for value in highlight_phrases if value.strip()},
        key=lambda value: (-len(value), value.casefold()),
    ):
        parts = [re.escape(part) for part in phrase.split()]
        if not parts:
            continue
        phrase_pattern = r"\s+".join(parts)
        pattern = re.compile(
            rf"(?<!\w){phrase_pattern}(?!\w)",
            flags=re.IGNORECASE,
        )
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text))

    selected: list[tuple[int, int]] = []
    for start, end in sorted(spans, key=lambda span: (span[0], -(span[1] - span[0]))):
        if any(start < chosen_end and end > chosen_start for chosen_start, chosen_end in selected):
            continue
        selected.append((start, end))
    selected.sort()

    parts_html: list[str] = []
    cursor = 0
    for start, end in selected:
        parts_html.append(html.escape(text[cursor:start], quote=True))
        parts_html.append(
            '<mark class="fr-ingredient-hit">'
            f"{html.escape(text[start:end], quote=True)}</mark>"
        )
        cursor = end
    parts_html.append(html.escape(text[cursor:], quote=True))
    return (
        '<div class="fr-source-line" translate="no" lang="en">'
        f"{safe_prefix} {''.join(parts_html)}</div>"
    )
