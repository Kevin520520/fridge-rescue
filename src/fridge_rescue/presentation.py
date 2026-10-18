"""Pure display helpers for authoritative recipe facts."""

from __future__ import annotations

import html

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


def authoritative_source_line_html(text: str, *, prefix: str) -> str:
    """Render a source line without browser translation changing measurements."""

    safe_prefix = html.escape(prefix, quote=True)
    safe_text = html.escape(text, quote=True)
    return (
        '<div class="fr-source-line" translate="no" lang="en">'
        f"{safe_prefix} {safe_text}</div>"
    )
