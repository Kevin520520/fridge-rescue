"""Pure display helpers for authoritative recipe facts."""

from __future__ import annotations

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


def ingredient_display_lines(recipe: RecipeRecord) -> tuple[list[str], bool]:
    """Return safe ingredient text and whether source quantities were used.

    Food.com stores ingredient names and quantities as separate arrays.  A
    quantity is paired with a name only when curation explicitly marked the
    arrays as aligned *and* their current lengths still agree.  This second
    check prevents a malformed record from silently displaying a quantity
    beside the wrong ingredient.
    """

    quantities_are_aligned = (
        recipe.quantity_alignment_status == "aligned"
        and len(recipe.ingredient_quantities_original) == len(recipe.ingredients_original)
        and all(
            isinstance(quantity, str) and bool(quantity.strip())
            for quantity in recipe.ingredient_quantities_original
        )
    )
    if not quantities_are_aligned:
        return list(recipe.ingredients_original), False

    return (
        [
            f"{quantity} {ingredient}".strip()
            for quantity, ingredient in zip(
                recipe.ingredient_quantities_original,
                recipe.ingredients_original,
                strict=True,
            )
        ],
        True,
    )
