"""Streamlit entry point for the Fridge Rescue recipe recommendation application."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from pydantic import ValidationError

from fridge_rescue.baseline import (
    BaselineError,
    BaselineRecommendation,
    ResponsesSinglePromptBaseline,
)
from fridge_rescue.input_normalization import (
    UnsupportedChineseIngredientError,
    contains_chinese,
    parse_ingredient_text,
)
from fridge_rescue.openai_selector import ResponsesRecommendationSelector
from fridge_rescue.presentation import (
    ai_privacy_notice,
    authoritative_source_line_html,
    ingredient_display_lines,
)
from fridge_rescue.repository import DEFAULT_CURATED_CORPUS_PATH, ParquetRecipeRepository
from fridge_rescue.retrieval import (
    DeterministicRecipeRetriever,
    RetrievalPolicy,
    RetrievalRules,
    ingredient_match_details,
)
from fridge_rescue.schemas import RecipeRecord, UserRequest
from fridge_rescue.settings import PROJECT_ROOT, get_settings
from fridge_rescue.workflow import FridgeRescueWorkflow, WorkflowOutcome

APP_TITLE = "Fridge Rescue"
OFFLINE_MODE = "Offline retrieval preview"
AGENTIC_MODE = "Agentic AI workflow"
BASELINE_MODE = "Single-prompt baseline"

MODE_LABELS = {
    AGENTIC_MODE: "✨ Smart recommendation · 智能推荐",
    BASELINE_MODE: "🧪 Single-prompt AI · 单次 AI",
    OFFLINE_MODE: "📚 Local retrieval · 本地检索",
}
MODE_DESCRIPTIONS = {
    AGENTIC_MODE: "Searches the curated recipe collection, then validates ingredient fit.",
    BASELINE_MODE: "Generates directly with AI, without recipe retrieval; included for comparison.",
    OFFLINE_MODE: "Searches the local recipe collection only, with no external AI call or API cost.",
}


def _inject_app_styles() -> None:
    """Apply application-specific styles without changing widget behavior."""

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 4%, rgba(244, 177, 98, 0.18), transparent 28rem),
                radial-gradient(circle at 92% 20%, rgba(111, 153, 91, 0.12), transparent 24rem),
                #fffaf4;
        }

        .block-container {
            max-width: 960px;
            padding-top: 2.25rem;
            padding-bottom: 3.5rem;
        }

        .fr-hero {
            position: relative;
            overflow: hidden;
            margin-bottom: 1.5rem;
            padding: 2.2rem 2.35rem;
            border: 1px solid rgba(213, 104, 61, 0.18);
            border-radius: 28px;
            background: linear-gradient(135deg, #fff2df 0%, #fffaf1 58%, #edf5e7 100%);
            box-shadow: 0 18px 45px rgba(91, 61, 38, 0.09);
        }

        .fr-hero::after {
            content: "🍅  🥕  🥬";
            position: absolute;
            right: 2rem;
            top: 1.9rem;
            font-size: 2rem;
            letter-spacing: .25rem;
            transform: rotate(3deg);
        }

        .fr-kicker {
            margin-bottom: .45rem;
            color: #a5482f;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-transform: uppercase;
        }

        .fr-hero h1 {
            max-width: 650px;
            margin: 0;
            color: #33251f;
            font-size: clamp(2.15rem, 5vw, 3.55rem);
            line-height: 1.05;
            letter-spacing: -.04em;
        }

        .fr-hero p {
            max-width: 590px;
            margin: .9rem 0 0;
            color: #69564b;
            font-size: 1.03rem;
            line-height: 1.65;
        }

        .fr-hero-cn {
            max-width: 590px;
            margin-top: .35rem;
            color: #8b7b72;
            font-size: .86rem;
            font-weight: 500;
            line-height: 1.55;
        }

        .fr-source-line {
            margin: .35rem 0;
            color: #3f332c;
            line-height: 1.55;
            white-space: pre-wrap;
        }

        .fr-zh {
            margin-left: .45rem;
            color: #8b7b72;
            font-size: .78em;
            font-weight: 500;
        }

        .fr-section-title {
            margin: 1.4rem 0 .5rem;
            color: #3c2b23;
            font-size: 1.18rem;
            font-weight: 800;
        }

        div[data-testid="stForm"] {
            margin-top: .65rem;
            padding: 1.45rem 1.5rem 1.25rem;
            border: 1px solid #eedbcb;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 14px 36px rgba(91, 61, 38, 0.07);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {
            border-radius: 12px;
        }

        div[data-testid="stButton"] button,
        div[data-testid="stFormSubmitButton"] button {
            min-height: 3rem;
            border: 0;
            border-radius: 999px;
            background: linear-gradient(135deg, #df603d, #ef8b45);
            color: white;
            font-weight: 800;
            box-shadow: 0 9px 20px rgba(214, 86, 49, 0.22);
        }

        div[data-testid="stButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background: linear-gradient(135deg, #cf5032, #e87b37);
            color: white;
            border: 0;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #edddcf;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.76);
            box-shadow: 0 10px 28px rgba(91, 61, 38, 0.06);
        }

        @media (max-width: 700px) {
            .block-container { padding-top: 1rem; }
            .fr-hero { padding: 1.6rem 1.35rem; border-radius: 22px; }
            .fr-hero::after { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _provider_label(provider: str) -> str:
    return {"deepseek": "DeepSeek", "openai": "OpenAI"}.get(provider, "LLM")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = yaml.safe_load(source) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def _reviewed_substitution_notes(rules: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    substitutions = rules.get("substitutions", {})
    if not isinstance(substitutions, dict):
        return notes
    for original, raw_rule in substitutions.items():
        if not isinstance(raw_rule, dict):
            continue
        substitute = str(raw_rule.get("substitute", "")).strip()
        warning = str(raw_rule.get("warning", "")).strip()
        contexts = raw_rule.get("allowed_contexts", [])
        if not substitute or not isinstance(contexts, list) or not contexts:
            continue
        context_text = ", ".join(str(item).strip() for item in contexts if str(item).strip())
        note = f"For {original}, {substitute} may work in: {context_text}."
        if warning:
            note = f"{note} {warning}"
        notes.append(note)
    return notes


@st.cache_resource(show_spinner=False)
def load_offline_runtime() -> tuple[
    ParquetRecipeRepository,
    DeterministicRecipeRetriever,
    tuple[str, ...],
    tuple[str, ...],
]:
    repository = ParquetRecipeRepository()
    retrieval_rules = RetrievalRules.from_yaml()
    retrieval_policy = RetrievalPolicy.from_yaml()
    retriever = DeterministicRecipeRetriever(
        repository,
        rules=retrieval_rules,
        policy=retrieval_policy,
    )
    rules = _load_yaml(PROJECT_ROOT / "rules.yaml")
    reviewed_notes = tuple(_reviewed_substitution_notes(rules))
    pantry_staples = tuple(retrieval_rules.pantry_staples)
    return repository, retriever, reviewed_notes, pantry_staples


def _coverage_summary(
    recipe: RecipeRecord,
    request: UserRequest,
    pantry_staples: Sequence[str],
    ingredient_aliases: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    missing: list[str] = []
    for normalized_line, original_line in zip(
        recipe.ingredients_normalized,
        recipe.ingredient_lines_original,
        strict=True,
    ):
        one_line_recipe = recipe.model_copy(
            update={
                "ingredient_lines_original": [original_line],
                "ingredients_normalized": [normalized_line],
            }
        )
        details = ingredient_match_details(
            one_line_recipe,
            request.available_ingredients,
            pantry_staples,
            ingredient_aliases,
        )
        if details.missing_count == 0:
            matched.append(original_line)
        else:
            missing.append(original_line)
    return matched, missing


def _display_recipe(
    recipe: RecipeRecord,
    request: UserRequest,
    pantry_staples: Sequence[str],
    ingredient_aliases: Mapping[str, str],
    *,
    reason: str,
    substitution_notes: Sequence[str] = (),
) -> None:
    matched, missing = _coverage_summary(
        recipe,
        request,
        pantry_staples,
        ingredient_aliases,
    )
    with st.container(border=True):
        st.subheader(f"🍽️ {recipe.title}")
        facts: list[str] = []
        if recipe.total_time_minutes is not None:
            facts.append(f"⏱️ {recipe.total_time_minutes} min")
        if recipe.rating is not None:
            facts.append(f"⭐ {recipe.rating:.1f}/5")
        facts.append(f"🧺 {len(matched)}/{len(recipe.ingredients_normalized)} ingredients covered")
        st.caption(" · ".join(facts))
        st.write(reason)

        if missing:
            st.info("You'll also need · 还需要：" + ", ".join(missing))
        else:
            st.success("You have the essentials — ready to cook! · 主要食材已齐")

        ingredient_tab, instruction_tab, evidence_tab = st.tabs(
            ["Ingredients · 食材", "Steps · 步骤", "Source · 来源"]
        )
        with ingredient_tab:
            for ingredient_line in ingredient_display_lines(recipe):
                st.markdown(
                    authoritative_source_line_html(ingredient_line, prefix="•"),
                    unsafe_allow_html=True,
                )
        with instruction_tab:
            for step_number, instruction in enumerate(recipe.instructions_original, start=1):
                st.markdown(
                    authoritative_source_line_html(
                        instruction,
                        prefix=f"{step_number}.",
                    ),
                    unsafe_allow_html=True,
                )
        with evidence_tab:
            st.code(f"Epicurious dataset row: {recipe.source_record_id}")
            st.write("Retrieved from the curated local collection of 2,000 Epicurious recipes.")
            st.caption(
                "Curated from the CC BY-SA 3.0 Food Ingredients and Recipe Dataset with Images."
            )
            if substitution_notes:
                st.markdown("**Reviewed substitution notes**")
                for note in substitution_notes:
                    st.markdown(f"- {note}")


def _display_workflow_outcome(
    outcome: WorkflowOutcome,
    request: UserRequest,
    pantry_staples: Sequence[str],
    ingredient_aliases: Mapping[str, str],
    *,
    max_recommendations: int,
) -> None:
    if outcome.status == "success":
        for recommendation in outcome.recommendations[:max_recommendations]:
            _display_recipe(
                recommendation.recipe,
                request,
                pantry_staples,
                ingredient_aliases,
                reason=recommendation.reason,
                substitution_notes=recommendation.substitution_notes,
            )
    elif outcome.status in {"no_result", "no_candidates"}:
        st.warning(
            outcome.no_result_reason
            or "No suitable recipe was found. Try adding one more ingredient."
        )
    elif outcome.status == "invalid_request":
        st.error(
            "The request could not be processed. Check the ingredients and filters, then try again."
        )
    elif outcome.status == "service_error":
        st.error(
            "The AI recommendation service is temporarily unavailable. Please try again later."
        )
    else:
        st.error("The result did not pass the safety checks. Adjust the request and try again.")


def _display_baseline_outcome(
    outcome: BaselineRecommendation,
    *,
    max_recommendations: int,
) -> None:
    """Display generated baseline recipes without implying corpus provenance."""

    st.caption(
        "🧪 Comparison mode: generated directly by AI without searching the local recipe collection."
    )
    if outcome.no_result_reason is not None:
        st.warning(outcome.no_result_reason)
        return

    for recipe in outcome.recipes[:max_recommendations]:
        with st.container(border=True):
            st.subheader(recipe.title)
            facts = [f"⏱️ {recipe.total_time_minutes} min"]
            if recipe.equipment:
                facts.append("🔧 " + ", ".join(recipe.equipment))
            st.caption(" · ".join(facts))
            st.write(recipe.reason)

            ingredient_tab, instruction_tab, provenance_tab = st.tabs(
                ["Ingredients · 食材", "Steps · 步骤", "Source · 来源"]
            )
            with ingredient_tab:
                for ingredient in recipe.ingredients:
                    st.text(f"• {ingredient}")
            with instruction_tab:
                for step_number, instruction in enumerate(recipe.instructions, start=1):
                    st.text(f"{step_number}. {instruction}")
            with provenance_tab:
                st.warning(
                    "This result has no local Recipe ID or corpus source and is shown for course comparison only."
                )


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="collapsed",
)
_inject_app_styles()
st.markdown(
    """
    <section class="fr-hero">
      <div class="fr-kicker">Fridge Rescue · Smart recipe finder</div>
      <h1>Turn what's in your fridge into tonight's dinner.</h1>
      <p>Find practical recipes from the ingredients you already have—less waste, less guesswork.</p>
      <div class="fr-hero-cn">用现有食材，找到今晚适合做的菜。</div>
    </section>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()
corpus_ready = DEFAULT_CURATED_CORPUS_PATH.is_file()
api_ready = settings.has_valid_llm_key
provider_label = _provider_label(settings.active_provider)

if not corpus_ready:
    st.error("Recipe data is not ready. Follow the data setup steps in the README first.")
    st.stop()

try:
    repository, retriever, reviewed_notes, pantry_staples = load_offline_runtime()
    app_config = _load_yaml(PROJECT_ROOT / "config.yaml")
    candidate_pool_size = int(app_config["retrieval"]["candidate_pool_size"])
    max_recommendations = int(app_config["retrieval"]["max_recommendations"])
    if candidate_pool_size < 1:
        raise ValueError("retrieval.candidate_pool_size must be positive")
    if not 1 <= max_recommendations <= 3:
        raise ValueError("retrieval.max_recommendations must be between 1 and 3")
except Exception:  # noqa: BLE001 - render a safe startup error
    st.error("Project data or settings could not be loaded. Run the setup check in the README.")
    st.stop()

mode_options = [OFFLINE_MODE]
if api_ready:
    mode_options = [AGENTIC_MODE, BASELINE_MODE, OFFLINE_MODE]
st.markdown(
    '<div class="fr-section-title">Recommendation mode<span class="fr-zh">推荐方式</span></div>',
    unsafe_allow_html=True,
)
mode = st.radio(
    "Run mode",
    mode_options,
    horizontal=True,
    format_func=lambda option: MODE_LABELS[option],
    label_visibility="collapsed",
)
st.caption(MODE_DESCRIPTIONS[mode])
privacy_notice = ai_privacy_notice(
    provider_label,
    uses_external_ai=mode in {AGENTIC_MODE, BASELINE_MODE},
)
if privacy_notice is not None:
    st.caption(privacy_notice)
if not api_ready:
    st.warning(
        f"Local retrieval is available. To enable smart recommendations, add "
        f"{settings.active_key_env_name} to the local .env file or the deployment platform's "
        "secret settings."
    )

st.markdown(
    '<div class="fr-section-title">What\'s in your fridge?<span class="fr-zh">冰箱里有什么？</span></div>',
    unsafe_allow_html=True,
)
with st.form("recipe_request"):
    ingredients_text = st.text_input(
        "Available ingredients · 可用食材",
        placeholder="e.g. eggs, tomatoes, rice · 例如：鸡蛋、番茄、米饭",
        help="English and Chinese are supported. Separate items with commas, semicolons, or new lines.",
    )
    excluded_text = st.text_input(
        "Ingredients to avoid · 需避开的食材",
        placeholder="e.g. peanuts, milk (optional) · 例如：花生、牛奶",
        help="Enter ingredient names only. Allergy filtering is a precautionary text check.",
    )
    submitted = st.form_submit_button(
        "🍽️ Find recipes",
        type="primary",
        use_container_width=True,
    )

if submitted:
    try:
        ingredient_aliases = retriever.rules.ingredient_aliases or {}
        available_ingredients = parse_ingredient_text(ingredients_text, ingredient_aliases)
        excluded_ingredients = parse_ingredient_text(excluded_text, ingredient_aliases)
        request = UserRequest(
            available_ingredients=available_ingredients,
            excluded_ingredients=excluded_ingredients,
            max_time_minutes=None,
        )
    except UnsupportedChineseIngredientError as error:
        st.error(str(error))
    except ValidationError as error:
        st.error(f"Please check the input: {error.errors()[0]['msg']}")
    else:
        if contains_chinese(ingredients_text) or contains_chinese(excluded_text):
            recognized = ", ".join(request.available_ingredients)
            st.caption(f"Recognized ingredients: {recognized}")
        try:
            with st.spinner("Finding recipes that fit…"):
                st.markdown(
                    '<div class="fr-section-title">Tonight\'s picks<span class="fr-zh">今晚推荐</span></div>',
                    unsafe_allow_html=True,
                )
                if mode == OFFLINE_MODE:
                    recipe_ids = retriever.retrieve(request, limit=max_recommendations)
                    recipes = repository.get_many(recipe_ids)
                    if not recipes:
                        st.warning("No recipe matches these ingredients and filters yet.")
                    for recipe in recipes:
                        matched, _ = _coverage_summary(
                            recipe,
                            request,
                            pantry_staples,
                            ingredient_aliases,
                        )
                        reason = f"Matches {len(matched)} of the ingredients you already have."
                        _display_recipe(
                            recipe,
                            request,
                            pantry_staples,
                            ingredient_aliases,
                            reason=reason,
                        )
                elif mode == BASELINE_MODE:
                    if settings.active_api_key is None:
                        raise BaselineError("API key is unavailable")
                    baseline = ResponsesSinglePromptBaseline(
                        model=settings.active_model,
                        api_key=settings.active_api_key.get_secret_value(),
                        base_url=settings.active_base_url,
                        provider_name=provider_label,
                        reasoning_effort=(
                            "none" if settings.active_provider == "deepseek" else None
                        ),
                    )
                    _display_baseline_outcome(
                        baseline.generate(request),
                        max_recommendations=max_recommendations,
                    )
                else:
                    if settings.active_api_key is None:
                        raise RuntimeError("API key is unavailable")
                    selector = ResponsesRecommendationSelector(
                        model=settings.active_model,
                        api_key=settings.active_api_key.get_secret_value(),
                        base_url=settings.active_base_url,
                        provider_name=provider_label,
                        reasoning_effort=(
                            "none" if settings.active_provider == "deepseek" else None
                        ),
                        allowed_substitution_notes=reviewed_notes,
                        ingredient_aliases=retriever.rules.ingredient_aliases,
                        pantry_staples=retriever.rules.pantry_staples,
                    )
                    workflow = FridgeRescueWorkflow(
                        repository=repository,
                        retriever=retriever,
                        selector=selector,
                        top_k=candidate_pool_size,
                        allowed_substitution_notes=reviewed_notes,
                        allergen_aliases=retriever.rules.allergen_aliases,
                        ingredient_aliases=retriever.rules.ingredient_aliases,
                        pantry_staples=retriever.rules.pantry_staples,
                    )
                    _display_workflow_outcome(
                        workflow.run(request),
                        request,
                        pantry_staples,
                        ingredient_aliases,
                        max_recommendations=max_recommendations,
                    )
        except BaselineError:
            st.error(
                "The single-prompt comparison is temporarily unavailable. Please try again later."
            )
        except Exception:  # noqa: BLE001 - render a safe request error
            st.error(
                "The recommendation could not be completed. Try again or switch to local retrieval."
            )

st.divider()
st.caption(
    "⚠️ The avoid-ingredient check is guidance only. Check labels carefully for severe allergies."
)
