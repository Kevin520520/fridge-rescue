"""Streamlit entry point for the Fridge Rescue recipe recommendation application."""

from __future__ import annotations

import logging
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
from fridge_rescue.cooking_constraints import request_from_inputs
from fridge_rescue.cooking_constraints_ui import (
    initialize_cooking_inputs,
    render_cooking_filters,
    render_natural_language_input,
)
from fridge_rescue.cooking_metadata import COOKING_METADATA_VERSION
from fridge_rescue.image_recognition_ui import render_image_ingredient_input
from fridge_rescue.ingredient_graph import RecipeIngredientGraph
from fridge_rescue.ingredient_graph_ui import (
    render_ingredient_graph_query,
    render_recipe_graph_evidence,
)
from fridge_rescue.input_normalization import (
    UnsupportedChineseIngredientError,
    contains_chinese,
)
from fridge_rescue.openai_selector import ResponsesRecommendationSelector
from fridge_rescue.presentation import (
    ai_privacy_notice,
    authoritative_source_line_html,
    ingredient_display_lines,
)
from fridge_rescue.repository import (
    DEFAULT_CURATED_CORPUS_PATH,
    ParquetRecipeRepository,
)
from fridge_rescue.retrieval import (
    DeterministicRecipeRetriever,
    RetrievalPolicy,
    RetrievalRules,
    ingredient_match_details,
    matched_input_surface_phrases,
)
from fridge_rescue.schemas import RecipeRecord, UserRequest
from fridge_rescue.settings import PROJECT_ROOT, get_settings
from fridge_rescue.workflow import FridgeRescueWorkflow, WorkflowOutcome

logger = logging.getLogger(__name__)

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
    OFFLINE_MODE: (
        "Recipe search uses only the local collection. Optional photo and language tools "
        "call AI only when clicked. / 菜谱检索仅使用本地数据；照片和自然语言工具仅在点击时调用 AI。"
    ),
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
            padding: .35rem .55rem;
            border-radius: 10px;
            color: #3f332c;
            line-height: 1.55;
            white-space: pre-wrap;
        }

        .fr-ingredient-hit {
            padding: .05rem .16rem;
            border-radius: .3rem;
            background: #dff1d8;
            color: #28522b;
            font-weight: 800;
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
def load_offline_runtime(metadata_version: str) -> tuple[
    ParquetRecipeRepository,
    DeterministicRecipeRetriever,
    tuple[str, ...],
    tuple[str, ...],
    RecipeIngredientGraph,
]:
    """Cache corpus resources by metadata policy version across Streamlit reruns."""

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
    ingredient_graph = RecipeIngredientGraph(
        repository.list_all(), ingredient_aliases=retrieval_rules.ingredient_aliases,
        substitutions=rules.get("substitutions", {}), pantry_staples=pantry_staples,
    )
    return repository, retriever, reviewed_notes, pantry_staples, ingredient_graph


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


def _requested_ingredient_summary(
    recipe: RecipeRecord,
    request: UserRequest,
    ingredient_aliases: Mapping[str, str],
) -> tuple[int, int]:
    """Return coverage of every ingredient explicitly entered by the user."""

    details = ingredient_match_details(
        recipe,
        request.available_ingredients,
        (),
        ingredient_aliases,
    )
    total_requested = len(details.matched_inputs) + len(details.unmatched_inputs)
    return details.matched_input_count, total_requested


def _requested_ingredient_line_highlights(
    recipe: RecipeRecord,
    request: UserRequest,
    ingredient_aliases: Mapping[str, str],
) -> list[tuple[str, ...]]:
    """Return source-side phrases matching ingredients explicitly entered by the user."""

    return _requested_text_line_highlights(
        recipe.ingredient_lines_original,
        request,
        ingredient_aliases,
    )


def _requested_text_line_highlights(
    lines: Sequence[str],
    request: UserRequest,
    ingredient_aliases: Mapping[str, str],
) -> list[tuple[str, ...]]:
    """Return visible phrases matching ingredients explicitly entered by the user."""

    return [
        matched_input_surface_phrases(
            line,
            request.available_ingredients,
            ingredient_aliases,
        )
        for line in lines
    ]


def _display_recipe(
    recipe: RecipeRecord,
    request: UserRequest,
    pantry_staples: Sequence[str],
    ingredient_aliases: Mapping[str, str],
    *,
    reason: str,
    substitution_notes: Sequence[str] = (),
) -> None:
    _, missing = _coverage_summary(
        recipe,
        request,
        pantry_staples,
        ingredient_aliases,
    )
    matched_requested, total_requested = _requested_ingredient_summary(
        recipe,
        request,
        ingredient_aliases,
    )
    requested_line_highlights = _requested_ingredient_line_highlights(
        recipe,
        request,
        ingredient_aliases,
    )
    with st.container(border=True):
        st.subheader(f"🍽️ {recipe.title}")
        facts: list[str] = []
        if recipe.total_time_minutes is not None:
            facts.append(f"⏱️ {recipe.total_time_minutes} min")
        elif recipe.estimated_time_minutes is not None:
            facts.append(f"⏱️ About {recipe.estimated_time_minutes} min (step estimate) · 约 {recipe.estimated_time_minutes} 分钟（步骤估算）")
        if request.diet_preference != "none":
            diet_label = "Vegan · 纯素" if request.diet_preference == "vegan" else "Vegetarian · 蛋奶素"
            source_field = (
                recipe.vegan_status_source
                if request.diet_preference == "vegan"
                else recipe.vegetarian_status_source
            )
            source = (
                "Runtime ingredient rules · 运行时食材规则"
                if source_field == "runtime_ingredient_rules"
                else "Curation rules · 数据整理规则"
            )
            facts.append(f"🥬 {diet_label}（{source}）")
        if recipe.rating is not None:
            facts.append(f"⭐ {recipe.rating:.1f}/5")
        facts.append(
            f"🥕 {matched_requested}/{total_requested} requested ingredients used"
        )
        if missing:
            facts.append(f"🛒 {len(missing)} extra ingredients needed")
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
            for ingredient_line, highlight_phrases in zip(
                ingredient_display_lines(recipe),
                requested_line_highlights,
                strict=True,
            ):
                st.markdown(
                    authoritative_source_line_html(
                        ingredient_line,
                        prefix="•",
                        highlight_phrases=highlight_phrases,
                    ),
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
            if recipe.total_time_minutes is None and recipe.time_estimate_basis:
                st.markdown("**时长估算依据 · Time estimate**")
                for basis in recipe.time_estimate_basis:
                    st.write(basis)
            if "runtime_ingredient_rules" in {
                recipe.vegetarian_status_source,
                recipe.vegan_status_source,
            }:
                st.caption("Missing diet labels are supplemented at runtime using animal-product markers, recognized ingredients and prepared-food rules in the original ingredient lines. / 缺失的饮食标签会在运行时根据原始食材行中的动物性成分、已知食材和复合食品规则补充判定。")
            if substitution_notes:
                st.markdown("**Allowed substitution notes**")
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
        st.error(
            "The AI selection could not be validated. Please try again or use Local retrieval."
        )


def _display_baseline_outcome(
    outcome: BaselineRecommendation,
    request: UserRequest,
    ingredient_aliases: Mapping[str, str],
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
        requested_line_highlights = _requested_text_line_highlights(
            recipe.ingredients,
            request,
            ingredient_aliases,
        )
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
                for ingredient, highlight_phrases in zip(
                    recipe.ingredients,
                    requested_line_highlights,
                    strict=True,
                ):
                    st.markdown(
                        authoritative_source_line_html(
                            ingredient,
                            prefix="•",
                            highlight_phrases=highlight_phrases,
                        ),
                        unsafe_allow_html=True,
                    )
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
    repository, retriever, reviewed_notes, pantry_staples, ingredient_graph = load_offline_runtime(COOKING_METADATA_VERSION)
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
initialize_cooking_inputs()
render_image_ingredient_input(settings, retriever.rules.ingredient_aliases or {})
render_natural_language_input(settings, retriever.rules.ingredient_aliases or {})
with st.container(border=True):
    ingredients_text = st.text_input(
        "Available ingredients · 可用食材",
        placeholder="e.g. eggs, tomatoes, rice · 例如：鸡蛋、番茄、米饭",
        help="English and Chinese are supported. Separate items with commas, semicolons, or new lines.",
        key="available_ingredients",
    )
    excluded_text = st.text_input(
        "Ingredients to avoid · 需避开的食材",
        placeholder="e.g. peanuts, milk (optional) · 例如：花生、牛奶",
        help="Enter ingredient names only. Allergy filtering is a precautionary text check.",
        key="excluded_ingredients",
    )
    render_cooking_filters()
    render_ingredient_graph_query(ingredient_graph, retriever, repository, st.session_state)
    submitted = st.button(
        "🍽️ Find recipes",
        key="find_recipes",
        type="primary",
        use_container_width=True,
    )

if submitted:
    try:
        ingredient_aliases = retriever.rules.ingredient_aliases or {}
        request = request_from_inputs(st.session_state, ingredient_aliases)
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
                    recipe_ids = retriever.retrieve(
                        request, limit=max_recommendations,
                        candidate_ids=ingredient_graph.find_recipe_ids(request.available_ingredients),
                    )
                    recipes = repository.get_many(recipe_ids)
                    if not recipes:
                        st.warning("No recipe matches these ingredients and filters yet.")
                    for recipe in recipes:
                        matched_requested, total_requested = _requested_ingredient_summary(
                            recipe,
                            request,
                            ingredient_aliases,
                        )
                        reason = (
                            f"Uses {matched_requested}/{total_requested} of your requested "
                            "ingredients."
                        )
                        _display_recipe(
                            recipe,
                            request,
                            pantry_staples,
                            ingredient_aliases,
                            reason=reason,
                        )
                    render_recipe_graph_evidence(
                        ingredient_graph, recipe_ids, request.available_ingredients, repository
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
                        request,
                        ingredient_aliases,
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
                        selection_pool_size=max_recommendations,
                        allowed_substitution_notes=reviewed_notes,
                        allergen_aliases=retriever.rules.allergen_aliases,
                        ingredient_aliases=retriever.rules.ingredient_aliases,
                        pantry_staples=retriever.rules.pantry_staples,
                        ingredient_graph=ingredient_graph,
                    )
                    outcome = workflow.run(request)
                    _display_workflow_outcome(
                        outcome,
                        request,
                        pantry_staples,
                        ingredient_aliases,
                        max_recommendations=max_recommendations,
                    )
                    render_recipe_graph_evidence(
                        ingredient_graph,
                        [item.recipe.recipe_id for item in outcome.recommendations],
                        request.available_ingredients, repository,
                    )
        except BaselineError:
            st.error(
                "The single-prompt comparison is temporarily unavailable. Please try again later."
            )
        except Exception:
            logger.exception("Unhandled recommendation request error")
            st.error(
                "The recommendation could not be completed. Try again or switch to local retrieval."
            )

st.divider()
st.caption(
    "⚠️ The avoid-ingredient check is guidance only. Check labels carefully for severe allergies."
)
