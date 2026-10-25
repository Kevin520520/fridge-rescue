"""User review of language parses, followed by shared editable cooking filters."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import streamlit as st

from fridge_rescue.cooking_constraints import (
    DIET_LABELS, EQUIPMENT_LABELS, MAX_REQUEST_LENGTH, ConstraintParsingError,
    ResponsesConstraintParser, input_defaults, parsed_input_updates,
)
from fridge_rescue.settings import Settings

logger = logging.getLogger(__name__)


def initialize_cooking_inputs() -> None:
    for key, value in input_defaults().items():
        st.session_state.setdefault(key, value)


def render_natural_language_input(settings: Settings, aliases: Mapping[str, str]) -> None:
    """Parse on request; require an explicit review click before changing any filter."""

    with st.expander("💬 Describe your cooking plan · 描述你的烹饪计划"):
        text = st.text_area(
            "Ingredients and conditions · 食材和烹饪条件",
            placeholder="e.g. Eggs and tomatoes, vegetarian, under 30 min, no oven or peanuts. / 例如：有鸡蛋和番茄，30 分钟内，蛋奶素，没有烤箱，不要花生。",
            max_chars=MAX_REQUEST_LENGTH, key="natural_language_request",
        )
        context = ("bilingual-v1", text, settings.active_provider, settings.active_model)
        if st.session_state.get("constraint_parse_context") != context:
            st.session_state.pop("constraint_parse_result", None)
            st.session_state.pop("constraint_parse_error", None)
            st.session_state["constraint_parse_context"] = context
        if settings.has_valid_llm_key:
            provider = "DeepSeek" if settings.active_provider == "deepseek" else "OpenAI"
            st.caption(
                f"Click Parse to send this request to {provider}. Review it before applying the editable ingredients and filters below. / "
                f"点击解析会将这句话发送给 {provider}；确认后填入下方可编辑的食材和筛选项。"
            )
        else:
            st.info(
                "Language parsing uses your configured AI key. You can enter ingredients and filters below. / "
                "自然语言解析使用现有 AI 密钥。当前可直接填写下方食材和筛选项。"
            )
        if st.button(
            "Parse conditions · 解析需求", key="parse_cooking_request",
            disabled=not text.strip() or not settings.has_valid_llm_key,
        ):
            st.session_state.pop("constraint_parse_result", None)
            st.session_state.pop("constraint_parse_error", None)
            try:
                with st.spinner("Parsing ingredients and conditions… / 正在解析食材和条件…"):
                    key = settings.active_api_key
                    parser = ResponsesConstraintParser(
                        model=settings.active_model,
                        api_key=key.get_secret_value() if key else None,
                        base_url=settings.active_base_url,
                        reasoning_effort="none" if settings.active_provider == "deepseek" else None,
                    )
                    st.session_state["constraint_parse_result"] = parser.parse(text)
            except ConstraintParsingError as error:
                logger.warning("Cooking request parsing failed: %s", type(error).__name__)
                st.session_state["constraint_parse_error"] = (
                    "Parsing is temporarily unavailable. Retry or enter the filters below. / "
                    "暂时无法解析，请重试或直接填写下方筛选项。"
                )
        if error := st.session_state.get("constraint_parse_error"):
            st.error(error)
        parsed = st.session_state.get("constraint_parse_result")
        if parsed is None:
            return

        st.markdown("**Review parsed conditions · 请确认解析结果**")
        if parsed.available_ingredients:
            st.write("Available ingredients · 可用食材：" + ", ".join(parsed.available_ingredients))
        if parsed.excluded_ingredients:
            st.write("Ingredients to avoid · 需避开：" + ", ".join(parsed.excluded_ingredients))
        if parsed.max_time_minutes is not None:
            st.write(f"Time limit: {parsed.max_time_minutes} min · 时间上限：{parsed.max_time_minutes} 分钟")
        if parsed.diet_preference is not None:
            st.write("Diet preference · 饮食偏好：" + DIET_LABELS[parsed.diet_preference])
        if parsed.available_equipment:
            st.write("Available equipment · 可用厨具：" + ", ".join(EQUIPMENT_LABELS[code] for code in parsed.available_equipment))
        if parsed.equipment_is_exhaustive:
            st.caption("This is the complete special-equipment list; other tagged devices will be excluded. / 以上是完整的专用厨具清单，其余已标注专用厨具将被排除。")
        if parsed.unavailable_equipment:
            st.write("Unavailable equipment · 不可用厨具：" + ", ".join(EQUIPMENT_LABELS[code] for code in parsed.unavailable_equipment))
        if parsed.clarifications:
            for question in parsed.clarifications:
                st.warning(question.display_text)
            st.caption("Update the description above and parse it again. / 请在上面的描述中补充或调整后重新解析。")
        elif not any((
            parsed.available_ingredients, parsed.excluded_ingredients,
            parsed.max_time_minutes, parsed.diet_preference,
            parsed.available_equipment, parsed.unavailable_equipment,
            parsed.equipment_is_exhaustive,
        )):
            st.info("No ingredients or conditions were found. Add more detail. / 没有识别到食材或烹饪条件，请补充描述。")
            return
        st.caption(
            "Ingredients and exclusions will merge with existing entries. Explicit time, diet and equipment conditions will update the filters. / "
            "食材和禁忌将与现有输入合并；句中明确的时间、饮食和厨具条件将更新对应筛选项。"
        )
        if st.button(
            "Confirm and apply · 确认并填入", key="apply_cooking_request",
            disabled=bool(parsed.clarifications),
        ):
            st.session_state.update(parsed_input_updates(parsed, st.session_state, aliases))
            st.success("Applied below. Edit the fields or find recipes. / 已填入下方，可以继续编辑或搜索菜谱。")


def render_cooking_filters() -> None:
    time_column, diet_column = st.columns(2)
    with time_column:
        st.checkbox("Limit cooking time · 限制烹饪时间", key="time_limit_enabled")
        st.number_input(
            "Maximum minutes · 最多几分钟", min_value=1, max_value=1440, step=5,
            key="max_time_minutes", disabled=not st.session_state["time_limit_enabled"],
        )
        st.checkbox(
            "Include time estimates · 包含步骤估算时长", key="allow_estimated_time",
            disabled=not st.session_state["time_limit_enabled"],
        )
    with diet_column:
        st.selectbox(
            "Diet preference · 饮食偏好", options=list(DIET_LABELS),
            format_func=DIET_LABELS.__getitem__, key="diet_preference",
        )
    st.checkbox("Filter by available equipment · 按可用厨具筛选", key="equipment_filter_enabled")
    st.multiselect(
        "Available special equipment · 可用的专用厨具", options=list(EQUIPMENT_LABELS),
        format_func=EQUIPMENT_LABELS.__getitem__, key="available_equipment",
        disabled=not st.session_state["equipment_filter_enabled"],
    )
    st.caption(
        "Basic pots, pans and a stovetop are assumed available. Equipment filtering covers the six tagged devices above. "
        "Source total times are missing, so step estimates are used by default; see each result's Source tab for the calculation. / "
        "基础锅具和灶台默认可用；厨具筛选覆盖上面六类菜谱已标注设备。当前语料缺少来源总时长，默认使用步骤估算；结果的“来源”中可查看依据。"
    )
