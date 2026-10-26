"""Streamlit photo upload, recognition and ingredient-selection flow."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping

import streamlit as st

from fridge_rescue.image_recognition import (
    MAX_BATCH_IMAGE_BYTES,
    MAX_IMAGES,
    IngredientRecognitionError,
    InvalidIngredientImage,
    ResponsesIngredientRecognizer,
    canonicalize_recognized_ingredients,
    prepare_ingredient_image,
)
from fridge_rescue.input_normalization import merge_ingredient_text
from fridge_rescue.settings import Settings

logger = logging.getLogger(__name__)
_PHOTO_CONTEXT = "ingredient_photo_context"
_PHOTO_RESULT = "ingredient_photo_result"
_PHOTO_ERROR = "ingredient_photo_error"
_PHOTO_SELECTION = "ingredient_photo_selection"


def _clear_photo_result() -> None:
    for key in (_PHOTO_RESULT, _PHOTO_ERROR, _PHOTO_SELECTION):
        st.session_state.pop(key, None)


def render_image_ingredient_input(
    settings: Settings, ingredient_aliases: Mapping[str, str]
) -> None:
    """Keep photos and results in this user's session; call AI only on a click."""

    st.session_state.setdefault("available_ingredients", "")
    with st.expander("📷 Ingredients from photos · 多张图片识别食材", expanded=True):
        uploaded = st.file_uploader(
            "Upload fridge, ingredients and seasoning photos · 上传冰箱、食材或调料照片",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            max_upload_size=10,
            key="ingredient_photo_upload",
            help="JPG/PNG/WebP; up to 5 photos, 10 MB each, 25 MB total. / 最多 5 张，每张 10 MB，总计 25 MB。",
        )
        uploads = list(uploaded or [])
        photos: list[tuple[str, bytes]] = []
        prepared = []
        valid_batch = True
        if len(uploads) > MAX_IMAGES:
            st.error("Upload up to 5 photos at a time. / 每次最多上传 5 张图片。")
            valid_batch = False
        elif sum(item.size for item in uploads) > MAX_BATCH_IMAGE_BYTES:
            st.error("Photos must total at most 25 MB. / 图片总大小不能超过 25 MB。")
            valid_batch = False
        else:
            photos = [(item.name, item.getvalue()) for item in uploads]

        context = (
            tuple((name, hashlib.sha256(data).hexdigest()) for name, data in photos),
            settings.active_provider,
            settings.active_vision_model,
        ) if photos else None
        if st.session_state.get(_PHOTO_CONTEXT) != context:
            _clear_photo_result()
            st.session_state[_PHOTO_CONTEXT] = context

        if photos:
            columns = st.columns(min(len(photos), 3))
            seen: set[str] = set()
            for index, (name, data) in enumerate(photos):
                with columns[index % len(columns)]:
                    try:
                        photo = prepare_ingredient_image(data)
                    except InvalidIngredientImage as error:
                        st.error(f"{name}: {error}")
                        valid_batch = False
                    else:
                        st.image(photo.data, caption=f"Photo {index + 1} · 图片 {index + 1}: {name}", width=260)
                        digest = hashlib.sha256(data).hexdigest()
                        if digest not in seen:
                            prepared.append(photo)
                            seen.add(digest)

        provider = "DeepSeek" if settings.active_provider == "deepseek" else "OpenAI"
        if settings.has_valid_llm_key:
            st.caption(
                f"Clicking Recognize sends these photos to {provider} in one request. / "
                f"点击“识别食材”后，全部图片将一次发送给 {provider}，合并为一份清单。"
            )
        else:
            st.info(
                "Photo recognition is not enabled yet. You can enter ingredients below. / "
                "图片识别尚未启用，可在下方手动填写食材。"
            )

        if st.button(
            "🔎 Recognize ingredients · 识别食材",
            key="recognize_ingredient_photo",
            disabled=not prepared or not valid_batch or not settings.has_valid_llm_key,
        ) and prepared and valid_batch:
            _clear_photo_result()
            try:
                with st.spinner("Identifying ingredients… / 正在识别食材…"):
                    key = settings.active_api_key
                    recognizer = ResponsesIngredientRecognizer(
                        model=settings.active_vision_model,
                        api_key=key.get_secret_value() if key is not None else None,
                        base_url=settings.active_base_url,
                        reasoning_effort="none" if settings.active_provider == "deepseek" else None,
                    )
                    st.session_state[_PHOTO_RESULT] = canonicalize_recognized_ingredients(
                        recognizer.recognize(prepared), ingredient_aliases
                    )
            except IngredientRecognitionError as error:
                logger.warning("Ingredient photo recognition failed: %s", error.reason)
                messages = {
                    "authentication": "Photo recognition is not configured correctly. / 图片识别配置有误，请联系应用维护者。",
                    "rate_limit": "Photo recognition is busy or has reached its usage limit. Try again later. / 图片识别繁忙或已达额度，请稍后重试。",
                    "timeout": "Photo recognition timed out. Try fewer or smaller photos. / 图片识别超时，请减少图片数量或大小后重试。",
                    "model_or_request": "The selected AI model cannot process these photos. / 当前 AI 模型无法处理这些图片。",
                    "connection": "The photo service could not be reached. Try again later. / 暂时无法连接图片识别服务，请稍后重试。",
                    "invalid_output": "Some photo labels could not be validated. Try a clearer photo. / 识别结果无法通过校验，请换一张更清晰的图片。",
                }
                st.session_state[_PHOTO_ERROR] = messages.get(
                    error.reason,
                    "Recognition could not be completed. Please try again. / 暂时无法完成识别，请重试。",
                )

        if error_message := st.session_state.get(_PHOTO_ERROR):
            st.error(error_message)
        result = st.session_state.get(_PHOTO_RESULT)
        if result is None:
            return
        if result.discarded_items:
            st.warning(
                f"Omitted {result.discarded_items} uncertain label(s). / "
                f"已忽略 {result.discarded_items} 个无法确认的标签。"
            )
        if not result.ingredients:
            if not result.discarded_items:
                st.info(
                    "No ingredients were identified. Try a clearer photo. / "
                    "未识别到食材，请换一张更清晰的图片。"
                )
            return

        labels = {item.name_en: f"{item.name_zh} · {item.name_en}" for item in result.ingredients}
        st.success(f"Found {len(labels)} ingredient types · 识别到 {len(labels)} 种食材")
        selected = st.multiselect(
            "Select ingredients to use · 选择要使用的食材",
            options=list(labels),
            default=list(labels),
            format_func=labels.__getitem__,
            key=_PHOTO_SELECTION,
        )
        if st.button(
            "Use selected ingredients · 使用所选食材",
            key="use_ingredient_photo_results",
            disabled=not selected,
        ):
            st.session_state["available_ingredients"] = merge_ingredient_text(
                st.session_state["available_ingredients"], selected, ingredient_aliases
            )
            st.success("Added to the ingredient field below. / 已填入下方食材输入框。")
