"""Exercise upload, recognition, selection and the existing recipe UI locally."""

from __future__ import annotations

import os
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from fridge_rescue.image_recognition import IngredientRecognitionError, IngredientRecognitionResult

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def photo(color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format="PNG")
    return output.getvalue()


def result() -> IngredientRecognitionResult:
    return IngredientRecognitionResult.model_validate({
        "ingredients": [
            {"name_en": "tomato", "name_zh": "番茄"},
            {"name_en": "egg", "name_zh": "鸡蛋"},
        ]
    })


def app(*, enabled: bool = True) -> AppTest:
    key = "test-key" if enabled else ""
    source = f'''
import streamlit as st
from fridge_rescue.settings import Settings
from fridge_rescue.image_recognition_ui import render_image_ingredient_input
settings = Settings(_env_file=None, llm_provider="deepseek", deepseek_api_key={key!r},
                    openai_api_key=None, vision_model=None)
render_image_ingredient_input(settings, {{"番茄": "tomato", "tomatoes": "tomato"}})
with st.container(border=True):
    st.text_input("Available ingredients", key="available_ingredients")
    st.button("Search recipes")
'''
    return AppTest.from_string(source, default_timeout=20).run()


class PhotoFlowTests(unittest.TestCase):
    def assert_healthy(self, at: AppTest) -> None:
        self.assertFalse(at.exception, [item.message for item in at.exception])

    def test_missing_key_keeps_manual_entry_available_and_recognition_disabled(self) -> None:
        at = app(enabled=False)
        at.file_uploader[0].set_value(("ingredients.png", photo(), "image/png")).run()
        self.assert_healthy(at)
        self.assertTrue(at.button(key="recognize_ingredient_photo").disabled)
        self.assertEqual(len(at.text_input), 1)

    def test_upload_alone_does_not_contact_the_model(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            at = app()
            at.file_uploader[0].set_value(("ingredients.png", photo(), "image/png")).run()
            self.assert_healthy(at)
            factory.assert_not_called()
            self.assertFalse(at.button(key="recognize_ingredient_photo").disabled)

    def test_recognition_returns_names_and_survives_unrelated_reruns(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            factory.return_value.recognize.return_value = result()
            at = app()
            at.file_uploader[0].set_value(("ingredients.png", photo(), "image/png")).run()
            at.button(key="recognize_ingredient_photo").click().run()
            self.assert_healthy(at)
            self.assertEqual(at.multiselect[0].value, ["tomato", "egg"])
            self.assertIn("番茄 · tomato", at.multiselect[0].options)
            at.run()
            factory.return_value.recognize.assert_called_once()
            self.assertEqual(at.multiselect[0].value, ["tomato", "egg"])

    def test_multiple_photos_are_recognized_together_and_merge_with_manual_input(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            factory.return_value.recognize.return_value = result()
            at = app()
            at.text_input[0].set_value("rice")
            at.file_uploader[0].set_value([
                ("fridge.png", photo("blue"), "image/png"),
                ("vegetables.png", photo("red"), "image/png"),
                ("seasoning.png", photo("green"), "image/png"),
            ]).run()
            at.button(key="recognize_ingredient_photo").click().run()
            self.assert_healthy(at)
            factory.return_value.recognize.assert_called_once()
            self.assertEqual(len(factory.return_value.recognize.call_args.args[0]), 3)
            at.button(key="use_ingredient_photo_results").click().run()
            self.assertEqual(at.text_input[0].value, "rice, tomato, egg")

    def test_one_corrupt_photo_blocks_partial_batch_results(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            at = app()
            at.file_uploader[0].set_value([
                ("valid.png", photo(), "image/png"), ("bad.png", b"bad", "image/png")
            ]).run()
            self.assert_healthy(at)
            self.assertTrue(at.error)
            self.assertTrue(at.button(key="recognize_ingredient_photo").disabled)
            factory.assert_not_called()

    def test_six_photos_are_rejected_before_calling_provider(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            at = app()
            at.file_uploader[0].set_value([
                (f"photo{index}.png", photo(), "image/png") for index in range(6)
            ]).run()
            self.assert_healthy(at)
            self.assertTrue(at.button(key="recognize_ingredient_photo").disabled)
            self.assertTrue(at.error)
            factory.assert_not_called()

    def test_selected_names_merge_into_existing_manual_input(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            factory.return_value.recognize.return_value = result()
            at = app()
            at.text_input[0].set_value("番茄, rice")
            at.button[-1].click().run()
            at.file_uploader[0].set_value(("ingredients.png", photo(), "image/png")).run()
            at.button(key="recognize_ingredient_photo").click().run()
            at.button(key="use_ingredient_photo_results").click().run()
            self.assert_healthy(at)
            self.assertEqual(at.text_input[0].value, "番茄, rice, egg")
            self.assertEqual(factory.call_args.kwargs["model"], "deepseek-flash")

    def test_manual_input_is_preserved_without_submitting_recipe_search_first(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            factory.return_value.recognize.return_value = result()
            at = app()
            at.text_input[0].set_value("rice")
            at.file_uploader[0].set_value(("ingredients.png", photo(), "image/png")).run()
            at.button(key="recognize_ingredient_photo").click().run()
            at.button(key="use_ingredient_photo_results").click().run()
            self.assert_healthy(at)
            self.assertEqual(at.text_input[0].value, "rice, tomato, egg")

    def test_deselected_names_are_not_added(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            factory.return_value.recognize.return_value = result()
            at = app()
            at.file_uploader[0].set_value(("ingredients.png", photo(), "image/png")).run()
            at.button(key="recognize_ingredient_photo").click().run()
            at.multiselect[0].set_value(["egg"]).run()
            at.button(key="use_ingredient_photo_results").click().run()
            self.assertEqual(at.text_input[0].value, "egg")
            at.multiselect[0].set_value([]).run()
            self.assertTrue(at.button(key="use_ingredient_photo_results").disabled)

    def test_replacing_or_removing_a_photo_clears_old_results(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            factory.return_value.recognize.return_value = result()
            at = app()
            at.file_uploader[0].set_value(("ingredients.png", photo(), "image/png")).run()
            at.button(key="recognize_ingredient_photo").click().run()
            at.file_uploader[0].set_value(("new.png", photo("blue"), "image/png")).run()
            self.assertFalse(at.multiselect)
            self.assertNotIn("ingredient_photo_result", at.session_state)
            at.file_uploader[0].set_value(None).run()
            self.assertTrue(at.button(key="recognize_ingredient_photo").disabled)
            factory.return_value.recognize.assert_called_once()

    def test_corrupt_image_cannot_start_recognition(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            at = app()
            at.file_uploader[0].set_value(("bad.png", b"not an image", "image/png")).run()
            self.assert_healthy(at)
            self.assertTrue(at.error)
            self.assertTrue(at.button(key="recognize_ingredient_photo").disabled)
            factory.assert_not_called()

    def test_no_food_result_has_an_explicit_empty_state(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            factory.return_value.recognize.return_value = IngredientRecognitionResult(ingredients=[])
            at = app()
            at.file_uploader[0].set_value(("empty.png", photo(), "image/png")).run()
            at.button(key="recognize_ingredient_photo").click().run()
            self.assert_healthy(at)
            self.assertFalse(at.multiselect)
            self.assertTrue(any("No ingredients were identified" in item.value for item in at.info))

    def test_service_error_is_readable_and_retryable(self) -> None:
        with patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory:
            factory.return_value.recognize.side_effect = IngredientRecognitionError("PRIVATE_DETAILS")
            at = app()
            at.file_uploader[0].set_value(("ingredients.png", photo(), "image/png")).run()
            with self.assertLogs("fridge_rescue.image_recognition_ui", level="WARNING") as logs:
                at.button(key="recognize_ingredient_photo").click().run()
            self.assert_healthy(at)
            self.assertTrue(at.error)
            self.assertNotIn("PRIVATE_DETAILS", at.error[0].value)
            self.assertNotIn("PRIVATE_DETAILS", " ".join(logs.output))
            factory.return_value.recognize.side_effect = None
            factory.return_value.recognize.return_value = result()
            at.button(key="recognize_ingredient_photo").click().run()
            self.assertFalse(at.error)
            self.assertEqual(at.multiselect[0].value, ["tomato", "egg"])


class ExistingAppRegressionTests(unittest.TestCase):
    def test_manual_english_and_chinese_recipe_search_still_works(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""}):
            at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
            self.assertFalse(at.exception)
            for ingredients in ("egg, tomato", "鸡蛋，番茄"):
                at.text_input(key="available_ingredients").set_value(ingredients)
                at.button(key="find_recipes").click().run()
                self.assertFalse(at.exception, [item.message for item in at.exception])
                self.assertFalse(at.error, [item.value for item in at.error])
                self.assertEqual(len(at.subheader), 3)

    def test_recognized_names_can_drive_a_real_local_recipe_search(self) -> None:
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key", "OPENAI_API_KEY": ""}),
            patch("fridge_rescue.image_recognition_ui.ResponsesIngredientRecognizer") as factory,
        ):
            factory.return_value.recognize.return_value = result()
            at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
            at.radio[0].set_value("Offline retrieval preview").run()
            at.file_uploader[0].set_value(("ingredients.png", photo(), "image/png")).run()
            at.button(key="recognize_ingredient_photo").click().run()
            at.button(key="use_ingredient_photo_results").click().run()
            self.assertEqual(at.text_input(key="available_ingredients").value, "tomato, egg")
            at.button(key="find_recipes").click().run()
            self.assertFalse(at.exception, [item.message for item in at.exception])
            self.assertFalse(at.error, [item.value for item in at.error])
            self.assertEqual(len(at.subheader), 3)
            factory.return_value.recognize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
