"""Exercise user confirmation, combined filters, graph queries and the agentic UI."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from fridge_rescue.cooking_constraints import ConstraintParsingError, ParsedCookingConstraints
from fridge_rescue.schemas import LLMRecommendation

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def parsed_request() -> ParsedCookingConstraints:
    return ParsedCookingConstraints(
        available_ingredients=["egg", "tomato"], excluded_ingredients=["peanut"],
        max_time_minutes=30, diet_preference="vegetarian", unavailable_equipment=["oven"],
    )


def constraint_app(*, enabled: bool = True) -> AppTest:
    return AppTest.from_string(f'''
import streamlit as st
from fridge_rescue.cooking_constraints_ui import initialize_cooking_inputs, render_natural_language_input, render_cooking_filters
from fridge_rescue.settings import Settings
initialize_cooking_inputs()
settings = Settings(_env_file=None, llm_provider="deepseek", deepseek_api_key={"test-key" if enabled else ""!r})
render_natural_language_input(settings, {{"番茄": "tomato", "tomatoes": "tomato"}})
st.text_input("Available ingredients", key="available_ingredients")
st.text_input("Excluded ingredients", key="excluded_ingredients")
render_cooking_filters()
''', default_timeout=20).run()


class LanguageReviewTests(unittest.TestCase):
    def assert_healthy(self, at: AppTest) -> None:
        self.assertFalse(at.exception, [item.message for item in at.exception])

    def test_parse_requires_confirmation_then_updates_editable_fields(self) -> None:
        with patch("fridge_rescue.cooking_constraints_ui.ResponsesConstraintParser") as factory:
            factory.return_value.parse.return_value = parsed_request()
            at = constraint_app()
            at.text_input(key="available_ingredients").set_value("rice, 番茄")
            at.text_area(key="natural_language_request").set_value("有鸡蛋和番茄，30分钟内，素食，没有烤箱").run()
            at.button(key="parse_cooking_request").click().run()
            self.assert_healthy(at)
            self.assertEqual(at.text_input(key="available_ingredients").value, "rice, 番茄")
            self.assertFalse(at.checkbox(key="time_limit_enabled").value)
            at.button(key="apply_cooking_request").click().run()
            self.assert_healthy(at)
            self.assertEqual(at.text_input(key="available_ingredients").value, "rice, 番茄, egg")
            self.assertEqual(at.text_input(key="excluded_ingredients").value, "peanut")
            self.assertEqual(at.number_input(key="max_time_minutes").value, 30)
            self.assertTrue(at.checkbox(key="time_limit_enabled").value)
            self.assertEqual(at.selectbox(key="diet_preference").value, "vegetarian")
            self.assertNotIn("oven", at.multiselect(key="available_equipment").value)
            factory.return_value.parse.assert_called_once()
            at.run()
            factory.return_value.parse.assert_called_once()

    def test_editing_description_clears_old_parse(self) -> None:
        with patch("fridge_rescue.cooking_constraints_ui.ResponsesConstraintParser") as factory:
            factory.return_value.parse.return_value = parsed_request()
            at = constraint_app()
            at.text_area[0].set_value("30分钟内").run()
            at.button(key="parse_cooking_request").click().run()
            at.text_area[0].set_value("15分钟内").run()
            self.assert_healthy(at)
            self.assertNotIn("constraint_parse_result", at.session_state)
            self.assertFalse(any(item.key == "apply_cooking_request" for item in at.button))

    def test_ambiguous_parse_displays_question_and_blocks_apply(self) -> None:
        with patch("fridge_rescue.cooking_constraints_ui.ResponsesConstraintParser") as factory:
            factory.return_value.parse.return_value = ParsedCookingConstraints(clarifications=[{
                "english": "You only have a microwave. Can you also use a stovetop?",
                "chinese": "只有微波炉，是否也能使用灶台？",
            }])
            at = constraint_app()
            at.text_area[0].set_value("只有微波炉").run()
            at.button(key="parse_cooking_request").click().run()
            self.assertTrue(at.warning)
            self.assertIn("Can you also use a stovetop?", at.warning[0].value)
            self.assertIn("是否也能使用灶台", at.warning[0].value)
            self.assertTrue(at.button(key="apply_cooking_request").disabled)

    def test_missing_key_keeps_manual_filters_usable(self) -> None:
        at = constraint_app(enabled=False)
        at.text_area[0].set_value("30分钟内").run()
        self.assertTrue(at.button(key="parse_cooking_request").disabled)
        at.checkbox(key="time_limit_enabled").check().run()
        self.assertFalse(at.number_input(key="max_time_minutes").disabled)

    def test_provider_failure_does_not_leak_details_or_change_existing_filters(self) -> None:
        with patch("fridge_rescue.cooking_constraints_ui.ResponsesConstraintParser") as factory:
            factory.return_value.parse.side_effect = ConstraintParsingError("PRIVATE_DETAILS")
            at = constraint_app()
            at.text_area[0].set_value("有鸡蛋").run()
            with self.assertLogs("fridge_rescue.cooking_constraints_ui", level="WARNING") as logs:
                at.button(key="parse_cooking_request").click().run()
            self.assert_healthy(at)
            self.assertTrue(at.error)
            self.assertNotIn("PRIVATE_DETAILS", at.error[0].value)
            self.assertNotIn("PRIVATE_DETAILS", " ".join(logs.output))
            self.assertEqual(at.text_input(key="available_ingredients").value, "")


class IntegratedAppTests(unittest.TestCase):
    def local_app(self) -> AppTest:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertFalse(at.exception)
        return at

    def test_combined_filters_return_real_corpus_recipes_with_estimate_labels(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""}):
            at = self.local_app()
            at.text_input(key="available_ingredients").set_value("鸡蛋，番茄")
            at.checkbox(key="time_limit_enabled").check()
            at.selectbox(key="diet_preference").set_value("vegetarian")
            at.checkbox(key="equipment_filter_enabled").check().run()
            at.multiselect(key="available_equipment").set_value(["microwave"])
            at.button(key="find_recipes").click().run()
            self.assertFalse(at.exception, [item.message for item in at.exception])
            self.assertFalse(at.error)
            self.assertTrue(at.subheader)
            self.assertTrue(any("步骤估算" in item.value for item in at.caption))
            self.assertTrue(any("食材规则判定" in item.value for item in at.caption))

    def test_disabling_estimates_explains_empty_time_filter_result(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""}):
            at = self.local_app()
            at.text_input(key="available_ingredients").set_value("egg, tomato")
            at.checkbox(key="time_limit_enabled").check().run()
            at.checkbox(key="allow_estimated_time").uncheck()
            at.button(key="find_recipes").click().run()
            self.assertFalse(at.exception)
            self.assertFalse(at.subheader)
            self.assertTrue(any("No recipe matches" in item.value for item in at.warning))

    def test_graph_recipe_query_returns_source_paths_and_invalidates_on_filter_change(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""}):
            at = self.local_app()
            at.text_input(key="ingredient_graph_query").set_value("鸡蛋，番茄").run()
            at.checkbox(key="graph_require_all").check()
            at.button(key="run_ingredient_graph_query").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(at.dataframe)
            rows = at.dataframe[-1].value
            self.assertIn("Original ingredient line · 原始食材行", rows.columns)
            self.assertTrue(rows["Recipe ID"].str.startswith("epicurious-").all())
            at.checkbox(key="time_limit_enabled").check().run()
            self.assertNotIn("ingredient_graph_result", at.session_state)

    def test_graph_companion_and_reviewed_substitution_queries(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""}):
            at = self.local_app()
            at.selectbox(key="ingredient_graph_tool").set_value("companions")
            at.text_input(key="ingredient_graph_query").set_value("tomato").run()
            at.button(key="run_ingredient_graph_query").click().run()
            self.assertFalse(at.exception)
            self.assertIn("Shared recipe count · 共同菜谱数", at.dataframe[-1].value.columns)
            at.selectbox(key="ingredient_graph_tool").set_value("substitutions")
            at.text_input(key="ingredient_graph_query").set_value("黄油").run()
            at.button(key="run_ingredient_graph_query").click().run()
            self.assertFalse(at.exception)
            rows = at.dataframe[-1].value
            self.assertEqual(rows.iloc[0]["Possible substitute · 可考虑替代"], "olive oil")
            self.assertIn("baking", rows.iloc[0]["Notes · 注意事项"])

    def test_reviewed_language_request_drives_agentic_graph_and_recipe_selection(self) -> None:
        def select(request, candidates, **kwargs):
            return LLMRecommendation(selections=[
                {"recipe_id": item.recipe_id, "reason": "Matches the requested ingredients."}
                for item in candidates
            ])

        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key", "OPENAI_API_KEY": ""}),
            patch("fridge_rescue.cooking_constraints_ui.ResponsesConstraintParser") as parser,
            patch("fridge_rescue.openai_selector.ResponsesRecommendationSelector") as selector,
        ):
            parser.return_value.parse.return_value = parsed_request()
            selector.return_value.select.side_effect = select
            at = self.local_app()
            at.text_area(key="natural_language_request").set_value("有鸡蛋和番茄，30分钟内，素食，不要花生，没有烤箱").run()
            at.button(key="parse_cooking_request").click().run()
            at.button(key="apply_cooking_request").click().run()
            at.button(key="find_recipes").click().run()
            self.assertFalse(at.exception, [item.message for item in at.exception])
            self.assertFalse(at.error, [item.value for item in at.error])
            self.assertEqual(len(at.subheader), 3)
            request = selector.return_value.select.call_args.args[0]
            self.assertEqual(request.max_time_minutes, 30)
            self.assertEqual(request.diet_preference, "vegetarian")
            self.assertIn("oven", request.unavailable_equipment)
            self.assertIn("peanut", request.excluded_ingredients)
            selector.return_value.select.assert_called_once()
            self.assertTrue(at.dataframe)


if __name__ == "__main__":
    unittest.main()
