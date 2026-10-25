"""Photo processing and vision API contract checks; no external requests."""

from __future__ import annotations

import base64
import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx2
from openai import OpenAI
from PIL import Image
from pydantic import ValidationError

from fridge_rescue.image_recognition import (
    MAX_IMAGE_BYTES,
    IngredientRecognitionError,
    IngredientRecognitionResult,
    InvalidIngredientImage,
    ResponsesIngredientRecognizer,
    prepare_ingredient_image,
    canonicalize_recognized_ingredients,
)
from fridge_rescue.input_normalization import merge_ingredient_text, parse_ingredient_text
from fridge_rescue.settings import Settings


def image_bytes(format: str = "PNG", *, size: tuple[int, int] = (32, 24)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "tomato").save(output, format=format)
    return output.getvalue()


def ingredient_payload() -> dict:
    return {
        "ingredients": [
            {"name_en": "tomato", "name_zh": "番茄"},
            {"name_en": "egg", "name_zh": "鸡蛋"},
        ]
    }


class PrepareImageTests(unittest.TestCase):
    def test_supported_formats_decode_to_jpeg(self) -> None:
        for format in ("PNG", "JPEG", "WEBP"):
            with self.subTest(format=format):
                prepared = prepare_ingredient_image(image_bytes(format))
                with Image.open(BytesIO(prepared.data)) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (32, 24))

    def test_empty_corrupt_and_unsupported_images_are_rejected(self) -> None:
        for data in (b"", b"<html>not a photo</html>", image_bytes("GIF")):
            with self.subTest(data_length=len(data)), self.assertRaises(InvalidIngredientImage):
                prepare_ingredient_image(data)

    def test_truncated_image_is_rejected(self) -> None:
        data = image_bytes("JPEG", size=(100, 100))
        with self.assertRaises(InvalidIngredientImage):
            prepare_ingredient_image(data[: len(data) // 2])

    def test_upload_size_limit_is_enforced_before_decode(self) -> None:
        with self.assertRaises(InvalidIngredientImage):
            prepare_ingredient_image(b"x" * (MAX_IMAGE_BYTES + 1))

    def test_decoded_pixel_limit_is_enforced(self) -> None:
        with patch("fridge_rescue.image_recognition.MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(InvalidIngredientImage):
                prepare_ingredient_image(image_bytes())

    def test_decompression_bomb_is_reported_as_an_invalid_photo(self) -> None:
        data = image_bytes()
        with patch.object(Image, "MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(InvalidIngredientImage):
                prepare_ingredient_image(data)

    def test_resize_preserves_aspect_ratio(self) -> None:
        prepared = prepare_ingredient_image(image_bytes(size=(2400, 1200)))
        with Image.open(BytesIO(prepared.data)) as image:
            self.assertEqual(image.size, (1600, 800))

    def test_exif_orientation_is_applied_and_metadata_removed(self) -> None:
        output = BytesIO()
        source = Image.new("RGB", (40, 20), "green")
        exif = Image.Exif()
        exif[274] = 6
        exif[315] = "private photographer name"
        source.save(output, format="JPEG", exif=exif)
        prepared = prepare_ingredient_image(output.getvalue())
        with Image.open(BytesIO(prepared.data)) as image:
            self.assertEqual(image.size, (20, 40))
            self.assertFalse(image.getexif())
        self.assertNotIn(b"private photographer name", prepared.data)

    def test_transparent_pixels_are_flattened_on_white(self) -> None:
        output = BytesIO()
        Image.new("RGBA", (32, 24), (255, 0, 0, 0)).save(output, format="PNG")
        prepared = prepare_ingredient_image(output.getvalue())
        with Image.open(BytesIO(prepared.data)) as image:
            self.assertTrue(all(channel >= 250 for channel in image.getpixel((0, 0))))

    def test_animated_webp_is_rejected(self) -> None:
        output = BytesIO()
        first = Image.new("RGB", (16, 16), "red")
        first.save(
            output,
            format="WEBP",
            save_all=True,
            append_images=[Image.new("RGB", (16, 16), "blue")],
            duration=100,
            loop=0,
        )
        with self.assertRaises(InvalidIngredientImage):
            prepare_ingredient_image(output.getvalue())


class VisionResponseTests(unittest.TestCase):
    def recognizer(self, payload: object, *, status: str = "completed") -> tuple:
        parse = Mock(return_value=SimpleNamespace(output_parsed=payload, status=status))
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        return ResponsesIngredientRecognizer(model="deepseek-flash", client=client), parse

    def test_returns_bilingual_names_and_deduplicates_types(self) -> None:
        payload = ingredient_payload()
        payload["ingredients"].append({"name_en": " Tomato ", "name_zh": "西红柿"})
        recognizer, parse = self.recognizer(payload)
        result = recognizer.recognize(prepare_ingredient_image(image_bytes()))
        self.assertEqual([item.name_en for item in result.ingredients], ["tomato", "egg"])
        self.assertEqual(result.ingredients[0].name_zh, "番茄")
        parse.assert_called_once()

    def test_empty_result_is_valid(self) -> None:
        recognizer, _ = self.recognizer({"ingredients": []})
        self.assertEqual(recognizer.recognize(prepare_ingredient_image(image_bytes())).ingredients, [])

    def test_multiple_photos_use_one_request_and_all_image_blocks(self) -> None:
        recognizer, parse = self.recognizer(ingredient_payload())
        images = [prepare_ingredient_image(image_bytes()) for _ in range(3)]
        recognizer.recognize(images)
        parse.assert_called_once()
        content = parse.call_args.kwargs["input"][1]["content"]
        self.assertEqual(len(content), 4)
        self.assertTrue(all(part["type"] == "input_image" for part in content[1:]))

    def test_invalid_batch_sizes_never_call_the_provider(self) -> None:
        recognizer, parse = self.recognizer(ingredient_payload())
        prepared = prepare_ingredient_image(image_bytes())
        for images in ([], [prepared] * 6):
            with self.assertRaises(InvalidIngredientImage):
                recognizer.recognize(images)
        parse.assert_not_called()

    def test_batch_synonyms_are_merged_using_local_aliases(self) -> None:
        result = IngredientRecognitionResult.model_validate({"ingredients": [
            {"name_en": "tomatoes", "name_zh": "番茄"},
            {"name_en": "tomato", "name_zh": "西红柿"},
            {"name_en": "egg", "name_zh": "鸡蛋"},
        ]})
        canonical = canonicalize_recognized_ingredients(result, {"tomatoes": "tomato"})
        self.assertEqual([item.name_en for item in canonical.ingredients], ["tomato", "egg"])

    def test_refused_and_incomplete_responses_are_rejected(self) -> None:
        for payload, status in ((None, "completed"), (ingredient_payload(), "incomplete")):
            with self.subTest(status=status):
                recognizer, _ = self.recognizer(payload, status=status)
                with self.assertRaises(IngredientRecognitionError):
                    recognizer.recognize(prepare_ingredient_image(image_bytes()))

    def test_malformed_names_and_extra_fields_are_rejected(self) -> None:
        invalid = [
            {"ingredients": [{"name_en": "", "name_zh": "鸡蛋"}]},
            {"ingredients": [{"name_en": "鸡蛋", "name_zh": "鸡蛋"}]},
            {"ingredients": [{"name_en": 12, "name_zh": "鸡蛋"}]},
            {"ingredients": [{"name_en": "egg, milk", "name_zh": "鸡蛋"}]},
            {"ingredients": [{"name_en": "egg", "name_zh": "  "}]},
            {"ingredients": [], "recipe": "unexpected extra content"},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                recognizer, _ = self.recognizer(payload)
                with self.assertRaises(IngredientRecognitionError):
                    recognizer.recognize(prepare_ingredient_image(image_bytes()))

    def test_service_failure_does_not_expose_request_details(self) -> None:
        recognizer, parse = self.recognizer(None)
        parse.side_effect = RuntimeError("PRIVATE_IMAGE_AND_KEY")
        with self.assertRaises(IngredientRecognitionError) as caught:
            recognizer.recognize(prepare_ingredient_image(image_bytes()))
        self.assertNotIn("PRIVATE_IMAGE_AND_KEY", str(caught.exception))

    def test_result_has_a_bounded_number_of_ingredients(self) -> None:
        with self.assertRaises(ValidationError):
            IngredientRecognitionResult.model_validate({
                "ingredients": [{"name_en": "egg", "name_zh": "鸡蛋"}] * 41
            })

    def test_real_sdk_serializes_image_and_parses_structured_response(self) -> None:
        requests: list[dict] = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            self.assertEqual(request.url.path, "/v1/responses")
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "deepseek-flash",
                "output": [{
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": json.dumps(ingredient_payload()),
                        "annotations": [],
                    }],
                }],
            })

        prepared = prepare_ingredient_image(image_bytes())
        with OpenAI(
            api_key="test-key",
            base_url="https://vision.test/v1",
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
            max_retries=0,
        ) as client:
            recognizer = ResponsesIngredientRecognizer(
                model="deepseek-flash", client=client, reasoning_effort="none"
            )
            result = recognizer.recognize(prepared)
        self.assertEqual(len(result.ingredients), 2)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertFalse(request["store"])
        self.assertEqual(request["reasoning"], {"effort": "none"})
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])
        image_part = request["input"][1]["content"][1]
        self.assertEqual(image_part["type"], "input_image")
        self.assertEqual(base64.b64decode(image_part["image_url"].split(",", 1)[1]), prepared.data)


class PhotoConfigurationTests(unittest.TestCase):
    def test_deepseek_vision_uses_flash_even_with_a_text_only_recipe_model(self) -> None:
        settings = Settings(
            _env_file=None, llm_provider="deepseek", deepseek_model="deepseek-v4-pro",
            vision_model=None,
        )
        self.assertEqual(settings.active_vision_model, "deepseek-flash")

    def test_openai_reuses_its_configured_model(self) -> None:
        settings = Settings(
            _env_file=None, llm_provider="openai", openai_model="configured-vision-model",
            vision_model=None,
        )
        self.assertEqual(settings.active_vision_model, "configured-vision-model")

    def test_optional_override_is_trimmed_and_blank_is_unset(self) -> None:
        self.assertEqual(
            Settings(_env_file=None, vision_model="  custom-vision  ").active_vision_model,
            "custom-vision",
        )
        self.assertIsNone(Settings(_env_file=None, vision_model="  ").vision_model)

    def test_merging_preserves_manual_names_and_deduplicates_bilingual_aliases(self) -> None:
        aliases = {"番茄": "tomato", "tomatoes": "tomato", "eggs": "egg"}
        merged = merge_ingredient_text("番茄， rice; eggs", ["tomato", "egg", "spinach"], aliases)
        self.assertEqual(merged, "番茄, rice, eggs, spinach")
        self.assertEqual(parse_ingredient_text(merged, aliases), ["tomato", "rice", "egg", "spinach"])

    def test_unknown_manual_chinese_is_preserved_for_normal_input_validation(self) -> None:
        merged = merge_ingredient_text("自选食材", ["egg"], {})
        self.assertEqual(merged, "自选食材, egg")


if __name__ == "__main__":
    unittest.main()
