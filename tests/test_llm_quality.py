import logging
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.engine.translation import dispatcher, groq_translate, llamacpp, lmstudio
from app.engine.translation.quality import (
    LLM_TEMPERATURE,
    LLM_TOP_P,
    chat_messages,
    openai_chat_payload,
    parse_numbered_output,
    repair_failed_outputs,
    needs_google_fallback,
)


class LlmFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_untranslated_fragment_is_repaired_by_google(self):
        regions = [{"text": "お前わざと俺を無視したんだろ！"}]
        with (
            patch.object(
                llamacpp,
                "translate_regions",
                AsyncMock(return_value=[{
                    "text": regions[0]["text"],
                    "translated_text": regions[0]["text"],
                }]),
            ),
            patch.object(
                dispatcher.google,
                "translate_regions",
                AsyncMock(return_value=[{
                    "text": regions[0]["text"],
                    "translated_text": "¡Me ignoraste a propósito!",
                }]),
            ) as google_repair,
        ):
            translated = await dispatcher.translate_regions(
                regions, source_lang="ja", target_lang="es", translator="llamacpp"
            )

        self.assertEqual(
            translated[0]["translated_text"],
            "¡Me ignoraste a propósito!",
        )
        google_repair.assert_awaited_once()

    def test_mana_cannot_disappear_from_english_spanish_translation(self):
        self.assertTrue(
            needs_google_fallback(
                "Mana hardening? A human can do that?",
                "¿Podría un humano hacer eso?",
                "en",
                "es",
            )
        )
        self.assertFalse(
            needs_google_fallback(
                "Mana hardening? A human can do that?",
                "¿Endurecimiento de maná? ¿Un humano puede hacer eso?",
                "en",
                "es",
            )
        )

    def test_model_refusal_is_repaired_instead_of_rendered(self):
        self.assertTrue(
            needs_google_fallback(
                "HE'S TOTALLY FINE!!",
                "No se proporcionó texto legible para traducir.",
                "en",
                "es",
            )
        )

    def test_valid_spanish_can_preserve_manga_names_and_honorifics(self):
        samples = [
            (
                "Is something wrong with Rimu-chan?",
                "¿Algo le pasa a Rimu-chan?",
            ),
            (
                "Rimu-chan is emitting light.",
                "Rimu-chan está emitiendo luz.",
            ),
            (
                "Fufufu, after buying a video camera onboard, I can look "
                "back on these memories anytime!",
                "Fufufu, después de comprar una cámara de video a bordo, "
                "puedo revivir estos recuerdos cuando quiera.",
            ),
        ]
        for source, translated in samples:
            self.assertFalse(
                needs_google_fallback(source, translated, "en", "es"),
                (source, translated),
            )

    def test_retained_english_phrase_still_requests_fallback(self):
        self.assertTrue(
            needs_google_fallback(
                "I will go with you tomorrow.",
                "Iré with you mañana.",
                "en",
                "es",
            )
        )

    def test_llamacpp_output_budget_scales_and_is_capped(self):
        self.assertEqual(llamacpp._output_token_budget([(0, "Hi")]), 64)
        self.assertLess(
            llamacpp._output_token_budget([(0, "Short"), (1, "page")]),
            1024,
        )
        self.assertEqual(
            llamacpp._output_token_budget([(0, "x" * 1000)]),
            640,
        )

    def test_llamacpp_streams_without_logits_stopping_callback(self):
        model = Mock()
        model.create_completion.return_value = iter([
            {"choices": [{"text": "Ho", "finish_reason": None}]},
            {"choices": [{"text": "la", "finish_reason": "stop"}]},
        ])
        with patch.object(llamacpp, "_llama_model", model):
            response = llamacpp._stream_completion("prompt", 16)

        self.assertEqual(response["choices"][0]["text"], "Hola")
        kwargs = model.create_completion.call_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertNotIn("stopping_criteria", kwargs)

    def test_chat_envelope_is_shared_and_explicit(self):
        messages = chat_messages("translate this")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "translate this")

    def test_all_llm_translators_share_groq_sampling_defaults(self):
        groq_payload = openai_chat_payload("llama-3.1-8b-instant", "translate this")
        lmstudio_payload = openai_chat_payload("qwen", "translate this")

        self.assertEqual(groq_payload["temperature"], LLM_TEMPERATURE)
        self.assertEqual(groq_payload["top_p"], LLM_TOP_P)
        self.assertEqual(lmstudio_payload["temperature"], LLM_TEMPERATURE)
        self.assertEqual(lmstudio_payload["top_p"], LLM_TOP_P)

    def test_numbered_parser_accepts_common_formats_and_multiline_text(self):
        raw = """1. Primera línea
continuación
[2] Segunda línea
3: Tercera línea
4) Cuarta línea"""
        self.assertEqual(
            parse_numbered_output(raw, 4),
            {
                0: "Primera línea continuación",
                1: "Segunda línea",
                2: "Tercera línea",
                3: "Cuarta línea",
            },
        )

    def test_missing_number_is_not_shifted_to_another_bubble(self):
        parsed = parse_numbered_output("1. Uno\n3. Tres", 3)
        self.assertEqual(parsed, {0: "Uno", 2: "Tres"})
        self.assertNotIn(1, parsed)

    async def test_shared_quality_gate_repairs_only_rejected_outputs(self):
        results = ["Hello there", "Todo está bien"]
        indexed = [(0, "Hello there"), (1, "Everything is fine")]
        with patch(
            "app.engine.translation.google.translate_text",
            new=AsyncMock(return_value="Hola"),
        ) as fallback:
            repaired = await repair_failed_outputs(
                results, indexed, "en", "es", "test", logging.getLogger("test")
            )
        self.assertEqual(repaired, ["Hola", "Todo está bien"])
        fallback.assert_awaited_once_with("Hello there", "en", "es")

    async def test_shared_quality_gate_rejects_fallback_that_copies_source(self):
        results = ["日本語の原文"]
        indexed = [(0, "日本語の原文")]
        with patch(
            "app.engine.translation.google.translate_text",
            new=AsyncMock(return_value="日本語の原文"),
        ):
            repaired = await repair_failed_outputs(
                results, indexed, "ja", "es", "test", logging.getLogger("test")
            )
        self.assertEqual(repaired, ["日本語の原文"])

    async def test_lmstudio_uses_the_same_chat_envelope_as_groq(self):
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hola"}}]
        }
        client = Mock()
        client.post = AsyncMock(return_value=response)

        with (
            patch.object(lmstudio, "get_client", AsyncMock(return_value=client)),
            patch.object(lmstudio, "resolve_model", AsyncMock(return_value="qwen")),
        ):
            translated = await lmstudio.translate_text(
                "Hello", "en", "es", host="http://lmstudio:1234/v1", model="qwen"
            )

        self.assertEqual(translated, "Hola")
        url = client.post.await_args.args[0]
        payload = client.post.await_args.kwargs["json"]
        self.assertEqual(url, "http://lmstudio:1234/v1/chat/completions")
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["system", "user"],
        )

    def test_lmstudio_api_base_does_not_duplicate_v1_segment(self):
        self.assertEqual(
            lmstudio._lmstudio_api_url(
                "http://lmstudio-host:1234/v1", "chat/completions"
            ),
            "http://lmstudio-host:1234/v1/chat/completions",
        )

    async def test_groq_auto_selects_an_available_model(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "data": [
                {"id": "whisper-large-v3"},
                {"id": "llama-3.1-8b-instant"},
            ]
        }
        client = Mock()
        client.get = AsyncMock(return_value=response)
        groq_translate._resolved_models.clear()

        with patch.object(
            groq_translate, "get_client", AsyncMock(return_value=client)
        ):
            selected = await groq_translate.resolve_model("secret", "auto")

        self.assertEqual(selected, "llama-3.1-8b-instant")

    async def test_groq_replaces_a_retired_requested_model(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "data": [{"id": "qwen/qwen3-32b"}]
        }
        client = Mock()
        client.get = AsyncMock(return_value=response)
        groq_translate._resolved_models.clear()

        with patch.object(
            groq_translate, "get_client", AsyncMock(return_value=client)
        ):
            selected = await groq_translate.resolve_model(
                "secret", "llama-3.3-70b-versatile"
            )

        self.assertEqual(selected, "qwen/qwen3-32b")

    async def test_lmstudio_auto_selects_a_loaded_model(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "data": [
                {"id": "qwen2.5-coder-3b"},
                {"id": "qwen3.5-2b"},
            ]
        }
        client = Mock()
        client.get = AsyncMock(return_value=response)
        lmstudio._resolved_models.clear()

        with patch.object(lmstudio, "get_client", AsyncMock(return_value=client)):
            selected = await lmstudio.resolve_model(
                "http://lmstudio-host:1234/v1", "auto"
            )

        self.assertEqual(selected, "qwen2.5-coder-3b")
        self.assertEqual(
            lmstudio._lmstudio_api_url("http://lmstudio-host:1234", "models"),
            "http://lmstudio-host:1234/v1/models",
        )

    async def test_llamacpp_uses_translation_model_native_completion(self):
        model = Mock()
        model.create_completion.return_value = iter([
            {"choices": [{"text": "Hola", "finish_reason": "stop"}]},
        ])
        with (
            patch.object(llamacpp, "_llama_model", model),
            patch.object(llamacpp, "_load_local_model", AsyncMock(return_value=model)),
        ):
            translated = await llamacpp.translate_text("Hello", "en", "es")

        self.assertEqual(translated, "Hola")
        kwargs = model.create_completion.call_args.kwargs
        self.assertNotIn("messages", kwargs)
        self.assertIn("Translate the following segment into Spanish", kwargs["prompt"])
        self.assertEqual(kwargs["top_k"], llamacpp.HY_TOP_K)
        self.assertEqual(kwargs["top_p"], llamacpp.HY_TOP_P)

    def test_llamacpp_batch_prompt_preserves_region_contract(self):
        prompt = llamacpp._batch_translation_prompt("es", "1. Uno\n2. Dos")
        self.assertEqual(
            prompt,
            "Translate the following segment into Spanish, without additional "
            "explanation.\n\n1. Uno\n2. Dos",
        )
        self.assertIn("1. Uno\n2. Dos", prompt)

    def test_llamacpp_names_korean_source_without_changing_default_prompt(self):
        korean = llamacpp._translation_prompt("es", "안녕하세요", "ko")
        default = llamacpp._translation_prompt("es", "Hello")
        self.assertIn("from Korean into Spanish", korean)
        self.assertNotIn("from Korean", default)

    def test_llamacpp_names_chinese_source(self):
        prompt = llamacpp._batch_translation_prompt("es", "1. 你好", "zh")
        self.assertIn("from Chinese into Spanish", prompt)

    def test_llamacpp_corrects_only_invalid_san_honorific(self):
        self.assertEqual(
            llamacpp._fix_japanese_honorific(
                "バルトロメウスさん", "El padre Bartolomé vendrá."
            ),
            "El señor Bartolomé vendrá.",
        )
        self.assertEqual(
            llamacpp._fix_japanese_honorific(
                "保護者みたいなものです", "Soy como un padre de familia."
            ),
            "Soy como un padre de familia.",
        )


if __name__ == "__main__":
    unittest.main()
