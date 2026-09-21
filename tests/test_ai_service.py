import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from ai import AIError, OpenAIFAQ
from config import Settings, read_env
from faq_engine import FAQ_FILE, UNKNOWN, load_faq
from service import BotService


def api_result(selection):
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(selection)},
        ]}],
    }


class AIResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.faq = load_faq(FAQ_FILE)

    def test_valid_selection_uses_only_original_answers(self):
        result = api_result({"faq_ids": [5, 1, 5]})
        self.assertEqual(OpenAIFAQ._parse_result(result, self.faq),
                         f"{self.faq[4][1]}\n\n{self.faq[0][1]}")

    def test_empty_selection_returns_unknown(self):
        self.assertEqual(OpenAIFAQ._parse_result(api_result({"faq_ids": []}), self.faq), UNKNOWN)

    def test_rejects_invalid_ids_and_shapes(self):
        invalid = (
            {"faq_ids": [0]}, {"faq_ids": [6]}, {"faq_ids": [-1]},
            {"faq_ids": [True]}, {"faq_ids": [False]}, {"faq_ids": [1.0]},
            {"faq_ids": ["1"]}, {"faq_ids": [None]}, {"faq_ids": [1] * 6},
            {"faq_ids": None}, {"faq_ids": "1"}, {"faq_ids": {}},
            {"faq_ids": [1], "answer": "invented"}, {}, [], None,
        )
        for selection in invalid:
            with self.subTest(selection=selection):
                with self.assertRaises(AIError):
                    OpenAIFAQ._parse_result(api_result(selection), self.faq)

    def test_refusal_returns_unknown(self):
        result = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "refusal", "refusal": "Refused"},
        ]}]}
        self.assertEqual(OpenAIFAQ._parse_result(result, self.faq), UNKNOWN)

    def test_incomplete_response_is_not_used(self):
        for status in ("incomplete", "failed", "in_progress", None):
            with self.subTest(status=status):
                result = api_result({"faq_ids": [1]})
                result["status"] = status
                with self.assertRaises(AIError):
                    OpenAIFAQ._parse_result(result, self.faq)

    def test_malformed_response_is_a_safe_error(self):
        malformed = (None, [], {}, {"status": "completed", "output": None},
                     {"status": "completed", "output": [{"type": "message", "content": None}]},
                     {"status": "completed", "output": [{"type": "message", "content": [
                         {"type": "output_text", "text": "not json"},
                     ]}]})
        for result in malformed:
            with self.subTest(result=result):
                with self.assertRaises(AIError):
                    OpenAIFAQ._parse_result(result, self.faq)

    @patch("ai.request.urlopen")
    def test_answer_sends_strict_schema_and_disables_storage(self, urlopen):
        urlopen.return_value = io.BytesIO(json.dumps(api_result({"faq_ids": [3]})).encode())
        client = OpenAIFAQ("fake-unit-test-key", model="test-model", timeout=11)
        self.assertEqual(client.answer("Что за направление?", self.faq), self.faq[2][1])
        req = urlopen.call_args.args[0]
        payload = json.loads(req.data)
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 11)
        self.assertEqual(payload["model"], "test-model")
        self.assertIs(payload["store"], False)
        self.assertIs(payload["text"]["format"]["strict"], True)
        self.assertEqual(json.loads(payload["input"])["question"], "Что за направление?")
        self.assertEqual(len(json.loads(payload["input"])["faq"]), 5)

    @patch("ai.request.urlopen")
    def test_missing_key_does_not_make_network_request(self, urlopen):
        with self.assertRaisesRegex(AIError, "OPENAI_API_KEY"):
            OpenAIFAQ("").answer("Вопрос", self.faq)
        urlopen.assert_not_called()

    @patch("ai.request.urlopen")
    def test_http_errors_are_safe_and_actionable(self, urlopen):
        messages = {401: "OPENAI_API_KEY", 403: "запрещён", 404: "OPENAI_MODEL",
                    429: "квота", 500: "HTTP 500"}
        for status, message in messages.items():
            with self.subTest(status=status):
                urlopen.side_effect = HTTPError("https://example.invalid", status,
                                               "private-server-detail", {}, None)
                with self.assertRaisesRegex(AIError, message) as caught:
                    OpenAIFAQ("fake-unit-test-key").answer("Вопрос", self.faq)
                self.assertNotIn("fake-unit-test-key", str(caught.exception))
                self.assertNotIn("private-server-detail", str(caught.exception))

    @patch("ai.request.urlopen")
    def test_transport_errors_and_timeouts_are_safe(self, urlopen):
        for failure in (TimeoutError("private-detail"), URLError("private-detail"), OSError("private-detail")):
            with self.subTest(failure=type(failure).__name__):
                urlopen.side_effect = failure
                with self.assertRaisesRegex(AIError, "подключение") as caught:
                    OpenAIFAQ("fake-unit-test-key").answer("Вопрос", self.faq)
                self.assertNotIn("private-detail", str(caught.exception))

    @patch("ai.request.urlopen")
    def test_non_json_http_body_is_safe(self, urlopen):
        urlopen.return_value = io.BytesIO(b"private-server-detail")
        with self.assertRaisesRegex(AIError, "некорректный ответ") as caught:
            OpenAIFAQ("fake-unit-test-key").answer("Вопрос", self.faq)
        self.assertNotIn("private-server-detail", str(caught.exception))


class BotServiceTests(unittest.TestCase):
    def setUp(self):
        self.faq = load_faq(FAQ_FILE)
        self.ai = Mock(spec=OpenAIFAQ)
        self.ai.answer.return_value = self.faq[1][1]
        self.service = BotService(self.faq, self.ai)

    def test_auto_prefers_classic_match_without_api_cost(self):
        self.assertEqual(self.service.answer("Как сдать?"), self.faq[3][1])
        self.ai.answer.assert_not_called()

    def test_auto_uses_ai_for_unrecognized_question(self):
        self.assertEqual(self.service.answer("  Допустимо объединиться?  "), self.faq[1][1])
        self.ai.answer.assert_called_once_with("Допустимо объединиться?", self.faq)

    def test_classic_never_uses_ai(self):
        self.assertEqual(self.service.answer("Как сдать?", "classic"), self.faq[3][1])
        self.assertEqual(self.service.answer("Привет", "classic"), UNKNOWN)
        self.ai.answer.assert_not_called()

    def test_forced_ai_uses_ai_even_for_classic_match(self):
        self.ai.answer.return_value = UNKNOWN
        self.assertEqual(self.service.answer("Сколько человек в команде?", "ai"), UNKNOWN)
        self.ai.answer.assert_called_once()

    def test_missing_key_keeps_classic_and_explains_forced_ai(self):
        service = BotService(self.faq)
        self.assertEqual(service.answer("Как сдать?"), self.faq[3][1])
        self.assertEqual(service.answer("Привет"), UNKNOWN)
        answer = service.answer("Как сдать?", "ai")
        self.assertTrue(answer.startswith(self.faq[3][1]))
        self.assertIn("OPENAI_API_KEY", answer)

    def test_ai_failure_falls_back_and_warns(self):
        self.ai.answer.side_effect = AIError("OpenAI: превышен лимит запросов.")
        with self.assertLogs("service", level="WARNING"):
            answer = self.service.answer("Как сдать?", "ai")
        self.assertTrue(answer.startswith(self.faq[3][1]))
        self.assertIn("ИИ временно недоступен", answer)
        with self.assertLogs("service", level="WARNING"):
            answer = self.service.answer("Посторонний вопрос", "auto")
        self.assertTrue(answer.startswith(UNKNOWN))

    def test_blank_and_oversized_questions_never_reach_ai(self):
        self.assertEqual(self.service.answer(" \n ", "ai"), "Напишите вопрос.")
        self.assertIn("4000", self.service.answer("я" * 4001, "ai"))
        self.ai.answer.assert_not_called()
        self.service.answer("я" * 4000, "ai")
        self.ai.answer.assert_called_once()

    def test_invalid_mode_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "режим"):
            self.service.answer("Привет", "invalid")
        self.ai.answer.assert_not_called()


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / ".env"
        clean_environment = patch.dict(os.environ, {}, clear=True)
        clean_environment.start()
        self.addCleanup(clean_environment.stop)

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def test_bom_quotes_comments_and_equals_in_values(self):
        self.write('\ufeff# Local settings\nOPENAI_API_KEY="fake=test"\n'
                   "TELEGRAM_BOT_TOKEN='fake-token'\nOPENAI_MODEL=test-model # comment\nAI_TIMEOUT=7\n")
        settings = Settings.load(self.path)
        self.assertEqual(settings.openai_key, "fake=test")
        self.assertEqual(settings.telegram_token, "fake-token")
        self.assertEqual(settings.openai_model, "test-model")
        self.assertEqual(settings.ai_timeout, 7)
        self.assertNotIn("fake=test", repr(settings))
        self.assertNotIn("fake-token", repr(settings))

    def test_environment_overrides_file_including_empty_values(self):
        self.write("OPENAI_API_KEY=fake-file-key\nOPENAI_MODEL=file-model\nAI_TIMEOUT=4\n")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_MODEL": " env-model ", "AI_TIMEOUT": "10"}):
            settings = Settings.load(self.path)
        self.assertEqual(settings.openai_key, "")
        self.assertEqual(settings.openai_model, "env-model")
        self.assertEqual(settings.ai_timeout, 10)

    def test_missing_file_uses_defaults(self):
        settings = Settings.load(self.path)
        self.assertEqual(settings.openai_key, "")
        self.assertEqual(settings.telegram_token, "")
        self.assertEqual(settings.openai_model, "gpt-4.1-mini")
        self.assertEqual(settings.ai_timeout, 20)

    def test_empty_model_uses_default(self):
        self.write("OPENAI_MODEL=\n")
        self.assertEqual(Settings.load(self.path).openai_model, "gpt-4.1-mini")

    def test_invalid_timeout_fails_without_echoing_value(self):
        for value in ("fake-private-value", "", "0", "0.99", "121", "-5", "nan", "inf"):
            with self.subTest(value=value):
                self.write(f"AI_TIMEOUT={value}\n")
                with self.assertRaisesRegex(ValueError, "от 1 до 120") as caught:
                    Settings.load(self.path)
                self.assertNotIn("fake-private-value", str(caught.exception))

    def test_timeout_boundaries_are_accepted(self):
        for value in ("1", "1.5", "120"):
            with self.subTest(value=value):
                self.write(f"AI_TIMEOUT={value}\n")
                self.assertEqual(Settings.load(self.path).ai_timeout, float(value))

    def test_unknown_setting_and_unclosed_quotes_are_safe_errors(self):
        for text in ("UNKNOWN=fake-private-value", 'OPENAI_API_KEY="fake-private-value'):
            with self.subTest(kind="unknown" if text.startswith("UNKNOWN") else "quotes"):
                self.write(text)
                with self.assertRaises(ValueError) as caught:
                    read_env(self.path)
                self.assertNotIn("fake-private-value", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
