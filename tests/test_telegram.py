"""Offline Telegram tests: no user tokens and no outgoing network requests."""

import io
import json
import unittest
import urllib.error
from unittest.mock import Mock, patch

from telegram_bot import TelegramBot, TelegramClient, TelegramError, run_telegram, split_message


def update(text="Когда?", chat_id=10, kind="private", **extra):
    message = {"text": text, "message_id": 5, "chat": {"id": chat_id, "type": kind},
               "from": {"id": 2, "is_bot": False}}
    message.update(extra)
    return {"update_id": 100, "message": message}


class TelegramBotTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.answer = Mock(return_value="Ответ FAQ")
        self.bot = TelegramBot(self.client, self.answer, "Все пять ответов", True, 777, "Test_Bot")

    def sent(self):
        return self.client.send_text.call_args.args[1]

    def test_private_question_uses_auto(self):
        self.bot.process_update(update())
        self.answer.assert_called_once_with("Когда?", "auto")
        self.assertEqual(self.sent(), "Ответ FAQ")

    def test_modes_are_scoped_to_chat(self):
        self.bot.process_update(update("/ai"))
        self.bot.process_update(update("Когда?"))
        self.answer.assert_called_with("Когда?", "ai")
        self.bot.process_update(update("Когда?", chat_id=20))
        self.answer.assert_called_with("Когда?", "auto")
        self.bot.process_update(update("/classic"))
        self.bot.process_update(update("Когда?"))
        self.answer.assert_called_with("Когда?", "classic")
        self.bot.process_update(update("/auto"))
        self.assertEqual(self.bot.modes[10], "auto")

    def test_unconfigured_ai_explains_and_preserves_mode(self):
        self.bot.ai_available = False
        self.bot.process_update(update("/classic"))
        self.bot.process_update(update("/ai"))
        self.assertIn("Ключ Telegram не является ключом ИИ", self.sent())
        self.assertEqual(self.bot.modes[10], "classic")
        self.answer.assert_not_called()

    def test_faq_and_help_are_local(self):
        self.bot.process_update(update("/faq"))
        self.assertEqual(self.sent(), "Все пять ответов")
        self.bot.process_update(update("/start"))
        self.assertIn("/classic", self.sent())
        self.answer.assert_not_called()

    def test_unaddressed_group_messages_and_commands_are_ignored(self):
        for text in ("Когда?", "/start", "/faq@different_bot", "@Test_Bot_extra когда?"):
            self.bot.process_update(update(text, kind="group"))
        self.client.send_text.assert_not_called()
        self.answer.assert_not_called()

    def test_group_addressed_command(self):
        self.bot.process_update(update("/faq@Test_Bot", kind="supergroup", message_thread_id=9))
        self.assertEqual(self.sent(), "Все пять ответов")
        self.assertEqual(self.client.send_text.call_args.kwargs["thread_id"], 9)

    def test_group_reply_to_bot(self):
        self.bot.process_update(update(kind="group", reply_to_message={"from": {"id": 777}}))
        self.answer.assert_called_once_with("Когда?", "auto")

    def test_group_mention_handles_utf16_emoji_offsets(self):
        self.bot.process_update(update("😀 @Test_Bot когда?", kind="group", entities=[
            {"type": "mention", "offset": 3, "length": 9},
        ]))
        self.answer.assert_called_once_with("😀  когда?", "auto")

    def test_other_bot_command_is_ignored_even_in_private(self):
        self.bot.process_update(update("/faq@other_bot"))
        self.client.send_text.assert_not_called()

    def test_oversized_question_does_not_reach_ai(self):
        self.bot.process_update(update("x" * 4001))
        self.assertIn("4000", self.sent())
        self.answer.assert_not_called()

    def test_answer_exception_never_reaches_user(self):
        self.answer.side_effect = RuntimeError("SECRET_PROVIDER_CREDENTIAL")
        self.bot.process_update(update())
        self.assertNotIn("SECRET", self.sent())
        self.assertIn("/faq", self.sent())

    def test_failed_delivery_reuses_answer_and_completed_chunks(self):
        self.answer.return_value = "a" * 5000
        self.client.send_text.side_effect = [None, TelegramError(), None]
        with self.assertRaises(TelegramError):
            self.bot.process_update(update())
        self.bot.process_update(update())
        self.answer.assert_called_once()
        self.assertEqual([len(call.args[1]) for call in self.client.send_text.call_args_list],
                         [4000, 1000, 1000])

    def test_nontext_and_bots_are_ignored(self):
        self.bot.process_update({"update_id": 3, "message": {"photo": []}})
        self.bot.process_update(update(**{"from": {"id": 999, "is_bot": True}}))
        self.client.send_text.assert_not_called()

    def test_message_split_preserves_text_and_utf16_limit(self):
        original = "Привет😀" * 1800
        chunks = split_message(original)
        self.assertEqual("".join(chunks), original)
        self.assertTrue(all(0 < len(chunk.encode("utf-16-le")) // 2 <= 4000 for chunk in chunks))


class TelegramClientTests(unittest.TestCase):
    @patch("telegram_bot.urllib.request.urlopen")
    def test_json_post_and_plain_text(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = b'{"ok": true, "result": {}}'
        TelegramClient("123:TEST_TOKEN").send_text(4, "**обычный текст**")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertNotIn("parse_mode", payload)
        self.assertEqual(payload["text"], "**обычный текст**")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 20)

    @patch("telegram_bot.urllib.request.urlopen")
    def test_http_error_does_not_disclose_token_or_api_body(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError(
            "https://api.telegram.org/bot123:SECRET/getMe", 401, "SECRET", {},
            io.BytesIO(b'{"ok": false, "error_code": 401, "description": "SECRET"}'))
        with self.assertRaises(TelegramError) as caught:
            TelegramClient("123:SECRET").call("getMe")
        self.assertEqual(caught.exception.code, 401)
        self.assertNotIn("SECRET", str(caught.exception))

    @patch("telegram_bot.urllib.request.urlopen")
    def test_rate_limit_preserves_retry_after(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError("ignored", 429, "rate", {},
            io.BytesIO(b'{"ok":false,"error_code":429,"parameters":{"retry_after":7}}'))
        with self.assertRaises(TelegramError) as caught:
            TelegramClient("123:TOKEN").call("getUpdates")
        self.assertEqual(caught.exception.retry_after, 7)
        self.assertTrue(caught.exception.transient)

    @patch("telegram_bot.print")
    @patch("telegram_bot.TelegramClient")
    def test_active_webhook_is_not_deleted(self, client_class, _print):
        client_class.return_value.call.side_effect = [
            {"id": 777, "username": "test_bot"}, {"url": "https://example.org/hook"},
        ]
        with self.assertRaisesRegex(RuntimeError, "webhook"):
            run_telegram("123:TOKEN", Mock(), "FAQ", False)
        self.assertEqual([call.args[0] for call in client_class.return_value.call.call_args_list],
                         ["getMe", "getWebhookInfo"])

    @patch("telegram_bot.print")
    @patch("telegram_bot.time.sleep")
    @patch("telegram_bot.TelegramClient")
    def test_poll_retries_rate_limit_and_advances_offset_after_processing(self, cls, sleep, _print):
        client = cls.return_value
        client.call.side_effect = [
            {"id": 777, "username": "test_bot"}, {"url": ""},
            TelegramError(429, 7), [update()], KeyboardInterrupt(),
        ]
        answer = Mock(return_value="Ответ")
        run_telegram("123:TOKEN", answer, "FAQ", True)
        answer.assert_called_once_with("Когда?", "auto")
        sleep.assert_called_once_with(7)
        polls = [call for call in client.call.call_args_list if call.args[0] == "getUpdates"]
        self.assertNotIn("offset", polls[0].args[1])
        self.assertEqual(polls[-1].args[1]["offset"], 101)

    @patch("telegram_bot.print")
    @patch("telegram_bot.TelegramClient")
    def test_conflict_fails_clearly_without_retry(self, cls, _print):
        cls.return_value.call.side_effect = [
            {"id": 777, "username": "test_bot"}, {"url": ""}, TelegramError(409),
        ]
        with self.assertRaisesRegex(TelegramError, "другой экземпляр"):
            run_telegram("123:TOKEN", Mock(), "FAQ", True)


if __name__ == "__main__":
    unittest.main()
