"""Telegram long polling using only Python's standard library.

Pending updates are preserved at startup. Modes live in memory until restart.
Telegram has no sendMessage idempotency key: a lost network acknowledgement can
cause a repeated reply, but confirmed chunks are not sent again in this process.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from http.client import HTTPException


class TelegramError(RuntimeError):
    """Safe error: never includes the request URL, token or response body."""

    def __init__(self, code: int = 0, retry_after: int = 0):
        self.code = code
        self.retry_after = retry_after
        messages = {
            0: "Нет соединения с Telegram. Проверьте интернет.",
            401: "Telegram отклонил токен. Проверьте TELEGRAM_BOT_TOKEN.",
            404: "Telegram не нашёл бота. Проверьте TELEGRAM_BOT_TOKEN.",
            403: "Бот не может отправить сообщение в этот чат.",
            409: "Telegram: конфликт запуска. Остановите другой экземпляр бота "
                 "или проверьте, не включён ли webhook.",
            429: "Telegram ограничил частоту запросов. Повторяем после паузы.",
        }
        super().__init__(messages.get(code, f"Ошибка Telegram API (код {code})."))

    @property
    def transient(self) -> bool:
        return self.code == 0 or self.code == 429 or self.code >= 500


def split_message(text: str, limit: int = 4000) -> list[str]:
    """Split below the Telegram limit, conservatively counting UTF-16 units."""
    chunks: list[str] = []
    start = 0
    units = 0
    for index, character in enumerate(text):
        width = 2 if ord(character) > 0xFFFF else 1
        if units + width > limit:
            chunks.append(text[start:index])
            start, units = index, 0
        units += width
    if start < len(text):
        chunks.append(text[start:])
    return chunks


class TelegramClient:
    def __init__(self, token: str):
        if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token):
            raise ValueError("Некорректный TELEGRAM_BOT_TOKEN.")
        self._token = token

    def call(self, method: str, payload: dict | None = None, *, timeout: int = 20):
        """One request; callers decide when to retry without losing an update."""
        if method not in {"getMe", "getWebhookInfo", "getUpdates", "sendMessage"}:
            raise ValueError("Неподдерживаемый метод Telegram API.")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self._token}/{method}",
            data=json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        status = 200
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(4_000_000)
        except urllib.error.HTTPError as error:
            status = error.code
            try:
                raw = error.read(4_000_000)
            except OSError:
                raise TelegramError(status) from None
            finally:
                error.close()
        except (urllib.error.URLError, OSError, TimeoutError, HTTPException):
            raise TelegramError() from None
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError):
            raise TelegramError(status if status != 200 else 502) from None
        if not isinstance(data, dict):
            raise TelegramError(502)
        if data.get("ok") is not True:
            code = data.get("error_code", status if status != 200 else 502)
            parameters = data.get("parameters") or {}
            retry = parameters.get("retry_after", 0) if isinstance(parameters, dict) else 0
            raise TelegramError(
                code if isinstance(code, int) else 502,
                max(0, retry) if isinstance(retry, int) else 0,
            )
        return data.get("result")

    def send_text(self, chat_id: int, text: str, *, reply_to: int | None = None,
                  thread_id: int | None = None) -> None:
        for chunk in split_message(text):
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "link_preview_options": {"is_disabled": True},
            }
            if reply_to is not None:
                payload["reply_parameters"] = {
                    "message_id": reply_to, "allow_sending_without_reply": True,
                }
            if thread_id is not None:
                payload["message_thread_id"] = thread_id
            self.call("sendMessage", payload)


HELP = (
    "Я отвечаю на вопросы о репетиции по пяти пунктам FAQ.\n\n"
    "/faq — показать все вопросы и ответы\n"
    "/auto — сначала поиск по FAQ, при необходимости ИИ (по умолчанию)\n"
    "/ai — ответы с помощью настроенного ИИ\n"
    "/classic — только поиск по ключевым словам, без ИИ\n"
    "/help — эта справка\n\n"
    "Напишите вопрос обычным сообщением. В группе упомяните меня через @имя_бота "
    "или ответьте на моё сообщение. Команды в группе: /faq@имя_бота.\n"
    "Режим общий для чата и сбрасывается на /auto после перезапуска. "
    "В режиме ИИ текст вопроса отправляется настроенному провайдеру ИИ."
)


class TelegramBot:
    def __init__(self, client: TelegramClient, answer: Callable[[str, str], str],
                 faq_text: str, ai_available: bool, bot_id: int, username: str):
        self.client = client
        self.answer = answer
        self.faq_text = faq_text
        self.ai_available = ai_available
        self.bot_id = bot_id
        self.username = username.lower()
        self.modes: dict[int, str] = {}
        self._pending_id: int | None = None
        self._pending_chunks: list[str] = []

    def _addressed_text(self, message: dict) -> str | None:
        text = message.get("text")
        if not isinstance(text, str) or message.get("from", {}).get("is_bot"):
            return None
        chat_type = message.get("chat", {}).get("type")
        command = re.match(r"^/[a-zA-Z0-9_]+(?:@([a-zA-Z0-9_]+))?(?=\s|$)", text)
        if command and command.group(1) and command.group(1).lower() != self.username:
            return None
        if chat_type == "private":
            return text.strip()
        if chat_type not in {"group", "supergroup"}:
            return None
        addressed = bool(command and command.group(1))
        addressed |= message.get("reply_to_message", {}).get("from", {}).get("id") == self.bot_id
        mentions = []
        encoded = text.encode("utf-16-le")
        for entity in message.get("entities", []):
            start = entity.get("offset", 0) * 2
            end = start + entity.get("length", 0) * 2
            word = encoded[start:end].decode("utf-16-le", errors="ignore")
            if (entity.get("type") == "mention" and word.lower() == f"@{self.username}") or (
                entity.get("type") == "text_mention"
                and entity.get("user", {}).get("id") == self.bot_id
            ):
                addressed = True
                mentions.append((start, end))
        if not addressed:
            return None
        for start, end in sorted(mentions, reverse=True):
            encoded = encoded[:start] + encoded[end:]
        return encoded.decode("utf-16-le").strip()

    def _reply(self, chat_id: int, text: str) -> str:
        if not text:
            return "Напишите вопрос о репетиции. Список вопросов: /faq."
        if len(text) > 4000:
            return "Вопрос слишком длинный. Сократите его до 4000 символов."
        if text.startswith("/"):
            command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
            if command in {"/start", "/help"}:
                return HELP
            if command == "/faq":
                return self.faq_text
            if command in {"/ai", "/classic", "/auto"}:
                mode = command[1:]
                if mode == "ai" and not self.ai_available:
                    return ("ИИ пока не настроен. Владельцу бота нужно добавить ключ "
                            "провайдера ИИ в .env по инструкции README и перезапустить бота. "
                            "Ключ Telegram не является ключом ИИ. Поиск по FAQ доступен: /classic.")
                self.modes[chat_id] = mode
                descriptions = {
                    "ai": "Включён режим ИИ. Напишите вопрос о репетиции.",
                    "classic": "Включён поиск по ключевым словам без ИИ.",
                    "auto": "Включён автоматический режим: FAQ, затем доступный ИИ.",
                }
                return descriptions[mode]
            return "Неизвестная команда. Доступные команды: /help."
        try:
            result = self.answer(text, self.modes.get(chat_id, "auto"))
        except Exception:
            # Do not expose provider diagnostics, prompts or credentials in chat/logs.
            return "Не удалось подготовить ответ. Попробуйте ещё раз или откройте /faq."
        return result if isinstance(result, str) and result.strip() else "Не знаю. Попробуйте /faq."

    def process_update(self, update: dict) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        text = self._addressed_text(message)
        if text is None:
            return
        chat_id = message.get("chat", {}).get("id")
        if not isinstance(chat_id, int):
            return
        update_id = update.get("update_id")
        if update_id != self._pending_id or not self._pending_chunks:
            self._pending_id = update_id
            self._pending_chunks = split_message(self._reply(chat_id, text))
        while self._pending_chunks:
            self.client.send_text(
                chat_id, self._pending_chunks[0], reply_to=message.get("message_id"),
                thread_id=message.get("message_thread_id"),
            )
            self._pending_chunks.pop(0)
        self._pending_id = None


def _retry_pause(error: TelegramError, delay: int) -> int:
    print(str(error), flush=True)
    time.sleep(max(delay, error.retry_after))
    return min(delay * 2, 30)


def run_telegram(token: str, answer: Callable[[str, str], str], faq_text: str,
                 ai_available: bool) -> None:
    """Run until Ctrl+C. Raise safe TelegramError for authentication/conflicts."""
    client = TelegramClient(token)
    try:
        delay = 2
        while True:
            try:
                me = client.call("getMe")
                webhook = client.call("getWebhookInfo")
                break
            except TelegramError as error:
                if not error.transient:
                    raise
                delay = _retry_pause(error, delay)
        if not isinstance(me, dict) or not isinstance(webhook, dict):
            raise TelegramError(502)
        if webhook.get("url"):
            raise RuntimeError("У бота настроен webhook. Отключите его вручную перед запуском "
                               "long polling; текущие настройки не изменены.")
        bot = TelegramBot(client, answer, faq_text, ai_available, me["id"], me["username"])
        print(f"Telegram-бот @{me['username']} запущен. "
              f"ИИ: {'настроен' if ai_available else 'не настроен'}. Остановка: Ctrl+C.", flush=True)
        offset = None
        delay = 2
        while True:
            payload = {"timeout": 30, "allowed_updates": ["message"]}
            if offset is not None:
                payload["offset"] = offset
            try:
                updates = client.call("getUpdates", payload, timeout=40)
                if not isinstance(updates, list):
                    raise TelegramError(502)
                delay = 2
            except TelegramError as error:
                if not error.transient:
                    raise
                delay = _retry_pause(error, delay)
                continue
            for update in updates:
                if not isinstance(update, dict) or not isinstance(update.get("update_id"), int):
                    continue
                update_id = update["update_id"]
                if offset is not None and update_id < offset:
                    continue
                delay = 2
                while True:
                    try:
                        bot.process_update(update)
                        break
                    except TelegramError as error:
                        if error.code in {401, 404, 409}:
                            raise
                        if not error.transient:
                            print(str(error), flush=True)
                            break  # Blocked/deleted chat must not block other users.
                        delay = _retry_pause(error, delay)
                offset = update_id + 1
    except KeyboardInterrupt:
        print("\nTelegram-бот остановлен.", flush=True)
