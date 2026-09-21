"""Shared answer logic for the terminal and Telegram."""
import logging

from ai import AIError, OpenAIFAQ
from faq_engine import UNKNOWN, find_answer

logger = logging.getLogger(__name__)


class BotService:
    def __init__(self, faq: list[tuple[str, str]], ai: OpenAIFAQ | None = None):
        self.faq = faq
        self.ai = ai

    def answer(self, question: str, mode: str = "auto") -> str:
        question = question.strip()
        if not question:
            return "Напишите вопрос."
        if len(question) > 4000:
            return "Вопрос слишком длинный. Сократите его до 4000 символов."
        if mode not in {"auto", "ai", "classic"}:
            raise ValueError("Неизвестный режим ответа.")
        classic_answer = find_answer(question, self.faq)
        if mode == "classic":
            return classic_answer
        if mode == "auto" and classic_answer != UNKNOWN:
            return classic_answer
        if self.ai is None:
            if mode == "ai":
                return f"{classic_answer}\n\nИИ не настроен: добавьте OPENAI_API_KEY в .env и перезапустите бота."
            return classic_answer
        try:
            return self.ai.answer(question, self.faq)
        except AIError as exc:
            logger.warning("%s", exc)
            return f"{classic_answer}\n\nИИ временно недоступен. Использован поиск по ключевым словам."
