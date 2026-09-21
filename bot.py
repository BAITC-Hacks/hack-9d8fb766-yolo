"""Run python bot.py for terminal chat or python bot.py --telegram."""
import argparse
import logging
import sys

from ai import OpenAIFAQ
from config import Settings
from faq_engine import FAQ_FILE, find_answer, format_faq, load_faq
from service import BotService


def terminal_chat(service: BotService, mode: str) -> None:
    print("FAQ-бот: время, команда, трек, сдача, призы.")
    print("Команды: /faq, /ai, /classic, /auto, выход. Режим:", mode)
    if service.ai is None:
        print("OpenAI пока не настроен; работает обычный FAQ.")
    while True:
        try:
            question = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nБот: До свидания!")
            return
        command = question.lower()
        if command in {"выход", "exit", "quit", "/exit"}:
            print("Бот: До свидания!")
            return
        if command == "/faq":
            print(format_faq(service.faq))
        elif command in {"/ai", "/classic", "/auto"}:
            if command == "/ai" and service.ai is None:
                print("Бот: Добавьте OPENAI_API_KEY в .env и перезапустите бота.")
                continue
            mode = command[1:]
            print("Бот: Режим", mode)
        else:
            print("Бот:", service.answer(question, mode))


def main() -> int:
    parser = argparse.ArgumentParser(description="FAQ-бот в терминале и Telegram с OpenAI.")
    parser.add_argument("--telegram", action="store_true", help="Запустить Telegram long polling")
    parser.add_argument("--mode", choices=("auto", "ai", "classic"), default="auto",
                        help="Режим терминала (по умолчанию auto)")
    parser.add_argument("--check", action="store_true", help="Проверить настройки и Telegram без чтения сообщений")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings = Settings.load()
        faq = load_faq(FAQ_FILE)
        ai = (OpenAIFAQ(settings.openai_key, settings.openai_model, settings.ai_timeout)
              if settings.openai_key else None)
        service = BotService(faq, ai)
        if args.telegram or args.check:
            from telegram_bot import TelegramClient, run_telegram
            if not settings.telegram_token:
                raise ValueError("Добавьте TELEGRAM_BOT_TOKEN в .env.")
            if args.check:
                client = TelegramClient(settings.telegram_token)
                me = client.call("getMe")
                webhook = client.call("getWebhookInfo")
                print("Telegram: @" + me["username"])
                print("Webhook:", "активен — long polling недоступен" if webhook.get("url") else "не установлен")
                print("FAQ: 5 пар загружены.")
                print("OpenAI:", "ключ задан (запрос к ИИ не выполнялся)" if ai else "ожидает OPENAI_API_KEY в .env")
                return 1 if webhook.get("url") else 0
            run_telegram(settings.telegram_token, service.answer, format_faq(faq), ai is not None)
        else:
            terminal_chat(service, args.mode)
        return 0
    except KeyboardInterrupt:
        print("\nБот остановлен.")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Ошибка запуска: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Ошибка Telegram: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
