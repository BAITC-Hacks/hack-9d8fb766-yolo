"""Небольшой поиск по пяти темам FAQ без внешних зависимостей."""

import re
from pathlib import Path


FAQ_FILE = Path(__file__).with_name("faq.txt")
UNKNOWN = "Не знаю."

# Порядок тем совпадает с порядком пяти пар в faq.txt.
# Общие слова вроде «сколько» и «куда» сами по себе тему не определяют.
KEYWORDS = {
    "время": {"врем", "долго", "длит", "минут", "час", "часа", "часов", "займ", "продолж"},
    "команда": {"команд", "вместе", "участник", "человек", "групп", "друз"},
    "трек": {"трек", "направлен", "llm", "тема", "теме", "тему", "темы"},
    "сдача": {"сдать", "сдава", "сдач", "загруз", "отправ", "репозитор", "github", "readme"},
    "призы": {"приз", "наград", "выигрыш", "деньги"},
}


def load_faq(path: Path) -> list[tuple[str, str]]:
    """Читает ровно пять непустых пар «Вопрос:» / «Ответ:».

    Пары разделяются пустыми строками; поддерживаются UTF-8 BOM и CRLF.
    Ошибочный блок не пропускается, чтобы не сместить соответствие тем.
    """
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    blocks = re.split(r"\n\s*\n", text) if text else []
    if len(blocks) != len(KEYWORDS):
        raise ValueError("В faq.txt должно быть ровно 5 пар вопрос–ответ.")

    faq: list[tuple[str, str]] = []
    for number, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines()]
        if len(lines) != 2 or not lines[0].startswith("Вопрос:") or not lines[1].startswith("Ответ:"):
            raise ValueError(f"Пара {number}: ожидаются две строки «Вопрос:» и «Ответ:».")
        question = lines[0].removeprefix("Вопрос:").strip()
        answer = lines[1].removeprefix("Ответ:").strip()
        if not question or not answer:
            raise ValueError(f"Пара {number}: вопрос и ответ не должны быть пустыми.")
        faq.append((question, answer))
    return faq


def normalize(text: str) -> set[str]:
    """Выделяет уникальные слова без учёта регистра и различия е/ё."""
    return set(re.findall(r"[а-яa-z0-9]+", text.lower().replace("ё", "е")))


def keyword_matches(word: str, keyword: str) -> bool:
    """Короткие ключи совпадают целиком, длинные могут быть основой слова."""
    return word == keyword or (len(keyword) >= 4 and word.startswith(keyword))


def find_answer(question: str, faq: list[tuple[str, str]]) -> str:
    """Возвращает ответ наиболее похожей темы или UNKNOWN при ничьей."""
    if len(faq) != len(KEYWORDS):
        raise ValueError("Для поиска нужны ровно 5 пар вопрос–ответ.")
    words = normalize(question)
    # Одно слово даёт теме не больше одного балла даже при нескольких ключах.
    scores = [
        sum(any(keyword_matches(word, keyword) for keyword in keywords) for word in words)
        for keywords in KEYWORDS.values()
    ]
    best_score = max(scores)
    if best_score == 0 or scores.count(best_score) != 1:
        return UNKNOWN
    return faq[scores.index(best_score)][1]


def format_faq(faq: list[tuple[str, str]]) -> str:
    """Список вопросов и ответов для команды /faq и терминала."""
    return "\n\n".join(
        f"{number}. {question}\n{answer}"
        for number, (question, answer) in enumerate(faq, start=1)
    )
