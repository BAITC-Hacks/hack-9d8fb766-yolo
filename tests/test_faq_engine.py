import tempfile
import unittest
from pathlib import Path

from faq_engine import FAQ_FILE, UNKNOWN, find_answer, format_faq, load_faq, normalize


class FAQMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.faq = load_faq(FAQ_FILE)

    def test_original_questions_find_their_answers(self):
        for question, answer in self.faq:
            with self.subTest(question=question):
                self.assertEqual(find_answer(question, self.faq), answer)

    def test_all_five_topics_with_natural_variants(self):
        variants = {
            0: ("Сколько это займёт?", "Как долго выполнять?", "Сколько минут нужно?"),
            1: ("Сколько человек в команде?", "Можно вместе с друзьями?", "Нужна команда?"),
            2: ("Какой ТРЕК?", "Что за направление?", "LLM-приложения?"),
            3: ("Куда загрузить решение?", "Как сдавать?", "Нужен README?"),
            4: ("Будут награды?", "Есть призы?", "Какой выигрыш?"),
        }
        for index, questions in variants.items():
            for question in questions:
                with self.subTest(question=question):
                    self.assertEqual(find_answer(question, self.faq), self.faq[index][1])

    def test_unknown_and_weak_words(self):
        for question in ("", "?!", "Какая погода?", "Сколько стоит билет?", "Куда поехать?", "Сколько?"):
            with self.subTest(question=question):
                self.assertEqual(find_answer(question, self.faq), UNKNOWN)

    def test_equal_topics_return_unknown(self):
        self.assertEqual(find_answer("Какой трек и какие призы?", self.faq), UNKNOWN)
        self.assertEqual(find_answer("Команда и время", self.faq), UNKNOWN)

    def test_duplicate_words_do_not_overweight_a_topic(self):
        self.assertEqual(find_answer("Трек трек трек призы", self.faq), UNKNOWN)

    def test_normalization(self):
        self.assertEqual(normalize("ЗАЙМЁТ, займёт! LLM"), {"займет", "llm"})

    def test_faq_format_includes_all_pairs_in_order(self):
        result = format_faq(self.faq)
        self.assertEqual(len(result.split("\n\n")), 5)
        for number, (question, answer) in enumerate(self.faq, start=1):
            self.assertIn(f"{number}. {question}\n{answer}", result)


class FAQFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "faq.txt"
        self.blocks = [f"Вопрос: Вопрос {i}?\nОтвет: Ответ {i}." for i in range(5)]

    def write(self, text):
        self.path.write_text(text, encoding="utf-8", newline="")

    def test_bom_crlf_and_whitespace_blank_lines(self):
        text = "\ufeff" + "\r\n \t\r\n\r\n".join(block.replace("\n", "\r\n") for block in self.blocks) + "\r\n"
        self.write(text)
        self.assertEqual(load_faq(self.path), [(f"Вопрос {i}?", f"Ответ {i}.") for i in range(5)])

    def test_rejects_missing_and_extra_pairs(self):
        for blocks in ([], self.blocks[:4], self.blocks + [self.blocks[0]]):
            with self.subTest(count=len(blocks)):
                self.write("\n\n".join(blocks))
                with self.assertRaisesRegex(ValueError, "ровно 5"):
                    load_faq(self.path)

    def test_rejects_malformed_or_empty_pairs(self):
        bad_blocks = ("Вопрос: ?\nНеверно: ответ", "Вопрос: \nОтвет: ответ", "Вопрос: вопрос\nОтвет: ", "Вопрос: вопрос\nОтвет: ответ\nЛишняя строка", "Ответ: ответ\nВопрос: вопрос")
        for block in bad_blocks:
            with self.subTest(block=block):
                self.write("\n\n".join([self.blocks[0], block, *self.blocks[2:]]))
                with self.assertRaisesRegex(ValueError, "Пара 2"):
                    load_faq(self.path)


if __name__ == "__main__":
    unittest.main()
