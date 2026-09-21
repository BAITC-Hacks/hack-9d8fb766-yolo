"""Local settings; no third-party dependencies and no secret logging."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ENV_FILE = Path(__file__).with_name(".env")
SETTING_NAMES = {"TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "OPENAI_MODEL", "AI_TIMEOUT"}


def read_env(path: Path = ENV_FILE) -> dict[str, str]:
    values = {}
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not separator or key not in SETTING_NAMES:
                raise ValueError(f"Проверьте строку {number} в .env: неизвестная настройка.")
            if value.startswith(("'", '"')):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError(f"Проверьте кавычки в строке {number} файла .env.")
                value = value[1:-1]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
            values[key] = value
    for key in SETTING_NAMES:
        if key in os.environ:
            values[key] = os.environ[key].strip()
    return values


@dataclass(frozen=True)
class Settings:
    telegram_token: str = field(default="", repr=False)
    openai_key: str = field(default="", repr=False)
    openai_model: str = "gpt-4.1-mini"
    ai_timeout: float = 20.0

    @classmethod
    def load(cls, path: Path = ENV_FILE) -> "Settings":
        env = read_env(path)
        try:
            timeout = float(env.get("AI_TIMEOUT", "20"))
            if not 1 <= timeout <= 120:
                raise ValueError
        except ValueError:
            raise ValueError("AI_TIMEOUT должен быть числом от 1 до 120 секунд.") from None
        return cls(
            telegram_token=env.get("TELEGRAM_BOT_TOKEN", ""),
            openai_key=env.get("OPENAI_API_KEY", ""),
            openai_model=env.get("OPENAI_MODEL", "") or "gpt-4.1-mini",
            ai_timeout=timeout,
        )
