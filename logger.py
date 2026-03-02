import os
import csv
from datetime import datetime

from config import LOGS_DIR
os.makedirs(LOGS_DIR, exist_ok=True)

# Заголовки CSV
_FIELDS = ["timestamp", "task", "model", "result", "prompt_tokens", "completion_tokens", "duration_sec"]


def _ensure_header(filepath: str):
    """Создаёт CSV с заголовком, если файла ещё нет."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_FIELDS)


def log_llm_call(
    agent_name: str,
    task: str,
    model: str,
    result: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    duration_sec: float = 0.0,
):
    """
    Дописывает одну строку в logs/<agent_name>.csv

    Parameters
    ----------
    agent_name : str   — имя агента/функции (станет именем CSV)
    task       : str   — входной текст пользователя
    model      : str   — название модели
    result     : str   — ответ LLM
    prompt_tokens / completion_tokens — из response.usage
    duration_sec — время вызова в секундах
    """
    filepath = os.path.join(LOGS_DIR, f"{agent_name}.csv")
    _ensure_header(filepath)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            task.replace("\n", " ").strip(),
            model,
            result.replace("\n", " ").strip(),
            prompt_tokens,
            completion_tokens,
            f"{duration_sec:.2f}",
        ])