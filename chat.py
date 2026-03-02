import os
import time
from datetime import date
import json
from openai import OpenAI

from config import get_model_cfg
from logger import log_llm_call

# Импортируем модели прогнозирования временных рядов
from models import Price, Trips, Sales
from config import DATA_DIR

# Реестр: series_name → (класс модели, файл данных, параметры)
MODEL_REGISTRY = {
    "price": {"cls": Price, "data": "price.csv", "historic_points": 8, "max_forecast_days": 90},
    "trips": {"cls": Trips, "data": "trips.csv", "historic_points": 8, "max_forecast_days": 90},
    "sales": {"cls": Sales, "data": "sales.csv", "historic_points": 8, "max_forecast_days": 90},
}

def get_client(model_key=None) -> tuple[OpenAI, str]:
    """Создаёт OpenAI-клиент для модели из config.yaml."""
    cfg = get_model_cfg(model_key)
    client = OpenAI(
        api_key=cfg.get("api_key", "none"),
        base_url=cfg["base_url"],
    )
    return client, cfg["name"]

SYSTEM_PROMPT = """ Ты - аналитический помощник по транспорту Москвы.
                """
# ─── Промпты ────────────────────────────────────────────

PARSE_TASK_TYPE_PROMPT = """
Ты - специализированный парсер запросов для системы прогнозирования временных рядов.

ТВОЯ ЗАДАЧА:
Получить текст запроса пользователя на русском языке и вернуть СТРОГО один JSON-объект с типом задачи:
<types> 
    - "sales" — продажи (выручка, количество проданных товаров и т.п.).
    - "trips" — поездки (поездки, поездок, trip, trips, поездка и т.п.).
    - "price" — цена (цена, стоимость, прайс и т.п.).
</types>
Если ряд не указан — ставь null.

ПОВЕДЕНИЕ:
    - Не объясняй свои действия.
    - Не добавляй никакой текст вне JSON.
    - Если чего‑то не хватает, ставь null или специальное значение, не выдумывай.

ВЫХОДНОЙ JSON:
Всегда возвращай объект вида:
    {
    "series": "..."
    }
"""

PARSE_DATE_PROMPT = """
Ты - агент по извлечению дат из запроса пользователя.

ТВОЯ ЗАДАЧА:
Извлечь из запроса пользователя временной диапазон для прогноза.
Сегодняшняя дата: {today} ({weekday}).

ПРАВИЛА:
- Если указана одна конкретная дата → date_from = date_to = эта дата.
- Если указан период ("следующая неделя", "март", "ближайшие 3 дня") →
  date_from = первый день периода, date_to = последний день периода.
- Если указано только "завтра", "послезавтра" → date_from = date_to.
- Относительные даты ("через неделю", "в следующий понедельник") вычисляй 
  относительно сегодняшнего дня.
- Если дату невозможно определить — ставь null.

Формат дат: YYYY-MM-DD

ПОВЕДЕНИЕ:
- Не объясняй свои действия.
- Не добавляй никакой текст вне JSON.

ВЫХОДНОЙ JSON:
{{
    "date_from": "YYYY-MM-DD",
    "date_to": "YYYY-MM-DD"
}}
"""


# ─── Универсальная функция ──────────────────────────────

def parse(task: str, client: OpenAI, model: str, prompt: str, agent_name: str):
    """Универсальный вызов LLM с JSON-ответом + логирование."""
    if len(task) < 1:
        return "empty"

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": task},
    ]

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    duration = time.perf_counter() - t0

    result = response.choices[0].message.content
    usage = response.usage

    log_llm_call(
        agent_name=agent_name,
        task=task,
        model=model,
        result=result,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        duration_sec=duration,
    )

    return result

def call_model(task_type: str, date_from: str, date_to: str):
    """Создаёт модель по типу задачи, обучает и прогнозирует."""
    if task_type not in MODEL_REGISTRY:
        raise ValueError(f"Неизвестный тип: {task_type}. Доступные: {list(MODEL_REGISTRY.keys())}")

    cfg = MODEL_REGISTRY[task_type]
    model = cfg["cls"](historic_points=cfg["historic_points"], max_forecast_days=cfg["max_forecast_days"])
    model.load_data(os.path.join(DATA_DIR, cfg["data"]))
    model.fit(model._full_data)

    result = model.predict(date_to)

    # Если диапазон — фильтруем траекторию
    df = result.to_dataframe()
    df = df[(df["date"] >= date_from) & (df["date"] <= date_to)]

    return df

# ─── Промпт для вежливого отказа ─────────────────────────

OFF_TOPIC_PROMPT = """
Ты - дружелюбный деловой ассистент-аналитик по транспорту Москвы.
Ты умеешь прогнозировать только три типа временных рядов: поездки (trips), продажи (sales), цену (price).

Пользователь задал вопрос не по теме прогнозирования.
Вежливо объясни, что ты можешь помочь только с прогнозами по этим трём рядам.
Приведи пример вопроса, который ты можешь обработать.
Отвечай коротко, по-русски, в дружеском деловом тоне.
"""


# ─── Шаблоны ответов ─────────────────────────────────────

SERIES_NAMES_RU = {
    "trips": "поездок",
    "sales": "продаж",
    "price": "цены",
}


def _format_forecast_answer(task_type: str, date_from: str, date_to: str, df) -> str:
    """Формирует шаблонный текстовый ответ по результатам прогноза."""
    series_ru = SERIES_NAMES_RU.get(task_type, task_type)

    if date_from == date_to:
        # Одна дата
        value = df["forecast"].iloc[-1]
        return (
            f"📊 Прогноз {series_ru} на {date_to}:\n"
            f"   Ожидаемое значение: {value:,.2f}\n"
            f"   (модель: {df['model'].iloc[0]})"
        )
    else:
        # Диапазон
        total = df["forecast"].sum()
        mean = df["forecast"].mean()
        days = len(df)
        # Построчный список прогнозов
        details = "\n".join(
            f"     {row['date'].strftime('%Y-%m-%d')}:  {row['forecast']:,.2f}"
            for _, row in df.iterrows()
        )

        return (
            f"📊 Прогноз {series_ru} с {date_from} по {date_to} ({days} дн.):\n"
            f"   Сумма:    {total:,.2f}\n"
            f"   Среднее:  {mean:,.2f}\n"
            f"   Мин:      {df['forecast'].min():,.2f}\n"
            f"   Макс:     {df['forecast'].max():,.2f}\n\n"
            f"   Детализация:\n{details}\n\n"
            f"   (модель: {df['model'].iloc[0]})"
        )


# ─── Главная функция ─────────────────────────────────────

def get_answer(task: str, client: OpenAI, model: str) -> str:
    """
    Оркестратор: принимает вопрос пользователя, возвращает текстовый ответ.

    Сценарии:
    1. Прогноз + дата → шаблонный ответ с данными модели
    2. Прогноз без даты → уточняющий вопрос
    3. Не прогноз → вежливый отказ через LLM
    """

    # ── Шаг 1: определяем тип задачи ────────────────────
    raw_series = parse(task, client, model, PARSE_TASK_TYPE_PROMPT, "parse_task_type")
    parsed_series = json.loads(raw_series)
    series = parsed_series.get("series")

    # ── Шаг 2: если не прогноз — вежливый отказ ─────────
    if series is None:
        messages = [
            {"role": "system", "content": OFF_TOPIC_PROMPT},
            {"role": "user", "content": task},
        ]
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        duration = time.perf_counter() - t0
        result = response.choices[0].message.content
        usage = response.usage

        log_llm_call(
            agent_name="off_topic",
            task=task,
            model=model,
            result=result,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            duration_sec=duration,
        )
        return result

    # ── Шаг 3: извлекаем дату ───────────────────────────
    today = date.today()
    date_prompt = PARSE_DATE_PROMPT.format(
        today=today.isoformat(),
        weekday=today.strftime("%A"),
    )
    raw_date = parse(task, client, model, date_prompt, "parse_date")
    parsed_date = json.loads(raw_date)
    date_from = parsed_date.get("date_from")
    date_to = parsed_date.get("date_to")

    # ── Шаг 4: если дата не определена — уточняем ──────
    if date_from is None or date_to is None:
        series_ru = SERIES_NAMES_RU.get(series, series)
        return (
            f"Я могу сделать прогноз {series_ru}, но не смог определить дату из вашего вопроса.\n"
            f"Уточните, пожалуйста, на какую дату или период нужен прогноз?\n"
            f"Например: «на завтра», «на 15 марта», «на следующую неделю»."
        )

    # ── Шаг 5: вызываем модель и формируем ответ ────────
    try:
        df = call_model(series, date_from, date_to)
        return _format_forecast_answer(series, date_from, date_to, df)
    except ValueError as e:
        return f"⚠️ Не удалось выполнить прогноз: {e}"

if __name__ == '__main__':
    client, model = get_client()

    while True:
        task = input("\nВы: ").strip()
        if task.lower() in ("выход", "exit", "quit"):
            print("До свидания!")
            break
        answer = get_answer(task, client, model)
        print(f"\nБот: {answer}")
    
