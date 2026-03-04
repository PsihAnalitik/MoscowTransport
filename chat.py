import os
import re
import time
from datetime import date
import json
from openai import OpenAI

from config import get_model_cfg, DATA_DIR
from logger import log_llm_call

# Импортируем модели прогнозирования временных рядов
from models import Price, Trips, Sales

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

# ─── Промпты ────────────────────────────────────────────

PARSE_INTENT_PROMPT = """
Ты - парсер запросов для системы прогнозирования временных рядов.
Сегодняшняя дата: {today} ({weekday}).

ТВОЯ ЗАДАЧА:
Разобрать запрос пользователя и извлечь ВСЕ задачи прогнозирования из него.

<series>
    - "sales" — продажи (выручка, количество проданных товаров и т.п.)
    - "trips" — поездки (поездки, поездок, trip, trips, поездка и т.п.)
    - "price" — цена (цена, стоимость, прайс и т.п.)
</series>

<rules>
- Один запрос может содержать несколько задач (например: "сравни поездки и продажи на завтра" = 2 задачи).
- Для каждой задачи определи ряд (series) и диапазон дат (date_from, date_to).
- Если дата одна → date_from = date_to.
- Если дата не указана → date_from = null, date_to = null.
- Если ряд не относится к sales/trips/price → tasks = пустой список.
- Относительные даты вычисляй от сегодняшнего дня.
- Формат дат: YYYY-MM-DD.
</rules>

<date_rules>
Правила интерпретации относительных периодов:
- "завтра" → date_from = date_to = завтрашняя дата.
- "послезавтра" → date_from = date_to = послезавтрашняя дата.
- "на этой неделе" → date_from = понедельник текущей недели, date_to = воскресенье текущей недели.
- "на следующей неделе" → date_from = понедельник следующей недели, date_to = воскресенье следующей недели.
- "через неделю" → date_from = date_to = сегодня + 7 дней.
- "в этом месяце" → date_from = 1-е число текущего месяца, date_to = последний день текущего месяца.
- "в следующем месяце" → date_from = 1-е число следующего месяца, date_to = последний день следующего месяца.
- "через месяц" → date_from = date_to = сегодня + 1 месяц (та же дата следующего месяца).
- "в марте", "в апреле" и т.п. → date_from = 1-е число указанного месяца, date_to = последний день указанного месяца. Если месяц уже прошёл в текущем году — бери следующий год.
- "на ближайшие N дней" → date_from = завтра, date_to = завтра + (N-1) дней.
- "в следующий понедельник" → date_from = date_to = ближайший будущий понедельник.
- Неделя начинается с понедельника (ISO).
</date_rules>

<need_llm_rules>
Поле "need_llm" определяет, нужен ли LLM для формулировки финального ответа.
- true — если вопрос аналитический, сравнительный или требует рассуждения
  (сравни, почему, стоит ли, объясни, проанализируй, оцени, рост, падение, тренд, динамика и т.п.)
- true — если задач больше одной.
- false — если вопрос простой и фактический ("сколько поездок завтра?", "какая цена на 10 марта?").
</need_llm_rules>

ПОВЕДЕНИЕ:
- Не объясняй свои действия.
- Не добавляй текст вне JSON.

<output_format>
{{
    "tasks": [
        {{"series": "...", "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD"}},
        ...
    ],
    "need_llm": true/false
}}
</output_format>

<examples>
- "Сколько поездок завтра?" → {{"tasks": [{{"series": "trips", "date_from": "...", "date_to": "..."}}], "need_llm": false}}
- "Сравни поездки и продажи за март" → {{"tasks": [{{"series": "trips", ...}}, {{"series": "sales", ...}}], "need_llm": true}}
- "Какая погода?" → {{"tasks": [], "need_llm": false}}
- "Какая будет цена?" → {{"tasks": [{{"series": "price", "date_from": null, "date_to": null}}], "need_llm": false}}
- "Стоит ли ожидать рост поездок?" → {{"tasks": [{{"series": "trips", "date_from": null, "date_to": null}}], "need_llm": true}}
- "Поездки на следующей неделе" → {{"tasks": [{{"series": "trips", "date_from": "<понедельник следующей недели>", "date_to": "<воскресенье следующей недели>"}}], "need_llm": false}}
- "Продажи в следующем месяце" → {{"tasks": [{{"series": "sales", "date_from": "<1-е число следующего месяца>", "date_to": "<последний день следующего месяца>"}}], "need_llm": false}}
</examples>
"""


# ─── Утилиты ─────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

def _strip_think(text: str) -> str:
    """Удаляет блок <think>...</think> из ответа LLM (reasoning-модели)."""
    return _THINK_RE.sub("", text).strip()


# ─── Универсальная функция ──────────────────────────────

def parse(task: str, client: OpenAI, model: str, prompt: str, agent_name: str):
    """Универсальный вызов LLM с JSON-ответом + логирование."""
    if not task.strip():
        return '{"tasks": [], "need_llm": false}'

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

    result = _strip_think(response.choices[0].message.content)
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

    model = MODEL_REGISTRY[task_type]
    try:
        result = model.predict(date_to)
    except RuntimeError as e:
        raise ValueError(f"Модель {task_type} не готова: {e}")

    df = result.to_dataframe()
    df = df[(df["date"] >= date_from) & (df["date"] <= date_to)]
    return df

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

SYNTHESIZE_PROMPT = """
Ты - аналитик по транспорту Москвы. Тебе предоставлены результаты прогнозных моделей.
Ответь на вопрос пользователя, используя ТОЛЬКО предоставленные данные.
Не выдумывай цифры. Если данных недостаточно для полного ответа — скажи об этом.
Отвечай по-русски, в деловом дружеском тоне, кратко и структурировано.
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



def _synthesize_answer(task: str, forecast_blocks: list[str], client: OpenAI, model: str) -> str:
    """LLM формулирует ответ на основе данных прогнозов."""
    data_text = "\n\n".join(forecast_blocks)
    messages = [
        {"role": "system", "content": SYNTHESIZE_PROMPT},
        {"role": "user", "content": (
            f"Вопрос пользователя: {task}\n\n"
            f"Данные прогнозов:\n{data_text}"
        )},
    ]

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    duration = time.perf_counter() - t0

    result = _strip_think(response.choices[0].message.content)
    usage = response.usage

    log_llm_call(
        agent_name="synthesize",
        task=task,
        model=model,
        result=result,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        duration_sec=duration,
    )

    return result


# ─── Главная функция ─────────────────────────────────────

def get_answer(task: str, client: OpenAI, model: str) -> str:
    """
    Оркестратор: принимает вопрос пользователя, возвращает текстовый ответ.

    Сценарии:
    1. Несколько прогнозов + даты → LLM-синтез с данными всех моделей
    2. Один простой прогноз + дата → шаблонный ответ
    3. Один аналитический прогноз + дата → LLM-синтез
    4. Прогноз без даты → уточняющий вопрос
    5. Не прогноз → вежливый отказ через LLM
    """

    # ── Шаг 1: единый парсинг (серии + даты за один вызов) ──
    today = date.today()
    intent_prompt = PARSE_INTENT_PROMPT.format(
        today=today.isoformat(),
        weekday=today.strftime("%A"),
    )
    raw = parse(task, client, model, intent_prompt, "parse_intent")
    parsed = json.loads(raw)
    tasks = parsed.get("tasks", [])
    need_llm = parsed.get("need_llm", False)

    # ── Шаг 2: нет задач → off-topic ───────────────────────
    if not tasks:
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
        result = _strip_think(response.choices[0].message.content)
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

    # ── Шаг 3: проверяем, есть ли задачи без даты ──────────
    missing_dates = [
        t for t in tasks
        if t.get("date_from") is None or t.get("date_to") is None
    ]
    if missing_dates:
        series_list = ", ".join(
            SERIES_NAMES_RU.get(t["series"], t["series"])
            for t in missing_dates
        )
        return (
            f"Я могу сделать прогноз ({series_list}), но не смог определить дату.\n"
            f"Уточните, на какую дату или период нужен прогноз?\n"
            f"Например: «на завтра», «на 15 марта», «на следующую неделю»."
        )

    # ── Шаг 4: выполняем все прогнозы ──────────────────────
    forecast_blocks = []
    for t in tasks:
        try:
            df = call_model(t["series"], t["date_from"], t["date_to"])
            block = _format_forecast_answer(t["series"], t["date_from"], t["date_to"], df)
            forecast_blocks.append(block)
        except ValueError as e:
            forecast_blocks.append(
                f"⚠️ {SERIES_NAMES_RU.get(t['series'], t['series'])}: {e}"
            )

    # ── Шаг 5: простой шаблон или LLM-синтез ─────────────────
    if need_llm:
        return _synthesize_answer(task, forecast_blocks, client, model)
    else:
        return "\n\n".join(forecast_blocks)

if __name__ == '__main__':
    client, model = get_client()

    while True:
        task = input("\nВы: ").strip()
        if task.lower() in ("выход", "exit", "quit"):
            print("До свидания!")
            break
        answer = get_answer(task, client, model)
        print(f"\nБот: {answer}")
    
