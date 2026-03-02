"""
Прогон тестовых вопросов через чат-бота.

Usage:
    python test_chat.py                         # все вопросы из test_questions.txt
    python test_chat.py --file my_questions.txt  # свой файл
    python test_chat.py --model qwen3-30b       # конкретная модель
"""

import os
import sys
import csv
import time
import argparse
from datetime import datetime

from config import ROOT_DIR, LOGS_DIR
from chat import get_client, get_answer


def load_questions(filepath: str) -> list[str]:
    """Загружает вопросы из текстового файла (пропускает пустые и комментарии)."""
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def run_tests(questions: list[str], client, model: str, output_csv: str):
    """Прогоняет список вопросов и сохраняет результаты в CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results = []
    total = len(questions)

    print(f"\n{'='*70}")
    print(f"  Тестирование чат-бота  |  модель: {model}")
    print(f"  Вопросов: {total}")
    print(f"{'='*70}\n")

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{total}] {question}")
        print(f"{'-'*50}")

        t0 = time.perf_counter()
        try:
            answer = get_answer(question, client, model)
            status = "OK"
            error = ""
        except Exception as e:
            answer = ""
            status = "ERROR"
            error = str(e)
        duration = time.perf_counter() - t0

        # Краткий вывод в консоль
        preview = answer.replace("\n", " ")[:120]
        print(f"  [{status}] ({duration:.2f}s) {preview}...")
        if error:
            print(f"  ⚠️ {error}")
        print()

        results.append({
            "question": question,
            "answer": answer,
            "status": status,
            "error": error,
            "duration_sec": round(duration, 2),
        })

    # Сохраняем CSV
    fields = ["question", "answer", "status", "error", "duration_sec"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    # Итоги
    ok = sum(1 for r in results if r["status"] == "OK")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    total_time = sum(r["duration_sec"] for r in results)
    avg_time = total_time / total if total else 0

    print(f"{'='*70}")
    print(f"  ИТОГИ")
    print(f"{'='*70}")
    print(f"  Всего:    {total}")
    print(f"  OK:       {ok}")
    print(f"  Ошибок:   {errors}")
    print(f"  Время:    {total_time:.1f}s (среднее {avg_time:.2f}s)")
    print(f"  Отчёт:    {output_csv}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Test chatbot with questions from file")
    parser.add_argument("--file", default=os.path.join(ROOT_DIR, "tests/test_questions.txt"),
                        help="Path to questions file (default: test_questions.txt)")
    parser.add_argument("--model", default=None,
                        help="Model key from config.yaml (default: default_model)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Файл не найден: {args.file}")
        sys.exit(1)

    questions = load_questions(args.file)
    if not questions:
        print("❌ Файл пуст или содержит только комментарии.")
        sys.exit(1)

    client, model = get_client(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(LOGS_DIR, f"test_run_{timestamp}.csv")

    run_tests(questions, client, model, output_csv)


if __name__ == "__main__":
    main()