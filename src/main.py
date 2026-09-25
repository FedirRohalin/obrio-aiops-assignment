import json
import logging
import os

from dotenv import load_dotenv

# Завантажуємо змінні оточення до імпорту клієнта
load_dotenv()

from src.classifier import TicketClassifier  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402

# Вимикаємо зайві логи, щоб таблиця була чистою
logging.getLogger("httpx").setLevel(logging.WARNING)


def evaluate():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Помилка: OPENAI_API_KEY не знайдено у .env")
        return

    client = LLMClient(api_key=api_key)
    classifier = TicketClassifier(client)

    try:
        with open("data/tickets.json", "r", encoding="utf-8") as f:
            tickets = json.load(f)
    except FileNotFoundError:
        print("Помилка: файл data/tickets.json не знайдено.")
        return

    print(
        f"{'ID':<10} | {'Expected':<18} | {'Actual':<18} | {'Match':<5} | {'Needs Human'}"
    )
    print("-" * 75)

    correct = 0
    total = len(tickets)

    for ticket in tickets:
        ticket_id = ticket.get("ticket_id", "UNKNOWN")
        expected_cat = ticket.get("expected_category", "UNKNOWN")
        user_text = ticket.get("user_text", "")

        # Запит до LLM
        result = classifier.classify(user_text)
        actual_cat = (
            result.category.value
            if hasattr(result.category, "value")
            else str(result.category)
        )

        # Порівняння
        match = "✅" if expected_cat == actual_cat else "❌"
        if match == "✅":
            correct += 1

        human_flag = "Yes" if result.needs_human else "No"

        print(
            f"{ticket_id:<10} | {expected_cat:<18} | {actual_cat:<18} | {match:<5} | {human_flag}"
        )

    accuracy = (correct / total) * 100 if total > 0 else 0
    print("-" * 75)
    print(f"Total processed: {total}")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")


if __name__ == "__main__":
    evaluate()
