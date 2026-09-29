import json
import sys
import time
from pathlib import Path

from openai import OpenAI

# Allow running as `python eval/run_eval.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assistant.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from src.assistant.schemas import Article  # noqa: E402

MODELS = ["gpt-4o", "gpt-4o-mini"]

# USD per 1M tokens. Verify against the current OpenAI pricing page before reporting results.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

DATASET: list[dict] = [
    {
        "ticket_text": "I forgot my password and the reset email never arrives. How can I get back into my account?",
        "expected_article_id": "KB-101",
        "articles": [
            Article(
                id="KB-101",
                title="Password Reset",
                tags=["account"],
                text="To reset your password, open Settings > Security and tap 'Reset password'. "
                "If the email does not arrive within 10 minutes, check your spam folder and confirm the address on your account.",
            ),
            Article(
                id="KB-102",
                title="Change Display Name",
                tags=["account"],
                text="You can change your display name at any time in Settings > Profile. "
                "Changes are applied immediately.",
            ),
            Article(
                id="KB-103",
                title="Invoices",
                tags=["billing"],
                text="Invoices are generated on the first day of each billing period and sent to the account email.",
            ),
        ],
    },
    {
        "ticket_text": "I was charged twice this month for my subscription. I want a refund for the duplicate payment.",
        "expected_article_id": "KB-201",
        "articles": [
            Article(
                id="KB-201",
                title="Duplicate Charges",
                tags=["billing", "refund"],
                text="If you were charged twice, contact support with the transaction IDs. "
                "Duplicate charges are refunded to the original payment method within 5-7 business days.",
            ),
            Article(
                id="KB-202",
                title="Cancel Subscription",
                tags=["billing"],
                text="To cancel your subscription, open Settings > Subscription and choose 'Cancel plan'. "
                "Access continues until the end of the paid period.",
            ),
            Article(
                id="KB-203",
                title="Payment Methods",
                tags=["billing"],
                text="We accept Visa, Mastercard, Apple Pay and Google Pay. Bank transfers are not supported.",
            ),
        ],
    },
    {
        "ticket_text": "The app crashes every time I open the Reports tab on my Android phone after the last update.",
        "expected_article_id": "KB-301",
        "articles": [
            Article(
                id="KB-301",
                title="App Crashes",
                tags=["troubleshooting", "android"],
                text="If the app crashes on launch or when opening a tab, update to the latest version, "
                "clear the app cache in system settings, and restart the device.",
            ),
            Article(
                id="KB-302",
                title="Push Notifications",
                tags=["settings"],
                text="Push notifications can be enabled in Settings > Notifications. "
                "Make sure notifications are also allowed in your phone's system settings.",
            ),
            Article(
                id="KB-303",
                title="Export Data",
                tags=["data"],
                text="You can export your data as CSV from Settings > Data > Export. Large exports may take a few minutes.",
            ),
        ],
    },
    {
        "ticket_text": "How do I delete my account and all my personal data permanently?",
        "expected_article_id": "KB-401",
        "articles": [
            Article(
                id="KB-401",
                title="Delete Account",
                tags=["privacy"],
                text="To permanently delete your account, go to Settings > Privacy > Delete account. "
                "All personal data is erased within 30 days and this action cannot be undone.",
            ),
            Article(
                id="KB-402",
                title="Deactivate Account",
                tags=["privacy"],
                text="You can temporarily deactivate your account in Settings > Privacy. "
                "Your data is kept and you can reactivate at any time by logging in.",
            ),
            Article(
                id="KB-403",
                title="Two-Factor Authentication",
                tags=["security"],
                text="Two-factor authentication can be enabled in Settings > Security using an authenticator app.",
            ),
        ],
    },
]


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING[model]
    return (
        prompt_tokens * rates["input"] + completion_tokens * rates["output"]
    ) / 1_000_000


def run_evaluation() -> list[dict]:
    client = OpenAI()
    results: list[dict] = []

    for model in MODELS:
        for index, item in enumerate(DATASET):
            user_prompt = build_user_prompt(item["ticket_text"], item["articles"])

            record: dict = {
                "model": model,
                "ticket_index": index,
                "expected_article_id": item["expected_article_id"],
                "predicted_article_id": None,
                "accuracy": 0,
                "latency_s": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "error": None,
            }

            start = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                record["latency_s"] = round(time.perf_counter() - start, 4)
            except Exception as exc:
                record["latency_s"] = round(time.perf_counter() - start, 4)
                record["error"] = f"{type(exc).__name__}: {exc}"
                results.append(record)
                continue

            usage = response.usage
            record["prompt_tokens"] = usage.prompt_tokens
            record["completion_tokens"] = usage.completion_tokens
            record["cost_usd"] = round(
                _calculate_cost(model, usage.prompt_tokens, usage.completion_tokens), 8
            )

            try:
                parsed = json.loads(response.choices[0].message.content)
                citation = parsed.get("citation") if isinstance(parsed, dict) else None
                predicted = (
                    citation.get("article_id") if isinstance(citation, dict) else None
                )
                record["predicted_article_id"] = predicted
                record["accuracy"] = int(predicted == item["expected_article_id"])
            except (json.JSONDecodeError, TypeError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"

            results.append(record)

    return results


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
