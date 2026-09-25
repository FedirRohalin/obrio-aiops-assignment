from src.classifier import TicketClassifier
from src.llm_client import LLMClient
from src.schemas import TicketCategory, TicketPriority


def test_fallback_result_is_safe():
    """Перевіряє, що у разі збою API система видає безпечний результат для ручного розбору."""
    client = LLMClient(api_key="dummy_key")
    classifier = TicketClassifier(client)

    # Викликаємо приватний метод для тестування ізольованої логіки
    fallback = classifier._build_fallback_result()

    assert fallback.category == TicketCategory.GENERAL_QUESTION
    assert fallback.priority == TicketPriority.HIGH
    assert fallback.needs_human is True
    assert "Fallback triggered" in fallback.reasoning
