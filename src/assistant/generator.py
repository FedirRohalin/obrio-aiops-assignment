from src.assistant.llm_client import LLMClient
from src.assistant.prompts import SYSTEM_PROMPT, build_user_prompt
from src.assistant.schemas import Article, AssistantOutput


class TicketGenerator:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate_response(
        self, ticket_text: str, articles: list[Article]
    ) -> AssistantOutput:
        user_prompt = build_user_prompt(ticket_text, articles)
        raw_data = self._client.generate(SYSTEM_PROMPT, user_prompt)
        output = AssistantOutput(**raw_data)

        if output.citation is not None:
            article = next(
                (a for a in articles if a.id == output.citation.article_id),
                None,
            )
            if article is None:
                raise ValueError("Hallucinated article_id")
            if output.citation.quote not in article.text:
                raise ValueError(
                    "Quote is not a verbatim substring of the article text"
                )

        return output
