import argparse
import logging
import sys

from src.assistant.config import get_settings
from src.assistant.generator import TicketGenerator
from src.assistant.kb import KnowledgeBase
from src.assistant.llm_client import LLMClient
from src.assistant.retriever import TicketRetriever

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nebula AI Support Assistant CLI")
    parser.add_argument(
        "ticket_text",
        nargs="?",
        type=str,
        help="Text of the support ticket. If not provided, reads from stdin.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output the final result as raw JSON."
    )

    args = parser.parse_args()

    # Read from CLI argument or stdin
    ticket_text = args.ticket_text
    if not ticket_text:
        if not sys.stdin.isatty():
            ticket_text = sys.stdin.read().strip()

    if not ticket_text:
        parser.error(
            "Ticket text must be provided either as a command line argument or via stdin."
        )

    # Load settings
    settings = get_settings()

    # Configure logging
    log_level = getattr(settings, "log_level", "INFO")
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("Initializing Nebula AI Support Assistant CLI...")

    try:
        # Load Knowledge Base
        logger.info(f"Loading Knowledge Base from {settings.kb_path}")
        kb = KnowledgeBase(settings.kb_path)
        kb.load()
        logger.info(
            f"Loaded {len(kb.get_all_articles())} articles into the Knowledge Base."
        )

        # Retrieve relevant articles
        logger.info("Initializing TicketRetriever and searching for relevant articles.")
        retriever = TicketRetriever(kb)
        scored_articles = retriever.retrieve(
            ticket_text,
            min_score=settings.retrieval_min_score,
            top_k=settings.retrieval_top_k,
        )

        # ДОДАНО: Витягуємо чисті об'єкти Article з ScoredArticle
        articles = [sa.article for sa in scored_articles]

        logger.info(f"Found {len(articles)} relevant article(s).")
        # Initialize LLM and Generator
        logger.info("Initializing LLMClient and TicketGenerator.")
        client = LLMClient(settings)
        generator = TicketGenerator(client)

        # Generate the response
        logger.info("Sending prompt to LLM to generate response...")
        result = generator.generate_response(ticket_text, articles)
        logger.info("Response generated successfully.")

        # Output the final result (ONLY using print)
        if args.json:
            print(result.model_dump_json())
        else:
            print("=" * 60)
            print("NEBULA AI ASSISTANT OUTPUT")
            print("=" * 60)
            print(f"\n[SUMMARY]\n{result.summary}")
            print(f"\n[FORMAL REPLY]\n{result.formal_reply}")
            print(f"\n[EMPATHETIC REPLY]\n{result.empathetic_reply}")
            print(f"\n[SHORT REPLY]\n{result.short_reply}")

            if result.citation:
                print(f"\n[CITATION - Article ID: {result.citation.article_id}]")
                print(f'"{result.citation.quote}"')
            else:
                print("\n[CITATION]\nNone provided.")
            print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
        sys.exit(1)
