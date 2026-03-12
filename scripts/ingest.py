import asyncio  # noqa: D100
from pathlib import Path  # noqa: D100
from loguru import logger
from production_rag.pipeline.embeddings import embed_documents
from production_rag.pipeline.parse import parse_file
from production_rag.core.database import init_db
from production_rag.core.config import load_config
from dotenv import load_dotenv
import argparse

load_dotenv()  # Load environment variables from .env file


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation execution."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation using the curated testset and the live RAG pipeline."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs/google-genai.yaml",
        help="Path to the experiment configuration file.",
    )
    return parser.parse_args()


async def _main():  # noqa: D103
    args = parse_args()
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized.")
    logger.info("Parsing documents.")
    documents = await parse_file("./data/raw/esp32_datasheet.pdf")
    logger.info(f"Parsed {len(documents)} documents.")
    logger.info("Embedding and storing documents.")
    config_path = Path(args.config_path)
    app_config = load_config(config_path)
    await embed_documents(documents, app_config.pipeline.embeddings)
    logger.info("Done.")


def main():  # noqa: D103
    """Synchronous entry point for the rag-ingest CLI command."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
