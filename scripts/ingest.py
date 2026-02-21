import asyncio  # noqa: D100
from loguru import logger
from production_rag.pipeline.embeddings import embed_documents
from production_rag.pipeline.parse import parse_file
from production_rag.core.database import init_db
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


async def _main():  # noqa: D103
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized.")
    logger.info("Parsing documents.")
    documents = await parse_file("../data/raw/esp32_datasheet.pdf")
    logger.info(f"Parsed {len(documents)} documents.")
    logger.info("Embedding and storing documents.")
    await embed_documents(documents)
    logger.info("Done.")


def main():  # noqa: D103
    """Synchronous entry point for the rag-ingest CLI command."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
