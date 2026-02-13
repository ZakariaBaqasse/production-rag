import asyncio
from loguru import logger
from src.embeddings import embed_documents
from src.parse import parse_file
from src.database import init_db
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


async def main():
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized.")
    logger.info("Parsing documents.")
    documents = await parse_file("./src/assets/esp32_datasheet.pdf")
    logger.info(f"Parsed {len(documents)} documents.")
    logger.info("Embedding and storing documents.")
    await embed_documents(documents)
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
