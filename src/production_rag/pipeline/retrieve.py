import asyncio  # noqa: D100
import asyncpg
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from dotenv import load_dotenv
from production_rag.core.config import (
    CHAT_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
)
from production_rag.core.database import get_db_connection

load_dotenv()  # Load environment variables from .env file


async def search_custom_docs(  # noqa: D103
    query_text: str, top_k: int = 5, model: str = EMBEDDING_MODEL_NAME
) -> list[asyncpg.Record]:
    # 1. Embed the query
    embeddings = OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
    query_vector = await embeddings.aembed_query(query_text)
    vector_str = "[" + ",".join(map(str, query_vector)) + "]"
    # 2. Run Cosine Similarity Search via SQL
    conn = await get_db_connection()
    try:
        # The <=> operator is "Cosine Distance" (Lower is better)
        rows = await conn.fetch(
            """
            SELECT content, page_number, 1 - (embedding <=> $1) as similarity
            FROM document_pages
            ORDER BY embedding <=> $1
            LIMIT $2;
        """,
            vector_str,
            top_k,
        )

        await conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to search documents: {e}")
        return []
    finally:
        if not conn.is_closed():
            await conn.close()


async def main():  # noqa: D103
    query = "What is the maximum operating temperature of the ESP32?"
    results = await search_custom_docs(query)
    logger.info(f"Top {len(results)} results for query: '{query}'")
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided document excerpts.",
        },
        {
            "role": "user",
            "content": f"Based on the following document excerpts, answer the question: '{query}'\n\n"
            + "\n\n".join(
                [
                    f"Excerpt {i + 1} (Page {row['page_number']}): {row['content']}"
                    for i, row in enumerate(results)
                ]
            ),
        },
    ]
    async for chunk in model.astream(messages):
        if chunk.content:
            print(chunk.content[0]["text"], end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
