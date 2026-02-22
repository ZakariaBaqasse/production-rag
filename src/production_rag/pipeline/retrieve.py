"""Retrieval and answer generation module for the production RAG pipeline.

This module provides functions to:
- search_custom_docs: retrieve relevant documents using vector similarity search
- generate_answer: generate answers using a language model based on retrieved documents
"""

import asyncpg
from langchain.chat_models import init_chat_model
from loguru import logger
from dotenv import load_dotenv
from tenacity import retry, wait_exponential
from production_rag.core.config import (
    ModelConfig,
)
from production_rag.core.database import get_db_connection
from production_rag.pipeline.utils import get_embedding_model

load_dotenv()  # Load environment variables from .env file


async def search_custom_docs(  # noqa: D103
    query_text: str, model_config: ModelConfig, top_k: int = 5
) -> list[asyncpg.Record]:
    # 1. Embed the query
    embeddings = get_embedding_model(model_config)
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


generation_system_prompt = """You are an assistant that answers questions based on the provided context from a document:"""


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def generate_answer(  # noqa: D103
    query_text: str, retrieved_docs: list[asyncpg.Record], model_config: ModelConfig
) -> str:
    # 1. Format the retrieved documents into a context string
    context = "\n\n".join(
        f"Page {doc['page_number']}:\n{doc['content']}"
        for doc in retrieved_docs
        if doc["content"]
    )
    # 2. Create the prompt for the language model
    prompt = generation_system_prompt.format(context=context, query_text=query_text)
    # 3. Generate the answer using the language model
    llm = init_chat_model(
        model=model_config.model,
        provider=model_config.provider,
        base_url=model_config.base_url,
    )
    try:
        response = await llm.agenerate(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"CONTEXT: {context} \n\n Question: {query_text}\nAnswer based on the context above.",
                },
            ]
        )
        return response.generations[0][0].text.strip()
    except Exception as e:
        logger.error(f"Failed to generate answer: {e}")
        return "Sorry, I couldn't generate an answer at this time."
