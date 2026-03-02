"""Retrieval and answer generation module for the production RAG pipeline.

This module provides functions to:
- search_custom_docs: retrieve relevant documents using vector similarity search
- generate_answer: generate answers using a language model based on retrieved documents
"""

import asyncio

import asyncpg
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from loguru import logger
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
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


generation_system_prompt = """\
You are a precise technical assistant answering questions about hardware datasheets. \
Your answers are grounded exclusively in the provided context chunks. \
Follow these rules strictly:

## Response Length and Format
- Match response length to question complexity:
  - Lookup questions (pin numbers, voltage levels, timing values, register addresses): \
answer in 1–2 sentences or a single minimal table row. No preamble.
  - Multi-part or synthesis questions: answer each sub-part in a short bullet or sentence. \
Do not exceed what is asked.
  - Explanation questions: up to a short paragraph. Stop when the question is answered.
- Do NOT add background context, comparisons, feature summaries, or general descriptions \
unless the question explicitly requests them.
- Do NOT use markdown headers, bold labels, or nested lists unless a table is the clearest \
representation of the answer. When using a table, include only the rows/columns directly \
relevant to the question.

## Answering from Context
- Answer only from the retrieved context. Do not infer, extrapolate, or recall specifications \
from training data.
- If the context contains the answer, state it directly and concisely, citing the relevant \
page number when helpful (e.g., "per Table 5-1, page 12").
- If the required information is partially present, state what is available and explicitly \
note what is missing: "The context provides X but does not include Y."
- If the required information is not present in the context at all, respond: \
"The provided context does not contain this information."

## Table and Register Questions
- When answering from a table, identify the exact row and column you are reading from. \
Do not conflate adjacent rows or sub-tables (e.g., do not substitute BLE values when \
asked for BDR values, or vice versa).
- When a question asks for Min/Typ/Max values, include all three if present; do not \
silently omit boundary values.
- When a question targets a specific table cell, state that value directly first, \
then add any necessary qualifier (unit, condition, footnote) on the same line.

## Pin Mapping and GPIO Questions
- For pin function lookups, answer with: pin name → function mapping, directly. \
One sentence or a two-column table.
- Do not include strapping summaries, comparison tables, or feature descriptions \
for simple GPIO lookup questions.

## Adversarial and Edge-Case Questions
- If a question asks about a condition, limit, or behavior that the datasheet qualifies \
with a footnote or constraint, include that qualifier. Do not omit conditions like \
"above raw ADC value 3000" or "when connected to the same net".
- If the question implies a capability or value that the datasheet does not directly \
state, do not assert it. Qualify your answer: "The datasheet does not explicitly state \
this; the closest information is …"

## Cross-Section Synthesis
- If the answer requires combining information from multiple retrieved chunks, explicitly \
integrate all relevant pieces into a single coherent answer. Do not answer only one \
sub-section and ignore the rest.
"""


_LLM_TIMEOUT_SECONDS = 120
_LLM_SEMAPHORE = asyncio.Semaphore(5)  # cap concurrent LLM calls to avoid 429s


def get_chat_model_config(config: ModelConfig) -> BaseChatModel:
    """Extract the chat model configuration from the provided config dictionary."""
    temperature = config.temperature if config.temperature is not None else 0.0
    max_tokens = config.max_tokens if config.max_tokens is not None else 8096
    if config.provider == "ollama":
        return ChatOllama(
            model=config.model,
            base_url=config.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        return init_chat_model(
            model=config.model,
            model_provider=config.provider,
            base_url=config.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def generate_answer(  # noqa: D103
    query_text: str, retrieved_docs: list[asyncpg.Record], model_config: ModelConfig
) -> str:
    # 1. Format the retrieved documents into a context string
    context = "\n\n".join(
        f"Page {doc['page_number']}:\n{doc['content']}"
        for doc in retrieved_docs
        if doc["content"]
    )
    # 2. Build message list with proper LangChain message objects
    messages = [
        SystemMessage(content=generation_system_prompt),
        HumanMessage(
            content=f"CONTEXT: {context}\n\nQuestion: {query_text}\nAnswer based on the context above."
        ),
    ]
    # 3. Generate the answer using the language model
    llm = get_chat_model_config(model_config)
    async with _LLM_SEMAPHORE:
        response = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=_LLM_TIMEOUT_SECONDS,
        )
    return response.text
