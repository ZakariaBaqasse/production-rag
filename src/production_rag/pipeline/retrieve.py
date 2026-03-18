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
from flashrank import Ranker, RerankRequest
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


async def retrieve_hybrid(
    query: str,
    model_config: ModelConfig,
    top_k: int = 10,
    candidate_k: int = 100,
) -> list[asyncpg.Record]:
    """Hybrid retrieval: RRF-fuse dense ANN + BM25, return top_k results.

    candidate_k candidates are fetched from each search leg before RRF merging.
    """
    embeddings = get_embedding_model(model_config)
    query_vector = await embeddings.aembed_query(query)
    vector_str = "[" + ",".join(map(str, query_vector)) + "]"
    conn = await get_db_connection()
    try:
        final_rows = await conn.fetch(
            """
            -- 1. The Dense Search (pgvector)
            WITH semantic_search AS (
                SELECT id, content, page_number,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
                FROM document_pages
                ORDER BY embedding <=> $1
                LIMIT $3
            ),
            -- 2. The Sparse Search (BM25 / tsvector)
            keyword_search AS (
                SELECT id, content, page_number,
                    ROW_NUMBER() OVER (ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', $2)) DESC) AS rank
                FROM document_pages
                WHERE content_tsv @@ websearch_to_tsquery('english', $2)
                ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', $2)) DESC
                LIMIT $3
            )
            -- 3. The RRF Merge
            SELECT
                COALESCE(s.id, k.id) AS document_id,
                COALESCE(s.content, k.content) AS content,
                COALESCE(s.page_number, k.page_number) AS page_number,
                -- The RRF Math (using standard k=60)
                (COALESCE(1.0 / (60 + s.rank), 0.0)) +
                (COALESCE(1.0 / (60 + k.rank), 0.0)) AS rrf_score
            FROM semantic_search s
            FULL OUTER JOIN keyword_search k ON s.id = k.id
            ORDER BY rrf_score DESC
            LIMIT $4;
            """,
            vector_str,
            query,
            candidate_k,
            top_k,
        )

        # Preserve RRF rank order
        return final_rows
    except Exception as e:
        logger.error(f"Failed to perform hybrid retrieval: {e}")
        return []
    finally:
        if not conn.is_closed():
            await conn.close()


async def retrieve_hybrid_with_reranking(
    query: str,
    model_config: ModelConfig,
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2",
    top_k: int = 10,
    candidate_k: int = 100,
) -> list[asyncpg.Record]:
    """Hybrid retrieval + LLM re-ranking.

    Retrieves candidate_k documents from each leg, merges with RRF, then re-ranks the top_k results using an LLM.
    """
    initial_results = await retrieve_hybrid(
        query, model_config, top_k=candidate_k, candidate_k=candidate_k
    )
    # Re-rank the initial results using the LLM
    ranker = Ranker(model_name=reranker_model)
    passages = [
        {
            "id": i,
            "text": row["content"],
            "metadata": {"page_number": row["page_number"]},
        }
        for i, row in enumerate(initial_results)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerank_request)
    return [initial_results[r["id"]] for r in reranked_results[:top_k]]


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
