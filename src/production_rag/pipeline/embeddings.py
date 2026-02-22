import json  # noqa: D100
import re

from loguru import logger
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from production_rag.core.config import ModelConfig
from production_rag.core.database import get_db_connection
from production_rag.pipeline.utils import get_embedding_model

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Tuned for technical markdown (datasheets) so sections/tables are less likely
# to be fragmented while still fitting embedding model limits.
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200

# Matches fenced code blocks (including mermaid) and markdown pipe tables.
# These structures should be treated as atomic units during recursive splitting.
ATOMIC_BLOCK_PATTERN = re.compile(
    r"```[\w+-]*\n[\s\S]*?```"
    r"|^\|.*\|\s*\n^\|(?:\s*:?-{3,}:?\s*\|)+\s*\n(?:^\|.*\|\s*(?:\n|$))+",
    re.MULTILINE,
)


def _protect_atomic_blocks(text: str) -> tuple[str, dict[str, str]]:
    """Replace atomic markdown blocks with tokens prior to size-based splitting.

    Returns the protected text and a token->original-block mapping that can be
    used to restore the original content after splitting.
    """
    replacements: dict[str, str] = {}
    protected_parts: list[str] = []
    cursor = 0

    for index, match in enumerate(ATOMIC_BLOCK_PATTERN.finditer(text)):
        token = f"<<ATOMIC_BLOCK_{index}>>"
        replacements[token] = match.group(0)
        protected_parts.append(text[cursor : match.start()])
        protected_parts.append(token)
        cursor = match.end()

    protected_parts.append(text[cursor:])
    return "".join(protected_parts), replacements


def _restore_atomic_blocks(text: str, replacements: dict[str, str]) -> str:
    """Restore fenced code/table blocks from token placeholders."""
    restored = text
    for token, block in replacements.items():
        restored = restored.replace(token, block)
    return restored


def split_markdown_documents(documents: list[Document]) -> list[Document]:
    """Split documents with a semantic-first, size-second markdown strategy.

    Stage 1: split on markdown headers to preserve section hierarchy metadata.
    Stage 2: recursively split only oversized sections by character limits.
    Atomic blocks (tables/fenced code) are protected so they stay intact.
    """
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[Document] = []

    for document in documents:
        # Semantic split first so chunks retain Header 1/2/3 metadata.
        section_documents = markdown_splitter.split_text(document.page_content)
        if not section_documents:
            section_documents = [
                Document(page_content=document.page_content, metadata={})
            ]

        for section in section_documents:
            merged_metadata = {**document.metadata, **section.metadata}
            # Protect non-splittable markdown structures before recursive splitting.
            protected_content, replacements = _protect_atomic_blocks(
                section.page_content
            )

            if len(protected_content) <= CHUNK_SIZE:
                chunks.append(
                    Document(
                        page_content=_restore_atomic_blocks(
                            protected_content,
                            replacements,
                        ),
                        metadata=merged_metadata,
                    )
                )
                continue

            split_contents = recursive_splitter.split_text(protected_content)
            for split_content in split_contents:
                chunks.append(
                    Document(
                        page_content=_restore_atomic_blocks(
                            split_content, replacements
                        ),
                        metadata=merged_metadata,
                    )
                )

    return chunks


async def embed_documents(
    documents: list[Document], model_config: ModelConfig
) -> list[float]:
    """Chunk markdown documents, embed chunks, and persist vectors in batches."""
    try:
        chunks = split_markdown_documents(documents)

        logger.info(f"--- 3. Embedding & Indexing {len(chunks)} Chunks ---")

        # Using your local Qwen model
        embeddings = get_embedding_model(model_config)

        # Generate embeddings for all chunks (run in thread to avoid blocking)
        logger.info("Generating embeddings and storing in database...")
        # Generate embeddings in batches to avoid overwhelming the model
        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_content = [chunk.page_content for chunk in batch_chunks]
            batch_embeddings = await embeddings.aembed_documents(batch_content)
            await store_embeddings_in_db(batch_embeddings, batch_chunks)
    except Exception as e:
        logger.error(f"Error getting embedding from {model_config.provider}: {e}")
        # Return empty list or re-raise depending on strategy.
        # Here we re-raise to catch upstream
        raise


async def store_embeddings_in_db(embeddings: list[float], chunks: list[Document]):
    """Store the generated embeddings in the PostgreSQL database."""
    logger.info("Inserting into custom 'document_pages' table...")

    for i, chunk in enumerate(chunks):
        vector = embeddings[i]
        content = chunk.page_content
        # Extract metadata safely (Assuming your chunk metadata has these keys)
        # You might need to adjust based on how you loaded the PDF
        page_num = chunk.metadata.get("page_number", 0)  # Default to 0 if not found
        try:
            conn = await get_db_connection()
            # Convert vector list to string format for pgvector
            vector_str = "[" + ",".join(map(str, vector)) + "]"
            await conn.execute(
                """
                INSERT INTO document_pages (page_number, content, embedding, metadata)
                VALUES ($1, $2, $3, $4)
            """,
                int(page_num),
                content,
                vector_str,
                json.dumps(chunk.metadata),
            )
        except Exception as e:
            logger.error(f"Failed to insert page {page_num}: {e}")
        finally:
            await conn.close()
    logger.info("✅ Custom Ingestion Complete.")
