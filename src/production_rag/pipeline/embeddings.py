import json  # noqa: D100
import re
from dataclasses import dataclass, field

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

# Matches fenced code blocks only — these remain indivisible atomic units
# during recursive splitting.  Pipe tables are no longer treated as atomic
# blocks; they are split at row boundaries via the PipeTable AST instead.
ATOMIC_BLOCK_PATTERN = re.compile(
    r"```[\w+-]*\n[\s\S]*?```",
    re.MULTILINE,
)

# Detection-only pattern: locates contiguous pipe-table blocks within section
# text.  Used by split_markdown_documents to find tables for row-boundary
# splitting.  Both header/separator rows and data rows match because they all
# start and end with a pipe character.
_PIPE_TABLE_PATTERN = re.compile(
    r"(?:^[ \t]*\|.+\|[ \t]*(?:\n|$))+",
    re.MULTILINE,
)

# "Cont'd on next page" / "Continued on next page" trailer emitted by LlamaParse
# when a table overflows a page boundary.
_CONT_TRAILER_RE = re.compile(
    r"\s*Cont(?:'?d|inued)\s+on\s+next\s+page\s*$", re.IGNORECASE
)
# Last row of a markdown pipe table at the bottom of page N.
_MD_TABLE_CLOSE_RE = re.compile(r"[ \t]*\|[^\n]+\|[ \t]*\s*$")
# Opening rows of a markdown pipe table at the top of page N+1: a header row
# followed by a separator row that LlamaParse emits on continuation pages.
_MD_TABLE_OPEN_RE = re.compile(
    r"^\s*\|[^\n]+\|\n[ \t]*\|[-|: \t]+\|\s*\n",
    re.IGNORECASE,
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


@dataclass
class PipeTable:
    """Lightweight AST node for a markdown pipe table.

    Provides schema comparison and row-boundary splitting so large tables can
    be divided into self-contained chunks without breaking the markdown
    structure.
    """

    header_row: str  # raw header line, e.g. "| Col A | Col B |\n"
    separator_row: str  # raw separator line, e.g. "|---|---|\n"
    data_rows: list[str]  # remaining data lines (each ends with \n)
    headers: list[str] = field(init=False)

    def __post_init__(self) -> None:
        """Derive normalised column names from the raw header row."""
        # Normalised column names used for schema comparison.
        # Empty cells (trailing pipes, merged-cell artefacts) are excluded so
        # that minor formatting variations like "| A | B | | |" and "| A | B |"
        # are treated as the same schema.
        self.headers = [
            cell.strip().lower()
            for cell in self.header_row.strip().strip("|").split("|")
            if cell.strip()
        ]

    @classmethod
    def parse(cls, raw: str) -> "PipeTable | None":
        """Parse a raw pipe-table string. Returns None if malformed."""
        lines = raw.splitlines(keepends=True)
        if len(lines) < 2:
            return None
        sep_idx = next(
            (
                i
                for i, ln in enumerate(lines)
                if re.match(r"^[ \t]*\|[-|: \t]+\|[ \t]*$", ln.rstrip())
            ),
            None,
        )
        if sep_idx != 1:  # separator must be the second line for a valid table
            return None
        return cls(
            header_row=lines[0],
            separator_row=lines[1],
            data_rows=lines[2:],
        )

    def same_schema(self, other: "PipeTable") -> bool:
        """True if both tables have the same normalised column names."""
        return self.headers == other.headers

    def render_chunk(self, rows: list[str]) -> str:
        """Render a subset of data rows as a self-contained table."""
        return self.header_row + self.separator_row + "".join(rows)

    def split_by_rows(self, chunk_size: int, overlap_rows: int = 1) -> list[str]:
        """Return a list of self-contained table strings, each <= chunk_size chars.

        Every output chunk starts with the original header + separator rows so
        it is readable in isolation.  The last ``overlap_rows`` data rows of
        chunk N are repeated at the start of chunk N+1 to preserve context
        across boundaries.

        A single data row that exceeds ``chunk_size`` on its own is kept intact
        (splitting mid-row would produce malformed markdown).
        """
        if not self.data_rows:
            return [self.render_chunk([])]

        results: list[str] = []
        header_overhead = len(self.header_row) + len(self.separator_row)
        start = 0

        while start < len(self.data_rows):
            batch: list[str] = []
            advance = 0
            for row in self.data_rows[start:]:
                candidate_size = header_overhead + sum(len(r) for r in batch) + len(row)
                if batch and candidate_size > chunk_size:
                    break
                batch.append(row)
                advance += 1

            if not batch:
                # Single row exceeds chunk_size — keep it intact rather than
                # producing malformed markdown.
                batch = [self.data_rows[start]]
                advance = 1

            results.append(self.render_chunk(batch))
            start += max(1, advance - overlap_rows)

        return results

    def _first_cell(self, row: str) -> str:
        """Return the normalised value of the first data cell in a pipe-table row."""
        cells = row.strip().strip("|").split("|")
        return cells[0].strip() if cells else ""

    def split_by_rows_with_range(
        self, chunk_size: int, overlap_rows: int = 1
    ) -> list[tuple[str, str, str]]:
        """Like split_by_rows but returns (chunk_text, key_start, key_end) tuples.

        key_start and key_end are the first-column values of the first and last
        data rows in each chunk, enabling downstream metadata pre-filters to
        locate the chunk that plausibly contains a specific identifier.
        """
        if not self.data_rows:
            return [(self.render_chunk([]), "", "")]

        results: list[tuple[str, str, str]] = []
        start = 0
        while start < len(self.data_rows):
            batch: list[str] = []
            for row in self.data_rows[start:]:
                candidate = self.render_chunk(batch + [row])
                if len(candidate) > chunk_size and batch:
                    break
                batch.append(row)
            key_start = self._first_cell(batch[0]) if batch else ""
            key_end = self._first_cell(batch[-1]) if batch else ""
            results.append((self.render_chunk(batch), key_start, key_end))
            advance = len(batch) if batch else 1
            start += max(1, advance - overlap_rows)
        return results


def _merge_split_tables(documents: list[Document]) -> list[Document]:
    """Merge consecutive documents where a table spans a page boundary.

    LlamaParse emits one Document per page.  Two split-table patterns are
    handled:

    * **Pipe tables**: previous page ends with a ``|...|`` row and the next
      page opens with a repeated header + separator row that LlamaParse emits
      on continuation pages.  The repeated header and separator are stripped
      so the result is a single continuous table.

    Metadata from the *first* document is preserved; a ``merged_pages`` key is
    added when a merge occurs.  Merges are applied iteratively so tables that
    span three or more pages are fully stitched.
    """
    if not documents:
        return documents

    merged: list[Document] = [documents[0]]
    for doc in documents[1:]:
        prev = merged[-1]
        prev_page = prev.metadata.get("page_number", "?")
        curr_page = doc.metadata.get("page_number", "?")

        # ── Markdown pipe-table continuation ────────────────────────────────
        # Strip the "Cont'd on next page" trailer first (it sits after the
        # last table row as plain text), then check whether the page ends
        # with a |...| row and the next page opens with a header+separator.
        prev_stripped = _CONT_TRAILER_RE.sub("", prev.page_content)
        if _MD_TABLE_CLOSE_RE.search(prev_stripped) and _MD_TABLE_OPEN_RE.match(
            doc.page_content.lstrip()
        ):
            # Schema-identity check: compare the column names of the trailing
            # table on page N with the incoming table header on page N+1.
            # LlamaParse repeats the header+separator on every continuation
            # page, so a mismatch means this is a *new* table, not a
            # continuation, and the merge must not fire.
            open_match = _MD_TABLE_OPEN_RE.match(doc.page_content.lstrip())
            next_table = PipeTable.parse(open_match.group(0)) if open_match else None
            prev_table_matches = list(_PIPE_TABLE_PATTERN.finditer(prev_stripped))
            prev_table = (
                PipeTable.parse(prev_table_matches[-1].group(0))
                if prev_table_matches
                else None
            )

            if prev_table and next_table and not prev_table.same_schema(next_table):
                logger.debug(
                    f"Not merging pages {prev_page} -> {curr_page}: "
                    f"table schema changed "
                    f"({prev_table.headers} → {next_table.headers})"
                )
                merged.append(doc)
                continue

            # Schemas match (or could not be parsed — fall through to merge).
            logger.debug(
                f"Merging pipe-table split across pages {prev_page} -> {curr_page}"
            )
            # Keep all rows from page N — the last row is data, not markup.
            prev_body = prev_stripped
            # Drop the repeated header+separator from page N+1 — keep only
            # the continuation data rows.
            curr_body = _MD_TABLE_OPEN_RE.sub("", doc.page_content.lstrip())
            merged[-1] = Document(
                page_content=prev_body + "\n" + curr_body,
                metadata={
                    **prev.metadata,
                    "merged_pages": f"{prev_page},{curr_page}",
                },
            )
            continue

        merged.append(doc)

    return merged


def split_markdown_documents(documents: list[Document]) -> list[Document]:
    """Split documents with a semantic-first, size-second markdown strategy.

    Pre-pass: merge consecutive documents where a table spans a page boundary.
    Stage 1:  split on markdown headers to preserve section hierarchy metadata.
    Stage 2:  recursively split only oversized sections by character limits.
              Atomic blocks (tables/fenced code) are protected so they stay
              intact.  The size gate uses the *original* section length, not the
              tokenized length, so a section containing a large table is never
              silently emitted as an oversized chunk.
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

    # Pre-pass: stitch tables that LlamaParse split across page boundaries.
    documents = _merge_split_tables(documents)

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

            # Gate on the *original* length — not the protected (tokenized)
            # length which shrinks large tables down to ~25-char placeholders
            # and produces a falsely small size estimate.
            if len(section.page_content) <= CHUNK_SIZE:
                chunks.append(
                    Document(
                        page_content=section.page_content,
                        metadata=merged_metadata,
                    )
                )
                continue

            # Section is too large: walk the section linearly, splitting pipe
            # tables at row boundaries via the PipeTable AST and recursively
            # splitting any non-table prose (fenced code blocks stay atomic).
            section_content = section.page_content
            last_end = 0

            for m in _PIPE_TABLE_PATTERN.finditer(section_content):
                # Emit any prose that precedes this table via the recursive
                # splitter, with code fences still protected as atomic blocks.
                pre_text = section_content[last_end : m.start()]
                if pre_text.strip():
                    protected, replacements = _protect_atomic_blocks(pre_text)
                    for piece in recursive_splitter.split_text(protected):
                        restored = _restore_atomic_blocks(piece, replacements)
                        chunks.append(
                            Document(page_content=restored, metadata=merged_metadata)
                        )

                # Parse the table and split it at row boundaries so every
                # output chunk is a self-contained, header-prefixed table.
                table_ast = PipeTable.parse(m.group(0))
                if table_ast is None:
                    logger.warning(
                        f"Could not parse pipe table on page "
                        f"{merged_metadata.get('page_number', '?')} — emitting as-is"
                    )
                    chunks.append(
                        Document(page_content=m.group(0), metadata=merged_metadata)
                    )
                else:
                    for (
                        chunk_text,
                        key_start,
                        key_end,
                    ) in table_ast.split_by_rows_with_range(CHUNK_SIZE):
                        chunk_metadata = {
                            **merged_metadata,
                            "table_header": table_ast.headers[0]
                            if table_ast.headers
                            else "",
                            "key_col_start": key_start,
                            "key_col_end": key_end,
                        }
                        chunks.append(
                            Document(page_content=chunk_text, metadata=chunk_metadata)
                        )

                last_end = m.end()

            # Emit any trailing prose after the last table (or the entire
            # section if it contained no pipe tables at all, e.g. oversized
            # prose sections or sections with only fenced code blocks).
            tail = section_content[last_end:]
            if tail.strip():
                protected, replacements = _protect_atomic_blocks(tail)
                for piece in recursive_splitter.split_text(protected):
                    restored = _restore_atomic_blocks(piece, replacements)
                    if len(restored) > CHUNK_SIZE:
                        # Restored chunk exceeds limit because it contains an
                        # atomic block (fenced code) larger than CHUNK_SIZE.
                        # Emit intact — splitting mid-block would be worse.
                        logger.warning(
                            f"Chunk exceeds CHUNK_SIZE ({len(restored)} > {CHUNK_SIZE}) "
                            f"because it contains an atomic block that cannot be split. "
                            f"Section header: {merged_metadata.get('Header 2') or merged_metadata.get('Header 1', '(unknown)')}"
                        )
                    chunks.append(
                        Document(page_content=restored, metadata=merged_metadata)
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
        connection = await get_db_connection()
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_content = [chunk.page_content for chunk in batch_chunks]
            batch_embeddings = await embeddings.aembed_documents(batch_content)
            await store_embeddings_in_db(connection, batch_embeddings, batch_chunks)
    except Exception as e:
        logger.error(f"Error getting embedding from {model_config.provider}: {e}")
        # Return empty list or re-raise depending on strategy.
        # Here we re-raise to catch upstream
        raise
    finally:
        if "connection" in locals():
            await connection.close()


async def store_embeddings_in_db(
    connection: any, embeddings: list[list[float]], chunks: list[Document]
):
    """Store the generated embeddings in the PostgreSQL database."""
    logger.info("Inserting into custom 'document_pages' table...")

    records = [
        (
            int(chunk.metadata.get("page_number", 0)),
            chunk.page_content,
            "[" + ",".join(map(str, embeddings[i])) + "]",
            json.dumps(chunk.metadata),
        )
        for i, chunk in enumerate(chunks)
    ]

    try:
        async with connection.transaction():
            await connection.executemany(
                """
                INSERT INTO document_pages (page_number, content, embedding, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                records,
            )
    except Exception as e:
        logger.error(f"Failed to insert batch of {len(chunks)} chunks: {e}")
        raise
    logger.info("✅ Custom Ingestion Complete.")
