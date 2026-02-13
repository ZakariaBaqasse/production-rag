import os
import hashlib
import json
from pathlib import Path
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.parsing_get_response import MarkdownPageMarkdownResultPage
from loguru import logger
from langchain_core.documents import Document

CACHE_DIR = Path(".cache/parsed")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def parse_file(file_path: str) -> list[Document]:
    file_hash = get_file_hash(file_path)
    cache_path = CACHE_DIR / f"{file_hash}.json"

    if cache_path.exists():
        logger.info(f"Using cached result for {file_path}")
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            markdown_pages = [MarkdownPageMarkdownResultPage(**page) for page in data]
            return [
                Document(
                    page.markdown,
                    metadata={"page_number": page.page_number, "file_hash": file_hash},
                )
                for page in markdown_pages
            ]
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    client = AsyncLlamaCloud(api_key=os.environ.get("LLAMA_CLOUD_API_KEY"))
    # Upload and parse a document
    try:
        file_obj = await client.files.create(file=file_path, purpose="parse")
        # client.files.get - removed as it seemed like dead code or typo
        result = await client.parsing.parse(
            file_id=file_obj.id,
            # The parsing tier. Options: fast, cost_effective, agentic, agentic_plus,
            tier="agentic",
            # The version of the parsing tier to use. Use 'latest' for the most recent version,
            version="latest",
            # 'expand' controls which result fields are returned in the response.,
            # Without it, only job metadata is returned. Available fields:,
            # - Content: "text", "markdown", "items", "metadata",
            # - Presigned URLs: "xlsx_content_metadata", "output_pdf_content_metadata", "images_content_metadata",
            expand=["text", "items", "markdown"],
        )

        # Cache the result
        try:
            # Handle Pydantic v1 vs v2
            pages_data = [
                p.model_dump(mode="json") if hasattr(p, "model_dump") else p.dict()
                for p in result.markdown.pages
            ]

            with open(cache_path, "w") as f:
                json.dump(pages_data, f)
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

        return [
            Document(
                page.markdown,
                metadata={"page_number": page.page_number, "file_hash": file_hash},
            )
            for page in result.markdown.pages
        ]
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        return []
