"""Terminal demo command for retrieval and answer generation."""

from __future__ import annotations

import argparse
import asyncio
from html import unescape
import logging
from pathlib import Path
import re
import sys

import asyncpg
from dotenv import load_dotenv
from loguru import logger

from production_rag.core.config import AppConfig, load_config
from production_rag.pipeline.retrieve import (
    generate_answer,
    retrieve_hybrid,
    retrieve_hybrid_with_reranking,
    search_custom_docs,
)

_DEFAULT_CONFIG_PATH = Path("configs/openai.yaml")
_DEFAULT_QUESTION = (
    "In the ESP32 GPIO_Matrix, what is signal number 63 and does it have a "
    "corresponding IO_MUX core input?"
)
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")
_LOW_SIGNAL_TOKENS = {
    "and",
    "connected",
    "corresponding",
    "does",
    "esp32",
    "function",
    "functions",
    "have",
    "input",
    "matrix",
    "other",
    "pin",
    "pins",
    "share",
    "signal",
    "that",
    "the",
    "these",
    "those",
    "what",
    "which",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the terminal demo."""
    parser = argparse.ArgumentParser(
        description="Run a terminal demo query against the RAG pipeline."
    )
    parser.add_argument(
        "--question",
        type=str,
        default=_DEFAULT_QUESTION,
        help="Question to ask the RAG system.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override the number of results shown and used for answer generation.",
    )
    parser.add_argument(
        "--show-full-context",
        action="store_true",
        help="Print full retrieved chunks instead of compact evidence lines.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable informational logs for debugging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Keep the terminal output recording-friendly by default."""
    level = "INFO" if verbose else "ERROR"
    logger.remove()
    logger.add(sys.stderr, level=level)

    for noisy_logger in (
        "httpx",
        "httpcore",
        "openai",
        "openai._base_client",
        "google_genai",
        "google_genai._api_client",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def normalize_text(text: str) -> str:
    """Normalize chunk content for compact terminal display."""
    normalized = re.sub(r"<(?:br|BR)\s*/?>", "\n", text)
    normalized = re.sub(
        r"</(?:tr|TR|p|P|div|DIV|li|LI|h[1-6]|H[1-6])>", "\n", normalized
    )
    normalized = re.sub(r"</(?:td|TD|th|TH)>", " | ", normalized)
    normalized = re.sub(r"<[^>]+>", "", normalized)
    normalized = unescape(normalized)
    normalized = re.sub(r"\\([_~|])", r"\1", normalized)
    normalized = normalized.replace("\r", "")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    cleaned_lines: list[str] = []
    for raw_line in normalized.splitlines():
        collapsed = _WHITESPACE_RE.sub(" ", raw_line).strip(" |\t")
        if collapsed:
            cleaned_lines.append(collapsed)
    return "\n".join(cleaned_lines)


def build_query_terms(question: str) -> list[str]:
    """Extract useful query tokens for selecting evidence lines."""
    terms: list[str] = []
    for token in _TOKEN_RE.findall(question.lower()):
        if token in _LOW_SIGNAL_TOKENS:
            continue
        if token.isdigit() or len(token) >= 3:
            if token not in terms:
                terms.append(token)
    return terms


def score_line(line: str, query_terms: list[str]) -> int:
    """Score a candidate evidence line against the demo question."""
    line_lower = line.lower()
    score = 0
    for token in query_terms:
        if token not in line_lower:
            continue
        if token.isdigit():
            score += 8
        elif (
            any(character.isdigit() for character in token)
            or "_" in token
            or "-" in token
        ):
            score += 6
        else:
            score += 2

    if "|" in line:
        score += 1
    if "pin no." in line_lower and "power supply pin" in line_lower:
        score -= 4
    return score


def truncate_line(text: str, max_chars: int = 140) -> str:
    """Keep evidence lines compact enough for a short recording clip."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def summarize_exception(error: Exception) -> str:
    """Render runtime failures as one short line for terminal users."""
    first_line = str(error).splitlines()[0].strip()
    if not first_line:
        return error.__class__.__name__
    return truncate_line(first_line, max_chars=180)


def select_relevant_excerpt(content: str, question: str) -> str:
    """Choose the most useful line from a chunk for on-screen evidence."""
    normalized = normalize_text(content)
    if not normalized:
        return "[empty chunk]"

    lines = [line for line in normalized.splitlines() if line.strip()]
    if not lines:
        return truncate_line(normalized)

    query_terms = build_query_terms(question)
    best_line = lines[0]
    best_score = -1

    for line in lines:
        score = score_line(line, query_terms)
        if score > best_score:
            best_score = score
            best_line = line

    if best_score <= 0:
        return truncate_line(best_line)

    return truncate_line(best_line)


def format_evidence(
    rows: list[asyncpg.Record],
    question: str,
    show_full_context: bool,
) -> list[str]:
    """Format retrieved rows for a clean terminal presentation."""
    formatted_rows: list[str] = []
    for index, row in enumerate(rows, start=1):
        page_number = row.get("page_number")
        page_label = f"Page {page_number}" if page_number is not None else "Page ?"
        content = str(row.get("content", ""))
        evidence_text = (
            normalize_text(content)
            if show_full_context
            else select_relevant_excerpt(content, question)
        )
        prefix = f"[{index}] {page_label} | "
        if show_full_context:
            indented = evidence_text.replace("\n", "\n    ")
            formatted_rows.append(f"{prefix}\n    {indented}")
        else:
            formatted_rows.append(prefix + evidence_text)
    return formatted_rows


async def retrieve_rows(
    question: str, app_config: AppConfig, top_k: int
) -> list[asyncpg.Record]:
    """Run the configured retrieval strategy for the demo question."""
    retrieval_config = app_config.retrieval
    embedding_model = app_config.pipeline.embeddings

    if (
        retrieval_config.perform_hybrid_retrieval
        and retrieval_config.reranker != "none"
    ):
        rows = await retrieve_hybrid_with_reranking(
            query=question,
            model_config=embedding_model,
            top_k=top_k,
            candidate_k=retrieval_config.candidate_k,
            reranker_model=retrieval_config.reranker,
        )
    elif retrieval_config.perform_hybrid_retrieval:
        rows = await retrieve_hybrid(
            query=question,
            model_config=embedding_model,
            top_k=top_k,
            candidate_k=retrieval_config.candidate_k,
        )
    else:
        rows = await search_custom_docs(
            query_text=question,
            model_config=embedding_model,
            top_k=top_k,
        )

    similarity_threshold = retrieval_config.similarity_threshold
    if similarity_threshold is None:
        return rows

    filtered_rows = [
        row for row in rows if float(row.get("similarity", 0.0)) >= similarity_threshold
    ]
    return filtered_rows


async def run_demo(args: argparse.Namespace) -> int:
    """Execute retrieval and answer generation for the terminal demo."""
    load_dotenv()
    app_config = load_config(args.config_path)
    default_top_k = min(app_config.retrieval.top_k, 20)
    top_k = args.top_k if args.top_k is not None else default_top_k

    print("Question")
    print(args.question)
    print()

    retrieved_rows = await retrieve_rows(args.question, app_config, top_k)

    print("Retrieved evidence")
    if not retrieved_rows:
        print("No documents were retrieved.")
        print()
        print("Answer")
        print("The system could not retrieve supporting context for this question.")
        return 1

    for line in format_evidence(retrieved_rows, args.question, args.show_full_context):
        print(line)

    print()
    print("Answer")
    try:
        answer = await generate_answer(
            query_text=args.question,
            retrieved_docs=retrieved_rows,
            model_config=app_config.pipeline.chat,
        )
    except Exception as error:
        if args.verbose:
            logger.exception("Answer generation failed")
        print(f"Answer generation failed: {summarize_exception(error)}")
        return 2

    print(answer.strip())
    return 0


def main() -> None:
    """Synchronous entry point for the terminal demo CLI."""
    args = parse_args()
    configure_logging(verbose=args.verbose)
    raise SystemExit(asyncio.run(run_demo(args)))


if __name__ == "__main__":
    main()
