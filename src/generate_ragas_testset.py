import argparse  # noqa: D100
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from loguru import logger
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms.extractors import NERExtractor
from src.constants import (
    CHAT_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
)
from src.parse import parse_file


def parse_args() -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Generate a baseline RAGAS testset from a source PDF."
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        default="./src/assets/esp32_datasheet.pdf",
        help="Path to the source PDF to parse and synthesize QA pairs from.",
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=50,
        help="Number of test samples to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/ragas",
        help="Directory where CSV/JSONL outputs will be written.",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default=CHAT_MODEL_NAME,
        help="LLM model used by RAGAS for testset generation.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="Embedding model used by RAGAS while building the testset.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional cap on parsed pages/chunks used by generator (0 means all).",
    )
    parser.add_argument(
        "--use-default-transforms",
        action="store_true",
        help=(
            "Use RAGAS default transform/query pipeline instead of the stable "
            "single-hop baseline mode."
        ),
    )
    return parser.parse_args()


async def main() -> None:  # noqa: D103
    load_dotenv()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Parsing source PDF for testset generation...")
    documents = await parse_file(str(pdf_path))
    if not documents:
        raise ValueError("No documents returned from parser. Check parser/API keys.")

    if args.max_docs > 0:
        documents = documents[: args.max_docs]

    logger.info(
        "Creating baseline RAGAS testset from {} parsed documents...",
        len(documents),
    )

    llm = ChatGoogleGenerativeAI(
        model=args.generator_model,
    )
    embeddings = OllamaEmbeddings(
        model=args.embedding_model,
        base_url=OLLAMA_BASE_URL,
    )

    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings,
    )
    generator.persona_list = [
        Persona(
            name="Embedded Firmware Engineer",
            role_description=(
                "Builds firmware for ESP32-class MCUs and needs precise datasheet facts."
            ),
        ),
        Persona(
            name="Hardware Validation Engineer",
            role_description=(
                "Validates electrical limits, timing, and register defaults against specs."
            ),
        ),
        Persona(
            name="IoT Systems Integrator",
            role_description=(
                "Integrates peripherals and pin mappings in production embedded systems."
            ),
        ),
    ]

    transforms = None
    query_distribution = None
    if not args.use_default_transforms:

        def _is_document_node(node) -> bool:
            return node.type.name == "DOCUMENT"

        transforms = [
            NERExtractor(
                llm=generator.llm,
                filter_nodes=_is_document_node,
            )
        ]
        query_distribution = [
            (
                SingleHopSpecificQuerySynthesizer(
                    llm=generator.llm,
                    property_name="entities",
                ),
                1.0,
            )
        ]
        logger.info(
            "Using stable single-hop baseline pipeline (NERExtractor + "
            "SingleHopSpecificQuerySynthesizer)."
        )
    else:
        logger.info("Using RAGAS default transform/query pipeline.")

    try:
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=args.testset_size,
            transforms=transforms,
            query_distribution=query_distribution,
            raise_exceptions=True,
        )
    except ValueError as exc:
        error_text = str(exc)
        if args.use_default_transforms is False:
            raise

        logger.warning("Default pipeline failed: {}", error_text)
        logger.warning(
            "Retrying with stable single-hop baseline pipeline "
            "(NERExtractor + SingleHopSpecificQuerySynthesizer)."
        )

        def _is_document_node(node) -> bool:
            return node.type.name == "DOCUMENT"

        fallback_transforms = [
            NERExtractor(
                llm=generator.llm,
                filter_nodes=_is_document_node,
            )
        ]
        fallback_query_distribution = [
            (
                SingleHopSpecificQuerySynthesizer(
                    llm=generator.llm,
                    property_name="entities",
                ),
                1.0,
            )
        ]
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=args.testset_size,
            transforms=fallback_transforms,
            query_distribution=fallback_query_distribution,
            raise_exceptions=True,
        )

    jsonl_path = output_dir / "baseline_testset.jsonl"
    csv_path = output_dir / "baseline_testset.csv"

    testset.to_jsonl(str(jsonl_path))
    testset.to_csv(str(csv_path))

    logger.info("Testset generated successfully.")
    logger.info("JSONL: {}", jsonl_path)
    logger.info("CSV: {}", csv_path)


if __name__ == "__main__":
    asyncio.run(main())
