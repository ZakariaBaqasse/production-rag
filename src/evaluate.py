"""Run and score curated RAG evaluations with RAGAS.

This module executes the end-to-end Phase 2 evaluation flow:
1) load curated test samples,
2) run retrieval + generation per sample,
3) compute RAGAS metrics,
4) aggregate overall/per-category results,
5) persist artifacts for experiment tracking.
"""

import argparse
import ast
import asyncio
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from openai import AsyncOpenAI

import pandas as pd
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from ragas.embeddings.base import embedding_factory
from loguru import logger
from pydantic import BaseModel
from ragas.dataset import Dataset
from ragas.dataset_schema import SingleTurnSample
from ragas.experiment import experiment
from ragas.metrics.collections import (
    Faithfulness,
    FactualCorrectness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)
from ragas.metrics.base import SimpleBaseMetric
from ragas.llms import llm_factory
from tenacity import retry, wait_exponential

from src.constants import (
    EMBEDDING_MODEL_NAME,
    EVAL_EMBEDDING_MODEL_NAME,
    EVAL_MODEL_NAME,
    OLLAMA_CHAT_MODEL,
    OLLAMA_CLOUD_BASE_URL,
)
from src.retrieve import search_custom_docs


@dataclass
class ExperimentConfig:
    """Configuration snapshot for a single evaluation run.

    Stored to disk so every experiment can be reproduced and compared.
    """

    experiment_name: str
    testset_path: str
    output_dir: str
    top_k: int
    similarity_threshold: float | None
    reranker: str
    chat_model: str
    eval_model: str
    embedding_model: str


class ExperimentResult(BaseModel):
    """Per-sample output row produced by the experiment runner.

    This model captures the generated answer and the fields required for
    downstream RAGAS scoring and category-level analysis.
    """

    user_input: str
    retrieved_contexts: list[str]
    response: str
    reference: str
    reference_contexts: list[str]
    question_category: str
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    factual_correctness: float | None = None
    answer_relevancy: float | None = None
    metric_error: str | None = None


METRIC_COLUMN_ORDER = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation execution."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation using the curated testset and the live RAG pipeline."
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="./artifacts/ragas/curated_testset.csv",
        help="Path to curated testset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/ragas/experiments",
        help="Directory where experiment artifacts are saved.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Optional experiment name. Defaults to a timestamp-based name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Optional similarity floor. Retrieved chunks below this are dropped.",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default="none",
        choices=["none"],
        help="Re-ranker strategy (Phase 2 supports only 'none').",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default=OLLAMA_CHAT_MODEL,
        help="Chat model used by your RAG pipeline for answer generation.",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=EVAL_MODEL_NAME,
        help="LLM judge model used by RAGAS metrics.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="Embedding model used by retrieval and answer_relevancy metric.",
    )
    return parser.parse_args()


def _safe_parse_list(value: Any) -> list[str]:
    """Parse a list-like CSV field robustly.

    The curated CSV may contain Python-list strings, JSON arrays, real lists,
    or plain text. This helper normalizes all of these into ``list[str]``.
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    return [text]


def load_testset(path: Path) -> list[dict[str, Any]]:
    """Load and normalize the curated testset from CSV.

    Returns rows containing only the fields needed for experiment execution.
    """
    if not path.exists():
        raise FileNotFoundError(f"Testset not found: {path}")

    df = pd.read_csv(path)
    required_columns = {
        "user_input",
        "reference_contexts",
        "reference",
        "question_category",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in testset: {missing}")

    # Defensive cleanup: some files include an accidental duplicated header row.
    if "synthesizer_name" in df.columns:
        df = df[df["synthesizer_name"] != "synthesizer_name"]

    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append(
            {
                "user_input": str(row.get("user_input", "")).strip(),
                "reference": str(row.get("reference", "")).strip(),
                "reference_contexts": _safe_parse_list(row.get("reference_contexts")),
                "question_category": str(row.get("question_category", "")).strip(),
            }
        )

    records = [r for r in records if r["user_input"]]
    if not records:
        raise ValueError("No valid rows found in testset after parsing.")

    return records


def build_generation_prompt(question: str, contexts: list[str]) -> str:
    """Build the generation prompt from the query and retrieved contexts."""
    context_block = "\n\n".join(
        [f"Excerpt {index + 1}: {ctx}" for index, ctx in enumerate(contexts)]
    )
    return (
        "Answer the question using only the provided excerpts. "
        "If the answer is not in the excerpts, say that explicitly.\n\n"
        f"Question: {question}\n\n"
        f"Excerpts:\n{context_block}"
    )


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def generate_answer(
    question: str,
    contexts: list[str],
    chat_model_name: str,
) -> str:
    """Generate an answer using the specified chat model and prompt."""
    model = ChatOllama(model=chat_model_name, base_url=OLLAMA_CLOUD_BASE_URL)
    prompt = build_generation_prompt(question, contexts)
    response = await model.ainvoke(prompt)
    return str(response.content)


@experiment(experiment_model=ExperimentResult)
async def run_sample(
    row: dict[str, Any],
    top_k: int,
    similarity_threshold: float | None,
    embedding_model: str,
    chat_model_name: str,
    metrics: list[Any],
) -> ExperimentResult:
    """Run retrieval and answer generation for one evaluation sample.

    This function is decorated with ``@experiment`` so RAGAS can execute it for
    each dataset row and collect typed outputs.
    """
    user_input = row["user_input"]

    retrieved_rows = await search_custom_docs(
        query_text=user_input,
        top_k=top_k,
        model=embedding_model,
    )

    # Optional hard filter to study precision/recall tradeoffs by similarity.
    if similarity_threshold is not None:
        retrieved_rows = [
            result
            for result in retrieved_rows
            if float(result.get("similarity", 0.0)) >= similarity_threshold
        ]

    retrieved_contexts = [str(result["content"]) for result in retrieved_rows]

    response_text = await generate_answer(
        question=user_input,
        contexts=retrieved_contexts,
        chat_model_name=chat_model_name,
    )

    sample = SingleTurnSample(
        user_input=user_input,
        retrieved_contexts=retrieved_contexts,
        reference_contexts=row["reference_contexts"],
        response=response_text,
        reference=row["reference"],
    )
    metric_scores, metric_error = await score_sample_metrics(
        metrics=metrics, sample=sample
    )

    return ExperimentResult(
        user_input=user_input,
        retrieved_contexts=retrieved_contexts,
        response=response_text,
        reference=row["reference"],
        reference_contexts=row["reference_contexts"],
        question_category=row["question_category"],
        context_precision=metric_scores.get("context_precision"),
        context_recall=metric_scores.get("context_recall"),
        faithfulness=metric_scores.get("faithfulness"),
        factual_correctness=metric_scores.get("factual_correctness"),
        answer_relevancy=metric_scores.get("answer_relevancy"),
        metric_error=metric_error,
    )


def build_ragas_metrics(
    eval_llm: str,
) -> list[Any]:
    """Construct the metric suite used for scoring.

    Notes:
    - This project uses ``ragas==0.4.3`` with LangChain model wrappers.
      The classic metric classes remain the compatible path for Gemini via
      ``ChatGoogleGenerativeAI``.
    """
    client = AsyncOpenAI()
    eval_model = llm_factory(
        client=client, model=eval_llm, temperature=0, max_tokens=8192
    )
    embeddings = embedding_factory(
        "openai", model=EVAL_EMBEDDING_MODEL_NAME, client=client
    )
    return [
        ContextPrecision(llm=eval_model),
        ContextRecall(llm=eval_model),
        Faithfulness(llm=eval_model),
        FactualCorrectness(llm=eval_model),
        AnswerRelevancy(llm=eval_model, embeddings=embeddings),
    ]


async def score_sample_metrics(
    metrics: list[SimpleBaseMetric],
    sample: SingleTurnSample,
) -> tuple[dict[str, float | None], str | None]:
    """Score one sample across all metrics without aborting on single failures."""
    scores: dict[str, float | None] = {}
    errors: list[str] = []

    sample_data = sample.model_dump()
    for metric in metrics:
        metric_name = str(metric.name)
        try:
            sig = inspect.signature(metric.ascore)
            valid_keys = set(sig.parameters.keys()) - {"self", "llm"}
            filtered = {k: v for k, v in sample_data.items() if k in valid_keys}
            score = await metric.ascore(**filtered)
            scores[metric_name] = float(score.value)
        except Exception as exc:
            scores[metric_name] = None
            errors.append(f"{metric_name}: {exc}")

    return scores, ("; ".join(errors) if errors else None)


def compute_report(scores_df: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    """Build the final JSON report payload.

    Includes overall means, per-category means, and a low-score failure list.
    """
    metric_columns = [col for col in METRIC_COLUMN_ORDER if col in scores_df.columns]

    overall = (
        scores_df[metric_columns].mean(numeric_only=True).fillna(0.0).to_dict()
        if metric_columns
        else {}
    )

    per_category: dict[str, Any] = {}
    if metric_columns and "question_category" in scores_df.columns:
        grouped = scores_df.groupby("question_category", dropna=False)
        for category, group in grouped:
            key = str(category) if pd.notna(category) else "unknown"
            values = group[metric_columns].mean(numeric_only=True).fillna(0.0).to_dict()
            values["count"] = int(len(group))
            per_category[key] = values

    failures: list[dict[str, Any]] = []
    if "factual_correctness" in scores_df.columns:
        bad = scores_df[scores_df["factual_correctness"] < 0.5]
        for row in bad.to_dict(orient="records"):
            failures.append(
                {
                    "user_input": row.get("user_input", ""),
                    "question_category": row.get("question_category", ""),
                    "factual_correctness": row.get("factual_correctness", None),
                    "context_recall": row.get("context_recall", None),
                }
            )

    return {
        "experiment_name": config.experiment_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "overall": overall,
        "per_category": per_category,
        "failures": failures,
    }


def ensure_experiment_dir(base_output_dir: Path, experiment_name: str) -> Path:
    """Create (or reuse) the output directory for a named experiment run."""
    run_dir = base_output_dir / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def print_summary(report: dict[str, Any]) -> None:
    """Log a compact human-readable summary of evaluation results."""
    logger.info("Experiment: {}", report["experiment_name"])
    logger.info("Overall metrics:")
    for metric, value in report.get("overall", {}).items():
        logger.info("  {} = {:.4f}", metric, float(value))

    logger.info("Per-category metrics:")
    for category, values in report.get("per_category", {}).items():
        count = values.get("count", 0)
        metric_text = ", ".join(
            [
                f"{metric}={float(score):.4f}"
                for metric, score in values.items()
                if metric != "count"
            ]
        )
        logger.info("  [{}] count={} | {}", category, count, metric_text)

    failures = report.get("failures", [])
    logger.info("Failure spotlight (factual_correctness < 0.5): {}", len(failures))


async def main() -> None:
    """CLI entrypoint for running one complete evaluation experiment."""
    load_dotenv()
    args = parse_args()

    testset_path = Path(args.testset)
    base_output_dir = Path(args.output_dir)
    experiment_name = (
        args.experiment_name
        if args.experiment_name
        else datetime.now(UTC).strftime("eval-%Y%m%d-%H%M%S")
    )

    config = ExperimentConfig(
        experiment_name=experiment_name,
        testset_path=str(testset_path),
        output_dir=str(base_output_dir),
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        reranker=args.reranker,
        chat_model=args.chat_model,
        eval_model=args.eval_model,
        embedding_model=args.embedding_model,
    )

    logger.info("Loading testset from {}", testset_path)
    rows = load_testset(testset_path)
    logger.info("Loaded {} evaluation samples", len(rows))

    dataset = Dataset(
        name="curated_testset",
        backend="inmemory",
        data_model=None,
        data=rows,
    )

    # Reuse a single deterministic evaluator across all rows.
    metrics = build_ragas_metrics(eval_llm=config.eval_model)

    logger.info("Running experiment with top_k={}...", config.top_k)
    experiment_view = await run_sample.arun(
        dataset=dataset,
        name=experiment_name,
        backend="inmemory",
        top_k=config.top_k,
        similarity_threshold=config.similarity_threshold,
        embedding_model=config.embedding_model,
        chat_model_name=config.chat_model,
        metrics=metrics,
    )
    experiment_df = experiment_view.to_pandas()

    logger.info(
        "Experiment completed with inline metric scoring across {} metrics.",
        len(metrics),
    )
    merged_scores_df = experiment_df.copy()

    report = compute_report(scores_df=merged_scores_df, config=config)
    print_summary(report)

    run_dir = ensure_experiment_dir(base_output_dir, experiment_name)
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.csv"
    scores_path = run_dir / "scores.csv"
    report_path = run_dir / "report.json"

    with open(config_path, "w") as config_file:
        json.dump(asdict(config), config_file, indent=2)

    experiment_df.to_csv(results_path, index=False)
    merged_scores_df.to_csv(scores_path, index=False)

    with open(report_path, "w") as report_file:
        json.dump(report, report_file, indent=2)

    logger.info("Saved config: {}", config_path)
    logger.info("Saved raw results: {}", results_path)
    logger.info("Saved scored rows: {}", scores_path)
    logger.info("Saved report: {}", report_path)


if __name__ == "__main__":
    asyncio.run(main())
