"""Run and score curated RAG evaluations with RAGAS.

This module executes the end-to-end Phase 2 evaluation flow:
1) load curated test samples,
2) run retrieval + generation per sample,
3) compute RAGAS metrics,
4) aggregate overall/per-category results,
5) persist artifacts for experiment tracking.
"""

import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from ragas.dataset import Dataset
from ragas.dataset_schema import SingleTurnSample
from ragas.experiment import experiment

from production_rag.core.config import (
    ModelConfig,
    load_config,
)
from production_rag.evals.cli import parse_args
from production_rag.evals.metrics import build_ragas_metrics, score_sample_metrics
from production_rag.evals.reporting import (
    compute_report,
    ensure_experiment_dir,
    print_summary,
)
from production_rag.evals.schemas import ExperimentConfig, ExperimentResult
from production_rag.evals.testset import load_testset
from production_rag.pipeline.retrieve import generate_answer, search_custom_docs


@experiment(experiment_model=ExperimentResult)
async def run_sample(
    row: dict[str, Any],
    top_k: int,
    similarity_threshold: float | None,
    embedding_model: ModelConfig,
    chat_model: ModelConfig,
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
        model_config=embedding_model,
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
        query_text=user_input,
        retrieved_docs=retrieved_rows,
        model_config=chat_model,
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


async def _main() -> None:
    """Async implementation of the evaluation experiment CLI."""
    load_dotenv()
    args = parse_args()

    testset_path = Path(args.testset)
    base_output_dir = Path(args.output_dir)
    experiment_name = (
        args.experiment_name
        if args.experiment_name
        else datetime.now(UTC).strftime("eval-%Y%m%d-%H%M%S")
    )
    config_path = Path(args.config_path)
    app_config = load_config(config_path)

    config = ExperimentConfig(
        experiment_name=experiment_name,
        testset_path=str(testset_path),
        output_dir=str(base_output_dir),
        top_k=app_config.retrieval.top_k,
        similarity_threshold=app_config.retrieval.similarity_threshold,
        reranker=app_config.retrieval.reranker,
        chat_model=app_config.pipeline.chat.model,
        eval_model=app_config.eval.llm.model,
        embedding_model=app_config.pipeline.embeddings.model,
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
    metrics = build_ragas_metrics(
        eval_llm=app_config.eval.llm, embedding_model=app_config.eval.embeddings
    )

    logger.info("Running experiment with top_k={}...", config.top_k)
    experiment_view = await run_sample.arun(
        dataset=dataset,
        name=experiment_name,
        backend="inmemory",
        top_k=config.top_k,
        similarity_threshold=config.similarity_threshold,
        embedding_model=app_config.pipeline.embeddings,
        chat_model=app_config.pipeline.chat,
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


def main() -> None:
    """Synchronous entry point for the rag-evaluate CLI command."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
