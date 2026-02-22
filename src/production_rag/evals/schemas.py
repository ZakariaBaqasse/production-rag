"""Schemas for evaluation experiments and results.

This module defines the data models used for storing experiment configurations
and evaluation results, including per-sample outputs with RAGAS scoring metrics.
"""

from dataclasses import dataclass

from pydantic import BaseModel


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
