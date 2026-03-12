"""Evaluation metrics module for RAG system.

This module provides functions to construct and score evaluation metrics using the RAGAS
framework. It supports multiple LLM and embedding providers including OpenAI, Google GenAI,
and Ollama.

Key functions:
- get_eval_model: Construct evaluation LLM based on provider config
- get_embedding_eval_model: Construct embedding model based on provider config
- build_ragas_metrics: Build the complete metric suite for scoring
- score_sample_metrics: Score a single sample across all metrics
"""

import inspect

from google import genai
from ollama import AsyncClient
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.base import SimpleBaseMetric

from typing import Any

from production_rag.core.config import ModelConfig
from ragas.metrics.collections import (
    Faithfulness,
    FactualCorrectness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)
from ragas.metrics.collections.factual_correctness.util import NLIStatementPrompt


class _FactualPayloadNLIPrompt(NLIStatementPrompt):
    """Custom NLI prompt that grades only the factual payload.

    Citation-source mismatches (e.g., a different Table Name) are not penalised
    as long as the core fact (numeric values, pin names, etc.) is correct.
    """

    instruction = (
        NLIStatementPrompt.instruction
        + "\nGrade ONLY the factual payload (e.g., numeric values, pin names). "
        "Do not penalize the response if the citation source (e.g., Table Name) "
        "differs from the ground truth, as long as the core fact is correct."
    )


def get_eval_model(model_config: ModelConfig) -> Any:
    """Helper to construct the evaluation LLM based on config."""
    if model_config.provider.lower() == "openai":
        client = AsyncOpenAI()
        return llm_factory(
            client=client, model=model_config.model, temperature=0, max_tokens=8192
        )
    elif model_config.provider.lower() == "googlegenai":
        client = genai.Client()
        return llm_factory(
            client=client,
            adapter="litellm",
            model=model_config.model,
            temperature=0,
            max_tokens=8192,
        )
    elif model_config.provider.lower() == "ollama":
        client = AsyncClient()
        return llm_factory(
            client=client,
            adapter="litellm",
            model=model_config.model,
            temperature=0,
            max_tokens=8192,
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_config.provider}")


def get_embedding_eval_model(model_config: ModelConfig) -> Any:
    """Helper to construct the embedding model client based on config."""
    if model_config.provider.lower() == "openai":
        client = AsyncOpenAI()
        return embedding_factory("openai", model=model_config.model, client=client)
    elif model_config.provider.lower() == "googlegenai":
        client = genai.Client()
        return embedding_factory("googlegenai", model=model_config.model, client=client)
    elif model_config.provider.lower() == "ollama":
        client = AsyncClient()
        return embedding_factory("ollama", model=model_config.model, client=client)
    else:
        raise ValueError(f"Unsupported embedding provider: {model_config.provider}")


def build_ragas_metrics(
    eval_llm: ModelConfig, embedding_model: ModelConfig
) -> list[Any]:
    """Construct the metric suite used for scoring.

    Notes:
    - This project uses ``ragas==0.4.3`` with LangChain model wrappers.
      The classic metric classes remain the compatible path for Gemini via
      ``ChatGoogleGenerativeAI``.
    """
    eval_model = get_eval_model(eval_llm)
    embeddings = get_embedding_eval_model(embedding_model)
    factual_correctness = FactualCorrectness(llm=eval_model)
    factual_correctness.nli_prompt = _FactualPayloadNLIPrompt()
    return [
        ContextPrecision(llm=eval_model),
        ContextRecall(llm=eval_model),
        Faithfulness(llm=eval_model),
        factual_correctness,
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
