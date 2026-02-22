"""Configuration management for the production RAG system.

This module provides:
- ModelConfig: configuration for language models and embeddings
- PipelineConfig: configuration for the RAG pipeline
- EvalConfig: configuration for evaluation
- RetrievalConfig: configuration for retrieval parameters
- AppConfig: main application configuration
- load_config: function to load configuration from YAML files
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class ModelConfig(BaseModel):
    """Configuration for language models and embeddings.

    ----------
    model : str
        The model identifier or name.
    provider : str
        The provider of the model (e.g., 'openai', 'anthropic').
    base_url : str | None
        Optional base URL for the model API endpoint.
    """

    model: str
    provider: str
    base_url: str | None = None


class PipelineConfig(BaseModel):
    """Configuration for the RAG pipeline.

    ----------
    chat : ModelConfig
        Configuration for the chat language model.
    embeddings : ModelConfig
        Configuration for the embeddings model.
    """

    chat: ModelConfig
    embeddings: ModelConfig


class EvalConfig(BaseModel):
    """Configuration for evaluation.

    ----------
    llm : ModelConfig
        Configuration for the language model used in evaluation.
    embeddings : ModelConfig
        Configuration for the embeddings model used in evaluation.
    """

    llm: ModelConfig
    embeddings: ModelConfig


class RetrievalConfig(BaseModel):
    """Configuration for retrieval parameters.

    ----------
    top_k : int
        The number of top results to retrieve (default: 5).
    similarity_threshold : float | None
        Optional threshold for similarity filtering.
    reranker : str
        The reranker to use (default: 'none').
    """

    top_k: int = 5
    similarity_threshold: float | None = None
    reranker: str = "none"


class AppConfig(BaseModel):
    """Main application configuration.

    ----------
    pipeline : PipelineConfig
        Configuration for the RAG pipeline.
    eval : EvalConfig
        Configuration for evaluation.
    retrieval : RetrievalConfig
        Configuration for retrieval parameters.
    """

    pipeline: PipelineConfig
    eval: EvalConfig
    retrieval: RetrievalConfig


def load_config(path: Path) -> AppConfig:
    """Load application configuration from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns:
    -------
    AppConfig
        Parsed application configuration object.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
