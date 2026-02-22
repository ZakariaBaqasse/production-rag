"""Utility functions for the production RAG pipeline.

This module provides functions for initializing embedding models
and other pipeline utilities.
"""

from langchain.embeddings import Embeddings

from production_rag.core.config import ModelConfig


def get_embedding_model(model_config: ModelConfig) -> Embeddings:
    """Initialize the embedding model based on the provided configuration."""
    if model_config.provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model_config.model, base_url=model_config.base_url
        )
    elif model_config.provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model_config.model, base_url=model_config.base_url
        )
    elif model_config.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_config.model, base_url=model_config.base_url
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {model_config.provider}")
