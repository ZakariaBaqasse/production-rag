"""MLflow experiment tracking for RAG evaluation runs.

Logs parameters, per-category and overall RAGAS metrics, and all
file artifacts produced by a single evaluation run into a local
(or remote) MLflow tracking server.
"""

from pathlib import Path
from typing import Any

import mlflow

from production_rag.evals.schemas import ExperimentConfig

# The single MLflow experiment that aggregates all RAG evaluation runs.
MLFLOW_EXPERIMENT_NAME = "rag-evals"


def log_run_to_mlflow(
    report: dict[str, Any],
    config: ExperimentConfig,
    run_dir: Path,
    tracking_uri: str,
) -> str:
    """Log a completed evaluation run to MLflow.

    Creates (or reuses) the experiment ``rag-evals``, opens a new run named
    after ``config.experiment_name``, and logs:

    - **Parameters**: every field from ``ExperimentConfig`` that is a
      meaningful hyperparameter (``top_k``, ``similarity_threshold``,
      ``reranker``, ``chat_model``, ``eval_model``, ``embedding_model``).
      ``similarity_threshold`` is serialised as the string ``"none"`` when
      ``None`` so MLflow stores it as a param rather than skipping it.

    - **Metrics (overall)**: each key in ``report["overall"]``
      (``context_precision``, ``context_recall``, ``faithfulness``,
      ``factual_correctness``, ``answer_relevancy``) logged flat, e.g.
      ``context_precision = 0.823``.

    - **Metrics (per-category)**: the same five metrics for every category
      in ``report["per_category"]``, prefixed with the category name, e.g.
      ``pin_mapping/factual_correctness = 0.638``.  The ``count`` key is
      also logged as ``{category}/count``.

    - **Tags**: ``experiment_name`` and ``testset_path`` for quick filtering
      in the UI without polluting the params namespace.

    - **Artifacts**: all four files in ``run_dir`` are uploaded:
      ``config.json``, ``results.csv``, ``scores.csv``, ``report.json``.

    Args:
        report: The dict returned by ``compute_report()``.
        config: The ``ExperimentConfig`` for this run.
        run_dir: The directory where the four artifact files were saved.
        tracking_uri: MLflow tracking URI (local path or http URL).

    Returns:
        The MLflow run ID as a string.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=config.experiment_name) as run:
        # --- Parameters ---
        mlflow.log_params(
            {
                "top_k": config.top_k,
                "similarity_threshold": str(config.similarity_threshold)
                if config.similarity_threshold is not None
                else "none",
                "reranker": config.reranker,
                "chat_model": config.chat_model,
                "eval_model": config.eval_model,
                "embedding_model": config.embedding_model,
            }
        )

        # --- Tags ---
        mlflow.set_tags(
            {
                "experiment_name": config.experiment_name,
                "testset_path": config.testset_path,
            }
        )

        # --- Overall metrics ---
        overall: dict[str, float] = report.get("overall", {})
        if overall:
            mlflow.log_metrics({k: float(v) for k, v in overall.items()})

        # --- Per-category metrics ---
        per_category: dict[str, Any] = report.get("per_category", {})
        for category, values in per_category.items():
            category_metrics: dict[str, float] = {
                f"{category}/{k}": float(v) for k, v in values.items() if k != "count"
            }
            category_metrics[f"{category}/count"] = float(values.get("count", 0))
            mlflow.log_metrics(category_metrics)

        # --- Artifacts ---
        for filename in ("config.json", "results.csv", "scores.csv", "report.json"):
            artifact_path = run_dir / filename
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        return run.info.run_id  # type: ignore[no-any-return]
