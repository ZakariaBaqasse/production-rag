"""Reporting module for evaluation experiments.

This module provides functions to:
- compute_report: Build JSON report payloads with overall and per-category metrics
- ensure_experiment_dir: Create output directories for experiment runs
- print_summary: Log human-readable summaries of evaluation results
"""
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attr import asdict
from loguru import logger
import pandas as pd

from production_rag.evals.schemas import ExperimentConfig

METRIC_COLUMN_ORDER = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
]
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
