"""CLI argument parsing for evaluation execution.

This module provides command-line argument parsing for running RAGAS
evaluation using the curated testset and the live RAG pipeline.
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation execution."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation using the curated testset and the live RAG pipeline."
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="./artifacts/testsets/final_ragas_testset.csv",
        help="Path to curated testset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/experiments",
        help="Directory where experiment artifacts are saved.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Optional experiment name. Defaults to a timestamp-based name.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./config/experiment_config.yaml",
        help="Path to the experiment configuration file.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="sqlite:///mlruns.db",
        help=(
            "MLflow tracking URI. Defaults to a local SQLite database 'mlruns.db'. "
            "Pass an http:// URL to log to a remote tracking server."
        ),
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        default=False,
        help="Disable MLflow tracking for this run.",
    )
    return parser.parse_args()
