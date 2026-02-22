"""Reporting module for evaluation experiments.

This module provides functions to:
- compute_report: Build JSON report payloads with overall and per-category metrics
- ensure_experiment_dir: Create output directories for experiment runs
- print_startup_panel: Render experiment config before the run starts
- print_summary: Render rich tables summarising evaluation results
- print_artifacts_panel: Render saved artifact paths after the run
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attr import asdict
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from production_rag.evals.schemas import ExperimentConfig

console = Console()

METRIC_COLUMN_ORDER = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
]


def _score_style(value: float | None) -> str:
    """Return a Rich style string based on the numeric score."""
    if value is None:
        return "dim"
    if value >= 0.7:
        return "bold green"
    if value >= 0.5:
        return "yellow"
    return "bold red"


def _fmt_score(value: float | None) -> Text:
    """Format a score as a Rich Text object with colour coding."""
    if value is None:
        return Text("N/A", style="dim")
    return Text(f"{value:.4f}", style=_score_style(value))


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


def print_startup_panel(
    experiment_name: str,
    config_file: Path,
    testset_path: Path,
    sample_count: int,
    chat_model: str,
    chat_provider: str,
    embedding_model: str,
    embedding_provider: str,
    eval_model: str,
    eval_provider: str,
    top_k: int,
    similarity_threshold: float | None,
) -> None:
    """Render a startup panel summarising the experiment configuration."""
    threshold_str = (
        str(similarity_threshold) if similarity_threshold is not None else "none"
    )
    content = (
        f"[bold cyan]Experiment[/bold cyan]       {experiment_name}\n"
        f"[bold cyan]Config file[/bold cyan]      {config_file}\n"
        f"[bold cyan]Testset[/bold cyan]          {testset_path}  ([bold]{sample_count}[/bold] samples)\n"
        "\n"
        f"[bold cyan]Chat model[/bold cyan]       {chat_model}  [dim]({chat_provider})[/dim]\n"
        f"[bold cyan]Embeddings[/bold cyan]       {embedding_model}  [dim]({embedding_provider})[/dim]\n"
        f"[bold cyan]Eval LLM[/bold cyan]         {eval_model}  [dim]({eval_provider})[/dim]\n"
        "\n"
        f"[bold cyan]Top-K[/bold cyan]            {top_k}\n"
        f"[bold cyan]Sim. threshold[/bold cyan]   {threshold_str}\n"
    )
    console.print(
        Panel(content, title="[bold]RAG Evaluation[/bold]", border_style="cyan")
    )


def print_summary(report: dict[str, Any]) -> None:
    """Render rich tables summarising evaluation results."""
    experiment_name = report["experiment_name"]

    # --- Overall metrics table ---
    overall = report.get("overall", {})
    if overall:
        table = Table(
            title=f"Overall Metrics — {experiment_name}",
            show_header=True,
            header_style="bold cyan",
            show_lines=False,
        )
        table.add_column("Metric", style="bold")
        table.add_column("Score", justify="right")
        for metric, value in overall.items():
            table.add_row(
                metric.replace("_", " ").title(),
                _fmt_score(float(value)),
            )
        console.print()
        console.print(table)

    # --- Per-category table ---
    per_category = report.get("per_category", {})
    if per_category:
        present_metrics = [
            col
            for col in METRIC_COLUMN_ORDER
            if any(col in values for values in per_category.values())
        ]
        cat_table = Table(
            title="Per-Category Metrics",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
        )
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Count", justify="right")
        for col in present_metrics:
            cat_table.add_column(col.replace("_", " ").title(), justify="right")
        for category, values in per_category.items():
            row: list[str | Text] = [category, str(values.get("count", 0))]
            for col in present_metrics:
                v = values.get(col)
                row.append(_fmt_score(float(v) if v is not None else None))
            cat_table.add_row(*row)
        console.print()
        console.print(cat_table)

    # --- Failures table ---
    failures = report.get("failures", [])
    console.print()
    if failures:
        fail_table = Table(
            title=f"[red]Failures — factual_correctness < 0.5[/red]  ({len(failures)} samples)",
            show_header=True,
            header_style="bold red",
            show_lines=True,
        )
        fail_table.add_column("Question", max_width=55, no_wrap=False)
        fail_table.add_column("Category")
        fail_table.add_column("Factual Corr.", justify="right")
        fail_table.add_column("Context Recall", justify="right")
        for failure in failures:
            fail_table.add_row(
                failure.get("user_input", ""),
                failure.get("question_category", ""),
                _fmt_score(failure.get("factual_correctness")),
                _fmt_score(failure.get("context_recall")),
            )
        console.print(fail_table)
    else:
        console.print(
            Panel(
                "[bold green]All samples passed (factual_correctness \u2265 0.5)[/bold green]",
                title="Failures",
                border_style="green",
            )
        )
    console.print()


def print_artifacts_panel(
    config_path: Path,
    results_path: Path,
    scores_path: Path,
    report_path: Path,
) -> None:
    """Render a closing panel listing all persisted artifact paths."""
    content = (
        f"[bold]Config  [/bold] {config_path}\n"
        f"[bold]Results [/bold] {results_path}\n"
        f"[bold]Scores  [/bold] {scores_path}\n"
        f"[bold]Report  [/bold] {report_path}\n"
    )
    console.print(
        Panel(
            content,
            title="[bold green]Artifacts Saved[/bold green]",
            border_style="green",
        )
    )
