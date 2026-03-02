"""Backfill an existing experiment directory into MLflow.

Run once to seed the tracking server with results produced before
MLflow integration was added:

    uv run python scripts/backfill_mlflow.py
"""

import json
from pathlib import Path

from production_rag.evals.mlflow_tracking import log_run_to_mlflow
from production_rag.evals.schemas import ExperimentConfig

EXPERIMENTS_DIR = Path("artifacts/experiments")
TRACKING_URI = "sqlite:///mlruns.db"


def backfill_experiment(run_dir: Path) -> None:
    """Log a single existing experiment directory to MLflow."""
    report = json.loads((run_dir / "report.json").read_text())
    config = ExperimentConfig(**json.loads((run_dir / "config.json").read_text()))
    run_id = log_run_to_mlflow(
        report=report,
        config=config,
        run_dir=run_dir,
        tracking_uri=TRACKING_URI,
    )
    print(f"Backfilled '{run_dir.name}'. Run ID: {run_id}")


if __name__ == "__main__":
    run_dirs = [
        d
        for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and (d / "report.json").exists()
    ]
    if not run_dirs:
        print(f"No experiment directories found under {EXPERIMENTS_DIR}")
    for run_dir in sorted(run_dirs):
        backfill_experiment(run_dir)
    print(f"\nDone. Launch UI with: mlflow ui --backend-store-uri {TRACKING_URI}")
