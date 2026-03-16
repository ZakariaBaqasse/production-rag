"""Generate a publication-ready line chart showing metric progression across the four chunking-improvement experiments.

    uv run python scripts/plot_experiment_progression.py

Output: scripts/experiment_progression.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPTS_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPTS_DIR.parent / "artifacts" / "experiments"

# Ordered experiment sequence (chronological)
EXPERIMENTS = [
    ("table-aware-chunking-with-subtables", "Table-aware\nchunking"),
    ("increase-top-k-to-20", "top_k → 20"),
    ("key-column-metadata-injection", "Key-column\nmetadata"),
    ("increase-row-overlap-to-4", "Row overlap\n1 → 4"),
]

METRICS = {
    "factual_correctness": ("Factual Correctness", "#2563EB"),  # blue
    "context_recall": ("Context Recall", "#16A34A"),  # green
    "context_precision": ("Context Precision", "#9333EA"),  # purple
}


def load_overall(experiment_dir: str) -> dict:
    """Load the 'overall' metrics from the experiment's report.json."""
    path = EXPERIMENTS_DIR / experiment_dir / "report.json"
    return json.loads(path.read_text())["overall"]


def main() -> None:
    """Generate the experiment progression plot."""
    labels = [label for _, label in EXPERIMENTS]
    data = {metric: [] for metric in METRICS}

    for exp_dir, _ in EXPERIMENTS:
        overall = load_overall(exp_dir)
        for metric in METRICS:
            data[metric].append(overall[metric])

    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    for metric, (display_name, color) in METRICS.items():
        values = data[metric]
        ax.plot(
            x,
            values,
            marker="o",
            markersize=7,
            linewidth=2.2,
            color=color,
            label=display_name,
        )
        # Annotate each point
        for i, v in enumerate(values):
            ax.annotate(
                f"{v:.3f}",
                xy=(i, v),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0.40, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "RAG Pipeline — Metric Progression Across Experiments",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    ax.legend(
        loc="lower left",
        fontsize=10,
        framealpha=0.7,
        edgecolor="#CBD5E1",
    )

    out_path = SCRIPTS_DIR / "experiment_progression.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
