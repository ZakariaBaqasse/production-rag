# ESP32 Datasheet RAG Pipeline

An advanced Retrieval-Augmented Generation (RAG) pipeline purpose-built for answering complex technical questions from the ESP32 datasheet — including questions about register maps, peripheral tables, memory layouts, and other structured content that standard PDF parsers struggle with.

The pipeline leverages **LlamaParse** (a VLM-powered OCR service) to faithfully extract tables and structured data from the datasheet, stores embeddings in **PostgreSQL + pgvector**, and uses [RAGAS](https://docs.ragas.io/) as a systematic evaluation framework to measure the impact of different retrieval and generation strategies.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Evaluation Framework](#evaluation-framework)
- [Experiment Artifacts](#experiment-artifacts)
- [Development](#development)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION                            │
│                                                             │
│  ESP32 PDF  →  LlamaParse (VLM/OCR)  →  Markdown Pages     │
│                                              │              │
│                                    Embed (Ollama/OpenAI)    │
│                                              │              │
│                                   PostgreSQL + pgvector     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        RETRIEVAL & GENERATION               │
│                                                             │
│  User Query  →  Embed Query  →  Cosine Similarity Search    │
│                                         │                   │
│                               Top-K Retrieved Pages         │
│                                         │                   │
│                             LLM (Ollama / OpenAI / Gemini)  │
│                                         │                   │
│                                    Final Answer             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        EVALUATION (RAGAS)                   │
│                                                             │
│  Curated Testset  →  Run Pipeline per Sample               │
│                              │                              │
│                   Score: Faithfulness, Context Precision,   │
│                   Context Recall, Factual Correctness,      │
│                   Answer Relevancy                          │
│                              │                              │
│              Experiment Artifacts (CSV + JSON report)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component            | Technology                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| PDF parsing / OCR    | [LlamaParse](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) (VLM, `agentic` tier) |
| Vector store         | [PostgreSQL 17](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector) |
| Embeddings           | Ollama (local), OpenAI, or Google GenAI — swappable via config                                  |
| LLM (generation)     | Ollama (local), OpenAI, or Google GenAI — swappable via config                                  |
| Evaluation           | [RAGAS v0.4.3](https://docs.ragas.io/)                                                          |
| Orchestration        | [LangChain](https://python.langchain.com/)                                                      |
| Package manager      | [uv](https://docs.astral.sh/uv/)                                                                |
| Linting / formatting | [Ruff](https://docs.astral.sh/ruff/)                                                            |
| Type checking        | [mypy](https://mypy.readthedocs.io/) (strict mode)                                              |

---

## Project Structure

```
production-rag/
├── configs/
│   └── cloud-ollama.yaml        # Example experiment configuration
├── artifacts/
│   ├── experiments/             # Per-run evaluation outputs (CSV + JSON)
│   └── testsets/
│       └── curated_testset.csv  # Hand-curated evaluation testset
├── data/
│   └── raw/                     # Place the ESP32 datasheet PDF here
├── scripts/
│   ├── ingest.py                # Standalone ingestion script
│   └── write_curated_testset.py # Helper to build the curated testset
├── src/production_rag/
│   ├── core/
│   │   ├── config.py            # Pydantic config models + YAML loader
│   │   └── database.py          # asyncpg connection helpers + schema init
│   ├── pipeline/
│   │   ├── parse.py             # LlamaParse VLM/OCR integration (with caching)
│   │   ├── embeddings.py        # Document embedding + DB storage
│   │   ├── retrieve.py          # Vector search + LLM answer generation
│   │   └── utils.py             # Shared embedding model factory
│   └── evals/
│       ├── cli.py               # Evaluation CLI argument parser
│       ├── evaluate.py          # End-to-end RAGAS evaluation runner
│       ├── generate_ragas_testset.py  # Automated testset generation via RAGAS
│       ├── metrics.py           # RAGAS metric construction + scoring
│       ├── reporting.py         # Aggregation, per-category breakdown, artifacts
│       ├── schemas.py           # ExperimentConfig / ExperimentResult models
│       └── testset.py           # Testset loading helpers
├── docker-compose.yml           # pgvector database service
└── pyproject.toml               # Project metadata, dependencies, tasks
```

---

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Ollama](https://ollama.com/) if using local models

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd production-rag
uv sync
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```bash
# Required for PDF parsing
LLAMA_CLOUD_API_KEY=llx-...

# Required depending on which LLM/embedding provider you use
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# PostgreSQL (override defaults if needed)
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=production_rag
POSTGRES_PORT=5432
```

### 3. Start the vector database

```bash
docker compose up -d
```

This starts a PostgreSQL 17 instance with the `pgvector` extension. Data is persisted in a named Docker volume.

### 4. Add the ESP32 datasheet

Place the PDF at:

```
data/raw/esp32_datasheet.pdf
```

---

## Configuration

Experiments are driven entirely by YAML config files. An example is provided at [configs/cloud-ollama.yaml](configs/cloud-ollama.yaml):

```yaml
pipeline:
  chat:
    model: "minimax-m2.5"
    provider: "ollama"
    base_url: "http://ollama.com"
  embeddings:
    model: "qwen3-embedding:0.6b"
    provider: "ollama"
    base_url: "http://localhost:11434"

eval:
  llm:
    model: "gpt-4o-mini"
    provider: "openai"
  embeddings:
    model: "text-embedding-3-small"
    provider: "openai"

retrieval:
  top_k: 5
  similarity_threshold: null # set a float (e.g. 0.7) to hard-filter by similarity
  reranker: "none"
```

Supported `provider` values: `ollama`, `openai`, `googlegenai`.

---

## Running the Pipeline

All commands are run from the project root with `uv run`.

### Step 1 — Ingest the datasheet

Parses the PDF via LlamaParse (VLM, `agentic` tier), embeds each page, and stores vectors in PostgreSQL. Parse results are cached locally under `.cache/parsed/` keyed by file hash to avoid redundant API calls.

```bash
rag-ingest
```

### Step 2 — Generate a testset (optional)

Automatically synthesize evaluation questions from the ingested pages using RAGAS and a configured LLM:

```bash
rag-generate-testset
```

A curated testset is already provided at `artifacts/testsets/curated_testset.csv`. Skip this step if you want to use it directly.

### Step 3 — Run an evaluation experiment

```bash
 rag-evaluate \
  --config-path configs/cloud-ollama.yaml \
  --testset artifacts/testsets/curated_testset.csv \
  --experiment-name "baseline-top5"
```

| Flag                | Default                                    | Description                        |
| ------------------- | ------------------------------------------ | ---------------------------------- |
| `--config-path`     | `./config/experiment_config.yaml`          | Path to the YAML config            |
| `--testset`         | `./artifacts/testsets/curated_testset.csv` | Evaluation testset CSV             |
| `--output-dir`      | `./artifacts/experiments`                  | Where to write result artifacts    |
| `--experiment-name` | timestamp                                  | Name for the experiment run folder |

---

## Evaluation Framework

Each evaluation run scores every testset sample across five RAGAS metrics:

| Metric                  | What it measures                                                              |
| ----------------------- | ----------------------------------------------------------------------------- |
| **Context Precision**   | Are the retrieved chunks actually relevant to the question?                   |
| **Context Recall**      | Does the retrieved context cover what the reference answer requires?          |
| **Faithfulness**        | Is the generated answer grounded in the retrieved context (no hallucination)? |
| **Factual Correctness** | Does the answer match the reference answer factually?                         |
| **Answer Relevancy**    | Is the answer on-topic and directly addressing the question?                  |

The evaluator LLM and embedding model used for scoring are configured separately under `eval:` in the config file, so you can score with a stronger model (e.g. GPT-4o) regardless of which model you use for generation.

Results are broken down **overall** and **per question category** (e.g. register maps, timing diagrams, pin configurations), making it straightforward to identify where a given retrieval strategy performs well or poorly.

---

## Experiment Artifacts

Each experiment run saves the following under `artifacts/experiments/<experiment-name>/`:

```
artifacts/experiments/baseline-top5/
├── config.json      # Full experiment configuration snapshot
├── results.csv      # Per-sample: question, retrieved contexts, answer, all metric scores
├── scores.csv       # Aggregated metric scores (overall + per category)
└── report.json      # JSON report: overall means, per-category means, low-score failures
```

This structure makes it easy to compare runs — change a single parameter (e.g. `top_k`, `similarity_threshold`, or the embedding model), re-run, and diff the reports.

---

## Development

```bash
# Lint
uv run task lint

# Auto-fix lint issues
uv run task lint-fix

# Format code
uv run task format

# Type check (strict mypy)
uv run task typecheck

# Run all checks (lint + typecheck + format check)
uv run task check
```
