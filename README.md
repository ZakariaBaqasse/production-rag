# Baseline RAGAS Testset Generation

This project includes a baseline testset generator so you can evaluate retrieval and answer quality before iterating on your RAG pipeline.

## Prerequisites

- `LLAMA_CLOUD_API_KEY` for PDF parsing
- `GOOGLE_API_KEY` for Gemini testset synthesis
- Ollama running locally at `http://localhost:11434` with your embedding model pulled

## Generate the baseline testset

From the project root:

```bash
uv run python -m src.generate_ragas_testset \
	--pdf-path ./src/assets/esp32_datasheet.pdf \
	--testset-size 50 \
	--output-dir ./artifacts/ragas
```

Outputs:

- `artifacts/ragas/baseline_testset.jsonl`
- `artifacts/ragas/baseline_testset.csv`

Optional flags:

- `--max-docs 20` to limit parsed pages used for generation
- `--generator-model <model_name>` to override default LLM
- `--embedding-model <model_name>` to override default embedding model
- `--use-default-transforms` to force RAGAS default transform/query pipeline

Notes:

- The script uses a stable single-hop baseline pipeline by default (`NERExtractor` + `SingleHopSpecificQuerySynthesizer`) to avoid `ragas==0.4.3` transform failures (`'headlines' property not found in this node` and cosine embedding shape errors).
- The script sets a fixed default persona list, avoiding RAGAS auto-persona generation that depends on additional graph properties.
- If `--use-default-transforms` is enabled and fails, the script automatically retries with the stable single-hop baseline pipeline.
