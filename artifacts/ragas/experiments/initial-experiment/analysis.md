# Experiment Analysis: initial-experiment

**Date:** 2026-02-19  
**Experiment Name:** `initial-experiment`  
**Timestamp:** 2026-02-19T13:59:48+00:00

---

## Experiment Configuration

| Parameter            | Value                                  |
| -------------------- | -------------------------------------- |
| Generation Model     | `minimax-m2.5`                         |
| Judge / Eval Model   | `gpt-4o-mini`                          |
| Embedding Model      | `qwen3-embedding:0.6b` (Ollama, local) |
| Vector Store         | pgvector (PostgreSQL)                  |
| Retrieval `top_k`    | 5                                      |
| Reranker             | None                                   |
| Similarity Threshold | None                                   |
| RAGAS Version        | 0.4.3+                                 |
| Testset Size         | 48 questions                           |

---

## Testset Breakdown

| Category                  | Count |
| ------------------------- | ----- |
| `table_extraction`        | 22    |
| `pin_mapping`             | 12    |
| `adversarial`             | 9     |
| `cross_section_synthesis` | 5     |

---

## Overall Results

| Metric                  | Overall   | adversarial | cross_section_synthesis | pin_mapping | table_extraction |
| ----------------------- | --------- | ----------- | ----------------------- | ----------- | ---------------- |
| Context Precision       | 0.823     | 0.892       | 0.801                   | 0.907       | 0.755            |
| Context Recall          | 0.918     | 1.000       | 0.933                   | 0.896       | 0.894            |
| Faithfulness            | 0.858     | 0.778       | 0.810                   | 0.861       | 0.900            |
| **Factual Correctness** | **0.668** | 0.683       | 0.652                   | 0.638       | 0.681            |
| **Answer Relevancy**    | **0.758** | 0.789       | 0.795                   | 0.718       | 0.758            |

**Retrieval metrics (Context Precision, Context Recall) are strong across all categories.**  
**Generation metrics (Factual Correctness, Answer Relevancy) are the primary weakness.**

---

## Metric-Level Diagnosis

### Factual Correctness (0.668) — Primary Weakness

The retrieval pipeline is largely delivering the right context (CR=0.918 overall), yet the model is failing to produce factually accurate answers. Three distinct root causes were identified:

1. **Verbose, markdown-heavy responses vs. terse reference answers**  
   The model structures its output with bold headers, bullet lists, and sections. RAGAS's factual correctness judge (an NLI-based scorer) computes claim-level precision and recall between the response and the reference. When a reference answer is a single sentence and the model produces five paragraphs — even if all claims are correct — the formatting divergence and paraphrase distance lowers the scored overlap. This is a consistent drag across all categories.

2. **Dense RF/Bluetooth compound-row table fragility**  
   The datasheet's RF and Bluetooth tables use compound rows (Min/Typ/Max columns across 6+ parameter rows, sometimes with separate BDR and BLE subtables on the same page). The model conflates sub-rows (e.g., answering BLE sensitivity values when asked for BDR) or omits boundary values (Min/Max) that the reference explicitly includes. These failures are concentrated in `table_extraction`.

3. **Cross-section synthesis gap**  
   For queries requiring information from two or more separate sections, the model correctly retrieves both chunks (CR=1.0) but fails to integrate them into a single coherent answer. The MTDI question is the clearest example: the model answers the VDD_SDIO strapping behavior correctly but omits the ADC2_CH4 / TOUCH4 pin multiplexing from a separate pin-function table (FC=0.18 despite CR=1.0).

---

### Answer Relevancy (0.758) — Secondary Weakness

Answer Relevancy is worst in `pin_mapping` (0.718). This metric is computed by having the model generate synthetic questions from its own response and measuring the cosine similarity (via the embedding model) between those synthetic questions and the original. It penalizes over-verbose and off-topic answers.

Pin mapping questions are precise lookups ("what is the GPIO number for MTDI?"). The model consistently responds with 3–5 paragraph answers including comparison tables, strapping pin summaries, and feature descriptions. This verbose output dilutes the embedding centroid, reducing the relevancy score even when the factual content is correct.

---

### Faithfulness (0.858) — Acceptable

Faithfulness measures whether every claim in the response can be attributed to the retrieved context. The score of 0.858 is acceptable for a baseline but `adversarial` (0.778) is notably lower. Adversarial questions test the model's ability to refuse or qualify answers when the datasheet doesn't directly support them. The lower score indicates the model occasionally makes claims not directly grounded in the retrieved chunks — likely over-reasoning or hallucinating specifications for edge-case adversarial prompts.

---

### Context Precision (0.823) — Good, but weakest in table_extraction (0.755)

Lower precision in `table_extraction` means the top-5 retrieved chunks contain more noise (non-relevant chunks) for table-heavy queries. This is consistent with table rows being split across chunks — a query about Row N of a table retrieves a chunk containing rows N±3 that aren't directly relevant to the answer, pushing precision down.

---

### Context Recall (0.918) — Strong Overall

Recall is the strongest metric. The one critical weakness is `pin_mapping` (0.896) dragged down by two complete retrieval misses (CR=0.0). Outside those failures the retrieval pipeline is reliably surfacing relevant content.

---

## Failure Case Analysis

11 samples scored below threshold. Full breakdown:

| Question (abbreviated)                         | Category                  | FC   | CR    | Root Cause                                                     |
| ---------------------------------------------- | ------------------------- | ---- | ----- | -------------------------------------------------------------- |
| Absolute max voltage + storage temp            | `table_extraction`        | 0.00 | 0.333 | Chunking boundary splits Table 5-1; partial table retrieved    |
| BDR sensitivity + adj. channel F0+1MHz         | `table_extraction`        | 0.40 | 1.000 | Model conflates BDR/BLE compound-row table structure           |
| BLE compliance / sensitivity / controller type | `table_extraction`        | 0.36 | 1.000 | Dense feature list extraction failure; model misses sub-specs  |
| GPIO_Matrix signal 63                          | `pin_mapping`             | 0.00 | 0.000 | Complete retrieval miss; 70-row table fragmentation            |
| ADC accuracy atten=3 degradation condition     | `adversarial`             | 0.40 | 1.000 | Model gives ±60mV range but omits "above raw 3000" footnote    |
| Wi-Fi sensitivity 802.11b + HT20 MCS7          | `table_extraction`        | 0.40 | 0.667 | Wi-Fi table rows split across chunks; partial context          |
| MTDI + ADC + touch sensor pin functions        | `cross_section_synthesis` | 0.18 | 1.000 | Model answers strapping section only; misses pin MUX table     |
| Touch sensor T5 and T8 GPIO mapping            | `pin_mapping`             | 0.44 | 1.000 | T8=GPIO33 (32K_XN) confusion; partial correctness only         |
| VDD_SDIO on same net as VDD3P3_RTC             | `adversarial`             | 0.29 | 1.000 | Implicit 6Ω bypass behavior requires inference beyond verbatim |
| ESD ratings (HBM and CDM)                      | `table_extraction`        | 0.25 | 0.000 | Complete retrieval miss; Table 5-5 not in top-5 results        |
| BLE TX power control + adj. channel F0±2MHz    | `table_extraction`        | 0.33 | 1.000 | BLE TX table multi-row structure not correctly parsed by model |

---

## Prioritized Recommendations

### 1. Fix Table Chunking (High Impact — targets CR=0 failures)

The two complete retrieval misses (GPIO_Matrix signal 63, ESD Table 5-5) and the partial miss (Absolute Max Table 5-1) are caused by table rows being fragmented across chunk boundaries. The fix is to implement table-aware splitting in the ingestion pipeline:

- Detect table boundaries during parsing and keep complete tables in a single chunk.
- If a table exceeds the maximum chunk size, duplicate the header row into each continuation chunk so the model can interpret rows without the parent table for context.

**Expected impact:** Fixes 3 CR=0/low-recall failures; improves `table_extraction` Context Precision.

---

### 2. Constrain Generation Verbosity (High Impact — fast to implement)

The system prompt currently generates verbose, markdown-rich responses regardless of question type. Adding explicit instructions to match response conciseness to question complexity would directly address the Answer Relevancy gap and partially address Factual Correctness.

Example system prompt addition:

```
Answer concisely. For lookup questions (GPIO numbers, voltage levels, timing values),
respond in 1-2 sentences or a minimal table. Do not add context, comparisons, or
background information unless the question explicitly asks for explanation.
```

**Expected impact:** Largest likely gain on Answer Relevancy (especially `pin_mapping`); secondary gain on Factual Correctness by reducing paraphrase distance from reference answers.

---

### 3. Add Hybrid Retrieval (BM25 + Dense) (Medium Impact — targets exact-match failures)

`qwen3-embedding:0.6b` is a small local model. It may not embed domain-specific acronyms (HBM, CDM, signal-63 GPIO_Matrix row numbers) close enough to query embeddings for exact-match retrieval. Adding sparse BM25 retrieval alongside pgvector similarity search would help surface rows and cells that contain exact token matches.

**Expected impact:** Likely fixes ESD (HBM/CDM) and GPIO_Matrix signal-63 retrieval misses; improves overall CR for dense technical tables.

---

### 4. Increase top_k for Cross-Section Queries (Medium Impact — targeted)

Cross-section synthesis failures have CR=1.0 (both required chunks are retrieved) but the model fails to synthesize across them. However, increasing top_k to 8–10 for synthesis-type queries (or adding a reranker) would provide more context overlap and marginally help. More importantly, testing top_k=8 globally would show whether additional context helps or hurts faithfulness.

**Expected impact:** Incremental improvement on `cross_section_synthesis`; negligible impact on `table_extraction`.

---

### 5. Evaluate a Stronger Generation Model (Model Isolation Test)

Several failures have CR=1.0 (context is correct) but FC≤0.40 — the required information is present but the model fails to extract and state it correctly. This points to model capability as a bottleneck, not the pipeline. Running the same experiment configuration with a stronger generation model (e.g., GPT-4o-mini or a larger open-weight model) would isolate pipeline issues from model issues.

**Expected impact:** Potentially large gain on RF/BT table extraction failures and cross-section synthesis; useful as a diagnostic to set an upper bound on what a better pipeline can achieve.

---

## Summary

| Area                    | Status                              | Priority Fix                   |
| ----------------------- | ----------------------------------- | ------------------------------ |
| Retrieval quality       | Good (CR=0.918) with 2 hard misses  | Hybrid retrieval (BM25)        |
| Table chunking          | Fragmented large tables             | Table-aware chunking in ingest |
| Generation verbosity    | Over-verbose for all question types | System prompt constraints      |
| Cross-section synthesis | Retrieval works; integration fails  | Verbosity fix + top_k increase |
| Model capability        | Likely bottleneck for RF/BT tables  | Model swap experiment          |

The fastest experiment to run next is **Recommendation 2** (system prompt verbosity fix) — zero infrastructure change, high expected impact on both primary weak metrics. The highest potential impact change is **Recommendation 1** (table chunking) but requires code changes to the ingestion pipeline.
