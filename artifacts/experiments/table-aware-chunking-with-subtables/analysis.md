# Experiment Analysis: `table-aware-chunking-with-subtables`

> **Date:** 2026-03-11  
> **Status:** Regression — all retrieval metrics degraded significantly  
> **Previous best baseline:** `markdown-tables-insteadof-html` (2026-03-09)

---

## 1. Executive Summary

Phase 1 of the implementation plan — row-boundary table splitting via the `PipeTable` AST — produced a **substantial regression across all retrieval metrics**, with the overall scores moving in the wrong direction on every axis that matters for end-user accuracy:

| Metric              | `implement-table-aware` | `markdown-tables-insteadof-html` | `table-aware-chunking-with-subtables` | Δ vs best baseline |
| ------------------- | ----------------------- | -------------------------------- | ------------------------------------- | ------------------ |
| context_precision   | 0.809                   | 0.797                            | **0.590**                             | −0.219 (−27.1 pp)  |
| context_recall      | 0.941                   | 0.920                            | **0.684**                             | −0.257 (−27.3 pp)  |
| faithfulness        | 0.830                   | 0.850                            | **0.877**                             | +0.027 (+3.2 pp)   |
| factual_correctness | 0.658                   | 0.660                            | **0.554**                             | −0.106 (−16.1 pp)  |
| answer_relevancy    | 0.627                   | 0.638                            | **0.524**                             | −0.114 (−17.9 pp)  |

The only metric that improved is **faithfulness**, which measures whether the generated answer is grounded in the retrieved context. This improvement is a direct and expected consequence of having smaller, more focused chunks — but faithfulness is the least important metric when the retrieval layer is already failing to bring back the right content.

The core problem: **the solution traded the oversized-chunk problem for a harder needle-in-a-haystack problem**. By splitting large tables into many small row-bounded chunks, the pipeline produced a corpus where the answer to a specific query lives in one of potentially 20–50 nearly-identical small chunks — and `top_k=5` is no longer enough to reliably surface it.

---

## 2. Confound: Eval Model Upgrade

Before interpreting the deltas, a critical experimental confound must be acknowledged: **`eval_model` changed from `gpt-4o-mini` to `gpt-4o`** between the baseline runs and this experiment.

`gpt-4o` is consistently stricter in RAGAS factual correctness scoring than `gpt-4o-mini`. It applies more nuanced claim-level verification and tends to down-score partially correct answers that `gpt-4o-mini` would accept. This means the reported `factual_correctness` regression (−0.106) is partially a measurement artifact, not a pure pipeline regression.

However, this confound **does not affect context_recall at all** — context_recall is computed purely from retrieved chunk coverage against reference contexts, with no LLM judgment. The context_recall collapse from 0.941 → 0.684 (−27.3 pp) is a genuine, unambiguous retrieval regression.

**Key implication:** The factual correctness decline is likely _worse than the numbers show_ due to the stricter scorer, meaning some of the remaining factual correctness score is an eval-model artefact. Re-running the baselines with `gpt-4o` as the eval model would be needed for a clean comparison of that metric.

---

## 3. Per-Category Breakdown

### 3.1 Context Recall

Context recall is the clearest signal of retrieval health. Every single category regressed:

| Category                  | `implement-table-aware` | `markdown-tables` | `subtables` | Δ          |
| ------------------------- | ----------------------- | ----------------- | ----------- | ---------- |
| adversarial               | 1.000                   | 1.000             | 0.611       | **−0.389** |
| adversarial_multi_hop     | 1.000                   | 0.889             | 0.750       | **−0.250** |
| cross_section_synthesis   | 1.000                   | 1.000             | 0.533       | **−0.467** |
| pin_mapping               | 1.000                   | 1.000             | 0.694       | **−0.306** |
| table_multi_row_reasoning | 0.850                   | 0.850             | 0.717       | **−0.133** |
| table_single_cell_lookup  | 0.889                   | 0.861             | 0.694       | **−0.195** |

`cross_section_synthesis` suffered the worst collapse (−47 pp), followed by `adversarial` (−39 pp) and `pin_mapping` (−31 pp). These are precisely the categories that contain table-heavy content — the categories the implementation was designed to improve.

### 3.2 Context Precision

Precision also regressed severely, meaning not only is the right content harder to find, but the 5 retrieved chunks now contain more irrelevant material:

| Category                  | `implement-table-aware` | `markdown-tables` | `subtables` | Δ          |
| ------------------------- | ----------------------- | ----------------- | ----------- | ---------- |
| adversarial               | 0.893                   | 0.789             | 0.629       | **−0.264** |
| adversarial_multi_hop     | 0.875                   | 0.825             | 0.669       | **−0.206** |
| cross_section_synthesis   | 0.764                   | 0.780             | 0.376       | **−0.404** |
| pin_mapping               | 0.912                   | 0.946             | 0.607       | **−0.339** |
| table_multi_row_reasoning | 0.666                   | 0.707             | 0.520       | **−0.187** |
| table_single_cell_lookup  | 0.788                   | 0.718             | 0.670       | **−0.118** |

The precision drop in `cross_section_synthesis` (−40 pp) is especially striking. Questions that require reasoning across multiple datasheet sections now find neither the relevant rows nor the relevant sections.

### 3.3 Factual Correctness (with eval-model caveat)

| Category                  | `implement-table-aware` | `markdown-tables` | `subtables` | Δ (note: gpt-4o vs gpt-4o-mini) |
| ------------------------- | ----------------------- | ----------------- | ----------- | ------------------------------- |
| adversarial               | 0.793                   | 0.813             | 0.687       | −0.126                          |
| adversarial_multi_hop     | 0.612                   | 0.522             | 0.578       | −0.034 (mixed)                  |
| cross_section_synthesis   | 0.754                   | 0.732             | **0.372**   | **−0.382**                      |
| pin_mapping               | 0.612                   | 0.755             | 0.597       | −0.158 vs best                  |
| table_multi_row_reasoning | 0.643                   | 0.564             | 0.554       | −0.089                          |
| table_single_cell_lookup  | 0.665                   | 0.646             | 0.542       | −0.123                          |

`cross_section_synthesis` fell from 0.754 to 0.372 — a near 50% collapse in answer quality for cross-table reasoning questions.

---

## 4. Failure Pattern Analysis

### 4.1 The "Retrieval Went to Zero" Pattern

The most alarming failure pattern in the new experiment is a set of questions that had **context_recall = 1.0 in the baseline but now have context_recall = 0.0**. These represent complete retrieval failures for content that was previously being found reliably:

| Question                                                                                | Baseline cr | New cr | Category                  |
| --------------------------------------------------------------------------------------- | ----------- | ------ | ------------------------- |
| "what pin number GPIO18 is?"                                                            | 1.0         | 0.0    | pin_mapping               |
| "In the ESP32 GPIO_Matrix, what is signal number 63..."                                 | 1.0         | 0.0    | pin_mapping               |
| "How does the ESP32 BLE receiver sensitivity compare to BT Classic..."                  | —           | 0.0    | cross_section_synthesis   |
| "What is the Wi-Fi receiver sensitivity for 802.11b at 1 Mbps and 802.11n HT20 MCS7..." | —           | 0.0    | table_multi_row_reasoning |
| "What is the Bluetooth Basic Data Rate receiver sensitivity..."                         | —           | 0.0    | table_single_cell_lookup  |
| "Hey... center pad for the ESP32 chip in the QFN 5×5 package?"                          | —           | 0.0    | pin_mapping               |

The "GPIO18 pin number" and "signal number 63" questions are canonical examples of the new failure mode. In the baseline, the pin table was either one large chunk (due to cross-page merging) or a few large chunks — and the embedding of the full table naturally captured all pin names, so any query for a specific GPIO would match. After row-boundary splitting, GPIO18's row lives in one of ~50 nearly-identical header+2-row chunks. The embedding of `| GPIO18 | 18 | ... |` alone is too sparse and generic to rank in the top-5 for a query about "GPIO18 pin number."

### 4.2 The Confidence-Recall Inversion

In the _baseline_ experiments, the dominant failure pattern was: **recall was high (context found), but factual correctness was low (model failed to extract the answer)**. The failures list showed many entries with cr=1.0 but fc=0.0–0.4.

In the _new_ experiment, the failure pattern flipped: **recall collapsed (context not found), and factual correctness collapsed as a downstream consequence**. This is a fundamentally worse failure mode because:

- A generative/prompting fix can address the cr=1.0/fc=0 pattern (better prompts, better model, chain-of-thought)
- Only a retrieval fix can address the cr=0.0 pattern

### 4.3 Cross-Section Synthesis Catastrophe

Questions like "Which ESP32 GPIO pins support both ADC and capacitive touch sensing simultaneously?" require retrieving content from the peripheral pin configuration table that shows both ADC channels and touch sensor assignments in a single view. In the baseline, the full pin table was one large (if oversized) chunk containing both columns — one retrieval hit answered the question. After splitting, the answer requires multiple specific row-chunks from the touch sensor sub-table AND the ADC sub-table. With `top_k=5`, the retrieval is now attempting to find 5 pins-compatible chunks from what is now a corpus of 50+ row-chunks, and the semantic similarity from the query to any individual row is too weak to guarantee coverage.

---

## 5. Root Cause Analysis

### 5.1 The Needle-in-a-Haystack Problem

Row-boundary splitting converted each large table (e.g., the 23,364-char Peripheral Pin Configurations table from page 49) into approximately **12–16 sub-chunks** of `~CHUNK_SIZE`. This means:

- The query "what pin number GPIO18 is?" must match one specific sub-chunk out of ~16 nearly identical pin-table chunks
- All 16 chunks have the same header row ("| Chip Pin | Type | Function | ... |")
- All 16 chunks differ only in which GPIO rows they contain
- The embedding of a header+2–3 pin rows is a weak, generic signal

The key insight is that **embedding similarity does not scale down to row granularity for structured tables**. A table row like `| 25 | GPIO18 | I/O/T | ... |` has almost no unique semantic content relative to adjacent rows. The embedding model cannot meaningfully distinguish "the chunk containing GPIO18" from "the chunk containing GPIO19" because the row-level semantic content is nearly identical. The query needs to match one specific needle in a haystack of structurally identical chunks.

This is the opposite of the problem that motivated the fix. Oversized chunks had diffuse embeddings because they averaged over too many topics. Row-split chunks have sparse embeddings because each chunk has too little content per topic.

### 5.2 `top_k=5` is Now Structurally Insufficient

With the baseline design:

- 1 large pin table chunk → 1 retrieval hit → entire table available for answer

With the new row-split design:

- 1 large pin table → 16 small chunks → need to hit 1–3 specific chunks → requires lucky draw from top-5

For cross-section synthesis questions that need rows from multiple tables, the math is even worse: 2–3 specific chunks needed from 30+ candidate chunks, with `top_k=5`.

The implementation plan anticipated the need for **hybrid search + key-column metadata** (Phase 4) to compensate for this. Phase 4 was not implemented before running this evaluation. The row-boundary splitting works well only when paired with better-than-cosine-similarity retrieval.

### 5.3 Schema Identity Check May Be Over-Blocking Merges

The new `same_schema` check in `_merge_split_tables` was designed to prevent the 8-page IO_MUX + GPIO Matrix chain from merging. However, it may be inadvertently blocking legitimate merges where LlamaParse introduces minor formatting variations across pages (e.g., an extra empty column in the separator row `|---|---||` vs `|---|---|`, or minor column name capitalisation differences). If pages 47–51 (the genuine Peripheral Pin Config continuation) are no longer merging due to a false schema mismatch, those pages become 5 independent moderate-sized documents instead of one large one — and the first page doesn't have enough context to answer questions about pins on pages 49–51.

This is recoverable (the logs should show "Not merging pages X → Y: table schema changed" for pages 47–51 if this is happening), but it would compound the row-splitting recall problem.

### 5.4 Context Precision Paradox

Intuitively, smaller, more focused chunks should _improve_ precision. The fact that precision also degraded suggests the vector similarity ranking became noisier: with many near-identical row chunks competing for the top-5 slots, the retriever surfaces the wrong 5 chunks more often. Previously, one large relevant chunk would dominate the similarity ranking. Now, the recall loss means the one right chunk isn't retrieving at all, and the 5 returned chunks are from the wrong rows of the same table — giving the appearance of an irrelevant result set despite containing semantically adjacent content.

---

## 6. Distinguishing Real Regressions from Eval-Model Noise

To isolate the implementation impact from the eval model change:

| Metric              | Likely pure implementation regression | Likely contaminated by eval model |
| ------------------- | ------------------------------------- | --------------------------------- |
| context_recall      | ✅ Yes — 27 pp drop, no LLM involved  | —                                 |
| context_precision   | ✅ Yes — 22 pp drop, no LLM involved  | —                                 |
| faithfulness        | Likely real improvement               | Modest shift                      |
| factual_correctness | Partially                             | Yes — strict gpt-4o scoring       |
| answer_relevancy    | Partially                             | Yes — LLM-judged                  |

The retrieval metrics (precision and recall) are the ground truth here. Both dropped ~27 pp. The implementation clearly regressed retrieval quality.

---

## 7. What Phase 1 Got Right

Despite the overall regression, two things did work:

1. **Faithfulness improved** (0.850 → 0.877, +3.2 pp). The smaller, focused chunks that do get retrieved are more topically coherent, so the LLM hallucinates less when it has relevant context.
2. **The schema-identity merge check** correctly prevents the 8-page IO_MUX+GPIO Matrix chain from producing a 45K monolithic chunk. This is a sound change — the regression comes from the row-splitting layer, not the merge check.

---

## 8. Recommendations

### 8.1 Short-Term: Do Not Deploy This Chunking to Production

The context recall and precision regressions are too severe. The `markdown-tables-insteadof-html` config remains the best-performing pipeline for the current evaluation set.

### 8.2 Fix the Retrieval-Granularity Mismatch Before Row Splitting

Row-boundary splitting is the right idea but must be paired with a retrieval strategy that can find specific rows. Three approaches, in order of implementation complexity:

**Option A — Increase `top_k` aggressively for table-heavy queries**  
Bump `top_k` to 15–20 for table-lookup and pin-mapping queries (use query classification or a simple heuristic). This increases the probability of hitting the right row-chunk. Cost: higher latency and token usage in the prompt.

**Option B — Parent-document pattern with dual indexing**  
Index two levels of granularity simultaneously: the full table (or merged table) for recall, and the row-split sub-chunks for precision. At retrieval time, use the small chunks for candidate selection but expand to the parent table for the LLM context window. This is the LangChain `ParentDocumentRetriever` pattern. This directly solves the recall/precision tension.

**Option C — Chunk-level metadata + structured retrieval for pin/register queries**  
Implement Phase 4 of the implementation plan: annotate each row-chunk with the key column values (GPIO number, signal name, register address) as searchable metadata, then use metadata pre-filtering before cosine similarity ranking. A query for "GPIO18" would pre-filter to chunks with `gpio_number=18` rather than relying on dense vector similarity to find a single row.

### 8.3 Validate the Merge Behaviour Post-Schema-Check

Run the ingest pipeline with `DEBUG` logging and verify that pages 47–51 (Peripheral Pin Config) are still merging. Add a test that the merged document for those pages has `merged_pages` containing all five page numbers. If the new schema check is blocking these, relax the comparison: a schema is "same" if the intersection of column names (not the exact set) covers the majority of columns.

### 8.4 Re-Run Baselines with `gpt-4o` as Eval Model

To have a clean comparison of factual correctness going forward, re-run at least `markdown-tables-insteadof-html` with `eval_model: gpt-4o`. The current factual correctness comparison is contaminated by the scorer change.

### 8.5 Add Chunk-Size Histogram Logging to the Ingest Script

Before committing to any new chunking strategy, add instrumentation that logs:

- Total number of chunks
- Max, P99, P95, P50 chunk size (characters)
- Number of chunks above `CHUNK_SIZE` and `2 × CHUNK_SIZE`
- Number of unique table-derived chunks versus prose chunks

This histogram should be logged to MLflow as a custom metric so that chunking experiments are directly comparable in the tracking UI without having to reread logs.

---

## 9. Summary

The Phase 1 implementation is technically correct — the `PipeTable` AST parses, splits, and renders self-contained table chunks as designed — but the assumption that row-granularity chunks would improve retrieval was incorrect for the current retrieval setup. The oversized-chunk problem was real, but the baseline pipeline had a compensating property: one large recall-heavy chunk could answer many question types. The new design requires the embedding-based retriever to pinpoint a specific 3-row sub-chunk from a table that may have 50+ nearly-identical siblings, and cosine similarity at that granularity is too noisy to do it reliably with `top_k=5`.

The next experiment should pair the row-splitting logic with either the parent-document retrieval pattern (Option B) or an increased `top_k` plus metadata filtering (Option C) before running a full RAGAS evaluation.
