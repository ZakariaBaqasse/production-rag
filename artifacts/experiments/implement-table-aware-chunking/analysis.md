# RAG Pipeline Evaluation Report: Table-Aware Chunking

## 1. Executive Summary

The implementation of table-aware chunking alongside the `qwen3-embedding:0.6b` model has yielded **excellent retrieval performance**, characterized by a highly impressive **Context Recall of 0.94**. The system is successfully fetching the relevant documents from the ESP32 datasheet for almost every query.

However, the downstream metrics—**Factual Correctness (0.65)** and **Answer Relevancy (0.62)**—indicate severe bottlenecks in the **Generation** and **Evaluation** phases. The pipeline is currently suffering from "Lost in the Middle" syndrome due to dense HTML tables, orphaned footnotes, and overly strict grading by the LLM-as-a-judge.

---

## 2. Deep-Dive: Metric Analysis by Category

| Category                | Context Precision | Context Recall | Faithfulness | Factual Correctness |
| ----------------------- | ----------------- | -------------- | ------------ | ------------------- |
| **Overall**             | **0.80**          | **0.94**       | **0.83**     | **0.65**            |
| `pin_mapping`           | 0.91              | **1.00**       | 0.65         | 0.61                |
| `table_single_cell`     | 0.78              | 0.88           | **0.92**     | 0.66                |
| `table_multi_row`       | **0.66**          | 0.85           | 0.86         | 0.64                |
| `adversarial_multi_hop` | 0.87              | **1.00**       | 0.79         | 0.61                |

### Key Takeaways from the Metrics:

- **The Retriever is NOT the main problem:** Categories like `pin_mapping` and `adversarial_multi_hop` achieved a perfect **1.0 Context Recall**. The system found the exact right pages and tables.
- **The Generator is choking on dense data:** Despite having a perfect 1.0 Context Recall, `pin_mapping` had a terrible 0.61 Factual Correctness and 0.65 Faithfulness.
- **Table Reasoning is structurally hard:** `table_multi_row_reasoning` suffered the lowest Context Precision (0.66), meaning the embedder struggled to align multi-condition queries with the correct chunks of complex tables.

---

## 3. Root Cause Analysis of Failures

By inspecting the raw `results.csv`, we can identify three primary failure modes corrupting the pipeline:

### Failure Mode A: Severe "Lost in the Middle" (HTML Bloat)

- **The Symptom:** High Context Recall, 0.0 Factual Correctness.
- **Example Query:** _"In the ESP32 GPIO_Matrix, what is signal number 63..."_
- **The Cause:** The `gemini-3.1-flash-lite-preview` model is a lightweight model. Llama Parse extracted complex tables (like `IO_MUX` and `GPIO_Matrix`) as raw HTML. When a chunk contains thousands of `<tr>` and `<td>` tags, the Flash-Lite model's attention mechanism breaks down. It physically "reads" the correct table row but hallucinates: _"The provided context does not contain information regarding signal number 63"_ because the target data is buried in HTML syntax noise.

### Failure Mode B: Ragas Strictness / False Negatives

- **The Symptom:** LLM answers correctly, but `gpt-4o-mini` grades it 0.0 for Factual Correctness.
- **Example Query:** _"what pin number GPIO18 is?"_
- **The Cause:** The Ground Truth reference states: _"According to the ESP32 Pin Layout table, GPIO18 corresponds to Pin Number 35."_ The Generator responded: _"Per the IO_MUX table on page 70, GPIO18 is Pin No. 35."_ The evaluator (`gpt-4o-mini`) scored this a 0.0 because the LLM cited the _IO_MUX_ table instead of the _Pin Layout_ table, even though the factual answer (Pin 35) is 100% correct.

### Failure Mode C: Orphaned Footnotes

- **The Symptom:** Missed context in multi-row and multi-hop questions.
- **Example Query:** _"what voltage ESP32-D0WDRH2-V3 need and how connect VDD_SDIO?"_
- **The Cause:** The answer to this lies in a specific footnote beneath Table 2-6. Table-aware chunkers often split the `<table>` element perfectly but leave the text immediately following the table (the footnotes) in a separate, isolated chunk. The isolated footnote lacks semantic context (e.g., it just says _"1. As the in-package flash..."_), so the retriever misses it.

---

## 4. Actionable Solutions (Ranked by Implementation Complexity)

To resolve these issues, here is a prioritized roadmap of fixes, from easiest to most complex.

### 🟢 Tier 1: Low Complexity (Quick Wins)

**1. Upgrade the Generator LLM**

- **The Fix:** Swap `gemini-3.1-flash-lite-preview` to a heavier model like standard `gemini-3.1-flash` or `gpt-4o` for the generation phase.
- **Why:** Larger models have significantly better needle-in-a-haystack recall and are much more resilient to HTML/Markdown syntax bloat. This alone will likely boost the Factual Correctness by 10-15%.

**2. Custom Ragas Prompting for Factual Correctness**

- **The Fix:** Override the default `factual_correctness` prompt in Ragas. Instruct the evaluator LLM: _"Grade ONLY the factual payload (e.g., numeric values, pin names). Do not penalize the response if the citation source (e.g., Table Name) differs from the ground truth, as long as the core fact is correct."_
- **Why:** Eliminates the False Negatives causing your `pin_mapping` scores to artificially tank.

### 🟡 Tier 2: Medium Complexity (Data Cleaning)

**3. Convert HTML Tables to Markdown / CSV / JSON**

- **The Fix:** Add a pre-processing step before embedding. Use BeautifulSoup or a regex script to parse the `<table>` outputs from Llama Parse and convert them into clean Markdown tables or flattened CSV strings.
- **Why:** HTML tags (`<tbody>`, `<tr>`, `<td>`) consume massive amounts of tokens and dilute the semantic density of the chunks. Clean Markdown allows both the Embedder and the Generator LLM to "read" the table structurally without distraction.

**4. Enhance the RAG System Prompt**

- **The Fix:** Add specific instructions to the Generation prompt regarding tables. For example: _"You are analyzing an ESP32 datasheet. Pay special attention to table footnotes. If a table cell contains a footnote marker (e.g., 'note 1', '3'), you must look for the corresponding footnote text in the provided context before answering."_

### 🔴 Tier 3: High Complexity (Advanced Pipeline Architecture)

**5. Parent-Document Retrieval (Auto-Merging)**

- **The Fix:** Implement a `ParentDocumentRetriever` (available in LangChain/LlamaIndex). Break the document into very small chunks (e.g., individual table rows or footnotes) for the _Embedding/Search_ phase, but link them to a larger Parent Chunk (e.g., the entire section including the table AND its footnotes).
- **Why:** This solves the "Orphaned Footnote" problem. If the user searches for a specific footnote condition, the retriever matches the small footnote chunk, but passes the entire table + footnote block to the LLM, ensuring perfect context.

**6. Contextual Table Summarization (Indexing Phase)**

- **The Fix:** During document ingestion, pass every extracted table to an LLM and ask it to write a textual summary of the table's purpose, its column headers, and its footnotes. Embed this LLM summary _alongside_ the raw table.
- **Why:** Standard embedding models (`qwen3-embedding`) are terrible at reading 2D table structures. By converting the structural logic into semantic text, your Context Precision for `table_multi_row_reasoning` will skyrocket.
