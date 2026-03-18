# Hybrid Search With Reranking Analysis

## Executive Summary

The reranked hybrid pipeline is an improvement over hybrid retrieval alone, but the gains are narrower than expected.

- Overall factual correctness improved from 0.655 to 0.696.
- Context precision improved from 0.568 to 0.584.
- Faithfulness improved from 0.914 to 0.932.
- Context recall declined slightly from 0.872 to 0.861.
- Answer relevancy declined slightly from 0.656 to 0.642.

This means the reranker is helping remove some noisy context and making answers more grounded, but it is not solving the main failure mode for datasheet QA: exact row, exact cell, and identifier-heavy retrieval.

## What Improved

### Multi-row reasoning improved materially

The strongest gain is in `table_multi_row_reasoning`.

- Context precision increased from 0.466 to 0.639.
- Context recall increased from 0.817 to 0.867.
- Factual correctness increased from 0.672 to 0.755.
- Answer relevancy increased from 0.680 to 0.766.

This suggests the reranker is effective when the right evidence is already present in the candidate pool and the task is to sort several partially relevant passages.

### Adversarial and multi-hop behavior is still solid

The reranked run remains strong on `adversarial_multi_hop`.

- Context precision is 0.832.
- Context recall is 1.0.
- Factual correctness is 0.795.

This is consistent with hybrid retrieval doing useful candidate expansion and reranking improving final ordering.

### Some noisy contexts became cleaner

Examples such as the ADC sampling-rate question show that the final answer quality improved even when the retrieved set still contained noise. The reranker is helping prioritize semantically relevant content over broadly related background sections.

## What Did Not Improve Enough

### Exact technical lookup remains weak

The remaining failures are concentrated in:

- Appendix and GPIO matrix lookups
- Exact table-cell questions
- Identifier-heavy pin mapping questions
- Questions requiring exact row extraction from comparison tables

Representative failures include:

- `In the ESP32 GPIO_Matrix, what is signal number 63 and does it have a corresponding IO_MUX core input?`
- `What is the BLE receiver sensitivity and co-channel C/I for the ESP32?`
- `Which pins are ADC2_CH5 and ADC1_CH4 connected to on the ESP32, and what other peripheral functions share those pins?`
- `As part of the hardware validation process ... identify the core type, package dimensions, and permissible VDD_SDIO voltage settings specifically for the ESP32-D0WD-V3 ...`

These are not generic semantic matching problems. They depend on exact identifiers, exact row retrieval, and precise technical extraction.

### Cross-section synthesis is still unstable

`cross_section_synthesis` improved only slightly in factual correctness, from 0.594 to 0.622, while recall fell from 0.933 to 0.733.

That pattern suggests the reranker is sometimes over-pruning evidence that should be combined across multiple chunks.

### Table single-cell lookup is still underperforming

`table_single_cell_lookup` did not materially recover.

- Context precision fell from 0.625 to 0.468.
- Context recall improved slightly from 0.778 to 0.819.
- Factual correctness improved modestly from 0.602 to 0.647.

This is better overall, but still not where a datasheet-focused system should be for exact value lookup.

This weakness should not be attributed only to chunk size. The earlier markdown-table experiment performed competitively while keeping tables intact, which suggests that single-cell lookup is currently limited by a combination of factors:

- some exact lookup questions still fail at retrieval time because the correct table is not surfaced at all
- technical identifiers and symbolic tokens are not handled strongly by the current sparse retrieval configuration
- the reranker is better at broad semantic passage relevance than exact technical row selection
- when the correct table is retrieved, the answer model can still extract the wrong row, wrong column, or an incomplete subset of requested values

In other words, chunk granularity is part of the story, but not the dominant explanation for the current single-cell lookup errors.

## Why The Reranker Underperformed Relative To Expectation

### 1. Row-aware chunking exists, but it is not applied to every table

The current chunker only applies pipe-table row splitting when a section exceeds `CHUNK_SIZE`.

Implication:

- Large sections get row-aware table chunks.
- Medium-sized sections under the size threshold are emitted as one chunk, even if they contain a table.

This does not mean whole-table chunks are inherently bad. In earlier experiments, keeping compact tables intact worked well because it preserved local table context and reduced ranking fragmentation. The real limitation is that the current pipeline has only one effective behavior for smaller tables: they remain whole even when the query would benefit from more row-targeted retrieval.

As a result, some exact table lookups are still searching across multi-row or whole-table chunks when the task would be easier with more targeted row-level evidence, while other questions may actually benefit from the larger table context remaining intact.

### 2. Retrieval ignores the row-aware metadata entirely

The chunker stores useful metadata for tables, including:

- `table_header`
- `key_col_start`
- `key_col_end`

That metadata is persisted into `document_pages.metadata`, but the retrieval and reranking pipeline does not use it at all.

Implication:

- The system pays the ingestion cost for structured chunk metadata.
- The retrieval stage behaves as if those chunks were plain text only.

This is the main reason row-aware chunking has not translated into stronger exact lookup performance.

### 3. The sparse retrieval leg is not tuned for technical identifiers

The BM25-style leg uses an English text-search configuration. That works reasonably well for prose, but many failing questions are dominated by identifiers and symbols such as:

- `ADC2_CH5`
- `GPIO_Matrix`
- `HT20 MCS7`
- `HBM`
- `CDM`
- `VDD_SDIO`
- signal numbers like `63`

These tokens do not behave like normal English text. The current sparse retrieval strategy is therefore weaker exactly where datasheet QA needs it to be strongest.

### 4. The reranker only sees raw chunk text

The reranker receives passage text and page number, but not the higher-signal metadata that would make technical ranking easier.

It does not explicitly see:

- the section title
- the table header
- the row key range
- whether the chunk came from a table or from prose

For technical ranking, those fields are often more informative than the raw text alone.

### 5. Some remaining failures are no longer retrieval failures

Several examples have strong or complete context recall but still low factual correctness. That means retrieval found the evidence, but the answering step did not extract or assemble it cleanly.

The Bluetooth LE compliance question is the clearest example: the retrieved evidence contains the needed capabilities, but the answer still under-specifies or incompletely enumerates them.

## Interpretation By System Layer

### Chunking

Partially effective.

- The design is directionally correct.
- The implementation is not universal enough because row-aware splitting is gated by section size.
- The earlier markdown-table results suggest that preserving whole tables can be beneficial for compact tables, so the likely target state is a mixed strategy rather than splitting every table as aggressively as possible.

### Hybrid retrieval

Useful for recall expansion.

- It continues to help on multi-hop and mixed-signal retrieval.
- It is not enough on its own for identifier-driven exact lookup.

### Reranking

Helpful, but limited.

- Good at reordering semantically related passages.
- Not strong enough for exact datasheet row selection without richer passage features.

### Generation

Still a material source of error.

- Strong faithfulness shows the model is mostly grounded.
- Lower factual correctness on some high-recall questions shows extraction and answer formatting remain inconsistent.

## Most Likely Root Causes

1. Exact lookup failures are caused by a mismatch between datasheet-style queries and the current retrieval signals.
2. Row-aware metadata is generated but unused.
3. Table splitting is only conditionally applied, so the system cannot choose between whole-table context and row-level targeting based on query type.
4. The reranker is optimizing plain-text semantic relevance instead of technical row-level relevance.
5. The answer model still misses required fields on multi-part technical questions even when evidence is present.

## Recommended Next Steps

The next round of work should focus less on adding another model and more on making the current structure observable to retrieval.

Priority direction:

- Make table handling query-aware or structure-aware instead of relying on a single strategy for all table shapes.
- Expose structured table metadata to ranking.
- Add identifier-aware retrieval support.
- Tighten answer generation for exact extraction tasks.

## Patch Plan

### Phase 1: Retrieval and Chunking Fixes

1. Rework table chunking with three explicit handling modes:
   - compact tables: keep intact when all of the following are true: total table length is at most about 900 to 1200 characters, row count is at most 8 to 10 data rows, column count is at most 4 to 6, and the table is not an appendix-style lookup table. These tables usually work better as a single chunk because they preserve local context and are often queried as a unit.
   - large appendix/comparison tables: split into row-aware chunks when any of the following are true: total table length exceeds about 1200 characters, row count exceeds about 10 rows, the table spans many identifiers or part numbers, or the section title suggests an appendix, comparison table, pin list, IO matrix, or lookup index. Use small row groups with repeated header rows, typically 3 to 6 rows per chunk with 1 overlapping row.
   - identifier-heavy tables: route to a stricter row-key-preserving mode when the first column or query contains tokens such as pin names, signal numbers, part numbers, register names, channel IDs, or mixed alphanumeric identifiers like `ADC2_CH5`, `GPIO27`, `ESP32-D0WD-V3`, or `63`. In this mode, keep the first-column key explicit in metadata, avoid splitting inside a logical row, and prefer 1 to 3 data rows per chunk.
   - fallback rule: if a table is compact but identifier-dense, prefer the identifier-heavy mode over the compact mode. If a table is both large and identifier-dense, prefer identifier-heavy row-group chunks rather than broad comparison chunks.
2. Preserve and surface chunk metadata during retrieval, including `table_header`, `key_col_start`, and `key_col_end`.
3. Expand retrieved fields so reranking and generation can see structured metadata, not just `content` and `page_number`.

### Phase 2: Ranking Improvements

4. Build reranker passages from enriched text that includes section headers and table metadata before the chunk body.
5. Add metadata-based boosts or filters for identifier-like queries, especially for first-column keys and table names.
6. Keep hybrid retrieval candidate expansion, but tune ranking to prefer exact row/table matches over general prose matches.

### Phase 3: Sparse Retrieval Improvements

7. Add a technical retrieval path for symbol-heavy queries such as pin names, channel names, table IDs, and signal numbers.
8. Revisit the PostgreSQL text-search configuration so identifier-heavy queries are not treated like ordinary English prose.

### Phase 4: Answer Generation Tightening

9. Refine generation instructions for exact lookup and multi-part questions so the model must enumerate all requested fields.
10. Prefer concise extraction-oriented outputs for technical questions rather than short summaries that may omit required values.

### Phase 5: Validation

11. Re-run the full evaluation after Phase 1 and Phase 2 before making more model-level changes.
12. Track a focused subset of failure classes separately: GPIO matrix lookups, comparison-table row retrieval, exact BLE/Wi-Fi numeric lookups, and cross-section synthesis.
