# Chempile Retrieval Datasets

This directory contains 9 retrieval dataset variants generated from StackExchange Q&A data.

## Variants

**Domain variants:**
- **A:** Chemistry only (~3.2k pairs)
- **B:** Chemistry + Matter Modeling (~3.6k pairs)
- **C:** All three domains (~8.3k pairs)

**Query modes:**
- **1:** Query = title
- **2:** Query = question body
- **3:** Query = title + question

## File Structure

Each variant (A1-C3) contains:
- `corpus.jsonl` - Answer documents (`{_id, text}`)
- `queries.jsonl` - Questions (`{_id, text}`)
- `default.jsonl` - Qrels (`{query-id, corpus-id, score}`)

## Loading

```python
from datasets import load_dataset

# Load corpus
corpus = load_dataset("json", data_files="A1/corpus.jsonl", split="train")

# Load queries
queries = load_dataset("json", data_files="A1/queries.jsonl", split="train")

# Load qrels
qrels = load_dataset("json", data_files="A1/default.jsonl", split="train")
```

## Stats

| Variant | Corpus | Queries | Qrels |
|---------|--------|---------|-------|
| A1-A3 | 3.1MB | 0.3-2.1MB | 218KB |
| B1-B3 | 3.6MB | 0.3-2.6MB | 249KB |
| C1-C3 | 8.5MB | 0.8-6.5MB | 570KB |

## Generation

Generated using `scripts/generate_datasets.py` from `jablonkagroup/chempile-reasoning`.

**Quality controls applied:**
- Deduplicated questions by exact match (title + q)
- No length filtering
- 1:1 query-corpus mapping
