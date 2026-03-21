# Chempile Retrieval Dataset Specification

**Status:** Specification phase (not yet implemented)
**Last updated:** 2026-03-21

## Source Dataset

- **Dataset:** `jablonkagroup/chempile-reasoning`
- **Configs used:**
  - `chemistry_stackexchange-raw_data`
  - `mattermodeling_stackexchange-raw_data`
  - `physics_stackexchange-raw_data`

**Important:** Only use columns `title`, `q`, `a`. Ignore all other columns including `split`, `text`, `__index_level_0__`.

## Output Datasets

**9 total datasets** (3 domain variants × 3 query modes):

### Domain Variants

| Variant | Domains | Config |
|---------|---------|--------|
| A | Chemistry only | `chemistry_stackexchange-raw_data` |
| B | Chemistry + Matter Modeling | `chemistry_stackexchange-raw_data` + `mattermodeling_stackexchange-raw_data` |
| C | All three | `chemistry_stackexchange-raw_data` + `mattermodeling_stackexchange-raw_data` + `physics_stackexchange-raw_data` |

### Query-Corpus Pairing Modes

| Mode | Query | Corpus | Description |
|------|-------|--------|-------------|
| 1 | `title` | `a` | Title-only queries |
| 2 | `q` | `a` | Question body only |
| 3 | `title + " " + q` | `a` | Combined title + question (space-separated) |

### Dataset Naming Convention

`{domain_variant}{query_mode}` → A1, A2, A3, B1, B2, B3, C1, C2, C3

Example: **B3** = chemistry + mattermodeling, query = title + " " + q

## Output Format

Mimic `BASF-AI/ChemRxivRetrieval` structure:

### Subset 1: `corpus` (train split)
```jsonl
{"_id": "...", "text": "..."}
```
- Contains answer documents
- One document per unique answer

### Subset 2: `queries` (train split)
```jsonl
{"_id": "...", "text": "..."}
```
- Contains question queries
- Format depends on query mode (see above)

### Subset 3: `default` (test split)
```jsonl
{"query-id": "...", "corpus-id": "...", "score": 1}
```
- Qrels mapping queries to corpus docs
- Binary relevance (score = 1)
- **Important:** Each query maps to exactly ONE corpus document (1:1 mapping)

## File Organization

```
chempile-retrieval/
├── data/
│   ├── A1/
│   │   ├── corpus.jsonl
│   │   ├── queries.jsonl
│   │   └── default.jsonl
│   ├── A2/
│   │   ├── corpus.jsonl
│   │   ├── queries.jsonl
│   │   └── default.jsonl
│   ├── A3/
│   │   ├── corpus.jsonl
│   │   ├── queries.jsonl
│   │   └── default.jsonl
│   ├── B1/
│   ├── B2/
│   ├── B3/
│   ├── C1/
│   ├── C2/
│   └── C3/
├── chempile_retrieval.py
├── SPEC.md
└── README.md
```

**Total:** 9 subdirectories × 3 files = 27 JSONL files

## Retrieval Task Design Notes

### 1:1 Mapping vs Multi-Relevant
- **Chempile:** Each query maps to exactly ONE relevant document (1:1)
- **ChemRxivRetrieval:** Also 1:1 (5k queries → 5k relevant docs in 70k corpus)

### Corpus Size Impact
- **ChemRxivRetrieval:** 5k queries, 70k corpus (7% relevant density)
- **Chempile:** ~N queries, ~N corpus (100% relevant density - each doc is relevant to exactly one query)

**Evaluation is identical** (MTEB uses pytrec_eval):
1. For each query, retrieve top-k from corpus
2. Check if relevant doc(s) appear in top-k
3. Compute metrics (NDCG@k, Recall@k, MRR@k, etc.)

**Difficulty difference:**
- Larger corpus with sparse relevance = harder (more distractors)
- Smaller corpus with dense relevance = easier
- But evaluation **logic** is the same

**Key insight:** The number of queries and corpus size being equal does NOT change the task structure. Each query is evaluated independently against the full corpus.

### What Makes This Different from ChemRxivRetrieval
- **Domain:** StackExchange Q/A (not scientific papers)
- **Query types:** title, question body, or combined
- **Corpus:** Full answer texts (not paragraphs)
- **Scale:** TBD (depends on source data filtering)

## Quality Controls

- [ ] Filter answers < 50 characters
- [ ] Deduplicate by title similarity (fuzzy matching)
- [ ] Use existing `split` column from source data
- [ ] Stable ID scheme (hash-based or sequential)

## HuggingFace Loading

```python
from datasets import load_dataset

# Load variant A1 (chemistry, title→a)
ds = load_dataset("HSILA/chempile-retrieval", "A1")

# Access subsets
corpus = ds["corpus"]      # train split
queries = ds["queries"]    # train split
qrels = ds["default"]      # test split
```

## Implementation Checklist

- [ ] Create data loading script (`chempile_retrieval.py`)
- [ ] Generate all 9 dataset variants
- [ ] Verify format matches ChemRxivRetrieval
- [ ] Push to HuggingFace Hub
- [ ] Update README with usage examples
