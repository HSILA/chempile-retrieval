# Chempile Retrieval - Specification

> **Internal file** - Keep in GitHub repo, exclude from HuggingFace Hub uploads.

---

## Dataset Source

- **Source:** `jablonkagroup/chempile-reasoning`
- **Configs:** `chemistry_stackexchange-raw_data`, `mattermodeling_stackexchange-raw_data`, `physics_stackexchange-raw_data`
- **Columns used:** `title`, `q`, `a` ONLY
- **Ignore:** `split`, `text`, `__index_level_0__`, all other columns

---

## Output: 9 Datasets (3 × 3)

### Domain Variants

- **A:** chemistry only
- **B:** chemistry + mattermodeling
- **C:** all three (chemistry + mattermodeling + physics)

### Query-Corpus Pairing Modes

- **Mode 1:** Query = `title` → Corpus = `a`
- **Mode 2:** Query = `q` → Corpus = `a`
- **Mode 3:** Query = `title + " " + q` → Corpus = `a`

### Naming

`{domain}{mode}` → A1, A2, A3, B1, B2, B3, C1, C2, C3

---

## Output Format (Mimic ChemRxivRetrieval)

Each dataset has 3 subsets:

- **corpus** (train split): `{_id, text}` - answer documents
- **queries** (train split): `{_id, text}` - questions
- **default** (test split): `{query-id, corpus-id, score}` - qrels (score=1)

### Mapping

- Each query maps to exactly ONE corpus document (1:1)
- Binary relevance (score = 1)

---

## File Structure (Local-First)

```
chempile-retrieval/
├── data/                      # Generated datasets (27 JSONL files)
│   ├── A1/
│   │   ├── A1.py              # Local HF loading script
│   │   ├── corpus.jsonl
│   │   ├── queries.jsonl
│   │   └── default.jsonl
│   ├── A2/
│   │   ├── A2.py
│   │   └── ...
│   ├── ... (9 subdirs total)
│   └── C3/
├── chempile_retrieval/        # Python package
│   ├── __init__.py
│   ├── tasks.py               # 9 task classes (A1-C3)
│   └── loader.py              # HuggingFace loading utilities
├── results/                   # Evaluation results
│   ├── nomic-ai__nomic-embed-text-v1/
│   │   ├── A1.json
│   │   ├── A2.json
│   │   └── ... (9 JSON files per model)
│   ├── BASF-AI__ChEmbed-vanilla/
│   ├── BASF-AI__ChEmbed-full/
│   ├── BASF-AI__ChEmbed-plug/
│   └── BASF-AI__ChEmbed-prog/
├── scripts/
│   ├── generate_datasets.py   # Create 9 datasets from source
│   └── run_evaluation.py      # Run MTEB on all 9 tasks × 5 models
├── SPEC.md                    # Internal spec (gitignored)
├── README.md
└── requirements.txt
```

**Total:** 9 subdirs × 4 files (script + 3 JSONL) = 36 files in data/

---

## Difficulty Comparison

**Chempile (N queries, N corpus):**
- Each query: 1 relevant + (N-1) distractors
- Example: 5k queries → 4,999 distractors per query
- Easier task (fewer distractors)

**ChemRxivRetrieval (5k queries, 70k corpus):**
- Each query: 1 relevant + 69,999 distractors
- Harder task (more distractors)

**Evaluation logic is identical** - MTEB evaluates each query independently against full corpus.

---

## Models (5 Total)

**Base model:**
- `nomic-ai/nomic-embed-text-v1`

**ChEmbed variants (4):**
- `BASF-AI/ChEmbed-vanilla`
- `BASF-AI/ChEmbed-full`
- `BASF-AI/ChEmbed-plug`
- `BASF-AI/ChEmbed-prog` (progressive)

---

## Metrics (MTEB Default)

**Provided automatically:**
- k_values = (1, 3, 5, 10, 20, 100, 1000)
- NDCG@k
- MAP@k
- Recall@k
- Precision@k
- MRR@k
- HitRate@k (success.k) - includes **HitRate@1** (equivalent to Accuracy@1)

No configuration needed. All metrics computed automatically.

---

## Quality Controls

1. **Keep all answers** (no length filtering)
2. **Deduplicate questions by exact match** on `title` + `q`
   - Use exact string comparison (no fuzzy matching)
   - Simple `set()` or `dict` deduplication
3. **Use existing `split` column** from source data
4. **Stable ID scheme** (hash-based or sequential)

**Do NOT deduplicate answers** - multiple questions can legitimately share the same answer.

---

## Local-First Approach

**Stage 1 (Current):**
- All datasets kept locally in `data/`
- Each dataset has local loading script (e.g., `data/A1/A1.py`)
- Task definitions point to local paths: `dataset={"path": "./data/A1", ...}`
- No HuggingFace Hub upload required
- Run evaluations locally

**Stage 2 (Future):**
- Once best dataset variant identified
- Push that ONE dataset to HuggingFace Hub
- Update task definition to point to HF Hub

**Benefits:**
- Fast iteration without HF uploads
- Works exactly like ChemRxivRetrieval
- Easy to test/compare all 9 variants
- Can push to HF Hub later (just change path)

---

## Task Definition Pattern

```python
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

class ChempileRetrievalA1(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChempileRetrievalA1",
        dataset={
            "path": "./data/A1",  # Local path
            "revision": "local",
        },
        description="Chempile retrieval - chemistry only, title as query",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        domains=["Chemistry"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        ...
    )
```

---

## Implementation Checklist

### Stage 1: Dataset Creation
- [ ] Write `scripts/generate_datasets.py`
  - Load from `jablonkagroup/chempile-reasoning`
  - Generate all 9 variants (A1-C3)
  - Apply quality controls (filtering, dedup)
  - Output JSONL files to `data/`
- [ ] Create local loading scripts (`data/A1/A1.py`, etc.)
- [ ] Verify format matches ChemRxivRetrieval

### Stage 2: Task Definitions
- [ ] Write `chempile_retrieval/tasks.py`
  - 9 task classes inheriting from `AbsTaskRetrieval`
  - Each points to corresponding local dataset
- [ ] Write `chempile_retrieval/__init__.py`
  - Export all task classes

### Stage 3: Evaluation
- [ ] Write `scripts/run_evaluation.py`
  - Load 5 models (nomic + 4 ChEmbed variants)
  - Run MTEB evaluation on all 9 tasks
  - Save results to `results/`
- [ ] Collect and analyze results
- [ ] Identify best dataset variant

### Stage 4: Publication (Future)
- [ ] Push best dataset to HuggingFace Hub
- [ ] Update task definition to point to HF Hub
- [ ] Document final dataset

---

## HuggingFace Hub Upload (Later)

When ready to upload best variant (e.g., B3):

1. Create dataset repo: `HSILA/chempile-retrieval-b3`
2. Upload files:
   - `chempile_retrieval.py` (loading script)
   - `corpus.jsonl`
   - `queries.jsonl`
   - `default.jsonl`
   - `README.md`
3. Update task:
   ```python
   dataset={"path": "HSILA/chempile-retrieval-b3", "revision": "main"}
   ```

---

## Notes

- Keep SPEC.md gitignored (internal only)
- README.md should NOT mention SPEC.md
- All evaluation runs use local datasets
- Results stored in `results/` directory
