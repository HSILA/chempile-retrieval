# Chempile Retrieval

MTEB-style retrieval dataset built from StackExchange chemistry/physics Q&A data.

## Dataset Overview

**9 retrieval datasets** constructed from `jablonkagroup/chempile-reasoning`:
- 3 domain variants (chemistry / chem+mattermodeling / all three)
- 3 query-corpus pairing modes (title / question / combined)

## Quick Start

```python
from datasets import load_dataset

# Load chemistry-only, title-as-query variant
# (Hub loading is optional; local eval uses ./data/* files)
ds = load_dataset("HSILA/chempile-retrieval", "A1")

corpus = ds["corpus"]    # {_id, text}
queries = ds["queries"]  # {_id, text}
qrels = ds["default"]    # {query-id, corpus-id, score}
```

## Evaluation (Local, CPU)

This repo supports running MTEB evaluations against the local JSONL files under `./data/{A1..C3}/`.

### 1) Create a UV virtualenv (CPU-only)

```bash
cd ~/research/chempile-retrieval
uv venv .venv --clear
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python -r requirements.txt
```

### 2) HuggingFace token (required for gated models)

Some models require accepting terms / having an email on HuggingFace. Export `HF_TOKEN` before running.

Two equivalent ways:

```bash
export HF_TOKEN="<your_hf_token>"
```

or inline for a single command:

```bash
HF_TOKEN="<your_hf_token>" <command>
```

### 3) Dump embeddings (for plotting)

On small machines, it can be easier to dump a small sample of embeddings to disk and then plot offline.

```bash
HF_TOKEN="<your_hf_token>" .venv/bin/python scripts/dump_embeddings.py \
  --variant A3 \
  --model nomic-ai/nomic-embed-text-v1 \
  --n-queries 200 \
  --n-corpus 200 \
  --batch-size 4 \
  --outdir results/embeddings
```

This writes:
- `results/embeddings/<variant>/<model_slug>/queries.npy`
- `results/embeddings/<variant>/<model_slug>/corpus.npy`
- `results/embeddings/<variant>/<model_slug>/meta.json`

### 4) Run one model on one or more tasks

```bash
HF_TOKEN="<your_hf_token>" .venv/bin/python scripts/run_evaluation.py \
  --model nomic-ai/nomic-embed-text-v1 \
  --tasks A1,A2,A3,B1,B2,B3,C1,C2,C3 \
  --batch-size 4
```

- `--tasks` accepts a comma-separated list of variants.
- `trust_remote_code` is **always enabled** for model loading (required for nomic + ChEmbed).

Outputs are written to:

- `./results/<model_name_with__>/...`

### 4) Run all 5 models (45 evaluations)

```bash
HF_TOKEN="<your_hf_token>" ./scripts/run_all_models_cpu.sh
```

## Dataset Variants

| ID | Domains | Query Mode |
|----|---------|------------|
| A1 | Chemistry | title |
| A2 | Chemistry | question |
| A3 | Chemistry | title + question |
| B1 | Chem + Matter | title |
| B2 | Chem + Matter | question |
| B3 | Chem + Matter | title + question |
| C1 | All three | title |
| C2 | All three | question |
| C3 | All three | title + question |

## Source Data

- **Dataset:** [jablonkagroup/chempile-reasoning](https://huggingface.co/datasets/jablonkagroup/chempile-reasoning)
- **Columns used:** `title`, `q`, `a` only
