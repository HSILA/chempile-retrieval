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
ds = load_dataset("HSILA/chempile-retrieval", "A1")

corpus = ds["corpus"]    # {_id, text}
queries = ds["queries"]  # {_id, text}
qrels = ds["default"]    # {query-id, corpus-id, score}
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

## Status

🚧 **In Progress** - Specification complete, implementation pending

See [SPEC.md](./SPEC.md) for full technical details.

## Source Data

- **Dataset:** [jablonkagroup/chempile-reasoning](https://huggingface.co/datasets/jablonkagroup/chempile-reasoning)
- **Columns used:** `title`, `q`, `a` only
