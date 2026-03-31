#!/usr/bin/env python3
"""Embedding diagnostics for ChempileRetrieval.

Computes, per model:
- embedding norm stats (mean/p95/max)
- collapse / anisotropy proxy: mean pairwise cosine (sample)
- dot vs cosine ranking agreement + HitRate@1 on a sampled eval

Designed to be run locally and write a small markdown report.

Example:
  .venv/bin/python scripts/embedding_diagnostics.py --variant A3 --queries 200 --corpus 2000 \
    --out results/embedding_diagnostics_A3.md

Notes:
- Uses the SAME model-loading logic as scripts/run_evaluation.py:
  - ChEmbed models -> ChEmbedWrapper
  - other models -> mteb.get_model(...)
- Does not print environment variables.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import sys

# Ensure repo root is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mteb
from chempile_retrieval.model_wrappers import ChEmbedWrapper


MODELS = [
    "nomic-ai/nomic-embed-text-v1",
    "BASF-AI/ChEmbed-vanilla",
]


def read_jsonl(path: Path, limit: int | None = None) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def l2norms(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


def cosine_mean_pairwise(x: np.ndarray, max_pairs: int = 50_000) -> float:
    # compute mean cosine over random pairs (avoid O(n^2))
    n = x.shape[0]
    if n < 2:
        return float("nan")

    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(0)
    pairs = min(max_pairs, n * (n - 1) // 2)

    # sample indices
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    mask = i != j
    i = i[mask]
    j = j[mask]
    if i.size == 0:
        return float("nan")
    sims = (x[i] * x[j]).sum(axis=1)
    return float(sims.mean())


def top1_ids(scores: np.ndarray, doc_ids: List[str]) -> List[str]:
    idx = np.argmax(scores, axis=1)
    return [doc_ids[i] for i in idx]


def hitrate_at_1(pred_doc_ids: List[str], rel: Dict[str, str]) -> float:
    # rel: query_id -> correct_corpus_id
    hits = 0
    n = 0
    for qi, di in enumerate(pred_doc_ids):
        qid = str(qi)
        # we will remap with external list; caller provides mapping
    raise RuntimeError("use hitrate_at_1_ids")


def hitrate_at_1_ids(pred_doc_ids: List[str], query_ids: List[str], rel: Dict[str, str]) -> float:
    hits = 0
    n = 0
    for qid, did in zip(query_ids, pred_doc_ids):
        n += 1
        if rel.get(qid) == did:
            hits += 1
    return hits / max(n, 1)


def summarize_norms(norms: np.ndarray) -> str:
    if norms.size == 0:
        return "n=0"
    return f"n={norms.size}, mean={norms.mean():.3f}, p95={np.percentile(norms,95):.3f}, max={norms.max():.3f}"


def load_model(model_name: str):
    trust_remote_code = True
    if "BASF-AI/ChEmbed" in model_name:
        return ChEmbedWrapper(model_name, trust_remote_code=trust_remote_code)
    return mteb.get_model(model_name, trust_remote_code=trust_remote_code)


def encode(model, texts: List[str], batch_size: int, prompt_name: str | None) -> np.ndarray:
    """Encode raw texts using the underlying SentenceTransformer.

    We intentionally bypass the MTEB wrapper's encode(DataLoader, task_metadata, ...)
    API to keep this diagnostic lightweight.

    If the underlying SentenceTransformer supports prompts, we pass prompt_name.
    """
    inner = getattr(model, "model", model)

    kw = dict(batch_size=batch_size, show_progress_bar=False)
    if prompt_name is not None:
        # sentence-transformers v5 supports prompt_name.
        try:
            emb = inner.encode(texts, prompt_name=prompt_name, **kw)
            return np.asarray(emb)
        except TypeError:
            pass

        # Fallback: manually prefix with the prompt text if present
        prompts = getattr(inner, "prompts", None)
        if isinstance(prompts, dict) and prompt_name in prompts:
            pref = prompts[prompt_name]
            texts = [pref + t for t in texts]

    emb = inner.encode(texts, **kw)
    return np.asarray(emb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="A1..C3")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--corpus", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    v = args.variant
    q_path = args.data_dir / v / "queries.jsonl"
    c_path = args.data_dir / v / "corpus.jsonl"
    r_path = args.data_dir / v / "default.jsonl"

    queries = read_jsonl(q_path, limit=args.queries)
    corpus = read_jsonl(c_path, limit=args.corpus)
    qrels = read_jsonl(r_path, limit=None)

    query_ids = [str(o.get("_id")) for o in queries]
    query_texts = [o.get("text", "") for o in queries]

    doc_ids = [str(o.get("_id")) for o in corpus]
    doc_texts = [o.get("text", "") for o in corpus]

    # Build 1:1 relevance mapping (query_id -> corpus_id) for the sampled queries.
    rel: Dict[str, str] = {}
    corpus_id_set = set(doc_ids)
    qid_set = set(query_ids)
    for row in qrels:
        qid = str(row.get("query-id"))
        cid = str(row.get("corpus-id"))
        score = row.get("score", 0)
        if qid in qid_set and cid in corpus_id_set and float(score) > 0:
            # if multiple, keep first (should be 1:1)
            rel.setdefault(qid, cid)

    lines: List[str] = []
    lines.append(f"# Embedding diagnostics – {v}\n")
    lines.append(f"Sampled queries: {len(query_texts)} | Sampled corpus: {len(doc_texts)}\n")
    lines.append("Metrics:")
    lines.append("- norm stats (query/doc)")
    lines.append("- mean pairwise cosine (query/doc) as a collapse proxy")
    lines.append("- dot vs cosine: top1 agreement + HitRate@1 (sampled)")
    lines.append("")

    for model_name in MODELS:
        lines.append(f"## {model_name}\n")
        model = load_model(model_name)

        q_emb = encode(model, query_texts, batch_size=args.batch_size, prompt_name="query")
        d_emb = encode(model, doc_texts, batch_size=args.batch_size, prompt_name="document")

        q_norm = l2norms(q_emb)
        d_norm = l2norms(d_emb)

        # collapse proxy
        q_pair = cosine_mean_pairwise(q_emb)
        d_pair = cosine_mean_pairwise(d_emb)

        # scoring comparisons
        # dot
        dot_scores = q_emb @ d_emb.T
        dot_top1 = top1_ids(dot_scores, doc_ids)

        # cosine
        qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        dn = d_emb / (np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-12)
        cos_scores = qn @ dn.T
        cos_top1 = top1_ids(cos_scores, doc_ids)

        agree = sum(1 for a, b in zip(dot_top1, cos_top1) if a == b) / max(1, len(dot_top1))

        hr_dot = hitrate_at_1_ids(dot_top1, query_ids, rel)
        hr_cos = hitrate_at_1_ids(cos_top1, query_ids, rel)

        lines.append(f"- Query norms: {summarize_norms(q_norm)}")
        lines.append(f"- Doc norms:   {summarize_norms(d_norm)}")
        lines.append(f"- Mean pairwise cosine (queries): {q_pair:.4f}")
        lines.append(f"- Mean pairwise cosine (docs):    {d_pair:.4f}")
        lines.append(f"- Dot vs Cosine top1 agreement: {agree*100:.1f}%")
        lines.append(f"- HitRate@1 (dot): {hr_dot:.4f}")
        lines.append(f"- HitRate@1 (cos): {hr_cos:.4f}")
        lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
