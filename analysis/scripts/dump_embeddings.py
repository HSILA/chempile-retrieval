#!/usr/bin/env python3
"""Dump embeddings for ChempileRetrieval to disk (for plotting / offline analysis).

Why:
- This repo runs on small CPU boxes; doing PCA/UMAP directly on-the-fly can OOM.
- This script writes small, reusable embedding artifacts so you can plot later without
  re-running model inference.

Outputs:
- <outdir>/<variant>/<model_slug>/queries.npy
- <outdir>/<variant>/<model_slug>/corpus.npy
- <outdir>/<variant>/<model_slug>/meta.json  (ids + basic stats)

Example:
  HF_TOKEN=... .venv/bin/python analysis/scripts/dump_embeddings.py \
    --variant A3 \
    --model nomic-ai/nomic-embed-text-v1 \
    --n-queries 200 \
    --n-corpus 200 \
    --batch-size 4 \
    --outdir results/embeddings

Notes:
- Uses the same wrapper logic as evaluation:
  - NomicWrapper for Nomic prompts
  - ChEmbedWrapper for ChEmbed (+ ChemVocab tokenizer swap for non-vanilla)
- Writes float32 arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mteb
from chempile_retrieval.model_wrappers import ChEmbedWrapper, NomicWrapper


def slugify(model_name: str) -> str:
    return model_name.replace("/", "__")


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


def load_model(model_name: str):
    trust_remote_code = True
    if "BASF-AI/ChEmbed" in model_name:
        return ChEmbedWrapper(model_name, trust_remote_code=trust_remote_code)
    if "nomic-ai/" in model_name:
        return NomicWrapper(model_name, trust_remote_code=trust_remote_code)
    return mteb.get_model(model_name, trust_remote_code=trust_remote_code)


def encode(model, texts: List[str], batch_size: int, prompt_name: str | None) -> np.ndarray:
    """Encode raw texts using the underlying SentenceTransformer.

    We use the underlying sentence-transformers model so we can easily pass prompt_name.
    """

    inner = getattr(model, "model", model)
    kw = dict(batch_size=batch_size, show_progress_bar=True)

    if prompt_name is not None:
        try:
            emb = inner.encode(texts, prompt_name=prompt_name, **kw)
            return np.asarray(emb)
        except TypeError:
            # older ST versions may not accept prompt_name
            pass

        prompts = getattr(inner, "prompts", None)
        if isinstance(prompts, dict) and prompt_name in prompts:
            pref = prompts[prompt_name]
            texts = [pref + t for t in texts]

    emb = inner.encode(texts, **kw)
    return np.asarray(emb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="A1..C3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--n-corpus", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--outdir", type=Path, default=Path("analysis/embeddings"))
    args = ap.parse_args()

    q_path = args.data_dir / args.variant / "queries.jsonl"
    c_path = args.data_dir / args.variant / "corpus.jsonl"

    queries = read_jsonl(q_path, limit=args.n_queries)
    corpus = read_jsonl(c_path, limit=args.n_corpus)

    query_ids = [str(o.get("_id")) for o in queries]
    doc_ids = [str(o.get("_id")) for o in corpus]

    query_texts = [o.get("text", "") for o in queries]
    doc_texts = [o.get("text", "") for o in corpus]

    model = load_model(args.model)

    q_emb = encode(model, query_texts, batch_size=args.batch_size, prompt_name="query").astype(
        np.float32
    )
    d_emb = encode(model, doc_texts, batch_size=args.batch_size, prompt_name="document").astype(
        np.float32
    )

    out_base = args.outdir / args.variant / slugify(args.model)
    out_base.mkdir(parents=True, exist_ok=True)

    np.save(out_base / "queries.npy", q_emb)
    np.save(out_base / "corpus.npy", d_emb)

    meta = {
        "variant": args.variant,
        "model": args.model,
        "n_queries": len(query_ids),
        "n_corpus": len(doc_ids),
        "dim": int(q_emb.shape[1]) if q_emb.ndim == 2 else None,
        "query_ids": query_ids,
        "corpus_ids": doc_ids,
        "query_norm_mean": float(np.linalg.norm(q_emb, axis=1).mean()) if q_emb.size else None,
        "doc_norm_mean": float(np.linalg.norm(d_emb, axis=1).mean()) if d_emb.size else None,
    }
    (out_base / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote embeddings to: {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
