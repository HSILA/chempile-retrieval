#!/usr/bin/env python3
"""Collect a reproducible embedding bundle for geometry comparisons.

Hardcoded (by design):
- Chempile variant: A3
- Models: BASF-AI/ChEmbed-vanilla and nomic-ai/nomic-embed-text-v1
- ChemRxiv retrieval dataset: BASF-AI/ChemRxivRetrieval
- General anchor datasets: mteb/hotpotqa and mteb/nq (50/50)

Outputs (under results/geometry_bundle/):
- <model_slug>/chempile_A3/{queries.npy, corpus.npy, meta.json}
- <model_slug>/chemrxiv/{queries.npy, corpus.npy, meta.json}
- <model_slug>/anchors_mixed_chem/{texts.jsonl, embeddings.npy, meta.json}
- <model_slug>/anchors_general/{texts.jsonl, embeddings.npy, meta.json}
- manifest.json (top-level)

Notes:
- Sampling is seeded for reproducibility.
- Uses the same wrapper logic as this repo for encoding (NomicWrapper / ChEmbedWrapper).
- Assumes you run with this repo's venv:
    HF_TOKEN=... .venv/bin/python scripts/collect_geometry_bundle.py
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mteb
from datasets import load_dataset
from chempile_retrieval.model_wrappers import ChEmbedWrapper, NomicWrapper


# ------------------------
# Hardcoded config knobs
# ------------------------

VARIANT = "A3"
MODELS: List[str] = [
    "BASF-AI/ChEmbed-vanilla",
    "nomic-ai/nomic-embed-text-v1",
]

CHEMRXIV_DATASET = "BASF-AI/ChemRxivRetrieval"
# HF retrieval datasets here use "configs" (queries/corpus/default), not plain splits.
CHEMRXIV_QUERIES_CONFIG = "queries"
CHEMRXIV_CORPUS_CONFIG = "corpus"

GENERAL_ANCHOR_DATASETS: List[str] = ["mteb/hotpotqa", "mteb/nq"]
# For the *general* anchor we prefer Wikipedia-like passages, not queries.
GENERAL_QUERIES_CONFIG = "queries"
GENERAL_CORPUS_CONFIG = "corpus"

OUTDIR = REPO_ROOT / "results" / "geometry_bundle"
SEED = 1337

# sizes (edit here only; no CLI args by design)
N_CHEMRXIV_QUERIES = 2000
N_CHEMRXIV_CORPUS = 10000

N_CHEMPILE_QUERIES = 2000
N_CHEMPILE_CORPUS = 10000

N_ANCHORS_MIXED_CHEM_TOTAL = 2000  # 50/50 chemrxiv vs chempile
N_ANCHORS_GENERAL_TOTAL = 2000     # 50/50 hotpotqa vs nq

# Match the repo's CPU-oriented defaults used in run_all_models_cpu.sh.
BATCH_SIZE = 4
# Force CPU by default for robustness on macOS; set COLLECT_GEOMETRY_DEVICE=mps/cuda to override.
DEVICE = os.environ.get("COLLECT_GEOMETRY_DEVICE", "cpu")


def slugify(model_name: str) -> str:
    return model_name.replace("/", "__")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_model(model_name: str):
    trust_remote_code = True
    common = dict(trust_remote_code=trust_remote_code, device=DEVICE)
    if "BASF-AI/ChEmbed" in model_name:
        return ChEmbedWrapper(model_name, **common)
    if "nomic-ai/" in model_name:
        return NomicWrapper(model_name, **common)
    return mteb.get_model(model_name, **common)


def encode_texts(model, texts: List[str], batch_size: int, prompt_name: Optional[str]) -> np.ndarray:
    """Encode raw texts using the underlying SentenceTransformer.

    Matches scripts/dump_embeddings.py behavior so prompts work for nomic + others.
    """

    inner = getattr(model, "model", model)
    kw = dict(batch_size=batch_size, show_progress_bar=True)

    if prompt_name is not None:
        try:
            emb = inner.encode(texts, prompt_name=prompt_name, **kw)
            return np.asarray(emb)
        except TypeError:
            pass

        prompts = getattr(inner, "prompts", None)
        if isinstance(prompts, dict) and prompt_name in prompts:
            pref = prompts[prompt_name]
            texts = [pref + t for t in texts]

    emb = inner.encode(texts, **kw)
    return np.asarray(emb)


def _seed_everything(seed: int) -> random.Random:
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    return rng


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _sample_indices(n: int, k: int, rng: random.Random) -> List[int]:
    k = min(k, n)
    if k <= 0:
        return []
    # stable sample
    return rng.sample(range(n), k)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------
# Dataset loaders
# ------------------------


def load_chempile_variant(variant: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load local chempile-retrieval jsonl for variant.

    Returns: query_ids, query_texts, doc_ids, doc_texts
    """
    q_path = REPO_ROOT / "data" / variant / "queries.jsonl"
    c_path = REPO_ROOT / "data" / variant / "corpus.jsonl"
    queries = _read_jsonl(q_path)
    corpus = _read_jsonl(c_path)

    query_ids = [str(o.get("_id")) for o in queries]
    query_texts = [o.get("text", "") for o in queries]
    doc_ids = [str(o.get("_id")) for o in corpus]
    doc_texts = [o.get("text", "") for o in corpus]
    return query_ids, query_texts, doc_ids, doc_texts


def load_chemrxiv_retrieval() -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load ChemRxiv retrieval from HF datasets.

    This HF dataset is an MTEB-style triple with three *configs*:
      - config="queries": split "train" with columns [_id, text]
      - config="corpus":  split "train" with columns [_id, text]
      - config="default": relevance triples (we don't need it for embedding)
    """

    q_ds = load_dataset(CHEMRXIV_DATASET, CHEMRXIV_QUERIES_CONFIG)
    c_ds = load_dataset(CHEMRXIV_DATASET, CHEMRXIV_CORPUS_CONFIG)

    q_split = q_ds["train"] if "train" in q_ds else q_ds[list(q_ds.keys())[0]]
    c_split = c_ds["train"] if "train" in c_ds else c_ds[list(c_ds.keys())[0]]

    q_ids = [str(x) for x in q_split["_id"]]
    q_texts = [str(x) for x in q_split["text"]]

    d_ids = [str(x) for x in c_split["_id"]]
    d_texts = [str(x) for x in c_split["text"]]

    return q_ids, q_texts, d_ids, d_texts


def load_general_anchor_texts(dataset_id: str, *, config: str, limit: int) -> List[str]:
    """Load a subset of *passage texts* from an MTEB-style HF dataset.

    For mteb/hotpotqa and mteb/nq we prefer the Wikipedia-like corpus passages:
      - config="corpus" with split "corpus" and columns [_id, title, text]

    We return the passage `text`.
    """

    ds = load_dataset(dataset_id, config)
    split = ds["corpus"] if "corpus" in ds else ds[list(ds.keys())[0]]
    texts = [str(x) for x in split["text"]]
    return texts[: min(limit, len(texts))]


# ------------------------
# Writers
# ------------------------


def dump_qc_embeddings(
    *,
    out_base: Path,
    model_name: str,
    dataset_name: str,
    query_ids: List[str],
    query_texts: List[str],
    doc_ids: List[str],
    doc_texts: List[str],
    n_queries: int,
    n_docs: int,
    rng: random.Random,
) -> Dict[str, Any]:
    """Sample, encode, and write query/corpus embeddings."""

    q_idx = _sample_indices(len(query_texts), n_queries, rng)
    d_idx = _sample_indices(len(doc_texts), n_docs, rng)

    q_ids_s = [query_ids[i] for i in q_idx]
    d_ids_s = [doc_ids[i] for i in d_idx]

    q_txt_s = [query_texts[i] for i in q_idx]
    d_txt_s = [doc_texts[i] for i in d_idx]

    model = load_model(model_name)

    q_emb = encode_texts(model, q_txt_s, batch_size=BATCH_SIZE, prompt_name="query").astype(
        np.float32
    )
    d_emb = encode_texts(model, d_txt_s, batch_size=BATCH_SIZE, prompt_name="document").astype(
        np.float32
    )

    _ensure_dir(out_base)
    np.save(out_base / "queries.npy", q_emb)
    np.save(out_base / "corpus.npy", d_emb)

    meta = {
        "dataset": dataset_name,
        "model": model_name,
        "n_queries": len(q_ids_s),
        "n_corpus": len(d_ids_s),
        "dim": int(q_emb.shape[1]) if q_emb.ndim == 2 else None,
        "query_ids": q_ids_s,
        "corpus_ids": d_ids_s,
        "seed": SEED,
        "query_norm_mean": float(np.linalg.norm(q_emb, axis=1).mean()) if q_emb.size else None,
        "doc_norm_mean": float(np.linalg.norm(d_emb, axis=1).mean()) if d_emb.size else None,
    }
    _write_json(out_base / "meta.json", meta)

    return {
        "out": str(out_base),
        "n_queries": len(q_ids_s),
        "n_corpus": len(d_ids_s),
        "dim": meta["dim"],
    }


def dump_anchor_embeddings(
    *,
    out_base: Path,
    model_name: str,
    anchor_name: str,
    texts: List[str],
) -> Dict[str, Any]:
    model = load_model(model_name)
    emb = encode_texts(model, texts, batch_size=BATCH_SIZE, prompt_name=None).astype(np.float32)

    _ensure_dir(out_base)
    rows = [{"_id": str(i), "text": t} for i, t in enumerate(texts)]
    _write_jsonl(out_base / "texts.jsonl", rows)
    np.save(out_base / "embeddings.npy", emb)

    meta = {
        "anchor": anchor_name,
        "model": model_name,
        "n": len(texts),
        "dim": int(emb.shape[1]) if emb.ndim == 2 else None,
        "seed": SEED,
    }
    _write_json(out_base / "meta.json", meta)

    return {"out": str(out_base), "n": len(texts), "dim": meta["dim"]}


def main() -> int:
    rng = _seed_everything(SEED)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load datasets once
    cx_q_ids, cx_q_txt, cx_d_ids, cx_d_txt = load_chemrxiv_retrieval()
    cp_q_ids, cp_q_txt, cp_d_ids, cp_d_txt = load_chempile_variant(VARIANT)

    # Prepare anchor texts (reproducibly)
    # Mixed-chem anchors: 50/50 sampled from corpora
    n_half = N_ANCHORS_MIXED_CHEM_TOTAL // 2
    cx_anchor_idx = _sample_indices(len(cx_d_txt), n_half, rng)
    cp_anchor_idx = _sample_indices(len(cp_d_txt), N_ANCHORS_MIXED_CHEM_TOTAL - n_half, rng)
    anchors_mixed_chem = [cx_d_txt[i] for i in cx_anchor_idx] + [cp_d_txt[i] for i in cp_anchor_idx]

    # General anchors: 50/50 from HotpotQA + NQ (use MTEB "corpus" config for passages)
    n_half_g = N_ANCHORS_GENERAL_TOTAL // 2
    hotpot = load_general_anchor_texts(
        GENERAL_ANCHOR_DATASETS[0], config=GENERAL_CORPUS_CONFIG, limit=n_half_g
    )
    nq = load_general_anchor_texts(
        GENERAL_ANCHOR_DATASETS[1], config=GENERAL_CORPUS_CONFIG, limit=N_ANCHORS_GENERAL_TOTAL - n_half_g
    )
    # Shuffle deterministically so we're not always taking the same prefix ordering.
    anchors_general = hotpot + nq
    rng.shuffle(anchors_general)

    # Write a top-level anchor manifest (shared across models)
    shared = {
        "seed": SEED,
        "variant": VARIANT,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "chemrxiv_dataset": CHEMRXIV_DATASET,
        "general_anchor_datasets": GENERAL_ANCHOR_DATASETS,
        "sizes": {
            "chemrxiv": {"queries": N_CHEMRXIV_QUERIES, "corpus": N_CHEMRXIV_CORPUS},
            "chempile": {"queries": N_CHEMPILE_QUERIES, "corpus": N_CHEMPILE_CORPUS},
            "anchors_mixed_chem": N_ANCHORS_MIXED_CHEM_TOTAL,
            "anchors_general": N_ANCHORS_GENERAL_TOTAL,
        },
        "generated_at": _now_utc_iso(),
    }
    _write_json(OUTDIR / "manifest.json", shared)

    # Dump per-model
    run_summary: Dict[str, Any] = {"models": {}, **shared}

    for model_name in MODELS:
        model_slug = slugify(model_name)
        base = OUTDIR / model_slug

        # Chempile
        chempile_out = dump_qc_embeddings(
            out_base=base / f"chempile_{VARIANT}",
            model_name=model_name,
            dataset_name=f"chempile_{VARIANT}",
            query_ids=cp_q_ids,
            query_texts=cp_q_txt,
            doc_ids=cp_d_ids,
            doc_texts=cp_d_txt,
            n_queries=N_CHEMPILE_QUERIES,
            n_docs=N_CHEMPILE_CORPUS,
            rng=rng,
        )

        # ChemRxiv
        chemrxiv_out = dump_qc_embeddings(
            out_base=base / "chemrxiv",
            model_name=model_name,
            dataset_name="chemrxiv",
            query_ids=cx_q_ids,
            query_texts=cx_q_txt,
            doc_ids=cx_d_ids,
            doc_texts=cx_d_txt,
            n_queries=N_CHEMRXIV_QUERIES,
            n_docs=N_CHEMRXIV_CORPUS,
            rng=rng,
        )

        # Anchors
        anchors_mixed_out = dump_anchor_embeddings(
            out_base=base / "anchors_mixed_chem",
            model_name=model_name,
            anchor_name="anchors_mixed_chem",
            texts=anchors_mixed_chem,
        )

        anchors_general_out = dump_anchor_embeddings(
            out_base=base / "anchors_general",
            model_name=model_name,
            anchor_name="anchors_general",
            texts=anchors_general,
        )

        run_summary["models"][model_name] = {
            "slug": model_slug,
            "chempile": chempile_out,
            "chemrxiv": chemrxiv_out,
            "anchors_mixed_chem": anchors_mixed_out,
            "anchors_general": anchors_general_out,
        }

    _write_json(OUTDIR / "run_summary.json", run_summary)
    print(f"Wrote geometry bundle to: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
