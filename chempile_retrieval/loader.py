from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_variant_from_local_files(base_dir: Path, variant: str) -> tuple[Dataset, Dataset, dict[str, dict[str, int]]]:
    """Load a variant (A1..C3) from ./data/{variant}/ JSONL files.

    Returns:
        queries_ds: Dataset with columns [id, text]
        corpus_ds:  Dataset with columns [id, text]
        qrels: mapping query_id -> {corpus_id: score}
    """

    vdir = base_dir / "data" / variant
    queries_path = vdir / "queries.jsonl"
    corpus_path = vdir / "corpus.jsonl"
    qrels_path = vdir / "default.jsonl"

    queries_raw = _read_jsonl(queries_path)
    corpus_raw = _read_jsonl(corpus_path)
    qrels_raw = _read_jsonl(qrels_path)

    # Normalize to MTEB v2 expected columns
    queries = [{"id": q["_id"], "text": q["text"]} for q in queries_raw]
    corpus = [{"id": d["_id"], "text": d["text"], "title": ""} for d in corpus_raw]

    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_raw:
        qid = r["query-id"]
        cid = r["corpus-id"]
        score = int(r["score"])
        qrels.setdefault(qid, {})[cid] = score

    return Dataset.from_list(queries), Dataset.from_list(corpus), qrels
