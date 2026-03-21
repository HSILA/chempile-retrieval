#!/usr/bin/env python3
"""Generate Chempile retrieval datasets (9 variants).

Usage:
    python scripts/generate_datasets.py

Notes:
- We build 1:1 query↔corpus pairs.
- IDs must be stable and unique even when the *query text* (e.g., title-only)
  is non-unique across rows.
- We pre-process at the domain level (A/B/C) before generating modes 1/2/3.
"""

import json
import uuid
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset

# Configuration
SOURCE_DATASET = "jablonkagroup/chempile-reasoning"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

# Domain variants
DOMAIN_CONFIGS = {
    "A": ["chemistry_stackexchange-raw_data"],
    "B": [
        "chemistry_stackexchange-raw_data",
        "mattermodeling_stackexchange-raw_data",
    ],
    "C": [
        "chemistry_stackexchange-raw_data",
        "mattermodeling_stackexchange-raw_data",
        "physics_stackexchange-raw_data",
    ],
}


def get_query(row, mode: str) -> str:
    """Extract query text based on mode."""
    if mode == "1":  # title only
        return row["title"]
    if mode == "2":  # question only
        return row["q"]
    if mode == "3":  # title + question
        return f"{row['title']} {row['q']}"
    raise ValueError(f"Unknown mode: {mode}")


def stable_uuid_hex(text: str) -> str:
    """Deterministic, effectively collision-free ID.

    uuid5 is deterministic for a given (namespace, text) and avoids the
    short-hash collision risk of truncating digests.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, text).hex


def load_domain_data(configs):
    """Load data from one or more domain configs into a list of dict rows."""
    all_rows = []
    for config in configs:
        print(f"  Loading {config}...")
        ds = load_dataset(
            SOURCE_DATASET,
            config,
            split="train",
        )
        all_rows.extend(ds)
    return all_rows


def preprocess_rows(rows):
    """Preprocess rows *once* per domain variant before building modes.

    Goals:
    - Ensure 1:1 mapping feasibility.
    - Remove exact duplicates.
    - Resolve duplicated questions (title+q identical) that may have different
      answers by keeping a single canonical row.

    Returns:
        unique_rows: list[dict]
        stats: dict with counts
    """
    # 1) Exact de-dup (title,q,a) to remove perfect repeats.
    seen_exact = set()
    exact_unique = []
    for r in rows:
        k = f"{r['title']}|||{r['q']}|||{r['a']}"
        if k in seen_exact:
            continue
        seen_exact.add(k)
        exact_unique.append(r)

    # 2) Question-level de-dup (title,q). If multiple answers exist for same
    # question, we keep the first encountered (stable as long as input order is
    # stable) and drop the rest to preserve 1:1 mapping.
    by_question = {}
    multi_answer_questions = 0

    for r in exact_unique:
        qk = f"{r['title']}|||{r['q']}"
        if qk not in by_question:
            by_question[qk] = r
        else:
            # We saw same (title,q) again; could be different answer.
            if r["a"] != by_question[qk]["a"]:
                multi_answer_questions += 1
            # Keep existing canonical row.

    unique_rows = list(by_question.values())

    stats = {
        "loaded": len(rows),
        "after_exact_dedup": len(exact_unique),
        "after_question_dedup": len(unique_rows),
        "multi_answer_question_collisions": multi_answer_questions,
    }

    # Deterministic ordering: sort by stable base key.
    unique_rows.sort(key=lambda r: stable_uuid_hex(f"{r['title']}|||{r['q']}"))
    return unique_rows, stats


def generate_variant(domain: str, mode: str):
    """Generate one variant (e.g., A1, B2, C3)."""
    print(f"\n{'=' * 60}")
    print(f"Generating variant {domain}{mode}")
    print(f"{'=' * 60}")

    # Load domain data
    configs = DOMAIN_CONFIGS[domain]
    rows = load_domain_data(configs)

    # Pre-process once (dedup + 1:1 enforcement)
    rows, stats = preprocess_rows(rows)

    print(f"  Total rows loaded: {stats['loaded']}")
    print(f"  After exact dedup (title,q,a): {stats['after_exact_dedup']}")
    print(f"  After question dedup (title,q): {stats['after_question_dedup']}")
    print(
        "  Multi-answer collisions (title,q with differing a): "
        f"{stats['multi_answer_question_collisions']}"
    )

    corpus = []
    queries = []
    qrels = []

    for row in rows:
        query_text = get_query(row, mode)

        # Base key ties together the 1:1 pair; do NOT derive IDs from query_text
        # or answer text directly, or we can get duplicate IDs in title-only mode
        # and/or when answers repeat.
        base_key = f"{row['title']}|||{row['q']}"
        base_id = stable_uuid_hex(base_key)

        query_id = f"q_{base_id}"
        corpus_id = f"c_{base_id}"

        corpus.append({"_id": corpus_id, "text": row["a"]})
        queries.append({"_id": query_id, "text": query_text})
        qrels.append({"query-id": query_id, "corpus-id": corpus_id, "score": 1})

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Qrels: {len(qrels)} pairs")

    return corpus, queries, qrels


def save_variant(variant_name, corpus, queries, qrels):
    """Save variant to JSONL files."""
    output_dir = OUTPUT_DIR / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save corpus
    corpus_file = output_dir / "corpus.jsonl"
    with open(corpus_file, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")
    print(f"  Saved: {corpus_file}")

    # Save queries
    queries_file = output_dir / "queries.jsonl"
    with open(queries_file, "w") as f:
        for query in queries:
            f.write(json.dumps(query) + "\n")
    print(f"  Saved: {queries_file}")

    # Save qrels (default split)
    qrels_file = output_dir / "default.jsonl"
    with open(qrels_file, "w") as f:
        for qrel in qrels:
            f.write(json.dumps(qrel) + "\n")
    print(f"  Saved: {qrels_file}")


def main():
    print("Chempile Retrieval Dataset Generator")
    print("=" * 60)

    for domain in ["A", "B", "C"]:
        for mode in ["1", "2", "3"]:
            variant_name = f"{domain}{mode}"
            corpus, queries, qrels = generate_variant(domain, mode)
            save_variant(variant_name, corpus, queries, qrels)

    print(f"\n{'=' * 60}")
    print("All 9 variants generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
