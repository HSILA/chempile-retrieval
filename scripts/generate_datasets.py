#!/usr/bin/env python3
"""
Generate Chempile retrieval datasets (9 variants).

Usage:
    python scripts/generate_datasets.py
"""

import json
import hashlib
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

# Configuration
SOURCE_DATASET = "jablonkagroup/chempile-reasoning"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

# Domain variants
DOMAIN_CONFIGS = {
    "A": ["chemistry_stackexchange-raw_data"],
    "B": ["chemistry_stackexchange-raw_data", "mattermodeling_stackexchange-raw_data"],
    "C": ["chemistry_stackexchange-raw_data", "mattermodeling_stackexchange-raw_data", "physics_stackexchange-raw_data"],
}

# Query modes
def get_query(row, mode):
    """Extract query based on mode."""
    if mode == "1":  # title only
        return row["title"]
    elif mode == "2":  # question only
        return row["q"]
    elif mode == "3":  # title + question
        return f"{row['title']} {row['q']}"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def generate_id(text):
    """Generate stable ID from text."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def load_domain_data(configs):
    """Load data from multiple domain configs."""
    all_rows = []
    
    for config in configs:
        print(f"  Loading {config}...")
        ds = load_dataset(SOURCE_DATASET, config, split="train", trust_remote_code=True)
        all_rows.extend(ds)
    
    return all_rows


def deduplicate_questions(rows):
    """Deduplicate questions by exact match on title + q."""
    seen = set()
    unique_rows = []
    
    for row in rows:
        key = f"{row['title']}|||{row['q']}"
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    
    return unique_rows


def generate_variant(domain, mode):
    """Generate one variant (e.g., A1, B2, C3)."""
    print(f"\n{'='*60}")
    print(f"Generating variant {domain}{mode}")
    print(f"{'='*60}")
    
    # Load domain data
    configs = DOMAIN_CONFIGS[domain]
    rows = load_domain_data(configs)
    print(f"  Total rows loaded: {len(rows)}")
    
    # Deduplicate questions
    rows = deduplicate_questions(rows)
    print(f"  After dedup: {len(rows)}")
    
    # Generate corpus, queries, qrels
    corpus = []
    queries = []
    qrels = []
    
    for row in rows:
        # Generate IDs
        query_text = get_query(row, mode)
        query_id = generate_id(query_text)
        corpus_id = generate_id(row['a'])
        
        # Corpus document
        corpus.append({
            "_id": corpus_id,
            "text": row['a']
        })
        
        # Query
        queries.append({
            "_id": query_id,
            "text": query_text
        })
        
        # Qrel (1:1 mapping)
        qrels.append({
            "query-id": query_id,
            "corpus-id": corpus_id,
            "score": 1
        })
    
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
    with open(corpus_file, 'w') as f:
        for doc in corpus:
            f.write(json.dumps(doc) + '\n')
    print(f"  Saved: {corpus_file}")
    
    # Save queries
    queries_file = output_dir / "queries.jsonl"
    with open(queries_file, 'w') as f:
        for query in queries:
            f.write(json.dumps(query) + '\n')
    print(f"  Saved: {queries_file}")
    
    # Save qrels (default split)
    qrels_file = output_dir / "default.jsonl"
    with open(qrels_file, 'w') as f:
        for qrel in qrels:
            f.write(json.dumps(qrel) + '\n')
    print(f"  Saved: {qrels_file}")


def main():
    """Generate all 9 variants."""
    print("Chempile Retrieval Dataset Generator")
    print("="*60)
    
    # Generate all variants
    for domain in ["A", "B", "C"]:
        for mode in ["1", "2", "3"]:
            variant_name = f"{domain}{mode}"
            
            # Generate
            corpus, queries, qrels = generate_variant(domain, mode)
            
            # Save
            save_variant(variant_name, corpus, queries, qrels)
    
    print(f"\n{'='*60}")
    print("All 9 variants generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
