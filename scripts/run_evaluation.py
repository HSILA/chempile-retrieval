#!/usr/bin/env python3
"""Run MTEB evaluation for ChempileRetrieval A1..C3.

Local-first: loads datasets from ./data/* JSONLs via chempile_retrieval.tasks.

Example:
    python scripts/run_evaluation.py --model nomic-ai/nomic-embed-text-v1

Notes:
- This script assumes mteb + sentence-transformers are installed in the active env.
- Results will be written under ./results/<model_name>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mteb import MTEB

import mteb
from chempile_retrieval.model_wrappers import ChEmbedWrapper
from chempile_retrieval.tasks import (
    ChempileRetrievalA1,
    ChempileRetrievalA2,
    ChempileRetrievalA3,
    ChempileRetrievalB1,
    ChempileRetrievalB2,
    ChempileRetrievalB3,
    ChempileRetrievalC1,
    ChempileRetrievalC2,
    ChempileRetrievalC3,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name")
    ap.add_argument("--revision", default=None, help="Optional HF revision/tag/commit")
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="If supported by the underlying model, set max_seq_length (default: 2048)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--tasks",
        default="A1,A2,A3,B1,B2,B3,C1,C2,C3",
        help="Comma-separated list of variants to run (e.g. A1,B3,C3). Default: all 9.",
    )
    # NOTE: This project always enables trust_remote_code for model loading.
    # ChEmbed + Nomic require it, and we want one consistent behavior.
    args = ap.parse_args()

    trust_remote_code = True

    task_map = {
        "A1": ChempileRetrievalA1,
        "A2": ChempileRetrievalA2,
        "A3": ChempileRetrievalA3,
        "B1": ChempileRetrievalB1,
        "B2": ChempileRetrievalB2,
        "B3": ChempileRetrievalB3,
        "C1": ChempileRetrievalC1,
        "C2": ChempileRetrievalC2,
        "C3": ChempileRetrievalC3,
    }

    selected = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in selected if t not in task_map]
    if unknown:
        raise SystemExit(f"Unknown tasks: {unknown}. Valid: {sorted(task_map.keys())}")

    tasks = [task_map[t]() for t in selected]

    # Mirror ChEmbed-Res behavior:
    # - ChEmbed models use our ChEmbedWrapper (prompt injection + optional ChemVocab tokenizer).
    # - Everything else (including nomic) uses mteb.get_model(...).
    model_name = args.model

    if "BASF-AI/ChEmbed" in model_name:
        model = ChEmbedWrapper(
            model_name,
            revision=args.revision,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = mteb.get_model(
            model_name,
            revision=args.revision,
            trust_remote_code=trust_remote_code,
        )

    inner = getattr(model, "model", model)
    if hasattr(inner, "max_seq_length"):
        inner.max_seq_length = args.max_seq_length

    out_dir = Path(__file__).resolve().parents[1] / "results" / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = MTEB(tasks=tasks)
    runner.run(
        model,
        output_folder=str(out_dir),
        encode_kwargs={"batch_size": args.batch_size},
    )


if __name__ == "__main__":
    main()
