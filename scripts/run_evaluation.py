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
from pathlib import Path

from mteb import MTEB
from mteb.models import SentenceTransformerEncoderWrapper

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
    ap.add_argument("--model", required=True, help="HF model name (SentenceTransformer compatible)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    tasks = [
        ChempileRetrievalA1(),
        ChempileRetrievalA2(),
        ChempileRetrievalA3(),
        ChempileRetrievalB1(),
        ChempileRetrievalB2(),
        ChempileRetrievalB3(),
        ChempileRetrievalC1(),
        ChempileRetrievalC2(),
        ChempileRetrievalC3(),
    ]

    model = SentenceTransformerEncoderWrapper(args.model, batch_size=args.batch_size)

    out_dir = Path(__file__).resolve().parents[1] / "results" / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    mteb = MTEB(tasks=tasks)
    mteb.run(model, output_folder=str(out_dir))


if __name__ == "__main__":
    main()
