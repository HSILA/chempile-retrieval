"""Chempile Retrieval (Reasoning-as-Retrieval) task package.

This repo is set up to be *local-first*:
- Datasets live under ./data/{A1..C3}/ as JSONL files.
- Task classes load from local files (no HF Hub required for Stage 2).

See SPEC.md (internal, not tracked) for the design notes.
"""

from .tasks import (
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

__all__ = [
    "ChempileRetrievalA1",
    "ChempileRetrievalA2",
    "ChempileRetrievalA3",
    "ChempileRetrievalB1",
    "ChempileRetrievalB2",
    "ChempileRetrievalB3",
    "ChempileRetrievalC1",
    "ChempileRetrievalC2",
    "ChempileRetrievalC3",
]
