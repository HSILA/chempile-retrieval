#!/usr/bin/env python3
"""Generate a compact markdown summary of ChempileRetrieval results.

Scans results/<model>/<model>/<revision>/<task>.json (or any deeper paths)
and produces a per-task table across models.

Metrics are pulled from the MTEB json "scores.test[0]" dict.

Usage:
  python scripts/generate_results_markdown.py \
    --results-dir results \
    --out results_summary.md

Notes:
- HitRate@k is reported as recall_at_k (for 1:1 matching tasks, this equals
  success/hitrate at k).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TASKS: List[str] = [
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

# Metrics to include in the markdown table.
# Keys are the json fields, values are display names.
METRICS: List[Tuple[str, str]] = [
    ("ndcg_at_10", "NDCG@10"),
    ("recall_at_10", "HitRate@10"),
    ("mrr_at_10", "MRR@10"),
    # Extra metric useful for 1:1 matching (success@1 / accuracy@1).
    ("recall_at_1", "HitRate@1"),
]


@dataclass(frozen=True)
class ModelResult:
    model: str
    task: str
    path: Path
    metrics: Dict[str, Optional[float]]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _read_task_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metrics(doc: Dict[str, Any]) -> Dict[str, Optional[float]]:
    scores = doc.get("scores", {})
    test = scores.get("test")
    if not isinstance(test, list) or not test:
        return {k: None for k, _ in METRICS}

    row = test[0]
    if not isinstance(row, dict):
        return {k: None for k, _ in METRICS}

    out: Dict[str, Optional[float]] = {}
    for key, _label in METRICS:
        out[key] = _safe_float(row.get(key))
    return out


def _find_latest_task_file(model_dir: Path, task: str) -> Optional[Path]:
    # Look for the task json anywhere under model_dir.
    candidates: List[Path] = list(model_dir.rglob(f"{task}.json"))
    if not candidates:
        return None

    # Prefer paths containing a revision dir (hash or no_revision_available),
    # otherwise just pick the newest mtime.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _discover_models(results_dir: Path) -> List[Tuple[str, Path]]:
    # Expected layout: results/<model>/<model>/...
    models: List[Tuple[str, Path]] = []
    if not results_dir.exists():
        return models

    for model in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        # Some repos use results/<model>/<model>; if missing, fall back to <model>.
        nested = model / model.name
        models.append((model.name, nested if nested.is_dir() else model))

    return models


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    # Keep 4 decimals; most of these are in [0,1]
    return f"{v:.4f}"


def render_markdown(
    results_dir: Path,
    tasks: Iterable[str],
) -> str:
    models = _discover_models(results_dir)

    lines: List[str] = []
    lines.append("# Chempile Retrieval – Results Summary")
    lines.append("")
    lines.append(f"Generated from: `{results_dir}`")
    lines.append("")
    lines.append("Metrics")
    lines.append("- NDCG@10")
    lines.append("- HitRate@10 (recall@10; for 1:1 matching this is success@10)")
    lines.append("- MRR@10")
    lines.append("- HitRate@1 (recall@1; success@1)")
    lines.append("")

    if not models:
        lines.append("No model directories found.")
        lines.append("")
        return "\n".join(lines)

    model_names = [m for m, _ in models]

    for task in tasks:
        lines.append(f"## {task}")
        lines.append("")

        header = ["Model"] + [label for _key, label in METRICS]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for model_name, model_path in models:
            task_file = _find_latest_task_file(model_path, task)
            if task_file is None:
                row_vals = ["—" for _k, _lbl in METRICS]
            else:
                doc = _read_task_json(task_file)
                m = _extract_metrics(doc)
                row_vals = [_fmt(m.get(k)) for k, _lbl in METRICS]

            lines.append("| " + " | ".join([f"`{model_name}`"] + row_vals) + " |")

        lines.append("")

    # Quick index at the bottom (handy for GitHub scrolling)
    lines.append("---")
    lines.append("")
    lines.append("## Tasks")
    for t in tasks:
        lines.append(f"- [{t}](#{t.lower()})")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results_summary.md"))
    args = ap.parse_args()

    md = render_markdown(args.results_dir, TASKS)
    args.out.write_text(md, encoding="utf-8")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
