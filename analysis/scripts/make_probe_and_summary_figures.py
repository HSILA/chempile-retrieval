"""Make simple paper-friendly figures for neighbor-mixing summaries.

Outputs into <run>/figs/:
- knn_same_rate_bar.png

This is intentionally static (matplotlib) for paper inclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run)
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists(), f"Missing {metrics_path}"

    metrics = json.loads(metrics_path.read_text())

    models = [
        "BASF-AI__ChEmbed-vanilla",
        "nomic-ai__nomic-embed-text-v1",
    ]

    # kNN same-dataset rates
    k10 = [metrics["models"][m]["dataset_shift"]["knn_same_dataset_rate"]["k=10"] for m in models]
    k50 = [metrics["models"][m]["dataset_shift"]["knn_same_dataset_rate"]["k=50"] for m in models]

    x = [0, 1]
    w = 0.35

    plt.figure(figsize=(6.8, 3.8))
    plt.bar([i - w/2 for i in x], k10, width=w, label="k=10", color="#2ca02c", alpha=0.85)
    plt.bar([i + w/2 for i in x], k50, width=w, label="k=50", color="#9467bd", alpha=0.85)
    plt.xticks(x, ["ChEmbed", "nomic"])
    plt.ylim(0.0, 1.0)
    for i, v in enumerate(k10):
        plt.text(i - w/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(k50):
        plt.text(i + w/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.title("Local clustering by dataset (kNN same-dataset neighbor rate)")
    plt.ylabel("fraction")
    plt.legend(frameon=False)
    plt.tight_layout()
    out2 = run_dir / "figs" / "knn_same_rate_bar.png"
    plt.savefig(out2, dpi=220)
    plt.close()

    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
