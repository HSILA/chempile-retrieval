"""Compute and visualize kNN mixing between datasets.

Outputs (written into <run>/figs/):
- knn_mixing_heatmap_<model_slug>_k10.png
- knn_mixing_heatmap_<model_slug>_k50.png
- knn_mixing_<model_slug>.json  (matrix values)

Design: static matplotlib PNGs are paper-friendly; no interactivity needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = REPO_ROOT / "analysis" / "geometry_bundle"

MODEL_SLUGS = [
    "BASF-AI__ChEmbed-vanilla",
    "nomic-ai__nomic-embed-text-v1",
]

DISPLAY_NAME = {
    "BASF-AI__ChEmbed-vanilla": "ChEmbed-vanilla",
    "nomic-ai__nomic-embed-text-v1": "nomic-embed-text-v1",
}

DATASET_ORDER = ["chemrxiv_corpus", "chempile_corpus", "anchors_general"]
LABELS = {"chemrxiv_corpus": 0, "chempile_corpus": 1, "anchors_general": 2}


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def load_sets(model_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chem = np.asarray(np.load(model_dir / "chemrxiv" / "corpus.npy", mmap_mode="r"))
    pile = np.asarray(np.load(model_dir / "chempile_A3" / "corpus.npy", mmap_mode="r"))
    anc = np.asarray(
        np.load(model_dir / "anchors_general" / "embeddings.npy", mmap_mode="r")
    )
    return chem, pile, anc


def mixing_matrix(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Return a 3x3 matrix M where M[i,j] is fraction of neighbors of class i that are class j."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    neigh_y = y[idx]

    M = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        mask = y == i
        if mask.sum() == 0:
            continue
        vals = neigh_y[mask].reshape(-1)
        for j in range(3):
            M[i, j] = float((vals == j).mean())
    return M


def plot_heatmap(M: np.ndarray, title: str, outpath: Path):
    plt.figure(figsize=(5.6, 4.6))
    im = plt.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")

    plt.xticks([0, 1, 2], ["ChemRxiv", "Chempile", "General"], rotation=20, ha="right")
    plt.yticks([0, 1, 2], ["ChemRxiv", "Chempile", "General"])

    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="white" if M[i,j] > 0.55 else "black")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 50])
    ap.add_argument(
        "--artifacts_dir",
        default=None,
        help="Where to write JSON artifacts (defaults to <run>/artifacts)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run)
    figs = run_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    artifacts = Path(args.artifacts_dir) if args.artifacts_dir else (run_dir / "artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    for slug in MODEL_SLUGS:
        md = BUNDLE_DIR / slug
        chem, pile, anc = load_sets(md)
        chem, pile, anc = l2n(chem), l2n(pile), l2n(anc)

        X = np.concatenate([chem, pile, anc], axis=0)
        y = np.concatenate([
            np.zeros(len(chem), dtype=np.int32),
            np.ones(len(pile), dtype=np.int32),
            np.full(len(anc), 2, dtype=np.int32),
        ])

        out = {"model": slug, "sizes": {"chemrxiv": int(len(chem)), "chempile": int(len(pile)), "anchors_general": int(len(anc))}, "k": {}}

        for k in args.k:
            M = mixing_matrix(X, y, k=k)
            out["k"][str(k)] = M.tolist()
            plot_heatmap(
                M,
                title=f"kNN mixing (k={k}) — {DISPLAY_NAME[slug]}",
                outpath=figs / f"knn_mixing_heatmap_{DISPLAY_NAME[slug]}_k{k}.png",
            )

        (artifacts / f"knn_mixing_{DISPLAY_NAME[slug]}.json").write_text(json.dumps(out, indent=2))

    print(f"wrote heatmaps + json into {figs}")


if __name__ == "__main__":
    main()
