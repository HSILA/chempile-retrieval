"""Make interactive PCA(3D) plots using plotly and export to HTML.

This script does NOT hand-write HTML.

Usage (absolute paths):
  /abs/path/to/.venv/bin/python /abs/path/to/analysis/scripts/make_interactive_pca_plotly.py \
    --run /abs/path/to/analysis/runs/<run_id> \
    --subsample 1500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


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

COLORS = {
    "chemrxiv": "#1f77b4",
    "chempile": "#ff7f0e",
    "anchors_general": "#2ca02c",
}

CENTROID_SYMBOL = {
    "chemrxiv": "diamond",
    "chempile": "x",
    "anchors_general": "square",
}


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def pca3_fit_transform(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :3] * S[:3]
    V = Vt[:3].T
    return Z, mu.squeeze(0), V


def load_arrays(model_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chem = np.asarray(np.load(model_dir / "chemrxiv" / "corpus.npy", mmap_mode="r"))
    pile = np.asarray(np.load(model_dir / "chempile_A3" / "corpus.npy", mmap_mode="r"))
    anc = np.asarray(
        np.load(model_dir / "anchors_general" / "embeddings.npy", mmap_mode="r")
    )
    return chem, pile, anc


def build_figure(Z: np.ndarray, labels: list[str], centroids_z: dict[str, np.ndarray], title: str) -> go.Figure:
    fig = go.Figure()

    for which in ("chemrxiv", "chempile", "anchors_general"):
        idx = [i for i, lab in enumerate(labels) if lab == which]
        fig.add_trace(
            go.Scatter3d(
                x=Z[idx, 0],
                y=Z[idx, 1],
                z=Z[idx, 2],
                mode="markers",
                name=which,
                marker=dict(size=2, opacity=0.30, color=COLORS[which]),
            )
        )

    for which in ("chemrxiv", "chempile", "anchors_general"):
        cz = centroids_z[which]
        fig.add_trace(
            go.Scatter3d(
                x=[cz[0]],
                y=[cz[1]],
                z=[cz[2]],
                mode="markers+text",
                name=f"{which}_centroid",
                text=[f"{which} centroid"],
                textposition="top center",
                marker=dict(
                    size=8,
                    opacity=1.0,
                    color=COLORS[which],
                    symbol=CENTROID_SYMBOL[which],
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        legend=dict(orientation="h"),
    )

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Output folder (e.g., analysis)")
    ap.add_argument("--subsample", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    run_dir = Path(args.run)
    figs_dir = run_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for slug in MODEL_SLUGS:
        md = BUNDLE_DIR / slug
        chem, pile, anc = load_arrays(md)

        chem = l2n(chem)
        pile = l2n(pile)
        anc = l2n(anc)

        def sub(x: np.ndarray) -> np.ndarray:
            # subsample<=0 means "use all points"
            if args.subsample <= 0 or args.subsample >= len(x):
                return x
            idx = rng.choice(len(x), size=args.subsample, replace=False)
            return x[idx]

        chem_s = sub(chem)
        pile_s = sub(pile)
        anc_s = sub(anc)

        X = np.concatenate([chem_s, pile_s, anc_s], axis=0)
        labels = (["chemrxiv"] * len(chem_s)) + (["chempile"] * len(pile_s)) + (["anchors_general"] * len(anc_s))

        Z, mu, V = pca3_fit_transform(X)

        # Project FULL centroids
        centroids = {
            "chemrxiv": chem.mean(axis=0),
            "chempile": pile.mean(axis=0),
            "anchors_general": anc.mean(axis=0),
        }
        centroids_z = {k: (centroids[k] - mu) @ V for k in centroids}

        name = DISPLAY_NAME[slug]
        fig = build_figure(Z, labels, centroids_z, title=f"{name} — PCA3 interactive")

        out_html = figs_dir / f"pca3_plotly_{name}.html"
        # Include plotly JS via CDN (portable). No hand-written HTML.
        fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
        print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
