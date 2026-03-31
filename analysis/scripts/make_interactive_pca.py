"""Generate interactive PCA(3D) plots from the geometry bundle.

Outputs are written into the given run folder under:
  <run>/figs/pca3_offline_<model_slug>.html

Design choices (hard-coded for this repo):
- Use Plotly via PUBLIC CDN so the HTML works for any user.
- Use a modest per-dataset subsample for browser performance.
- Overlay centroids with distinct marker symbols.

Usage:
  /abs/path/to/.venv/bin/python /abs/path/to/analysis/scripts/make_interactive_pca.py \
    --run /abs/path/to/analysis/runs/<run_id> \
    --subsample 1500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = REPO_ROOT / "results" / "geometry_bundle"

MODEL_SLUGS = [
    "BASF-AI__ChEmbed-vanilla",
    "nomic-ai__nomic-embed-text-v1",
]

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
    """Return (Z, mean, V) where Z is (n,3) and V is (d,3) basis."""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :3] * S[:3]
    V = Vt[:3].T
    return Z, mu.squeeze(0), V


def load_arrays(model_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chem = np.asarray(np.load(model_dir / "chemrxiv" / "corpus.npy", mmap_mode="r"))
    pile = np.asarray(
        np.load(model_dir / "chempile_A3" / "corpus.npy", mmap_mode="r")
    )
    anc = np.asarray(
        np.load(model_dir / "anchors_general" / "embeddings.npy", mmap_mode="r")
    )
    return chem, pile, anc


def write_model_html(
    out_html: Path,
    title: str,
    Z: np.ndarray,
    labels: list[str],
    centroids_z: dict[str, np.ndarray],
):
    # Split points
    idx = {
        k: [i for i, lab in enumerate(labels) if lab == k]
        for k in ("chemrxiv", "chempile", "anchors_general")
    }

    def coords(which: str):
        ii = idx[which]
        return (
            [float(Z[i, 0]) for i in ii],
            [float(Z[i, 1]) for i in ii],
            [float(Z[i, 2]) for i in ii],
        )

    traces = []
    for which in ("chemrxiv", "chempile", "anchors_general"):
        x, y, z = coords(which)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": which,
                "x": x,
                "y": y,
                "z": z,
                "marker": {"size": 2, "opacity": 0.30, "color": COLORS[which]},
            }
        )

    # centroid overlays
    for which in ("chemrxiv", "chempile", "anchors_general"):
        cz = centroids_z[which]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": f"{which}_centroid",
                "x": [float(cz[0])],
                "y": [float(cz[1])],
                "z": [float(cz[2])],
                "text": [f"{which} centroid"],
                "textposition": "top center",
                "marker": {
                    "size": 9,
                    "opacity": 1.0,
                    "color": COLORS[which],
                    "symbol": CENTROID_SYMBOL[which],
                },
                "showlegend": False,
            }
        )

    html = []
    html.append("<html><head><meta charset='utf-8' />")
    html.append(f"<title>{title}</title>")
    html.append(
        "<style>body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px;} </style>"
    )
    html.append("</head><body>")
    html.append(f"<h2>{title}</h2>")
    html.append(
        "<p>Interactive PCA(3D) scatter of corpus embeddings. Centroids are overlaid with distinct symbols (diamond/x/square). This HTML uses the public Plotly CDN.</p>"
    )

    html.append("<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>")

    html.append("<div id='plot' style='width: 1100px; height: 750px;'></div>")
    html.append("<script>")
    html.append(f"const data = {traces};")
    html.append(
        "Plotly.newPlot('plot', data, {margin:{l:0,r:0,b:0,t:40}, scene:{xaxis:{title:'PC1'}, yaxis:{title:'PC2'}, zaxis:{title:'PC3'}}, legend:{orientation:'h'}});"
    )
    html.append("</script>")

    html.append("</body></html>")

    out_html.write_text("\n".join(html), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder like analysis/runs/<run_id>")
    ap.add_argument(
        "--subsample",
        type=int,
        default=1500,
        help="Subsample this many points per dataset for browser interactivity.",
    )
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    run_dir = Path(args.run)
    figs = run_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for slug in MODEL_SLUGS:
        md = BUNDLE_DIR / slug
        chem, pile, anc = load_arrays(md)

        # normalize (they’re already unit-norm but keep it explicit)
        chem = l2n(chem)
        pile = l2n(pile)
        anc = l2n(anc)

        def sub(x: np.ndarray) -> np.ndarray:
            n = min(len(x), args.subsample)
            if n < len(x):
                idx = rng.choice(len(x), size=n, replace=False)
                return x[idx]
            return x

        chem_s = sub(chem)
        pile_s = sub(pile)
        anc_s = sub(anc)

        X = np.concatenate([chem_s, pile_s, anc_s], axis=0)
        labels = (["chemrxiv"] * len(chem_s)) + (["chempile"] * len(pile_s)) + (
            ["anchors_general"] * len(anc_s)
        )

        Z, mu, V = pca3_fit_transform(X)

        # project full centroids into PCA basis
        centroids = {
            "chemrxiv": chem.mean(axis=0),
            "chempile": pile.mean(axis=0),
            "anchors_general": anc.mean(axis=0),
        }
        centroids_z = {k: (centroids[k] - mu) @ V for k in centroids}

        out_html = figs / f"pca3_offline_{slug}.html"
        write_model_html(
            out_html,
            title=f"{slug} — PCA3 interactive",
            Z=Z,
            labels=labels,
            centroids_z=centroids_z,
        )
        print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
