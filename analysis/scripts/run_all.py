from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = REPO_ROOT / "results" / "geometry_bundle"
MANIFEST_PATH = BUNDLE_DIR / "manifest.json"
RUN_SUMMARY_PATH = BUNDLE_DIR / "run_summary.json"

# Hardcoded model slugs present in the bundle
MODEL_SLUGS = [
    "BASF-AI__ChEmbed-vanilla",
    "nomic-ai__nomic-embed-text-v1",
]

DATASETS = {
    "chemrxiv": {
        "queries": "chemrxiv/queries.npy",
        "corpus": "chemrxiv/corpus.npy",
    },
    "chempile_A3": {
        "queries": "chempile_A3/queries.npy",
        "corpus": "chempile_A3/corpus.npy",
    },
    "anchors_general": {
        "corpus": "anchors_general/embeddings.npy",
        "texts": "anchors_general/texts.jsonl",
    },
    "anchors_mixed_chem": {
        "corpus": "anchors_mixed_chem/embeddings.npy",
        "texts": "anchors_mixed_chem/texts.jsonl",
    },
}

# kNN params
K_LIST = [10, 50]

# Probe params
PCA_DIMS = 64
PROBE_TEST_SIZE = 0.2
PROBE_SEED = 1337

# Pairwise distance correlation subsample
DIST_CORR_N = 1500


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def load_memmap(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def safe_np(a: np.ndarray, max_rows: int | None = None) -> np.ndarray:
    # Convert memmap to in-memory (optionally capped)
    if max_rows is not None:
        a = a[:max_rows]
    return np.asarray(a)


def cosine_centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    # expects normalized rows
    ma = a.mean(axis=0)
    mb = b.mean(axis=0)
    ma = ma / (np.linalg.norm(ma) + 1e-12)
    mb = mb / (np.linalg.norm(mb) + 1e-12)
    return float(1.0 - np.dot(ma, mb))


def knn_same_dataset_rates(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    # X: normalized
    # Use cosine distance via brute force. Requires RAM; user will run on their machine.
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)
    idx = idx[:, 1:]  # drop self
    neigh_labels = labels[idx]
    same = (neigh_labels == labels[:, None]).mean()
    return float(same)


def dataset_source_probe(X: np.ndarray, y: np.ndarray) -> dict:
    # Balanced train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PROBE_TEST_SIZE, random_state=PROBE_SEED, stratify=y
    )

    # PCA -> logistic regression
    pca = PCA(n_components=min(PCA_DIMS, X_train.shape[1]), random_state=PROBE_SEED)
    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)

    # NOTE: keep args minimal for broad scikit-learn compatibility.
    # Also: avoid setting n_jobs to silence sklearn>=1.8 FutureWarning.
    clf = LogisticRegression(
        max_iter=2000,
    )
    clf.fit(X_train_p, y_train)
    y_pred = clf.predict(X_test_p)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    return {
        "macro_f1": float(macro_f1),
        "pca_dims": int(X_train_p.shape[1]),
        "explained_var_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def length_only_probe(lengths: np.ndarray, y: np.ndarray) -> dict:
    # Simple baseline: logistic regression on 1D length feature
    L = lengths.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        L, y, test_size=PROBE_TEST_SIZE, random_state=PROBE_SEED, stratify=y
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    return {"macro_f1": float(macro_f1)}


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    # Linear CKA with centered Gram matrices; O(n^2) in memory.
    # Assumes X,Y are in-memory arrays.
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    # Center
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic = (Kc * Lc).sum()
    norm = np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return float(hsic / (norm + 1e-12))


def neighbor_overlap(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    # overlap of top-k neighbors under cosine distance (excluding self)
    nn_a = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(Xa)
    nn_b = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(Xb)
    ia = nn_a.kneighbors(Xa, return_distance=False)[:, 1:]
    ib = nn_b.kneighbors(Xb, return_distance=False)[:, 1:]
    overlaps = []
    for ra, rb in zip(ia, ib):
        overlaps.append(len(set(ra.tolist()) & set(rb.tolist())) / k)
    return float(np.mean(overlaps))


def spearman_distance_corr(Xa: np.ndarray, Xb: np.ndarray, n: int, seed: int = 1337) -> float:
    # subsample n points, compute pairwise cosine distances, Spearman correlate
    rng = np.random.default_rng(seed)
    idx = rng.choice(Xa.shape[0], size=min(n, Xa.shape[0]), replace=False)
    A = Xa[idx]
    B = Xb[idx]
    # pairwise cosine distance = 1 - cosine sim
    As = A @ A.T
    Bs = B @ B.T
    da = 1.0 - As[np.triu_indices_from(As, k=1)]
    db = 1.0 - Bs[np.triu_indices_from(Bs, k=1)]
    # Spearman via rank corr
    ra = da.argsort().argsort().astype(np.float64)
    rb = db.argsort().argsort().astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    corr = (ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12)
    return float(corr)


def plot_pca_scatter(X: np.ndarray, labels: np.ndarray, label_names: dict[int, str], outpath: Path, title: str):
    pca = PCA(n_components=2, random_state=PROBE_SEED)
    Z = pca.fit_transform(X)
    plt.figure(figsize=(7, 6))
    for lab in sorted(np.unique(labels)):
        m = labels == lab
        plt.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.35, label=label_names[int(lab)])
    plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180)
    plt.close()


def main():
    assert MANIFEST_PATH.exists(), f"Missing {MANIFEST_PATH}"
    assert RUN_SUMMARY_PATH.exists(), f"Missing {RUN_SUMMARY_PATH}"

    run_id = _utc_run_id()
    outdir = REPO_ROOT / "analysis" / "runs" / run_id
    figs = outdir / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_PATH.read_text())
    run_summary = json.loads(RUN_SUMMARY_PATH.read_text())

    metrics = {
        "run_id": run_id,
        "bundle_manifest": manifest,
        "bundle_run_summary": run_summary,
        "models": {},
    }

    # Load anchor texts (for length-only baseline). We only have texts for anchors.
    # For other sets we don’t have texts in the bundle; length baseline will be "N/A" unless user adds text exports.

    for model_slug in MODEL_SLUGS:
        mp = BUNDLE_DIR / model_slug
        if not mp.exists():
            raise FileNotFoundError(mp)

        model_metrics = {
            "sanity": {},
            "dataset_shift": {},
            "model_space": {},
        }

        # Load corpora embeddings
        chem_corpus = safe_np(load_memmap(mp / "chemrxiv" / "corpus.npy"))
        pile_corpus = safe_np(load_memmap(mp / "chempile_A3" / "corpus.npy"))
        anc_gen = safe_np(load_memmap(mp / "anchors_general" / "embeddings.npy"))

        # Normalize
        chem_corpus_n = l2_normalize(chem_corpus)
        pile_corpus_n = l2_normalize(pile_corpus)
        anc_gen_n = l2_normalize(anc_gen)

        # Sanity
        def norm_stats(x: np.ndarray):
            n = np.linalg.norm(x, axis=1)
            return {
                "mean": float(np.mean(n)),
                "std": float(np.std(n)),
                "p95": float(np.quantile(n, 0.95)),
            }

        model_metrics["sanity"] = {
            "chemrxiv_corpus_shape": list(chem_corpus.shape),
            "chempile_corpus_shape": list(pile_corpus.shape),
            "anchors_general_shape": list(anc_gen.shape),
            "norms_raw": {
                "chemrxiv": norm_stats(chem_corpus),
                "chempile": norm_stats(pile_corpus),
                "anchors_general": norm_stats(anc_gen),
            },
        }

        # Centroid cosine distances
        model_metrics["dataset_shift"]["centroid_cos_dist"] = {
            "chemrxiv_vs_chempile": cosine_centroid_distance(chem_corpus_n, pile_corpus_n),
            "chemrxiv_vs_anchors_general": cosine_centroid_distance(chem_corpus_n, anc_gen_n),
            "chempile_vs_anchors_general": cosine_centroid_distance(pile_corpus_n, anc_gen_n),
        }

        # kNN mixing on pooled corpora
        pooled = np.concatenate([chem_corpus_n, pile_corpus_n, anc_gen_n], axis=0)
        labels = np.concatenate([
            np.zeros(len(chem_corpus_n), dtype=np.int32),
            np.ones(len(pile_corpus_n), dtype=np.int32),
            np.full(len(anc_gen_n), 2, dtype=np.int32),
        ])
        label_names = {0: "chemrxiv_corpus", 1: "chempile_corpus", 2: "anchors_general"}

        shift_knn = {}
        for k in K_LIST:
            shift_knn[f"k={k}"] = knn_same_dataset_rates(pooled, labels, k=k)
        model_metrics["dataset_shift"]["knn_same_dataset_rate"] = shift_knn

        # Dataset-source probe (embeddings)
        # Balance by downsampling to smallest class
        n0, n1, n2 = (labels == 0).sum(), (labels == 1).sum(), (labels == 2).sum()
        nmin = int(min(n0, n1, n2))
        rng = np.random.default_rng(PROBE_SEED)
        idx0 = rng.choice(np.where(labels == 0)[0], size=nmin, replace=False)
        idx1 = rng.choice(np.where(labels == 1)[0], size=nmin, replace=False)
        idx2 = rng.choice(np.where(labels == 2)[0], size=nmin, replace=False)
        idx = np.concatenate([idx0, idx1, idx2])
        X_probe = pooled[idx]
        y_probe = labels[idx]
        model_metrics["dataset_shift"]["source_probe"] = dataset_source_probe(X_probe, y_probe)

        # Length-only baseline: only available for anchors_general (texts.jsonl). Mark N/A.
        model_metrics["dataset_shift"]["source_probe_length_only"] = {
            "note": "N/A (bundle does not include raw texts for chemrxiv/chempile corpora; only anchors have texts.jsonl)."
        }

        # PCA scatter figure
        plot_pca_scatter(
            X_probe,
            y_probe,
            label_names,
            outpath=figs / f"pca_scatter_{model_slug}.png",
            title=f"PCA scatter (balanced) — {model_slug}",
        )

        metrics["models"][model_slug] = model_metrics

    # Model-space comparisons between the two models on the same datasets
    # NOTE: We can only do this reliably if counts match. For chemrxiv corpus they do (10k). For chempile they should match (3185).
    # Use full arrays (user will run on their machine). These operations are heavy.

    ms = {}
    mA, mB = MODEL_SLUGS
    mpA, mpB = BUNDLE_DIR / mA, BUNDLE_DIR / mB

    for ds_name, rel in [("chemrxiv_corpus", ("chemrxiv/corpus.npy",)), ("chempile_corpus", ("chempile_A3/corpus.npy",))]:
        XA = l2_normalize(safe_np(load_memmap(mpA / rel[0])))
        XB = l2_normalize(safe_np(load_memmap(mpB / rel[0])))
        assert XA.shape == XB.shape

        # CKA (very heavy due to NxN). Guardrail: if N too big, skip.
        cka = None
        if XA.shape[0] <= 4000:
            cka = linear_cka(XA, XB)
        else:
            cka = None

        ms[ds_name] = {
            "shape": list(XA.shape),
            "linear_cka": cka,
            "neighbor_overlap": {f"k={k}": neighbor_overlap(XA, XB, k=k) for k in K_LIST},
            "spearman_distance_corr": spearman_distance_corr(XA, XB, n=DIST_CORR_N, seed=PROBE_SEED),
            "notes": "linear_cka computed only if N<=4000 (otherwise skipped due to O(N^2) memory).",
        }

    metrics["model_space"] = ms

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Simple markdown report
    lines = []
    lines.append(f"# Geometry analysis report\n")
    lines.append(f"Run id: `{run_id}`\n")
    lines.append("## Dataset shift (per model)\n")
    for model_slug, mm in metrics["models"].items():
        lines.append(f"### {model_slug}\n")
        lines.append("**Centroid cosine distances**\n")
        for k, v in mm["dataset_shift"]["centroid_cos_dist"].items():
            lines.append(f"- {k}: {v:.6f}")
        lines.append("\n**kNN same-dataset neighbor rate (pooled corpora)**\n")
        for k, v in mm["dataset_shift"]["knn_same_dataset_rate"].items():
            lines.append(f"- {k}: {v:.4f}")
        sp = mm["dataset_shift"]["source_probe"]
        lines.append("\n**Dataset-source probe (PCA→logreg; macro-F1)**\n")
        lines.append(f"- macro_f1: {sp['macro_f1']:.4f}")
        lines.append(f"- pca_dims: {sp['pca_dims']} (explained_var_sum={sp['explained_var_ratio_sum']:.3f})\n")
        lines.append(f"Figure: `figs/pca_scatter_{model_slug}.png`\n")

    lines.append("## Model-space comparisons (ChEmbed vs nomic)\n")
    for ds_name, info in metrics["model_space"].items():
        lines.append(f"### {ds_name}\n")
        lines.append(f"- shape: {info['shape']}")
        lines.append(f"- linear_cka: {info['linear_cka']}")
        lines.append(f"- spearman_distance_corr (n={DIST_CORR_N}): {info['spearman_distance_corr']:.4f}")
        lines.append("- neighbor_overlap:")
        for k, v in info["neighbor_overlap"].items():
            lines.append(f"  - {k}: {v:.4f}")
        lines.append(f"- notes: {info['notes']}\n")

    (outdir / "report.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote: {outdir}")


if __name__ == "__main__":
    main()
