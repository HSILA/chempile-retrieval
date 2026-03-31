# Analysis (Geometry bundle)

This folder contains **contained analysis code** to generate geometry metrics + figures from the embedding bundle under:

- `analysis/geometry_bundle/`

## What this produces
Outputs are written to:

- `analysis/`
  - `report.md` (main report)
  - `artifacts/metrics.json` (machine-readable)
  - `figs/` (PNGs + HTML interactives)
  - `artifacts/` (JSON artifacts)

## Quick start
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) core metrics + 2D PCA
python analysis/scripts/run_all.py

# (optional) if you need to (re)generate the geometry bundle first:
#   python analysis/scripts/dump_embeddings.py
#   python analysis/scripts/collect_geometry_bundle.py

# 2) kNN mixing heatmaps (+ JSON matrices into analysis/artifacts/)
python analysis/scripts/make_knn_mixing_figures.py --run analysis

# 3) neighbor-mixing summary bar
python analysis/scripts/make_probe_and_summary_figures.py --run analysis

# 4) interactive PCA3 (Plotly)
python analysis/scripts/make_interactive_pca_plotly.py --run analysis --subsample 0
```

After it finishes, commit `analysis/` outputs (report.md, artifacts/metrics.json, figs/, artifacts/).

## Scripts (what to run when)
- `analysis/scripts/run_all.py`
  - Writes: `analysis/artifacts/metrics.json`
  - Figures: `analysis/figs/pca_scatter_<model>.png`
- `analysis/scripts/make_knn_mixing_figures.py`
  - Figures: `analysis/figs/knn_mixing_heatmap_<model>_k10.png` and `_k50.png`
  - JSON: `analysis/artifacts/knn_mixing_<model>.json`
- `analysis/scripts/make_probe_and_summary_figures.py`
  - Figure: `analysis/figs/knn_same_rate_bar.png`
- `analysis/scripts/make_interactive_pca_plotly.py`
  - HTML: `analysis/figs/pca3_plotly_<model>.html` (includes centroids; uses Plotly CDN)

## Notes
- The scripts use **hard-coded paths** and assumptions for this repo layout.
- This is intended to be run on a machine with enough RAM/CPU for kNN + PCA on the full bundle.
