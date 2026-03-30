# Analysis (Geometry bundle)

This folder contains **contained analysis code** to generate geometry metrics + figures from the already-generated embedding bundle under:

- `results/geometry_bundle/`

## What this produces
Outputs are written to:

- `analysis/runs/<run_id>/`
  - `report.md` (main report)
  - `metrics.json` (machine-readable)
  - `figs/` (PNGs)

## Quick start
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r analysis/requirements.txt

python analysis/scripts/run_all.py
```

After it finishes, commit the generated folder under `analysis/runs/<run_id>/`.

## Notes
- The scripts use **hard-coded paths** and assumptions for this repo layout.
- This is intended to be run on a machine with enough RAM/CPU for kNN + PCA on the full bundle.
