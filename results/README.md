# Results folder (tracked)

This repository keeps evaluation outputs under `results/` in git so results can be shared and analyzed.

## Layout

Results are stored per model, with `/` replaced by `__`:

- `results/<model_name_with__>/...`

Examples:
- `results/nomic-ai__nomic-embed-text-v1/`
- `results/BASF-AI__ChEmbed-vanilla/`

Within each model folder, MTEB writes JSON outputs per task.

## Notes

- This directory may grow large. If it becomes too big for regular git, we can switch to Git LFS later.
