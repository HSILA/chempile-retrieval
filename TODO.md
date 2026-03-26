# TODO

## Done
- [x] Build the 9 Chempile retrieval dataset variants from `jablonkagroup/chempile-reasoning`
- [x] Add local evaluation flow for the retrieval tasks
- [x] Add small embedding-dump utility for offline plotting/debugging
- [x] Decide the primary comparison task: **A3**
- [x] Decide the comparison models:
  - `BASF-AI/ChEmbed-vanilla`
  - `nomic-ai/nomic-embed-text-v1`
- [x] Decide anchor strategy:
  - `anchors_mixed_chem` = 50/50 ChemRxiv corpus + Chempile A3 corpus
  - `anchors_general` = 50/50 HotpotQA corpus + NQ corpus
- [x] Keep **queries + corpus** for ChemRxiv and Chempile A3
- [x] Keep **corpus-only** for anchor sets
- [x] Make geometry collection reproducible with fixed seed
- [x] Build `scripts/collect_geometry_bundle.py`
- [x] Switch geometry collection default device to **CPU** (avoid MPS OOM)
- [x] Match collector batch size to repo CPU default
- [x] Run the geometry bundle collection and push results

## Current focus
- [ ] Inspect the pushed geometry bundle outputs and verify they are complete and usable
- [ ] Summarize what the collected bundle contains at a glance (counts, shapes, model folders, missing artifacts if any)

## Next
- [ ] Extract / review meeting notes relevant to this benchmark thread
- [ ] Build a plan for what to do next based on:
  - geometry bundle contents
  - current open questions
  - missing evidence
- [ ] Decide the next analysis step for:
  - dataset geometry / distribution shift
  - model representation geometry

## Later
- [ ] Add an analysis script or notebook for geometry metrics + plots
- [ ] Decide whether query-anchor variants are needed later
- [ ] Consider screening B3/C3 only if A3 proves insufficient for the comparison story
