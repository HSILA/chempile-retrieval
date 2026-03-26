# Decisions & next steps

This file captures the decisions finalized during the geometry-comparison setup work so we do not have to re-derive them later.

## Finalized decisions

### Chempile retrieval task choice
- Use **A3** as the primary Chempile task for geometry comparison.
- Rationale: chemistry-only + combined title/question query mode is the most relevant first comparison point against our chemistry retrieval setup.

### Comparison goals
We want to support two distinct analyses:
1. **Dataset geometry / distribution shift**
   - Compare how different the distributions are across:
     - ChemRxiv retrieval
     - Chempile A3
     - general Wikipedia-like retrieval passages
2. **Model representation geometry**
   - Compare whether the two embedding models induce different spaces / representations on the same texts.

### Models to use
Use both:
- `BASF-AI/ChEmbed-vanilla`
- `nomic-ai/nomic-embed-text-v1`

### Dataset sources
#### Chemistry retrieval datasets
- **ChemRxiv retrieval** from Hugging Face:
  - dataset: `BASF-AI/ChemRxivRetrieval`
  - config `queries` → split `train`, columns `_id`, `text`
  - config `corpus` → split `train`, columns `_id`, `text`
- **Chempile A3** from local repo files:
  - `data/A3/queries.jsonl`
  - `data/A3/corpus.jsonl`

#### General-domain anchor datasets
Use 50/50 mixture of corpus passages from:
- `mteb/hotpotqa` config `corpus`
- `mteb/nq` config `corpus`

### Query vs corpus policy
#### Retrieval datasets (ChemRxiv + Chempile A3)
Keep **both**:
- query embeddings
- corpus embeddings

Reason:
- corpus embeddings are useful for dataset/distribution geometry
- query + corpus together preserve the option to analyze retrieval-specific structure later

#### Anchor sets
Use **corpus/passages only**.
Do **not** use query anchors by default.

Reason:
- anchors are meant to represent a stable content distribution reference
- query text is shorter and stylistically different; for ChemRxiv it is also synthesized

### Anchor sets to build
Build two anchor sets:
1. **anchors_mixed_chem**
   - 50/50 mixture of:
     - ChemRxiv corpus passages
     - Chempile A3 corpus passages
2. **anchors_general**
   - 50/50 mixture of:
     - HotpotQA corpus passages
     - NQ corpus passages

### Reproducibility
- Use a fixed random seed: **1337**
- Keep dataset/model choices hardcoded inside the collection script for reproducibility

### Runtime / device choice
- Default device: **CPU**
- Reason: MPS caused out-of-memory failures on long corpus passages and did not provide reliable end-to-end speedups for this workflow.

### Batch size choice
- Match the repo's existing CPU workflow default used by `scripts/run_all_models_cpu.sh`
- Current collector batch size: **4**

## Output bundle
Collector script:
- `scripts/collect_geometry_bundle.py`

Outputs under:
- `results/geometry_bundle/<model_slug>/...`

Per model it writes:
- `chemrxiv/queries.npy`
- `chemrxiv/corpus.npy`
- `chemrxiv/meta.json`
- `chempile_A3/queries.npy`
- `chempile_A3/corpus.npy`
- `chempile_A3/meta.json`
- `anchors_mixed_chem/texts.jsonl`
- `anchors_mixed_chem/embeddings.npy`
- `anchors_mixed_chem/meta.json`
- `anchors_general/texts.jsonl`
- `anchors_general/embeddings.npy`
- `anchors_general/meta.json`

Top-level outputs:
- `results/geometry_bundle/manifest.json`
- `results/geometry_bundle/run_summary.json`

## Status
### Done
- Built the 9 Chempile retrieval tasks / dataset variants
- Added embedding dump utility for plotting/debugging
- Added geometry bundle collector script
- Finalized data/model/anchor decisions above
- Ran the geometry bundle collection and pushed results to the repo

### Next for this project slice
1. Inspect the pushed geometry bundle artifacts and verify completeness/shape
2. Review meeting notes / extracted notes / planning notes relevant to this benchmark thread
3. Decide the next analysis plan based on:
   - what geometry bundle now enables
   - what evidence is still missing
4. Then move to the next requested task: build or refine the workflow for extracting meeting notes / TODO planning and deciding next actions
