# Geometry analysis report (expanded)

Run id: `20260330-022831`

## Setting
- Purpose: quantify distribution shift between ChemRxiv literature paragraphs, Chempile (chemistry StackExchange-style QA answers), and general Wikipedia QA corpora; and compare how ChEmbed vs base nomic organize these texts in embedding space.
- Embedding spaces:
  - `BASF-AI/ChEmbed-vanilla`
  - `nomic-ai/nomic-embed-text-v1`
- Data used (corpus-only for plots/shift metrics):
  - ChemRxiv corpus: 10,000 passages
  - Chempile A3 corpus: 3,185 answers
  - anchors_general corpus: 2,000 passages (50/50 HotpotQA + Natural Questions)
  - Notes: this run intentionally avoids plotting/query-mixing; anchors are corpus/passages only.

## Core observations
### BASF-AI__ChEmbed-vanilla

**Sanity: vector norms**
- chemrxiv mean norm: 1.000000 (std 6.39e-08)
- chempile mean norm: 1.000000 (std 6.40e-08)
- anchors_general mean norm: 1.000000 (std 6.47e-08)
- Interpretation: embeddings are effectively unit-normalized already, so cosine-space analyses are appropriate and norm/length artifacts via magnitude are unlikely here.

**Centroid cosine distances (corpus means)**
- chemrxiv vs chempile: 0.414
- chemrxiv vs anchors_general: 0.571
- chempile vs anchors_general: 0.474
- Interpretation: in ChEmbed space, ChemRxiv and Chempile are far apart, and both are far from general anchors. This is consistent with strong domain/task specialization and a large representation shift.

**kNN same-dataset neighbor rate (pooled corpora)**
- k=10: 0.881
- k=50: 0.829
- Interpretation: high values mean that, locally, passages mostly neighbor passages from the same source dataset. This indicates a strong distribution shift signal (datasets cluster in embedding space).

**Dataset-source probe (PCA→logreg, macro-F1)**
- macro-F1: 0.961
- PCA dims used: 64 (explained variance sum 0.366)
- Interpretation: near-perfect separability means these corpora are easily distinguishable in embedding space even after reducing to 64 PCs. This supports the "distribution shift" premise directly.

**PCA scatter**
- See: `figs/pca_scatter_BASF-AI__ChEmbed-vanilla.png`
- Qualitative note: in 2D PCA, clusters overlap more. This does not contradict separability: PCA(2D) is a projection, while the probe result reflects separability in a higher-dimensional subspace.

### nomic-ai__nomic-embed-text-v1

**Sanity: vector norms**
- chemrxiv mean norm: 1.000000 (std 6.81e-08)
- chempile mean norm: 1.000000 (std 6.77e-08)
- anchors_general mean norm: 1.000000 (std 6.78e-08)
- Interpretation: embeddings are effectively unit-normalized already, so cosine-space analyses are appropriate and norm/length artifacts via magnitude are unlikely here.

**Centroid cosine distances (corpus means)**
- chemrxiv vs chempile: 0.045
- chemrxiv vs anchors_general: 0.146
- chempile vs anchors_general: 0.148
- Interpretation: in nomic space, ChemRxiv and Chempile centroids are much closer, and both are relatively closer to general anchors. This suggests a more domain-agnostic geometry (or at least less centroid drift between the two chemistry sources).

**kNN same-dataset neighbor rate (pooled corpora)**
- k=10: 0.929
- k=50: 0.892
- Interpretation: high values mean that, locally, passages mostly neighbor passages from the same source dataset. This indicates a strong distribution shift signal (datasets cluster in embedding space).

**Dataset-source probe (PCA→logreg, macro-F1)**
- macro-F1: 0.969
- PCA dims used: 64 (explained variance sum 0.490)
- Interpretation: near-perfect separability means these corpora are easily distinguishable in embedding space even after reducing to 64 PCs. This supports the "distribution shift" premise directly.

**PCA scatter**
- See: `figs/pca_scatter_nomic-ai__nomic-embed-text-v1.png`
- Qualitative note: in 2D PCA, anchors_general separates cleanly from the two chemistry corpora along PC1, while ChemRxiv vs Chempile separate along PC2. This is consistent with strong dataset clustering in the top variance directions.

## Cross-model comparison (ChEmbed vs nomic)
### chemrxiv_corpus
- shape: [10000, 768]
- linear CKA: None
- Spearman distance correlation (subsample): 0.0017
- neighbor overlap:
  - k=10: 0.0010
  - k=50: 0.0048
- Interpretation: extremely low neighbor-overlap and near-zero distance-correlation imply the two models induce very different local geometries on the same texts (not a simple rotation/scale). This is consistent with domain adaptation changing what is considered "similar".

### chempile_corpus
- shape: [3185, 768]
- linear CKA: 0.04159276714969064
- Spearman distance correlation (subsample): -0.0094
- neighbor overlap:
  - k=10: 0.0036
  - k=50: 0.0161
- Interpretation: extremely low neighbor-overlap and near-zero distance-correlation imply the two models induce very different local geometries on the same texts (not a simple rotation/scale). This is consistent with domain adaptation changing what is considered "similar".

## What this means for the paper story (careful version)
- These results support that ChemRxiv-literature retrieval, Chempile QA-style chemistry retrieval, and Wikipedia-style corpora are distinct distributions in embedding space.
- ChEmbed shows large centroid shifts between ChemRxiv and Chempile, which is consistent with specialization to literature-like retrieval. However, geometry alone does not prove retrieval performance; we still need task metrics to tie this to “works/doesn’t work”.
- The PCA overlap in ChEmbed does not imply the model is bad; separability is still very high in higher dimensions, and PCA(2D) is not a decision boundary.

## Next improvements (to reduce reviewer skepticism)
- Add raw text exports (or at least length/token counts) for ChemRxiv + Chempile corpora in the bundle so we can run length-matched controls and a length-only baseline probe.
- Add a 3D interactive visualization (Plotly) for corpus centroids and/or a sampled scatter to illustrate the “triangle” intuition without over-interpreting 2D PCA.

