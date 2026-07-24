# 04.07 — References: Manifold Learning

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1 | Manifold hypothesis | Tenenbaum et al. (2000); Cayton (2005) |
| §3 | Isomap, classical MDS | Tenenbaum, de Silva & Langford (2000) |
| §4 | LLE, Laplacian eigenmaps | Roweis & Saul (2000); Belkin & Niyogi (2003) |
| §5 | t-SNE | van der Maaten & Hinton (2008) |
| §6 | UMAP | McInnes, Healy & Melville (2018) |
| §7 | Reading t-SNE plots | Wattenberg, Viégas & Johnson (2016) |

---

## Papers

- **Tenenbaum, J. B., de Silva, V. & Langford, J. C. (2000).** "A Global Geometric Framework for
  Nonlinear Dimensionality Reduction." *Science* 290(5500), 2319-2323. — **Isomap** (§3): geodesic
  distances via graph shortest paths + MDS. Free at
  <https://web.mit.edu/cocosci/Papers/sci_reprint.pdf>.
- **Roweis, S. T. & Saul, L. K. (2000).** "Nonlinear Dimensionality Reduction by Locally Linear
  Embedding." *Science* 290(5500), 2323-2326. — **LLE** (§4). The companion to Isomap in the same
  Science issue that launched manifold learning.
- **Belkin, M. & Niyogi, P. (2003).** "Laplacian Eigenmaps for Dimensionality Reduction and Data
  Representation." *Neural Computation* 15(6). — **Laplacian eigenmaps** (§4); the tie to spectral
  clustering ([04.05](../05-spectral-clustering/)).
- **van der Maaten, L. & Hinton, G. (2008).** "Visualizing Data using t-SNE." *JMLR* 9, 2579-2605. —
  **the t-SNE paper** (§5): perplexity-calibrated $P$, Student-t $Q$, the KL objective and gradient.
  Free at <https://jmlr.org/papers/v9/vandermaaten08a.html>.
- **McInnes, L., Healy, J. & Melville, J. (2018).** "UMAP: Uniform Manifold Approximation and
  Projection for Dimension Reduction." *arXiv:1802.03426*. — **UMAP** (§6). Free at
  <https://arxiv.org/abs/1802.03426>.
- **Wattenberg, M., Viégas, F. & Johnson, I. (2016).** "How to Use t-SNE Effectively."
  *Distill.pub*. — **the essential guide to reading t-SNE plots** (§7): cluster sizes, distances, and
  perplexity are all shown to be misleading. Read this before ever presenting a t-SNE plot. Free at
  <https://distill.pub/2016/misread-tsne/>.
- **Cayton, L. (2005).** "Algorithms for manifold learning." *UCSD Tech Report*. — a clear survey of
  the classical methods (§2-§4).
- **Coifman, R. R. & Lafon, S. (2006).** "Diffusion maps." *Applied and Computational Harmonic
  Analysis* 21(1). — the diffusion-geometry view connecting spectral methods and manifold learning.

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. §14.9 "Nonlinear Dimension Reduction and Local
Multidimensional Scaling" covers Isomap, LLE, and Laplacian eigenmaps (§3-§4).

**Bishop, C. (2006). *Pattern Recognition and Machine Learning*.** §12.4 (nonlinear latent variable
models) and the MDS/PCA connections behind classical MDS (§3).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.manifold.Isomap`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_isomap.py) | k-NN graph, shortest paths, kernel PCA on the geodesic distances; verified against here |
| [`sklearn.manifold.TSNE`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_t_sne.py) | perplexity calibration, Barnes-Hut approximation, early exaggeration |
| [`sklearn.manifold.LocallyLinearEmbedding`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_locally_linear.py) | LLE and its variants (§4) |
| [`umap-learn`](https://github.com/lmcinnes/umap) | the reference UMAP implementation (§6) |
| [openTSNE](https://github.com/pavlin-policar/openTSNE) | a fast, extensible t-SNE with out-of-sample transform |

---

## Deferred to later chapters

- **PCA — the linear baseline and a common preprocessing step** → [04.06](../06-linear-dimensionality-reduction/)
- **Spectral clustering — Laplacian eigenmaps for clustering** → [04.05](../05-spectral-clustering/)
- **Autoencoders — nonlinear dimensionality reduction with neural nets** → [07.xx / 12.xx]
- **Self-supervised representation learning — learned embeddings** → [11.xx]
- **The curse of dimensionality that motivates dimension reduction** → [03.06](../../03-supervised-learning/06-knn/)
