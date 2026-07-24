# 04.05 — References: Spectral Clustering

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§5 | Algorithm, Laplacian, embedding | Ng, Jordan & Weiss (2002); von Luxburg (2007) |
| §3-§4 | Laplacian properties | Chung (1997); von Luxburg (2007) |
| §6 | Graph cuts (NCut, RatioCut) | Shi & Malik (2000); Hagen & Kahng (1992) |
| §7 | Eigengap heuristic | von Luxburg (2007) |
| §8 | Graph construction, self-tuning | Zelnik-Manor & Perona (2004) |
| §9 | Connections, scaling | von Luxburg (2007); Belkin & Niyogi (2003) |

---

## Papers

- **von Luxburg, U. (2007).** "A Tutorial on Spectral Clustering." *Statistics and Computing* 17(4),
  395-416. — **the definitive tutorial and the backbone of this chapter.** Covers the three
  Laplacians, the graph-cut relaxations (§6), the eigengap (§7), graph construction (§8), and the
  consistency results that explain why normalized Laplacians are preferred. Read this. Free at
  <https://arxiv.org/abs/0711.0189>.
- **Shi, J. & Malik, J. (2000).** "Normalized Cuts and Image Segmentation." *IEEE TPAMI* 22(8),
  888-905. — **Normalized Cut** (§6) and its spectral relaxation; the paper that made spectral methods
  mainstream (via image segmentation). Free at
  <https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf>.
- **Ng, A. Y., Jordan, M. I. & Weiss, Y. (2002).** "On Spectral Clustering: Analysis and an
  Algorithm." *NeurIPS*. — **the NJW algorithm** (§5): the symmetric Laplacian with row normalization
  and k-means on the embedding. Free at
  <https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html>.
- **Hagen, L. & Kahng, A. B. (1992).** "New spectral methods for ratio cut partitioning and
  clustering." *IEEE TCAD* 11(9). — **RatioCut** and its eigenvector relaxation (§6).
- **Zelnik-Manor, L. & Perona, P. (2004).** "Self-Tuning Spectral Clustering." *NeurIPS*. — local
  scaling of the affinity to handle varying density (§8).
- **Belkin, M. & Niyogi, P. (2003).** "Laplacian Eigenmaps for Dimensionality Reduction and Data
  Representation." *Neural Computation* 15(6). — the same Laplacian eigenvectors used for **manifold
  learning** ([04.07](../07-manifold-learning/)); the embedding view of §9.

---

## Books

**Chung, F. R. K. (1997). *Spectral Graph Theory*.** The mathematical foundation: the Laplacian, its
spectrum, and the connected-components / Cheeger-inequality theory behind §3-§4 and §7.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
§14.5.3 introduces spectral clustering concisely and connects it to kernel methods (§9).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.cluster.SpectralClustering`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_spectral.py) | affinity construction, the normalized Laplacian, `assign_labels`; verified against here |
| [`sklearn.manifold.spectral_embedding`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_spectral_embedding.py) | the eigenvector embedding step (§5), shared with Laplacian eigenmaps |
| [`scipy.sparse.linalg.eigsh`](https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/) | the sparse eigensolver for large $k$-NN graphs (§9) |

---

## Deferred to later chapters

- **k-Means — used on the spectral embedding** → [04.01](../01-kmeans/)
- **PCA — eigenvectors of the covariance, the largest ones** → [04.06](../06-linear-dimensionality-reduction/)
- **Laplacian eigenmaps — the same embedding for dimensionality reduction** → [04.07](../07-manifold-learning/)
- **DBSCAN — another non-convex clusterer, via density** → [04.03](../03-density-clustering/)
- **Graph neural networks — the Laplacian as a learnable operator** → [14.xx graph ML]
