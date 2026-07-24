# 04.06 — References: Linear Dimensionality Reduction (PCA)

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§4 | PCA, max variance, covariance eigendecomposition | Pearson (1901); Hotelling (1933); ESL §14.5 |
| §3 | Minimum reconstruction error | Jolliffe (2002) §1-2 |
| §5 | SVD and numerical computation | Golub & Van Loan (2013); ESL §14.5 |
| §6-§7 | Choosing k, scaling, whitening | Jolliffe (2002) §6; ESL §14.5 |
| §8, §10 | Limits, probabilistic PCA, kernel PCA | Tipping & Bishop (1999); Schölkopf et al. (1998) |

---

## Books

**Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed.** — **the definitive book on PCA.**
Every section here (the two derivations, choosing components, scaling, rotations, robust and sparse
variants) is treated in depth. The reference if you want everything about PCA in one place.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. §14.5 "Principal Components, Curves and
Surfaces" gives the concise modern treatment: the SVD view (§5), the reconstruction-error derivation
(§3), and the connection to principal curves (nonlinear, §10).

**Bishop, C. (2006). *Pattern Recognition and Machine Learning*.** §12.1 PCA (both derivations), §12.2
probabilistic PCA and factor analysis (§10), §12.3 kernel PCA (§10) — the cleanest unified treatment.

**Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.** The reference for the SVD
and why it is the numerically stable way to compute PCA (§5).

---

## Papers

- **Pearson, K. (1901).** "On Lines and Planes of Closest Fit to Systems of Points in Space."
  *Philosophical Magazine* 2(11), 559-572. — **the origin of PCA** as the best-fitting line/plane
  (the minimum-reconstruction view, §3).
- **Hotelling, H. (1933).** "Analysis of a complex of statistical variables into principal
  components." *J. Educational Psychology* 24. — the **maximum-variance** formulation (§2) and the name
  "principal components."
- **Tipping, M. E. & Bishop, C. M. (1999).** "Probabilistic Principal Component Analysis." *JRSS-B*
  61(3), 611-622. — **PPCA** (§10): PCA as a latent-Gaussian model, giving a likelihood and handling
  missing data. Free at <https://www.microsoft.com/en-us/research/publication/probabilistic-principal-component-analysis/>.
- **Schölkopf, B., Smola, A. & Müller, K.-R. (1998).** "Nonlinear Component Analysis as a Kernel
  Eigenvalue Problem." *Neural Computation* 10(5). — **kernel PCA** (§10): PCA in a feature space for
  nonlinear structure.
- **Zou, H., Hastie, T. & Tibshirani, R. (2006).** "Sparse Principal Component Analysis." *JCGS*
  15(2). — **sparse PCA** (§9-§10) for interpretable components.
- **Halko, N., Martinsson, P.-G. & Tropp, J. A. (2011).** "Finding Structure with Randomness:
  Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." *SIAM Review* 53(2). —
  **randomized SVD/PCA** for large data (§10). Free at <https://arxiv.org/abs/0909.4061>.
- **Candès, E. J., Li, X., Ma, Y. & Wright, J. (2011).** "Robust Principal Component Analysis?"
  *JACM* 58(3). — **robust PCA** (low-rank + sparse decomposition) for outlier-corrupted data (§10).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.decomposition.PCA`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_pca.py) | the SVD-based implementation, `explained_variance_ratio_`, whitening, the `svd_solver` options; verified against here |
| [`sklearn.decomposition.TruncatedSVD`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_truncated_svd.py) | PCA without centering (for sparse data / LSA) |
| [`sklearn.decomposition.KernelPCA`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_kernel_pca.py) | nonlinear PCA via kernels (§10) |
| [`sklearn.decomposition.IncrementalPCA`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py) | streaming / out-of-core PCA |

---

## Deferred to later chapters

- **Manifold learning (t-SNE, UMAP, Isomap, LLE) — nonlinear dimensionality reduction** → [04.07](../07-manifold-learning/)
- **Spectral clustering — Laplacian eigenvectors, a cousin of PCA's covariance eigenvectors** → [04.05](../05-spectral-clustering/)
- **LDA — the supervised linear projection** → [03.05](../../03-supervised-learning/05-generative-classifiers/)
- **Autoencoders — nonlinear PCA with neural networks** → [07.xx / 12.xx]
- **The curse of dimensionality PCA helps fight** → [03.06](../../03-supervised-learning/06-knn/)
- **Probabilistic PCA & factor analysis — the latent-variable view** → [04.04](../04-gaussian-mixtures/) (EM)
