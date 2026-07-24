# 04.01 — References: k-Means Clustering

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-§4 | Objective, Lloyd's algorithm, convergence | Lloyd (1957/1982); MacQueen (1967); ESL §14.3 |
| §5 | k-means++ | Arthur & Vassilvitskii (2007) |
| §6 | Choosing k (silhouette, gap) | Rousseeuw (1987); Tibshirani et al. (2001) |
| §7-§8 | Assumptions, GMM connection | ESL §14.3; Bishop §9.1 |
| §9 | k-medoids, variants | Kaufman & Rousseeuw (1990); Sculley (2010) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. **§14.3 "Cluster Analysis"** is the reference:
§14.3.6 k-means, §14.3.9 the connection to Gaussian mixtures (§8), and §14.3.11 the practical issues.

**Bishop, C. (2006). *Pattern Recognition and Machine Learning*.** §9.1 derives k-means and then §9.2
shows it as the zero-variance limit of the Gaussian-mixture EM (§8) — the clearest treatment of that
connection.

**Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster
Analysis*.** The book on k-medoids (PAM) and the silhouette (§6, §9).

---

## Papers

- **Lloyd, S. P. (1982).** "Least squares quantization in PCM." *IEEE Trans. Information Theory* 28(2),
  129-137. (Written 1957 at Bell Labs.) — **the algorithm** (§3); k-means as vector quantization.
- **MacQueen, J. (1967).** "Some methods for classification and analysis of multivariate
  observations." *Berkeley Symp.* — coined the name "k-means" and gave the online variant.
- **Arthur, D. & Vassilvitskii, S. (2007).** "k-means++: The Advantages of Careful Seeding." *SODA*. —
  **k-means++** (§5): the $D^2$ seeding and its $O(\log k)$ approximation guarantee. Free at
  <https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf>.
- **Rousseeuw, P. J. (1987).** "Silhouettes: a graphical aid to the interpretation and validation of
  cluster analysis." *J. Computational and Applied Mathematics* 20, 53-65. — **the silhouette** (§6).
- **Tibshirani, R., Walther, G. & Hastie, T. (2001).** "Estimating the number of clusters in a data
  set via the gap statistic." *JRSS-B* 63(2), 411-423. — **the gap statistic** (§6). Free at
  <https://hastie.su.domains/Papers/gap.pdf>.
- **Sculley, D. (2010).** "Web-scale k-means clustering." *WWW*. — **mini-batch k-means** (§9).
- **Elkan, C. (2003).** "Using the triangle inequality to accelerate k-means." *ICML*. — a fast exact
  k-means; useful implementation reading.
- **Kanungo, T. et al. (2002).** "A local search approximation algorithm for k-means clustering."
  *Computational Geometry* 28. — theory on the hardness and approximation of k-means (§2).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.cluster.KMeans`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py) | Lloyd + Elkan algorithms, k-means++ seeding, `n_init`; verified against here |
| [`sklearn.cluster.MiniBatchKMeans`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py) | the mini-batch variant (§9) |
| [`sklearn_extra.cluster.KMedoids`](https://github.com/scikit-learn-contrib/scikit-learn-extra) | k-medoids / PAM (§9) |
| [`sklearn.metrics.silhouette_score`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/cluster/_unsupervised.py) | the silhouette, verified against here |

---

## Deferred to later chapters

- **Gaussian mixtures & EM — the soft, full-covariance generalization** → [04.04](../04-gaussian-mixtures/)
- **DBSCAN — density clustering for arbitrary shapes** → [04.03](../03-density-clustering/)
- **Spectral clustering — non-convex clusters via the graph Laplacian** → [04.05](../05-spectral-clustering/)
- **Hierarchical clustering — no need to fix k up front** → [04.02](../02-hierarchical-clustering/)
- **Feature scaling — essential preprocessing for distance-based methods** → [02.04](../../02-data/04-scaling-and-transformation/)
