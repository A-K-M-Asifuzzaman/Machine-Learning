# 04.03 — References: Density Clustering (DBSCAN)

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§4 | DBSCAN algorithm, core/border/noise | Ester, Kriegel, Sander & Xu (1996) |
| §5 | eps via k-distance | Ester et al. (1996); Schubert et al. (2017) |
| §7 | OPTICS | Ankerst, Breunig, Kriegel & Sander (1999) |
| §7 | HDBSCAN | Campello, Moulavi & Sander (2013) |
| §6, §8 | Failure modes, high-dim, complexity | Schubert et al. (2017) |

---

## Papers

- **Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996).** "A Density-Based Algorithm for Discovering
  Clusters in Large Spatial Databases with Noise." *KDD*. — **the DBSCAN paper**, and the source for
  §1-§5. Introduces core/border/noise, density-reachability, the algorithm, and the k-distance
  heuristic. Won the 2014 KDD Test of Time award. Free at
  <https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf>.
- **Schubert, E., Sander, J., Ester, M., Kriegel, H.-P. & Xu, X. (2017).** "DBSCAN Revisited,
  Revisited: Why and How You Should (Still) Use DBSCAN." *ACM TODS* 42(3). — **the modern reference**:
  clears up misconceptions, parameter selection (§5), complexity (§8), and when DBSCAN does and does
  not work (§6). Read this alongside the original.
- **Ankerst, M., Breunig, M. M., Kriegel, H.-P. & Sander, J. (1999).** "OPTICS: Ordering Points To
  Identify the Clustering Structure." *SIGMOD*. — **OPTICS** (§7): the reachability plot that handles
  varying density.
- **Campello, R. J. G. B., Moulavi, D. & Sander, J. (2013).** "Density-Based Clustering Based on
  Hierarchical Density Estimates." *PAKDD*. — **HDBSCAN** (§7): hierarchical density clustering with
  stability-based extraction. The practical successor to DBSCAN.
- **McInnes, L., Healy, J. & Astels, S. (2017).** "hdbscan: Hierarchical density based clustering."
  *JOSS* 2(11). — the widely-used HDBSCAN implementation and its clear documentation.

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
§14.3 places DBSCAN among clustering methods; the density view contrasts with the centroid (k-means)
and linkage (hierarchical) views.

**Aggarwal, C. C. & Reddy, C. K. (eds.) (2013). *Data Clustering: Algorithms and Applications*.**
Chapter 5 (density-based clustering) is a thorough survey of DBSCAN, OPTICS, and their descendants.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.cluster.DBSCAN`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_dbscan.py) | the core-point/expansion logic and neighbor queries; verified against here |
| [`sklearn.cluster.OPTICS`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_optics.py) | the reachability ordering (§7) |
| [`hdbscan`](https://github.com/scikit-learn-contrib/hdbscan) | HDBSCAN — varying-density clustering (§7) |
| [ELKI](https://elki-project.github.io/) | reference implementations of DBSCAN, OPTICS, and many variants, by the authors of the "revisited" paper |

---

## Deferred to later chapters

- **Gaussian mixtures — soft, model-based clustering** → [04.04](../04-gaussian-mixtures/)
- **Spectral clustering — another non-convex method, via the graph Laplacian** → [04.05](../05-spectral-clustering/)
- **Dimensionality reduction before density clustering in high-d** → [04.06](../06-linear-dimensionality-reduction/)
- **Anomaly detection — density estimation for outliers** → [04.08](../08-anomaly-detection/)
- **The curse of dimensionality and distance concentration** → [03.06](../../03-supervised-learning/06-knn/)
