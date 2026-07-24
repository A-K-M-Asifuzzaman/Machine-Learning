# 04.02 — References: Hierarchical Clustering

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-§4 | Agglomerative clustering, linkages | ESL §14.3.12; Kaufman & Rousseeuw (1990) |
| §4 | Ward's method | Ward (1963) |
| §5 | Lance-Williams update | Lance & Williams (1967) |
| §6-§7 | Dendrogram, cophenetic correlation | Sokal & Rohlf (1962) |
| §8 | Fast algorithms | Müllner (2011); Sibson (1973, SLINK) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. §14.3.12 "Hierarchical Clustering" covers
agglomerative merging, the linkages, and dendrograms (§2-§6).

**Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data*.** Chapters 5-6 on agglomerative
(AGNES) and divisive (DIANA) clustering, with the linkage comparisons of §4.

**Everitt, B., Landau, S., Leese, M. & Stahl, D. (2011). *Cluster Analysis*, 5th ed.** The most
thorough single reference on clustering; Chapter 4 is hierarchical methods.

---

## Papers

- **Ward, J. H. (1963).** "Hierarchical Grouping to Optimize an Objective Function." *JASA* 58(301),
  236-244. — **Ward's method** (§4): merge to minimize the increase in within-cluster variance; the
  sum-of-squares link to k-means (§9).
- **Lance, G. N. & Williams, W. T. (1967).** "A general theory of classificatory sorting strategies."
  *The Computer Journal* 9(4). — **the Lance-Williams recurrence** (§5) that unifies all linkages under
  one update formula.
- **Sibson, R. (1973).** "SLINK: an optimally efficient algorithm for the single-link cluster method."
  *The Computer Journal* 16(1). — an $O(n^2)$ single-linkage algorithm (§8).
- **Defays, D. (1977).** "An efficient algorithm for a complete link method." *The Computer Journal*
  20(4). — CLINK, the complete-linkage counterpart.
- **Müllner, D. (2011).** "Modern hierarchical, agglomerative clustering algorithms."
  *arXiv:1109.2378*. — **the reference on fast agglomerative algorithms** (nearest-neighbor chain),
  and the basis of `scipy`'s and `fastcluster`'s implementations (§8). Free at
  <https://arxiv.org/abs/1109.2378>.
- **Sokal, R. R. & Rohlf, F. J. (1962).** "The comparison of dendrograms by objective methods."
  *Taxon* 11(2). — the **cophenetic correlation** (§7).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`scipy.cluster.hierarchy`](https://github.com/scipy/scipy/blob/main/scipy/cluster/hierarchy.py) | `linkage`, `dendrogram`, `fcluster`, `cophenet` — the linkage matrix format and all four methods, verified against here |
| [`sklearn.cluster.AgglomerativeClustering`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_agglomerative.py) | connectivity-constrained agglomerative clustering; Ward, complete, average, single |
| [`fastcluster`](https://github.com/dmuellner/fastcluster) | Müllner's fast C++ implementation of the nearest-neighbor-chain algorithm (§8) |

---

## Deferred to later chapters

- **k-Means — the flat, centroid-based counterpart** → [04.01](../01-kmeans/)
- **DBSCAN — density clustering, also handles arbitrary shapes** → [04.03](../03-density-clustering/)
- **Spectral clustering — non-convex clusters via the graph Laplacian** → [04.05](../05-spectral-clustering/)
- **HDBSCAN — a hierarchical density method combining both ideas** → [04.03](../03-density-clustering/)
