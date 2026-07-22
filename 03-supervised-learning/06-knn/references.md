# 03.06 — References: k-Nearest Neighbours

Exact sections used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §1-§2 | Instance-based learning | Hastie et al., *ESL*, §2.3.2, §13.3; Mitchell, *Machine Learning*, Ch. 8 |
| §3 | Distance metrics | Deza & Deza, *Encyclopedia of Distances* |
| §5 | Choosing k, bias-variance | Hastie et al., *ESL*, §2.3.2, §7.1 |
| §7 | KNN regression | Hastie et al., *ESL*, §2.3.2 |
| §8 | Curse of dimensionality | Bellman (1961); Hastie et al., *ESL*, §2.5; Beyer et al. (1999) |
| §8.2 | Distance concentration | Beyer et al. (1999); Aggarwal, Hinneburg & Keim (2001) |
| §9 | Cover-Hart bound | Cover & Hart (1967); Devroye, Györfi & Lugosi (1996), Ch. 5 |
| §10.1 | KD-trees, ball trees | Bentley (1975); Friedman, Bentley & Finkel (1977) |
| §10.2 | Approximate search | Indyk & Motwani (1998); Malkov & Yashunin (2018) |

---

## Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>
§2.3.2 introduces KNN as the counterpoint to linear regression; **§2.5 is the classic
curse-of-dimensionality section** and the source of §8.1's edge-length table; §13.3 covers
prototype methods and the practical variants.

**Devroye, L., Györfi, L. & Lugosi, G. (1996). *A Probabilistic Theory of Pattern Recognition*.
Springer.**
The rigorous treatment of nearest-neighbour theory. Chapter 5 has the Cover-Hart proof and the
consistency results of §9; Chapter 6 covers the rates, including why convergence is exponentially
slow in $d$.

**Shakhnarovich, G., Darrell, T. & Indyk, P., eds. (2006). *Nearest-Neighbor Methods in Learning
and Vision*. MIT Press.**
The reference for the search side: LSH, tree structures, and metric learning.

**Deza, M. M. & Deza, E. (2016). *Encyclopedia of Distances*, 4th ed. Springer.**
Exhaustive catalogue of metrics. Useful when your data is not real-valued vectors — strings,
graphs, distributions, sets.

---

## Papers

### Theory
- **Cover, T. M. & Hart, P. E. (1967).** "Nearest neighbor pattern classification." *IEEE Trans.
  Information Theory* 13(1), 21-27. — **the $R^{*}\le R_{1NN}\le 2R^{*}$ result of §9.** Short and
  readable; the proof is worth working through once.
- **Fix, E. & Hodges, J. L. (1951).** "Discriminatory analysis, nonparametric discrimination."
  USAF School of Aviation Medicine, Report 4. — the original, unpublished for decades.
- **Stone, C. J. (1977).** "Consistent Nonparametric Regression." *Annals of Statistics* 5(4),
  595-620. — universal consistency for $k\to\infty$, $k/n\to0$.

### The curse of dimensionality
- **Beyer, K., Goldstein, J., Ramakrishnan, R. & Shaft, U. (1999).** "When Is 'Nearest Neighbor'
  Meaningful?" *ICDT*. — **the paper behind §8.2.** Proves that under broad conditions the ratio
  of farthest to nearest distance converges to 1, and asks the title's question seriously. Read
  this after Experiment 1.
- **Aggarwal, C. C., Hinneburg, A. & Keim, D. A. (2001).** "On the Surprising Behavior of Distance
  Metrics in High Dimensional Space." *ICDT*. — the follow-up showing $\ell_1$ concentrates less
  than $\ell_2$, and fractional norms $\ell_p$ with $p<1$ less still. Relevant to §3's metric
  choice.
- **Bellman, R. (1961). *Adaptive Control Processes*.** — where the phrase comes from.
- **Radovanović, M., Nanopoulos, A. & Ivanović, M. (2010).** "Hubs in Space: Popular Nearest
  Neighbors in High-Dimensional Data." *JMLR* 11, 2487-2531. — **hubness**, a fourth
  high-dimensional pathology this chapter does not cover: some points become the nearest neighbour
  of a disproportionate number of queries, distorting KNN badly.

### Search structures
- **Bentley, J. L. (1975).** "Multidimensional binary search trees used for associative
  searching." *CACM* 18(9), 509-517. — KD-trees.
- **Friedman, J. H., Bentley, J. L. & Finkel, R. A. (1977).** "An Algorithm for Finding Best
  Matches in Logarithmic Expected Time." *ACM TOMS* 3(3), 209-226. — the query algorithm and its
  complexity, including the dimension dependence of §10.1.
- **Omohundro, S. M. (1989).** "Five Balltree Construction Algorithms." ICSI Technical Report.
- **Indyk, P. & Motwani, R. (1998).** "Approximate nearest neighbors: towards removing the curse
  of dimensionality." *STOC*. — LSH.
- **Malkov, Y. A. & Yashunin, D. A. (2018).** "Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs." *IEEE TPAMI* 42(4), 824-836. —
  **HNSW**, the algorithm behind essentially every modern vector database. If you read one paper
  from §10.2, read this.
- **Jégou, H., Douze, M. & Schmid, C. (2011).** "Product Quantization for Nearest Neighbor
  Search." *IEEE TPAMI* 33(1), 117-128. — IVF-PQ, how FAISS compresses billions of vectors.
- **Johnson, J., Douze, M. & Jégou, H. (2019).** "Billion-scale similarity search with GPUs."
  *IEEE Trans. Big Data*. — the FAISS paper.

### Metric learning
- **Weinberger, K. Q. & Saul, L. K. (2009).** "Distance Metric Learning for Large Margin Nearest
  Neighbor Classification." *JMLR* 10, 207-244. — LMNN: learn the Mahalanobis matrix from labels.
- **Goldberger, J. et al. (2004).** "Neighbourhood Components Analysis." *NeurIPS*. — NCA;
  available as `sklearn.neighbors.NeighborhoodComponentsAnalysis`.
- **Schroff, F., Kalenichenko, D. & Philbin, J. (2015).** "FaceNet: A Unified Embedding for Face
  Recognition and Clustering." *CVPR*. — the triplet loss: learn an embedding *so that* KNN works.
  This is §8.4's escape route, made industrial.

---

## Reference implementations

| Source | What to look at |
|---|---|
| [scikit-learn `_kd_tree.pyx` / `_ball_tree.pyx`](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/neighbors) | production tree implementations; note the `algorithm="auto"` heuristic and where it gives up on trees |
| [scikit-learn `_classification.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neighbors/_classification.py) | the weighting logic, including the exact-hit special case |
| [`faiss`](https://github.com/facebookresearch/faiss) | the industry standard for large-scale similarity search — IVF, PQ, HNSW, GPU |
| [`hnswlib`](https://github.com/nmslib/hnswlib) | a small, readable HNSW implementation; good for understanding the algorithm |
| [`annoy`](https://github.com/spotify/annoy) | random projection forests; memory-mappable, which matters at scale |
| [`pgvector`](https://github.com/pgvector/pgvector) | vector search inside Postgres — often the right answer for moderate scale |

---

## Benchmarks worth knowing

**ANN-Benchmarks** — <https://ann-benchmarks.com/> — standardized recall-vs-throughput comparisons
across every major library and dataset. The right place to look before choosing an index, and a
good corrective to marketing claims.

---

## Deferred to later chapters

- **Dimensionality reduction: PCA, t-SNE, UMAP** → [04.06](../../04-unsupervised-learning/06-linear-dim-reduction/), [04.07](../../04-unsupervised-learning/07-manifold-learning/)
- **k-means, which uses the same distance machinery** → [04.01](../../04-unsupervised-learning/01-kmeans/)
- **Density-based clustering (DBSCAN) — nearest neighbours again** → [04.03](../../04-unsupervised-learning/03-density-clustering/)
- **Local Outlier Factor — anomaly detection by neighbour density** → [04.08](../../04-unsupervised-learning/08-anomaly-detection/)
- **Kernel methods: the same "similarity" idea, done implicitly** → [03.07](../07-svm/)
- **Learned embeddings, and why they rescue KNN** → [10.03](../../10-nlp/03-word-embeddings/), [08.05](../../08-computer-vision/05-vision-transformers/)
- **Vector search and RAG — KNN's most important modern application** → [11.08](../../11-transformers-and-llms/08-rag-and-agents/)
- **Recommender candidate generation by ANN retrieval** → [16.02](../../16-recommender-systems/02-modern-recommenders/)
