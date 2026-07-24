# 04.04 — References: Gaussian Mixtures & EM

Exact sources used, so every claim in this chapter can be traced and checked.

---

## Primary sources by section

| README § | Topic | Source |
|---|---|---|
| §2-§4 | GMM, EM algorithm | Bishop §9.2; Dempster, Laird & Rubin (1977) |
| §5 | ELBO, EM as coordinate ascent | Neal & Hinton (1998); Bishop §9.4 |
| §6 | Covariance types | Bishop §9.2; ESL §14.3.7 |
| §7 | k-means as hard EM | Bishop §9.3.2 |
| §8 | BIC / model selection | Schwarz (1978); Fraley & Raftery (2002) |
| §9 | Singularities, regularization | Bishop §9.2.1 |

---

## Books

**Bishop, C. (2006). *Pattern Recognition and Machine Learning*.** — **Chapter 9 is the definitive
treatment** and the backbone of this chapter: §9.2 GMMs and EM, §9.3 the k-means connection (§7), §9.4
the general EM / ELBO view (§5). If you read one thing, read Chapter 9. The singularity discussion
(§9.2.1) is exactly §9 here.

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.**
— free at <https://hastie.su.domains/ElemStatLearn/>. §8.5 "The EM Algorithm" gives the concise
statistical view; §14.3.7 places mixtures among clustering methods.

**Murphy, K. (2012). *Machine Learning: A Probabilistic Perspective*.** Chapter 11 covers mixture
models and EM with the latent-variable and MAP-regularization perspective of §9.

---

## Papers

- **Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977).** "Maximum Likelihood from Incomplete Data
  via the EM Algorithm." *JRSS-B* 39(1), 1-38. — **the EM paper**: the general framework and the
  monotonic-ascent guarantee (§4-§5). One of the most-cited statistics papers ever.
- **Neal, R. M. & Hinton, G. E. (1998).** "A View of the EM Algorithm that Justifies Incremental,
  Sparse, and Other Variants." In *Learning in Graphical Models*. — **the ELBO / free-energy view**
  (§5): EM as coordinate ascent on a lower bound, the bridge to variational inference. Free at
  <https://www.cs.toronto.edu/~radford/ftp/emk.pdf>.
- **Schwarz, G. (1978).** "Estimating the Dimension of a Model." *Annals of Statistics* 6(2). — the
  **Bayesian Information Criterion** (§8).
- **Fraley, C. & Raftery, A. E. (2002).** "Model-Based Clustering, Discriminant Analysis, and Density
  Estimation." *JASA* 97(458), 611-631. — **model-based (GMM) clustering with BIC selection** over
  both $K$ and covariance type (§6, §8); the `mclust` framework.
- **Wu, C. F. J. (1983).** "On the Convergence Properties of the EM Algorithm." *Annals of Statistics*
  11(1). — a careful analysis of when EM converges (and to what).

---

## Reference implementations

| Source | What to look at |
|---|---|
| [`sklearn.mixture.GaussianMixture`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_gaussian_mixture.py) | the E/M steps, all four covariance types, `reg_covar`, BIC/AIC; verified against here |
| [`sklearn.mixture.BayesianGaussianMixture`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_bayesian_mixture.py) | the variational / Dirichlet-process version that infers the number of components |
| [`mclust` (R)](https://cran.r-project.org/package=mclust) | Fraley & Raftery's model-based clustering with automatic BIC selection over $K$ and covariance |

---

## Deferred to later chapters

- **k-Means — the hard-assignment special case** → [04.01](../01-kmeans/)
- **Spectral clustering — for non-Gaussian, non-convex clusters** → [04.05](../05-spectral-clustering/)
- **PCA — a related latent-Gaussian model (probabilistic PCA)** → [04.06](../06-linear-dimensionality-reduction/)
- **Anomaly detection — GMM as a density model for outliers** → [04.08](../08-anomaly-detection/)
- **Variational inference — EM generalized when the posterior is intractable** → [12.xx generative models]
- **Hidden Markov models — EM (Baum-Welch) with temporal latent states** → [09.xx sequence models]
