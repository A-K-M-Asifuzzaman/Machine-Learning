# 04.04 — Gaussian Mixtures & the EM Algorithm

> **Prerequisites**: [04.01](../01-kmeans/) (k-means, the hard-assignment special case),
> [00.03](../../00-mathematical-foundations/03-probability/) (multivariate Gaussian, Bayes),
> [00.04](../../00-mathematical-foundations/04-statistics-and-inference/) (maximum likelihood),
> [00.02](../../00-mathematical-foundations/02-calculus-and-optimization/) (Jensen's inequality).
> **You will be able to**: write the GMM likelihood, derive the EM algorithm as ascent on a
> lower bound, implement it, choose the number of components by BIC, and explain the singularities
> that make unregularized GMMs blow up.

---

## Table of contents

1. [From hard clusters to soft, probabilistic ones](#1-from-hard-clusters-to-soft-probabilistic-ones)
2. [The generative model](#2-the-generative-model)
3. [Why maximum likelihood is hard](#3-why-maximum-likelihood-is-hard)
4. [The EM algorithm](#4-the-em-algorithm)
5. [Why EM works: the ELBO](#5-why-em-works-the-elbo)
6. [Covariance types](#6-covariance-types)
7. [k-means is hard EM](#7-k-means-is-hard-em)
8. [Choosing the number of components](#8-choosing-the-number-of-components)
9. [Singularities and how to prevent them](#9-singularities-and-how-to-prevent-them)
10. [What GMMs are good for, and their limits](#10-what-gmms-are-good-for-and-their-limits)
11. [Common misconceptions](#11-common-misconceptions)

---

## 1. From hard clusters to soft, probabilistic ones

k-means ([04.01](../01-kmeans/)) makes a **hard** decision: each point belongs to exactly one cluster,
full stop. But real data is often ambiguous — a point midway between two clusters *should* be reported
as "60% this one, 40% that one," not forced into a single bin. And k-means' spherical, equal-size
assumption cannot represent clusters that are elongated, correlated, or of different spreads.

A **Gaussian mixture model (GMM)** fixes both. It is a *probabilistic* model of the data: the clusters
are Gaussians (with their own means *and* full covariances, so they can be elongated and tilted), and
each point gets a **soft assignment** — a probability of belonging to each cluster, called its
**responsibility**. Fitting the GMM is done by the **EM algorithm**, one of the most important
algorithms in all of machine learning: a general recipe for maximum likelihood when there are hidden
variables (here, which Gaussian each point came from). Understanding EM here pays off everywhere it
recurs — HMMs, topic models, missing-data imputation, variational inference.

---

## 2. The generative model

A GMM says the data was generated like this: to make a point, first **pick a cluster** $k$ with
probability $\pi_k$ (the *mixing weights*, $\sum_k \pi_k = 1$), then **draw the point** from that
cluster's Gaussian $\mathcal{N}(\boldsymbol\mu_k, \boldsymbol\Sigma_k)$. Marginalizing over the
hidden choice gives the density of any point as a weighted sum of Gaussians:

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k, \boldsymbol\Sigma_k).
$$

The hidden cluster choice is a **latent variable** $z \in \lbrace 1,\dots,K\rbrace$ with
$p(z=k)=\pi_k$ and $p(\mathbf{x}\mid z=k) = \mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)$.
We never observe $z$ — we only see $\mathbf{x}$ — and inferring the distribution of $z$ given
$\mathbf{x}$ is exactly the "soft assignment." The parameters to learn are
$\boldsymbol\theta = \lbrace \pi_k, \boldsymbol\mu_k, \boldsymbol\Sigma_k\rbrace_{k=1}^K$.

This is more than clustering: a fitted GMM is a full **density model** $p(\mathbf{x})$, so it can also
score new points (anomaly detection, [04.08](../08-anomaly-detection/)) and generate synthetic data.

---

## 3. Why maximum likelihood is hard

The natural way to fit $\boldsymbol\theta$ is maximum likelihood — maximize the log-likelihood of the
data:

$$
\ell(\boldsymbol\theta) = \sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k).
$$

The trouble is the **log of a sum**. For a single Gaussian, the log cancels the exponential and MLE is
closed-form (the sample mean and covariance). Here the sum inside the log couples all the components,
the derivative has no closed-form zero, and the objective is **non-convex** with many local maxima.
Direct gradient ascent is possible but awkward (it must respect $\sum\pi_k=1$ and positive-definite
covariances). EM sidesteps all of this with a clean, constraint-respecting, monotonically-improving
iteration — by exploiting the latent-variable structure the log-of-sum is hiding.

---

## 4. The EM algorithm

EM alternates two steps that mirror k-means' assignment/update, but *soft*:

**E-step (Expectation)** — with parameters fixed, compute the **responsibility** of each component $k$
for each point $n$: the posterior probability that $\mathbf{x}_n$ came from component $k$, by Bayes'
rule:

$$
\gamma_{nk} = p(z_n = k\mid\mathbf{x}_n) = \frac{\pi_k\, \mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{\sum_{j=1}^{K}\pi_j\, \mathcal{N}(\mathbf{x}_n\mid\boldsymbol\mu_j,\boldsymbol\Sigma_j)}.
$$

Each $\gamma_{nk}\in[0,1]$ and $\sum_k\gamma_{nk}=1$ — the soft assignment. (k-means' hard assignment
is $\gamma_{nk}\in\lbrace0,1\rbrace$.)

**M-step (Maximization)** — with responsibilities fixed, update the parameters by
**responsibility-weighted** maximum likelihood. Let $N_k = \sum_n \gamma_{nk}$ be the *effective
number of points* in component $k$:

$$
\pi_k = \frac{N_k}{N}, \qquad \boldsymbol\mu_k = \frac{1}{N_k}\sum_n \gamma_{nk}\mathbf{x}_n, \qquad \boldsymbol\Sigma_k = \frac{1}{N_k}\sum_n \gamma_{nk}(\mathbf{x}_n-\boldsymbol\mu_k)(\mathbf{x}_n-\boldsymbol\mu_k)^\top.
$$

Each is the ordinary MLE, but with every point weighted by how much it belongs to the component. A
point that is 60% component 1 contributes 0.6 of itself to component 1's mean and covariance.

**Iterate** E and M until the log-likelihood stops increasing. Each full iteration **provably does not
decrease** $\ell$ (§5), so EM converges — to a local maximum. Like k-means, it depends on
initialization, and the standard trick is to **initialize with k-means** and then let EM soften and
refine. Experiment 2 shows the log-likelihood climbing monotonically to a plateau.

---

## 5. Why EM works: the ELBO

EM's monotonic improvement is not luck; it follows from a lower bound. For any distribution
$q(z)$ over the latent variable, Jensen's inequality gives the **evidence lower bound (ELBO)**:

$$
\ell(\boldsymbol\theta) = \sum_n \log\sum_{z} p(\mathbf{x}_n, z\mid\boldsymbol\theta) \ge \sum_n \mathbb{E}_{q(z)}\big[\log p(\mathbf{x}_n, z\mid\boldsymbol\theta)\big] + H(q) =: \mathcal{L}(q, \boldsymbol\theta).
$$

The gap between $\ell$ and the ELBO is exactly $\sum_n \mathrm{KL}\big(q(z)\,\Vert\, p(z\mid\mathbf{x}_n,\boldsymbol\theta)\big) \ge 0$.
EM is **coordinate ascent on** $\mathcal{L}(q, \boldsymbol\theta)$:

- **E-step** maximizes $\mathcal{L}$ over $q$ with $\boldsymbol\theta$ fixed. The KL gap is zero when
  $q(z) = p(z\mid\mathbf{x}_n,\boldsymbol\theta)$ — which is exactly the responsibilities $\gamma_{nk}$.
  So the E-step makes the bound **tight** (touching $\ell$ at the current $\boldsymbol\theta$).
- **M-step** maximizes $\mathcal{L}$ over $\boldsymbol\theta$ with $q$ fixed. Because the bound was
  tight before the step and can only rise, $\ell$ itself rises by at least as much.

Hence $\ell(\boldsymbol\theta^{t+1}) \ge \ell(\boldsymbol\theta^{t})$ every iteration — the guarantee.
This ELBO view is the whole of EM, and it is the same object that variational inference maximizes when
the exact posterior is intractable. Seeing EM as "make the bound tight (E), then push it up (M)" is the
single most useful mental model in latent-variable modelling.

---

## 6. Covariance types

The shape of $\boldsymbol\Sigma_k$ controls what clusters the GMM can represent, and trades
flexibility against the number of parameters:

| Covariance | Each $\boldsymbol\Sigma_k$ is | Cluster shape | Params per component ($d$ dims) |
|---|---|---|---|
| **Full** | any positive-definite matrix | elongated, tilted ellipsoids | $d(d+1)/2$ |
| **Diagonal** | diagonal | axis-aligned ellipsoids | $d$ |
| **Spherical** | $\sigma_k^2\mathbf{I}$ | balls (varying radius) | $1$ |
| **Tied** | shared $\boldsymbol\Sigma$ across all $k$ | same shape for all clusters (LDA-like) | $d(d+1)/2$ total |

**Full** covariance is the most expressive — it captures the elongated, correlated clusters that break
k-means — but needs the most data (a $d\times d$ matrix per component) and is most prone to
singularities (§9). **Diagonal** and **spherical** are cheaper and more stable in high dimensions.
**Spherical with equal variance** recovers k-means (§7). Experiment 4 shows full covariance capturing
tilted ellipses that spherical (and k-means) cannot.

---

## 7. k-means is hard EM

k-means is not a different algorithm from EM — it is EM's **zero-temperature limit**. Take a GMM with
**spherical, equal, fixed covariance** $\boldsymbol\Sigma_k = \sigma^2\mathbf{I}$ and let $\sigma^2\to0$.
Then in the E-step the responsibility

$$
\gamma_{nk} \propto \exp\!\left(-\frac{\lVert\mathbf{x}_n-\boldsymbol\mu_k\rVert^2}{2\sigma^2}\right)
$$

becomes infinitely peaked: as $\sigma^2\to0$, all the weight goes to the single nearest center, so
$\gamma_{nk}\to 1$ for the nearest $k$ and $0$ otherwise — a **hard assignment**. The M-step's
weighted mean then becomes the plain cluster mean. EM has collapsed exactly into Lloyd's algorithm
([04.01 §3](../01-kmeans/)). This is why k-means is fast but rigid (hard assignments, spherical equal
clusters) and GMM is slower but flexible (soft assignments, full covariances): **they are the same
algorithm at different temperatures**, EM being the general case. Experiment 6 measures the soft GMM
responsibilities hardening toward k-means as the covariance shrinks.

---

## 8. Choosing the number of components

Because a GMM is a probabilistic model, choosing $K$ is a **model-selection** problem with a
principled answer — unlike k-means' heuristic elbow. More components always fit the training data
better (higher likelihood), so we penalize complexity with an **information criterion**:

$$
\mathrm{BIC} = -2\,\ell(\hat{\boldsymbol\theta}) + p\log N, \qquad \mathrm{AIC} = -2\,\ell(\hat{\boldsymbol\theta}) + 2p,
$$

where $p$ is the number of free parameters (which grows with $K$ and the covariance type) and $N$ the
sample size. **Pick the $K$ that minimizes BIC** (or AIC). BIC's $\log N$ penalty is heavier, so it
prefers simpler models and is the usual choice for selecting the number of clusters. Experiment 5
shows BIC minimized at the true $K$. This is a real advantage of GMMs over k-means: the number of
clusters comes from a likelihood-based criterion, not a kink in a curve.

---

## 9. Singularities and how to prevent them

Maximum likelihood for a GMM with full covariance has a catastrophic failure mode: a component can
**collapse onto a single point**. If $\boldsymbol\mu_k$ sits exactly on one data point and
$\boldsymbol\Sigma_k\to 0$, that Gaussian becomes an infinitely tall spike, its density at that point
$\to\infty$, and the likelihood $\to\infty$. So the global maximum of the likelihood is *degenerate* —
$\ell=+\infty$ at these useless singular solutions, and EM can wander into one, with a covariance going
to zero and the log-likelihood exploding.

The fix is **regularization**: add a small constant to the diagonal of each covariance in the M-step,
$\boldsymbol\Sigma_k \leftarrow \boldsymbol\Sigma_k + \epsilon\mathbf{I}$ (scikit-learn's
`reg_covar`). This floors the variance, so no Gaussian can collapse, and turns the degenerate MLE into
a well-behaved MAP estimate (equivalently, a prior on the covariances). Experiment 3 triggers a
singularity without regularization (log-likelihood diverging) and shows the floor preventing it.
Always run GMMs with `reg_covar > 0` — the default exists for this reason.

---

## 10. What GMMs are good for, and their limits

**Good for:**

- **Soft clustering** — probabilistic assignments, honest about ambiguous points.
- **Elongated / correlated clusters** — full covariance fits tilted ellipses k-means cannot (§6).
- **Density estimation** — a fitted GMM is a smooth $p(\mathbf{x})$; useful for anomaly detection
  (score = low density) and as a generative model.
- **Model selection** — BIC gives a principled number of clusters (§8).

**Limits:**

- **Assumes Gaussian components.** Non-Gaussian clusters (curved manifolds, heavy tails) are fit
  poorly — a banana-shaped cluster needs several Gaussians to approximate. For arbitrary shapes, use
  DBSCAN ([04.03](../03-density-clustering/)) or spectral clustering ([04.05](../05-spectral-clustering/)).
- **Local optima and initialization** — like k-means; initialize with k-means and use several restarts.
- **Singularities** — require regularization (§9).
- **High dimensions** — full covariance needs $O(d^2)$ parameters per component; use diagonal/spherical
  or reduce dimension first ([04.06](../06-linear-dimensionality-reduction/)).

The one-line summary: a GMM is k-means with probabilities and ellipses — more flexible and more
informative, but slower, and only as good as the Gaussian assumption.

---

## 11. Common misconceptions

**"A GMM is just k-means with probabilities."**
It is that *and* it allows full covariances (elongated, tilted clusters) and gives a density model, not
just a partition (§1, §6). k-means is the spherical, hard, zero-variance special case (§7).

**"EM finds the global maximum likelihood."**
It finds a *local* maximum, and depends on initialization (§4). The true global maximum is actually
degenerate ($+\infty$ at singularities, §9) — you do **not** want it.

**"More components is always better."**
Training likelihood always improves with $K$; BIC/AIC penalize complexity so you pick the right $K$
(§8), not the largest.

**"Run GMM with default settings and it's fine."**
Without covariance regularization, EM can hit a singularity and blow up (§9). Keep `reg_covar > 0`.

**"EM is specific to Gaussian mixtures."**
EM is a *general* framework for maximum likelihood with latent variables — the same E/M/ELBO structure
drives HMMs, topic models, and missing-data imputation (§5). GMM is the canonical example.

**"Soft and hard clustering give the same answer."**
They differ exactly on ambiguous points near boundaries — where soft assignment reports the
uncertainty and hard assignment hides it (§1). That difference is often the point.

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — a full-covariance GMM fit by EM (k-means initialization,
  responsibility E-step, weighted M-step, `reg_covar` floor) in NumPy, verified against
  `sklearn.mixture.GaussianMixture` (log-likelihood and labels). Six experiments: (1) GMM's soft, full
  covariance beating k-means on elongated/overlapping clusters (ARI); (2) EM's monotonic
  log-likelihood ascent; (3) a covariance singularity diverging without regularization and the floor
  fixing it; (4) covariance types — spherical ≈ k-means, full captures ellipses; (5) BIC minimized at
  the true $K$; (6) responsibilities hardening toward k-means as $\sigma^2\to 0$.
- **[exercises.md](exercises.md)** — derive the M-step updates and the ELBO, implement the E/M steps,
  reproduce every experiment.
- **[references.md](references.md)** — Dempster-Laird-Rubin (EM), Bishop Ch. 9, Neal & Hinton.

**Next**: [04.05 — Spectral Clustering](../05-spectral-clustering/) — clustering via the eigenvectors
of a graph Laplacian, for the non-convex structure GMMs cannot fit.
