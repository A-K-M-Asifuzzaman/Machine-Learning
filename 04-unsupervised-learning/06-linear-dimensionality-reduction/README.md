# 04.06 — Linear Dimensionality Reduction (PCA)

> **Prerequisites**: [00.01](../../00-mathematical-foundations/01-linear-algebra/) (eigenvectors, SVD,
> symmetric matrices), [00.03](../../00-mathematical-foundations/03-probability/) (covariance),
> [00.04](../../00-mathematical-foundations/04-statistics-and-inference/) (variance).
> **You will be able to**: derive PCA two ways (max variance and min reconstruction error), compute
> it by covariance eigendecomposition and by SVD, choose the number of components from explained
> variance, and say precisely when PCA helps and when it fails.

---

## Table of contents

1. [The goal: fewer dimensions, most of the information](#1-the-goal-fewer-dimensions-most-of-the-information)
2. [Derivation 1: maximum variance](#2-derivation-1-maximum-variance)
3. [Derivation 2: minimum reconstruction error](#3-derivation-2-minimum-reconstruction-error)
4. [The covariance eigendecomposition](#4-the-covariance-eigendecomposition)
5. [PCA via the SVD](#5-pca-via-the-svd)
6. [Choosing the number of components](#6-choosing-the-number-of-components)
7. [Centering, scaling, and whitening](#7-centering-scaling-and-whitening)
8. [What PCA assumes, and where it fails](#8-what-pca-assumes-and-where-it-fails)
9. [Interpretation and uses](#9-interpretation-and-uses)
10. [Beyond linear PCA](#10-beyond-linear-pca)
11. [Common misconceptions](#11-common-misconceptions)

---

## 1. The goal: fewer dimensions, most of the information

High-dimensional data is hard to store, visualize, and model — and much of it is redundant, because
features are correlated. **Principal Component Analysis** finds a small set of new axes — the
*principal components* — that capture as much of the data's variation as possible, and projects the
data onto them. It answers: *if I could keep only $k$ directions, which $k$ retain the most
information?*

PCA's answer is: keep the directions of **maximum variance**. The first principal component is the
single direction along which the data varies most; the second is the direction of most *remaining*
variance orthogonal to the first; and so on. Projecting onto the top $k$ of these gives the best
$k$-dimensional linear summary of the data. This one idea powers visualization (project to 2–3D),
compression, denoising, decorrelation, and preprocessing for nearly every other algorithm. And
remarkably, "maximum variance" and "minimum information lost" turn out to be the *same* objective
(§2–§3), solved by a single eigendecomposition.

---

## 2. Derivation 1: maximum variance

Center the data (subtract the mean, §7). The first principal component is the unit vector
$\mathbf{w}$ that maximizes the variance of the projected data $\mathbf{X}\mathbf{w}$:

$$
\max_{\lVert\mathbf{w}\rVert=1}\ \mathrm{Var}(\mathbf{X}\mathbf{w}) = \max_{\lVert\mathbf{w}\rVert=1}\ \mathbf{w}^\top \mathbf{C}\, \mathbf{w}, \qquad \mathbf{C} = \tfrac{1}{n}\mathbf{X}^\top\mathbf{X}.
$$

$\mathbf{C}$ is the sample **covariance matrix**. Maximizing $\mathbf{w}^\top\mathbf{C}\mathbf{w}$
subject to $\lVert\mathbf{w}\rVert=1$ is a constrained optimization; the Lagrangian
$\mathbf{w}^\top\mathbf{C}\mathbf{w} - \lambda(\mathbf{w}^\top\mathbf{w}-1)$ has stationary condition

$$
\mathbf{C}\mathbf{w} = \lambda\mathbf{w}.
$$

So the optimal $\mathbf{w}$ is an **eigenvector of the covariance matrix**, and the variance it
captures is its **eigenvalue** $\lambda = \mathbf{w}^\top\mathbf{C}\mathbf{w}$. To maximize, pick the
eigenvector with the *largest* eigenvalue — the first principal component. The second PC maximizes
variance subject to being orthogonal to the first, giving the second-largest eigenvector, and so on.
**The principal components are the eigenvectors of the covariance matrix, ordered by eigenvalue.**

---

## 3. Derivation 2: minimum reconstruction error

A completely different-sounding goal gives the *same* answer. Suppose we want the $k$-dimensional
subspace onto which projecting the data loses the least — minimizes the squared **reconstruction
error**. Writing $\mathbf{W}_k$ for the orthonormal basis of the subspace, we project each point down
($\mathbf{z} = \mathbf{W}_k^\top\mathbf{x}$) and reconstruct it back ($\hat{\mathbf{x}} = \mathbf{W}_k\mathbf{z}$),
and minimize

$$
\min_{\mathbf{W}_k^\top\mathbf{W}_k = \mathbf{I}}\ \sum_{n} \lVert \mathbf{x}_n - \mathbf{W}_k\mathbf{W}_k^\top\mathbf{x}_n \rVert^2.
$$

By the Pythagorean theorem, total variance = variance kept (in the subspace) + variance lost
(reconstruction error). So **minimizing reconstruction error is exactly maximizing retained
variance** — the same problem as §2, with the same solution: $\mathbf{W}_k$ = the top $k$
eigenvectors of $\mathbf{C}$. The reconstruction error equals the **sum of the discarded eigenvalues**,
$\sum_{j>k}\lambda_j$ — an exact identity `from_scratch.py` verifies (Experiment 3). That the
"compress with least loss" and "find the wiggliest directions" objectives coincide is the elegant
heart of PCA.

---

## 4. The covariance eigendecomposition

The direct algorithm follows from §2:

1. **Center** the data: $\mathbf{X} \leftarrow \mathbf{X} - \bar{\mathbf{x}}$ (and usually scale, §7).
2. **Covariance**: $\mathbf{C} = \tfrac{1}{n}\mathbf{X}^\top\mathbf{X}$ (a $d\times d$ symmetric PSD
   matrix).
3. **Eigendecompose**: $\mathbf{C} = \mathbf{V}\boldsymbol\Lambda\mathbf{V}^\top$, with eigenvectors
   $\mathbf{V}$ (the principal components) and eigenvalues $\boldsymbol\Lambda$ (the variances) sorted
   descending.
4. **Project**: keep the top $k$ eigenvectors $\mathbf{V}_k$ and transform
   $\mathbf{Z} = \mathbf{X}\mathbf{V}_k$ (the *scores*, the data in PC coordinates).

The eigenvalues give the **explained variance**: $\lambda_j / \sum_i\lambda_i$ is the fraction of
total variance captured by PC $j$ (§6). This is conceptually clean but numerically not ideal — forming
$\mathbf{X}^\top\mathbf{X}$ squares the condition number, losing precision on ill-conditioned data.
The SVD (§5) avoids that.

---

## 5. PCA via the SVD

The **singular value decomposition** of the centered data, $\mathbf{X} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top$,
gives PCA directly and more stably. Because
$\mathbf{X}^\top\mathbf{X} = \mathbf{V}\boldsymbol\Sigma^2\mathbf{V}^\top$, the **right singular
vectors $\mathbf{V}$ are exactly the principal components**, and the singular values relate to the
eigenvalues by $\lambda_j = \sigma_j^2 / n$. So:

- Principal components = right singular vectors $\mathbf{V}$.
- Explained variance of PC $j$ = $\sigma_j^2 / n$.
- Scores = $\mathbf{X}\mathbf{V} = \mathbf{U}\boldsymbol\Sigma$.

The SVD **never forms $\mathbf{X}^\top\mathbf{X}$**, so it does not square the condition number — it is
numerically superior and is what scikit-learn and every serious implementation use. Experiment 4 shows
the covariance-eigendecomposition losing accuracy on ill-conditioned data where the SVD holds. For
$n \ll d$ (few samples, many features), the SVD also works on the smaller Gram matrix, saving compute.
**Compute PCA with the SVD.**

---

## 6. Choosing the number of components

The eigenvalues tell you how many components to keep:

- **Explained-variance ratio.** $\lambda_j/\sum_i\lambda_i$ is PC $j$'s share of total variance. Plot
  the cumulative sum and keep enough PCs to reach a threshold — **95%** (or 90%, 99%) is common.
- **Scree plot / elbow.** Plot eigenvalues in descending order and look for the "elbow" where they
  level off; components past it capture little. Like k-means' elbow, it is a visual heuristic.
- **Kaiser criterion.** Keep PCs with eigenvalue $> 1$ (on standardized data, i.e. capturing more than
  one original feature's worth of variance). A rough rule, often over- or under-selects.

For **visualization** you keep 2 or 3 regardless; for **compression/preprocessing** you pick $k$ by the
variance threshold. Experiment 2 draws the scree and cumulative-variance curves and reads $k$ off them.
Unlike clustering's $k$, this choice is well-grounded: the eigenvalues directly quantify what each
component is worth.

---

## 7. Centering, scaling, and whitening

Three preprocessing points, in order of importance:

- **Centering is mandatory.** PCA finds directions of variance *about the mean*; without subtracting
  the mean, the first component points at the mean vector, not the variation. Always center.
- **Scaling is usually necessary.** PCA maximizes variance, and variance depends on units — a feature
  measured in millimeters has $10^6\times$ the variance of the same feature in meters, so it would
  dominate the components for no good reason. **Standardize** (zero mean, unit variance) unless the
  features are already in comparable units. This is equivalent to running PCA on the *correlation*
  matrix instead of the covariance matrix. Experiment 5 shows the principal components changing
  entirely when one feature's scale is inflated.
- **Whitening** (optional) rescales each PC to unit variance: $\mathbf{Z} \leftarrow \mathbf{Z}/\sqrt{\lambda}$.
  This *decorrelates and equalizes* the features — useful as preprocessing for algorithms that assume
  isotropic inputs, at the cost of amplifying low-variance (often noisy) directions.

The scaling decision is the one people get wrong most: on heterogeneous features, unscaled PCA
silently reports the loudest units as the "principal" structure.

---

## 8. What PCA assumes, and where it fails

PCA is powerful but narrow, and its assumptions are strong:

- **It is linear.** PCA finds a linear subspace, so it captures only *linear* correlations. Data on a
  curved manifold — a Swiss roll, a circle, a spiral — is not summarized by any linear projection;
  PCA flattens it and loses the structure. Experiment 7 shows PCA failing to unroll a nonlinear
  manifold, motivating kernel PCA and manifold learning ([04.07](../07-manifold-learning/)).
- **Variance is assumed to equal importance.** PCA keeps high-variance directions, but variance is not
  the same as *usefulness* — a low-variance direction can carry the discriminative signal, and a
  high-variance one can be noise. PCA is unsupervised: it does not know your task. (For a supervised
  linear projection, use LDA, [03.05](../../03-supervised-learning/05-generative-classifiers/).)
- **Sensitive to scale and outliers.** Unscaled features and gross outliers both distort the
  components (§7); use standardization and, for contaminated data, robust PCA.
- **Components are linear combinations of all features** — often hard to interpret (§9).

Know the failure modes: PCA is the right tool for decorrelated linear compression and the wrong tool
for nonlinear structure or task-specific projection.

---

## 9. Interpretation and uses

**Interpretation.** Each PC is a **loading vector** — a weighted combination of the original features.
Large-magnitude loadings show which features drive that component; a *biplot* overlays the loadings on
the score scatter. But because a PC mixes all features, it is often not cleanly interpretable, which
is a genuine drawback (sparse PCA, §10, trades some variance for interpretability).

**Uses**, all following from "linear compression that keeps variance":

- **Visualization** — project to 2–3 PCs to see high-dimensional structure.
- **Compression / denoising** — keep the top $k$ PCs; the discarded low-variance directions are often
  noise, so reconstruction *denoises* (Experiment 6).
- **Decorrelation / whitening** — feed decorrelated features to downstream models (§7).
- **Preprocessing** — reduce dimension before clustering ([04.03](../03-density-clustering/) in high-d),
  regression, or nearest neighbors, fighting the curse of dimensionality.
- **Eigenfaces / eigen-anything** — PCA on images gives a compact basis for faces, digits, etc.

---

## 10. Beyond linear PCA

PCA is the linear base case of a large family:

- **Kernel PCA** — run PCA in a feature space via a kernel, capturing *nonlinear* structure (the
  circle, the Swiss roll). The dimensionality-reduction cousin of the kernel trick
  ([03.07](../../03-supervised-learning/07-svm/)).
- **Probabilistic PCA / factor analysis** — PCA as a latent-Gaussian generative model (a linear-Gaussian
  cousin of the GMM, [04.04](../04-gaussian-mixtures/)); handles missing data and gives a likelihood.
- **Sparse PCA** — components with few nonzero loadings, for interpretability (§9).
- **Randomized / incremental PCA** — fast approximate SVD for huge or streaming data.
- **Robust PCA** — decomposes data into low-rank + sparse, robust to gross outliers.
- **Autoencoders** — a neural network with a bottleneck is *nonlinear PCA*; a linear autoencoder with
  squared loss recovers PCA exactly ([07.xx](../../07-deep-learning/)).
- **Manifold learning** (t-SNE, UMAP, Isomap, LLE) — nonlinear embeddings for visualization
  ([04.07](../07-manifold-learning/)).

---

## 11. Common misconceptions

**"PCA selects the most important features."**
It creates new features (linear combinations of all originals), it does not select among the existing
ones (§9). For feature *selection*, use a different method.

**"You don't need to scale before PCA."**
On features with different units you almost always must standardize, or the highest-variance *unit*
dominates the components for no real reason (§7).

**"High variance means important."**
PCA keeps high-variance directions, but variance ≠ task-relevance — the signal can live in a
low-variance direction, and noise in a high-variance one (§8). PCA is unsupervised.

**"The covariance eigendecomposition and the SVD are equivalent, so use either."**
Mathematically equivalent, numerically not: the SVD avoids squaring the condition number and is
strictly preferred (§5, Experiment 4).

**"PCA works on any data."**
It captures only *linear* structure; nonlinear manifolds need kernel PCA or manifold learning (§8, §10).

**"The principal components are interpretable."**
They are linear combinations of all features and often are not cleanly interpretable; sparse PCA
trades variance for interpretability (§9).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — PCA via both covariance eigendecomposition and SVD in
  NumPy, verified against `sklearn.decomposition.PCA` (components up to sign, explained variance,
  transform). Seven experiments: (1) the two derivations (max variance = min reconstruction) giving
  the same components; (2) explained-variance and scree curves for choosing $k$; (3) the exact
  identity reconstruction error = sum of discarded eigenvalues; (4) SVD's numerical superiority on
  ill-conditioned data; (5) scaling changing the components entirely; (6) compression/denoising by
  truncated reconstruction; (7) PCA failing on a nonlinear manifold.
- **[exercises.md](exercises.md)** — derive both objectives, prove the reconstruction identity,
  implement PCA by SVD, reproduce every experiment.
- **[references.md](references.md)** — Pearson, Hotelling, Jolliffe's PCA book, Tipping & Bishop (PPCA).

**Next**: [04.07 — Manifold Learning](../07-manifold-learning/) — nonlinear dimensionality reduction
(t-SNE, UMAP, Isomap) for the curved structure PCA cannot capture.
