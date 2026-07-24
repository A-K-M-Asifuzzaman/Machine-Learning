# 04.01 — k-Means Clustering

> **Prerequisites**: [00.01](../../00-mathematical-foundations/01-linear-algebra/) (norms, distances),
> [00.02](../../00-mathematical-foundations/02-calculus-and-optimization/) (coordinate descent),
> [00.03](../../00-mathematical-foundations/03-probability/) (the Gaussian k-means is a hard special
> case of — [04.04](../04-gaussian-mixtures/)).
> **You will be able to**: derive Lloyd's algorithm as coordinate descent on the within-cluster
> sum of squares, explain why k-means++ initialization matters, choose $k$ from the data, and say
> precisely which cluster shapes k-means cannot find.

---

## Table of contents

1. [The clustering problem](#1-the-clustering-problem)
2. [The k-means objective](#2-the-k-means-objective)
3. [Lloyd's algorithm](#3-lloyds-algorithm)
4. [Why it converges — and only to a local optimum](#4-why-it-converges--and-only-to-a-local-optimum)
5. [Initialization: k-means++](#5-initialization-k-means)
6. [Choosing k](#6-choosing-k)
7. [What k-means assumes, and where it fails](#7-what-k-means-assumes-and-where-it-fails)
8. [k-means is hard-assignment GMM](#8-k-means-is-hard-assignment-gmm)
9. [Variants](#9-variants)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. The clustering problem

Clustering is the canonical **unsupervised** task: given data $\lbrace\mathbf{x}_1,\dots,\mathbf{x}_n\rbrace$
with **no labels**, partition it into groups so that points in the same group are similar and points
in different groups are not. There is no ground truth to fit — "similar" is a modelling choice, and
different definitions of similarity give genuinely different, equally valid clusterings. This is what
makes clustering both useful (it finds structure you did not specify) and treacherous (there is no
single right answer, and every method imposes its own idea of what a cluster is).

k-means is the simplest and most used clustering method. Its idea of a cluster is specific and worth
stating up front: **a cluster is a group of points near a common center**, and "near" means squared
Euclidean distance. Everything about k-means — its speed, its algorithm, and its failure modes —
follows from that one definition.

---

## 2. The k-means objective

k-means partitions the data into $k$ clusters $C_1,\dots,C_k$ with centers (centroids)
$\boldsymbol\mu_1,\dots,\boldsymbol\mu_k$, chosen to minimize the **within-cluster sum of squares**
(WCSS), also called *inertia*:

$$
J(\lbrace C_k\rbrace, \lbrace\boldsymbol\mu_k\rbrace) = \sum_{k=1}^{K}\sum_{\mathbf{x}_i \in C_k} \lVert \mathbf{x}_i - \boldsymbol\mu_k \rVert^2 .
$$

Every point contributes the squared distance to its cluster's center; $J$ is the total. Minimizing it
asks for a partition where points sit as close as possible to their center. Two facts about this
objective drive everything:

- **The squared distance** makes the mean the optimal center (§3) and makes k-means, like the mean
  itself, sensitive to outliers — a single far point pulls its centroid.
- **The problem is NP-hard** to solve exactly (for general $k$ and dimension): there are exponentially
  many partitions and $J$ is non-convex in the assignment. So we do not solve it exactly; we descend
  to a *local* minimum with Lloyd's algorithm (§3–§4).

---

## 3. Lloyd's algorithm

The standard k-means algorithm (Lloyd, 1957) minimizes $J$ by **alternating** between the two sets of
variables — the assignments and the centroids — holding one fixed while optimizing the other:

**Initialize** $k$ centroids (§5). Then repeat until assignments stop changing:

1. **Assignment step** — fix the centroids; assign each point to its **nearest** centroid:
   $C_k = \lbrace \mathbf{x}_i : k = \arg\min_j \lVert \mathbf{x}_i - \boldsymbol\mu_j \rVert^2 \rbrace$.
   This minimizes $J$ over the assignments with centroids fixed.
2. **Update step** — fix the assignments; move each centroid to the **mean** of its points:
   $\boldsymbol\mu_k = \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$.
   This minimizes $J$ over the centroids with assignments fixed, because the mean is the unique
   minimizer of $\sum_i \lVert \mathbf{x}_i - \mathbf{c} \rVert^2$ (set the gradient to zero:
   $\sum_i (\mathbf{x}_i - \mathbf{c}) = 0 \Rightarrow \mathbf{c} = \bar{\mathbf{x}}$).

Each step is an exact minimization over one block of variables with the other fixed — so Lloyd's
algorithm is **coordinate descent** ([00.02](../../00-mathematical-foundations/02-calculus-and-optimization/))
on $J$. The assignment step draws a **Voronoi partition** of space (each cluster is the region closest
to its centroid), so k-means boundaries are always straight lines / hyperplanes — a fact that explains
its failure modes (§7).

---

## 4. Why it converges — and only to a local optimum

**Convergence.** Both steps *decrease* (never increase) $J$: the assignment step moves each point to a
closer center, and the update step moves each center to the point minimizing its cluster's spread. So
$J$ is monotonically non-increasing across iterations. Since there are only finitely many possible
assignments ($k^n$) and $J$ is bounded below by 0, the algorithm cannot cycle and must **converge in
a finite number of steps** — usually very few. Experiment 2 shows $J$ dropping monotonically to a
plateau.

**Only local.** Convergence is to a *local* minimum, not the global one. The objective is non-convex,
and the partition Lloyd's algorithm lands in depends entirely on the initialization — a bad start
gives a bad clustering that is nonetheless a fixed point. This is why **initialization is the single
most important practical detail** (§5), and why k-means is run several times with different seeds,
keeping the lowest-inertia result (`n_init` in scikit-learn).

---

## 5. Initialization: k-means++

Naive initialization — pick $k$ random data points as centroids — frequently produces bad local
optima: two initial centers can land in the same true cluster while another true cluster gets none,
and Lloyd's algorithm cannot recover. **k-means++** (Arthur & Vassilvitskii, 2007) fixes this by
seeding centers that are **spread out**:

1. Pick the first center uniformly at random from the data.
2. For each remaining center, pick a data point with probability **proportional to its squared
   distance** $D(\mathbf{x})^2$ to the nearest already-chosen center. Points far from all current
   centers are far more likely to be picked.
3. Repeat until $k$ centers are chosen, then run Lloyd's algorithm.

The $D^2$ weighting makes it likely that each true cluster gets its own seed, so Lloyd's algorithm
starts near a good partition. This is not just a heuristic: k-means++ gives an **$O(\log k)$
approximation** to the optimal inertia in expectation, versus no guarantee for random init. It is the
default in every serious implementation, and Experiment 1 shows it reaching lower inertia far more
reliably than random seeding, with far fewer catastrophic runs.

---

## 6. Choosing k

$k$ is a hyperparameter, and with no labels there is no cross-validation score to optimize — choosing
it is inherently heuristic. The standard tools:

- **The elbow method.** Plot inertia $J$ against $k$. Inertia always decreases as $k$ grows (more
  centers fit tighter), reaching 0 at $k = n$. The "elbow" — the $k$ where the decrease sharply
  flattens — marks diminishing returns, a reasonable $k$. It is often ambiguous, which is its main
  weakness.
- **Silhouette score.** For each point, compare its mean distance to its own cluster ($a$) against the
  mean distance to the nearest *other* cluster ($b$): $s = (b - a)/\max(a, b) \in [-1, 1]$. High
  silhouette means tight, well-separated clusters. Averaged over points, it gives a score per $k$;
  pick the $k$ that maximizes it. More principled than the elbow, and it also flags points that may be
  misclustered ($s < 0$).
- **Gap statistic.** Compare the observed inertia to the inertia expected under a null (uniform)
  reference distribution; the $k$ with the largest gap is chosen. More rigorous, more expensive.

Experiment 3 computes the elbow and silhouette on data with a known $k$ and shows both pointing to the
right answer. In practice, use silhouette or the gap statistic when the elbow is unclear, and always
remember that **the "best" $k$ depends on what you mean by a cluster** — domain knowledge beats any
statistic.

---

## 7. What k-means assumes, and where it fails

k-means is fast and often excellent, but its cluster model is narrow, and knowing its assumptions is
what separates using it from misusing it. Minimizing squared Euclidean distance to a center implicitly
assumes clusters that are:

- **Spherical (isotropic).** Equal spread in every direction. k-means cannot represent elongated or
  correlated clusters — its Voronoi boundaries are straight, so it slices an elongated cluster in half
  rather than following its shape.
- **Equal-sized (in variance).** The squared-distance objective favors clusters of similar spread; a
  tight cluster next to a diffuse one gets mis-split, with the boundary drawn too close to the diffuse
  cluster's points.
- **Roughly equal in population.** Because a large cluster contributes more total inertia, k-means
  tends to carve big clusters and merge small ones to balance the objective.
- **Linearly separable by Voronoi cells / convex.** Non-convex shapes (two interlocking moons,
  concentric rings) are impossible: no set of centers produces those regions with straight boundaries.

Experiment 4 measures k-means failing (low Adjusted Rand Index against the truth) on elongated,
unequal-variance, and non-convex data — exactly the cases that motivate Gaussian mixtures
([04.04](../04-gaussian-mixtures/), which allow full covariance), density clustering
([04.03](../03-density-clustering/), which follows arbitrary shapes), and spectral clustering
([04.05](../05-spectral-clustering/), which handles non-convex structure). k-means is the right first
tool for compact, well-separated, blob-like clusters and the wrong tool for anything else.

---

## 8. k-means is hard-assignment GMM

k-means is not an isolated trick — it is the **hard-assignment, equal-spherical-covariance limit of a
Gaussian mixture model** ([04.04](../04-gaussian-mixtures/)). A GMM fit by EM does exactly Lloyd's two
steps in *soft* form: the E-step assigns each point a *probability* of belonging to each cluster
(instead of a hard nearest-center assignment), and the M-step updates means (and covariances and
mixing weights) as *weighted* averages. Take the GMM, fix every covariance to $\sigma^2 \mathbf{I}$
with $\sigma^2 \to 0$, and the soft responsibilities collapse to hard 0/1 assignments — recovering
k-means exactly.

This connection is worth holding onto: it explains k-means' spherical assumption (fixed isotropic
covariance), tells you what to reach for when that assumption breaks (a full-covariance GMM), and
frames both as instances of the **EM algorithm** ([04.04](../04-gaussian-mixtures/)) — alternating
between inferring assignments and updating parameters. Lloyd's algorithm *is* EM with the temperature
turned to zero.

---

## 9. Variants

The k-means idea spawns a family, each relaxing one assumption:

| Variant | Change | Good for |
|---|---|---|
| **k-means++** | $D^2$ seeding (§5) | the default — always use it |
| **k-medoids (PAM)** | centers are actual data points; any distance metric | robustness to outliers; non-Euclidean distances |
| **k-medians** | minimize $L_1$; centers are per-dimension medians | outlier resistance |
| **Mini-batch k-means** | update on random mini-batches | very large datasets (scalability) |
| **Fuzzy c-means** | soft membership weights | overlapping clusters, uncertainty |
| **Kernel k-means** | k-means in a feature space via a kernel | non-linear cluster boundaries |

**k-medoids** deserves emphasis: by using an actual point as each center (a *medoid*) and any distance
you like, it is robust to outliers (which cannot drag a medoid the way they drag a mean) and works on
data where only pairwise distances are defined. Experiment 5 shows k-medoids holding its clustering
under outliers that break k-means. **Kernel k-means** connects to spectral clustering
([04.05](../05-spectral-clustering/)); **fuzzy c-means** connects to the soft GMM (§8).

---

## 10. Common misconceptions

**"k-means finds the globally optimal clustering."**
It finds a *local* minimum of a non-convex objective; the result depends on initialization (§4).
Run it multiple times (`n_init`) and keep the lowest inertia.

**"Random initialization is fine."**
Random seeding routinely produces bad local optima; k-means++ spreads the seeds and gives an
$O(\log k)$ guarantee (§5). Use it — it is the default for a reason.

**"Lower inertia always means a better clustering."**
Inertia decreases monotonically with $k$ and reaches 0 at $k=n$, so you cannot pick $k$ by minimizing
it (§6). Use the elbow, silhouette, or gap statistic.

**"k-means works on any data."**
It assumes spherical, similarly-sized, convex clusters (§7). On elongated, unequal-density, or
non-convex data it fails — use a GMM, DBSCAN, or spectral clustering instead.

**"k-means and GMM are unrelated."**
k-means is GMM with hard assignments and fixed isotropic covariance in the zero-variance limit (§8).
They are the same algorithm at different temperatures.

**"You should scale features before k-means... maybe."**
You almost always should. k-means uses Euclidean distance, so a feature on a larger scale dominates
the objective. Standardize features unless their scales are meaningfully comparable
([02.04](../../02-data/04-scaling-and-transformation/)).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — k-means with k-means++ initialization and k-medoids in
  NumPy, verified against `sklearn.cluster.KMeans` (inertia and labels up to permutation). Five
  experiments: (1) k-means++ vs random init — lower inertia, fewer catastrophic runs; (2) Lloyd's
  monotonic descent of $J$; (3) choosing $k$ by elbow and silhouette; (4) the failure modes —
  elongated, unequal-variance, and non-convex clusters measured by Adjusted Rand Index; (5) k-medoids'
  robustness to outliers vs k-means.
- **[exercises.md](exercises.md)** — derive the mean as the optimal center, prove convergence,
  implement k-means++ and the silhouette score, reproduce every experiment.
- **[references.md](references.md)** — Lloyd, Arthur & Vassilvitskii (k-means++), the elbow/silhouette
  literature.

**Next**: [04.02 — Hierarchical Clustering](../02-hierarchical-clustering/) — clustering without
committing to $k$ up front, building a full tree of nested clusters instead.
