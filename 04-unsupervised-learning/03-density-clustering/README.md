# 04.03 — Density Clustering (DBSCAN)

> **Prerequisites**: [04.01](../01-kmeans/) (clustering and its failure modes),
> [03.06](../../03-supervised-learning/06-knn/) (neighborhoods and the curse of dimensionality).
> **You will be able to**: define core / border / noise points, run DBSCAN, choose `eps` from a
> k-distance plot, explain why DBSCAN finds arbitrary shapes and labels outliers natively, and say
> exactly when its single global density threshold fails.

---

## Table of contents

1. [Clusters as dense regions](#1-clusters-as-dense-regions)
2. [Core, border, and noise points](#2-core-border-and-noise-points)
3. [Density-reachability and the cluster definition](#3-density-reachability-and-the-cluster-definition)
4. [The DBSCAN algorithm](#4-the-dbscan-algorithm)
5. [Choosing eps and minPts](#5-choosing-eps-and-minpts)
6. [Strengths and failure modes](#6-strengths-and-failure-modes)
7. [Varying density: OPTICS and HDBSCAN](#7-varying-density-optics-and-hdbscan)
8. [Complexity](#8-complexity)
9. [DBSCAN vs k-means vs hierarchical](#9-dbscan-vs-k-means-vs-hierarchical)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Clusters as dense regions

k-means and Ward define a cluster by a *center*; DBSCAN (Ester et al., 1996) defines it by *density*:

> A cluster is a **maximal region of high point density**, separated from other clusters by regions
> of low density.

This one change of definition fixes k-means' three biggest limitations at once. Because a cluster is
"a connected dense region" rather than "points near a center," DBSCAN:

- finds **arbitrarily shaped** clusters (the two moons, concentric rings) that no centroid method can;
- needs **no $k$** — the number of clusters falls out of the density structure;
- **labels outliers natively** — points in low-density regions are simply not part of any cluster,
  marked *noise*.

The price is that "density" needs two parameters — a radius and a count — and a *single* density
threshold, which is DBSCAN's own characteristic failure mode (§6). But for clean data with
well-separated clusters of similar density, DBSCAN is often the best tool available.

---

## 2. Core, border, and noise points

DBSCAN's density is local and combinatorial, defined by two parameters:

- **`eps`** ($\varepsilon$) — a neighborhood radius.
- **`minPts`** — the minimum number of points (including the point itself) that must lie within
  `eps` for a region to count as dense.

Every point is then one of three types:

- **Core point** — has **at least `minPts`** points within distance `eps`. It sits in the interior of
  a dense region and is what clusters grow from.
- **Border point** — has *fewer* than `minPts` neighbors, but lies within `eps` of a **core** point.
  It is on the edge of a cluster: part of it, but not dense enough to expand it.
- **Noise point** — neither core nor border: it is in a sparse region and belongs to **no** cluster.

That third category is what makes DBSCAN an outlier detector for free — noise points are the outliers,
identified as a byproduct of clustering rather than a separate step.

---

## 3. Density-reachability and the cluster definition

Clusters are built by connecting core points through their neighborhoods:

- A point $q$ is **directly density-reachable** from a core point $p$ if $q$ is within `eps` of $p$.
- $q$ is **density-reachable** from $p$ if there is a chain of core points $p = p_1, \dots, p_m = q$
  where each $p_{i+1}$ is directly density-reachable from $p_i$. (Reachability flows *through* core
  points, which is why a border point cannot extend a cluster.)
- Two points are **density-connected** if both are density-reachable from some common core point.

A **cluster** is then a maximal set of density-connected points. This definition — connect dense
regions through overlapping `eps`-neighborhoods — is exactly what lets a cluster snake along a curved,
non-convex shape: as long as there is a continuous chain of dense neighborhoods, the cluster follows
it, wherever it goes.

---

## 4. The DBSCAN algorithm

**Input**: data, `eps`, `minPts`.

1. For each unvisited point $p$:
   1. Mark $p$ visited. Find its `eps`-neighborhood $N(p)$.
   2. If $|N(p)| < \texttt{minPts}$, tentatively mark $p$ **noise** (it may become a border point
      later).
   3. Otherwise $p$ is a **core** point — start a new cluster $C$, add $p$, and **expand**: process a
      queue of $p$'s neighbors. For each neighbor $q$:
      - if $q$ was noise, add it to $C$ as a **border** point;
      - if $q$ is unvisited, mark it visited, add it to $C$, and if $q$ is itself a core point
        ($|N(q)| \ge \texttt{minPts}$), add $q$'s neighbors to the queue (the cluster keeps growing
        through core points).
2. Points never added to a cluster remain **noise**.

The expansion is a flood fill through connected dense neighborhoods. Note a subtlety the algorithm
exposes: a **border point reachable from two clusters** is assigned to whichever reaches it first, so
DBSCAN's clustering of border points is mildly order-dependent — the one place it is not fully
deterministic. `from_scratch.py` reproduces scikit-learn's result exactly on core points.

---

## 5. Choosing eps and minPts

DBSCAN's quality lives and dies by these two parameters:

- **`minPts`** — a smoothing / robustness knob. Larger `minPts` requires denser regions to form
  clusters, so it ignores more noise but can miss small clusters. A standard rule: `minPts` $\ge d+1$,
  and often `minPts` $= 2d$ for $d$-dimensional data; the 2D default is 4–5.
- **`eps`** — the critical one, chosen with the **k-distance plot**. Fix $k = \texttt{minPts}$, compute
  each point's distance to its $k$-th nearest neighbor, sort these distances ascending, and plot them.
  The curve is flat for points inside clusters (small $k$-distance) and rises sharply at the "knee"
  where you transition to noise points (large $k$-distance). **Set `eps` to the $k$-distance at the
  knee** — it separates dense from sparse. Experiment 4 draws this plot and reads the knee.

The k-distance plot is the closest DBSCAN has to a principled parameter choice, and it is far more
reliable than guessing. Even so, `eps` is a *single global* value — the assumption that one density
threshold fits every cluster is DBSCAN's Achilles' heel (§6–§7).

---

## 6. Strengths and failure modes

**Strengths** (the reasons to reach for DBSCAN):

- **Arbitrary cluster shapes** — connected dense regions, not blobs. Experiment 1 shows it clustering
  two moons and concentric rings perfectly, where k-means scores near zero.
- **No $k$** — the number of clusters emerges from the data.
- **Native noise/outlier labeling** — sparse points are flagged automatically (Experiment 2).
- **Robust to outliers** — outliers become noise and do not distort clusters (unlike k-means' means).

**Failure modes** (the reasons it sometimes cannot be used):

- **Varying density.** A *single* `eps`/`minPts` cannot fit clusters of different densities: an `eps`
  large enough to connect a sparse cluster will merge nearby dense clusters, and one small enough to
  separate dense clusters will shatter the sparse one into noise. Experiment 3 shows this directly —
  no single `eps` recovers clusters of very different density. This is the motivation for OPTICS and
  HDBSCAN (§7).
- **High dimensions.** DBSCAN relies on distances, and in high dimensions distances concentrate
  ([03.06](../../03-supervised-learning/06-knn/)) — every point becomes roughly equidistant, so
  "dense" loses meaning and `eps` becomes impossible to set. Reduce dimension first
  ([04.06](../06-linear-dimensionality-reduction/)).
- **Parameter sensitivity.** The result can change sharply with `eps`; too small marks everything
  noise, too large merges everything into one cluster (Experiment 4).

---

## 7. Varying density: OPTICS and HDBSCAN

The varying-density failure (§6) is important enough to have spawned two successors:

- **OPTICS** (Ordering Points To Identify the Clustering Structure) does not commit to one `eps`.
  Instead it produces a **reachability plot**: an ordering of points where valleys correspond to
  clusters and their depth reflects density. You extract clusters at *different* density thresholds
  from the same ordering, so clusters of different densities are all found. It is DBSCAN generalized
  over all `eps` at once.
- **HDBSCAN** (Hierarchical DBSCAN) builds a full hierarchy of DBSCAN clusterings across density
  levels, then extracts the clusters that are most **stable** across a range of thresholds. It needs
  essentially only `minPts` (`min_cluster_size`), handles varying density, and is often the best
  off-the-shelf density clusterer today — the practical recommendation when plain DBSCAN's single
  `eps` is too rigid.

Both keep DBSCAN's core idea (clusters are dense regions, sparse points are noise) while removing its
single-threshold limitation.

---

## 8. Complexity

Each point requires a **region query** (find all neighbors within `eps`):

- **Naive**: $O(n^2)$ — compare every pair.
- **With a spatial index** (kd-tree, ball-tree, R-tree): each query is $O(\log n)$ on low-dimensional
  data, giving $O(n\log n)$ overall — which is why DBSCAN scales far better than hierarchical
  clustering's $O(n^2)$ memory.

The index degrades in high dimensions (the curse again), where queries revert toward linear scans — a
second reason to reduce dimension before density clustering.

---

## 9. DBSCAN vs k-means vs hierarchical

| | DBSCAN | k-means | Hierarchical |
|---|---|---|---|
| **Cluster definition** | dense connected region | near a centroid | linkage-dependent |
| **Choose $k$?** | no | yes | no (cut the tree) |
| **Arbitrary shapes?** | **yes** | no | single-linkage only |
| **Outliers** | **labeled as noise** | forced into a cluster | forced into a cluster |
| **Varying density** | **fails** (one `eps`) | n/a | partial |
| **Scales to large $n$?** | yes (with index) | yes | no |
| **Parameters** | `eps`, `minPts` | $k$ | linkage, cut |

Reach for DBSCAN when clusters are irregularly shaped, of *similar* density, and possibly surrounded
by noise; for k-means when clusters are blob-like and $n$ is huge; for hierarchical when you want the
nesting; and for HDBSCAN when densities vary.

---

## 10. Common misconceptions

**"DBSCAN needs the number of clusters."**
No — $k$ emerges from the density structure (§1). You choose `eps` and `minPts` instead, which
determine how many clusters appear.

**"DBSCAN has no parameters to tune."**
It has two, and `eps` is delicate (§5). Use the k-distance plot; do not guess.

**"DBSCAN always beats k-means."**
It fails on varying-density clusters and in high dimensions, where k-means or a GMM may do better
(§6). It shines specifically on irregular shapes of similar density with noise.

**"Every point gets a cluster."**
Noise points get **no** cluster (label $-1$ in scikit-learn) — that is a feature, not a bug (§2).

**"DBSCAN is fully deterministic."**
Core-point clustering is deterministic, but **border points** reachable from two clusters are assigned
by processing order (§4).

**"Just set eps by intuition."**
`eps` interacts with the data's scale and density; the k-distance knee is the principled choice, and
the result is highly sensitive to getting it wrong (§5–§6).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — DBSCAN with core/border/noise labeling in NumPy, plus the
  k-distance plot, verified against `sklearn.cluster.DBSCAN`. Five experiments: (1) DBSCAN clustering
  two moons and rings where k-means fails (ARI); (2) native noise detection recovering clusters and
  flagging injected outliers; (3) the varying-density failure — no single `eps` works; (4) `eps`
  sensitivity and the k-distance knee; (5) the effect of `minPts`.
- **[exercises.md](exercises.md)** — define reachability precisely, implement the k-distance heuristic,
  reproduce every experiment.
- **[references.md](references.md)** — Ester et al. (DBSCAN), Ankerst et al. (OPTICS), Campello et al.
  (HDBSCAN).

**Next**: [04.04 — Gaussian Mixtures & EM](../04-gaussian-mixtures/) — soft, probabilistic clustering
with full covariance, and the EM algorithm that k-means is the hard limit of.
