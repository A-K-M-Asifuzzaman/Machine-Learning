# 04.02 — Hierarchical Clustering

> **Prerequisites**: [04.01](../01-kmeans/) (the clustering problem and its failure modes),
> [00.01](../../00-mathematical-foundations/01-linear-algebra/) (distances).
> **You will be able to**: build a dendrogram by agglomerative merging, choose a linkage criterion
> from the cluster shapes you expect, read and cut a dendrogram to get $k$ clusters, and say why
> single linkage finds chains while Ward finds blobs.

---

## Table of contents

1. [Clustering without committing to k](#1-clustering-without-committing-to-k)
2. [Agglomerative vs divisive](#2-agglomerative-vs-divisive)
3. [The agglomerative algorithm](#3-the-agglomerative-algorithm)
4. [Linkage criteria](#4-linkage-criteria)
5. [The Lance-Williams update](#5-the-lance-williams-update)
6. [The dendrogram and where to cut it](#6-the-dendrogram-and-where-to-cut-it)
7. [Cophenetic correlation](#7-cophenetic-correlation)
8. [Complexity and scaling](#8-complexity-and-scaling)
9. [Hierarchical vs k-means](#9-hierarchical-vs-k-means)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Clustering without committing to k

k-means ([04.01](../01-kmeans/)) forces you to choose the number of clusters $k$ before you start.
Hierarchical clustering does not: it builds a **whole tree of nested clusterings** — from every point
in its own cluster at the bottom, to all points in one cluster at the top — and lets you choose $k$
*afterward* by cutting the tree at whatever level you like. The output is a **dendrogram**, a binary
tree whose merge heights encode how similar the merged groups were.

This buys two things k-means lacks: you see the cluster structure at *all* scales at once (are there 3
big groups, or 3 big groups each splitting into 2?), and the result is **deterministic** — no random
initialization, no local optima. The cost is scale: it needs the full pairwise distance matrix, so it
does not run on millions of points (§8). Hierarchical clustering is the tool when $n$ is modest, when
the number of clusters is genuinely unknown, and when the *nesting* of clusters is itself meaningful
(taxonomy, phylogenetics, document topics).

---

## 2. Agglomerative vs divisive

Two directions build the tree:

- **Agglomerative (bottom-up)** — start with $n$ singleton clusters and repeatedly **merge** the two
  closest clusters until one remains. This is the standard, and everything below is agglomerative.
- **Divisive (top-down)** — start with all points in one cluster and repeatedly **split**. Splitting
  optimally is expensive (choosing the best split of a cluster is itself hard), so divisive methods
  are rare in practice.

Agglomerative clustering is greedy: each merge is locally optimal (merge the closest pair now) and
never undone. Like k-means' local optimum, this greediness means the result is not globally optimal —
an early merge that looks good can be regretted later — but it is fast enough and good enough to be the
default.

---

## 3. The agglomerative algorithm

**Input**: $n$ points and a distance between points (usually Euclidean), plus a **linkage** rule for
the distance between two *clusters* (§4).

1. Start with $n$ clusters, one per point.
2. Compute all pairwise cluster distances.
3. **Merge** the two clusters with the smallest distance into one, recording the merge and its
   height (the distance at which they merged).
4. **Update** the distances from the new cluster to all others (via linkage, §5).
5. Repeat 3–4 until a single cluster remains.

The sequence of merges, with their heights, *is* the dendrogram. Cutting it at a chosen height (or to
get a chosen number of clusters) gives a flat clustering. The only real choice is the linkage rule in
step 4 — and it changes everything.

---

## 4. Linkage criteria

The **linkage** defines the distance between two clusters $A$ and $B$ from the distances between their
points. It is the single most consequential choice, because it determines what shape of cluster the
method prefers:

- **Single linkage** — the distance between the **nearest** pair:
  $d(A, B) = \min_{a\in A, b\in B} d(a, b)$. Merges clusters that have *any* close pair, so it can
  follow long, thin, non-convex shapes ("chaining"). Its strength (finds elongated clusters, like a
  minimum spanning tree) is also its weakness: a single bridge of noise points links two genuine
  clusters into one.
- **Complete linkage** — the distance between the **farthest** pair:
  $d(A, B) = \max_{a\in A, b\in B} d(a, b)$. Merges only clusters that are close *everywhere*, so it
  produces **compact, roughly spherical** clusters of similar diameter. Sensitive to outliers (one far
  point inflates the max).
- **Average linkage (UPGMA)** — the **mean** pairwise distance:
  $d(A, B) = \frac{1}{|A||B|}\sum_{a,b} d(a,b)$. A compromise between single and complete; less prone
  to chaining than single, less outlier-sensitive than complete.
- **Ward's linkage** — merge the pair that **increases the total within-cluster variance the least**.
  Equivalently, it minimizes the same sum-of-squares objective as k-means ([04.01](../01-kmeans/)), so
  it strongly prefers **compact, equal-sized, spherical** clusters and is usually the best default for
  blob-like data. It is the most popular choice.

Single vs Ward is the key contrast: single linkage chases connectivity (finds chains and non-convex
shapes, breaks on noise), Ward chases compactness (finds blobs, ignores non-convex structure).
Experiment 1 measures each linkage's clustering quality on blob-like vs chain-like data, and the
tradeoff is stark.

---

## 5. The Lance-Williams update

Recomputing linkage distances from scratch after each merge is wasteful. The **Lance-Williams
formula** updates them recursively: when clusters $i$ and $j$ merge into $i\cup j$, the distance from
the merged cluster to any other cluster $k$ is a fixed linear combination of the old distances,

$$
d(i\cup j, k) = \alpha_i\, d(i,k) + \alpha_j\, d(j,k) + \beta\, d(i,j) + \gamma\, |d(i,k) - d(j,k)|,
$$

where the coefficients $(\alpha_i, \alpha_j, \beta, \gamma)$ depend only on the linkage (and, for Ward
and average, the cluster sizes). Single linkage is $\alpha_i=\alpha_j=\tfrac12$, $\beta=0$,
$\gamma=-\tfrac12$ (which yields the min); complete linkage flips $\gamma=+\tfrac12$ (the max); Ward
has size-weighted $\alpha$'s. This one formula implements *every* linkage with a single update rule,
and `from_scratch.py` uses it — verified to reproduce scipy's dendrograms exactly.

---

## 6. The dendrogram and where to cut it

The **dendrogram** draws the merge history: leaves are points, each internal node is a merge drawn at
the *height* of that merge's linkage distance. Reading it:

- **Height** = how dissimilar the two merged groups were. Low merges join similar things; high merges
  join dissimilar things.
- **Cutting** the tree with a horizontal line at height $h$ yields the clusters that exist below $h$;
  cutting to leave $k$ branches yields $k$ clusters.

**Where to cut?** Look for the **largest vertical gap** between consecutive merge heights — a big jump
means the next merge is joining two genuinely dissimilar clusters, so cut just below it. This "biggest
gap" heuristic is the dendrogram analogue of the elbow method ([04.01 §6](../01-kmeans/)). Experiment 4
recovers the true $k$ from the largest gap in the merge heights.

---

## 7. Cophenetic correlation

How faithfully does the dendrogram represent the original distances? The **cophenetic distance**
between two points is the height at which they *first* end up in the same cluster (the height of their
lowest common ancestor). The **cophenetic correlation** is the Pearson correlation between all these
tree-distances and the original pairwise distances — a single number in $[0, 1]$ measuring how well the
hierarchy preserves the true geometry. A high value means the dendrogram is trustworthy; a low value
means the tree distorts the data and its clusters should be doubted. It is also a way to *compare
linkages*: pick the one with the highest cophenetic correlation. Experiment 5 computes it for each
linkage.

---

## 8. Complexity and scaling

The cost is the reason hierarchical clustering is a small-to-medium-$n$ tool:

- **Space**: $O(n^2)$ — the full pairwise distance matrix. This is the binding constraint: 100k points
  is a 10-billion-entry matrix, which does not fit in memory.
- **Time**: naive is $O(n^3)$; with priority queues and the Lance-Williams update, $O(n^2\log n)$;
  single linkage has an $O(n^2)$ algorithm (SLINK) via the minimum spanning tree.

So hierarchical clustering is practical up to roughly tens of thousands of points. Beyond that, use
k-means / mini-batch k-means ([04.01](../01-kmeans/)) or DBSCAN ([04.03](../03-density-clustering/)),
or cluster a representative subsample and assign the rest.

---

## 9. Hierarchical vs k-means

| | Hierarchical (agglomerative) | k-means |
|---|---|---|
| **Choose $k$ up front?** | no — cut the tree afterward | yes |
| **Output** | a full tree of nested clusters | a flat partition |
| **Determinism** | deterministic (no init) | depends on initialization |
| **Cluster shape** | depends on linkage (single: any; Ward: blobs) | spherical only |
| **Scales to large $n$?** | no ($O(n^2)$ memory) | yes |
| **Cluster centers** | none (except Ward implicitly) | explicit centroids |

Ward linkage and k-means optimize the *same* sum-of-squares objective, so on blob data they give
similar clusterings — Ward is essentially k-means with a hierarchy and no random restarts. Choose
hierarchical when $n$ is modest and you want the tree (unknown $k$, meaningful nesting); choose k-means
when $n$ is large and you know roughly how many clusters you want.

---

## 10. Common misconceptions

**"Hierarchical clustering doesn't need any choices."**
It needs the **linkage**, which is as consequential as $k$ is for k-means (§4): single vs Ward give
completely different clusterings.

**"Single linkage is best because it finds any shape."**
Its chaining finds non-convex clusters but breaks catastrophically on noise — one bridge of points
merges two real clusters. Use it only on clean, well-separated data (§4).

**"The dendrogram tells you the number of clusters."**
It suggests candidates (via the largest merge-height gap), but $k$ is still a judgment call (§6), just
as with the elbow method.

**"It's deterministic, so it's more reliable than k-means."**
Determinism is not correctness — the greedy merges are not globally optimal, and a bad linkage choice
gives a bad, *reproducibly* bad, clustering (§2).

**"Ward and k-means are unrelated."**
Both minimize within-cluster sum of squares; Ward is the agglomerative version and gives similar
results on blobs (§9).

**"It scales like k-means."**
No — the $O(n^2)$ distance matrix caps it at tens of thousands of points (§8).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — agglomerative clustering with single / complete / average /
  Ward linkage via the Lance-Williams update, the dendrogram (merge list + heights), cophenetic
  correlation, and dendrogram cutting, all in NumPy, verified against `scipy.cluster.hierarchy`. Five
  experiments: (1) linkage comparison on blobs vs chains (ARI); (2) single-linkage chaining finding
  non-convex clusters and breaking on a noise bridge; (3) Ward ≈ k-means on blobs; (4) recovering $k$
  from the largest merge-height gap; (5) cophenetic correlation per linkage.
- **[exercises.md](exercises.md)** — derive the Lance-Williams coefficients, implement the dendrogram
  and cophenetic distance, reproduce every experiment.
- **[references.md](references.md)** — Ward, Lance & Williams, Müllner's fast algorithms.

**Next**: [04.03 — Density Clustering (DBSCAN)](../03-density-clustering/) — clusters as dense regions
separated by sparse ones, with no $k$ and native outlier detection.
