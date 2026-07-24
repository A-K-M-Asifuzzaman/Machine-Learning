# 04.07 — Manifold Learning

> **Prerequisites**: [04.06](../06-linear-dimensionality-reduction/) (PCA, the linear baseline),
> [04.05](../05-spectral-clustering/) (Laplacian eigenmaps — the same eigenvectors),
> [00.05](../../00-mathematical-foundations/05-information-theory/) (KL divergence, for t-SNE).
> **You will be able to**: state the manifold hypothesis, unfold a nonlinear manifold with Isomap,
> run t-SNE/UMAP for visualization, and — most importantly — read a t-SNE plot without being fooled
> by its artifacts.

---

## Table of contents

1. [The manifold hypothesis](#1-the-manifold-hypothesis)
2. [Two families of methods](#2-two-families-of-methods)
3. [Isomap: geodesic distances](#3-isomap-geodesic-distances)
4. [LLE and Laplacian eigenmaps](#4-lle-and-laplacian-eigenmaps)
5. [t-SNE](#5-t-sne)
6. [UMAP](#6-umap)
7. [How to read a t-SNE / UMAP plot — the cautions](#7-how-to-read-a-t-sne--umap-plot--the-cautions)
8. [Choosing a method](#8-choosing-a-method)
9. [Limits](#9-limits)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. The manifold hypothesis

Real high-dimensional data — images, audio, text embeddings — rarely fills its space. A 100×100
grayscale image lives in a 10,000-dimensional space, but the set of *natural face images* forms a
tiny, curved, low-dimensional surface within it: change pose, lighting, expression, and you move
along a handful of intrinsic directions. This is the **manifold hypothesis**: high-dimensional data
concentrates near a low-dimensional, usually *nonlinear*, manifold.

PCA ([04.06](../06-linear-dimensionality-reduction/)) cannot capture this, because a manifold is
curved and PCA only finds *linear* subspaces — flatten a rolled-up sheet of paper with a linear
projection and you get overlapping garbage. **Manifold learning** is nonlinear dimensionality
reduction: methods that *unfold* the manifold, mapping the curved high-dimensional structure to a
faithful low-dimensional representation. They power the 2-D scatter plots that reveal clusters in
high-dimensional data — with important caveats about how to read them (§7).

---

## 2. Two families of methods

Manifold methods split by *what they try to preserve*, which determines what they are good for:

- **Structure-preserving embeddings** (Isomap, LLE, Laplacian eigenmaps) preserve *distances* or
  *local geometry* and produce a genuine low-dimensional coordinate system. They are true
  dimensionality reduction — the output is meaningful for downstream use, and often has a clear
  intrinsic interpretation (the angle and height of a Swiss roll).
- **Neighbor-embedding methods for visualization** (t-SNE, UMAP) preserve *neighborhoods* — who is
  close to whom — and are optimized to make clusters *visually* separate in 2-D. They produce
  stunning plots but distort global geometry (distances, sizes, densities), so their output is for
  *looking at*, not for computing with (§7).

The distinction matters: reach for Isomap/LLE when you need a faithful low-dimensional representation,
and t-SNE/UMAP when you need to *see* the cluster structure of high-dimensional data.

---

## 3. Isomap: geodesic distances

Isomap's insight: on a curved manifold, straight-line (Euclidean) distance is misleading — two points
on opposite ends of a rolled-up Swiss roll are close in space but far *along the surface*. The right
distance is the **geodesic** — the distance measured *along the manifold*. Isomap approximates it and
then embeds:

1. **Neighborhood graph.** Connect each point to its $k$ nearest neighbors, weighting edges by
   Euclidean distance (which is accurate *locally*, for nearby points).
2. **Geodesic distances.** Compute shortest-path distances between all pairs in this graph (Dijkstra /
   Floyd-Warshall). A path that hops neighbor-to-neighbor follows the manifold, so the graph distance
   approximates the geodesic.
3. **Classical MDS.** Find the low-dimensional coordinates whose Euclidean distances best match those
   geodesic distances — an eigendecomposition of the double-centered squared-distance matrix.

The result **unfolds** the manifold: the Swiss roll becomes a flat rectangle whose axes are the true
intrinsic coordinates. Experiment 1 shows Isomap recovering the roll's underlying parameter where PCA
(which sees only the tangled 3-D shape) cannot. Isomap is elegant and interpretable, but sensitive to
the neighbor count $k$ (too large and edges "short-circuit" across the manifold's folds).

---

## 4. LLE and Laplacian eigenmaps

Two other structure-preserving methods, each with a different local principle:

- **Locally Linear Embedding (LLE)** assumes each point lies in the *linear span of its neighbors*:
  it finds weights $w_{ij}$ that reconstruct each point from its neighbors, then finds low-dimensional
  coordinates that preserve those same reconstruction weights. It captures local linear structure
  without ever computing global distances — cheaper than Isomap, but can distort.
- **Laplacian Eigenmaps** embed using the smallest eigenvectors of the graph Laplacian — *exactly* the
  construction of spectral clustering ([04.05](../05-spectral-clustering/)). Points connected in the
  neighborhood graph are pulled together in the embedding. The tight link between spectral clustering
  and manifold learning is that both read cluster/manifold structure from the same Laplacian spectrum.

These methods, with Isomap, are the classical (2000-era) manifold learners: principled, based on
eigendecompositions, giving meaningful coordinates — but largely superseded for *visualization* by
t-SNE and UMAP.

---

## 5. t-SNE

**t-SNE** (t-distributed Stochastic Neighbor Embedding, van der Maaten & Hinton 2008) is the method
behind most striking high-dimensional visualizations. It preserves *neighborhoods probabilistically*:

1. In the high-dimensional space, convert distances to **probabilities**: $p_{ij}$ is the chance that
   point $i$ would pick $j$ as a neighbor, under a Gaussian centered on $i$ whose width is set by the
   **perplexity** (roughly, the effective number of neighbors).
2. In the low-dimensional (2-D) space, define analogous probabilities $q_{ij}$ using a **Student-t
   distribution** (heavy-tailed) instead of a Gaussian.
3. Move the low-dimensional points by gradient descent to **minimize the KL divergence**
   $\mathrm{KL}(P\Vert Q)$ — making the low-D neighbor probabilities match the high-D ones.

Two design choices make it work. The KL divergence is **asymmetric** — it heavily penalizes putting
*near* points *far* apart, so t-SNE fiercely preserves local neighborhoods (tight clusters). The
**heavy-tailed Student-t** in low-D gives distant points room, solving the "crowding problem" (in 2-D
there is not enough space to place all the moderately-distant points a high-D dataset has). The result
is beautifully separated clusters — but at the cost of global geometry (§7). Experiment 2 shows t-SNE
cleanly separating clusters that overlap under PCA.

---

## 6. UMAP

**UMAP** (Uniform Manifold Approximation and Projection, McInnes et al. 2018) is the modern default,
built on a topological/graph foundation: it constructs a **fuzzy neighbor graph** in high-D and finds
a low-D layout whose fuzzy graph matches it, by minimizing a **cross-entropy** (attractive forces for
neighbors, repulsive for non-neighbors). Compared to t-SNE:

- **Faster and scales better** — the practical reason it has largely replaced t-SNE.
- **Preserves more global structure** — inter-cluster arrangement is a bit more trustworthy (though
  still not to be over-read).
- **Can transform new points** — unlike t-SNE, it learns a mapping (some out-of-sample capability).
- The main knob is **`n_neighbors`** (local vs global emphasis) and `min_dist` (cluster tightness).

UMAP and t-SNE share the same purpose (2-D visualization of neighborhood structure) and the same
cautions (§7). UMAP is usually the better choice today for speed; t-SNE remains excellent for fine
local cluster structure.

---

## 7. How to read a t-SNE / UMAP plot — the cautions

This is the most important section. t-SNE and UMAP produce gorgeous, persuasive plots, and they are
**routinely over-interpreted**. What you may and may not conclude:

- **Cluster *presence* is meaningful.** Well-separated blobs usually reflect real cluster structure.
- **Cluster *sizes* are NOT meaningful.** t-SNE expands dense clusters and shrinks sparse ones to
  equalize them; a big blob is not a bigger or more important cluster. Experiment 3 shows clusters of
  identical size rendered at different sizes.
- **Distances *between* clusters are NOT meaningful.** Two clusters far apart in a t-SNE plot are not
  necessarily more different than two that are close. Global geometry is sacrificed to local fidelity.
- **Density is NOT meaningful.** The apparent tightness of points within a cluster is an artifact.
- **The plot changes with perplexity / `n_neighbors`.** Different settings give different, sometimes
  contradictory pictures; always look at several (Experiment 4). There is no single "true" t-SNE plot.
- **Do not cluster on, or feed downstream, the 2-D coordinates.** The embedding distorts distances by
  design; run clustering on the *original* (or PCA-reduced) data, and use t-SNE only to *look*
  (§8–§9).

The one-sentence rule: **t-SNE/UMAP tell you *whether* clusters exist, not how big, how far apart, or
how dense they are.** Read them as qualitative maps, never as quantitative geometry.

---

## 8. Choosing a method

| Need | Use |
|---|---|
| Fast linear compression / denoising / preprocessing | **PCA** ([04.06](../06-linear-dimensionality-reduction/)) |
| Faithful low-D coordinates of a nonlinear manifold | **Isomap / LLE / Laplacian eigenmaps** |
| 2-D visualization of cluster structure | **UMAP** (or t-SNE) |
| Fine local cluster detail in a plot | **t-SNE** |
| Speed + some global structure + new-point transform | **UMAP** |

A common and effective pipeline: **PCA first** to reduce to ~50 dimensions (denoise and speed up),
*then* t-SNE/UMAP to 2-D. PCA removes noise directions that would otherwise confuse the neighbor
graph, and shrinks the compute. For actual dimensionality reduction feeding a model, prefer PCA or
Isomap (meaningful coordinates) over t-SNE/UMAP (visualization only).

---

## 9. Limits

- **Visualization methods are not dimensionality reduction for modelling.** t-SNE/UMAP coordinates
  distort distance and are unstable; do not train on them or measure distances in them (§7).
- **No out-of-sample map (mostly).** Classical methods and t-SNE embed a fixed dataset; a new point
  requires re-running (UMAP is the partial exception).
- **Hyperparameter sensitivity.** Perplexity / `n_neighbors` materially change the output (§7); the
  neighbor count $k$ short-circuits Isomap if too large (§3).
- **Compute.** Pairwise methods are $O(n^2)$ or worse; Barnes-Hut t-SNE and UMAP use approximations,
  but very large $n$ still needs subsampling or PCA preprocessing.
- **Stochastic and non-convex.** t-SNE's result depends on the random seed and initialization; run it
  a few times.

---

## 10. Common misconceptions

**"t-SNE clusters that are far apart are very different."**
Inter-cluster distances are not meaningful (§7). t-SNE preserves local neighborhoods, not global
geometry.

**"The bigger t-SNE cluster is the more important / more populous one."**
Cluster sizes are artifacts of the crowding correction (§7, Experiment 3).

**"I'll cluster on the t-SNE 2-D coordinates."**
Don't — the embedding distorts distances by construction. Cluster on the original or PCA-reduced data
and use t-SNE only to visualize (§7–§8).

**"There is one correct t-SNE plot."**
The plot depends on perplexity, seed, and iterations; inspect several settings (§7, Experiment 4).

**"PCA and manifold learning do the same thing."**
PCA is linear and cannot unfold curved manifolds (§1); manifold methods are nonlinear (Experiment 1).

**"UMAP/t-SNE preserve density."**
They do not — apparent within-cluster density is an artifact (§7).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — **Isomap** (k-NN graph → shortest-path geodesics →
  classical MDS) and a compact **t-SNE** (perplexity-calibrated $P$, Student-t $Q$, KL-gradient
  descent) in NumPy, checked against `sklearn.manifold`. Five experiments: (1) Isomap unfolding the
  Swiss roll where PCA cannot (correlation with the true manifold coordinate); (2) t-SNE separating
  clusters that overlap under PCA; (3) the size artifact — equal clusters rendered at different sizes;
  (4) perplexity changing the picture; (5) t-SNE distances not preserving the original distances.
- **[exercises.md](exercises.md)** — derive classical MDS and the t-SNE gradient, implement Isomap,
  reproduce every experiment.
- **[references.md](references.md)** — Tenenbaum (Isomap), Roweis & Saul (LLE), van der Maaten &
  Hinton (t-SNE), McInnes et al. (UMAP).

**Next**: [04.08 — Anomaly Detection](../08-anomaly-detection/) — finding the points that *don't*
belong, with density, distance, and isolation methods.
