# 04.05 — Spectral Clustering

> **Prerequisites**: [04.01](../01-kmeans/) (k-means, used on the embedding),
> [00.01](../../00-mathematical-foundations/01-linear-algebra/) (eigenvectors, symmetric matrices),
> [04.03](../03-density-clustering/) (the non-convex clusters this also handles).
> **You will be able to**: build a similarity graph and its Laplacian, explain why the smallest
> eigenvectors reveal clusters, run the Ng-Jordan-Weiss algorithm, use the eigengap to choose $k$,
> and connect spectral clustering to the graph-cut problem it approximates.

---

## Table of contents

1. [Clustering by graph structure](#1-clustering-by-graph-structure)
2. [The similarity graph](#2-the-similarity-graph)
3. [The graph Laplacian](#3-the-graph-laplacian)
4. [Properties of the Laplacian](#4-properties-of-the-laplacian)
5. [The algorithm](#5-the-algorithm)
6. [Why it works: graph cuts](#6-why-it-works-graph-cuts)
7. [Choosing k: the eigengap](#7-choosing-k-the-eigengap)
8. [Building the graph well](#8-building-the-graph-well)
9. [Strengths, costs, and connections](#9-strengths-costs-and-connections)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Clustering by graph structure

k-means and GMMs cluster by *geometry* — distance to a center. They cannot separate two interlocking
moons or concentric rings, because those clusters are not linearly separable and not blob-shaped.
Spectral clustering takes a completely different route: it treats the data as a **graph** — points are
nodes, and edges connect *similar* points — and clusters by the graph's **connectivity** rather than
raw distance. Two points on opposite ends of a curved cluster are far in Euclidean distance but
*connected* through a chain of similar neighbors, so the graph view keeps them together.

The magic is that this connectivity is captured by the **eigenvectors of the graph Laplacian**. The
few smallest eigenvectors give a new coordinate system — an *embedding* — in which the tangled,
non-convex clusters become simple, well-separated blobs. You then run plain k-means in that embedding.
Spectral clustering is, in one line: **embed with Laplacian eigenvectors, then k-means**. It is one of
the most elegant algorithms in machine learning, tying together graph theory, linear algebra, and
clustering.

---

## 2. The similarity graph

The first step turns the data into a weighted graph via a **similarity (affinity) matrix** $W$, where
$W_{ij} \ge 0$ measures how similar points $i$ and $j$ are (and $W_{ii}=0$). Three standard
constructions:

- **Fully connected with a Gaussian (RBF) kernel**:
  $W_{ij} = \exp(-\lVert\mathbf{x}_i-\mathbf{x}_j\rVert^2 / (2\sigma^2))$. Every pair is connected,
  weighted by closeness; the bandwidth $\sigma$ sets the scale of "similar."
- **$k$-nearest-neighbor graph**: connect $i$ and $j$ if either is among the other's $k$ nearest
  neighbors. Sparse, local, and often the most robust choice.
- **$\varepsilon$-neighborhood graph**: connect points within distance $\varepsilon$. Sparse but
  sensitive to $\varepsilon$ (like DBSCAN's `eps`).

The graph construction is the most consequential modelling choice in spectral clustering (§8) — it
*defines* what "connected" means, and a bad graph gives a bad clustering no matter how the eigenvectors
are computed. The kernel bandwidth $\sigma$ or the neighbor count $k$ must match the data's scale.

---

## 3. The graph Laplacian

From $W$ and the **degree matrix** $D$ (diagonal, $D_{ii} = \sum_j W_{ij}$ = total similarity at node
$i$), form the **graph Laplacian**. There are three versions:

$$
L = D - W \quad(\text{unnormalized}), \qquad L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}, \qquad L_{\mathrm{rw}} = I - D^{-1} W.
$$

The unnormalized $L = D - W$ is the basic object; the two **normalized** Laplacians rescale by degree
and usually cluster better (they balance cluster sizes, §6). The Laplacian earns its name from the
identity that makes everything work — for any vector $\mathbf{f}$,

$$
\mathbf{f}^\top L\, \mathbf{f} = \tfrac12 \sum_{i,j} W_{ij}\, (f_i - f_j)^2 \ge 0.
$$

This quadratic form is small exactly when $\mathbf{f}$ takes **similar values on connected
(high-$W_{ij}$) points** — i.e. when $\mathbf{f}$ is *smooth over the graph*. Minimizing it (subject to
constraints) finds the smoothest functions on the graph, which are precisely the cluster-indicator-like
vectors. That is why we want the *smallest* eigenvectors.

---

## 4. Properties of the Laplacian

The Laplacian's spectrum encodes the cluster structure directly:

- **Symmetric and positive semidefinite** — all eigenvalues are real and $\ge 0$ (from the quadratic
  form above). So there is a smallest eigenvalue, and it is $0$.
- **The smallest eigenvalue is $0$**, with the constant vector $\mathbf{1}$ as eigenvector (every point
  the same — perfectly smooth, zero cut).
- **The multiplicity of the eigenvalue $0$ equals the number of connected components** of the graph,
  and the corresponding eigenvectors are the **indicator vectors** of those components. This is the
  key theorem: if the graph splits into $k$ disconnected pieces, $L$ has exactly $k$ zero eigenvalues,
  and their eigenvectors *are* the clusters. Experiment 2 verifies this by counting near-zero
  eigenvalues.
- **The Fiedler vector** — the eigenvector of the *second*-smallest eigenvalue — gives the best
  bipartition of a connected graph (its sign splits the graph into two well-separated halves). Its
  eigenvalue (the *algebraic connectivity*) measures how well-connected the graph is.

Real clusters are not perfectly disconnected, but they are *weakly* connected between clusters and
*strongly* connected within. So the $k$ smallest eigenvalues are near zero (not exactly), and their
eigenvectors are approximately piecewise-constant on the clusters — good enough that k-means on them
recovers the clustering.

---

## 5. The algorithm

The **Ng-Jordan-Weiss** spectral clustering algorithm (2002), for $k$ clusters:

1. Build the similarity matrix $W$ (§2) and the degree matrix $D$.
2. Form the normalized Laplacian $L_{\mathrm{sym}} = I - D^{-1/2}WD^{-1/2}$ (or use $L_{\mathrm{rw}}$).
3. Compute the **$k$ eigenvectors with the smallest eigenvalues**, and stack them as the columns of a
   matrix $U \in \mathbb{R}^{n\times k}$. Each *row* of $U$ is the new $k$-dimensional embedding of a
   data point.
4. (For $L_{\mathrm{sym}}$) **normalize each row** of $U$ to unit length.
5. Run **k-means** on the rows of $U$ to get the final clusters.

Steps 1–4 produce an embedding in which the non-convex clusters have become tight, separated blobs;
step 5 is ordinary k-means finishing the job. `from_scratch.py` builds each piece and verifies the
result against scikit-learn, and Experiment 6 visualizes the embedding making the two moons linearly
separable.

---

## 6. Why it works: graph cuts

Spectral clustering is a **continuous relaxation of graph partitioning**. Partitioning a graph into
clusters means cutting edges; a good clustering cuts *few, weak* edges (little similarity crosses
cluster boundaries). The **cut** between sets $A$ and its complement is $\mathrm{cut}(A) = \sum_{i\in A, j\notin A} W_{ij}$.
Minimizing the raw cut is bad — it just lops off a single point. Two balanced objectives fix this:

- **RatioCut** divides each cut by the *number of nodes* in the cluster, and
- **Normalized Cut (NCut)** divides by the cluster's *total degree* (Shi & Malik).

Both favor clusters that are internally dense and externally sparse *and* reasonably balanced in size.
The catch: minimizing RatioCut or NCut over discrete cluster assignments is **NP-hard**. Spectral
clustering **relaxes** the discrete indicator vectors to real-valued ones — and the relaxed problem's
solution is exactly the smallest eigenvectors of the Laplacian ($L$ for RatioCut, $L_{\mathrm{rw}}$ /
$L_{\mathrm{sym}}$ for NCut). So computing eigenvectors and running k-means is a tractable
approximation to an NP-hard balanced-cut problem. This is the theoretical heart of the method, and it
is why the *normalized* Laplacians (which relax NCut) usually outperform the unnormalized one
(RatioCut): NCut's degree-balancing handles clusters of different sizes better.

---

## 7. Choosing k: the eigengap

Spectral clustering needs the number of clusters $k$, and its own eigenvalues suggest it. Sort the
Laplacian's eigenvalues $0 = \lambda_1 \le \lambda_2 \le \dots$. The **eigengap heuristic**: choose $k$
where there is a **large gap** between $\lambda_k$ and $\lambda_{k+1}$. The intuition is §4's theorem —
with $k$ well-separated clusters, the first $k$ eigenvalues are near zero (approximating the $k$
disconnected-component case) and $\lambda_{k+1}$ jumps up. The size of the gap reflects how clean the
cluster separation is. Experiment 2 shows $k$ near-zero eigenvalues followed by a clear jump, reading
off the right $k$. This is more principled than k-means' elbow, though it too can be ambiguous when
clusters overlap.

---

## 8. Building the graph well

The quality of spectral clustering hinges on the similarity graph, more than on any other choice:

- **Kernel bandwidth $\sigma$** (RBF graph). Too small and the graph fragments (every point isolated,
  many spurious clusters); too large and everything connects (one blob). Rule of thumb: set $\sigma$
  near the typical distance to a point's $k$-th neighbor. Experiment 4 shows the result swinging with
  $\sigma$.
- **Neighbor count $k$** ($k$-NN graph). Must be large enough to keep each true cluster connected,
  small enough not to bridge clusters. The $k$-NN graph is often more robust than the fully-connected
  RBF graph because it adapts to local density.
- **Mutual vs symmetric $k$-NN, and self-tuning kernels** (Zelnik-Manor & Perona) adapt the scale per
  point, helping with varying density.

Because the graph *is* the model, spend your effort here. A well-built $k$-NN graph on data of similar
scale is a reliable default.

---

## 9. Strengths, costs, and connections

**Strengths.** Finds **arbitrarily shaped, non-convex clusters** (moons, rings, spirals) that k-means
and GMMs cannot (Experiment 1); makes no assumption about cluster shape, only about connectivity;
grounded in a clear optimization (graph cuts).

**Costs.** The eigendecomposition is **$O(n^3)$** for a dense Laplacian (or $O(n^2)$ for a sparse
$k$-NN graph with iterative eigensolvers), so it does not scale to millions of points without
approximation (Nyström, landmark methods). It needs $k$, is **sensitive to graph construction** (§8),
and gives no out-of-sample rule (a new point is not naturally embedded — you must re-run or use an
extension).

**Connections.** Spectral clustering is **kernel k-means** in disguise, and its Laplacian embedding is
exactly **Laplacian eigenmaps** ([04.07](../07-manifold-learning/)) — the same eigenvectors used for
non-linear dimensionality reduction. The Fiedler vector links to graph theory and PageRank; the whole
method is a cousin of **diffusion maps** and of **PCA** ([04.06](../06-linear-dimensionality-reduction/),
which uses eigenvectors of a covariance instead of a Laplacian). Learning spectral clustering unlocks a
whole family of eigenvector-based methods.

---

## 10. Common misconceptions

**"Spectral clustering is just a fancy k-means."**
It runs k-means only at the *end*, on a Laplacian-eigenvector embedding where non-convex clusters have
become separable (§5). The embedding is the whole point; plain k-means on the raw data fails on the
same data (§1).

**"Use the largest eigenvectors, like PCA."**
No — the *smallest* eigenvectors of the Laplacian (the smoothest graph functions) carry the cluster
structure (§3–§4). PCA uses the largest eigenvectors of a *covariance*; the objects are different.

**"The similarity graph doesn't matter much."**
It matters most (§8). The bandwidth $\sigma$ or neighbor count $k$ can change the clustering entirely;
the graph *is* the model.

**"Normalized and unnormalized Laplacians are interchangeable."**
The normalized Laplacians relax NCut and balance cluster degrees; they usually cluster better,
especially with uneven cluster sizes (§6). Prefer $L_{\mathrm{sym}}$ or $L_{\mathrm{rw}}$.

**"Spectral clustering scales like k-means."**
No — the eigendecomposition is $O(n^3)$ dense, $O(n^2)$ sparse (§9). For large $n$ use approximations
or a different method.

**"It gives a rule to cluster new points."**
It does not — the embedding is defined only on the training points (§9). New points need an
out-of-sample extension or a re-run.

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the full pipeline in NumPy: RBF and $k$-NN affinity graphs,
  all three Laplacians, the eigenvector embedding, and k-means on it, verified against
  `sklearn.cluster.SpectralClustering`. Six experiments: (1) spectral clustering two moons and rings
  where k-means fails (ARI); (2) the eigengap — $k$ near-zero eigenvalues then a jump; (3) normalized
  vs unnormalized Laplacian; (4) sensitivity to the RBF bandwidth $\sigma$; (5) the Fiedler vector
  bipartitioning; (6) the embedding making the two moons linearly separable.
- **[exercises.md](exercises.md)** — derive the quadratic-form identity and the RatioCut relaxation,
  implement the Laplacians and the eigengap, reproduce every experiment.
- **[references.md](references.md)** — Shi & Malik (NCut), Ng-Jordan-Weiss, von Luxburg's tutorial.

**Next**: [04.06 — Linear Dimensionality Reduction (PCA)](../06-linear-dimensionality-reduction/) —
eigenvectors again, now of the covariance matrix, to compress data onto its directions of maximum
variance.
