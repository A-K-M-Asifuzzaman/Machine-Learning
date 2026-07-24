# 04.07 — Exercises: Manifold Learning

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** State the manifold hypothesis and explain why linear PCA cannot capture a curved manifold.

**D2.** Describe Isomap's three steps and explain why graph shortest paths approximate geodesic
distances.

**D3.** Derive classical MDS: given a squared-distance matrix, show that double-centering gives a
Gram matrix whose top eigenvectors are the embedding coordinates.

**D4.** Explain LLE's two optimizations (reconstruction weights, then embedding) and what local
structure it preserves.

**D5.** Show that Laplacian eigenmaps use the same eigenvectors as spectral clustering
([04.05](../05-spectral-clustering/)).

**D6.** Write t-SNE's high-D probabilities $p_{ij}$ (Gaussian, perplexity-calibrated) and low-D
$q_{ij}$ (Student-t). Explain the role of perplexity.

**D7.** Explain why t-SNE minimizes the *asymmetric* $\mathrm{KL}(P\Vert Q)$ and what that asymmetry
does to local vs global structure.

**D8.** Explain the crowding problem and how the heavy-tailed Student-t in low-D solves it.

**D9.** Derive the t-SNE gradient $\frac{\partial C}{\partial y_i} = 4\sum_j (p_{ij}-q_{ij})(y_i-y_j)(1+\lVert y_i-y_j\rVert^2)^{-1}$.

**D10.** Explain, mechanistically, why t-SNE cluster sizes, densities, and inter-cluster distances are
not meaningful.

---

## Tier 2 — Implementation

**I1.** Implement Isomap (k-NN graph, shortest paths, classical MDS). Verify it recovers the Swiss
roll coordinate and matches `sklearn.manifold.Isomap`.

**I2.** Implement classical MDS and verify it reproduces PCA on Euclidean distances.

**I3.** Implement the perplexity binary search for the Gaussian precisions, and verify the entropy
matches $\log(\text{perplexity})$.

**I4.** Implement t-SNE (P, Q, KL gradient, momentum, early exaggeration). Verify it separates blobs.

**I5.** Reproduce Experiment 1: Isomap vs PCA on the Swiss roll (rank correlation with the manifold
coordinate).

**I6.** Reproduce Experiment 3: two clusters of equal size but different spread, and show t-SNE
rendering them at the same visual size.

**I7.** Reproduce Experiment 4: run t-SNE at several perplexities and show the layout changing.

**I8.** Reproduce Experiment 5: clusters at graded distances, and show t-SNE scrambling the
inter-cluster ordering while PCA preserves it.

**I9.** Implement LLE and compare its Swiss-roll unfolding to Isomap.

**I10.** *(Pipeline.)* PCA to 50-D then t-SNE to 2-D on a digit dataset; compare to t-SNE directly on
raw pixels in speed and quality.

---

## Tier 3 — Interview

**Q1.** What is the manifold hypothesis?

**Q2.** Why can't PCA unfold a Swiss roll?

**Q3.** How does Isomap work?

**Q4.** What does t-SNE optimize?

**Q5.** What is perplexity?

**Q6.** Can you read cluster sizes off a t-SNE plot?

**Q7.** Can you read distances between clusters off a t-SNE plot?

**Q8.** Should you cluster on t-SNE coordinates?

**Q9.** t-SNE vs UMAP — how do they differ?

**Q10.** When would you use Isomap instead of t-SNE?

**Q11.** Why run PCA before t-SNE?

**Q12.** What are the main pitfalls of interpreting a t-SNE plot?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the manifold hypothesis and PCA's linear limitation
- [ ] Implement Isomap and classical MDS
- [ ] Explain t-SNE's P, Q, and KL objective
- [ ] List exactly what is and isn't meaningful in a t-SNE plot
- [ ] Choose Isomap vs t-SNE/UMAP from the task
- [ ] Explain why you never cluster on or measure distances in a t-SNE embedding
