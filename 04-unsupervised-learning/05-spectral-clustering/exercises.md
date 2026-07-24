# 04.05 — Exercises: Spectral Clustering

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Prove the quadratic-form identity $\mathbf{f}^\top L\mathbf{f} = \tfrac12\sum_{ij}W_{ij}(f_i-f_j)^2$
for the unnormalized Laplacian $L=D-W$, and conclude $L$ is positive semidefinite.

**D2.** Show the smallest eigenvalue of $L$ is $0$ with eigenvector $\mathbf{1}$, and that the
multiplicity of $0$ equals the number of connected components.

**D3.** Define RatioCut and Normalized Cut. Explain why minimizing the raw cut is degenerate and how
each balances cluster size/degree.

**D4.** Derive the spectral relaxation of RatioCut: show the relaxed problem is minimized by the
smallest eigenvectors of $L$.

**D5.** Show that Normalized Cut relaxes to the smallest eigenvectors of $L_{\mathrm{rw}}$ (or the
generalized eigenproblem $L\mathbf{u}=\lambda D\mathbf{u}$).

**D6.** Define the Fiedler vector and algebraic connectivity. Why does the Fiedler vector's sign give
a good bipartition?

**D7.** State the eigengap heuristic and justify it from the connected-components theorem (D2).

**D8.** Explain why the *smallest* Laplacian eigenvectors carry cluster structure, in contrast to
PCA's *largest* covariance eigenvectors.

**D9.** Explain why the RBF bandwidth $\sigma$ (or $k$-NN count) is the most important parameter, and
give a rule of thumb for setting it.

**D10.** Explain the relationship between spectral clustering and kernel k-means.

---

## Tier 2 — Implementation

**I1.** Implement RBF and $k$-NN affinity graphs and all three Laplacians. Verify a full spectral
clustering against `sklearn.cluster.SpectralClustering`.

**I2.** Reproduce Experiment 1: cluster two moons and rings; compare to k-means by ARI.

**I3.** Reproduce Experiment 2: compute the Laplacian eigenvalues and read the number of clusters from
the eigengap.

**I4.** Reproduce Experiment 3: build a graph with outliers and show the unnormalized Laplacian
isolating them while normalized Laplacians do not.

**I5.** Reproduce Experiment 4: sweep the RBF bandwidth and show the clustering swinging from one blob
to fragments.

**I6.** Reproduce Experiment 5: bipartition the two moons with the Fiedler vector's sign alone.

**I7.** Reproduce Experiment 6: show a linear classifier failing on raw moons but succeeding on the
Laplacian embedding.

**I8.** Implement a self-tuning affinity (Zelnik-Manor & Perona local scaling) and show it handling
varying density better than a fixed $\sigma$.

**I9.** *(Scaling.)* Use a sparse $k$-NN graph and `scipy.sparse.linalg.eigsh` to cluster a larger
dataset; measure the speedup over the dense eigendecomposition.

**I10.** Implement the Nyström approximation for out-of-sample extension and embed new points.

---

## Tier 3 — Interview

**Q1.** What is spectral clustering in one sentence?

**Q2.** What is the graph Laplacian and why is it useful?

**Q3.** Why do you use the smallest eigenvectors, not the largest?

**Q4.** How does spectral clustering find non-convex clusters?

**Q5.** What graph-cut problem does it approximate?

**Q6.** Normalized vs unnormalized Laplacian — which and why?

**Q7.** How do you choose the number of clusters?

**Q8.** What's the most important parameter to tune?

**Q9.** What is the Fiedler vector?

**Q10.** What are the computational limits of spectral clustering?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Prove the Laplacian quadratic form and the connected-components theorem
- [ ] Explain the RatioCut / NCut relaxation
- [ ] Run the NJW algorithm and know why the embedding step matters
- [ ] Use the eigengap to choose $k$
- [ ] Build a good similarity graph and pick the normalization
- [ ] Connect spectral clustering to Laplacian eigenmaps and PCA
