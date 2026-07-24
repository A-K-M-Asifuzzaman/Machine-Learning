# 04.01 — Exercises: k-Means Clustering

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the k-means objective (WCSS/inertia) and prove that, with assignments fixed, the optimal
center of a cluster is the mean of its points.

**D2.** Show that both steps of Lloyd's algorithm (assignment, update) never increase the objective,
and conclude that the algorithm converges in finitely many steps.

**D3.** Explain why convergence is only to a *local* minimum. Give a small example (a few points) where
a bad initialization gives a suboptimal fixed point.

**D4.** Describe the k-means++ seeding distribution and explain intuitively why $D^2$ weighting spreads
the seeds across true clusters. State its approximation guarantee.

**D5.** Show that the assignment step induces a Voronoi partition, and hence that k-means cluster
boundaries are piecewise linear. What does this imply about the cluster shapes it can represent?

**D6.** Derive k-means as the hard-assignment, isotropic-covariance ($\sigma^2\mathbf I$, $\sigma^2\to0$)
limit of a Gaussian mixture fit by EM ([04.04](../04-gaussian-mixtures/)).

**D7.** Define the silhouette coefficient and explain what $s<0$, $s\approx0$, and $s\approx1$ mean for
a point.

**D8.** Explain why inertia cannot be used directly to choose $k$, and why the elbow and silhouette
can.

**D9.** Explain why k-means is sensitive to feature scaling, and why k-medoids can use any distance
metric.

**D10.** Analyze the time complexity of one Lloyd iteration in terms of $n$, $k$, $d$.

---

## Tier 2 — Implementation

**I1.** Implement Lloyd's algorithm with vectorized distance computation. Verify inertia and labels
(up to permutation) against `sklearn.cluster.KMeans`.

**I2.** Implement k-means++ seeding. Reproduce Experiment 1: compare mean inertia and the fraction of
catastrophic runs against random init.

**I3.** Reproduce Experiment 2: record the objective at each iteration and confirm it never increases.

**I4.** Implement the silhouette score. Reproduce Experiment 3: compute the elbow and silhouette curves
and identify $k$.

**I5.** Reproduce Experiment 4: build elongated, unequal-variance, and two-moons data and measure
k-means' ARI against the truth.

**I6.** Implement k-medoids (PAM). Reproduce Experiment 5: show it holds under outliers that break
k-means.

**I7.** Implement mini-batch k-means and compare its inertia and runtime to full k-means on a large
dataset.

**I8.** Implement the gap statistic and compare its chosen $k$ to the silhouette's on the same data.

**I9.** *(Kernel k-means.)* Implement k-means in an RBF feature space (via the kernel trick) and show
it separating two moons that plain k-means cannot.

**I10.** Standardize vs not: cluster a dataset with mismatched feature scales with and without
standardization, and compare the results.

---

## Tier 3 — Interview

**Q1.** What does k-means optimize?

**Q2.** Walk me through Lloyd's algorithm.

**Q3.** Does k-means find the global optimum?

**Q4.** Why is k-means++ better than random initialization?

**Q5.** How do you choose $k$?

**Q6.** What cluster shapes can k-means not find?

**Q7.** How is k-means related to Gaussian mixtures?

**Q8.** When would you use k-medoids?

**Q9.** Should you scale features before k-means?

**Q10.** Your k-means clusters look bad. What could be wrong?

**Q11.** What is the silhouette score?

**Q12.** How does k-means scale to millions of points?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the mean as the optimal center and prove convergence
- [ ] Explain why initialization matters and how k-means++ helps
- [ ] Choose $k$ with the elbow and silhouette, and know their limits
- [ ] State the spherical/equal-variance/convex assumptions and their failure modes
- [ ] Connect k-means to GMM and EM
- [ ] Pick k-medoids or kernel k-means when the assumptions break
