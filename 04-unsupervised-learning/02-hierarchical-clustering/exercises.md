# 04.02 — Exercises: Hierarchical Clustering

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the agglomerative algorithm and explain why it is greedy and not globally optimal.

**D2.** Define single, complete, average, and Ward linkage. For each, describe the cluster shape it
prefers and its main weakness.

**D3.** State the Lance-Williams formula and give the $(\alpha_i,\alpha_j,\beta,\gamma)$ coefficients
for single and complete linkage. Show they yield the min and max respectively.

**D4.** Derive the Ward linkage update: show that merging the pair minimizing the increase in
within-cluster sum of squares corresponds to the Lance-Williams coefficients with cluster sizes.

**D5.** Show that Ward linkage optimizes the same objective as k-means, and explain why they give
similar clusterings on blob data.

**D6.** Define the cophenetic distance and cophenetic correlation. What does a low value tell you?

**D7.** Explain why single linkage "chains" and why complete linkage does not. When is chaining a
feature, and when a bug?

**D8.** Analyze the time and space complexity of agglomerative clustering, and explain why space is
the binding constraint.

**D9.** Explain how to choose the number of clusters from a dendrogram (the largest merge-height gap),
and relate it to the elbow method.

**D10.** Why is divisive clustering rarely used despite being the natural top-down counterpart?

---

## Tier 2 — Implementation

**I1.** Implement agglomerative clustering with the Lance-Williams update for all four linkages.
Verify the merge heights and flat clusters against `scipy.cluster.hierarchy`.

**I2.** Implement dendrogram cutting (`fcluster`) to get $k$ clusters or to cut at a height.

**I3.** Implement the cophenetic distance and correlation; verify against `scipy.cluster.hierarchy.cophenet`.

**I4.** Reproduce Experiment 1: compare linkages on blobs vs two moons by ARI.

**I5.** Reproduce Experiment 2: single linkage on concentric rings, then add a noise bridge and show it
break.

**I6.** Reproduce Experiment 3: show Ward ≈ k-means on blobs.

**I7.** Reproduce Experiment 4: recover $k$ from the largest merge-height gap.

**I8.** Reproduce Experiment 5: compute cophenetic correlation per linkage and pick the most faithful.

**I9.** Draw an actual dendrogram (matplotlib) and annotate the cut. Compare to `scipy`'s
`dendrogram`.

**I10.** *(Scaling.)* Implement the nearest-neighbor-chain algorithm for Ward to reduce time, and
measure the speedup over the naive $O(n^3)$ version.

---

## Tier 3 — Interview

**Q1.** What does hierarchical clustering give you that k-means does not?

**Q2.** Agglomerative vs divisive — which is standard and why?

**Q3.** What is linkage, and how do you choose it?

**Q4.** Why does single linkage find non-convex clusters?

**Q5.** What is the weakness of single linkage?

**Q6.** How is Ward linkage related to k-means?

**Q7.** How do you choose the number of clusters from a dendrogram?

**Q8.** What is cophenetic correlation?

**Q9.** Why doesn't hierarchical clustering scale to millions of points?

**Q10.** Is hierarchical clustering deterministic, and does that make it better than k-means?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Run agglomerative merging by hand and via Lance-Williams
- [ ] Pick a linkage from the cluster shape you expect
- [ ] Explain single-linkage chaining and its fragility
- [ ] Connect Ward to k-means
- [ ] Cut a dendrogram to choose $k$
- [ ] State the $O(n^2)$ memory limit and when to use k-means/DBSCAN instead
