# 04.03 — Exercises: Density Clustering (DBSCAN)

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Define core, border, and noise points in terms of `eps` and `minPts`.

**D2.** Define directly density-reachable, density-reachable, and density-connected. Why must
reachability flow through core points?

**D3.** Define a DBSCAN cluster as a maximal density-connected set, and explain why this definition
allows arbitrarily shaped clusters.

**D4.** Explain why DBSCAN needs no $k$ and labels noise natively, unlike k-means.

**D5.** Explain the k-distance heuristic for `eps`: why does the sorted $k$-distance curve have a knee,
and why does the knee separate cluster points from noise?

**D6.** Give the rule of thumb for `minPts` and explain the tradeoff as it grows.

**D7.** *(Varying density.)* Construct (on paper) two clusters of different density and argue that no
single `eps` can both separate the dense clusters and capture the sparse one.

**D8.** Explain why border points make DBSCAN mildly non-deterministic, and why core-point clustering
is deterministic.

**D9.** Explain why DBSCAN degrades in high dimensions (relate to distance concentration,
[03.06](../../03-supervised-learning/06-knn/)).

**D10.** Describe how OPTICS and HDBSCAN remove the single-`eps` limitation.

---

## Tier 2 — Implementation

**I1.** Implement DBSCAN with core/border/noise labeling. Verify the core-point set and clustering
against `sklearn.cluster.DBSCAN`.

**I2.** Implement the k-distance plot. Reproduce Experiment 4: read the knee and confirm it sets `eps`
well.

**I3.** Reproduce Experiment 1: cluster two moons and concentric rings, and compare to k-means by ARI.

**I4.** Reproduce Experiment 2: inject outliers and show DBSCAN flagging them as noise while
recovering the clusters.

**I5.** Reproduce Experiment 3: build a dense pair plus a sparse cluster and show no single `eps`
handles both.

**I6.** Reproduce Experiment 5: sweep `minPts` and show the noise fraction rising.

**I7.** Add a kd-tree / ball-tree region query and measure the speedup over the naive $O(n^2)$ scan.

**I8.** *(OPTICS.)* Implement the reachability-distance ordering and plot it; extract clusters at two
different density thresholds.

**I9.** Compare DBSCAN, k-means, and Ward on the same three datasets (blobs, moons, varying density)
and tabulate ARI.

**I10.** Run DBSCAN on high-dimensional data and show `eps` becoming impossible to set as dimension
grows.

---

## Tier 3 — Interview

**Q1.** How does DBSCAN define a cluster?

**Q2.** What are core, border, and noise points?

**Q3.** How is `eps` chosen?

**Q4.** Why does DBSCAN find non-convex clusters?

**Q5.** What is DBSCAN's main weakness?

**Q6.** How does DBSCAN handle outliers?

**Q7.** DBSCAN vs k-means — when each?

**Q8.** What do OPTICS and HDBSCAN add?

**Q9.** Why does DBSCAN struggle in high dimensions?

**Q10.** Is DBSCAN deterministic?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Define core/border/noise and density-reachability precisely
- [ ] Set `eps` from a k-distance plot rather than guessing
- [ ] Explain why DBSCAN finds arbitrary shapes and labels noise
- [ ] State the varying-density failure and demonstrate it
- [ ] Choose DBSCAN vs k-means vs HDBSCAN from the data
- [ ] Explain the high-dimensional and scaling limits
