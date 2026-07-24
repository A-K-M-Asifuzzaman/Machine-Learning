# 04.08 — Exercises: Anomaly Detection

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Distinguish outlier detection from novelty detection, and point vs contextual vs collective
anomalies.

**D2.** Explain the density view of anomalies and why a single Gaussian fails on multimodal data.

**D3.** Define the k-NN distance score and explain why a global threshold fails under varying density.

**D4.** Define LOF: local reachability density and the LOF ratio. Show LOF $\approx 1$ for normal
points and $\gg 1$ for local anomalies.

**D5.** Explain the Isolation Forest principle: why do anomalies have shorter expected path lengths?

**D6.** Derive the Isolation Forest anomaly score $2^{-\mathbb{E}[h]/c(n)}$ and the normalization
$c(n) = 2H(n-1) - 2(n-1)/n$.

**D7.** Explain the One-Class SVM objective (separate data from the origin with margin) and its
$\nu$ parameter.

**D8.** Explain reconstruction-based detection and why PCA reconstruction error flags off-subspace
points.

**D9.** Explain why average precision / precision@k is preferred to ROC-AUC for anomaly evaluation
(relate to imbalance, [05.03](../../05-model-evaluation/03-classification-metrics/)).

**D10.** Explain the role of the contamination parameter as a decision threshold.

---

## Tier 2 — Implementation

**I1.** Implement Isolation Forest (random isolation trees + path-length scoring). Verify the ROC-AUC
and ranking against `sklearn.ensemble.IsolationForest`.

**I2.** Implement LOF (reachability distance, local reachability density, ratio). Verify against
`sklearn.neighbors.LocalOutlierFactor`.

**I3.** Reproduce Experiment 1: measure normal vs anomaly path lengths and the ROC-AUC.

**I4.** Reproduce Experiment 2: plant a local anomaly and show LOF catching it where global k-NN
distance misses.

**I5.** Reproduce Experiment 3: multimodal normal data with anomalies at the global mean; show the
single Gaussian failing (AUC < 0.5) and Isolation Forest succeeding.

**I6.** Reproduce Experiment 4: PCA reconstruction error on low-rank normal data with off-subspace
anomalies.

**I7.** Reproduce Experiment 5: sweep contamination and plot precision vs recall.

**I8.** Implement a GMM density scorer ([04.04](../04-gaussian-mixtures/)) and compare to the single
Gaussian on multimodal data.

**I9.** Implement a One-Class SVM (or use sklearn) for novelty detection on clean training data, and
compare to Isolation Forest.

**I10.** *(Ensembling.)* Average the ranks of Isolation Forest, LOF, and PCA reconstruction, and show
the ensemble is more robust than any single detector.

---

## Tier 3 — Interview

**Q1.** Why not treat anomaly detection as supervised classification?

**Q2.** How does Isolation Forest work?

**Q3.** Why is Isolation Forest good in high dimensions?

**Q4.** What is LOF and when do you need it?

**Q5.** Why does a single Gaussian fail on multimodal data?

**Q6.** What is novelty detection, and what method fits it?

**Q7.** How does reconstruction-based detection work?

**Q8.** How do you evaluate an anomaly detector without labels?

**Q9.** Why prefer average precision to ROC-AUC here?

**Q10.** What does the contamination parameter do?

**Q11.** Which method would you start with on a new problem?

**Q12.** How would you combine multiple detectors?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Frame outlier vs novelty detection and the anomaly types
- [ ] Explain and implement Isolation Forest and LOF
- [ ] Explain why global distance and single-Gaussian methods fail
- [ ] Use PCA reconstruction error as an anomaly score
- [ ] Evaluate detectors with AP/precision@k and the contamination threshold
- [ ] Choose a method from the data's shape and dimension
