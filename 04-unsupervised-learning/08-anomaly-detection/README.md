# 04.08 — Anomaly Detection

> **Prerequisites**: [04.04](../04-gaussian-mixtures/) (density models),
> [04.06](../06-linear-dimensionality-reduction/) (PCA reconstruction),
> [06.02](../../06-ensembles/02-random-forests/) (random trees, for Isolation Forest),
> [05.03](../../05-model-evaluation/03-classification-metrics/) (evaluating rare-class detection).
> **You will be able to**: frame outlier vs novelty detection, apply density, distance, isolation,
> boundary, and reconstruction methods, explain why Isolation Forest and LOF work, and evaluate
> detectors when labels are scarce.

---

## Table of contents

1. [Finding what doesn't belong](#1-finding-what-doesnt-belong)
2. [Outliers, novelties, and types of anomaly](#2-outliers-novelties-and-types-of-anomaly)
3. [Density and statistical methods](#3-density-and-statistical-methods)
4. [Distance methods and LOF](#4-distance-methods-and-lof)
5. [Isolation Forest](#5-isolation-forest)
6. [One-Class SVM](#6-one-class-svm)
7. [Reconstruction-based detection](#7-reconstruction-based-detection)
8. [Evaluation without labels](#8-evaluation-without-labels)
9. [Choosing a method](#9-choosing-a-method)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Finding what doesn't belong

Anomaly detection finds the **rare, unusual points** that differ from the majority — fraudulent
transactions among millions of normal ones, intrusions in network traffic, defects on a production
line, disease in medical scans. It is one of the most valuable unsupervised tasks, and one of the
hardest, for a structural reason: **anomalies are rare and often unlabeled**. You usually cannot train
a supervised classifier (there are too few positives, and new anomaly types appear that were never
labeled), so you model **what "normal" looks like** and flag whatever deviates.

Every method is a different answer to "what does normal look like, and how far is this point from it?"
— low probability under a density model, far from its neighbors, easy to isolate, outside a learned
boundary, or poorly reconstructed. The methods differ in what "normal" means and how they measure
deviation, and no single one dominates; the right choice depends on the data's shape, dimension, and
whether you have clean training data.

---

## 2. Outliers, novelties, and types of anomaly

Two problem settings, often conflated:

- **Outlier detection** — the training data is *contaminated* (it contains anomalies), and you find the
  anomalies *within it*. Unsupervised; the method must be robust to the anomalies it is trying to find.
- **Novelty detection** — the training data is *clean* (all normal), and you flag *new* points that
  differ. Semi-supervised (one-class); you learn the normal region, then test new points against it.

And three kinds of anomaly:

- **Point anomalies** — a single point far from the rest (the fraudulent \$1M charge). The most common
  target.
- **Contextual anomalies** — normal in general but anomalous *in context* (35°C is normal in summer,
  anomalous in winter). Needs context features.
- **Collective anomalies** — a *group* of points anomalous together though each is individually normal
  (a burst of small transactions). Needs sequence/group modelling.

Most classical methods (this chapter) target *point* anomalies; contextual and collective anomalies
need domain-specific features or sequence models.

---

## 3. Density and statistical methods

The oldest idea: **anomalies live in low-density regions**. Fit a model of the data's density and flag
points with low probability.

- **Gaussian / z-score.** Fit a Gaussian (or per-feature z-scores); flag points beyond a few standard
  deviations. Simple and fast, but assumes a single Gaussian blob — it fails on multimodal or
  non-Gaussian data.
- **Gaussian Mixture Model** ([04.04](../04-gaussian-mixtures/)). A fitted GMM is a flexible density
  $p(\mathbf{x})$; low $p(\mathbf{x})$ = anomaly. Handles multimodal normal data.
- **Kernel density estimation.** Nonparametric density; flag low-density points. Flexible but scales
  poorly and struggles in high dimensions.

Density methods are principled (the anomaly score is a probability) and interpretable, but they inherit
their density model's assumptions — a single Gaussian is often too rigid, and all density estimation
degrades in high dimensions (the curse: everything looks low-density). They work best on
low-dimensional data whose "normal" is well-described by a simple density.

---

## 4. Distance methods and LOF

If modelling the whole density is hard, measure **local** deviation instead. Distance-based methods
flag points that are *far from their neighbors*:

- **k-NN distance.** Score each point by its distance to its $k$-th nearest neighbor (or the mean of
  its $k$ neighbors). Far points score high. Simple and effective on uniform-density data, but it uses
  a *global* distance threshold, so it fails when normal regions have *different densities* — a point
  in a naturally-sparse region looks anomalous.

- **Local Outlier Factor (LOF).** The fix for varying density. LOF compares a point's local density to
  the *densities of its neighbors*. Precisely: the **local reachability density** of a point is
  (roughly) the inverse of its average distance to its neighbors; LOF is the *ratio* of the neighbors'
  average density to the point's own. LOF $\approx 1$ means "as dense as my neighbors" (normal); LOF
  $\gg 1$ means "in a much sparser region than my neighbors" (a **local** anomaly). Because it is a
  *ratio*, LOF adapts to each region's local density — it can flag a point that is anomalous *relative
  to its own neighborhood* even if globally it sits in a moderately dense area. Experiment 2 shows LOF
  catching a local anomaly that global k-NN distance misses.

LOF is the classic answer when normal data has clusters of different densities. Its cost is the
neighbor computation ($O(n^2)$ naive) and sensitivity to the neighbor count $k$.

---

## 5. Isolation Forest

**Isolation Forest** (Liu, Ting & Zhou, 2008) is the most widely used modern method, and it inverts
the usual logic. Instead of modelling *normal* and measuring deviation, it directly exploits what
makes anomalies special: **anomalies are few and different, so they are *easy to isolate*.**

Build many random binary trees ("isolation trees"): at each node pick a random feature and a random
split value, recursively partitioning the data until each point is alone in a leaf. An **anomaly**,
being far from the crowd, gets separated after only a **few** random splits — its **path length** from
root to leaf is short. A **normal** point, buried among many similar points, takes **many** splits to
isolate. Average the path length over many random trees, and:

$$
\text{anomaly score} = 2^{-\,\mathbb{E}[\text{path length}] / c(n)},
$$

where $c(n)$ normalizes by the expected path length in a tree of $n$ points. Short expected path →
score near 1 (anomaly); long path → score near 0.5 (normal). Experiment 1 confirms anomalies have
markedly shorter path lengths.

Isolation Forest's advantages are decisive in practice: it is **fast** ($O(n\log n)$, sub-samples the
data), needs **no distance computation** (so it scales to **high dimensions** where distance methods
break), makes **no density assumption**, and has few parameters. It is the sensible default for a new
anomaly-detection problem, especially in high dimensions.

---

## 6. One-Class SVM

For **novelty detection** with clean training data, the **One-Class SVM** learns a boundary that
encloses the normal region. It maps the data to a feature space (via a kernel, usually RBF) and finds
the maximum-margin hyperplane separating the data from the *origin* — equivalently, the smallest region
containing a specified fraction $\nu$ of the training points. New points *outside* the learned boundary
are novelties.

It is the kernelized, boundary-based cousin of the SVM ([03.07](../../03-supervised-learning/07-svm/)),
and it can model complex non-linear normal regions. But it is **sensitive to hyperparameters** ($\nu$
and the kernel width $\gamma$), scales poorly ($O(n^2)$–$O(n^3)$), and is not robust to contamination
in the "clean" training set. Use it for novelty detection on modest, genuinely clean data; reach for
Isolation Forest otherwise.

---

## 7. Reconstruction-based detection

A different principle: **if you can compress and rebuild normal data well, anomalies will rebuild
badly**. Fit a model that reconstructs the data through a bottleneck; the **reconstruction error** is
the anomaly score.

- **PCA reconstruction** ([04.06](../06-linear-dimensionality-reduction/)). Project onto the top $k$
  principal components and back; normal points (which live near the low-dimensional subspace)
  reconstruct with small error, while anomalies (off the subspace) reconstruct poorly. Cheap and
  effective when normal data has linear low-rank structure. Experiment 4 uses PCA reconstruction error
  as an anomaly score.
- **Autoencoders** ([07.xx](../../07-deep-learning/)). A neural network with a bottleneck learns a
  *nonlinear* low-dimensional manifold of normal data; anomalies off that manifold reconstruct poorly.
  The deep-learning generalization, for images and complex structured data.

Reconstruction methods shine when "normal" has strong (linear or nonlinear) low-dimensional structure —
images, sensor readings, structured records — and the anomalies violate it.

---

## 8. Evaluation without labels

Evaluating anomaly detectors is genuinely hard because you usually **lack labels** — if you knew which
points were anomalies, you might not need the detector. Strategies:

- **When some labels exist** (a validation set of known anomalies): use **ROC-AUC** and **average
  precision** ([05.03](../../05-model-evaluation/03-classification-metrics/)). Because anomalies are
  rare, **average precision / precision@k** is more informative than ROC-AUC (the imbalance argument of
  [05.03 §7](../../05-model-evaluation/03-classification-metrics/)). Experiment 1 scores detectors by
  ROC-AUC against injected labels.
- **The contamination parameter.** Most methods output a continuous score; turning it into a decision
  needs a threshold, usually set by an assumed **contamination rate** (the expected fraction of
  anomalies). Getting this wrong trades false alarms against misses — the same cost tradeoff as any
  classifier's threshold ([05.03 §11](../../05-model-evaluation/03-classification-metrics/)).
- **Rank and inspect.** With no labels at all, rank points by anomaly score and have a domain expert
  inspect the top ones — the practical reality in most deployments.

The honest summary: anomaly detection is often evaluated informally, and a detector that produces a
useful *ranking* of suspicious points is frequently all you can ask for.

---

## 9. Choosing a method

| Situation | Method |
|---|---|
| New problem, high dimensions, want a fast default | **Isolation Forest** |
| Normal data has clusters of varying density | **LOF** |
| Low-dim data, normal is a simple density | **Gaussian / GMM** ([04.04](../04-gaussian-mixtures/)) |
| Clean training data, novelty detection | **One-Class SVM** |
| Normal data has low-rank (linear) structure | **PCA reconstruction error** |
| Images / complex structured data | **Autoencoder reconstruction** |

Practical guidance: **start with Isolation Forest** — it is fast, high-dimensional-friendly,
assumption-light, and usually competitive. Add **LOF** if densities vary, a **density model** if the
data is low-dimensional and you want probabilistic scores, and **reconstruction** if normal data is
strongly low-rank. Ensembling several detectors (averaging ranks) is a robust practical trick.

---

## 10. Common misconceptions

**"Anomaly detection is just classification."**
Anomalies are rare and often unlabeled, and new types appear unseen — so you model *normal* and flag
deviations rather than training a supervised classifier (§1–§2).

**"A single Gaussian / z-score is enough."**
It assumes one Gaussian blob and fails on multimodal or non-Gaussian normal data (§3). Use a GMM,
Isolation Forest, or LOF.

**"Global distance to neighbors always works."**
It uses one global threshold and fails when normal regions have different densities; LOF's local
*ratio* is the fix (§4).

**"Isolation Forest models the normal distribution."**
It does the opposite — it *isolates* anomalies via short random-partition path lengths, needing no
density model and no distances (§5).

**"Use ROC-AUC to evaluate."**
Under heavy imbalance, average precision / precision@k is more informative than ROC-AUC (§8,
[05.03 §7](../../05-model-evaluation/03-classification-metrics/)).

**"Set the contamination parameter to a default and forget it."**
It sets the decision threshold and directly trades false alarms against misses; tune it to the cost of
each error (§8).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — Isolation Forest (random isolation trees + path-length
  scoring), LOF (local reachability density ratio), a Gaussian density scorer, and PCA reconstruction
  error, all in NumPy, verified against `sklearn`. Five experiments: (1) Isolation Forest — anomalies
  have shorter path lengths, scored by ROC-AUC; (2) LOF catching a *local* anomaly that global k-NN
  distance misses; (3) Isolation Forest vs a single-Gaussian detector on multimodal data; (4) PCA
  reconstruction error as an anomaly score; (5) the contamination threshold trading precision against
  recall.
- **[exercises.md](exercises.md)** — derive the Isolation Forest score and LOF, implement each
  detector, reproduce every experiment.
- **[references.md](references.md)** — Liu et al. (Isolation Forest), Breunig et al. (LOF), Schölkopf
  et al. (One-Class SVM).

**Next**: [04.09 — Association Rule Mining](../09-association-rules/) — finding frequent itemsets and
"if-then" rules in transactional data (market-basket analysis), the last piece of unsupervised learning.
