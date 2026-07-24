# 06.02 — Random Forests

> **Prerequisites**: [06.01](../01-bagging/) — this chapter is bagging plus one idea, and it attacks
> the exact number ([06.01 §3](../01-bagging/)) that bagging could not.
> **You will be able to**: explain why subsampling features at each split lowers the variance floor,
> tune `max_features` from the bias-variance tradeoff rather than folklore, read (and distrust) the
> importances correctly, and say why a forest cannot extrapolate.

---

## Table of contents

1. [Bagging plus one idea](#1-bagging-plus-one-idea)
2. [The algorithm](#2-the-algorithm)
3. [Why feature subsampling works](#3-why-feature-subsampling-works)
4. [Tuning max_features](#4-tuning-max_features)
5. [The other hyperparameters](#5-the-other-hyperparameters)
6. [Out-of-bag error, again](#6-out-of-bag-error-again)
7. [Feature importance — two flavours, both with caveats](#7-feature-importance--two-flavours-both-with-caveats)
8. [Extremely randomized trees](#8-extremely-randomized-trees)
9. [Proximities and the forest as a kernel](#9-proximities-and-the-forest-as-a-kernel)
10. [Strengths, and the two real weaknesses](#10-strengths-and-the-two-real-weaknesses)
11. [Random forests vs gradient boosting](#11-random-forests-vs-gradient-boosting)
12. [Common misconceptions](#12-common-misconceptions)

---

## 1. Bagging plus one idea

[06.01](../01-bagging/) ended on a precise limitation. Bagged trees have variance

$$\rho\sigma^{2} + \frac{1-\rho}{B}\sigma^{2}
\;\xrightarrow{\;B\to\infty\;}\; \rho\sigma^{2}$$

The second term vanishes with more trees, but the first — the floor $\rho\sigma^2$ — does not, and
it is set by how *correlated* the trees are. Plain bagging leaves $\rho$ high, because every tree
tends to split on the same few strong features near the root, so the trees look alike.

The random forest (Breiman, 2001) adds **one ingredient** to attack exactly that:

> **At each split, consider only a random subset of the features.**

By hiding the strong features from most splits, different trees are *forced* to use different
features, so they disagree more — $\rho$ drops, the floor $\rho\sigma^2$ drops, and the ensemble's
variance drops with it. Same trees, same number of them, lower error. That is the entire idea, and
[06.01's Experiment 4](../01-bagging/) already measured it working.

**Nothing else changes.** A random forest *is* bagging with this one addition. Everything from
[06.01](../01-bagging/) — the bootstrap, OOB estimation, "reduces variance not bias," "use deep
trees" — carries over unchanged.

---

## 2. The algorithm

```
random_forest(data, B, max_features):
    for b in 1..B:
        sample_b = bootstrap(data)                 # bagging: resample the ROWS
        tree_b   = grow_tree(sample_b,
                             feature_subset = random max_features at EACH split)
    return ensemble

predict(x):
    regression:     mean of tree_b(x)
    classification: soft vote (mean of predicted probabilities)
```

Two sources of randomness, and both matter:

| Randomness | What it randomizes | Effect |
|---|---|---|
| **Bootstrap** (from bagging) | the rows | different trees see different data |
| **Feature subsampling** (new) | the features at each split | different trees use different features |

The feature subset is redrawn **at every split**, not once per tree — this is the detail that makes
it work. A single random subset per tree would barely decorrelate them; redrawing at each node means
even two trees that started identically diverge as they grow deeper.

The trees are grown **fully / deep and unpruned**, for the same reason as bagging
([06.01 §8](../01-bagging/)): you want low bias, because averaging fixes variance and not bias.

---

## 3. Why feature subsampling works

The mechanism is entirely the $\rho$ in $\rho\sigma^2$, and it is worth being precise about how the
subsampling lowers it.

Suppose two features, $x_1$ and $x_2$, are both strongly predictive, with $x_1$ marginally the
stronger. In **plain bagging**, nearly every tree picks $x_1$ for its root split — it is the best
split on almost every bootstrap sample — so the trees share their most important decision and are
highly correlated. In a **random forest** with `max_features` small, $x_1$ is simply *not available*
at many splits (it was not in the random subset), forcing those trees to split on $x_2$ (or
something else). Now the trees genuinely differ, and $\rho$ falls.

**There is a cost, and it sets up the tradeoff.** Hiding the best feature from a split means that
split is made on a *worse* feature, so each individual tree is slightly weaker — higher bias,
higher $\sigma^2$. So feature subsampling:

- **lowers $\rho$** (good — the trees decorrelate), but
- **raises $\sigma^2$** (bad — each tree is a little worse).

The floor is $\rho\sigma^2$, a *product*, so `max_features` trades one factor against the other. Too
many features and the trees stay correlated ($\rho$ high); too few and each tree is too weak
($\sigma^2$ high). The optimum is in between — which is §4, and which
[06.01's Experiment 4](../01-bagging/) already showed as an accuracy peak at intermediate
`max_features`.

---

## 4. Tuning max_features

`max_features` is the one hyperparameter unique to random forests, and the defaults are good
starting points backed by both theory and Breiman's experiments:

| Task | Default `max_features` | Reasoning |
|---|---|---|
| **Classification** | $\sqrt{d}$ | strong decorrelation; empirically robust |
| **Regression** | $d/3$ | regression tolerates less decorrelation; targets are noisier |

| `max_features` | $\rho$ | Per-tree strength | Net effect |
|---|---|---|---|
| $d$ (all) | high | strongest | plain bagging — floor stays up |
| $\sqrt{d}$ | low | slightly weaker | usually optimal for classification |
| $1$ | lowest | much weaker | over-decorrelated; trees too weak |

**Do tune it** — it is the highest-leverage knob a random forest has — but tune it on a log-ish grid
around the default, not from scratch. And note the interaction with the data: if only a *few*
features are informative and the rest are noise, a small `max_features` will often pick a useless
feature at a split, so you may need it larger; if *many* features are informative, small works well.

---

## 5. The other hyperparameters

Everything else is inherited from the trees ([03.08](../../03-supervised-learning/08-decision-trees/))
and from bagging, and matters less than `max_features`:

| Hyperparameter | Effect | Guidance |
|---|---|---|
| `n_estimators` ($B$) | more trees → lower variance, to the floor | more is never worse; 100-500 typical. A **compute** knob, not a regularizer |
| `max_depth` | tree depth | usually leave unlimited (deep = low bias) |
| `min_samples_leaf` | minimum leaf size | the main *pruning* knob if you must regularize; 1-5 |
| `max_samples` | bootstrap size | < 1.0 speeds training, adds a little decorrelation |
| `bootstrap` | resample rows? | `True` (off = no OOB, less decorrelation) |

> **The single most important non-obvious fact**: `n_estimators` **cannot overfit**. Adding trees
> only drives variance toward its floor; test error decreases monotonically and plateaus. So set
> $B$ as high as your compute budget allows and never worry about it as a source of overfitting.
> This is emphatically **not** true of gradient boosting (§11), where more estimators *can* overfit
> — a distinction that trips people up constantly.

---

## 6. Out-of-bag error, again

Random forests inherit **OOB estimation** unchanged from bagging
([06.01 §6](../01-bagging/)): each point is out-of-bag for ~37% of the trees, so it can be scored by
only the trees that never trained on it, giving a nearly-free, nearly-unbiased test-error estimate.

For a random forest this is especially valuable because forests are usually used on datasets where
you would rather not sacrifice rows to a validation set. `oob_score=True` in sklearn. The same
caveat holds: reliable only once $B$ is a few hundred.

---

## 7. Feature importance — two flavours, both with caveats

Random forests are prized for telling you which features matter. They offer two methods, and
[03.08 §12](../../03-supervised-learning/08-decision-trees/) already delivered the warning that
applies to the first.

**Mean decrease in impurity (MDI)** — sum each feature's impurity reduction over all splits in all
trees. This is `feature_importances_`. **It is biased**, in the same direction as for a single tree
([03.08 §12](../../03-supervised-learning/08-decision-trees/)): it inflates high-cardinality and
continuous features. Averaging over a forest **reduces** this bias but does **not eliminate** it —
a forest tolerates a stronger real signal than a single tree before being fooled, but a *weak-enough*
real feature buried among high-cardinality noise still loses. Experiment 3 measures exactly this:
sweeping the real signal's strength, MDI ranks it correctly when strong, but by a class rate of
0.42/0.58 the continuous noise's MDI *overtakes* the genuinely predictive feature.

**Permutation importance** — shuffle one feature across the OOB (or test) rows and measure the drop
in accuracy. Unbiased with respect to cardinality, model-agnostic, and the method you should
actually report.

> ⚠️ **Both methods share a distinct failure on *correlated* features.** If $x_1$ and $x_2$ are
> duplicates, the forest splits on each about half the time, so MDI *splits* the importance between
> them — each looks half as important as it is, and a naive reading concludes neither matters.
> Permutation has the mirror problem: shuffling $x_1$ barely hurts, because the forest falls back
> on its correlated twin $x_2$, so *both* look unimportant. Neither method handles correlated
> features honestly. The fixes — conditional permutation importance, or SHAP with a correlation-aware
> background — are in [17.02](../../17-explainable-ai/02-post-hoc/). Experiment 4 measures both
> failures.

**The practical rule**: use permutation importance on held-out data for the headline story; be
deeply suspicious of any importance ranking when features are correlated; and for anything
consequential, reach for SHAP.

---

## 8. Extremely randomized trees

**Extra-Trees** (Geurts et al., 2006) push the randomization one step further: instead of searching
for the *best* threshold on each candidate feature, they pick a **random** threshold and keep the
best among those random choices.

| | Random forest | Extra-Trees |
|---|---|---|
| Row sampling | bootstrap | **whole dataset** (usually) |
| Split threshold | best (searched) | **random**, then best-of-random |
| $\rho$ | low | **lower** |
| Per-tree bias | low | slightly higher |
| Training speed | fast | **faster** (no threshold search) |

The extra randomness lowers $\rho$ further and eliminates the expensive threshold search, so
Extra-Trees are faster to train and sometimes generalize better — the same $\rho$-vs-$\sigma^2$
tradeoff pushed toward more decorrelation. Worth trying as a drop-in alternative; `ExtraTreesClassifier`
in sklearn.

---

## 9. Proximities and the forest as a kernel

A less-known but genuinely useful byproduct. Define the **proximity** of two points as the fraction
of trees in which they land in the **same leaf**:

$$\text{prox}(\mathbf{x}_i, \mathbf{x}_j) = \frac{1}{B}\sum_{b} \mathbb{1}[\text{leaf}_b(\mathbf{x}_i) = \text{leaf}_b(\mathbf{x}_j)]$$

This is a **learned, supervised similarity measure** — two points are "close" if the forest keeps
routing them together, which happens when they behave similarly with respect to the *target*. It is
a valid kernel, and it gives a random forest capabilities most models lack:

- **Missing-value imputation** — fill in using proximity-weighted neighbours.
- **Outlier detection** — points with low proximity to everything are anomalies.
- **Visualization** — feed the proximity matrix to MDS or t-SNE
  ([04.07](../../04-unsupervised-learning/07-manifold-learning/)).
- **Clustering** — cluster on proximities to get a supervised-similarity clustering.

The connection to [03.06](../../03-supervised-learning/06-knn/) is exact: a random forest is, in a
sense, an *adaptive nearest-neighbour* method where the notion of "near" is learned from the labels
rather than fixed as Euclidean distance — which is precisely the "learned metric" that rescues KNN
in high dimensions ([03.06 §8.4](../../03-supervised-learning/06-knn/)).

---

## 10. Strengths, and the two real weaknesses

**Strengths** — this is why the random forest is the default first model for tabular data:

- **Excellent out of the box.** Strong accuracy with default hyperparameters; forgiving.
- **Almost no preprocessing.** Scale-invariant, handles mixed types, robust to outliers and
  irrelevant features (inherited from trees).
- **Cannot overfit via `n_estimators`** (§5) — a rare and comforting property.
- **Free OOB error** and useful **importances** (with §7's caveats).
- **Parallel** — trees are independent, unlike boosting.
- **Robust** — Breiman's forests are famously hard to break.

**The two weaknesses that actually matter:**

1. **It cannot extrapolate.** Every prediction is an average of leaf means, so it is bounded by the
   training targets' range — exactly like KNN ([03.06 §7](../../03-supervised-learning/06-knn/)) and
   a single tree ([03.08 §13](../../03-supervised-learning/08-decision-trees/)). For a regression
   target that trends beyond the training data, a forest flatlines. Experiment 5 shows it.
2. **Gradient boosting usually beats it on accuracy.** On most tabular benchmarks a well-tuned
   XGBoost/LightGBM edges out a random forest, because boosting reduces *bias* as well as variance
   (§11). The forest's advantage is that it gets 95% of the way there with almost no tuning.

Lesser drawbacks: large memory footprint ($B$ full trees), slower inference than a linear model,
and loss of the single tree's interpretability.

---

## 11. Random forests vs gradient boosting

The two dominant tree ensembles, and the contrast is the whole story of Part 6:

| | Random forest | Gradient boosting |
|---|---|---|
| Trees are | **independent** (parallel) | **sequential** (each fixes the last) |
| Reduces | **variance** | **bias** (and variance) |
| Base trees | **deep**, low-bias | **shallow**, high-bias (stumps) |
| `n_estimators` overfits? | **no** | **yes** |
| Tuning | easy, forgiving | fiddly, high-leverage |
| Out of the box | **excellent** | good, needs care |
| Best tuned accuracy | very good | **usually best** |
| Parallel training | **yes** | no (sequential) |

**The one-line summary**: a random forest averages many *strong, independent* trees to cut variance;
gradient boosting adds many *weak, dependent* trees to cut bias. They sit at opposite ends of the
bias-variance strategy, and knowing which problem you have — too much variance or too much bias —
tells you which to reach for. Boosting is [06.03](../03-boosting-theory/) onward.

> **Practical default**: start with a random forest to get a strong, tuning-free baseline and a
> feel for the problem, then switch to gradient boosting if you need to squeeze out the last few
> points and can afford the tuning.

---

## 12. Common misconceptions

**"A random forest is a fundamentally different algorithm from bagging."**
It is bagging plus feature subsampling at each split. One ingredient (§1).

**"More trees can overfit a random forest."**
No. `n_estimators` only reduces variance toward the floor; test error never increases with $B$
(§5). (This *is* true of boosting — do not confuse them.)

**"`feature_importances_` tells me which features matter."**
MDI is biased toward high-cardinality features and splits importance across correlated ones (§7).
Use permutation importance, and be wary even then.

**"Feature subsampling always helps."**
It helps up to a point; too small a `max_features` makes each tree too weak and hurts (§3-§4).

**"Random forests don't overfit at all."**
They resist overfitting, but a forest of fully-grown trees on noisy data still has irreducible
variance and can overfit the noise — `min_samples_leaf` is the knob if so.

**"A random forest can predict trends beyond its training data."**
It cannot extrapolate — predictions are bounded by the training targets (§10).

**"Random forests are obsolete now that we have XGBoost."**
They remain the best *tuning-free* tabular baseline, give free OOB error and proximities, and are
easier to reason about. Boosting wins on tuned accuracy, not on effort-to-value (§11).

**"Averaging many trees fixes MDI's importance bias."**
It does not — the bias is systematic and survives averaging (§7).

---

## Files in this chapter

| File | Contents |
|---|---|
| [`from_scratch.py`](from_scratch.py) | A random forest (classifier and regressor) built on the fast tree from [06.01](../01-bagging/), with per-split feature subsampling, OOB scoring, MDI and permutation importance, and proximities — verified against sklearn, with experiments on the $\rho$-floor drop, `max_features` tuning, MDI bias, the correlated-feature importance failure, and the no-extrapolation limit |
| [`exercises.md`](exercises.md) | Derivation, implementation, and interview questions |
| [`references.md`](references.md) | Exact sources used |

**Previous**: [06.01 — Bagging](../01-bagging/) ·
**Next**: [06.03 — Boosting Theory & AdaBoost](../03-boosting-theory/) — the bias-reducing opposite.
