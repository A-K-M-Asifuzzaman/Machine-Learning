# 03.08 — Decision Trees

> **Prerequisites**: [00.05 §9](../../00-mathematical-foundations/05-information-theory/) (entropy,
> information gain — a tree split *is* mutual information).
> **You will be able to**: derive the splitting criteria from first principles, explain why greedy
> trees are provably suboptimal yet universally used, prune correctly, and say exactly why a single
> tree overfits — which sets up every ensemble in [Part 6](../../06-ensembles/).

---

## Table of contents

1. [The idea](#1-the-idea)
2. [Anatomy of a tree](#2-anatomy-of-a-tree)
3. [The greedy recursion](#3-the-greedy-recursion)
4. [Splitting criteria for classification](#4-splitting-criteria-for-classification)
5. [Gini vs entropy](#5-gini-vs-entropy)
6. [Splitting criteria for regression](#6-splitting-criteria-for-regression)
7. [Finding the best split](#7-finding-the-best-split)
8. [Categorical features](#8-categorical-features)
9. [Why trees overfit](#9-why-trees-overfit)
10. [Pruning](#10-pruning)
11. [Missing values](#11-missing-values)
12. [Feature importance, and why it lies](#12-feature-importance-and-why-it-lies)
13. [Strengths and the fatal weakness](#13-strengths-and-the-fatal-weakness)
14. [CART, ID3, C4.5](#14-cart-id3-c45)
15. [Common misconceptions](#15-common-misconceptions)

---

## 1. The idea

> Ask a sequence of yes/no questions about the features. Each answer narrows down the region of
> input space until you reach a pure enough group to make a prediction.

That is the whole model, and its appeal is that it matches how people actually reason: "Is income
> \$50k? If yes, is age > 30? If yes, approve." A decision tree is a flowchart that a machine
learns from data.

Three properties make trees unlike everything else in Part 3:

- **They partition the input space into axis-aligned boxes**, and predict a constant in each box.
  No smoothness, no linearity — a *piecewise-constant* function
  ([03.03](../03-basis-expansion/), but with the pieces learned).
- **They need almost no preprocessing.** Scaling is irrelevant (a split at income > 50,000 is the
  same after standardizing), and they handle mixed numeric/categorical data natively. This is a
  genuine practical advantage over everything else in this part.
- **They are the base learner of the models that actually win on tabular data.** A single tree is
  mediocre; that is *why* [Part 6](../../06-ensembles/) exists. Understanding exactly how a tree
  overfits (§9) is the foundation for understanding why bagging and boosting fix it.

---

## 2. Anatomy of a tree

| Term | Meaning |
|---|---|
| **Root** | the top node; sees all the data |
| **Internal node** | a test on one feature, e.g. $x_j \le t$ |
| **Branch** | an outcome of a test (left = true, right = false) |
| **Leaf** | a terminal node; holds a prediction |
| **Depth** | longest root-to-leaf path |

A prediction routes an input from the root to a leaf by answering each test, then returns the
leaf's value: the **majority class** (classification) or the **mean target** (regression) of the
training points that landed there.

Each internal node splits on **one feature at a time**, so every decision boundary is
**perpendicular to a feature axis**. This is the source of both the tree's interpretability and
its fatal weakness (§13): a diagonal boundary must be approximated by a staircase.

---

## 3. The greedy recursion

Building the optimal tree is **NP-complete** (Hyafil & Rivest, 1976) — you cannot search all
possible trees. So every practical algorithm is **greedy**: at each node, pick the single split
that looks best *right now*, and recurse, never reconsidering.

```
build(node, data):
    if stopping_condition(data):
        make node a leaf, prediction = majority / mean
        return
    (feature, threshold) = best_split(data)      # greedy: best immediate gain
    split data into left, right by the test
    build(left_child,  left)
    build(right_child, right)
```

**Stopping conditions** (any one triggers a leaf):

- the node is pure (all one class), or
- fewer than `min_samples_split` points remain, or
- `max_depth` is reached, or
- no split improves the criterion by at least `min_impurity_decrease`.

> **Greedy is provably suboptimal and universally used.** A split that looks worse now can enable
> two excellent splits below it — think XOR, where no single feature is informative alone but the
> pair is decisive. The greedy tree cannot see this and makes a bad first split. Yet exhaustive
> search is intractable, and the greedy tree is *good enough* that ensembling it beats any
> practical attempt at optimality. Experiment 4 shows the XOR failure directly. This is a recurring
> pattern in ML: a tractable greedy heuristic plus ensembling beats an intractable exact method.

---

## 4. Splitting criteria for classification

"Best split" means "most reduces impurity." Let $p_k$ be the fraction of class $k$ at a node.

**Entropy** ([00.05 §3](../../00-mathematical-foundations/05-information-theory/)):

$$H = -\sum_{k=1}^{K} p_k\log_2 p_k$$

**Gini impurity** — the probability that two random draws (with replacement) from the node have
*different* labels:

$$G = 1 - \sum_{k=1}^{K} p_k^{2} = \sum_{k\ne k'} p_kp_{k'}$$

Both are 0 for a pure node and maximal for a uniform mix.

**Information gain** — the *reduction* in impurity from a split:

$$\mathrm{IG} = I(\text{parent}) - \sum_{\text{children } c}\frac{n_c}{n}\,I(c)$$

where $I$ is entropy or Gini. **The tree picks the split maximizing information gain.**

> **A split's information gain is exactly the mutual information between the feature and the label**
> ([00.05 §9](../../00-mathematical-foundations/05-information-theory/)). "The tree splits on the
> most informative feature" is not a metaphor — it is $\arg\max_j I(Y; X_j)$, and the tree is a
> greedy mutual-information maximizer. That chapter also showed the trap: information gain is biased
> toward high-cardinality features, which §7 and §12 return to.

---

## 5. Gini vs entropy

They almost always pick the same split. Both are concave functions of $p$, maximal at uniformity,
zero at purity; they differ only in shape.

| | Gini | Entropy |
|---|---|---|
| Formula | $1-\sum p_k^{2}$ | $-\sum p_k\log_2 p_k$ |
| Range (binary) | $[0, 0.5]$ | $[0, 1]$ |
| Cost | cheap (no logs) | logarithm per class |
| Default in | **CART / sklearn** | ID3 / C4.5 |
| Bias | slightly toward larger, purer partitions | slightly toward balanced partitions |

**Which to use? It rarely matters** — empirically they agree on >98% of splits, and no study has
found one reliably better. Gini is the sklearn default because it avoids logarithms. Choose based
on speed, not accuracy, and do not spend hyperparameter budget here.

There is a clean unifying view: Gini is the **first-order Taylor approximation** of entropy around
$p_k = 1/K$ (Experiment 2 verifies this numerically), which is why they behave so similarly.

---

## 6. Splitting criteria for regression

For a continuous target, impurity is **variance**, and the reduction being maximized is exactly
the regression counterpart of information gain:

$$I(\text{node}) = \frac{1}{n}\sum_{i\in\text{node}}(y_i - \bar{y})^{2} = \mathrm{Var}(y)$$

Splitting to minimize the weighted child variance is equivalent to minimizing total squared error —
so a regression tree is doing **greedy piecewise-constant least squares**, and the leaf prediction
$\bar{y}$ is the value minimizing squared error in that box
([00.03 §9.4](../../00-mathematical-foundations/03-probability/)).

Variants:

- **MSE** (the default): leaf predicts the mean; sensitive to outliers.
- **MAE**: leaf predicts the median; robust but slower (no closed-form update during the scan).
- **Friedman MSE**: a modified criterion used by gradient boosting
  ([06.04](../../06-ensembles/04-gradient-boosting/)).

---

## 7. Finding the best split

For each feature, for each candidate threshold, compute the impurity reduction; keep the best over
all (feature, threshold) pairs.

**Continuous features.** Sort the values; the only thresholds worth testing are **midpoints
between adjacent distinct values** — impurity only changes when a point crosses the split. That is
$n-1$ candidates per feature.

**The efficient scan.** Naively, evaluating each threshold from scratch is $O(n)$, giving
$O(n^{2})$ per feature. But if you sort once and sweep left to right, moving one point across the
boundary at a time, the class counts (and hence the impurity) update in $O(1)$:

$$\text{total per feature} = \underbrace{O(n\log n)}_{\text{sort}} + \underbrace{O(n)}_{\text{sweep}}$$

Across $d$ features and $O(n)$ nodes, a balanced tree costs $O(d\,n\log^{2} n)$. This incremental
sweep is *the* implementation detail that makes trees fast; Experiment 1 shows it agreeing with
the naive version at a fraction of the cost.

> **Information gain's cardinality bias, again.** A feature with many distinct values offers many
> thresholds and can nearly always find one that separates the training data — an ID column
> achieves perfect information gain and zero generalization
> ([00.05 §9](../../00-mathematical-foundations/05-information-theory/), Experiment 5 there). C4.5's
> **gain ratio** divides information gain by the split's own entropy to penalize this; CART sidesteps
> it with binary splits and Gini. Either way: watch for it, and never let an identifier column into
> a tree.

---

## 8. Categorical features

A categorical feature with $q$ levels has $2^{q-1}-1$ possible binary partitions — exponential, so
exhaustive search is infeasible for large $q$.

| Approach | How | Used by |
|---|---|---|
| **Multiway split** | one branch per category | ID3, C4.5 |
| **Binary + sorting trick** | for binary classification, sort categories by mean label; the optimal split is among the $q-1$ contiguous cuts | CART |
| **One-hot first** | encode, then treat as binary numerics | sklearn (which has no native categorical support) |

The sorting trick (Breiman 1984) is elegant: for a two-class problem, ordering the $q$ categories
by their positive-class rate reduces the exponential search to a linear one, *provably* without
losing the optimum. It does not generalize to multiclass or regression with the same guarantee.

⚠️ **One-hot encoding hurts trees.** It shatters one informative feature into many sparse binary
ones, each carrying a fraction of the signal, so the tree needs many splits to reconstruct what one
categorical split would have captured — and each sparse column has weak individual gain, so the
tree may never select it. This is why **LightGBM and CatBoost handle categoricals natively**
([06.05](../../06-ensembles/05-modern-gbdts/)) and consistently beat one-hot + sklearn on
categorical data.

---

## 9. Why trees overfit

A tree grown to purity **memorizes the training set**: keep splitting and every leaf eventually
contains one point, achieving zero training error and terrible test error — the exact signature of
overfitting.

Two independent mechanisms:

**1. Unlimited capacity.** With enough depth a tree can isolate every training point, so its
effective complexity grows without bound. Nothing in the greedy recursion stops it.

**2. High variance.** This is the deeper problem and the one that motivates
[Part 6](../../06-ensembles/). Trees are **unstable**: a small change in the data can change which
split wins at the root, and because every descendant depends on that choice, the *entire tree below*
reorganizes. A different bootstrap sample gives a wildly different tree.

> **This instability is not a bug to be fixed in isolation — it is the property ensembles exploit.**
> Averaging many high-variance, low-bias, *decorrelated* trees cuts variance without adding bias
> ([00.03 §4.3](../../00-mathematical-foundations/03-probability/)): that is bagging and random
> forests ([06.01](../../06-ensembles/01-bagging/), [06.02](../../06-ensembles/02-random-forests/)).
> The tree's greatest weakness as a standalone model is exactly what makes it the ideal ensemble
> member. Experiment 3 measures the instability directly — retraining on bootstrap samples and
> watching the root split flip.

**Controlling a single tree:** `max_depth`, `min_samples_leaf`, `min_samples_split`,
`max_leaf_nodes`, `min_impurity_decrease`, or post-pruning (§10). All trade fit for generalization.

---

## 10. Pruning

Two philosophies:

**Pre-pruning (early stopping).** Stop growing when a limit is hit (`max_depth`, etc.). Fast, but
**short-sighted** — it can stop before a split that would have enabled valuable splits below it,
the same greedy blindness as §3.

**Post-pruning.** Grow the full tree, then remove subtrees that do not earn their complexity. More
reliable because it sees the whole tree before deciding.

**Cost-complexity pruning** (CART's method, `ccp_alpha` in sklearn). Minimize

$$R_\alpha(T) = R(T) + \alpha\,|T|$$

where $R(T)$ is the tree's error, $|T|$ its number of leaves, and $\alpha \ge 0$ the complexity
price. This is **exactly the regularized-objective pattern** of
[03.02](../02-regularized-linear-models/): fit plus a penalty on complexity, with $\alpha$ playing
the role of $\lambda$. As $\alpha$ increases from 0, the tree collapses from full to a single node
along a finite sequence of subtrees, and cross-validation picks the $\alpha$ minimizing validation
error. Experiment 5 traces that path.

---

## 11. Missing values

Trees handle missing data more gracefully than most models, which do not handle it at all.

| Strategy | How |
|---|---|
| **Surrogate splits** (CART) | at each node, store backup features correlated with the primary; if the primary is missing, use a surrogate |
| **Default direction** (XGBoost, LightGBM) | learn, per node, which way missing values should go, by trying both and keeping the better |
| **Missing as a category** | treat "missing" as its own value — valid when missingness is informative (MNAR, [02.02](../../02-data/02-cleaning-and-missing-data/)) |
| Imputation first | fill in, then treat as complete — loses the missingness signal |

The **default-direction** approach is why gradient boosting libraries need no imputation and often
beat carefully-imputed pipelines: missingness is frequently *informative*, and learning where to
send it captures that signal instead of destroying it.

---

## 12. Feature importance, and why it lies

The default (**mean decrease in impurity**, MDI): sum the impurity reduction each feature
contributes across all its splits, weighted by node size. sklearn's `feature_importances_`.

**It is biased, and the bias is severe enough to mislead you.** MDI systematically inflates:

- **high-cardinality features** — more thresholds means more chances to reduce impurity by luck,
  the §7 bias yet again;
- **continuous over categorical** features, for the same reason;
- features that happen to be split **near the root** (larger node sizes carry more weight).

A feature can rank as "most important" purely because it has many unique values, even when it is
pure noise. Experiment 6 demonstrates this concretely: a **continuous noise column** (no
relationship to the label) scores an MDI importance of **0.85 — the highest of all features** —
while the one binary feature the label actually depends on scores 0.11. Permutation importance
flips the ranking correctly (real 0.05, noise −0.01), because shuffling a column the model never
truly relied on cannot hurt a held-out score.

**Trustworthy alternatives:**

| Method | Idea | Cost |
|---|---|---|
| **Permutation importance** | shuffle one feature, measure the accuracy drop | retrain-free, model-agnostic |
| **Drop-column importance** | retrain without the feature | expensive, most direct |
| **SHAP** | game-theoretic attribution | principled, the current standard ([17.02](../../17-explainable-ai/02-post-hoc/)) |

> **Never report MDI importance for anything that matters** — feature selection, a stakeholder
> story, a scientific claim. Use permutation importance or SHAP, computed on a *held-out* set.

---

## 13. Strengths and the fatal weakness

**Strengths:**
- **Interpretable** — you can read the rules (for a small tree)
- **No preprocessing** — scale-invariant, handles mixed types, no encoding needed for many libraries
- **Handles nonlinearity and interactions** automatically — the whole point, versus a linear model
- **Fast prediction** — $O(\text{depth})$, a handful of comparisons
- **Native missing-value and categorical handling** (in good implementations)

**The fatal weakness — high variance (§9).** A single tree is rarely competitive: unstable,
overfits easily, and its axis-aligned boxes approximate a diagonal boundary with an ugly staircase.

**The second weakness — no smoothness.** Piecewise-constant predictions mean trees **cannot
extrapolate** (like KNN, [03.06 §7](../06-knn/)) and produce discontinuous jumps. For a genuinely
smooth relationship, a tree is the wrong tool.

> **This is why the chapter matters more than the model.** You will almost never deploy a single
> decision tree. You will deploy **thousands of them** — random forests, gradient boosting, XGBoost,
> LightGBM — and every one of those is built on exactly the mechanics above. The single tree is the
> unit of study; [Part 6](../../06-ensembles/) is where it becomes state of the art.

---

## 14. CART, ID3, C4.5

| | ID3 (1986) | C4.5 (1993) | CART (1984) |
|---|---|---|---|
| Author | Quinlan | Quinlan | Breiman et al. |
| Splits | multiway | multiway | **binary only** |
| Criterion | information gain | **gain ratio** | **Gini** (classif.), MSE (regr.) |
| Categorical | native multiway | native | via sorting trick |
| Regression | ✗ | ✗ | **✓** |
| Missing values | ✗ | fractional instances | surrogate splits |
| Pruning | ✗ | error-based | **cost-complexity** |

**CART is what `sklearn` implements** (binary, Gini/MSE, cost-complexity pruning) and what every
modern ensemble is built on. C4.5's gain ratio is the principled fix for the cardinality bias (§7)
but is rarely used now; ID3 is of historical interest. When someone says "decision tree" in a
modern ML context, they mean CART.

---

## 15. Common misconceptions

**"Decision trees need feature scaling."**
No. A split at $x_j \le t$ is unaffected by scaling — one of the few models where this is genuinely
true (§1).

**"Gini and entropy give very different trees."**
They agree on >98% of splits; the choice is about speed, not accuracy (§5).

**"A deeper tree is always better."**
A tree grown to purity memorizes the training set (§9). Depth is the primary overfitting knob.

**"`feature_importances_` tells me which features matter."**
MDI importance is biased toward high-cardinality and continuous features and can rank noise first
(§12). Use permutation importance or SHAP.

**"Trees find the optimal split sequence."**
Greedy and provably suboptimal (§3). Optimal trees are NP-complete.

**"One-hot encode categoricals for your tree."**
It fragments the signal and weakens each split. Prefer native categorical handling (§8).

**"A single tree is a strong model."**
It is a *weak* learner — high variance, mediocre alone. Its value is as an ensemble member (§13).

**"Trees can't handle missing data."**
Good implementations handle it natively and often better than imputation (§11).

**"Trees can extrapolate."**
Piecewise-constant: predictions are bounded by the training targets, exactly like KNN (§13).

---

## Files in this chapter

| File | Contents |
|---|---|
| [`from_scratch.py`](from_scratch.py) | CART classifier and regressor with Gini/entropy/MSE, the efficient incremental split scan, cost-complexity pruning, and permutation importance — verified against sklearn, with experiments on the fast scan, Gini-as-entropy's-Taylor-approximation, tree instability, greedy failure on XOR, and MDI's cardinality bias |
| [`exercises.md`](exercises.md) | Derivation, implementation, and interview questions |
| [`references.md`](references.md) | Exact sections used |

**Previous**: [03.07 — Support Vector Machines](../07-svm/) ·
**Next**: [03.09 — Perceptron](../09-perceptron/)
