# 06.01 — Bagging

> **Prerequisites**: [00.03 §4.3](../../00-mathematical-foundations/03-probability/) (variance of a
> sum, correlation), [00.04 §12](../../00-mathematical-foundations/04-statistics-and-inference/)
> (the bootstrap), [03.08 §9](../../03-supervised-learning/08-decision-trees/) (tree instability —
> the raw material).
> **You will be able to**: derive exactly how much variance averaging removes and why correlation
> caps it, explain why bagging helps trees but not linear regression, and use the free out-of-bag
> estimate instead of a validation set.

---

## Table of contents

1. [The idea, and where it comes from](#1-the-idea-and-where-it-comes-from)
2. [The variance reduction, derived](#2-the-variance-reduction-derived)
3. [Why correlation is the enemy](#3-why-correlation-is-the-enemy)
4. [The bootstrap, and the 63% fact](#4-the-bootstrap-and-the-63-fact)
5. [The algorithm](#5-the-algorithm)
6. [Out-of-bag estimation — a free validation set](#6-out-of-bag-estimation--a-free-validation-set)
7. [Which models benefit — and which do not](#7-which-models-benefit--and-which-do-not)
8. [Bagging and the bias-variance decomposition](#8-bagging-and-the-bias-variance-decomposition)
9. [Bagging for classification](#9-bagging-for-classification)
10. [Limitations, and the segue to random forests](#10-limitations-and-the-segue-to-random-forests)
11. [Common misconceptions](#11-common-misconceptions)

---

## 1. The idea, and where it comes from

[03.08 §9](../../03-supervised-learning/08-decision-trees/) ended on a problem: a single decision
tree is **unstable** — resample the data and the whole tree reorganizes. That is low bias (a deep
tree fits anything) bought at the cost of enormous variance.

Bagging (Breiman, 1996) turns that liability into an asset with one move:

> **Train many models on random resamples of the data, and average their predictions.**

The name is short for **b**ootstrap **agg**regat**ing**. The insight is purely statistical: if you
have many noisy-but-unbiased estimates, their average is far less noisy and just as unbiased
([00.03 §12](../../00-mathematical-foundations/03-probability/)). Averaging cannot reduce bias — the
average of $B$ trees fits the same average shape a single tree does — but it can dramatically reduce
variance, and variance is exactly what plagues a single tree.

This is why [03.08](../../03-supervised-learning/08-decision-trees/) insisted the single tree was
only a "unit of study." The unit is high-variance on purpose; bagging is the machine that averages
the variance away.

---

## 2. The variance reduction, derived

Take $B$ estimators, each with variance $\sigma^{2}$, and average them. How much variance does the
average have? This is the entire theory of bagging, and it is one identity from
[00.03 §4.3](../../00-mathematical-foundations/03-probability/).

**The independent case first.** If the $B$ estimators were *independent*, then

$$\mathrm{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} f_b\right)
= \frac{1}{B^{2}}\sum_b \mathrm{Var}(f_b) = \frac{1}{B^{2}}\cdot B\sigma^{2} = \frac{\sigma^{2}}{B}$$

Variance falls by a factor of $B$. Ten independent trees have a tenth the variance; a hundred, a
hundredth. Bias is untouched. This is the dream, and if you could get independent trees you would
simply crank $B$ to infinity.

**The realistic case.** The trees are *not* independent — they are trained on bootstrap samples of
the *same* dataset, so they are positively correlated. Let $\rho$ be the pairwise correlation.
Using $\mathrm{Var}(\sum f_b) = \sum_b\mathrm{Var}(f_b) + \sum_{b\ne b'}\mathrm{Cov}(f_b, f_{b'})$:

$$\boxed{\;\mathrm{Var}\!\left(\frac{1}{B}\sum_b f_b\right)
= \rho\sigma^{2} + \frac{1-\rho}{B}\sigma^{2}\;}$$

*Derivation.* There are $B$ variance terms each $\sigma^2$, and $B(B-1)$ covariance terms each
$\rho\sigma^2$. Dividing by $B^2$:

$$\frac{1}{B^2}\big[B\sigma^2 + B(B-1)\rho\sigma^2\big]
= \frac{\sigma^2}{B} + \frac{(B-1)\rho\sigma^2}{B}
\;\xrightarrow{\;B\to\infty\;}\; \rho\sigma^{2}\ \blacksquare$$

**Read the two terms.** The second term, $\frac{1-\rho}{B}\sigma^2$, vanishes as $B\to\infty$ — this
is the part averaging removes, and it is why more trees help. The first term, $\rho\sigma^2$, does
**not** depend on $B$ at all — it is a **floor** that no amount of averaging can break through.

$$\lim_{B\to\infty}\mathrm{Var}(\text{bagged}) = \rho\sigma^{2}$$

Experiment 1 measures both terms directly: the total variance falling as $\frac{1-\rho}{B}\sigma^2$
and flattening out at exactly $\rho\sigma^2$.

---

## 3. Why correlation is the enemy

The floor $\rho\sigma^2$ is the single most important fact about bagging, and it dictates everything
that comes after:

- **If the trees were independent** ($\rho=0$), variance would go to zero and bagging would be
  perfect.
- **Because they share data** ($\rho > 0$), variance stops falling at $\rho\sigma^2$, and adding
  more trees past a few hundred does essentially nothing.

So the way to make bagging *better* is not more trees — it is **less correlated** trees. Anything
that decorrelates the trees lowers the floor.

> **This is precisely what a random forest does, and why it exists.** Bagging alone bootstraps the
> *rows*. A random forest adds a second source of randomness — it considers only a random subset of
> *features* at each split — which forces different trees to use different features and drives
> $\rho$ down. Lowering $\rho$ lowers the floor $\rho\sigma^2$, so a random forest achieves lower
> variance than plain bagging with the same number of trees. The entire content of
> [06.02](../02-random-forests/) is: *bagging + feature subsampling to attack the $\rho$ in this
> formula.* Experiment 4 shows the floor dropping as decorrelation increases.

---

## 4. The bootstrap, and the 63% fact

Bagging's randomness comes from the **bootstrap** ([00.04 §12](../../00-mathematical-foundations/04-statistics-and-inference/)):
draw $n$ samples from your $n$ training points **with replacement**. Each bootstrap sample is the
same size as the original but contains duplicates and omits some points.

**How many points does a bootstrap sample omit?** The probability a specific point is *not* drawn
in one pick is $1 - 1/n$; over $n$ independent picks,

$$P(\text{point omitted}) = \left(1 - \frac{1}{n}\right)^{n} \;\xrightarrow{\;n\to\infty\;}\; e^{-1} \approx 0.368$$

So each bootstrap sample contains about **63.2%** of the distinct original points, and omits about
**36.8%**. The omitted points are called **out-of-bag** (OOB) for that tree, and §6 turns them into
a free validation set.

Two consequences worth noting: the ~37% overlap between any two bootstrap samples is exactly the
source of the tree correlation $\rho$ in §2 (they share most of their data), and the duplication
within a sample is what makes each tree see a slightly different data distribution.

---

## 5. The algorithm

```
bagging(data, B, base_learner):
    for b in 1..B:
        sample_b = bootstrap(data)           # n draws with replacement
        model_b  = base_learner.fit(sample_b)
    return ensemble of {model_b}

predict(x):
    regression:     mean_b   model_b(x)
    classification: majority_vote or mean of predicted probabilities
```

Three properties make it attractive in practice:

- **Embarrassingly parallel.** The $B$ models are independent, so training scales linearly across
  cores or machines. (Boosting, [06.03](../03-boosting-theory/), is sequential and cannot do this.)
- **Almost no hyperparameters.** $B$ (more is never worse, just slower) and the base learner's own
  settings. There is no learning rate, no early stopping to tune.
- **The base learner should be deep / unpruned.** You *want* high variance and low bias in the base
  learner, because averaging fixes variance and cannot fix bias (§8). This is the opposite of what
  you would do with a single tree.

---

## 6. Out-of-bag estimation — a free validation set

Here is bagging's most elegant feature. Each point is out-of-bag for the ~37% of trees that did not
train on it. So you can evaluate each point using **only the trees that never saw it** — which is
exactly the honest, held-out prediction a validation set gives you, at **zero extra cost**.

$$\hat{y}_i^{\text{OOB}} = \text{aggregate}\big\lbrace \text{model}_b(\mathbf{x}_i) : i \notin \text{sample}_b \big\rbrace$$

The OOB error, averaged over all points, is a nearly unbiased estimate of test error — comparable to
$k$-fold cross-validation ([05.04](../../05-model-evaluation/04-cross-validation/)) but for **free**,
because the resampling was going to happen anyway.

> **Use it.** OOB error means you often do not need a separate validation set for a bagged model,
> which is a real saving on small datasets where every held-out row hurts. `sklearn`'s
> `oob_score=True` computes it. Experiment 2 shows the OOB estimate tracking the true test error
> closely.

One caveat: at small $B$, some points are out-of-bag for very few trees (or none), making their OOB
prediction noisy or undefined. OOB estimates are reliable only once $B$ is large enough — a few
hundred trees.

---

## 7. Which models benefit — and which do not

Bagging reduces variance and does nothing for bias (§2, §8). So it helps **high-variance, low-bias**
models and is useless — or slightly harmful — for stable ones.

| Base learner | Variance | Bagging helps? |
|---|---|---|
| **Deep decision tree** | very high | **enormously** — the canonical case |
| Neural network | high | yes (though expensive) |
| k-NN with small $k$ | moderate | a little |
| **Linear / logistic regression** | low | **essentially not at all** |
| k-NN with large $k$ | low | no |

**Why linear regression gains nothing** is worth understanding, because it makes the mechanism
concrete. OLS is a *stable, low-variance* estimator — a small data change moves the fitted line only
slightly. So the trees-are-different effect that bagging exploits barely exists: every bootstrapped
linear fit is nearly identical, $\rho \approx 1$, and by §2 the variance floor $\rho\sigma^2$ is
essentially the original variance. Averaging near-identical models achieves near-nothing.
Experiment 3 measures this: bagged trees improve dramatically, bagged linear regression does not
budge.

**The rule:** bag models that overfit. If your base learner is already stable, bagging is wasted
compute.

---

## 8. Bagging and the bias-variance decomposition

Test error decomposes as ([00.04 §3](../../00-mathematical-foundations/04-statistics-and-inference/),
[05.01](../../05-model-evaluation/01-bias-variance-and-theory/)):

$$\text{error} = \text{bias}^{2} + \text{variance} + \text{irreducible noise}$$

Bagging acts on exactly one term:

| Term | Effect of bagging |
|---|---|
| Bias² | **unchanged** — the ensemble fits the same average shape |
| Variance | **reduced** toward the floor $\rho\sigma^2$ |
| Noise | untouched (irreducible) |

This is the whole story in one table, and it explains the design choices: use a **low-bias** base
learner (deep, unpruned trees) so the one term bagging cannot fix is already small, and let
averaging crush the variance term it *can* fix. Pruning the base trees would be
counterproductive — it trades away the low bias you want to keep in exchange for reducing variance
that bagging removes for free.

---

## 9. Bagging for classification

Two aggregation choices:

- **Hard voting**: each tree votes for a class, majority wins.
- **Soft voting**: average the predicted probabilities, then take the argmax.

**Soft voting is almost always better** because it uses each tree's *confidence*, not just its
decision — a tree that is 51% sure and one that is 99% sure count equally under hard voting but
differently (and correctly) under soft voting. It also yields usable probability estimates, whereas
hard voting gives only a discrete class.

A classic result (Breiman): for classification, bagging can improve accuracy even when individual
trees are only slightly better than random, provided their errors are somewhat independent — an
early glimpse of the "many weak learners" principle that boosting ([06.03](../03-boosting-theory/))
pushes much further.

---

## 10. Limitations, and the segue to random forests

**What bagging does not do:**

- **It does not reduce bias.** If your base learner underfits, bagging underfits too (§8). Boosting
  is the answer there.
- **It hits the $\rho\sigma^2$ floor.** Row-resampling alone leaves the trees quite correlated — they
  tend to split on the same few strong features near the root — so $\rho$ stays high and the floor
  stays up (§3).
- **It costs interpretability.** One tree is a readable flowchart; 500 averaged trees are not.
- **It is compute- and memory-heavy.** $B$ full models to train, store, and query.

**The segue.** The dominant limitation is the correlation floor, and the fix is a second layer of
randomness. A **random forest** ([06.02](../02-random-forests/)) is bagging plus **feature
subsampling at each split**: restrict each split to a random subset of features, so different trees
are forced to use different features and $\rho$ drops. Lower $\rho$, lower floor, lower variance —
same trees, better ensemble. Everything you learned here carries over; the random forest just adds
the one ingredient that attacks the number this chapter identified as the enemy.

---

## 11. Common misconceptions

**"Bagging reduces bias and variance."**
Only variance. Bias is unchanged — the ensemble fits the same average shape as one tree (§2, §8).

**"More trees can overfit."**
No. Adding trees only drives variance toward its floor $\rho\sigma^2$; it never increases test
error. $B$ is a compute knob, not a regularization knob. (This is *not* true of boosting.)

**"Bagging works for any model."**
Only high-variance ones. On stable models like linear regression it does essentially nothing
(§7).

**"You should prune the base trees."**
The opposite. You want deep, low-bias, high-variance trees, because averaging fixes the variance
and cannot fix bias (§8).

**"OOB error is a rough heuristic."**
It is a nearly unbiased estimate of test error, comparable to cross-validation and free (§6).

**"Bagging and boosting are variations on the same idea."**
They target different terms: bagging reduces variance (parallel, independent trees), boosting
reduces bias (sequential, dependent trees). See [06.03](../03-boosting-theory/).

**"Hard voting and soft voting are equivalent."**
Soft voting uses confidence and is almost always better, and gives probabilities (§9).

---

## Files in this chapter

| File | Contents |
|---|---|
| [`from_scratch.py`](from_scratch.py) | A generic bagging ensemble (any base learner), the bootstrap with OOB tracking, soft/hard voting, and OOB scoring — verified against sklearn, with experiments measuring the $\rho\sigma^2$ variance floor, OOB-vs-test agreement, the trees-vs-linear-regression contrast, and decorrelation lowering the floor |
| [`exercises.md`](exercises.md) | Derivation, implementation, and interview questions |
| [`references.md`](references.md) | Exact sources used |

**Next**: [06.02 — Random Forests](../02-random-forests/) attacks the correlation floor directly.
