# 06.03 — Boosting Theory & AdaBoost

> **Prerequisites**: [06.01](../01-bagging/)–[06.02](../02-random-forests/) (the variance-reduction
> ensembles this chapter contrasts with), [03.08](../../03-supervised-learning/08-decision-trees/)
> (the weak learner), [00.02 §6](../../00-mathematical-foundations/02-calculus-and-optimization/)
> (convexity).
> **You will be able to**: derive AdaBoost from forward stagewise additive modelling, prove the
> training error decays exponentially, explain the exponential-loss view and why it makes AdaBoost
> fragile to noise, and say precisely why boosting *can* overfit while bagging cannot.

---

## Table of contents

1. [A different question entirely](#1-a-different-question-entirely)
2. [Weak learnability — the founding question](#2-weak-learnability--the-founding-question)
3. [AdaBoost, the algorithm](#3-adaboost-the-algorithm)
4. [Worked intuition](#4-worked-intuition)
5. [Why those exact weights? The derivation](#5-why-those-exact-weights-the-derivation)
6. [The training error bound](#6-the-training-error-bound)
7. [AdaBoost is minimizing exponential loss](#7-adaboost-is-minimizing-exponential-loss)
8. [Why AdaBoost resists overfitting — and when it doesn't](#8-why-adaboost-resists-overfitting--and-when-it-doesnt)
9. [The margin explanation](#9-the-margin-explanation)
10. [Sensitivity to noise](#10-sensitivity-to-noise)
11. [Boosting vs bagging — the complete contrast](#11-boosting-vs-bagging--the-complete-contrast)
12. [Beyond AdaBoost](#12-beyond-adaboost)
13. [Common misconceptions](#13-common-misconceptions)

---

## 1. A different question entirely

Bagging and random forests
([06.01](../01-bagging/)–[06.02](../02-random-forests/)) reduce **variance**: average many
*strong, independent, low-bias* trees so their errors cancel. They cannot reduce bias — the
ensemble fits the same average shape as one tree.

Boosting asks the opposite question:

> **Can we combine many *weak*, high-bias models into one strong, low-bias model?**

The answer — remarkably — is yes, and the mechanism is the mirror image of bagging:

| | Bagging / RF | Boosting |
|---|---|---|
| Trees are trained | in **parallel**, independently | **sequentially**, each fixing the last's mistakes |
| Base learner | **strong** (deep tree) | **weak** (a stump) |
| Reduces | variance | **bias** (primarily) |
| Combining | equal-weight average | **weighted** sum |

Where bagging averages away noise, boosting **stacks corrections**: each new model is trained to fix
exactly what the current ensemble gets wrong. It is a fundamentally sequential, adaptive process,
and it produces the models that win on tabular data
([06.05](../05-modern-gbdts/)) — but it inherits a fragility bagging does not have (§8, §10).

---

## 2. Weak learnability — the founding question

Boosting was born from a precise theoretical question (Kearns & Valiant, 1988):

> Is a class that can be learned only **slightly better than chance** (a *weak* learner, error
> $\le \tfrac12 - \gamma$ for some small $\gamma > 0$) also learnable to **arbitrary accuracy** (a
> *strong* learner)?

Schapire (1990) proved the answer is **yes** — weak and strong learnability are equivalent — and the
*constructive* proof was the first boosting algorithm. This is a genuinely surprising result: a
model barely better than a coin flip, applied repeatedly and combined correctly, can be driven to
any accuracy the data supports. AdaBoost (Freund & Schapire, 1997) is the practical, adaptive
realization, and it won its authors the Gödel Prize.

The theoretical framing matters because it sets the design: the base learner should be **weak on
purpose** — a decision *stump* (depth-1 tree) is the canonical choice. A strong base learner would
leave nothing for subsequent rounds to correct, and would overfit
([06.01 §8](../01-bagging/)'s logic, inverted).

---

## 3. AdaBoost, the algorithm

**AdaBoost** (Adaptive Boosting) for binary classification with labels $y_i\in\{-1,+1\}$:

```
initialize weights  w_i = 1/n  for all i
for m = 1..M:
    1. fit weak learner h_m to the data weighted by w
    2. compute its weighted error:
           err_m = sum_i w_i * 1[h_m(x_i) != y_i]  /  sum_i w_i
    3. compute its vote:
           alpha_m = 1/2 * ln( (1 - err_m) / err_m )
    4. reweight:  increase weights on misclassified points
           w_i <- w_i * exp( -alpha_m * y_i * h_m(x_i) )
    5. renormalize w to sum to 1

final:  H(x) = sign( sum_m alpha_m * h_m(x) )
```

Three ideas, and each is derived (not chosen) in §5:

- **The vote $\alpha_m$.** Accurate learners (low $\mathrm{err}_m$) get a large vote; a learner at
  chance ($\mathrm{err}_m = \tfrac12$) gets $\alpha_m = 0$ (ignored); a learner *worse* than chance
  gets a *negative* vote (its predictions are flipped). This is exactly $\tfrac12\ln\frac{1-\mathrm{err}}{\mathrm{err}}$.
- **The reweighting.** $\exp(-\alpha_m y_i h_m(x_i))$ is $>1$ when $h_m$ is *wrong* on point $i$
  ($y_i h_m \ne$ agree) and $<1$ when right. So misclassified points get **heavier**, and the next
  learner is forced to focus on them.
- **The sequential dependence.** Each learner is trained on a distribution shaped by all previous
  ones. This is what makes boosting un-parallelizable — and what makes it powerful.

---

## 4. Worked intuition

Picture a 2-D binary problem a single stump cannot solve.

**Round 1.** All points weigh equally. The stump draws one axis-aligned line, getting most points
right but some wrong. The wrong ones get their weights boosted.

**Round 2.** The second stump *sees the boosted weights*, so it prioritizes the points round 1
missed — it draws a different line, one that fixes those specific errors (while possibly creating new
ones). Its errors get boosted in turn.

**Round 3, 4, …** Each stump attacks the current frontier of mistakes. The final classifier is a
weighted vote: regions handled confidently by many stumps are decided firmly; contested regions are
decided by the weighted balance.

The effect is that the ensemble's **decision boundary becomes arbitrarily complex** — a weighted
combination of many simple lines — even though every component is a single line. That is bias
reduction: the ensemble represents functions no individual stump can. Experiment 1 visualizes the
weight redistribution.

---

## 5. Why those exact weights? The derivation

AdaBoost is not a bag of heuristics — every formula falls out of one principle: **forward stagewise
additive modelling** of an exponential loss.

**Setup.** Build an additive model $F_M(\mathbf{x}) = \sum_{m=1}^{M}\alpha_m h_m(\mathbf{x})$ greedily,
one term at a time, minimizing the **exponential loss**

$$L(y, F) = \exp(-y\,F(\mathbf{x}))$$

At stage $m$, with $F_{m-1}$ fixed, we choose the next $(\alpha_m, h_m)$ to minimize

$$\sum_{i=1}^{n}\exp\!\big(-y_i(F_{m-1}(\mathbf{x}_i) + \alpha h(\mathbf{x}_i))\big)
= \sum_{i=1}^{n}\underbrace{\exp(-y_iF_{m-1}(\mathbf{x}_i))}_{\displaystyle w_i^{(m)}}\ \exp(-\alpha y_i h(\mathbf{x}_i))$$

The weight $w_i^{(m)} = \exp(-y_iF_{m-1}(\mathbf{x}_i))$ **appears automatically** — it is the
current exponential loss on point $i$. This is the AdaBoost weight, derived rather than posited.

**Solving for $h$.** Split the sum by whether $h$ is right ($y_ih = +1$) or wrong ($y_ih = -1$):

$$= e^{-\alpha}\sum_{y_i = h_i} w_i^{(m)} + e^{\alpha}\sum_{y_i\ne h_i} w_i^{(m)}
= e^{-\alpha}\sum_i w_i^{(m)} + (e^{\alpha} - e^{-\alpha})\sum_{y_i\ne h_i}w_i^{(m)}$$

For any fixed $\alpha > 0$, this is minimized by the $h$ with the smallest **weighted error**
$\mathrm{err}_m = \sum_{y_i\ne h_i}w_i^{(m)} / \sum_i w_i^{(m)}$ — which is exactly what "fit $h_m$
to the weighted data" (step 1) does.

**Solving for $\alpha$.** Differentiate w.r.t. $\alpha$ and set to zero:

$$-e^{-\alpha}(1-\mathrm{err}_m) + e^{\alpha}\,\mathrm{err}_m = 0
\;\Longrightarrow\;
e^{2\alpha} = \frac{1-\mathrm{err}_m}{\mathrm{err}_m}
\;\Longrightarrow\;
\boxed{\;\alpha_m = \tfrac12\ln\frac{1-\mathrm{err}_m}{\mathrm{err}_m}\;}$$

The vote formula, derived. And the weight update for the next round is
$w_i^{(m+1)} = \exp(-y_iF_m(\mathbf{x}_i)) = w_i^{(m)}\exp(-\alpha_m y_i h_m(\mathbf{x}_i))$ —
exactly step 4.

> **This is the single most important idea in the chapter.** AdaBoost *is* greedy exponential-loss
> minimization. Every step — the weighted fit, the vote, the reweighting — is a consequence of that
> one objective, not a design choice. Experiment 4 verifies that the AdaBoost weights match the
> exponential-loss gradient numerically. Once you see this, gradient boosting
> ([06.04](../04-gradient-boosting/)) is the obvious generalization: *keep forward stagewise
> additive modelling, but swap exponential loss for any differentiable loss.*

---

## 6. The training error bound

AdaBoost's original claim to fame: the training error decays **exponentially** in the number of
rounds.

**Theorem.** If each weak learner has weighted error $\mathrm{err}_m = \tfrac12 - \gamma_m$ (i.e.
$\gamma_m$ better than chance), the training error of the final classifier satisfies

$$\frac{1}{n}\sum_i \mathbb{1}[H(\mathbf{x}_i)\ne y_i]
\ \le\ \prod_{m=1}^{M}2\sqrt{\mathrm{err}_m(1-\mathrm{err}_m)}
\ =\ \prod_m\sqrt{1 - 4\gamma_m^{2}}
\ \le\ \exp\!\Big(-2\sum_m \gamma_m^{2}\Big)$$

*Proof sketch.* The $0/1$ error is upper-bounded by the exponential loss (which dominates it
pointwise), and the exponential loss factorizes across rounds as
$\prod_m Z_m$ where $Z_m = 2\sqrt{\mathrm{err}_m(1-\mathrm{err}_m)}$ is the normalizer at round $m$.
Each factor is $< 1$ whenever $\gamma_m > 0$. $\blacksquare$

**Read the consequence.** As long as *every* weak learner is even slightly better than chance
($\gamma_m > \gamma > 0$), the training error is driven to **zero exponentially fast** — it needs
only $O(\log n / \gamma^2)$ rounds. This is the constructive proof of §2's weak⇒strong theorem, and
Experiment 2 measures the exponential decay directly.

---

## 7. AdaBoost is minimizing exponential loss

From §5, AdaBoost minimizes $\sum_i \exp(-y_i F(\mathbf{x}_i))$. Comparing this to the losses of
[03.04](../../03-supervised-learning/04-logistic-regression/) and
[03.07](../../03-supervised-learning/07-svm/) is illuminating — the margin $yF$ is on the x-axis:

| Loss | $L(yF)$ | At $yF\to+\infty$ | At $yF\to-\infty$ |
|---|---|---|---|
| $0/1$ | $\mathbb{1}[yF<0]$ | 0 | 1 |
| **Exponential (AdaBoost)** | $e^{-yF}$ | $\to 0$ | $\to\infty$ **exponentially** |
| Logistic | $\ln(1+e^{-yF})$ | $\to 0$ | $\to\infty$ **linearly** |
| Hinge (SVM) | $\max(0, 1-yF)$ | 0 | $\to\infty$ linearly |

All four are **convex upper bounds on the $0/1$ loss** — which is why minimizing any of them is a
tractable surrogate for minimizing error. But the exponential loss punishes misclassified points
($yF < 0$) *exponentially* harder than logistic or hinge, and that single fact explains AdaBoost's
character:

- **It drives training error down aggressively** (§6) — the steep penalty forces the ensemble to fix
  every mistake.
- **It is fragile to label noise and outliers** (§10) — a mislabelled point is *always*
  misclassified, so its weight compounds exponentially round after round until it dominates the
  entire fit.

> **This is why the logistic loss version (LogitBoost) and, later, gradient boosting with robust
> losses displaced AdaBoost.** The exponential loss's aggression is a liability on real, noisy data.
> Gradient boosting ([06.04](../04-gradient-boosting/)) lets you pick a loss with a gentler tail.

Note the connection to [00.03 §9.4](../../00-mathematical-foundations/03-probability/): the logistic
loss corresponds to a proper probability model, and AdaBoost's $F(\mathbf{x})$ is in fact related to
the log-odds — $p(y=1\mid\mathbf{x}) = 1/(1+e^{-2F(\mathbf{x})})$ — which is how you extract
calibrated probabilities from a boosted model.

---

## 8. Why AdaBoost resists overfitting — and when it doesn't

Here is the most counterintuitive empirical fact about AdaBoost, and the honest account of it.

**The surprise.** Naively, adding rounds increases model complexity, so test error should eventually
rise. Yet AdaBoost's test error often keeps *falling* long after training error hits zero — sometimes
for thousands of rounds. This baffled the field in the 1990s.

**The resolution (partial).** The margin theory (§9) explains that even after training error is zero,
AdaBoost keeps increasing the *margins* — the confidence of correct classifications — which improves
generalization. Test error tracks margin distribution, not training error.

**But it absolutely can overfit** — and this is the critical contrast with bagging:

| | Bagging / RF | AdaBoost |
|---|---|---|
| More estimators | **never** increases test error | **can** increase test error |
| Reason | drives variance to a floor | keeps adding capacity that can fit noise |

On **noisy data**, AdaBoost overfits, and it does so badly, because the exponential loss forces it to
memorize mislabelled points (§10). The "resistance to overfitting" holds on clean, separable-ish
data; it fails on noise.

> **The practical consequence**: `n_estimators` is a **regularization** hyperparameter for boosting
> and must be tuned (with early stopping on a validation set) — the exact opposite of a random
> forest, where more trees is always safe ([06.02 §5](../02-random-forests/)). Confusing these two
> is one of the most common and costly mistakes in applied ensembling. Experiment 3 shows AdaBoost's
> test error turning up on noisy data while a forest's plateaus.

---

## 9. The margin explanation

Why does test error keep improving after training error is zero? Schapire et al. (1998) answered with
a **margin** argument, echoing the SVM ([03.07 §1](../../03-supervised-learning/07-svm/)).

Define the (normalized) margin of point $i$ as

$$\text{margin}(\mathbf{x}_i) = \frac{y_i\sum_m\alpha_m h_m(\mathbf{x}_i)}{\sum_m\alpha_m} \in [-1, 1]$$

A positive margin means correctly classified; a *larger* margin means classified by a larger
weighted majority — more confidently. The key empirical finding: **AdaBoost keeps increasing the
*minimum* margin even after every point is correctly classified.** Additional rounds do not change
the *decisions* (training error stays zero) but push the *worst-case* margin up, and generalization
bounds that depend on the minimum margin (not on the number of rounds) then predict continued
improvement.

> ⚠️ **It is specifically the minimum margin — the lower tail — that improves, not the whole
> distribution.** Experiment 5 measures this carefully: as rounds increase past zero training error,
> the minimum margin and the 5th percentile climb, but the *median* margin drifts slightly *down*.
> So the common shorthand "AdaBoost keeps increasing the margins" is too strong; the precise claim
> is that it increases the worst-case margin, which is exactly the quantity the bound rewards. The
> improvement is real but modest and confined to the tail.

This is the same principle as the SVM's max-margin objective — maximize the *smallest* margin —
arrived at from a completely different direction, and it is why AdaBoost is sometimes described as
approximately maximizing the margin.

---

## 10. Sensitivity to noise

The flip side of the exponential loss (§7). Consider a **mislabelled** training point. It is, by
definition, on the wrong side of the true boundary, so a good learner classifies it "wrong" every
round. Its weight is therefore multiplied by $e^{\alpha_m} > 1$ round after round, growing
**exponentially**. Within a few dozen rounds it can carry more weight than hundreds of correct
points combined, and the entire ensemble contorts to fit it.

Symptoms and fixes:

| Symptom | Cause | Fix |
|---|---|---|
| A few points dominate the weights | mislabels / outliers | clean the labels; cap weights |
| Test error rises with rounds | overfitting noise | early stopping; fewer rounds |
| Unstable across reruns | high-weight noisy points | **use a robust loss** — gradient boosting |

The clean statement: **AdaBoost assumes low label noise.** On clean data it is excellent; on noisy
data it is one of the *worse* choices, and this is exactly why gradient boosting with robust losses
(Huber, log-loss) superseded it ([06.04](../04-gradient-boosting/)). Experiment 3 demonstrates the
noise sensitivity directly.

---

## 11. Boosting vs bagging — the complete contrast

The whole of Part 6 so far, in one table:

| | Bagging / Random Forest | AdaBoost / Boosting |
|---|---|---|
| **Goal** | reduce variance | reduce bias |
| **Base learner** | strong (deep tree) | weak (stump) |
| **Training** | parallel, independent | sequential, dependent |
| **Combining** | equal-weight average | weighted vote |
| **Data weighting** | uniform (bootstrap) | adaptive (reweight on errors) |
| **More estimators** | can't overfit | **can** overfit |
| **Noise robustness** | robust | **fragile** (exponential loss) |
| **Parallelizable** | yes | no |
| **Tuning** | easy | needs care (esp. `n_estimators`) |
| **Out of the box** | excellent | good, but tune |

They are not competing versions of one idea — they attack **opposite terms** of the bias-variance
decomposition. The decision is diagnostic: if your model has high variance (overfitting a flexible
learner), bag it; if it has high bias (a weak learner underfitting), boost it. Modern gradient
boosting ([06.04](../04-gradient-boosting/)–[06.05](../05-modern-gbdts/)) even folds in a little
bagging (subsampling) to get some variance reduction on top of the bias reduction.

---

## 12. Beyond AdaBoost

AdaBoost is the historical and conceptual root; the modern lineage generalizes it:

| Method | The generalization |
|---|---|
| **AdaBoost** (1997) | forward stagewise + **exponential** loss, classification |
| **LogitBoost** (2000) | same, but **logistic** loss — more robust to noise |
| **Gradient Boosting** (2001) | forward stagewise + **any differentiable** loss, via functional gradient descent ([06.04](../04-gradient-boosting/)) |
| **XGBoost / LightGBM / CatBoost** | gradient boosting + 2nd-order info, regularization, and systems engineering ([06.05](../05-modern-gbdts/)) |

The through-line is §5: **forward stagewise additive modelling**. AdaBoost fixes the loss to be
exponential; gradient boosting frees it. That one generalization — swap the loss, take a functional
gradient — is the whole of [06.04](../04-gradient-boosting/), and it is why understanding AdaBoost's
derivation deeply is worth more than memorizing its formulas.

---

## 13. Common misconceptions

**"Boosting and bagging are similar ensemble methods."**
Opposite goals (bias vs variance), opposite training (sequential vs parallel), opposite base
learners (weak vs strong). §11.

**"AdaBoost's weights are heuristic."**
They are derived exactly from greedy minimization of exponential loss (§5).

**"More rounds always helps, like a random forest."**
No. AdaBoost can overfit; `n_estimators` is a regularizer that needs tuning (§8).

**"AdaBoost can't overfit — its test error keeps dropping."**
Only on clean data. On noisy data it overfits badly (§8, §10).

**"AdaBoost works on any data."**
It assumes low label noise. The exponential loss makes it fragile to mislabels and outliers (§10).

**"Boosting needs deep trees."**
The opposite — weak learners (stumps) on purpose, so each round has something to correct (§2).

**"AdaBoost and gradient boosting are unrelated."**
Gradient boosting *is* AdaBoost's forward-stagewise framework with the loss generalized (§5, §12).

**"Boosting can be parallelized like a random forest."**
No — each learner depends on all previous ones (§3, §11). Modern libraries parallelize *within* a
tree's split-finding, not across trees.

---

## Files in this chapter

| File | Contents |
|---|---|
| [`from_scratch.py`](from_scratch.py) | AdaBoost (SAMME and SAMME.R for multiclass) on decision stumps, the exponential-loss connection verified numerically, the training-error bound checked, and margin distributions — verified against sklearn, with experiments on weight redistribution, exponential error decay, noise-driven overfitting vs a forest, and the margin-after-zero-training-error phenomenon |
| [`exercises.md`](exercises.md) | Derivation (including the full AdaBoost derivation), implementation, and interview questions |
| [`references.md`](references.md) | Exact sources used |

**Previous**: [06.02 — Random Forests](../02-random-forests/) ·
**Next**: [06.04 — Gradient Boosting](../04-gradient-boosting/) generalizes §5 to any loss.
