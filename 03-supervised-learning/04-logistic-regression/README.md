# 03.04 — Logistic Regression

> **Prerequisites**: [03.01](../01-linear-regression/), [03.02](../02-regularized-linear-models/);
> [00.02 §6, §12](../../00-mathematical-foundations/02-calculus-and-optimization/) (convexity,
> Newton); [00.05 §6](../../00-mathematical-foundations/05-information-theory/) (cross-entropy);
> [00.06 §9](../../00-mathematical-foundations/06-numerical-methods/) (stable sigmoid and BCE).
> **You will be able to**: derive the loss from maximum likelihood, prove it is convex, implement
> IRLS, explain why perfectly separable data breaks the model, and interpret a coefficient as an
> odds ratio without hand-waving.

---

## Table of contents

1. [Why not just use linear regression](#1-why-not-just-use-linear-regression)
2. [The model](#2-the-model)
3. [Interpreting the coefficients](#3-interpreting-the-coefficients)
4. [The loss, from maximum likelihood](#4-the-loss-from-maximum-likelihood)
5. [Gradient and Hessian](#5-gradient-and-hessian)
6. [Convexity](#6-convexity)
7. [Why there is no closed form](#7-why-there-is-no-closed-form)
8. [IRLS — Newton's method in disguise](#8-irls--newtons-method-in-disguise)
9. [Perfect separation](#9-perfect-separation)
10. [Regularization](#10-regularization)
11. [Multiclass](#11-multiclass)
12. [The decision threshold](#12-the-decision-threshold)
13. [Calibration](#13-calibration)
14. [Complexity, and when to use it](#14-complexity-and-when-to-use-it)
15. [Common misconceptions](#15-common-misconceptions)

---

## 1. Why not just use linear regression

Encode the classes as 0/1 and fit OLS. Three things break:

**1. Predictions leave $[0,1]$.** A linear function is unbounded, so you get "probabilities" of
$-0.3$ and $1.4$. There is no principled way to interpret them.

**2. The wrong noise model.** OLS assumes $\varepsilon\sim\mathcal{N}(0,\sigma^{2})$ with constant
variance. For a binary outcome, $\mathrm{Var}(y\mid\mathbf{x}) = p(1-p)$ — it depends on
$\mathbf{x}$ and is maximal at $p=0.5$. Squared error is the negative log-likelihood of the wrong
distribution ([00.03 §9.4](../../00-mathematical-foundations/03-probability/)).

**3. Outliers in $\mathbf{x}$ rotate the boundary.** Because OLS penalizes squared distance from
0/1 targets, a correctly-classified point far from the boundary is *penalized for being too
correct* — its fitted value overshoots 1 and contributes a large residual. Adding such points
drags the decision boundary away from a perfectly good position.

Point 3 is the one that surprises people, and Experiment 1 measures it: adding a cluster of
unambiguous, correctly-labelled positives makes an OLS classifier *worse*, while logistic
regression is unmoved.

The fix is not a patch — it is to model the **probability** directly, and to choose a link that
maps $\mathbb{R}$ into $[0,1]$.

---

## 2. The model

$$p(y=1\mid\mathbf{x}) = \sigma(z) = \frac{1}{1+e^{-z}},
\qquad z = \mathbf{w}^{\top}\mathbf{x}+b$$

The **logistic sigmoid** squashes $\mathbb{R}\to(0,1)$, is monotone, and has the convenient
derivative

$$\sigma'(z) = \sigma(z)\big(1-\sigma(z)\big)$$

which is why every gradient below is so clean.

### 2.1 The log-odds view — the right way to think about it

Invert the sigmoid:

$$z = \log\frac{p}{1-p} = \mathrm{logit}(p)$$

$$\boxed{\;\log\frac{p(y=1\mid\mathbf{x})}{p(y=0\mid\mathbf{x})} = \mathbf{w}^{\top}\mathbf{x}+b\;}$$

**Logistic regression is a linear model for the log-odds.** That is the honest one-sentence
description, and it explains everything else:

- The model is linear — in the log-odds, not in the probability.
- The decision boundary $p=0.5$ is $z=0$, a **hyperplane** — logistic regression is a *linear
  classifier*.
- The probability is a nonlinear (sigmoid) function of a linear score.

This also places it in the GLM family: a linear predictor $\mathbf{w}^{\top}\mathbf{x}$, a
Bernoulli response, and the **logit link**. Swap the link for a probit, or the response for a
Poisson, and you have the rest of the GLM catalogue.

---

## 3. Interpreting the coefficients

From §2.1, increasing $x_j$ by one unit adds $w_j$ to the log-odds — so it **multiplies the odds
by $e^{w_j}$**.

$$\text{odds ratio} = e^{w_j}$$

| $w_j$ | $e^{w_j}$ | Reading |
|---|---|---|
| $0$ | 1.00 | no effect |
| $0.1$ | 1.11 | +11% odds per unit |
| $0.69$ | 2.00 | doubles the odds |
| $-0.69$ | 0.50 | halves the odds |
| $2.0$ | 7.39 | 7.4× the odds |

**Two cautions that matter in practice:**

⚠️ **Odds ratio ≠ risk ratio.** Doubling the *odds* is not doubling the *probability*. From
$p=0.1$ (odds 1:9), doubling the odds to 2:9 gives $p=0.18$ — a 1.8× risk ratio. From $p=0.4$
(odds 2:3), doubling to 4:3 gives $p=0.57$ — only 1.4×. The two coincide only when $p$ is small.
Medical papers conflate them constantly, and it systematically overstates effects.

⚠️ **"Holding the others constant"** carries the same caveats as in OLS
([03.01 §12](../01-linear-regression/)): with correlated features the individual coefficients are
unstable and the phrase describes a comparison that may not exist in your data.

---

## 4. The loss, from maximum likelihood

Each observation is Bernoulli with parameter $p_i = \sigma(\mathbf{w}^{\top}\mathbf{x}_i)$:

$$p(y_i\mid\mathbf{x}_i) = p_i^{y_i}(1-p_i)^{1-y_i}$$

(The exponent trick just selects the right factor: $y_i=1$ gives $p_i$, $y_i=0$ gives $1-p_i$.)

Log-likelihood over an i.i.d. sample:

$$\ell(\mathbf{w}) = \sum_{i=1}^{n}\Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big]$$

Negate to get a loss:

$$\boxed{\;J(\mathbf{w}) = -\sum_{i=1}^{n}\Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big]\;}$$

**This is binary cross-entropy** ([00.05 §6](../../00-mathematical-foundations/05-information-theory/)),
and it is also the negative log-likelihood, and it is also the KL divergence to the empirical
distribution up to a constant. Three derivations, one loss — the same convergence that made
squared error trustworthy for regression.

> ⚠️ **Never implement this formula literally.** Once $\sigma(z)$ saturates to exactly 0 or 1 —
> around $|z| = 37$ in float64, $|z| = 17$ in float32 — you get $\log 0 = -\infty$. Fold the
> sigmoid into the loss:
>
> $$J_i = \max(z_i, 0) - z_iy_i + \log\big(1+e^{-|z_i|}\big)$$
>
> which is bounded at every $z$. This is exactly what `BCEWithLogitsLoss` computes and why it
> exists separately from `Sigmoid` + `BCELoss`. See
> [00.06 §9](../../00-mathematical-foundations/06-numerical-methods/).

---

## 5. Gradient and Hessian

Using $\sigma' = \sigma(1-\sigma)$, the algebra collapses beautifully:

$$\frac{\partial J}{\partial w_j} = \sum_{i=1}^{n}(p_i - y_i)x_{ij}
\qquad\Longrightarrow\qquad
\boxed{\;\nabla J = \mathbf{X}^{\top}(\mathbf{p}-\mathbf{y})\;}$$

**Compare with linear regression**, whose gradient is $\mathbf{X}^{\top}(\hat{\mathbf{y}}-\mathbf{y})$.
*Identical in form* — features times residuals. This is not a coincidence: it is a general
property of GLMs with their canonical link, and it is why the same optimization code works for
both.

The Hessian:

$$\boxed{\;\mathbf{H} = \mathbf{X}^{\top}\mathbf{S}\mathbf{X},
\qquad \mathbf{S} = \mathrm{diag}\big(p_i(1-p_i)\big)\;}$$

$\mathbf{S}$ weights each observation by $p_i(1-p_i)$ — maximal at $p_i = 0.5$ (the uncertain
points, near the boundary) and near zero for confident points. **The model learns almost entirely
from the examples it is unsure about**, which is the same principle that support vectors formalize
([03.07](../07-svm/)) and that hard-example mining exploits.

---

## 6. Convexity

$$\mathbf{v}^{\top}\mathbf{H}\mathbf{v} = \mathbf{v}^{\top}\mathbf{X}^{\top}\mathbf{S}\mathbf{X}\mathbf{v}
= \Vert \mathbf{S}^{1/2}\mathbf{X}\mathbf{v}\Vert _2^{2} \ge 0$$

using $p_i(1-p_i)>0$, so $\mathbf{S}^{1/2}$ is real. Hence $\mathbf{H}\succeq 0$ and $J$ is
**convex** ([00.02 §6.2](../../00-mathematical-foundations/02-calculus-and-optimization/)).

Therefore:

- Every local minimum is global ([00.02 §6.3](../../00-mathematical-foundations/02-calculus-and-optimization/)).
- $\nabla J = \mathbf{0}$ is a certificate of optimality.
- Any convergent descent method finds *the* answer — no initialization sensitivity, no restarts.

**This is the property neural networks give up**, and it is worth appreciating what it is worth: a
logistic regression fitted twice on the same data gives bitwise the same answer. A neural network
does not.

Note $J$ is convex but **not strictly convex** in general — if $\mathbf{X}$ is rank-deficient, or
the data is separable (§9), the minimum is not unique. Adding an $\ell_2$ penalty makes it
strictly convex and restores uniqueness, exactly as in ridge
([03.02 §2.1](../02-regularized-linear-models/)).

---

## 7. Why there is no closed form

Setting $\nabla J = \mathbf{X}^{\top}(\mathbf{p}-\mathbf{y}) = \mathbf{0}$ gives

$$\mathbf{X}^{\top}\sigma(\mathbf{X}\mathbf{w}) = \mathbf{X}^{\top}\mathbf{y}$$

The unknown $\mathbf{w}$ is **inside a nonlinear function**. In linear regression the analogous
equation is linear in $\mathbf{w}$ and inverts; here it does not. These are *transcendental*
equations with no algebraic solution.

So we iterate. And because the problem is convex with a cheap, always-PSD Hessian, we can afford
second-order methods — which is exactly what §8 does.

---

## 8. IRLS — Newton's method in disguise

Apply Newton's method ([00.02 §12.1](../../00-mathematical-foundations/02-calculus-and-optimization/)):

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \mathbf{H}^{-1}\nabla J
= \mathbf{w}^{(t)} + (\mathbf{X}^{\top}\mathbf{S}\mathbf{X})^{-1}\mathbf{X}^{\top}(\mathbf{y}-\mathbf{p})$$

Now rewrite it. Define the **working response**

$$\mathbf{z} = \mathbf{X}\mathbf{w}^{(t)} + \mathbf{S}^{-1}(\mathbf{y}-\mathbf{p})$$

Then a line of algebra gives

$$\boxed{\;\mathbf{w}^{(t+1)} = (\mathbf{X}^{\top}\mathbf{S}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{S}\mathbf{z}\;}$$

which is **exactly the weighted least squares solution** for regressing $\mathbf{z}$ on
$\mathbf{X}$ with weights $\mathbf{S}$.

> **So logistic regression is a sequence of weighted linear regressions.** That is
> *Iteratively Reweighted Least Squares*, and it is the standard algorithm for every GLM. Each
> iteration reweights toward the uncertain points ($p\approx0.5$, weight $0.25$) and away from the
> confident ones (weight $\approx 0$), then refits.

**Convergence.** Newton converges *quadratically* near the optimum — the number of correct digits
doubles each step — so IRLS typically needs **5-8 iterations** where gradient descent needs
thousands. Experiment 2 measures exactly this.

**Cost.** $O(nd^{2}+d^{3})$ per iteration, because you solve a $d\times d$ system. Fine for
$d$ in the thousands; hopeless beyond. That is why:

| $d$ | Solver | Why |
|---|---|---|
| small (< ~1,000) | **IRLS / Newton** | few iterations, no learning rate |
| medium | **L-BFGS** | $O(md)$ memory, sklearn's default |
| large, sparse | **SAGA / SGD** | never touches a $d\times d$ matrix |

`sklearn.linear_model.LogisticRegression` defaults to `lbfgs` for exactly this reason
([00.02 §12.2](../../00-mathematical-foundations/02-calculus-and-optimization/)).

---

## 9. Perfect separation

Here is the failure mode you should know before you meet it.

If a hyperplane separates the classes perfectly, then scaling $\mathbf{w}$ by $c>1$ makes every
$p_i$ more confident, so the likelihood **strictly increases** — and it keeps increasing forever:

$$\Vert \mathbf{w}\Vert \to\infty, \qquad J\to 0, \qquad \text{no finite minimum exists}$$

The MLE does not exist. What you see in practice:

- Coefficients in the hundreds or thousands
- Enormous standard errors (often larger than the coefficients)
- Convergence warnings, or an iteration limit hit
- Predicted probabilities of exactly 0 and 1

**When does it happen?** More often than you would think:
- Small $n$ with many features — with $d > n$ some hyperplane always separates
- A feature that is a perfect proxy for the label (**usually a data leak**,
  [02.06](../../02-data/06-data-leakage/))
- A rare category that appears with only one class
- After one-hot encoding a high-cardinality feature

**Fixes**, in order of preference:

1. **$\ell_2$ regularization.** The penalty grows as $\Vert \mathbf{w}\Vert ^{2}$ while the
   likelihood gain saturates, so a finite minimum always exists. This is the standard answer, and
   it is why `sklearn` regularizes **by default** (`C=1.0`) — a decision that surprises people
   coming from statsmodels, but which prevents exactly this.
2. **Investigate the separating feature.** Perfect separation on real data is very often a leak.
3. **Firth's penalized likelihood** — a principled bias-reduction method used in epidemiology when
   you need coefficients you can interpret.

Experiment 3 shows coefficients diverging without regularization and stabilizing with it.

---

## 10. Regularization

Identical to [03.02](../02-regularized-linear-models/), applied to the logistic loss:

$$J(\mathbf{w}) = -\ell(\mathbf{w}) + \lambda\Omega(\mathbf{w})$$

| Penalty | Effect | sklearn |
|---|---|---|
| $\ell_2$ | shrinks, guarantees a unique finite solution | `penalty="l2"` (default) |
| $\ell_1$ | sparse; feature selection | `penalty="l1"`, needs `liblinear`/`saga` |
| elastic net | both | `penalty="elasticnet"`, `saga` |

⚠️ **sklearn parameterizes by $C = 1/\lambda$**, so **smaller $C$ = more regularization**. This is
backwards from every other library and is a persistent source of confusion. And the default is
$C=1.0$, i.e. regularization is **on** unless you set `penalty=None`. If your sklearn coefficients
disagree with statsmodels, this is why.

Standardize features first, for the same reason as
[03.02 §10](../02-regularized-linear-models/) — the penalty is not scale-invariant.

---

## 11. Multiclass

### 11.1 Softmax (multinomial) regression

Generalize the sigmoid to $K$ classes:

$$p(y=k\mid\mathbf{x}) = \frac{e^{\mathbf{w}_k^{\top}\mathbf{x}}}{\sum_{j=1}^{K}e^{\mathbf{w}_j^{\top}\mathbf{x}}}$$

Loss is categorical cross-entropy, and the gradient has the same form again:

$$\nabla_{\mathbf{w}_k}J = \mathbf{X}^{\top}(\mathbf{p}_k - \mathbf{y}_k)$$

**Note the model is over-parameterized**: adding a constant vector to every $\mathbf{w}_k$ leaves
all probabilities unchanged (the softmax is shift-invariant,
[00.06 §8](../../00-mathematical-foundations/06-numerical-methods/)). So the solution is not
unique without regularization, and one class's weights can be fixed to zero without loss of
generality. With $K=2$ that reduction gives back exactly binary logistic regression.

### 11.2 One-vs-rest

Train $K$ independent binary classifiers, take the argmax. Simpler and parallelizable, but the $K$
scores are **not calibrated against each other** — each was trained on a different, usually
imbalanced, problem — so the probabilities need renormalizing and are less trustworthy.

**Prefer multinomial** when you want probabilities; OvR is acceptable when you only need the
argmax.

---

## 12. The decision threshold

The model outputs a probability. Turning it into a decision requires a threshold, and **0.5 is a
default, not a law.**

The optimal threshold depends on your costs. If a false negative costs $C_{FN}$ and a false
positive costs $C_{FP}$, the expected-cost-minimizing rule is

$$\text{predict } 1 \iff p > \frac{C_{FP}}{C_{FP}+C_{FN}}$$

Fraud detection ($C_{FN} \gg C_{FP}$) wants a low threshold; a costly intervention wants a high
one.

> **This is why you should not "fix" class imbalance by resampling if all you want is a better
> threshold.** Moving the threshold is free, reversible, and does not distort your probabilities.
> Resampling changes the model's implied base rate and therefore its calibration. See
> [02.05](../../02-data/05-class-imbalance/) and
> [05.03](../../05-model-evaluation/03-classification-metrics/).

---

## 13. Calibration

Logistic regression is fitted by minimizing a **proper scoring rule**
([00.05 §6.2](../../00-mathematical-foundations/05-information-theory/)) — a loss uniquely
minimized by reporting your true beliefs. A direct consequence: on the training distribution,
logistic regression is **calibrated essentially by construction**.

Concretely, the gradient at the optimum is $\mathbf{X}^{\top}(\mathbf{p}-\mathbf{y}) = \mathbf{0}$.
Taking the intercept column (all ones) gives

$$\sum_i p_i = \sum_i y_i$$

**The predicted probabilities sum to the observed number of positives — exactly.** No other common
classifier gives you that for free. Experiment 4 verifies it and contrasts with an SVM and a
random forest, whose scores are not probabilities at all.

Caveats: this holds *in-sample*, on the training distribution, and heavy regularization degrades
it (shrinking coefficients pulls probabilities toward the base rate). Under distribution shift, all
bets are off. See [05.06](../../05-model-evaluation/06-calibration/).

---

## 14. Complexity, and when to use it

| Operation | Cost |
|---|---|
| Fit (IRLS) | $O(k(nd^{2}+d^{3}))$, $k\approx 5$-$10$ |
| Fit (L-BFGS) | $O(knd)$, $k$ in the hundreds |
| Fit (SGD) | $O(\text{epochs}\cdot nd)$ |
| Predict | $O(d)$ |

**Use it when:**
- You need **calibrated probabilities** (§13)
- You need **interpretable** effects — odds ratios with confidence intervals
- The problem is regulated (credit, clinical, insurance) and you must explain decisions
- Latency matters — $O(d)$ inference, a dot product and an exponential
- **As a baseline.** Always. A model that cannot beat regularized logistic regression on tabular
  data is telling you something.

**Don't when:**
- The boundary is genuinely nonlinear and you can't engineer the features (use trees/boosting)
- Interactions matter and there are too many to specify
- The data is images, audio, or text at scale (though TF-IDF + logistic regression remains a
  shockingly strong text baseline)

---

## 15. Common misconceptions

**"Logistic regression is a regression algorithm."**
It is a classification algorithm. The name refers to fitting a *regression on the log-odds* (§2.1).

**"It's linear, so it can only draw straight lines."**
The **decision boundary** is linear in the feature space you give it. Add polynomial or
interaction features and it draws whatever that basis can express
([03.03](../03-basis-expansion/)).

**"Coefficients give the change in probability."**
They give the change in **log-odds**. The probability change depends on where you start (§3).

**"An odds ratio of 2 means the probability doubles."**
Only when $p$ is small (§3).

**"There's a closed-form solution like OLS."**
There is not — the score equation is transcendental (§7).

**"My coefficients are 500, the model must be great."**
Perfect separation. The MLE does not exist and your fit is meaningless (§9).

**"sklearn's LogisticRegression is unregularized by default."**
It has $C=1.0$, i.e. $\ell_2$ regularization on. Use `penalty=None` for the true MLE (§10).

**"Bigger C means more regularization."**
$C = 1/\lambda$. Smaller $C$ means more (§10).

**"The threshold is 0.5."**
0.5 is the default; the optimal threshold depends on your costs (§12).

**"Any classifier's output is a probability."**
Only if it was trained with a proper scoring rule. SVM decision values and unadjusted tree votes
are not probabilities (§13).

---

## Files in this chapter

| File | Contents |
|---|---|
| [`from_scratch.py`](from_scratch.py) | Binary logistic regression with four solvers (gradient descent, Newton/IRLS, L-BFGS, SGD), $\ell_1$/$\ell_2$ regularization, softmax regression, odds ratios with confidence intervals, and a numerically stable loss. Verified against sklearn and statsmodels |
| [`exercises.md`](exercises.md) | Derivation, implementation, and interview questions |
| [`references.md`](references.md) | Exact sections used |

**Previous**: [03.03 — Basis Expansion](../03-basis-expansion/) ·
**Next**: [03.05 — Generative Classifiers](../05-generative-classifiers/)
