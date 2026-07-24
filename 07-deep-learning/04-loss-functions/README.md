# 07.04 — Loss Functions

> **Prerequisites**: [07.03](../03-activations/) (output activations — sigmoid, softmax),
> [07.02](../02-backpropagation/) (how the loss's gradient starts the backward pass),
> [00.04](../../00-mathematical-foundations/04-statistics-and-inference/) (maximum likelihood),
> [05.02](../../05-model-evaluation/02-regression-metrics/)–[05.03](../../05-model-evaluation/03-classification-metrics/)
> (the metrics these losses relate to).
> **You will be able to**: derive each loss as a negative log-likelihood, match a loss to its output
> activation, and explain the single most important loss result in deep learning — why cross-entropy,
> not MSE, is used for classification.

---

## Table of contents

1. [The loss is the training objective](#1-the-loss-is-the-training-objective)
2. [Every loss is a negative log-likelihood](#2-every-loss-is-a-negative-log-likelihood)
3. [Regression losses](#3-regression-losses)
4. [Classification losses](#4-classification-losses)
5. [The softmax + cross-entropy gradient](#5-the-softmax--cross-entropy-gradient)
6. [Why cross-entropy, not MSE, for classification](#6-why-cross-entropy-not-mse-for-classification)
7. [Numerical stability](#7-numerical-stability)
8. [Other losses](#8-other-losses)
9. [Matching loss to task](#9-matching-loss-to-task)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. The loss is the training objective

The loss function $L(\hat{\mathbf{y}}, \mathbf{y})$ measures how wrong a prediction is, and it is
*the* thing training minimizes — backprop computes $\partial L/\partial\hat{\mathbf{y}}$ at the output
and propagates it back ([07.02](../02-backpropagation/)). So the loss **defines what the network learns
to do**: change the loss and you change the task, even with the same architecture.

The loss is related to, but distinct from, the *metric* you report
([05.02 §1](../../05-model-evaluation/02-regression-metrics/)). The metric can be anything (accuracy,
AUC); the loss must be **differentiable** (so gradients exist) and well-behaved for optimization. You
train on the loss and evaluate on the metric, and they are chosen to align — but the loss's job is to
give *useful gradients everywhere*, which, as §6 shows, is a stronger requirement than merely
measuring error.

---

## 2. Every loss is a negative log-likelihood

The losses are not arbitrary — nearly every standard one is the **negative log-likelihood** of the
data under a probabilistic assumption about the output. Assume the target is generated as
$\mathbf{y} \sim p(\mathbf{y}\mid \hat{\mathbf{y}})$ for some distribution, and maximizing the
likelihood is minimizing $-\log p$:

$$
L(\hat{\mathbf{y}}, \mathbf{y}) = -\log p(\mathbf{y}\mid\hat{\mathbf{y}}).
$$

- **Gaussian** output noise → squared error (§3).
- **Laplace** output noise → absolute error (§3).
- **Bernoulli** output → binary cross-entropy (§4).
- **Categorical** output → categorical cross-entropy (§4).

This is the unifying principle: **choosing a loss is choosing a probabilistic model of the output**,
and the matching **output activation** is whatever maps the network's real-valued logits to that
distribution's parameter (identity → Gaussian mean; sigmoid → Bernoulli probability; softmax →
categorical probabilities). Losses and output activations come in *matched pairs* for this reason, and
it explains why some pairings (softmax + cross-entropy) are natural and others (sigmoid + MSE) are
mistakes (§6). Experiment 1 confirms each loss's minimizing constant is the corresponding estimator
(mean for MSE, median for MAE), exactly as the NLL view predicts.

---

## 3. Regression losses

For real-valued targets:

- **Mean squared error (MSE / L2)**: $\frac1n\sum_i (\hat y_i - y_i)^2$. The Gaussian NLL; its
  minimizer is the **conditional mean**. Smooth gradient $2(\hat y - y)$, but **sensitive to
  outliers** (a far point contributes a large squared error and a large gradient). Pair with an
  **identity** output.
- **Mean absolute error (MAE / L1)**: $\frac1n\sum_i |\hat y_i - y_i|$. The Laplace NLL; minimizer is
  the **conditional median**, so it is **robust to outliers**. Its gradient is $\pm 1$ (constant
  magnitude) — robust, but non-smooth at zero and less informative near the optimum.
- **Huber loss**: quadratic within a band $|\hat y - y|\le\delta$, linear outside. A smooth blend —
  MSE's gradient near the fit, MAE's bounded gradient in the tails. The robust default for regression
  ([05.02 §8](../../05-model-evaluation/02-regression-metrics/), [06.04 §5](../../06-ensembles/04-gradient-boosting/)).

The MSE-vs-MAE choice is the mean-vs-median choice ([05.02 §4](../../05-model-evaluation/02-regression-metrics/));
Huber gets both.

---

## 4. Classification losses

For discrete targets, the loss operates on **probabilities** (produced by the output activation):

- **Binary cross-entropy (BCE / log loss)** for two classes, paired with a **sigmoid** output
  $\hat p = \sigma(z)$:

$$
L = -\big[y\log\hat p + (1-y)\log(1-\hat p)\big].
$$

  The Bernoulli NLL. It goes to $0$ for a confident correct prediction and to $+\infty$ for a confident
  wrong one — a savage penalty on overconfident mistakes ([05.03 §8](../../05-model-evaluation/03-classification-metrics/)).

- **Categorical cross-entropy (CCE)** for $K$ classes, paired with a **softmax** output
  $\hat{\mathbf{p}} = \mathrm{softmax}(\mathbf{z})$ and a one-hot target $\mathbf{y}$:

$$
L = -\sum_{k=1}^{K} y_k \log \hat p_k = -\log \hat p_{\text{true class}}.
$$

  The categorical NLL — the negative log-probability the model assigned to the *correct* class. This
  is the standard loss for multiclass classification.

Cross-entropy is not just "a loss for classification" — it is the *right* one, for the gradient reason
in §5–§6.

---

## 5. The softmax + cross-entropy gradient

Softmax and cross-entropy are designed to be used together, and the reason is a small miracle of
cancellation. Take the categorical cross-entropy $L = -\sum_k y_k\log\hat p_k$ with
$\hat{\mathbf{p}} = \mathrm{softmax}(\mathbf{z})$. The gradient of the loss with respect to the
**logits** $\mathbf{z}$ — the thing backprop needs to start — is stunningly simple:

$$
\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{p}} - \mathbf{y}.
$$

Just the **predicted probabilities minus the one-hot target**. All the messy softmax and log
derivatives cancel exactly. (The same holds for sigmoid + BCE: $\partial L/\partial z = \hat p - y$.)
This is why frameworks provide a *combined* `softmax_cross_entropy` / `BCEWithLogits` op that takes
logits directly — it is both numerically stable (§7) and computes this clean gradient in one step.
Experiment 2 verifies $\partial L/\partial\mathbf{z} = \hat{\mathbf{p}} - \mathbf{y}$ against
finite differences to machine precision. The elegance is not cosmetic: this clean gradient is exactly
what gives cross-entropy its good learning behavior (§6).

---

## 6. Why cross-entropy, not MSE, for classification

This is the single most important loss result in deep learning, and it follows directly from §5. You
*could* train a classifier with a sigmoid output and MSE loss — but you should not, and here is the
gradient-level reason.

For a sigmoid output $\hat p = \sigma(z)$, the gradient of the loss w.r.t. the logit $z$ is
$\partial L/\partial z = (\partial L/\partial\hat p)\cdot\sigma'(z)$, and $\sigma'(z) = \hat p(1-\hat p)$
**saturates to 0** when the output is confident ($\hat p\approx 0$ or $1$). Now compare the two losses
when the model is **confidently wrong** ($\hat p\approx 1$ but $y = 0$):

- **MSE**: $\partial L/\partial z = (\hat p - y)\cdot\hat p(1-\hat p) \approx 1\cdot 0 = 0$. The
  gradient **vanishes** — the model is badly wrong but *cannot learn from it*, because the sigmoid has
  saturated. Training stalls.
- **Cross-entropy**: $\partial L/\partial z = \hat p - y \approx 1$. The $\sigma'$ factor **cancels**
  (§5), so the gradient is *large* exactly when the error is large. The model learns fastest from its
  worst mistakes.

Cross-entropy's log shape is engineered so its derivative cancels the output activation's saturating
derivative, leaving a clean $\hat p - y$ that never vanishes on a wrong prediction. MSE has no such
cancellation, so paired with a saturating output it produces vanishing gradients on confident errors.
Experiment 3 measures exactly this: MSE's gradient collapsing toward 0 on confidently-wrong examples
while cross-entropy's stays large, and MSE-trained classification converging far slower. **Use
cross-entropy for classification** — this is why.

---

## 7. Numerical stability

Cross-entropy involves `exp` (softmax) and `log`, both of which overflow/underflow naively:
$e^{z}$ for a large logit overflows to `inf`; $\log(\hat p)$ for a tiny $\hat p$ is `-inf`. Computing
softmax then log separately is a recipe for `NaN`. Two standard fixes:

- **Log-sum-exp / softmax with max-subtraction**: $\mathrm{softmax}(\mathbf{z})_k = e^{z_k - m}/\sum_j e^{z_j - m}$
  with $m = \max_j z_j$. Subtracting the max makes the largest exponent $0$, preventing overflow, and
  leaves the result unchanged ([07.03 §8](../03-activations/)).
- **Combine softmax and cross-entropy analytically**: $-\log\hat p_c = -z_c + \log\sum_j e^{z_j}$,
  computed with log-sum-exp on the logits directly — never forming the probabilities. This is what
  `cross_entropy_with_logits` / `BCEWithLogitsLoss` do, and it is both stable *and* gives the clean
  gradient of §5 in one op.

Experiment 4 shows the naive "softmax then log" producing `inf`/`NaN` on large logits where the stable
combined form gives the correct finite loss. **Always use the framework's `*_with_logits` loss** —
feed it logits, not probabilities.

---

## 8. Other losses

Beyond the NLL staples:

- **Hinge loss** ($\max(0, 1 - y\cdot z)$) — the SVM's margin loss
  ([03.07](../../03-supervised-learning/07-svm/)); cares about the margin, not calibrated
  probabilities.
- **Focal loss** — cross-entropy down-weighted on easy examples,
  $-(1-\hat p_t)^\gamma\log\hat p_t$, for extreme class imbalance (object detection); it focuses
  training on the hard, rare cases ([05.03 §2](../../05-model-evaluation/03-classification-metrics/)).
- **Label smoothing** — replace one-hot targets with $(1-\epsilon)$ on the true class and $\epsilon/K$
  elsewhere; regularizes by discouraging overconfidence, improving calibration
  ([05.06](../../05-model-evaluation/06-calibration/)).
- **KL divergence** — for matching a target *distribution* (knowledge distillation, variational
  methods); cross-entropy is KL plus a constant.
- **Contrastive / triplet losses** — for metric/representation learning: pull similar pairs together,
  push dissimilar apart ([11.xx](../../)).

Each is a tool for a specific structure — imbalance, distributions, embeddings — layered on the NLL
foundation.

---

## 9. Matching loss to task

The task determines the loss, and the loss determines the output activation — the matched pairs of §2:

| Task | Output activation | Loss |
|---|---|---|
| Regression | **identity** | **MSE** (or Huber if outliers) |
| Robust regression | identity | **MAE / Huber** |
| Binary classification | **sigmoid** | **binary cross-entropy** (with logits) |
| Multiclass (single label) | **softmax** | **categorical cross-entropy** (with logits) |
| Multi-label | **sigmoid** (per label) | **binary cross-entropy** per label |
| Imbalanced detection | sigmoid/softmax | **focal loss** |
| Distribution matching | softmax | **KL divergence** |

The rules: identity + MSE for regression; sigmoid + BCE for binary; softmax + cross-entropy for
multiclass — and always the numerically stable *with-logits* version (§7). Get the pairing right and
the gradients are clean (§5); get it wrong (sigmoid + MSE) and training stalls (§6).

---

## 10. Common misconceptions

**"The loss and the metric are the same thing."**
The loss is the differentiable training objective; the metric is what you report and may be
non-differentiable (accuracy, AUC) (§1). They align but are distinct.

**"You can use MSE for classification."**
You can, but paired with a saturating output it vanishes the gradient on confident errors, so training
stalls — cross-entropy's clean $\hat p - y$ gradient does not (§6). Use cross-entropy.

**"Losses are arbitrary design choices."**
Nearly every standard loss is a negative log-likelihood under a noise/output assumption (§2). The loss
encodes a probabilistic model.

**"Apply softmax, then compute cross-entropy on the probabilities."**
That is numerically unstable and error-prone; use the combined `cross_entropy_with_logits` on the
logits directly (§7).

**"The softmax + cross-entropy gradient is complicated."**
It is exactly $\hat{\mathbf{p}} - \mathbf{y}$ — all the softmax/log derivatives cancel (§5).

**"MAE and MSE give the same regression."**
MSE targets the mean, MAE the median; they differ on skewed data and under outliers (§3,
[05.02 §4](../../05-model-evaluation/02-regression-metrics/)).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — MSE, MAE, Huber, binary and categorical cross-entropy (with
  the numerically stable with-logits forms) in NumPy, with gradients verified against PyTorch autograd.
  Five experiments: (1) each loss's minimizing constant is its estimator (MSE→mean, MAE→median); (2)
  the softmax+CE gradient equals $\hat{\mathbf{p}} - \mathbf{y}$ to machine precision; (3) **cross-
  entropy vs MSE for classification** — MSE's gradient vanishing on confident errors while CE's stays
  large, and CE training faster; (4) numerical stability — naive softmax→log giving `NaN` where the
  stable form is correct; (5) focal loss / class weighting focusing on the rare class.
- **[exercises.md](exercises.md)** — derive each loss as an NLL and the softmax+CE gradient, implement
  stable cross-entropy, reproduce every experiment.
- **[references.md](references.md)** — Goodfellow et al. Ch. 6, Bishop Ch. 4-5, the focal-loss and
  label-smoothing papers.

**Next**: [07.05 — Initialization](../05-initialization/) — how to set the initial weights so that
signals and gradients neither vanish nor explode, and training can even begin.
