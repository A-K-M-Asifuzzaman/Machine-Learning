# 07.06 — Optimizers

> **Prerequisites**: [07.02](../02-backpropagation/) (the gradients being used),
> [00.02](../../00-mathematical-foundations/02-calculus-and-optimization/) (gradient descent, momentum,
> Adam's bias correction — derived there), [05.05](../../05-model-evaluation/05-hyperparameter-optimization/)
> (tuning the learning rate).
> **You will be able to**: implement SGD, momentum, RMSProp, and Adam, explain what each fixes,
> choose an optimizer and learning-rate schedule, and understand why SGD's noise is a feature.

---

## Table of contents

1. [From gradient to update](#1-from-gradient-to-update)
2. [Stochastic gradient descent](#2-stochastic-gradient-descent)
3. [The learning rate is everything](#3-the-learning-rate-is-everything)
4. [Momentum](#4-momentum)
5. [Adaptive learning rates: AdaGrad and RMSProp](#5-adaptive-learning-rates-adagrad-and-rmsprop)
6. [Adam](#6-adam)
7. [Learning-rate schedules and warmup](#7-learning-rate-schedules-and-warmup)
8. [Why not second-order methods](#8-why-not-second-order-methods)
9. [Choosing an optimizer](#9-choosing-an-optimizer)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. From gradient to update

Backprop ([07.02](../02-backpropagation/)) gives the gradient $\nabla_{\boldsymbol\theta}L$; the
**optimizer** turns it into a weight update. The simplest is **gradient descent**: step downhill,

$$
\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\,\nabla_{\boldsymbol\theta}L,
$$

with **learning rate** $\eta$. Backprop computes the *direction*; the optimizer decides *how far and how
cleverly* to move. That "cleverly" is the whole chapter: plain gradient descent is slow and fragile on
the ill-conditioned, high-dimensional, stochastic loss landscapes of deep learning, and momentum,
adaptive rates, and schedules are the accumulated fixes that make training fast and robust. Backprop +
optimizer = training; this chapter is the second half.

---

## 2. Stochastic gradient descent

Computing the gradient over the **entire** training set (batch gradient descent) is exact but
expensive — one update per full pass. **Stochastic gradient descent (SGD)** instead estimates the
gradient from a small **mini-batch** (32–512 examples), taking many noisy updates per epoch:

$$
\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\,\nabla_{\boldsymbol\theta}L_{\text{batch}}.
$$

This is faster (an update per mini-batch, not per epoch) and scales to huge datasets. Crucially, the
mini-batch gradient is a **noisy** estimate of the true gradient — and that noise is a **feature**, not
a bug:

- it helps the optimizer **escape saddle points** and sharp/narrow minima (which are the real obstacle
  in high dimensions, §8), and
- it acts as an implicit regularizer, biasing SGD toward **flatter minima** that generalize better.

Experiment 5 shows SGD's noise escaping a saddle point where full-batch gradient descent stalls. Pure
batch GD is rarely used for deep nets; mini-batch SGD (and its accelerated variants) is the norm.

---

## 3. The learning rate is everything

The learning rate $\eta$ is **the single most important hyperparameter** in deep learning. It sets the
step size, and its effect is dramatic:

- **Too small**: training is correct but agonizingly slow — the loss creeps down over far more steps
  than necessary.
- **Too large**: steps overshoot the minimum; the loss oscillates, plateaus high, or **diverges to
  `NaN`**.
- **Just right**: fast, stable descent.

The good range is often narrow and problem-dependent, which is why finding it is the first thing to
tune ([05.05](../../05-model-evaluation/05-hyperparameter-optimization/)) — a **learning-rate range
test** (sweep $\eta$ and watch the loss) is standard practice. Experiment 1 shows the same network
stalling, diverging, and converging under three learning rates. Everything else in this chapter — the
adaptive methods and schedules — is partly about making training *less* sensitive to this one fragile
number.

---

## 4. Momentum

Plain SGD struggles in **ravines** — regions where the loss curves much more steeply in some directions
than others (ill-conditioning). It oscillates across the steep walls while creeping slowly along the
gentle valley floor. **Momentum** fixes this by accumulating a **velocity** — an exponentially-decaying
running average of past gradients — and stepping with the velocity instead of the raw gradient:

$$
\mathbf{v} \leftarrow \beta\,\mathbf{v} + \nabla_{\boldsymbol\theta}L, \qquad \boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\,\mathbf{v},
$$

with $\beta\approx 0.9$. In directions where the gradient is **consistent**, the velocity *accumulates*
and the optimizer accelerates (like a ball rolling downhill gaining speed); in directions where the
gradient **oscillates**, the opposing contributions *cancel* in the average, damping the zig-zag. The
result is faster, smoother convergence along the valley. Experiment 2 shows momentum reaching the
minimum of an ill-conditioned quadratic in far fewer steps than plain SGD. **Nesterov momentum** is a
refinement that evaluates the gradient at the *look-ahead* position $\boldsymbol\theta - \eta\beta\mathbf{v}$,
giving a slightly better correction. Momentum is nearly always worth using.

---

## 5. Adaptive learning rates: AdaGrad and RMSProp

A single global $\eta$ is a blunt tool — different parameters may need different step sizes (frequent
features want small steps, rare features large ones). **Adaptive** methods give each parameter its
*own* effective learning rate, scaled by its gradient history:

- **AdaGrad** accumulates the *sum* of squared gradients per parameter and divides the step by its
  square root: $\eta / \sqrt{G_t + \epsilon}$ where $G_t = \sum_{\tau\le t} g_\tau^2$. Parameters with
  large past gradients get smaller steps. Great for **sparse** features, but its fatal flaw is that
  $G_t$ only *grows*, so the effective learning rate **monotonically decays to 0** and learning stops
  prematurely (Experiment 6).

- **RMSProp** fixes this by using an *exponentially-weighted moving average* of squared gradients
  instead of a cumulative sum: $v_t = \gamma v_{t-1} + (1-\gamma)g_t^2$, step $= \eta/\sqrt{v_t+\epsilon}$.
  Because it *forgets* old gradients, the denominator does not grow without bound, so the learning rate
  stays alive. RMSProp is AdaGrad that keeps working.

Experiment 6 shows AdaGrad's effective learning rate decaying toward 0 while RMSProp's stabilizes — the
one change that made adaptive methods practical.

---

## 6. Adam

**Adam** (Adaptive Moment Estimation, 2015) combines the two big ideas — **momentum** (first moment) and
**RMSProp** (second moment) — plus a bias correction, and is the **default optimizer** for most deep
learning. It maintains running averages of the gradient and its square:

$$
\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t, \qquad \mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2,
$$

then **bias-corrects** them ($\hat{\mathbf{m}}_t = \mathbf{m}_t/(1-\beta_1^t)$,
$\hat{\mathbf{v}}_t = \mathbf{v}_t/(1-\beta_2^t)$) and updates:

$$
\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}.
$$

The momentum term $\hat{\mathbf{m}}$ gives smooth, accelerated descent; the $1/\sqrt{\hat{\mathbf{v}}}$
term gives per-parameter adaptive scaling. **The bias correction is essential**: $\mathbf{m}$ and
$\mathbf{v}$ start at 0, so early in training they are biased *toward zero*, and dividing by
$(1-\beta_1^t)$ — which is tiny early on — *inflates* the early steps to compensate (the effect derived
and measured in [00.02](../../00-mathematical-foundations/02-calculus-and-optimization/), and reproduced
in Experiment 4). Default hyperparameters $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ work
remarkably often — a big reason for Adam's popularity.

**AdamW** decouples weight decay from the adaptive update (applying it directly to the weights rather
than through the gradient), which fixes a subtle interaction bug and generalizes better; **AdamW is the
modern default for Transformers**. Experiment 3 shows Adam converging faster and more robustly than
plain SGD on an ill-conditioned problem.

---

## 7. Learning-rate schedules and warmup

A fixed learning rate is rarely optimal for the whole of training — you want large steps early (fast
progress) and small steps late (fine convergence). **Schedules** vary $\eta$ over training:

- **Step decay** — drop $\eta$ by a factor (e.g. 10×) at set epochs.
- **Cosine annealing** — smoothly decay $\eta$ following a cosine curve to near 0; the common modern
  choice, often with **warm restarts**.
- **Warmup** — *increase* $\eta$ linearly from ~0 over the first few hundred/thousand steps, then
  decay. Warmup is essential for **Transformers**: early on, the adaptive second-moment estimate
  $\hat{\mathbf{v}}$ is unreliable (few samples), so large steps are dangerous; warmup lets it stabilize
  first.
- **One-cycle** — ramp up then down within a single run, enabling large maximum rates (super-convergence).

The schedule interacts with the optimizer (Adam needs less aggressive scheduling than SGD, but still
benefits) and is often as important as the optimizer choice. Cosine decay with a short warmup is a
strong default for large models.

---

## 8. Why not second-order methods

Newton's method uses the **Hessian** (curvature) to take better-scaled steps —
$\boldsymbol\theta \leftarrow \boldsymbol\theta - \mathbf{H}^{-1}\nabla L$ — and converges in far fewer
iterations on well-behaved problems. So why does deep learning use first-order (gradient-only) methods?

- **The Hessian is enormous.** For $P$ parameters it is $P\times P$; for a billion-parameter model that
  is $10^{18}$ entries — impossible to form, let alone invert.
- **Quasi-Newton methods (L-BFGS)** approximate the Hessian from gradients and work well for small,
  full-batch problems ([00.06](../../00-mathematical-foundations/06-numerical-methods/)) — but they
  clash with *stochastic* mini-batch gradients (the noisy curvature estimates are unstable).
- **Adaptive methods are cheap second-order-ish approximations.** Adam's $1/\sqrt{\hat{\mathbf{v}}}$ is
  a *diagonal* approximation of curvature — capturing much of the benefit at $O(P)$ cost instead of
  $O(P^2)$.

So the field settled on first-order methods with adaptive, diagonal curvature scaling: the sweet spot
of cost and benefit at scale. The high-dimensional obstacle, incidentally, is not local minima but
**saddle points** (far more numerous in high dimensions), which SGD's noise (§2) helps escape.

---

## 9. Choosing an optimizer

| Situation | Optimizer |
|---|---|
| Default, most problems | **Adam / AdamW** |
| Transformers / large models | **AdamW + warmup + cosine decay** |
| CNNs / vision (chasing best generalization) | **SGD + momentum** (often generalizes slightly better) |
| Sparse features / NLP embeddings | **Adam / AdaGrad** |
| Small, full-batch, smooth problem | **L-BFGS** |

Practical guidance: **start with Adam/AdamW** — it is robust to the learning rate and usually trains
fast with default $\beta$'s. Tune the **learning rate** first (it matters more than the optimizer
choice), add a **schedule** (cosine + warmup for large models), and consider **SGD + momentum** if you
are optimizing the last bit of generalization on a vision task (a well-documented case where SGD's
implicit regularization edges out Adam). The optimizer is important, but the learning rate and schedule
are where most of the tuning payoff is.

---

## 10. Common misconceptions

**"The optimizer is where learning happens."**
Backprop computes the gradients; the optimizer turns them into updates (§1). Both are needed.

**"Batch gradient descent is best because it uses the exact gradient."**
Mini-batch SGD is faster *and* its noise helps escape saddle points and find flatter minima (§2). Exact
gradients are not the goal.

**"Adam is always better than SGD."**
Adam is the robust default, but SGD + momentum often *generalizes* slightly better on vision tasks (§9).
"Faster training loss" ≠ "better test accuracy."

**"The bias correction in Adam is a minor detail."**
Without it, the first-moment/second-moment estimates are biased toward zero early on, and the early
updates are wrong — the correction inflates them appropriately (§6,
[00.02](../../00-mathematical-foundations/02-calculus-and-optimization/)).

**"AdaGrad's adaptive rate is strictly good."**
Its cumulative sum makes the learning rate decay to 0 and stop learning; RMSProp/Adam fix this by
forgetting old gradients (§5).

**"A fixed learning rate is fine."**
Schedules (warmup + decay) materially improve training, and warmup is essential for Transformers (§7).

**"Second-order methods would be better if we could afford them."**
The Hessian is intractable at scale, and quasi-Newton clashes with mini-batch noise; adaptive
first-order methods capture much of the benefit cheaply (§8).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — SGD, momentum, Nesterov, AdaGrad, RMSProp, and Adam in
  NumPy, with the Adam update verified step-for-step against PyTorch. Six experiments: (1) the learning
  rate — stall / diverge / converge; (2) momentum accelerating on an ill-conditioned ravine; (3) Adam
  vs SGD convergence; (4) Adam's bias correction inflating early steps; (5) SGD noise escaping a saddle
  point where full-batch GD stalls; (6) AdaGrad's learning rate decaying to 0 while RMSProp's survives.
- **[exercises.md](exercises.md)** — derive momentum and the Adam update, implement each optimizer,
  reproduce every experiment.
- **[references.md](references.md)** — Kingma & Ba (Adam), Loshchilov & Hutter (AdamW), Sutskever et al.
  (momentum), Smith (one-cycle).

**Next**: [07.07 — Normalization](../07-normalization/) — batch and layer normalization, which
stabilize the activations these optimizers descend on.
