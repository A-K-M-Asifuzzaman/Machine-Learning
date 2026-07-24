# 07.05 — Weight Initialization

> **Prerequisites**: [07.02 §8](../02-backpropagation/) (vanishing/exploding gradients as a product of
> Jacobians), [07.03](../03-activations/) (activation saturation),
> [00.03](../../00-mathematical-foundations/03-probability/) (variance of sums).
> **You will be able to**: explain why all-zeros init fails, derive the variance-preservation
> condition, choose He vs Glorot init from the activation, and see how bad init makes training vanish,
> explode, or never start.

---

## Table of contents

1. [Why initialization matters](#1-why-initialization-matters)
2. [The symmetry problem: never initialize to zero](#2-the-symmetry-problem-never-initialize-to-zero)
3. [The goal: preserve variance across layers](#3-the-goal-preserve-variance-across-layers)
4. [The forward-pass derivation](#4-the-forward-pass-derivation)
5. [Xavier / Glorot initialization](#5-xavier--glorot-initialization)
6. [He / Kaiming initialization](#6-he--kaiming-initialization)
7. [What bad initialization does](#7-what-bad-initialization-does)
8. [Biases, and other schemes](#8-biases-and-other-schemes)
9. [Initialization in the modern stack](#9-initialization-in-the-modern-stack)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Why initialization matters

Before training can improve the weights, they must be *set to something*, and that starting point
matters enormously. A network is a deep composition, so the initial weights determine whether the
forward signal and the backward gradient **maintain a healthy scale as they pass through the layers**,
or whether they **vanish** (shrink toward zero) or **explode** (blow up) — the same product-of-factors
problem as vanishing gradients ([07.02 §8](../02-backpropagation/)), now at initialization.

Get it wrong and training never even starts: if activations vanish, every gradient is ~0 and nothing
updates; if they explode, you get `NaN`s immediately. Get it right and the network begins in a regime
where gradients flow and gradient descent can make progress. Initialization is not a minor detail — for
deep networks (especially without normalization, §9), it is the difference between a network that
trains and one that does not. This chapter derives the *right* scale for the initial random weights from
a single principle: **keep the variance constant across layers.**

---

## 2. The symmetry problem: never initialize to zero

The most tempting init — all weights zero — is catastrophic, for a reason that has nothing to do with
scale. If every weight in a layer is identical (zero or any constant), then **every unit in that layer
computes exactly the same thing**: same output, and — because backprop treats them symmetrically — the
same gradient. So they update identically and *stay identical forever*. A layer of 100 units behaves
like a single unit; the network can never use its capacity. This is the **symmetry problem**, and it
means the initial weights must be **random** to *break the symmetry* — to make units different so they
can specialize into different features.

Randomness is therefore mandatory, not optional. The only question is the *scale* of that
randomness — too small and signals vanish, too large and they explode (§7) — which §3–§6 pin down
exactly. Experiment 1 shows an all-zeros network's units remaining identical and the network failing to
learn, versus random init breaking the symmetry.

---

## 3. The goal: preserve variance across layers

Given that weights must be random, how large should they be? The guiding principle is **variance
preservation**: choose the weight scale so that the *variance of the activations stays roughly constant*
as the signal passes forward through the layers, and the *variance of the gradients stays constant* as
it passes backward. If each layer preserves variance, then a signal or gradient can traverse many
layers without shrinking to zero or growing without bound.

This is the key idea. A network that neither vanishes nor explodes its signals is one where every layer
has variance-preservation "gain" near 1 — and the initialization is what sets that gain at the start of
training. The whole theory (Glorot, He) is just: *solve for the weight variance that makes the
per-layer gain 1.* Experiment 2 measures activation variance across a deep network and shows good init
holding it constant while bad init makes it vanish or explode geometrically with depth.

---

## 4. The forward-pass derivation

Consider one linear layer $z_j = \sum_{i=1}^{n_{\text{in}}} W_{ji} x_i$, with inputs $x_i$ and weights
$W_{ji}$ drawn independently with mean 0. Treating them as independent random variables, the variance of
each output is a sum of $n_{\text{in}}$ independent terms:

$$
\mathrm{Var}(z_j) = n_{\text{in}}\cdot \mathrm{Var}(W)\cdot \mathrm{Var}(x).
$$

For the output variance to **equal** the input variance ($\mathrm{Var}(z) = \mathrm{Var}(x)$, i.e. the
layer preserves variance forward), we need

$$
\boxed{\ \mathrm{Var}(W) = \frac{1}{n_{\text{in}}}\ }.
$$

So the weights should be scaled by $1/\sqrt{n_{\text{in}}}$: a wider layer (more inputs summed) needs
*smaller* weights, so the sum doesn't blow up. This is the core result. The **backward** pass gives the
symmetric condition $\mathrm{Var}(W) = 1/n_{\text{out}}$ (to preserve gradient variance), and the two
different requirements are what Glorot and He reconcile (§5–§6). Everything else is which $n$ to use and
a constant factor for the activation.

---

## 5. Xavier / Glorot initialization

**Xavier/Glorot** initialization (2010) splits the difference between the forward requirement
($\mathrm{Var}(W) = 1/n_{\text{in}}$) and the backward one ($1/n_{\text{out}}$) by averaging:

$$
\mathrm{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}.
$$

Drawn either from a uniform or a normal distribution with this variance. It is designed for activations
that are **linear near zero and symmetric** — **tanh** (and, roughly, sigmoid) — where the derivation's
assumption that the activation preserves variance (gain ≈ 1 near 0) holds. Glorot init was the first
principled scheme and made training deep tanh networks reliable. But it *undershoots* for ReLU, because
ReLU is not linear near zero — it kills half the signal (§6).

---

## 6. He / Kaiming initialization

**ReLU zeroes out the negative half** of its inputs, so on average it **halves the variance** of the
signal passing through. Glorot's derivation assumed a variance-preserving activation and so
under-scales for ReLU, letting the signal shrink layer by layer. **He/Kaiming** initialization (2015)
corrects this with a factor of 2:

$$
\mathrm{Var}(W) = \frac{2}{n_{\text{in}}}.
$$

The extra 2 exactly compensates for ReLU discarding half the units, so the forward variance is
preserved through ReLU layers. **He init is the standard for ReLU networks** (and its Leaky/ELU/GELU
cousins) — which is to say, the standard for most modern networks. Experiment 3 shows He preserving
activation variance through a deep ReLU network while Glorot lets it decay, confirming the factor-of-2
matters.

The pairing to remember: **He for ReLU-family activations, Glorot for tanh/sigmoid.** Using the wrong
one (Glorot with ReLU) still works with normalization but is measurably worse without it.

---

## 7. What bad initialization does

Initialization has a narrow good regime, and both sides of it fail:

- **Too small** ($\mathrm{Var}(W) \ll 1/n_{\text{in}}$): activations shrink geometrically with depth
  toward 0, so gradients vanish and the network **barely learns** — the loss sits nearly flat.
- **Too large** ($\mathrm{Var}(W) \gg 1/n_{\text{in}}$): activations grow geometrically and, with
  saturating activations, push into the flat regions (or overflow); gradients explode or vanish, and
  training **diverges to `NaN`** or stalls.
- **Just right** (He/Glorot): variance is preserved, gradients flow, and training **converges**.

Experiment 4 trains the *same* network from too-small, too-large, and correct init and shows the loss
staying flat, diverging, and converging respectively — the same architecture, three fates decided
entirely by the initial weight scale. This is the concrete reason initialization is worth getting right.

---

## 8. Biases, and other schemes

- **Biases** are almost always initialized to **0** — there is no symmetry problem for biases (the
  random weights already break it), and 0 is a neutral starting point. (Exceptions: sometimes a small
  positive bias for ReLU to reduce dead units, or specific biases for LSTM forget gates.)
- **Orthogonal initialization** — initialize weight matrices to be orthogonal, which exactly preserves
  norms (gain 1) and is especially helpful for **RNNs** and very deep networks, where it keeps the
  repeated multiplication ([07.02 §8](../02-backpropagation/)) from vanishing/exploding.
- **LSUV** (Layer-Sequential Unit-Variance) — data-driven: initialize, then rescale each layer so its
  output has unit variance on a real batch.
- **Fixup / T-Fixup** — careful init schemes that let very deep residual networks / Transformers train
  *without* normalization.

The mainstream default remains He (ReLU) or Glorot (tanh) with zero biases; the others are tools for
specific hard cases.

---

## 9. Initialization in the modern stack

Does initialization still matter now that we have batch/layer normalization ([07.07](../07-normalization/))
and residual connections? **Less, but yes.** Normalization rescales activations at every layer, so it
*corrects* a poorly-scaled signal on the fly and makes networks far more robust to the init choice —
one reason modern practice is more forgiving than the pre-2015 era. Residual connections similarly
create a "highway" that preserves gradient flow regardless of the weights.

But initialization still matters: at the *very first* step (before normalization statistics stabilize),
for networks *without* normalization, for very deep or recurrent networks where errors compound, and
for training stability and speed even when convergence is eventually reached. The right default costs
nothing and only helps, so use He/Glorot. Initialization went from "make-or-break" to "still worth
getting right" — a downgrade in criticality, not in relevance.

---

## 10. Common misconceptions

**"Just initialize weights to zero (or a constant)."**
Then every unit in a layer is identical and stays identical — the symmetry problem (§2). Weights must
be random.

**"Any small random values will do."**
The *scale* is what matters: too small vanishes, too large explodes (§7). Use the variance-preservation
scale ($1/\sqrt{n_{\text{in}}}$-ish).

**"He and Glorot are interchangeable."**
He has an extra factor of 2 for ReLU's half-killing (§6). Glorot under-scales for ReLU; use He for
ReLU, Glorot for tanh.

**"Bias initialization matters as much as weights."**
Biases are fine at 0 (§8); the random *weights* break symmetry.

**"With batch norm, initialization is irrelevant."**
Normalization makes init *less* critical but not irrelevant — the first steps, no-norm networks, and
very deep/recurrent nets still need good init (§9).

**"Bigger initial weights help the network learn faster."**
Too-large init pushes activations into saturation / overflow and explodes gradients — training diverges
(§7). The right scale, not a big scale, is what helps.

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the initialization schemes (zeros, naive scaled, Glorot, He,
  orthogonal) in NumPy, with tools to measure activation and gradient variance across a deep network,
  and He/Glorot scales checked against PyTorch's `nn.init`. Five experiments: (1) the symmetry problem —
  zero init keeps units identical and fails to learn; (2) variance preservation across depth (good init
  holds it, bad init vanishes/explodes); (3) He vs Glorot for ReLU — He preserves variance, Glorot
  decays; (4) bad init → training vanishes / diverges / converges from the same net; (5) activation
  standard deviation by layer for each scheme.
- **[exercises.md](exercises.md)** — derive the variance-preservation condition and the He factor,
  implement the schemes, reproduce every experiment.
- **[references.md](references.md)** — Glorot & Bengio, He et al., Saxe et al. (orthogonal), Mishkin &
  Matas (LSUV).

**Next**: [07.06 — Optimizers](../06-optimizers/) — gradient descent and its accelerated variants
(momentum, RMSProp, Adam) that turn gradients into weight updates.
