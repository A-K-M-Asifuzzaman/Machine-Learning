# 07.03 — Activation Functions

> **Prerequisites**: [07.01](../01-neural-network-basics/) (why nonlinearity),
> [07.02](../02-backpropagation/) (how $\sigma'$ enters the gradient and causes vanishing).
> **You will be able to**: explain what each activation does to the forward signal *and* the
> backward gradient, diagnose saturation and dying units, and choose an activation — ReLU by default,
> GELU/Swish for Transformers — for the right reasons.

---

## Table of contents

1. [The activation shapes the gradient, not just the output](#1-the-activation-shapes-the-gradient-not-just-the-output)
2. [Sigmoid](#2-sigmoid)
3. [Tanh](#3-tanh)
4. [ReLU](#4-relu)
5. [The dying ReLU, and Leaky/PReLU](#5-the-dying-relu-and-leakyprelu)
6. [ELU and SELU](#6-elu-and-selu)
7. [GELU and Swish](#7-gelu-and-swish)
8. [Softmax is an output activation](#8-softmax-is-an-output-activation)
9. [Choosing an activation](#9-choosing-an-activation)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. The activation shapes the gradient, not just the output

An activation function has *two* jobs, and the second is the one people forget. Forward, it injects the
**nonlinearity** that gives depth its power ([07.01 §4](../01-neural-network-basics/)). Backward, its
**derivative $\sigma'$ multiplies the gradient at every layer** — backprop's (BP2) is
$\boldsymbol\delta^{(\ell)} = ((\mathbf{W}^{(\ell+1)})^\top\boldsymbol\delta^{(\ell+1)}) \odot \sigma'(\mathbf{z}^{(\ell)})$
([07.02 §4](../02-backpropagation/)). So the *shape of $\sigma'$* determines whether gradients flow or
vanish.

This is the lens for the whole chapter: judge an activation not just by its curve but by **where its
derivative is large, small, or zero**. An activation that saturates ($\sigma' \to 0$ over wide regions)
chokes the gradient and stalls deep training; one whose derivative stays near 1 lets gradients flow.
The move from sigmoid/tanh to ReLU — the change that made deep networks trainable — is entirely a
story about $\sigma'$. Experiment 1 measures each activation's derivative saturating (or not), and
Experiment 2 traces the consequence for gradient flow through depth.

---

## 2. Sigmoid

$\sigma(z) = 1/(1+e^{-z})$ squashes any input to $(0, 1)$. Historically the default, now avoided in
hidden layers for three reasons:

- **It saturates.** For $|z| \gtrsim 5$, $\sigma(z)$ is flat and its derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$
  is nearly 0. **Its maximum derivative is only $0.25$** (at $z=0$), so *even in the best case* it
  shrinks the gradient by 4× per layer — the direct cause of vanishing gradients in deep sigmoid
  networks ([07.02 §8](../02-backpropagation/)).
- **It is not zero-centered.** Outputs are all positive, so the gradients w.r.t. a layer's weights all
  share a sign, forcing inefficient zig-zag updates.
- **`exp` is relatively expensive.**

Sigmoid survives in exactly one place: the **output** of a binary classifier, where you *want* a
probability in $(0,1)$ (and pair it with the log loss, [07.04](../04-loss-functions/)). As a hidden
activation in a deep net, it is a mistake.

---

## 3. Tanh

$\tanh(z)$ squashes to $(-1, 1)$ and is a rescaled sigmoid ($\tanh(z) = 2\sigma(2z)-1$). It fixes
sigmoid's zero-centering problem — outputs are centered at 0, so gradients are not sign-biased — and
its derivative $1 - \tanh^2(z)$ peaks at **1** (vs sigmoid's 0.25), so it shrinks gradients less. That
makes tanh strictly better than sigmoid for hidden layers, and it was the standard before ReLU.

But tanh **still saturates**: for $|z| \gtrsim 3$ its derivative is near 0, so deep tanh networks still
suffer vanishing gradients, just less severely than sigmoid. Tanh remains useful in specific places —
inside LSTM/GRU gates ([09.xx](../../)), and where a bounded, zero-centered output is genuinely wanted
— but for generic deep hidden layers, ReLU superseded it.

---

## 4. ReLU

$\mathrm{ReLU}(z) = \max(0, z)$ is the modern default, and its dominance is a direct consequence of §1.
Its derivative is **exactly 1 for $z > 0$** and **0 for $z < 0$** — so on the active half it does not
shrink the gradient *at all*. Stacking many ReLU layers does not vanish the gradient the way sigmoid
does (the surviving factors are 1, not 0.25), which is why ReLU made training deep networks practical.
Its other virtues:

- **Cheap** — a single comparison, no `exp`.
- **Sparse activations** — about half the units output 0 for any input, a form of implicit
  regularization and efficiency.
- **Non-saturating (on the positive side)** — no upper bound, so no vanishing for large positive
  pre-activations.

Experiment 2 shows a deep ReLU network preserving gradient magnitude across depth where sigmoid's
collapses. ReLU has one failure mode, though — the dying ReLU (§5).

---

## 5. The dying ReLU, and Leaky/PReLU

ReLU's derivative is **0 for $z < 0$**, and that creates a pathology: if a unit's pre-activation is
negative for *every* input (pushed there by a large gradient step or bad init), it outputs 0 always,
its gradient is 0 always, and it **never updates again** — a permanently **dead** unit. In a badly
trained network a substantial fraction of units can die, wasting capacity. Experiment 3 measures the
dead-unit fraction and shows it rising with the learning rate / bad init.

The fix is to give the negative side a small nonzero slope so the gradient never fully vanishes:

- **Leaky ReLU**: $\max(\alpha z, z)$ with a small fixed $\alpha$ (e.g. 0.01) — units on the negative
  side still get a small gradient and can recover.
- **PReLU**: the same, but $\alpha$ is *learned* per channel.

Experiment 3 shows Leaky ReLU keeping units alive where plain ReLU kills them. Leaky ReLU is a cheap,
safe default when you suspect dying units; PReLU adds a few parameters for a small potential gain.

---

## 6. ELU and SELU

Two smooth activations with negative saturation:

- **ELU** (Exponential Linear Unit): $z$ for $z>0$, $\alpha(e^z - 1)$ for $z\le0$. Smooth, and its
  negative values push the mean activation toward zero (like tanh's centering), which can speed
  training. It saturates gently on the negative side (bounded below), avoiding dead units.
- **SELU** (Scaled ELU): ELU with specific fixed constants chosen so that, under certain conditions
  (careful init, no other normalization), activations **self-normalize** — automatically keep zero
  mean and unit variance across layers, removing the need for batch normalization
  ([07.07](../07-normalization/)). Elegant in theory, but finicky in practice (it needs its own init
  and breaks with dropout), so it never displaced ReLU + normalization.

These are worth knowing but are niche; ReLU (or GELU) plus normalization is the mainstream path.

---

## 7. GELU and Swish

The activations that power modern architectures, both smooth and non-monotonic:

- **GELU** (Gaussian Error Linear Unit): $z\cdot\Phi(z)$, where $\Phi$ is the standard-normal CDF. It
  smoothly gates the input by how far it is above zero — like a soft, probabilistic ReLU. **GELU is
  the standard activation in Transformers** (BERT, GPT), and its smoothness helps optimization.
- **Swish / SiLU**: $z\cdot\sigma(z)$ (sigmoid-weighted linear unit). Very close to GELU, found by
  neural-architecture search to slightly outperform ReLU on deep networks. Also non-monotonic (it dips
  slightly negative before rising).

Their advantage over ReLU is subtle — smoother gradients, a small nonzero response for slightly
negative inputs (no hard dead zone), and non-monotonicity that seems to help very deep models. The
gains are modest on small networks but consistent on large ones, which is why **GELU/Swish are the
default in large Transformers** while ReLU remains fine (and cheaper) for CNNs and MLPs. Experiment 4
compares their curves and gradients to ReLU.

---

## 8. Softmax is an output activation

**Softmax** turns a vector of scores (logits) into a probability distribution:
$\mathrm{softmax}(\mathbf{z})_k = e^{z_k}/\sum_j e^{z_j}$, with outputs positive and summing to 1. It
is **not a hidden-layer activation** — it is the **output** layer for multiclass classification, paired
with the cross-entropy loss ([07.04](../04-loss-functions/)). It couples all the outputs (each depends
on all logits), which is exactly what you want for mutually-exclusive classes. Numerically it must be
computed with the log-sum-exp trick (subtract the max logit) to avoid overflow — a detail
[07.04](../04-loss-functions/) covers. Do not use softmax between hidden layers; use it once, at the
output, for classification.

---

## 9. Choosing an activation

A practical decision guide, all following from §1's "what does $\sigma'$ do":

| Situation | Activation |
|---|---|
| Default for hidden layers (CNNs, MLPs) | **ReLU** |
| Suspect dead units / small networks | **Leaky ReLU** |
| Transformers / large models | **GELU** (or Swish) |
| Binary classifier output | **Sigmoid** (+ log loss) |
| Multiclass output | **Softmax** (+ cross-entropy) |
| Regression output | **Identity** (no activation) |
| Inside LSTM/GRU gates | **Sigmoid / tanh** (by design) |

The rules of thumb: **use ReLU by default**; switch to **Leaky ReLU** if you see dead units; use
**GELU/Swish** in Transformers and very deep networks; **never use sigmoid or tanh as a hidden
activation in a deep feedforward net** (they vanish gradients). Match the *output* activation to the
task and loss ([07.04](../04-loss-functions/)). The activation matters most through its effect on
gradient flow — which is why the whole field moved to non-saturating functions.

---

## 10. Common misconceptions

**"The activation just adds nonlinearity."**
It also shapes the *gradient* via $\sigma'$ (§1). Its saturation behavior — where $\sigma'\to0$ — is
what determines whether deep training works.

**"Sigmoid is a fine hidden activation."**
Its max derivative is 0.25 and it saturates, so deep sigmoid nets vanish gradients (§2). Use it only
for a binary output.

**"ReLU has no downsides."**
Dead units: a unit stuck at negative pre-activation never updates (§5). Leaky ReLU / good init /
sensible learning rates mitigate it.

**"GELU/Swish are always better than ReLU."**
The gains are modest and mostly matter for large/deep models (Transformers); ReLU is fine and cheaper
elsewhere (§7).

**"Softmax is an activation you can put anywhere."**
It is an *output* activation for multiclass classification, not a hidden-layer nonlinearity (§8).

**"Zero-centering doesn't matter."**
Sigmoid's all-positive outputs bias the weight gradients to one sign, causing zig-zag updates; tanh and
zero-centered activations avoid this (§2–§3).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — every activation and its derivative in NumPy (sigmoid, tanh,
  ReLU, Leaky ReLU, ELU, GELU, Swish, softmax), verified against PyTorch. Five experiments: (1)
  saturation — sigmoid/tanh derivatives collapsing to 0 for large $|z|$ while ReLU's stays 1; (2)
  gradient flow through a deep network by activation (ReLU preserves magnitude, sigmoid vanishes); (3)
  the dying-ReLU fraction and Leaky ReLU fixing it; (4) GELU/Swish vs ReLU curves and gradients; (5)
  ReLU training faster than sigmoid on the same task.
- **[exercises.md](exercises.md)** — derive each derivative, implement numerically stable softmax,
  reproduce every experiment.
- **[references.md](references.md)** — Glorot et al. (ReLU), Maas et al. (Leaky ReLU), Hendrycks &
  Gimpel (GELU), Ramachandran et al. (Swish).

**Next**: [07.04 — Loss Functions](../04-loss-functions/) — the objective each task minimizes, its
matching output activation, and the clean gradients that result.
