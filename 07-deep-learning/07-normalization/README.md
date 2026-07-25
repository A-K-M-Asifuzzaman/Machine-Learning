# 07.07 — Normalization

> **Prerequisites**: [07.05](../05-initialization/) (keeping activation variance stable),
> [07.02 §8](../02-backpropagation/) (gradient flow), [07.06](../06-optimizers/) (the landscape
> optimizers descend).
> **You will be able to**: derive batch normalization's forward and backward passes, explain why it
> helps (a smoother landscape, not "covariate shift"), handle the train/inference distinction, and
> choose batch vs layer vs group norm.

---

## Table of contents

1. [The problem normalization solves](#1-the-problem-normalization-solves)
2. [Batch normalization](#2-batch-normalization)
3. [Why it actually works](#3-why-it-actually-works)
4. [Train vs inference: running statistics](#4-train-vs-inference-running-statistics)
5. [The batch-size dependence](#5-the-batch-size-dependence)
6. [Layer normalization](#6-layer-normalization)
7. [Instance and group normalization](#7-instance-and-group-normalization)
8. [Where to place normalization](#8-where-to-place-normalization)
9. [Choosing a normalization](#9-choosing-a-normalization)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. The problem normalization solves

Good initialization ([07.05](../05-initialization/)) sets the activation scale correctly at the *start*
of training — but as the weights change, the distribution of each layer's inputs *drifts*, and can
again grow, shrink, or become badly-scaled. This makes training sensitive to the learning rate and slow
to converge, because each layer is constantly chasing a moving input distribution.

**Normalization** fixes this *during* training, not just at init: it inserts a step that **re-centers
and re-scales** each layer's activations to a stable distribution (roughly zero mean, unit variance) on
every forward pass. The signal is kept in a healthy range throughout training, no matter how the weights
evolve. This one idea — normalize the activations inside the network — was one of the most impactful
techniques of the 2010s: it lets you train deeper networks, use much larger learning rates, worry less
about initialization, and converge faster. Batch normalization ([§2](#2-batch-normalization)) is the
original; layer norm ([§6](#6-layer-normalization)) is the one inside every Transformer.

---

## 2. Batch normalization

**Batch normalization** (Ioffe & Szegedy, 2015) normalizes each feature (channel) across the **batch**.
For a mini-batch of pre-activations, for each feature $j$ it computes the batch mean $\mu_j$ and
variance $\sigma_j^2$, normalizes, and then applies a **learnable** scale $\gamma_j$ and shift
$\beta_j$:

$$
\hat x_{ij} = \frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \qquad y_{ij} = \gamma_j\,\hat x_{ij} + \beta_j.
$$

The normalization ($\hat x$) forces every feature to zero mean and unit variance across the batch —
stabilizing the scale. The learnable $\gamma, \beta$ then let the network *undo* the normalization if
it wants to (setting $\gamma = \sigma, \beta = \mu$ recovers the original), so **no representational
power is lost** — the network chooses the best scale/shift by gradient descent. Both are learned like
any other parameter.

The backward pass is more involved than most layers (the mean and variance couple all examples in the
batch, so each output's gradient depends on all inputs), but it has a clean closed form that
`from_scratch.py` derives and verifies against PyTorch. Placed after (or before, §8) each linear/conv
layer, batch norm keeps activations well-scaled throughout the network.

---

## 3. Why it actually works

Batch norm's original justification was reducing **"internal covariate shift"** — the drift in each
layer's input distribution as earlier layers update. This is intuitive but turns out to be **largely
wrong** as the explanation. Santurkar et al. (2018) showed that BN helps even when covariate shift is
artificially *increased*, and that the real mechanism is:

> Batch norm makes the **loss landscape smoother** — it reduces the Lipschitz constant of the loss and
> its gradients, so the loss changes more predictably as you step. A smoother landscape means gradients
> point more reliably toward the minimum and you can take **larger steps** without overshooting.

The concrete consequences, all measured in the experiments:

- **Higher learning rates** are stable, so training is faster (Experiment 2).
- **Less sensitivity to initialization** — BN corrects a poorly-scaled signal on the fly, so a bad init
  that would otherwise fail can still train (Experiment 3, and why [07.05 §9](../05-initialization/)
  says init matters *less* with normalization).
- **A mild regularization effect** — the batch statistics are noisy (they depend on which examples are
  in the batch), injecting noise that acts like a small dropout ([07.08](../08-regularization/)).

Whatever the precise mechanism, the empirical effect is decisive: normalization makes deep networks
much easier and faster to train. The "covariate shift" story is a useful mnemonic but not the truth.

---

## 4. Train vs inference: running statistics

Batch norm has a subtlety that trips up beginners: it behaves **differently in training and inference**.

- **Training**: normalize using the **current mini-batch's** mean and variance (so each example's
  normalization depends on its batchmates).
- **Inference**: you may have a single example, or want deterministic outputs, so you cannot use batch
  statistics. Instead BN uses **running (exponential moving average) estimates** of the mean and
  variance, accumulated over training, as fixed population statistics.

This is why frameworks have a **train / eval mode** (`model.train()` vs `model.eval()`), and why
*forgetting to switch to eval mode* is a classic bug: using batch statistics at inference makes the
prediction for one example depend on the others in its batch — nonsensical and non-deterministic.
Experiment 4 shows the two modes giving different outputs, and how using batch stats at inference makes
a single example's prediction depend on its (arbitrary) batchmates. Always call `.eval()` before
inference.

---

## 5. The batch-size dependence

Because batch norm's statistics are estimated **from the batch**, its behavior depends on the batch
size — and this is its main weakness. With a **large** batch the mean/variance are accurate; with a
**tiny** batch (say 2–4, common in memory-limited settings like large images or detection) they are
**noisy and unreliable**, and BN degrades or destabilizes training. Batch size 1 is degenerate (zero
variance). Experiment 5 shows BN's effective normalization getting noisier as the batch shrinks.

This batch-dependence is exactly what the other normalizations (§6–§7) avoid: they normalize over
dimensions *within* each example, so they are **independent of the batch**. When batches must be small,
or the model must not depend on batch composition (recurrent nets, generative models), batch norm is
the wrong choice.

---

## 6. Layer normalization

**Layer normalization** (Ba et al., 2016) normalizes across the **features of each individual example**
instead of across the batch. For each example, it computes the mean and variance over that example's
own feature vector, normalizes, and applies learnable $\gamma, \beta$:

$$
\hat x_{ij} = \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \qquad \mu_i = \frac1d\sum_j x_{ij},\ \ \sigma_i^2 = \frac1d\sum_j (x_{ij}-\mu_i)^2.
$$

The crucial difference: the statistics are per-**example**, computed over the feature dimension, so
layer norm is **completely independent of the batch** — the same input gives the same output regardless
of what else is in the batch, and there is **no train/inference distinction** (no running statistics
needed). This makes it ideal for:

- **Transformers** — layer norm is the normalization inside every Transformer block
  ([11.xx](../../)); it works with variable sequence lengths and any batch size.
- **RNNs** — where batch norm's cross-time-step statistics are awkward.
- **Small or variable batch sizes** — no batch dependence.

Experiment 6 verifies that layer norm gives identical per-example outputs regardless of batch
composition, where batch norm does not. LayerNorm's batch-independence is why it, not BatchNorm, powers
modern sequence models.

---

## 7. Instance and group normalization

Two more, mainly for vision:

- **Instance normalization** — normalize each channel of each example *spatially* (per-image,
  per-channel). Removes instance-specific contrast, the key to **style transfer**.
- **Group normalization** (Wu & He, 2018) — split channels into groups and normalize within each group,
  per example. A **batch-independent** middle ground between layer norm (one group) and instance norm
  (one channel per group). It matches batch norm's accuracy at **small batch sizes** (detection,
  segmentation), where BN fails (§5) — the standard choice when vision batches are tiny.

These, with BN and LN, cover the normalization design space: *which dimensions you average over*
(batch, channel, feature, group) is the only real choice, and it is dictated by the task and the batch
constraints.

---

## 8. Where to place normalization

Two placement questions, both consequential:

- **Before or after the activation?** The original BN paper placed it *before* the activation
  (normalize the pre-activation, then apply ReLU); in practice both are used and the difference is
  small. Placing it before keeps the activation's input well-scaled.
- **Pre-norm vs post-norm (Transformers).** In a residual block, **post-norm** applies normalization
  *after* adding the residual (original Transformer); **pre-norm** applies it *inside* the residual
  branch, before the sublayer. **Pre-norm is more stable to train** (it keeps a clean gradient path
  through the residual, avoiding the need for careful warmup) and is standard in modern large models,
  though post-norm can reach slightly better final quality with careful tuning. This placement choice
  materially affects trainability of very deep Transformers.

The placement interacts with residual connections and warmup ([07.06 §7](../06-optimizers/)); pre-norm
is the safe default for deep stacks.

---

## 9. Choosing a normalization

| Situation | Normalization |
|---|---|
| CNNs / vision, reasonable batch size | **Batch norm** |
| Transformers, RNNs, sequence models | **Layer norm** |
| Small / variable batch (detection, segmentation) | **Group norm** |
| Style transfer | **Instance norm** |
| Must be batch-independent / deterministic | **Layer / group norm** |

The rules: **batch norm for CNNs** with decent batches; **layer norm for Transformers and RNNs**;
**group norm when batches are tiny**. The deciding factor is almost always *what dimensions you can
safely average over* — the batch (BN) only when it is large and representative, otherwise per-example
(LN/GN). Modern large-model practice is overwhelmingly **layer norm** (Transformers), which is why it is
the one to know best.

---

## 10. Common misconceptions

**"Batch norm works by reducing internal covariate shift."**
That was the original story, but the real mechanism is a smoother loss landscape (§3, Santurkar et al.);
BN helps even when covariate shift is increased.

**"Batch norm behaves the same in training and inference."**
No — training uses batch statistics, inference uses running (EMA) statistics (§4). Forgetting
`.eval()` is a classic bug.

**"Normalization loses information by forcing zero mean / unit variance."**
The learnable $\gamma, \beta$ can undo the normalization, so no representational power is lost (§2).

**"Batch norm works for any batch size."**
It degrades with small batches because its statistics become noisy (§5); use group or layer norm then.

**"Layer norm and batch norm are interchangeable."**
They average over different dimensions — batch vs features — so LN is batch-independent and has no
train/inference split (§6). LN is for sequences; BN is for vision.

**"Where you place normalization doesn't matter."**
Pre-norm vs post-norm materially affects the trainability of deep Transformers (§8).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — batch norm (forward *and* the nontrivial backward) and layer
  norm in NumPy, with forward and gradients verified against PyTorch. Six experiments: (1) BN
  stabilizing activation distributions across depth; (2) BN enabling higher learning rates / faster
  convergence; (3) BN letting a badly-initialized network still train; (4) train vs eval mode and the
  batch-statistics-at-inference bug; (5) BN's statistics degrading with tiny batches while LN is
  unaffected; (6) layer norm giving batch-independent per-example outputs.
- **[exercises.md](exercises.md)** — derive the BN backward pass, implement group norm, reproduce every
  experiment.
- **[references.md](references.md)** — Ioffe & Szegedy (BN), Ba et al. (LN), Santurkar et al. (why BN
  works), Wu & He (group norm).

**Next**: [07.08 — Regularization](../08-regularization/) — dropout, weight decay, early stopping, and
data augmentation: the tools that make an over-parametrized network generalize.
