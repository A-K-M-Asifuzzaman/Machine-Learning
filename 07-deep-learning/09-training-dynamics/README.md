# 07.09 — Training Dynamics & Debugging

> **Prerequisites**: all of [07.01](../01-neural-network-basics/)–[07.08](../08-regularization/) — this
> chapter puts them together into a practical recipe and diagnostic toolkit.
> **You will be able to**: follow a disciplined training recipe, read loss curves and gradient/update
> statistics to diagnose failures, recognize the common bugs by their signatures, and find a working
> learning rate.

---

## Table of contents

1. [Deep learning is debugging](#1-deep-learning-is-debugging)
2. [The recipe](#2-the-recipe)
3. [The initial-loss sanity check](#3-the-initial-loss-sanity-check)
4. [Overfit a single batch](#4-overfit-a-single-batch)
5. [Reading the loss curves](#5-reading-the-loss-curves)
6. [Gradient and update statistics](#6-gradient-and-update-statistics)
7. [The learning-rate range test](#7-the-learning-rate-range-test)
8. [Common bugs and their signatures](#8-common-bugs-and-their-signatures)
9. [Monitoring and reproducibility](#9-monitoring-and-reproducibility)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Deep learning is debugging

Here is the truth no course tells you at the start: **most of deep learning in practice is debugging.**
A neural network is a "leaky abstraction" — it will *silently* fail. Unlike ordinary code that crashes
on a bug, a network with a wrong loss scale, a mislabeled dataset, a forgotten `eval()`, or a bad
learning rate will still *run*, still produce a loss curve, and still look like it is "training" — just
badly, or subtly wrong. There is no exception, no stack trace; the model just quietly does not learn
what you think it does.

So the skill that separates practitioners is not writing the model — frameworks make that easy — but
**diagnosing why it is not working.** This chapter is that skill: a disciplined recipe (§2) that
catches bugs early, a set of diagnostic checks (§3–§7) that localize them, and a catalog of common
failures by their signature (§8). Every technique from the earlier chapters (init, activations, loss,
normalization, optimizers, regularization) shows up here as a *thing that can go wrong* — and knowing
how each fails is what lets you fix it.

---

## 2. The recipe

A disciplined process (following Karpathy's "A Recipe for Training Neural Networks") prevents most
disasters by catching bugs when they are still cheap to find:

1. **Become one with the data.** Before any modeling, *look at the data* — inspect examples, labels,
   distributions, duplicates, corrupt samples. A huge fraction of "model" bugs are data bugs.
2. **Set up the end-to-end skeleton + a dumb baseline.** Get the full pipeline running (data →
   model → loss → eval) with a trivial model, and verify a baseline (predict the mean / majority
   class). This checks the plumbing before you add complexity.
3. **Overfit a single batch (§4).** The single most valuable check: confirm the model can drive the
   loss on a *tiny* fixed batch to ~0. If it cannot, the model, loss, or gradient flow is broken — fix
   *that* before anything else.
4. **Add regularization, then tune.** Only once the model can fit do you add data, regularization
   ([07.08](../08-regularization/)), and hyperparameter tuning to make it *generalize*.

The order matters: **make it work, make it right, make it fast** — get the model to learn *something*
first, then make it learn the *right* thing, then optimize. Skipping to step 4 (throwing a big
regularized model at the data and tuning) with a bug in step 1–3 wastes days.

---

## 3. The initial-loss sanity check

The very first number a model produces is a free bug detector. **At initialization, before any
training, the loss should equal the loss of a model that predicts the class prior** — because a
freshly-initialized network outputs roughly uniform/random predictions. For $K$-class classification
with cross-entropy, that is

$$
L_{\text{init}} \approx -\log\frac1K = \log K.
$$

For binary, $\log 2 \approx 0.693$; for 10 classes, $\log 10 \approx 2.303$. If your initial loss is
**far** from this, something is wrong *before training even starts*: the wrong loss (or its scale), a
bug in the output layer, a label off-by-one, or a bad initialization saturating the outputs.
Experiment 1 verifies the initial cross-entropy matches $\log K$ across class counts. This 10-second
check catches a surprising number of bugs — always compute it first.

---

## 4. Overfit a single batch

If you do one diagnostic, do this. Take a **tiny fixed batch** (say 4–10 examples), turn *off*
regularization, and train on *only that batch* for many steps. A correct model with correct gradients
**must** be able to drive the loss to essentially **zero** — it has more than enough capacity to
memorize a handful of examples.

- **If it reaches ~0 loss**: the model can represent the mapping, the loss is wired correctly, and
  gradients flow. The architecture and backward pass are sound; any remaining problem is about
  *generalization or optimization*, not correctness.
- **If it cannot reach ~0**: something is fundamentally broken — a detached gradient, a wrong loss, a
  data/label mismatch, a dead network (bad init/activation), or a learning rate so small nothing moves.
  Fix this before touching anything else.

Experiment 2 shows a healthy model overfitting a single batch to ~0 while a **broken** one (gradients
not flowing) stays stuck — the exact signal you look for. This check localizes bugs to *correctness*
vs *generalization* faster than anything else, which is why it is the first real experiment every time.

---

## 5. Reading the loss curves

The training and validation loss curves are the primary diagnostic instrument. Their shapes map to
specific problems:

| Curve shape | Diagnosis | Fix |
|---|---|---|
| Loss **flat / barely moves** | learning rate too low, dead network, gradient bug | raise LR, check init/activations, gradient-check |
| Loss → **`NaN`** / diverges | learning rate too high, exploding gradients, numerical bug | lower LR, clip gradients, use stable loss (§8) |
| Train ↓ to ~0, **val high** (big gap) | overfitting | regularize ([07.08](../08-regularization/)), more data |
| Train & val both **high and flat** | underfitting | more capacity, longer, higher LR, less regularization |
| Loss drops then **plateaus** | LR too high for fine convergence | decay the LR ([07.06 §7](../06-optimizers/)) |
| Train loss **spiky / noisy** | LR high or batch too small | lower LR, larger batch |

Experiment 3 produces the too-low / too-high / just-right curves side by side. Reading these shapes is
the fastest way to a diagnosis — the loss curve tells you *which* knob is wrong before you check
anything else. Plot train *and* val (the gap is the overfitting signal,
[05.01 §6](../../05-model-evaluation/01-bias-variance-and-theory/)), and plot on a log scale to see
early dynamics.

---

## 6. Gradient and update statistics

When the loss curve is ambiguous, look *inside* the network. Two cheap, powerful diagnostics:

- **The update-to-weight ratio.** For each layer, track $\lVert\eta\,\Delta W\rVert / \lVert W\rVert$ —
  the relative size of the update. A healthy value is around **$10^{-3}$** (updates ~0.1% of the
  weights per step). **Much larger** ($\gg 10^{-2}$) means the learning rate is too high (unstable);
  **much smaller** ($\ll 10^{-4}$) means it is too low (glacial) or gradients are vanishing. Experiment
  4 measures this ratio at three learning rates and shows the healthy band.
- **Per-layer gradient magnitudes.** Vanishing gradients show up as tiny gradients in early layers
  ([07.02 §8](../02-backpropagation/)); exploding as huge ones. Dead ReLU units show up as zero
  activations ([07.03 §5](../03-activations/)). Watching these localizes the failing layer.

Also useful: **activation statistics** (mean/std per layer — should be stable, not vanishing/exploding,
[07.05](../05-initialization/)), the **gradient norm** (a spike precedes a `NaN`), and the fraction of
**dead units**. These internal signals catch problems the loss curve hides.

---

## 7. The learning-rate range test

Since the learning rate is the most important and most fragile hyperparameter
([07.06 §3](../06-optimizers/)), find a good one *systematically* rather than by guessing. The **LR
range test** (Smith): start from a tiny learning rate and **increase it exponentially** over a few
hundred steps, recording the loss at each rate. The loss will:

- barely move at first (LR too small),
- then **drop steeply** (the good range),
- then **blow up** (LR too large).

Pick a learning rate near the point of **steepest descent** — roughly an order of magnitude below where
the loss starts diverging. Experiment 5 runs the range test and identifies the sweet spot. This one
sweep replaces a lot of blind trial-and-error and is standard practice before a serious training run.

---

## 8. Common bugs and their signatures

A field guide to the failures you *will* hit, with the tell-tale signature of each:

| Bug | Signature |
|---|---|
| **Learning rate wrong** | flat loss (too low) or `NaN` (too high) — check first |
| **Forgot to zero gradients** | gradients accumulate; loss erratic or diverges |
| **Forgot `eval()` mode** | BN/dropout use train behavior at test; val loss inflated / non-deterministic ([07.07 §4](../07-normalization/)) |
| **Wrong loss / output activation** | initial loss $\ne \log K$ (§3); e.g. softmax applied twice |
| **Data/label misalignment** | can't overfit a single batch (§4); train loss stuck |
| **Data not shuffled** | loss oscillates with the label ordering; poor convergence |
| **Numerical instability** | `NaN` from `exp`/`log`; use `*_with_logits` ([07.04 §7](../04-loss-functions/)) |
| **Data leakage** | val/test accuracy suspiciously perfect ([05.04 §7](../../05-model-evaluation/04-cross-validation/)) |
| **Wrong input normalization** | slow/unstable training; check input mean/std |
| **Bad init** | can't start learning; vanishing/exploding activations ([07.05](../05-initialization/)) |

The meta-lesson: each earlier chapter's topic is a bug waiting to happen, and its *signature* is how you
recognize it. When training fails, run down this list against the symptoms.

---

## 9. Monitoring and reproducibility

- **Monitor** during training: train and val loss, the learning rate (with schedule), the gradient
  norm, per-layer update-to-weight ratios (§6), and the actual task metric ([05.02](../../05-model-evaluation/02-regression-metrics/)–[05.03](../../05-model-evaluation/03-classification-metrics/)).
  Tools (TensorBoard, Weights & Biases) make these curves easy to watch — and watching them live is how
  you catch a divergence at step 500 instead of after an overnight run.
- **Reproducibility**: set all random **seeds** (data, init, dropout, augmentation), and enable
  deterministic ops if you need bit-exact runs. Non-reproducible training makes debugging nearly
  impossible — you cannot tell whether a change helped or you just got a lucky seed. Log the config,
  code version, and data version with every run.
- **Scale up carefully**: once a small model trains correctly on a subset, scale the model and data
  *incrementally*, re-checking the diagnostics at each step. Bugs are far cheaper to find on a small,
  fast model than on the full run.

---

## 10. Common misconceptions

**"If it runs without errors, it's training correctly."**
Neural nets fail *silently* — a wrong loss, bad LR, or data bug produces a plausible loss curve while
learning the wrong thing (§1). Verify with the sanity checks.

**"Start by tuning hyperparameters."**
Start by making sure the model can *learn at all* — overfit a single batch first (§2, §4). Tuning a
buggy model is wasted effort.

**"A decreasing loss means everything is fine."**
It can decrease while a bug (unshuffled data, wrong metric, leakage) makes the result meaningless.
Check the initial loss, the train-val gap, and the actual metric (§3, §5, §8).

**"Guess and adjust the learning rate."**
Use the LR range test to find it systematically (§7). It is the most important knob and worth the one
sweep.

**"The loss curve is all I need to look at."**
Gradient/update statistics and activation distributions catch problems the loss hides (§6).

**"Reproducibility is a nice-to-have."**
Without fixed seeds you cannot tell whether a change helped or a seed did — it is essential for
debugging (§9).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — a small trainable classifier plus the diagnostic toolkit in
  NumPy. Five experiments: (1) the initial-loss sanity check (CE ≈ $\log K$ across class counts); (2)
  overfitting a single batch — a healthy model → ~0 loss, a broken (no-gradient-flow) model stuck; (3)
  the loss-curve signatures (too-low / too-high / just-right learning rate); (4) the update-to-weight
  ratio and its healthy ~$10^{-3}$ band; (5) the learning-rate range test finding the sweet spot.
- **[exercises.md](exercises.md)** — implement each diagnostic, plant and detect bugs, reproduce every
  experiment.
- **[references.md](references.md)** — Karpathy's recipe, the CS231n practical notes, Smith's LR range
  test.

**This completes Part 7 — Deep Learning Foundations.** From the MLP and backprop through activations,
losses, initialization, optimizers, normalization, and regularization to this diagnostic recipe, you
can now build, train, and *debug* a deep network from scratch. **Next**: the architectures that
specialize these foundations — [Part 8 — Computer Vision (CNNs)](../../), sequence models, and
[Transformers](../../).
