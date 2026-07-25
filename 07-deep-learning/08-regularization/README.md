# 07.08 — Regularization

> **Prerequisites**: [05.01](../../05-model-evaluation/01-bias-variance-and-theory/) (bias-variance,
> overfitting), [03.02](../../03-supervised-learning/02-regularized-linear-models/) (L2/L1 penalties),
> [07.06](../06-optimizers/) (weight decay in the optimizer).
> **You will be able to**: apply weight decay, dropout, early stopping, and data augmentation, explain
> each as a way to trade bias for variance, and combine them to make an over-parametrized network
> generalize.

---

## Table of contents

1. [Capacity, overfitting, and the goal](#1-capacity-overfitting-and-the-goal)
2. [Weight decay (L2)](#2-weight-decay-l2)
3. [Dropout](#3-dropout)
4. [Dropout at inference](#4-dropout-at-inference)
5. [Early stopping](#5-early-stopping)
6. [Data augmentation](#6-data-augmentation)
7. [Other regularizers](#7-other-regularizers)
8. [Implicit regularization](#8-implicit-regularization)
9. [Combining regularizers](#9-combining-regularizers)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. Capacity, overfitting, and the goal

A deep network has enormous capacity — enough to fit *any* training set, including its noise, and even
random labels ([05.01 §10](../../05-model-evaluation/01-bias-variance-and-theory/)). Left unchecked, it
**overfits**: the training loss goes to zero while the test loss climbs, because the network memorizes
the training examples instead of learning the underlying pattern. **Regularization** is the collection
of techniques that fight this — that reduce the gap between training and test performance so the network
*generalizes*.

Every regularizer is, in the language of [05.01](../../05-model-evaluation/01-bias-variance-and-theory/),
a way to **trade a little bias for a large reduction in variance**: constrain the model (add bias) so
it is less sensitive to the particular training sample (less variance), and the total error drops. The
methods differ in *how* they constrain — penalize large weights (§2), inject noise (§3), stop early
(§5), expand the data (§6) — but they share that one purpose. Experiment 1 establishes the baseline: a
big network on small data overfitting (train loss → 0, validation loss high), which the rest of the
chapter then reduces.

---

## 2. Weight decay (L2)

The simplest regularizer penalizes large weights by adding an **L2 penalty** to the loss:

$$
L_{\text{reg}} = L + \frac{\lambda}{2}\sum_w w^2.
$$

The gradient of the penalty is $\lambda w$, so each update shrinks every weight toward zero by a factor
proportional to its size — hence **weight decay**:

$$
w \leftarrow w - \eta(\nabla_w L + \lambda w) = (1 - \eta\lambda)\,w - \eta\nabla_w L.
$$

This is exactly ridge regression's penalty ([03.02](../../03-supervised-learning/02-regularized-linear-models/))
applied to a neural net. Smaller weights mean a smoother, lower-variance function (large weights create
sharp, wiggly decision boundaries), so weight decay reduces overfitting — Experiment 2 shows the
validation loss improving as $\lambda$ grows, up to a point.

One subtlety: with **adaptive optimizers** (Adam), adding $\lambda w$ to the gradient is *not* the same
as true weight decay, because the adaptive $1/\sqrt{v}$ scaling distorts it. **AdamW**
([07.06 §6](../06-optimizers/)) fixes this by applying the decay $(1-\eta\lambda)w$ *directly* to the
weights, decoupled from the gradient — which is why AdamW generalizes better and is the modern default.
$\lambda$ is a key hyperparameter (typical $10^{-4}$–$10^{-2}$).

---

## 3. Dropout

**Dropout** (Srivastava et al., 2014) is the deep-learning-native regularizer, and it works by
**injecting noise**: during training, each unit is randomly **set to zero** with probability $p$ (the
*dropout rate*, e.g. 0.5) on each forward pass. A different random subset is dropped every step.

Two ways to understand why this regularizes:

- **It prevents co-adaptation.** A unit cannot rely on any *specific* other unit being present (it might
  be dropped), so units must learn features that are useful *on their own*, not brittle combinations
  that only work together. This redundancy generalizes better.
- **It is an implicit ensemble.** Each dropout mask defines a different "thinned" sub-network sharing
  weights; training with dropout trains an exponential number of these sub-networks, and inference
  (§4) approximately averages them — an ensemble ([06.01](../../06-ensembles/01-bagging/)) for free.

Dropout is applied to hidden layers (rate 0.5 is classic for dense layers; smaller, ~0.1–0.2, for
convolutional/modern nets, which use less dropout now that batch norm regularizes too). Experiment 3
shows dropout closing the train–validation gap. It is most valuable when data is limited relative to
model size.

---

## 4. Dropout at inference

Dropout, like batch norm ([07.07 §4](../07-normalization/)), behaves differently in training and
inference — and getting this wrong silently breaks the model. At **inference you want the full network**
(no random dropping, deterministic output), but if you simply turn dropout off, each unit now receives
input from *all* its predecessors, whereas during training it received input from only a fraction
$(1-p)$ of them — so the expected activation is too large by a factor $1/(1-p)$.

Two equivalent fixes keep the expected activation consistent:

- **Standard dropout**: at inference, **scale activations by $(1-p)$** to match the training
  expectation.
- **Inverted dropout** (the common implementation): during *training*, scale the kept activations by
  $1/(1-p)$; then inference needs no change at all. This keeps the expected activation at $1/(1-p)\cdot(1-p) = 1$
  during training, so test time is just the plain forward pass.

Experiment 4 verifies that inverted dropout keeps the expected activation constant between train and
test, so the two modes are consistent. As with batch norm, **switch to eval mode for inference** — the
framework handles the scaling, but only if you tell it you are evaluating.

---

## 5. Early stopping

The cheapest and one of the most effective regularizers: **stop training when the validation loss stops
improving.** As training proceeds, the training loss keeps falling, but the validation loss follows a
**U-shape** — it drops while the model learns the signal, then rises as the model starts memorizing
noise ([05.01 §4](../../05-model-evaluation/01-bias-variance-and-theory/)). Early stopping monitors the
validation loss and **keeps the weights from the epoch where it was lowest**, halting once it has not
improved for a set "patience" number of epochs.

This is regularization by *limiting how long* the model is allowed to fit — equivalent to constraining
the weights to stay near their (small) initial values, a soft capacity limit. It is essentially free
(you need a validation set anyway), requires no hyperparameter beyond patience, and is the same
principle as choosing the number of boosting rounds ([06.04 §10](../../06-ensembles/04-gradient-boosting/)).
Experiment 5 shows the validation U-curve and early stopping catching its minimum. **Always use early
stopping** — it costs nothing and protects against over-training.

---

## 6. Data augmentation

The most powerful regularizer of all, especially in vision: **expand the training set with
label-preserving transformations**. A cat photo flipped horizontally, cropped, slightly rotated, or
color-jittered is still a cat — so each real image becomes many training examples, and the network
learns to be **invariant** to these nuisances instead of memorizing pixels.

- **Vision**: random flips, crops, rotations, scaling, color jitter, and stronger schemes (RandAugment,
  AutoAugment).
- **Audio**: time/frequency masking, pitch/speed shift.
- **Text**: synonym replacement, back-translation (harder — most edits change meaning).

Augmentation works because it directly attacks the root cause of overfitting — *too little data
relative to capacity* — by manufacturing more (correlated but useful) data, and it encodes real
invariances the task should respect. It is often the single highest-impact regularizer in practice,
frequently worth more than all the weight-decay/dropout tuning combined. Its cost is domain knowledge
(you must know which transforms preserve the label) and compute.

---

## 7. Other regularizers

- **Batch normalization** — its noisy batch statistics have a mild regularizing effect
  ([07.07 §3](../07-normalization/)); using BN often lets you use less dropout.
- **Label smoothing** — soften one-hot targets to discourage overconfidence
  ([07.04 §8](../04-loss-functions/)); improves calibration and generalization.
- **Mixup / CutMix** — train on convex combinations (or spliced patches) of pairs of examples and their
  labels; a strong modern augmentation-style regularizer.
- **Stochastic depth** — randomly drop whole *layers* (in residual nets) during training; dropout for
  depth.
- **Max-norm** — hard-constrain each weight vector's norm; sometimes paired with dropout.

These are tools layered on the core four (weight decay, dropout, early stopping, augmentation).

---

## 8. Implicit regularization

Not all regularization is explicit. **SGD itself regularizes**: its mini-batch noise
([07.06 §2](../06-optimizers/)) biases training toward **flat minima**, which generalize better than
sharp ones — a major reason over-parametrized networks trained with SGD generalize *at all* despite
having the capacity to memorize (the double-descent puzzle,
[05.01 §10](../../05-model-evaluation/01-bias-variance-and-theory/)). Smaller batch sizes and larger
learning rates increase this implicit regularization.

Other implicit effects: **early stopping** limits how far weights move from their small init;
architectural choices (convolution's weight sharing, [08.xx](../../)) constrain the hypothesis class;
and even the initialization scale biases the solution. The lesson is that a network's generalization is
shaped by the *whole training procedure*, not just the explicit penalty terms — which is why "just add
more regularization" is not always the answer, and why the interaction of batch size, learning rate,
and explicit regularizers matters.

---

## 9. Combining regularizers

In practice you **combine several**, and they are complementary:

1. **Data augmentation** first — it attacks the root cause (too little data) and usually helps most.
2. **Weight decay** (AdamW) — a cheap, always-on baseline; tune $\lambda$.
3. **Early stopping** — free, always use it.
4. **Dropout** — when still overfitting, especially with limited data and dense layers (less needed
   with heavy BN/augmentation).
5. **Label smoothing / mixup** — for the last bit of generalization and calibration.

The right amount is set by the **train–validation gap**: a large gap means overfitting (add
regularization); training and validation both high and close means *under*fitting (reduce
regularization, add capacity) — the learning-curve diagnosis of
[05.01 §6](../../05-model-evaluation/01-bias-variance-and-theory/). Tune regularization against a
validation set ([05.04](../../05-model-evaluation/04-cross-validation/)), and remember that more
regularization is not always better — past the sweet spot it just adds bias.

---

## 10. Common misconceptions

**"Regularization always improves the model."**
It reduces *variance* at the cost of *bias*; past the sweet spot it under-fits and hurts (§1, §9). Tune
it against validation.

**"Dropout is applied at inference too."**
No — inference uses the full network with the activations scaled (or inverted-dropout's train-time
scaling), so training and test expectations match (§4). Forgetting eval mode is a bug.

**"Weight decay and L2 are identical for any optimizer."**
For plain SGD, yes; for adaptive optimizers (Adam), adding $\lambda w$ to the gradient differs from true
decay — AdamW decouples them (§2).

**"More dropout is always better."**
Too high a dropout rate under-fits (too much noise); modern nets with BN and augmentation often need
little or none (§3, §9).

**"Data augmentation is a minor trick."**
It is often the *highest-impact* regularizer, attacking the root cause (too little data) directly (§6).

**"Over-parametrized networks can't generalize."**
They do, thanks to explicit regularization *and* SGD's implicit regularization toward flat minima (§8,
double descent).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — weight decay, dropout (inverted, forward + backward), and
  early stopping in NumPy, in a trainable network, with dropout verified against PyTorch. Six
  experiments: (1) the overfitting baseline (big net, small data — train → 0, val high); (2) weight
  decay reducing the train–val gap; (3) dropout closing the gap; (4) inverted dropout keeping the
  expected activation constant train vs test; (5) the validation U-curve and early stopping catching
  its minimum; (6) dropout as an implicit ensemble (averaging over masks).
- **[exercises.md](exercises.md)** — derive weight decay and the dropout inference scaling, implement
  each regularizer, reproduce every experiment.
- **[references.md](references.md)** — Srivastava et al. (dropout), Loshchilov & Hutter (AdamW),
  Zhang et al. (mixup, rethinking generalization).

**Next**: [07.09 — Training Dynamics & Debugging](../09-training-dynamics/) — putting it all together:
diagnosing why a network won't train, and the practical recipe for getting deep learning to work.
