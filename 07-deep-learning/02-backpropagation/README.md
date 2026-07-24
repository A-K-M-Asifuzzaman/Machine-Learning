# 07.02 — Backpropagation

> **Prerequisites**: [07.01](../01-neural-network-basics/) (the MLP and its forward pass as a
> computational graph), [00.02](../../00-mathematical-foundations/02-calculus-and-optimization/) (chain
> rule, gradients), [00.01](../../00-mathematical-foundations/01-linear-algebra/) (Jacobians).
> **You will be able to**: derive the four backprop equations, implement forward and backward passes
> from scratch, verify gradients by finite differences, explain backprop as reverse-mode autodiff, and
> understand where vanishing gradients come from.

---

## Table of contents

1. [The problem: gradients, efficiently](#1-the-problem-gradients-efficiently)
2. [The chain rule is the whole idea](#2-the-chain-rule-is-the-whole-idea)
3. [Forward pass, then backward pass](#3-forward-pass-then-backward-pass)
4. [The four backpropagation equations](#4-the-four-backpropagation-equations)
5. [Backprop as reverse-mode autodiff](#5-backprop-as-reverse-mode-autodiff)
6. [Gradient checking](#6-gradient-checking)
7. [Why reverse mode](#7-why-reverse-mode)
8. [Where vanishing and exploding gradients come from](#8-where-vanishing-and-exploding-gradients-come-from)
9. [Autograd: how frameworks do it](#9-autograd-how-frameworks-do-it)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. The problem: gradients, efficiently

To train a network we minimize a loss $C(\boldsymbol\theta)$ over its parameters $\boldsymbol\theta$
(all the weights and biases) by gradient descent, which needs $\nabla_{\boldsymbol\theta} C$ — the
gradient of the loss with respect to *every* parameter. A modern network has millions to billions of
parameters, so this had better be cheap.

The naive way — **finite differences**, nudging each parameter and re-running the forward pass — costs
one forward pass *per parameter*: $O(P)$ forward passes for $P$ parameters. For a million parameters
that is a million forward passes per gradient step: hopelessly slow. **Backpropagation** computes the
*entire* gradient — all $P$ partial derivatives — in a **single backward pass** costing about the same
as one forward pass. That efficiency is what makes training deep networks possible at all, and it is
the single most important algorithm in deep learning. Experiment 3 measures backprop giving identical
gradients to finite differences at a tiny fraction of the cost.

---

## 2. The chain rule is the whole idea

Backpropagation is nothing more than the **chain rule of calculus, applied systematically to a
composition of functions**. A network is a composition: the loss $C$ is a function of the last layer's
output, which is a function of the previous layer's, and so on back to the parameters. For a
composition $C = f(g(h(\boldsymbol\theta)))$, the chain rule gives

$$
\frac{\partial C}{\partial \boldsymbol\theta} = \frac{\partial C}{\partial f}\cdot\frac{\partial f}{\partial g}\cdot\frac{\partial g}{\partial h}\cdot\frac{\partial h}{\partial \boldsymbol\theta}.
$$

The key insight that makes this *efficient* is to compute the product **right to left** — starting
from the loss and multiplying the local derivatives backward. That way, the running product (the
gradient of the loss w.r.t. each intermediate value) is *reused* across all the parameters that feed
into it, instead of being recomputed. Backprop is exactly this: propagate the loss gradient backward
through the layers, reusing each layer's gradient to compute the ones before it.

---

## 3. Forward pass, then backward pass

Backprop has two phases, and the forward pass must come first because the backward pass needs the
values it computes:

- **Forward pass.** Compute and **cache** every layer's pre-activation $\mathbf{z}^{(\ell)}$ and
  activation $\mathbf{a}^{(\ell)}$, up to the loss ([07.01 §3](../01-neural-network-basics/)). These
  cached values are needed to evaluate the local derivatives on the way back.
- **Backward pass.** Starting from $\partial C/\partial \mathbf{a}^{(L)}$ at the output, apply the
  chain rule layer by layer *backward*, computing at each layer the gradient of the loss w.r.t. that
  layer's pre-activation (the "error" $\boldsymbol\delta^{(\ell)}$), and from it the gradients w.r.t.
  that layer's weights and biases.

The forward pass is inference; the backward pass is learning. Every deep-learning framework runs
exactly these two passes, and `from_scratch.py` implements both explicitly for an MLP and trains it.

---

## 4. The four backpropagation equations

For an MLP with layers $\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$,
$\mathbf{a}^{(\ell)} = \sigma(\mathbf{z}^{(\ell)})$, define the **error** at layer $\ell$ as
$\boldsymbol\delta^{(\ell)} = \partial C / \partial \mathbf{z}^{(\ell)}$ — the gradient of the loss
w.r.t. that layer's pre-activation. The whole algorithm is four equations (Nielsen's notation):

**(BP1) Output-layer error** — how wrong the final layer is, scaled by the output nonlinearity's slope:

$$
\boldsymbol\delta^{(L)} = \nabla_{\mathbf{a}} C \;\odot\; \sigma'\!\big(\mathbf{z}^{(L)}\big).
$$

**(BP2) Backpropagate the error** — push the next layer's error back through the weights and the local
slope:

$$
\boldsymbol\delta^{(\ell)} = \Big(\big(\mathbf{W}^{(\ell+1)}\big)^\top \boldsymbol\delta^{(\ell+1)}\Big)\;\odot\;\sigma'\!\big(\mathbf{z}^{(\ell)}\big).
$$

**(BP3) Gradient w.r.t. biases** — the error itself:

$$
\frac{\partial C}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol\delta^{(\ell)}.
$$

**(BP4) Gradient w.r.t. weights** — the error times the incoming activation:

$$
\frac{\partial C}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol\delta^{(\ell)}\big(\mathbf{a}^{(\ell-1)}\big)^\top.
$$

Read the structure: (BP1) starts the error at the output, (BP2) transports it backward one layer at a
time (the $(\mathbf{W}^{(\ell+1)})^\top$ is where "back" propagation gets its name — the *transpose* of
the forward weight sends gradients the opposite way), and (BP3)–(BP4) read off the parameter gradients
from the error and the cached activations. Two matrix multiplies per layer, backward. `from_scratch.py`
implements these verbatim and gradient-checks them.

---

## 5. Backprop as reverse-mode autodiff

Backprop is a special case of a general algorithm: **reverse-mode automatic differentiation**. Think of
the network as a computational graph ([07.01 §8](../01-neural-network-basics/)) of primitive
operations (matmul, add, activation, loss). Each operation knows two things: how to compute its output
(forward), and how to turn a gradient on its output into a gradient on its inputs (backward) — its
**vector-Jacobian product** (VJP). Reverse-mode autodiff:

1. runs the forward pass, recording the graph and caching values;
2. seeds the gradient of the loss w.r.t. itself as $1$;
3. traverses the graph **in reverse topological order**, at each node applying its VJP to convert the
   incoming (downstream) gradient into gradients on its inputs, accumulating gradients where a value
   feeds multiple places.

Backprop for an MLP *is* this, with the primitives being dense layers and activations. The power of the
autodiff view is generality: it works for *any* differentiable computation, not just MLPs — which is
why frameworks can differentiate arbitrary code (RNNs, attention, custom losses) with the same
machinery (§9). The four equations of §4 are just the VJPs of the linear layer and the activation,
composed.

---

## 6. Gradient checking

Because a bug in the backward pass produces *wrong gradients* that still "train" (just badly, or
subtly), you must **verify gradients numerically**. Gradient checking compares the analytic gradient
from backprop against a finite-difference estimate:

$$
\frac{\partial C}{\partial \theta_i} \approx \frac{C(\boldsymbol\theta + \epsilon\,\mathbf{e}_i) - C(\boldsymbol\theta - \epsilon\,\mathbf{e}_i)}{2\epsilon},
$$

the **central difference** (which has $O(\epsilon^2)$ error, far more accurate than the one-sided
version). With $\epsilon \approx 10^{-5}$ and float64, the analytic and numerical gradients should
agree to a relative error of about $10^{-7}$; a much larger discrepancy means a bug. Experiment 1
gradient-checks the from-scratch backprop and confirms agreement to $\sim 10^{-9}$, and Experiment 2
uses the verified gradients to actually train XOR. **Always gradient-check a hand-written backward
pass** — it is the single most valuable debugging tool in deep learning, and cheap to run on a small
network once.

---

## 7. Why reverse mode

Automatic differentiation comes in two modes, and the choice matters. For a function
$f: \mathbb{R}^n \to \mathbb{R}^m$:

- **Forward mode** propagates derivatives *input-to-output*, computing one column of the Jacobian per
  pass — efficient when there are **few inputs** ($n$ small).
- **Reverse mode** (backprop) propagates derivatives *output-to-input*, computing one row of the
  Jacobian per pass — efficient when there are **few outputs** ($m$ small).

Neural-network training has a **scalar loss** ($m = 1$) and **millions of parameters** ($n$ huge) —
exactly the regime where reverse mode wins overwhelmingly: **one** backward pass yields the gradient
w.r.t. *all* parameters, whereas forward mode would need $n$ passes. This asymmetry is the whole reason
backprop, not forward-mode differentiation, is the algorithm of deep learning. The cost is memory:
reverse mode must **store** the forward-pass activations for the backward pass (the source of the
memory pressure that gradient checkpointing later trades off).

---

## 8. Where vanishing and exploding gradients come from

Backprop's (BP2) reveals a fundamental training difficulty. Propagating the error back through $L$
layers multiplies it by $L$ factors of $(\mathbf{W}^{(\ell+1)})^\top$ and $L$ factors of
$\sigma'(\mathbf{z}^{(\ell)})$. This **product of many terms** either shrinks or grows exponentially in
depth:

- If the factors are consistently **< 1** (e.g. sigmoid/tanh saturate, so $\sigma' \ll 1$; or small
  weights), the gradient **vanishes** — early layers receive almost no signal and barely learn.
- If the factors are consistently **> 1** (large weights), the gradient **explodes** — updates blow up
  and training diverges.

Experiment 5 measures the gradient magnitude shrinking layer by layer in a deep sigmoid network — the
classic vanishing-gradient signature. This single observation motivates much of the rest of Part 7:
non-saturating activations (**ReLU**, [07.03](../03-activations/)), careful **initialization** that
keeps the factors near 1 ([07.05](../05-initialization/)), **normalization** that rescales
pre-activations ([07.07](../07-normalization/)), and architectural fixes (residual connections). All of
them are, at heart, ways to keep backprop's product of Jacobians from vanishing or exploding.

---

## 9. Autograd: how frameworks do it

You will almost never write backprop by hand in practice — frameworks do it for you via **autograd**.
When you run a forward computation in PyTorch/TensorFlow/JAX, the framework records the graph of
operations (a *tape*). Calling `.backward()` traverses that tape in reverse, applying each operation's
VJP (§5), and deposits the gradient of the loss into each parameter's `.grad`. You write only the
forward pass; the backward pass is derived automatically for *any* differentiable code.

Understanding backprop by hand is still essential, though — it is what lets you reason about vanishing
gradients (§8), debug `NaN`s and dead units, design custom layers, and understand memory costs. The
framework hides the mechanism, not the consequences. `from_scratch.py` verifies its hand-written
gradients against PyTorch's autograd, closing the loop: the four equations of §4 produce exactly what
the framework computes.

---

## 10. Common misconceptions

**"Backprop is a learning algorithm."**
Backprop only *computes gradients*. The *learning* is the optimizer's update using those gradients
([07.06](../06-optimizers/)). Backprop + gradient descent = training.

**"Backprop is just finite differences."**
No — it computes exact analytic gradients via the chain rule, in one backward pass, versus finite
differences' $O(P)$ forward passes and truncation error (§1, §6).

**"You need to derive gradients by hand for every network."**
Autograd derives them automatically for any differentiable computation (§9). You implement the forward
pass; the framework handles the backward.

**"The transpose in (BP2) is a coincidence."**
It is fundamental: the backward pass through a linear layer $\mathbf{W}$ is multiplication by
$\mathbf{W}^\top$ — the adjoint that sends gradients the opposite direction (§4–§5).

**"Vanishing gradients are a solved historical problem."**
They are *managed*, not eliminated — by ReLU, good init, normalization, and residual connections, all
of which exist precisely to control backprop's product of Jacobians (§8).

**"If the loss goes down, the gradients must be correct."**
A subtly wrong gradient can still decrease the loss (in a wrong direction that happens to help
sometimes). Always gradient-check (§6).

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — a full MLP with forward *and* backward passes (the four
  equations of §4) in NumPy, that actually **trains** by gradient descent. Verified two ways: by
  **finite-difference gradient checking** (analytic vs numerical to ~1e-9) and against **PyTorch
  autograd** (identical gradients). Five experiments: (1) gradient checking; (2) training XOR from
  scratch with the verified gradients; (3) backprop vs finite differences — identical gradients, and
  backprop's $O(1)$-pass speed vs $O(P)$ passes; (4) reverse mode giving all parameter gradients in one
  pass; (5) vanishing gradients in a deep sigmoid network.
- **[exercises.md](exercises.md)** — derive the four equations, implement backprop and gradient
  checking, reproduce every experiment.
- **[references.md](references.md)** — Rumelhart et al., Nielsen's chapter, Baydin et al. (autodiff
  survey).

**Next**: [07.03 — Activation Functions](../03-activations/) — which nonlinearity to use, and how each
shapes the gradients backprop propagates.
