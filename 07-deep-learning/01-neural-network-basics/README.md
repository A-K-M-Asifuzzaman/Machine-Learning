# 07.01 — Neural Network Basics

> **Prerequisites**: [03.09](../../03-supervised-learning/09-perceptron/) (the perceptron and the XOR
> problem), [00.01](../../00-mathematical-foundations/01-linear-algebra/) (matrix multiplication),
> [00.02](../../00-mathematical-foundations/02-calculus-and-optimization/) (composition of functions).
> **You will be able to**: describe an MLP and its forward pass, explain *why* nonlinearity is
> essential, state the universal approximation theorem and its catch, and see how hidden layers learn
> a representation that makes a hard problem easy.

---

## Table of contents

1. [From perceptron to multilayer network](#1-from-perceptron-to-multilayer-network)
2. [The multilayer perceptron](#2-the-multilayer-perceptron)
3. [The forward pass](#3-the-forward-pass)
4. [Why nonlinearity is essential](#4-why-nonlinearity-is-essential)
5. [Hidden layers learn a representation](#5-hidden-layers-learn-a-representation)
6. [The universal approximation theorem](#6-the-universal-approximation-theorem)
7. [Depth vs width](#7-depth-vs-width)
8. [The computational graph](#8-the-computational-graph)
9. [Capacity, and what comes next](#9-capacity-and-what-comes-next)
10. [Common misconceptions](#10-common-misconceptions)

---

## 1. From perceptron to multilayer network

The perceptron ([03.09](../../03-supervised-learning/09-perceptron/)) is a single linear unit: it
computes $\mathrm{sign}(\mathbf{w}^\top\mathbf{x} + b)$ and can only separate linearly separable
classes. Its famous failure is **XOR** — no straight line separates $\lbrace(0,0),(1,1)\rbrace$ from
$\lbrace(0,1),(1,0)\rbrace$ — and that failure stalled neural-network research for a decade.

The fix is to **stack** units into layers: feed the outputs of one layer of units as inputs to the
next. The intermediate ("hidden") layer computes new features — combinations of the inputs — and in
that new feature space the problem becomes linearly separable. A network with one or more hidden
layers of nonlinear units is a **multilayer perceptron (MLP)**, and it can represent XOR, and far more:
with enough units it can approximate *any* continuous function (§6). The whole of deep learning is
built on this one move — compose many simple nonlinear units — and this chapter is its foundation.

---

## 2. The multilayer perceptron

An MLP is a sequence of **layers**. Each layer $\ell$ takes the previous layer's output
$\mathbf{a}^{(\ell-1)}$, applies a **linear transformation** (a weight matrix $\mathbf{W}^{(\ell)}$
and bias $\mathbf{b}^{(\ell)}$), then a **nonlinear activation** $\sigma$ elementwise:

$$
\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \qquad \mathbf{a}^{(\ell)} = \sigma\big(\mathbf{z}^{(\ell)}\big).
$$

- The input is $\mathbf{a}^{(0)} = \mathbf{x}$; the final layer's output $\mathbf{a}^{(L)}$ is the
  prediction.
- $\mathbf{z}^{(\ell)}$ is the **pre-activation** (the linear part); $\mathbf{a}^{(\ell)}$ the
  **activation** (after the nonlinearity).
- **Hidden layers** are the intermediate ones; their units learn features. The **output layer** uses
  an activation matched to the task (softmax for classification, identity for regression, §[07.04](../04-loss-functions/)).

The layer is called *fully connected* (or *dense*) because every unit connects to every input. An MLP
is defined by its architecture (number of layers, units per layer, activation) and its parameters (the
weights and biases), which are learned by minimizing a loss via gradient descent — the gradients coming
from **backpropagation** ([07.02](../02-backpropagation/)).

---

## 3. The forward pass

Computing the network's output for an input is the **forward pass**: apply the layers in order. In
matrix form, for a batch $\mathbf{X}$ of inputs (one per row), each layer is a matrix multiply plus a
bias broadcast plus an elementwise activation. This is the entire inference computation — a stack of
`activation(X @ W + b)` operations — and it is why neural networks map so cleanly onto GPUs (dense
matrix multiplies).

`from_scratch.py` implements this forward pass and verifies it exactly against PyTorch: the same
weights produce the same outputs to machine precision. Everything else in deep learning — training,
regularization, fancy architectures — is built on top of this simple layered forward computation.

---

## 4. Why nonlinearity is essential

The activation function is not a detail; it is what makes depth meaningful. **Without it, a deep
network collapses to a single linear layer.** If every layer were just $\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}$
with no activation, then composing two layers gives

$$
\mathbf{W}^{(2)}(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)} = \underbrace{(\mathbf{W}^{(2)}\mathbf{W}^{(1)})}_{\text{one matrix}}\mathbf{x} + \underbrace{(\mathbf{W}^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)})}_{\text{one bias}},
$$

which is *exactly* a single linear layer. No matter how many linear layers you stack, the result is
one linear function — it can only draw straight boundaries and cannot solve XOR. The **nonlinear
activation between layers is what lets the composition express nonlinear functions.** Experiment 1
verifies that a stack of linear layers equals a single linear map (so it fails XOR), and Experiment 2
shows a nonlinearity fixing it. This is the single most important reason activation functions exist —
their other properties (gradient behavior, saturation) are the subject of [07.03](../03-activations/).

---

## 5. Hidden layers learn a representation

Here is the deep idea, made concrete on XOR. A hidden layer transforms the input into a new feature
space, and the network's job is to find a transformation that makes the problem **linearly separable**
in that space — so the output layer (a linear classifier) can finish.

For XOR, a 2-unit hidden layer can compute features like "$x_1$ OR $x_2$" and "$x_1$ AND $x_2$"; in the
plane spanned by those two features, the four XOR points become linearly separable, and a single output
unit solves it. Experiment 3 hand-constructs exactly such a network — no training needed — and shows
the hidden representation making XOR separable, the same "map to a space where the problem is easy"
idea behind kernels ([03.07](../../03-supervised-learning/07-svm/)) and spectral embeddings
([04.05](../../04-unsupervised-learning/05-spectral-clustering/)).

This is **representation learning**: the hidden layers are not hand-designed features but *learned*
ones, and their composition builds increasingly abstract representations (edges → shapes → objects in
vision). It is why deep networks are so powerful — they learn the features, not just the classifier on
top of them.

---

## 6. The universal approximation theorem

How expressive is an MLP? The **universal approximation theorem** (Cybenko 1989; Hornik 1991) answers:

> A feedforward network with a **single hidden layer** containing **enough units** and a non-polynomial
> activation can approximate **any continuous function** on a compact domain to arbitrary accuracy.

So MLPs are, in principle, universal function approximators — one hidden layer is already enough to
represent anything continuous. Experiment 4 demonstrates this by fitting a wiggly target with a
single-hidden-layer network and showing the approximation error shrink as the layer widens.

**The catch** is *how many* units. The theorem guarantees existence, not efficiency: a shallow network
may need an **exponentially large** hidden layer to approximate a function that a **deep** network
represents with far fewer units (§7). And it says nothing about whether gradient descent will *find*
the right weights. Universality is a statement about representational capacity, not about learnability
or efficiency — which is exactly why, despite one layer being "enough," we use *deep* networks.

---

## 7. Depth vs width

If one wide hidden layer is universal, why go deep? Because **depth is exponentially more efficient**
for the compositional, hierarchical functions that real data exhibits. Many functions that a deep
network represents with $O(\text{depth})$ units require a shallow network to use $O(2^{\text{depth}})$
units to match.

The canonical example is a highly oscillatory function (a "sawtooth"): each layer of a deep network can
*fold* the input, doubling the number of oscillations, so $k$ layers produce $2^k$ oscillations with
$O(k)$ units — while a shallow network needs one unit per oscillation, i.e. $2^k$ units. Experiment 5
fits such an oscillatory target and shows a shallow (random-feature) network needing many more units
than the depth-based construction to represent the same number of folds.

The intuition: real-world functions are **compositions** — pixels compose into edges compose into parts
compose into objects — and a deep network mirrors that compositional structure, reusing lower-level
features across the higher levels. Depth buys parameter efficiency for exactly the kind of structured
functions that matter. This is the theoretical case for "deep" learning over "wide" learning.

---

## 8. The computational graph

An MLP's forward pass is a **computational graph**: a directed graph where nodes are operations
(matrix multiply, bias add, activation) and edges carry values. The input flows forward through the
graph to produce the output and the loss.

This graph view is not just bookkeeping — it is the foundation of **backpropagation**
([07.02](../02-backpropagation/)). Because each node is a simple differentiable operation, the chain
rule can be applied node by node, *backward* through the graph, to compute the gradient of the loss
with respect to every weight efficiently. Modern frameworks (PyTorch, TensorFlow, JAX) build this graph
automatically (autograd) and traverse it to get gradients. Understanding the forward pass *as a graph*
is what makes the backward pass make sense.

---

## 9. Capacity, and what comes next

An MLP's capacity grows with its width and depth — more units, more parameters, more functions it can
represent. That capacity is a double-edged sword: enough to fit anything (§6), including noise
(overfitting, [05.01](../../05-model-evaluation/01-bias-variance-and-theory/)). The rest of Part 7 is
about making this capacity *learnable and generalizable*:

- **[07.02](../02-backpropagation/)** — how to compute the gradients that train the network.
- **[07.03](../03-activations/)** — which nonlinearity, and its effect on gradients.
- **[07.04](../04-loss-functions/)** — the loss for each task and its output activation.
- **[07.05](../05-initialization/)** — how to initialize weights so training even starts.
- **[07.06](../06-optimizers/)** — gradient descent and its accelerated variants.
- **[07.07](../07-normalization/)**–**[07.08](../08-regularization/)** — stabilizing and regularizing.
- **[07.09](../09-training-dynamics/)** — diagnosing and debugging training.

This chapter is the skeleton: layers, a forward pass, nonlinearity, and the promise of universality.
Everything else makes it work.

---

## 10. Common misconceptions

**"More layers always means more power, even without activations."**
Without nonlinear activations, any number of linear layers collapses to a single linear map (§4). The
activation is what gives depth its expressive power.

**"A single hidden layer is enough, so depth is pointless."**
One layer is universal *in principle* (§6) but may need exponentially many units; depth is
exponentially more efficient for compositional functions (§7). Universality ≠ efficiency ≠
learnability.

**"Universal approximation means a network can learn any function."**
It means a network can *represent* any continuous function — it says nothing about whether gradient
descent will *find* those weights, or how much data you need (§6).

**"Hidden units are hand-designed features."**
They are *learned* features — the network discovers a representation that makes the task easy (§5).
That learning is the point.

**"The forward pass is where the learning happens."**
The forward pass just computes outputs; learning happens in the *backward* pass (gradients) and the
optimizer update ([07.02](../02-backpropagation/), [07.06](../06-optimizers/)).

**"Neural networks are a black box with no structure."**
They are a precise, differentiable computational graph of simple operations (§8) — that structure is
exactly what makes them trainable and analyzable.

---

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — an MLP forward pass (dense layers + activations) in NumPy,
  verified against PyTorch to machine precision. Five experiments: (1) a stack of linear layers equals
  a single linear map (and so fails XOR); (2) adding a nonlinearity lets it solve XOR; (3) a
  hand-constructed 2-2-1 network computing XOR, with the hidden layer making it linearly separable;
  (4) universal approximation — a single hidden layer fitting a wiggly target with error shrinking as
  it widens; (5) depth vs width — an oscillatory function that a shallow network needs far more units
  to represent.
- **[exercises.md](exercises.md)** — prove the linear-collapse and hand-derive XOR weights, implement
  the forward pass, reproduce every experiment.
- **[references.md](references.md)** — Rumelhart et al., Cybenko/Hornik (universal approximation),
  Goodfellow et al. (Deep Learning).

**Next**: [07.02 — Backpropagation](../02-backpropagation/) — the chain rule on the computational
graph, computed from scratch and verified by gradient checking.
