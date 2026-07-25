# 09.01 — Recurrent Neural Networks

> **An RNN is a loop: one small network applied at every timestep, passing a hidden state to its future
> self.** That loop lets it process sequences of any length with a fixed number of parameters — and it
> is also its downfall. Unrolled, the loop is a very deep network whose gradient is a *product* of many
> Jacobians, so it vanishes or explodes exponentially with sequence length. This chapter derives the
> recurrence, its backpropagation-through-time, and the gradient pathology that motivates everything
> after it (LSTMs, then attention).

A feedforward net maps a fixed-size input to an output. Sequences — text, audio, time series — have
*variable length* and *order matters*. The RNN's answer: apply the same cell repeatedly, threading a
**hidden state** $h_t$ that summarizes everything seen so far.

## Table of contents

1. [The recurrence](#1-the-recurrence)
2. [Unrolling: an RNN is a deep, weight-shared net](#2-unrolling-an-rnn-is-a-deep-weight-shared-net)
3. [Backpropagation through time](#3-backpropagation-through-time)
4. [The vanishing/exploding gradient](#4-the-vanishingexploding-gradient)
5. [Gradient clipping](#5-gradient-clipping)
6. [Short vs long dependencies](#6-short-vs-long-dependencies)
7. [Architectures: many-to-one, seq2seq, bidirectional](#7-architectures-many-to-one-seq2seq-bidirectional)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The recurrence

At each timestep the cell combines the current input $x_t$ with the previous hidden state $h_{t-1}$:

$$
h_t = \tanh\!\big(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b\big), \qquad y_t = W_{hy}\,h_t + b_y.
$$

The **same** weights $W_{xh}, W_{hh}, b$ are used at every step — that weight sharing is what makes the
model handle arbitrary lengths and generalize across positions (the sequence analogue of a
convolution's weight sharing, [08.01](../../08-computer-vision/01-convolution/)). The hidden state is
the RNN's only memory. [`from_scratch.py`](from_scratch.py) implements this and matches
`torch.nn.RNN` to $7\times10^{-16}$ (Experiment 1).

## 2. Unrolling: an RNN is a deep, weight-shared net

To process a length-$T$ sequence, **unroll** the loop into $T$ copies of the cell, chained by the
hidden state:

$$
h_0 \to \boxed{\text{cell}} \to h_1 \to \boxed{\text{cell}} \to h_2 \to \cdots \to h_T.
$$

This is a depth-$T$ feedforward network in which **every layer shares the same weights**. Everything
about training deep nets ([Part 7](../../07-deep-learning/)) applies — including, acutely, the
gradient-flow problem (§4). The difference from a CNN's depth is that here depth = sequence length, so
a long sentence *is* a very deep net.

## 3. Backpropagation through time

Training is ordinary backprop on the unrolled graph, called **backpropagation through time (BPTT)**.
The gradient of the loss w.r.t. the shared weights **accumulates over all timesteps**, and the gradient
w.r.t. each hidden state has two contributions — from that step's output and from the future:

$$
\frac{\partial \mathcal{L}}{\partial h_t} = \underbrace{\frac{\partial \mathcal{L}_t}{\partial h_t}}_{\text{this step's output}} + \underbrace{W_{hh}^{\top}\,\big(\text{diag}(1-h_{t+1}^2)\big)\,\frac{\partial \mathcal{L}}{\partial h_{t+1}}}_{\text{from the future}} .
$$

Because $W_{hh}$ is reused, its gradient sums each step's contribution:
$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_t \frac{\partial \mathcal{L}}{\partial h_t}\frac{\partial h_t}{\partial W_{hh}}$. The from-scratch BPTT matches PyTorch autograd to $2\times10^{-15}$
(Experiment 1). In practice, long sequences use **truncated BPTT** — backprop only $k$ steps back — to
bound memory and compute.

## 4. The vanishing/exploding gradient

Here is the defining problem. Propagating the gradient from step $T$ back to step $0$ multiplies **one
Jacobian per step**:

$$
\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=1}^{T} \text{diag}\!\big(1-h_t^2\big)\,W_{hh}.
$$

A product of $T$ matrices behaves like the $T$-th power of its typical factor. If that factor's size is
below $1$, the product **vanishes** toward $0$; if above $1$, it **explodes**. Experiment 2 measures
$\lVert \partial\mathcal{L}/\partial h_0 \rVert$ as the sequence lengthens:

| Spectral radius of $W_{hh}$ | $T=5$ | $T=10$ | $T=25$ | $T=50$ |
|---|---:|---:|---:|---:|
| 0.5 (vanish) | 4.7e−2 | 7.3e−4 | 3.0e−9 | **1.2e−18** |
| 0.9 (vanish) | 7.5e−1 | 1.7e−1 | 1.1e−3 | 2.8e−7 |
| 1.0 (vanish) | 1.1 | 3.7e−1 | 1.0e−2 | 1.7e−5 |
| 1.5 (explode) | 3.3 | 3.7 | 1.8e+1 | **1.4e+1** |

At radius $0.5$ the gradient is $10^{-18}$ by $T=50$ — utterly gone. At $1.5$ it explodes.
**Crucially, even spectral radius $1.0$ vanishes**, because the $\tanh$ term $\text{diag}(1-h_t^2)$ has
entries $\le 1$ (§3, Experiment 3) — so the borderline is *below* 1, and vanishing is the default. A
zero gradient at step 0 means **the early inputs cannot influence the loss**: the network cannot learn
long-range dependencies.

## 5. Gradient clipping

Exploding gradients are the easier half to fix: **clip** the gradient to a maximum norm before the
update,

$$
g \leftarrow g \cdot \min\!\left(1, \frac{\tau}{\lVert g \rVert}\right),
$$

so no single step can blow up the weights. Experiment 4 shows a raw gradient reaching $4\times10^{15}$
capped cleanly at $\tau=5$. Clipping is standard practice for RNNs. But it does **nothing for
vanishing** gradients — you cannot un-shrink a zero. That half needs a better *architecture*: gated
recurrences (LSTM/GRU, [09.02](../02-lstm-gru/)) that give the gradient a near-identity path through
time, exactly as residual connections did for depth ([08.02 §4](../../08-computer-vision/02-cnn-architectures/)).

## 6. Short vs long dependencies

The vanishing gradient is not abstract — it caps how far back an RNN can actually learn. Experiment 5
trains an RNN to recall a bit planted at step $0$, through $T$ steps of interfering noise:

| Lag $T$ | Train accuracy | |
|:--:|:--:|:--|
| 3 | 1.00 | learned |
| 7 | 1.00 | learned |
| 15 | 0.67 | failing |
| 25 | 0.51 | **failed (≈ chance)** |
| 40 | 0.58 | **failed** |

Short lags are learned perfectly; by $T=25$ the accuracy is at chance. The gradient that should teach
step 0 to store the bit has vanished before it propagates back, so the early weights never learn to
protect it. **Plain RNNs have a short effective memory** — empirically a handful to a couple dozen
steps. Extending it is what the next chapter is about.

## 7. Architectures: many-to-one, seq2seq, bidirectional

The same cell wires up for different tasks:

- **Many-to-one** — read the whole sequence, output once (sentiment classification): use $h_T$.
- **Many-to-many (aligned)** — output per step (part-of-speech tagging): use every $y_t$.
- **Seq2seq (encoder–decoder)** — read one sequence, generate another (translation): an encoder RNN
  compresses the input into a context vector that seeds a decoder RNN ([09.03](../03-seq2seq-and-attention/)).
- **Bidirectional** — run one RNN forward and one backward and concatenate, so each position sees both
  past and future context (only for non-streaming tasks).
- **Stacked/deep** — feed one RNN's outputs into another for more capacity.

## 8. Common misconceptions

- **"RNNs remember everything."** Their memory is a fixed-size vector and, in practice, short — the
  vanishing gradient (§4, §6) limits learnable range to a few dozen steps.
- **"Exploding and vanishing are the same bug."** Opposite ends of the same product; clipping fixes
  explosion, only architecture fixes vanishing (§5).
- **"Use ReLU to avoid vanishing."** ReLU/identity recurrences avoid the $\tanh$ shrink but explode more
  readily and can produce unbounded states; they need careful initialization (IRNN).
- **"BPTT is a different algorithm."** It is plain backprop on the unrolled graph — the only twist is
  summing the shared-weight gradient over steps (§3).
- **"RNNs are obsolete."** Transformers ([Part 11](../../11-transformers-llms/)) dominate NLP, but RNNs
  remain useful for streaming, low-latency, and small-footprint settings, and the gradient-flow lessons
  here underpin all of them.

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — an RNN cell with forward and BPTT, verified against
  `torch.nn.RNN` and autograd to machine precision. Five experiments: (1) forward+BPTT match PyTorch;
  (2) gradients vanish/explode exponentially with sequence length; (3) $\tanh$ saturation drives the
  vanishing; (4) gradient clipping caps explosion; (5) a trained RNN learns short but not long
  dependencies.
- **[exercises.md](exercises.md)** — derive BPTT, analyze the Jacobian product, implement clipping and
  truncated BPTT.
- **[references.md](references.md)** — the foundational RNN and gradient-flow papers.

## Where this leads

- **LSTM & GRU — gates that fix the vanishing gradient** → [09.02](../02-lstm-gru/)
- **Seq2seq and attention — and why attention replaced recurrence** → [09.03](../03-seq2seq-and-attention/)
- **The same gradient-flow fix for depth (residuals)** → [08.02 §4](../../08-computer-vision/02-cnn-architectures/)
- **Backprop, the general algorithm BPTT specializes** → [07.02](../../07-deep-learning/02-backpropagation/)
- **Transformers — sequence modeling without recurrence** → [Part 11](../../11-transformers-llms/)
