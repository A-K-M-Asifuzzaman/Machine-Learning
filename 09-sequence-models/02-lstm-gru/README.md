# 09.02 — LSTM & GRU

> **The fix for the vanishing gradient is a road the gradient can travel without shrinking.** An LSTM
> adds a *cell state* whose update is nearly linear — $c_t = f_t\,c_{t-1} + i_t\,g_t$ — so the gradient
> from step $T$ back to step $0$ is a product of **forget gates**, not a product of weight-matrix
> Jacobians. Keep the forget gates near 1 and the gradient survives. This is the same idea as a
> residual connection, applied through time, and it is what let recurrent nets learn long-range
> dependencies for the first time.

The plain RNN ([09.01](../01-rnn/)) forgets after a few dozen steps because
$\partial h_T/\partial h_0 = \prod_t \text{diag}(1-h_t^2)\,W_{hh}$ vanishes. The LSTM (Hochreiter &
Schmidhuber, 1997) and GRU (Cho et al., 2014) redesign the cell so that product can stay near 1.

## Table of contents

1. [The idea: a protected memory channel](#1-the-idea-a-protected-memory-channel)
2. [The LSTM cell](#2-the-lstm-cell)
3. [The GRU cell](#3-the-gru-cell)
4. [Why gates fix the gradient](#4-why-gates-fix-the-gradient)
5. [Long dependencies, learned](#5-long-dependencies-learned)
6. [The forget gate is the memory dial](#6-the-forget-gate-is-the-memory-dial)
7. [LSTM vs GRU, and practical notes](#7-lstm-vs-gru-and-practical-notes)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The idea: a protected memory channel

A plain RNN overwrites its entire state every step — information must survive by passing *through* the
nonlinearity and weight matrix again and again, and it doesn't. The gated cells add a **separate memory
channel** (the cell state $c_t$, or the GRU's directly-carried $h_t$) that is edited by **addition and
element-wise gates** rather than a full matrix multiply. Gates are sigmoids in $[0,1]$ that decide, per
component, how much to **keep**, **write**, and **read**. The result: memory can persist untouched for
many steps, and so can the gradient.

## 2. The LSTM cell

The LSTM maintains a hidden state $h_t$ *and* a cell state $c_t$, controlled by three gates (input
$i$, forget $f$, output $o$) plus a candidate $g$:

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) & \text{(input: what to write)}\\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) & \text{(forget: what to keep)}\\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g) & \text{(candidate content)}\\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) & \text{(output: what to expose)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t & \text{(update the cell)}\\
h_t &= o_t \odot \tanh(c_t) & \text{(read the cell)}
\end{aligned}
$$

The one line that matters is $c_t = f_t \odot c_{t-1} + i_t \odot g_t$: the old memory is **kept
(scaled by $f$) and added to**, not transformed. [`from_scratch.py`](from_scratch.py) implements the
forward and the full hand-derived backward, matching `torch.nn.LSTM` and its autograd to
$\sim 10^{-16}$ (Experiment 1).

## 3. The GRU cell

The GRU merges the cell and hidden state and uses only two gates — **update** $z$ and **reset** $r$:

$$
\begin{aligned}
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) & \text{(reset: how much past to use)}\\
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) & \text{(update: keep vs write)}\\
n_t &= \tanh(W_n x_t + r_t \odot (U_n h_{t-1} + b_n)) & \text{(candidate)}\\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1} & \text{(interpolate)}
\end{aligned}
$$

The update gate $z$ **interpolates** between keeping the old state and writing the new candidate — the
same "keep + add" structure as the LSTM's cell, in one gate. The GRU has ~25% fewer parameters (two
gates instead of three, no separate cell/output) and usually comparable accuracy. It matches
`torch.nn.GRU` to $\sim 10^{-17}$ (Experiment 2).

## 4. Why gates fix the gradient

The whole point. Differentiate the cell update:

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t \quad(\text{element-wise}), \qquad\Longrightarrow\qquad \frac{\partial c_T}{\partial c_0} = \prod_{t=1}^{T} f_t .
$$

The long-range gradient is a product of **forget gates** — a *diagonal*, data-dependent factor — not
the plain RNN's product of full Jacobians with their $\tanh$ shrink. If the forget gates stay near 1,
the product stays near 1: a **constant error carousel**. Experiment 3 measures the gradient back to
step 0:

| $T$ | Plain RNN | LSTM ($f \approx 1$) |
|:--:|---:|---:|
| 10 | 1.7e−1 | 2.7 |
| 25 | 1.1e−3 | 1.25 |
| 50 | 2.8e−7 | 3.5e−1 |
| 100 | **1.0e−11** | **2.8e−2** |

At 100 steps the RNN's gradient has vanished to $10^{-11}$ while the LSTM's is still $\sim 10^{-2}$ —
alive and trainable. This is structurally identical to a residual connection's $I + F'$ Jacobian
([08.02 §4](../../08-computer-vision/02-cnn-architectures/)): an additive, near-identity path that the
gradient rides through many steps.

## 5. Long dependencies, learned

Does the preserved gradient actually let the model *learn* long dependencies? Experiment 4 runs the
exact recall-a-bit task from [09.01](../01-rnn/) — remember a bit planted at step 0 through interfering
noise — on an LSTM:

| Lag $T$ | Plain RNN (09.01) | **LSTM** |
|:--:|:--:|:--:|
| 7 | 1.00 | 1.00 |
| 15 | 0.67 | **1.00** |
| 25 | 0.51 (chance) | **0.85** |
| 40 | 0.58 (chance) | **1.00** |

Where the plain RNN collapsed to chance past ~15 steps, the LSTM recalls the bit at 40 steps. The
gradient reaches step 0, so the cell learns to hold the bit and the forget gate learns to stay open on
that unit. Gated recurrences turned an unsolvable long-range credit-assignment problem into a solvable
one — the reason LSTMs powered a decade of translation, speech, and captioning before transformers.
(Trained with Adam — LSTMs optimize poorly under plain SGD.)

## 6. The forget gate is the memory dial

The forget gate directly sets the memory time-constant. Experiment 5 varies its bias and reads the
cell-state gradient after 50 steps:

| Forget bias | Mean forget gate $f$ | $\lVert$grad$\rVert$ at step 0 |
|:--:|:--:|:--:|
| −2.0 | 0.119 | 6.5e−47 |
| 0.0 | 0.500 | 8.9e−16 |
| 1.0 | 0.731 | 1.6e−7 |
| 3.0 | 0.953 | **8.8e−2** |

A low forget gate erases memory almost immediately ($f^{50}$ with $f=0.12$ is $10^{-47}$); a high one
preserves it. This is exactly why **LSTMs are initialized with a positive forget-gate bias** (~1):
start by remembering, and let training decide what to forget. The gate makes the memory length a
**learnable quantity** instead of a fixed property of $W_{hh}$ (Jozefowicz et al., 2015).

## 7. LSTM vs GRU, and practical notes

- **Choice.** GRU is simpler and faster (fewer parameters); LSTM is slightly more expressive. On most
  tasks they are within noise of each other — try both.
- **Forget-bias init.** Set it to 1–2. This one line meaningfully improves long-range learning.
- **Gradient clipping still helps.** Gates fix *vanishing*; clipping ([09.01 §5](../01-rnn/)) still
  guards against occasional *exploding* gradients.
- **Stacking + bidirectional.** Both cells stack into deep RNNs and run bidirectionally, as in §7 of
  [09.01](../01-rnn/).
- **Peephole / variants** exist (connect $c_t$ to the gates) but rarely matter.
- **They still process sequentially.** No parallelism across time — the limitation transformers remove
  ([Part 11](../../11-transformers-llms/)).

## 8. Common misconceptions

- **"LSTMs never forget."** They forget *by design* — the forget gate is trained to drop irrelevant
  information; that is a feature, not a bug (§6).
- **"The gates are the memory."** The **cell state** is the memory; the gates only regulate reading,
  writing, and keeping (§2).
- **"GRU is strictly worse because it has fewer gates."** Usually comparable, sometimes better, and
  cheaper (§7).
- **"Gates eliminate exploding gradients too."** They fix vanishing; exploding can still happen — keep
  clipping (§7).
- **"LSTMs solved long-range memory."** They *extended* it (dozens→hundreds of steps) but still
  degrade over very long ranges and cannot parallelize — which is why attention replaced them (§7,
  [09.03](../03-seq2seq-and-attention/)).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — LSTM (forward + hand-derived backward) and GRU (forward),
  verified against `torch.nn.LSTM`/`GRU` and autograd to machine precision. Five experiments:
  (1) LSTM forward+backward match PyTorch; (2) GRU forward matches; (3) the cell-state gradient is a
  product of forget gates that survives 100 steps where the RNN's vanishes; (4) an LSTM learns the
  long-lag task the RNN failed; (5) the forget-gate bias as the memory dial.
- **[exercises.md](exercises.md)** — derive the LSTM/GRU backward, analyze the forget-gate gradient,
  implement both cells.
- **[references.md](references.md)** — the LSTM, GRU, and forget-gate papers.

## Where this leads

- **The vanishing-gradient problem these cells solve** → [09.01](../01-rnn/)
- **Seq2seq, attention, and why attention replaced recurrence** → [09.03](../03-seq2seq-and-attention/)
- **The same additive-path idea for depth (residuals)** → [08.02 §4](../../08-computer-vision/02-cnn-architectures/)
- **Transformers — sequence modeling without recurrence** → [Part 11](../../11-transformers-llms/)
