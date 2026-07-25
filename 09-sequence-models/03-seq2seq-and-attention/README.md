# 09.03 — Seq2seq & Attention

> **Attention is a differentiable lookup table.** Instead of cramming an entire input sequence into one
> fixed vector, the decoder issues a *query* at each step and retrieves a relevance-weighted blend of
> *all* the encoder's states. That one idea removed the seq2seq bottleneck, gave translation its first
> interpretable alignments — and, generalized to let a sequence attend to itself, became the
> transformer. This chapter builds attention from the scoring function up and measures why it was the
> pivotal idea in modern sequence modeling.

A sequence-to-sequence (seq2seq) model maps an input sequence to an output sequence of possibly
different length: translation, summarization, speech-to-text. The encoder–decoder architecture
(Sutskever et al., 2014) reads the input with one RNN and generates the output with another — but its
single hand-off vector is a bottleneck that attention (Bahdanau et al., 2015) dissolves.

## Table of contents

1. [The encoder–decoder](#1-the-encoderdecoder)
2. [The bottleneck](#2-the-bottleneck)
3. [Attention: a differentiable lookup](#3-attention-a-differentiable-lookup)
4. [Attention is alignment](#4-attention-is-alignment)
5. [Score scaling and soft vs hard attention](#5-score-scaling-and-soft-vs-hard-attention)
6. [Decoding: greedy vs beam search](#6-decoding-greedy-vs-beam-search)
7. [The bridge to transformers](#7-the-bridge-to-transformers)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The encoder–decoder

The classic seq2seq:

1. An **encoder** RNN ([09.02](../02-lstm-gru/)) reads the input $x_1, \dots, x_{T}$ and produces
   hidden states $s_1, \dots, s_T$.
2. The final state $s_T$ (the **context** $c$) is handed to a **decoder** RNN.
3. The decoder generates the output autoregressively — each step conditions on $c$ and the previously
   generated token — until an end-of-sequence token.

It works for short sequences and was the first neural machine translation system to rival statistical
MT. Its flaw is step 2.

## 2. The bottleneck

The entire input must be squeezed into the single fixed-size vector $c$. A $d$-dimensional vector holds
only $d$ numbers, no matter how long the input — so past a point it *must* discard information.
Experiment 3 makes this exact: the best-possible (PCA-optimal) reconstruction of an $L$-state sequence
from one $d=16$ context vector loses more and more as $L$ grows:

| Input length $L$ | Info (numbers) | Fixed context — variance lost | Attention |
|:--:|:--:|:--:|:--:|
| 2 | 8 | 0.000 | 0.000 |
| 4 | 16 | 0.000 | 0.000 |
| 8 | 32 | **0.500** | 0.000 |
| 16 | 64 | 0.750 | 0.000 |
| 32 | 128 | **0.874** | 0.000 |

While the input fits in $d$ numbers, the context is lossless; once it exceeds $d$, half the information
is gone by $L=8$ and 87% by $L=32$. This is *why* plain seq2seq degrades on long sentences. Attention
keeps **all** encoder states and reads them on demand, so it has no fixed bottleneck (loss 0).

## 3. Attention: a differentiable lookup

Attention replaces the single context with a **per-decoder-step** context. At decoder step $i$, with a
query $q_i$ (the decoder's state) and the encoder's keys/values $\{k_j, v_j\}$:

$$
e_{ij} = \text{score}(q_i, k_j), \qquad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{j'} \exp(e_{ij'})}, \qquad c_i = \sum_j \alpha_{ij}\, v_j .
$$

The weights $\alpha_{ij}$ **sum to 1** — a soft selection over inputs — and $c_i$ is the retrieved
blend. Two scoring functions dominate (Experiment 1 confirms both give valid distributions):

| Name | Score | Notes |
|---|---|---|
| **Additive (Bahdanau)** | $v^\top \tanh(W_q q_i + W_k k_j)$ | a tiny MLP; the original |
| **Dot / multiplicative (Luong)** | $q_i^\top k_j$ (optionally $q_i^\top W k_j$) | cheap; what transformers use |

The whole operation is differentiable, so the model **learns what to attend to** end-to-end. This is
the query–key–value pattern that [self-attention](../../11-transformers-llms/01-attention/) generalizes.

## 4. Attention is alignment

Because $\alpha_{ij}$ says how much output step $i$ draws on input position $j$, the weight matrix is a
soft **alignment**. Experiment 2 uses content-based queries on a 6-symbol sequence:

- **Copy** (output $i$ wants input $i$): alignment argmax $= [0,1,2,3,4,5]$ — a **diagonal**.
- **Reverse** (output $i$ wants input $L{-}1{-}i$): argmax $= [5,4,3,2,1,0]$ — an **anti-diagonal**.

with 99.6% of the weight on the intended position. This is **content-based addressing**: the decoder
forms a query and retrieves the encoder states whose keys match. On real translation these alignments
are interpretable — they recover word correspondences across languages — which is how attention was
first validated and why it is a built-in explainability tool.

## 5. Score scaling and soft vs hard attention

The *magnitude* of the scores controls how peaked the attention is. Dot products of $d$-dimensional
vectors grow like $\sqrt{d}$, so in high dimensions raw scores get large and the softmax collapses
toward one-hot — a near-**hard** selection with vanishing gradients. Experiment 4 ($d=64$, raw score
std $\approx 5.6$):

| Scaling | Max attention weight | Entropy (bits, max = 3) |
|---|:--:|:--:|
| $\times\sqrt{d}$ (huge) | 1.000 | 0.000 |
| $\times 1$ (raw) | 0.906 | 0.453 |
| $/\sqrt{d}$ (used) | 0.328 | 2.614 |
| $/d$ (tiny) | 0.145 | 2.994 |

Dividing by $\sqrt{d}$ keeps the scores $O(1)$, the attention **soft** (entropy 2.6 of 3), and the
gradients healthy. This is exactly the $1/\sqrt{d}$ in the transformer's **scaled dot-product
attention** ([11.01](../../11-transformers-llms/01-attention/)) — not a detail but a fix for a real
gradient problem. (Hard attention — sampling one position — exists but is non-differentiable and
trained with REINFORCE.)

## 6. Decoding: greedy vs beam search

Generation is a search for the most probable output sequence. **Greedy** decoding takes the
highest-probability token at each step — fast, but it can commit early to a token that leads nowhere.
**Beam search** keeps the top-$k$ partial sequences ("beams") at every step, expanding and re-ranking
them, so it can recover from a bad first token. Experiment 5 builds a 3-step model where greedy is
trapped:

| Decoder | Output | Probability |
|---|:--:|:--:|
| greedy | "ACA" | 0.170 |
| **beam search** (width 3) | **"BCA"** | **0.450** |

Greedy takes the locally-best first token (A, 0.5) and lands in a mediocre region; beam search retains
the B branch and finds the far more probable "BCA". Beam search approximates the maximum-probability
sequence and is standard for translation and any seq2seq generation, though very large beams can hurt
(favoring short or generic outputs), so widths of 4–10 are typical.

## 7. The bridge to transformers

Attention began as an *add-on* to a recurrent encoder–decoder. Two observations turned it into a
replacement:

1. **The RNN is barely needed.** If attention can read any input position directly, the sequential
   recurrence that made RNNs slow (and forgetful) is doing little.
2. **A sequence can attend to itself.** Apply the same query–key–value mechanism *within* one sequence
   — **self-attention** — and you get contextual representations with no recurrence at all, fully
   parallel across positions.

"Attention Is All You Need" (2017) took exactly this step: drop the RNN, keep the attention, add
positional encodings ([08.05 §4](../../08-computer-vision/05-vision-transformers/)) since attention is
order-agnostic ([08.05 §3](../../08-computer-vision/05-vision-transformers/)). Everything in this
chapter — scoring, the softmax weights, $1/\sqrt{d}$, the alignment view — is a direct ancestor of the
transformer ([Part 11](../../11-transformers-llms/)).

## 8. Common misconceptions

- **"Attention is a neural network layer with weights."** The core operation has *no learned weights*
  (dot-product) — it is a weighted average whose weights are computed from the data. The learning is in
  producing good queries/keys/values.
- **"The context vector is fine for long sequences."** It is a hard information bottleneck (§2); that is
  the whole reason attention exists.
- **"Beam search finds the optimal sequence."** It is a heuristic; it can miss the true maximum and can
  bias toward short/generic outputs (§6).
- **"Additive and dot attention are very different."** They compute alignment differently but serve the
  same role; dot-product won for speed (§3).
- **"The $1/\sqrt{d}$ is a minor detail."** Without it, high-dimensional scores saturate the softmax and
  kill gradients (§5).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — attention and decoding in NumPy. Five experiments:
  (1) dot and additive scoring produce valid distributions; (2) attention is content-based alignment
  (diagonal for copy, anti-diagonal for reverse); (3) the fixed-context bottleneck loses up to 87% of
  the input while attention loses none; (4) score scaling controls attention entropy — why transformers
  divide by $\sqrt{d}$; (5) beam search finds a 2.6× more probable sequence than greedy.
- **[exercises.md](exercises.md)** — derive the attention weights, implement both scorers and beam
  search, analyze the bottleneck.
- **[references.md](references.md)** — Sutskever seq2seq, Bahdanau/Luong attention, the transformer.

## Where this leads

- **Self-attention and the full transformer** → [11.01](../../11-transformers-llms/01-attention/), [11.02](../../11-transformers-llms/02-transformer-architecture/)
- **The RNNs attention was bolted onto** → [09.01](../01-rnn/), [09.02](../02-lstm-gru/)
- **Positional encodings (attention is order-agnostic)** → [08.05](../../08-computer-vision/05-vision-transformers/)
- **Decoding strategies for LLMs (sampling, top-k, nucleus)** → [11.06](../../11-transformers-llms/06-decoding-generation/)
- **NLP tasks these models solve** → [Part 10](../../10-nlp/)
