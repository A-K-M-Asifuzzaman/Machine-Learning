# 11.01 — The Transformer

> **Attention is all you need — literally.** Strip out the recurrence that made RNNs slow and
> forgetful, keep only the attention that let a decoder read any input position, apply it *within* a
> sequence, and add a position-wise MLP. That is the transformer: two sub-layers repeated, fully
> parallel over the sequence, and it is the architecture behind every large language model, every
> modern vision model, and most of the last decade's breakthroughs. This chapter derives it and builds
> every piece from scratch, verified against PyTorch to machine precision.

Attention began as an add-on to a recurrent encoder–decoder ([09.03](../../09-sequence-models/03-seq2seq-and-attention/)).
Vaswani et al. (2017) made the radical move: **drop the RNN entirely.** If attention can relate any two
positions directly, the sequential recurrence is doing little — and removing it makes the whole model
parallelizable, which is what let transformers scale to billions of parameters.

## Table of contents

1. [Self-attention: the core operation](#1-self-attention-the-core-operation)
2. [Scaled dot-product attention](#2-scaled-dot-product-attention)
3. [Multi-head attention](#3-multi-head-attention)
4. [Masking: encoder vs decoder](#4-masking-encoder-vs-decoder)
5. [Positional encodings](#5-positional-encodings)
6. [The full block and why it works](#6-the-full-block-and-why-it-works)
7. [The complete transformer](#7-the-complete-transformer)
8. [Common misconceptions](#8-common-misconceptions)

## 1. Self-attention: the core operation

In seq2seq attention, a *decoder* query attended over *encoder* states. **Self-attention** applies the
same mechanism *within a single sequence*: every token builds a query, and attends over the keys and
values of **all tokens in the same sequence** (including itself). Each token's new representation is a
weighted blend of every token's value, weighted by relevance. The result is a **contextual**
representation — the vector for "bank" now depends on whether "river" or "money" is nearby, fixing the
polysemy that static embeddings ([10.03](../../10-nlp/03-word-embeddings/)) could not.

Each token produces three vectors by linear projection:

- **Query** $q$ — what this token is looking for.
- **Key** $k$ — what this token offers to others.
- **Value** $v$ — the information this token carries.

## 2. Scaled dot-product attention

For queries $Q$, keys $K$, values $V$ (each a matrix of stacked vectors):

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V .
$$

Read it right to left: $QK^\top$ scores every query against every key; divide by $\sqrt{d_k}$; softmax
into weights that sum to 1; multiply by $V$ to get the blend. The $\sqrt{d_k}$ scaling is not cosmetic —
dot products grow like $\sqrt{d_k}$, and without the division the softmax saturates into a near one-hot
with vanishing gradients ([09.03 §5](../../09-sequence-models/03-seq2seq-and-attention/)).
[`from_scratch.py`](from_scratch.py) implements this and matches
`torch.nn.functional.scaled_dot_product_attention` to $7\times10^{-16}$ (Experiment 1). It is a
differentiable, content-based lookup — the entire heart of the transformer.

## 3. Multi-head attention

One attention operation forces the model to blend all relationships into a single weighted average.
**Multi-head attention** runs $h$ attentions in **parallel** on different learned projections, then
concatenates:

$$
\text{MHA}(X) = \big[\text{head}_1; \dots; \text{head}_h\big]\,W^O, \qquad \text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V).
$$

Each head operates in a $d/h$-dimensional subspace, so total compute is unchanged, but each can
**specialize** — heads empirically learn to track syntax, coreference, positional offsets, rare tokens.
The from-scratch multi-head attention matches `torch.nn.MultiheadAttention` (packed QKV projection and
all) to $6\times10^{-17}$ (Experiment 2).

## 4. Masking: encoder vs decoder

The same attention serves two roles, distinguished by a **mask** added to the scores before the
softmax:

- **Encoder (bidirectional)** — no mask; every token attends to every token. Used for understanding
  tasks (BERT, [11.02](../02-pretraining/)).
- **Decoder (causal / autoregressive)** — a mask sets scores for *future* positions to $-\infty$, so
  token $i$ attends only to tokens $\le i$. Used for generation (GPT).

Experiment 3 confirms the causal mask puts **exactly 0** weight on every future token while each row
still sums to 1:

```
pos 0: 1.00  0.00  0.00  0.00  0.00
pos 1: 0.38  0.62  0.00  0.00  0.00
pos 2: 0.23  0.27  0.49  0.00  0.00
...
```

This single mask is what turns an encoder into a next-token language model — the difference between
BERT and GPT is largely this triangle of $-\infty$.

## 5. Positional encodings

Self-attention is **permutation-invariant** ([08.05 §3](../../08-computer-vision/05-vision-transformers/)):
shuffle the tokens and the outputs shuffle identically — it has no notion of order. Since word order is
meaning, transformers **add a positional encoding** to each token. The original uses fixed sinusoids at
geometrically spaced frequencies:

$$
\text{PE}_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad \text{PE}_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right).
$$

Experiment 4 confirms the two properties that matter: every position gets a **distinct** encoding
(minimum pairwise distance $> 0$), and the **dot product between positions decays smoothly with their
distance** (32.0 → 30.9 → 28.3 → 23.9 → 19.4 as the offset grows), so attention can express "the
previous token" or "$k$ words back" as a relative signal. Learned absolute embeddings (GPT-2), and later
relative schemes (RoPE, ALiBi — [11.03](../03-efficient-attention/)), are alternatives; the need for
*some* positional signal is universal.

## 6. The full block and why it works

A transformer layer wraps attention and an MLP in residual connections and normalization. The modern
**pre-norm** form:

$$
\begin{aligned}
x &\leftarrow x + \text{MHA}(\text{LayerNorm}(x)) \quad &\text{(mix information across tokens)}\\
x &\leftarrow x + \text{MLP}(\text{LayerNorm}(x)) \quad &\text{(process each token's features)}
\end{aligned}
$$

- **Attention** moves information *between* tokens; the **MLP** (a 2-layer GELU network applied
  identically to each position) transforms *within* each token. Alternating them is the whole design.
- **Residual connections** ([08.02 §4](../../08-computer-vision/02-cnn-architectures/)) give the
  gradient a clean path through a deep stack — the same fix that made deep CNNs and RNNs trainable.
- **LayerNorm** ([07.07](../../07-deep-learning/07-normalization/)) keeps activations well-scaled;
  **pre-norm** (LayerNorm *inside* the residual branch) is more stable to train than the original
  post-norm and is now standard.

Experiment 5 builds this block and matches `torch.nn.TransformerEncoderLayer` to $4\times10^{-16}$. The
entire transformer is these two sub-layers, repeated.

## 7. The complete transformer

Stack $N$ identical blocks, and you have an encoder (or, with causal masking, a decoder). The full
model: tokenize ([10.01](../../10-nlp/01-text-preprocessing/)) → embed + positional encoding → $N$
transformer blocks → task head. The payoff is **parallelism**. Experiment 6 contrasts the cost:

| Model | Sequential steps | Compute per layer |
|---|:--:|:--:|
| RNN | $O(n)$ | $O(n\,d^2)$ |
| self-attention | $O(1)$ | $O(n^2 d)$ |

An RNN processes tokens **one at a time** ($O(n)$ sequential steps), so it cannot parallelize over the
sequence. Self-attention computes all pairwise interactions **at once** — $O(1)$ sequential depth,
fully parallel on a GPU — which is *the* reason transformers train fast enough to scale to billions of
parameters. The price is the $n \times n$ attention matrix: $O(n^2)$ compute and memory (67M scores at
$n = 8192$), the bottleneck that [efficient attention](../03-efficient-attention/) exists to attack.

## 8. Common misconceptions

- **"Attention has learned weights."** The attention *operation* (softmax of dot products) has none;
  the learning is in the Q/K/V/output projections that produce good queries, keys, and values.
- **"Multi-head attention is more compute."** Splitting $d$ into $h$ heads keeps total compute constant;
  it buys *diversity*, not size (§3).
- **"Positional encodings are a minor add-on."** Without them a transformer is order-blind — it would
  read a sentence as a bag of words (§5).
- **"Pre-norm vs post-norm doesn't matter."** Pre-norm is markedly more stable for deep stacks and is
  why modern LLMs train without the warmup gymnastics the original needed (§6).
- **"Transformers replaced everything because they're smarter."** They win largely because they
  **parallelize** and therefore *scale* — the architecture plus scale, not cleverness per parameter
  (§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the full transformer in NumPy, verified against PyTorch. Six
  experiments: (1) scaled dot-product attention == `F.scaled_dot_product_attention`; (2) multi-head ==
  `nn.MultiheadAttention`; (3) causal masking puts exactly 0 on the future; (4) sinusoidal positional
  encodings are distinct and encode relative distance; (5) a full block == `nn.TransformerEncoderLayer`;
  (6) the $O(1)$-depth / $O(n^2)$-cost trade-off.
- **[exercises.md](exercises.md)** — derive attention and its gradient, implement multi-head and
  masking, analyze complexity.
- **[references.md](references.md)** — the transformer papers and the annotated implementations.

## Where this leads

- **Pretraining objectives: BERT, GPT, T5** → [11.02](../02-pretraining/)
- **Efficient & long-context attention (KV cache, FlashAttention, RoPE)** → [11.03](../03-efficient-attention/)
- **The attention it generalizes (seq2seq)** → [09.03](../../09-sequence-models/03-seq2seq-and-attention/)
- **Vision transformers — the same block on image patches** → [08.05](../../08-computer-vision/05-vision-transformers/)
- **Normalization and residuals that make it trainable** → [07.07](../../07-deep-learning/07-normalization/), [08.02](../../08-computer-vision/02-cnn-architectures/)
