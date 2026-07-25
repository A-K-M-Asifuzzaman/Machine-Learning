# 11.03 — Efficient Attention

> **Attention is O(n²) and recomputes everything every step — and both facts are fixable without
> changing the math.** The story of making transformers fast and long-context is a story of
> *bookkeeping*: cache the keys and values you already computed, share them across heads, stream the
> softmax so you never store the n×n matrix, and encode position as a rotation or a bias so it
> extrapolates. Every trick here produces the *same* attention output — verified to machine precision —
> just far cheaper. Together they are what turned a 512-token research model into a 1M-token production
> LLM.

The vanilla transformer ([11.01 §7](../01-transformer/)) has two costs: the $n \times n$ attention
matrix ($O(n^2)$ compute and memory), and, during generation, recomputing all past keys/values at every
step. This chapter attacks both.

## Table of contents

1. [The two costs](#1-the-two-costs)
2. [The KV cache](#2-the-kv-cache)
3. [MQA and GQA](#3-mqa-and-gqa)
4. [Positional tricks: RoPE and ALiBi](#4-positional-tricks-rope-and-alibi)
5. [FlashAttention](#5-flashattention)
6. [Sparse and linear attention](#6-sparse-and-linear-attention)
7. [Long context in practice](#7-long-context-in-practice)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The two costs

- **Quadratic attention.** The score matrix is $n \times n$: doubling the sequence quadruples compute
  and memory. At $n = 8192$ that is 67M scores *per head per layer* — the wall for long context.
- **Redundant recomputation in generation.** A decoder generates one token at a time. Naively, each step
  re-projects the keys and values of *all* previous tokens — but those never change. Over $n$ tokens
  that is $O(n^2)$ wasted projections.

Everything below keeps attention's output identical and cuts one of these costs.

## 2. The KV cache

Past tokens' keys and values are fixed once computed, so **cache them**. Store each token's $K, V$ once;
at each generation step, project only the *new* token's $K, V$ and append. Experiment 1 generates 8
tokens both ways:

| Method | K,V projections | Output |
|---|:--:|---|
| recompute (naive) | 36 ($O(n^2)$) | — |
| **KV cache** | **8 ($O(n)$)** | identical (max diff $9\times10^{-16}$) |

Identical output, linear cost. **The KV cache is the single most important LLM-inference optimization** —
without it, generation would be quadratic. Its price is *memory*: the cache grows with sequence length
and, at long context, dominates GPU memory — which is what the next trick attacks.

## 3. MQA and GQA

The KV cache stores one key and value **per head** per token, so multi-head attention's cache is large.
The fix: **share** keys and values across heads.

- **Multi-Query Attention (MQA)** — all query heads share a *single* $K, V$ head.
- **Grouped-Query Attention (GQA)** — a few $K, V$ heads, each shared by a *group* of query heads (the
  middle ground).

Experiment 2 (32 heads, head-dim 128, $n = 4096$, fp16):

| Scheme | KV heads | KV cache |
|---|:--:|:--:|
| MHA | 32 | 67.1 MB |
| **GQA** | 8 | **16.8 MB** (4× smaller) |
| MQA | 1 | 2.1 MB (32× smaller) |

MQA shrinks the cache dramatically but can cost quality; **GQA** is the sweet spot — a 4× smaller cache
with almost no quality loss — and is used by Llama-2/3, Mistral, and most modern LLMs.

## 4. Positional tricks: RoPE and ALiBi

Sinusoidal/learned absolute positions ([11.01 §5](../01-transformer/)) don't extrapolate past the
training length. Two schemes fix this by making position **relative**:

- **RoPE (Rotary Position Embedding)** — *rotate* each query and key by an angle proportional to its
  position before the dot product. Because rotating $q$ by $m$ and $k$ by $n$ leaves only the difference
  $m - n$, the score depends **only on relative distance**. Experiment 3 confirms: every $(m, n)$ pair
  with $m - n = 2$ gives the *identical* score $-4.630102$ (spread $2\times10^{-15}$):

  | $(m, n)$ | RoPE$(q,m)\cdot$RoPE$(k,n)$ |
  |:--:|:--:|
  | (5, 3), (6, 4), (10, 8), (2, 0), (100, 98) | −4.630102 (all equal) |

  No learned parameters, and it extrapolates to longer sequences. RoPE is used by Llama, PaLM, GPT-NeoX,
  and most current LLMs.

- **ALiBi (Attention with Linear Biases)** — add a penalty $-\text{slope}\cdot(i - j)$ to the score
  (larger for more distant keys), with a per-head slope, and *no* position embeddings at all.
  Experiment 5 shows the resulting recency bias — the last query weights its nearest key 0.306 vs the
  farthest 0.108. Even simpler than RoPE, and also extrapolates.

## 5. FlashAttention

The $n \times n$ matrix is an $O(n^2)$ **memory** problem, and memory bandwidth — not compute — is the
real bottleneck on GPUs. **FlashAttention** (Dao et al., 2022) never materializes the matrix. It streams
over blocks of keys/values, maintaining a running max and running sum — an **online softmax** — and
rescales the accumulator whenever a larger score appears:

$$
m^{\text{new}} = \max(m^{\text{old}}, \max_j s_j), \quad \ell \leftarrow \ell\,e^{m^{\text{old}} - m^{\text{new}}} + \textstyle\sum_j e^{s_j - m^{\text{new}}}, \quad O \leftarrow O\,e^{m^{\text{old}} - m^{\text{new}}} + \textstyle\sum_j e^{s_j - m^{\text{new}}} v_j.
$$

Experiment 4 confirms the blockwise result **equals** the full-matrix softmax to $9\times10^{-16}$ — it
is exact, not an approximation. By keeping the working set in fast on-chip SRAM, it also runs several
times faster in wall-clock. FlashAttention (and its v2/v3) is why training with long contexts is
feasible, and it is now the default attention kernel.

## 6. Sparse and linear attention

To break the $O(n^2)$ *compute* asymptotically, restrict *which* pairs attend:

- **Sparse / windowed attention** — each token attends only to a local window plus a few global tokens
  (Longformer, BigBird, Sparse Transformer). Cost $O(n\,w)$ — linear in $n$. Swin
  ([08.05 §6](../../08-computer-vision/05-vision-transformers/)) is the vision analogue.
- **Linear attention** — replace $\text{softmax}(QK^\top)V$ with a kernel feature map
  $\phi(Q)(\phi(K)^\top V)$, reordering the matmuls to $O(n)$ (Performer, Linear Transformer). Cheaper,
  but usually a quality hit.
- **State-space models** (Mamba, S4) — sidestep attention entirely with a recurrence that is $O(n)$ and
  parallelizable, a serious recent challenger for long sequences.

In practice, exact methods (FlashAttention + GQA + RoPE) plus modest sparsity have carried most
production long-context models, because approximations tend to lose quality.

## 7. Long context in practice

Modern LLMs reach 100K–1M+ token contexts by *stacking* these: **GQA** (small cache) + **RoPE** (with
frequency scaling / YaRN to stretch beyond training length) + **FlashAttention** (no $n^2$ memory) +
sometimes sparsity. The remaining challenges are *quality* over long context ("lost in the middle" —
models under-use the center of a long prompt) and the KV-cache memory that still grows linearly, which
motivates cache compression and eviction.

## 8. Common misconceptions

- **"Efficient attention approximates attention."** The KV cache, MQA/GQA, RoPE, and FlashAttention are
  **exact** (verified to $10^{-16}$); only sparse/linear methods approximate (§2–§5 vs §6).
- **"FlashAttention changes the math."** It reorders the softmax computation for memory; the result is
  bit-comparable to standard attention (§5).
- **"MQA is strictly better because the cache is smaller."** It can cost quality; GQA is the standard
  compromise (§3).
- **"RoPE/ALiBi are just alternative position encodings."** Their key property is **extrapolation** to
  unseen lengths, which absolute encodings lack (§4).
- **"Bigger context is free now."** Cache memory still grows linearly, and quality degrades in the middle
  of very long contexts (§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the efficiency tricks in NumPy. Five experiments: (1) KV
  cache gives identical output at $O(n)$ instead of $O(n^2)$; (2) MQA/GQA shrink the KV cache 4–32×;
  (3) RoPE makes scores depend only on relative distance (identical to $10^{-15}$); (4) FlashAttention's
  online softmax equals standard attention to $10^{-16}$; (5) ALiBi's distance-linear recency bias.
- **[exercises.md](exercises.md)** — derive the online softmax, implement the KV cache and RoPE, analyze
  memory.
- **[references.md](references.md)** — FlashAttention, RoPE, ALiBi, GQA, and long-context papers.

## Where this leads

- **The attention it optimizes** → [11.01](../01-transformer/)
- **Scaling laws and model design** → [11.04](../04-scaling-and-architecture/)
- **Inference/serving where the KV cache lives** → [11.07](../07-inference/)
- **Windowed attention in vision (Swin)** → [08.05](../../08-computer-vision/05-vision-transformers/)
- **Positional encodings, the starting point** → [11.01 §5](../01-transformer/)
