# 11.04 — Scaling Laws & Modern Architecture

> **The most consequential fact about LLMs is that they are predictable.** Loss falls as a smooth power
> law in model size, data, and compute — so you can forecast a $100M training run from cheap small ones,
> and spend compute *optimally* by balancing parameters against data. This chapter derives the scaling
> laws, works the Chinchilla compute-optimal trade-off (and why GPT-3 was undertrained), and covers the
> two ideas — Mixture-of-Experts and a handful of architecture refinements — that define a modern LLM.

Once the transformer ([11.01](../01-transformer/)) and pretraining ([11.02](../02-pretraining/)) were
fixed, progress became an *engineering* question: how big a model, on how much data, for how much
compute? Scaling laws turned that from guesswork into a formula.

## Table of contents

1. [Scaling laws: loss is a power law](#1-scaling-laws-loss-is-a-power-law)
2. [The Chinchilla compute-optimal frontier](#2-the-chinchilla-compute-optimal-frontier)
3. [GPT-3 was undertrained](#3-gpt-3-was-undertrained)
4. [Emergence and its debate](#4-emergence-and-its-debate)
5. [Mixture-of-Experts](#5-mixture-of-experts)
6. [The modern LLM recipe](#6-the-modern-llm-recipe)
7. [Limits of scaling](#7-limits-of-scaling)
8. [Common misconceptions](#8-common-misconceptions)

## 1. Scaling laws: loss is a power law

Kaplan et al. (2020) found that test loss falls as a **power law** in each of model size $N$, dataset
size $D$, and compute $C$ — over *seven orders of magnitude*. The reducible loss (above an irreducible
floor $E$) is straight in log-log space. Experiment 1 fits the compute-optimal frontier of the
Chinchilla parametric model $L(N,D) = E + A/N^\alpha + B/D^\beta$:

| Compute $C$ (FLOPs) | Optimal loss |
|:--:|:--:|
| $10^{18}$ | 3.535 |
| $10^{20}$ | 2.543 |
| $10^{22}$ | 2.084 |
| $10^{24}$ | ~1.9 |

$$
L - E \;\propto\; C^{-0.15}, \qquad R^2 = 1.00000 \text{ (log-log linear fit)}.
$$

A perfect straight line. The practical payoff: **you can forecast the loss of a model $100\times$ bigger
by fitting a line to small models.** This predictability is what justified nine-figure training runs —
the payoff was known before the run started.

## 2. The Chinchilla compute-optimal frontier

Compute is roughly $C \approx 6ND$ (6 FLOPs per parameter per token). For a *fixed* budget $C$, you
trade model size against data: a bigger model sees fewer tokens. Experiment 2 sweeps model size at
$C = 10^{21}$ and finds loss is **U-shaped** with a clear optimum:

| Params $N$ | Tokens $D$ | Loss |
|:--:|:--:|:--:|
| 3.2 × 10⁸ (too small) | 5.3 × 10¹¹ | 2.428 |
| **1.3 × 10⁹ (optimum)** | 1.2 × 10¹¹ | **2.332** |
| 1.0 × 10¹¹ (too big) | 1.7 × 10⁹ | 2.839 |

Too small a model **underfits** (the $A/N^\alpha$ term dominates); too big a model is **data-starved**
(the $B/D^\beta$ term dominates). Fitting this minimum across compute budgets is exactly how Chinchilla
(Hoffmann et al., 2022) derived the compute-optimal rule: **scale parameters and data *together*.**

## 3. GPT-3 was undertrained

The headline consequence. Experiment 3 spends GPT-3's compute budget two ways:

| Model | Params | Tokens | Tokens/param | Loss |
|---|:--:|:--:|:--:|:--:|
| GPT-3 (as trained) | 175 B | 300 B | 1.7 | 2.002 |
| **compute-optimal** | **25 B** | **2.1 T** | 87 | **1.954** |

At the *same* compute, the optimal model is far **smaller** (25B vs 175B) but trained on far **more**
data (2.1T vs 0.3T tokens), and reaches **lower** loss. GPT-3 poured its budget into parameters and
starved on data. Chinchilla proved this by training a 70B model on 1.4T tokens (~20 tokens/param) that
beat the 175B–280B giants. Modern models (Llama 2/3) push tokens/param even higher — often
$100$–$1000{\times}$ — because a smaller, data-rich model is not only compute-optimal to *train* but
much cheaper to *serve* (inference cost scales with $N$). Chinchilla-scaling is now standard.

## 4. Emergence and its debate

Scaling laws predict *loss* smoothly. But some *downstream abilities* (multi-step arithmetic, in-context
learning, chain-of-thought) appeared to switch on suddenly at a scale threshold — "emergent abilities"
(Wei et al., 2022). A 2023 counter-argument (Schaeffer et al.) showed many "emergences" are artifacts of
**discontinuous metrics** (exact-match jumps from 0 to 1) — under smooth metrics the improvement is
gradual. The truth is nuanced: capabilities improve with scale, sharply under some metrics, smoothly
under others. Either way, **you cannot always predict which capability a bigger model unlocks**, only
that loss will drop.

## 5. Mixture-of-Experts

Scaling laws want more parameters — but parameters cost FLOPs. **Mixture-of-Experts (MoE)** breaks that
link: replace one MLP with $E$ expert MLPs and a **router** that sends each token to only its top-$k$
experts. Experiment 4 (8 experts, top-2):

| Quantity | Value |
|---|:--:|
| total expert params (capacity) | 2,176 |
| **active params per token (FLOPs)** | **640** (3.4× fewer) |

The total parameter count — the model's *knowledge* — grows with the number of experts, while the
**compute per token stays fixed** at $k$ experts. This is how Mixtral (8 experts, top-2) and
GPT-4-class systems hold hundreds of billions of parameters at the inference cost of a much smaller
dense model. The catch is **load balancing**: a lazy router that sends everything to one expert wastes
the rest, so an auxiliary balancing loss (and capacity limits) are required.

## 6. The modern LLM recipe

Beyond scale, today's LLMs share a set of refinements over the 2017 transformer — each a small,
well-motivated change:

| Component | Original | Modern | Why |
|---|---|---|---|
| Normalization | LayerNorm, post-norm | **RMSNorm**, **pre-norm** | cheaper, more stable deep ([11.01 §6](../01-transformer/)) |
| Position | sinusoidal | **RoPE** / ALiBi | relative, extrapolates ([11.03 §4](../03-efficient-attention/)) |
| Attention | MHA | **GQA** + FlashAttention | small KV cache, fast ([11.03](../03-efficient-attention/)) |
| MLP activation | ReLU/GELU | **SwiGLU** | gated, better quality per param |
| KV heads | full | grouped | inference memory |
| Sometimes | dense | **MoE** | params ≫ FLOPs (§5) |

**RMSNorm** drops the mean-centering of LayerNorm (just scales by the RMS) — cheaper, equally
effective. **SwiGLU** is a gated MLP ($\text{SiLU}(xW_1) \odot (xW_2)$ then project) that consistently
beats a plain MLP per parameter. None is revolutionary alone; together they are the "Llama recipe" that
most open models now follow.

## 7. Limits of scaling

Scaling is not infinite. **Data is finite** — high-quality text is being exhausted, motivating synthetic
data, multi-epoch training, and multimodal corpora. **Inference cost** favors smaller, longer-trained
models (§3). **Returns are sublinear** — the power-law exponent ($\sim 0.15$) means each order of
magnitude of compute buys a fixed *fractional* loss reduction, so gains get expensive. And loss is not
the goal — *usefulness* requires alignment ([11.06](../06-alignment/)), which scale alone does not
provide. The frontier is shifting from "bigger" to "better data, better post-training, and better
inference efficiency."

## 8. Common misconceptions

- **"Scaling laws guarantee better models."** They predict lower *loss*; whether that yields a specific
  *capability* is not guaranteed (§4).
- **"Bigger is always better."** Compute-optimal (Chinchilla) often means *smaller* with more data — and
  smaller is cheaper to serve (§3).
- **"MoE models are as expensive as their parameter count."** Inference cost tracks *active* params
  ($k$ experts), not total (§5).
- **"Emergent abilities are magic."** Many are artifacts of discontinuous metrics; underlying loss
  improves smoothly (§4).
- **"The modern recipe is a new architecture."** It is the same transformer with cheaper norm, better
  position encoding, gated MLPs, and grouped attention (§6).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the scaling math and MoE in NumPy. Four experiments:
  (1) compute-optimal loss is a power law ($R^2 = 1.0$ in log-log); (2) the IsoFLOP U-curve and its
  optimum; (3) GPT-3 undertrained vs the Chinchilla-optimal split; (4) MoE decoupling total params from
  active FLOPs.
- **[exercises.md](exercises.md)** — derive the compute-optimal frontier, implement MoE routing, reason
  about the modern recipe.
- **[references.md](references.md)** — Kaplan, Chinchilla, MoE, and modern-architecture papers.

## Where this leads

- **Efficient attention (GQA, RoPE, FlashAttention) in the recipe** → [11.03](../03-efficient-attention/)
- **Adapting the pretrained model cheaply** → [11.05](../05-adaptation/)
- **Alignment — turning loss into usefulness** → [11.06](../06-alignment/)
- **Inference cost that Chinchilla-scaling optimizes** → [11.07](../07-inference/)
- **The transformer block being scaled** → [11.01](../01-transformer/)
