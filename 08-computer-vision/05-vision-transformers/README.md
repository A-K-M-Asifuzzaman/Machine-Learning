# 08.05 — Vision Transformers

> **A Vision Transformer treats an image as a sentence of patches.** It cuts the image into a grid of
> patches, projects each into a token, adds positional information, and feeds the sequence to a plain
> transformer — the same architecture as language models. In doing so it *throws away* the
> convolution's built-in priors of locality and translation equivariance, betting that with enough
> data the model learns better priors on its own. This chapter builds the vision-specific machinery
> and measures exactly what that bet costs and buys.

The convolution ([08.01](../01-convolution/)) bakes in two assumptions: features are local, and a
feature means the same thing anywhere ([08.01 §7](../01-convolution/)). Those priors make CNNs
data-efficient. The ViT (Dosovitskiy et al., 2020) discards them and lets **self-attention** relate any
patch to any other from the first layer. The attention mechanism itself is derived in
[11.01](../../11-transformers-and-llms/01-transformer/); here we focus on what is new for *vision*.

## Table of contents

1. [The idea: an image is a sequence of patches](#1-the-idea-an-image-is-a-sequence-of-patches)
2. [Patch embedding is a strided convolution](#2-patch-embedding-is-a-strided-convolution)
3. [Attention has no sense of order](#3-attention-has-no-sense-of-order)
4. [Positional embeddings](#4-positional-embeddings)
5. [Global receptive field, and the inductive-bias trade](#5-global-receptive-field-and-the-inductive-bias-trade)
6. [The quadratic cost and Swin](#6-the-quadratic-cost-and-swin)
7. [Self-supervised vision: DINO, MAE, CLIP](#7-self-supervised-vision-dino-mae-clip)
8. [Common misconceptions](#8-common-misconceptions)

## 1. The idea: an image is a sequence of patches

The full ViT pipeline:

1. **Patchify:** split a $H \times W$ image into a grid of non-overlapping $P \times P$ patches
   (e.g. $16\times16$), giving $N = HW/P^2$ patches.
2. **Embed:** flatten each patch and linearly project it to a $D$-dim **token** (§2).
3. **Add position:** add a positional embedding to each token so order is not lost (§4).
4. **Prepend a `[CLS]` token:** a learned token whose final state is used for classification.
5. **Transformer encoder:** $L$ layers of multi-head self-attention + MLP
   ([11.01](../../11-transformers-and-llms/01-transformer/)).
6. **Head:** classify from the `[CLS]` token (or a global average of tokens).

Everything after step 2 is a standard transformer. The vision-specific parts are steps 1–4.

## 2. Patch embedding is a strided convolution

"Patch embedding" sounds like a new operation. It is not: splitting an image into $P\times P$ patches
and linearly projecting each is **exactly a convolution with kernel size = stride = $P$**. Experiment 1
computes the patch-extract-plus-projection and a `Conv2d(kernel=P, stride=P)` with the same weights and
finds them **identical to $0$**. This is literally how ViT is implemented — one strided conv produces
the tokens. So a ViT's *first* layer is a convolution; the novelty is what happens next: the output is
treated as an **unordered set** of tokens, not a spatial grid.

## 3. Attention has no sense of order

Self-attention is **permutation-equivariant**: permute the input tokens and the outputs permute the
same way, with nothing else changed. Experiment 2 shuffles $6$ tokens by a random permutation and
confirms

$$
\text{attention}(\Pi X) = \Pi\,\text{attention}(X) \quad\text{to } 9 \times 10^{-16}.
$$

Attention treats its input as a **set**. It has no built-in notion of *where* a token is — a patch in
the top-left and the same patch in the bottom-right are indistinguishable to it. For images (and text)
that is unacceptable: spatial layout is information. This is *why* transformers need the next
ingredient.

## 4. Positional embeddings

To restore order, ViT **adds a positional embedding** $\text{pos}_i$ to each patch token before the
transformer. Experiment 3 adds positions, then shuffles the patches while keeping positions in place,
and finds the output now genuinely changes (difference $\approx 9.1$, no longer zero):

$$
\text{attention}(\Pi X + \text{pos}) \neq \Pi\,\text{attention}(X + \text{pos}).
$$

The permutation-invariance of §3 is broken — the model can now tell a top-left patch from a
bottom-right one. ViT uses **learned** 1-D positional embeddings (one vector per grid position);
sinusoidal and 2-D-aware variants exist. Without any positional embedding, a ViT would be blind to
spatial arrangement.

## 5. Global receptive field, and the inductive-bias trade

A single self-attention layer relates **every** patch to every other. Experiment 4 confirms that in one
layer each token attends to all $16$ patches (softmax weights are all strictly positive) — the
receptive field is the **whole image at layer 1**. Contrast a $3\times3$ convolution, which sees only
$9$ neighbours and needs many stacked layers to grow its receptive field
([08.01 §6](../01-convolution/)).

This is the central trade:

| | CNN | ViT |
|---|---|---|
| Locality prior | built in | none (global from layer 1) |
| Translation equivariance | built in | must be learned |
| Receptive field at layer 1 | small (e.g. 3×3) | whole image |
| Data efficiency | high (strong prior) | low (needs lots of data) |
| Ceiling with huge data | good | higher |

ViT's lack of the convolution's priors means it **needs more data** to match a CNN — the original ViT
only beat CNNs when pretrained on 300M images (JFT). **DeiT** (2021) closed the gap on ImageNet-scale
data with strong augmentation and **distillation** from a CNN teacher, showing the data requirement was
about training recipe as much as architecture. The pattern: **strong priors win with little data;
flexible models win with lots.**

## 6. The quadratic cost and Swin

Global attention is expensive. Computing the $N \times N$ attention matrix costs $O(N^2)$ in the number
of patches, so cost grows **quadratically** with patches — quadruple the patches, $16\times$ the cost.
Experiment 5 tabulates it:

| Grid | # patches $N$ | Full ($N^2$) | Windowed ($N \cdot w$, $w{=}49$) | Speedup |
|---|---:|---:|---:|---:|
| 14×14 | 196 | 38,416 | 9,604 | 4× |
| 28×28 | 784 | 614,656 | 38,416 | 16× |
| 56×56 | 3,136 | 9,834,496 | 153,664 | 64× |
| 112×112 | 12,544 | 157,351,936 | 614,656 | **256×** |

For high-resolution images (needed for detection and segmentation), $O(N^2)$ is prohibitive. **Swin
Transformer** (2021) restricts attention to local **windows** of $w$ patches, making the cost **linear**
in $N$, and **shifts** the windows between layers so information still mixes globally over depth. This
hierarchical, windowed design made transformers practical as **general-purpose vision backbones** —
usable for detection and segmentation, not just classification — and is why ViTs now compete with (and
often replace) CNNs across vision.

## 7. Self-supervised vision: DINO, MAE, CLIP

ViT's appetite for data made **label-free pretraining** essential, and transformers turned out to be
excellent at it:

- **MAE (Masked Autoencoders, 2021)** — mask ~75% of patches and train the ViT to reconstruct them.
  A pretext task with no labels that learns strong features; the vision analogue of masked language
  modeling ([11.02](../../11-transformers-and-llms/02-pretraining/)).
- **DINO (2021)** — self-distillation with no labels; the resulting attention maps segment objects for
  free, revealing that ViTs learn object structure unsupervised.
- **CLIP (2021)** — train an image encoder and a text encoder to agree on 400M image–caption pairs.
  The result is **zero-shot** classification (classify by comparing to text prompts) and the backbone
  behind text-to-image models ([12.xx]). Not a ViT-only method, but ViT is its standard image encoder.

These connect vision to [transfer learning](../03-transfer-learning/) and to language pretraining — the
same pretrain-then-adapt paradigm, now without labels.

## 8. Common misconceptions

- **"ViT has no convolutions."** Its patch-embedding layer *is* a strided convolution (§2). Hybrids add
  more.
- **"Transformers are strictly better than CNNs for vision."** Only with enough data/compute or good
  distillation (§5). With little data, a CNN's priors still win. ConvNeXt
  ([08.02](../02-cnn-architectures/)) shows a well-tuned CNN matches ViT.
- **"Attention knows where patches are."** No — attention is permutation-equivariant (§3); position
  comes entirely from the added embeddings (§4).
- **"Bigger images just work."** Full attention is $O(N^2)$ in patches (§6); high resolution needs
  windowed/linear-attention variants.
- **"ViTs need labels to pretrain."** Self-supervised methods (MAE, DINO, CLIP) pretrain without
  classification labels and often produce *better* features (§7).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — the vision-specific ViT machinery in NumPy: (1) patch
  embedding shown identical to a strided convolution; (2) self-attention's permutation-equivariance,
  verified to $10^{-16}$; (3) positional embeddings breaking that invariance; (4) attention's global
  (all-patch) receptive field at layer 1; (5) the quadratic patch cost that motivates windowed
  attention.
- **[exercises.md](exercises.md)** — derive permutation equivariance, count attention FLOPs, reason
  about the inductive-bias/data trade.
- **[references.md](references.md)** — ViT, DeiT, Swin, MAE, DINO, CLIP.

## Where this leads

- **The attention mechanism in full** → [11.01](../../11-transformers-and-llms/01-transformer/)
- **The transformer encoder block** → [11.01](../../11-transformers-and-llms/01-transformer/)
- **CNNs, the prior ViT drops (and ConvNeXt's answer)** → [08.02](../02-cnn-architectures/)
- **Transfer / self-supervised pretraining** → [08.03](../03-transfer-learning/), [11.02](../../11-transformers-and-llms/02-pretraining/)
- **CLIP-conditioned text-to-image generation** → [Part 12](../../12-generative-models/)
