# 08.01 — Convolution

> **A convolution is a matrix multiply with two constraints bolted on: locality and weight sharing.**
> Those two constraints are the entire reason a network can look at a 224×224 image without a
> trillion parameters. This chapter derives the operation, its gradient, and the three properties —
> parameter economy, growing receptive fields, and translation equivariance — that make it the right
> primitive for images. Everything is verified against PyTorch to machine precision.

A dense layer ([07.01](../../07-deep-learning/01-neural-network-basics/)) connects every input to
every output. On a 224×224×3 image mapping to 64 channels, that is **483 billion** parameters for one
layer (measured below). It also throws away the one thing we know about images: **nearby pixels are
related, and a cat is a cat wherever it appears.** Convolution encodes both facts directly.

## Table of contents

1. [From dense to convolution: the two constraints](#1-from-dense-to-convolution-the-two-constraints)
2. [A kernel is a feature detector](#2-a-kernel-is-a-feature-detector)
3. [The operation, precisely](#3-the-operation-precisely)
4. [Convolution as one matrix multiply (im2col)](#4-convolution-as-one-matrix-multiply-im2col)
5. [Parameter economy](#5-parameter-economy)
6. [Receptive fields](#6-receptive-fields)
7. [Translation equivariance — the inductive bias](#7-translation-equivariance--the-inductive-bias)
8. [Pooling](#8-pooling)
9. [The backward pass](#9-the-backward-pass)
10. [Common misconceptions](#10-common-misconceptions)

## 1. From dense to convolution: the two constraints

Start with a dense layer $\mathbf{y} = W\mathbf{x}$ on a flattened image. Impose two restrictions on
$W$:

1. **Locality (sparsity).** An output only connects to a small $k \times k$ neighbourhood of the
   input, not the whole image. Every weight outside that window is forced to zero.
2. **Weight sharing.** The *same* $k \times k$ set of weights is reused at every spatial location.

The result is convolution. It is *still a linear map* — you can write it as a (huge, structured)
matrix — but it has $O \cdot C \cdot k^2$ free parameters instead of $(C H W)(O H W)$, and it has a
built-in prior that features are local and position-independent. Those are not tricks; they are
assumptions about images that happen to be true.

## 2. A kernel is a feature detector

One filter (kernel) slides over the image and, at each position, computes a weighted sum of the local
patch. The weights *are* the feature it looks for. The classic hand-designed example is the **Sobel**
operator:

$$
K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \qquad
K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}.
$$

$K_x$ responds to *horizontal* intensity change — a vertical edge; $K_y$ responds to a horizontal
edge. On a test image that is dark on the left and bright on the right (one vertical edge),
Experiment 5 measures the peak responses:

| Filter | Tuned for | Peak $\lvert$response$\rvert$ |
|---|---|:--:|
| $K_x$ (Sobel-x) | vertical edges | **4.0** — fires exactly at the edge |
| $K_y$ (Sobel-y) | horizontal edges | **0.0** — nothing to detect |

The edge-detector fires at the edge; the wrong-orientation detector is silent. **Training a CNN does
not hand-design these kernels — it learns them.** The first convolutional layer of a trained network
reliably converges to edge and colour-blob detectors that look just like Sobel/Gabor filters
([Zeiler & Fergus 2014](references.md)).

## 3. The operation, precisely

For input $X \in \mathbb{R}^{C \times H \times W}$ and a filter bank $W \in \mathbb{R}^{O \times C
\times k_H \times k_W}$ with bias $b \in \mathbb{R}^O$, the output at channel $o$, position $(p, q)$
is

$$
Y_{o,p,q} = b_o + \sum_{c=1}^{C} \sum_{u=1}^{k_H} \sum_{v=1}^{k_W} W_{o,c,u,v}\, X_{c,\; sp+d(u-1)-P,\; sq+d(v-1)-P},
$$

where $s$ is the **stride**, $P$ the **padding**, and $d$ the **dilation**. (Deep learning calls this
"convolution" but does *not* flip the kernel — it is technically cross-correlation. The distinction is
immaterial because the weights are learned.) The output size along each axis is

$$
H_{\text{out}} = \left\lfloor \frac{H + 2P - d(k_H - 1) - 1}{s} \right\rfloor + 1.
$$

- **Stride $s$** subsamples the output (downsampling, fewer positions).
- **Padding $P$** adds a zero border so output size can match input size ("same" convolution).
- **Dilation $d$** spaces out the taps to enlarge the receptive field without more weights (§6).

## 4. Convolution as one matrix multiply (im2col)

The efficient implementation makes the "it's really a matrix multiply" claim literal. **im2col**
gathers every $k_H \times k_W$ receptive field into a column:

$$
X \;\xrightarrow{\text{im2col}}\; \text{cols} \in \mathbb{R}^{(C k_H k_W) \times L}, \qquad L = H_{\text{out}} W_{\text{out}},
$$

then the whole layer is a single GEMM:

$$
Y_{\text{flat}} = W_{\text{flat}} \,\cdot\, \text{cols}, \qquad W_{\text{flat}} \in \mathbb{R}^{O \times (C k_H k_W)}.
$$

This is exactly how [`from_scratch.py`](from_scratch.py) computes it, and it matches
`torch.nn.functional.conv2d` to **~$10^{-14}$** across every configuration of stride, padding,
dilation, and channel count (Experiment 1). It is also why convolutions run fast on GPUs: they reduce
to the one operation hardware is best at.

## 5. Parameter economy

The headline benefit. A dense layer preserving spatial size has $(CHW)(OHW)$ weights; a $3\times3$
convolution has $O \cdot C \cdot 9 + O$ — **independent of image size**. Experiment 3 measures the gap:

| Image | $C \to O$ | Dense params | Conv $3\times3$ params | Ratio |
|---|---|---:|---:|---:|
| 32×32 | 3 → 64 | 201,326,592 | 1,792 | 1.1 × 10⁵ |
| 64×64 | 64 → 128 | 137,438,953,472 | 73,856 | 1.9 × 10⁶ |
| 224×224 | 3 → 64 | 483,385,147,392 | 1,792 | 2.7 × 10⁸ |

A single dense layer on an ImageNet-sized image would need **half a trillion** weights; the
convolution needs **1,792**. Weight sharing is not an optimization — it is what makes vision networks
exist at all.

## 6. Receptive fields

The **receptive field** is the region of the input that influences one output unit. A single
$3\times3$ conv sees $3\times3$. Stacking convolutions *grows* it — but slowly. Experiment 4 measures
both regimes:

| Depth $L$ | Plain $3\times3$ (RF) | Dilated $1,2,4,\dots$ (RF) |
|:--:|:--:|:--:|
| 1 | 3×3 | 3×3 |
| 2 | 5×5 | 7×7 |
| 3 | 7×7 | 15×15 |
| 4 | 9×9 | 31×31 |
| 5 | 11×11 | 63×63 |

Stacking plain $3\times3$ convs grows the receptive field **linearly** ($1 + 2L$) — seeing a large
context needs many layers, which is exactly why deep stacks of small filters (VGG) replaced single
large filters. **Dilated** convolutions space the taps by $1, 2, 4, \dots$ and grow the receptive
field **exponentially** ($1 + 2(2^L - 1)$) with no extra parameters — the trick behind dense
prediction (segmentation) and WaveNet-style audio models.

## 7. Translation equivariance — the inductive bias

Because one kernel is shared across all positions, **shifting the input shifts the output by the same
amount**:

$$
\text{conv}(\text{shift}_\Delta(X)) = \text{shift}_\Delta(\text{conv}(X)).
$$

Experiment 6 shifts an input by 4 pixels and checks:

| Layer | Output is the input's shift? |
|---|:--:|
| Convolution | **True** — same response, translated |
| Dense layer | **False** — a shifted input gives an unrelated output |

This is *equivariance*: a feature is detected wherever it appears, at no extra parameter cost. A dense
layer has an independent weight per position, so it would have to re-learn every feature at every
location. Equivariance — then **pooling** to convert it into *invariance* (§8) — is the core inductive
bias that makes CNNs data-efficient on images.

## 8. Pooling

Pooling downsamples a feature map by summarizing each window:

- **Max pooling** keeps the strongest activation — "was this feature present anywhere in the window?"
- **Average pooling** takes the mean — a smooth downsample.

Both match PyTorch to machine precision (Experiment 7). Max-pool's **backward pass** routes the
gradient *only* to the position that won the max (the argmax); every other input in the window gets
zero. Pooling gives **local translation invariance**: a small shift usually does not change which
value is the maximum, so the pooled output is unchanged. Modern architectures increasingly replace
pooling with strided convolutions, but the invariance argument is the same.

## 9. The backward pass

Convolution is linear, so its gradients are themselves convolutions. With upstream gradient
$\partial \mathcal{L}/\partial Y = \delta$:

- **Bias:** $\dfrac{\partial \mathcal{L}}{\partial b_o} = \sum_{n,p,q} \delta_{n,o,p,q}$ (sum over batch and space).
- **Weights:** $\dfrac{\partial \mathcal{L}}{\partial W_{\text{flat}}} = \delta_{\text{flat}} \cdot \text{cols}^\top$ — a correlation of the input patches with the upstream gradient.
- **Input:** $\dfrac{\partial \mathcal{L}}{\partial X} = \text{col2im}\!\left(W_{\text{flat}}^\top \cdot \delta_{\text{flat}}\right)$ — scatter-add the gradient back through the same patches (a *full, flipped* convolution). Overlapping patches accumulate, which is why `col2im` uses `np.add.at`.

[`from_scratch.py`](from_scratch.py) implements all three and matches PyTorch autograd to **~$10^{-14}$**
(the gradient w.r.t. the input is *exactly* $0$ error). Getting the backward pass right to machine
precision is the real proof that the im2col/col2im pair are correct adjoints.

## 10. Common misconceptions

- **"Convolution flips the kernel."** Mathematical convolution does; deep-learning "convolution" is
  cross-correlation (no flip). Since weights are learned, it makes no difference — but it explains why
  the backward pass involves a genuine flip.
- **"Bigger filters see more, so use large kernels."** Two stacked $3\times3$ convs have the same
  $5\times5$ receptive field as one $5\times5$ conv, with fewer parameters and an extra nonlinearity.
  Small-filter stacks (§6) won.
- **"Padding is just about output size."** Zero-padding also injects an artificial edge at the border
  (the Sobel-y filter fired on the padded border until Experiment 5 dropped the padding) — a real
  source of boundary artifacts.
- **"Pooling is essential."** It is one way to get invariance; strided convolutions achieve
  downsampling with learnable weights and are now often preferred.
- **"1×1 convolutions do nothing."** A $1\times1$ conv is a per-pixel dense layer across channels — it
  mixes channels and changes their count. It is the workhorse of bottlenecks and Inception blocks
  ([08.02](../02-cnn-architectures/)).

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — conv2d built on im2col, forward *and* full backward
  (dX, dW, db), plus max/avg pooling, all verified against `torch.nn.functional` and autograd to
  machine precision. Seven experiments: (1-2) forward+backward match PyTorch; (3) the parameter saving
  vs a dense layer; (4) receptive-field growth, plain vs dilated; (5) a Sobel kernel detecting edges;
  (6) translation equivariance vs a dense layer; (7) pooling forward+backward.
- **[exercises.md](exercises.md)** — derive the output-size and gradient formulas, implement conv via
  im2col, count parameters and FLOPs, reason about receptive fields.
- **[references.md](references.md)** — the exact sources behind every section.

## Where this leads

- **CNN architectures built from these blocks (ResNet, Inception, …)** → [08.02](../02-cnn-architectures/)
- **Transfer learning — reusing the learned filters** → [08.03](../03-transfer-learning/)
- **Backprop, the chain rule this chapter specializes** → [07.02](../../07-deep-learning/02-backpropagation/)
- **Normalization and initialization for deep conv stacks** → [07.05](../../07-deep-learning/05-initialization/), [07.07](../../07-deep-learning/07-normalization/)
- **Attention as an alternative to convolution's inductive bias** → [08.05](../05-vision-transformers/), [11.01](../../11-transformers-llms/01-attention/)
