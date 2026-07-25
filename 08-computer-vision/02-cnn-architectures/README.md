# 08.02 — CNN Architectures

> **Every landmark CNN is one new idea bolted onto the last, each fixing the previous generation's
> bottleneck.** Deeper stalled → residuals. Wide-and-deep got expensive → bottlenecks and depthwise
> convs. Fully-connected heads ballooned → global average pooling. This chapter tells that story as a
> sequence of *measured* innovations, not a gallery of diagrams — each idea is demonstrated
> numerically in [`from_scratch.py`](from_scratch.py).

The convolution ([08.01](../01-convolution/)) is the primitive. An *architecture* is how you stack it:
how deep, how wide, what connects to what. From 1998 to 2022 the field converged on a handful of
reusable patterns, and knowing *why* each was introduced is worth more than memorizing any single net.

## Table of contents

1. [The evolution at a glance](#1-the-evolution-at-a-glance)
2. [LeNet and AlexNet — the template](#2-lenet-and-alexnet--the-template)
3. [VGG — go deep with small filters](#3-vgg--go-deep-with-small-filters)
4. [ResNet — residuals solve degradation](#4-resnet--residuals-solve-degradation)
5. [Bottlenecks — cheaper depth](#5-bottlenecks--cheaper-depth)
6. [Depthwise-separable — cheaper still](#6-depthwise-separable--cheaper-still)
7. [Global average pooling — kill the FC head](#7-global-average-pooling--kill-the-fc-head)
8. [Inception, DenseNet, EfficientNet, ConvNeXt](#8-inception-densenet-efficientnet-convnext)
9. [Common misconceptions](#9-common-misconceptions)

## 1. The evolution at a glance

| Year | Network | The one new idea | What it fixed |
|---|---|---|---|
| 1998 | **LeNet-5** | conv + pool + FC, trained by backprop | hand-designed features |
| 2012 | **AlexNet** | ReLU, dropout, GPU training, depth | proved deep CNNs win (ImageNet) |
| 2014 | **VGG** | only 3×3 convs, very deep | big filters (§3) |
| 2014 | **Inception/GoogLeNet** | multi-scale blocks, 1×1 bottlenecks | fixed compute budget (§8) |
| 2015 | **ResNet** | residual connections | the degradation problem (§4) |
| 2016 | **DenseNet** | connect every layer to every later layer | feature reuse, gradient flow (§8) |
| 2017 | **MobileNet/Xception** | depthwise-separable convolution | compute cost (§6) |
| 2019 | **EfficientNet** | compound depth/width/resolution scaling | how to scale (§8) |
| 2022 | **ConvNeXt** | a CNN modernized with transformer tricks | closed the gap to ViT (§8) |

Four ideas from that table — residuals, bottlenecks, depthwise-separable convs, and global average
pooling — recur in almost every modern network. The rest of this chapter measures each.

## 2. LeNet and AlexNet — the template

**LeNet-5** (LeCun, 1998) set the template still used today: `[conv → nonlinearity → pool]` blocks to
extract a feature hierarchy, then fully-connected layers to classify. It worked on digits but did not
scale — sigmoid activations saturated and compute was scarce.

**AlexNet** (2012) kept the template and changed three things that mattered: **ReLU** (no saturation,
fast — [07.03](../../07-deep-learning/03-activations/)), **dropout** in the FC head
([07.08](../../07-deep-learning/08-regularization/)), and **GPU training** on ImageNet. It cut the
ImageNet error nearly in half overnight and started the deep-learning era. The lesson: the template was
already right; it needed a trainable nonlinearity and enough data and compute.

## 3. VGG — go deep with small filters

VGG asked: what if we use *only* $3\times3$ convolutions and just go deeper? The justification is a
receptive-field identity ([08.01 §6](../01-convolution/)): **two stacked $3\times3$ convs have the
same $5\times5$ receptive field as one $5\times5$ conv** — with fewer parameters and an extra
nonlinearity in between. Experiment 5 measures it (64→64 channels):

| Configuration | Receptive field | Parameters |
|---|:--:|---:|
| one $5\times5$ conv | 5×5 | 102,400 |
| **two $3\times3$ convs** | 5×5 | **73,728** (28% fewer) |
| one $7\times7$ conv | 7×7 | 200,704 |
| **three $3\times3$ convs** | 7×7 | **110,592** (45% fewer) |

Same field of view, fewer weights, more nonlinearity — so more representational power per parameter.
VGG turned this into a rule (3×3 everywhere, go deep) that has held ever since. Its weakness was the FC
head (§7).

## 4. ResNet — residuals solve degradation

Stacking more plain layers eventually made networks *worse* — and not from overfitting. A 56-layer
plain net had **higher training error** than a 20-layer one. The cause is the **degradation problem**:
in a deep plain stack the gradient must pass through a long product of Jacobians, and with anything
short of perfect initialization it vanishes (or explodes) exponentially. Experiment 1 measures the
gradient reaching layer 0 of an $L$-layer ReLU net (slightly imperfect init):

| Depth $L$ | Plain grad@0 | Residual grad@0 |
|:--:|---:|---:|
| 10 | 6.6 × 10⁻¹ | 4.1 |
| 30 | 2.6 × 10⁻¹ | 13.3 |
| 50 | 2.0 × 10⁻² | 28.4 |
| 100 | **4.9 × 10⁻⁵** | **125** |

By 100 layers the plain net's gradient at layer 0 is ~$5\times10^{-5}$ — the early layers are frozen.
**ResNet's fix is a single change:** a block computes

$$
\mathbf{y} = \mathbf{x} + F(\mathbf{x}),
$$

so its Jacobian is $I + F'$. The identity $I$ gives the gradient a path that **skips** the nonlinear
branch entirely:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}\left(I + F'\right) = \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{y}}}_{\text{never vanishes}} + \frac{\partial \mathcal{L}}{\partial \mathbf{y}}F'.
$$

The gradient can never fully vanish because the identity term passes it straight through. In the
measurement the residual gradient stays $\geq O(1)$ even at 100 layers. This is what made 100- and
1000-layer networks trainable; residual connections are now in essentially every deep architecture,
including transformers ([11.01](../../11-transformers-and-llms/01-transformer/)). Real ResNets
also add **BatchNorm** ([07.07](../../07-deep-learning/07-normalization/)) inside the block to keep the
*forward* activations bounded (the from-scratch version scales the residual branch to the same end).

## 5. Bottlenecks — cheaper depth

Going deep is only affordable if each block is cheap. A plain $3\times3$ conv on $C$ channels costs
$\sim C^2$ parameters. The **bottleneck** block (ResNet-50+) squeezes the channels down with a
$1\times1$ conv, does the expensive $3\times3$ in that thin space, then expands back with another
$1\times1$. Experiment 2 (256 channels, squeeze to 64):

| Block | Parameters |
|---|---:|
| plain $3\times3$ (256 → 256) | 589,824 |
| $1\times1$(256→64) + $3\times3$(64) + $1\times1$(64→256) | **69,632** (8.5× fewer) |

Same input/output shape, **8.5× fewer parameters** — the saving that let ResNet reach 50, 101, and 152
layers. The $1\times1$ convolution here is doing real work: it is a per-pixel dense layer *across
channels* (§8, Inception), the cheapest way to change channel count.

## 6. Depthwise-separable — cheaper still

MobileNet/Xception push the factorization further. A standard conv does two jobs at once — filter each
spatial neighbourhood *and* mix channels — costing $O\,C\,k^2$. **Depthwise-separable** splits them:

1. a **depthwise** conv filters each channel with its own $k\times k$ kernel ($C\,k^2$ params),
2. a **pointwise** $1\times1$ conv mixes channels ($O\,C$ params).

Experiment 3 implements this and verifies it against PyTorch's grouped convolution to **9.8 × 10⁻¹⁵**,
then counts the cost (16→32 channels, $k=3$):

| Conv | Parameters |
|---|---:|
| standard ($O\,C\,k^2$) | 4,608 |
| depthwise-separable ($C\,k^2 + O\,C$) | **656** (7.0× fewer) |

The reduction is $\big(\tfrac1O + \tfrac1{k^2}\big)^{-1}$, approaching $k^2$ for many channels. This is
the core of MobileNet, Xception, and EfficientNet — near-equal accuracy at a fraction of the compute,
which is what put CNNs on phones.

## 7. Global average pooling — kill the FC head

VGG's real weight lives not in its convolutions but in its **fully-connected head**: flattening a
$7\times7\times512$ feature map into a 4096-unit dense layer. Experiment 4 counts it:

| Head | Parameters |
|---|---:|
| flatten + dense(4096) + dense(1000) | 106,856,448 |
| **global average pool + dense(1000)** | **512,000** (209× fewer) |

The first dense layer *alone* is ~103M parameters — most of the whole network — spent purely to
flatten. **Global average pooling** (GAP) instead collapses each channel to its spatial mean, yielding
a 512-vector fed straight to the classifier: **209× fewer parameters**, no overfitting-prone FC layers,
and it accepts any input resolution. Every modern CNN (ResNet, Inception, EfficientNet) uses GAP.

## 8. Inception, DenseNet, EfficientNet, ConvNeXt

- **Inception/GoogLeNet** — run several filter sizes ($1\times1$, $3\times3$, $5\times5$, pool) *in
  parallel* in each block and concatenate, so the net picks the scale it needs. $1\times1$ convs
  reduce channels first to keep the compute budget fixed — the bottleneck idea (§5), introduced here.
- **DenseNet** — connect every layer to *every* later layer by concatenation. Maximizes feature reuse
  and gives every layer a short path to the loss (an extreme of the residual gradient argument, §4),
  with very few parameters per layer.
- **EfficientNet** — instead of scaling depth *or* width *or* resolution ad hoc, scale all three
  together by a **compound coefficient** found by search. Gets far better accuracy-per-FLOP; built on
  depthwise-separable blocks (§6).
- **ConvNeXt** (2022) — take a ResNet and modernize it with tricks borrowed from vision transformers
  (large kernels, LayerNorm, GELU, fewer activations, inverted bottlenecks). Matches ViT accuracy,
  showing the CNN was not obsolete — it was under-tuned.

## 9. Common misconceptions

- **"Deeper is always better."** Not for *plain* nets — beyond a point they degrade (§4). Deeper helps
  only once residuals let the gradient through.
- **"Residual connections prevent overfitting."** They fix *optimization* (gradient flow), not
  generalization. A ResNet can still overfit; that is what regularization is for.
- **"$1\times1$ convolutions are pointless."** They are per-pixel channel-mixing dense layers — the
  cheapest way to change channel count, and the heart of bottlenecks and Inception (§5, §8).
- **"Bigger filters see more, so use them."** Stacks of $3\times3$ match a big filter's receptive field
  with fewer parameters and more nonlinearity (§3).
- **"Transformers made CNNs obsolete."** ConvNeXt (§8) matches ViT by adopting its training recipe —
  much of ViT's edge was the recipe, not the architecture.

## What's in this chapter

- **[from_scratch.py](from_scratch.py)** — five experiments measuring the four recurring ideas:
  (1) residual connections keep the gradient alive to 100 layers where a plain net's vanishes to
  $5\times10^{-5}$; (2) the bottleneck's 8.5× parameter saving; (3) depthwise-separable convolution,
  verified against PyTorch grouped conv to machine precision, at 7× fewer parameters; (4) global
  average pooling's 209× smaller head; (5) VGG's small-filter parameter saving.
- **[exercises.md](exercises.md)** — derive the residual Jacobian, count bottleneck/depthwise FLOPs,
  reason about receptive fields and scaling.
- **[references.md](references.md)** — the landmark papers, one per architecture.

## Where this leads

- **The convolution primitive underneath all of this** → [08.01](../01-convolution/)
- **Transfer learning — reusing these pretrained backbones** → [08.03](../03-transfer-learning/)
- **Detection/segmentation heads on top of these backbones** → [08.04](../04-detection-and-segmentation/)
- **Vision transformers — the architecture ConvNeXt was answering** → [08.05](../05-vision-transformers/)
- **Residual connections and normalization in transformers** → [11.01](../../11-transformers-and-llms/01-transformer/)
- **Why residuals work: the gradient-flow analysis** → [07.02](../../07-deep-learning/02-backpropagation/), [07.05](../../07-deep-learning/05-initialization/)
