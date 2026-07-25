# 08.02 — Exercises: CNN Architectures

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive the Jacobian of a residual block $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$ and show that
$\partial \mathcal{L}/\partial \mathbf{x}$ contains an additive term equal to $\partial
\mathcal{L}/\partial \mathbf{y}$ that cannot vanish. Contrast with a plain block $\mathbf{y} =
F(\mathbf{x})$.

**D2.** For a plain deep net with per-layer Jacobian of spectral norm $\sigma$, show the gradient at
layer 0 scales like $\sigma^L$. State the condition for vanishing / exploding and why it makes plain
nets fragile to initialization.

**D3.** Show two stacked $3\times3$ convs have a $5\times5$ receptive field, and derive the parameter
counts for one $n\times n$ conv vs $\lfloor n/2\rfloor$ stacked $3\times3$ convs.

**D4.** Derive the parameter count of a bottleneck block $1\times1(C\to r)$, $3\times3(r)$,
$1\times1(r\to C)$ and find the reduction factor vs a plain $3\times3(C\to C)$ as a function of
$C$ and $r$.

**D5.** Derive the parameter and FLOP reduction of depthwise-separable vs standard convolution, and
show it equals $\big(1/O + 1/k^2\big)^{-1}$.

**D6.** Compare the parameter count of a flatten+dense head to a global-average-pool head for a
$H\times W\times C \to K$ classifier, and explain why GAP also removes the fixed-input-size constraint.

**D7.** Explain what a $1\times1$ convolution computes and why it is exactly a per-pixel dense layer
across channels. Give its parameter and FLOP count.

**D8.** Explain DenseNet's connectivity and why concatenating all previous feature maps gives every
layer a short path to the loss. Relate to the residual gradient argument (D1).

**D9.** State EfficientNet's compound-scaling rule and explain why scaling depth, width, and resolution
*together* beats scaling any one alone.

**D10.** Explain why the "degradation problem" is an *optimization* failure, not overfitting, using the
observation that the deeper plain net had higher *training* error.

---

## Tier 2 — Implementation

**I1.** Reproduce Experiment 1: build $L$-layer plain and residual nets and measure the gradient norm
reaching layer 0 as a function of depth. Show plain vanishes and residual does not.

**I2.** Implement a grouped conv2d via im2col; verify `groups=C` gives a depthwise conv identical to
PyTorch's (Experiment 3).

**I3.** Reproduce Experiment 2: count parameters for a plain $3\times3$ vs a bottleneck block at several
channel counts.

**I4.** Reproduce Experiment 3: build depthwise-separable convolution, verify against PyTorch grouped
conv, and measure the parameter/FLOP saving.

**I5.** Reproduce Experiment 4: count parameters for a flatten+FC head vs a GAP head.

**I6.** Reproduce Experiment 5: parameter counts for small-filter stacks vs one large filter.

**I7.** Implement a small ResNet block (conv → BN → ReLU → conv → BN, + skip) and confirm it is the
identity at initialization when the last BN's scale is zero.

**I8.** Implement an Inception block (parallel $1\times1$, $3\times3$, $5\times5$, pool branches +
concat) and count its parameters with and without the $1\times1$ bottleneck reductions.

**I9.** Build a tiny full CNN (a few conv blocks + GAP + linear) in a framework and train it on
CIFAR-10; compare a plain vs a residual variant at increasing depth.

**I10.** *(Scaling.)* For a fixed FLOP budget, empirically compare making a small net deeper vs wider
vs higher-resolution, and relate to EfficientNet's compound scaling.

---

## Tier 3 — Interview

**Q1.** What problem do residual connections solve, and how?

**Q2.** Why did deeper plain networks perform worse, and why is that not overfitting?

**Q3.** What is a bottleneck block and why is it cheaper?

**Q4.** What is a depthwise-separable convolution and how much does it save?

**Q5.** What does a $1\times1$ convolution do?

**Q6.** Why do modern CNNs use global average pooling instead of a fully-connected head?

**Q7.** Why does VGG use only $3\times3$ filters?

**Q8.** What is the key idea of Inception? Of DenseNet?

**Q9.** How does EfficientNet decide how to scale a network?

**Q10.** Are CNNs obsolete now that we have vision transformers?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the residual Jacobian and explain why the gradient survives depth
- [ ] Explain the degradation problem as an optimization failure
- [ ] Count parameters/FLOPs for bottleneck and depthwise-separable blocks
- [ ] Explain what a $1\times1$ conv does and where it is used
- [ ] Justify global average pooling over a flatten+FC head
- [ ] Name each landmark architecture's one contribution
- [ ] Reason about depth/width/resolution scaling trade-offs
