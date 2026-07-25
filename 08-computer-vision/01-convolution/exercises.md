# 08.01 — Exercises: Convolution

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Derive the output-size formula
$H_{\text{out}} = \lfloor (H + 2P - d(k-1) - 1)/s \rfloor + 1$ from the requirement that the last
valid window fits inside the padded, dilated input. Check it against a $7\times7$ input, $k=3$, $s=2$,
$P=1$, $d=1$.

**D2.** Show that convolution is a linear map, and write the (sparse, block-Toeplitz) matrix for a 1-D
convolution with kernel $[a, b, c]$ on a length-5 input, "same" padding.

**D3.** Derive the three backward-pass gradients: $\partial \mathcal{L}/\partial b$,
$\partial \mathcal{L}/\partial W$, and $\partial \mathcal{L}/\partial X$. Show that the gradient
w.r.t. the input is a *full convolution* of the upstream gradient with the flipped kernel.

**D4.** Prove translation equivariance:
$\text{conv}(\text{shift}_\Delta X) = \text{shift}_\Delta(\text{conv}(X))$ (ignoring boundaries), and
explain precisely where weight sharing enters the proof.

**D5.** Show that two stacked $3\times3$ convolutions have a $5\times5$ receptive field but fewer
parameters than one $5\times5$ conv. Generalize to $n$ stacked $3\times3$ convs.

**D6.** Derive the receptive-field size after $L$ layers with dilations $1, 2, 4, \dots, 2^{L-1}$ and
show it is $1 + 2(2^L - 1)$.

**D7.** Count the multiply-adds (FLOPs) of a conv layer with input $C \times H \times W$, $O$ output
channels, kernel $k$, stride $s$. Compare to the parameter count and explain why they scale
differently with image size.

**D8.** Show that a $1\times1$ convolution is exactly a per-pixel dense layer across channels, and
give its parameter count.

**D9.** Derive max-pooling's backward pass and explain why it routes the gradient only to the argmax.
Contrast with average-pooling's backward pass.

**D10.** Explain, using the receptive field, why zero-padding creates boundary artifacts, and what
"reflect" or "replicate" padding does differently.

---

## Tier 2 — Implementation

**I1.** Implement `im2col` and `col2im` for arbitrary stride, padding, dilation. Verify `col2im` is
the exact adjoint of `im2col` (i.e. $\langle A x, y\rangle = \langle x, A^\top y\rangle$) to machine
precision.

**I2.** Implement `conv2d_forward` via im2col and verify it against `torch.nn.functional.conv2d` across
several configs (Experiment 1).

**I3.** Implement `conv2d_backward` (dX, dW, db) and verify against PyTorch autograd (Experiment 2).

**I4.** Reproduce Experiment 3: count parameters for a dense vs a $3\times3$ conv layer at several
image sizes, and confirm the conv count is size-independent.

**I5.** Reproduce Experiment 4: measure the receptive field of a stack of plain vs dilated convs.

**I6.** Reproduce Experiment 5: apply hand-set Sobel-x / Sobel-y kernels and confirm each fires only
on its orientation.

**I7.** Reproduce Experiment 6: verify convolution is translation-equivariant and a random dense layer
is not.

**I8.** Implement max and average pooling (forward + backward) and verify against PyTorch
(Experiment 7).

**I9.** Build a tiny 2-conv-layer classifier (conv → ReLU → pool → conv → ReLU → pool → dense) in
NumPy, train it on a toy image dataset by hand-rolled backprop, and confirm it learns.

**I10.** *(Efficiency.)* Compare the wall-clock time of your im2col conv against a naive 6-nested-loop
convolution; explain the speedup in terms of the single GEMM.

---

## Tier 3 — Interview

**Q1.** Why do we use convolutions instead of dense layers for images?

**Q2.** What is a receptive field, and how do you grow it?

**Q3.** What does stride / padding / dilation each do?

**Q4.** Why do modern CNNs stack small $3\times3$ filters instead of large ones?

**Q5.** What is the difference between equivariance and invariance, and how does a CNN get each?

**Q6.** What does a $1\times1$ convolution do?

**Q7.** How is convolution implemented efficiently on a GPU?

**Q8.** What does the backward pass of a convolution look like?

**Q9.** Why is pooling used, and what does its backward pass do?

**Q10.** How many parameters and FLOPs does a conv layer have?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive the output-size formula and use it fluently
- [ ] Explain convolution as a sparse, weight-shared matrix multiply
- [ ] Implement conv forward + backward via im2col, verified to machine precision
- [ ] Reason about receptive fields (plain and dilated)
- [ ] Explain translation equivariance and where weight sharing creates it
- [ ] Count parameters and FLOPs for a conv layer
- [ ] Explain what pooling does and derive its backward pass
