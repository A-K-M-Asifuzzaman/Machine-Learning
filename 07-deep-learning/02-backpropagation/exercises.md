# 07.02 — Exercises: Backpropagation

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** State the chain rule for a composition $C = f(g(h(\boldsymbol\theta)))$ and explain why
evaluating it right-to-left is efficient.

**D2.** Derive (BP1): $\boldsymbol\delta^{(L)} = \nabla_{\mathbf{a}} C \odot \sigma'(\mathbf{z}^{(L)})$.

**D3.** Derive (BP2): $\boldsymbol\delta^{(\ell)} = ((\mathbf{W}^{(\ell+1)})^\top \boldsymbol\delta^{(\ell+1)}) \odot \sigma'(\mathbf{z}^{(\ell)})$,
and explain why the transpose appears.

**D4.** Derive (BP3) and (BP4): the gradients w.r.t. biases and weights from the error and cached
activations.

**D5.** Show that backprop is reverse-mode autodiff: identify the vector-Jacobian product of a dense
layer and of an activation.

**D6.** Compare the cost of computing the full gradient by backprop vs finite differences, in forward
passes.

**D7.** Explain when forward-mode autodiff beats reverse-mode, and why neural-net training is the
reverse-mode regime.

**D8.** Derive the central-difference formula and show its error is $O(\epsilon^2)$ (vs $O(\epsilon)$
for the one-sided difference).

**D9.** Using (BP2), show that the gradient through $L$ layers is a product of $L$ Jacobians, and
explain when it vanishes or explodes.

**D10.** Explain the memory cost of reverse mode (storing activations) and how gradient checkpointing
trades compute for memory.

---

## Tier 2 — Implementation

**I1.** Implement the forward pass with caching and the backward pass (the four equations). Train a
small MLP.

**I2.** Implement central-difference gradient checking and verify your backprop to ~1e-7.

**I3.** Verify your analytic gradients against PyTorch autograd on the same weights.

**I4.** Reproduce Experiment 2: train XOR from scratch with backprop.

**I5.** Reproduce Experiment 3: measure the backprop speedup over finite differences and confirm the
gradients match.

**I6.** Reproduce Experiment 5: show gradients vanishing with depth in a sigmoid network, and then
show ReLU + good init keeping them stable.

**I7.** Implement a tiny reverse-mode autograd engine (scalar values with `.grad` and a backward
graph, à la micrograd) and rebuild the MLP on top of it.

**I8.** Add softmax + cross-entropy and derive/implement its (especially clean) output gradient
$\hat{\mathbf{p}} - \mathbf{y}$.

**I9.** Implement gradient accumulation over mini-batches and verify it equals the full-batch
gradient.

**I10.** *(Debugging.)* Deliberately introduce a backward-pass bug (e.g. forget a $\sigma'$) and show
gradient checking catching it.

---

## Tier 3 — Interview

**Q1.** What does backpropagation compute?

**Q2.** Why is backprop better than finite differences?

**Q3.** Walk through the four backprop equations.

**Q4.** Why does the transpose of the weight matrix appear in the backward pass?

**Q5.** What is reverse-mode automatic differentiation?

**Q6.** Why does deep learning use reverse mode, not forward mode?

**Q7.** How do you verify a hand-written backward pass?

**Q8.** Where do vanishing / exploding gradients come from?

**Q9.** What must the forward pass store for the backward pass, and why?

**Q10.** Is backprop the same as gradient descent?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive all four backprop equations
- [ ] Implement forward + backward and gradient-check them
- [ ] Explain backprop as reverse-mode autodiff (VJPs)
- [ ] Explain why reverse mode is efficient for a scalar loss
- [ ] Trace vanishing/exploding gradients to the product of Jacobians
- [ ] Debug a backward pass with gradient checking
