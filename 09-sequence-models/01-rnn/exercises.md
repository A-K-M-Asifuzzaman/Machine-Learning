# 09.01 — Exercises: Recurrent Neural Networks

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the RNN recurrence and unroll it for $T=3$ steps into an explicit feedforward graph.
Identify which weights are shared.

**D2.** Derive the BPTT gradient $\partial \mathcal{L}/\partial h_t$ showing the two contributions (this
step's output and the future), and the accumulated $\partial \mathcal{L}/\partial W_{hh}$.

**D3.** Derive $\partial h_T/\partial h_0 = \prod_t \text{diag}(1 - h_t^2)\,W_{hh}$ and explain why a
product of $T$ Jacobians vanishes or explodes.

**D4.** Show that even with spectral radius $\rho(W_{hh}) = 1$, the gradient still vanishes because
$\tanh'(z) = 1 - \tanh^2(z) \le 1$. Bound the shrink over $T$ steps.

**D5.** Derive gradient clipping by norm and show it preserves the gradient *direction* while capping
its magnitude.

**D6.** Explain truncated BPTT and its bias–variance / memory trade-off vs full BPTT.

**D7.** Compare the RNN's vanishing-gradient problem to the deep-net degradation problem
([08.02 §4](../../08-computer-vision/02-cnn-architectures/)) and explain why a near-identity path fixes
both.

**D8.** For a linear recurrence $h_t = W h_{t-1}$, express $h_T$ via the eigendecomposition of $W$ and
state the exact vanish/explode condition.

**D9.** Derive the forward and backward equations for a bidirectional RNN.

**D10.** Explain why weight sharing across time is necessary for variable-length sequences and what it
implies for generalization.

---

## Tier 2 — Implementation

**I1.** Implement the RNN forward pass; verify against `torch.nn.RNN` (Experiment 1).

**I2.** Implement BPTT (dWxh, dWhh, db); verify against PyTorch autograd (Experiment 1).

**I3.** Reproduce Experiment 2: measure $\lVert \partial \mathcal{L}/\partial h_0 \rVert$ vs sequence
length for several spectral radii of $W_{hh}$.

**I4.** Reproduce Experiment 3: tabulate $\tanh'$ and show the product collapses over many steps.

**I5.** Implement gradient clipping and reproduce Experiment 4.

**I6.** Reproduce Experiment 5: train an RNN on the recall-a-bit task and show accuracy collapses with
lag.

**I7.** Implement truncated BPTT with truncation length $k$ and compare training to full BPTT.

**I8.** Implement a character-level RNN language model and sample text from it.

**I9.** Implement a bidirectional RNN and apply it to a tagging task.

**I10.** *(Init.)* Try an identity-initialized ReLU RNN (IRNN) on the long-lag task and compare its
memory range to the tanh RNN.

---

## Tier 3 — Interview

**Q1.** What is an RNN and how does it process a sequence?

**Q2.** What is backpropagation through time?

**Q3.** What causes vanishing and exploding gradients in RNNs?

**Q4.** Why does the borderline for vanishing sit below spectral radius 1?

**Q5.** How does gradient clipping help, and what does it *not* fix?

**Q6.** What is truncated BPTT and why use it?

**Q7.** How long a dependency can a plain RNN actually learn?

**Q8.** What is a bidirectional RNN and when can you use one?

**Q9.** How does an RNN's vanishing-gradient problem relate to deep-net gradient flow?

**Q10.** Are RNNs still used given transformers?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Write and unroll the RNN recurrence
- [ ] Derive BPTT and the shared-weight gradient accumulation
- [ ] Derive the Jacobian-product form of the long-range gradient
- [ ] Explain why vanishing is the default failure (tanh saturation)
- [ ] Implement and reason about gradient clipping and truncated BPTT
- [ ] Demonstrate the short-memory limit empirically
- [ ] Choose the right RNN wiring for a given task
