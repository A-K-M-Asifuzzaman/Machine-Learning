# 07.05 — Exercises: Weight Initialization

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Explain the symmetry problem: show that if all weights in a layer are equal, every unit stays
identical through training.

**D2.** Derive the forward variance-preservation condition $\mathrm{Var}(W) = 1/n_{\text{in}}$ from
$\mathrm{Var}(z) = n_{\text{in}}\mathrm{Var}(W)\mathrm{Var}(x)$.

**D3.** Derive the backward condition $\mathrm{Var}(W) = 1/n_{\text{out}}$ and explain why forward and
backward give different requirements.

**D4.** Derive Xavier/Glorot init as the compromise $\mathrm{Var}(W) = 2/(n_{\text{in}}+n_{\text{out}})$.

**D5.** Show that ReLU halves the variance of a zero-mean symmetric input, and derive He init
$\mathrm{Var}(W) = 2/n_{\text{in}}$.

**D6.** Explain, using the product-of-Jacobians view ([07.02 §8](../02-backpropagation/)), why too-small
init vanishes and too-large explodes.

**D7.** Show that an orthogonal weight matrix has gain exactly 1 (preserves norms), and explain why
this helps RNNs.

**D8.** Explain why biases can be initialized to 0 without a symmetry problem.

**D9.** Explain how batch/layer normalization reduces sensitivity to initialization.

**D10.** Derive the variance a signal has at layer $\ell$ under a per-layer gain $g$, and solve for the
$g$ that keeps it constant.

---

## Tier 2 — Implementation

**I1.** Implement zeros, small, Glorot, He, and orthogonal init. Verify the He/Glorot std against
theory and PyTorch `nn.init`.

**I2.** Reproduce Experiment 1: show zero init keeping units identical and failing to learn.

**I3.** Reproduce Experiment 2: measure activation std across a deep tanh network for small/large/Glorot
init.

**I4.** Reproduce Experiment 3: He vs Glorot for a deep ReLU network; show He preserving variance.

**I5.** Reproduce Experiment 4: train from small/large/He init and show stall/explode/converge.

**I6.** Reproduce Experiment 5: tabulate deep-layer activation std by scheme.

**I7.** Implement the backward variance measurement and show gradient variance is also preserved by
He/Glorot.

**I8.** Implement LSUV (data-driven unit-variance init) and compare to He on a deep net.

**I9.** Sweep the init scale continuously and plot deep-layer activation std, marking the He value.

**I10.** Show that with batch normalization, training succeeds from a wider range of init scales than
without.

---

## Tier 3 — Interview

**Q1.** Why can't you initialize all weights to zero?

**Q2.** What is the goal of a good initialization?

**Q3.** Derive the $1/n_{\text{in}}$ variance rule.

**Q4.** He vs Glorot — when do you use each?

**Q5.** Why does He have an extra factor of 2?

**Q6.** What happens with too-small init? Too-large?

**Q7.** How do you initialize biases?

**Q8.** What is orthogonal init good for?

**Q9.** Does initialization still matter with batch norm?

**Q10.** How would you debug a deep network that won't start learning?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the symmetry problem and why weights must be random
- [ ] Derive the variance-preservation condition and the He factor
- [ ] Choose He vs Glorot from the activation
- [ ] Predict vanishing/exploding from the init scale
- [ ] Measure activation variance across depth and read the diagnosis
- [ ] Explain how normalization changes init's importance
