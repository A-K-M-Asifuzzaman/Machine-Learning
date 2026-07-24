# 07.03 — Exercises: Activation Functions

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Explain the two jobs of an activation function (forward nonlinearity, backward gradient
factor) and why the derivative's shape matters most.

**D2.** Derive $\sigma'(z) = \sigma(z)(1-\sigma(z))$ for the sigmoid and show its maximum is 0.25.

**D3.** Derive $\tanh'(z) = 1 - \tanh^2(z)$ and show its maximum is 1. Show $\tanh(z) = 2\sigma(2z)-1$.

**D4.** State ReLU's derivative and explain why it neither vanishes (on the active side) nor saturates.

**D5.** Explain the dying-ReLU problem mechanistically and how Leaky ReLU / PReLU fix it.

**D6.** Explain why sigmoid's non-zero-centered output biases the weight gradients to one sign.

**D7.** Write GELU ($z\Phi(z)$) and Swish ($z\sigma(z)$) and explain in what sense each is a smooth,
gated ReLU.

**D8.** Derive the numerically stable softmax (subtract the max) and show it is invariant to adding a
constant to all logits.

**D9.** Using backprop's (BP2), explain why a saturating activation causes vanishing gradients through
depth.

**D10.** Explain SELU's self-normalizing property at a high level and why it needs its own init.

---

## Tier 2 — Implementation

**I1.** Implement all activations and their derivatives. Verify the values against PyTorch and the
derivatives by finite differences.

**I2.** Reproduce Experiment 1: tabulate each derivative at increasing $|z|$ and identify saturation.

**I3.** Reproduce Experiment 2: measure gradient flow through a deep network for sigmoid, tanh, ReLU.

**I4.** Reproduce Experiment 3: count dead ReLU units after aggressive training, and show Leaky ReLU
fixing it.

**I5.** Reproduce Experiment 4: plot GELU/Swish vs ReLU and their derivatives; find the non-monotonic
dip.

**I6.** Reproduce Experiment 5: train the same net with sigmoid, tanh, ReLU and compare convergence.

**I7.** Implement a numerically stable softmax + log-softmax and verify against PyTorch on extreme
logits.

**I8.** Implement PReLU (learnable $\alpha$) and add its gradient to backprop.

**I9.** Implement SELU with the standard constants and verify activations stay ~zero-mean/unit-variance
through layers under the right init.

**I10.** Sweep the initialization scale and show how it interacts with dying ReLU.

---

## Tier 3 — Interview

**Q1.** What does an activation function do, beyond adding nonlinearity?

**Q2.** Why did the field move from sigmoid/tanh to ReLU?

**Q3.** What is the vanishing-gradient problem's connection to activations?

**Q4.** What is the dying-ReLU problem, and how do you fix it?

**Q5.** Why is tanh better than sigmoid for hidden layers?

**Q6.** What is GELU and where is it used?

**Q7.** When would you not use ReLU?

**Q8.** Is softmax a hidden-layer activation?

**Q9.** Why must softmax be computed with the max-subtraction trick?

**Q10.** What output activation goes with each task (binary, multiclass, regression)?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain each activation by what its derivative does to the gradient
- [ ] Diagnose saturation, vanishing gradients, and dead units
- [ ] Derive every activation's derivative and stable softmax
- [ ] Choose ReLU / Leaky ReLU / GELU for the right reasons
- [ ] Match the output activation to the task and loss
- [ ] Explain why non-saturating activations train faster
