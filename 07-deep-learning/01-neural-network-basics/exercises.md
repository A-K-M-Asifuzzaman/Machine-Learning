# 07.01 — Exercises: Neural Network Basics

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the forward pass of an $L$-layer MLP in matrix form, defining pre-activations and
activations.

**D2.** Prove that a stack of linear layers (no activations) is equivalent to a single linear layer.
Give the composed weight matrix and bias.

**D3.** Prove that XOR is not linearly separable. Then hand-derive weights for a 2-2-1 network (with a
step or sign activation) that computes XOR, and identify the features the hidden layer computes.

**D4.** State the universal approximation theorem precisely (single hidden layer, non-polynomial
activation). What does "enough units" mean, and what does the theorem *not* guarantee?

**D5.** Explain, with the tent-map / folding argument, why a depth-$k$ network can represent
$2^{k-1}$ oscillations with $O(k)$ units while a shallow network needs $O(2^{k})$.

**D6.** Show that the ReLU tent map $T(x) = 1 - \mathrm{relu}(2x-1) - \mathrm{relu}(1-2x)$ equals
$1 - |2x-1|$ on $[0,1]$.

**D7.** Count the parameters of an MLP with layer sizes $[d_0, d_1, \dots, d_L]$ (weights + biases).

**D8.** Explain the difference between representational capacity, learnability, and sample complexity
for neural networks.

**D9.** Explain why the forward pass maps naturally onto GPU hardware.

**D10.** Draw the computational graph of a 2-layer MLP with a squared-error loss, and mark where the
chain rule will be applied in backprop ([07.02](../02-backpropagation/)).

---

## Tier 2 — Implementation

**I1.** Implement the MLP forward pass (dense layers + activations). Verify it against PyTorch with
identical weights.

**I2.** Reproduce Experiment 1: show a stack of linear layers equals a single linear map, and that a
linear model cannot fit XOR.

**I3.** Reproduce Experiment 2: solve XOR with a nonlinear hidden layer (random features + least
squares, no backprop).

**I4.** Reproduce Experiment 3: hand-build a 2-2-1 XOR network and verify the hidden representation is
linearly separable.

**I5.** Reproduce Experiment 4: approximate a wiggly target with a single hidden layer and show the
error shrinking with width.

**I6.** Implement the deep tent-fold network and reproduce Experiment 5: represent a 16-oscillation
function exactly with ~10 units, and show a shallow net needing far more.

**I7.** Implement a batched forward pass and verify it matches the per-sample loop.

**I8.** *(Random features.)* Show that a fixed random hidden layer + least-squares output ("extreme
learning machine") is a universal approximator as width grows.

**I9.** Visualize the hidden-layer representation of a trained (or random-feature) XOR network in 2-D.

**I10.** Build an MLP for a small real dataset (forward pass with random-feature output) and compare
its decision boundary to a linear model's.

---

## Tier 3 — Interview

**Q1.** What is a multilayer perceptron?

**Q2.** Why do we need activation functions?

**Q3.** What happens if you stack linear layers without activations?

**Q4.** How does an MLP solve XOR?

**Q5.** What is the universal approximation theorem?

**Q6.** If one hidden layer is universal, why go deep?

**Q7.** Does universal approximation mean a network can learn anything?

**Q8.** What do hidden layers represent?

**Q9.** What is the forward pass?

**Q10.** What is a computational graph and why does it matter?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Write the MLP forward pass and implement it
- [ ] Explain and prove why nonlinearity is essential
- [ ] Hand-build an XOR network and see the hidden representation
- [ ] State the universal approximation theorem and its limits
- [ ] Explain depth vs width with the folding argument
- [ ] See the forward pass as a computational graph ready for backprop
