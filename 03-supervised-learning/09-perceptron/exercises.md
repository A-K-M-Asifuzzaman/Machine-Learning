# 03.09 — Exercises: The Perceptron

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that after the update $\mathbf{w}\leftarrow\mathbf{w}+\eta y_i\mathbf{x}_i$ on a
misclassified point, that point's score $y_i(\mathbf{w}^{\top}\mathbf{x}_i)$ increases by
$\eta\Vert\mathbf{x}_i\Vert^{2}$.

**D2.** Prove the learning rate is irrelevant on separable data: show scaling $\eta$ scales
$\mathbf{w}$ and changes no sign.

**D3.** *(The full convergence proof.)* Assume separability with unit $\mathbf{w}^{\star}$ and
margin $\gamma$. Prove, from $\mathbf{w}_0=\mathbf{0}$:
(a) $\mathbf{w}_k^{\top}\mathbf{w}^{\star}\ge k\gamma$;
(b) $\Vert\mathbf{w}_k\Vert^{2}\le kR^{2}$;
(c) combine with Cauchy-Schwarz to get $k\le(R/\gamma)^{2}$.
Identify where each of separability and "the update happened because the point was misclassified"
is used.

**D4.** Explain geometrically what parts (a) and (b) of the proof each control, and why "alignment
cannot exceed length" gives the bound.

**D5.** Show that the bound $(R/\gamma)^{2}$ is independent of $n$ and of the dimension $d$. Why is
that remarkable?

**D6.** Prove the perceptron never converges on non-separable data. *Hint*: if it did, it would
have found a separator.

**D7.** Prove that a single perceptron cannot compute XOR: show no $(\mathbf{w}, b)$ classifies all
four points correctly.

**D8.** Construct a two-layer network that computes XOR, giving explicit weights. Show
$\text{XOR} = \text{OR}\land\lnot\text{AND}$ and that OR and AND are each linearly separable.

**D9.** Write the perceptron loss $\max(0, -yf)$ and show that SGD on it recovers the perceptron
update rule.

**D10.** Compare the perceptron loss with the hinge loss $\max(0, 1-yf)$. Show they differ only by
a margin shift, and explain what that shift buys the SVM.

**D11.** Derive the ADALINE / delta rule from squared loss $\tfrac12(y-\mathbf{w}^{\top}\mathbf{x})^{2}$
and show it is exactly an SGD step. Contrast the error term with the perceptron's.

**D12.** Explain why ADALINE, not the perceptron, is the true ancestor of backpropagation.

**D13.** Show the perceptron's weights are always a linear combination of training points, and use
this to derive the kernel perceptron.

**D14.** Explain, via the ensembling/variance argument, why the averaged perceptron generalizes
better than the plain one.

---

## Tier 2 — Implementation

**I1.** Implement the perceptron. Verify it reaches zero training error on separable data and
matches sklearn's decisions.

**I2.** Reproduce Experiment 1: construct problems with a known margin and verify the mistake count
stays under $(R/\gamma)^{2}$.

**I3.** Reproduce Experiment 2: sweep the margin toward zero and show mistakes climb as
$1/\gamma^{2}$, and that a non-separable problem never converges.

**I4.** Implement the pocket algorithm. On non-separable data, show it returns better weights than
the plain perceptron at the same iteration budget.

**I5.** Implement the averaged perceptron. Reproduce Experiment 4 and confirm the gap over the
plain version widens with noise.

**I6.** Implement ADALINE. Verify a delta-rule step equals an SGD step on squared loss to machine
precision.

**I7.** Reproduce Experiment 3: show a single perceptron and ADALINE are at chance on XOR, and a
two-layer `tanh` network solves it.

**I8.** Implement the kernel perceptron with an RBF kernel. Show it solves XOR and concentric
circles, connecting to [03.07 §8](../07-svm/).

**I9.** *(Order dependence.)* Train the plain perceptron on the same separable data in 20 different
orders. Measure how much the final hyperplane varies, and contrast with logistic regression's
unique solution.

**I10.** *(Margin perceptron.)* Modify the update to fire whenever $y_if < \gamma$ (not just
$< 0$). Show the resulting hyperplane has a larger margin, approaching the SVM's.

**I11.** Visualize the two-layer XOR network's hidden-unit boundaries and show how they combine to
carve out the XOR regions.

**I12.** *(The bridge.)* Take your `TwoLayerNet`, swap `tanh` for a sigmoid, and show it is a
stack of "perceptrons with a smooth activation" — then relate each piece to
[07.01](../../07-deep-learning/01-neural-network-basics/).

---

## Tier 3 — Interview

**Q1.** What is a perceptron, and how does it learn?

**Q2.** State and sketch the proof of the convergence theorem.

**Q3.** What does the bound $(R/\gamma)^{2}$ depend on, and — notably — what does it *not* depend
on?

**Q4.** What happens if the data is not linearly separable?

**Q5.** Why can't a perceptron learn XOR? Is that a training problem or a representation problem?

**Q6.** How do you fix the XOR problem?

**Q7.** What was missing in 1969 that stalled multilayer networks, and what supplied it?

**Q8.** How do the perceptron, logistic regression, and the SVM relate?

**Q9.** Does the learning rate matter for a perceptron?

**Q10.** What is the difference between the perceptron and ADALINE, and why does it matter
historically?

**Q11.** What is the averaged perceptron and why does it generalize better?

**Q12.** How is a perceptron related to a neuron in a deep network?

**Q13.** Why do we still teach the perceptron if it is obsolete?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Reproduce the convergence proof from memory
- [ ] Explain what the bound depends on and why it is dimension-independent
- [ ] State exactly what XOR proved and what it did not
- [ ] Build a two-layer XOR solver and explain why depth fixes representation
- [ ] Explain why ADALINE, not the perceptron, leads to backpropagation
- [ ] Place the perceptron, logistic regression, and SVM on one loss spectrum
- [ ] Explain why every neuron is a perceptron with a smooth activation
