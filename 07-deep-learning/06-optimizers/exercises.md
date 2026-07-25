# 07.06 — Exercises: Optimizers

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the gradient-descent update and explain the role of the learning rate.

**D2.** Explain why mini-batch SGD's gradient is a noisy estimate, and why that noise helps (saddle
escape, flat minima).

**D3.** Derive the stability limit of gradient descent on a quadratic $f = \frac12 a x^2$ (the largest
$\eta$ for which it converges).

**D4.** Write the momentum update and explain why it accelerates in consistent directions and damps
oscillations. Contrast with Nesterov.

**D5.** Explain AdaGrad's per-parameter rate and why its cumulative sum makes the learning rate decay
to 0.

**D6.** Write RMSProp's EMA update and explain why it fixes AdaGrad's decay.

**D7.** Write the full Adam update including bias correction, and identify the momentum (first moment)
and RMSProp (second moment) components.

**D8.** Derive Adam's bias correction: show $\mathbb{E}[\mathbf{m}_t] = (1-\beta_1^t)\mathbb{E}[\mathbf{g}]$
and why dividing by $(1-\beta_1^t)$ removes the bias.

**D9.** Explain AdamW's decoupled weight decay and why it differs from L2 regularization inside Adam.

**D10.** Explain why second-order methods (Newton, L-BFGS) are impractical for large stochastic deep
learning, and how Adam approximates curvature cheaply.

---

## Tier 2 — Implementation

**I1.** Implement SGD, momentum, Nesterov, AdaGrad, RMSProp, and Adam. Verify Adam step-for-step
against `torch.optim.Adam`.

**I2.** Reproduce Experiment 1: show stall / diverge / converge across learning rates, and find the
stability limit.

**I3.** Reproduce Experiment 2: momentum and Nesterov beating SGD on an ill-conditioned ravine.

**I4.** Reproduce Experiment 3: Adam vs SGD on a badly-conditioned problem.

**I5.** Reproduce Experiment 4: measure Adam's early updates with and without bias correction.

**I6.** Reproduce Experiment 5: SGD noise escaping a saddle where deterministic GD stalls.

**I7.** Reproduce Experiment 6: AdaGrad's effective learning rate decaying vs RMSProp's surviving.

**I8.** Implement AdamW (decoupled weight decay) and compare to Adam-with-L2 on a small net.

**I9.** Implement a cosine learning-rate schedule with linear warmup and show its effect on training.

**I10.** Train a small MLP with SGD+momentum and with Adam; compare training-loss speed and test
accuracy (the "Adam trains faster but SGD generalizes" phenomenon).

---

## Tier 3 — Interview

**Q1.** What does an optimizer do, given the gradient?

**Q2.** Why is mini-batch SGD used instead of full-batch?

**Q3.** Why is the learning rate so important?

**Q4.** What problem does momentum solve?

**Q5.** What is Adam, and what does it combine?

**Q6.** Why does Adam need bias correction?

**Q7.** What is the difference between Adam and AdamW?

**Q8.** Why did AdaGrad fall out of favor?

**Q9.** When would you use SGD+momentum over Adam?

**Q10.** Why not use Newton's method for deep learning?

**Q11.** What is learning-rate warmup and when is it needed?

**Q12.** In high dimensions, what is the main obstacle — local minima or saddle points?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Implement every optimizer and verify Adam against PyTorch
- [ ] Explain the learning-rate stability limit and tune it first
- [ ] Explain momentum and adaptive rates mechanistically
- [ ] Write the Adam update and its bias correction
- [ ] Choose an optimizer and schedule for a given problem
- [ ] Explain why SGD noise helps and why second-order methods don't scale
