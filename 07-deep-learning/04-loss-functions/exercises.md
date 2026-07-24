# 07.04 — Exercises: Loss Functions

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that MSE is the Gaussian negative log-likelihood and its minimizer is the conditional
mean; that MAE is the Laplace NLL and its minimizer is the median.

**D2.** Derive binary cross-entropy as the Bernoulli NLL, and categorical cross-entropy as the
categorical NLL.

**D3.** Derive the softmax + cross-entropy gradient w.r.t. the logits and show it equals
$\hat{\mathbf{p}} - \mathbf{y}$. Where do the softmax and log derivatives cancel?

**D4.** Derive the sigmoid + BCE gradient and show it equals $\hat p - y$.

**D5.** *(The key result.)* For a sigmoid output, compute $\partial L/\partial z$ for MSE and for BCE.
Show MSE's contains $\sigma'(z)$ (vanishes on confident errors) and BCE's does not.

**D6.** Derive the numerically stable softmax (max-subtraction) and the combined
$-z_c + \log\sum_j e^{z_j}$ form of cross-entropy from logits.

**D7.** Show that cross-entropy equals KL divergence plus the (constant) entropy of the target.

**D8.** Write the focal loss and show the modulating factor $(1-\hat p_t)^\gamma$ down-weights easy
examples.

**D9.** Derive the effect of label smoothing on the target and explain why it reduces overconfidence.

**D10.** Write the Huber loss and its gradient, and explain how it interpolates MSE and MAE.

---

## Tier 2 — Implementation

**I1.** Implement MSE, MAE, Huber, BCE-with-logits, and CE-with-logits with their gradients. Verify
values and gradients against PyTorch autograd.

**I2.** Reproduce Experiment 1: find each loss's minimizing constant on skewed data.

**I3.** Reproduce Experiment 2: verify the softmax+CE gradient equals $\hat{\mathbf{p}} - \mathbf{y}$
against finite differences.

**I4.** Reproduce Experiment 3: tabulate MSE vs CE gradient magnitude on confidently-wrong examples,
and then train two classifiers to show CE converging faster.

**I5.** Reproduce Experiment 4: show naive softmax→log producing NaN on large logits and the stable
form giving the correct loss.

**I6.** Implement focal loss and reproduce Experiment 5; then train on imbalanced data and compare to
plain CE.

**I7.** Implement label smoothing and measure its effect on calibration
([05.06](../../05-model-evaluation/06-calibration/)).

**I8.** Implement stable `log_softmax` and verify against PyTorch on extreme logits.

**I9.** Implement KL-divergence loss for distribution matching (soft targets / distillation).

**I10.** *(Multi-label.)* Implement per-label BCE for a multi-label problem and contrast with softmax
CE.

---

## Tier 3 — Interview

**Q1.** What is the difference between a loss and a metric?

**Q2.** Why is every standard loss a negative log-likelihood?

**Q3.** What loss and output activation go with regression? Binary? Multiclass?

**Q4.** Why use cross-entropy instead of MSE for classification?

**Q5.** What is the softmax + cross-entropy gradient?

**Q6.** Why do frameworks combine softmax and cross-entropy into one op?

**Q7.** How do you compute cross-entropy stably?

**Q8.** What is focal loss for?

**Q9.** What does label smoothing do?

**Q10.** MSE or MAE for regression — how do you choose?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive each loss as a negative log-likelihood
- [ ] Derive the softmax+CE gradient $\hat{\mathbf{p}}-\mathbf{y}$
- [ ] Explain why MSE fails for classification at the gradient level
- [ ] Compute cross-entropy numerically stably from logits
- [ ] Match every task to its loss and output activation
- [ ] Choose focal loss / label smoothing when appropriate
