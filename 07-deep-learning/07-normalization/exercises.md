# 07.07 — Exercises: Normalization

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the batch-norm forward pass and explain the role of the learnable $\gamma, \beta$.

**D2.** Show that $\gamma, \beta$ let the network recover the un-normalized activation, so no
representational power is lost.

**D3.** Derive the batch-norm backward pass
$\mathrm{d}x = \frac{1}{N\sigma}\big(N\,\mathrm{d}\hat x - \sum\mathrm{d}\hat x - \hat x\sum \hat x\,\mathrm{d}\hat x\big)$
and explain why the batch mean/variance couple all examples.

**D4.** Explain the "internal covariate shift" story and why Santurkar et al. showed it is not the real
mechanism. What is?

**D5.** Explain why batch norm needs different behavior at training and inference, and how running
statistics are accumulated.

**D6.** Explain why batch norm degrades at small batch sizes.

**D7.** Write the layer-norm forward pass and contrast the axis it normalizes over with batch norm's.

**D8.** Explain why layer norm is batch-independent and needs no running statistics.

**D9.** Describe group norm and instance norm as choices of which dimensions to average over. Draw the
"which axes" picture for BN/LN/IN/GN.

**D10.** Explain pre-norm vs post-norm in a Transformer residual block and why pre-norm is more stable.

---

## Tier 2 — Implementation

**I1.** Implement batch-norm forward and backward. Verify both against `torch.nn.BatchNorm1d`.

**I2.** Implement layer-norm forward and backward. Verify against `torch.nn.LayerNorm`.

**I3.** Reproduce Experiment 1: show BN stabilizing activation std across a deep network.

**I4.** Reproduce Experiment 2: BN enabling a high learning rate that diverges without it.

**I5.** Reproduce Experiment 3: BN letting a badly-initialized network train.

**I6.** Reproduce Experiment 4: show train vs eval outputs and the batch-stats-at-inference bug.

**I7.** Reproduce Experiment 5: measure BN's statistic noise vs batch size.

**I8.** Reproduce Experiment 6: layer norm's batch-independent outputs vs batch norm's.

**I9.** Implement group norm and show it matches BN at large batches and beats it at batch size 2.

**I10.** *(Placement.)* Compare pre-activation vs post-activation BN, and pre-norm vs post-norm on a
small residual net.

---

## Tier 3 — Interview

**Q1.** What does batch normalization do?

**Q2.** Why does it help? (Give the modern answer, not covariate shift.)

**Q3.** Why does BN behave differently in training and inference?

**Q4.** What's the bug if you forget `model.eval()`?

**Q5.** Why does BN struggle with small batches?

**Q6.** How is layer norm different from batch norm?

**Q7.** Why do Transformers use layer norm, not batch norm?

**Q8.** What is group norm for?

**Q9.** Does normalization lose information?

**Q10.** Where do you place normalization — before or after the activation? Pre- or post-norm?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Derive BN's forward and backward passes
- [ ] Explain the real mechanism (smoother landscape) and its consequences
- [ ] Handle the train/inference distinction correctly
- [ ] Explain BN's batch-size dependence and its fixes
- [ ] Contrast BN, LN, IN, GN by the axes they normalize
- [ ] Choose a normalization and placement for a given architecture
