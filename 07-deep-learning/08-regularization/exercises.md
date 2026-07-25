# 07.08 — Exercises: Regularization

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that adding an L2 penalty $\frac{\lambda}{2}\sum w^2$ gives the weight-decay update
$w \leftarrow (1-\eta\lambda)w - \eta\nabla_w L$.

**D2.** Explain, in bias-variance terms, why shrinking weights reduces variance and how it can
under-fit if $\lambda$ is too large.

**D3.** Explain why weight decay $\ne$ L2 penalty for adaptive optimizers, and how AdamW decouples
them.

**D4.** Derive the dropout inference scaling: show that to keep $\mathbb{E}[\text{activation}]$
consistent, you either scale by $(1-p)$ at test or by $1/(1-p)$ at train (inverted dropout).

**D5.** Explain the two views of dropout: preventing co-adaptation, and an implicit ensemble of
weight-sharing sub-networks.

**D6.** Explain early stopping as a regularizer and relate it to constraining the weights near their
initial values.

**D7.** Explain why data augmentation reduces overfitting, and what property of the task it must
respect (label preservation).

**D8.** Explain SGD's implicit regularization toward flat minima and its connection to double descent
([05.01 §10](../../05-model-evaluation/01-bias-variance-and-theory/)).

**D9.** Explain how batch normalization has a regularizing effect.

**D10.** Given train and validation curves, state the rule for whether to add or reduce
regularization.

---

## Tier 2 — Implementation

**I1.** Implement L2 weight decay in the backward pass. Reproduce Experiment 2: show validation
improving as $\lambda$ grows.

**I2.** Implement inverted dropout (forward + backward). Verify the mean-preservation against PyTorch.

**I3.** Reproduce Experiment 1: overfit a big net on limited data.

**I4.** Reproduce Experiment 3: show dropout closing the train-val gap; find the best rate.

**I5.** Reproduce Experiment 4: confirm inverted dropout keeps the expected activation constant.

**I6.** Implement early stopping with patience. Reproduce Experiment 5.

**I7.** Reproduce Experiment 6: MC-dropout ensemble vs a single sub-network vs deterministic
inference.

**I8.** Implement AdamW (decoupled decay) and compare to Adam-with-L2 on an overfitting task.

**I9.** Implement mixup (convex combinations of pairs) and measure its effect.

**I10.** *(Combining.)* Combine weight decay + dropout + early stopping and tune them against a
validation set; report the best combination.

---

## Tier 3 — Interview

**Q1.** What is regularization, in bias-variance terms?

**Q2.** How does weight decay work?

**Q3.** What is dropout and why does it help?

**Q4.** How does dropout behave at inference?

**Q5.** What is early stopping?

**Q6.** Why is data augmentation so effective?

**Q7.** What is the difference between weight decay and L2 for Adam?

**Q8.** Is more regularization always better?

**Q9.** What is implicit regularization?

**Q10.** How do you decide how much to regularize?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain every regularizer as a bias-variance trade
- [ ] Derive weight decay and the dropout inference scaling
- [ ] Implement dropout and weight decay correctly
- [ ] Diagnose over- vs under-fitting from the train-val gap
- [ ] Explain SGD's implicit regularization
- [ ] Combine regularizers and tune them against validation
