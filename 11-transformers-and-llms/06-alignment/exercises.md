# 11.06 — Exercises: Alignment

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** State the alignment problem (HHH) and explain why pretraining does not achieve it.

**D2.** Write the Bradley-Terry preference model and the reward-model training loss. Show fitting it
recovers a ranking.

**D3.** Write the KL-constrained RLHF objective and derive its closed-form optimum
$\pi^*(y) \propto \pi_{\text{ref}}(y)\exp(r(y)/\beta)$ (Lagrangian / calculus-of-variations).

**D4.** Explain the role of $\beta$: the reward-vs-KL trade-off and what happens at the extremes.

**D5.** Define reward hacking and explain it via Goodhart's law; sketch the true-reward-vs-optimization
curve.

**D6.** Invert the RLHF optimum to write $r(y) = \beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} + C$, and
substitute into Bradley-Terry to derive the DPO loss.

**D7.** Explain why DPO needs no reward model, no sampling, and no RL, yet targets the same optimum.

**D8.** Sketch PPO's clipped objective and how the KL penalty enters RLHF.

**D9.** Explain Constitutional AI / RLAIF and how AI feedback replaces human labels.

**D10.** Explain scalable oversight: why aligning models on superhuman tasks is hard.

---

## Tier 2 — Implementation

**I1.** Fit a Bradley-Terry reward model from preference pairs; reproduce Experiment 1 (recover the
ranking).

**I2.** Verify the RLHF closed form against direct optimization of the KL-constrained objective
(Experiment 2).

**I3.** Reproduce Experiment 3: sweep $\beta$ and plot reward vs KL.

**I4.** Reproduce Experiment 4: over-optimize an imperfect proxy reward and show true reward peaks then
falls.

**I5.** Implement the DPO loss and verify it recovers the RLHF optimum (Experiment 5).

**I6.** Implement a minimal PPO loop (sample, score, clipped update, KL penalty) on a toy policy.

**I7.** Implement IPO or KTO and compare to DPO on the same preferences.

**I8.** Train a reward model on a real preference dataset and evaluate its accuracy on held-out pairs.

**I9.** Fine-tune a small model with DPO on a preference dataset and evaluate helpfulness.

**I10.** *(Overoptimization.)* Empirically reproduce a reward-model overoptimization curve (true vs
proxy reward vs KL budget).

---

## Tier 3 — Interview

**Q1.** What is alignment and why is it needed?

**Q2.** What are the stages of RLHF?

**Q3.** How is a reward model trained from human feedback?

**Q4.** Why does RLHF include a KL penalty to the reference model?

**Q5.** What is the closed-form solution of the RLHF objective?

**Q6.** What is reward hacking?

**Q7.** How does DPO work, and why is it simpler than RLHF?

**Q8.** Do DPO and RLHF optimize the same thing?

**Q9.** What is Constitutional AI / RLAIF?

**Q10.** What are the open problems in alignment?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the alignment problem and the RLHF pipeline
- [ ] Derive and fit a Bradley-Terry reward model
- [ ] Derive the KL-constrained optimum and the role of $\beta$
- [ ] Explain and demonstrate reward hacking
- [ ] Derive the DPO loss from the RLHF optimum
- [ ] Explain why DPO and RLHF share an optimum
- [ ] Describe Constitutional AI and scalable-oversight challenges
