# 11.04 — Exercises: Scaling Laws & Modern Architecture

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the Chinchilla parametric loss $L(N,D) = E + A/N^\alpha + B/D^\beta$ and interpret each
term, including the irreducible floor $E$.

**D2.** Given $C = 6ND$, derive the compute-optimal $N^*(C)$ and $D^*(C)$ by minimizing $L$ subject to
fixed $C$. Show they are power laws in $C$.

**D3.** Show that the reducible loss is a power law in compute along the optimal frontier, i.e. straight
in log-log space.

**D4.** Explain why loss is U-shaped in model size at fixed compute (under- vs over-parameterized
regimes).

**D5.** Compute GPT-3's tokens-per-parameter and contrast with Chinchilla's ~20; explain the
undertraining conclusion.

**D6.** Explain why compute-optimal (train) and serving-optimal (inference) point toward even
smaller-and-longer-trained models.

**D7.** Derive the MoE FLOP saving: total vs active parameters for $E$ experts, top-$k$ routing.

**D8.** Explain the MoE load-balancing problem and the auxiliary loss that addresses it.

**D9.** Explain the "emergent abilities" debate: how discontinuous metrics can manufacture apparent
emergence.

**D10.** Justify RMSNorm and SwiGLU as improvements over LayerNorm and a plain MLP.

---

## Tier 2 — Implementation

**I1.** Implement the Chinchilla loss and `compute_optimal`; reproduce Experiment 1 (power-law fit).

**I2.** Reproduce Experiment 2: sweep model size at fixed compute and find the U-curve minimum.

**I3.** Reproduce Experiment 3: compare GPT-3 to the compute-optimal split at its budget.

**I4.** Implement a MoE layer (router + top-$k$ + experts) and reproduce Experiment 4's sparsity.

**I5.** Add a load-balancing auxiliary loss to your MoE and measure expert utilization with and without
it.

**I6.** Fit your own scaling law: train a family of small models of increasing size on a fixed task and
fit loss vs $N$.

**I7.** Implement RMSNorm and verify it matches an RMSNorm reference; compare cost to LayerNorm.

**I8.** Implement a SwiGLU MLP and compare it to a GELU MLP on a toy task at matched parameter count.

**I9.** Use a fitted scaling law to *predict* a larger model's loss, then train it and check the
prediction.

**I10.** *(Serving.)* Estimate inference FLOPs/token for a dense model vs an MoE of equal total params.

---

## Tier 3 — Interview

**Q1.** What are neural scaling laws?

**Q2.** Why are scaling laws so important practically?

**Q3.** What is the Chinchilla compute-optimal result?

**Q4.** Why was GPT-3 undertrained?

**Q5.** How does compute relate to parameters and tokens ($C \approx 6ND$)?

**Q6.** What is a Mixture-of-Experts model?

**Q7.** Why does MoE decouple parameters from compute?

**Q8.** What is the load-balancing problem in MoE?

**Q9.** What are emergent abilities, and are they real?

**Q10.** What are the main components of a modern LLM architecture beyond the original transformer?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] State and use the power-law scaling relationships
- [ ] Derive the compute-optimal frontier and the U-curve
- [ ] Explain the Chinchilla result and GPT-3's undertraining
- [ ] Implement MoE routing and compute its FLOP saving
- [ ] Explain load balancing and emergent-ability debates
- [ ] Name and justify the modern architecture refinements
- [ ] Reason about the limits of scaling
