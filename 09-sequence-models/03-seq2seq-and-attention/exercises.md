# 09.03 — Exercises: Seq2seq & Attention

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Write the encoder–decoder equations and identify precisely where the fixed-context bottleneck
occurs.

**D2.** Derive the attention output $c_i = \sum_j \alpha_{ij} v_j$ with $\alpha = \text{softmax}(e)$ and
show the weights sum to 1.

**D3.** Write the additive (Bahdanau) and dot-product (Luong) scoring functions and compare their
parameter counts and cost.

**D4.** Show that dot products of $d$-dimensional vectors with unit-variance entries have variance
$\propto d$, and hence justify the $1/\sqrt{d}$ scaling.

**D5.** Explain, via an information/rank argument, why a $d$-dimensional context cannot losslessly
represent an $L$-state sequence once $L \cdot k > d$ (Experiment 3).

**D6.** Derive the gradient of the attention output w.r.t. the scores and show attention is fully
differentiable (soft attention).

**D7.** Explain why hard attention (selecting one position) is non-differentiable and how it is trained
(REINFORCE / straight-through).

**D8.** Write out beam search and its complexity in beam width $k$ and vocabulary $|V|$; contrast with
greedy and exhaustive search.

**D9.** Explain the length bias of beam search and how length normalization corrects it.

**D10.** Show how attention's query–key–value mechanism generalizes to self-attention, and what must be
added because attention is order-agnostic.

---

## Tier 2 — Implementation

**I1.** Implement dot-product and additive attention; verify both produce valid distributions and the
output equals $A V$ (Experiment 1).

**I2.** Reproduce Experiment 2: build content-based queries and show the alignment is diagonal (copy)
and anti-diagonal (reverse).

**I3.** Reproduce Experiment 3: measure the PCA-optimal reconstruction loss of a fixed context vs
sequence length.

**I4.** Reproduce Experiment 4: measure attention entropy vs score scaling and confirm $1/\sqrt{d}$
keeps it soft.

**I5.** Implement greedy and beam search; reproduce Experiment 5 where beam beats greedy.

**I6.** Build a full seq2seq with attention (encoder + decoder + Bahdanau attention) and train it on a
copy/reverse task; show it beats a no-attention baseline on long sequences.

**I7.** Visualize the attention alignment matrix on a trained translation-like toy task.

**I8.** Add length normalization to beam search and measure its effect on output length.

**I9.** Implement Luong's global vs local attention and compare.

**I10.** *(Bridge.)* Modify your attention to attend a sequence to itself (self-attention) and confirm
it matches the transformer's scaled dot-product attention.

---

## Tier 3 — Interview

**Q1.** What problem does attention solve in seq2seq models?

**Q2.** What is the fixed-context bottleneck?

**Q3.** How does attention compute its output?

**Q4.** What is the difference between additive and dot-product attention?

**Q5.** Why do transformers divide the scores by $\sqrt{d}$?

**Q6.** What does an attention weight matrix represent?

**Q7.** What is the difference between greedy and beam search?

**Q8.** What are the failure modes of beam search?

**Q9.** What is the difference between soft and hard attention?

**Q10.** How did attention lead to the transformer?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain the encoder–decoder and its bottleneck
- [ ] Derive the attention weights and output
- [ ] Implement additive and dot-product scoring
- [ ] Explain the $1/\sqrt{d}$ scaling from the score variance
- [ ] Interpret an attention alignment matrix
- [ ] Implement greedy and beam search and explain the trade-off
- [ ] Trace the path from attention to self-attention and transformers
