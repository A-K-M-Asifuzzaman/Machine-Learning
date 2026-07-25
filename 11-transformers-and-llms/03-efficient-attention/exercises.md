# 11.03 — Exercises: Efficient Attention

Three tiers. **Derivation** (paper), **implementation** (code), **interview** (explain out loud).

---

## Tier 1 — Derivation

**D1.** Show that during autoregressive generation, past keys and values are fixed, and derive the
$O(n^2) \to O(n)$ saving of the KV cache.

**D2.** Compute the KV-cache memory for MHA, GQA, and MQA as a function of heads, head-dim, length, and
precision. Derive the reduction factors.

**D3.** Derive RoPE's relative-position property: show $\text{RoPE}(q, m) \cdot \text{RoPE}(k, n)$
depends only on $m - n$.

**D4.** Explain why RoPE and ALiBi extrapolate to unseen sequence lengths while absolute encodings do
not.

**D5.** Derive the online (streaming) softmax: the running-max/running-sum recurrence and the rescaling
correction, and show it equals the batch softmax.

**D6.** Explain why FlashAttention is memory-bound-optimal (SRAM vs HBM) and exact, not approximate.

**D7.** Derive the cost of windowed attention ($O(nw)$) and of linear attention ($O(n)$ via the
kernel-feature reordering).

**D8.** Explain the quality/efficiency trade-off of MQA vs GQA vs MHA.

**D9.** Explain the "lost in the middle" phenomenon and why long context is hard even when it fits in
memory.

**D10.** Compare the long-context approaches: exact (Flash+GQA+RoPE), sparse, linear, and state-space
(Mamba).

---

## Tier 2 — Implementation

**I1.** Implement a KV cache and verify identical output to recompute-from-scratch (Experiment 1).

**I2.** Compute KV-cache sizes for MHA/GQA/MQA and reproduce Experiment 2.

**I3.** Implement RoPE and verify the relative-position property (Experiment 3).

**I4.** Implement the online softmax / FlashAttention and verify it equals standard attention
(Experiment 4).

**I5.** Implement ALiBi biases and reproduce the recency pattern (Experiment 5).

**I6.** Implement GQA (grouped K,V) and verify it is a valid attention against a reference grouped
computation.

**I7.** Implement windowed (local) attention and measure its cost vs full attention across lengths.

**I8.** Implement linear attention (kernel feature map) and compare quality/cost to softmax attention.

**I9.** Implement RoPE frequency scaling (NTK/YaRN) to extend context beyond training length and test
extrapolation.

**I10.** *(Bench.)* Measure wall-clock and peak memory of standard vs FlashAttention-style attention as
length grows.

---

## Tier 3 — Interview

**Q1.** What is the KV cache and why is it essential for LLM inference?

**Q2.** What is the difference between MHA, MQA, and GQA?

**Q3.** What is RoPE and what property does it have?

**Q4.** How does ALiBi encode position?

**Q5.** Why do RoPE/ALiBi extrapolate but absolute encodings don't?

**Q6.** What does FlashAttention do, and is it an approximation?

**Q7.** What is the online softmax?

**Q8.** How do sparse and linear attention reduce the $O(n^2)$ cost?

**Q9.** What limits context length even with efficient attention?

**Q10.** How do modern LLMs achieve 100K+ token contexts?

---

## Checkpoints

You have understood this chapter if you can:

- [ ] Explain and implement the KV cache and its $O(n)$ saving
- [ ] Compute KV-cache memory for MHA/GQA/MQA
- [ ] Derive and implement RoPE's relative-position property
- [ ] Derive and implement the online softmax (FlashAttention)
- [ ] Distinguish exact vs approximate efficiency methods
- [ ] Explain long-context extrapolation and its limits
- [ ] Name the stack modern LLMs use for long context
